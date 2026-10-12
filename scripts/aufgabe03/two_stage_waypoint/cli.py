"""Arena localization, mandatory Nav2 staging, then isolated waypoint motion."""
import argparse
import math
import shlex
import subprocess
import sys
import time
from dataclasses import fields
from datetime import datetime
from pathlib import Path

from .model import (ArenaActiveSpinConfig, ArenaGeometryConfig, CoordinatorConfig,
                    SPIN_SHARED_FIELDS, geometry_arguments, spin_arguments)
from .pure import build_follower_command, load_waypoints, staging_goal_from_waypoints

# Measured arena tuning changes exploration limits, never motion authorization.
REAL_SHADOW_DEFAULTS = {
    "arena_active_explore_max_attempts": 12,
    "arena_active_explore_max_single_move_m": 0.80,
    "arena_active_explore_max_total_distance_m": 5.00,
    "arena_active_explore_max_candidate_path_m": 3.00,
    "arena_active_explore_grid_size_m": 5.0,
    "arena_active_explore_inflation_radius_m": 0.12,
    "arena_active_explore_soft_clearance_radius_m": 0.15,
    "arena_active_explore_soft_clearance_weight": 2.0,
    "arena_active_explore_max_path_segments": 24,
    "arena_active_explore_curve_lookahead_m": 0.16,
    "arena_active_explore_curve_linear_speed_mps": 0.05,
    "arena_active_explore_curve_max_angular_rad_s": 0.40,
    "arena_active_explore_map_max_samples": 1500,
}
CHOICES = {
    "arena_active_spin_direction": ("ccw", "cw"),
    "arena_active_recovery_mode": ("none", "active_explore"),
    "arena_active_recovery_executor": ("dry_run", "cmd_vel"),
    "heater_wall_side": ("+x", "-x"),
    "arena_force_short_wall_side": ("axis_negative", "axis_positive"),
    "arena_force_short_wall_type": ("heater", "clean"),
}
NONNEGATIVE = {
    "amcl_settle_min_sec", "min_waypoint_spacing_m", "arena_width_match_min_margin_m",
    "arena_max_short_wall_range_sum_error_m", "arena_active_explore_soft_clearance_radius_m",
    "arena_active_explore_soft_clearance_weight",
}
SIGNED = {"arena_map_center_x", "arena_map_center_y", "arena_map_yaw_deg"}


def add_options(parser, defaults, mapping):
    for field, name in mapping.items():
        default = getattr(defaults, field)
        option = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(option, default=default, action=argparse.BooleanOptionalAction)
        else:
            value_type = (str if name in CHOICES else float) if default is None else type(default)
            parser.add_argument(option, default=default, type=value_type,
                                choices=CHOICES.get(name))


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run arena-prior localization, Nav2 staging, and waypoint following.")
    add_options(parser, CoordinatorConfig(), {field.name: field.name
                                             for field in fields(CoordinatorConfig)})
    add_options(parser, ArenaGeometryConfig(), geometry_arguments())
    add_options(parser, ArenaActiveSpinConfig("", Path()), {
        field: name for field, name in spin_arguments().items()
        if field not in SPIN_SHARED_FIELDS
    })
    parser.add_argument("--arena-active-diagnostics-json", type=Path)
    parser.add_argument("--arena-active-profile", choices=("default", "real_shadow"), default="default")
    parser.set_defaults(arena_active_require_operator_confirmation=False)
    parser.add_argument("--no-arena-active-operator-confirmation",
                        dest="arena_active_require_operator_confirmation", action="store_false")
    parser.add_argument("--arena-active-explore-allow-unknown",
                        dest="arena_active_explore_unknown_blocked", action="store_false")
    # Apply preset before the final parse so explicit arguments win.
    profile = argparse.ArgumentParser(add_help=False)
    profile.add_argument("--arena-active-profile", default="default")
    if profile.parse_known_args(argv)[0].arena_active_profile == "real_shadow":
        parser.set_defaults(**REAL_SHADOW_DEFAULTS)
    args = parser.parse_args(argv)
    if not args.run_id:
        args.run_id = datetime.now().strftime("arena_prior_two_stage_%Y%m%d_%H%M%S")
    validate_args(parser, args)
    return args


def validate_args(parser, args):
    for name, value in vars(args).items():
        if type(value) not in (int, float):
            continue
        if not math.isfinite(value):
            parser.error(f"--{name.replace('_', '-')} must be finite")
        if name in SIGNED:
            continue
        if value < 0 or (value == 0 and name not in NONNEGATIVE):
            parser.error(f"--{name.replace('_', '-')} must be "
                         + ("non-negative" if name in NONNEGATIVE else "greater than zero"))
    if args.arena_active_recovery_executor != "dry_run" and args.arena_active_recovery_mode != "active_explore":
        parser.error("--arena-active-recovery-executor cmd_vel requires active_explore mode")
    if (args.arena_force_short_wall_side is None) != (args.arena_force_short_wall_type is None):
        parser.error("--arena-force-short-wall-side and --arena-force-short-wall-type must be provided together")
    if args.arena_active_explore_curve_goal_tolerance_m > args.arena_active_explore_curve_lookahead_m:
        parser.error("exploration goal tolerance must not exceed lookahead")
    if args.arena_active_explore_curve_linear_speed_mps > 0.10:
        parser.error("exploration linear speed must be <= 0.10 m/s")
    if args.arena_active_explore_curve_max_angular_rad_s > 0.80:
        parser.error("exploration angular speed must be <= 0.80 rad/s")
    for name in ("arena_min_parallel_score", "arena_min_short_wall_confidence",
                 "arena_min_classification_margin",
                 "arena_active_center_reposition_heater_approach_min_selected_score",
                 "arena_active_center_reposition_heater_approach_max_opposite_score",
                 "arena_active_center_reposition_heater_approach_min_delta"):
        if getattr(args, name) > 1:
            parser.error(f"--{name.replace('_', '-')} must be <= 1")


def run_follower_command(command, runner=subprocess.run):
    return runner(command, check=False, shell=False)


def cleanup_motion(node):
    try:
        node.cancel_active_goal()
    finally:
        node.stop_repeatedly()


def require_motion_confirmation(args, staging_goal, follower_command):
    if args.yes:
        return True
    print("This command moves the TurtleBot through localization, Nav2 staging, and path following.")
    print("Keep the arena clear and physical stop available.")
    print(f"Staging: ({staging_goal.waypoint.x:.3f}, {staging_goal.waypoint.y:.3f}), "
          f"yaw {staging_goal.yaw_deg:.1f} deg")
    print(f"Exploration: {args.arena_active_recovery_mode}/{args.arena_active_recovery_executor}")
    if args.verbose:
        print("Follower command:", shlex.join(follower_command))
    return input("Type RUN to start: ").strip() == "RUN"


def print_dry_run(args, waypoints, staging_goal, follower_command):
    print("Arena-prior two-stage waypoint run dry run")
    print(f"Waypoints: {len(waypoints)} from {args.waypoints}")
    print(f"Selected waypoint 0: {staging_goal.waypoint.x:.3f}, {staging_goal.waypoint.y:.3f}")
    print(f"Computed staging yaw: {staging_goal.yaw_deg:.1f} deg")
    print(f"Exploration: {args.arena_active_recovery_mode}/{args.arena_active_recovery_executor}")
    print("Follower command:", shlex.join(follower_command))
    if args.verbose:
        for name, value in sorted(vars(args).items()):
            print(f"  {name}: {value}")


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        waypoints = load_waypoints(args.waypoints)
        staging_goal = staging_goal_from_waypoints(waypoints)
    except (OSError, ValueError, KeyError) as exc:
        print(f"two_stage_waypoint_run.py: error: {exc}", file=sys.stderr)
        return 2
    follower_command = build_follower_command(args)
    if args.dry_run:
        print_dry_run(args, waypoints, staging_goal, follower_command)
        return 0
    if not require_motion_confirmation(args, staging_goal, follower_command):
        print("Run cancelled.")
        return 130
    from . import ros_runtime
    if ros_runtime.rclpy is None:
        print("ROS 2 Python modules are unavailable. Source ROS 2 Humble first.", file=sys.stderr)
        return 2
    node = None
    initialized = False
    return_code = 1
    try:
        ros_runtime.rclpy.init()
        initialized = True
        node = ros_runtime.TwoStageCoordinator(args)
        node.preflight_before_motion()
        arena_result = node.perform_arena_active_spin()
        if not arena_result.success:
            raise RuntimeError(f"arena-prior localization failed: {arena_result.failure_reason}")
        phase_start = time.time()
        node.publish_arena_active_initial_pose(arena_result.pose_prior, arena_result)
        node.wait_for_amcl_validation(args.arena_active_validation_timeout_sec,
                                      min_received_sec=phase_start,
                                      min_settle_sec=args.amcl_settle_min_sec)
        pose, frame = node.validate_post_localization_tf()
        node.validate_post_amcl_pose_prior(arena_result.pose_prior, pose, arena_result, frame)
        # Always send a Nav2 staging goal, including when the localized pose is on-path.
        node.navigate_to_staging(staging_goal)
        node.verify_arrival(staging_goal, waypoints)
        node.stop_repeatedly()
        result = run_follower_command(follower_command)
        if result.returncode != 0:
            raise RuntimeError(f"Follower exited with return code {result.returncode}")
        return_code = 0
    except KeyboardInterrupt:
        print("Interrupted; cancelling navigation and stopping.")
        return_code = 130
    except Exception as exc:
        print(f"two_stage_waypoint_run.py: error: {exc}", file=sys.stderr)
    finally:
        try:
            if node is not None:
                try:
                    cleanup_motion(node)
                except Exception as exc:
                    print(f"Motion cleanup failed: {exc}", file=sys.stderr)
                    return_code = 1
                finally:
                    node.destroy_node()
        finally:
            if initialized:
                ros_runtime.rclpy.shutdown()
    return return_code
