import argparse
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .experiment_log import append_csv_row, build_log_row
from .model import (
    CSV_HEADER,
    DEFAULT_AMCL_SETTLE_MIN_SEC,
    DEFAULT_ARENA_ACTIVE_MAX_POST_AMCL_PRIOR_POSITION_ERROR_M,
    DEFAULT_ARENA_ACTIVE_MAX_POST_AMCL_PRIOR_YAW_ERROR_DEG,
    DEFAULT_ARENA_ACTIVE_VALIDATION_TIMEOUT_SEC,
    DEFAULT_ARRIVAL_TOLERANCE_M,
    DEFAULT_ARRIVAL_YAW_TOLERANCE_DEG,
    DEFAULT_CONTROL_RATE_HZ,
    DEFAULT_FOLLOWER_SCRIPT,
    DEFAULT_FOLLOWER_START_ON_PATH_TOLERANCE_M,
    DEFAULT_FOLLOWER_STARTUP_TIMEOUT_SEC,
    DEFAULT_GOAL_TOLERANCE_M,
    DEFAULT_MAX_AMCL_AGE_SEC,
    DEFAULT_MAX_AMCL_VAR_X,
    DEFAULT_MAX_AMCL_VAR_Y,
    DEFAULT_MAX_AMCL_VAR_YAW_RAD2,
    DEFAULT_MAX_POSE_AGE_SEC,
    DEFAULT_MAX_SCAN_AGE_SEC,
    DEFAULT_MAX_STABLE_POSE_JUMP_M,
    DEFAULT_MAX_STABLE_YAW_JUMP_DEG,
    DEFAULT_MIN_WAYPOINT_SPACING_M,
    DEFAULT_NAV_TO_START_TIMEOUT_SEC,
    DEFAULT_MAX_GOAL_SNAP_M,
    DEFAULT_MAX_REPLAN_PATH_LENGTH_RATIO,
    DEFAULT_MAX_REPLAN_SCAN_AGE_SEC,
    DEFAULT_MAX_REPLAN_TF_AGE_SEC,
    DEFAULT_MAX_START_SNAP_M,
    DEFAULT_OBSTACLE_ANGLE_WINDOW_DEG,
    DEFAULT_OBSTACLE_FORWARD_DISTANCE_M,
    DEFAULT_OBSTACLE_FORWARD_HALF_WIDTH_M,
    DEFAULT_OBSTACLE_INFLATE_RADIUS_M,
    DEFAULT_OBSTACLE_MIN_CLUSTER_SIZE,
    DEFAULT_OBSTACLE_MIN_CLUSTER_WIDTH_M,
    DEFAULT_OBSTACLE_MIN_RANGE_M,
    DEFAULT_PREFLIGHT_TIMEOUT_SEC,
    DEFAULT_REPLAN_OUTPUT_DIR,
    DEFAULT_REPLAN_TIMEOUT_SEC,
    DEFAULT_RESULTS_CSV,
    DEFAULT_ROBOT_FOOTPRINT_RADIUS_M,
    DEFAULT_RUN_LOCAL_MAP_CLEARANCE_MARGIN_M,
    DEFAULT_RUN_LOCAL_MAP_CORRIDOR_CHECK_DISTANCE_M,
    DEFAULT_RUN_LOCAL_MAP_INITIAL_SCAN_COUNT,
    DEFAULT_RUN_LOCAL_MAP_INITIAL_SCAN_MODE,
    DEFAULT_RUN_LOCAL_MAP_INFLATION_RADIUS_M,
    DEFAULT_RUN_LOCAL_MAP_MAX_REJECTED_RATIO,
    DEFAULT_RUN_LOCAL_MAP_MAX_SCAN_AGE_SEC,
    DEFAULT_RUN_LOCAL_MAP_MAX_TF_AGE_SEC,
    DEFAULT_RUN_LOCAL_MAP_MAX_UPDATES,
    DEFAULT_RUN_LOCAL_MAP_MIN_HIT_COUNT,
    DEFAULT_RUN_LOCAL_MAP_MIN_USED_POINTS,
    DEFAULT_RUN_LOCAL_MAP_UPDATE_MODE,
    DEFAULT_SPIN_MIN_SCAN_RANGE_M,
    DEFAULT_STATIC_MAP,
    DEFAULT_SPIN_MIN_VALID_SCAN_COUNT,
    DEFAULT_STABLE_AMCL_SAMPLES,
    DEFAULT_TF_LOOKUP_RETRY_PERIOD_SEC,
    DEFAULT_TF_LOOKUP_TIMEOUT_SEC,
    DEFAULT_TF_READY_TIMEOUT_SEC,
    DEFAULT_WAYPOINTS_CSV,
    DEFAULT_WAYPOINT_TOLERANCE_M,
    RunDiagnostics,
)
from .pure import (
    build_follower_command,
    load_waypoints,
    staging_goal_from_waypoints,
    timestamp_now,
)


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
    print("\nThis command may move the physical TurtleBot.")
    print("Safety requirements:")
    print("  - clear the arena and keep an operator near the robot")
    print("  - keep Ctrl+C and physical stop available")
    print("  - ensure no other controller is intentionally publishing /cmd_vel")
    if args.arena_active_enable_center_reposition:
        print("  - reposition recovery may drive before /initialpose is published")
    print(f"Run ID: {args.run_id}")
    print(
        "Staging goal: "
        f"x={staging_goal.waypoint.x:.3f}, "
        f"y={staging_goal.waypoint.y:.3f}, "
        f"yaw={staging_goal.yaw_deg:.1f} deg"
    )
    print("Follower command:", shlex.join(follower_command))
    response = input("Type RUN to start arena-prior two-stage run: ").strip()
    return response == "RUN"


def print_dry_run(args, waypoints, staging_goal, follower_command):
    print("Arena-prior two-stage waypoint run dry run")
    print(f"Waypoint CSV: {args.waypoints}")
    print(f"Waypoints: {len(waypoints)}")
    print(
        "Selected waypoint 0: "
        f"x={staging_goal.waypoint.x:.3f}, y={staging_goal.waypoint.y:.3f}"
    )
    print(f"Computed staging yaw: {staging_goal.yaw_deg:.1f} deg")
    print("ROS interfaces:")
    print(f"  navigate action: {args.navigate_action}")
    print(f"  initial pose topic: {args.initial_pose_topic}")
    print(f"  amcl topic: {args.amcl_topic}")
    print(f"  cmd_vel topic: {args.cmd_vel_topic}")
    print(f"  scan topic: {args.scan_topic}")
    print(f"  odom topic: {args.odom_topic}")
    print(f"LiDAR map replan: {'enabled' if args.enable_lidar_map_replan else 'disabled'}")
    if args.enable_lidar_map_replan:
        print(f"  artifact only: {'yes' if args.lidar_replan_artifact_only else 'no'}")
        print(f"  static map: {args.static_map}")
        print(f"  output dir: {args.replan_output_dir}")
        print(
            "  initial scans: "
            f"{args.run_local_map_initial_scan_mode} x "
            f"{args.run_local_map_initial_scan_count}"
        )
        print(f"  update mode: {args.run_local_map_update_mode}")
        print(f"  min hit count: {args.run_local_map_min_hit_count}")
        print(f"  inflation radius: {args.run_local_map_inflation_radius_m:.3f} m")
    print(f"Follower command: {shlex.join(follower_command)}")
    print(f"Log path: {args.results_csv}")
    from .ros_runtime import rclpy

    print(f"ROS imports available: {'yes' if rclpy is not None else 'no'}")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Run arena-prior localization, Nav2 staging, and waypoint following."
        ),
    )
    parser.add_argument("--waypoints", default=DEFAULT_WAYPOINTS_CSV, type=Path)
    parser.add_argument("--results-csv", default=DEFAULT_RESULTS_CSV, type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--notes", default="arena_prior_two_stage_run")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--base-frame", default="base_footprint")
    parser.add_argument("--fallback-base-frame", default="base_link")

    parser.add_argument("--preflight-timeout-sec", default=DEFAULT_PREFLIGHT_TIMEOUT_SEC, type=float)
    parser.add_argument(
        "--nav-to-start-timeout-sec",
        default=DEFAULT_NAV_TO_START_TIMEOUT_SEC,
        type=float,
    )
    parser.add_argument("--tf-ready-timeout-sec", default=DEFAULT_TF_READY_TIMEOUT_SEC, type=float)
    parser.add_argument(
        "--tf-lookup-timeout-sec",
        default=DEFAULT_TF_LOOKUP_TIMEOUT_SEC,
        type=float,
    )
    parser.add_argument(
        "--tf-lookup-retry-period-sec",
        default=DEFAULT_TF_LOOKUP_RETRY_PERIOD_SEC,
        type=float,
    )

    parser.add_argument("--navigate-action", default="/navigate_to_pose")
    parser.add_argument("--initial-pose-topic", default="/initialpose")
    parser.add_argument("--amcl-topic", default="/amcl_pose")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--follower-script", default=DEFAULT_FOLLOWER_SCRIPT, type=Path)
    parser.add_argument("--python-executable", default="python3")
    parser.add_argument(
        "--follower-startup-timeout-sec",
        default=DEFAULT_FOLLOWER_STARTUP_TIMEOUT_SEC,
        type=float,
    )
    parser.add_argument(
        "--follower-start-on-path-tolerance-m",
        default=DEFAULT_FOLLOWER_START_ON_PATH_TOLERANCE_M,
        type=float,
        help=(
            "Maximum distance from the planned waypoint path that is accepted "
            "for handing off from Nav2 staging to the custom follower."
        ),
    )

    parser.add_argument("--max-pose-age-sec", default=DEFAULT_MAX_POSE_AGE_SEC, type=float)
    parser.add_argument("--max-scan-age-sec", default=DEFAULT_MAX_SCAN_AGE_SEC, type=float)
    parser.add_argument("--max-amcl-age-sec", default=DEFAULT_MAX_AMCL_AGE_SEC, type=float)
    parser.add_argument("--max-amcl-var-x", default=DEFAULT_MAX_AMCL_VAR_X, type=float)
    parser.add_argument("--max-amcl-var-y", default=DEFAULT_MAX_AMCL_VAR_Y, type=float)
    parser.add_argument(
        "--max-amcl-var-yaw-rad2",
        default=DEFAULT_MAX_AMCL_VAR_YAW_RAD2,
        type=float,
    )
    parser.add_argument("--stable-amcl-samples", default=DEFAULT_STABLE_AMCL_SAMPLES, type=int)
    parser.add_argument("--amcl-settle-min-sec", default=DEFAULT_AMCL_SETTLE_MIN_SEC, type=float)
    parser.add_argument(
        "--max-stable-pose-jump-m",
        default=DEFAULT_MAX_STABLE_POSE_JUMP_M,
        type=float,
    )
    parser.add_argument(
        "--max-stable-yaw-jump-deg",
        default=DEFAULT_MAX_STABLE_YAW_JUMP_DEG,
        type=float,
    )
    parser.add_argument(
        "--spin-min-scan-range-m",
        default=DEFAULT_SPIN_MIN_SCAN_RANGE_M,
        type=float,
    )
    parser.add_argument(
        "--spin-min-valid-scan-count",
        default=DEFAULT_SPIN_MIN_VALID_SCAN_COUNT,
        type=int,
    )
    parser.add_argument("--arrival-tolerance-m", default=DEFAULT_ARRIVAL_TOLERANCE_M, type=float)
    parser.add_argument(
        "--arrival-yaw-tolerance-deg",
        default=DEFAULT_ARRIVAL_YAW_TOLERANCE_DEG,
        type=float,
    )
    parser.add_argument("--waypoint-tolerance-m", default=DEFAULT_WAYPOINT_TOLERANCE_M, type=float)
    parser.add_argument("--goal-tolerance-m", default=DEFAULT_GOAL_TOLERANCE_M, type=float)
    parser.add_argument(
        "--min-waypoint-spacing-m",
        default=DEFAULT_MIN_WAYPOINT_SPACING_M,
        type=float,
    )
    parser.add_argument("--control-rate-hz", default=DEFAULT_CONTROL_RATE_HZ, type=float)
    parser.add_argument("--enable-lidar-map-replan", action="store_true")
    parser.add_argument("--lidar-replan-artifact-only", action="store_true")
    parser.add_argument("--static-map", default=DEFAULT_STATIC_MAP, type=Path)
    parser.add_argument("--replan-output-dir", default=DEFAULT_REPLAN_OUTPUT_DIR, type=Path)
    parser.add_argument("--max-replans", default=1, type=int)
    parser.add_argument("--replan-timeout-sec", default=DEFAULT_REPLAN_TIMEOUT_SEC, type=float)
    parser.add_argument("--max-replan-scan-age-sec", default=DEFAULT_MAX_REPLAN_SCAN_AGE_SEC, type=float)
    parser.add_argument("--max-replan-tf-age-sec", default=DEFAULT_MAX_REPLAN_TF_AGE_SEC, type=float)
    parser.add_argument("--allow-latest-tf-replan-fallback", action="store_true")
    parser.add_argument("--obstacle-forward-distance-m", default=DEFAULT_OBSTACLE_FORWARD_DISTANCE_M, type=float)
    parser.add_argument("--obstacle-forward-half-width-m", default=DEFAULT_OBSTACLE_FORWARD_HALF_WIDTH_M, type=float)
    parser.add_argument("--obstacle-angle-window-deg", default=DEFAULT_OBSTACLE_ANGLE_WINDOW_DEG, type=float)
    parser.add_argument("--obstacle-min-range-m", default=DEFAULT_OBSTACLE_MIN_RANGE_M, type=float)
    parser.add_argument("--robot-footprint-radius-m", default=DEFAULT_ROBOT_FOOTPRINT_RADIUS_M, type=float)
    parser.add_argument("--obstacle-min-cluster-size", default=DEFAULT_OBSTACLE_MIN_CLUSTER_SIZE, type=int)
    parser.add_argument("--obstacle-min-cluster-width-m", default=DEFAULT_OBSTACLE_MIN_CLUSTER_WIDTH_M, type=float)
    parser.add_argument("--obstacle-inflate-radius-m", default=DEFAULT_OBSTACLE_INFLATE_RADIUS_M, type=float)
    parser.add_argument("--max-start-snap-m", default=DEFAULT_MAX_START_SNAP_M, type=float)
    parser.add_argument("--max-goal-snap-m", default=DEFAULT_MAX_GOAL_SNAP_M, type=float)
    parser.add_argument(
        "--max-replan-path-length-ratio",
        default=DEFAULT_MAX_REPLAN_PATH_LENGTH_RATIO,
        type=float,
    )
    parser.add_argument(
        "--run-local-map-initial-scan-mode",
        default=DEFAULT_RUN_LOCAL_MAP_INITIAL_SCAN_MODE,
        choices=["none", "full"],
    )
    parser.add_argument(
        "--run-local-map-initial-scan-count",
        default=DEFAULT_RUN_LOCAL_MAP_INITIAL_SCAN_COUNT,
        type=int,
    )
    parser.add_argument(
        "--run-local-map-update-mode",
        default=DEFAULT_RUN_LOCAL_MAP_UPDATE_MODE,
        choices=["none", "forward", "full"],
    )
    parser.add_argument(
        "--run-local-map-min-hit-count",
        default=DEFAULT_RUN_LOCAL_MAP_MIN_HIT_COUNT,
        type=int,
    )
    parser.add_argument(
        "--run-local-map-inflation-radius-m",
        default=DEFAULT_RUN_LOCAL_MAP_INFLATION_RADIUS_M,
        type=float,
    )
    parser.add_argument(
        "--run-local-map-max-tf-age-sec",
        default=DEFAULT_RUN_LOCAL_MAP_MAX_TF_AGE_SEC,
        type=float,
    )
    parser.add_argument(
        "--run-local-map-max-scan-age-sec",
        default=DEFAULT_RUN_LOCAL_MAP_MAX_SCAN_AGE_SEC,
        type=float,
    )
    parser.add_argument(
        "--run-local-map-min-used-points",
        default=DEFAULT_RUN_LOCAL_MAP_MIN_USED_POINTS,
        type=int,
    )
    parser.add_argument(
        "--run-local-map-max-rejected-ratio",
        default=DEFAULT_RUN_LOCAL_MAP_MAX_REJECTED_RATIO,
        type=float,
    )
    parser.add_argument(
        "--run-local-map-corridor-check-distance-m",
        default=DEFAULT_RUN_LOCAL_MAP_CORRIDOR_CHECK_DISTANCE_M,
        type=float,
    )
    parser.add_argument("--run-local-map-corridor-radius-m", type=float)
    parser.add_argument(
        "--run-local-map-clearance-margin-m",
        default=DEFAULT_RUN_LOCAL_MAP_CLEARANCE_MARGIN_M,
        type=float,
    )
    parser.add_argument(
        "--run-local-map-max-updates",
        default=DEFAULT_RUN_LOCAL_MAP_MAX_UPDATES,
        type=int,
    )
    parser.add_argument("--run-local-map-artifact-prefix")

    parser.add_argument(
        "--arena-active-spin-direction",
        default="ccw",
        choices=["ccw", "cw"],
    )
    parser.add_argument("--arena-active-angular-speed-rad-s", default=0.25, type=float)
    parser.add_argument("--arena-active-max-spin-sec", default=30.0, type=float)
    parser.add_argument("--arena-active-spin-complete-tolerance-deg", default=5.0, type=float)
    parser.add_argument("--arena-active-min-angular-progress-rad-s", default=0.05, type=float)
    parser.add_argument("--arena-active-progress-check-sec", default=2.0, type=float)
    parser.add_argument("--arena-active-min-scan-samples", default=20, type=int)
    parser.add_argument("--arena-active-max-odom-scan-age-sec", default=0.20, type=float)
    parser.add_argument("--arena-active-stop-settle-sec", default=0.5, type=float)
    parser.add_argument("--arena-active-min-front-clearance-m", default=0.35, type=float)
    parser.add_argument("--arena-active-min-side-clearance-m", default=0.20, type=float)
    parser.add_argument("--arena-active-min-rear-clearance-m", default=0.20, type=float)
    parser.add_argument(
        "--arena-active-require-operator-confirmation",
        dest="arena_active_require_operator_confirmation",
        action="store_true",
    )
    parser.add_argument(
        "--no-arena-active-operator-confirmation",
        dest="arena_active_require_operator_confirmation",
        action="store_false",
    )
    parser.set_defaults(arena_active_require_operator_confirmation=True)
    parser.add_argument("--arena-active-allow-extra-cmd-vel-publishers", action="store_true")
    parser.add_argument(
        "--arena-active-validation-timeout-sec",
        default=DEFAULT_ARENA_ACTIVE_VALIDATION_TIMEOUT_SEC,
        type=float,
    )
    parser.add_argument(
        "--arena-active-max-post-amcl-prior-position-error-m",
        default=DEFAULT_ARENA_ACTIVE_MAX_POST_AMCL_PRIOR_POSITION_ERROR_M,
        type=float,
    )
    parser.add_argument(
        "--arena-active-max-post-amcl-prior-yaw-error-deg",
        default=DEFAULT_ARENA_ACTIVE_MAX_POST_AMCL_PRIOR_YAW_ERROR_DEG,
        type=float,
    )
    parser.add_argument("--arena-active-diagnostics-json", type=Path)
    parser.add_argument("--arena-active-range-stride", default=6, type=int)
    parser.add_argument("--arena-active-max-points", default=3000, type=int)
    parser.add_argument(
        "--arena-active-enable-center-reposition",
        "--arena-active-enable-reposition",
        dest="arena_active_enable_center_reposition",
        action="store_true",
    )
    parser.add_argument("--arena-active-center-reposition-max-attempts", default=1, type=int)
    parser.add_argument(
        "--arena-active-center-reposition-target-nearest-short-wall-range-m",
        default=1.65,
        type=float,
    )
    parser.add_argument("--arena-active-center-reposition-min-step-m", default=0.25, type=float)
    parser.add_argument("--arena-active-center-reposition-max-step-m", default=1.10, type=float)
    parser.add_argument(
        "--arena-active-center-reposition-linear-speed-mps",
        default=0.08,
        type=float,
    )
    parser.add_argument(
        "--arena-active-center-reposition-angular-speed-rad-s",
        default=0.25,
        type=float,
    )
    parser.add_argument(
        "--arena-active-center-reposition-heading-tolerance-deg",
        default=8.0,
        type=float,
    )
    parser.add_argument(
        "--arena-active-center-reposition-min-front-clearance-m",
        default=0.45,
        type=float,
    )
    parser.add_argument(
        "--arena-active-center-reposition-lateral-offset-threshold-m",
        default=0.25,
        type=float,
    )
    parser.add_argument(
        "--arena-active-center-reposition-lateral-target-offset-m",
        default=0.10,
        type=float,
    )
    parser.add_argument(
        "--arena-active-center-reposition-lateral-min-step-m",
        default=0.15,
        type=float,
    )
    parser.add_argument(
        "--arena-active-center-reposition-lateral-max-step-m",
        default=0.55,
        type=float,
    )
    parser.add_argument(
        "--arena-active-disable-heater-approach-reposition",
        dest="arena_active_center_reposition_enable_heater_approach",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--arena-active-center-reposition-heater-approach-max-attempts",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--arena-active-center-reposition-heater-approach-target-range-m",
        default=1.05,
        type=float,
    )
    parser.add_argument(
        "--arena-active-center-reposition-heater-approach-min-selected-score",
        default=0.50,
        type=float,
    )
    parser.add_argument(
        "--arena-active-center-reposition-heater-approach-max-opposite-score",
        default=0.30,
        type=float,
    )
    parser.add_argument(
        "--arena-active-center-reposition-heater-approach-min-delta",
        default=0.35,
        type=float,
    )
    parser.add_argument(
        "--arena-active-center-reposition-heater-approach-min-step-m",
        default=0.25,
        type=float,
    )
    parser.add_argument(
        "--arena-active-center-reposition-heater-approach-max-step-m",
        default=1.10,
        type=float,
    )
    parser.add_argument("--arena-length-m", default=3.90, type=float)
    parser.add_argument("--arena-width-m", type=float)
    parser.add_argument("--arena-heater-wall-width-m", default=2.016, type=float)
    parser.add_argument("--arena-clean-wall-width-m", default=1.967, type=float)
    parser.add_argument("--arena-width-match-min-margin-m", default=0.015, type=float)
    parser.add_argument("--arena-max-short-wall-range-sum-error-m", default=0.15, type=float)
    parser.add_argument("--arena-map-center-x", default=0.0, type=float)
    parser.add_argument("--arena-map-center-y", default=0.0, type=float)
    parser.add_argument("--arena-map-yaw-deg", default=0.0, type=float)
    parser.add_argument("--heater-wall-side", default="+x", choices=["+x", "-x"])
    parser.add_argument("--arena-min-wall-points", default=20, type=int)
    parser.add_argument("--arena-max-wall-separation-error-m", default=0.20, type=float)
    parser.add_argument("--arena-max-line-rmse-m", default=0.08, type=float)
    parser.add_argument("--arena-min-parallel-score", default=0.90, type=float)
    parser.add_argument("--arena-min-short-wall-confidence", default=0.75, type=float)
    parser.add_argument("--arena-min-classification-margin", default=0.15, type=float)
    parser.add_argument(
        "--arena-force-short-wall-side",
        choices=["axis_negative", "axis_positive"],
    )
    parser.add_argument(
        "--arena-force-short-wall-type",
        choices=["heater", "clean"],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args(argv)

    if not args.run_id:
        args.run_id = datetime.now().strftime("arena_prior_two_stage_%Y%m%d_%H%M%S")
    validate_args(parser, args)
    return args


def validate_args(parser, args):
    positive_float_fields = [
        "preflight_timeout_sec",
        "nav_to_start_timeout_sec",
        "tf_ready_timeout_sec",
        "tf_lookup_timeout_sec",
        "tf_lookup_retry_period_sec",
        "follower_startup_timeout_sec",
        "follower_start_on_path_tolerance_m",
        "arena_active_validation_timeout_sec",
        "arena_active_max_post_amcl_prior_position_error_m",
        "arena_active_max_post_amcl_prior_yaw_error_deg",
        "max_pose_age_sec",
        "max_scan_age_sec",
        "max_amcl_age_sec",
        "max_amcl_var_x",
        "max_amcl_var_y",
        "max_amcl_var_yaw_rad2",
        "max_stable_pose_jump_m",
        "max_stable_yaw_jump_deg",
        "spin_min_scan_range_m",
        "arrival_tolerance_m",
        "arrival_yaw_tolerance_deg",
        "waypoint_tolerance_m",
        "goal_tolerance_m",
        "control_rate_hz",
        "replan_timeout_sec",
        "max_replan_scan_age_sec",
        "max_replan_tf_age_sec",
        "obstacle_forward_distance_m",
        "obstacle_forward_half_width_m",
        "obstacle_angle_window_deg",
        "obstacle_min_range_m",
        "robot_footprint_radius_m",
        "obstacle_min_cluster_width_m",
        "obstacle_inflate_radius_m",
        "max_start_snap_m",
        "max_goal_snap_m",
        "max_replan_path_length_ratio",
        "run_local_map_inflation_radius_m",
        "run_local_map_max_tf_age_sec",
        "run_local_map_max_scan_age_sec",
        "run_local_map_corridor_check_distance_m",
        "run_local_map_clearance_margin_m",
        "arena_active_angular_speed_rad_s",
        "arena_active_max_spin_sec",
        "arena_active_spin_complete_tolerance_deg",
        "arena_active_min_angular_progress_rad_s",
        "arena_active_progress_check_sec",
        "arena_active_max_odom_scan_age_sec",
        "arena_active_stop_settle_sec",
        "arena_active_min_front_clearance_m",
        "arena_active_min_side_clearance_m",
        "arena_active_min_rear_clearance_m",
        "arena_active_center_reposition_target_nearest_short_wall_range_m",
        "arena_active_center_reposition_min_step_m",
        "arena_active_center_reposition_max_step_m",
        "arena_active_center_reposition_linear_speed_mps",
        "arena_active_center_reposition_angular_speed_rad_s",
        "arena_active_center_reposition_heading_tolerance_deg",
        "arena_active_center_reposition_min_front_clearance_m",
        "arena_active_center_reposition_lateral_offset_threshold_m",
        "arena_active_center_reposition_lateral_target_offset_m",
        "arena_active_center_reposition_lateral_min_step_m",
        "arena_active_center_reposition_lateral_max_step_m",
        "arena_active_center_reposition_heater_approach_target_range_m",
        "arena_active_center_reposition_heater_approach_min_selected_score",
        "arena_active_center_reposition_heater_approach_max_opposite_score",
        "arena_active_center_reposition_heater_approach_min_delta",
        "arena_active_center_reposition_heater_approach_min_step_m",
        "arena_active_center_reposition_heater_approach_max_step_m",
        "arena_length_m",
        "arena_heater_wall_width_m",
        "arena_clean_wall_width_m",
        "arena_max_wall_separation_error_m",
        "arena_max_line_rmse_m",
        "arena_min_parallel_score",
        "arena_min_short_wall_confidence",
        "arena_min_classification_margin",
    ]
    for field in positive_float_fields:
        if getattr(args, field) <= 0.0:
            parser.error(f"--{field.replace('_', '-')} must be greater than zero")
    if args.stable_amcl_samples < 1:
        parser.error("--stable-amcl-samples must be >= 1")
    if args.amcl_settle_min_sec < 0.0:
        parser.error("--amcl-settle-min-sec must be non-negative")
    if args.spin_min_valid_scan_count < 1:
        parser.error("--spin-min-valid-scan-count must be >= 1")
    if args.max_replans < 1:
        parser.error("--max-replans must be >= 1")
    if args.run_local_map_initial_scan_count < 1:
        parser.error("--run-local-map-initial-scan-count must be >= 1")
    if args.run_local_map_min_hit_count < 1:
        parser.error("--run-local-map-min-hit-count must be >= 1")
    if args.run_local_map_min_used_points < 1:
        parser.error("--run-local-map-min-used-points must be >= 1")
    if not (0.0 <= args.run_local_map_max_rejected_ratio <= 1.0):
        parser.error("--run-local-map-max-rejected-ratio must be between 0 and 1")
    if (
        args.run_local_map_corridor_radius_m is not None
        and args.run_local_map_corridor_radius_m <= 0.0
    ):
        parser.error("--run-local-map-corridor-radius-m must be greater than zero")
    if args.run_local_map_max_updates < 1:
        parser.error("--run-local-map-max-updates must be >= 1")
    if args.obstacle_min_cluster_size < 1:
        parser.error("--obstacle-min-cluster-size must be >= 1")
    if args.arena_active_min_scan_samples < 1:
        parser.error("--arena-active-min-scan-samples must be >= 1")
    if args.arena_active_range_stride < 1:
        parser.error("--arena-active-range-stride must be >= 1")
    if args.arena_active_max_points < 1:
        parser.error("--arena-active-max-points must be >= 1")
    if args.arena_active_center_reposition_max_attempts < 1:
        parser.error("--arena-active-center-reposition-max-attempts must be >= 1")
    if args.arena_active_center_reposition_heater_approach_max_attempts < 1:
        parser.error(
            "--arena-active-center-reposition-heater-approach-max-attempts must be >= 1"
        )
    if (
        args.arena_active_center_reposition_min_step_m
        > args.arena_active_center_reposition_max_step_m
    ):
        parser.error(
            "--arena-active-center-reposition-min-step-m must be <= "
            "--arena-active-center-reposition-max-step-m"
        )
    if (
        args.arena_active_center_reposition_lateral_min_step_m
        > args.arena_active_center_reposition_lateral_max_step_m
    ):
        parser.error(
            "--arena-active-center-reposition-lateral-min-step-m must be <= "
            "--arena-active-center-reposition-lateral-max-step-m"
        )
    if (
        args.arena_active_center_reposition_heater_approach_min_step_m
        > args.arena_active_center_reposition_heater_approach_max_step_m
    ):
        parser.error(
            "--arena-active-center-reposition-heater-approach-min-step-m must be <= "
            "--arena-active-center-reposition-heater-approach-max-step-m"
        )
    for field in [
        "arena_active_center_reposition_heater_approach_min_selected_score",
        "arena_active_center_reposition_heater_approach_max_opposite_score",
        "arena_active_center_reposition_heater_approach_min_delta",
    ]:
        value = getattr(args, field)
        if value > 1.0:
            parser.error(f"--{field.replace('_', '-')} must be <= 1")
    if args.arena_width_m is not None and args.arena_width_m <= 0.0:
        parser.error("--arena-width-m must be greater than zero")
    if args.arena_width_match_min_margin_m < 0.0:
        parser.error("--arena-width-match-min-margin-m must be non-negative")
    if args.arena_max_short_wall_range_sum_error_m < 0.0:
        parser.error("--arena-max-short-wall-range-sum-error-m must be non-negative")
    if args.arena_min_wall_points < 1:
        parser.error("--arena-min-wall-points must be >= 1")
    if (args.arena_force_short_wall_side is None) != (
        args.arena_force_short_wall_type is None
    ):
        parser.error(
            "--arena-force-short-wall-side and --arena-force-short-wall-type "
            "must be provided together"
        )
    if args.min_waypoint_spacing_m < 0.0:
        parser.error("--min-waypoint-spacing-m must be non-negative")


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        waypoints = load_waypoints(args.waypoints)
        staging_goal = staging_goal_from_waypoints(waypoints)
    except Exception as exc:
        print(f"two_stage_waypoint_run.py: error: {exc}", file=sys.stderr)
        return 2

    follower_command = build_follower_command(args)
    if args.dry_run:
        print_dry_run(args, waypoints, staging_goal, follower_command)
        return 0

    if not require_motion_confirmation(args, staging_goal, follower_command):
        print("Arena-prior two-stage waypoint run cancelled.")
        return 130

    from . import ros_runtime

    if ros_runtime.rclpy is None:
        print("ROS 2 Python modules are unavailable. Source ROS 2 Humble first.", file=sys.stderr)
        return 2

    diagnostics = RunDiagnostics(
        timestamp=timestamp_now(),
        start_wall_time=timestamp_now(),
        follower_command=shlex.join(follower_command),
        notes=args.notes,
    )
    start_monotonic = time.time()
    node = None
    return_code = 1

    try:
        ros_runtime.rclpy.init()
        node = ros_runtime.TwoStageCoordinator(args)
        node.preflight_before_motion()

        phase_start = time.time()
        arena_result = node.perform_arena_active_spin()
        diagnostics.arena_spin_duration_sec = time.time() - phase_start
        if not arena_result.success:
            raise RuntimeError(
                "arena-prior localization failed: "
                f"{arena_result.failure_reason}"
            )

        phase_start = time.time()
        node.publish_arena_active_initial_pose(
            arena_result.pose_prior,
            arena_result,
        )
        stability = node.wait_for_amcl_validation(
            args.arena_active_validation_timeout_sec,
            min_received_sec=phase_start,
            min_settle_sec=args.amcl_settle_min_sec,
        )
        diagnostics.amcl_var_x = stability.cov_x
        diagnostics.amcl_var_y = stability.cov_y
        diagnostics.amcl_var_yaw_rad2 = stability.cov_yaw_rad2
        diagnostics.stable_samples = stability.stable_count
        diagnostics.max_pose_jump_m = stability.max_pose_jump_m
        diagnostics.max_yaw_jump_deg = stability.max_yaw_jump_deg

        post_amcl_pose, frame = node.validate_post_localization_tf()
        diagnostics.selected_base_frame = frame
        node.validate_post_amcl_pose_prior(
            arena_result.pose_prior,
            post_amcl_pose,
            arena_result,
            frame,
        )

        phase_start = time.time()
        diagnostics.nav2_result_status = node.navigate_to_staging(staging_goal)
        diagnostics.nav2_duration_sec = time.time() - phase_start

        arrival = node.verify_arrival(staging_goal, waypoints)
        diagnostics.selected_base_frame = arrival.base_frame
        diagnostics.tf_arrival_x = arrival.pose.x
        diagnostics.tf_arrival_y = arrival.pose.y
        diagnostics.tf_arrival_yaw_deg = arrival.pose.yaw_deg
        diagnostics.arrival_position_error_m = arrival.position_error_m
        diagnostics.arrival_yaw_error_deg = arrival.yaw_error_deg
        node.stop_repeatedly()

        phase_start = time.time()
        follower_result = run_follower_command(follower_command)
        diagnostics.follower_duration_sec = time.time() - phase_start
        diagnostics.follower_return_code = follower_result.returncode
        if follower_result.returncode != 0:
            raise RuntimeError(f"Follower exited with return code {follower_result.returncode}")

        final_pose, _frame = node.lookup_pose(description="final TF")
        diagnostics.final_tf_x = final_pose.x
        diagnostics.final_tf_y = final_pose.y
        diagnostics.final_tf_yaw_deg = final_pose.yaw_deg
        diagnostics.status = "completed"
        diagnostics.final_status_reason = "completed"
        return_code = 0

    except KeyboardInterrupt:
        diagnostics.status = "interrupted"
        diagnostics.final_status_reason = "keyboard_interrupt"
        print("Interrupted. Cancelling navigation and sending stop...")
        if node is not None:
            cleanup_motion(node)
        return_code = 130

    except Exception as exc:
        diagnostics.status = "failed"
        diagnostics.final_status_reason = str(exc)
        print(f"two_stage_waypoint_run.py: error: {exc}", file=sys.stderr)
        if node is not None:
            cleanup_motion(node)
        return_code = 1

    finally:
        diagnostics.end_wall_time = timestamp_now()
        diagnostics.duration_sec = time.time() - start_monotonic
        if node is not None:
            try:
                final_pose, _frame = node.lookup_pose(description="final TF logging")
                diagnostics.final_tf_x = final_pose.x
                diagnostics.final_tf_y = final_pose.y
                diagnostics.final_tf_yaw_deg = final_pose.yaw_deg
                diagnostics.selected_base_frame = node.selected_base_frame
            except Exception:
                pass
            try:
                node.destroy_node()
            finally:
                ros_runtime.rclpy.shutdown()
        if not args.no_log:
            try:
                append_csv_row(
                    args.results_csv,
                    CSV_HEADER,
                    build_log_row(args, staging_goal, diagnostics),
                )
            except Exception as log_exc:
                print(f"Could not write two-stage run log: {log_exc}", file=sys.stderr)

    return return_code
