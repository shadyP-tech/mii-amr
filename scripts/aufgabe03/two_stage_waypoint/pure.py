import csv
import math
from datetime import datetime
from pathlib import Path

from .model import (
    DEFAULT_SPIN_MIN_SCAN_RANGE_M,
    DEFAULT_SPIN_MIN_VALID_SCAN_COUNT,
    GOAL_STATUS_NAMES,
    MIN_ARENA_ACTIVE_VAR_XY,
    MIN_ARENA_ACTIVE_VAR_YAW_RAD2,
    AmclCovariance,
    PreflightRequirements,
    ScanSafety,
    StagingGoal,
    StabilityState,
    Waypoint,
)


def timestamp_now():
    return datetime.now().isoformat(timespec="seconds")


def empty_if_none(value):
    return "" if value is None else value


def shortest_angle_delta_deg(start_deg, end_deg):
    return (end_deg - start_deg + 180.0) % 360.0 - 180.0


def yaw_to_quaternion_values(yaw_deg):
    half = math.radians(yaw_deg) / 2.0
    return 0.0, 0.0, math.sin(half), math.cos(half)


def quaternion_to_yaw_deg(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def heading_between(a, b):
    return math.degrees(math.atan2(b.y - a.y, b.x - a.x))


def load_waypoints(path):
    path = Path(path)
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        required = {"index", "world_x_m", "world_y_m"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required column(s): {', '.join(sorted(missing))}")

        waypoints = []
        for row in reader:
            waypoint = Waypoint(
                index=int(row["index"]),
                x=float(row["world_x_m"]),
                y=float(row["world_y_m"]),
            )
            if waypoints and waypoint.x == waypoints[-1].x and waypoint.y == waypoints[-1].y:
                continue
            waypoints.append(waypoint)

    if len(waypoints) < 2:
        raise ValueError(f"{path} needs at least two waypoints for two-stage mode")
    return waypoints


def staging_goal_from_waypoints(waypoints):
    if len(waypoints) < 2:
        raise ValueError("two-stage mode needs at least two waypoints")
    return StagingGoal(waypoint=waypoints[0], yaw_deg=heading_between(waypoints[0], waypoints[1]))


def valid_scan_ranges(ranges, range_min=None, range_max=None):
    valid = []
    for value in ranges:
        if value is None or not math.isfinite(value):
            continue
        if range_min is not None and value < range_min:
            continue
        if range_max is not None and value > range_max:
            continue
        valid.append(float(value))
    return valid


def evaluate_spin_scan_safety(
    ranges,
    range_min=None,
    range_max=None,
    min_scan_range_m=DEFAULT_SPIN_MIN_SCAN_RANGE_M,
    min_valid_scan_count=DEFAULT_SPIN_MIN_VALID_SCAN_COUNT,
):
    valid = valid_scan_ranges(ranges, range_min=range_min, range_max=range_max)
    if len(valid) < min_valid_scan_count:
        return ScanSafety(False, "insufficient_valid_scan", len(valid), None)
    minimum = min(valid)
    if minimum < min_scan_range_m:
        return ScanSafety(False, "unsafe_proximity", len(valid), minimum)
    return ScanSafety(True, "ok", len(valid), minimum)


def amcl_covariances(covariance):
    if len(covariance) < 36:
        raise ValueError("AMCL covariance must have 36 entries")
    return AmclCovariance(
        x=float(covariance[0]),
        y=float(covariance[7]),
        yaw_rad2=float(covariance[35]),
    )


def pose_distance_m(a, b):
    return math.hypot(b.x - a.x, b.y - a.y)


def update_amcl_stability(
    state,
    pose,
    covariance,
    max_var_x,
    max_var_y,
    max_var_yaw_rad2,
    max_pose_jump_m,
    max_yaw_jump_deg,
):
    cov = amcl_covariances(covariance)
    samples_seen = state.samples_seen + 1
    if cov.x > max_var_x or cov.y > max_var_y or cov.yaw_rad2 > max_var_yaw_rad2:
        return StabilityState(
            stable_count=0,
            previous_pose=pose,
            max_pose_jump_m=state.max_pose_jump_m,
            max_yaw_jump_deg=state.max_yaw_jump_deg,
            cov_x=cov.x,
            cov_y=cov.y,
            cov_yaw_rad2=cov.yaw_rad2,
            samples_seen=samples_seen,
            reason="covariance_above_threshold",
        )

    pose_jump = 0.0
    yaw_jump = 0.0
    stable_count = state.stable_count + 1
    reason = "stable"
    if state.previous_pose is not None:
        pose_jump = pose_distance_m(state.previous_pose, pose)
        yaw_jump = abs(shortest_angle_delta_deg(state.previous_pose.yaw_deg, pose.yaw_deg))
        if pose_jump > max_pose_jump_m or yaw_jump > max_yaw_jump_deg:
            stable_count = 1
            reason = "pose_jump_above_threshold"

    return StabilityState(
        stable_count=stable_count,
        previous_pose=pose,
        max_pose_jump_m=max(state.max_pose_jump_m, pose_jump),
        max_yaw_jump_deg=max(state.max_yaw_jump_deg, yaw_jump),
        cov_x=cov.x,
        cov_y=cov.y,
        cov_yaw_rad2=cov.yaw_rad2,
        samples_seen=samples_seen,
        reason=reason,
    )


def amcl_validation_timed_out(start_sec, now_sec, timeout_sec):
    return now_sec - start_sec > timeout_sec


def required_preflight_interfaces(args):
    services = []
    if args.localization_mode == "global":
        services.append(args.global_localization_service)
    if (
        args.localization_mode == "arena-active"
        and args.arena_active_on_failure == "global"
        and not args.arena_active_dry_run
    ):
        services.append(args.global_localization_service)
    actions = []
    if not (args.localization_mode == "arena-active" and args.arena_active_dry_run):
        actions.append(args.navigate_action)
    return PreflightRequirements(
        services=services,
        actions=actions,
        topics=[args.scan_topic],
        requires_tf_before_localization=False,
    )


def arena_active_diagnostics_path(args):
    if args.arena_active_diagnostics_json is not None:
        return args.arena_active_diagnostics_json
    return Path("results/aufgabe03") / f"{args.run_id}_arena_active_result.json"


def validate_pose_prior_for_initialpose(pose_prior):
    if pose_prior is None:
        raise RuntimeError("Arena-active localizer did not return a pose prior")
    if not (
        math.isfinite(pose_prior.x_m)
        and math.isfinite(pose_prior.y_m)
        and math.isfinite(pose_prior.yaw_rad)
    ):
        raise RuntimeError("Arena-active pose prior contains non-finite pose values")
    if len(pose_prior.covariance) < 36:
        raise RuntimeError("Arena-active pose prior covariance must have 36 entries")
    var_x = float(pose_prior.covariance[0])
    var_y = float(pose_prior.covariance[7])
    var_yaw = float(pose_prior.covariance[35])
    for name, value in [
        ("x", var_x),
        ("y", var_y),
        ("yaw", var_yaw),
    ]:
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"Arena-active pose prior has invalid {name} covariance")
    return (
        max(var_x, MIN_ARENA_ACTIVE_VAR_XY),
        max(var_y, MIN_ARENA_ACTIVE_VAR_XY),
        max(var_yaw, MIN_ARENA_ACTIVE_VAR_YAW_RAD2),
    )


def build_follower_command(args):
    return [
        str(args.python_executable),
        str(args.follower_script),
        "--waypoints",
        str(args.waypoints),
        "--run-id",
        str(args.run_id),
        "--yes",
        "--start-selection",
        "path-progress",
        "--map-frame",
        args.map_frame,
        "--base-frame",
        args.base_frame,
        "--fallback-base-frame",
        args.fallback_base_frame,
        "--max-pose-age-sec",
        str(args.max_pose_age_sec),
        "--max-scan-age-sec",
        str(args.max_scan_age_sec),
        "--max-amcl-age-sec",
        str(args.max_amcl_age_sec),
        "--max-amcl-var-x",
        str(args.max_amcl_var_x),
        "--max-amcl-var-y",
        str(args.max_amcl_var_y),
        "--max-amcl-var-yaw",
        str(args.max_amcl_var_yaw_rad2),
        "--waypoint-tolerance-m",
        str(args.waypoint_tolerance_m),
        "--goal-tolerance-m",
        str(args.goal_tolerance_m),
        "--min-waypoint-spacing-m",
        str(args.min_waypoint_spacing_m),
        "--notes",
        f"{args.notes};two_stage_handoff",
    ]


def goal_status_name(status):
    return GOAL_STATUS_NAMES.get(int(status), f"STATUS_{status}")
