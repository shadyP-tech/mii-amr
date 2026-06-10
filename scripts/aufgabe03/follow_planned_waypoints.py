#!/usr/bin/env python3
"""
Follow simplified A* waypoints with a conservative TF-based controller.

This script executes a static waypoint CSV in the map frame. It assumes Nav2
localization/AMCL is already running, but it publishes /cmd_vel itself.
"""

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
    from rclpy.time import Time
    from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, Twist
    from nav_msgs.msg import Path as NavPath
    from sensor_msgs.msg import LaserScan
    from visualization_msgs.msg import Marker, MarkerArray
    import tf2_ros
except ImportError:
    rclpy = None
    Node = object
    DurabilityPolicy = None
    QoSProfile = None
    qos_profile_sensor_data = None
    Time = None
    Point = None
    PoseStamped = None
    PoseWithCovarianceStamped = object
    NavPath = None
    Twist = None
    LaserScan = object
    Marker = None
    MarkerArray = None
    tf2_ros = None

import lidar_obstacle_map
import map_path_planner
import replan_runtime
from waypoint_following.command_smoothing import (  # noqa: E402
    COMMAND_SMOOTHING_MODES,
    COMMAND_SMOOTHING_OFF,
    COMMAND_SMOOTHING_RATE_LIMIT,
    CommandSmoother,
    CommandSmoothingConfig,
)
from waypoint_following.controllers import (  # noqa: E402
    PathController,
    PurePursuitController,
    StopGoController,
    build_path_controller,
    should_rotate,
    velocity_command,
)
from waypoint_following.math_utils import (  # noqa: E402
    clamp,
    normalize_angle_rad,
    quaternion_to_yaw_deg,
    shortest_angle_delta_deg,
)
from waypoint_following.models import (  # noqa: E402
    AmclHealth,
    ControllerStep,
    Pose2D,
    RouteState,
    ScanSafety,
    StartSelection,
    TargetState,
    TrackingPathValidation,
    TwistCommand,
    Waypoint,
)
from waypoint_following.path_curves import (  # noqa: E402
    polyline_lookahead_target,
    pure_pursuit_curve_command,
    select_curve_lookahead_target,
)
from waypoint_following.path_progress import (  # noqa: E402
    distance_point_to_segment_m,
    downsample_waypoints,
    heading_between,
    is_heading_change,
    load_tracking_path_csv,
    load_waypoints,
    nearest_path_segment,
    prepare_executable_waypoints,
    select_executable_waypoints,
    select_path_progress_waypoints,
    target_state,
    validate_tracking_path_geometry,
    validate_tracking_point_structure,
    waypoint_distance,
    waypoint_reached,
)
from waypoint_following.lookahead_guard import (  # noqa: E402
    LOOKAHEAD_GUARD_MODES,
    LOOKAHEAD_GUARD_OFF,
    LOOKAHEAD_GUARD_STATIC_AND_RUN_LOCAL,
    LOOKAHEAD_GUARD_STATIC_MAP,
    LookaheadGuard,
    dense_route_signature,
    guard_block_signature,
    static_inflated_blocked_cells,
)
from waypoint_following.replanning import ReplanManager  # noqa: E402
from waypoint_following.run_logging import (  # noqa: E402
    BASE_CSV_HEADER,
    CSV_HEADER,
    RuntimeDiagnostics,
    append_csv_row,
    build_log_row,
    migrate_csv_header,
    pose_fields,
)
from waypoint_following.velocity_scheduler import (  # noqa: E402
    SCHEDULER_STATUS_DEADBAND,
    SPEED_PROFILE_CURVATURE_AWARE,
    SPEED_PROFILE_FIXED,
    SPEED_PROFILE_MODES,
    PurePursuitGeometry,
    PurePursuitVelocityConfig,
    PurePursuitVelocityScheduler,
    VelocityScheduleResult,
    pure_pursuit_geometry,
)
from waypoint_following.scan_safety import (  # noqa: E402
    FORWARD_SOFT_STOP_MIN_CLOSE_RANGES,
    evaluate_scan_safety,
    percentile,
    valid_scan_ranges,
)



DEFAULT_WAYPOINTS_CSV = Path("results/aufgabe03/aufgabe03_waypoints.csv")
DEFAULT_RESULTS_CSV = Path("results/aufgabe03/aufgabe03_waypoint_follow_runs.csv")
DEFAULT_STATIC_MAP = Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml")
DEFAULT_REPLAN_OUTPUT_DIR = Path("results/aufgabe03")
DEFAULT_RVIZ_PATH_TOPIC = "/mii_amr/planned_path"
DEFAULT_RVIZ_WAYPOINT_MARKER_TOPIC = "/mii_amr/planned_waypoints"
DEFAULT_RVIZ_OBSTACLE_MARKER_TOPIC = "/mii_amr/run_local_obstacles"

DEFAULT_LINEAR_SPEED_MPS = 0.04
DEFAULT_PURE_PURSUIT_LINEAR_SPEED_MPS = 0.06
DEFAULT_MIN_LINEAR_SPEED_MPS = 0.012
DEFAULT_LINEAR_GAIN = 0.25
DEFAULT_MAX_ANGULAR_SPEED_RADPS = 0.09
DEFAULT_PURE_PURSUIT_MAX_ANGULAR_SPEED_RADPS = 0.20
DEFAULT_YAW_GAIN = 0.35
DEFAULT_WAYPOINT_TOLERANCE_M = 0.12
DEFAULT_GOAL_TOLERANCE_M = 0.12
DEFAULT_ROTATE_START_HEADING_ERROR_DEG = 20.0
DEFAULT_ROTATE_STOP_HEADING_ERROR_DEG = 7.0
DEFAULT_FORWARD_YAW_DEADBAND_DEG = 3.0
DEFAULT_FORWARD_STOP_HEADING_ERROR_DEG = 18.0
DEFAULT_MIN_WAYPOINT_SPACING_M = 0.12
DEFAULT_START_SELECTION = "path-progress"
DEFAULT_START_ON_PATH_TOLERANCE_M = 0.25
DEFAULT_CONTROLLER = "stop-go"
DEFAULT_PATH_LOOKAHEAD_M = 0.18
DEFAULT_PURE_PURSUIT_LOOKAHEAD_GUARD = LOOKAHEAD_GUARD_OFF
DEFAULT_PURE_PURSUIT_MIN_GUARDED_LOOKAHEAD_M = 0.12
DEFAULT_PURE_PURSUIT_LOOKAHEAD_GUARD_STATIC_INFLATION_RADIUS_M = (
    map_path_planner.DEFAULT_INFLATE_RADIUS_M
)
DEFAULT_PURE_PURSUIT_COMMAND_SMOOTHING = COMMAND_SMOOTHING_RATE_LIMIT
DEFAULT_PURE_PURSUIT_MAX_LINEAR_ACCEL_MPS2 = 0.06
DEFAULT_PURE_PURSUIT_MAX_LINEAR_DECEL_MPS2 = 0.12
DEFAULT_PURE_PURSUIT_MAX_ANGULAR_ACCEL_RADPS2 = 0.18
DEFAULT_PURE_PURSUIT_MAX_ANGULAR_DECEL_RADPS2 = 0.36
DEFAULT_PURE_PURSUIT_FINAL_DECEL_DISTANCE_M = 0.30
DEFAULT_PURE_PURSUIT_SPEED_PROFILE = SPEED_PROFILE_FIXED
DEFAULT_PURE_PURSUIT_MAX_LATERAL_ACCEL_MPS2 = 0.04
DEFAULT_PURE_PURSUIT_TURN_SPEED_MARGIN = 0.85
DEFAULT_PURE_PURSUIT_ROTATE_START_HEADING_ERROR_DEG = 75.0
DEFAULT_PURE_PURSUIT_ROTATE_STOP_HEADING_ERROR_DEG = 35.0
DEFAULT_PURE_PURSUIT_HEADING_DEADBAND_DEG = 4.0
DEFAULT_PURE_PURSUIT_LATERAL_DEADBAND_M = 0.03
DEFAULT_PURE_PURSUIT_CURVATURE_LIMIT_START_HEADING_ERROR_DEG = 12.0
DEFAULT_PURE_PURSUIT_CURVATURE_LIMIT_FULL_HEADING_ERROR_DEG = 30.0
DEFAULT_TRACKING_ENDPOINT_TOLERANCE_M = 0.10
DEFAULT_TRACKING_START_TOLERANCE_M = 0.20
DEFAULT_TRACKING_MAX_SEGMENT_M = 0.30
DEFAULT_ODOM_FRAME = "odom"
DEFAULT_SCAN_HALF_ANGLE_DEG = 35.0
DEFAULT_HARD_STOP_RANGE_M = 0.16
DEFAULT_MIN_SCAN_RANGE_M = 0.40
DEFAULT_ROTATION_STOP_RANGE_M = 0.18
FORWARD_SOFT_STOP_MIN_CLOSE_RANGES = 2
DEFAULT_MAX_POSE_AGE_SEC = 10.0
DEFAULT_MAX_SCAN_AGE_SEC = 8.0
DEFAULT_MAX_AMCL_AGE_SEC = 15.0
DEFAULT_MAX_AMCL_VAR_X = 0.05
DEFAULT_MAX_AMCL_VAR_Y = 0.05
DEFAULT_MAX_AMCL_VAR_YAW = 0.10
DEFAULT_MAX_WAYPOINT_TIME_SEC = 180.0
DEFAULT_MAX_TF_UPDATE_GAP_SEC = 5.0
DEFAULT_TF_RECOVERY_TIME_SEC = 5.0
DEFAULT_LOCALIZATION_RECOVERY_TIME_SEC = 5.0
DEFAULT_CONTROL_RATE_HZ = 20.0
DEFAULT_SETTLE_SEC = 0.5
DEFAULT_REPLAN_TIMEOUT_SEC = 5.0
DEFAULT_MAX_REPLAN_SCAN_AGE_SEC = 1.0
DEFAULT_MAX_REPLAN_TF_AGE_SEC = 1.0
DEFAULT_OBSTACLE_FORWARD_DISTANCE_M = 0.75
DEFAULT_OBSTACLE_FORWARD_HALF_WIDTH_M = 0.25
DEFAULT_OBSTACLE_ANGLE_WINDOW_DEG = 45.0
DEFAULT_OBSTACLE_MIN_RANGE_M = 0.12
DEFAULT_ROBOT_FOOTPRINT_RADIUS_M = 0.10
DEFAULT_OBSTACLE_MIN_CLUSTER_SIZE = 3
DEFAULT_OBSTACLE_MIN_CLUSTER_WIDTH_M = 0.05
DEFAULT_OBSTACLE_INFLATE_RADIUS_M = 0.15
DEFAULT_MAX_START_SNAP_M = 0.20
DEFAULT_MAX_GOAL_SNAP_M = 0.30
DEFAULT_MAX_REPLAN_PATH_LENGTH_RATIO = 3.0
DEFAULT_RUN_LOCAL_MAP_INITIAL_SCAN_MODE = "forward"
DEFAULT_RUN_LOCAL_MAP_INITIAL_SCAN_COUNT = 5
DEFAULT_RUN_LOCAL_MAP_UPDATE_MODE = "forward"
DEFAULT_RUN_LOCAL_MAP_MIN_HIT_COUNT = 2
DEFAULT_RUN_LOCAL_MAP_INFLATION_RADIUS_M = DEFAULT_OBSTACLE_INFLATE_RADIUS_M
DEFAULT_RUN_LOCAL_MAP_MAX_TF_AGE_SEC = 1.0
DEFAULT_RUN_LOCAL_MAP_MAX_SCAN_AGE_SEC = 1.0
DEFAULT_RUN_LOCAL_MAP_MIN_USED_POINTS = 3
DEFAULT_RUN_LOCAL_MAP_MAX_REJECTED_RATIO = 0.90
DEFAULT_RUN_LOCAL_MAP_CORRIDOR_CHECK_DISTANCE_M = 0.75
DEFAULT_RUN_LOCAL_MAP_CLEARANCE_MARGIN_M = 0.04
DEFAULT_RUN_LOCAL_MAP_MAX_UPDATES = 3
DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_COUNT = 2
DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_FORWARD_HALF_WIDTH_M = 0.40
DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_ANGLE_WINDOW_DEG = 75.0
DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_FORWARD_DISTANCE_M = 1.00
DEFAULT_RUN_LOCAL_MAP_PRUNE_BEHIND_DISTANCE_M = 0.20

INITIAL_RUN_LOCAL_MAP_NONFATAL_REASONS = {
    lidar_obstacle_map.RUN_LOCAL_FAILURE_TOO_FEW_SCAN_POINTS,
    lidar_obstacle_map.RUN_LOCAL_FAILURE_TOO_MANY_REJECTED_POINTS,
}

REPLAN_TRIGGER_SCAN_BLOCKAGE = "scan_blockage"
REPLAN_TRIGGER_KNOWN_CORRIDOR = "known_corridor"
REPLAN_TRIGGER_LOOKAHEAD_GUARD = "lookahead_guard"


def run_local_map_has_confirmed_obstacles(run_local_map):
    if run_local_map is None:
        return False
    return bool(getattr(run_local_map, "confirmed_raw_cells", None))


def lidar_replan_failure(reason):
    message = str(reason)
    if message.startswith("lidar_replan_failed:"):
        return RuntimeError(message)
    return RuntimeError(f"lidar_replan_failed:{message}")


def warn_logger(logger, message):
    if logger is None:
        return
    warn = getattr(logger, "warn", None)
    if warn is not None:
        warn(message)


def build_sparse_tracking_validation(source, point_count, status):
    return TrackingPathValidation(
        source=source,
        point_count=point_count,
        validation_status=status,
    )


def prepare_tracking_setup(
    args,
    route_waypoints,
    current_pose=None,
    logger=None,
    structural_only=False,
):
    route_waypoints = list(route_waypoints)
    if getattr(args, "controller", DEFAULT_CONTROLLER) != "pure-pursuit":
        return None, build_sparse_tracking_validation(
            source="ignored_stop_go",
            point_count=0,
            status="ignored",
        )

    if not getattr(args, "tracking_path_csv", None):
        message = (
            "Pure-pursuit has no --tracking-path-csv; "
            "falling back to sparse waypoint geometry."
        )
        warn_logger(logger, message)
        return None, build_sparse_tracking_validation(
            source="waypoints",
            point_count=len(route_waypoints),
            status="fallback_sparse_waypoints",
        )

    tracking_points, warnings = load_tracking_path_csv(
        args.tracking_path_csv,
        max_segment_m=args.tracking_max_segment_m,
    )
    for warning in warnings:
        warn_logger(logger, warning)
    if structural_only:
        return tracking_points, TrackingPathValidation(
            source="csv",
            point_count=len(tracking_points),
            validation_status="structural_ok",
            warnings=tuple(warnings),
        )
    validation = validate_tracking_path_geometry(
        route_waypoints,
        tracking_points,
        endpoint_tolerance_m=args.tracking_endpoint_tolerance_m,
        start_tolerance_m=args.tracking_start_tolerance_m,
        allow_mismatch=args.allow_tracking_path_mismatch,
        current_pose=current_pose,
        source="csv",
        structural_warnings=warnings,
    )
    for warning in validation.warnings:
        warn_logger(logger, warning)
    return tracking_points, validation


def format_optional_m(value):
    return "n/a" if value is None else f"{value:.3f}"


def notes_with_tracking_metadata(notes, args, tracking_validation):
    if (
        getattr(args, "controller", DEFAULT_CONTROLLER) != "pure-pursuit"
        or tracking_validation is None
    ):
        return notes
    return (
        f"{notes};controller={args.controller};"
        f"tracking_source={tracking_validation.source};"
        f"tracking_point_count={tracking_validation.point_count};"
        f"tracking_validation_status={tracking_validation.validation_status}"
    )


def build_lookahead_guard(args, run_local_map_fn=None):
    if getattr(args, "controller", DEFAULT_CONTROLLER) != "pure-pursuit":
        return None
    guard_mode = getattr(
        args,
        "pure_pursuit_lookahead_guard",
        DEFAULT_PURE_PURSUIT_LOOKAHEAD_GUARD,
    )
    if guard_mode == LOOKAHEAD_GUARD_OFF:
        return None
    return LookaheadGuard.from_static_map(
        args.static_map,
        args.pure_pursuit_lookahead_guard_static_inflation_radius_m,
        mode=guard_mode,
        run_local_map_fn=run_local_map_fn,
    )


def command_smoothing_active(args):
    return (
        getattr(args, "controller", DEFAULT_CONTROLLER) == "pure-pursuit"
        and getattr(
            args,
            "pure_pursuit_command_smoothing",
            DEFAULT_PURE_PURSUIT_COMMAND_SMOOTHING,
        )
        == COMMAND_SMOOTHING_RATE_LIMIT
    )


def build_command_smoother(args):
    if not command_smoothing_active(args):
        return None
    return CommandSmoother(
        CommandSmoothingConfig(
            max_linear_accel_mps2=args.pure_pursuit_max_linear_accel_mps2,
            max_linear_decel_mps2=args.pure_pursuit_max_linear_decel_mps2,
            max_angular_accel_radps2=args.pure_pursuit_max_angular_accel_radps2,
            max_angular_decel_radps2=args.pure_pursuit_max_angular_decel_radps2,
            final_decel_distance_m=args.pure_pursuit_final_decel_distance_m,
            min_smoothed_linear_speed_mps=(
                args.pure_pursuit_min_smoothed_linear_speed_mps
            ),
        )
    )


def reset_command_smoother(node):
    smoother = getattr(node, "command_smoother", None)
    if smoother is not None:
        smoother.reset()
    if hasattr(node, "last_smoothed_command_time_sec"):
        node.last_smoothed_command_time_sec = None
    if hasattr(node, "last_velocity_scheduler_status"):
        node.last_velocity_scheduler_status = None
    if hasattr(node, "last_velocity_scheduler_log_sec"):
        node.last_velocity_scheduler_log_sec = None


def smoothing_dt_sec(node, now_sec):
    args = node.args
    default_dt = 1.0 / args.control_rate_hz
    max_dt = 2.0 / args.control_rate_hz
    previous_sec = getattr(node, "last_smoothed_command_time_sec", None)
    if previous_sec is None:
        return default_dt
    dt_sec = now_sec - previous_sec
    if not math.isfinite(dt_sec):
        return default_dt
    return clamp(dt_sec, 0.0, max_dt)


def smoothed_step_command(node, step, now_sec):
    smoother = getattr(node, "command_smoother", None)
    if smoother is None:
        return step.command
    if step.command.linear_x == 0.0 and step.command.angular_z == 0.0:
        reset_command_smoother(node)
        return step.command
    dt_sec = smoothing_dt_sec(node, now_sec)
    command = smoother.apply(
        step.command,
        dt_sec,
        step.distance_m,
        node.args.pure_pursuit_goal_tolerance_m,
    )
    node.last_smoothed_command_time_sec = now_sec
    return command


def notes_with_smoothing_metadata(notes, args):
    if not command_smoothing_active(args):
        return notes
    return (
        f"{notes};pure_pursuit_command_smoothing="
        f"{args.pure_pursuit_command_smoothing};"
        "pure_pursuit_max_linear_accel_mps2="
        f"{args.pure_pursuit_max_linear_accel_mps2:.3f};"
        "pure_pursuit_max_linear_decel_mps2="
        f"{args.pure_pursuit_max_linear_decel_mps2:.3f};"
        "pure_pursuit_max_angular_accel_radps2="
        f"{args.pure_pursuit_max_angular_accel_radps2:.3f};"
        "pure_pursuit_max_angular_decel_radps2="
        f"{args.pure_pursuit_max_angular_decel_radps2:.3f};"
        "pure_pursuit_final_decel_distance_m="
        f"{args.pure_pursuit_final_decel_distance_m:.3f};"
        "pure_pursuit_min_smoothed_linear_speed_mps="
        f"{args.pure_pursuit_min_smoothed_linear_speed_mps:.3f}"
    )


def notes_with_velocity_scheduler_metadata(notes, args):
    if getattr(args, "controller", DEFAULT_CONTROLLER) != "pure-pursuit":
        return notes
    return (
        f"{notes};pure_pursuit_speed_profile="
        f"{args.pure_pursuit_speed_profile};"
        "pure_pursuit_default_linear_speed_resolved_mps="
        f"{args.linear_speed:.3f};"
        "pure_pursuit_default_max_angular_speed_resolved_radps="
        f"{args.max_angular_speed:.3f};"
        "pure_pursuit_max_lateral_accel_mps2="
        f"{args.pure_pursuit_max_lateral_accel_mps2:.3f};"
        "pure_pursuit_turn_speed_margin="
        f"{args.pure_pursuit_turn_speed_margin:.3f};"
        "pure_pursuit_heading_deadband_deg="
        f"{args.pure_pursuit_heading_deadband_deg:.3f};"
        "pure_pursuit_lateral_deadband_m="
        f"{args.pure_pursuit_lateral_deadband_m:.3f};"
        "pure_pursuit_curvature_limit_start_heading_error_deg="
        f"{args.pure_pursuit_curvature_limit_start_heading_error_deg:.3f};"
        "pure_pursuit_curvature_limit_full_heading_error_deg="
        f"{args.pure_pursuit_curvature_limit_full_heading_error_deg:.3f};"
        "pure_pursuit_rotate_start_heading_error_deg="
        f"{args.pure_pursuit_rotate_start_heading_error_deg:.3f};"
        "pure_pursuit_rotate_stop_heading_error_deg="
        f"{args.pure_pursuit_rotate_stop_heading_error_deg:.3f}"
    )


def notes_with_guard_metadata(notes, args, guard_result):
    if (
        getattr(args, "controller", DEFAULT_CONTROLLER) != "pure-pursuit"
        or getattr(args, "pure_pursuit_lookahead_guard", LOOKAHEAD_GUARD_OFF)
        == LOOKAHEAD_GUARD_OFF
        or guard_result is None
    ):
        return notes
    return (
        f"{notes};lookahead_guard={args.pure_pursuit_lookahead_guard};"
        f"lookahead_guard_status={guard_result.status};"
        "lookahead_guard_selected_distance_m="
        f"{format_optional_m(guard_result.selected_target_distance_m)};"
        f"lookahead_guard_blocked_cell_count={guard_result.blocked_cell_count}"
    )


DEFAULT_STARTUP_TIMEOUT_SEC = 20.0
STOP_PUBLISH_COUNT = 10
STOP_PUBLISH_HZ = 10.0

RVIZ_COLOR_PATH = (0.0, 0.55, 1.0, 0.95)
RVIZ_COLOR_WAYPOINT = (0.0, 0.75, 1.0, 0.85)
RVIZ_COLOR_CURRENT = (1.0, 0.82, 0.16, 0.95)
RVIZ_COLOR_GOAL = (0.95, 0.18, 0.14, 0.95)
RVIZ_COLOR_LABEL = (0.95, 0.95, 0.95, 1.0)
RVIZ_COLOR_CONFIRMED_OBSTACLE = (0.05, 0.95, 0.22, 0.85)
RVIZ_COLOR_INFLATED_OBSTACLE = (0.9, 0.25, 1.0, 0.30)
RVIZ_COLOR_BLOCKED_CORRIDOR = (1.0, 0.45, 0.0, 0.80)

def rviz_messages_available():
    return all(
        message_type is not None
        for message_type in (NavPath, PoseStamped, Point, Marker, MarkerArray)
    )


def rviz_qos_profile():
    if QoSProfile is None or DurabilityPolicy is None:
        return 1
    qos = QoSProfile(depth=1)
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    return qos


def set_header(message, frame_id, stamp):
    message.header.frame_id = frame_id
    message.header.stamp = stamp


def set_pose_xy(pose, x, y, z=0.0):
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = 0.0
    pose.orientation.w = 1.0


def point_msg(x, y, z=0.0):
    point = Point()
    point.x = float(x)
    point.y = float(y)
    point.z = float(z)
    return point


def set_marker_color(marker, color):
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]


def marker_delete_all(frame_id, stamp):
    marker = Marker()
    set_header(marker, frame_id, stamp)
    marker.action = Marker.DELETEALL
    return marker


def apply_marker_common(marker, frame_id, stamp, namespace, marker_id, marker_type, color):
    set_header(marker, frame_id, stamp)
    marker.ns = namespace
    marker.id = int(marker_id)
    marker.type = marker_type
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    set_marker_color(marker, color)


def build_pose_stamped(frame_id, stamp, x, y):
    pose = PoseStamped()
    set_header(pose, frame_id, stamp)
    set_pose_xy(pose.pose, x, y)
    return pose


def build_rviz_path_message(waypoints, frame_id, stamp, current_pose=None):
    if NavPath is None or PoseStamped is None:
        raise RuntimeError("ROS nav_msgs/geometry_msgs are unavailable.")
    path = NavPath()
    set_header(path, frame_id, stamp)
    if current_pose is not None:
        path.poses.append(build_pose_stamped(frame_id, stamp, current_pose.x, current_pose.y))
    for waypoint in waypoints:
        path.poses.append(build_pose_stamped(frame_id, stamp, waypoint.x, waypoint.y))
    return path


def waypoint_point(waypoint, z=0.04):
    return point_msg(waypoint.x, waypoint.y, z)


def build_point_layer_marker(
    frame_id,
    stamp,
    namespace,
    marker_id,
    marker_type,
    points,
    color,
    scale_m,
):
    if not points:
        return None
    marker = Marker()
    apply_marker_common(marker, frame_id, stamp, namespace, marker_id, marker_type, color)
    marker.scale.x = scale_m
    marker.scale.y = scale_m
    marker.scale.z = scale_m
    marker.points = list(points)
    return marker


def build_single_waypoint_marker(frame_id, stamp, namespace, marker_id, waypoint, color, scale_m, z):
    marker = Marker()
    apply_marker_common(marker, frame_id, stamp, namespace, marker_id, Marker.SPHERE, color)
    marker.scale.x = scale_m
    marker.scale.y = scale_m
    marker.scale.z = scale_m
    set_pose_xy(marker.pose, waypoint.x, waypoint.y, z)
    return marker


def build_waypoint_label_marker(frame_id, stamp, marker_id, waypoint):
    marker = Marker()
    apply_marker_common(
        marker,
        frame_id,
        stamp,
        "planned_waypoint_labels",
        marker_id,
        Marker.TEXT_VIEW_FACING,
        RVIZ_COLOR_LABEL,
    )
    set_pose_xy(marker.pose, waypoint.x, waypoint.y, 0.20)
    marker.scale.z = 0.08
    marker.text = str(waypoint.index)
    return marker


def build_rviz_waypoint_markers(waypoints, frame_id, stamp, current_waypoint_index=0):
    if Marker is None or MarkerArray is None or Point is None:
        raise RuntimeError("ROS visualization messages are unavailable.")
    waypoints = list(waypoints)
    markers = [marker_delete_all(frame_id, stamp)]
    points = [waypoint_point(waypoint) for waypoint in waypoints]
    waypoint_layer = build_point_layer_marker(
        frame_id,
        stamp,
        "planned_waypoints",
        1,
        Marker.SPHERE_LIST,
        points,
        RVIZ_COLOR_WAYPOINT,
        0.07,
    )
    if waypoint_layer is not None:
        markers.append(waypoint_layer)
    if waypoints:
        current_index = max(0, min(int(current_waypoint_index), len(waypoints) - 1))
        markers.append(
            build_single_waypoint_marker(
                frame_id,
                stamp,
                "current_waypoint",
                2,
                waypoints[current_index],
                RVIZ_COLOR_CURRENT,
                0.14,
                0.08,
            )
        )
        markers.append(
            build_single_waypoint_marker(
                frame_id,
                stamp,
                "goal_waypoint",
                3,
                waypoints[-1],
                RVIZ_COLOR_GOAL,
                0.12,
                0.10,
            )
        )
        for label_index, waypoint in enumerate(waypoints):
            markers.append(
                build_waypoint_label_marker(
                    frame_id,
                    stamp,
                    1000 + label_index,
                    waypoint,
                )
            )
    return MarkerArray(markers=markers)


def build_cell_layer_marker(
    run_local_map,
    frame_id,
    stamp,
    namespace,
    marker_id,
    cells,
    color,
    z,
    height_m,
):
    cells = sorted(cells or ())
    if not cells:
        return None
    metadata = run_local_map.static_map.metadata
    marker = Marker()
    apply_marker_common(marker, frame_id, stamp, namespace, marker_id, Marker.CUBE_LIST, color)
    marker.scale.x = metadata.resolution
    marker.scale.y = metadata.resolution
    marker.scale.z = height_m
    marker.points = [
        point_msg(
            *lidar_obstacle_map.planner.grid_to_world(cell[0], cell[1], metadata),
            z,
        )
        for cell in cells
    ]
    return marker


def append_marker(markers, marker):
    if marker is not None:
        markers.append(marker)


def build_rviz_obstacle_markers(run_local_map, frame_id, stamp, blocked_cells=None):
    if Marker is None or MarkerArray is None or Point is None:
        raise RuntimeError("ROS visualization messages are unavailable.")
    markers = [marker_delete_all(frame_id, stamp)]
    if run_local_map is None:
        return MarkerArray(markers=markers)
    append_marker(
        markers,
        build_cell_layer_marker(
            run_local_map,
            frame_id,
            stamp,
            "run_local_inflated_obstacle_cells",
            1,
            run_local_map.inflated_obstacle_cells,
            RVIZ_COLOR_INFLATED_OBSTACLE,
            0.005,
            0.02,
        ),
    )
    append_marker(
        markers,
        build_cell_layer_marker(
            run_local_map,
            frame_id,
            stamp,
            "run_local_confirmed_obstacle_cells",
            2,
            run_local_map.confirmed_raw_cells,
            RVIZ_COLOR_CONFIRMED_OBSTACLE,
            0.045,
            0.05,
        ),
    )
    append_marker(
        markers,
        build_cell_layer_marker(
            run_local_map,
            frame_id,
            stamp,
            "run_local_blocked_corridor_cells",
            3,
            blocked_cells or set(),
            RVIZ_COLOR_BLOCKED_CORRIDOR,
            0.075,
            0.06,
        ),
    )
    return MarkerArray(markers=markers)


def publish_rviz_route_if_available(
    node,
    waypoints,
    current_pose=None,
    current_waypoint_index=0,
):
    publish = getattr(node, "publish_rviz_route", None)
    if callable(publish):
        publish(
            waypoints,
            current_pose=current_pose,
            current_waypoint_index=current_waypoint_index,
        )


def publish_rviz_obstacles_if_available(node, blocked_cells=None):
    publish = getattr(node, "publish_rviz_obstacles", None)
    if callable(publish):
        publish(blocked_cells=blocked_cells)


def stamp_to_sec(stamp):
    if stamp is None:
        return None
    return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0


def amcl_covariances(covariance):
    return float(covariance[0]), float(covariance[7]), float(covariance[35])


def evaluate_amcl_health(
    covariance,
    age_sec,
    max_age_sec,
    max_var_x,
    max_var_y,
    max_var_yaw,
    fail_on_bad_localization=False,
):
    cov_x, cov_y, cov_yaw = amcl_covariances(covariance)
    warnings = []
    if age_sec is None or age_sec > max_age_sec:
        warnings.append("stale_amcl")
    if cov_x > max_var_x:
        warnings.append("high_cov_x")
    if cov_y > max_var_y:
        warnings.append("high_cov_y")
    if cov_yaw > max_var_yaw:
        warnings.append("high_cov_yaw")
    return AmclHealth(
        ok=not warnings or not fail_on_bad_localization,
        warnings=warnings,
        cov_x=cov_x,
        cov_y=cov_y,
        cov_yaw=cov_yaw,
        age_sec=age_sec,
    )


def age_ok(age_sec, max_age_sec):
    return age_sec is not None and age_sec <= max_age_sec


def ordered_base_frames(base_frame, fallback_base_frame):
    frames = []
    for frame in [base_frame, fallback_base_frame]:
        if frame and frame not in frames:
            frames.append(frame)
    return frames


class WaypointFollower(Node):
    def __init__(self, args):
        if rclpy is None:
            raise RuntimeError(
                "ROS 2 Python modules are unavailable. Source ROS 2 Humble before "
                "running the waypoint follower."
            )
        super().__init__("follow_planned_waypoints")
        self.args = args
        self.last_scan = None
        self.last_scan_received_sec = None
        self.last_amcl = None
        self.last_amcl_received_sec = None
        self.base_frame_used = ""
        self.reached_count = 0
        self.start_pose = None
        self.final_pose = None
        self.last_scan_safety = None
        self.last_amcl_health = None
        self.diagnostics = RuntimeDiagnostics(
            max_tf_update_gap_sec=args.max_tf_update_gap_sec,
        )
        self.last_tf_stamp_sec = None
        self.last_tf_stamp_change_local_sec = None
        self.run_local_map = None
        self.live_replan_attempt_count = 0
        self.known_corridor_repair_count = 0
        self.last_known_corridor_repair_signature = None
        self.suppressed_known_corridor_signature = None
        self.last_scan_block_budget_repair_signature = None
        self.last_lookahead_guard_block_signature = None
        self.last_lookahead_guard_result = None
        self.command_smoother = build_command_smoother(args)
        self.last_smoothed_command_time_sec = None
        self.last_velocity_scheduler_status = None
        self.last_velocity_scheduler_log_sec = None
        self.last_replan_tracking_points = None
        self.last_replan_tracking_source = "waypoints"
        self.last_replan_tracking_validation = None
        self.rviz_last_blocked_cells = set()
        self.replan_manager = ReplanManager(self)
        self.lookahead_guard = build_lookahead_guard(
            args,
            run_local_map_fn=lambda: self.run_local_map,
        )
        if self.lookahead_guard is not None and args.verbose:
            self.get_logger().info(
                "Pure-pursuit lookahead guard enabled: "
                f"mode={args.pure_pursuit_lookahead_guard}, "
                "unknown_cells=blocked, "
                "static_inflation_radius_m="
                f"{args.pure_pursuit_lookahead_guard_static_inflation_radius_m:.3f}, "
                f"static_blocked_cells={len(self.lookahead_guard.static_blocked_cells)}"
            )
        if args.controller == "pure-pursuit" and args.verbose:
            self.get_logger().info(
                "Pure-pursuit speed profile: "
                f"profile={args.pure_pursuit_speed_profile}, "
                f"resolved_linear_speed={args.linear_speed:.3f}, "
                f"resolved_max_angular_speed={args.max_angular_speed:.3f}, "
                f"max_lateral_accel={args.pure_pursuit_max_lateral_accel_mps2:.3f}, "
                f"turn_speed_margin={args.pure_pursuit_turn_speed_margin:.3f}, "
                f"heading_deadband={args.pure_pursuit_heading_deadband_deg:.1f} deg, "
                f"lateral_deadband={args.pure_pursuit_lateral_deadband_m:.3f} m, "
                "curvature_limit_start="
                f"{args.pure_pursuit_curvature_limit_start_heading_error_deg:.1f} deg, "
                "curvature_limit_full="
                f"{args.pure_pursuit_curvature_limit_full_heading_error_deg:.1f} deg, "
                "rotate_start="
                f"{args.pure_pursuit_rotate_start_heading_error_deg:.1f} deg, "
                "rotate_stop="
                f"{args.pure_pursuit_rotate_stop_heading_error_deg:.1f} deg"
            )
        if self.command_smoother is not None and args.verbose:
            self.get_logger().info(
                "Pure-pursuit command smoothing enabled: "
                f"mode={args.pure_pursuit_command_smoothing}, "
                f"linear_accel={args.pure_pursuit_max_linear_accel_mps2:.3f}, "
                f"linear_decel={args.pure_pursuit_max_linear_decel_mps2:.3f}, "
                f"angular_accel={args.pure_pursuit_max_angular_accel_radps2:.3f}, "
                f"angular_decel={args.pure_pursuit_max_angular_decel_radps2:.3f}, "
                f"final_decel_distance={args.pure_pursuit_final_decel_distance_m:.3f}, "
                "dt_clamp=[0, 2/control_rate_hz]"
            )

        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.rviz_path_pub = None
        self.rviz_waypoint_marker_pub = None
        self.rviz_obstacle_marker_pub = None
        if not args.no_rviz_visualization:
            if not rviz_messages_available():
                raise RuntimeError(
                    "ROS RViz message types are unavailable. Source ROS 2 Humble "
                    "before enabling RViz visualization."
                )
            rviz_qos = rviz_qos_profile()
            self.rviz_path_pub = self.create_publisher(
                NavPath,
                args.rviz_path_topic,
                rviz_qos,
            )
            self.rviz_waypoint_marker_pub = self.create_publisher(
                MarkerArray,
                args.rviz_waypoint_marker_topic,
                rviz_qos,
            )
            self.rviz_obstacle_marker_pub = self.create_publisher(
                MarkerArray,
                args.rviz_obstacle_marker_topic,
                rviz_qos,
            )
            if args.verbose:
                self.get_logger().info(
                    "Publishing RViz visualization: "
                    f"path={args.rviz_path_topic}, "
                    f"waypoints={args.rviz_waypoint_marker_topic}, "
                    f"obstacles={args.rviz_obstacle_marker_topic}"
                )
        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self.amcl_callback,
            10,
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        time.sleep(1.0)

    def rviz_visualization_enabled(self):
        return (
            not getattr(self.args, "no_rviz_visualization", False)
            and self.rviz_path_pub is not None
            and self.rviz_waypoint_marker_pub is not None
            and self.rviz_obstacle_marker_pub is not None
        )

    def rviz_stamp(self):
        return self.get_clock().now().to_msg()

    def publish_rviz_route(self, waypoints, current_pose=None, current_waypoint_index=0):
        if not self.rviz_visualization_enabled():
            return
        waypoints = list(waypoints)
        stamp = self.rviz_stamp()
        self.rviz_path_pub.publish(
            build_rviz_path_message(
                waypoints,
                self.args.map_frame,
                stamp,
                current_pose=current_pose,
            )
        )
        self.rviz_waypoint_marker_pub.publish(
            build_rviz_waypoint_markers(
                waypoints,
                self.args.map_frame,
                stamp,
                current_waypoint_index=current_waypoint_index,
            )
        )

    def publish_rviz_obstacles(self, blocked_cells=None):
        if not self.rviz_visualization_enabled():
            return
        if blocked_cells is not None:
            self.rviz_last_blocked_cells = set(blocked_cells)
        self.rviz_obstacle_marker_pub.publish(
            build_rviz_obstacle_markers(
                self.run_local_map,
                self.args.map_frame,
                self.rviz_stamp(),
                blocked_cells=self.rviz_last_blocked_cells,
            )
        )

    def scan_callback(self, msg):
        self.last_scan = msg
        self.last_scan_received_sec = time.time()

    def amcl_callback(self, msg):
        self.last_amcl = msg
        self.last_amcl_received_sec = time.time()

    def publish_velocity(self, linear_x, angular_z):
        if linear_x != 0.0 or angular_z != 0.0:
            self.last_scan_block_budget_repair_signature = None
            self.last_lookahead_guard_block_signature = None
        else:
            reset_command_smoother(self)
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.pub.publish(msg)

    def maybe_log_velocity_scheduler_result(self, result, now_sec):
        if (
            result is None
            or not self.args.verbose
            or self.args.controller != "pure-pursuit"
            or self.args.pure_pursuit_speed_profile != SPEED_PROFILE_CURVATURE_AWARE
            or result.status == SCHEDULER_STATUS_DEADBAND
        ):
            return
        status_changed = result.status != self.last_velocity_scheduler_status
        log_due = (
            self.last_velocity_scheduler_log_sec is None
            or now_sec - self.last_velocity_scheduler_log_sec >= 2.0
        )
        if not status_changed and not log_due:
            return
        self.last_velocity_scheduler_status = result.status
        self.last_velocity_scheduler_log_sec = now_sec
        self.get_logger().info(
            "Pure-pursuit scheduler: "
            f"status={result.status}, "
            f"alpha_deg={result.alpha_deg:.2f}, "
            f"lateral_error_m={result.lateral_error_m:.3f}, "
            f"angular_scale={result.angular_scale:.3f}, "
            f"speed_limit_blend={result.speed_limit_blend:.3f}, "
            f"raw_v={result.raw_linear_x:.3f}, "
            f"scheduled_v={result.scheduled_linear_x:.3f}, "
            f"raw_omega={result.raw_angular_z:.3f}, "
            f"scheduled_omega={result.scheduled_angular_z:.3f}"
        )

    def stop_repeatedly(self):
        reset_command_smoother(self)
        msg = Twist()
        sleep_sec = 1.0 / STOP_PUBLISH_HZ
        for _ in range(STOP_PUBLISH_COUNT):
            if rclpy.ok():
                self.pub.publish(msg)
            self.spin_for(sleep_sec)

    def spin_once(self, timeout_sec):
        rclpy.spin_once(self, timeout_sec=timeout_sec)

    def spin_for(self, duration_sec, step_sec=0.05):
        deadline = time.time() + max(0.0, duration_sec)
        while rclpy.ok() and time.time() < deadline:
            timeout_sec = min(step_sec, max(0.0, deadline - time.time()))
            rclpy.spin_once(self, timeout_sec=timeout_sec)

    def wait_for_startup_gate(self, timeout_sec=None):
        if timeout_sec is None:
            timeout_sec = self.args.startup_timeout_sec
        require_amcl = (
            self.args.require_amcl_startup
            or self.args.fail_on_bad_localization
            or self.args.pause_on_bad_localization
        )
        start = time.time()
        while rclpy.ok():
            have_scan = self.last_scan is not None
            have_amcl = self.last_amcl is not None
            have_tf = False
            try:
                _pose, frame = self.lookup_pose()
                self.base_frame_used = frame
                have_tf = True
            except RuntimeError:
                have_tf = False

            if have_scan and have_tf and (have_amcl or not require_amcl):
                return
            if time.time() - start > timeout_sec:
                missing = []
                if not have_scan:
                    missing.append("/scan")
                if require_amcl and not have_amcl:
                    missing.append("/amcl_pose")
                if not have_tf:
                    missing.append(
                        f"TF {self.args.map_frame}->{self.args.base_frame}/"
                        f"{self.args.fallback_base_frame}"
                    )
                raise RuntimeError(
                    "Timed out waiting for startup data: " + ", ".join(missing)
                )
            rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError("ROS shutdown during startup gate.")

    def lookup_pose(self):
        errors = []
        for frame in ordered_base_frames(self.args.base_frame, self.args.fallback_base_frame):
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.args.map_frame,
                    frame,
                    Time(),
                )
                self.base_frame_used = frame
                return transform_to_pose2d(transform, frame), frame
            except Exception as exc:
                errors.append(f"{frame}: {exc}")
        odom_frame = getattr(self.args, "odom_frame", DEFAULT_ODOM_FRAME)
        for frame in ordered_base_frames(self.args.base_frame, self.args.fallback_base_frame):
            try:
                map_from_odom = self.tf_buffer.lookup_transform(
                    self.args.map_frame,
                    odom_frame,
                    Time(),
                )
                odom_from_base = self.tf_buffer.lookup_transform(
                    odom_frame,
                    frame,
                    Time(),
                )
                self.base_frame_used = frame
                return compose_2d_pose(
                    map_from_odom,
                    odom_from_base,
                    frame,
                ), frame
            except Exception as exc:
                errors.append(f"split {self.args.map_frame}->{odom_frame}->{frame}: {exc}")
        raise RuntimeError("Could not lookup TF pose: " + "; ".join(errors))

    def update_tf_tracking(self, pose):
        if pose.stamp_sec is None:
            return None
        now = time.time()
        if self.last_tf_stamp_sec is None or pose.stamp_sec != self.last_tf_stamp_sec:
            self.last_tf_stamp_sec = pose.stamp_sec
            self.last_tf_stamp_change_local_sec = now
        if self.last_tf_stamp_change_local_sec is None:
            self.last_tf_stamp_change_local_sec = now
        return now - self.last_tf_stamp_change_local_sec

    def reset_tf_tracking(self):
        self.last_tf_stamp_sec = None
        self.last_tf_stamp_change_local_sec = None

    def refresh_after_operator_wait(self, min_scan_stamp_sec, timeout_sec=None):
        self.reset_tf_tracking()
        timeout_sec = timeout_sec or self.args.startup_timeout_sec
        deadline = time.time() + timeout_sec
        while rclpy.ok() and time.time() <= deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            scan_stamp_sec = (
                None
                if self.last_scan is None
                else replan_runtime.scan_stamp_sec(self.last_scan)
            )
            if (
                self.last_scan is not None
                and self.last_scan_received_sec is not None
                and self.last_scan_received_sec >= min_scan_stamp_sec
                and (
                    scan_stamp_sec is None
                    or scan_stamp_sec
                    >= min_scan_stamp_sec - replan_runtime.FRESH_SCAN_STAMP_SLACK_SEC
                )
            ):
                return
        raise RuntimeError(
            "Timed out waiting for fresh stamped /scan after handoff pause."
        )

    def record_motion_sample(self, yaw_error_deg, linear_x, angular_z, sample_seconds):
        abs_error = abs(yaw_error_deg)
        self.diagnostics.max_abs_yaw_error_deg = max(
            self.diagnostics.max_abs_yaw_error_deg,
            abs_error,
        )
        self.diagnostics.yaw_error_sum_deg += abs_error
        self.diagnostics.yaw_error_count += 1
        if abs(linear_x) <= 1e-9 and abs(angular_z) > 1e-9:
            self.diagnostics.rotate_seconds += sample_seconds
        else:
            self.diagnostics.forward_seconds += sample_seconds

    def current_amcl_health(self):
        if self.last_amcl is None:
            return AmclHealth(
                ok=not self.args.fail_on_bad_localization,
                warnings=["missing_amcl"],
                cov_x=None,
                cov_y=None,
                cov_yaw=None,
                age_sec=None,
            )
        age_sec = (
            None if self.last_amcl_received_sec is None
            else time.time() - self.last_amcl_received_sec
        )
        covariance = self.last_amcl.pose.covariance
        return evaluate_amcl_health(
            covariance,
            age_sec,
            self.args.max_amcl_age_sec,
            self.args.max_amcl_var_x,
            self.args.max_amcl_var_y,
            self.args.max_amcl_var_yaw,
            fail_on_bad_localization=self.args.fail_on_bad_localization,
        )

    def check_health_or_raise(self):
        try:
            pose, frame = self.lookup_pose()
        except RuntimeError as exc:
            raise RecoverableHealthError(
                "tf_lookup",
                self.args.tf_recovery_time_sec,
                str(exc),
            ) from exc
        if pose.stamp_sec is None:
            raise RuntimeError("TF pose has no usable timestamp.")
        pose_age = time.time() - pose.stamp_sec
        self.diagnostics.tf_pose_age_sec = pose_age
        if pose_age > self.args.max_pose_age_sec:
            self.diagnostics.tf_stale_warning_count += 1
            message = f"TF pose is stale: age={pose_age:.3f} sec"
            if self.args.fail_on_stale_tf:
                raise RuntimeError(message)
            self.get_logger().warn(message)

        tf_update_gap_sec = self.update_tf_tracking(pose)
        if (
            tf_update_gap_sec is not None
            and tf_update_gap_sec > self.args.max_tf_update_gap_sec
        ):
            raise RecoverableHealthError(
                "tf_update_gap",
                self.args.tf_recovery_time_sec,
                "TF transform stamp stopped updating: "
                f"gap={tf_update_gap_sec:.3f} sec, "
                f"limit={self.args.max_tf_update_gap_sec:.3f} sec",
            )

        scan_age = (
            None if self.last_scan_received_sec is None
            else time.time() - self.last_scan_received_sec
        )
        if not age_ok(scan_age, self.args.max_scan_age_sec):
            raise RecoverableHealthError(
                "scan_stale",
                self.args.max_scan_age_sec,
                f"/scan is stale: age={scan_age}",
            )

        amcl_health = self.current_amcl_health()
        if amcl_health.warnings:
            self.diagnostics.localization_warning_count += 1
            message = "AMCL localization warning(s): " + ",".join(amcl_health.warnings)
            if not amcl_health.ok:
                raise RuntimeError(message)
            if self.args.pause_on_bad_localization:
                raise RecoverableHealthError(
                    "bad_localization",
                    self.args.localization_recovery_time_sec,
                    message,
                )
            self.get_logger().warn(message)

        return pose, frame, amcl_health

    def check_health_or_recover(self):
        while True:
            try:
                return self.check_health_or_raise()
            except RecoverableHealthError as exc:
                self.diagnostics.recovery_pause_count += 1
                self.stop_repeatedly()
                self.get_logger().warn(
                    f"{exc}; pausing for up to {exc.timeout_sec:.1f} sec"
                )
                deadline = time.time() + exc.timeout_sec
                last_message = str(exc)
                while time.time() < deadline and rclpy.ok():
                    rclpy.spin_once(self, timeout_sec=0.1)
                    time.sleep(0.1)
                    try:
                        return self.check_health_or_raise()
                    except RecoverableHealthError as retry_exc:
                        last_message = str(retry_exc)
                raise RuntimeError(
                    f"{exc.reason} did not recover within "
                    f"{exc.timeout_sec:.1f} sec: {last_message}"
                )

    def check_scan_or_raise(self, mode):
        if self.last_scan is None:
            raise RuntimeError("No /scan sample is available.")
        safety = evaluate_scan_safety(
            self.last_scan.ranges,
            self.last_scan.angle_min,
            self.last_scan.angle_increment,
            self.last_scan.range_min,
            self.last_scan.range_max,
            mode,
            self.args.scan_half_angle_deg,
            self.args.hard_stop_range_m,
            self.args.min_scan_range_m,
            self.args.rotation_stop_range_m,
        )
        if not safety.safe:
            raise BlockedByScanError(safety)
        return safety

    def update_replan_diagnostics(self, result, count_replan=True):
        diag = result.diagnostics
        if count_replan:
            self.diagnostics.replan_count += 1
        self.diagnostics.last_replan_reason = result.reason
        self.diagnostics.updated_map_yaml = result.updated_map_yaml or ""
        self.diagnostics.updated_waypoints_csv = result.updated_waypoints_csv or ""
        self.diagnostics.detected_obstacle_count = diag.detected_obstacle_count
        self.diagnostics.candidate_scan_points = diag.candidate_scan_points
        self.diagnostics.filtered_obstacle_points = diag.filtered_obstacle_points
        self.diagnostics.raw_obstacle_cells = diag.raw_obstacle_cells
        self.diagnostics.free_obstacle_cells = diag.free_obstacle_cells
        self.diagnostics.inflated_cells_total = diag.inflated_cells_total
        self.diagnostics.inflated_cells_newly_occupied = diag.inflated_cells_newly_occupied
        self.diagnostics.inflated_cells_over_static_occupied = diag.inflated_cells_over_static_occupied
        self.diagnostics.scan_frame = diag.scan_frame
        self.diagnostics.scan_age_sec = diag.scan_age_sec
        self.diagnostics.tf_age_sec = diag.tf_age_sec
        self.diagnostics.tf_lookup_mode = diag.tf_lookup_mode
        self.diagnostics.start_snap_distance_m = diag.start_snap_distance_m
        self.diagnostics.goal_snap_distance_m = diag.goal_snap_distance_m
        self.diagnostics.old_remaining_waypoint_count = diag.old_remaining_waypoint_count
        self.diagnostics.new_waypoint_count = diag.new_waypoint_count
        self.diagnostics.old_path_length_m = diag.old_path_length_m
        self.diagnostics.new_path_length_m = diag.new_path_length_m
        self.diagnostics.replan_duration_sec = diag.replan_duration_sec
        self.diagnostics.run_local_map_updates = diag.run_local_map_updates
        self.diagnostics.run_local_replan_count += diag.run_local_replan_count
        self.diagnostics.run_local_last_replan_reason = diag.run_local_last_replan_reason
        self.diagnostics.run_local_no_path_reason = diag.run_local_no_path_reason
        self.diagnostics.run_local_start_cell_blocked = diag.run_local_start_cell_blocked
        self.diagnostics.run_local_goal_cell_blocked = diag.run_local_goal_cell_blocked
        self.diagnostics.run_local_path_blocked_cell_count = diag.run_local_path_blocked_cell_count
        self.diagnostics.run_local_scan_points_valid = diag.run_local_scan_points_valid
        self.diagnostics.run_local_scan_points_used = diag.run_local_scan_points_used
        self.diagnostics.run_local_scan_points_rejected_invalid_range = (
            diag.run_local_scan_points_rejected_invalid_range
        )
        self.diagnostics.run_local_scan_points_rejected_static = diag.run_local_scan_points_rejected_static
        self.diagnostics.run_local_scan_points_rejected_bounds = diag.run_local_scan_points_rejected_bounds
        self.diagnostics.run_local_scan_points_rejected_wall_band = (
            diag.run_local_scan_points_rejected_wall_band
        )
        self.diagnostics.run_local_scan_points_rejected_low_confidence = (
            diag.run_local_scan_points_rejected_low_confidence
        )
        self.diagnostics.run_local_update_rejected_reason = diag.run_local_update_rejected_reason
        self.diagnostics.run_local_initial_scan_count = max(
            self.diagnostics.run_local_initial_scan_count,
            diag.run_local_initial_scan_count,
        )
        self.diagnostics.run_local_corridor_check_distance_m = (
            diag.run_local_corridor_check_distance_m
        )
        self.diagnostics.run_local_inflation_radius_m = diag.run_local_inflation_radius_m
        self.diagnostics.run_local_map_yaml = diag.run_local_map_yaml
        self.diagnostics.run_local_waypoints_csv = diag.run_local_waypoints_csv
        self.diagnostics.run_local_cell_source_counts = diag.run_local_cell_source_counts
        if result.run_local_map is not None:
            self.run_local_map = result.run_local_map
            publish_rviz_obstacles_if_available(self)

    def replanned_waypoints_from_result(self, result):
        return [
            Waypoint(index, x, y)
            for index, x, y in result.waypoints
        ]

    def replanned_tracking_points_from_result(self, result):
        path_points = getattr(result, "path_points", None) or []
        converted = []
        for point in path_points:
            if isinstance(point, Waypoint):
                converted.append(point)
            else:
                converted.append(Waypoint(point[0], point[1], point[2]))
        return converted

    def remember_replan_tracking_replacement(self, result, replanned, current_pose):
        self.last_replan_tracking_points = None
        self.last_replan_tracking_source = "waypoints"
        self.last_replan_tracking_validation = None
        if getattr(self.args, "controller", DEFAULT_CONTROLLER) != "pure-pursuit":
            return

        path_points = WaypointFollower.replanned_tracking_points_from_result(
            self,
            result,
        )
        if not path_points:
            self.last_replan_tracking_points = list(replanned)
            self.last_replan_tracking_source = "replan_sparse_fallback"
            self.last_replan_tracking_validation = TrackingPathValidation(
                source="replan_sparse_fallback",
                point_count=len(replanned),
                validation_status="fallback_sparse_waypoints",
            )
            self.get_logger().warn(
                "Pure-pursuit LiDAR replan did not include dense path_points; "
                "falling back to sparse replanned waypoints for tracking."
            )
            return

        structural_warnings = validate_tracking_point_structure(
            path_points,
            max_segment_m=getattr(
                self.args,
                "tracking_max_segment_m",
                DEFAULT_TRACKING_MAX_SEGMENT_M,
            ),
            label="replan tracking path",
        )
        validation = validate_tracking_path_geometry(
            replanned,
            path_points,
            endpoint_tolerance_m=getattr(
                self.args,
                "tracking_endpoint_tolerance_m",
                DEFAULT_TRACKING_ENDPOINT_TOLERANCE_M,
            ),
            start_tolerance_m=getattr(
                self.args,
                "tracking_start_tolerance_m",
                DEFAULT_TRACKING_START_TOLERANCE_M,
            ),
            allow_mismatch=getattr(
                self.args,
                "allow_tracking_path_mismatch",
                False,
            ),
            current_pose=current_pose,
            source="replan",
            structural_warnings=structural_warnings,
        )
        for warning in validation.warnings:
            self.get_logger().warn(warning)
        self.last_replan_tracking_points = path_points
        self.last_replan_tracking_source = validation.source
        self.last_replan_tracking_validation = validation

    def first_motion_waypoint(self, replanned, current_pose):
        for waypoint in replanned:
            distance_m = math.hypot(
                waypoint.x - current_pose.x,
                waypoint.y - current_pose.y,
            )
            if distance_m > self.args.waypoint_tolerance_m:
                return waypoint
        return replanned[-1]

    def first_motion_waypoint_index(self, replanned, current_pose):
        for index, waypoint in enumerate(replanned):
            distance_m = math.hypot(
                waypoint.x - current_pose.x,
                waypoint.y - current_pose.y,
            )
            if distance_m > self.args.waypoint_tolerance_m:
                return index
        return max(0, len(replanned) - 1)

    def replan_start_artifact_distance_limit_m(self):
        start_on_path_tolerance_m = getattr(
            self.args,
            "start_on_path_tolerance_m",
            self.args.waypoint_tolerance_m,
        )
        return max(
            self.args.waypoint_tolerance_m,
            min(0.35, max(0.0, float(start_on_path_tolerance_m))),
        )

    def first_forward_motion_waypoint_index(self, replanned, current_pose):
        if not replanned:
            return 0
        first_motion_index = WaypointFollower.first_motion_waypoint_index(
            self,
            replanned,
            current_pose,
        )
        artifact_distance_limit_m = (
            WaypointFollower.replan_start_artifact_distance_limit_m(self)
        )
        pose = lidar_obstacle_map.Pose2D(
            current_pose.x,
            current_pose.y,
            current_pose.yaw_deg,
        )
        for index in range(first_motion_index, len(replanned)):
            waypoint = replanned[index]
            distance_m = math.hypot(
                waypoint.x - current_pose.x,
                waypoint.y - current_pose.y,
            )
            first_base = lidar_obstacle_map.map_point_to_base(
                waypoint.x,
                waypoint.y,
                pose,
            )
            if first_base.x >= -self.args.robot_footprint_radius_m:
                return index
            if distance_m > artifact_distance_limit_m:
                return index
        return max(0, len(replanned) - 1)

    def prune_replanned_waypoints_for_progress(self, replanned, current_pose):
        if not replanned:
            return replanned
        index = self.first_motion_waypoint_index(replanned, current_pose)
        return replanned[index:]

    def route_signature(self, waypoints):
        waypoints = list(waypoints)
        if not waypoints:
            return ()
        first = waypoints[0]
        goal = waypoints[-1]
        return (
            len(waypoints),
            round(first.x, 3),
            round(first.y, 3),
            round(goal.x, 3),
            round(goal.y, 3),
        )

    def remember_known_corridor_repair(self, waypoints):
        self.last_known_corridor_repair_signature = self.route_signature(waypoints)
        self.suppressed_known_corridor_signature = None

    def suppress_repeated_known_corridor_repair(self, waypoints):
        signature = self.route_signature(waypoints)
        if (
            not signature
            or signature != getattr(self, "last_known_corridor_repair_signature", None)
        ):
            return False
        if signature != getattr(self, "suppressed_known_corridor_signature", None):
            self.get_logger().warn(
                "Known corridor blockage still overlaps the same repaired route; "
                "continuing under live scan safety instead of replanning again."
            )
        self.suppressed_known_corridor_signature = signature
        return True

    def scan_block_budget_repair_signature(self, current_pose, waypoints):
        return (
            round(current_pose.x, 2),
            round(current_pose.y, 2),
            WaypointFollower.route_signature(self, waypoints),
        )

    def remember_scan_block_budget_repair(self, current_pose, waypoints):
        signature = WaypointFollower.scan_block_budget_repair_signature(
            self,
            current_pose,
            waypoints,
        )
        if (
            signature
            and signature
            == getattr(self, "last_scan_block_budget_repair_signature", None)
        ):
            raise RuntimeError(
                "lidar_replan_failed:"
                "persistent_scan_blockage_after_existing_map_repair"
            )
        self.last_scan_block_budget_repair_signature = signature

    def validate_replan_result(
        self,
        result,
        current_pose,
        old_remaining_waypoints,
        goal_waypoint,
        require_changed=True,
    ):
        if not result.success:
            raise RuntimeError(f"lidar_replan_failed:{result.reason}")
        replanned = self.replanned_waypoints_from_result(result)
        if not replanned:
            raise RuntimeError("lidar_replan_failed:empty_waypoint_list")
        final_error = waypoint_distance(replanned[-1], goal_waypoint)
        if final_error > self.args.goal_tolerance_m:
            raise RuntimeError(
                "lidar_replan_failed:final_goal_mismatch "
                f"error={final_error:.3f}"
            )
        old_pairs = [(round(wp.x, 3), round(wp.y, 3)) for wp in old_remaining_waypoints]
        new_pairs = [(round(wp.x, 3), round(wp.y, 3)) for wp in replanned]
        if require_changed and old_pairs == new_pairs:
            raise RuntimeError("lidar_replan_failed:updated_path_matches_old_path")
        first_motion_index = WaypointFollower.first_motion_waypoint_index(
            self,
            replanned,
            current_pose,
        )
        forward_motion_index = WaypointFollower.first_forward_motion_waypoint_index(
            self,
            replanned,
            current_pose,
        )
        motion_waypoint = replanned[forward_motion_index]
        first_base = lidar_obstacle_map.map_point_to_base(
            motion_waypoint.x,
            motion_waypoint.y,
            lidar_obstacle_map.Pose2D(current_pose.x, current_pose.y, current_pose.yaw_deg),
        )
        if first_base.x < -self.args.robot_footprint_radius_m:
            raise RuntimeError("lidar_replan_failed:first_waypoint_behind_robot")
        first_heading = math.degrees(math.atan2(first_base.y, first_base.x))
        first_heading_error = abs(shortest_angle_delta_deg(0.0, first_heading))
        if not math.isfinite(first_heading_error) or first_heading_error > 180.0:
            raise RuntimeError("lidar_replan_failed:first_segment_heading_unreachable")
        if result.inflated_obstacle_cells and result.path_cells:
            obstacle_path_overlap = set(result.path_cells).intersection(result.inflated_obstacle_cells)
            if obstacle_path_overlap:
                raise RuntimeError("lidar_replan_failed:path_crosses_obstacle_cells")
        if forward_motion_index > first_motion_index:
            logger = self.get_logger() if hasattr(self, "get_logger") else None
            if logger is not None:
                logger.warn(
                    "Pruned behind-the-robot startup waypoint(s) from LiDAR replan: "
                    f"removed={forward_motion_index}"
                )
            replanned = replanned[forward_motion_index:]
        WaypointFollower.remember_replan_tracking_replacement(
            self,
            result,
            replanned,
            current_pose,
        )
        return replanned

    def initialize_run_local_route(self, current_pose, waypoints):
        if self.args.run_local_map_initial_scan_mode == "none":
            return list(waypoints)
        self.stop_repeatedly()
        goal_waypoint = waypoints[-1]
        result = replan_runtime.perform_initial_run_local_replan(
            self,
            self.args,
            current_pose,
            goal_waypoint,
            waypoints,
        )
        self.update_replan_diagnostics(result, count_replan=result.success)
        if not result.success and result.reason in INITIAL_RUN_LOCAL_MAP_NONFATAL_REASONS:
            self.stop_repeatedly()
            if not run_local_map_has_confirmed_obstacles(self.run_local_map):
                self.run_local_map = None
            self.get_logger().warn(
                "Initial run-local obstacle map did not find a confirmed "
                f"free-space obstacle; continuing with the static route. reason={result.reason}"
            )
            return list(waypoints)
        replanned = self.validate_replan_result(
            result,
            current_pose,
            waypoints,
            goal_waypoint,
            require_changed=False,
        )
        self.stop_repeatedly()
        self.get_logger().info(
            "Initial run-local obstacle map completed: "
            f"waypoints={len(replanned)}, map={result.updated_map_yaml}"
        )
        return replanned

    def corridor_blocked_cells(self, current_pose, remaining_waypoints):
        if self.run_local_map is None or not remaining_waypoints:
            return set()
        check_distance_m = self.args.run_local_map_corridor_check_distance_m
        corridor_radius_m = (
            self.args.run_local_map_corridor_radius_m
            if self.args.run_local_map_corridor_radius_m is not None
            else 0.0
        )
        blocked = lidar_obstacle_map.path_corridor_blocked_cells(
            self.run_local_map.static_map,
            lidar_obstacle_map.Pose2D(
                current_pose.x,
                current_pose.y,
                current_pose.yaw_deg,
            ),
            remaining_waypoints,
            self.run_local_map.inflated_obstacle_cells,
            check_distance_m,
            corridor_radius_m,
        )
        self.diagnostics.run_local_corridor_check_distance_m = check_distance_m
        self.diagnostics.run_local_path_blocked_cell_count = len(blocked)
        return blocked

    def prune_run_local_obstacles_after_progress(self, current_pose, remaining_waypoints):
        if self.run_local_map is None or not run_local_map_has_confirmed_obstacles(self.run_local_map):
            return None
        prune_distance_m = getattr(
            self.args,
            "run_local_map_prune_behind_distance_m",
            DEFAULT_RUN_LOCAL_MAP_PRUNE_BEHIND_DISTANCE_M,
        )
        if prune_distance_m <= 0.0:
            return None
        pose = lidar_obstacle_map.Pose2D(
            current_pose.x,
            current_pose.y,
            current_pose.yaw_deg,
        )
        candidate_cells = set()
        metadata = self.run_local_map.static_map.metadata
        for cell in self.run_local_map.confirmed_raw_cells:
            world_x, world_y = lidar_obstacle_map.planner.grid_to_world(
                cell[0],
                cell[1],
                metadata,
            )
            base_point = lidar_obstacle_map.map_point_to_base(world_x, world_y, pose)
            if base_point.x < -prune_distance_m:
                candidate_cells.add(cell)
        if not candidate_cells:
            return None

        corridor_radius_m = (
            self.args.run_local_map_corridor_radius_m
            if self.args.run_local_map_corridor_radius_m is not None
            else 0.0
        )
        corridor_cells = lidar_obstacle_map.path_corridor_cells(
            self.run_local_map.static_map,
            pose,
            remaining_waypoints,
            self.args.run_local_map_corridor_check_distance_m,
            corridor_radius_m,
        )
        protected_cells = set()
        for cell in candidate_cells:
            inflated = lidar_obstacle_map.inflate_cells(
                self.run_local_map.static_map,
                {cell},
                self.run_local_map.config.inflation_radius_m,
            )
            if inflated.intersection(corridor_cells):
                protected_cells.add(cell)

        prune_cells = candidate_cells.difference(protected_cells)
        if not prune_cells:
            return None
        result = self.run_local_map.remove_raw_cells(prune_cells)
        if result.removed_raw_cells:
            self.diagnostics.run_local_pruned_raw_cells += result.removed_raw_cells
            self.diagnostics.run_local_pruned_inflated_cells += result.removed_inflated_cells
            self.diagnostics.run_local_cell_source_counts = self.run_local_map.cell_source_counts()
            self.get_logger().info(
                "Pruned passed run-local obstacle cells: "
                f"raw={result.removed_raw_cells}, "
                f"inflated={result.removed_inflated_cells}"
            )
            publish_rviz_obstacles_if_available(self)
        return result

    def plan_with_existing_run_local_map(
        self,
        current_pose,
        old_remaining_waypoints,
        sequence=None,
        count_replan=True,
    ):
        if self.run_local_map is None:
            raise RuntimeError("lidar_replan_failed:no_run_local_map")
        if not run_local_map_has_confirmed_obstacles(self.run_local_map):
            raise RuntimeError("lidar_replan_failed:no_confirmed_run_local_obstacles")
        sequence = sequence or self.live_replan_attempt_count + 1
        goal_waypoint = old_remaining_waypoints[-1]
        result = replan_runtime.plan_existing_run_local_map(
            self.args,
            self.run_local_map,
            current_pose,
            goal_waypoint,
            old_remaining_waypoints,
            sequence=sequence,
        )
        self.update_replan_diagnostics(result, count_replan=count_replan)
        return self.validate_replan_result(
            result,
            current_pose,
            old_remaining_waypoints,
            goal_waypoint,
            require_changed=True,
        )

    def sparse_retry_scan_args(self):
        return replan_runtime.args_with_obstacle_roi(
            self.args,
            forward_distance_m=getattr(
                self.args,
                "run_local_map_sparse_retry_forward_distance_m",
                DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_FORWARD_DISTANCE_M,
            ),
            forward_half_width_m=getattr(
                self.args,
                "run_local_map_sparse_retry_forward_half_width_m",
                DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_FORWARD_HALF_WIDTH_M,
            ),
            angle_window_deg=getattr(
                self.args,
                "run_local_map_sparse_retry_angle_window_deg",
                DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_ANGLE_WINDOW_DEG,
            ),
        )

    def retry_sparse_lidar_replan(
        self,
        current_pose,
        goal_waypoint,
        old_remaining_waypoints,
        sequence,
    ):
        retry_limit = getattr(
            self.args,
            "run_local_map_sparse_retry_count",
            DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_COUNT,
        )
        last_result = None
        retry_args = self.sparse_retry_scan_args()
        retry_mode = (
            "expanded_forward:"
            f"distance={retry_args.obstacle_forward_distance_m:.3f},"
            f"half_width={retry_args.obstacle_forward_half_width_m:.3f},"
            f"angle={retry_args.obstacle_angle_window_deg:.1f}"
        )
        for retry_index in range(1, retry_limit + 1):
            self.stop_repeatedly()
            self.get_logger().warn(
                "LiDAR map update returned too few accepted scan points; "
                f"retrying with expanded forward ROI ({retry_index}/{retry_limit})."
            )
            result = replan_runtime.perform_lidar_replan(
                self,
                self.args,
                current_pose,
                goal_waypoint,
                old_remaining_waypoints,
                sequence=sequence,
                scan_args=retry_args,
            )
            self.diagnostics.run_local_sparse_retry_count = retry_index
            self.diagnostics.run_local_sparse_retry_mode = retry_mode
            last_result = result
            if result.success:
                self.get_logger().info(
                    "Sparse LiDAR map update retry succeeded: "
                    f"attempt={retry_index}"
                )
                return result
            if result.reason != lidar_obstacle_map.RUN_LOCAL_FAILURE_TOO_FEW_SCAN_POINTS:
                self.get_logger().warn(
                    "Sparse LiDAR map update retry stopped on non-sparse failure: "
                    f"reason={result.reason}"
                )
                return result
        self.get_logger().warn(
            "Sparse LiDAR map update retries exhausted; "
            f"reason={lidar_obstacle_map.RUN_LOCAL_FAILURE_TOO_FEW_SCAN_POINTS}"
        )
        return last_result

    def replan_after_blockage(
        self,
        current_pose,
        old_remaining_waypoints,
        trigger=REPLAN_TRIGGER_SCAN_BLOCKAGE,
    ):
        known_corridor_repair_count = getattr(self, "known_corridor_repair_count", 0)
        sequence = self.live_replan_attempt_count + known_corridor_repair_count + 1
        goal_waypoint = old_remaining_waypoints[-1]
        if trigger == REPLAN_TRIGGER_KNOWN_CORRIDOR and self.run_local_map is not None:
            replanned = self.plan_with_existing_run_local_map(
                current_pose,
                old_remaining_waypoints,
                sequence=sequence,
            )
            self.known_corridor_repair_count = known_corridor_repair_count + 1
            self.get_logger().info(
                "Replanned with existing run-local map for known corridor blockage."
            )
            return replanned
        if self.live_replan_attempt_count >= self.args.max_replans:
            if (
                trigger == REPLAN_TRIGGER_SCAN_BLOCKAGE
                and run_local_map_has_confirmed_obstacles(self.run_local_map)
            ):
                replanned = self.plan_with_existing_run_local_map(
                    current_pose,
                    old_remaining_waypoints,
                    sequence=sequence,
                )
                WaypointFollower.remember_scan_block_budget_repair(
                    self,
                    current_pose,
                    replanned,
                )
                self.known_corridor_repair_count = known_corridor_repair_count + 1
                self.get_logger().warn(
                    "LiDAR replan budget exhausted; repaired route with existing "
                    "run-local map after scan blockage."
                )
                return replanned
            raise RuntimeError("lidar_replan_failed:max_replans_exceeded")
        if self.args.run_local_map_update_mode == "none":
            replanned = self.plan_with_existing_run_local_map(
                current_pose,
                old_remaining_waypoints,
                sequence=sequence,
            )
            self.live_replan_attempt_count += 1
            self.get_logger().info("Replanned with existing run-local map.")
            return replanned
        try:
            result = replan_runtime.perform_lidar_replan(
                self,
                self.args,
                current_pose,
                goal_waypoint,
                old_remaining_waypoints,
                sequence=sequence,
            )
        except RuntimeError as exc:
            if trigger == REPLAN_TRIGGER_SCAN_BLOCKAGE:
                raise lidar_replan_failure(exc) from exc
            if self.run_local_map is None:
                raise
            replanned = self.plan_with_existing_run_local_map(
                current_pose,
                old_remaining_waypoints,
                sequence=sequence,
            )
            self.live_replan_attempt_count += 1
            self.get_logger().warn(
                "LiDAR map update failed; replanned with existing run-local map."
            )
            return replanned
        if (
            trigger == REPLAN_TRIGGER_SCAN_BLOCKAGE
            and not result.success
            and result.reason == lidar_obstacle_map.RUN_LOCAL_FAILURE_TOO_FEW_SCAN_POINTS
            and getattr(
                self.args,
                "run_local_map_sparse_retry_count",
                DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_COUNT,
            ) > 0
        ):
            result = self.retry_sparse_lidar_replan(
                current_pose,
                goal_waypoint,
                old_remaining_waypoints,
                sequence,
            )
        self.update_replan_diagnostics(result)
        if not result.success and self.run_local_map is not None:
            rejected_reasons = {
                lidar_obstacle_map.RUN_LOCAL_FAILURE_STALE_TF,
                lidar_obstacle_map.RUN_LOCAL_FAILURE_STALE_SCAN,
                lidar_obstacle_map.RUN_LOCAL_FAILURE_TOO_FEW_SCAN_POINTS,
                lidar_obstacle_map.RUN_LOCAL_FAILURE_TOO_MANY_REJECTED_POINTS,
                lidar_obstacle_map.RUN_LOCAL_FAILURE_MAX_UPDATES_EXCEEDED,
            }
            if result.reason in rejected_reasons:
                if trigger == REPLAN_TRIGGER_SCAN_BLOCKAGE:
                    raise lidar_replan_failure(result.reason)
                self.get_logger().warn(
                    "LiDAR map update rejected; "
                    f"replanning with existing run-local map. reason={result.reason}"
                )
                replanned = self.plan_with_existing_run_local_map(
                    current_pose,
                    old_remaining_waypoints,
                    sequence=sequence,
                )
                self.live_replan_attempt_count += 1
                return replanned
        replanned = self.validate_replan_result(
            result,
            current_pose,
            old_remaining_waypoints,
            goal_waypoint,
        )
        self.live_replan_attempt_count += 1
        self.get_logger().info(
            "LiDAR obstacle replan completed: "
            f"waypoints={len(replanned)}, map={result.updated_map_yaml}"
        )
        return replanned

    def follow_waypoints(
        self,
        waypoints,
        tracking_points=None,
        tracking_validation=None,
    ):
        reached_count = 0
        start_pose, _frame, amcl_health = self.check_health_or_recover()
        final_pose = start_pose
        last_scan_safety = None
        self.start_pose = start_pose
        self.final_pose = final_pose
        self.last_amcl_health = amcl_health
        if not hasattr(self, "command_smoother"):
            self.command_smoother = build_command_smoother(self.args)
        if not hasattr(self, "last_smoothed_command_time_sec"):
            self.last_smoothed_command_time_sec = None
        reset_command_smoother(self)

        waypoints = list(waypoints)
        continuous_tracking = self.args.controller == "pure-pursuit"
        if not continuous_tracking:
            tracking_points = None
            tracking_validation = build_sparse_tracking_validation(
                source="ignored_stop_go",
                point_count=0,
                status="ignored",
            )
        tracking_source = (
            tracking_validation.source
            if tracking_validation is not None
            else ("csv" if tracking_points is not None else "waypoints")
        )
        route_state = RouteState(
            waypoints,
            tracking_points=tracking_points,
            tracking_source=tracking_source,
            tracking_validation=tracking_validation,
        )
        controller = build_path_controller(
            self.args,
            lookahead_guard=getattr(self, "lookahead_guard", None),
        )
        replan_manager = getattr(self, "replan_manager", ReplanManager(self))
        publish_rviz_route_if_available(self, waypoints, current_pose=start_pose)
        publish_rviz_obstacles_if_available(self)
        if self.args.enable_lidar_map_replan:
            waypoints = replan_manager.initialize_route(start_pose, waypoints)
            last_replan_tracking_points = getattr(
                self,
                "last_replan_tracking_points",
                None,
            )
            replacement_tracking_points = (
                last_replan_tracking_points
                if last_replan_tracking_points is not None
                else tracking_points
            )
            replacement_tracking_source = (
                getattr(self, "last_replan_tracking_source", "waypoints")
                if last_replan_tracking_points is not None
                else tracking_source
            )
            replacement_tracking_validation = (
                getattr(self, "last_replan_tracking_validation", None)
                if last_replan_tracking_points is not None
                else tracking_validation
            )
            route_state.replace_route(
                waypoints,
                tracking_points=replacement_tracking_points,
                tracking_source=replacement_tracking_source,
                tracking_validation=replacement_tracking_validation,
            )
            reset_command_smoother(self)
            controller = build_path_controller(
                self.args,
                lookahead_guard=getattr(self, "lookahead_guard", None),
            )
            publish_rviz_route_if_available(self, waypoints, current_pose=start_pose)
            if self.args.lidar_replan_artifact_only:
                self.stop_repeatedly()
                return {
                    "reached_count": reached_count,
                    "start_pose": start_pose,
                    "final_pose": final_pose,
                    "scan_safety": last_scan_safety,
                    "amcl_health": amcl_health,
                    "base_frame_used": self.base_frame_used,
                    "status": "replan_artifact_only_complete",
                }

        while not route_state.complete:
            waypoint = route_state.current_waypoint()
            publish_rviz_route_if_available(
                self,
                route_state.remaining(),
                current_pose=final_pose,
                current_waypoint_index=0,
            )
            self.get_logger().info(
                f"[{route_state.current_waypoint_index + 1}/{len(route_state.waypoints)}] "
                f"target waypoint {waypoint.index}: "
                f"x={waypoint.x:.3f}, y={waypoint.y:.3f}"
            )
            waypoint_start = time.time()
            reached_current = False
            replanned_current = False

            while rclpy.ok():
                pose, _frame, amcl_health = self.check_health_or_recover()
                final_pose = pose
                self.final_pose = final_pose
                self.last_amcl_health = amcl_health
                step = controller.compute(pose, route_state)
                self.last_lookahead_guard_result = step.guard_result
                if hasattr(self, "maybe_log_velocity_scheduler_result"):
                    self.maybe_log_velocity_scheduler_result(
                        step.velocity_schedule_result,
                        time.time(),
                    )
                if (
                    self.args.verbose
                    and step.guard_result is not None
                    and step.guard_result.status != "clear"
                ):
                    self.get_logger().info(
                        "Pure-pursuit lookahead guard result: "
                        f"mode={self.args.pure_pursuit_lookahead_guard}, "
                        f"status={step.guard_result.status}, "
                        "selected_distance_m="
                        f"{format_optional_m(step.guard_result.selected_target_distance_m)}, "
                        f"blocked_cells={step.guard_result.blocked_cell_count}"
                    )

                if step.reached:
                    if continuous_tracking:
                        route_state.mark_complete()
                        reached_count = len(route_state.waypoints)
                    else:
                        reached_count += 1
                    self.reached_count = reached_count
                    if self.args.enable_lidar_map_replan:
                        replan_manager.prune_after_progress(
                            pose,
                            route_state.waypoints[
                                route_state.current_waypoint_index + 1:
                            ],
                        )
                    self.last_known_corridor_repair_signature = None
                    self.suppressed_known_corridor_signature = None
                    self.last_scan_block_budget_repair_signature = None
                    self.last_lookahead_guard_block_signature = None
                    reset_command_smoother(self)
                    self.stop_repeatedly()
                    self.spin_for(self.args.settle_sec)
                    if not continuous_tracking:
                        route_state.advance()
                    reached_current = True
                    break

                if continuous_tracking:
                    before_index = route_state.current_waypoint_index
                    if route_state.advance_if_reached(
                        pose,
                        self.args.waypoint_tolerance_m,
                        self.args.pure_pursuit_goal_tolerance_m,
                    ):
                        reached_count = max(
                            reached_count,
                            route_state.current_waypoint_index,
                        )
                        self.reached_count = reached_count
                    if route_state.current_waypoint_index != before_index:
                        waypoint_start = time.time()

                if time.time() - waypoint_start > self.args.max_waypoint_time_sec:
                    raise WaypointTimeoutError(waypoint)

                if step.mode == "blocked":
                    if self.args.verbose and step.guard_result is not None:
                        self.get_logger().warn(
                            "Pure-pursuit lookahead guard blocked motion: "
                            f"status={step.guard_result.status}, "
                            f"blocked_cells={step.guard_result.blocked_cell_count}"
                        )
                    reset_command_smoother(self)
                    self.stop_repeatedly()
                    last_scan_safety = self.check_scan_or_raise("forward")
                    self.last_scan_safety = last_scan_safety
                    if not self.args.enable_lidar_map_replan:
                        raise RuntimeError("pure_pursuit_lookahead_blocked")
                    guard_signature = guard_block_signature(
                        pose,
                        route_state.remaining_tracking_points(),
                    )
                    if (
                        guard_signature
                        == getattr(
                            self,
                            "last_lookahead_guard_block_signature",
                            None,
                        )
                    ):
                        raise RuntimeError(
                            "pure_pursuit_lookahead_blocked_after_unchanged_replan"
                        )
                    self.last_lookahead_guard_block_signature = guard_signature
                    remaining = route_state.remaining()
                    replanned = replan_manager.replan_after_blockage(
                        pose,
                        remaining,
                        trigger=REPLAN_TRIGGER_LOOKAHEAD_GUARD,
                    )
                    publish_rviz_route_if_available(
                        self,
                        replanned,
                        current_pose=pose,
                        current_waypoint_index=0,
                    )
                    if self.args.lidar_replan_artifact_only:
                        self.stop_repeatedly()
                        return {
                            "reached_count": reached_count,
                            "start_pose": start_pose,
                            "final_pose": final_pose,
                            "scan_safety": last_scan_safety,
                            "amcl_health": amcl_health,
                            "base_frame_used": self.base_frame_used,
                            "status": "replan_artifact_only_complete",
                        }
                    waypoints = self.prune_replanned_waypoints_for_progress(
                        replanned,
                        pose,
                    )
                    route_state.replace_route(
                        waypoints,
                        tracking_points=getattr(
                            self,
                            "last_replan_tracking_points",
                            None,
                        ),
                        tracking_source=getattr(
                            self,
                            "last_replan_tracking_source",
                            "waypoints",
                        ),
                        tracking_validation=getattr(
                            self,
                            "last_replan_tracking_validation",
                            None,
                        ),
                    )
                    reset_command_smoother(self)
                    controller = build_path_controller(
                        self.args,
                        lookahead_guard=getattr(self, "lookahead_guard", None),
                    )
                    replanned_current = True
                    break

                try:
                    last_scan_safety = self.check_scan_or_raise(step.mode)
                    self.last_scan_safety = last_scan_safety
                    self.last_scan_block_budget_repair_signature = None
                except BlockedByScanError as exc:
                    reset_command_smoother(self)
                    if self.args.enable_lidar_map_replan:
                        remaining = route_state.remaining()
                        replanned = replan_manager.replan_after_blockage(
                            pose,
                            remaining,
                            trigger=REPLAN_TRIGGER_SCAN_BLOCKAGE,
                        )
                        publish_rviz_route_if_available(
                            self,
                            replanned,
                            current_pose=pose,
                            current_waypoint_index=0,
                        )
                        if self.args.lidar_replan_artifact_only:
                            self.stop_repeatedly()
                            return {
                                "reached_count": reached_count,
                                "start_pose": start_pose,
                                "final_pose": final_pose,
                                "scan_safety": exc.scan_safety,
                                "amcl_health": amcl_health,
                                "base_frame_used": self.base_frame_used,
                                "status": "replan_artifact_only_complete",
                            }
                        waypoints = self.prune_replanned_waypoints_for_progress(replanned, pose)
                        route_state.replace_route(
                            waypoints,
                            tracking_points=getattr(
                                self,
                                "last_replan_tracking_points",
                                None,
                            ),
                            tracking_source=getattr(
                                self,
                                "last_replan_tracking_source",
                                "waypoints",
                            ),
                            tracking_validation=getattr(
                                self,
                                "last_replan_tracking_validation",
                                None,
                            ),
                        )
                        reset_command_smoother(self)
                        controller = build_path_controller(
                            self.args,
                            lookahead_guard=getattr(self, "lookahead_guard", None),
                        )
                        replanned_current = True
                        break
                    raise BlockedByScanError(exc.scan_safety, waypoint) from exc
                if self.args.enable_lidar_map_replan and self.run_local_map is not None:
                    remaining = route_state.remaining()
                    replan_manager.prune_after_progress(
                        pose,
                        remaining,
                    )
                    blocked_cells = replan_manager.corridor_blocked_cells(pose, remaining)
                    if blocked_cells and not self.suppress_repeated_known_corridor_repair(remaining):
                        publish_rviz_obstacles_if_available(self, blocked_cells)
                        self.stop_repeatedly()
                        replanned = replan_manager.replan_after_blockage(
                            pose,
                            remaining,
                            trigger=REPLAN_TRIGGER_KNOWN_CORRIDOR,
                        )
                        publish_rviz_route_if_available(
                            self,
                            replanned,
                            current_pose=pose,
                            current_waypoint_index=0,
                        )
                        if self.args.lidar_replan_artifact_only:
                            self.stop_repeatedly()
                            return {
                                "reached_count": reached_count,
                                "start_pose": start_pose,
                                "final_pose": final_pose,
                                "scan_safety": last_scan_safety,
                                "amcl_health": amcl_health,
                                "base_frame_used": self.base_frame_used,
                                "status": "replan_artifact_only_complete",
                            }
                        waypoints = self.prune_replanned_waypoints_for_progress(replanned, pose)
                        route_state.replace_route(
                            waypoints,
                            tracking_points=getattr(
                                self,
                                "last_replan_tracking_points",
                                None,
                            ),
                            tracking_source=getattr(
                                self,
                                "last_replan_tracking_source",
                                "waypoints",
                            ),
                            tracking_validation=getattr(
                                self,
                                "last_replan_tracking_validation",
                                None,
                            ),
                        )
                        reset_command_smoother(self)
                        controller = build_path_controller(
                            self.args,
                            lookahead_guard=getattr(self, "lookahead_guard", None),
                        )
                        self.remember_known_corridor_repair(waypoints)
                        replanned_current = True
                        break
                command = smoothed_step_command(self, step, time.time())
                self.record_motion_sample(
                    step.yaw_error_deg,
                    command.linear_x,
                    command.angular_z,
                    1.0 / self.args.control_rate_hz,
                )
                self.publish_velocity(command.linear_x, command.angular_z)
                rclpy.spin_once(self, timeout_sec=1.0 / self.args.control_rate_hz)
                time.sleep(1.0 / self.args.control_rate_hz)

            if replanned_current:
                continue
            if reached_current:
                continue
            raise RuntimeError("ROS shutdown while following waypoints")

        return {
            "reached_count": reached_count,
            "start_pose": start_pose,
            "final_pose": final_pose,
            "scan_safety": last_scan_safety,
            "amcl_health": amcl_health,
            "base_frame_used": self.base_frame_used,
        }


class BlockedByScanError(RuntimeError):
    def __init__(self, scan_safety, waypoint=None):
        super().__init__(
            "Blocked by /scan safety: "
            f"reason={scan_safety.reason}, "
            f"min={scan_safety.min_range_m}, p05={scan_safety.percentile_5_m}"
        )
        self.scan_safety = scan_safety
        self.waypoint = waypoint


class WaypointTimeoutError(RuntimeError):
    def __init__(self, waypoint):
        super().__init__(f"Timed out trying to reach waypoint {waypoint.index}")
        self.waypoint = waypoint


class RecoverableHealthError(RuntimeError):
    def __init__(self, reason, timeout_sec, message):
        super().__init__(message)
        self.reason = reason
        self.timeout_sec = timeout_sec


def transform_to_pose2d(transform, frame_id):
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    stamp_sec = stamp_to_sec(transform.header.stamp)
    return Pose2D(
        x=float(translation.x),
        y=float(translation.y),
        yaw_deg=quaternion_to_yaw_deg(
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        ),
        stamp_sec=stamp_sec,
        frame_id=frame_id,
    )


def compose_2d_pose(parent_from_mid, mid_from_child, child_frame_id):
    parent_pose = transform_to_pose2d(parent_from_mid, parent_from_mid.header.frame_id)
    child_pose = transform_to_pose2d(mid_from_child, child_frame_id)
    yaw_rad = math.radians(parent_pose.yaw_deg)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    x = parent_pose.x + cos_yaw * child_pose.x - sin_yaw * child_pose.y
    y = parent_pose.y + sin_yaw * child_pose.x + cos_yaw * child_pose.y
    yaw_deg = shortest_angle_delta_deg(0.0, parent_pose.yaw_deg + child_pose.yaw_deg)
    return Pose2D(
        x=x,
        y=y,
        yaw_deg=yaw_deg,
        stamp_sec=child_pose.stamp_sec,
        frame_id=child_frame_id,
    )


def require_motion_confirmation(args, waypoints):
    if args.yes:
        return True

    print("\nThis command will publish /cmd_vel to the physical TurtleBot.")
    print("Safety requirements:")
    print("  - Nav2 localization is running with the saved map")
    print("  - RViz pose estimate is set and /scan aligns with the map")
    print("  - no active Nav2 goal/controller is publishing /cmd_vel")
    print("  - clear the path and keep an operator near the robot")
    print("  - keep Ctrl+C and physical stop available")
    print(f"Run ID: {args.run_id}")
    print(f"Waypoints: {len(waypoints)} from {args.waypoints}")
    response = input("Type RUN to start waypoint following: ").strip()
    return response == "RUN"


def wait_before_follow_confirmation(args, current_pose, executable_waypoints, input_fn=input):
    if not args.wait_before_follow:
        return True

    print("\nWaypoint follower handoff is ready.")
    print("The robot is stopped after Nav2 staging and before custom waypoint following.")
    print("Place the temporary obstacle on the planned path now.")
    print("Safety requirements:")
    print("  - keep the path area clear except for the test obstacle")
    print("  - keep Ctrl+C and physical stop available")
    print(
        "Current pose: "
        f"x={current_pose.x:.3f}, y={current_pose.y:.3f}, yaw={current_pose.yaw_deg:.1f} deg"
    )
    if executable_waypoints:
        first = executable_waypoints[0]
        print(
            "First follower waypoint: "
            f"index={first.index}, x={first.x:.3f}, y={first.y:.3f}"
        )
    response = input_fn("Type RUN to start custom waypoint following: ").strip()
    return response == "RUN"


def print_dry_run(
    args,
    raw_waypoints,
    executable_waypoints,
    tracking_validation=None,
    lookahead_guard=None,
):
    print("Waypoint follower dry run")
    print(f"Waypoint CSV: {args.waypoints}")
    print(f"Raw waypoints: {len(raw_waypoints)}")
    print(f"Executable waypoints: {len(executable_waypoints)}")
    if executable_waypoints:
        first = executable_waypoints[0]
        last = executable_waypoints[-1]
        print(
            "First executable waypoint: "
            f"source index {first.index}, x={first.x:.3f}, y={first.y:.3f}"
        )
        print(
            "Last executable waypoint: "
            f"source index {last.index}, x={last.x:.3f}, y={last.y:.3f}"
        )
    print(f"Start selection: {args.start_selection}")
    print(f"Wait before follow: {'yes' if args.wait_before_follow else 'no'}")
    print(f"LiDAR map replan: {'enabled' if args.enable_lidar_map_replan else 'disabled'}")
    print(f"Log path: {args.results_csv}")
    if not args.verbose:
        print("Detailed route/config hidden; rerun with --verbose to print it.")
        return

    print(f"Map frame: {args.map_frame}")
    print(f"Base frame: {args.base_frame}, fallback: {args.fallback_base_frame}")
    print(f"Controller: {args.controller}")
    if tracking_validation is not None:
        print(f"controller={args.controller}")
        print(f"tracking_source={tracking_validation.source}")
        print(f"tracking_point_count={tracking_validation.point_count}")
        print(
            "tracking_endpoint_error_m="
            f"{format_optional_m(tracking_validation.endpoint_error_m)}"
        )
        if tracking_validation.start_projection_error_m is None:
            print(
                "tracking_start_error_m="
                f"{format_optional_m(tracking_validation.start_error_m)}"
            )
        else:
            print(
                "tracking_start_projection_error_m="
                f"{format_optional_m(tracking_validation.start_projection_error_m)}"
            )
        print(
            "tracking_validation_status="
            f"{tracking_validation.validation_status}"
        )
    print(f"Linear speed: {args.linear_speed:.3f} m/s")
    print(f"Max angular speed: {args.max_angular_speed:.3f} rad/s")
    print(f"Waypoint tolerance: {args.waypoint_tolerance_m:.3f} m")
    print(f"Goal tolerance: {args.goal_tolerance_m:.3f} m")
    if args.controller == "pure-pursuit":
        print(f"Path lookahead: {args.path_lookahead_m:.3f} m")
        print(
            "Pure-pursuit goal tolerance: "
            f"{args.pure_pursuit_goal_tolerance_m:.3f} m"
        )
        print(f"pure_pursuit_speed_profile={args.pure_pursuit_speed_profile}")
        print(
            "pure_pursuit_default_linear_speed_resolved_mps="
            f"{args.linear_speed:.3f}"
        )
        print(
            "pure_pursuit_default_max_angular_speed_resolved_radps="
            f"{args.max_angular_speed:.3f}"
        )
        print(
            "pure_pursuit_max_lateral_accel_mps2="
            f"{args.pure_pursuit_max_lateral_accel_mps2:.3f}"
        )
        print(
            "pure_pursuit_turn_speed_margin="
            f"{args.pure_pursuit_turn_speed_margin:.3f}"
        )
        print(
            "pure_pursuit_heading_deadband_deg="
            f"{args.pure_pursuit_heading_deadband_deg:.3f}"
        )
        print(
            "pure_pursuit_lateral_deadband_m="
            f"{args.pure_pursuit_lateral_deadband_m:.3f}"
        )
        print(
            "pure_pursuit_curvature_limit_start_heading_error_deg="
            f"{args.pure_pursuit_curvature_limit_start_heading_error_deg:.3f}"
        )
        print(
            "pure_pursuit_curvature_limit_full_heading_error_deg="
            f"{args.pure_pursuit_curvature_limit_full_heading_error_deg:.3f}"
        )
        print(
            "pure_pursuit_rotate_start_heading_error_deg="
            f"{args.pure_pursuit_rotate_start_heading_error_deg:.3f}"
        )
        print(
            "pure_pursuit_rotate_stop_heading_error_deg="
            f"{args.pure_pursuit_rotate_stop_heading_error_deg:.3f}"
        )
        print(
            "pure_pursuit_min_curvature_linear_speed_mps="
            f"{args.pure_pursuit_min_curvature_linear_speed_mps:.3f}"
        )
        print(f"Tracking path CSV: {args.tracking_path_csv or 'none'}")
        print(f"pure_pursuit_lookahead_guard={args.pure_pursuit_lookahead_guard}")
        print(
            "pure_pursuit_min_guarded_lookahead_m="
            f"{args.pure_pursuit_min_guarded_lookahead_m:.3f}"
        )
        if args.pure_pursuit_lookahead_guard != LOOKAHEAD_GUARD_OFF:
            print(
                "lookahead_guard_static_inflation_radius_m="
                f"{args.pure_pursuit_lookahead_guard_static_inflation_radius_m:.3f}"
            )
            print("lookahead_guard_unknown_cells=blocked")
            print(
                "lookahead_guard_static_blocked_cell_count="
                f"{len(lookahead_guard.static_blocked_cells) if lookahead_guard else 'n/a'}"
            )
            print("lookahead_guard_status=configured")
            print("lookahead_guard_selected_target_distance_m=n/a")
            print("lookahead_guard_blocked_cell_count=n/a")
        print(
            "pure_pursuit_command_smoothing="
            f"{args.pure_pursuit_command_smoothing}"
        )
        if args.pure_pursuit_command_smoothing == COMMAND_SMOOTHING_RATE_LIMIT:
            print(
                "pure_pursuit_max_linear_accel_mps2="
                f"{args.pure_pursuit_max_linear_accel_mps2:.3f}"
            )
            print(
                "pure_pursuit_max_linear_decel_mps2="
                f"{args.pure_pursuit_max_linear_decel_mps2:.3f}"
            )
            print(
                "pure_pursuit_max_angular_accel_radps2="
                f"{args.pure_pursuit_max_angular_accel_radps2:.3f}"
            )
            print(
                "pure_pursuit_max_angular_decel_radps2="
                f"{args.pure_pursuit_max_angular_decel_radps2:.3f}"
            )
            print(
                "pure_pursuit_final_decel_distance_m="
                f"{args.pure_pursuit_final_decel_distance_m:.3f}"
            )
            print(
                "pure_pursuit_min_smoothed_linear_speed_mps="
                f"{args.pure_pursuit_min_smoothed_linear_speed_mps:.3f}"
            )
            print("pure_pursuit_smoothing_dt_clamp=[0,2/control_rate_hz]")
    print(f"RViz visualization: {'disabled' if args.no_rviz_visualization else 'enabled'}")
    if not args.no_rviz_visualization:
        print(f"  path topic: {args.rviz_path_topic}")
        print(f"  waypoint markers: {args.rviz_waypoint_marker_topic}")
        print(f"  obstacle markers: {args.rviz_obstacle_marker_topic}")
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
        print(f"  sparse retry count: {args.run_local_map_sparse_retry_count}")
        print(f"  prune behind distance: {args.run_local_map_prune_behind_distance_m:.3f} m")
    if args.start_selection == "path-progress":
        print(
            "Runtime route selection uses live TF after startup; "
            "the route below is a fixed-skip preview."
        )
    print("Executable route:")
    for index, waypoint in enumerate(executable_waypoints, start=1):
        print(f"  {index}. source index {waypoint.index}: x={waypoint.x:.3f}, y={waypoint.y:.3f}")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Follow planned A* waypoints using TF pose and /cmd_vel.",
    )
    parser.add_argument("--waypoints", default=DEFAULT_WAYPOINTS_CSV, type=Path)
    parser.add_argument("--run-id", help="Run ID for logging.")
    parser.add_argument("--results-csv", default=DEFAULT_RESULTS_CSV, type=Path)
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--base-frame", default="base_footprint")
    parser.add_argument("--fallback-base-frame", default="base_link")
    parser.add_argument("--linear-speed", type=float)
    parser.add_argument("--min-linear-speed", default=DEFAULT_MIN_LINEAR_SPEED_MPS, type=float)
    parser.add_argument("--linear-gain", default=DEFAULT_LINEAR_GAIN, type=float)
    parser.add_argument("--max-angular-speed", type=float)
    parser.add_argument("--yaw-gain", default=DEFAULT_YAW_GAIN, type=float)
    parser.add_argument("--forward-yaw-deadband-deg", default=DEFAULT_FORWARD_YAW_DEADBAND_DEG, type=float)
    parser.add_argument("--forward-stop-heading-error-deg", default=DEFAULT_FORWARD_STOP_HEADING_ERROR_DEG, type=float)
    parser.add_argument("--waypoint-tolerance-m", default=DEFAULT_WAYPOINT_TOLERANCE_M, type=float)
    parser.add_argument("--goal-tolerance-m", default=DEFAULT_GOAL_TOLERANCE_M, type=float)
    parser.add_argument(
        "--controller",
        default=DEFAULT_CONTROLLER,
        choices=["stop-go", "pure-pursuit"],
    )
    parser.add_argument("--path-lookahead-m", default=DEFAULT_PATH_LOOKAHEAD_M, type=float)
    parser.add_argument("--pure-pursuit-goal-tolerance-m", type=float)
    parser.add_argument(
        "--pure-pursuit-lookahead-guard",
        default=DEFAULT_PURE_PURSUIT_LOOKAHEAD_GUARD,
        choices=LOOKAHEAD_GUARD_MODES,
    )
    parser.add_argument(
        "--pure-pursuit-min-guarded-lookahead-m",
        default=DEFAULT_PURE_PURSUIT_MIN_GUARDED_LOOKAHEAD_M,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-lookahead-guard-static-inflation-radius-m",
        default=DEFAULT_PURE_PURSUIT_LOOKAHEAD_GUARD_STATIC_INFLATION_RADIUS_M,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-command-smoothing",
        default=DEFAULT_PURE_PURSUIT_COMMAND_SMOOTHING,
        choices=COMMAND_SMOOTHING_MODES,
    )
    parser.add_argument(
        "--pure-pursuit-speed-profile",
        default=DEFAULT_PURE_PURSUIT_SPEED_PROFILE,
        choices=SPEED_PROFILE_MODES,
    )
    parser.add_argument(
        "--pure-pursuit-max-lateral-accel-mps2",
        default=DEFAULT_PURE_PURSUIT_MAX_LATERAL_ACCEL_MPS2,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-turn-speed-margin",
        default=DEFAULT_PURE_PURSUIT_TURN_SPEED_MARGIN,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-heading-deadband-deg",
        default=DEFAULT_PURE_PURSUIT_HEADING_DEADBAND_DEG,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-lateral-deadband-m",
        default=DEFAULT_PURE_PURSUIT_LATERAL_DEADBAND_M,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-curvature-limit-start-heading-error-deg",
        default=DEFAULT_PURE_PURSUIT_CURVATURE_LIMIT_START_HEADING_ERROR_DEG,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-curvature-limit-full-heading-error-deg",
        default=DEFAULT_PURE_PURSUIT_CURVATURE_LIMIT_FULL_HEADING_ERROR_DEG,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-rotate-start-heading-error-deg",
        default=DEFAULT_PURE_PURSUIT_ROTATE_START_HEADING_ERROR_DEG,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-rotate-stop-heading-error-deg",
        default=DEFAULT_PURE_PURSUIT_ROTATE_STOP_HEADING_ERROR_DEG,
        type=float,
    )
    parser.add_argument("--pure-pursuit-min-curvature-linear-speed-mps", type=float)
    parser.add_argument(
        "--pure-pursuit-max-linear-accel-mps2",
        default=DEFAULT_PURE_PURSUIT_MAX_LINEAR_ACCEL_MPS2,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-max-linear-decel-mps2",
        default=DEFAULT_PURE_PURSUIT_MAX_LINEAR_DECEL_MPS2,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-max-angular-accel-radps2",
        default=DEFAULT_PURE_PURSUIT_MAX_ANGULAR_ACCEL_RADPS2,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-max-angular-decel-radps2",
        default=DEFAULT_PURE_PURSUIT_MAX_ANGULAR_DECEL_RADPS2,
        type=float,
    )
    parser.add_argument(
        "--pure-pursuit-final-decel-distance-m",
        default=DEFAULT_PURE_PURSUIT_FINAL_DECEL_DISTANCE_M,
        type=float,
    )
    parser.add_argument("--pure-pursuit-min-smoothed-linear-speed-mps", type=float)
    parser.add_argument("--tracking-path-csv", type=Path)
    parser.add_argument(
        "--tracking-endpoint-tolerance-m",
        default=DEFAULT_TRACKING_ENDPOINT_TOLERANCE_M,
        type=float,
    )
    parser.add_argument(
        "--tracking-start-tolerance-m",
        default=DEFAULT_TRACKING_START_TOLERANCE_M,
        type=float,
    )
    parser.add_argument(
        "--tracking-max-segment-m",
        default=DEFAULT_TRACKING_MAX_SEGMENT_M,
        type=float,
    )
    parser.add_argument("--allow-tracking-path-mismatch", action="store_true")
    parser.add_argument(
        "--rotate-start-heading-error-deg",
        default=DEFAULT_ROTATE_START_HEADING_ERROR_DEG,
        type=float,
    )
    parser.add_argument(
        "--rotate-stop-heading-error-deg",
        default=DEFAULT_ROTATE_STOP_HEADING_ERROR_DEG,
        type=float,
    )
    parser.add_argument("--min-waypoint-spacing-m", default=DEFAULT_MIN_WAYPOINT_SPACING_M, type=float)
    parser.add_argument(
        "--start-selection",
        default=DEFAULT_START_SELECTION,
        choices=["path-progress", "fixed-skip"],
    )
    parser.add_argument("--start-on-path-tolerance-m", default=DEFAULT_START_ON_PATH_TOLERANCE_M, type=float)
    parser.add_argument("--odom-frame", default=DEFAULT_ODOM_FRAME)
    parser.add_argument("--scan-half-angle-deg", default=DEFAULT_SCAN_HALF_ANGLE_DEG, type=float)
    parser.add_argument("--hard-stop-range-m", default=DEFAULT_HARD_STOP_RANGE_M, type=float)
    parser.add_argument("--min-scan-range-m", default=DEFAULT_MIN_SCAN_RANGE_M, type=float)
    parser.add_argument("--rotation-stop-range-m", default=DEFAULT_ROTATION_STOP_RANGE_M, type=float)
    parser.add_argument("--max-pose-age-sec", default=DEFAULT_MAX_POSE_AGE_SEC, type=float)
    parser.add_argument("--max-scan-age-sec", default=DEFAULT_MAX_SCAN_AGE_SEC, type=float)
    parser.add_argument("--max-amcl-age-sec", default=DEFAULT_MAX_AMCL_AGE_SEC, type=float)
    parser.add_argument("--max-amcl-var-x", default=DEFAULT_MAX_AMCL_VAR_X, type=float)
    parser.add_argument("--max-amcl-var-y", default=DEFAULT_MAX_AMCL_VAR_Y, type=float)
    parser.add_argument("--max-amcl-var-yaw", default=DEFAULT_MAX_AMCL_VAR_YAW, type=float)
    parser.add_argument("--max-waypoint-time-sec", default=DEFAULT_MAX_WAYPOINT_TIME_SEC, type=float)
    parser.add_argument("--max-tf-update-gap-sec", default=DEFAULT_MAX_TF_UPDATE_GAP_SEC, type=float)
    parser.add_argument("--tf-recovery-time-sec", default=DEFAULT_TF_RECOVERY_TIME_SEC, type=float)
    parser.add_argument(
        "--localization-recovery-time-sec",
        default=DEFAULT_LOCALIZATION_RECOVERY_TIME_SEC,
        type=float,
    )
    parser.add_argument("--control-rate-hz", default=DEFAULT_CONTROL_RATE_HZ, type=float)
    parser.add_argument("--settle-sec", default=DEFAULT_SETTLE_SEC, type=float)
    parser.add_argument("--startup-timeout-sec", default=DEFAULT_STARTUP_TIMEOUT_SEC, type=float)
    parser.add_argument("--notes", default="follow_planned_waypoints")
    parser.add_argument("--fail-on-bad-localization", action="store_true")
    parser.add_argument("--pause-on-bad-localization", action="store_true")
    parser.add_argument("--require-amcl-startup", action="store_true")
    parser.add_argument("--fail-on-stale-tf", action="store_true")
    parser.add_argument("--no-skip-first-waypoint", action="store_true")
    parser.add_argument("--rviz-path-topic", default=DEFAULT_RVIZ_PATH_TOPIC)
    parser.add_argument(
        "--rviz-waypoint-marker-topic",
        default=DEFAULT_RVIZ_WAYPOINT_MARKER_TOPIC,
    )
    parser.add_argument(
        "--rviz-obstacle-marker-topic",
        default=DEFAULT_RVIZ_OBSTACLE_MARKER_TOPIC,
    )
    parser.add_argument("--no-rviz-visualization", action="store_true")
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
        choices=["none", "forward", "full"],
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
    parser.add_argument(
        "--run-local-map-sparse-retry-count",
        default=DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_COUNT,
        type=int,
    )
    parser.add_argument(
        "--run-local-map-sparse-retry-forward-half-width-m",
        default=DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_FORWARD_HALF_WIDTH_M,
        type=float,
    )
    parser.add_argument(
        "--run-local-map-sparse-retry-angle-window-deg",
        default=DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_ANGLE_WINDOW_DEG,
        type=float,
    )
    parser.add_argument(
        "--run-local-map-sparse-retry-forward-distance-m",
        default=DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_FORWARD_DISTANCE_M,
        type=float,
    )
    parser.add_argument(
        "--run-local-map-prune-behind-distance-m",
        default=DEFAULT_RUN_LOCAL_MAP_PRUNE_BEHIND_DISTANCE_M,
        type=float,
    )
    parser.add_argument("--run-local-map-artifact-prefix")
    parser.add_argument("--wait-before-follow", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed route/configuration output.",
    )
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args(argv)

    if not args.run_id:
        args.run_id = datetime.now().strftime("waypoint_follow_%Y%m%d_%H%M%S")
    if args.linear_speed is None:
        args.linear_speed = (
            DEFAULT_PURE_PURSUIT_LINEAR_SPEED_MPS
            if args.controller == "pure-pursuit"
            else DEFAULT_LINEAR_SPEED_MPS
        )
    if args.max_angular_speed is None:
        args.max_angular_speed = (
            DEFAULT_PURE_PURSUIT_MAX_ANGULAR_SPEED_RADPS
            if args.controller == "pure-pursuit"
            else DEFAULT_MAX_ANGULAR_SPEED_RADPS
        )
    if args.pure_pursuit_goal_tolerance_m is None:
        args.pure_pursuit_goal_tolerance_m = args.goal_tolerance_m
    if args.pure_pursuit_min_curvature_linear_speed_mps is None:
        args.pure_pursuit_min_curvature_linear_speed_mps = args.min_linear_speed
    if args.pure_pursuit_min_smoothed_linear_speed_mps is None:
        args.pure_pursuit_min_smoothed_linear_speed_mps = args.min_linear_speed
    validate_args(parser, args)
    return args


def validate_args(parser, args):
    positive_fields = [
        "linear_speed",
        "min_linear_speed",
        "linear_gain",
        "max_angular_speed",
        "yaw_gain",
        "waypoint_tolerance_m",
        "goal_tolerance_m",
        "path_lookahead_m",
        "pure_pursuit_goal_tolerance_m",
        "pure_pursuit_min_guarded_lookahead_m",
        "pure_pursuit_max_lateral_accel_mps2",
        "pure_pursuit_rotate_start_heading_error_deg",
        "pure_pursuit_rotate_stop_heading_error_deg",
        "tracking_endpoint_tolerance_m",
        "tracking_start_tolerance_m",
        "tracking_max_segment_m",
        "rotate_start_heading_error_deg",
        "rotate_stop_heading_error_deg",
        "scan_half_angle_deg",
        "hard_stop_range_m",
        "min_scan_range_m",
        "rotation_stop_range_m",
        "start_on_path_tolerance_m",
        "max_pose_age_sec",
        "max_scan_age_sec",
        "max_amcl_age_sec",
        "max_amcl_var_x",
        "max_amcl_var_y",
        "max_amcl_var_yaw",
        "max_waypoint_time_sec",
        "max_tf_update_gap_sec",
        "tf_recovery_time_sec",
        "localization_recovery_time_sec",
        "control_rate_hz",
        "startup_timeout_sec",
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
        "run_local_map_sparse_retry_forward_half_width_m",
        "run_local_map_sparse_retry_angle_window_deg",
        "run_local_map_sparse_retry_forward_distance_m",
        "run_local_map_prune_behind_distance_m",
    ]
    for field in positive_fields:
        if getattr(args, field) <= 0.0:
            parser.error(f"--{field.replace('_', '-')} must be greater than zero")
    if args.min_linear_speed > args.linear_speed:
        parser.error("--min-linear-speed must be <= --linear-speed")
    if args.rotate_stop_heading_error_deg >= args.rotate_start_heading_error_deg:
        parser.error("--rotate-stop-heading-error-deg must be < --rotate-start-heading-error-deg")
    if args.forward_yaw_deadband_deg < 0.0:
        parser.error("--forward-yaw-deadband-deg must be non-negative")
    if args.forward_yaw_deadband_deg >= args.forward_stop_heading_error_deg:
        parser.error("--forward-yaw-deadband-deg must be < --forward-stop-heading-error-deg")
    if args.forward_stop_heading_error_deg >= args.rotate_start_heading_error_deg:
        parser.error(
            "--forward-stop-heading-error-deg must be < "
            "--rotate-start-heading-error-deg"
        )
    if args.hard_stop_range_m >= args.min_scan_range_m:
        parser.error("--hard-stop-range-m must be < --min-scan-range-m")
    if args.hard_stop_range_m >= args.rotation_stop_range_m:
        parser.error("--hard-stop-range-m must be < --rotation-stop-range-m")
    if not (0.0 < args.scan_half_angle_deg <= 90.0):
        parser.error("--scan-half-angle-deg must be > 0 and <= 90")
    if args.settle_sec < 0.0:
        parser.error("--settle-sec must be non-negative")
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
    if args.run_local_map_sparse_retry_count < 0:
        parser.error("--run-local-map-sparse-retry-count must be >= 0")
    if args.run_local_map_sparse_retry_angle_window_deg > 90.0:
        parser.error("--run-local-map-sparse-retry-angle-window-deg must be <= 90")
    if args.obstacle_min_cluster_size < 1:
        parser.error("--obstacle-min-cluster-size must be >= 1")
    if args.pure_pursuit_lookahead_guard_static_inflation_radius_m < 0.0:
        parser.error(
            "--pure-pursuit-lookahead-guard-static-inflation-radius-m "
            "must be non-negative"
        )
    non_negative_fields = [
        "pure_pursuit_max_linear_accel_mps2",
        "pure_pursuit_max_linear_decel_mps2",
        "pure_pursuit_max_angular_accel_radps2",
        "pure_pursuit_max_angular_decel_radps2",
        "pure_pursuit_min_smoothed_linear_speed_mps",
        "pure_pursuit_min_curvature_linear_speed_mps",
        "pure_pursuit_heading_deadband_deg",
        "pure_pursuit_lateral_deadband_m",
    ]
    for field in non_negative_fields:
        if getattr(args, field) < 0.0:
            parser.error(f"--{field.replace('_', '-')} must be non-negative")
    if not (0.0 < args.pure_pursuit_turn_speed_margin <= 1.0):
        parser.error("--pure-pursuit-turn-speed-margin must be > 0 and <= 1")
    if not (
        args.pure_pursuit_heading_deadband_deg
        < args.pure_pursuit_curvature_limit_start_heading_error_deg
        < args.pure_pursuit_curvature_limit_full_heading_error_deg
    ):
        parser.error(
            "--pure-pursuit-heading-deadband-deg must be < "
            "--pure-pursuit-curvature-limit-start-heading-error-deg must be < "
            "--pure-pursuit-curvature-limit-full-heading-error-deg"
        )
    if (
        args.pure_pursuit_curvature_limit_full_heading_error_deg
        >= args.pure_pursuit_rotate_start_heading_error_deg
    ):
        parser.error(
            "--pure-pursuit-curvature-limit-full-heading-error-deg must be < "
            "--pure-pursuit-rotate-start-heading-error-deg"
        )
    if (
        args.pure_pursuit_rotate_stop_heading_error_deg
        >= args.pure_pursuit_rotate_start_heading_error_deg
    ):
        parser.error(
            "--pure-pursuit-rotate-stop-heading-error-deg must be < "
            "--pure-pursuit-rotate-start-heading-error-deg"
        )
    if args.pure_pursuit_min_smoothed_linear_speed_mps > args.linear_speed:
        parser.error(
            "--pure-pursuit-min-smoothed-linear-speed-mps must be <= "
            "--linear-speed"
        )
    if args.pure_pursuit_min_curvature_linear_speed_mps > args.linear_speed:
        parser.error(
            "--pure-pursuit-min-curvature-linear-speed-mps must be <= "
            "--linear-speed"
        )
    if args.pure_pursuit_final_decel_distance_m <= args.goal_tolerance_m:
        parser.error(
            "--pure-pursuit-final-decel-distance-m must be greater than "
            "--goal-tolerance-m"
        )


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    try:
        raw_waypoints = load_waypoints(args.waypoints)
        preview_waypoints = prepare_executable_waypoints(
            raw_waypoints,
            skip_first=not args.no_skip_first_waypoint,
            min_spacing_m=args.min_waypoint_spacing_m,
        )
        preview_tracking_points, preview_tracking_validation = prepare_tracking_setup(
            args,
            raw_waypoints,
        )
        preview_lookahead_guard = build_lookahead_guard(args)
    except Exception as exc:
        print(f"follow_planned_waypoints.py: error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print_dry_run(
            args,
            raw_waypoints,
            preview_waypoints,
            tracking_validation=preview_tracking_validation,
            lookahead_guard=preview_lookahead_guard,
        )
        return 0

    if not require_motion_confirmation(args, preview_waypoints):
        print("Waypoint following cancelled.")
        return 130

    if rclpy is None:
        print(
            "ROS 2 Python modules are unavailable. Source ROS 2 Humble before running.",
            file=sys.stderr,
        )
        return 2

    rclpy.init()
    node = WaypointFollower(args)
    status = "failed"
    notes = args.notes
    reached_count = 0
    executable_waypoints = preview_waypoints
    tracking_points = preview_tracking_points
    tracking_validation = preview_tracking_validation
    start_pose = None
    final_pose = None
    blocked_waypoint = None
    timeout_waypoint = None
    scan_safety = None
    amcl_health = None
    return_code = 1

    try:
        node.wait_for_startup_gate()
        start_pose, _frame, amcl_health = node.check_health_or_recover()
        start_selection = select_executable_waypoints(
            raw_waypoints,
            start_pose,
            args.start_selection,
            args.start_on_path_tolerance_m,
            args.waypoint_tolerance_m,
            args.goal_tolerance_m,
            args.min_waypoint_spacing_m,
            skip_first=not args.no_skip_first_waypoint,
        )
        executable_waypoints = start_selection.waypoints
        tracking_points, tracking_validation = prepare_tracking_setup(
            args,
            raw_waypoints,
            current_pose=start_pose,
            logger=node.get_logger(),
        )
        node.diagnostics.selected_start_segment_index = (
            start_selection.selected_segment_index
        )
        node.diagnostics.selected_start_waypoint_index = (
            start_selection.selected_waypoint_index
        )
        node.diagnostics.distance_to_path_m = start_selection.distance_to_path_m
        if args.verbose:
            node.get_logger().info(
                "Selected executable route: "
                f"segment={start_selection.selected_segment_index}, "
                f"first_waypoint={start_selection.selected_waypoint_index}, "
                f"distance_to_path={start_selection.distance_to_path_m}"
            )
        node.publish_rviz_route(executable_waypoints, current_pose=start_pose)
        node.publish_rviz_obstacles()
        if not wait_before_follow_confirmation(args, start_pose, executable_waypoints):
            status = "interrupted"
            notes = f"{args.notes};wait_before_follow_cancelled"
            notes = notes_with_velocity_scheduler_metadata(notes, args)
            notes = notes_with_smoothing_metadata(notes, args)
            node.diagnostics.final_status_reason = "wait_before_follow_cancelled"
            print("Waypoint following cancelled before custom follower start.")
            return_code = 130
        else:
            if args.wait_before_follow:
                node.refresh_after_operator_wait(time.time())
            result = node.follow_waypoints(
                executable_waypoints,
                tracking_points=tracking_points,
                tracking_validation=tracking_validation,
            )
            reached_count = result["reached_count"]
            start_pose = result["start_pose"]
            final_pose = result["final_pose"]
            scan_safety = result["scan_safety"]
            amcl_health = result["amcl_health"]
            status = result.get("status", "completed")
            notes = notes_with_tracking_metadata(notes, args, tracking_validation)
            notes = notes_with_velocity_scheduler_metadata(notes, args)
            notes = notes_with_smoothing_metadata(notes, args)
            notes = notes_with_guard_metadata(
                notes,
                args,
                getattr(node, "last_lookahead_guard_result", None),
            )
            node.diagnostics.final_status_reason = status
            return_code = 0

    except KeyboardInterrupt:
        status = "interrupted"
        notes = f"{args.notes};keyboard_interrupt"
        notes = notes_with_velocity_scheduler_metadata(notes, args)
        notes = notes_with_smoothing_metadata(notes, args)
        notes = notes_with_guard_metadata(
            notes,
            args,
            getattr(node, "last_lookahead_guard_result", None),
        )
        node.diagnostics.final_status_reason = "keyboard_interrupt"
        print("Interrupted. Sending stop command...")
        return_code = 130

    except BlockedByScanError as exc:
        status = "blocked"
        notes = f"{args.notes};{exc}"
        notes = notes_with_velocity_scheduler_metadata(notes, args)
        notes = notes_with_smoothing_metadata(notes, args)
        notes = notes_with_guard_metadata(
            notes,
            args,
            getattr(node, "last_lookahead_guard_result", None),
        )
        node.diagnostics.final_status_reason = str(exc)
        reached_count = node.reached_count
        start_pose = node.start_pose
        final_pose = node.final_pose
        amcl_health = node.last_amcl_health
        scan_safety = exc.scan_safety
        blocked_waypoint = exc.waypoint
        node.get_logger().error(str(exc))
        return_code = 1

    except WaypointTimeoutError as exc:
        status = "timeout"
        notes = f"{args.notes};{exc}"
        notes = notes_with_velocity_scheduler_metadata(notes, args)
        notes = notes_with_smoothing_metadata(notes, args)
        notes = notes_with_guard_metadata(
            notes,
            args,
            getattr(node, "last_lookahead_guard_result", None),
        )
        node.diagnostics.final_status_reason = str(exc)
        reached_count = node.reached_count
        start_pose = node.start_pose
        final_pose = node.final_pose
        scan_safety = node.last_scan_safety
        amcl_health = node.last_amcl_health
        timeout_waypoint = exc.waypoint
        node.get_logger().error(str(exc))
        return_code = 1

    except Exception as exc:
        status = "failed"
        notes = f"{args.notes};{exc}"
        notes = notes_with_velocity_scheduler_metadata(notes, args)
        notes = notes_with_smoothing_metadata(notes, args)
        notes = notes_with_guard_metadata(
            notes,
            args,
            getattr(node, "last_lookahead_guard_result", None),
        )
        node.diagnostics.final_status_reason = str(exc)
        reached_count = node.reached_count
        start_pose = node.start_pose
        final_pose = node.final_pose
        scan_safety = node.last_scan_safety
        amcl_health = node.last_amcl_health
        node.get_logger().error(str(exc))
        return_code = 1

    finally:
        try:
            node.stop_repeatedly()
            if final_pose is None:
                try:
                    final_pose, _frame = node.lookup_pose()
                except Exception:
                    final_pose = None
        finally:
            if not args.no_log:
                try:
                    row = build_log_row(
                        args,
                        len(executable_waypoints),
                        reached_count,
                        status,
                        notes,
                        start_pose=start_pose,
                        final_pose=final_pose,
                        blocked_waypoint=blocked_waypoint,
                        timeout_waypoint=timeout_waypoint,
                        base_frame_used=node.base_frame_used,
                        scan_safety=scan_safety,
                        amcl_health=amcl_health,
                        diagnostics=node.diagnostics,
                    )
                    append_csv_row(args.results_csv, CSV_HEADER, row)
                    node.get_logger().info(f"Saved run log to {args.results_csv}")
                except Exception as log_exc:
                    print(f"Could not write waypoint-follow log: {log_exc}", file=sys.stderr)
            node.destroy_node()
            rclpy.shutdown()

    return return_code


if __name__ == "__main__":
    sys.exit(main())
