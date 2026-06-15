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
    from nav_msgs.msg import Odometry
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
    Odometry = object
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
    FORWARD_CONTROL_MODES,
    FORWARD_CONTROL_ROUTE_DAMPED,
    FORWARD_CONTROL_TARGET_BEARING,
    PATH_PROFILE_SCHEDULING_MODES,
    PATH_PROFILE_SCHEDULING_OFF,
    PATH_PROFILE_SCHEDULING_ON,
    PATH_PROFILE_STATUS_APPROACH_BEND,
    PATH_PROFILE_STATUS_BASE,
    PATH_PROFILE_STATUS_FORCE_ROTATE_HANDOFF,
    PATH_PROFILE_STATUS_FORCE_ROTATE_PENDING,
    PATH_PROFILE_STATUS_OFF,
    PATH_PROFILE_STATUS_SHORT_SEGMENT,
    PATH_PROFILE_STATUS_STRAIGHT_FAST,
    POST_ROTATE_BRANCH_END_TOLERANCE_M,
    POST_ROTATE_BRANCH_END_LATERAL_TOLERANCE_M,
    POST_ROTATE_BRANCH_HEADING_TOLERANCE_DEG,
    POST_ROTATE_BRANCH_MIN_RELEASE_PROGRESS_M,
    POST_ROTATE_BRANCH_RELEASE_STABLE_SAMPLES,
    POST_ROTATE_ZERO_LINEAR_EPS_MPS,
    ROTATE_ANCHOR_ROUTE_HEADING_EXIT_SAMPLES,
    PathProfileScheduleResult,
    PathController,
    PROJECTION_LOCK_PROGRESS_TOLERANCE_M,
    PROJECTION_LOCK_REQUIRED_SAMPLES,
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
    BranchCompatiblePath,
    PATH_SEGMENT_EPS_M,
    ROUTE_HEADING_LOOKAHEAD_M,
    RouteHeading,
    RouteProjection,
    branch_compatible_path_from_projection,
    lookahead_target_from_route_anchor,
    polyline_lookahead_target,
    project_point_to_route,
    project_point_to_route_branch_window,
    project_point_to_route_progress_window,
    pure_pursuit_curve_command,
    route_cumulative_distances,
    route_heading_at_progress,
    route_heading_from_projection,
    route_points_from_projection,
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
from waypoint_following import controller_runtime  # noqa: E402
from waypoint_following import facade_contexts  # noqa: E402
from waypoint_following import follower_defaults  # noqa: E402
from waypoint_following import follower_cli  # noqa: E402
from waypoint_following import follow_loop  # noqa: E402
from waypoint_following import node_setup  # noqa: E402
from waypoint_following import replanning  # noqa: E402
from waypoint_following.replanning import ReplanManager  # noqa: E402
from waypoint_following import post_replan_recovery  # noqa: E402
from waypoint_following import ros_node_wiring  # noqa: E402
from waypoint_following import ros_runtime  # noqa: E402
from waypoint_following import run_logging  # noqa: E402
from waypoint_following import rviz_visualization  # noqa: E402
from waypoint_following import run_session  # noqa: E402
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
    evaluate_scan_safety,
    percentile,
    valid_scan_ranges,
)
from waypoint_following.follower_defaults import (  # noqa: E402
    DEFAULT_WAYPOINTS_CSV,
    DEFAULT_RESULTS_CSV,
    DEFAULT_STATIC_MAP,
    DEFAULT_REPLAN_OUTPUT_DIR,
    DEFAULT_RVIZ_PATH_TOPIC,
    DEFAULT_RVIZ_WAYPOINT_MARKER_TOPIC,
    DEFAULT_RVIZ_OBSTACLE_MARKER_TOPIC,
    DEFAULT_CMD_VEL_TOPIC,
    DEFAULT_SCAN_TOPIC,
    DEFAULT_AMCL_TOPIC,
    DEFAULT_ODOM_TOPIC,
    DEFAULT_MAX_ODOM_AGE_SEC,
    DEFAULT_LINEAR_SPEED_MPS,
    DEFAULT_PURE_PURSUIT_LINEAR_SPEED_MPS,
    DEFAULT_MIN_LINEAR_SPEED_MPS,
    DEFAULT_LINEAR_GAIN,
    DEFAULT_MAX_ANGULAR_SPEED_RADPS,
    DEFAULT_PURE_PURSUIT_MAX_ANGULAR_SPEED_RADPS,
    DEFAULT_YAW_GAIN,
    DEFAULT_WAYPOINT_TOLERANCE_M,
    DEFAULT_GOAL_TOLERANCE_M,
    DEFAULT_ROTATE_START_HEADING_ERROR_DEG,
    DEFAULT_ROTATE_STOP_HEADING_ERROR_DEG,
    DEFAULT_FORWARD_YAW_DEADBAND_DEG,
    DEFAULT_FORWARD_STOP_HEADING_ERROR_DEG,
    DEFAULT_MIN_WAYPOINT_SPACING_M,
    DEFAULT_START_SELECTION,
    DEFAULT_START_ON_PATH_TOLERANCE_M,
    DEFAULT_CONTROLLER,
    DEFAULT_PATH_LOOKAHEAD_M,
    DEFAULT_PURE_PURSUIT_LOOKAHEAD_GUARD,
    DEFAULT_PURE_PURSUIT_MIN_GUARDED_LOOKAHEAD_M,
    DEFAULT_PURE_PURSUIT_LOOKAHEAD_GUARD_STATIC_INFLATION_RADIUS_M,
    DEFAULT_PURE_PURSUIT_COMMAND_SMOOTHING,
    DEFAULT_PURE_PURSUIT_MAX_LINEAR_ACCEL_MPS2,
    DEFAULT_PURE_PURSUIT_MAX_LINEAR_DECEL_MPS2,
    DEFAULT_PURE_PURSUIT_MAX_ANGULAR_ACCEL_RADPS2,
    DEFAULT_PURE_PURSUIT_MAX_ANGULAR_DECEL_RADPS2,
    DEFAULT_PURE_PURSUIT_FINAL_DECEL_DISTANCE_M,
    DEFAULT_PURE_PURSUIT_SPEED_PROFILE,
    DEFAULT_PURE_PURSUIT_FORWARD_CONTROL,
    DEFAULT_PURE_PURSUIT_PATH_PROFILE_SCHEDULING,
    DEFAULT_PURE_PURSUIT_PATH_PROFILE_STRAIGHT_SPEED_MPS,
    DEFAULT_PURE_PURSUIT_PATH_PROFILE_SHORT_SPEED_CAP_MPS,
    DEFAULT_PURE_PURSUIT_PATH_PROFILE_BEND_SPEED_CAP_MPS,
    DEFAULT_PURE_PURSUIT_ROUTE_HEADING_BLEND,
    DEFAULT_PURE_PURSUIT_CROSS_TRACK_GAIN,
    DEFAULT_PURE_PURSUIT_CROSS_TRACK_SPEED_FLOOR_MPS,
    DEFAULT_PURE_PURSUIT_MAX_CROSS_TRACK_CORRECTION_DEG,
    DEFAULT_PURE_PURSUIT_ANGULAR_FEASIBILITY_SPEED_LIMIT,
    DEFAULT_PURE_PURSUIT_ANGULAR_FEASIBILITY_MARGIN,
    DEFAULT_PURE_PURSUIT_MAX_LATERAL_ACCEL_MPS2,
    DEFAULT_PURE_PURSUIT_TURN_SPEED_MARGIN,
    DEFAULT_PURE_PURSUIT_ROTATE_START_HEADING_ERROR_DEG,
    DEFAULT_PURE_PURSUIT_ROTATE_STOP_HEADING_ERROR_DEG,
    DEFAULT_PURE_PURSUIT_ROUTE_HEADING_ROTATE_START_DEG,
    DEFAULT_PURE_PURSUIT_ROUTE_HEADING_ROTATE_STOP_DEG,
    DEFAULT_PURE_PURSUIT_MAX_TRACK_ANGULAR_SPEED_RADPS,
    DEFAULT_PURE_PURSUIT_HEADING_DEADBAND_DEG,
    DEFAULT_PURE_PURSUIT_LATERAL_DEADBAND_M,
    DEFAULT_PURE_PURSUIT_CURVATURE_LIMIT_START_HEADING_ERROR_DEG,
    DEFAULT_PURE_PURSUIT_CURVATURE_LIMIT_FULL_HEADING_ERROR_DEG,
    DEFAULT_PURE_PURSUIT_CROSS_TRACK_WARNING_M,
    DEFAULT_PURE_PURSUIT_MAX_CROSS_TRACK_ERROR_M,
    DEFAULT_PURE_PURSUIT_TRACKING_PROGRESS_TOLERANCE_M,
    DEFAULT_TRACKING_ENDPOINT_TOLERANCE_M,
    DEFAULT_TRACKING_START_TOLERANCE_M,
    DEFAULT_TRACKING_MAX_SEGMENT_M,
    DEFAULT_ODOM_FRAME,
    DEFAULT_SCAN_HALF_ANGLE_DEG,
    DEFAULT_HARD_STOP_RANGE_M,
    DEFAULT_MIN_SCAN_RANGE_M,
    DEFAULT_ROTATION_STOP_RANGE_M,
    FORWARD_SOFT_STOP_MIN_CLOSE_RANGES,
    DEFAULT_MAX_POSE_AGE_SEC,
    DEFAULT_MAX_SCAN_AGE_SEC,
    DEFAULT_MAX_AMCL_AGE_SEC,
    DEFAULT_MAX_AMCL_VAR_X,
    DEFAULT_MAX_AMCL_VAR_Y,
    DEFAULT_MAX_AMCL_VAR_YAW,
    DEFAULT_MAX_WAYPOINT_TIME_SEC,
    DEFAULT_MAX_TF_UPDATE_GAP_SEC,
    DEFAULT_TF_RECOVERY_TIME_SEC,
    DEFAULT_LOCALIZATION_RECOVERY_TIME_SEC,
    DEFAULT_CONTROL_RATE_HZ,
    DEFAULT_SETTLE_SEC,
    DEFAULT_REPLAN_TIMEOUT_SEC,
    DEFAULT_MAX_REPLAN_SCAN_AGE_SEC,
    DEFAULT_MAX_REPLAN_TF_AGE_SEC,
    DEFAULT_OBSTACLE_FORWARD_DISTANCE_M,
    DEFAULT_OBSTACLE_FORWARD_HALF_WIDTH_M,
    DEFAULT_OBSTACLE_ANGLE_WINDOW_DEG,
    DEFAULT_OBSTACLE_MIN_RANGE_M,
    DEFAULT_ROBOT_FOOTPRINT_RADIUS_M,
    DEFAULT_OBSTACLE_MIN_CLUSTER_SIZE,
    DEFAULT_OBSTACLE_MIN_CLUSTER_WIDTH_M,
    DEFAULT_OBSTACLE_INFLATE_RADIUS_M,
    DEFAULT_MAX_START_SNAP_M,
    DEFAULT_MAX_GOAL_SNAP_M,
    DEFAULT_MAX_REPLAN_PATH_LENGTH_RATIO,
    DEFAULT_RUN_LOCAL_MAP_INITIAL_SCAN_MODE,
    DEFAULT_RUN_LOCAL_MAP_INITIAL_SCAN_COUNT,
    DEFAULT_RUN_LOCAL_MAP_UPDATE_MODE,
    DEFAULT_RUN_LOCAL_MAP_MIN_HIT_COUNT,
    DEFAULT_RUN_LOCAL_MAP_INFLATION_RADIUS_M,
    DEFAULT_RUN_LOCAL_MAP_MAX_TF_AGE_SEC,
    DEFAULT_RUN_LOCAL_MAP_MAX_SCAN_AGE_SEC,
    DEFAULT_RUN_LOCAL_MAP_MIN_USED_POINTS,
    DEFAULT_RUN_LOCAL_MAP_MAX_REJECTED_RATIO,
    DEFAULT_RUN_LOCAL_MAP_CORRIDOR_CHECK_DISTANCE_M,
    DEFAULT_RUN_LOCAL_MAP_CLEARANCE_MARGIN_M,
    DEFAULT_RUN_LOCAL_MAP_MAX_UPDATES,
    DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_COUNT,
    DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_FORWARD_HALF_WIDTH_M,
    DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_ANGLE_WINDOW_DEG,
    DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_FORWARD_DISTANCE_M,
    DEFAULT_RUN_LOCAL_MAP_PRUNE_BEHIND_DISTANCE_M,
    DEFAULT_STARTUP_TIMEOUT_SEC,
    STOP_PUBLISH_COUNT,
    STOP_PUBLISH_HZ,
)

DEFAULT_POST_REPLAN_RECOVERY = post_replan_recovery.DEFAULT_POST_REPLAN_RECOVERY
POST_REPLAN_RECOVERY_MODES = post_replan_recovery.POST_REPLAN_RECOVERY_MODES
DEFAULT_POST_REPLAN_CLEARANCE_MODE = (
    post_replan_recovery.DEFAULT_POST_REPLAN_CLEARANCE_MODE
)
POST_REPLAN_CLEARANCE_MODES = post_replan_recovery.POST_REPLAN_CLEARANCE_MODES
DEFAULT_POST_REPLAN_ROUTE_CLEARANCE_PREVIEW_DISTANCE_M = (
    post_replan_recovery.DEFAULT_POST_REPLAN_ROUTE_CLEARANCE_PREVIEW_DISTANCE_M
)
post_replan_route_clearance_preview_distance_m = (
    post_replan_recovery.post_replan_route_clearance_preview_distance_m
)
DEFAULT_POST_REPLAN_CLEAR_SCAN_SAMPLES = (
    post_replan_recovery.DEFAULT_POST_REPLAN_CLEAR_SCAN_SAMPLES
)
DEFAULT_POST_REPLAN_TIMEOUT_SEC = post_replan_recovery.DEFAULT_POST_REPLAN_TIMEOUT_SEC
DEFAULT_POST_REPLAN_ESCAPE_DISTANCE_M = (
    post_replan_recovery.DEFAULT_POST_REPLAN_ESCAPE_DISTANCE_M
)
DEFAULT_POST_REPLAN_ESCAPE_LINEAR_SPEED_MPS = (
    post_replan_recovery.DEFAULT_POST_REPLAN_ESCAPE_LINEAR_SPEED_MPS
)
DEFAULT_POST_REPLAN_ALIGN_HEADING_ERROR_DEG = (
    post_replan_recovery.DEFAULT_POST_REPLAN_ALIGN_HEADING_ERROR_DEG
)
POST_REPLAN_MIN_ROUTE_SEGMENT_M = post_replan_recovery.POST_REPLAN_MIN_ROUTE_SEGMENT_M
POST_REPLAN_ROUTE_HEADING_LOOKAHEAD_M = (
    post_replan_recovery.POST_REPLAN_ROUTE_HEADING_LOOKAHEAD_M
)
POST_REPLAN_CLEARANCE_MAX_YAW_DEG = (
    post_replan_recovery.POST_REPLAN_CLEARANCE_MAX_YAW_DEG
)
POST_REPLAN_CLEARANCE_IMPROVEMENT_M = (
    post_replan_recovery.POST_REPLAN_CLEARANCE_IMPROVEMENT_M
)
POST_REPLAN_CLEARANCE_MAX_ANGULAR_RADPS = (
    post_replan_recovery.POST_REPLAN_CLEARANCE_MAX_ANGULAR_RADPS
)
POST_REPLAN_CLEARANCE_SIDE_DIFF_M = (
    post_replan_recovery.POST_REPLAN_CLEARANCE_SIDE_DIFF_M
)
POST_REPLAN_ESCAPE_COMPLETION_TOLERANCE_M = (
    post_replan_recovery.POST_REPLAN_ESCAPE_COMPLETION_TOLERANCE_M
)
POST_REPLAN_ESCAPE_TIMEOUT_MARGIN_SEC = (
    post_replan_recovery.POST_REPLAN_ESCAPE_TIMEOUT_MARGIN_SEC
)
POST_REPLAN_ESCAPE_MIN_TIMEOUT_SEC = (
    post_replan_recovery.POST_REPLAN_ESCAPE_MIN_TIMEOUT_SEC
)
POST_REPLAN_ESCAPE_ANGULAR_HINT_CAP_RADPS = (
    post_replan_recovery.POST_REPLAN_ESCAPE_ANGULAR_HINT_CAP_RADPS
)
POST_REPLAN_ESCAPE_STRAIGHT_UNTIL_PROGRESS_M = (
    post_replan_recovery.POST_REPLAN_ESCAPE_STRAIGHT_UNTIL_PROGRESS_M
)
POST_REPLAN_ESCAPE_NO_MOTION_EPS_M = (
    post_replan_recovery.POST_REPLAN_ESCAPE_NO_MOTION_EPS_M
)
POST_REPLAN_ESCAPE_NO_MOTION_TIMEOUT_ODOM_SEC = (
    post_replan_recovery.POST_REPLAN_ESCAPE_NO_MOTION_TIMEOUT_ODOM_SEC
)
POST_REPLAN_ESCAPE_NO_MOTION_TIMEOUT_MAP_SEC = (
    post_replan_recovery.POST_REPLAN_ESCAPE_NO_MOTION_TIMEOUT_MAP_SEC
)
DEFAULT_POST_REPLAN_ESCAPE_STEERING_MODE = (
    post_replan_recovery.DEFAULT_POST_REPLAN_ESCAPE_STEERING_MODE
)
POST_REPLAN_ESCAPE_STEERING_MODES = (
    post_replan_recovery.POST_REPLAN_ESCAPE_STEERING_MODES
)
resolve_post_replan_escape_steering_mode = (
    post_replan_recovery.resolve_post_replan_escape_steering_mode
)
POST_REPLAN_ACTIVATION_MIN_TARGET_FLOOR_M = (
    post_replan_recovery.POST_REPLAN_ACTIVATION_MIN_TARGET_FLOOR_M
)
POST_REPLAN_RECOVERY_ALIGN = post_replan_recovery.POST_REPLAN_RECOVERY_ALIGN
POST_REPLAN_RECOVERY_CLEARANCE_SEARCH = (
    post_replan_recovery.POST_REPLAN_RECOVERY_CLEARANCE_SEARCH
)
POST_REPLAN_RECOVERY_WAIT_CLEAR = post_replan_recovery.POST_REPLAN_RECOVERY_WAIT_CLEAR
POST_REPLAN_RECOVERY_ESCAPE = post_replan_recovery.POST_REPLAN_RECOVERY_ESCAPE
POST_REPLAN_RECOVERY_DONE = post_replan_recovery.POST_REPLAN_RECOVERY_DONE
POST_REPLAN_PRE_CONTROLLER_RECOVERY_PHASES = (
    post_replan_recovery.POST_REPLAN_PRE_CONTROLLER_RECOVERY_PHASES
)
PostReplanRecoveryState = post_replan_recovery.PostReplanRecoveryState
PostReplanAlignmentHeading = post_replan_recovery.PostReplanAlignmentHeading
post_replan_recovery_should_preempt_controller = (
    post_replan_recovery.post_replan_recovery_should_preempt_controller
)

INITIAL_RUN_LOCAL_MAP_NONFATAL_REASONS = (
    replanning.INITIAL_RUN_LOCAL_MAP_NONFATAL_REASONS
)
REPLAN_TRIGGER_SCAN_BLOCKAGE = replanning.REPLAN_TRIGGER_SCAN_BLOCKAGE
REPLAN_TRIGGER_KNOWN_CORRIDOR = replanning.REPLAN_TRIGGER_KNOWN_CORRIDOR
REPLAN_TRIGGER_LOOKAHEAD_GUARD = replanning.REPLAN_TRIGGER_LOOKAHEAD_GUARD
PostReplanActivationRoute = replanning.PostReplanActivationRoute


def run_local_map_has_confirmed_obstacles(run_local_map):
    return replanning.run_local_map_has_confirmed_obstacles(run_local_map)


def lidar_replan_failure(reason):
    return replanning.lidar_replan_failure(reason)


def warn_logger(logger, message):
    return controller_runtime.warn_logger(
        logger,
        message,
        _controller_runtime_context(),
    )


def build_sparse_tracking_validation(source, point_count, status):
    return controller_runtime.build_sparse_tracking_validation(
        source,
        point_count,
        status,
        _controller_runtime_context(),
    )


def prepare_tracking_setup(
    args,
    route_waypoints,
    current_pose=None,
    logger=None,
    structural_only=False,
):
    return controller_runtime.prepare_tracking_setup(
        args,
        route_waypoints,
        _controller_runtime_context(),
        current_pose=current_pose,
        logger=logger,
        structural_only=structural_only,
    )


def format_optional_m(value):
    return controller_runtime.format_optional_m(value, _controller_runtime_context())


def notes_with_tracking_metadata(notes, args, tracking_validation):
    return controller_runtime.notes_with_tracking_metadata(
        notes,
        args,
        tracking_validation,
        _controller_runtime_context(),
    )


def build_lookahead_guard(args, run_local_map_fn=None):
    return controller_runtime.build_lookahead_guard(
        args,
        _controller_runtime_context(),
        run_local_map_fn=run_local_map_fn,
    )


def command_smoothing_active(args):
    return controller_runtime.command_smoothing_active(
        args,
        _controller_runtime_context(),
    )


def build_command_smoother(args):
    return controller_runtime.build_command_smoother(
        args,
        _controller_runtime_context(),
    )


def reset_command_smoother(node):
    return controller_runtime.reset_command_smoother(
        node,
        _controller_runtime_context(),
    )


def reset_route_projection_controller(controller):
    return controller_runtime.reset_route_projection_controller(
        controller,
        _controller_runtime_context(),
    )


def smoothing_dt_sec(node, now_sec):
    return controller_runtime.smoothing_dt_sec(
        node,
        now_sec,
        _controller_runtime_context(),
    )


def smoothed_step_command(node, step, now_sec):
    return controller_runtime.smoothed_step_command(
        node,
        step,
        now_sec,
        _controller_runtime_context(),
    )


def notes_with_smoothing_metadata(notes, args):
    return controller_runtime.notes_with_smoothing_metadata(
        notes,
        args,
        _controller_runtime_context(),
    )


def notes_with_velocity_scheduler_metadata(notes, args):
    return controller_runtime.notes_with_velocity_scheduler_metadata(
        notes,
        args,
        _controller_runtime_context(),
    )


def notes_with_route_projection_metadata(notes, args, node):
    return controller_runtime.notes_with_route_projection_metadata(
        notes,
        args,
        node,
        _controller_runtime_context(),
    )


def notes_with_guard_metadata(notes, args, guard_result):
    return controller_runtime.notes_with_guard_metadata(
        notes,
        args,
        guard_result,
        _controller_runtime_context(),
    )


def post_replan_recovery_active_for_args(args):
    return post_replan_recovery.post_replan_recovery_active_for_args(
        args,
        default_controller=DEFAULT_CONTROLLER,
    )


def notes_with_post_replan_recovery_metadata(notes, args, node):
    return post_replan_recovery.notes_with_post_replan_recovery_metadata(
        notes,
        args,
        node,
    )

RVIZ_COLOR_PATH = rviz_visualization.RVIZ_COLOR_PATH
RVIZ_COLOR_WAYPOINT = rviz_visualization.RVIZ_COLOR_WAYPOINT
RVIZ_COLOR_CURRENT = rviz_visualization.RVIZ_COLOR_CURRENT
RVIZ_COLOR_GOAL = rviz_visualization.RVIZ_COLOR_GOAL
RVIZ_COLOR_LABEL = rviz_visualization.RVIZ_COLOR_LABEL
RVIZ_COLOR_CONFIRMED_OBSTACLE = rviz_visualization.RVIZ_COLOR_CONFIRMED_OBSTACLE
RVIZ_COLOR_INFLATED_OBSTACLE = rviz_visualization.RVIZ_COLOR_INFLATED_OBSTACLE
RVIZ_COLOR_BLOCKED_CORRIDOR = rviz_visualization.RVIZ_COLOR_BLOCKED_CORRIDOR


def _rviz_message_types():
    return facade_contexts.rviz_message_types(globals())


def rviz_messages_available():
    return rviz_visualization.rviz_messages_available(_rviz_message_types())


def rviz_qos_profile():
    return rviz_visualization.rviz_qos_profile(_rviz_message_types())


def set_header(message, frame_id, stamp):
    return rviz_visualization.set_header(message, frame_id, stamp)


def set_pose_xy(pose, x, y, z=0.0):
    return rviz_visualization.set_pose_xy(pose, x, y, z)


def point_msg(x, y, z=0.0):
    return rviz_visualization.point_msg(_rviz_message_types(), x, y, z)


def set_marker_color(marker, color):
    return rviz_visualization.set_marker_color(marker, color)


def marker_delete_all(frame_id, stamp):
    return rviz_visualization.marker_delete_all(
        _rviz_message_types(),
        frame_id,
        stamp,
    )


def apply_marker_common(marker, frame_id, stamp, namespace, marker_id, marker_type, color):
    return rviz_visualization.apply_marker_common(
        _rviz_message_types(),
        marker,
        frame_id,
        stamp,
        namespace,
        marker_id,
        marker_type,
        color,
    )


def build_pose_stamped(frame_id, stamp, x, y):
    return rviz_visualization.build_pose_stamped(
        _rviz_message_types(),
        frame_id,
        stamp,
        x,
        y,
    )


def build_rviz_path_message(waypoints, frame_id, stamp, current_pose=None):
    return rviz_visualization.build_rviz_path_message(
        _rviz_message_types(),
        waypoints,
        frame_id,
        stamp,
        current_pose=current_pose,
    )


def waypoint_point(waypoint, z=0.04):
    return rviz_visualization.waypoint_point(_rviz_message_types(), waypoint, z)


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
    return rviz_visualization.build_point_layer_marker(
        _rviz_message_types(),
        frame_id,
        stamp,
        namespace,
        marker_id,
        marker_type,
        points,
        color,
        scale_m,
    )


def build_single_waypoint_marker(frame_id, stamp, namespace, marker_id, waypoint, color, scale_m, z):
    return rviz_visualization.build_single_waypoint_marker(
        _rviz_message_types(),
        frame_id,
        stamp,
        namespace,
        marker_id,
        waypoint,
        color,
        scale_m,
        z,
    )


def build_waypoint_label_marker(frame_id, stamp, marker_id, waypoint):
    return rviz_visualization.build_waypoint_label_marker(
        _rviz_message_types(),
        frame_id,
        stamp,
        marker_id,
        waypoint,
    )


def build_rviz_waypoint_markers(waypoints, frame_id, stamp, current_waypoint_index=0):
    return rviz_visualization.build_rviz_waypoint_markers(
        _rviz_message_types(),
        waypoints,
        frame_id,
        stamp,
        current_waypoint_index=current_waypoint_index,
    )


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
    return rviz_visualization.build_cell_layer_marker(
        _rviz_message_types(),
        lidar_obstacle_map.planner.grid_to_world,
        run_local_map,
        frame_id,
        stamp,
        namespace,
        marker_id,
        cells,
        color,
        z,
        height_m,
    )


def append_marker(markers, marker):
    return rviz_visualization.append_marker(markers, marker)


def build_rviz_obstacle_markers(run_local_map, frame_id, stamp, blocked_cells=None):
    return rviz_visualization.build_rviz_obstacle_markers(
        _rviz_message_types(),
        lidar_obstacle_map.planner.grid_to_world,
        run_local_map,
        frame_id,
        stamp,
        blocked_cells=blocked_cells,
    )


def publish_rviz_route_if_available(
    node,
    waypoints,
    current_pose=None,
    current_waypoint_index=0,
):
    return rviz_visualization.publish_rviz_route_if_available(
        node,
        waypoints,
        current_pose=current_pose,
        current_waypoint_index=current_waypoint_index,
    )


def publish_rviz_obstacles_if_available(node, blocked_cells=None):
    return rviz_visualization.publish_rviz_obstacles_if_available(
        node,
        blocked_cells=blocked_cells,
    )


def _rviz_node_context():
    return facade_contexts.rviz_node_context(globals())


def stamp_to_sec(stamp):
    return ros_runtime.stamp_to_sec(stamp)


def amcl_covariances(covariance):
    return ros_runtime.amcl_covariances(covariance)


def evaluate_amcl_health(
    covariance,
    age_sec,
    max_age_sec,
    max_var_x,
    max_var_y,
    max_var_yaw,
    fail_on_bad_localization=False,
):
    return ros_runtime.evaluate_amcl_health(
        covariance,
        age_sec,
        max_age_sec,
        max_var_x,
        max_var_y,
        max_var_yaw,
        fail_on_bad_localization=fail_on_bad_localization,
    )


def age_ok(age_sec, max_age_sec):
    return ros_runtime.age_ok(age_sec, max_age_sec)


def ordered_base_frames(base_frame, fallback_base_frame):
    return ros_runtime.ordered_base_frames(base_frame, fallback_base_frame)


class WaypointFollower(Node):
    def __init__(self, args):
        if rclpy is None:
            raise RuntimeError(
                "ROS 2 Python modules are unavailable. Source ROS 2 Humble before "
                "running the waypoint follower."
        )
        super().__init__("follow_planned_waypoints")
        self.args = args
        node_setup.initialize_runtime_state(self, args, _node_setup_context())
        node_setup.log_startup_configuration(self, args, _node_setup_context())
        ros_node_wiring.initialize_ros_interfaces(
            self,
            args,
            _ros_node_wiring_context(),
        )

    def rviz_visualization_enabled(self):
        return rviz_visualization.rviz_visualization_enabled(self)

    def rviz_stamp(self):
        return rviz_visualization.rviz_stamp(self)

    def publish_rviz_route(self, waypoints, current_pose=None, current_waypoint_index=0):
        return rviz_visualization.publish_rviz_route(
            self,
            waypoints,
            current_pose=current_pose,
            current_waypoint_index=current_waypoint_index,
            context=_rviz_node_context(),
        )

    def publish_rviz_obstacles(self, blocked_cells=None):
        return rviz_visualization.publish_rviz_obstacles(
            self,
            blocked_cells=blocked_cells,
            context=_rviz_node_context(),
        )

    def scan_callback(self, msg):
        return ros_runtime.scan_callback(self, _ros_runtime_context(), msg)

    def amcl_callback(self, msg):
        return ros_runtime.amcl_callback(self, _ros_runtime_context(), msg)

    def odom_callback(self, msg):
        return ros_runtime.odom_callback(self, _ros_runtime_context(), msg)

    def fresh_direct_odom_pose(self, now_sec=None):
        return ros_runtime.fresh_direct_odom_pose(
            self,
            now_sec=now_sec,
        )

    def publish_velocity(self, linear_x, angular_z):
        return ros_runtime.publish_velocity(
            self,
            _ros_runtime_context(),
            linear_x,
            angular_z,
        )

    def maybe_log_velocity_scheduler_result(self, result, now_sec):
        return controller_runtime.maybe_log_velocity_scheduler_result(
            self,
            result,
            now_sec,
            _controller_runtime_context(),
        )

    def record_route_projection_result(self, step):
        return controller_runtime.record_route_projection_result(
            self,
            step,
            _controller_runtime_context(),
        )

    def maybe_log_route_projection_result(self, step, now_sec):
        return controller_runtime.maybe_log_route_projection_result(
            self,
            step,
            now_sec,
            _controller_runtime_context(),
        )

    def stop_repeatedly(self):
        return ros_runtime.stop_repeatedly(self, _ros_runtime_context())

    def spin_once(self, timeout_sec):
        return ros_runtime.spin_once(self, _ros_runtime_context(), timeout_sec)

    def spin_for(self, duration_sec, step_sec=0.05):
        return ros_runtime.spin_for(
            self,
            _ros_runtime_context(),
            duration_sec,
            step_sec=step_sec,
        )

    def wait_for_startup_gate(self, timeout_sec=None):
        return ros_runtime.wait_for_startup_gate(
            self,
            _ros_runtime_context(),
            timeout_sec=timeout_sec,
        )

    def lookup_pose(self):
        return ros_runtime.lookup_pose(self, _ros_runtime_context())

    def lookup_odom_pose(self):
        return ros_runtime.lookup_odom_pose(self, _ros_runtime_context())

    def try_lookup_odom_pose(self):
        return ros_runtime.try_lookup_odom_pose(self, _ros_runtime_context())

    def update_tf_tracking(self, pose):
        return ros_runtime.update_tf_tracking(self, _ros_runtime_context(), pose)

    def reset_tf_tracking(self):
        return ros_runtime.reset_tf_tracking(self, _ros_runtime_context())

    def refresh_after_operator_wait(self, min_scan_stamp_sec, timeout_sec=None):
        return ros_runtime.refresh_after_operator_wait(
            self,
            _ros_runtime_context(),
            min_scan_stamp_sec,
            timeout_sec=timeout_sec,
        )

    def record_motion_sample(self, yaw_error_deg, linear_x, angular_z, sample_seconds):
        return run_logging.record_motion_sample(
            self,
            yaw_error_deg,
            linear_x,
            angular_z,
            sample_seconds,
        )

    def current_amcl_health(self):
        return ros_runtime.current_amcl_health(self, _ros_runtime_context())

    def check_health_or_raise(self):
        return ros_runtime.check_health_or_raise(self, _ros_runtime_context())

    def check_health_or_recover(self):
        return ros_runtime.check_health_or_recover(self, _ros_runtime_context())

    def check_scan_or_raise(self, mode):
        return ros_runtime.check_scan_or_raise(self, _ros_runtime_context(), mode)

    def evaluate_current_scan_safety(self, mode):
        return ros_runtime.evaluate_current_scan_safety(
            self,
            _ros_runtime_context(),
            mode,
        )

    def current_scan_identity(self):
        return post_replan_recovery.current_scan_identity(self)

    def scan_is_fresh_for_post_replan_recovery(self, recovery):
        return post_replan_recovery.scan_is_fresh_for_post_replan_recovery(
            self,
            recovery,
        )

    def scan_already_counted_for_post_replan_recovery(self, recovery):
        return post_replan_recovery.scan_already_counted_for_post_replan_recovery(
            self,
            recovery,
        )

    def reset_post_replan_recovery(self, status=""):
        return post_replan_recovery.reset_post_replan_recovery(self, status)

    def post_replan_recovery_route_points(self, route_state):
        return post_replan_recovery.post_replan_recovery_route_points(
            self,
            route_state,
        )

    def local_post_replan_alignment_heading(self, route_points, segment_index):
        return post_replan_recovery.local_post_replan_alignment_heading(
            self,
            route_points,
            segment_index,
        )

    def post_replan_alignment_heading(self, pose, route_state):
        return post_replan_recovery.post_replan_alignment_heading(
            self,
            pose,
            route_state,
        )

    def route_heading_for_post_replan_recovery(self, pose, route_state):
        return post_replan_recovery.route_heading_for_post_replan_recovery(
            self,
            pose,
            route_state,
        )

    def activate_post_replan_recovery(self, pose, route_state):
        return post_replan_recovery.activate_post_replan_recovery(
            self,
            pose,
            route_state,
        )

    def post_replan_recovery_timeout_reason(self, recovery):
        return post_replan_recovery.post_replan_recovery_timeout_reason(
            self,
            recovery,
        )

    def post_replan_recovery_timed_out(self, recovery, now_sec):
        return post_replan_recovery.post_replan_recovery_timed_out(
            self,
            recovery,
            now_sec,
        )

    def post_replan_escape_timeout_sec(self):
        return post_replan_recovery.post_replan_escape_timeout_sec(self)

    def post_replan_escape_timed_out(self, recovery, now_sec):
        return post_replan_recovery.post_replan_escape_timed_out(
            self,
            recovery,
            now_sec,
        )

    def post_replan_escape_measurement(self, recovery, pose, now_sec=None):
        return post_replan_recovery.post_replan_escape_measurement(
            self,
            recovery,
            pose,
            now_sec=now_sec,
        )

    def update_post_replan_escape_progress(self, recovery, measurement, now_sec):
        return post_replan_recovery.update_post_replan_escape_progress(
            self,
            recovery,
            measurement,
            now_sec,
        )

    def post_replan_escape_no_motion_timed_out(self, recovery, linear_x):
        return post_replan_recovery.post_replan_escape_no_motion_timed_out(
            self,
            recovery,
            linear_x,
        )

    def wait_one_control_cycle(self):
        return ros_runtime.wait_one_control_cycle(self, _ros_runtime_context())

    def maybe_log_post_replan_recovery(self, safety=None, heading_error_deg=None):
        return post_replan_recovery.maybe_log_post_replan_recovery(
            self,
            safety=safety,
            heading_error_deg=heading_error_deg,
        )

    def post_replan_escape_angular_hint(self, step):
        return post_replan_recovery.post_replan_escape_angular_hint(self, step)

    def post_replan_forward_side_p05(self):
        return post_replan_recovery.post_replan_forward_side_p05(self)

    def post_replan_clearance_search_direction(self, heading_error_deg):
        return post_replan_recovery.post_replan_clearance_search_direction(
            self,
            heading_error_deg,
        )

    def start_post_replan_clearance_search(self, recovery, pose, safety, heading_error_deg):
        return post_replan_recovery.start_post_replan_clearance_search(
            self,
            recovery,
            pose,
            safety,
            heading_error_deg,
        )

    def post_replan_clearance_scan_is_new(self, recovery):
        return post_replan_recovery.post_replan_clearance_scan_is_new(
            self,
            recovery,
        )

    def enter_post_replan_wait_clear(self, recovery, reason):
        return post_replan_recovery.enter_post_replan_wait_clear(
            self,
            recovery,
            reason,
        )

    def fail_post_replan_clearance_search(self, recovery, reason):
        return post_replan_recovery.fail_post_replan_clearance_search(
            self,
            recovery,
            reason,
        )

    def handle_post_replan_recovery(self, step, pose, now_sec, route_state=None):
        return post_replan_recovery.handle_post_replan_recovery(
            self,
            step,
            pose,
            now_sec,
            route_state=route_state,
            blocked_error_type=BlockedByScanError,
        )

    @staticmethod
    def _compat_method(target, name, *args, **kwargs):
        method = getattr(target, name, None)
        if callable(method):
            return method(*args, **kwargs)
        return getattr(WaypointFollower, name)(target, *args, **kwargs)

    def _replan_manager(self):
        manager = getattr(self, "replan_manager", None)
        if manager is None:
            manager = ReplanManager(self)
            self.replan_manager = manager
        return manager

    def update_replan_diagnostics(self, result, count_replan=True):
        return WaypointFollower._replan_manager(self).update_diagnostics(
            result,
            count_replan=count_replan,
        )

    def replanned_waypoints_from_result(self, result):
        return replanning.replanned_waypoints_from_result(self, result)

    def replanned_tracking_points_from_result(self, result):
        return replanning.replanned_tracking_points_from_result(self, result)

    def remember_replan_tracking_replacement(self, result, replanned, current_pose):
        return replanning.remember_replan_tracking_replacement(
            self,
            result,
            replanned,
            current_pose,
        )

    def first_motion_waypoint(self, replanned, current_pose):
        return replanning.first_motion_waypoint(self, replanned, current_pose)

    def first_motion_waypoint_index(self, replanned, current_pose):
        return replanning.first_motion_waypoint_index(self, replanned, current_pose)

    def replan_start_artifact_distance_limit_m(self):
        return replanning.replan_start_artifact_distance_limit_m(self)

    def first_forward_motion_waypoint_index(self, replanned, current_pose):
        return replanning.first_forward_motion_waypoint_index(
            self,
            replanned,
            current_pose,
        )

    def prune_replanned_waypoints_for_progress(self, replanned, current_pose):
        return replanning.prune_replanned_waypoints_for_progress(
            self,
            replanned,
            current_pose,
        )

    def post_replan_activation_min_target_distance_m(self):
        return replanning.post_replan_activation_min_target_distance_m(self)

    @staticmethod
    def _waypoint_xy(point):
        return replanning._waypoint_xy(point)

    @staticmethod
    def _projection_on_xy_segment(point, segment_start, segment_end):
        return replanning._projection_on_xy_segment(
            point,
            segment_start,
            segment_end,
        )

    @staticmethod
    def _waypoint_progress_on_route(
        route_points,
        cumulative,
        waypoint,
        min_progress_m=None,
    ):
        return replanning._waypoint_progress_on_route(
            route_points,
            cumulative,
            waypoint,
            min_progress_m=min_progress_m,
        )

    @staticmethod
    def _activation_tracking_validation(validation, tracking_source, point_count):
        return replanning._activation_tracking_validation(
            validation,
            tracking_source,
            point_count,
        )

    def _record_post_replan_activation_route(self, activation):
        return replanning._record_post_replan_activation_route(self, activation)

    def prepare_run_local_route_activation(
        self,
        replanned,
        current_pose,
        goal_waypoint,
        trigger,
    ):
        return replanning.prepare_run_local_route_activation(
            self,
            replanned,
            current_pose,
            goal_waypoint,
            trigger,
        )

    def route_signature(self, waypoints):
        return replanning.route_signature(self, waypoints)

    def remember_known_corridor_repair(self, waypoints):
        return replanning.remember_known_corridor_repair(self, waypoints)

    def suppress_repeated_known_corridor_repair(self, waypoints):
        return replanning.suppress_repeated_known_corridor_repair(self, waypoints)

    def scan_block_budget_repair_signature(self, current_pose, waypoints):
        return replanning.scan_block_budget_repair_signature(
            self,
            current_pose,
            waypoints,
        )

    def remember_scan_block_budget_repair(self, current_pose, waypoints):
        return replanning.remember_scan_block_budget_repair(
            self,
            current_pose,
            waypoints,
        )

    def validate_replan_result(
        self,
        result,
        current_pose,
        old_remaining_waypoints,
        goal_waypoint,
        require_changed=True,
    ):
        return WaypointFollower._replan_manager(self).validate_result(
            result,
            current_pose,
            old_remaining_waypoints,
            goal_waypoint,
            require_changed=require_changed,
        )

    def initialize_run_local_route(self, current_pose, waypoints):
        return WaypointFollower._replan_manager(self).initialize_route(
            current_pose,
            waypoints,
        )

    def corridor_blocked_cells(self, current_pose, remaining_waypoints):
        return WaypointFollower._replan_manager(self).corridor_blocked_cells(
            current_pose,
            remaining_waypoints,
        )

    def prune_run_local_obstacles_after_progress(self, current_pose, remaining_waypoints):
        return WaypointFollower._replan_manager(self).prune_after_progress(
            current_pose,
            remaining_waypoints,
        )

    def plan_with_existing_run_local_map(
        self,
        current_pose,
        old_remaining_waypoints,
        sequence=None,
        count_replan=True,
    ):
        return replanning.plan_with_existing_run_local_map(
            self,
            current_pose,
            old_remaining_waypoints,
            sequence=sequence,
            count_replan=count_replan,
        )

    def sparse_retry_scan_args(self):
        return replanning.sparse_retry_scan_args(self)

    def retry_sparse_lidar_replan(
        self,
        current_pose,
        goal_waypoint,
        old_remaining_waypoints,
        sequence,
    ):
        return replanning.retry_sparse_lidar_replan(
            self,
            current_pose,
            goal_waypoint,
            old_remaining_waypoints,
            sequence,
        )

    def replan_after_blockage(
        self,
        current_pose,
        old_remaining_waypoints,
        trigger=REPLAN_TRIGGER_SCAN_BLOCKAGE,
        guard_signature=None,
    ):
        return WaypointFollower._replan_manager(self).replan_after_blockage(
            current_pose,
            old_remaining_waypoints,
            trigger=trigger,
            guard_signature=guard_signature,
        )

    def follow_waypoints(
        self,
        waypoints,
        tracking_points=None,
        tracking_validation=None,
    ):
        return follow_loop.follow_waypoints(
            self,
            waypoints,
            tracking_points=tracking_points,
            tracking_validation=tracking_validation,
            context=_follow_loop_context(),
        )

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


def _node_setup_context():
    return facade_contexts.node_setup_context(globals())


def _controller_runtime_context():
    return facade_contexts.controller_runtime_context(globals())


def _ros_runtime_context():
    return facade_contexts.ros_runtime_context(globals())


def _ros_node_wiring_context():
    return facade_contexts.ros_node_wiring_context(globals())


def _follow_loop_context():
    return facade_contexts.follow_loop_context(globals())


RecoverableHealthError = ros_runtime.RecoverableHealthError


def transform_to_pose2d(transform, frame_id):
    return ros_runtime.transform_to_pose2d(
        transform,
        frame_id,
        _ros_runtime_context(),
    )


def compose_2d_pose(parent_from_mid, mid_from_child, child_frame_id):
    return ros_runtime.compose_2d_pose(
        parent_from_mid,
        mid_from_child,
        child_frame_id,
        _ros_runtime_context(),
    )


def _cli_context():
    return globals()


def _run_session_context():
    return facade_contexts.run_session_context(globals())


def require_motion_confirmation(args, waypoints):
    return follower_cli.require_motion_confirmation(args, waypoints)


def wait_before_follow_confirmation(
    args,
    current_pose,
    executable_waypoints,
    input_fn=input,
):
    return follower_cli.wait_before_follow_confirmation(
        args,
        current_pose,
        executable_waypoints,
        input_fn=input_fn,
    )


def print_dry_run(
    args,
    raw_waypoints,
    executable_waypoints,
    tracking_validation=None,
    lookahead_guard=None,
):
    return follower_cli.print_dry_run(
        args,
        raw_waypoints,
        executable_waypoints,
        tracking_validation=tracking_validation,
        lookahead_guard=lookahead_guard,
        context=_cli_context(),
    )


def parse_args(argv):
    return follower_cli.parse_args(argv, context=_cli_context())


def validate_args(parser, args):
    return follower_cli.validate_args(parser, args, context=_cli_context())


def main(argv=None):
    return run_session.run(argv, _run_session_context())


if __name__ == "__main__":
    sys.exit(main())
