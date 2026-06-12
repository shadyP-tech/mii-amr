from __future__ import annotations

import math
import time
from dataclasses import dataclass

import replan_runtime

from .math_utils import (
    clamp,
    normalize_angle_rad,
    quaternion_to_yaw_deg,
    shortest_angle_delta_deg,
)
from .models import Pose2D, ScanSafety
from .path_curves import (
    project_point_to_route,
    route_heading_from_projection,
    route_points_from_projection,
    truncate_polyline_by_distance,
)
from .scan_safety import percentile


DEFAULT_POST_REPLAN_RECOVERY = "on"
POST_REPLAN_RECOVERY_MODES = ("on", "off")
POST_REPLAN_CLEARANCE_CONE = "cone"
POST_REPLAN_CLEARANCE_ROUTE_AWARE = "route-aware"
POST_REPLAN_CLEARANCE_MODES = (
    POST_REPLAN_CLEARANCE_CONE,
    POST_REPLAN_CLEARANCE_ROUTE_AWARE,
)
DEFAULT_POST_REPLAN_CLEARANCE_MODE = POST_REPLAN_CLEARANCE_CONE
DEFAULT_POST_REPLAN_ROUTE_CLEARANCE_PREVIEW_DISTANCE_M = 0.0
POST_REPLAN_ROUTE_CLEARANCE_MIN_PREVIEW_M = 0.25
POST_REPLAN_ROUTE_CLEARANCE_MAX_AUTO_PREVIEW_M = 0.60
POST_REPLAN_ROUTE_CLEARANCE_REASON_BLOCKED = "route_corridor_blocked"
POST_REPLAN_ROUTE_CLEARANCE_REASON_SIDE_OBSTACLE = "route_clear_side_obstacle"
POST_REPLAN_ROUTE_CLEARANCE_REASON_UNAVAILABLE = "route_clearance_unavailable"
DEFAULT_POST_REPLAN_CLEAR_SCAN_SAMPLES = 2
DEFAULT_POST_REPLAN_TIMEOUT_SEC = 4.0
DEFAULT_POST_REPLAN_ESCAPE_DISTANCE_M = 0.12
DEFAULT_POST_REPLAN_ESCAPE_LINEAR_SPEED_MPS = 0.02
POST_REPLAN_ESCAPE_STEERING_AUTO = "auto"
POST_REPLAN_ESCAPE_STEERING_ROUTE_HINT = "route-hint"
POST_REPLAN_ESCAPE_STEERING_STRAIGHT_UNTIL_PROGRESS = "straight-until-progress"
POST_REPLAN_ESCAPE_STEERING_MODES = (
    POST_REPLAN_ESCAPE_STEERING_AUTO,
    POST_REPLAN_ESCAPE_STEERING_ROUTE_HINT,
    POST_REPLAN_ESCAPE_STEERING_STRAIGHT_UNTIL_PROGRESS,
)
DEFAULT_POST_REPLAN_ESCAPE_STEERING_MODE = POST_REPLAN_ESCAPE_STEERING_AUTO
DEFAULT_POST_REPLAN_ALIGN_HEADING_ERROR_DEG = 25.0
POST_REPLAN_MIN_ROUTE_SEGMENT_M = 0.05
POST_REPLAN_ROUTE_HEADING_LOOKAHEAD_M = 0.12
POST_REPLAN_CLEARANCE_MAX_YAW_DEG = 12.0
POST_REPLAN_CLEARANCE_IMPROVEMENT_M = 0.03
POST_REPLAN_CLEARANCE_MAX_ANGULAR_RADPS = 0.12
POST_REPLAN_CLEARANCE_SIDE_DIFF_M = 0.03
POST_REPLAN_ESCAPE_COMPLETION_TOLERANCE_M = 0.005
POST_REPLAN_ESCAPE_TIMEOUT_MARGIN_SEC = 0.75
POST_REPLAN_ESCAPE_MIN_TIMEOUT_SEC = 4.0
POST_REPLAN_ESCAPE_ANGULAR_HINT_CAP_RADPS = 0.05
POST_REPLAN_ESCAPE_STRAIGHT_UNTIL_PROGRESS_M = 0.010
POST_REPLAN_ESCAPE_NO_MOTION_EPS_M = 0.003
POST_REPLAN_ESCAPE_NO_MOTION_TIMEOUT_ODOM_SEC = 3.0
POST_REPLAN_ESCAPE_NO_MOTION_TIMEOUT_MAP_SEC = 4.0
POST_REPLAN_ACTIVATION_MIN_TARGET_FLOOR_M = 0.08
POST_REPLAN_RECOVERY_ALIGN = "align"
POST_REPLAN_RECOVERY_CLEARANCE_SEARCH = "clearance_search"
POST_REPLAN_RECOVERY_WAIT_CLEAR = "wait_clear"
POST_REPLAN_RECOVERY_ESCAPE = "escape"
POST_REPLAN_RECOVERY_DONE = "done"
POST_REPLAN_PRE_CONTROLLER_RECOVERY_PHASES = (
    POST_REPLAN_RECOVERY_ALIGN,
    POST_REPLAN_RECOVERY_WAIT_CLEAR,
    POST_REPLAN_RECOVERY_CLEARANCE_SEARCH,
)


@dataclass
class PostReplanRecoveryState:
    route_generation_id: int
    activation_pose: Pose2D
    activation_time_sec: float
    activation_scan_stamp_sec: float | None
    activation_scan_received_sec: float | None
    route_heading_deg: float
    phase: str = POST_REPLAN_RECOVERY_ALIGN
    clear_scan_count: int = 0
    last_counted_scan_identity: tuple[float | None, float | None] | None = None
    escape_start_pose: Pose2D | None = None
    escape_start_odom_pose: Pose2D | None = None
    escape_start_direct_odom_pose: Pose2D | None = None
    escape_start_tf_odom_pose: Pose2D | None = None
    escape_start_time_sec: float | None = None
    last_escape_timeout_sec: float | None = None
    last_escape_elapsed_sec: float | None = None
    last_scan_reason: str = ""
    last_heading_error_deg: float | None = None
    last_alignment_heading_deg: float | None = None
    last_alignment_heading_source: str = ""
    last_alignment_projection_segment_index: int | None = None
    last_alignment_projection_segment_ratio: float | None = None
    last_escape_distance_m: float = 0.0
    best_escape_distance_m: float = 0.0
    last_progress_distance_m: float = 0.0
    last_progress_time_sec: float | None = None
    first_escape_command_time_sec: float | None = None
    last_escape_distance_source: str = ""
    last_escape_no_motion_elapsed_sec: float | None = None
    escape_straight_until_progress_active: bool = False
    last_escape_command_linear_mps: float = 0.0
    last_escape_command_angular_radps: float = 0.0
    last_escape_angular_hint_source: str = ""
    last_escape_steering_mode_resolved: str = ""
    last_escape_odom_distance_m: float | None = None
    last_escape_map_distance_m: float | None = None
    last_escape_odom_stamp_delta_sec: float | None = None
    last_escape_progress_source: str = ""
    last_escape_no_motion_reason: str = ""
    last_escape_odom_source: str = ""
    last_escape_odom_source_fallback_reason: str = ""
    last_escape_direct_odom_distance_m: float | None = None
    last_escape_tf_odom_distance_m: float | None = None
    last_escape_direct_odom_age_sec: float | None = None
    last_escape_direct_odom_stamp_delta_sec: float | None = None
    last_escape_tf_odom_stamp_delta_sec: float | None = None
    last_escape_direct_odom_frame_id: str = ""
    last_escape_direct_odom_child_frame_id: str = ""
    last_escape_odom_disagreement: str = ""
    clearance_search_attempted: bool = False
    clearance_search_direction: float = 0.0
    clearance_search_start_yaw_deg: float | None = None
    clearance_search_baseline_p05_m: float | None = None
    clearance_search_best_p05_m: float | None = None
    clearance_search_baseline_min_m: float | None = None
    clearance_search_best_min_m: float | None = None
    clearance_search_last_scan_identity: tuple[float | None, float | None] | None = None
    clearance_search_yaw_delta_deg: float = 0.0
    clearance_search_result: str = ""
    clearance_search_direction_source: str = ""
    route_clearance_reason: str = ""
    route_corridor_min_distance_m: float | None = None
    route_corridor_blocked_count: int = 0
    route_clear_side_obstacle_count: int = 0
    route_corridor_preview_distance_m: float = 0.0
    final_status: str = "active"


@dataclass(frozen=True)
class PostReplanAlignmentHeading:
    heading_deg: float
    source: str
    projection_segment_index: int | None = None
    projection_segment_ratio: float | None = None


@dataclass(frozen=True)
class PostReplanRouteClearance:
    safe: bool
    reason: str
    preview_distance_m: float
    corridor_radius_m: float
    valid_scan_points: int
    blocked_count: int
    side_obstacle_count: int
    min_corridor_distance_m: float | None
    min_scan_range_m: float | None


@dataclass(frozen=True)
class PostReplanEscapeMeasurement:
    progress_distance_m: float
    progress_source: str
    odom_distance_m: float | None
    map_distance_m: float | None
    odom_stamp_delta_sec: float | None
    odom_source: str = "unavailable"
    odom_source_fallback_reason: str = "progress_unavailable"
    direct_odom_distance_m: float | None = None
    tf_odom_distance_m: float | None = None
    direct_odom_age_sec: float | None = None
    direct_odom_stamp_delta_sec: float | None = None
    tf_odom_stamp_delta_sec: float | None = None
    direct_odom_frame_id: str = ""
    direct_odom_child_frame_id: str = ""
    odom_disagreement: str = ""


def _format_optional_m(value):
    return "n/a" if value is None else f"{value:.3f}"


def _reset_command_smoother(node):
    smoother = getattr(node, "command_smoother", None)
    if smoother is not None:
        smoother.reset()
    if hasattr(node, "last_smoothed_command_time_sec"):
        node.last_smoothed_command_time_sec = None
    if hasattr(node, "last_velocity_scheduler_status"):
        node.last_velocity_scheduler_status = None
    if hasattr(node, "last_velocity_scheduler_log_sec"):
        node.last_velocity_scheduler_log_sec = None
    if hasattr(node, "last_smoothed_motion_mode"):
        node.last_smoothed_motion_mode = None


def _stamp_to_sec(stamp):
    if stamp is None:
        return None
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _ordered_base_frames(base_frame, fallback_base_frame):
    frames = []
    for frame in (base_frame, fallback_base_frame):
        if frame and frame not in frames:
            frames.append(frame)
    return frames


def _transform_to_pose2d(transform, frame_id):
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    stamp_sec = _stamp_to_sec(transform.header.stamp)
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


def _lookup_odom_pose(node):
    odom_frame = getattr(node.args, "odom_frame", "odom")
    for frame in _ordered_base_frames(node.args.base_frame, node.args.fallback_base_frame):
        transform = node.tf_buffer.lookup_transform(odom_frame, frame, None)
        return _transform_to_pose2d(transform, frame)
    raise RuntimeError("Could not lookup odom TF pose: no base frames configured")


def try_lookup_odom_pose(node):
    try:
        lookup = getattr(node, "lookup_odom_pose", None)
        if callable(lookup):
            return lookup()
        return _lookup_odom_pose(node)
    except Exception:
        return None


def fresh_direct_odom_pose(node, now_sec=None):
    helper = getattr(node, "fresh_direct_odom_pose", None)
    if callable(helper):
        return helper(now_sec=now_sec)
    pose = getattr(node, "last_odom_pose", None)
    received_sec = getattr(node, "last_odom_received_sec", None)
    if pose is None or received_sec is None:
        return None, None, "direct_odom_start_unavailable"
    if now_sec is None:
        now_sec = time.time()
    age_sec = max(0.0, now_sec - received_sec)
    if age_sec > float(getattr(node.args, "max_odom_age_sec", 1.0)):
        return None, age_sec, "direct_odom_stale"
    return pose, age_sec, "none"


def _pose_distance_m(start_pose, current_pose):
    if start_pose is None or current_pose is None:
        return None
    return math.hypot(
        current_pose.x - start_pose.x,
        current_pose.y - start_pose.y,
    )


def _pose_stamp_delta_sec(start_pose, current_pose):
    if start_pose is None or current_pose is None:
        return None
    if start_pose.stamp_sec is None or current_pose.stamp_sec is None:
        return None
    return current_pose.stamp_sec - start_pose.stamp_sec


def _escape_odom_disagreement(direct_distance_m, tf_distance_m):
    direct_moved = (
        direct_distance_m is not None
        and direct_distance_m >= POST_REPLAN_ESCAPE_NO_MOTION_EPS_M
    )
    tf_static = tf_distance_m is None or tf_distance_m < POST_REPLAN_ESCAPE_NO_MOTION_EPS_M
    if direct_moved and tf_static:
        return "direct_moved_tf_static"
    return ""


def resolve_post_replan_escape_steering_mode(args):
    configured = getattr(
        args,
        "post_replan_escape_steering_mode",
        DEFAULT_POST_REPLAN_ESCAPE_STEERING_MODE,
    )
    if configured == POST_REPLAN_ESCAPE_STEERING_ROUTE_HINT:
        return POST_REPLAN_ESCAPE_STEERING_ROUTE_HINT
    if configured == POST_REPLAN_ESCAPE_STEERING_STRAIGHT_UNTIL_PROGRESS:
        return POST_REPLAN_ESCAPE_STEERING_STRAIGHT_UNTIL_PROGRESS
    if (
        getattr(
            args,
            "post_replan_clearance_mode",
            DEFAULT_POST_REPLAN_CLEARANCE_MODE,
        )
        == POST_REPLAN_CLEARANCE_ROUTE_AWARE
    ):
        return POST_REPLAN_ESCAPE_STEERING_ROUTE_HINT
    return POST_REPLAN_ESCAPE_STEERING_STRAIGHT_UNTIL_PROGRESS


def post_replan_recovery_should_preempt_controller(recovery, args=None):
    if recovery is None:
        return False
    phase = getattr(recovery, "phase", "")
    if phase in POST_REPLAN_PRE_CONTROLLER_RECOVERY_PHASES:
        return True
    if (
        phase == POST_REPLAN_RECOVERY_ESCAPE
        and args is not None
        and resolve_post_replan_escape_steering_mode(args)
        == POST_REPLAN_ESCAPE_STEERING_ROUTE_HINT
    ):
        return False
    return (
        phase == POST_REPLAN_RECOVERY_ESCAPE
        and getattr(recovery, "best_escape_distance_m", 0.0)
        < POST_REPLAN_ESCAPE_STRAIGHT_UNTIL_PROGRESS_M
    )


def post_replan_recovery_active_for_args(args, default_controller="stop-go"):
    return (
        getattr(args, "controller", default_controller) == "pure-pursuit"
        and getattr(args, "enable_lidar_map_replan", False)
        and getattr(args, "post_replan_recovery", DEFAULT_POST_REPLAN_RECOVERY) == "on"
    )
def notes_with_post_replan_recovery_metadata(notes, args, node):
    if not getattr(args, "enable_lidar_map_replan", False):
        return notes
    recovery = getattr(node, "post_replan_recovery", None)
    last_heading_error = (
        getattr(recovery, "last_heading_error_deg", None)
        if recovery is not None
        else getattr(node, "last_post_replan_recovery_heading_error_deg", None)
    )
    last_alignment_heading = (
        getattr(recovery, "last_alignment_heading_deg", None)
        if recovery is not None
        else getattr(
            node,
            "last_post_replan_recovery_alignment_heading_deg",
            None,
        )
    )
    last_alignment_source = (
        getattr(recovery, "last_alignment_heading_source", "")
        if recovery is not None
        else getattr(
            node,
            "last_post_replan_recovery_alignment_heading_source",
            "",
        )
    )
    last_escape_command_linear = (
        getattr(recovery, "last_escape_command_linear_mps", 0.0)
        if recovery is not None
        else getattr(node, "last_post_replan_recovery_escape_command_linear_mps", 0.0)
    )
    last_escape_command_angular = (
        getattr(recovery, "last_escape_command_angular_radps", 0.0)
        if recovery is not None
        else getattr(
            node,
            "last_post_replan_recovery_escape_command_angular_radps",
            0.0,
        )
    )
    last_escape_angular_hint_source = (
        getattr(recovery, "last_escape_angular_hint_source", "")
        if recovery is not None
        else getattr(
            node,
            "last_post_replan_recovery_escape_angular_hint_source",
            "",
        )
    )
    last_escape_elapsed = (
        getattr(recovery, "last_escape_elapsed_sec", None)
        if recovery is not None
        else getattr(node, "last_post_replan_recovery_escape_elapsed_sec", None)
    )
    last_escape_timeout = (
        getattr(recovery, "last_escape_timeout_sec", None)
        if recovery is not None
        else getattr(node, "last_post_replan_recovery_escape_timeout_sec", None)
    )
    last_escape_distance = (
        getattr(recovery, "last_escape_distance_m", 0.0)
        if recovery is not None
        else getattr(node, "last_post_replan_recovery_escape_distance_m", 0.0)
    )
    last_escape_source = (
        getattr(recovery, "last_escape_distance_source", "")
        if recovery is not None
        else getattr(node, "last_post_replan_recovery_escape_distance_source", "")
    )
    best_escape_distance = (
        getattr(recovery, "best_escape_distance_m", 0.0)
        if recovery is not None
        else getattr(node, "last_post_replan_recovery_best_escape_distance_m", 0.0)
    )
    no_motion_elapsed = (
        getattr(recovery, "last_escape_no_motion_elapsed_sec", None)
        if recovery is not None
        else getattr(
            node,
            "last_post_replan_recovery_escape_no_motion_elapsed_sec",
            None,
        )
    )
    steering_mode_configured = getattr(
        args,
        "post_replan_escape_steering_mode",
        DEFAULT_POST_REPLAN_ESCAPE_STEERING_MODE,
    )
    steering_mode_resolved = (
        getattr(recovery, "last_escape_steering_mode_resolved", "")
        if recovery is not None
        else getattr(
            node,
            "last_post_replan_recovery_escape_steering_mode_resolved",
            "",
        )
    )
    if not steering_mode_resolved:
        steering_mode_resolved = resolve_post_replan_escape_steering_mode(args)
    escape_odom_distance = (
        getattr(recovery, "last_escape_odom_distance_m", None)
        if recovery is not None
        else getattr(
            node,
            "last_post_replan_recovery_escape_odom_distance_m",
            None,
        )
    )
    escape_map_distance = (
        getattr(recovery, "last_escape_map_distance_m", None)
        if recovery is not None
        else getattr(
            node,
            "last_post_replan_recovery_escape_map_distance_m",
            None,
        )
    )
    escape_odom_stamp_delta = (
        getattr(recovery, "last_escape_odom_stamp_delta_sec", None)
        if recovery is not None
        else getattr(
            node,
            "last_post_replan_recovery_escape_odom_stamp_delta_sec",
            None,
        )
    )
    escape_progress_source = (
        getattr(recovery, "last_escape_progress_source", "")
        if recovery is not None
        else getattr(
            node,
            "last_post_replan_recovery_escape_progress_source",
            "",
        )
    )
    escape_no_motion_reason = (
        getattr(recovery, "last_escape_no_motion_reason", "")
        if recovery is not None
        else getattr(
            node,
            "last_post_replan_recovery_escape_no_motion_reason",
            "",
        )
    )
    escape_odom_source = (
        getattr(recovery, "last_escape_odom_source", "")
        if recovery is not None
        else getattr(node, "last_post_replan_recovery_escape_odom_source", "")
    )
    escape_odom_source_fallback_reason = (
        getattr(recovery, "last_escape_odom_source_fallback_reason", "")
        if recovery is not None
        else getattr(
            node,
            "last_post_replan_recovery_escape_odom_source_fallback_reason",
            "",
        )
    )
    escape_direct_odom_distance = (
        getattr(recovery, "last_escape_direct_odom_distance_m", None)
        if recovery is not None
        else getattr(
            node,
            "last_post_replan_recovery_escape_direct_odom_distance_m",
            None,
        )
    )
    escape_tf_odom_distance = (
        getattr(recovery, "last_escape_tf_odom_distance_m", None)
        if recovery is not None
        else getattr(
            node,
            "last_post_replan_recovery_escape_tf_odom_distance_m",
            None,
        )
    )
    escape_direct_odom_age = (
        getattr(recovery, "last_escape_direct_odom_age_sec", None)
        if recovery is not None
        else getattr(
            node,
            "last_post_replan_recovery_escape_direct_odom_age_sec",
            None,
        )
    )
    escape_direct_odom_stamp_delta = (
        getattr(recovery, "last_escape_direct_odom_stamp_delta_sec", None)
        if recovery is not None
        else getattr(
            node,
            "last_post_replan_recovery_escape_direct_odom_stamp_delta_sec",
            None,
        )
    )
    escape_tf_odom_stamp_delta = (
        getattr(recovery, "last_escape_tf_odom_stamp_delta_sec", None)
        if recovery is not None
        else getattr(
            node,
            "last_post_replan_recovery_escape_tf_odom_stamp_delta_sec",
            None,
        )
    )
    escape_direct_odom_frame_id = (
        getattr(recovery, "last_escape_direct_odom_frame_id", "")
        if recovery is not None
        else getattr(
            node,
            "last_post_replan_recovery_escape_direct_odom_frame_id",
            "",
        )
    )
    escape_direct_odom_child_frame_id = (
        getattr(recovery, "last_escape_direct_odom_child_frame_id", "")
        if recovery is not None
        else getattr(
            node,
            "last_post_replan_recovery_escape_direct_odom_child_frame_id",
            "",
        )
    )
    escape_odom_disagreement = (
        getattr(recovery, "last_escape_odom_disagreement", "")
        if recovery is not None
        else getattr(
            node,
            "last_post_replan_recovery_escape_odom_disagreement",
            "",
        )
    )
    straight_active = (
        getattr(recovery, "escape_straight_until_progress_active", False)
        if recovery is not None
        else getattr(node, "last_post_replan_recovery_escape_straight_active", False)
    )
    clearance_attempted = (
        getattr(recovery, "clearance_search_attempted", False)
        if recovery is not None
        else getattr(node, "last_post_replan_clearance_search_attempted", False)
    )
    clearance_direction = (
        getattr(recovery, "clearance_search_direction", 0.0)
        if recovery is not None
        else getattr(node, "last_post_replan_clearance_search_direction", 0.0)
    )
    clearance_yaw_delta = (
        getattr(recovery, "clearance_search_yaw_delta_deg", 0.0)
        if recovery is not None
        else getattr(node, "last_post_replan_clearance_search_yaw_delta_deg", 0.0)
    )
    clearance_baseline_p05 = (
        getattr(recovery, "clearance_search_baseline_p05_m", None)
        if recovery is not None
        else getattr(node, "last_post_replan_clearance_search_baseline_p05_m", None)
    )
    clearance_best_p05 = (
        getattr(recovery, "clearance_search_best_p05_m", None)
        if recovery is not None
        else getattr(node, "last_post_replan_clearance_search_best_p05_m", None)
    )
    clearance_baseline_min = (
        getattr(recovery, "clearance_search_baseline_min_m", None)
        if recovery is not None
        else getattr(node, "last_post_replan_clearance_search_baseline_min_m", None)
    )
    clearance_best_min = (
        getattr(recovery, "clearance_search_best_min_m", None)
        if recovery is not None
        else getattr(node, "last_post_replan_clearance_search_best_min_m", None)
    )
    clearance_result = (
        getattr(recovery, "clearance_search_result", "")
        if recovery is not None
        else getattr(node, "last_post_replan_clearance_search_result", "")
    )
    clearance_direction_source = (
        getattr(recovery, "clearance_search_direction_source", "")
        if recovery is not None
        else getattr(node, "last_post_replan_clearance_search_direction_source", "")
    )
    activation_projection_progress = getattr(
        node,
        "last_post_replan_activation_projection_progress_m",
        None,
    )
    activation_first_target_distance = getattr(
        node,
        "last_post_replan_activation_first_target_distance_m",
        None,
    )
    route_clearance_reason = (
        getattr(recovery, "route_clearance_reason", "")
        if recovery is not None
        else getattr(node, "last_post_replan_route_clearance_reason", "")
    )
    route_corridor_min_distance = (
        getattr(recovery, "route_corridor_min_distance_m", None)
        if recovery is not None
        else getattr(node, "last_post_replan_route_corridor_min_distance_m", None)
    )
    route_corridor_blocked_count = (
        getattr(recovery, "route_corridor_blocked_count", 0)
        if recovery is not None
        else getattr(node, "last_post_replan_route_corridor_blocked_count", 0)
    )
    route_clear_side_obstacle_count = (
        getattr(recovery, "route_clear_side_obstacle_count", 0)
        if recovery is not None
        else getattr(node, "last_post_replan_route_clear_side_obstacle_count", 0)
    )
    route_corridor_preview_distance = (
        getattr(recovery, "route_corridor_preview_distance_m", 0.0)
        if recovery is not None
        else getattr(node, "last_post_replan_route_corridor_preview_distance_m", 0.0)
    )
    return (
        f"{notes};post_replan_recovery={args.post_replan_recovery};"
        "post_replan_clearance_mode="
        f"{getattr(args, 'post_replan_clearance_mode', DEFAULT_POST_REPLAN_CLEARANCE_MODE)};"
        "post_replan_route_clearance_reason="
        f"{route_clearance_reason};"
        "route_corridor_min_distance_m="
        f"{_format_optional_m(route_corridor_min_distance)};"
        "route_corridor_blocked_count="
        f"{route_corridor_blocked_count};"
        "route_clear_side_obstacle_count="
        f"{route_clear_side_obstacle_count};"
        "route_corridor_preview_distance_m="
        f"{route_corridor_preview_distance:.3f};"
        "post_replan_recovery_activations="
        f"{getattr(node, 'post_replan_recovery_activations', 0)};"
        "post_replan_recovery_last_status="
        f"{getattr(node, 'last_post_replan_recovery_status', '')};"
        "post_replan_recovery_last_phase="
        f"{getattr(node, 'last_post_replan_recovery_phase', '')};"
        "post_replan_recovery_clear_scan_count="
        f"{getattr(recovery, 'clear_scan_count', 0) if recovery is not None else getattr(node, 'last_post_replan_recovery_clear_count', 0)};"
        "post_replan_recovery_max_clear_scan_count="
        f"{getattr(node, 'max_post_replan_recovery_clear_count', 0)};"
        "post_replan_recovery_timeout_sec="
        f"{args.post_replan_timeout_sec:.3f};"
        "post_replan_recovery_escape_distance_m="
        f"{args.post_replan_escape_distance_m:.3f};"
        "post_replan_recovery_last_escape_distance_m="
        f"{last_escape_distance:.3f};"
        "post_replan_recovery_best_escape_distance_m="
        f"{best_escape_distance:.3f};"
        "post_replan_recovery_escape_distance_source="
        f"{last_escape_source};"
        "post_replan_escape_steering_mode_configured="
        f"{steering_mode_configured};"
        "post_replan_escape_steering_mode_resolved="
        f"{steering_mode_resolved};"
        "post_replan_escape_odom_distance_m="
        f"{_format_optional_m(escape_odom_distance)};"
        "post_replan_escape_map_distance_m="
        f"{_format_optional_m(escape_map_distance)};"
        "post_replan_escape_odom_stamp_delta_sec="
        f"{_format_optional_m(escape_odom_stamp_delta)};"
        "post_replan_escape_progress_source="
        f"{escape_progress_source};"
        "post_replan_escape_no_motion_reason="
        f"{escape_no_motion_reason};"
        "post_replan_escape_odom_source="
        f"{escape_odom_source};"
        "post_replan_escape_odom_source_fallback_reason="
        f"{escape_odom_source_fallback_reason};"
        "post_replan_escape_direct_odom_distance_m="
        f"{_format_optional_m(escape_direct_odom_distance)};"
        "post_replan_escape_tf_odom_distance_m="
        f"{_format_optional_m(escape_tf_odom_distance)};"
        "post_replan_escape_direct_odom_age_sec="
        f"{_format_optional_m(escape_direct_odom_age)};"
        "post_replan_escape_direct_odom_stamp_delta_sec="
        f"{_format_optional_m(escape_direct_odom_stamp_delta)};"
        "post_replan_escape_tf_odom_stamp_delta_sec="
        f"{_format_optional_m(escape_tf_odom_stamp_delta)};"
        "post_replan_escape_direct_odom_frame_id="
        f"{escape_direct_odom_frame_id};"
        "post_replan_escape_direct_odom_child_frame_id="
        f"{escape_direct_odom_child_frame_id};"
        "post_replan_escape_odom_disagreement="
        f"{escape_odom_disagreement};"
        "post_replan_recovery_last_heading_error_deg="
        f"{_format_optional_m(last_heading_error)};"
        "post_replan_recovery_last_alignment_heading_deg="
        f"{_format_optional_m(last_alignment_heading)};"
        "post_replan_recovery_last_alignment_heading_source="
        f"{last_alignment_source};"
        "post_replan_recovery_escape_linear_speed_mps="
        f"{args.post_replan_escape_linear_speed_mps:.3f};"
        "post_replan_escape_completion_tolerance_m="
        f"{POST_REPLAN_ESCAPE_COMPLETION_TOLERANCE_M:.3f};"
        "post_replan_recovery_last_escape_elapsed_sec="
        f"{_format_optional_m(last_escape_elapsed)};"
        "post_replan_recovery_last_escape_timeout_sec="
        f"{_format_optional_m(last_escape_timeout)};"
        "post_replan_escape_angular_hint_cap_radps="
        f"{POST_REPLAN_ESCAPE_ANGULAR_HINT_CAP_RADPS:.3f};"
        "post_replan_escape_straight_until_progress_m="
        f"{POST_REPLAN_ESCAPE_STRAIGHT_UNTIL_PROGRESS_M:.3f};"
        "post_replan_escape_straight_until_progress_active="
        f"{straight_active};"
        "post_replan_escape_no_motion_eps_m="
        f"{POST_REPLAN_ESCAPE_NO_MOTION_EPS_M:.3f};"
        "post_replan_escape_no_motion_timeout_odom_sec="
        f"{POST_REPLAN_ESCAPE_NO_MOTION_TIMEOUT_ODOM_SEC:.3f};"
        "post_replan_escape_no_motion_timeout_map_sec="
        f"{POST_REPLAN_ESCAPE_NO_MOTION_TIMEOUT_MAP_SEC:.3f};"
        "post_replan_escape_no_motion_elapsed_sec="
        f"{_format_optional_m(no_motion_elapsed)};"
        "post_replan_recovery_last_escape_command_linear_mps="
        f"{last_escape_command_linear:.3f};"
        "post_replan_recovery_last_escape_command_angular_radps="
        f"{last_escape_command_angular:.3f};"
        "post_replan_recovery_last_escape_angular_hint_source="
        f"{last_escape_angular_hint_source};"
        "post_replan_clearance_search_attempted="
        f"{clearance_attempted};"
        "post_replan_clearance_search_direction="
        f"{clearance_direction:.1f};"
        "post_replan_clearance_search_yaw_delta_deg="
        f"{clearance_yaw_delta:.3f};"
        "post_replan_clearance_search_baseline_p05_m="
        f"{_format_optional_m(clearance_baseline_p05)};"
        "post_replan_clearance_search_best_p05_m="
        f"{_format_optional_m(clearance_best_p05)};"
        "post_replan_clearance_search_baseline_min_m="
        f"{_format_optional_m(clearance_baseline_min)};"
        "post_replan_clearance_search_best_min_m="
        f"{_format_optional_m(clearance_best_min)};"
        "post_replan_clearance_search_result="
        f"{clearance_result};"
        "post_replan_clearance_search_direction_source="
        f"{clearance_direction_source};"
        "post_replan_activation_min_target_distance_m="
        f"{getattr(node, 'last_post_replan_activation_min_target_distance_m', 0.0):.3f};"
        "post_replan_activation_pruned_sparse_count="
        f"{getattr(node, 'last_post_replan_activation_pruned_sparse_count', 0)};"
        "post_replan_activation_pruned_dense_count="
        f"{getattr(node, 'last_post_replan_activation_pruned_dense_count', 0)};"
        "post_replan_activation_projection_progress_m="
        f"{_format_optional_m(activation_projection_progress)};"
        "post_replan_activation_first_target_distance_m="
        f"{_format_optional_m(activation_first_target_distance)};"
        "post_replan_activation_status="
        f"{getattr(node, 'last_post_replan_activation_status', '')};"
        "post_replan_recovery_align_heading_error_deg="
        f"{args.post_replan_align_heading_error_deg:.3f}"
    )

def current_scan_identity(node):
    return (
        replan_runtime.scan_stamp_sec(node.last_scan)
        if node.last_scan is not None
        else None,
        node.last_scan_received_sec,
    )

def scan_is_fresh_for_post_replan_recovery(node, recovery):
    stamp_sec, received_sec = node.current_scan_identity()
    activation_stamp = recovery.activation_scan_stamp_sec
    activation_received = recovery.activation_scan_received_sec
    epsilon = 1e-6
    if stamp_sec is not None and activation_stamp is not None:
        return stamp_sec > activation_stamp + epsilon
    if stamp_sec is not None and activation_stamp is None:
        return True
    if received_sec is not None and activation_received is not None:
        return received_sec > activation_received + epsilon
    return False

def scan_already_counted_for_post_replan_recovery(node, recovery):
    return recovery.last_counted_scan_identity == node.current_scan_identity()

def reset_post_replan_recovery(node, status=""):
    recovery = getattr(node, "post_replan_recovery", None)
    if recovery is not None:
        node.last_post_replan_recovery_phase = recovery.phase
        node.last_post_replan_recovery_clear_count = recovery.clear_scan_count
        node.max_post_replan_recovery_clear_count = max(
            node.max_post_replan_recovery_clear_count,
            recovery.clear_scan_count,
        )
        node.last_post_replan_recovery_escape_distance_m = (
            recovery.last_escape_distance_m
        )
        node.last_post_replan_recovery_best_escape_distance_m = (
            recovery.best_escape_distance_m
        )
        node.last_post_replan_recovery_escape_distance_source = (
            recovery.last_escape_distance_source
        )
        node.last_post_replan_recovery_escape_no_motion_elapsed_sec = (
            recovery.last_escape_no_motion_elapsed_sec
        )
        node.last_post_replan_recovery_escape_straight_active = (
            recovery.escape_straight_until_progress_active
        )
        node.last_post_replan_recovery_escape_elapsed_sec = (
            recovery.last_escape_elapsed_sec
        )
        node.last_post_replan_recovery_escape_timeout_sec = (
            recovery.last_escape_timeout_sec
        )
        node.last_post_replan_recovery_heading_error_deg = (
            recovery.last_heading_error_deg
        )
        node.last_post_replan_recovery_alignment_heading_deg = (
            recovery.last_alignment_heading_deg
        )
        node.last_post_replan_recovery_alignment_heading_source = (
            recovery.last_alignment_heading_source
        )
        node.last_post_replan_recovery_alignment_segment_index = (
            recovery.last_alignment_projection_segment_index
        )
        node.last_post_replan_recovery_alignment_segment_ratio = (
            recovery.last_alignment_projection_segment_ratio
        )
        node.last_post_replan_recovery_escape_command_linear_mps = (
            recovery.last_escape_command_linear_mps
        )
        node.last_post_replan_recovery_escape_command_angular_radps = (
            recovery.last_escape_command_angular_radps
        )
        node.last_post_replan_recovery_escape_angular_hint_source = (
            recovery.last_escape_angular_hint_source
        )
        node.last_post_replan_recovery_escape_steering_mode_resolved = (
            recovery.last_escape_steering_mode_resolved
        )
        node.last_post_replan_recovery_escape_odom_distance_m = (
            recovery.last_escape_odom_distance_m
        )
        node.last_post_replan_recovery_escape_map_distance_m = (
            recovery.last_escape_map_distance_m
        )
        node.last_post_replan_recovery_escape_odom_stamp_delta_sec = (
            recovery.last_escape_odom_stamp_delta_sec
        )
        node.last_post_replan_recovery_escape_progress_source = (
            recovery.last_escape_progress_source
        )
        node.last_post_replan_recovery_escape_no_motion_reason = (
            recovery.last_escape_no_motion_reason
        )
        node.last_post_replan_recovery_escape_odom_source = (
            recovery.last_escape_odom_source
        )
        node.last_post_replan_recovery_escape_odom_source_fallback_reason = (
            recovery.last_escape_odom_source_fallback_reason
        )
        node.last_post_replan_recovery_escape_direct_odom_distance_m = (
            recovery.last_escape_direct_odom_distance_m
        )
        node.last_post_replan_recovery_escape_tf_odom_distance_m = (
            recovery.last_escape_tf_odom_distance_m
        )
        node.last_post_replan_recovery_escape_direct_odom_age_sec = (
            recovery.last_escape_direct_odom_age_sec
        )
        node.last_post_replan_recovery_escape_direct_odom_stamp_delta_sec = (
            recovery.last_escape_direct_odom_stamp_delta_sec
        )
        node.last_post_replan_recovery_escape_tf_odom_stamp_delta_sec = (
            recovery.last_escape_tf_odom_stamp_delta_sec
        )
        node.last_post_replan_recovery_escape_direct_odom_frame_id = (
            recovery.last_escape_direct_odom_frame_id
        )
        node.last_post_replan_recovery_escape_direct_odom_child_frame_id = (
            recovery.last_escape_direct_odom_child_frame_id
        )
        node.last_post_replan_recovery_escape_odom_disagreement = (
            recovery.last_escape_odom_disagreement
        )
        node.last_post_replan_clearance_search_attempted = (
            recovery.clearance_search_attempted
        )
        node.last_post_replan_clearance_search_direction = (
            recovery.clearance_search_direction
        )
        node.last_post_replan_clearance_search_yaw_delta_deg = (
            recovery.clearance_search_yaw_delta_deg
        )
        node.last_post_replan_clearance_search_baseline_p05_m = (
            recovery.clearance_search_baseline_p05_m
        )
        node.last_post_replan_clearance_search_best_p05_m = (
            recovery.clearance_search_best_p05_m
        )
        node.last_post_replan_clearance_search_baseline_min_m = (
            recovery.clearance_search_baseline_min_m
        )
        node.last_post_replan_clearance_search_best_min_m = (
            recovery.clearance_search_best_min_m
        )
        node.last_post_replan_clearance_search_result = (
            recovery.clearance_search_result
        )
        node.last_post_replan_clearance_search_direction_source = (
            recovery.clearance_search_direction_source
        )
        node.last_post_replan_route_clearance_reason = (
            recovery.route_clearance_reason
        )
        node.last_post_replan_route_corridor_min_distance_m = (
            recovery.route_corridor_min_distance_m
        )
        node.last_post_replan_route_corridor_blocked_count = (
            recovery.route_corridor_blocked_count
        )
        node.last_post_replan_route_clear_side_obstacle_count = (
            recovery.route_clear_side_obstacle_count
        )
        node.last_post_replan_route_corridor_preview_distance_m = (
            recovery.route_corridor_preview_distance_m
        )
    if status:
        node.last_post_replan_recovery_status = status
    node.post_replan_recovery = None
    node.last_post_replan_recovery_log_sec = None

def post_replan_recovery_route_points(node, route_state):
    points = route_state.remaining_tracking_points()
    if len(points) < 2:
        points = route_state.remaining()
    return [
        (
            float(point.x if hasattr(point, "x") else point[0]),
            float(point.y if hasattr(point, "y") else point[1]),
        )
        for point in points
    ]


def post_replan_route_clearance_preview_distance_m(args):
    configured = float(
        getattr(
            args,
            "post_replan_route_clearance_preview_distance_m",
            DEFAULT_POST_REPLAN_ROUTE_CLEARANCE_PREVIEW_DISTANCE_M,
        )
    )
    if configured > 0.0:
        return configured
    auto_distance = max(
        POST_REPLAN_ROUTE_CLEARANCE_MIN_PREVIEW_M,
        float(getattr(args, "min_scan_range_m", 0.0)),
        float(getattr(args, "post_replan_escape_distance_m", 0.0)),
    )
    return min(POST_REPLAN_ROUTE_CLEARANCE_MAX_AUTO_PREVIEW_M, auto_distance)


def _map_point_to_base(point, pose):
    dx = float(point[0]) - float(pose.x)
    dy = float(point[1]) - float(pose.y)
    yaw = math.radians(float(pose.yaw_deg))
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        cos_yaw * dx + sin_yaw * dy,
        -sin_yaw * dx + cos_yaw * dy,
    )


def post_replan_route_preview_base_points(node, pose, route_state):
    if route_state is None or pose is None:
        return ()
    route_points = post_replan_recovery_route_points(node, route_state)
    if len(route_points) < 2:
        return ()
    try:
        projection = project_point_to_route(
            route_points,
            pose,
            allow_backward=True,
            projection_status="post_replan_route_clearance",
        )
        preview_points = route_points_from_projection(route_points, projection)
        preview_points = truncate_polyline_by_distance(
            preview_points,
            post_replan_route_clearance_preview_distance_m(node.args),
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return ()
    if len(preview_points) < 2:
        return ()
    try:
        return tuple(_map_point_to_base(point, pose) for point in preview_points)
    except (AttributeError, TypeError, ValueError):
        return ()


def _valid_scan_base_points(scan):
    if scan is None:
        return ()
    try:
        ranges = tuple(getattr(scan, "ranges", ()))
        angle_min = float(scan.angle_min)
        angle_increment = float(scan.angle_increment)
        range_min = float(scan.range_min)
        range_max = float(scan.range_max)
    except (AttributeError, TypeError, ValueError):
        return ()
    points = []
    for index, raw_range in enumerate(ranges):
        try:
            range_m = float(raw_range)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(range_m):
            continue
        if range_m < range_min or range_m > range_max:
            continue
        angle = angle_min + index * angle_increment
        points.append((
            range_m * math.cos(angle),
            range_m * math.sin(angle),
            range_m,
            normalize_angle_rad(angle),
        ))
    return tuple(points)


def _point_segment_projection_distance_m(point, start, end):
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return None
    ratio = ((px - sx) * dx + (py - sy) * dy) / length_sq
    if ratio < -1e-9 or ratio > 1.0 + 1e-9:
        return None
    ratio = clamp(ratio, 0.0, 1.0)
    closest = (sx + ratio * dx, sy + ratio * dy)
    return math.hypot(px - closest[0], py - closest[1])


def _point_preview_distance_m(point, preview_points):
    best_distance = None
    for index in range(len(preview_points) - 1):
        distance = _point_segment_projection_distance_m(
            point,
            preview_points[index],
            preview_points[index + 1],
        )
        if distance is None:
            continue
        if best_distance is None or distance < best_distance:
            best_distance = distance
    return best_distance


def evaluate_post_replan_route_clearance(node, pose, route_state):
    preview_distance = post_replan_route_clearance_preview_distance_m(node.args)
    corridor_radius = (
        float(getattr(node.args, "robot_footprint_radius_m", 0.0))
        + float(getattr(node.args, "run_local_map_clearance_margin_m", 0.0))
    )
    if corridor_radius <= 0.0:
        return PostReplanRouteClearance(
            False,
            POST_REPLAN_ROUTE_CLEARANCE_REASON_UNAVAILABLE,
            preview_distance,
            corridor_radius,
            0,
            0,
            0,
            None,
            None,
        )
    preview_points = post_replan_route_preview_base_points(node, pose, route_state)
    if len(preview_points) < 2:
        return PostReplanRouteClearance(
            False,
            POST_REPLAN_ROUTE_CLEARANCE_REASON_UNAVAILABLE,
            preview_distance,
            corridor_radius,
            0,
            0,
            0,
            None,
            None,
        )
    scan_points = _valid_scan_base_points(getattr(node, "last_scan", None))
    if not scan_points:
        return PostReplanRouteClearance(
            False,
            POST_REPLAN_ROUTE_CLEARANCE_REASON_UNAVAILABLE,
            preview_distance,
            corridor_radius,
            0,
            0,
            0,
            None,
            None,
        )

    min_distance = None
    min_range = None
    blocked_count = 0
    side_obstacle_count = 0
    half_angle_rad = math.radians(float(node.args.scan_half_angle_deg))
    min_scan_range_m = float(node.args.min_scan_range_m)
    for x, y, range_m, angle in scan_points:
        distance = _point_preview_distance_m((x, y), preview_points)
        blocked = distance is not None and distance <= corridor_radius + 1e-9
        if distance is not None:
            min_distance = (
                distance
                if min_distance is None
                else min(min_distance, distance)
            )
        if min_range is None or range_m < min_range:
            min_range = range_m
        if blocked:
            blocked_count += 1
        elif abs(angle) <= half_angle_rad and range_m < min_scan_range_m:
            side_obstacle_count += 1

    if blocked_count > 0:
        reason = POST_REPLAN_ROUTE_CLEARANCE_REASON_BLOCKED
    else:
        reason = POST_REPLAN_ROUTE_CLEARANCE_REASON_SIDE_OBSTACLE
    return PostReplanRouteClearance(
        blocked_count == 0,
        reason,
        preview_distance,
        corridor_radius,
        len(scan_points),
        blocked_count,
        side_obstacle_count,
        min_distance,
        min_range,
    )


def _record_route_clearance_result(node, recovery, result):
    node.last_post_replan_route_clearance_reason = result.reason
    node.last_post_replan_route_corridor_min_distance_m = (
        result.min_corridor_distance_m
    )
    node.last_post_replan_route_corridor_blocked_count = result.blocked_count
    node.last_post_replan_route_clear_side_obstacle_count = (
        result.side_obstacle_count
    )
    node.last_post_replan_route_corridor_preview_distance_m = (
        result.preview_distance_m
    )
    if recovery is not None:
        recovery.route_clearance_reason = result.reason
        recovery.route_corridor_min_distance_m = result.min_corridor_distance_m
        recovery.route_corridor_blocked_count = result.blocked_count
        recovery.route_clear_side_obstacle_count = result.side_obstacle_count
        recovery.route_corridor_preview_distance_m = result.preview_distance_m


def post_replan_forward_clearance_safety(node, pose, route_state, cone_safety=None):
    cone_safety = cone_safety or node.evaluate_current_scan_safety("forward")
    recovery = getattr(node, "post_replan_recovery", None)
    mode = getattr(
        node.args,
        "post_replan_clearance_mode",
        DEFAULT_POST_REPLAN_CLEARANCE_MODE,
    )
    if mode != POST_REPLAN_CLEARANCE_ROUTE_AWARE:
        preview_distance = post_replan_route_clearance_preview_distance_m(node.args)
        _record_route_clearance_result(
            node,
            recovery,
            PostReplanRouteClearance(
                cone_safety.safe,
                cone_safety.reason,
                preview_distance,
                0.0,
                cone_safety.valid_count,
                0,
                0,
                None,
                cone_safety.min_range_m,
            ),
        )
        return cone_safety

    if cone_safety.reason == "hard_stop":
        preview_distance = post_replan_route_clearance_preview_distance_m(node.args)
        _record_route_clearance_result(
            node,
            recovery,
            PostReplanRouteClearance(
                False,
                "hard_stop",
                preview_distance,
                0.0,
                cone_safety.valid_count,
                0,
                0,
                None,
                cone_safety.min_range_m,
            ),
        )
        return cone_safety

    route_clearance = evaluate_post_replan_route_clearance(
        node,
        pose,
        route_state,
    )
    _record_route_clearance_result(node, recovery, route_clearance)
    if route_clearance.reason == POST_REPLAN_ROUTE_CLEARANCE_REASON_UNAVAILABLE:
        return cone_safety
    if route_clearance.blocked_count > 0:
        return ScanSafety(
            False,
            POST_REPLAN_ROUTE_CLEARANCE_REASON_BLOCKED,
            route_clearance.valid_scan_points,
            route_clearance.min_scan_range_m,
            cone_safety.percentile_5_m,
        )
    if cone_safety.safe:
        _record_route_clearance_result(
            node,
            recovery,
            PostReplanRouteClearance(
                True,
                cone_safety.reason,
                route_clearance.preview_distance_m,
                route_clearance.corridor_radius_m,
                route_clearance.valid_scan_points,
                0,
                route_clearance.side_obstacle_count,
                route_clearance.min_corridor_distance_m,
                route_clearance.min_scan_range_m,
            ),
        )
        return cone_safety
    if cone_safety.reason == "soft_stop":
        return ScanSafety(
            True,
            POST_REPLAN_ROUTE_CLEARANCE_REASON_SIDE_OBSTACLE,
            route_clearance.valid_scan_points,
            cone_safety.min_range_m,
            cone_safety.percentile_5_m,
        )
    return cone_safety


def local_post_replan_alignment_heading(node, route_points, segment_index):
    if len(route_points) < 2 or segment_index is None:
        return None
    segment_index = max(0, min(int(segment_index), len(route_points) - 2))
    candidates = []
    for offset in range(0, 3):
        if offset == 0:
            candidates.append(segment_index)
            continue
        candidates.extend([segment_index + offset, segment_index - offset])
    seen = set()
    for index in candidates:
        if index in seen or index < 0 or index >= len(route_points) - 1:
            continue
        seen.add(index)
        start = route_points[index]
        end = route_points[index + 1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        if math.hypot(dx, dy) >= POST_REPLAN_MIN_ROUTE_SEGMENT_M:
            return PostReplanAlignmentHeading(
                math.degrees(math.atan2(dy, dx)),
                "local_projection_fallback",
                index,
                0.0,
            )
    return None

def post_replan_alignment_heading(node, pose, route_state):
    route_points = post_replan_recovery_route_points(
        node,
        route_state,
    )
    if len(route_points) < 2:
        return None
    try:
        projection = project_point_to_route(
            route_points,
            pose,
            allow_backward=True,
            projection_status="post_replan_recovery_align",
        )
    except RuntimeError:
        return None
    route_heading = route_heading_from_projection(
        route_points,
        projection,
        pose.yaw_deg,
        heading_lookahead_m=POST_REPLAN_ROUTE_HEADING_LOOKAHEAD_M,
    )
    if route_heading.heading_deg is not None:
        return PostReplanAlignmentHeading(
            float(route_heading.heading_deg),
            f"route_projection_{route_heading.source}",
            projection.segment_index,
            projection.segment_ratio,
        )
    return local_post_replan_alignment_heading(
        node,
        route_points,
        projection.segment_index,
    )

def route_heading_for_post_replan_recovery(node, pose, route_state):
    alignment = post_replan_alignment_heading(
        node,
        pose,
        route_state,
    )
    return None if alignment is None else alignment.heading_deg

def activate_post_replan_recovery(node, pose, route_state):
    if not post_replan_recovery_active_for_args(node.args):
        reset_post_replan_recovery(node, "disabled")
        return
    alignment = post_replan_alignment_heading(
        node,
        pose,
        route_state,
    )
    if alignment is None:
        reset_post_replan_recovery(
            node,
            "post_replan_alignment_unavailable",
        )
        raise RuntimeError("post_replan_alignment_unavailable")
    node.post_replan_recovery = PostReplanRecoveryState(
        route_generation_id=node.active_route_generation_id,
        activation_pose=pose,
        activation_time_sec=time.time(),
        activation_scan_stamp_sec=(
            replan_runtime.scan_stamp_sec(getattr(node, "last_scan", None))
            if getattr(node, "last_scan", None) is not None
            else None
        ),
        activation_scan_received_sec=getattr(node, "last_scan_received_sec", None),
        route_heading_deg=alignment.heading_deg,
        last_alignment_heading_deg=alignment.heading_deg,
        last_alignment_heading_source=alignment.source,
        last_alignment_projection_segment_index=(
            alignment.projection_segment_index
        ),
        last_alignment_projection_segment_ratio=(
            alignment.projection_segment_ratio
        ),
    )
    node.post_replan_recovery_activations += 1
    node.last_post_replan_recovery_status = "active"
    node.last_post_replan_recovery_phase = POST_REPLAN_RECOVERY_ALIGN
    node.last_post_replan_recovery_clear_count = 0
    node.last_post_replan_recovery_escape_distance_m = 0.0
    node.last_post_replan_recovery_best_escape_distance_m = 0.0
    node.last_post_replan_recovery_escape_distance_source = ""
    node.last_post_replan_recovery_escape_no_motion_elapsed_sec = None
    node.last_post_replan_recovery_escape_straight_active = False
    node.last_post_replan_recovery_escape_elapsed_sec = None
    node.last_post_replan_recovery_escape_timeout_sec = None
    node.last_post_replan_recovery_escape_steering_mode_resolved = ""
    node.last_post_replan_recovery_escape_odom_distance_m = None
    node.last_post_replan_recovery_escape_map_distance_m = None
    node.last_post_replan_recovery_escape_odom_stamp_delta_sec = None
    node.last_post_replan_recovery_escape_progress_source = ""
    node.last_post_replan_recovery_escape_no_motion_reason = ""
    node.last_post_replan_recovery_escape_odom_source = ""
    node.last_post_replan_recovery_escape_odom_source_fallback_reason = ""
    node.last_post_replan_recovery_escape_direct_odom_distance_m = None
    node.last_post_replan_recovery_escape_tf_odom_distance_m = None
    node.last_post_replan_recovery_escape_direct_odom_age_sec = None
    node.last_post_replan_recovery_escape_direct_odom_stamp_delta_sec = None
    node.last_post_replan_recovery_escape_tf_odom_stamp_delta_sec = None
    node.last_post_replan_recovery_escape_direct_odom_frame_id = ""
    node.last_post_replan_recovery_escape_direct_odom_child_frame_id = ""
    node.last_post_replan_recovery_escape_odom_disagreement = ""
    node.last_post_replan_route_clearance_reason = ""
    node.last_post_replan_route_corridor_min_distance_m = None
    node.last_post_replan_route_corridor_blocked_count = 0
    node.last_post_replan_route_clear_side_obstacle_count = 0
    node.last_post_replan_route_corridor_preview_distance_m = 0.0
    _reset_command_smoother(node)
    node.publish_velocity(0.0, 0.0)

def post_replan_recovery_timeout_reason(node, recovery):
    if recovery.phase == POST_REPLAN_RECOVERY_ALIGN:
        return "post_replan_align_timeout"
    if recovery.phase == POST_REPLAN_RECOVERY_CLEARANCE_SEARCH:
        return "post_replan_clearance_search_failed"
    if recovery.phase == POST_REPLAN_RECOVERY_ESCAPE:
        return "post_replan_escape_timeout"
    return "post_replan_scan_still_blocked"

def post_replan_recovery_timed_out(node, recovery, now_sec):
    return now_sec - recovery.activation_time_sec > node.args.post_replan_timeout_sec

def post_replan_escape_timeout_sec(node):
    speed_mps = max(0.0, float(node.args.post_replan_escape_linear_speed_mps))
    distance_m = max(0.0, float(node.args.post_replan_escape_distance_m))
    if speed_mps <= 0.0:
        return POST_REPLAN_ESCAPE_MIN_TIMEOUT_SEC
    return max(
        POST_REPLAN_ESCAPE_MIN_TIMEOUT_SEC,
        distance_m / speed_mps + POST_REPLAN_ESCAPE_TIMEOUT_MARGIN_SEC,
    )

def post_replan_escape_timed_out(node, recovery, now_sec):
    escape_start_sec = recovery.escape_start_time_sec
    if escape_start_sec is None:
        escape_start_sec = recovery.activation_time_sec
    escape_timeout_sec = post_replan_escape_timeout_sec(node)
    recovery.last_escape_timeout_sec = escape_timeout_sec
    recovery.last_escape_elapsed_sec = max(0.0, now_sec - escape_start_sec)
    total_deadline_sec = (
        recovery.activation_time_sec + node.args.post_replan_timeout_sec
    )
    escape_deadline_sec = escape_start_sec + escape_timeout_sec
    effective_deadline_sec = max(total_deadline_sec, escape_deadline_sec)
    return now_sec > effective_deadline_sec

def post_replan_escape_measurement(node, recovery, pose, now_sec=None):
    direct_pose, direct_age_sec, direct_reason = fresh_direct_odom_pose(
        node,
        now_sec=now_sec,
    )
    tf_odom_pose = try_lookup_odom_pose(node)
    direct_start_pose = recovery.escape_start_direct_odom_pose
    tf_start_pose = recovery.escape_start_tf_odom_pose
    if tf_start_pose is None:
        tf_start_pose = recovery.escape_start_odom_pose

    direct_odom_distance = None
    direct_odom_stamp_delta = None
    if direct_start_pose is not None and direct_pose is not None:
        direct_odom_distance = _pose_distance_m(direct_start_pose, direct_pose)
        direct_odom_stamp_delta = _pose_stamp_delta_sec(
            direct_start_pose,
            direct_pose,
        )

    tf_odom_distance = None
    tf_odom_stamp_delta = None
    if tf_start_pose is not None and tf_odom_pose is not None:
        tf_odom_distance = _pose_distance_m(tf_start_pose, tf_odom_pose)
        tf_odom_stamp_delta = _pose_stamp_delta_sec(
            tf_start_pose,
            tf_odom_pose,
        )

    start_pose = recovery.escape_start_pose or pose
    map_distance = None
    if start_pose is not None and pose is not None:
        map_distance = math.hypot(
            pose.x - start_pose.x,
            pose.y - start_pose.y,
        )

    direct_failure_reason = "none"
    if direct_start_pose is None:
        direct_failure_reason = "direct_odom_start_unavailable"
    elif direct_pose is None:
        direct_failure_reason = direct_reason or "progress_unavailable"

    odom_disagreement = _escape_odom_disagreement(
        direct_odom_distance,
        tf_odom_distance,
    )
    direct_frame_id = getattr(node, "last_odom_frame_id", "")
    direct_child_frame_id = getattr(node, "last_odom_child_frame_id", "")

    if direct_odom_distance is not None:
        return PostReplanEscapeMeasurement(
            direct_odom_distance,
            "direct_odom",
            direct_odom_distance,
            map_distance,
            direct_odom_stamp_delta,
            odom_source="direct_odom",
            odom_source_fallback_reason="none",
            direct_odom_distance_m=direct_odom_distance,
            tf_odom_distance_m=tf_odom_distance,
            direct_odom_age_sec=direct_age_sec,
            direct_odom_stamp_delta_sec=direct_odom_stamp_delta,
            tf_odom_stamp_delta_sec=tf_odom_stamp_delta,
            direct_odom_frame_id=direct_frame_id,
            direct_odom_child_frame_id=direct_child_frame_id,
            odom_disagreement=odom_disagreement,
        )
    if tf_odom_distance is not None:
        fallback_reason = (
            direct_failure_reason
            if direct_failure_reason != "none"
            else "none"
        )
        return PostReplanEscapeMeasurement(
            tf_odom_distance,
            "tf_odom",
            tf_odom_distance,
            map_distance,
            tf_odom_stamp_delta,
            odom_source="tf_odom",
            odom_source_fallback_reason=fallback_reason,
            direct_odom_distance_m=direct_odom_distance,
            tf_odom_distance_m=tf_odom_distance,
            direct_odom_age_sec=direct_age_sec,
            direct_odom_stamp_delta_sec=direct_odom_stamp_delta,
            tf_odom_stamp_delta_sec=tf_odom_stamp_delta,
            direct_odom_frame_id=direct_frame_id,
            direct_odom_child_frame_id=direct_child_frame_id,
            odom_disagreement=odom_disagreement,
        )

    if direct_start_pose is None and tf_start_pose is None:
        fallback_reason = "progress_unavailable"
    elif direct_failure_reason != "none":
        fallback_reason = direct_failure_reason
    elif tf_start_pose is None:
        fallback_reason = "tf_odom_start_unavailable"
    else:
        fallback_reason = "progress_unavailable"
    return PostReplanEscapeMeasurement(
        0.0,
        "unavailable",
        None,
        map_distance,
        None,
        odom_source="unavailable",
        odom_source_fallback_reason=fallback_reason,
        direct_odom_distance_m=direct_odom_distance,
        tf_odom_distance_m=tf_odom_distance,
        direct_odom_age_sec=direct_age_sec,
        direct_odom_stamp_delta_sec=direct_odom_stamp_delta,
        tf_odom_stamp_delta_sec=tf_odom_stamp_delta,
        direct_odom_frame_id=direct_frame_id,
        direct_odom_child_frame_id=direct_child_frame_id,
        odom_disagreement=odom_disagreement,
    )

def update_post_replan_escape_progress(node, recovery, measurement, now_sec):
    recovery.last_escape_distance_m = measurement.progress_distance_m
    recovery.last_escape_distance_source = measurement.progress_source
    recovery.last_escape_odom_distance_m = measurement.odom_distance_m
    recovery.last_escape_map_distance_m = measurement.map_distance_m
    recovery.last_escape_odom_stamp_delta_sec = measurement.odom_stamp_delta_sec
    recovery.last_escape_progress_source = measurement.progress_source
    recovery.last_escape_odom_source = measurement.odom_source
    recovery.last_escape_odom_source_fallback_reason = (
        measurement.odom_source_fallback_reason
    )
    recovery.last_escape_direct_odom_distance_m = (
        measurement.direct_odom_distance_m
    )
    recovery.last_escape_tf_odom_distance_m = measurement.tf_odom_distance_m
    recovery.last_escape_direct_odom_age_sec = measurement.direct_odom_age_sec
    recovery.last_escape_direct_odom_stamp_delta_sec = (
        measurement.direct_odom_stamp_delta_sec
    )
    recovery.last_escape_tf_odom_stamp_delta_sec = (
        measurement.tf_odom_stamp_delta_sec
    )
    recovery.last_escape_direct_odom_frame_id = measurement.direct_odom_frame_id
    recovery.last_escape_direct_odom_child_frame_id = (
        measurement.direct_odom_child_frame_id
    )
    recovery.last_escape_odom_disagreement = measurement.odom_disagreement
    recovery.best_escape_distance_m = max(
        recovery.best_escape_distance_m,
        measurement.progress_distance_m,
    )
    if recovery.first_escape_command_time_sec is None:
        recovery.last_escape_no_motion_elapsed_sec = None
        return
    if recovery.last_progress_time_sec is None:
        recovery.last_progress_time_sec = recovery.first_escape_command_time_sec
        recovery.last_progress_distance_m = recovery.best_escape_distance_m
    if (
        recovery.best_escape_distance_m - recovery.last_progress_distance_m
        >= POST_REPLAN_ESCAPE_NO_MOTION_EPS_M
    ):
        recovery.last_progress_distance_m = recovery.best_escape_distance_m
        recovery.last_progress_time_sec = now_sec
    recovery.last_escape_no_motion_elapsed_sec = max(
        0.0,
        now_sec - recovery.last_progress_time_sec,
    )

def post_replan_escape_no_motion_reason(recovery):
    odom_distance = recovery.last_escape_odom_distance_m
    map_distance = recovery.last_escape_map_distance_m
    stamp_delta = recovery.last_escape_odom_stamp_delta_sec
    map_moved = (
        map_distance is not None
        and map_distance >= POST_REPLAN_ESCAPE_NO_MOTION_EPS_M
    )
    odom_moved = (
        odom_distance is not None
        and odom_distance >= POST_REPLAN_ESCAPE_NO_MOTION_EPS_M
    )
    if odom_moved:
        return ""
    if odom_distance is None and map_distance is None:
        return "progress_unavailable"
    if stamp_delta is None or stamp_delta <= 1e-6:
        return "odom_static_map_moved" if map_moved else "odom_stale"
    if map_moved:
        return "odom_static_map_moved"
    return "cmd_vel_no_odom_motion"


def post_replan_escape_no_motion_timed_out(node, recovery, linear_x):
    if linear_x <= 0.0 or recovery.first_escape_command_time_sec is None:
        recovery.last_escape_no_motion_reason = ""
        return False
    if recovery.last_escape_no_motion_elapsed_sec is None:
        recovery.last_escape_no_motion_reason = ""
        return False
    recovery.last_escape_no_motion_reason = post_replan_escape_no_motion_reason(
        recovery,
    )
    timeout_sec = (
        POST_REPLAN_ESCAPE_NO_MOTION_TIMEOUT_ODOM_SEC
        if recovery.last_escape_distance_source in ("odom", "direct_odom", "tf_odom")
        else POST_REPLAN_ESCAPE_NO_MOTION_TIMEOUT_MAP_SEC
    )
    return recovery.last_escape_no_motion_elapsed_sec >= timeout_sec

def maybe_log_post_replan_recovery(node, safety=None, heading_error_deg=None):
    if not node.args.verbose:
        return
    recovery = getattr(node, "post_replan_recovery", None)
    if recovery is None:
        return
    now_sec = time.time()
    phase_changed = recovery.phase != node.last_post_replan_recovery_phase
    log_due = (
        node.last_post_replan_recovery_log_sec is None
        or now_sec - node.last_post_replan_recovery_log_sec >= 1.0
    )
    if not phase_changed and not log_due:
        return
    node.last_post_replan_recovery_phase = recovery.phase
    node.last_post_replan_recovery_log_sec = now_sec
    node.get_logger().info(
        "Post-replan recovery: "
        f"phase={recovery.phase}, "
        f"scan_reason={getattr(safety, 'reason', '') if safety else ''}, "
        f"scan_identity={node.current_scan_identity()}, "
        "alignment_heading_deg="
        f"{_format_optional_m(recovery.last_alignment_heading_deg)}, "
        "alignment_heading_source="
        f"{recovery.last_alignment_heading_source}, "
        "projection_segment_index="
        f"{recovery.last_alignment_projection_segment_index}, "
        "projection_segment_ratio="
        f"{_format_optional_m(recovery.last_alignment_projection_segment_ratio)}, "
        "heading_error_deg="
        f"{_format_optional_m(heading_error_deg)}, "
        f"clear_scan_count={recovery.clear_scan_count}, "
        f"escape_distance_m={recovery.last_escape_distance_m:.3f}, "
        f"best_escape_distance_m={recovery.best_escape_distance_m:.3f}, "
        "escape_distance_source="
        f"{recovery.last_escape_distance_source}, "
        "escape_steering_mode_resolved="
        f"{recovery.last_escape_steering_mode_resolved}, "
        "escape_odom_distance_m="
        f"{_format_optional_m(recovery.last_escape_odom_distance_m)}, "
        "escape_map_distance_m="
        f"{_format_optional_m(recovery.last_escape_map_distance_m)}, "
        "escape_odom_stamp_delta_sec="
        f"{_format_optional_m(recovery.last_escape_odom_stamp_delta_sec)}, "
        "escape_progress_source="
        f"{recovery.last_escape_progress_source}, "
        "escape_no_motion_reason="
        f"{recovery.last_escape_no_motion_reason}, "
        "escape_odom_source="
        f"{recovery.last_escape_odom_source}, "
        "escape_odom_source_fallback_reason="
        f"{recovery.last_escape_odom_source_fallback_reason}, "
        "escape_direct_odom_distance_m="
        f"{_format_optional_m(recovery.last_escape_direct_odom_distance_m)}, "
        "escape_tf_odom_distance_m="
        f"{_format_optional_m(recovery.last_escape_tf_odom_distance_m)}, "
        "escape_direct_odom_age_sec="
        f"{_format_optional_m(recovery.last_escape_direct_odom_age_sec)}, "
        "escape_direct_odom_stamp_delta_sec="
        f"{_format_optional_m(recovery.last_escape_direct_odom_stamp_delta_sec)}, "
        "escape_tf_odom_stamp_delta_sec="
        f"{_format_optional_m(recovery.last_escape_tf_odom_stamp_delta_sec)}, "
        "escape_direct_odom_frame_id="
        f"{recovery.last_escape_direct_odom_frame_id}, "
        "escape_direct_odom_child_frame_id="
        f"{recovery.last_escape_direct_odom_child_frame_id}, "
        "escape_odom_disagreement="
        f"{recovery.last_escape_odom_disagreement}, "
        "escape_elapsed_sec="
        f"{_format_optional_m(recovery.last_escape_elapsed_sec)}, "
        "escape_timeout_sec="
        f"{_format_optional_m(recovery.last_escape_timeout_sec)}, "
        "escape_completion_tolerance_m="
        f"{POST_REPLAN_ESCAPE_COMPLETION_TOLERANCE_M:.3f}, "
        "escape_command_linear_mps="
        f"{recovery.last_escape_command_linear_mps:.3f}, "
        "escape_command_angular_radps="
        f"{recovery.last_escape_command_angular_radps:.3f}, "
        "escape_angular_hint_source="
        f"{recovery.last_escape_angular_hint_source}, "
        "escape_angular_hint_cap_radps="
        f"{POST_REPLAN_ESCAPE_ANGULAR_HINT_CAP_RADPS:.3f}, "
        "escape_straight_until_progress_active="
        f"{recovery.escape_straight_until_progress_active}, "
        "escape_no_motion_elapsed_sec="
        f"{_format_optional_m(recovery.last_escape_no_motion_elapsed_sec)}, "
        "clearance_search_attempted="
        f"{recovery.clearance_search_attempted}, "
        "clearance_search_direction="
        f"{recovery.clearance_search_direction:.1f}, "
        "clearance_search_direction_source="
        f"{recovery.clearance_search_direction_source}, "
        "clearance_search_yaw_delta_deg="
        f"{recovery.clearance_search_yaw_delta_deg:.3f}, "
        "clearance_search_baseline_p05_m="
        f"{_format_optional_m(recovery.clearance_search_baseline_p05_m)}, "
        "clearance_search_best_p05_m="
        f"{_format_optional_m(recovery.clearance_search_best_p05_m)}, "
        "clearance_search_baseline_min_m="
        f"{_format_optional_m(recovery.clearance_search_baseline_min_m)}, "
        "clearance_search_best_min_m="
        f"{_format_optional_m(recovery.clearance_search_best_min_m)}, "
        "clearance_search_result="
        f"{recovery.clearance_search_result}, "
        "route_clearance_reason="
        f"{recovery.route_clearance_reason}, "
        "route_corridor_min_distance_m="
        f"{_format_optional_m(recovery.route_corridor_min_distance_m)}, "
        "route_corridor_blocked_count="
        f"{recovery.route_corridor_blocked_count}, "
        "route_clear_side_obstacle_count="
        f"{recovery.route_clear_side_obstacle_count}, "
        "route_corridor_preview_distance_m="
        f"{recovery.route_corridor_preview_distance_m:.3f}"
    )

def post_replan_escape_angular_hint(node, step):
    if step is None or getattr(step, "command", None) is None:
        return 0.0, "route_hint_unavailable"
    mode = getattr(step, "mode", "")
    if mode == "blocked":
        raise RuntimeError("post_replan_escape_controller_blocked")
    if mode == "off_route":
        raise RuntimeError("post_replan_escape_off_route")
    try:
        angular_z = float(getattr(step.command, "angular_z", 0.0))
    except (TypeError, ValueError):
        return 0.0, "nonfinite"
    if not math.isfinite(angular_z):
        return 0.0, "nonfinite"
    angular_cap = min(
        node.args.pure_pursuit_max_track_angular_speed_radps,
        POST_REPLAN_ESCAPE_ANGULAR_HINT_CAP_RADPS,
    )
    return (
        clamp(
            angular_z,
            -angular_cap,
            angular_cap,
        ),
        "controller",
    )

def post_replan_forward_side_p05(node):
    scan = getattr(node, "last_scan", None)
    if scan is None:
        return None, None
    half_angle_rad = math.radians(node.args.scan_half_angle_deg)
    left_ranges = []
    right_ranges = []
    for index, raw_range in enumerate(scan.ranges):
        if not math.isfinite(raw_range):
            continue
        if raw_range < scan.range_min or raw_range > scan.range_max:
            continue
        angle = normalize_angle_rad(scan.angle_min + index * scan.angle_increment)
        if abs(angle) > half_angle_rad:
            continue
        if angle > 0.0:
            left_ranges.append(float(raw_range))
        elif angle < 0.0:
            right_ranges.append(float(raw_range))
    left_p05 = percentile(left_ranges, 5.0) if left_ranges else None
    right_p05 = percentile(right_ranges, 5.0) if right_ranges else None
    return left_p05, right_p05

def post_replan_clearance_search_direction(node, heading_error_deg):
    left_p05, right_p05 = post_replan_forward_side_p05(node)
    if left_p05 is not None and right_p05 is not None:
        if left_p05 + POST_REPLAN_CLEARANCE_SIDE_DIFF_M < right_p05:
            return -1.0, "left_obstacle"
        if right_p05 + POST_REPLAN_CLEARANCE_SIDE_DIFF_M < left_p05:
            return 1.0, "right_obstacle"
    if heading_error_deg is not None and abs(heading_error_deg) > 1e-6:
        return (1.0 if heading_error_deg > 0.0 else -1.0), "route_heading"
    return 1.0, "deterministic_left"

def start_post_replan_clearance_search(node, recovery, pose, safety, heading_error_deg):
    direction, direction_source = post_replan_clearance_search_direction(
        node,
        heading_error_deg,
    )
    recovery.phase = POST_REPLAN_RECOVERY_CLEARANCE_SEARCH
    recovery.clear_scan_count = 0
    recovery.clearance_search_attempted = True
    recovery.clearance_search_direction = direction
    recovery.clearance_search_direction_source = direction_source
    recovery.clearance_search_start_yaw_deg = pose.yaw_deg
    recovery.clearance_search_baseline_p05_m = safety.percentile_5_m
    recovery.clearance_search_best_p05_m = safety.percentile_5_m
    recovery.clearance_search_baseline_min_m = safety.min_range_m
    recovery.clearance_search_best_min_m = safety.min_range_m
    recovery.clearance_search_last_scan_identity = node.current_scan_identity()
    recovery.clearance_search_yaw_delta_deg = 0.0
    recovery.clearance_search_result = "active"
    _reset_command_smoother(node)

def post_replan_clearance_scan_is_new(node, recovery):
    return (
        scan_is_fresh_for_post_replan_recovery(node, recovery)
        and recovery.clearance_search_last_scan_identity != node.current_scan_identity()
    )

def enter_post_replan_wait_clear(node, recovery, reason):
    recovery.phase = POST_REPLAN_RECOVERY_WAIT_CLEAR
    recovery.clear_scan_count = 0
    recovery.clearance_search_result = reason
    _reset_command_smoother(node)
    node.publish_velocity(0.0, 0.0)
    node.wait_one_control_cycle()
    return True

def fail_post_replan_clearance_search(node, recovery, reason):
    recovery.clearance_search_result = reason
    _reset_command_smoother(node)
    node.publish_velocity(0.0, 0.0)
    reset_post_replan_recovery(
        node,
        "post_replan_clearance_search_failed",
    )
    raise RuntimeError("post_replan_clearance_search_failed")

def handle_post_replan_recovery(
    node,
    step,
    pose,
    now_sec,
    route_state=None,
    blocked_error_type=RuntimeError,
):
    recovery = getattr(node, "post_replan_recovery", None)
    if recovery is None:
        return False
    if recovery.route_generation_id != node.active_route_generation_id:
        reset_post_replan_recovery(
            node,
            "route_generation_changed",
        )
        return False

    if recovery.phase == POST_REPLAN_RECOVERY_ALIGN:
        safety = node.evaluate_current_scan_safety("rotate")
        recovery.last_scan_reason = safety.reason
        if safety.reason == "hard_stop":
            reset_post_replan_recovery(node, "hard_stop")
            raise blocked_error_type(safety)
        if not safety.safe:
            recovery.clear_scan_count = 0
            node.maybe_log_post_replan_recovery(
                safety,
                recovery.last_heading_error_deg,
            )
            if post_replan_recovery_timed_out(
                node,
                recovery,
                now_sec,
            ):
                reason = post_replan_recovery_timeout_reason(
                    node,
                    recovery,
                )
                _reset_command_smoother(node)
                node.publish_velocity(0.0, 0.0)
                reset_post_replan_recovery(node, reason)
                raise RuntimeError(reason)
            node.publish_velocity(0.0, 0.0)
            node.wait_one_control_cycle()
            return True
        alignment = (
            post_replan_alignment_heading(
                node,
                pose,
                route_state,
            )
            if route_state is not None
            else None
        )
        if alignment is None:
            reason = "post_replan_alignment_unavailable"
            _reset_command_smoother(node)
            node.publish_velocity(0.0, 0.0)
            reset_post_replan_recovery(node, reason)
            raise RuntimeError(reason)
        recovery.route_heading_deg = alignment.heading_deg
        recovery.last_alignment_heading_deg = alignment.heading_deg
        recovery.last_alignment_heading_source = alignment.source
        recovery.last_alignment_projection_segment_index = (
            alignment.projection_segment_index
        )
        recovery.last_alignment_projection_segment_ratio = (
            alignment.projection_segment_ratio
        )
        heading_error_deg = shortest_angle_delta_deg(
            pose.yaw_deg,
            alignment.heading_deg,
        )
        recovery.last_heading_error_deg = heading_error_deg
        node.maybe_log_post_replan_recovery(safety, heading_error_deg)
        if post_replan_recovery_timed_out(
            node,
            recovery,
            now_sec,
        ):
            reason = post_replan_recovery_timeout_reason(
                node,
                recovery,
            )
            _reset_command_smoother(node)
            node.publish_velocity(0.0, 0.0)
            reset_post_replan_recovery(node, reason)
            raise RuntimeError(reason)
        if abs(heading_error_deg) > node.args.post_replan_align_heading_error_deg:
            angular_z = clamp(
                math.radians(heading_error_deg) * node.args.yaw_gain,
                -node.args.pure_pursuit_max_rotate_angular_speed_radps,
                node.args.pure_pursuit_max_rotate_angular_speed_radps,
            )
            node.publish_velocity(0.0, angular_z)
            node.wait_one_control_cycle()
            return True
        forward_safety = post_replan_forward_clearance_safety(
            node,
            pose,
            route_state,
        )
        recovery.last_scan_reason = forward_safety.reason
        if forward_safety.reason == "hard_stop":
            reset_post_replan_recovery(node, "hard_stop")
            raise blocked_error_type(forward_safety)
        if (
            forward_safety.reason == "soft_stop"
            and not recovery.clearance_search_attempted
        ):
            start_post_replan_clearance_search(
                node,
                recovery,
                pose,
                forward_safety,
                heading_error_deg,
            )
            node.maybe_log_post_replan_recovery(forward_safety, heading_error_deg)
            node.publish_velocity(0.0, 0.0)
            node.wait_one_control_cycle()
            return True
        recovery.phase = POST_REPLAN_RECOVERY_WAIT_CLEAR
        recovery.clear_scan_count = 0
        _reset_command_smoother(node)
        node.publish_velocity(0.0, 0.0)
        node.wait_one_control_cycle()
        return True

    if recovery.phase == POST_REPLAN_RECOVERY_CLEARANCE_SEARCH:
        rotate_safety = node.evaluate_current_scan_safety("rotate")
        recovery.last_scan_reason = rotate_safety.reason
        if not rotate_safety.safe:
            _reset_command_smoother(node)
            node.publish_velocity(0.0, 0.0)
            reset_post_replan_recovery(node, rotate_safety.reason)
            raise blocked_error_type(rotate_safety)
        if post_replan_recovery_timed_out(
            node,
            recovery,
            now_sec,
        ):
            fail_post_replan_clearance_search(
                node,
                recovery,
                "timeout",
            )
        forward_safety = post_replan_forward_clearance_safety(
            node,
            pose,
            route_state,
        )
        recovery.last_scan_reason = forward_safety.reason
        if forward_safety.reason == "hard_stop":
            _reset_command_smoother(node)
            node.publish_velocity(0.0, 0.0)
            reset_post_replan_recovery(node, "hard_stop")
            raise blocked_error_type(forward_safety)
        if post_replan_clearance_scan_is_new(node, recovery):
            recovery.clearance_search_last_scan_identity = node.current_scan_identity()
            if forward_safety.percentile_5_m is not None:
                if recovery.clearance_search_best_p05_m is None:
                    recovery.clearance_search_best_p05_m = forward_safety.percentile_5_m
                else:
                    recovery.clearance_search_best_p05_m = max(
                        recovery.clearance_search_best_p05_m,
                        forward_safety.percentile_5_m,
                    )
            if forward_safety.min_range_m is not None:
                if recovery.clearance_search_best_min_m is None:
                    recovery.clearance_search_best_min_m = forward_safety.min_range_m
                else:
                    recovery.clearance_search_best_min_m = max(
                        recovery.clearance_search_best_min_m,
                        forward_safety.min_range_m,
                    )
            baseline_p05 = recovery.clearance_search_baseline_p05_m
            if forward_safety.safe:
                return enter_post_replan_wait_clear(
                    node,
                    recovery,
                    forward_safety.reason,
                )
            if (
                baseline_p05 is not None
                and forward_safety.percentile_5_m is not None
                and forward_safety.percentile_5_m
                >= baseline_p05 + POST_REPLAN_CLEARANCE_IMPROVEMENT_M - 1e-9
            ):
                return enter_post_replan_wait_clear(
                    node,
                    recovery,
                    "p05_improved",
                )
        start_yaw = (
            pose.yaw_deg
            if recovery.clearance_search_start_yaw_deg is None
            else recovery.clearance_search_start_yaw_deg
        )
        recovery.clearance_search_yaw_delta_deg = abs(
            shortest_angle_delta_deg(start_yaw, pose.yaw_deg)
        )
        node.maybe_log_post_replan_recovery(
            forward_safety,
            recovery.last_heading_error_deg,
        )
        if recovery.clearance_search_yaw_delta_deg >= POST_REPLAN_CLEARANCE_MAX_YAW_DEG:
            fail_post_replan_clearance_search(
                node,
                recovery,
                "yaw_limit",
            )
        angular_limit = min(
            node.args.pure_pursuit_max_rotate_angular_speed_radps,
            POST_REPLAN_CLEARANCE_MAX_ANGULAR_RADPS,
        )
        angular_z = recovery.clearance_search_direction * angular_limit
        node.publish_velocity(0.0, angular_z)
        node.wait_one_control_cycle()
        return True

    if recovery.phase == POST_REPLAN_RECOVERY_WAIT_CLEAR:
        safety = post_replan_forward_clearance_safety(
            node,
            pose,
            route_state,
        )
        recovery.last_scan_reason = safety.reason
        node.maybe_log_post_replan_recovery(safety, recovery.last_heading_error_deg)
        if safety.reason == "hard_stop":
            reset_post_replan_recovery(node, "hard_stop")
            raise blocked_error_type(safety)
        if post_replan_recovery_timed_out(
            node,
            recovery,
            now_sec,
        ):
            reason = post_replan_recovery_timeout_reason(
                node,
                recovery,
            )
            _reset_command_smoother(node)
            node.publish_velocity(0.0, 0.0)
            reset_post_replan_recovery(node, reason)
            raise RuntimeError(reason)
        if not safety.safe:
            recovery.clear_scan_count = 0
            if (
                safety.reason == "soft_stop"
                and not recovery.clearance_search_attempted
                and scan_is_fresh_for_post_replan_recovery(
                    node,
                    recovery,
                )
            ):
                start_post_replan_clearance_search(
                    node,
                    recovery,
                    pose,
                    safety,
                    recovery.last_heading_error_deg,
                )
                node.maybe_log_post_replan_recovery(
                    safety,
                    recovery.last_heading_error_deg,
                )
            node.publish_velocity(0.0, 0.0)
            node.wait_one_control_cycle()
            return True
        if (
            scan_is_fresh_for_post_replan_recovery(node, recovery)
            and not scan_already_counted_for_post_replan_recovery(
                node,
                recovery,
            )
        ):
            recovery.clear_scan_count += 1
            recovery.last_counted_scan_identity = node.current_scan_identity()
            node.max_post_replan_recovery_clear_count = max(
                node.max_post_replan_recovery_clear_count,
                recovery.clear_scan_count,
            )
        if recovery.clear_scan_count >= node.args.post_replan_clear_scan_samples:
            recovery.phase = POST_REPLAN_RECOVERY_ESCAPE
            recovery.escape_start_pose = pose
            direct_odom_pose, direct_odom_age_sec, direct_odom_reason = (
                fresh_direct_odom_pose(node, now_sec=now_sec)
            )
            recovery.escape_start_direct_odom_pose = direct_odom_pose
            recovery.escape_start_tf_odom_pose = try_lookup_odom_pose(node)
            recovery.escape_start_odom_pose = recovery.escape_start_tf_odom_pose
            recovery.escape_start_time_sec = now_sec
            recovery.last_escape_distance_m = 0.0
            recovery.best_escape_distance_m = 0.0
            recovery.last_progress_distance_m = 0.0
            recovery.last_progress_time_sec = None
            recovery.first_escape_command_time_sec = None
            if recovery.escape_start_direct_odom_pose is not None:
                recovery.last_escape_distance_source = "direct_odom"
                recovery.last_escape_odom_source_fallback_reason = "none"
            elif recovery.escape_start_tf_odom_pose is not None:
                recovery.last_escape_distance_source = "tf_odom"
                recovery.last_escape_odom_source_fallback_reason = (
                    direct_odom_reason or "direct_odom_start_unavailable"
                )
            else:
                recovery.last_escape_distance_source = "unavailable"
                recovery.last_escape_odom_source_fallback_reason = (
                    "progress_unavailable"
                )
            recovery.last_escape_no_motion_elapsed_sec = None
            recovery.last_escape_steering_mode_resolved = (
                resolve_post_replan_escape_steering_mode(node.args)
            )
            recovery.last_escape_odom_distance_m = None
            recovery.last_escape_map_distance_m = None
            recovery.last_escape_odom_stamp_delta_sec = None
            recovery.last_escape_progress_source = recovery.last_escape_distance_source
            recovery.last_escape_no_motion_reason = ""
            recovery.last_escape_odom_source = recovery.last_escape_distance_source
            recovery.last_escape_direct_odom_distance_m = None
            recovery.last_escape_tf_odom_distance_m = None
            recovery.last_escape_direct_odom_age_sec = direct_odom_age_sec
            recovery.last_escape_direct_odom_stamp_delta_sec = None
            recovery.last_escape_tf_odom_stamp_delta_sec = None
            recovery.last_escape_direct_odom_frame_id = getattr(
                node,
                "last_odom_frame_id",
                "",
            )
            recovery.last_escape_direct_odom_child_frame_id = getattr(
                node,
                "last_odom_child_frame_id",
                "",
            )
            recovery.last_escape_odom_disagreement = ""
            recovery.escape_straight_until_progress_active = (
                recovery.last_escape_steering_mode_resolved
                == POST_REPLAN_ESCAPE_STEERING_STRAIGHT_UNTIL_PROGRESS
            )
            recovery.last_escape_elapsed_sec = 0.0
            recovery.last_escape_timeout_sec = (
                post_replan_escape_timeout_sec(node)
            )
            _reset_command_smoother(node)
        node.publish_velocity(0.0, 0.0)
        node.wait_one_control_cycle()
        return True

    if recovery.phase == POST_REPLAN_RECOVERY_ESCAPE:
        if step is not None and getattr(step, "reached", False):
            _reset_command_smoother(node)
            reset_post_replan_recovery(node, "reached")
            return False
        safety = post_replan_forward_clearance_safety(
            node,
            pose,
            route_state,
        )
        recovery.last_scan_reason = safety.reason
        if safety.reason == "hard_stop":
            _reset_command_smoother(node)
            node.publish_velocity(0.0, 0.0)
            reset_post_replan_recovery(node, "hard_stop")
            raise blocked_error_type(safety)
        if not safety.safe:
            _reset_command_smoother(node)
            node.publish_velocity(0.0, 0.0)
            reset_post_replan_recovery(
                node,
                "post_replan_escape_blocked",
            )
            raise RuntimeError("post_replan_escape_blocked")
        escape_measurement = post_replan_escape_measurement(
            node,
            recovery,
            pose,
            now_sec=now_sec,
        )
        update_post_replan_escape_progress(
            node,
            recovery,
            escape_measurement,
            now_sec,
        )
        node.last_post_replan_recovery_escape_distance_m = (
            recovery.last_escape_distance_m
        )
        resolved_steering = resolve_post_replan_escape_steering_mode(node.args)
        recovery.last_escape_steering_mode_resolved = resolved_steering
        recovery.escape_straight_until_progress_active = (
            resolved_steering
            == POST_REPLAN_ESCAPE_STEERING_STRAIGHT_UNTIL_PROGRESS
            and recovery.best_escape_distance_m
            < POST_REPLAN_ESCAPE_STRAIGHT_UNTIL_PROGRESS_M
        )
        escape_timed_out = post_replan_escape_timed_out(
            node,
            recovery,
            now_sec,
        )
        if (
            recovery.best_escape_distance_m + POST_REPLAN_ESCAPE_COMPLETION_TOLERANCE_M
            >= node.args.post_replan_escape_distance_m
        ):
            recovery.phase = POST_REPLAN_RECOVERY_DONE
            reset_post_replan_recovery(node, "done")
            _reset_command_smoother(node)
            return False
        linear_x = max(0.0, node.args.post_replan_escape_linear_speed_mps)
        if resolved_steering == POST_REPLAN_ESCAPE_STEERING_ROUTE_HINT:
            try:
                angular_z, angular_hint_source = (
                    post_replan_escape_angular_hint(node, step)
                )
            except RuntimeError as exc:
                reason = str(exc)
                _reset_command_smoother(node)
                node.publish_velocity(0.0, 0.0)
                reset_post_replan_recovery(node, reason)
                raise
        elif recovery.escape_straight_until_progress_active:
            angular_z = 0.0
            angular_hint_source = "straight_until_progress"
        else:
            try:
                angular_z, angular_hint_source = (
                    post_replan_escape_angular_hint(node, step)
                )
            except RuntimeError as exc:
                reason = str(exc)
                _reset_command_smoother(node)
                node.publish_velocity(0.0, 0.0)
                reset_post_replan_recovery(node, reason)
                raise
        recovery.last_escape_command_linear_mps = linear_x
        recovery.last_escape_command_angular_radps = angular_z
        recovery.last_escape_angular_hint_source = angular_hint_source
        no_motion_timed_out = post_replan_escape_no_motion_timed_out(
            node,
            recovery,
            linear_x,
        )
        node.maybe_log_post_replan_recovery(safety, recovery.last_heading_error_deg)
        if no_motion_timed_out:
            reason = "post_replan_escape_no_motion"
            _reset_command_smoother(node)
            node.publish_velocity(0.0, 0.0)
            reset_post_replan_recovery(node, reason)
            raise RuntimeError(reason)
        if escape_timed_out:
            reason = post_replan_recovery_timeout_reason(
                node,
                recovery,
            )
            _reset_command_smoother(node)
            node.publish_velocity(0.0, 0.0)
            reset_post_replan_recovery(node, reason)
            raise RuntimeError(reason)
        node.record_motion_sample(
            getattr(step, "yaw_error_deg", 0.0) if step is not None else 0.0,
            linear_x,
            angular_z,
            1.0 / node.args.control_rate_hz,
        )
        node.publish_velocity(linear_x, angular_z)
        if recovery.first_escape_command_time_sec is None:
            recovery.first_escape_command_time_sec = now_sec
            recovery.last_progress_time_sec = now_sec
            recovery.last_progress_distance_m = recovery.best_escape_distance_m
            recovery.last_escape_no_motion_elapsed_sec = 0.0
        node.wait_one_control_cycle()
        return True

    reset_post_replan_recovery(node, "unknown_phase")
    return False
