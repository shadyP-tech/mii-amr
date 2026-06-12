from __future__ import annotations

import math
from dataclasses import dataclass

import lidar_obstacle_map
import replan_runtime

from . import rviz_visualization
from .math_utils import clamp, shortest_angle_delta_deg
from .models import TrackingPathValidation, Waypoint
from .path_curves import PATH_SEGMENT_EPS_M, project_point_to_route, route_cumulative_distances
from .path_progress import (
    validate_tracking_path_geometry,
    validate_tracking_point_structure,
    waypoint_distance,
)
from .post_replan_recovery import POST_REPLAN_ACTIVATION_MIN_TARGET_FLOOR_M


DEFAULT_CONTROLLER = "stop-go"
DEFAULT_PATH_LOOKAHEAD_M = 0.18
DEFAULT_TRACKING_ENDPOINT_TOLERANCE_M = 0.10
DEFAULT_TRACKING_START_TOLERANCE_M = 0.20
DEFAULT_TRACKING_MAX_SEGMENT_M = 0.30
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


@dataclass(frozen=True)
class PostReplanActivationRoute:
    waypoints: list[Waypoint]
    tracking_points: list[Waypoint] | None
    tracking_source: str
    tracking_validation: TrackingPathValidation | None
    status: str
    goal_reached: bool
    min_target_distance_m: float
    pruned_sparse_count: int
    pruned_dense_count: int
    projection_progress_m: float | None
    first_target_distance_m: float | None


def _format_optional_m(value):
    return "n/a" if value is None else f"{value:.3f}"


def run_local_map_has_confirmed_obstacles(run_local_map):
    if run_local_map is None:
        return False
    return bool(getattr(run_local_map, "confirmed_raw_cells", None))


def lidar_replan_failure(reason):
    message = str(reason)
    if message.startswith("lidar_replan_failed:"):
        return RuntimeError(message)
    return RuntimeError(f"lidar_replan_failed:{message}")


class ReplanManager:
    def __init__(self, runtime):
        self.runtime = runtime

    def update_diagnostics(self, result, count_replan=True):
        return update_replan_diagnostics(self.runtime, result, count_replan=count_replan)

    def initialize_route(self, current_pose, waypoints):
        return initialize_run_local_route(self.runtime, current_pose, waypoints)

    def replan_after_blockage(self, current_pose, old_remaining_waypoints, trigger):
        return replan_after_blockage(
            self.runtime,
            current_pose,
            old_remaining_waypoints,
            trigger=trigger,
        )

    def prune_after_progress(self, current_pose, remaining_waypoints):
        return prune_run_local_obstacles_after_progress(
            self.runtime,
            current_pose,
            remaining_waypoints,
        )

    def corridor_blocked_cells(self, current_pose, remaining_waypoints):
        return corridor_blocked_cells(
            self.runtime,
            current_pose,
            remaining_waypoints,
        )

    def validate_result(
        self,
        result,
        current_pose,
        old_remaining_waypoints,
        goal_waypoint,
        require_changed=True,
    ):
        return validate_replan_result(
            self.runtime,
            result,
            current_pose,
            old_remaining_waypoints,
            goal_waypoint,
            require_changed=require_changed,
        )

def update_replan_diagnostics(node, result, count_replan=True):
    diag = result.diagnostics
    if count_replan:
        node.diagnostics.replan_count += 1
    node.diagnostics.last_replan_reason = result.reason
    node.diagnostics.updated_map_yaml = result.updated_map_yaml or ""
    node.diagnostics.updated_waypoints_csv = result.updated_waypoints_csv or ""
    node.diagnostics.detected_obstacle_count = diag.detected_obstacle_count
    node.diagnostics.candidate_scan_points = diag.candidate_scan_points
    node.diagnostics.filtered_obstacle_points = diag.filtered_obstacle_points
    node.diagnostics.raw_obstacle_cells = diag.raw_obstacle_cells
    node.diagnostics.free_obstacle_cells = diag.free_obstacle_cells
    node.diagnostics.inflated_cells_total = diag.inflated_cells_total
    node.diagnostics.inflated_cells_newly_occupied = diag.inflated_cells_newly_occupied
    node.diagnostics.inflated_cells_over_static_occupied = diag.inflated_cells_over_static_occupied
    node.diagnostics.scan_frame = diag.scan_frame
    node.diagnostics.scan_age_sec = diag.scan_age_sec
    node.diagnostics.tf_age_sec = diag.tf_age_sec
    node.diagnostics.tf_lookup_mode = diag.tf_lookup_mode
    node.diagnostics.start_snap_distance_m = diag.start_snap_distance_m
    node.diagnostics.goal_snap_distance_m = diag.goal_snap_distance_m
    node.diagnostics.old_remaining_waypoint_count = diag.old_remaining_waypoint_count
    node.diagnostics.new_waypoint_count = diag.new_waypoint_count
    node.diagnostics.old_path_length_m = diag.old_path_length_m
    node.diagnostics.new_path_length_m = diag.new_path_length_m
    node.diagnostics.replan_duration_sec = diag.replan_duration_sec
    node.diagnostics.run_local_map_updates = diag.run_local_map_updates
    node.diagnostics.run_local_replan_count += diag.run_local_replan_count
    node.diagnostics.run_local_last_replan_reason = diag.run_local_last_replan_reason
    node.diagnostics.run_local_no_path_reason = diag.run_local_no_path_reason
    node.diagnostics.run_local_start_cell_blocked = diag.run_local_start_cell_blocked
    node.diagnostics.run_local_goal_cell_blocked = diag.run_local_goal_cell_blocked
    node.diagnostics.run_local_path_blocked_cell_count = diag.run_local_path_blocked_cell_count
    node.diagnostics.run_local_scan_points_valid = diag.run_local_scan_points_valid
    node.diagnostics.run_local_scan_points_used = diag.run_local_scan_points_used
    node.diagnostics.run_local_scan_points_rejected_invalid_range = (
        diag.run_local_scan_points_rejected_invalid_range
    )
    node.diagnostics.run_local_scan_points_rejected_static = diag.run_local_scan_points_rejected_static
    node.diagnostics.run_local_scan_points_rejected_bounds = diag.run_local_scan_points_rejected_bounds
    node.diagnostics.run_local_scan_points_rejected_wall_band = (
        diag.run_local_scan_points_rejected_wall_band
    )
    node.diagnostics.run_local_scan_points_rejected_low_confidence = (
        diag.run_local_scan_points_rejected_low_confidence
    )
    node.diagnostics.run_local_update_rejected_reason = diag.run_local_update_rejected_reason
    node.diagnostics.run_local_initial_scan_count = max(
        node.diagnostics.run_local_initial_scan_count,
        diag.run_local_initial_scan_count,
    )
    node.diagnostics.run_local_corridor_check_distance_m = (
        diag.run_local_corridor_check_distance_m
    )
    node.diagnostics.run_local_inflation_radius_m = diag.run_local_inflation_radius_m
    node.diagnostics.run_local_map_yaml = diag.run_local_map_yaml
    node.diagnostics.run_local_waypoints_csv = diag.run_local_waypoints_csv
    node.diagnostics.run_local_cell_source_counts = diag.run_local_cell_source_counts
    if result.run_local_map is not None:
        node.run_local_map = result.run_local_map
        rviz_visualization.publish_rviz_obstacles_if_available(node)

def replanned_waypoints_from_result(node, result):
    return [
        Waypoint(index, x, y)
        for index, x, y in result.waypoints
    ]

def replanned_tracking_points_from_result(node, result):
    path_points = getattr(result, "path_points", None) or []
    converted = []
    for point in path_points:
        if isinstance(point, Waypoint):
            converted.append(point)
        else:
            converted.append(Waypoint(point[0], point[1], point[2]))
    return converted

def remember_replan_tracking_replacement(node, result, replanned, current_pose):
    node.last_replan_tracking_points = None
    node.last_replan_tracking_source = "waypoints"
    node.last_replan_tracking_validation = None
    if getattr(node.args, "controller", DEFAULT_CONTROLLER) != "pure-pursuit":
        return

    path_points = replanned_tracking_points_from_result(
        node,
        result,
    )
    if not path_points:
        node.last_replan_tracking_points = list(replanned)
        node.last_replan_tracking_source = "replan_sparse_fallback"
        node.last_replan_tracking_validation = TrackingPathValidation(
            source="replan_sparse_fallback",
            point_count=len(replanned),
            validation_status="fallback_sparse_waypoints",
        )
        node.get_logger().warn(
            "Pure-pursuit LiDAR replan did not include dense path_points; "
            "falling back to sparse replanned waypoints for tracking."
        )
        return

    structural_warnings = validate_tracking_point_structure(
        path_points,
        max_segment_m=getattr(
            node.args,
            "tracking_max_segment_m",
            DEFAULT_TRACKING_MAX_SEGMENT_M,
        ),
        label="replan tracking path",
    )
    validation = validate_tracking_path_geometry(
        replanned,
        path_points,
        endpoint_tolerance_m=getattr(
            node.args,
            "tracking_endpoint_tolerance_m",
            DEFAULT_TRACKING_ENDPOINT_TOLERANCE_M,
        ),
        start_tolerance_m=getattr(
            node.args,
            "tracking_start_tolerance_m",
            DEFAULT_TRACKING_START_TOLERANCE_M,
        ),
        allow_mismatch=getattr(
            node.args,
            "allow_tracking_path_mismatch",
            False,
        ),
        current_pose=current_pose,
        source="replan",
        structural_warnings=structural_warnings,
    )
    for warning in validation.warnings:
        node.get_logger().warn(warning)
    node.last_replan_tracking_points = path_points
    node.last_replan_tracking_source = validation.source
    node.last_replan_tracking_validation = validation

def first_motion_waypoint(node, replanned, current_pose):
    for waypoint in replanned:
        distance_m = math.hypot(
            waypoint.x - current_pose.x,
            waypoint.y - current_pose.y,
        )
        if distance_m > node.args.waypoint_tolerance_m:
            return waypoint
    return replanned[-1]

def first_motion_waypoint_index(node, replanned, current_pose):
    for index, waypoint in enumerate(replanned):
        distance_m = math.hypot(
            waypoint.x - current_pose.x,
            waypoint.y - current_pose.y,
        )
        if distance_m > node.args.waypoint_tolerance_m:
            return index
    return max(0, len(replanned) - 1)

def replan_start_artifact_distance_limit_m(node):
    start_on_path_tolerance_m = getattr(
        node.args,
        "start_on_path_tolerance_m",
        node.args.waypoint_tolerance_m,
    )
    return max(
        node.args.waypoint_tolerance_m,
        min(0.35, max(0.0, float(start_on_path_tolerance_m))),
    )

def first_forward_motion_waypoint_index(node, replanned, current_pose):
    if not replanned:
        return 0
    first_motion_index = first_motion_waypoint_index(
        node,
        replanned,
        current_pose,
    )
    artifact_distance_limit_m = (
        replan_start_artifact_distance_limit_m(node)
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
        if first_base.x >= -node.args.robot_footprint_radius_m:
            return index
        if distance_m > artifact_distance_limit_m:
            return index
    return max(0, len(replanned) - 1)

def prune_replanned_waypoints_for_progress(node, replanned, current_pose):
    if not replanned:
        return replanned
    index = first_motion_waypoint_index(node, replanned, current_pose)
    return replanned[index:]

def post_replan_activation_min_target_distance_m(node):
    return max(
        float(node.args.goal_tolerance_m),
        0.5
        * float(getattr(node.args, "path_lookahead_m", DEFAULT_PATH_LOOKAHEAD_M)),
        float(getattr(node.args, "post_replan_escape_distance_m", 0.0)),
        POST_REPLAN_ACTIVATION_MIN_TARGET_FLOOR_M,
    )

def _waypoint_xy(point):
    return (
        float(point.x if hasattr(point, "x") else point[0]),
        float(point.y if hasattr(point, "y") else point[1]),
    )

def _projection_on_xy_segment(point, segment_start, segment_end):
    dx = segment_end[0] - segment_start[0]
    dy = segment_end[1] - segment_start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return (
            math.hypot(point[0] - segment_start[0], point[1] - segment_start[1]),
            0.0,
        )
    ratio = (
        (point[0] - segment_start[0]) * dx
        + (point[1] - segment_start[1]) * dy
    ) / length_sq
    ratio = clamp(ratio, 0.0, 1.0)
    closest = (
        segment_start[0] + ratio * dx,
        segment_start[1] + ratio * dy,
    )
    return math.hypot(point[0] - closest[0], point[1] - closest[1]), ratio

def _waypoint_progress_on_route(route_points, cumulative, waypoint, min_progress_m=None):
    if len(route_points) < 2 or not cumulative or cumulative[-1] <= 1e-9:
        return None
    point = _waypoint_xy(waypoint)
    best = None
    progress_floor = None if min_progress_m is None else float(min_progress_m)
    for index in range(len(route_points) - 1):
        start_progress = cumulative[index]
        end_progress = cumulative[index + 1]
        segment_length = end_progress - start_progress
        if segment_length <= 1e-9:
            continue
        distance_m, ratio = _projection_on_xy_segment(
            point,
            route_points[index],
            route_points[index + 1],
        )
        progress_m = start_progress + ratio * segment_length
        if progress_floor is not None and progress_m + 0.03 < progress_floor:
            continue
        candidate = (distance_m, progress_m, index)
        if best is None or candidate < best:
            best = candidate
    if best is None and progress_floor is not None:
        return _waypoint_progress_on_route(
            route_points,
            cumulative,
            waypoint,
            min_progress_m=None,
        )
    return None if best is None else best[1]

def _activation_tracking_validation(validation, tracking_source, point_count):
    if validation is None:
        return TrackingPathValidation(
            source=tracking_source,
            point_count=point_count,
            validation_status="activation_pruned",
        )
    return TrackingPathValidation(
        source=validation.source,
        point_count=point_count,
        endpoint_error_m=validation.endpoint_error_m,
        start_error_m=validation.start_error_m,
        start_projection_error_m=validation.start_projection_error_m,
        validation_status=validation.validation_status,
        warnings=validation.warnings,
    )

def _record_post_replan_activation_route(node, activation):
    node.last_post_replan_activation_min_target_distance_m = (
        activation.min_target_distance_m
    )
    node.last_post_replan_activation_pruned_sparse_count = (
        activation.pruned_sparse_count
    )
    node.last_post_replan_activation_pruned_dense_count = (
        activation.pruned_dense_count
    )
    node.last_post_replan_activation_projection_progress_m = (
        activation.projection_progress_m
    )
    node.last_post_replan_activation_first_target_distance_m = (
        activation.first_target_distance_m
    )
    node.last_post_replan_activation_status = activation.status

def prepare_run_local_route_activation(
    node,
    replanned,
    current_pose,
    goal_waypoint,
    trigger,
):
    replanned = list(replanned)
    trigger = str(trigger)
    min_target_distance_m = (
        post_replan_activation_min_target_distance_m(node)
    )
    tracking_points = getattr(node, "last_replan_tracking_points", None)
    tracking_source = getattr(node, "last_replan_tracking_source", "waypoints")
    tracking_validation = getattr(node, "last_replan_tracking_validation", None)
    projection_progress_m = None
    pruned_dense_count = 0
    pruned_sparse_count = 0
    pruned_tracking_points = tracking_points
    status = "unchanged"

    dense_points = []
    if tracking_points is not None:
        dense_points = [
            _waypoint_xy(point)
            for point in tracking_points
        ]

    if len(dense_points) >= 2:
        projection = project_point_to_route(
            dense_points,
            current_pose,
            allow_backward=True,
            projection_status="post_replan_activation",
        )
        projection_progress_m = projection.route_progress_m
        cumulative = route_cumulative_distances(dense_points)
        target_progress_m = min(
            cumulative[-1],
            projection.route_progress_m + min_target_distance_m,
        )
        projected = projection.projected_point
        pruned_tracking_points = [
            Waypoint(-1, float(projected[0]), float(projected[1])),
        ]
        first_kept_index = None
        for index, point in enumerate(tracking_points):
            progress_m = cumulative[index]
            if (
                progress_m <= projection.route_progress_m + 1e-9
                or progress_m < target_progress_m - 1e-9
            ):
                continue
            if (
                math.hypot(point.x - projected[0], point.y - projected[1])
                <= PATH_SEGMENT_EPS_M
            ):
                continue
            first_kept_index = index
            break
        if first_kept_index is not None:
            pruned_tracking_points.extend(tracking_points[first_kept_index:])
            pruned_dense_count = first_kept_index
        else:
            pruned_dense_count = len(tracking_points)
        tracking_validation = _activation_tracking_validation(
            tracking_validation,
            tracking_source,
            len(pruned_tracking_points),
        )
        pruned_waypoints = []
        previous_progress_m = projection.route_progress_m
        for waypoint in replanned:
            progress_m = _waypoint_progress_on_route(
                dense_points,
                cumulative,
                waypoint,
                min_progress_m=previous_progress_m,
            )
            if progress_m is None:
                distance_m = math.hypot(
                    waypoint.x - current_pose.x,
                    waypoint.y - current_pose.y,
                )
                if distance_m < min_target_distance_m - 1e-9:
                    pruned_sparse_count += 1
                    continue
                pruned_waypoints.append(waypoint)
                continue
            previous_progress_m = max(previous_progress_m, progress_m)
            if (
                progress_m - projection.route_progress_m
                < min_target_distance_m - 1e-9
            ):
                pruned_sparse_count += 1
                continue
            pruned_waypoints.append(waypoint)
        status = "dense_progress_pruned"
    else:
        pruned_waypoints = []
        for waypoint in replanned:
            distance_m = math.hypot(
                waypoint.x - current_pose.x,
                waypoint.y - current_pose.y,
            )
            if distance_m < min_target_distance_m - 1e-9:
                pruned_sparse_count += 1
                continue
            pruned_waypoints.append(waypoint)
        status = "sparse_distance_pruned"

    if pruned_sparse_count == 0 and pruned_dense_count == 0:
        status = "no_prune_needed"

    escape_distance_m = float(
        getattr(node.args, "post_replan_escape_distance_m", 0.0)
    )
    while pruned_waypoints:
        first_distance_m = math.hypot(
            pruned_waypoints[0].x - current_pose.x,
            pruned_waypoints[0].y - current_pose.y,
        )
        if first_distance_m + 1e-9 >= escape_distance_m:
            break
        pruned_waypoints = pruned_waypoints[1:]
        pruned_sparse_count += 1

    if (
        pruned_tracking_points is not None
        and len(pruned_tracking_points) < 2
        and pruned_waypoints
    ):
        first = pruned_waypoints[0]
        anchor = pruned_tracking_points[0]
        if math.hypot(first.x - anchor.x, first.y - anchor.y) > PATH_SEGMENT_EPS_M:
            pruned_tracking_points = list(pruned_tracking_points) + [first]
            tracking_validation = _activation_tracking_validation(
                tracking_validation,
                tracking_source,
                len(pruned_tracking_points),
            )

    first_target_distance_m = (
        None
        if not pruned_waypoints
        else math.hypot(
            pruned_waypoints[0].x - current_pose.x,
            pruned_waypoints[0].y - current_pose.y,
        )
    )
    if not pruned_waypoints:
        goal_distance_m = (
            math.inf
            if goal_waypoint is None
            else math.hypot(
                goal_waypoint.x - current_pose.x,
                goal_waypoint.y - current_pose.y,
            )
        )
        goal_reached = goal_distance_m <= node.args.goal_tolerance_m
        status = "goal_reached" if goal_reached else "no_meaningful_target"
        status = f"{trigger}_{status}"
        activation = PostReplanActivationRoute(
            waypoints=[],
            tracking_points=pruned_tracking_points,
            tracking_source=tracking_source,
            tracking_validation=tracking_validation,
            status=status,
            goal_reached=goal_reached,
            min_target_distance_m=min_target_distance_m,
            pruned_sparse_count=pruned_sparse_count,
            pruned_dense_count=pruned_dense_count,
            projection_progress_m=projection_progress_m,
            first_target_distance_m=None,
        )
        _record_post_replan_activation_route(node, activation)
        if node.args.verbose:
            node.get_logger().info(
                "Post-replan route activation pruning: "
                f"status={status}, min_target_distance_m={min_target_distance_m:.3f}, "
                f"pruned_sparse={pruned_sparse_count}, pruned_dense={pruned_dense_count}, "
                "projection_progress_m="
                f"{_format_optional_m(projection_progress_m)}, "
                "first_target_distance_m=n/a"
        )
        return activation

    status = f"{trigger}_{status}"
    activation = PostReplanActivationRoute(
        waypoints=pruned_waypoints,
        tracking_points=pruned_tracking_points,
        tracking_source=tracking_source,
        tracking_validation=tracking_validation,
        status=status,
        goal_reached=False,
        min_target_distance_m=min_target_distance_m,
        pruned_sparse_count=pruned_sparse_count,
        pruned_dense_count=pruned_dense_count,
        projection_progress_m=projection_progress_m,
        first_target_distance_m=first_target_distance_m,
    )
    _record_post_replan_activation_route(node, activation)
    if node.args.verbose:
        node.get_logger().info(
            "Post-replan route activation pruning: "
            f"status={status}, min_target_distance_m={min_target_distance_m:.3f}, "
            f"pruned_sparse={pruned_sparse_count}, pruned_dense={pruned_dense_count}, "
            "projection_progress_m="
            f"{_format_optional_m(projection_progress_m)}, "
            "first_target_distance_m="
            f"{_format_optional_m(first_target_distance_m)}"
        )
    return activation

def route_signature(node, waypoints):
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

def remember_known_corridor_repair(node, waypoints):
    node.last_known_corridor_repair_signature = route_signature(node, waypoints)
    node.suppressed_known_corridor_signature = None

def suppress_repeated_known_corridor_repair(node, waypoints):
    signature = route_signature(node, waypoints)
    if (
        not signature
        or signature != getattr(node, "last_known_corridor_repair_signature", None)
    ):
        return False
    if signature != getattr(node, "suppressed_known_corridor_signature", None):
        node.get_logger().warn(
            "Known corridor blockage still overlaps the same repaired route; "
            "continuing under live scan safety instead of replanning again."
        )
    node.suppressed_known_corridor_signature = signature
    return True

def scan_block_budget_repair_signature(node, current_pose, waypoints):
    return (
        round(current_pose.x, 2),
        round(current_pose.y, 2),
        route_signature(node, waypoints),
    )

def remember_scan_block_budget_repair(node, current_pose, waypoints):
    signature = scan_block_budget_repair_signature(
        node,
        current_pose,
        waypoints,
    )
    if (
        signature
        and signature
        == getattr(node, "last_scan_block_budget_repair_signature", None)
    ):
        raise RuntimeError(
            "lidar_replan_failed:"
            "persistent_scan_blockage_after_existing_map_repair"
        )
    node.last_scan_block_budget_repair_signature = signature

def validate_replan_result(
    node,
    result,
    current_pose,
    old_remaining_waypoints,
    goal_waypoint,
    require_changed=True,
):
    if not result.success:
        raise RuntimeError(f"lidar_replan_failed:{result.reason}")
    replanned = replanned_waypoints_from_result(node, result)
    if not replanned:
        raise RuntimeError("lidar_replan_failed:empty_waypoint_list")
    final_error = waypoint_distance(replanned[-1], goal_waypoint)
    if final_error > node.args.goal_tolerance_m:
        raise RuntimeError(
            "lidar_replan_failed:final_goal_mismatch "
            f"error={final_error:.3f}"
        )
    old_pairs = [(round(wp.x, 3), round(wp.y, 3)) for wp in old_remaining_waypoints]
    new_pairs = [(round(wp.x, 3), round(wp.y, 3)) for wp in replanned]
    if require_changed and old_pairs == new_pairs:
        raise RuntimeError("lidar_replan_failed:updated_path_matches_old_path")
    first_motion_index = first_motion_waypoint_index(
        node,
        replanned,
        current_pose,
    )
    forward_motion_index = first_forward_motion_waypoint_index(
        node,
        replanned,
        current_pose,
    )
    motion_waypoint = replanned[forward_motion_index]
    first_base = lidar_obstacle_map.map_point_to_base(
        motion_waypoint.x,
        motion_waypoint.y,
        lidar_obstacle_map.Pose2D(current_pose.x, current_pose.y, current_pose.yaw_deg),
    )
    if first_base.x < -node.args.robot_footprint_radius_m:
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
        logger = node.get_logger() if hasattr(node, "get_logger") else None
        if logger is not None:
            logger.warn(
                "Pruned behind-the-robot startup waypoint(s) from LiDAR replan: "
                f"removed={forward_motion_index}"
            )
        replanned = replanned[forward_motion_index:]
    remember_replan_tracking_replacement(
        node,
        result,
        replanned,
        current_pose,
    )
    return replanned

def initialize_run_local_route(node, current_pose, waypoints):
    if node.args.run_local_map_initial_scan_mode == "none":
        return list(waypoints)
    node.stop_repeatedly()
    goal_waypoint = waypoints[-1]
    result = replan_runtime.perform_initial_run_local_replan(
        node,
        node.args,
        current_pose,
        goal_waypoint,
        waypoints,
    )
    update_replan_diagnostics(node, result, count_replan=result.success)
    if not result.success and result.reason in INITIAL_RUN_LOCAL_MAP_NONFATAL_REASONS:
        node.stop_repeatedly()
        if not run_local_map_has_confirmed_obstacles(node.run_local_map):
            node.run_local_map = None
        node.get_logger().warn(
            "Initial run-local obstacle map did not find a confirmed "
            f"free-space obstacle; continuing with the static route. reason={result.reason}"
        )
        return list(waypoints)
    replanned = validate_replan_result(
        node,
        result,
        current_pose,
        waypoints,
        goal_waypoint,
        require_changed=False,
    )
    node.stop_repeatedly()
    node.get_logger().info(
        "Initial run-local obstacle map completed: "
        f"waypoints={len(replanned)}, map={result.updated_map_yaml}"
    )
    return replanned

def corridor_blocked_cells(node, current_pose, remaining_waypoints):
    if node.run_local_map is None or not remaining_waypoints:
        return set()
    check_distance_m = node.args.run_local_map_corridor_check_distance_m
    corridor_radius_m = (
        node.args.run_local_map_corridor_radius_m
        if node.args.run_local_map_corridor_radius_m is not None
        else 0.0
    )
    blocked = lidar_obstacle_map.path_corridor_blocked_cells(
        node.run_local_map.static_map,
        lidar_obstacle_map.Pose2D(
            current_pose.x,
            current_pose.y,
            current_pose.yaw_deg,
        ),
        remaining_waypoints,
        node.run_local_map.inflated_obstacle_cells,
        check_distance_m,
        corridor_radius_m,
    )
    node.diagnostics.run_local_corridor_check_distance_m = check_distance_m
    node.diagnostics.run_local_path_blocked_cell_count = len(blocked)
    return blocked

def prune_run_local_obstacles_after_progress(node, current_pose, remaining_waypoints):
    if node.run_local_map is None or not run_local_map_has_confirmed_obstacles(node.run_local_map):
        return None
    prune_distance_m = getattr(
        node.args,
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
    metadata = node.run_local_map.static_map.metadata
    for cell in node.run_local_map.confirmed_raw_cells:
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
        node.args.run_local_map_corridor_radius_m
        if node.args.run_local_map_corridor_radius_m is not None
        else 0.0
    )
    corridor_cells = lidar_obstacle_map.path_corridor_cells(
        node.run_local_map.static_map,
        pose,
        remaining_waypoints,
        node.args.run_local_map_corridor_check_distance_m,
        corridor_radius_m,
    )
    protected_cells = set()
    for cell in candidate_cells:
        inflated = lidar_obstacle_map.inflate_cells(
            node.run_local_map.static_map,
            {cell},
            node.run_local_map.config.inflation_radius_m,
        )
        if inflated.intersection(corridor_cells):
            protected_cells.add(cell)

    prune_cells = candidate_cells.difference(protected_cells)
    if not prune_cells:
        return None
    result = node.run_local_map.remove_raw_cells(prune_cells)
    if result.removed_raw_cells:
        node.diagnostics.run_local_pruned_raw_cells += result.removed_raw_cells
        node.diagnostics.run_local_pruned_inflated_cells += result.removed_inflated_cells
        node.diagnostics.run_local_cell_source_counts = node.run_local_map.cell_source_counts()
        node.get_logger().info(
            "Pruned passed run-local obstacle cells: "
            f"raw={result.removed_raw_cells}, "
            f"inflated={result.removed_inflated_cells}"
        )
        rviz_visualization.publish_rviz_obstacles_if_available(node)
    return result

def plan_with_existing_run_local_map(
    node,
    current_pose,
    old_remaining_waypoints,
    sequence=None,
    count_replan=True,
):
    if node.run_local_map is None:
        raise RuntimeError("lidar_replan_failed:no_run_local_map")
    if not run_local_map_has_confirmed_obstacles(node.run_local_map):
        raise RuntimeError("lidar_replan_failed:no_confirmed_run_local_obstacles")
    sequence = sequence or node.live_replan_attempt_count + 1
    goal_waypoint = old_remaining_waypoints[-1]
    result = replan_runtime.plan_existing_run_local_map(
        node.args,
        node.run_local_map,
        current_pose,
        goal_waypoint,
        old_remaining_waypoints,
        sequence=sequence,
    )
    update_replan_diagnostics(node, result, count_replan=count_replan)
    return validate_replan_result(
        node,
        result,
        current_pose,
        old_remaining_waypoints,
        goal_waypoint,
        require_changed=True,
    )

def sparse_retry_scan_args(node):
    return replan_runtime.args_with_obstacle_roi(
        node.args,
        forward_distance_m=getattr(
            node.args,
            "run_local_map_sparse_retry_forward_distance_m",
            DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_FORWARD_DISTANCE_M,
        ),
        forward_half_width_m=getattr(
            node.args,
            "run_local_map_sparse_retry_forward_half_width_m",
            DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_FORWARD_HALF_WIDTH_M,
        ),
        angle_window_deg=getattr(
            node.args,
            "run_local_map_sparse_retry_angle_window_deg",
            DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_ANGLE_WINDOW_DEG,
        ),
    )

def retry_sparse_lidar_replan(
    node,
    current_pose,
    goal_waypoint,
    old_remaining_waypoints,
    sequence,
):
    retry_limit = getattr(
        node.args,
        "run_local_map_sparse_retry_count",
        DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_COUNT,
    )
    last_result = None
    retry_args = sparse_retry_scan_args(node)
    retry_mode = (
        "expanded_forward:"
        f"distance={retry_args.obstacle_forward_distance_m:.3f},"
        f"half_width={retry_args.obstacle_forward_half_width_m:.3f},"
        f"angle={retry_args.obstacle_angle_window_deg:.1f}"
    )
    for retry_index in range(1, retry_limit + 1):
        node.stop_repeatedly()
        node.get_logger().warn(
            "LiDAR map update returned too few accepted scan points; "
            f"retrying with expanded forward ROI ({retry_index}/{retry_limit})."
        )
        result = replan_runtime.perform_lidar_replan(
            node,
            node.args,
            current_pose,
            goal_waypoint,
            old_remaining_waypoints,
            sequence=sequence,
            scan_args=retry_args,
        )
        node.diagnostics.run_local_sparse_retry_count = retry_index
        node.diagnostics.run_local_sparse_retry_mode = retry_mode
        last_result = result
        if result.success:
            node.get_logger().info(
                "Sparse LiDAR map update retry succeeded: "
                f"attempt={retry_index}"
            )
            return result
        if result.reason != lidar_obstacle_map.RUN_LOCAL_FAILURE_TOO_FEW_SCAN_POINTS:
            node.get_logger().warn(
                "Sparse LiDAR map update retry stopped on non-sparse failure: "
                f"reason={result.reason}"
            )
            return result
    node.get_logger().warn(
        "Sparse LiDAR map update retries exhausted; "
        f"reason={lidar_obstacle_map.RUN_LOCAL_FAILURE_TOO_FEW_SCAN_POINTS}"
    )
    return last_result

def replan_after_blockage(
    node,
    current_pose,
    old_remaining_waypoints,
    trigger=REPLAN_TRIGGER_SCAN_BLOCKAGE,
):
    known_corridor_repair_count = getattr(node, "known_corridor_repair_count", 0)
    sequence = node.live_replan_attempt_count + known_corridor_repair_count + 1
    goal_waypoint = old_remaining_waypoints[-1]
    if trigger == REPLAN_TRIGGER_KNOWN_CORRIDOR and node.run_local_map is not None:
        replanned = plan_with_existing_run_local_map(
            node,
            current_pose,
            old_remaining_waypoints,
            sequence=sequence,
        )
        node.known_corridor_repair_count = known_corridor_repair_count + 1
        node.get_logger().info(
            "Replanned with existing run-local map for known corridor blockage."
        )
        return replanned
    if node.live_replan_attempt_count >= node.args.max_replans:
        if (
            trigger == REPLAN_TRIGGER_SCAN_BLOCKAGE
            and run_local_map_has_confirmed_obstacles(node.run_local_map)
        ):
            replanned = plan_with_existing_run_local_map(
                node,
                current_pose,
                old_remaining_waypoints,
                sequence=sequence,
            )
            remember_scan_block_budget_repair(
                node,
                current_pose,
                replanned,
            )
            node.known_corridor_repair_count = known_corridor_repair_count + 1
            node.get_logger().warn(
                "LiDAR replan budget exhausted; repaired route with existing "
                "run-local map after scan blockage."
            )
            return replanned
        raise RuntimeError("lidar_replan_failed:max_replans_exceeded")
    if node.args.run_local_map_update_mode == "none":
        replanned = plan_with_existing_run_local_map(
            node,
            current_pose,
            old_remaining_waypoints,
            sequence=sequence,
        )
        node.live_replan_attempt_count += 1
        node.get_logger().info("Replanned with existing run-local map.")
        return replanned
    try:
        result = replan_runtime.perform_lidar_replan(
            node,
            node.args,
            current_pose,
            goal_waypoint,
            old_remaining_waypoints,
            sequence=sequence,
        )
    except RuntimeError as exc:
        if trigger == REPLAN_TRIGGER_SCAN_BLOCKAGE:
            raise lidar_replan_failure(exc) from exc
        if node.run_local_map is None:
            raise
        replanned = plan_with_existing_run_local_map(
            node,
            current_pose,
            old_remaining_waypoints,
            sequence=sequence,
        )
        node.live_replan_attempt_count += 1
        node.get_logger().warn(
            "LiDAR map update failed; replanned with existing run-local map."
        )
        return replanned
    if (
        trigger == REPLAN_TRIGGER_SCAN_BLOCKAGE
        and not result.success
        and result.reason == lidar_obstacle_map.RUN_LOCAL_FAILURE_TOO_FEW_SCAN_POINTS
        and getattr(
            node.args,
            "run_local_map_sparse_retry_count",
            DEFAULT_RUN_LOCAL_MAP_SPARSE_RETRY_COUNT,
        ) > 0
    ):
        result = retry_sparse_lidar_replan(
            node,
            current_pose,
            goal_waypoint,
            old_remaining_waypoints,
            sequence,
        )
    update_replan_diagnostics(node, result)
    if not result.success and node.run_local_map is not None:
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
            node.get_logger().warn(
                "LiDAR map update rejected; "
                f"replanning with existing run-local map. reason={result.reason}"
            )
            replanned = plan_with_existing_run_local_map(
                node,
                current_pose,
                old_remaining_waypoints,
                sequence=sequence,
            )
            node.live_replan_attempt_count += 1
            return replanned
    replanned = validate_replan_result(
        node,
        result,
        current_pose,
        old_remaining_waypoints,
        goal_waypoint,
    )
    node.live_replan_attempt_count += 1
    node.get_logger().info(
        "LiDAR obstacle replan completed: "
        f"waypoints={len(replanned)}, map={result.updated_map_yaml}"
    )
    return replanned
