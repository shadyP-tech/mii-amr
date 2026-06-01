#!/usr/bin/env python3
"""
Spin-only active arena localization helper.

This module owns the experimental live spin, scan/odom pairing, safety checks,
diagnostics, and call into the offline arena geometry localizer. It deliberately
does not publish /initialpose or interact with Nav2.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from arena_geometry_localizer import (
    ArenaGeometryConfig,
    Pose2D,
    ScanSample,
    analyze_scan_samples,
)
from arena_active_explore import (
    ActiveExploreConfig,
    ActiveExplorePlan,
    CELL_FREE,
    CELL_INFLATED,
    CELL_OCCUPIED,
    CELL_UNKNOWN,
    build_local_grid_from_scan_samples,
    geometry_is_recoverable,
    in_bounds,
    plan_active_explore_recovery,
    world_to_cell,
)


DEFAULT_STOP_COUNT = 10
DEFAULT_STOP_HZ = 10.0
ACTIVE_EXPLORE_FRONTIER_CLUSTER_MATCH_M = 0.35
ACTIVE_EXPLORE_FRONTIER_TARGET_MATCH_M = 0.45
ACTIVE_EXPLORE_FRONTIER_REACHED_PATH_M = 0.15
ACTIVE_EXPLORE_SHADOW_EMPTY_REPLANS_TO_COMPLETE = 2
ACTIVE_EXPLORE_PHASE_SHADOW = "shadow_explore"
ACTIVE_EXPLORE_PHASE_LOCALIZATION_POSE = "localization_pose"
ACTIVE_EXPLORE_PHASE_LOCALIZATION_SPIN = "localization_spin"
ACTIVE_EXPLORE_LOCALIZATION_CANDIDATE_KINDS = (
    "suspected_heater_approach",
    "provisional_center",
    "lateral_recenter",
)
ACTIVE_EXPLORE_FRONTIER_UNREACHABLE_REASONS = {
    "no_connected_path",
    "path_too_long",
}
LOCALIZER_FILTER_WALL_MARGIN_CELLS = 2
LOCALIZER_FILTER_WALL_EXPAND_CELLS = 1
LOCALIZER_FILTER_MIN_WALL_LENGTH_M = 0.45
LOCALIZER_FILTER_MIN_WALL_ASPECT_RATIO = 3.0
LOCALIZER_FILTER_MAX_WALL_THICKNESS_M = 0.20


class ActiveExploreMotionError(RuntimeError):
    def __init__(self, reason, record):
        super().__init__(reason)
        self.reason = reason
        self.record = record


@dataclass(frozen=True)
class PosePrior:
    x_m: float
    y_m: float
    yaw_rad: float
    covariance: list[float]


@dataclass
class ArenaActiveSpinResult:
    success: bool
    failure_reason: str | None
    pose_prior: PosePrior | None
    diagnostics: dict
    diagnostics_path: str | None = None


@dataclass(frozen=True)
class SectorClearance:
    ok: bool
    reason: str
    front_min_m: float | None
    left_min_m: float | None
    right_min_m: float | None
    rear_min_m: float | None


@dataclass(frozen=True)
class CenterRepositionStep:
    kind: str
    reason: str
    planned_distance_m: float
    local_heading_rad: float | None
    odom_heading_rad: float
    dynamic_heading: bool = False
    dynamic_heading_source: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CenterRepositionAction:
    ok: bool
    reason: str
    nearest_axis_side: str | None = None
    away_axis_side: str | None = None
    suspected_heater_axis_side: str | None = None
    nearest_short_wall_range_m: float | None = None
    far_short_wall_range_m: float | None = None
    suspected_heater_range_m: float | None = None
    target_nearest_short_wall_range_m: float | None = None
    heater_approach_target_range_m: float | None = None
    planned_distance_m: float | None = None
    local_heading_rad: float | None = None
    odom_heading_rad: float | None = None
    range_sum_error_m: float | None = None
    heater_scores: dict[str, float] | None = None
    selected_heater_score: float | None = None
    opposite_heater_score: float | None = None
    heater_profile_delta: float | None = None
    lateral_offset_m: float | None = None
    lateral_target_offset_m: float | None = None
    lateral_planned_distance_m: float | None = None
    lateral_step_skipped: bool = True
    lateral_skip_reason: str | None = None
    steps: tuple[CenterRepositionStep, ...] = ()

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ArenaActiveSpinConfig:
    run_id: str
    diagnostics_path: Path
    cmd_vel_topic: str = "/cmd_vel"
    scan_topic: str = "/scan"
    odom_topic: str = "/odom"
    spin_direction: str = "ccw"
    angular_speed_rad_s: float = 0.25
    max_spin_sec: float = 30.0
    spin_complete_tolerance_deg: float = 5.0
    min_angular_progress_rad_s: float = 0.05
    progress_check_sec: float = 2.0
    min_scan_samples: int = 20
    max_odom_scan_age_sec: float = 0.20
    stop_settle_sec: float = 0.5
    min_front_clearance_m: float = 0.35
    min_side_clearance_m: float = 0.20
    min_rear_clearance_m: float = 0.20
    require_operator_confirmation: bool = True
    allow_extra_cmd_vel_publishers: bool = False
    on_failure: str = "abort"
    dry_run: bool = False
    range_stride: int = 6
    max_points: int = 3000
    control_rate_hz: float = 10.0
    recovery_mode: str = "none"
    recovery_executor: str = "dry_run"
    enable_center_reposition: bool = False
    center_reposition_max_attempts: int = 1
    center_reposition_target_nearest_short_wall_range_m: float = 1.65
    center_reposition_min_step_m: float = 0.25
    center_reposition_max_step_m: float = 1.10
    center_reposition_linear_speed_mps: float = 0.08
    center_reposition_angular_speed_rad_s: float = 0.25
    center_reposition_heading_tolerance_deg: float = 8.0
    center_reposition_min_front_clearance_m: float = 0.45
    center_reposition_lateral_offset_threshold_m: float = 0.25
    center_reposition_lateral_target_offset_m: float = 0.10
    center_reposition_lateral_min_step_m: float = 0.15
    center_reposition_lateral_max_step_m: float = 0.55
    center_reposition_enable_heater_approach: bool = True
    center_reposition_heater_approach_max_attempts: int = 1
    center_reposition_heater_approach_target_range_m: float = 1.05
    center_reposition_heater_approach_min_selected_score: float = 0.50
    center_reposition_heater_approach_max_opposite_score: float = 0.30
    center_reposition_heater_approach_min_delta: float = 0.35
    center_reposition_heater_approach_min_step_m: float = 0.25
    center_reposition_heater_approach_max_step_m: float = 1.10
    active_explore_max_attempts: int = 2
    active_explore_max_single_move_m: float = 0.45
    active_explore_max_total_distance_m: float = 0.90
    active_explore_max_candidate_path_m: float | None = None
    active_explore_grid_resolution_m: float = 0.05
    active_explore_grid_size_m: float = 4.0
    active_explore_inflation_radius_m: float = 0.28
    active_explore_soft_clearance_radius_m: float = 0.35
    active_explore_soft_clearance_weight: float = 3.0
    active_explore_unknown_blocked: bool = True
    active_explore_max_path_segments: int = 3
    active_explore_use_accumulated_map: bool = True
    active_explore_map_max_samples: int = 240
    active_explore_temporary_map_publish_period_sec: float = 1.0
    active_explore_curve_lookahead_m: float = 0.18
    active_explore_curve_goal_tolerance_m: float = 0.05
    active_explore_curve_linear_speed_mps: float = 0.06
    active_explore_curve_max_angular_rad_s: float = 0.45
    active_explore_min_progress_before_spin_m: float = 0.05
    arena_config: ArenaGeometryConfig = field(default_factory=ArenaGeometryConfig)


def normalize_angle_rad(angle_rad):
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def shortest_angle_delta_rad(start_rad, end_rad):
    return normalize_angle_rad(end_rad - start_rad)


def clamp(value, low, high):
    return max(low, min(high, value))


def distance_2d(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def finite_point_2d(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x = value[0]
    y = value[1]
    try:
        x = float(x)
        y = float(y)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return [x, y]


def candidate_visible_shadow_count(candidate):
    if candidate is None:
        return 0
    metadata = candidate.metadata or {}
    value = metadata.get("visible_cluster_shadow_count")
    if value is None:
        value = candidate.score_components.get("visible_shadow_unknown_count")
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def candidate_cluster_centroid(candidate):
    return finite_point_2d((candidate.metadata or {}).get("cluster_centroid_world"))


def candidate_path_needs_motion(candidate):
    path_length = None if candidate is None else candidate.path_length_m
    return path_length is None or path_length > ACTIVE_EXPLORE_FRONTIER_REACHED_PATH_M


def candidate_is_accepted_shadow_frontier(candidate):
    return (
        candidate is not None
        and candidate.accepted
        and candidate.kind == "obstacle_shadow_frontier"
    )


def candidate_is_moving_shadow_frontier(candidate):
    return (
        candidate_is_accepted_shadow_frontier(candidate)
        and candidate_visible_shadow_count(candidate) > 0
        and candidate_path_needs_motion(candidate)
    )


def candidate_is_localization_pose_candidate(candidate):
    return (
        candidate is not None
        and candidate.accepted
        and candidate.kind in ACTIVE_EXPLORE_LOCALIZATION_CANDIDATE_KINDS
    )


def candidate_is_accepted_open_corridor(candidate):
    return (
        candidate is not None
        and candidate.accepted
        and candidate.kind == "open_corridor"
    )


def active_explore_curve_execution_record(
    candidate,
    path_points,
    curve_samples,
    driven_distance_m,
    duration_sec,
    stop_reason,
    **extra,
):
    record = {
        "executor": "cmd_vel_curve",
        "executed": True,
        "candidate_kind": candidate.kind,
        "candidate_score": candidate.score,
        "path_length_m": candidate.path_length_m,
        "curve_path_world": [[float(x), float(y)] for x, y in path_points],
        "curve_samples": list(curve_samples),
        "driven_distance_m": float(driven_distance_m),
        "duration_sec": float(duration_sec),
        "stop_reason": stop_reason,
    }
    record.update(extra)
    return record


def truncate_polyline_by_distance(points, max_distance_m):
    if len(points) < 2:
        raise RuntimeError("active_explore_curve_path_too_short")
    if max_distance_m <= 0.0:
        raise RuntimeError("active_explore_distance_limit_exhausted")
    truncated = [points[0]]
    remaining = float(max_distance_m)
    previous = points[0]
    for point in points[1:]:
        segment = distance_2d(previous, point)
        if segment <= 1e-9:
            previous = point
            continue
        if segment <= remaining + 1e-9:
            truncated.append(point)
            remaining -= segment
            previous = point
            if remaining <= 1e-9:
                break
            continue
        ratio = remaining / segment
        truncated.append(
            (
                previous[0] + ratio * (point[0] - previous[0]),
                previous[1] + ratio * (point[1] - previous[1]),
            )
        )
        break
    if len(truncated) < 2:
        raise RuntimeError("active_explore_curve_path_too_short")
    return tuple(truncated)


def active_explore_curve_path(candidate, current_pose, max_distance_m):
    source = list(candidate.path_world)
    if len(source) < 2:
        source = list(candidate.simplified_path_world)
    if len(source) < 2:
        raise RuntimeError("active_explore_curve_path_too_short")
    start = (float(current_pose.x), float(current_pose.y))
    if distance_2d(start, source[0]) <= 0.10:
        points = [start, *source[1:]]
    else:
        points = [start, *source]
    return truncate_polyline_by_distance(points, max_distance_m)


def select_curve_lookahead_target(path_points, current_point, lookahead_m):
    if not path_points:
        raise RuntimeError("active_explore_curve_path_too_short")
    nearest_index = min(
        range(len(path_points)),
        key=lambda index: distance_2d(current_point, path_points[index]),
    )
    for point in path_points[nearest_index + 1 :]:
        if distance_2d(current_point, point) >= lookahead_m:
            return point
    return path_points[-1]


def pure_pursuit_curve_command(
    current_pose,
    target_point,
    lookahead_m,
    linear_speed_mps,
    max_angular_rad_s,
):
    dx = float(target_point[0]) - float(current_pose.x)
    dy = float(target_point[1]) - float(current_pose.y)
    target_heading = math.atan2(dy, dx)
    yaw = math.radians(float(current_pose.yaw_deg))
    alpha = normalize_angle_rad(target_heading - yaw)
    linear_scale = clamp(math.cos(abs(alpha)), 0.35, 1.0)
    linear_x = abs(linear_speed_mps) * linear_scale
    angular_z = clamp(
        2.0 * linear_x * math.sin(alpha) / max(0.01, lookahead_m),
        -abs(max_angular_rad_s),
        abs(max_angular_rad_s),
    )
    return linear_x, angular_z, alpha


def temporary_map_cell_to_occupancy(value):
    if value == CELL_UNKNOWN:
        return -1
    if value == CELL_FREE:
        return 0
    if value == CELL_INFLATED:
        return 70
    if value == CELL_OCCUPIED:
        return 100
    return -1


def temporary_map_occupancy_data(grid):
    return [
        temporary_map_cell_to_occupancy(value)
        for row in grid.cells
        for value in row
    ]


def valid_scan_range_count(samples):
    count = 0
    for sample in samples:
        for value in sample.ranges:
            if value is None or not math.isfinite(value):
                continue
            if value < sample.range_min or value > sample.range_max:
                continue
            count += 1
    return count


def scan_endpoint_world(sample, index, raw_range):
    pose = sample.odom_pose
    if pose is None:
        return None
    angle = float(sample.angle_min) + index * float(sample.angle_increment)
    local_x = float(raw_range) * math.cos(angle)
    local_y = float(raw_range) * math.sin(angle)
    yaw = math.radians(float(pose.yaw_deg))
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        float(pose.x) + cos_yaw * local_x - sin_yaw * local_y,
        float(pose.y) + sin_yaw * local_x + cos_yaw * local_y,
    )


def neighbors_8_cells(cell):
    x, y = cell
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            yield (x + dx, y + dy)


def cluster_cells_8(cells):
    remaining = set(cells)
    clusters = []
    while remaining:
        start = remaining.pop()
        cluster = {start}
        stack = [start]
        while stack:
            cell = stack.pop()
            for neighbor in neighbors_8_cells(cell):
                if neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                cluster.add(neighbor)
                stack.append(neighbor)
        clusters.append(frozenset(cluster))
    return tuple(clusters)


def occupied_cells_from_grid(grid):
    cells = []
    for y, row in enumerate(grid.cells):
        for x, value in enumerate(row):
            if value == CELL_OCCUPIED:
                cells.append((x, y))
    return tuple(cells)


def blocked_cells_from_grid(grid):
    cells = []
    for y, row in enumerate(grid.cells):
        for x, value in enumerate(row):
            if value in {CELL_OCCUPIED, CELL_INFLATED}:
                cells.append((x, y))
    return tuple(cells)


def cell_bounds(cells):
    xs = [cell[0] for cell in cells]
    ys = [cell[1] for cell in cells]
    return min(xs), max(xs), min(ys), max(ys)


def cluster_is_wall_like(grid, cluster, occupied_envelope):
    min_x, max_x, min_y, max_y = cell_bounds(cluster)
    env_min_x, env_max_x, env_min_y, env_max_y = occupied_envelope
    margin = LOCALIZER_FILTER_WALL_MARGIN_CELLS
    near_outer_envelope = (
        min_x <= env_min_x + margin
        or max_x >= env_max_x - margin
        or min_y <= env_min_y + margin
        or max_y >= env_max_y - margin
    )
    near_grid_boundary = (
        min_x <= margin
        or min_y <= margin
        or max_x >= grid.width - 1 - margin
        or max_y >= grid.height - 1 - margin
    )
    if not near_outer_envelope and not near_grid_boundary:
        return False

    span_x = max_x - min_x + 1
    span_y = max_y - min_y + 1
    long_cells = max(span_x, span_y)
    short_cells = max(1, min(span_x, span_y))
    long_m = long_cells * grid.resolution_m
    short_m = short_cells * grid.resolution_m
    aspect = long_cells / short_cells
    return (
        long_m >= LOCALIZER_FILTER_MIN_WALL_LENGTH_M
        and short_m <= LOCALIZER_FILTER_MAX_WALL_THICKNESS_M
        and aspect >= LOCALIZER_FILTER_MIN_WALL_ASPECT_RATIO
    )


def expand_cells(grid, cells, radius_cells):
    expanded = set()
    for cell_x, cell_y in cells:
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                cell = (cell_x + dx, cell_y + dy)
                if in_bounds(grid, cell):
                    expanded.add(cell)
    return expanded


def temporary_grid_localizer_obstacle_mask(grid):
    occupied = occupied_cells_from_grid(grid)
    if not occupied:
        return set(), set(), {
            "occupied_cluster_count": 0,
            "protected_wall_cluster_count": 0,
        }

    occupied_envelope = cell_bounds(occupied)
    protected_wall_cells = set()
    protected_wall_cluster_count = 0
    clusters = cluster_cells_8(occupied)
    for cluster in clusters:
        if not cluster_is_wall_like(grid, cluster, occupied_envelope):
            continue
        protected_wall_cluster_count += 1
        protected_wall_cells.update(cluster)

    protected_wall_cells = expand_cells(
        grid,
        protected_wall_cells,
        LOCALIZER_FILTER_WALL_EXPAND_CELLS,
    )
    obstacle_mask = set(blocked_cells_from_grid(grid)) - protected_wall_cells
    diagnostics = {
        "occupied_cluster_count": len(clusters),
        "protected_wall_cluster_count": protected_wall_cluster_count,
    }
    return obstacle_mask, protected_wall_cells, diagnostics


def filter_scan_samples_with_temporary_obstacle_map(samples, grid, obstacle_mask):
    filtered = []
    filtered_range_count = 0
    for sample in samples:
        ranges = list(sample.ranges)
        for index, raw_range in enumerate(ranges):
            if raw_range is None or not math.isfinite(raw_range):
                continue
            if raw_range < sample.range_min or raw_range > sample.range_max:
                continue
            endpoint = scan_endpoint_world(sample, index, raw_range)
            if endpoint is None:
                continue
            cell = world_to_cell(grid, endpoint[0], endpoint[1])
            if not in_bounds(grid, cell):
                continue
            if cell not in obstacle_mask:
                continue
            ranges[index] = float("inf")
            filtered_range_count += 1
        filtered.append(
            ScanSample(
                ranges=ranges,
                angle_min=sample.angle_min,
                angle_increment=sample.angle_increment,
                range_min=sample.range_min,
                range_max=sample.range_max,
                odom_pose=sample.odom_pose,
            )
        )
    return filtered, filtered_range_count


def opposite_axis_side(axis_side):
    if axis_side == "axis_negative":
        return "axis_positive"
    if axis_side == "axis_positive":
        return "axis_negative"
    return None


def spin_diagnostics_template():
    return {
        "target_rad": 2.0 * math.pi,
        "accumulated_rad": 0.0,
        "duration_sec": 0.0,
        "timeout": False,
    }


def candidate_range(candidate):
    if candidate is None:
        return None
    value = getattr(candidate, "short_wall_candidate_range_m", None)
    if value is None or not math.isfinite(value) or value <= 0.0:
        return None
    return float(value)


def candidate_heater_score(candidate):
    if candidate is None:
        return None
    value = getattr(candidate, "heater_profile_score", None)
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def candidate_profile_valid(candidate):
    if candidate is None:
        return False
    reason = getattr(candidate, "validity_failed_reason", None)
    if reason is not None:
        return False
    features = getattr(candidate, "profile_features", None) or {}
    return features.get("validity_failed_reason") is None


def short_wall_ranges_and_error(result, config: ArenaActiveSpinConfig):
    candidates = result.short_wall_candidates or {}
    negative = candidates.get("axis_negative")
    positive = candidates.get("axis_positive")
    negative_range = candidate_range(negative)
    positive_range = candidate_range(positive)
    if negative_range is None or positive_range is None:
        return None, None, None, None, None, None
    range_sum_error = negative_range + positive_range - config.arena_config.arena_length_m
    return candidates, negative, positive, negative_range, positive_range, range_sum_error


def choose_center_reposition_action(result, config: ArenaActiveSpinConfig, origin_yaw_rad=0.0):
    if effective_recovery_mode(config) != "legacy":
        return CenterRepositionAction(False, "center_reposition_disabled")
    if result.success or result.failure_reason != "pose_not_unique":
        return CenterRepositionAction(False, "center_reposition_not_pose_not_unique")
    long_fit = result.long_wall_fit
    if not getattr(long_fit, "ok", False) or long_fit.axis_angle_rad is None:
        return CenterRepositionAction(False, "center_reposition_invalid_long_wall_fit")

    (
        _candidates,
        negative,
        positive,
        negative_range,
        positive_range,
        range_sum_error,
    ) = short_wall_ranges_and_error(result, config)
    if negative_range is None or positive_range is None:
        return CenterRepositionAction(False, "center_reposition_missing_short_wall_ranges")

    if abs(range_sum_error) > config.arena_config.max_short_wall_range_sum_error_m:
        return CenterRepositionAction(
            False,
            "center_reposition_range_sum_invalid",
            range_sum_error_m=range_sum_error,
        )

    if negative_range <= positive_range:
        nearest_side = "axis_negative"
        nearest_range = negative_range
        far_range = positive_range
    else:
        nearest_side = "axis_positive"
        nearest_range = positive_range
        far_range = negative_range
    away_side = opposite_axis_side(nearest_side)

    heater_scores = {
        "axis_negative": getattr(negative, "heater_profile_score", None),
        "axis_positive": getattr(positive, "heater_profile_score", None),
    }
    steps = []

    raw_step = config.center_reposition_target_nearest_short_wall_range_m - nearest_range
    planned_distance = None
    local_heading = None
    odom_heading = None
    if raw_step >= config.center_reposition_min_step_m:
        planned_distance = clamp(
            raw_step,
            config.center_reposition_min_step_m,
            config.center_reposition_max_step_m,
        )
        local_heading = long_fit.axis_angle_rad
        if away_side == "axis_negative":
            local_heading += math.pi
        odom_heading = normalize_angle_rad(origin_yaw_rad + local_heading)
        steps.append(
            CenterRepositionStep(
                kind="longitudinal",
                reason="center_reposition_away_from_nearest_short_wall",
                planned_distance_m=planned_distance,
                local_heading_rad=normalize_angle_rad(local_heading),
                odom_heading_rad=odom_heading,
            )
        )

    lateral_offset = getattr(long_fit, "lateral_offset_m", None)
    lateral_target = config.center_reposition_lateral_target_offset_m
    lateral_planned_distance = None
    lateral_step_skipped = True
    lateral_skip_reason = "center_reposition_lateral_offset_unavailable"
    if lateral_offset is not None and math.isfinite(lateral_offset):
        lateral_error = abs(float(lateral_offset))
        if lateral_error <= config.center_reposition_lateral_offset_threshold_m:
            lateral_skip_reason = "center_reposition_lateral_offset_within_threshold"
        else:
            normal_angle = getattr(long_fit, "normal_angle_rad", None)
            if normal_angle is None or not math.isfinite(normal_angle):
                return CenterRepositionAction(
                    False,
                    "center_reposition_missing_lateral_normal",
                    nearest_axis_side=nearest_side,
                    away_axis_side=away_side,
                    nearest_short_wall_range_m=nearest_range,
                    far_short_wall_range_m=far_range,
                    target_nearest_short_wall_range_m=(
                        config.center_reposition_target_nearest_short_wall_range_m
                    ),
                    planned_distance_m=planned_distance if planned_distance is not None else max(0.0, raw_step),
                    range_sum_error_m=range_sum_error,
                    heater_scores=heater_scores,
                    lateral_offset_m=float(lateral_offset),
                    lateral_target_offset_m=lateral_target,
                    lateral_planned_distance_m=None,
                    lateral_step_skipped=True,
                    lateral_skip_reason="center_reposition_missing_lateral_normal",
                    steps=tuple(steps),
                )
            lateral_raw_step = max(0.0, lateral_error - lateral_target)
            lateral_planned_distance = clamp(
                lateral_raw_step,
                config.center_reposition_lateral_min_step_m,
                config.center_reposition_lateral_max_step_m,
            )
            lateral_heading = normal_angle if lateral_offset < 0.0 else normal_angle + math.pi
            lateral_odom_heading = normalize_angle_rad(origin_yaw_rad + lateral_heading)
            steps.append(
                CenterRepositionStep(
                    kind="lateral",
                    reason="center_reposition_reduce_lateral_offset_dynamic",
                    planned_distance_m=lateral_planned_distance,
                    local_heading_rad=normalize_angle_rad(lateral_heading),
                    odom_heading_rad=lateral_odom_heading,
                    dynamic_heading=True,
                    dynamic_heading_source="live_side_clearance",
                )
            )
            lateral_step_skipped = False
            lateral_skip_reason = None

    if not steps:
        return CenterRepositionAction(
            False,
            "center_reposition_not_useful_already_near_target",
            nearest_axis_side=nearest_side,
            away_axis_side=away_side,
            nearest_short_wall_range_m=nearest_range,
            far_short_wall_range_m=far_range,
            target_nearest_short_wall_range_m=(
                config.center_reposition_target_nearest_short_wall_range_m
            ),
            planned_distance_m=max(0.0, raw_step),
            range_sum_error_m=range_sum_error,
            heater_scores=heater_scores,
            lateral_offset_m=lateral_offset,
            lateral_target_offset_m=lateral_target,
            lateral_planned_distance_m=lateral_planned_distance,
            lateral_step_skipped=lateral_step_skipped,
            lateral_skip_reason=lateral_skip_reason,
        )

    if planned_distance is None:
        first_step = steps[0]
        planned_distance = first_step.planned_distance_m
        local_heading = first_step.local_heading_rad
        odom_heading = first_step.odom_heading_rad

    return CenterRepositionAction(
        True,
        "center_reposition_toward_arena_center",
        nearest_axis_side=nearest_side,
        away_axis_side=away_side,
        nearest_short_wall_range_m=nearest_range,
        far_short_wall_range_m=far_range,
        target_nearest_short_wall_range_m=(
            config.center_reposition_target_nearest_short_wall_range_m
        ),
        planned_distance_m=planned_distance,
        local_heading_rad=None if local_heading is None else normalize_angle_rad(local_heading),
        odom_heading_rad=odom_heading,
        range_sum_error_m=range_sum_error,
        heater_scores=heater_scores,
        lateral_offset_m=lateral_offset,
        lateral_target_offset_m=lateral_target,
        lateral_planned_distance_m=lateral_planned_distance,
        lateral_step_skipped=lateral_step_skipped,
        lateral_skip_reason=lateral_skip_reason,
        steps=tuple(steps),
    )


def choose_heater_approach_reposition_action(
    result,
    config: ArenaActiveSpinConfig,
    origin_yaw_rad=0.0,
):
    if effective_recovery_mode(config) != "legacy":
        return CenterRepositionAction(False, "heater_approach_reposition_disabled")
    if not config.center_reposition_enable_heater_approach:
        return CenterRepositionAction(False, "heater_approach_reposition_disabled")
    if result.success or result.failure_reason != "pose_not_unique":
        return CenterRepositionAction(False, "heater_approach_not_pose_not_unique")
    long_fit = result.long_wall_fit
    if not getattr(long_fit, "ok", False) or long_fit.axis_angle_rad is None:
        return CenterRepositionAction(False, "heater_approach_invalid_long_wall_fit")

    (
        _candidates,
        negative,
        positive,
        negative_range,
        positive_range,
        range_sum_error,
    ) = short_wall_ranges_and_error(result, config)
    if negative_range is None or positive_range is None:
        return CenterRepositionAction(False, "heater_approach_missing_short_wall_ranges")
    if abs(range_sum_error) > config.arena_config.max_short_wall_range_sum_error_m:
        return CenterRepositionAction(
            False,
            "heater_approach_range_sum_invalid",
            range_sum_error_m=range_sum_error,
        )
    if not candidate_profile_valid(negative) or not candidate_profile_valid(positive):
        return CenterRepositionAction(
            False,
            "heater_approach_profile_invalid",
            range_sum_error_m=range_sum_error,
        )

    negative_score = candidate_heater_score(negative)
    positive_score = candidate_heater_score(positive)
    if negative_score is None or positive_score is None:
        return CenterRepositionAction(
            False,
            "heater_approach_missing_heater_scores",
            range_sum_error_m=range_sum_error,
        )

    if negative_score >= positive_score:
        selected_side = "axis_negative"
        selected_range = negative_range
        selected_score = negative_score
        opposite_score = positive_score
    else:
        selected_side = "axis_positive"
        selected_range = positive_range
        selected_score = positive_score
        opposite_score = negative_score
    delta = selected_score - opposite_score
    heater_scores = {
        "axis_negative": negative_score,
        "axis_positive": positive_score,
    }
    common = {
        "suspected_heater_axis_side": selected_side,
        "suspected_heater_range_m": selected_range,
        "heater_approach_target_range_m": (
            config.center_reposition_heater_approach_target_range_m
        ),
        "range_sum_error_m": range_sum_error,
        "heater_scores": heater_scores,
        "selected_heater_score": selected_score,
        "opposite_heater_score": opposite_score,
        "heater_profile_delta": delta,
    }
    if selected_score < config.center_reposition_heater_approach_min_selected_score:
        return CenterRepositionAction(
            False,
            "heater_approach_selected_score_too_low",
            **common,
        )
    if opposite_score > config.center_reposition_heater_approach_max_opposite_score:
        return CenterRepositionAction(
            False,
            "heater_approach_opposite_score_too_high",
            **common,
        )
    if delta < config.center_reposition_heater_approach_min_delta:
        return CenterRepositionAction(
            False,
            "heater_approach_delta_too_low",
            **common,
        )

    raw_step = selected_range - config.center_reposition_heater_approach_target_range_m
    if raw_step < config.center_reposition_heater_approach_min_step_m:
        return CenterRepositionAction(
            False,
            "heater_approach_not_useful_already_near_target",
            planned_distance_m=max(0.0, raw_step),
            **common,
        )

    planned_distance = clamp(
        raw_step,
        config.center_reposition_heater_approach_min_step_m,
        config.center_reposition_heater_approach_max_step_m,
    )
    local_heading = long_fit.axis_angle_rad
    if selected_side == "axis_negative":
        local_heading += math.pi
    odom_heading = normalize_angle_rad(origin_yaw_rad + local_heading)
    step = CenterRepositionStep(
        kind="heater_approach",
        reason="heater_approach_toward_suspected_heater",
        planned_distance_m=planned_distance,
        local_heading_rad=normalize_angle_rad(local_heading),
        odom_heading_rad=odom_heading,
    )
    return CenterRepositionAction(
        True,
        "heater_approach_toward_suspected_heater",
        planned_distance_m=planned_distance,
        local_heading_rad=step.local_heading_rad,
        odom_heading_rad=odom_heading,
        steps=(step,),
        **common,
    )


def yaw_from_quaternion(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def odom_pose_from_msg(msg):
    pose = msg.pose.pose
    orientation = pose.orientation
    return Pose2D(
        x=float(pose.position.x),
        y=float(pose.position.y),
        yaw_deg=math.degrees(
            yaw_from_quaternion(
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            )
        ),
    )


def scan_sample_from_msg(msg, odom_pose):
    return ScanSample(
        ranges=list(msg.ranges),
        angle_min=float(msg.angle_min),
        angle_increment=float(msg.angle_increment),
        range_min=float(msg.range_min),
        range_max=float(msg.range_max),
        odom_pose=odom_pose,
    )


def valid_range(value, range_min, range_max):
    return (
        value is not None
        and math.isfinite(value)
        and value >= range_min
        and value <= range_max
    )


def angle_in_sector(angle_deg, ranges):
    return any(lower <= angle_deg <= upper for lower, upper in ranges)


def min_sector_range(scan, sectors):
    values = []
    for index, raw_range in enumerate(scan.ranges):
        if not valid_range(raw_range, scan.range_min, scan.range_max):
            continue
        angle_rad = scan.angle_min + index * scan.angle_increment
        angle_deg = math.degrees(normalize_angle_rad(angle_rad))
        if angle_in_sector(angle_deg, sectors):
            values.append(float(raw_range))
    return min(values) if values else None


def evaluate_clearance(scan, config: ArenaActiveSpinConfig):
    front = min_sector_range(scan, [(-30.0, 30.0)])
    left = min_sector_range(scan, [(60.0, 120.0)])
    right = min_sector_range(scan, [(-120.0, -60.0)])
    rear = min_sector_range(scan, [(150.0, 180.0), (-180.0, -150.0)])
    checks = [
        ("front_clearance_missing", "front_clearance_below_limit", front, config.min_front_clearance_m),
        ("left_clearance_missing", "left_clearance_below_limit", left, config.min_side_clearance_m),
        ("right_clearance_missing", "right_clearance_below_limit", right, config.min_side_clearance_m),
        ("rear_clearance_missing", "rear_clearance_below_limit", rear, config.min_rear_clearance_m),
    ]
    for missing_reason, low_reason, value, limit in checks:
        if value is None:
            return SectorClearance(False, missing_reason, front, left, right, rear)
        if value < limit:
            return SectorClearance(False, low_reason, front, left, right, rear)
    return SectorClearance(True, "ok", front, left, right, rear)


def evaluate_reposition_clearance(scan, config: ArenaActiveSpinConfig):
    front = min_sector_range(scan, [(-30.0, 30.0)])
    left = min_sector_range(scan, [(60.0, 120.0)])
    right = min_sector_range(scan, [(-120.0, -60.0)])
    rear = min_sector_range(scan, [(150.0, 180.0), (-180.0, -150.0)])
    checks = [
        (
            "front_clearance_missing",
            "front_clearance_below_limit",
            front,
            config.center_reposition_min_front_clearance_m,
        ),
        ("left_clearance_missing", "left_clearance_below_limit", left, config.min_side_clearance_m),
        ("right_clearance_missing", "right_clearance_below_limit", right, config.min_side_clearance_m),
        ("rear_clearance_missing", "rear_clearance_below_limit", rear, config.min_rear_clearance_m),
    ]
    for missing_reason, low_reason, value, limit in checks:
        if value is None:
            return SectorClearance(False, missing_reason, front, left, right, rear)
        if value < limit:
            return SectorClearance(False, low_reason, front, left, right, rear)
    return SectorClearance(True, "ok", front, left, right, rear)


def min_valid_scan_range(scan):
    values = [
        float(value)
        for value in getattr(scan, "ranges", [])
        if value is not None
        and math.isfinite(float(value))
        and float(value) >= float(scan.range_min)
        and float(value) <= float(scan.range_max)
    ]
    return min(values) if values else None


def dynamic_lateral_heading_from_scan(scan, current_yaw_rad):
    left = min_sector_range(scan, [(60.0, 120.0)])
    right = min_sector_range(scan, [(-120.0, -60.0)])
    if left is None or right is None:
        raise RuntimeError("center_reposition_dynamic_lateral_clearance_missing")
    if left >= right:
        return {
            "odom_heading_rad": normalize_angle_rad(current_yaw_rad + math.pi / 2.0),
            "direction": "left",
            "left_clearance_m": left,
            "right_clearance_m": right,
        }
    return {
        "odom_heading_rad": normalize_angle_rad(current_yaw_rad - math.pi / 2.0),
        "direction": "right",
        "left_clearance_m": left,
        "right_clearance_m": right,
    }


def covariance_list_from_localizer(covariance):
    values = [0.0] * 36
    values[0] = float(covariance["x_m2"])
    values[7] = float(covariance["y_m2"])
    values[35] = float(covariance["yaw_rad2"])
    return values


def pose_prior_from_localizer_result(result):
    pose = result.estimated_pose_prior
    covariance = result.estimated_covariance
    if pose is None or covariance is None:
        return None
    return PosePrior(
        x_m=float(pose.x),
        y_m=float(pose.y),
        yaw_rad=math.radians(float(pose.yaw_deg)),
        covariance=covariance_list_from_localizer(covariance),
    )


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    return value


def write_diagnostics_json(path: Path | str | None, diagnostics):
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(json_safe(diagnostics), file, indent=2, sort_keys=True)
        file.write("\n")
    return str(path)


def config_diagnostics(config: ArenaActiveSpinConfig):
    data = asdict(config)
    data["diagnostics_path"] = str(config.diagnostics_path)
    data["arena_config"] = asdict(config.arena_config)
    data["effective_recovery_mode"] = effective_recovery_mode(config)
    return data


def effective_recovery_mode(config: ArenaActiveSpinConfig):
    if config.recovery_mode != "none":
        return config.recovery_mode
    if config.enable_center_reposition:
        return "legacy"
    return "none"


def active_explore_config_from_arena_config(config: ArenaActiveSpinConfig):
    return ActiveExploreConfig(
        max_attempts=config.active_explore_max_attempts,
        max_single_move_m=config.active_explore_max_single_move_m,
        max_total_distance_m=config.active_explore_max_total_distance_m,
        max_candidate_path_m=config.active_explore_max_candidate_path_m,
        grid_resolution_m=config.active_explore_grid_resolution_m,
        grid_size_m=config.active_explore_grid_size_m,
        inflation_radius_m=config.active_explore_inflation_radius_m,
        soft_clearance_radius_m=config.active_explore_soft_clearance_radius_m,
        soft_clearance_weight=config.active_explore_soft_clearance_weight,
        unknown_blocked=config.active_explore_unknown_blocked,
        max_path_segments=config.active_explore_max_path_segments,
        target_nearest_short_wall_range_m=(
            config.center_reposition_target_nearest_short_wall_range_m
        ),
        center_min_step_m=config.center_reposition_min_step_m,
        lateral_offset_threshold_m=config.center_reposition_lateral_offset_threshold_m,
        lateral_target_offset_m=config.center_reposition_lateral_target_offset_m,
        heater_approach_target_range_m=(
            config.center_reposition_heater_approach_target_range_m
        ),
        heater_approach_min_selected_score=(
            config.center_reposition_heater_approach_min_selected_score
        ),
        heater_approach_max_opposite_score=(
            config.center_reposition_heater_approach_max_opposite_score
        ),
        heater_approach_min_delta=config.center_reposition_heater_approach_min_delta,
        arena_length_m=config.arena_config.arena_length_m,
        max_short_wall_range_sum_error_m=(
            config.arena_config.max_short_wall_range_sum_error_m
        ),
    )


def initial_diagnostics(config: ArenaActiveSpinConfig):
    recovery_mode = effective_recovery_mode(config)
    return {
        "mode": "arena-active",
        "success": False,
        "failure_reason": "",
        "fallback_used": False,
        "config": config_diagnostics(config),
        "spin": spin_diagnostics_template(),
        "spin_attempts": [],
        "reposition": {
            "enabled": recovery_mode == "legacy",
            "attempts": [],
        },
        "active_explore": {
            "enabled": recovery_mode == "active_explore",
            "mode": recovery_mode,
            "executor": config.recovery_executor,
            "active_explore_phase": ACTIVE_EXPLORE_PHASE_SHADOW,
            "use_accumulated_map": config.active_explore_use_accumulated_map,
            "map_max_samples": config.active_explore_map_max_samples,
            "temporary_map": {
                "frame": "odom",
                "scan_samples_stored": 0,
                "grid": None,
            },
            "attempts": [],
            "total_distance_m": 0.0,
            "persistent_frontier_goal": None,
            "shadow_frontier_empty_replans": 0,
            "shadow_explore_complete": False,
            "shadow_frontier_status": None,
            "localization_candidate_policy": None,
            "localizer_filter": {
                "enabled": False,
                "reason": "not_run",
            },
        },
        "samples": {
            "scan_samples_collected": 0,
            "scan_samples_used": 0,
            "rejected_scan_samples": 0,
        },
        "safety": {
            "min_front_range_m": None,
            "min_left_range_m": None,
            "min_right_range_m": None,
            "min_rear_range_m": None,
        },
        "cmd_vel_publishers": {
            "count": None,
            "unexpected_count": None,
            "allowed": config.allow_extra_cmd_vel_publishers,
        },
        "localizer_result": None,
        "exception": None,
        "initialpose": {
            "published": False,
            "reason": "not_reached",
        },
    }


def update_safety_minima(diagnostics, clearance: SectorClearance):
    safety = diagnostics["safety"]
    for key, value in [
        ("min_front_range_m", clearance.front_min_m),
        ("min_left_range_m", clearance.left_min_m),
        ("min_right_range_m", clearance.right_min_m),
        ("min_rear_range_m", clearance.rear_min_m),
    ]:
        if value is None:
            continue
        current = safety.get(key)
        safety[key] = value if current is None else min(current, value)


def stop_repeatedly(
    publisher,
    twist_factory: Callable[[], object],
    sleep_fn: Callable[[float], None] = time.sleep,
    count=DEFAULT_STOP_COUNT,
    hz=DEFAULT_STOP_HZ,
):
    delay = 1.0 / hz
    for _ in range(count):
        publisher.publish(twist_factory())
        sleep_fn(delay)


class ArenaActiveSpinSession:
    def __init__(
        self,
        node,
        config: ArenaActiveSpinConfig,
        rclpy_module,
        twist_factory,
        scan_msg_type,
        odom_msg_type,
        qos_profile,
        input_fn=input,
        time_fn=time.time,
        sleep_fn=time.sleep,
        analyze_fn=analyze_scan_samples,
        temporary_map_callback=None,
        active_explore_plan_callback=None,
    ):
        self.node = node
        self.config = config
        self.rclpy = rclpy_module
        self.twist_factory = twist_factory
        self.input_fn = input_fn
        self.time_fn = time_fn
        self.sleep_fn = sleep_fn
        self.analyze_fn = analyze_fn
        self.temporary_map_callback = temporary_map_callback
        self.active_explore_plan_callback = active_explore_plan_callback
        self.last_temporary_map_publish_sec = None
        self.latest_scan = None
        self.latest_scan_received_sec = None
        self.latest_odom_pose = None
        self.latest_odom_yaw_rad = None
        self.latest_odom_received_sec = None
        self.collecting = False
        self.collecting_explore_map = False
        self.samples = []
        self.explore_samples = []
        self.rejected_samples = 0
        self.active_explore_frontier_goal = None
        self.active_explore_phase = ACTIVE_EXPLORE_PHASE_SHADOW
        self.shadow_frontier_empty_replans = 0
        self.shadow_explore_complete = False
        self.diagnostics = initial_diagnostics(config)
        self.scan_subscription = node.create_subscription(
            scan_msg_type,
            config.scan_topic,
            self.scan_callback,
            qos_profile,
        )
        self.odom_subscription = node.create_subscription(
            odom_msg_type,
            config.odom_topic,
            self.odom_callback,
            10,
        )

    def now(self):
        return self.time_fn()

    def scan_callback(self, msg):
        received_sec = self.now()
        self.latest_scan = msg
        self.latest_scan_received_sec = received_sec
        collecting_localizer = self.collecting
        collecting_explore = (
            effective_recovery_mode(self.config) == "active_explore"
            and self.config.active_explore_use_accumulated_map
            and (self.collecting or self.collecting_explore_map)
        )
        if not collecting_localizer and not collecting_explore:
            return
        if self.latest_odom_pose is None or self.latest_odom_received_sec is None:
            if collecting_localizer:
                self.rejected_samples += 1
            return
        if received_sec - self.latest_odom_received_sec > self.config.max_odom_scan_age_sec:
            if collecting_localizer:
                self.rejected_samples += 1
            return
        sample = scan_sample_from_msg(msg, self.latest_odom_pose)
        if collecting_localizer:
            self.samples.append(sample)
        if collecting_explore:
            self.append_explore_sample(sample)

    def append_explore_sample(self, sample):
        self.explore_samples.append(sample)
        max_samples = max(1, int(self.config.active_explore_map_max_samples))
        if len(self.explore_samples) > max_samples:
            del self.explore_samples[: len(self.explore_samples) - max_samples]
        self.diagnostics["active_explore"]["temporary_map"]["scan_samples_stored"] = (
            len(self.explore_samples)
        )
        self.publish_temporary_map_if_ready()

    def update_temporary_map_diagnostics(self, grid):
        self.diagnostics["active_explore"]["temporary_map"] = {
            "frame": "odom",
            "source": "accumulated_spin_and_recovery_scans",
            "scan_samples_stored": len(self.explore_samples),
            "grid": grid.to_dict(),
        }

    def publish_temporary_map_if_ready(self, force=False, grid=None):
        if self.temporary_map_callback is None:
            return
        if (
            effective_recovery_mode(self.config) != "active_explore"
            or not self.config.active_explore_use_accumulated_map
            or not self.explore_samples
            or self.latest_odom_pose is None
        ):
            return
        now = self.now()
        period_sec = self.config.active_explore_temporary_map_publish_period_sec
        if (
            not force
            and self.last_temporary_map_publish_sec is not None
            and now - self.last_temporary_map_publish_sec < period_sec
        ):
            return
        if grid is None:
            grid = build_local_grid_from_scan_samples(
                self.explore_samples,
                self.latest_odom_pose,
                active_explore_config_from_arena_config(self.config),
            )
        self.update_temporary_map_diagnostics(grid)
        self.last_temporary_map_publish_sec = now
        try:
            self.temporary_map_callback(grid)
        except Exception as exc:
            self.diagnostics["active_explore"]["temporary_map"]["publish_error"] = str(exc)

    def publish_active_explore_plan_if_ready(self, plan, move_limit_m):
        if self.active_explore_plan_callback is None or self.latest_odom_pose is None:
            return
        try:
            self.active_explore_plan_callback(
                plan,
                self.latest_odom_pose,
                move_limit_m,
            )
        except Exception as exc:
            self.diagnostics["active_explore"]["path_viz_publish_error"] = str(exc)

    def active_explore_spin_safety(self):
        self.wait_for_fresh_inputs()
        clearance = evaluate_clearance(self.latest_scan, self.config)
        update_safety_minima(self.diagnostics, clearance)
        full_min_range = min_valid_scan_range(self.latest_scan)
        required = self.config.min_front_clearance_m
        if full_min_range is None:
            ok = False
            reason = "spin_clearance_missing"
        elif full_min_range < required:
            ok = False
            reason = "spin_full_clearance_below_front_limit"
        else:
            ok = True
            reason = "ok"
        return {
            "ok": ok,
            "reason": reason,
            "full_min_range_m": full_min_range,
            "required_min_range_m": required,
            "sector_clearance": asdict(clearance),
        }

    def print_active_explore_spin_skip(self, spin_safety):
        print("\nArena-active post-motion spin skipped")
        print(f"  reason: {spin_safety['reason']}")
        print(f"  full min range: {spin_safety['full_min_range_m']}")
        print(f"  required min range: {spin_safety['required_min_range_m']}")
        print("  expected action: replan toward active-explore frontier without rotating")

    def print_active_explore_phase_spin_skip(self, reason):
        print("\nArena-active post-motion spin skipped")
        print(f"  reason: {reason}")
        print("  expected action: keep exploring obstacle shadow without rotating")

    def odom_callback(self, msg):
        self.latest_odom_pose = odom_pose_from_msg(msg)
        self.latest_odom_yaw_rad = math.radians(self.latest_odom_pose.yaw_deg)
        self.latest_odom_received_sec = self.now()

    def fresh_scan_age_sec(self):
        if self.latest_scan_received_sec is None:
            return None
        return self.now() - self.latest_scan_received_sec

    def fresh_odom_age_sec(self):
        if self.latest_odom_received_sec is None:
            return None
        return self.now() - self.latest_odom_received_sec

    def wait_for_fresh_inputs(self):
        deadline = self.now() + min(5.0, self.config.max_spin_sec)
        while self.rclpy.ok() and self.now() <= deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
            scan_age = self.fresh_scan_age_sec()
            odom_age = self.fresh_odom_age_sec()
            if (
                self.latest_scan is not None
                and scan_age is not None
                and scan_age <= self.config.max_odom_scan_age_sec
                and self.latest_odom_pose is not None
                and odom_age is not None
                and odom_age <= self.config.max_odom_scan_age_sec
            ):
                return
        raise RuntimeError("fresh_scan_or_odom_unavailable")

    def refresh_fresh_inputs_after_prompt(self):
        deadline = self.now() + min(2.0, self.config.max_spin_sec)
        while self.rclpy.ok() and self.now() <= deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
            scan_age = self.fresh_scan_age_sec()
            odom_age = self.fresh_odom_age_sec()
            if (
                self.latest_scan is not None
                and scan_age is not None
                and scan_age <= self.config.max_odom_scan_age_sec
                and self.latest_odom_pose is not None
                and odom_age is not None
                and odom_age <= self.config.max_odom_scan_age_sec
            ):
                return
        raise RuntimeError("fresh_scan_or_odom_unavailable_after_prompt")

    def cmd_vel_publisher_check(self):
        count = None
        if hasattr(self.node, "count_publishers"):
            count = self.node.count_publishers(self.config.cmd_vel_topic)
        unexpected = None if count is None else max(0, int(count) - 1)
        self.diagnostics["cmd_vel_publishers"] = {
            "count": count,
            "unexpected_count": unexpected,
            "allowed": self.config.allow_extra_cmd_vel_publishers,
        }
        if (
            unexpected is not None
            and unexpected > 0
            and not self.config.allow_extra_cmd_vel_publishers
        ):
            raise RuntimeError("unexpected_cmd_vel_publishers")

    def print_operator_prompt(self):
        scan_age = self.fresh_scan_age_sec()
        odom_age = self.fresh_odom_age_sec()
        clearance = evaluate_clearance(self.latest_scan, self.config)
        update_safety_minima(self.diagnostics, clearance)
        print("\nArena-active spin-only startup")
        print(f"  angular speed: {self.config.angular_speed_rad_s:.3f} rad/s")
        print(f"  direction: {self.config.spin_direction}")
        print(f"  max spin time: {self.config.max_spin_sec:.1f} s")
        print(f"  front clearance: {clearance.front_min_m}")
        print(f"  left clearance: {clearance.left_min_m}")
        print(f"  right clearance: {clearance.right_min_m}")
        print(f"  rear clearance: {clearance.rear_min_m}")
        print(f"  latest scan age: {scan_age}")
        print(f"  latest odom age: {odom_age}")
        print(f"  cmd_vel publisher check: {self.diagnostics['cmd_vel_publishers']}")
        print("  expected action: rotate in place 360 degrees")
        if not clearance.ok:
            raise RuntimeError(f"scan_clearance_failed:{clearance.reason}")
        if self.config.require_operator_confirmation:
            self.input_fn("Press Enter to start arena-active spin, or Ctrl+C to abort: ")

    def publish_spin_command(self, publisher):
        command = self.twist_factory()
        sign = 1.0 if self.config.spin_direction == "ccw" else -1.0
        command.angular.z = sign * abs(self.config.angular_speed_rad_s)
        publisher.publish(command)

    def run_spin(self, publisher):
        self.wait_for_fresh_inputs()
        self.cmd_vel_publisher_check()
        self.print_operator_prompt()
        self.refresh_fresh_inputs_after_prompt()

        previous_yaw = self.latest_odom_yaw_rad
        if previous_yaw is None:
            raise RuntimeError("fresh_odom_unavailable")
        self.collecting = True
        accumulated = 0.0
        target = 2.0 * math.pi - math.radians(self.config.spin_complete_tolerance_deg)
        period = 1.0 / self.config.control_rate_hz
        start = self.now()
        last_progress_time = start
        last_progress_yaw = 0.0

        while self.rclpy.ok():
            if self.now() - start > self.config.max_spin_sec:
                self.diagnostics["spin"]["timeout"] = True
                raise RuntimeError("arena_active_spin_timeout")
            self.publish_spin_command(publisher)
            self.rclpy.spin_once(self.node, timeout_sec=period)
            now = self.now()
            scan_age = self.fresh_scan_age_sec()
            odom_age = self.fresh_odom_age_sec()
            if scan_age is None or scan_age > self.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_scan_during_spin")
            if odom_age is None or odom_age > self.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_odom_during_spin")

            clearance = evaluate_clearance(self.latest_scan, self.config)
            update_safety_minima(self.diagnostics, clearance)
            if not clearance.ok:
                raise RuntimeError(f"scan_clearance_failed:{clearance.reason}")

            current_yaw = self.latest_odom_yaw_rad
            delta = shortest_angle_delta_rad(previous_yaw, current_yaw)
            accumulated += delta
            previous_yaw = current_yaw
            self.diagnostics["spin"]["accumulated_rad"] = accumulated
            self.diagnostics["spin"]["duration_sec"] = now - start
            if abs(accumulated) >= target:
                return accumulated, now - start

            if now - last_progress_time >= self.config.progress_check_sec:
                progress_rate = abs(accumulated - last_progress_yaw) / (now - last_progress_time)
                if progress_rate < self.config.min_angular_progress_rad_s:
                    raise RuntimeError("insufficient_angular_progress")
                last_progress_time = now
                last_progress_yaw = accumulated

        raise RuntimeError("ros_shutdown_during_arena_active_spin")

    def reset_spin_collection(self, attempt_index):
        self.collecting = False
        self.samples = []
        self.rejected_samples = 0
        self.diagnostics["spin"] = {
            **spin_diagnostics_template(),
            "attempt_index": attempt_index,
        }

    def run_spin_attempt(self, publisher, attempt_index):
        self.reset_spin_collection(attempt_index)
        self.run_spin(publisher)
        self.collecting = False
        stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
        self.sleep_fn(self.config.stop_settle_sec)
        self.diagnostics["spin_attempts"].append(
            {
                **self.diagnostics["spin"],
                "scan_samples_collected": len(self.samples),
                "scan_samples_used": len(self.samples),
                "rejected_scan_samples": self.rejected_samples,
            }
        )

    def active_explore_localizer_filter_reason_disabled(self):
        if effective_recovery_mode(self.config) != "active_explore":
            return "not_active_explore"
        if not self.config.active_explore_use_accumulated_map:
            return "accumulated_map_disabled"
        attempt_index = self.diagnostics.get("spin", {}).get("attempt_index")
        if attempt_index == 0:
            return "first_spin"
        if self.active_explore_phase != ACTIVE_EXPLORE_PHASE_LOCALIZATION_SPIN:
            return "not_final_active_explore_localization_spin"
        if not self.shadow_explore_complete:
            return "shadow_explore_not_complete"
        return None

    def active_explore_localizer_filter_grid(self):
        if not self.explore_samples:
            return None, "no_temporary_map_samples"
        if self.latest_odom_pose is None:
            return None, "missing_latest_odom_pose"
        grid = build_local_grid_from_scan_samples(
            self.explore_samples,
            self.latest_odom_pose,
            active_explore_config_from_arena_config(self.config),
        )
        self.update_temporary_map_diagnostics(grid)
        return grid, "ok"

    def active_explore_filtered_localizer_samples(self):
        diagnostics = {
            "enabled": False,
            "reason": "",
            "input_sample_count": len(self.samples),
            "output_sample_count": len(self.samples),
            "valid_ranges_before": valid_scan_range_count(self.samples),
            "valid_ranges_after": valid_scan_range_count(self.samples),
            "filtered_range_count": 0,
            "obstacle_mask_cell_count": 0,
            "protected_wall_cell_count": 0,
            "temporary_grid_cell_counts": None,
            "final_spin_attempt_index": self.diagnostics.get("spin", {}).get(
                "attempt_index"
            ),
        }
        disabled_reason = self.active_explore_localizer_filter_reason_disabled()
        if disabled_reason is not None:
            diagnostics["reason"] = disabled_reason
            self.diagnostics["active_explore"]["localizer_filter"] = diagnostics
            return self.samples

        grid, grid_reason = self.active_explore_localizer_filter_grid()
        if grid is None:
            diagnostics["reason"] = grid_reason
            self.diagnostics["active_explore"]["localizer_filter"] = diagnostics
            return self.samples

        diagnostics["temporary_grid_cell_counts"] = grid.to_dict()["cell_counts"]
        obstacle_mask, protected_wall_cells, mask_diagnostics = (
            temporary_grid_localizer_obstacle_mask(grid)
        )
        diagnostics.update(mask_diagnostics)
        diagnostics["obstacle_mask_cell_count"] = len(obstacle_mask)
        diagnostics["protected_wall_cell_count"] = len(protected_wall_cells)
        if not obstacle_mask:
            diagnostics["reason"] = "no_temporary_obstacle_mask"
            self.diagnostics["active_explore"]["localizer_filter"] = diagnostics
            return self.samples

        filtered_samples, filtered_range_count = (
            filter_scan_samples_with_temporary_obstacle_map(
                self.samples,
                grid,
                obstacle_mask,
            )
        )
        diagnostics["enabled"] = True
        diagnostics["reason"] = "filtered_temporary_obstacles"
        diagnostics["output_sample_count"] = len(filtered_samples)
        diagnostics["filtered_range_count"] = filtered_range_count
        diagnostics["valid_ranges_after"] = valid_scan_range_count(filtered_samples)
        self.diagnostics["active_explore"]["localizer_filter"] = diagnostics
        return filtered_samples

    def analyze_result(self):
        if len(self.samples) < self.config.min_scan_samples:
            raise RuntimeError(
                "insufficient_scan_samples:"
                f"{len(self.samples)}<{self.config.min_scan_samples}"
            )
        localizer_samples = self.active_explore_filtered_localizer_samples()
        result = self.analyze_fn(
            localizer_samples,
            self.config.arena_config,
            range_stride=self.config.range_stride,
            max_points=self.config.max_points,
        )
        self.diagnostics["localizer_result"] = result.to_dict()
        return result

    def pose_prior_from_result_or_raise(self, result):
        if not result.success:
            raise RuntimeError(f"arena_localizer_failed:{result.failure_reason}")
        pose_prior = pose_prior_from_localizer_result(result)
        if pose_prior is None:
            raise RuntimeError("arena_localizer_missing_pose_prior")
        return pose_prior

    def first_sample_origin_yaw_rad(self):
        for sample in self.samples:
            if sample.odom_pose is not None:
                return math.radians(sample.odom_pose.yaw_deg)
        return 0.0

    def publish_turn_command(self, publisher, target_yaw_rad):
        if self.latest_odom_yaw_rad is None:
            raise RuntimeError("fresh_odom_unavailable_during_reposition_turn")
        command = self.twist_factory()
        delta = shortest_angle_delta_rad(self.latest_odom_yaw_rad, target_yaw_rad)
        command.angular.z = (
            1.0 if delta >= 0.0 else -1.0
        ) * abs(self.config.center_reposition_angular_speed_rad_s)
        publisher.publish(command)

    def turn_to_heading(self, publisher, target_yaw_rad):
        tolerance = math.radians(self.config.center_reposition_heading_tolerance_deg)
        deadline = self.now() + max(
            8.0,
            math.pi / max(0.01, abs(self.config.center_reposition_angular_speed_rad_s))
            + 3.0,
        )
        period = 1.0 / self.config.control_rate_hz
        while self.rclpy.ok() and self.now() <= deadline:
            self.rclpy.spin_once(self.node, timeout_sec=period)
            scan_age = self.fresh_scan_age_sec()
            odom_age = self.fresh_odom_age_sec()
            if scan_age is None or scan_age > self.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_scan_during_reposition_turn")
            if odom_age is None or odom_age > self.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_odom_during_reposition_turn")
            clearance = evaluate_clearance(self.latest_scan, self.config)
            update_safety_minima(self.diagnostics, clearance)
            if not clearance.ok:
                raise RuntimeError(f"reposition_turn_clearance_failed:{clearance.reason}")
            delta = shortest_angle_delta_rad(self.latest_odom_yaw_rad, target_yaw_rad)
            if abs(delta) <= tolerance:
                return
            self.publish_turn_command(publisher, target_yaw_rad)
        raise RuntimeError("center_reposition_turn_timeout")

    def publish_drive_command(self, publisher):
        command = self.twist_factory()
        command.linear.x = abs(self.config.center_reposition_linear_speed_mps)
        publisher.publish(command)

    def drive_forward(self, publisher, distance_m):
        if self.latest_odom_pose is None:
            raise RuntimeError("fresh_odom_unavailable_before_reposition_drive")
        start_x = self.latest_odom_pose.x
        start_y = self.latest_odom_pose.y
        deadline = self.now() + max(
            8.0,
            distance_m / max(0.01, abs(self.config.center_reposition_linear_speed_mps))
            + 3.0,
        )
        period = 1.0 / self.config.control_rate_hz
        while self.rclpy.ok() and self.now() <= deadline:
            self.rclpy.spin_once(self.node, timeout_sec=period)
            scan_age = self.fresh_scan_age_sec()
            odom_age = self.fresh_odom_age_sec()
            if scan_age is None or scan_age > self.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_scan_during_reposition_drive")
            if odom_age is None or odom_age > self.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_odom_during_reposition_drive")
            clearance = evaluate_reposition_clearance(self.latest_scan, self.config)
            update_safety_minima(self.diagnostics, clearance)
            if not clearance.ok:
                raise RuntimeError(f"reposition_drive_clearance_failed:{clearance.reason}")
            dx = self.latest_odom_pose.x - start_x
            dy = self.latest_odom_pose.y - start_y
            if math.hypot(dx, dy) >= distance_m:
                return math.hypot(dx, dy)
            self.publish_drive_command(publisher)
        raise RuntimeError("center_reposition_drive_timeout")

    def print_reposition_prompt(self, action: CenterRepositionAction):
        print("\nArena-active reposition recovery")
        print(f"  nearest short wall: {action.nearest_axis_side}")
        print(f"  away direction: {action.away_axis_side}")
        print(f"  nearest range: {action.nearest_short_wall_range_m}")
        print(f"  target nearest range: {action.target_nearest_short_wall_range_m}")
        print(f"  suspected heater wall: {action.suspected_heater_axis_side}")
        print(f"  suspected heater range: {action.suspected_heater_range_m}")
        print(f"  target heater range: {action.heater_approach_target_range_m}")
        print(f"  heater scores: {action.heater_scores}")
        print(f"  heater delta: {action.heater_profile_delta}")
        print(f"  lateral offset: {action.lateral_offset_m}")
        print(f"  target lateral offset: {action.lateral_target_offset_m}")
        steps = list(action.steps)
        if not steps and action.odom_heading_rad is not None and action.planned_distance_m is not None:
            steps = [
                CenterRepositionStep(
                    kind="legacy",
                    reason=action.reason,
                    planned_distance_m=action.planned_distance_m,
                    local_heading_rad=action.local_heading_rad,
                    odom_heading_rad=action.odom_heading_rad,
                )
            ]
        for index, step in enumerate(steps, start=1):
            heading_text = f"{math.degrees(step.odom_heading_rad):.1f} deg"
            if step.dynamic_heading:
                heading_text = f"dynamic ({step.dynamic_heading_source}), initial estimate {heading_text}"
            print(
                f"  step {index} {step.kind}: "
                f"distance={step.planned_distance_m:.3f} m, "
                f"target odom heading={heading_text}"
            )
        if action.lateral_step_skipped:
            print(f"  lateral step: skipped ({action.lateral_skip_reason})")
        print("  expected action: turn, drive, optionally turn sideways, drive, then spin again")
        if self.config.require_operator_confirmation:
            self.input_fn("Press Enter to start center reposition, or Ctrl+C to abort: ")

    def execute_center_reposition(self, publisher, action: CenterRepositionAction):
        steps = list(action.steps)
        if not steps and action.odom_heading_rad is not None and action.planned_distance_m is not None:
            steps = [
                CenterRepositionStep(
                    kind="legacy",
                    reason=action.reason,
                    planned_distance_m=action.planned_distance_m,
                    local_heading_rad=action.local_heading_rad,
                    odom_heading_rad=action.odom_heading_rad,
                )
            ]
        if not action.ok or not steps:
            raise RuntimeError(action.reason)
        self.wait_for_fresh_inputs()
        self.print_reposition_prompt(action)
        self.refresh_fresh_inputs_after_prompt()
        clearance = evaluate_reposition_clearance(self.latest_scan, self.config)
        update_safety_minima(self.diagnostics, clearance)
        if not clearance.ok:
            raise RuntimeError(f"reposition_precheck_clearance_failed:{clearance.reason}")

        start = self.now()
        total_driven = 0.0
        step_records = []
        for index, step in enumerate(steps):
            if index > 0:
                self.wait_for_fresh_inputs()
            step_start = self.now()
            step_record = step.to_dict()
            target_heading = step.odom_heading_rad
            if step.dynamic_heading:
                if self.latest_odom_yaw_rad is None:
                    raise RuntimeError("fresh_odom_unavailable_before_dynamic_lateral_turn")
                dynamic_heading = dynamic_lateral_heading_from_scan(
                    self.latest_scan,
                    self.latest_odom_yaw_rad,
                )
                target_heading = dynamic_heading["odom_heading_rad"]
                step_record["dynamic_heading_result"] = dynamic_heading
                step_record["odom_heading_rad"] = target_heading
            self.turn_to_heading(publisher, target_heading)
            stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
            self.wait_for_fresh_inputs()
            driven = self.drive_forward(publisher, step.planned_distance_m)
            stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
            total_driven += driven
            step_records.append(
                {
                    **step_record,
                    "driven_distance_m": driven,
                    "duration_sec": self.now() - step_start,
                }
            )
        record = action.to_dict()
        record["steps"] = step_records
        record["driven_distance_m"] = total_driven
        record["duration_sec"] = self.now() - start
        return record

    def plan_active_explore_recovery(self, result):
        active_config = active_explore_config_from_arena_config(self.config)
        geometry_ok, reason = geometry_is_recoverable(result, active_config)
        if not geometry_ok:
            return ActiveExplorePlan(False, reason, None, (), None)
        self.wait_for_fresh_inputs()
        origin_yaw = self.first_sample_origin_yaw_rad()
        grid = None
        if (
            self.config.active_explore_use_accumulated_map
            and self.explore_samples
            and self.latest_odom_pose is not None
        ):
            grid = build_local_grid_from_scan_samples(
                self.explore_samples,
                self.latest_odom_pose,
                active_config,
            )
            self.update_temporary_map_diagnostics(grid)
            self.publish_temporary_map_if_ready(force=True, grid=grid)
        return plan_active_explore_recovery(
            result,
            self.latest_scan,
            self.latest_odom_pose,
            active_config,
            origin_yaw_rad=origin_yaw,
            grid=grid,
        )

    def active_explore_frontier_goal_diagnostics(self):
        if self.active_explore_frontier_goal is None:
            return None
        goal = dict(self.active_explore_frontier_goal)
        if goal.get("cluster_centroid_world") is not None:
            goal["cluster_centroid_world"] = list(goal["cluster_centroid_world"])
        return goal

    def clear_active_explore_frontier_goal(self, _reason):
        self.active_explore_frontier_goal = None
        self.diagnostics["active_explore"]["persistent_frontier_goal"] = None

    def store_active_explore_frontier_goal(self, candidate, attempt_index):
        metadata = candidate.metadata or {}
        previous = self.active_explore_frontier_goal or {}
        cluster_centroid = candidate_cluster_centroid(candidate)
        goal = {
            "target_x": float(candidate.target_x),
            "target_y": float(candidate.target_y),
            "cluster_centroid_world": cluster_centroid,
            "cluster_size": metadata.get("cluster_size"),
            "visible_cluster_shadow_count": candidate_visible_shadow_count(candidate),
            "created_attempt_index": previous.get(
                "created_attempt_index",
                attempt_index,
            ),
            "last_matched_attempt_index": attempt_index,
            "driven_toward_goal_m": float(previous.get("driven_toward_goal_m", 0.0)),
        }
        self.active_explore_frontier_goal = goal
        self.diagnostics["active_explore"]["persistent_frontier_goal"] = (
            self.active_explore_frontier_goal_diagnostics()
        )
        return goal

    def update_active_explore_frontier_progress(self, driven_distance_m):
        if self.active_explore_frontier_goal is None:
            return
        self.active_explore_frontier_goal["driven_toward_goal_m"] = float(
            self.active_explore_frontier_goal.get("driven_toward_goal_m", 0.0)
        ) + max(0.0, float(driven_distance_m))
        self.diagnostics["active_explore"]["persistent_frontier_goal"] = (
            self.active_explore_frontier_goal_diagnostics()
        )

    def set_active_explore_phase(self, phase):
        self.active_explore_phase = phase
        self.diagnostics["active_explore"]["active_explore_phase"] = phase

    def update_active_explore_phase_diagnostics(self):
        self.diagnostics["active_explore"]["active_explore_phase"] = (
            self.active_explore_phase
        )
        self.diagnostics["active_explore"]["shadow_frontier_empty_replans"] = (
            self.shadow_frontier_empty_replans
        )
        self.diagnostics["active_explore"]["shadow_explore_complete"] = (
            self.shadow_explore_complete
        )

    def shadow_frontier_status_from_plan(self, plan):
        frontier_candidates = [
            candidate
            for candidate in plan.candidates
            if candidate is not None and candidate.kind == "obstacle_shadow_frontier"
        ]
        accepted_frontiers = [
            candidate
            for candidate in frontier_candidates
            if candidate.accepted
        ]
        rejected_frontiers = [
            candidate
            for candidate in frontier_candidates
            if not candidate.accepted
        ]
        visible_frontiers = [
            candidate
            for candidate in accepted_frontiers
            if candidate_visible_shadow_count(candidate) > 0
        ]
        moving_frontiers = [
            candidate
            for candidate in visible_frontiers
            if candidate_path_needs_motion(candidate)
        ]
        path_lengths = [
            candidate.path_length_m
            for candidate in visible_frontiers
            if candidate.path_length_m is not None
        ]
        all_visible_counts = [
            candidate_visible_shadow_count(candidate)
            for candidate in frontier_candidates
            if candidate_visible_shadow_count(candidate) > 0
        ]
        rejection_reasons = {}
        unreachable_frontiers = []
        for candidate in rejected_frontiers:
            reason = candidate.rejection_reason or "unknown"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            if reason in ACTIVE_EXPLORE_FRONTIER_UNREACHABLE_REASONS:
                unreachable_frontiers.append(candidate)
        if moving_frontiers:
            shadow_frontier_state = "reachable"
        elif frontier_candidates:
            shadow_frontier_state = "unreachable"
        else:
            shadow_frontier_state = "absent"
        status = {
            "frontier_candidate_count": len(frontier_candidates),
            "accepted_frontier_count": len(accepted_frontiers),
            "rejected_frontier_count": len(rejected_frontiers),
            "frontier_rejection_reasons": rejection_reasons,
            "unreachable_frontier_count": len(unreachable_frontiers),
            "visible_shadow_frontier_count": len(visible_frontiers),
            "moving_shadow_frontier_count": len(moving_frontiers),
            "best_visible_shadow_count": (
                max(all_visible_counts)
                if all_visible_counts
                else 0
            ),
            "min_visible_frontier_path_m": min(path_lengths) if path_lengths else None,
            "max_visible_frontier_path_m": max(path_lengths) if path_lengths else None,
            "frontier_motion_threshold_m": ACTIVE_EXPLORE_FRONTIER_REACHED_PATH_M,
            "empty_replans_required": (
                ACTIVE_EXPLORE_SHADOW_EMPTY_REPLANS_TO_COMPLETE
            ),
            "shadow_frontier_state": shadow_frontier_state,
            "empty": shadow_frontier_state == "absent",
        }
        return status

    def update_shadow_explore_phase_from_plan(self, plan):
        status = self.shadow_frontier_status_from_plan(plan)
        if self.active_explore_phase != ACTIVE_EXPLORE_PHASE_SHADOW:
            status["empty_replans"] = self.shadow_frontier_empty_replans
            status["complete"] = self.shadow_explore_complete
            self.diagnostics["active_explore"]["shadow_frontier_status"] = status
            self.update_active_explore_phase_diagnostics()
            return status

        if status["shadow_frontier_state"] == "reachable":
            self.shadow_frontier_empty_replans = 0
            self.shadow_explore_complete = False
        elif status["shadow_frontier_state"] == "absent":
            self.shadow_frontier_empty_replans += 1
            if (
                self.shadow_frontier_empty_replans
                >= ACTIVE_EXPLORE_SHADOW_EMPTY_REPLANS_TO_COMPLETE
            ):
                self.shadow_explore_complete = True
                self.clear_active_explore_frontier_goal("shadow_frontier_exhausted")
                self.set_active_explore_phase(ACTIVE_EXPLORE_PHASE_LOCALIZATION_POSE)
        else:
            self.shadow_frontier_empty_replans = 0
            self.shadow_explore_complete = False

        status["empty_replans"] = self.shadow_frontier_empty_replans
        status["complete"] = self.shadow_explore_complete
        self.diagnostics["active_explore"]["shadow_frontier_status"] = status
        self.update_active_explore_phase_diagnostics()
        return status

    def moving_shadow_frontier_candidates(self, plan):
        return tuple(
            candidate
            for candidate in plan.candidates
            if candidate_is_moving_shadow_frontier(candidate)
        )

    def best_scored_candidate(self, candidates):
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda candidate: (
                -(candidate.score if candidate.score is not None else -math.inf),
                (
                    candidate.path_length_m
                    if candidate.path_length_m is not None
                    else math.inf
                ),
            ),
        )[0]

    def shadow_approach_fallback_candidate(self, plan):
        candidates = [
            candidate
            for candidate in plan.candidates
            if candidate_is_accepted_open_corridor(candidate)
        ]
        return self.best_scored_candidate(candidates)

    def localization_pose_candidate(self, plan):
        candidates = [
            candidate
            for candidate in plan.candidates
            if candidate_is_localization_pose_candidate(candidate)
        ]
        policy = {
            "eligible_kinds": list(ACTIVE_EXPLORE_LOCALIZATION_CANDIDATE_KINDS),
            "candidate_count": len(candidates),
            "selected_kind": None,
            "reason": "",
        }
        if not candidates:
            policy["reason"] = "no_localization_pose_candidate"
            return None, policy

        priority = {
            "suspected_heater_approach": 0,
            "provisional_center": 1,
            "lateral_recenter": 1,
        }
        candidates.sort(
            key=lambda candidate: (
                priority.get(candidate.kind, 99),
                -(candidate.score if candidate.score is not None else -math.inf),
                (
                    candidate.path_length_m
                    if candidate.path_length_m is not None
                    else math.inf
                ),
            )
        )
        selected = candidates[0]
        policy["selected_kind"] = selected.kind
        policy["reason"] = "selected"
        return selected, policy

    def frontier_goal_candidate_match(self, goal, candidate):
        match = {
            "matched": False,
            "reason": "",
            "target_distance_m": None,
            "cluster_centroid_distance_m": None,
            "visible_cluster_shadow_count": candidate_visible_shadow_count(candidate),
            "candidate": None if candidate is None else candidate.to_dict(),
        }
        if candidate is None or candidate.kind != "obstacle_shadow_frontier":
            match["reason"] = "not_obstacle_shadow_frontier"
            return match
        if not candidate.accepted:
            match["reason"] = candidate.rejection_reason or "candidate_rejected"
            return match
        if match["visible_cluster_shadow_count"] <= 0:
            match["reason"] = "visible_shadow_zero"
            return match

        goal_target = [goal.get("target_x"), goal.get("target_y")]
        if all(value is not None for value in goal_target):
            match["target_distance_m"] = distance_2d(
                goal_target,
                [candidate.target_x, candidate.target_y],
            )

        goal_cluster = finite_point_2d(goal.get("cluster_centroid_world"))
        candidate_cluster = candidate_cluster_centroid(candidate)
        if goal_cluster is not None and candidate_cluster is not None:
            match["cluster_centroid_distance_m"] = distance_2d(
                goal_cluster,
                candidate_cluster,
            )

        target_ok = (
            match["target_distance_m"] is not None
            and match["target_distance_m"] <= ACTIVE_EXPLORE_FRONTIER_TARGET_MATCH_M
        )
        cluster_ok = (
            match["cluster_centroid_distance_m"] is not None
            and match["cluster_centroid_distance_m"]
            <= ACTIVE_EXPLORE_FRONTIER_CLUSTER_MATCH_M
        )
        if not target_ok and not cluster_ok:
            match["reason"] = "frontier_goal_mismatch"
            return match
        match["matched"] = True
        match["reason"] = "matched"
        return match

    def matching_active_explore_frontier_candidate(self, plan, candidates=None):
        if self.active_explore_frontier_goal is None:
            return None, None
        matches = []
        candidate_iterable = plan.candidates if candidates is None else candidates
        for candidate in candidate_iterable:
            match = self.frontier_goal_candidate_match(
                self.active_explore_frontier_goal,
                candidate,
            )
            if match["matched"]:
                target_distance = match["target_distance_m"]
                path_length = candidate.path_length_m
                matches.append(
                    (
                        (
                            target_distance if target_distance is not None else math.inf,
                            -match["visible_cluster_shadow_count"],
                            path_length if path_length is not None else math.inf,
                        ),
                        candidate,
                        match,
                    )
                )
        if not matches:
            return None, None
        matches.sort(key=lambda item: item[0])
        return matches[0][1], matches[0][2]

    def apply_active_explore_persistent_selection(self, plan, attempt_index):
        default_selected = plan.selected
        effective_selected = default_selected
        selection_policy = "score_best"
        persistent_match = None
        abandon_reason = None

        if not plan.ok or default_selected is None:
            if self.active_explore_frontier_goal is not None:
                abandon_reason = plan.reason or "plan_not_ok"
                self.clear_active_explore_frontier_goal(abandon_reason)
            return plan, {
                "default_selected": None,
                "effective_selected": None,
                "selection_policy": selection_policy,
                "persistent_frontier_goal": self.active_explore_frontier_goal_diagnostics(),
                "persistent_frontier_match": None,
                "persistent_frontier_abandon_reason": abandon_reason,
            }

        if self.active_explore_frontier_goal is not None:
            matched_candidate, persistent_match = (
                self.matching_active_explore_frontier_candidate(plan)
            )
            if matched_candidate is not None:
                effective_selected = matched_candidate
                selection_policy = "persistent_frontier"
            else:
                abandon_reason = "no_matching_accepted_frontier"
                self.clear_active_explore_frontier_goal(abandon_reason)

        if effective_selected.kind == "obstacle_shadow_frontier":
            visible_count = candidate_visible_shadow_count(effective_selected)
            path_length = effective_selected.path_length_m
            if visible_count <= 0:
                abandon_reason = abandon_reason or "selected_frontier_visible_shadow_zero"
                self.clear_active_explore_frontier_goal(abandon_reason)
            elif (
                path_length is not None
                and path_length <= ACTIVE_EXPLORE_FRONTIER_REACHED_PATH_M
            ):
                abandon_reason = abandon_reason or "persistent_frontier_goal_reached"
                self.clear_active_explore_frontier_goal(abandon_reason)
            else:
                self.store_active_explore_frontier_goal(
                    effective_selected,
                    attempt_index,
                )
        elif self.active_explore_frontier_goal is not None:
            abandon_reason = abandon_reason or "selected_candidate_not_frontier"
            self.clear_active_explore_frontier_goal(abandon_reason)

        effective_plan = plan
        if effective_selected is not default_selected:
            effective_plan = ActiveExplorePlan(
                plan.ok,
                plan.reason,
                effective_selected,
                plan.candidates,
                plan.grid,
            )

        return effective_plan, {
            "default_selected": default_selected.to_dict(),
            "effective_selected": effective_selected.to_dict(),
            "selection_policy": selection_policy,
            "persistent_frontier_goal": self.active_explore_frontier_goal_diagnostics(),
            "persistent_frontier_match": persistent_match,
            "persistent_frontier_abandon_reason": abandon_reason,
        }

    def apply_active_explore_phase_selection(self, plan, attempt_index):
        default_selected = plan.selected
        shadow_status = self.update_shadow_explore_phase_from_plan(plan)
        localization_policy = None
        persistent_match = None
        abandon_reason = None
        continue_without_motion = False

        def diagnostics(effective_selected, selection_policy):
            return {
                "active_explore_phase": self.active_explore_phase,
                "shadow_frontier_empty_replans": self.shadow_frontier_empty_replans,
                "shadow_explore_complete": self.shadow_explore_complete,
                "shadow_frontier_status": shadow_status,
                "default_selected": (
                    None if default_selected is None else default_selected.to_dict()
                ),
                "effective_selected": (
                    None if effective_selected is None else effective_selected.to_dict()
                ),
                "selection_policy": selection_policy,
                "persistent_frontier_goal": self.active_explore_frontier_goal_diagnostics(),
                "persistent_frontier_match": persistent_match,
                "persistent_frontier_abandon_reason": abandon_reason,
                "localization_candidate_policy": localization_policy,
                "continue_without_motion": continue_without_motion,
            }

        if not plan.ok:
            if self.active_explore_frontier_goal is not None:
                abandon_reason = plan.reason or "plan_not_ok"
                self.clear_active_explore_frontier_goal(abandon_reason)
            return plan, diagnostics(None, "plan_not_ok")

        if self.active_explore_phase == ACTIVE_EXPLORE_PHASE_SHADOW:
            moving_frontiers = self.moving_shadow_frontier_candidates(plan)
            if not moving_frontiers:
                if self.active_explore_frontier_goal is not None:
                    abandon_reason = "no_moving_shadow_frontier"
                    self.clear_active_explore_frontier_goal(abandon_reason)
                if shadow_status["shadow_frontier_state"] == "unreachable":
                    fallback = self.shadow_approach_fallback_candidate(plan)
                    if fallback is None:
                        gated_plan = ActiveExplorePlan(
                            False,
                            "shadow_frontier_unreachable_no_approach_candidate",
                            None,
                            plan.candidates,
                            plan.grid,
                        )
                        return gated_plan, diagnostics(
                            None,
                            "shadow_approach_fallback",
                        )
                    effective_plan = ActiveExplorePlan(
                        True,
                        plan.reason,
                        fallback,
                        plan.candidates,
                        plan.grid,
                    )
                    return effective_plan, diagnostics(
                        fallback,
                        "shadow_approach_fallback",
                    )
                continue_without_motion = True
                gated_plan = ActiveExplorePlan(
                    False,
                    "shadow_frontier_empty_replan_wait",
                    None,
                    plan.candidates,
                    plan.grid,
                )
                return gated_plan, diagnostics(None, "shadow_frontier_required")

            effective_selected = None
            selection_policy = "shadow_frontier_best"
            if self.active_explore_frontier_goal is not None:
                effective_selected, persistent_match = (
                    self.matching_active_explore_frontier_candidate(
                        plan,
                        candidates=moving_frontiers,
                    )
                )
                if effective_selected is not None:
                    selection_policy = "persistent_frontier"
                else:
                    abandon_reason = "no_matching_accepted_frontier"
                    self.clear_active_explore_frontier_goal(abandon_reason)

            if effective_selected is None:
                effective_selected = self.best_scored_candidate(moving_frontiers)

            if effective_selected is not None:
                self.store_active_explore_frontier_goal(effective_selected, attempt_index)
                effective_plan = ActiveExplorePlan(
                    True,
                    plan.reason,
                    effective_selected,
                    plan.candidates,
                    plan.grid,
                )
                return effective_plan, diagnostics(effective_selected, selection_policy)

        localization_candidate, localization_policy = (
            self.localization_pose_candidate(plan)
        )
        self.diagnostics["active_explore"]["localization_candidate_policy"] = (
            localization_policy
        )
        if self.active_explore_frontier_goal is not None:
            abandon_reason = abandon_reason or "shadow_explore_complete"
            self.clear_active_explore_frontier_goal(abandon_reason)
        if localization_candidate is None:
            no_pose_plan = ActiveExplorePlan(
                False,
                localization_policy["reason"],
                None,
                plan.candidates,
                plan.grid,
            )
            return no_pose_plan, diagnostics(None, "localization_pose_required")

        effective_plan = ActiveExplorePlan(
            True,
            plan.reason,
            localization_candidate,
            plan.candidates,
            plan.grid,
        )
        return effective_plan, diagnostics(
            localization_candidate,
            "localization_pose",
        )

    def active_explore_steps_from_candidate(self, candidate, distance_limit_m=None):
        points = list(candidate.simplified_path_world or candidate.path_world)
        if len(points) < 2:
            raise RuntimeError("active_explore_no_motion_steps")
        if self.latest_odom_pose is None:
            raise RuntimeError("fresh_odom_unavailable_before_active_explore")

        limit = self.config.active_explore_max_single_move_m
        if distance_limit_m is not None:
            limit = min(limit, max(0.0, distance_limit_m))
        if limit <= 0.0:
            raise RuntimeError("active_explore_distance_limit_exhausted")

        previous = (float(self.latest_odom_pose.x), float(self.latest_odom_pose.y))
        remaining = limit
        steps = []
        for point in points[1:]:
            if len(steps) >= self.config.active_explore_max_path_segments:
                break
            dx = point[0] - previous[0]
            dy = point[1] - previous[1]
            distance = math.hypot(dx, dy)
            if distance <= 1e-6:
                previous = point
                continue
            planned_distance = min(distance, remaining)
            if planned_distance <= 1e-6:
                break
            heading = math.atan2(dy, dx)
            steps.append(
                CenterRepositionStep(
                    kind="active_explore",
                    reason=f"active_explore_{candidate.kind}",
                    planned_distance_m=planned_distance,
                    local_heading_rad=None,
                    odom_heading_rad=normalize_angle_rad(heading),
                )
            )
            remaining -= planned_distance
            if planned_distance < distance or remaining <= 1e-6:
                break
            previous = point

        if not steps:
            raise RuntimeError("active_explore_no_motion_steps")
        return steps

    def print_active_explore_prompt(self, candidate, path_points):
        print("\nArena-active active-explore recovery")
        print(f"  executor: {self.config.recovery_executor}")
        print(f"  selected candidate: {candidate.kind}")
        print(f"  score: {candidate.score}")
        print(f"  score components: {candidate.score_components}")
        print(f"  path length: {candidate.path_length_m}")
        print(
            "  curve follower: "
            f"lookahead={self.config.active_explore_curve_lookahead_m:.3f} m, "
            f"linear={self.config.active_explore_curve_linear_speed_mps:.3f} m/s, "
            f"max angular={self.config.active_explore_curve_max_angular_rad_s:.3f} rad/s"
        )
        print(f"  curve path points: {len(path_points)}")
        if candidate.kind == "obstacle_shadow_frontier":
            print(
                "  expected action: follow short odom-frame curve, "
                "update temporary map, then replan without forced spin"
            )
        elif self.active_explore_phase == ACTIVE_EXPLORE_PHASE_LOCALIZATION_POSE:
            print(
                "  expected action: follow localization-friendly curve, "
                "stop, then spin if safe"
            )
        else:
            print("  expected action: follow short odom-frame curve and stop")
        if self.config.require_operator_confirmation:
            self.input_fn("Press Enter to start active-explore recovery, or Ctrl+C to abort: ")

    def publish_curve_command(self, publisher, linear_x, angular_z):
        command = self.twist_factory()
        command.linear.x = float(linear_x)
        command.angular.z = float(angular_z)
        publisher.publish(command)

    def execute_active_explore_cmd_vel(self, publisher, candidate, distance_limit_m=None):
        previous_collecting = self.collecting_explore_map
        self.collecting_explore_map = True
        try:
            self.wait_for_fresh_inputs()
            move_limit = self.config.active_explore_max_single_move_m
            if distance_limit_m is not None:
                move_limit = min(move_limit, max(0.0, distance_limit_m))
            path_points = active_explore_curve_path(
                candidate,
                self.latest_odom_pose,
                move_limit,
            )
            self.print_active_explore_prompt(candidate, path_points)
            self.refresh_fresh_inputs_after_prompt()

            start = self.now()
            deadline = self.now() + max(
                8.0,
                move_limit
                / max(0.01, abs(self.config.active_explore_curve_linear_speed_mps))
                + 5.0,
            )
            period = 1.0 / self.config.control_rate_hz
            final_target = path_points[-1]
            previous_point = (
                float(self.latest_odom_pose.x),
                float(self.latest_odom_pose.y),
            )
            total_driven = 0.0
            curve_samples = []

            while self.rclpy.ok() and self.now() <= deadline:
                self.rclpy.spin_once(self.node, timeout_sec=period)
                scan_age = self.fresh_scan_age_sec()
                odom_age = self.fresh_odom_age_sec()
                if scan_age is None or scan_age > self.config.max_odom_scan_age_sec:
                    raise RuntimeError("stale_scan_during_active_explore_curve")
                if odom_age is None or odom_age > self.config.max_odom_scan_age_sec:
                    raise RuntimeError("stale_odom_during_active_explore_curve")
                if self.latest_odom_pose is None:
                    raise RuntimeError("fresh_odom_unavailable_during_active_explore_curve")

                current_point = (
                    float(self.latest_odom_pose.x),
                    float(self.latest_odom_pose.y),
                )
                delta = distance_2d(previous_point, current_point)
                if math.isfinite(delta):
                    total_driven += delta
                previous_point = current_point

                clearance = evaluate_reposition_clearance(self.latest_scan, self.config)
                update_safety_minima(self.diagnostics, clearance)
                if not clearance.ok:
                    stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
                    if total_driven >= self.config.active_explore_min_progress_before_spin_m:
                        return active_explore_curve_execution_record(
                            candidate,
                            path_points,
                            curve_samples,
                            total_driven,
                            self.now() - start,
                            "clearance_stop_after_progress",
                            clearance_failure_reason=clearance.reason,
                        )
                    raise RuntimeError(
                        f"active_explore_curve_clearance_failed:{clearance.reason}"
                    )

                if (
                    total_driven >= move_limit
                    or distance_2d(current_point, final_target)
                    <= self.config.active_explore_curve_goal_tolerance_m
                ):
                    stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
                    return active_explore_curve_execution_record(
                        candidate,
                        path_points,
                        curve_samples,
                        total_driven,
                        self.now() - start,
                        "completed",
                    )

                target = select_curve_lookahead_target(
                    path_points,
                    current_point,
                    self.config.active_explore_curve_lookahead_m,
                )
                linear_x, angular_z, alpha = pure_pursuit_curve_command(
                    self.latest_odom_pose,
                    target,
                    self.config.active_explore_curve_lookahead_m,
                    self.config.active_explore_curve_linear_speed_mps,
                    self.config.active_explore_curve_max_angular_rad_s,
                )
                remaining = max(0.0, move_limit - total_driven)
                linear_x = min(linear_x, remaining / max(period, 1e-6))
                curve_samples.append(
                    {
                        "odom_x": float(self.latest_odom_pose.x),
                        "odom_y": float(self.latest_odom_pose.y),
                        "odom_yaw_rad": math.radians(float(self.latest_odom_pose.yaw_deg)),
                        "target_x": float(target[0]),
                        "target_y": float(target[1]),
                        "alpha_rad": alpha,
                        "linear_x_mps": linear_x,
                        "angular_z_rad_s": angular_z,
                        "front_clearance_m": clearance.front_min_m,
                        "left_clearance_m": clearance.left_min_m,
                        "right_clearance_m": clearance.right_min_m,
                    }
                )
                self.publish_curve_command(publisher, linear_x, angular_z)

            timeout_sec = deadline - start
            record = active_explore_curve_execution_record(
                candidate,
                path_points,
                curve_samples,
                total_driven,
                self.now() - start,
                "timeout_stop_after_progress",
                timeout_sec=timeout_sec,
            )
            if total_driven >= self.config.active_explore_min_progress_before_spin_m:
                stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
                return record
            record["executed"] = False
            record["stop_reason"] = "active_explore_curve_timeout_before_progress"
            raise ActiveExploreMotionError(
                "active_explore_curve_timeout_before_progress",
                record,
            )
        except Exception:
            stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
            raise
        finally:
            self.collecting_explore_map = previous_collecting

    def run_legacy_recovery(self, publisher, result):
        reposition_attempts = 0
        while (
            not result.success
            and reposition_attempts < self.config.center_reposition_max_attempts
        ):
            origin_yaw = self.first_sample_origin_yaw_rad()
            action = choose_center_reposition_action(result, self.config, origin_yaw)
            attempt_record = {
                "attempt_index": len(self.diagnostics["reposition"]["attempts"]),
                "stage": "center",
                "previous_failure_reason": result.failure_reason,
                "previous_classifier_reason": result.short_wall_classification.reason,
                "action": action.to_dict(),
            }
            self.diagnostics["reposition"]["attempts"].append(attempt_record)
            if not action.ok:
                break
            self.diagnostics["fallback_used"] = True
            motion_record = self.execute_center_reposition(publisher, action)
            attempt_record["motion"] = motion_record
            reposition_attempts += 1
            self.run_spin_attempt(publisher, attempt_index=reposition_attempts)
            result = self.analyze_result()
            attempt_record["post_reposition_success"] = result.success
            attempt_record["post_reposition_failure_reason"] = result.failure_reason
            attempt_record["post_reposition_classifier_reason"] = (
                result.short_wall_classification.reason
            )

        heater_approach_attempts = 0
        while (
            not result.success
            and self.config.center_reposition_enable_heater_approach
            and heater_approach_attempts
            < self.config.center_reposition_heater_approach_max_attempts
        ):
            origin_yaw = self.first_sample_origin_yaw_rad()
            action = choose_heater_approach_reposition_action(
                result,
                self.config,
                origin_yaw,
            )
            attempt_record = {
                "attempt_index": len(self.diagnostics["reposition"]["attempts"]),
                "stage": "heater_approach",
                "previous_failure_reason": result.failure_reason,
                "previous_classifier_reason": result.short_wall_classification.reason,
                "action": action.to_dict(),
            }
            self.diagnostics["reposition"]["attempts"].append(attempt_record)
            if not action.ok:
                break
            self.diagnostics["fallback_used"] = True
            motion_record = self.execute_center_reposition(publisher, action)
            attempt_record["motion"] = motion_record
            heater_approach_attempts += 1
            self.run_spin_attempt(
                publisher,
                attempt_index=len(self.diagnostics["spin_attempts"]),
            )
            result = self.analyze_result()
            attempt_record["post_reposition_success"] = result.success
            attempt_record["post_reposition_failure_reason"] = result.failure_reason
            attempt_record["post_reposition_classifier_reason"] = (
                result.short_wall_classification.reason
            )
        return result

    def run_active_explore_recovery(self, publisher, result):
        if self.config.recovery_executor not in {"dry_run", "cmd_vel", "nav2_follow_path"}:
            raise RuntimeError(f"active_explore_executor_unknown:{self.config.recovery_executor}")

        attempts = 0
        total_distance = float(self.diagnostics["active_explore"].get("total_distance_m", 0.0))
        while (
            not result.success
            and attempts < self.config.active_explore_max_attempts
            and total_distance < self.config.active_explore_max_total_distance_m
        ):
            attempt_index = len(self.diagnostics["active_explore"]["attempts"])
            plan = self.plan_active_explore_recovery(result)
            plan, selection_diagnostics = self.apply_active_explore_phase_selection(
                plan,
                attempt_index,
            )
            plan_dict = plan.to_dict()
            rejected_unknown = [
                candidate
                for candidate in plan.candidates
                if candidate.rejection_reason == "goal_unknown"
            ]
            attempt_record = {
                "attempt_index": attempt_index,
                "stage": "active_explore",
                "executor": self.config.recovery_executor,
                "previous_failure_reason": result.failure_reason,
                "previous_classifier_reason": result.short_wall_classification.reason,
                "plan": plan_dict,
                **selection_diagnostics,
                "local_grid_stats": (
                    None if plan.grid is None else plan.grid.to_dict()["cell_counts"]
                ),
                "rejected_unknown_space_candidates": len(rejected_unknown),
                "execution": {
                    "executed": False,
                    "stop_reason": "not_started",
                    "driven_distance_m": 0.0,
                },
            }
            self.diagnostics["active_explore"]["attempts"].append(attempt_record)
            preview_limit = min(
                self.config.active_explore_max_single_move_m,
                max(0.0, self.config.active_explore_max_total_distance_m - total_distance),
            )
            self.publish_active_explore_plan_if_ready(plan, preview_limit)

            if not plan.ok or plan.selected is None:
                attempt_record["execution"]["stop_reason"] = plan.reason
                attempts += 1
                if selection_diagnostics.get("continue_without_motion"):
                    continue
                break

            if self.config.recovery_executor == "dry_run":
                attempt_record["execution"] = {
                    "executor": "dry_run",
                    "executed": False,
                    "stop_reason": "dry_run",
                    "driven_distance_m": 0.0,
                }
                break

            if self.config.recovery_executor == "nav2_follow_path":
                attempt_record["execution"] = {
                    "executor": "nav2_follow_path",
                    "executed": False,
                    "stop_reason": "nav2_follow_path_unverified_pre_amcl",
                    "driven_distance_m": 0.0,
                }
                raise RuntimeError("nav2_follow_path_unverified_pre_amcl")

            remaining_distance = (
                self.config.active_explore_max_total_distance_m - total_distance
            )
            self.diagnostics["fallback_used"] = True
            try:
                motion_record = self.execute_active_explore_cmd_vel(
                    publisher,
                    plan.selected,
                    distance_limit_m=remaining_distance,
                )
            except ActiveExploreMotionError as exc:
                motion_record = exc.record
                attempt_record["execution"] = motion_record
                total_distance += float(motion_record.get("driven_distance_m", 0.0))
                self.diagnostics["active_explore"]["total_distance_m"] = total_distance
                self.update_active_explore_frontier_progress(
                    motion_record.get("driven_distance_m", 0.0)
                )
                self.clear_active_explore_frontier_goal(exc.reason)
                raise
            except Exception:
                self.clear_active_explore_frontier_goal("active_explore_motion_failed")
                raise
            total_distance += float(motion_record.get("driven_distance_m", 0.0))
            self.update_active_explore_frontier_progress(
                motion_record.get("driven_distance_m", 0.0)
            )
            if total_distance > self.config.active_explore_max_total_distance_m + 1e-6:
                motion_record["stop_reason"] = "active_explore_total_distance_exceeded"
                attempt_record["execution"] = motion_record
                self.diagnostics["active_explore"]["total_distance_m"] = total_distance
                self.clear_active_explore_frontier_goal(
                    "active_explore_total_distance_exceeded"
                )
                raise RuntimeError("active_explore_total_distance_exceeded")

            attempt_record["execution"] = motion_record
            self.diagnostics["active_explore"]["total_distance_m"] = total_distance
            attempts += 1
            if (
                plan.selected.kind == "obstacle_shadow_frontier"
                or attempt_record.get("selection_policy") == "shadow_approach_fallback"
            ):
                decision = {
                    "action": "skip",
                    "reason": "shadow_exploration_not_complete",
                    "active_explore_phase": self.active_explore_phase,
                    "shadow_explore_complete": self.shadow_explore_complete,
                    "shadow_frontier_status": self.diagnostics["active_explore"].get(
                        "shadow_frontier_status"
                    ),
                }
                attempt_record["post_motion_spin_decision"] = decision
                attempt_record["post_recovery_spin_skipped"] = True
                attempt_record["post_recovery_spin_skip_reason"] = decision["reason"]
                self.print_active_explore_phase_spin_skip(decision["reason"])
                continue

            spin_safety = self.active_explore_spin_safety()
            attempt_record["post_motion_spin_safety"] = spin_safety
            if not spin_safety["ok"]:
                decision = {
                    "action": "skip",
                    "reason": spin_safety["reason"],
                    "active_explore_phase": self.active_explore_phase,
                    "shadow_explore_complete": self.shadow_explore_complete,
                    "spin_safety": spin_safety,
                }
                attempt_record["post_motion_spin_decision"] = decision
                stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
                attempt_record["post_recovery_spin_skipped"] = True
                attempt_record["post_recovery_spin_skip_reason"] = spin_safety["reason"]
                self.print_active_explore_spin_skip(spin_safety)
                continue

            decision = {
                "action": "spin",
                "reason": "localization_pose_reached",
                "active_explore_phase": ACTIVE_EXPLORE_PHASE_LOCALIZATION_SPIN,
                "shadow_explore_complete": self.shadow_explore_complete,
                "spin_safety": spin_safety,
            }
            attempt_record["post_motion_spin_decision"] = decision
            attempt_record["post_recovery_spin_skipped"] = False
            self.set_active_explore_phase(ACTIVE_EXPLORE_PHASE_LOCALIZATION_SPIN)
            self.run_spin_attempt(
                publisher,
                attempt_index=len(self.diagnostics["spin_attempts"]),
            )
            result = self.analyze_result()
            attempt_record["post_recovery_spin_result"] = result.to_dict()
            attempt_record["post_recovery_success"] = result.success
            attempt_record["post_recovery_failure_reason"] = result.failure_reason
            attempt_record["post_recovery_classifier_reason"] = (
                result.short_wall_classification.reason
            )
            if result.success:
                self.clear_active_explore_frontier_goal("localization_success")
            else:
                self.set_active_explore_phase(ACTIVE_EXPLORE_PHASE_LOCALIZATION_POSE)
        return result

    def finish_failure(self, reason, exception=None):
        self.diagnostics["success"] = False
        self.diagnostics["failure_reason"] = reason
        if exception is not None:
            self.diagnostics["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
            }
        self.diagnostics["samples"]["scan_samples_collected"] = len(self.samples)
        self.diagnostics["samples"]["scan_samples_used"] = len(self.samples)
        self.diagnostics["samples"]["rejected_scan_samples"] = self.rejected_samples
        path = write_diagnostics_json(self.config.diagnostics_path, self.diagnostics)
        return ArenaActiveSpinResult(False, reason, None, self.diagnostics, path)

    def finish_success(self, pose_prior):
        self.diagnostics["success"] = True
        self.diagnostics["failure_reason"] = ""
        self.diagnostics["samples"]["scan_samples_collected"] = len(self.samples)
        self.diagnostics["samples"]["scan_samples_used"] = len(self.samples)
        self.diagnostics["samples"]["rejected_scan_samples"] = self.rejected_samples
        self.diagnostics["initialpose"] = {
            "published": False,
            "reason": "dry_run" if self.config.dry_run else "pending_runner_publication",
        }
        path = write_diagnostics_json(self.config.diagnostics_path, self.diagnostics)
        return ArenaActiveSpinResult(True, None, pose_prior, self.diagnostics, path)

    def run(self, publisher):
        try:
            stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
            self.run_spin_attempt(publisher, attempt_index=0)
            result = self.analyze_result()
            recovery_mode = effective_recovery_mode(self.config)
            if not result.success and recovery_mode == "legacy":
                result = self.run_legacy_recovery(publisher, result)
            elif not result.success and recovery_mode == "active_explore":
                result = self.run_active_explore_recovery(publisher, result)
            pose_prior = self.pose_prior_from_result_or_raise(result)
            return self.finish_success(pose_prior)
        except KeyboardInterrupt:
            self.collecting = False
            stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
            return self.finish_failure("keyboard_interrupt")
        except Exception as exc:
            self.collecting = False
            stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
            return self.finish_failure(str(exc), exception=exc)


def run_arena_active_spin(
    node,
    publisher,
    config: ArenaActiveSpinConfig,
    rclpy_module,
    twist_factory,
    scan_msg_type,
    odom_msg_type,
    qos_profile,
    input_fn=input,
    time_fn=time.time,
    sleep_fn=time.sleep,
    analyze_fn=analyze_scan_samples,
    temporary_map_callback=None,
    active_explore_plan_callback=None,
):
    session = ArenaActiveSpinSession(
        node,
        config,
        rclpy_module,
        twist_factory,
        scan_msg_type,
        odom_msg_type,
        qos_profile,
        input_fn=input_fn,
        time_fn=time_fn,
        sleep_fn=sleep_fn,
        analyze_fn=analyze_fn,
        temporary_map_callback=temporary_map_callback,
        active_explore_plan_callback=active_explore_plan_callback,
    )
    return session.run(publisher)
