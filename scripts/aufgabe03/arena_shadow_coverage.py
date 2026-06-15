#!/usr/bin/env python3
"""
ROS-free shadow coverage helpers for experimental arena map-collection motion.

The grid built here is temporary odom-frame evidence. It is used only to choose
short coverage motions while an external mapper creates the saved static map.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Sequence

from arena_active_explore import (
    ActiveExploreConfig,
    blocked_distance_field,
    build_local_grid_from_scan_samples,
    clearance_distance_for_cell,
    generate_obstacle_shadow_frontier_candidates,
    grid_cell_counts,
    in_bounds,
    obstacle_shadow_unknown_cells,
    plan_candidate,
    world_to_cell,
)
from arena_active_spin_core.curve_following import (
    active_explore_curve_path,
    select_curve_lookahead_target,
)


NO_SHADOW_REASONS = {
    "no_shadow_frontier",
}


@dataclass(frozen=True)
class ShadowCoverageConfig:
    max_attempts: int = 12
    max_single_move_m: float = 0.80
    max_total_distance_m: float = 5.0
    max_candidate_path_m: float | None = 3.0
    grid_resolution_m: float = 0.05
    grid_size_m: float = 5.0
    inflation_radius_m: float = 0.12
    soft_clearance_radius_m: float = 0.15
    soft_clearance_weight: float = 2.0
    unknown_blocked: bool = True
    max_path_segments: int = 24
    max_samples: int = 1500
    max_sample_age_sec: float = 30.0
    max_sample_travel_m: float = 1.25
    max_sample_yaw_span_deg: float = 420.0
    min_visible_shadow_cells: int = 3
    min_move_length_m: float = 0.12
    recent_target_radius_m: float = 0.25
    min_endpoint_clearance_m: float = 0.16
    completion_confirmations: int = 2
    max_initial_heading_error_deg: float = 80.0
    allow_initial_preturn: bool = False
    preturn_max_heading_error_deg: float = 135.0
    preturn_handoff_heading_error_deg: float = 45.0
    curve_lookahead_m: float = 0.16
    preturn_handoff_deadband_deg: float = 3.0

    def active_explore_config(self):
        return ActiveExploreConfig(
            max_attempts=self.max_attempts,
            max_single_move_m=self.max_single_move_m,
            max_total_distance_m=self.max_total_distance_m,
            max_candidate_path_m=self.max_candidate_path_m,
            grid_resolution_m=self.grid_resolution_m,
            grid_size_m=self.grid_size_m,
            inflation_radius_m=self.inflation_radius_m,
            soft_clearance_radius_m=self.soft_clearance_radius_m,
            soft_clearance_weight=self.soft_clearance_weight,
            unknown_blocked=self.unknown_blocked,
            max_path_segments=self.max_path_segments,
        )


@dataclass(frozen=True)
class ShadowScanSample:
    ranges: Sequence[float]
    angle_min: float
    angle_increment: float
    range_min: float = 0.0
    range_max: float = float("inf")
    odom_pose: object | None = None
    stamp_sec: float = 0.0
    segment_index: int = 0

    @classmethod
    def from_scan_sample(cls, sample, stamp_sec, segment_index):
        return cls(
            ranges=tuple(sample.ranges),
            angle_min=float(sample.angle_min),
            angle_increment=float(sample.angle_increment),
            range_min=float(sample.range_min),
            range_max=float(sample.range_max),
            odom_pose=sample.odom_pose,
            stamp_sec=float(stamp_sec),
            segment_index=int(segment_index),
        )


@dataclass(frozen=True)
class ShadowMovePlan:
    ok: bool
    reason: str
    selected: object | None
    candidates: tuple
    grid: object | None
    sample_window: dict
    shadow_status: dict
    candidate_rejections: dict

    def to_dict(self):
        return {
            "ok": self.ok,
            "reason": self.reason,
            "selected": None if self.selected is None else self.selected.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "grid": None if self.grid is None else self.grid.to_dict(),
            "sample_window": dict(self.sample_window),
            "shadow_status": dict(self.shadow_status),
            "candidate_rejections": dict(self.candidate_rejections),
        }


@dataclass(frozen=True)
class ShadowCoverageSummary:
    attempts: int = 0
    moves_executed: int = 0
    total_distance_m: float = 0.0
    spin_count: int = 0
    fallback_used: bool = False
    stop_reason: str = ""
    final_phase: str = ""

    def to_dict(self):
        return asdict(self)


def _pose_xy(pose):
    if pose is None:
        return None
    return (float(pose.x), float(pose.y))


def _pose_yaw_deg(pose):
    if pose is None:
        return None
    return float(getattr(pose, "yaw_deg", 0.0))


def _distance_xy(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _shortest_angle_delta_deg(start_deg, end_deg):
    return (float(end_deg) - float(start_deg) + 180.0) % 360.0 - 180.0


def sample_window_stats(samples, now_sec=None):
    ordered = tuple(sorted(samples, key=lambda sample: float(sample.stamp_sec)))
    if not ordered:
        return {
            "count": 0,
            "oldest_age_sec": None,
            "newest_age_sec": None,
            "odom_travel_span_m": 0.0,
            "yaw_span_deg": 0.0,
            "segment_min": None,
            "segment_max": None,
        }

    if now_sec is None:
        now_sec = max(float(sample.stamp_sec) for sample in ordered)
    oldest_stamp = min(float(sample.stamp_sec) for sample in ordered)
    newest_stamp = max(float(sample.stamp_sec) for sample in ordered)
    segments = [int(getattr(sample, "segment_index", 0)) for sample in ordered]

    travel_span = 0.0
    yaw_span = 0.0
    previous_pose = ordered[0].odom_pose
    previous_xy = _pose_xy(previous_pose)
    previous_yaw = _pose_yaw_deg(previous_pose)
    for sample in ordered[1:]:
        xy = _pose_xy(sample.odom_pose)
        yaw = _pose_yaw_deg(sample.odom_pose)
        if previous_xy is not None and xy is not None:
            travel_span += _distance_xy(previous_xy, xy)
        if previous_yaw is not None and yaw is not None:
            yaw_span += abs(_shortest_angle_delta_deg(previous_yaw, yaw))
        previous_xy = xy
        previous_yaw = yaw

    return {
        "count": len(ordered),
        "oldest_age_sec": max(0.0, float(now_sec) - oldest_stamp),
        "newest_age_sec": max(0.0, float(now_sec) - newest_stamp),
        "odom_travel_span_m": travel_span,
        "yaw_span_deg": yaw_span,
        "segment_min": min(segments),
        "segment_max": max(segments),
    }


def prune_shadow_samples(samples, config: ShadowCoverageConfig, now_sec=None, current_segment=None):
    kept = tuple(sorted(samples, key=lambda sample: float(sample.stamp_sec)))
    if not kept:
        return (), sample_window_stats(())

    if current_segment is None:
        current_segment = max(int(getattr(sample, "segment_index", 0)) for sample in kept)
    min_segment = int(current_segment) - 1
    kept = tuple(
        sample
        for sample in kept
        if int(getattr(sample, "segment_index", 0)) >= min_segment
    )

    if now_sec is None and kept:
        now_sec = max(float(sample.stamp_sec) for sample in kept)
    if now_sec is not None and config.max_sample_age_sec > 0.0:
        kept = tuple(
            sample
            for sample in kept
            if float(now_sec) - float(sample.stamp_sec) <= config.max_sample_age_sec
        )

    max_samples = max(1, int(config.max_samples))
    if len(kept) > max_samples:
        kept = kept[-max_samples:]

    while len(kept) > 1:
        stats = sample_window_stats(kept, now_sec=now_sec)
        if (
            stats["odom_travel_span_m"] <= config.max_sample_travel_m + 1e-9
            and stats["yaw_span_deg"] <= config.max_sample_yaw_span_deg + 1e-9
        ):
            break
        kept = kept[1:]

    return kept, sample_window_stats(kept, now_sec=now_sec)


def candidate_visible_shadow_count(candidate):
    if candidate is None:
        return 0
    value = candidate.metadata.get("visible_cluster_shadow_count")
    if value is None:
        value = candidate.score_components.get("visible_shadow_unknown_count")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _recent_attempt_targets(recent_attempts):
    targets = []
    for attempt in recent_attempts or ():
        if isinstance(attempt, dict):
            x = attempt.get("target_x")
            y = attempt.get("target_y")
        else:
            x = getattr(attempt, "target_x", None)
            y = getattr(attempt, "target_y", None)
        try:
            x = float(x)
            y = float(y)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            targets.append((x, y))
    return tuple(targets)


def _recently_attempted(candidate, recent_attempts, radius_m):
    target = (float(candidate.target_x), float(candidate.target_y))
    return any(
        _distance_xy(target, previous) <= radius_m
        for previous in _recent_attempt_targets(recent_attempts)
    )


def _endpoint_clearance_m(candidate, grid, clearance_distance_field):
    target_cell = world_to_cell(grid, candidate.target_x, candidate.target_y)
    if not in_bounds(grid, target_cell):
        return 0.0
    return clearance_distance_for_cell(clearance_distance_field, target_cell)


def _reject_candidate(candidate, reason, extra_metadata=None):
    metadata = dict(candidate.metadata)
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata["shadow_coverage_rejection_reason"] = reason
    return replace(
        candidate,
        accepted=False,
        rejection_reason=reason,
        metadata=metadata,
    )


def _annotate_candidate(candidate, extra_metadata):
    metadata = dict(candidate.metadata)
    metadata.update(extra_metadata)
    return replace(candidate, metadata=metadata)


def _increment(counts, reason):
    counts[reason] = counts.get(reason, 0) + 1


def _shadow_status(grid, candidates, raw_shadow_unknown_count):
    frontier_candidates = [
        candidate
        for candidate in candidates
        if candidate is not None and candidate.kind == "obstacle_shadow_frontier"
    ]
    accepted_frontiers = [
        candidate
        for candidate in frontier_candidates
        if candidate.accepted
    ]
    moving_frontiers = [
        candidate
        for candidate in accepted_frontiers
        if candidate_visible_shadow_count(candidate) > 0
        and (candidate.path_length_m or 0.0) > 0.0
    ]
    if moving_frontiers:
        state = "reachable"
    elif frontier_candidates:
        state = "unreachable"
    else:
        state = "absent"
    return {
        "frontier_state": state,
        "raw_shadow_unknown_cell_count": raw_shadow_unknown_count,
        "shadow_unknown_cell_count": raw_shadow_unknown_count if moving_frontiers else 0,
        "frontier_candidate_count": len(frontier_candidates),
        "accepted_frontier_count": len(accepted_frontiers),
        "moving_frontier_count": len(moving_frontiers),
        "cell_counts": None if grid is None else grid_cell_counts(grid),
    }


def heading_error_to_point_deg(robot_pose, target_point):
    dx = float(target_point[0]) - float(robot_pose.x)
    dy = float(target_point[1]) - float(robot_pose.y)
    target_heading = math.atan2(dy, dx)
    yaw = math.radians(float(robot_pose.yaw_deg))
    return math.degrees((target_heading - yaw + math.pi) % (2.0 * math.pi) - math.pi)


def candidate_initial_heading_metadata(candidate, robot_pose, config: ShadowCoverageConfig):
    path_points = active_explore_curve_path(
        candidate,
        robot_pose,
        config.max_single_move_m,
    )
    current_point = (float(robot_pose.x), float(robot_pose.y))
    lookahead_target = select_curve_lookahead_target(
        path_points,
        current_point,
        config.curve_lookahead_m,
    )
    heading_error_deg = heading_error_to_point_deg(robot_pose, lookahead_target)
    return {
        "initial_heading_error_deg": heading_error_deg,
        "initial_lookahead_target": [float(lookahead_target[0]), float(lookahead_target[1])],
        "requires_initial_preturn": False,
        "heading_gate_stage": "planner",
    }


def apply_initial_heading_gate(candidate, robot_pose, config: ShadowCoverageConfig):
    try:
        metadata = candidate_initial_heading_metadata(candidate, robot_pose, config)
    except RuntimeError as exc:
        return _reject_candidate(
            candidate,
            "initial_heading_error_unavailable",
            {
                "heading_gate_stage": "planner",
                "heading_gate_reason": str(exc),
            },
        )

    error_abs = abs(metadata["initial_heading_error_deg"])
    if error_abs <= config.max_initial_heading_error_deg:
        metadata["heading_gate_reason"] = "accepted"
        return _annotate_candidate(candidate, metadata)
    if (
        config.allow_initial_preturn
        and error_abs <= config.preturn_max_heading_error_deg
    ):
        metadata["requires_initial_preturn"] = True
        metadata["heading_gate_reason"] = "preturn_required"
        return _annotate_candidate(candidate, metadata)
    return _reject_candidate(
        candidate,
        "initial_heading_error_too_large",
        {
            **metadata,
            "heading_gate_reason": "initial_heading_error_too_large",
            "max_initial_heading_error_deg": config.max_initial_heading_error_deg,
            "preturn_max_heading_error_deg": config.preturn_max_heading_error_deg,
        },
    )


def plan_shadow_coverage_move(samples, config: ShadowCoverageConfig, recent_attempts=()):
    samples, sample_window = prune_shadow_samples(samples, config)
    if not samples:
        return ShadowMovePlan(
            False,
            "no_samples",
            None,
            (),
            None,
            sample_window,
            _shadow_status(None, (), 0),
            {},
        )

    robot_pose = getattr(samples[-1], "odom_pose", None)
    if robot_pose is None:
        return ShadowMovePlan(
            False,
            "missing_latest_sample_pose",
            None,
            (),
            None,
            sample_window,
            _shadow_status(None, (), 0),
            {},
        )

    active_config = config.active_explore_config()
    grid = build_local_grid_from_scan_samples(samples, robot_pose, active_config)
    raw_shadow_unknown_count = len(obstacle_shadow_unknown_cells(grid))
    clearance_distance_field = blocked_distance_field(grid)
    raw_candidates = generate_obstacle_shadow_frontier_candidates(
        grid,
        active_config,
        clearance_distance_field=clearance_distance_field,
    )
    rejections = {}
    candidates = []

    for raw in raw_candidates:
        candidate = plan_candidate(
            raw,
            grid,
            active_config,
            clearance_distance_field=clearance_distance_field,
        )
        if not candidate.accepted:
            _increment(rejections, candidate.rejection_reason or "blocked")
            candidates.append(candidate)
            continue

        visible_count = candidate_visible_shadow_count(candidate)
        if visible_count < config.min_visible_shadow_cells:
            reason = "no_positive_gain"
            _increment(rejections, reason)
            candidates.append(
                _reject_candidate(
                    candidate,
                    reason,
                    {"visible_shadow_count": visible_count},
                )
            )
            continue

        if (candidate.path_length_m or 0.0) < config.min_move_length_m:
            reason = "too_short"
            _increment(rejections, reason)
            candidates.append(candidate if config.min_move_length_m <= 0.0 else _reject_candidate(candidate, reason))
            continue

        if _recently_attempted(candidate, recent_attempts, config.recent_target_radius_m):
            reason = "recently_attempted"
            _increment(rejections, reason)
            candidates.append(_reject_candidate(candidate, reason))
            continue

        clearance_m = _endpoint_clearance_m(candidate, grid, clearance_distance_field)
        if clearance_m < config.min_endpoint_clearance_m:
            reason = "too_close_to_obstacle"
            _increment(rejections, reason)
            candidates.append(
                _reject_candidate(
                    candidate,
                    reason,
                    {"endpoint_clearance_m": clearance_m},
                )
            )
            continue

        candidate = apply_initial_heading_gate(candidate, robot_pose, config)
        if not candidate.accepted:
            _increment(rejections, candidate.rejection_reason or "initial_heading_error_too_large")
            candidates.append(candidate)
            continue

        candidates.append(candidate)

    candidates = tuple(candidates)
    accepted = tuple(candidate for candidate in candidates if candidate.accepted)
    status = _shadow_status(grid, candidates, raw_shadow_unknown_count)

    if not accepted:
        reason = (
            "no_shadow_frontier"
            if not raw_candidates or raw_shadow_unknown_count <= 0
            else "shadow_blocked_or_incomplete"
        )
        return ShadowMovePlan(
            False,
            reason,
            None,
            candidates,
            grid,
            sample_window,
            status,
            rejections,
        )

    selected = max(
        accepted,
        key=lambda candidate: (
            candidate.score if candidate.score is not None else -math.inf,
            candidate_visible_shadow_count(candidate),
            -(candidate.path_length_m or math.inf),
        ),
    )
    return ShadowMovePlan(
        True,
        "selected",
        selected,
        candidates,
        grid,
        sample_window,
        status,
        rejections,
    )
