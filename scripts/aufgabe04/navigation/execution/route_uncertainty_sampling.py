"""Canonical route sampling for uncertainty admission.

This module is deliberately ROS-free.  It turns a map-frame route polyline into
short geometric subsegments for clearance/uncertainty budgeting without
changing the executable route itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from scripts.aufgabe04.navigation.approach.dynamic_approach_planner import (
    point_clearance_to_blocked_m,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.planning.costmap import Costmap


MAX_SAMPLE_INTERVALS_PER_ROUTE_SEGMENT = 1_000_000

_GEOMETRY_EPSILON_M = 1.0e-10
_COLLINEAR_ABSOLUTE_TOLERANCE_M2 = 1.0e-10


@dataclass(frozen=True)
class SampledRouteSegment:
    canonical_index: int
    subsegment_index: int
    start: Pose2D
    end: Pose2D
    length_m: float
    normal_x: float
    normal_y: float
    parent_interval_count: int
    actual_spacing_m: float
    samples: tuple[tuple[float, float, float], ...]
    minimum_sampled_clearance_m: float
    lipschitz_deduction_m: float
    clearance_lower_bound_m: float
    ends_at_route_corner: bool


@dataclass(frozen=True)
class RouteUncertaintySamplingProfile:
    source_pose_count: int
    canonical_poses: tuple[Pose2D, ...]
    segments: tuple[SampledRouteSegment, ...]


def sample_route_for_uncertainty_admission(
    costmap: Costmap,
    poses: Sequence[Pose2D],
    maximum_spacing_m: float,
) -> tuple[RouteUncertaintySamplingProfile | None, str | None]:
    """Return canonical short subsegments or a fail-closed error code."""

    source_error = _source_route_geometry_error(tuple(poses))
    if source_error is not None:
        return None, source_error
    canonical = _canonical_route_poses(tuple(poses))
    result: list[SampledRouteSegment] = []
    for index, (start, end) in enumerate(zip(canonical, canonical[1:])):
        dx = end.x_m - start.x_m
        dy = end.y_m - start.y_m
        length_m = math.hypot(dx, dy)
        if not math.isfinite(length_m):
            return None, f"map_route_segment_{index}_length_nonfinite"
        if length_m <= 0.0:
            return None, f"map_route_segment_{index}_zero_length_ambiguous"

        ratio = length_m / maximum_spacing_m
        if not math.isfinite(ratio):
            return None, f"map_route_segment_{index}_sample_count_nonfinite"
        interval_count = max(1, int(math.ceil(ratio)))
        if interval_count > MAX_SAMPLE_INTERVALS_PER_ROUTE_SEGMENT:
            return None, f"map_route_segment_{index}_sample_count_excessive"
        subsegment_length_m = length_m / interval_count
        for subsegment_index in range(interval_count):
            start_fraction = subsegment_index / interval_count
            end_fraction = (subsegment_index + 1) / interval_count
            sub_start = Pose2D(
                x_m=start.x_m + start_fraction * dx,
                y_m=start.y_m + start_fraction * dy,
                yaw_rad=start.yaw_rad,
            )
            sub_end = Pose2D(
                x_m=start.x_m + end_fraction * dx,
                y_m=start.y_m + end_fraction * dy,
                yaw_rad=end.yaw_rad,
            )
            samples = []
            try:
                for sample_pose in (sub_start, sub_end):
                    clearance_m = point_clearance_to_blocked_m(
                        costmap,
                        Pose2D(x_m=sample_pose.x_m, y_m=sample_pose.y_m),
                    )
                    if not math.isfinite(clearance_m) or clearance_m < 0.0:
                        return None, (
                            f"map_route_segment_{index}_sample_clearance_invalid"
                        )
                    samples.append(
                        (sample_pose.x_m, sample_pose.y_m, clearance_m)
                    )
            except (TypeError, ValueError, OverflowError):
                return None, f"map_route_segment_{index}_sampling_failed"

            minimum_sampled = min(item[2] for item in samples)
            lipschitz_deduction = subsegment_length_m / 2.0
            result.append(
                SampledRouteSegment(
                    canonical_index=index,
                    subsegment_index=subsegment_index,
                    start=sub_start,
                    end=sub_end,
                    length_m=subsegment_length_m,
                    normal_x=-dy / length_m,
                    normal_y=dx / length_m,
                    parent_interval_count=interval_count,
                    actual_spacing_m=subsegment_length_m,
                    samples=tuple(samples),
                    minimum_sampled_clearance_m=minimum_sampled,
                    lipschitz_deduction_m=lipschitz_deduction,
                    clearance_lower_bound_m=max(
                        0.0,
                        minimum_sampled - lipschitz_deduction,
                    ),
                    ends_at_route_corner=(
                        subsegment_index + 1 == interval_count
                        and index + 1 < len(canonical) - 1
                    ),
                )
            )
    return (
        RouteUncertaintySamplingProfile(
            source_pose_count=len(poses),
            canonical_poses=canonical,
            segments=tuple(result),
        ),
        None,
    )


def _source_route_geometry_error(poses: tuple[Pose2D, ...]) -> str | None:
    """Preserve fail-closed validation before removing redundant vertices.

    Canonicalization is an analysis-only representation change.  It must not
    hide a zero-length or non-finite segment present in the executable source
    route, even when a later collinear vertex would otherwise absorb it.
    """

    for index, (start, end) in enumerate(zip(poses, poses[1:])):
        length_m = math.hypot(end.x_m - start.x_m, end.y_m - start.y_m)
        if not math.isfinite(length_m):
            return f"map_route_segment_{index}_length_nonfinite"
        if length_m <= _GEOMETRY_EPSILON_M:
            return f"map_route_segment_{index}_zero_length_ambiguous"
    return None


def _canonical_route_poses(poses: tuple[Pose2D, ...]) -> tuple[Pose2D, ...]:
    if len(poses) < 3:
        return poses
    canonical = [poses[0]]
    for pose in poses[1:]:
        if _same_position(canonical[-1], pose):
            canonical.append(pose)
            continue
        while (
            len(canonical) >= 2
            and _is_redundant_collinear_vertex(canonical[-2], canonical[-1], pose)
        ):
            canonical.pop()
        canonical.append(pose)
    return tuple(canonical)


def _same_position(left: Pose2D, right: Pose2D) -> bool:
    return math.hypot(left.x_m - right.x_m, left.y_m - right.y_m) <= (
        _GEOMETRY_EPSILON_M
    )


def _is_redundant_collinear_vertex(
    previous: Pose2D,
    current: Pose2D,
    following: Pose2D,
) -> bool:
    ax = current.x_m - previous.x_m
    ay = current.y_m - previous.y_m
    bx = following.x_m - current.x_m
    by = following.y_m - current.y_m
    cross = ax * by - ay * bx
    scale = max(math.hypot(ax, ay) * math.hypot(bx, by), 1.0)
    if abs(cross) > _COLLINEAR_ABSOLUTE_TOLERANCE_M2 * scale:
        return False
    return ax * bx + ay * by >= -_GEOMETRY_EPSILON_M


__all__ = [
    "MAX_SAMPLE_INTERVALS_PER_ROUTE_SEGMENT",
    "RouteUncertaintySamplingProfile",
    "SampledRouteSegment",
    "sample_route_for_uncertainty_admission",
]
