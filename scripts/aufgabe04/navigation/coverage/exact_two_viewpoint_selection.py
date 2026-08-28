"""Pure geometry for selecting two longitudinally diverse LiDAR viewpoints.

The selector is deliberately independent of ROS and route execution. Its
inputs are already-snapped, traversable map cells plus the visibility sets
computed by the coverage planner. This keeps the geometry policy testable
without weakening later route-certificate or motion-admission gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from scripts.aufgabe04.navigation.foundation.models import GridCell


_EPSILON = 1.0e-12
DEFAULT_EXACT_TWO_CANDIDATE_SPACING_M = 0.40
DEFAULT_MINIMUM_EXACT_TWO_VIEWPOINT_BASELINE_M = 1.00


@dataclass(frozen=True)
class ExactTwoViewpointCandidate:
    """One snapped viewpoint and its map-derived visibility evidence."""

    cell: GridCell
    x_m: float
    y_m: float
    visible_cells: tuple[GridCell, ...]


def select_exact_two_viewpoint_cells(
    candidates: Sequence[ExactTwoViewpointCandidate],
    *,
    surveyable_world_xy: Mapping[GridCell, tuple[float, float]],
    coverage_threshold: float,
    minimum_viewpoint_baseline_m: float,
    start_x_m: float,
    start_y_m: float,
) -> tuple[GridCell, GridCell]:
    """Choose an ordered exact-two pair with different observation bearings.

    A pair is admitted only when it uses distinct snapped cells, has shared
    visibility, reaches the requested union-coverage threshold, and provides
    the persisted minimum world-space baseline. Among admitted pairs, the
    mean axis-incidence diversity ``abs(sin(delta bearing))`` is maximized.
    The metric treats opposite views of a thin stand head as the same
    incidence and rewards genuinely different observation bearings. Since
    coverage and baseline are hard gates, shorter travel is preferred next;
    union/shared coverage and source order are deterministic final tie-breaks.
    """

    _validate_selection_inputs(
        candidates,
        surveyable_world_xy=surveyable_world_xy,
        coverage_threshold=coverage_threshold,
        minimum_viewpoint_baseline_m=minimum_viewpoint_baseline_m,
        start_x_m=start_x_m,
        start_y_m=start_y_m,
    )
    surveyable_count = len(surveyable_world_xy)
    visibility_sets = tuple(
        frozenset(candidate.visible_cells) for candidate in candidates
    )

    best: tuple[
        tuple[float, float, int, int, int, int, int, int, int, int],
        tuple[GridCell, GridCell],
    ] | None = None
    for first_index, first in enumerate(candidates):
        for second_index in range(first_index + 1, len(candidates)):
            second = candidates[second_index]
            if first.cell == second.cell:
                continue
            viewpoint_baseline_m = math.hypot(
                second.x_m - first.x_m,
                second.y_m - first.y_m,
            )
            if (
                viewpoint_baseline_m + _EPSILON
                < minimum_viewpoint_baseline_m
            ):
                continue

            shared = visibility_sets[first_index].intersection(
                visibility_sets[second_index]
            )
            if not shared:
                continue
            union_count = len(
                visibility_sets[first_index].union(
                    visibility_sets[second_index]
                )
            )
            if union_count / surveyable_count + _EPSILON < coverage_threshold:
                continue

            diversity = _mean_axis_incidence_diversity(
                first,
                second,
                shared,
                surveyable_world_xy,
            )
            travel_distance_m = math.hypot(
                first.x_m - start_x_m,
                first.y_m - start_y_m,
            ) + viewpoint_baseline_m
            # Round computed floats only for deterministic equality/tie
            # handling. Hard gates above always use unrounded values.
            score = (
                -round(diversity, 12),
                round(travel_distance_m, 12),
                -union_count,
                -len(shared),
                first_index,
                second_index,
                first.cell.x,
                first.cell.y,
                second.cell.x,
                second.cell.y,
            )
            selected = (score, (first.cell, second.cell))
            if best is None or selected[0] < best[0]:
                best = selected

    if best is None:
        raise ValueError(
            "no valid longitudinal exact-two inspection-point pair satisfies "
            "distinct cells, shared visibility, coverage threshold, and "
            "minimum world-space viewpoint baseline"
        )
    return best[1]


def _mean_axis_incidence_diversity(
    first: ExactTwoViewpointCandidate,
    second: ExactTwoViewpointCandidate,
    shared_cells: frozenset[GridCell],
    surveyable_world_xy: Mapping[GridCell, tuple[float, float]],
) -> float:
    values = []
    for cell in sorted(shared_cells):
        target_x_m, target_y_m = surveyable_world_xy[cell]
        first_dx = target_x_m - first.x_m
        first_dy = target_y_m - first.y_m
        second_dx = target_x_m - second.x_m
        second_dy = target_y_m - second.y_m
        if (
            math.hypot(first_dx, first_dy) <= _EPSILON
            or math.hypot(second_dx, second_dy) <= _EPSILON
        ):
            continue
        first_bearing = math.atan2(first_dy, first_dx)
        second_bearing = math.atan2(second_dy, second_dx)
        values.append(abs(math.sin(first_bearing - second_bearing)))
    return sum(values) / len(values) if values else 0.0


def _validate_selection_inputs(
    candidates: Sequence[ExactTwoViewpointCandidate],
    *,
    surveyable_world_xy: Mapping[GridCell, tuple[float, float]],
    coverage_threshold: float,
    minimum_viewpoint_baseline_m: float,
    start_x_m: float,
    start_y_m: float,
) -> None:
    if len(candidates) < 2:
        raise ValueError("exact-two selection requires at least two candidates")
    if not surveyable_world_xy:
        raise ValueError("exact-two selection requires surveyable cells")
    if not math.isfinite(coverage_threshold) or not (
        0.0 < coverage_threshold <= 1.0
    ):
        raise ValueError("coverage_threshold must be in (0, 1]")
    if (
        isinstance(minimum_viewpoint_baseline_m, bool)
        or not isinstance(minimum_viewpoint_baseline_m, (int, float))
        or not math.isfinite(minimum_viewpoint_baseline_m)
        or minimum_viewpoint_baseline_m <= 0.0
    ):
        raise ValueError(
            "minimum_viewpoint_baseline_m must be finite and positive"
        )
    if not all(math.isfinite(value) for value in (start_x_m, start_y_m)):
        raise ValueError("start geometry must be finite")

    surveyable_cells = frozenset(surveyable_world_xy)
    for cell, world_xy in surveyable_world_xy.items():
        if len(world_xy) != 2 or not all(
            math.isfinite(value) for value in world_xy
        ):
            raise ValueError(f"surveyable cell {cell!r} has invalid world geometry")
    for candidate in candidates:
        if not all(math.isfinite(value) for value in (candidate.x_m, candidate.y_m)):
            raise ValueError("exact-two candidate coordinates must be finite")
        if not set(candidate.visible_cells).issubset(surveyable_cells):
            raise ValueError(
                "exact-two candidate visibility contains a non-surveyable cell"
            )
