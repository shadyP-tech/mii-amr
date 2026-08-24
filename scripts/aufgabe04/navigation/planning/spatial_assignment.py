"""Deterministic, dependency-free assignment of nearby 2D points.

The Aufgabe 04 survey observes only a small number of stand candidates in one
stopped epoch.  A bit-mask dynamic program is therefore a compact way to get a
global assignment without adding a numeric/optimization dependency.  The mask
always represents the smaller input side, so an unbalanced input remains
cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class SpatialAssignment:
    """One left-to-right match selected under a distance gate."""

    left_index: int
    right_index: int
    distance_m: float


@dataclass(frozen=True)
class _PartialAssignment:
    total_distance_m: float
    pairs: tuple[tuple[int, int], ...]


def assign_spatial_points(
    left_points: Sequence[tuple[float, float]],
    right_points: Sequence[tuple[float, float]],
    *,
    maximum_distance_m: float,
) -> tuple[SpatialAssignment, ...]:
    """Return a maximum-cardinality, minimum-distance one-to-one assignment.

    Only pairs at or below ``maximum_distance_m`` are eligible.  Among
    assignments with the same cardinality, the one with the smallest total
    Euclidean distance wins.  Exact cost ties are resolved by the
    lexicographically smallest tuple of ``(left_index, right_index)`` pairs.
    This makes the result deterministic for canonically ordered inputs.

    Runtime is ``O(max(L, R) * min(L, R) * 2**min(L, R))`` and memory is
    ``O(2**min(L, R))``.  Survey epochs contain a small number of detections;
    using the smaller side as the mask also keeps large, unbalanced registries
    bounded by the epoch size.
    """

    _validate_maximum_distance(maximum_distance_m)
    normalized_left = _validated_points(left_points, "left_points")
    normalized_right = _validated_points(right_points, "right_points")
    if not normalized_left or not normalized_right:
        return ()

    small_is_left = len(normalized_left) <= len(normalized_right)
    if small_is_left:
        small_points = normalized_left
        large_points = normalized_right
    else:
        small_points = normalized_right
        large_points = normalized_left

    eligible_by_large: list[tuple[tuple[int, float], ...]] = []
    for large_point in large_points:
        eligible_small_points: list[tuple[int, float]] = []
        for small_index, small_point in enumerate(small_points):
            distance_m = math.hypot(
                small_point[0] - large_point[0],
                small_point[1] - large_point[1],
            )
            if distance_m <= maximum_distance_m:
                eligible_small_points.append((small_index, distance_m))
        eligible_by_large.append(tuple(eligible_small_points))
    states: dict[int, _PartialAssignment] = {
        0: _PartialAssignment(total_distance_m=0.0, pairs=())
    }
    for large_index, eligible_small_points in enumerate(eligible_by_large):
        next_states = dict(states)
        for mask, partial in states.items():
            for small_index, distance_m in eligible_small_points:
                bit = 1 << small_index
                if mask & bit:
                    continue
                if small_is_left:
                    pair = (small_index, large_index)
                else:
                    pair = (large_index, small_index)
                next_mask = mask | bit
                proposal = _PartialAssignment(
                    total_distance_m=partial.total_distance_m + distance_m,
                    pairs=tuple(sorted((*partial.pairs, pair))),
                )
                incumbent = next_states.get(next_mask)
                if incumbent is None or _is_better_same_mask(
                    proposal,
                    incumbent,
                ):
                    next_states[next_mask] = proposal
        states = next_states

    _best_mask, best = min(
        states.items(),
        key=lambda item: (
            -item[0].bit_count(),
            item[1].total_distance_m,
            item[1].pairs,
        ),
    )
    return tuple(
        SpatialAssignment(
            left_index=left_index,
            right_index=right_index,
            distance_m=math.hypot(
                normalized_left[left_index][0]
                - normalized_right[right_index][0],
                normalized_left[left_index][1]
                - normalized_right[right_index][1],
            ),
        )
        for left_index, right_index in best.pairs
    )


def _is_better_same_mask(
    proposal: _PartialAssignment,
    incumbent: _PartialAssignment,
) -> bool:
    return (proposal.total_distance_m, proposal.pairs) < (
        incumbent.total_distance_m,
        incumbent.pairs,
    )


def _validated_points(
    points: Sequence[tuple[float, float]],
    name: str,
) -> tuple[tuple[float, float], ...]:
    validated: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        if len(point) != 2:
            raise ValueError(f"{name}[{index}] must contain exactly two coordinates")
        x_m, y_m = point
        if not math.isfinite(x_m) or not math.isfinite(y_m):
            raise ValueError(f"{name}[{index}] coordinates must be finite")
        validated.append((x_m, y_m))
    return tuple(validated)


def _validate_maximum_distance(maximum_distance_m: float) -> None:
    if not math.isfinite(maximum_distance_m) or maximum_distance_m < 0.0:
        raise ValueError("maximum_distance_m must be finite and non-negative")
