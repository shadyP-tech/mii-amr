"""Collision-checked line-of-sight smoothing for planned route geometry.

The caller must pass the same already-inflated configuration-space costmap
used for planning.  This module is ROS-free and only removes vertices when the
complete replacement segment is traversable.  Runtime pursuit policy and
execution certificates remain unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

from scripts.aufgabe04.navigation.costmap import Costmap
from scripts.aufgabe04.navigation.global_planner import PlanRouteResult
from scripts.aufgabe04.navigation.models import GridCell, Pose2D, Route, RoutePoint


_EPSILON = 1.0e-10


@dataclass(frozen=True)
class RouteSmoothingSummary:
    enabled: bool
    input_point_count: int
    output_point_count: int
    input_length_m: float
    output_length_m: float
    optimized: bool
    skipped_reason: str = ""

    def to_metadata(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "input_point_count": self.input_point_count,
            "output_point_count": self.output_point_count,
            "input_length_m": self.input_length_m,
            "output_length_m": self.output_length_m,
            "optimized": self.optimized,
            "skipped_reason": self.skipped_reason,
        }


@dataclass(frozen=True)
class SmoothedPlanRouteResult:
    result: PlanRouteResult
    summary: RouteSmoothingSummary


def _finite_pose(pose: Pose2D, *, name: str) -> None:
    if not isinstance(pose, Pose2D):
        raise ValueError(f"{name} must be a Pose2D")
    if not all(math.isfinite(value) for value in (pose.x_m, pose.y_m)):
        raise ValueError(f"{name} position must be finite")


def _segment_intersects_closed_cell(
    costmap: Costmap,
    start: Pose2D,
    end: Pose2D,
    cell: GridCell,
) -> bool:
    origin_x, origin_y, _ = costmap.metadata.origin
    x_min = origin_x + cell.x * costmap.resolution
    y_min = origin_y + cell.y * costmap.resolution
    x_max = x_min + costmap.resolution
    y_max = y_min + costmap.resolution
    dx = end.x_m - start.x_m
    dy = end.y_m - start.y_m
    t_min = 0.0
    t_max = 1.0
    for coordinate, delta, lower, upper in (
        (start.x_m, dx, x_min, x_max),
        (start.y_m, dy, y_min, y_max),
    ):
        if abs(delta) <= _EPSILON:
            if coordinate < lower - _EPSILON or coordinate > upper + _EPSILON:
                return False
            continue
        near = (lower - coordinate) / delta
        far = (upper - coordinate) / delta
        if near > far:
            near, far = far, near
        t_min = max(t_min, near)
        t_max = min(t_max, far)
        if t_min > t_max + _EPSILON:
            return False
    return True


def supercover_segment_cells(
    costmap: Costmap,
    start: Pose2D,
    end: Pose2D,
) -> tuple[GridCell, ...]:
    """Return every grid square touched by a closed world-space segment.

    Closed-square intersection includes both sides of a grid boundary and all
    four cells at a corner, so shortcuts cannot graze occupied geometry.
    """

    _finite_pose(start, name="segment start")
    _finite_pose(end, name="segment end")
    origin_x, origin_y, _ = costmap.metadata.origin
    resolution = costmap.resolution
    min_x = math.floor((min(start.x_m, end.x_m) - origin_x) / resolution) - 1
    max_x = math.floor((max(start.x_m, end.x_m) - origin_x) / resolution) + 1
    min_y = math.floor((min(start.y_m, end.y_m) - origin_y) / resolution) - 1
    max_y = math.floor((max(start.y_m, end.y_m) - origin_y) / resolution) + 1
    touched = []
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            cell = GridCell(x, y)
            if _segment_intersects_closed_cell(costmap, start, end, cell):
                touched.append(cell)
    return tuple(sorted(set(touched)))


def segment_is_collision_free(
    costmap: Costmap,
    start: Pose2D,
    end: Pose2D,
) -> bool:
    """Validate a segment against an already configuration-space costmap."""

    return all(
        costmap.is_traversable(cell)
        for cell in supercover_segment_cells(costmap, start, end)
    )


def greedy_line_of_sight_retained_indices(
    costmap: Costmap,
    poses: Sequence[Pose2D],
    *,
    protected_indices: Iterable[int] = (),
) -> tuple[int, ...]:
    """Return a greedy collision-free subsequence of polyline indices.

    ``protected_indices`` divide the route into independent smoothing spans.
    They are retained even when adjacent protected vertices have the same
    position, which preserves semantic zero-length handoffs and certified
    exact-start connector anchors.
    """

    if not poses:
        return ()
    for index, pose in enumerate(poses):
        _finite_pose(pose, name=f"poses[{index}]")
    protected = {0, len(poses) - 1}
    for value in protected_indices:
        if type(value) is not int or not 0 <= value < len(poses):
            raise ValueError("protected route index is outside the polyline")
        protected.add(value)

    retained = [0]
    for boundary in sorted(protected)[1:]:
        anchor = retained[-1]
        while anchor < boundary:
            selected = None
            for candidate in range(boundary, anchor, -1):
                if segment_is_collision_free(
                    costmap,
                    poses[anchor],
                    poses[candidate],
                ):
                    selected = candidate
                    break
            if selected is None:
                raise ValueError("input polyline contains a colliding segment")
            retained.append(selected)
            anchor = selected
    return tuple(retained)


def greedy_line_of_sight_shortcut(
    costmap: Costmap,
    poses: Sequence[Pose2D],
    *,
    protected_indices: Iterable[int] = (),
) -> tuple[Pose2D, ...]:
    """Remove vertices only when each full replacement chord is collision-free."""

    indices = greedy_line_of_sight_retained_indices(
        costmap,
        poses,
        protected_indices=protected_indices,
    )
    return tuple(poses[index] for index in indices)


def smooth_plan_route_result(
    result: PlanRouteResult,
    *,
    costmap: Costmap,
    protected_indices: Iterable[int] = (),
) -> PlanRouteResult:
    """Rebuild a successful plan from its collision-checked retained vertices.

    ``PlanningDiagnostics.path_cell_count`` remains the raw A* cell count so
    diagnostics retain planner effort.  The rebuilt route and its length are
    the geometry written to artifacts and later certified for execution.
    """

    if result.route is None:
        return result
    original = result.route
    if not original.points:
        raise ValueError("cannot smooth an empty successful route")
    retained = greedy_line_of_sight_retained_indices(
        costmap,
        tuple(point.pose for point in original.points),
        protected_indices=protected_indices,
    )

    points = []
    cumulative_m = 0.0
    previous_pose = None
    for output_index, original_index in enumerate(retained):
        original_point = original.points[original_index]
        segment_m = (
            0.0
            if previous_pose is None
            else math.hypot(
                original_point.pose.x_m - previous_pose.x_m,
                original_point.pose.y_m - previous_pose.y_m,
            )
        )
        cumulative_m += segment_m
        points.append(
            RoutePoint(
                index=output_index,
                cell=original_point.cell,
                pose=original_point.pose,
                segment_length_m=segment_m,
                cumulative_length_m=cumulative_m,
            )
        )
        previous_pose = original_point.pose

    route = Route(
        points=tuple(points),
        requested_start=original.requested_start,
        requested_goal=original.requested_goal,
        snapped_start=original.snapped_start,
        snapped_goal=original.snapped_goal,
        length_m=cumulative_m,
    )
    diagnostics = replace(result.diagnostics, route_length_m=route.length_m)
    return PlanRouteResult(
        route=route,
        diagnostics=diagnostics,
        failure=result.failure,
    )


def smooth_plan_route_after_certified_prefix(
    result: PlanRouteResult,
    *,
    costmap: Costmap,
    prefix_end_index: int,
) -> PlanRouteResult:
    """Smooth only the tail following a separately certified route prefix.

    Prefix segments are retained byte-for-byte and intentionally are not
    revalidated by the cell-level costmap. This narrow exception is for an
    exact metric start connector whose continuous-clearance proof can admit a
    pose inside a conservatively inflated grid cell. The caller must validate
    that proof and any non-static overlays before calling this function.
    """

    if result.route is None:
        return result
    original = result.route
    if (
        type(prefix_end_index) is not int
        or prefix_end_index < 1
        or prefix_end_index >= len(original.points)
    ):
        raise ValueError("certified prefix end index is outside the route")
    tail = original.points[prefix_end_index:]
    retained_tail = greedy_line_of_sight_retained_indices(
        costmap,
        tuple(point.pose for point in tail),
    )
    selected = [*original.points[:prefix_end_index]]
    selected.extend(tail[index] for index in retained_tail)

    points = []
    cumulative_m = 0.0
    previous_pose = None
    for index, point in enumerate(selected):
        segment_m = (
            0.0
            if previous_pose is None
            else math.hypot(
                point.pose.x_m - previous_pose.x_m,
                point.pose.y_m - previous_pose.y_m,
            )
        )
        cumulative_m += segment_m
        points.append(
            replace(
                point,
                index=index,
                segment_length_m=segment_m,
                cumulative_length_m=cumulative_m,
            )
        )
        previous_pose = point.pose
    route = replace(
        original,
        points=tuple(points),
        length_m=cumulative_m,
    )
    return PlanRouteResult(
        route=route,
        diagnostics=replace(result.diagnostics, route_length_m=route.length_m),
        failure=result.failure,
    )


def smooth_plan_route_from_exact_start(
    result: PlanRouteResult,
    *,
    costmap: Costmap,
    exact_start: Pose2D,
) -> PlanRouteResult:
    """Smooth from the live metric start on the complete planning costmap.

    The exact pose is inserted before shortcut selection, so a long first
    chord is accepted only when the same inflated costmap and live keepout
    overlays used by A* certify the entire segment.  This avoids turning the
    base-map-only exact-start connector into an unchecked distant shortcut.
    """

    if result.route is None:
        return result
    _finite_pose(exact_start, name="exact_start")
    if not segment_is_collision_free(costmap, exact_start, exact_start):
        raise ValueError("exact route start is not traversable on the planning costmap")

    original = result.route
    source_points = [
        RoutePoint(
            index=0,
            cell=costmap.world_to_grid(exact_start),
            pose=exact_start,
        )
    ]
    for point in original.points:
        previous = source_points[-1].pose
        if math.hypot(
            point.pose.x_m - previous.x_m,
            point.pose.y_m - previous.y_m,
        ) <= _EPSILON:
            continue
        source_points.append(
            RoutePoint(
                index=len(source_points),
                cell=point.cell,
                pose=point.pose,
            )
        )
    cumulative_m = 0.0
    rebuilt_points = []
    previous_pose = None
    for index, point in enumerate(source_points):
        segment_m = (
            0.0
            if previous_pose is None
            else math.hypot(
                point.pose.x_m - previous_pose.x_m,
                point.pose.y_m - previous_pose.y_m,
            )
        )
        cumulative_m += segment_m
        rebuilt_points.append(
            replace(
                point,
                index=index,
                segment_length_m=segment_m,
                cumulative_length_m=cumulative_m,
            )
        )
        previous_pose = point.pose

    exact_route = Route(
        points=tuple(rebuilt_points),
        requested_start=exact_start,
        requested_goal=original.requested_goal,
        snapped_start=original.snapped_start,
        snapped_goal=original.snapped_goal,
        length_m=cumulative_m,
    )
    exact_result = PlanRouteResult(
        route=exact_route,
        diagnostics=replace(
            result.diagnostics,
            start_cell=costmap.world_to_grid(exact_start),
            route_length_m=exact_route.length_m,
        ),
        failure=result.failure,
    )
    return smooth_plan_route_result(exact_result, costmap=costmap)


def smooth_plan_route_from_exact_start_with_summary(
    result: PlanRouteResult,
    *,
    costmap: Costmap,
    exact_start: Pose2D,
) -> SmoothedPlanRouteResult:
    """Return exact-start smoothing plus artifact-ready reduction evidence."""

    if result.route is None:
        return SmoothedPlanRouteResult(
            result=result,
            summary=RouteSmoothingSummary(
                enabled=True,
                input_point_count=0,
                output_point_count=0,
                input_length_m=0.0,
                output_length_m=0.0,
                optimized=False,
                skipped_reason="failed_route",
            ),
        )
    original = result.route
    first_distance_m = math.hypot(
        original.points[0].pose.x_m - exact_start.x_m,
        original.points[0].pose.y_m - exact_start.y_m,
    )
    start_was_inserted = first_distance_m > _EPSILON
    input_count = len(original.points) + int(start_was_inserted)
    input_length_m = original.length_m + (
        first_distance_m if start_was_inserted else 0.0
    )
    output = smooth_plan_route_from_exact_start(
        result,
        costmap=costmap,
        exact_start=exact_start,
    )
    assert output.route is not None
    output_count = len(output.route.points)
    return SmoothedPlanRouteResult(
        result=output,
        summary=RouteSmoothingSummary(
            enabled=True,
            input_point_count=input_count,
            output_point_count=output_count,
            input_length_m=input_length_m,
            output_length_m=output.route.length_m,
            optimized=output_count < input_count,
            skipped_reason=(
                "already_minimal" if output_count == input_count else ""
            ),
        ),
    )


def smooth_plan_route_result_with_summary(
    result: PlanRouteResult,
    *,
    costmap: Costmap,
    enabled: bool = True,
    protected_indices: Iterable[int] = (),
) -> SmoothedPlanRouteResult:
    if result.route is None:
        return SmoothedPlanRouteResult(
            result=result,
            summary=RouteSmoothingSummary(
                enabled=enabled,
                input_point_count=0,
                output_point_count=0,
                input_length_m=0.0,
                output_length_m=0.0,
                optimized=False,
                skipped_reason="failed_route",
            ),
        )
    input_count = len(result.route.points)
    input_length = result.route.length_m
    if not enabled:
        output = result
        skipped_reason = "disabled"
    else:
        try:
            output = smooth_plan_route_result(
                result,
                costmap=costmap,
                protected_indices=protected_indices,
            )
        except ValueError as exc:
            return SmoothedPlanRouteResult(
                result=result,
                summary=RouteSmoothingSummary(
                    enabled=True,
                    input_point_count=input_count,
                    output_point_count=input_count,
                    input_length_m=input_length,
                    output_length_m=input_length,
                    optimized=False,
                    skipped_reason=str(exc),
                ),
            )
        assert output.route is not None
        skipped_reason = (
            "already_minimal" if len(output.route.points) == input_count else ""
        )
    assert output.route is not None
    output_count = len(output.route.points)
    return SmoothedPlanRouteResult(
        result=output,
        summary=RouteSmoothingSummary(
            enabled=enabled,
            input_point_count=input_count,
            output_point_count=output_count,
            input_length_m=input_length,
            output_length_m=output.route.length_m,
            optimized=output_count < input_count,
            skipped_reason=skipped_reason,
        ),
    )


def smooth_plan_route_results(
    results: Sequence[PlanRouteResult],
    *,
    costmap: Costmap,
    enabled: bool = True,
) -> tuple[SmoothedPlanRouteResult, ...]:
    smoothed = []
    for result in results:
        smoothed.append(
            smooth_plan_route_result_with_summary(
                result,
                costmap=costmap,
                enabled=enabled,
            )
        )
    return tuple(smoothed)
