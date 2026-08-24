"""Build a smoothed route without discarding its exact metric start pose.

Grid planning snaps a requested start to a cell centre.  This pure helper
combines the already-existing continuous connector proof with line-of-sight
smoothing so every caller uses the same conservative treatment of a start
that lies in static inflation but not in a live obstacle overlay.
"""

from __future__ import annotations

from scripts.aufgabe04.navigation.planning.costmap import CELL_SOURCE_INFLATED, Costmap
from scripts.aufgabe04.navigation.planning.exact_start_connector import (
    ExactStartConnectorEvidence,
    prepend_certified_exact_start,
)
from scripts.aufgabe04.navigation.planning.global_planner import PlanRouteResult
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.planning.route_smoothing import (
    RouteSmoothingSummary,
    segment_is_collision_free,
    smooth_plan_route_after_certified_prefix,
    smooth_plan_route_from_exact_start_with_summary,
    supercover_segment_cells,
)


def certify_and_smooth_exact_start_route(
    result: PlanRouteResult,
    *,
    base_costmap: Costmap,
    planning_costmap: Costmap,
    exact_start: Pose2D,
    required_clearance_m: float,
) -> tuple[
    PlanRouteResult,
    ExactStartConnectorEvidence,
    RouteSmoothingSummary,
]:
    """Return a route beginning at ``exact_start`` plus its safety evidence.

    ``base_costmap`` supplies the continuous physical-clearance proof;
    ``planning_costmap`` must be the exact inflated/keepout map used by A*.
    A start inside static inflation may use the separately certified connector,
    but no live overlay (for example a candidate keepout) is exempted.
    """

    smoothed = None
    if segment_is_collision_free(planning_costmap, exact_start, exact_start):
        try:
            smoothed = smooth_plan_route_from_exact_start_with_summary(
                result,
                costmap=planning_costmap,
                exact_start=exact_start,
            )
        except ValueError:
            # A conservative grid boundary can reject the metric first chord.
            # The fallback retains the separately sampled connector.
            smoothed = None
    if smoothed is not None:
        route_result, connector = prepend_certified_exact_start(
            smoothed.result,
            base_costmap=base_costmap,
            start=exact_start,
            required_clearance_m=required_clearance_m,
        )
        return route_result, connector, smoothed.summary

    joined, connector = prepend_certified_exact_start(
        result,
        base_costmap=base_costmap,
        start=exact_start,
        required_clearance_m=required_clearance_m,
    )
    if not connector.required or joined.route is None:
        raise ValueError("nontraversable exact start lacks a certified connector")
    connector_cells = supercover_segment_cells(
        planning_costmap,
        joined.route.points[0].pose,
        joined.route.points[1].pose,
    )
    unsupported_sources = sorted(
        {
            str(planning_costmap.cell_sources.get(cell, "out_of_bounds"))
            for cell in connector_cells
            if not planning_costmap.is_traversable(cell)
            and planning_costmap.cell_sources.get(cell) != CELL_SOURCE_INFLATED
        }
    )
    if unsupported_sources:
        raise ValueError(
            "exact-start connector intersects a live planning overlay: "
            + ", ".join(unsupported_sources)
        )
    input_count = len(joined.route.points)
    input_length_m = joined.route.length_m
    route_result = smooth_plan_route_after_certified_prefix(
        joined,
        costmap=planning_costmap,
        prefix_end_index=1,
    )
    assert route_result.route is not None
    output_count = len(route_result.route.points)
    return (
        route_result,
        connector,
        RouteSmoothingSummary(
            enabled=True,
            input_point_count=input_count,
            output_point_count=output_count,
            input_length_m=input_length_m,
            output_length_m=route_result.route.length_m,
            optimized=output_count < input_count,
            skipped_reason="exact_start_prefix_preserved",
        ),
    )
