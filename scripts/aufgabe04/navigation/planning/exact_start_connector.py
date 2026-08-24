"""Certify and persist the exact-pose join to a grid-snapped A* route.

A* operates on cell centres, while AMCL reports a continuous world pose.  A
route must not silently discard that offset: the first segment is executable
only after its full continuous clearance has been certified against the
uninflated static map and measured arena boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.approach.dynamic_approach_planner import (
    point_clearance_to_blocked_m,
)
from scripts.aufgabe04.navigation.planning.global_planner import PlanRouteResult
from scripts.aufgabe04.navigation.foundation.models import Pose2D, Route, RoutePoint


_EPSILON_M = 1.0e-9


@dataclass(frozen=True)
class ExactStartConnectorEvidence:
    required: bool
    validated: bool
    exact_start: Pose2D
    anchor: Pose2D
    connector_length_m: float
    required_clearance_m: float
    minimum_sampled_clearance_m: float
    minimum_continuous_clearance_m: float
    minimum_margin_m: float
    sample_spacing_m: float
    sample_count: int

    def to_metadata(self) -> dict[str, object]:
        return asdict(self)


def _segment_clearance_evidence(
    base_costmap: Costmap,
    start: Pose2D,
    anchor: Pose2D,
    *,
    required_clearance_m: float,
) -> ExactStartConnectorEvidence:
    if not math.isfinite(required_clearance_m) or required_clearance_m < 0.0:
        raise ValueError(
            "exact-start required clearance must be finite and non-negative"
        )
    values = (start.x_m, start.y_m, start.yaw_rad, anchor.x_m, anchor.y_m)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("exact-start connector poses must be finite")

    length_m = math.hypot(anchor.x_m - start.x_m, anchor.y_m - start.y_m)
    required = length_m > _EPSILON_M
    maximum_spacing_m = min(0.005, base_costmap.resolution / 10.0)
    interval_count = max(1, int(math.ceil(length_m / maximum_spacing_m)))
    actual_spacing_m = length_m / interval_count
    clearances = []
    for index in range(interval_count + 1):
        fraction = index / interval_count
        pose = Pose2D(
            start.x_m + fraction * (anchor.x_m - start.x_m),
            start.y_m + fraction * (anchor.y_m - start.y_m),
            start.yaw_rad,
        )
        clearances.append(point_clearance_to_blocked_m(base_costmap, pose))

    minimum_sampled_m = min(clearances)
    # Distance to a closed obstacle set is 1-Lipschitz.  Subtracting half an
    # interval turns dense point samples into a lower bound for the complete
    # continuous segment, including the points between samples.
    continuous_lower_bound_m = max(
        0.0,
        minimum_sampled_m - actual_spacing_m / 2.0,
    )
    margin_m = continuous_lower_bound_m - required_clearance_m
    validated = margin_m > _EPSILON_M
    evidence = ExactStartConnectorEvidence(
        required=required,
        validated=validated,
        exact_start=start,
        anchor=anchor,
        connector_length_m=length_m,
        required_clearance_m=required_clearance_m,
        minimum_sampled_clearance_m=minimum_sampled_m,
        minimum_continuous_clearance_m=continuous_lower_bound_m,
        minimum_margin_m=margin_m,
        sample_spacing_m=actual_spacing_m,
        sample_count=len(clearances),
    )
    if not validated:
        raise ValueError(
            "exact AMCL-to-A* connector lacks continuous static clearance: "
            f"certified={continuous_lower_bound_m:.6f} m "
            f"required={required_clearance_m:.6f} m"
        )
    return evidence


def prepend_certified_exact_start(
    result: PlanRouteResult,
    *,
    base_costmap: Costmap,
    start: Pose2D,
    required_clearance_m: float,
) -> tuple[PlanRouteResult, ExactStartConnectorEvidence]:
    """Return a route beginning at ``start`` after proving its grid join."""

    if result.route is None or not result.route.points:
        raise ValueError("cannot certify an exact start for a failed or empty route")
    original = result.route
    anchor_index = 0
    first = original.points[0]
    exact_to_first_m = math.hypot(
        first.pose.x_m - start.x_m,
        first.pose.y_m - start.y_m,
    )
    if (
        len(original.points) >= 2
        and exact_to_first_m > _EPSILON_M
        and first.cell == base_costmap.world_to_grid(start)
    ):
        # The exact pose already occupies the A* start cell. Executing a short
        # detour to that cell's centre can create a near reversal before the
        # first real A* edge (152.61 degrees in the physical regression). Skip
        # only this redundant centre and certify the complete metric segment
        # directly to the first neighboring A* vertex. If that segment lacks
        # clearance, _segment_clearance_evidence fails closed.
        anchor_index = 1
    anchor = original.points[anchor_index].pose
    evidence = _segment_clearance_evidence(
        base_costmap,
        start,
        anchor,
        required_clearance_m=required_clearance_m,
    )
    if not evidence.required:
        return result, evidence

    poses_and_cells = [(start, base_costmap.world_to_grid(start))]
    poses_and_cells.extend(
        (point.pose, point.cell)
        for point in original.points[anchor_index:]
    )
    points = []
    cumulative_m = 0.0
    previous_pose = None
    for index, (pose, cell) in enumerate(poses_and_cells):
        segment_m = (
            0.0
            if previous_pose is None
            else math.hypot(
                pose.x_m - previous_pose.x_m,
                pose.y_m - previous_pose.y_m,
            )
        )
        cumulative_m += segment_m
        points.append(
            RoutePoint(
                index=index,
                cell=cell,
                pose=pose,
                segment_length_m=segment_m,
                cumulative_length_m=cumulative_m,
            )
        )
        previous_pose = pose

    route = Route(
        points=tuple(points),
        requested_start=start,
        requested_goal=original.requested_goal,
        snapped_start=original.snapped_start,
        snapped_goal=original.snapped_goal,
        length_m=cumulative_m,
    )
    diagnostics = replace(result.diagnostics, route_length_m=route.length_m)
    return (
        PlanRouteResult(route=route, diagnostics=diagnostics, failure=result.failure),
        evidence,
    )
