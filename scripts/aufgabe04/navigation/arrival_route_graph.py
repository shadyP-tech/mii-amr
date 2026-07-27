"""Collision-checked directed route graph for catalogued stand arrivals.

This layer deliberately separates path computation from route-order
optimization.  Every edge is planned to one exact, already-selected arrival
face.  A failed A* or terminal-corridor validation is represented as an
explicitly unreachable edge; no opposite face or snapped goal is substituted.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

from scripts.aufgabe04.navigation.costmap import CELL_SOURCE_RUN_LOCAL, Costmap
from scripts.aufgabe04.navigation.dynamic_approach_planner import (
    DynamicApproachConfig,
    DynamicApproachPlanResult,
    FaceNormalCandidate,
    circular_keepout_cells,
    plan_fixed_approach,
    point_clearance_to_blocked_m,
    segment_is_collision_free,
    supercover_segment_cells,
)
from scripts.aufgabe04.navigation.models import GridCell, Pose2D


_KEEPOUT_EPSILON_M = 1.0e-10
_EGRESS_SEARCH_RADIUS_CELLS = 4


def _as_navigation_pose(value: object, *, name: str) -> Pose2D:
    """Normalize a structurally compatible catalog/geometry pose once.

    ``arrival_pose_geometry`` deliberately owns a ROS-free catalog pose model,
    while the costmap uses ``isinstance(Pose2D)`` to distinguish a pose from a
    scalar x coordinate.  Letting the catalog shape pass through unchanged
    therefore fails later with a misleading ``y_m is required`` error.  The
    graph boundary is the narrowest place to establish the navigation type.
    """

    try:
        x_m = float(getattr(value, "x_m"))
        y_m = float(getattr(value, "y_m"))
        yaw_rad = float(getattr(value, "yaw_rad"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must provide numeric x_m, y_m, and yaw_rad") from exc
    if not all(math.isfinite(component) for component in (x_m, y_m, yaw_rad)):
        raise ValueError(f"{name} must be finite")
    return Pose2D(x_m, y_m, yaw_rad)


@dataclass(frozen=True)
class ArrivalRouteNode:
    station_id: str
    arrival_id: str
    stand: Pose2D
    face: FaceNormalCandidate
    config: DynamicApproachConfig

    def __post_init__(self) -> None:
        station_id = str(self.station_id).strip()
        arrival_id = str(self.arrival_id).strip()
        if not station_id:
            raise ValueError("station_id must be non-empty")
        if not arrival_id:
            raise ValueError("arrival_id must be non-empty")
        try:
            face_id = self.face.face_id
            normal_rad = float(self.face.normal_rad)
            target = _as_navigation_pose(
                self.face.target, name="arrival face target"
            )
            entry = _as_navigation_pose(
                self.face.entry, name="arrival face corridor entry"
            )
        except AttributeError as exc:
            raise ValueError(
                "face must provide face_id, normal_rad, target, and entry"
            ) from exc
        if not math.isfinite(normal_rad):
            raise ValueError("arrival face normal_rad must be finite")

        object.__setattr__(self, "station_id", station_id)
        object.__setattr__(self, "arrival_id", arrival_id)
        object.__setattr__(
            self,
            "stand",
            _as_navigation_pose(self.stand, name="stand pose"),
        )
        object.__setattr__(
            self,
            "face",
            FaceNormalCandidate(
                face_id=face_id,
                normal_rad=normal_rad,
                target=target,
                entry=entry,
            ),
        )


@dataclass(frozen=True)
class ArrivalRouteEdge:
    source_id: str
    target_id: str
    result: DynamicApproachPlanResult
    non_target_clearances: tuple["NonTargetStandClearance", ...] = ()
    non_target_overlay: "NonTargetStandOverlayDiagnostics | None" = None

    @property
    def cost_m(self) -> float | None:
        return None if self.result.plan is None else self.result.plan.length_m


@dataclass(frozen=True)
class ArrivalRouteGraph:
    start_id: str
    nodes: Mapping[str, ArrivalRouteNode]
    edges: Mapping[tuple[str, str], ArrivalRouteEdge]

    @property
    def directed_costs(self) -> dict[tuple[str, str], float | None]:
        return {edge_id: edge.cost_m for edge_id, edge in self.edges.items()}


@dataclass(frozen=True)
class NonTargetStandKeepout:
    station_id: str
    stand: Pose2D
    radius_m: float


@dataclass(frozen=True)
class NonTargetStandClearance:
    station_id: str
    x_m: float
    y_m: float
    radius_m: float
    minimum_route_clearance_m: float


@dataclass(frozen=True)
class NonTargetStandOverlayDiagnostics:
    """Evidence for the run-local non-target keepout raster.

    A closed metric keepout can overlap the square containing a safe exact
    arrival pose even when both that pose and the cell centre are outside the
    keepout.  ``start_cell_exempted`` records the narrowly scoped correction
    for that rasterization artefact.  The static/inflated base map is never
    modified by the exemption.
    """

    rasterized_cell_count: int
    blocked_cell_count: int
    start_cell: GridCell
    start_cell_was_rasterized: bool
    start_cell_exempted: bool
    exact_start_minimum_margin_m: float | None
    cell_center_minimum_margin_m: float | None
    start_connector_minimum_margin_m: float | None
    egress_anchor: Pose2D | None
    egress_anchor_cell: GridCell | None
    egress_cells: tuple[GridCell, ...]
    egress_connector_minimum_margin_m: float | None
    egress_continuous_clearance_validated: bool
    egress_failure_reason: str | None


def _non_target_stand_keepouts(
    nodes: Sequence[ArrivalRouteNode],
    *,
    target_station_id: str,
) -> tuple[NonTargetStandKeepout, ...]:
    radii: dict[tuple[str, float, float], float] = {}
    for node in nodes:
        if node.station_id == target_station_id:
            continue
        key = (node.station_id, node.stand.x_m, node.stand.y_m)
        radii[key] = max(
            radii.get(key, 0.0),
            node.config.non_target_stand_keepout_radius_m,
        )
    return tuple(
        NonTargetStandKeepout(
            station_id=station_id,
            stand=Pose2D(x_m, y_m),
            radius_m=radius_m,
        )
        for (station_id, x_m, y_m), radius_m in sorted(radii.items())
    )


def _with_non_target_stand_keepouts(
    costmap: Costmap,
    keepouts: Sequence[NonTargetStandKeepout],
    *,
    start: Pose2D,
    connector_keepouts: Sequence[NonTargetStandKeepout] = (),
) -> tuple[Costmap, NonTargetStandOverlayDiagnostics]:
    cells = set()
    for keepout in keepouts:
        cells.update(
            circular_keepout_cells(
                costmap,
                keepout.stand,
                keepout.radius_m,
            )
        )
    rasterized_cell_count = len(cells)
    start_cell = costmap.world_to_grid(start)
    start_cell_was_rasterized = start_cell in cells
    start_cell_exempted = False

    exact_margin = _minimum_point_keepout_margin_m(start, keepouts)
    cell_center_margin = None
    connector_margin = None
    egress_anchor = None
    egress_anchor_cell = None
    egress_cells: tuple[GridCell, ...] = ()
    egress_connector_margin = None
    egress_continuous_clearance_validated = False
    egress_failure_reason = None
    if start_cell_was_rasterized and costmap.in_bounds(start_cell):
        cell_center = costmap.grid_to_world(start_cell)
        cell_center_margin = _minimum_point_keepout_margin_m(
            cell_center,
            keepouts,
        )
        connector_margin = _minimum_segment_keepout_margin_m(
            start,
            cell_center,
            keepouts,
        )

        # The exact pose, the A* start vertex, and their connector must all be
        # physically outside every non-target stand.  Requiring the connector
        # here avoids reproducing the unsafe case where a nominally safe exact
        # pose is joined to a cell centre through the exclusion disk.  Static
        # occupancy/inflation independently vetoes the exemption.
        if (
            costmap.is_traversable(start_cell)
            and segment_is_collision_free(costmap, start, cell_center)
            and exact_margin is not None
            and exact_margin > _KEEPOUT_EPSILON_M
            and cell_center_margin is not None
            and cell_center_margin > _KEEPOUT_EPSILON_M
            and connector_margin is not None
            and connector_margin > _KEEPOUT_EPSILON_M
        ):
            cells.remove(start_cell)
            start_cell_exempted = True
        elif (
            costmap.is_traversable(start_cell)
            and exact_margin is not None
            and exact_margin > _KEEPOUT_EPSILON_M
        ):
            anchor_result = _find_safe_egress_anchor(
                costmap,
                cells,
                (*keepouts, *connector_keepouts),
                start,
            )
            if anchor_result is None:
                egress_failure_reason = "start_egress_no_safe_anchor"
            else:
                (
                    egress_anchor,
                    egress_anchor_cell,
                    egress_cells,
                    egress_connector_margin,
                ) = anchor_result
                egress_continuous_clearance_validated = True
        elif exact_margin is None or exact_margin <= _KEEPOUT_EPSILON_M:
            egress_failure_reason = "start_egress_exact_pose_inside_keepout"
        else:
            egress_failure_reason = "start_egress_static_start_blocked"

    overlay = costmap.with_blocked_cells(
        cells,
        source=CELL_SOURCE_RUN_LOCAL,
    )
    return overlay, NonTargetStandOverlayDiagnostics(
        rasterized_cell_count=rasterized_cell_count,
        blocked_cell_count=len(cells),
        start_cell=start_cell,
        start_cell_was_rasterized=start_cell_was_rasterized,
        start_cell_exempted=start_cell_exempted,
        exact_start_minimum_margin_m=exact_margin,
        cell_center_minimum_margin_m=cell_center_margin,
        start_connector_minimum_margin_m=connector_margin,
        egress_anchor=egress_anchor,
        egress_anchor_cell=egress_anchor_cell,
        egress_cells=egress_cells,
        egress_connector_minimum_margin_m=egress_connector_margin,
        egress_continuous_clearance_validated=(
            egress_continuous_clearance_validated
        ),
        egress_failure_reason=egress_failure_reason,
    )


def _point_to_segment_distance_m(
    point: Pose2D,
    start: Pose2D,
    end: Pose2D,
) -> float:
    dx = end.x_m - start.x_m
    dy = end.y_m - start.y_m
    denominator = dx * dx + dy * dy
    if denominator <= 1.0e-20:
        return math.hypot(point.x_m - start.x_m, point.y_m - start.y_m)
    fraction = max(
        0.0,
        min(
            1.0,
            (
                (point.x_m - start.x_m) * dx
                + (point.y_m - start.y_m) * dy
            )
            / denominator,
        ),
    )
    return math.hypot(
        point.x_m - (start.x_m + fraction * dx),
        point.y_m - (start.y_m + fraction * dy),
    )


def _minimum_point_keepout_margin_m(
    point: Pose2D,
    keepouts: Sequence[NonTargetStandKeepout],
) -> float | None:
    if not keepouts:
        return None
    return min(
        math.hypot(point.x_m - keepout.stand.x_m, point.y_m - keepout.stand.y_m)
        - keepout.radius_m
        for keepout in keepouts
    )


def _minimum_segment_keepout_margin_m(
    start: Pose2D,
    end: Pose2D,
    keepouts: Sequence[NonTargetStandKeepout],
) -> float | None:
    if not keepouts:
        return None
    return min(
        _point_to_segment_distance_m(keepout.stand, start, end)
        - keepout.radius_m
        for keepout in keepouts
    )


def _find_safe_egress_anchor(
    costmap: Costmap,
    rasterized_cells: set[GridCell],
    keepouts: Sequence[NonTargetStandKeepout],
    start: Pose2D,
) -> tuple[Pose2D, GridCell, tuple[GridCell, ...], float] | None:
    """Find a nearby exterior cell joined by a continuously safe segment.

    The exact source pose may be physically outside a stand disk while its
    containing grid-cell centre lies inside the conservative closed-square
    raster. In that case planning must begin at an exterior anchor; the raster
    itself remains untouched.
    """

    start_cell = costmap.world_to_grid(start)
    candidates = []
    radius = _EGRESS_SEARCH_RADIUS_CELLS
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            cell = GridCell(start_cell.x + dx, start_cell.y + dy)
            if cell in rasterized_cells or not costmap.is_traversable(cell):
                continue
            anchor = costmap.grid_to_world(cell)
            distance_m = math.hypot(
                anchor.x_m - start.x_m,
                anchor.y_m - start.y_m,
            )
            if distance_m <= _KEEPOUT_EPSILON_M:
                continue
            candidates.append((distance_m, abs(dx) + abs(dy), cell, anchor))
    for _distance_m, _manhattan, cell, anchor in sorted(candidates):
        if not segment_is_collision_free(costmap, start, anchor):
            continue
        connector_margin = _minimum_segment_keepout_margin_m(
            start,
            anchor,
            keepouts,
        )
        if (
            connector_margin is None
            or connector_margin <= _KEEPOUT_EPSILON_M
        ):
            continue
        return (
            anchor,
            cell,
            supercover_segment_cells(costmap, start, anchor),
            connector_margin,
        )
    return None


def _prepend_certified_non_target_egress(
    result: DynamicApproachPlanResult,
    *,
    source_start: Pose2D,
    target: ArrivalRouteNode,
    overlay: NonTargetStandOverlayDiagnostics,
    static_costmap: Costmap,
) -> DynamicApproachPlanResult:
    """Restore the exact source before a plan made from its exterior anchor."""

    anchor = overlay.egress_anchor
    if anchor is None or result.plan is None:
        return result
    if not overlay.egress_continuous_clearance_validated:
        raise ValueError("source egress anchor lacks continuous validation")
    if not result.plan.waypoints:
        raise ValueError("source egress plan has no anchor waypoint")
    first = result.plan.waypoints[0].pose
    if math.hypot(first.x_m - anchor.x_m, first.y_m - anchor.y_m) > 1.0e-8:
        raise ValueError("source egress plan lost its certified anchor")
    target_connector_margin = (
        _point_to_segment_distance_m(
            target.stand,
            source_start,
            anchor,
        )
        - target.config.stand_keepout_radius_m
    )
    if target_connector_margin <= _KEEPOUT_EPSILON_M:
        diagnostics = replace(
            result.diagnostics,
            failure_reason="start_egress_intersects_target_keepout",
        )
        return DynamicApproachPlanResult(None, diagnostics)
    static_clearance = point_clearance_to_blocked_m(
        static_costmap,
        source_start,
    )
    connector_margin = overlay.egress_connector_minimum_margin_m
    if (
        static_clearance <= _KEEPOUT_EPSILON_M
        or connector_margin is None
        or connector_margin <= _KEEPOUT_EPSILON_M
    ):
        diagnostics = replace(
            result.diagnostics,
            failure_reason="start_egress_lacks_positive_clearance",
        )
        return DynamicApproachPlanResult(None, diagnostics)
    start_join_clearance = min(
        static_clearance,
        connector_margin,
        target_connector_margin,
    )
    diagnostics = replace(
        result.diagnostics,
        start_join_clearance_m=start_join_clearance,
    )
    exact_start = replace(
        result.plan.waypoints[0],
        pose=Pose2D(source_start.x_m, source_start.y_m, math.nan),
        protected=False,
        corridor=False,
    )
    egress_length = math.hypot(
        anchor.x_m - source_start.x_m,
        anchor.y_m - source_start.y_m,
    )
    plan = replace(
        result.plan,
        waypoints=(exact_start, *result.plan.waypoints),
        length_m=result.plan.length_m + egress_length,
        diagnostics=diagnostics,
    )
    return DynamicApproachPlanResult(plan, diagnostics)


def _continuous_non_target_clearances(
    plan,
    keepouts: Sequence[NonTargetStandKeepout],
) -> tuple[NonTargetStandClearance, ...]:
    poses = tuple(waypoint.pose for waypoint in plan.waypoints)
    if not poses:
        raise ValueError("arrival route has no waypoints")
    clearances = []
    for keepout in keepouts:
        if len(poses) == 1:
            minimum_m = math.hypot(
                poses[0].x_m - keepout.stand.x_m,
                poses[0].y_m - keepout.stand.y_m,
            )
        else:
            minimum_m = min(
                _point_to_segment_distance_m(
                    keepout.stand,
                    segment_start,
                    segment_end,
                )
                for segment_start, segment_end in zip(poses, poses[1:])
            )
        if minimum_m <= keepout.radius_m + _KEEPOUT_EPSILON_M:
            raise ValueError(
                "non_target_stand_route_clearance_failed:"
                f"station_id={keepout.station_id}:"
                f"clearance_m={minimum_m:.9f}:"
                f"radius_m={keepout.radius_m:.9f}"
            )
        clearances.append(
            NonTargetStandClearance(
                station_id=keepout.station_id,
                x_m=keepout.stand.x_m,
                y_m=keepout.stand.y_m,
                radius_m=keepout.radius_m,
                minimum_route_clearance_m=minimum_m,
            )
        )
    return tuple(clearances)


def _plan_arrival_edge(
    costmap: Costmap,
    start: Pose2D,
    target: ArrivalRouteNode,
    nodes: Sequence[ArrivalRouteNode],
    *,
    source_id: str,
) -> ArrivalRouteEdge:
    keepouts = _non_target_stand_keepouts(
        nodes,
        target_station_id=target.station_id,
    )
    target_keepout = NonTargetStandKeepout(
        station_id=target.station_id,
        stand=target.stand,
        radius_m=target.config.stand_keepout_radius_m,
    )
    planning_costmap, overlay_diagnostics = _with_non_target_stand_keepouts(
        costmap,
        keepouts,
        start=start,
        connector_keepouts=(target_keepout,),
    )
    planning_start = overlay_diagnostics.egress_anchor or start
    result = plan_fixed_approach(
        planning_costmap,
        planning_start,
        target.stand,
        target.face,
        config=target.config,
    )
    if (
        overlay_diagnostics.start_cell_was_rasterized
        and not overlay_diagnostics.start_cell_exempted
        and overlay_diagnostics.egress_anchor is None
        and overlay_diagnostics.egress_failure_reason is not None
    ):
        result = DynamicApproachPlanResult(
            None,
            replace(
                result.diagnostics,
                failure_reason=overlay_diagnostics.egress_failure_reason,
            ),
        )
    result = _prepend_certified_non_target_egress(
        result,
        source_start=start,
        target=target,
        overlay=overlay_diagnostics,
        static_costmap=costmap,
    )
    clearances: tuple[NonTargetStandClearance, ...] = ()
    if result.plan is not None:
        try:
            clearances = _continuous_non_target_clearances(
                result.plan,
                keepouts,
            )
        except ValueError as exc:
            result = DynamicApproachPlanResult(
                None,
                replace(result.diagnostics, failure_reason=str(exc)),
            )
    return ArrivalRouteEdge(
        source_id=source_id,
        target_id=target.arrival_id,
        result=result,
        non_target_clearances=clearances,
        non_target_overlay=overlay_diagnostics,
    )


def build_arrival_route_graph(
    costmap: Costmap,
    start: Pose2D,
    nodes: Sequence[ArrivalRouteNode],
    *,
    start_id: str = "mission_start",
) -> ArrivalRouteGraph:
    """Plan every directed transition needed by the exact route optimizer."""

    start_id = str(start_id).strip()
    if not start_id:
        raise ValueError("start_id must be non-empty")
    node_list = tuple(nodes)
    if not node_list:
        raise ValueError("at least one arrival route node is required")
    by_id: dict[str, ArrivalRouteNode] = {}
    for node in node_list:
        if node.arrival_id == start_id:
            raise ValueError("arrival_id conflicts with start_id")
        if node.arrival_id in by_id:
            raise ValueError(f"duplicate arrival_id: {node.arrival_id}")
        by_id[node.arrival_id] = node

    edges: dict[tuple[str, str], ArrivalRouteEdge] = {}
    for target in node_list:
        edge = _plan_arrival_edge(
            costmap,
            start,
            target,
            node_list,
            source_id=start_id,
        )
        edge_id = (start_id, target.arrival_id)
        edges[edge_id] = edge

    for source in node_list:
        for target in node_list:
            if source.station_id == target.station_id:
                continue
            edge = _plan_arrival_edge(
                costmap,
                source.face.target,
                target,
                node_list,
                source_id=source.arrival_id,
            )
            edge_id = (source.arrival_id, target.arrival_id)
            edges[edge_id] = edge

    return ArrivalRouteGraph(start_id=start_id, nodes=by_id, edges=edges)


def resolve_station_arrival_order(
    nodes: Sequence[ArrivalRouteNode],
    station_order: Sequence[str],
) -> tuple[str, ...]:
    """Resolve semantic task station IDs to one unambiguous frozen arrival."""

    normalized_order = tuple(str(station_id).strip() for station_id in station_order)
    if not normalized_order or any(not station_id for station_id in normalized_order):
        raise ValueError("station_order must contain non-empty station IDs")
    by_station: dict[str, list[ArrivalRouteNode]] = {}
    for node in nodes:
        by_station.setdefault(node.station_id, []).append(node)
    arrivals: list[str] = []
    for station_id in normalized_order:
        matches = by_station.get(station_id, [])
        if not matches:
            raise ValueError(f"task station has no frozen arrival: {station_id}")
        if len(matches) != 1:
            arrival_ids = sorted(node.arrival_id for node in matches)
            raise ValueError(
                "task station has ambiguous frozen arrivals: "
                f"{station_id}: {arrival_ids}"
            )
        arrivals.append(matches[0].arrival_id)
    return tuple(arrivals)


def build_required_arrival_route_graph(
    costmap: Costmap,
    start: Pose2D,
    nodes: Sequence[ArrivalRouteNode],
    arrival_order: Sequence[str],
    *,
    start_id: str = "mission_start",
) -> ArrivalRouteGraph:
    """Plan only start->first and consecutive edges for an immutable task order.

    This is the logistics counterpart to :func:`build_arrival_route_graph`.
    Survey exploration may compute all pairs and optimize their order; mission
    execution must preserve the QR/task-server order and therefore has no
    reason to plan or expose shortcut edges.
    """

    start_id = str(start_id).strip()
    if not start_id:
        raise ValueError("start_id must be non-empty")
    node_list = tuple(nodes)
    if not node_list:
        raise ValueError("at least one arrival route node is required")
    by_id: dict[str, ArrivalRouteNode] = {}
    for node in node_list:
        if node.arrival_id == start_id:
            raise ValueError("arrival_id conflicts with start_id")
        if node.arrival_id in by_id:
            raise ValueError(f"duplicate arrival_id: {node.arrival_id}")
        by_id[node.arrival_id] = node

    required = tuple(str(arrival_id).strip() for arrival_id in arrival_order)
    if not required or any(not arrival_id for arrival_id in required):
        raise ValueError("arrival_order must contain non-empty arrival IDs")
    missing = [arrival_id for arrival_id in required if arrival_id not in by_id]
    if missing:
        raise ValueError(f"arrival_order contains unknown arrivals: {missing}")

    edges: dict[tuple[str, str], ArrivalRouteEdge] = {}
    source_id = start_id
    source_pose = start
    for target_id in required:
        target = by_id[target_id]
        edge = _plan_arrival_edge(
            costmap,
            source_pose,
            target,
            node_list,
            source_id=source_id,
        )
        edges[(source_id, target_id)] = edge
        source_id = target_id
        source_pose = target.face.target
    return ArrivalRouteGraph(start_id=start_id, nodes=by_id, edges=edges)


def selected_edges(
    graph: ArrivalRouteGraph,
    arrival_order: Sequence[str],
) -> tuple[ArrivalRouteEdge, ...]:
    """Resolve an optimized arrival order to its prevalidated route legs."""

    source_id = graph.start_id
    selected = []
    for target_id in arrival_order:
        edge_id = (source_id, target_id)
        try:
            edge = graph.edges[edge_id]
        except KeyError as exc:
            raise ValueError(f"route graph is missing selected edge {edge_id}") from exc
        if edge.result.plan is None:
            raise ValueError(f"selected edge is unreachable: {source_id}->{target_id}")
        selected.append(edge)
        source_id = target_id
    return tuple(selected)
