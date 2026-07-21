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
    segment_is_collision_free,
)
from scripts.aufgabe04.navigation.models import GridCell, Pose2D


_KEEPOUT_EPSILON_M = 1.0e-10


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
    planning_costmap, overlay_diagnostics = _with_non_target_stand_keepouts(
        costmap,
        keepouts,
        start=start,
    )
    result = plan_fixed_approach(
        planning_costmap,
        start,
        target.stand,
        target.face,
        config=target.config,
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
