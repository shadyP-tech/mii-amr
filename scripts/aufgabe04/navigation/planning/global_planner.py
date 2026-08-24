"""Deterministic A* route planning for Aufgabe 04 dry-run navigation."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.foundation.models import (
    GridCell,
    PlanningDiagnostics,
    PlanningFailure,
    Pose2D,
    Route,
    RoutePoint,
)


STATUS_OK = "ok"
STATUS_FAILED = "failed"
FAILURE_START_BLOCKED = "start_blocked"
FAILURE_GOAL_BLOCKED = "goal_blocked"
FAILURE_START_SNAP_FAILED = "start_snap_failed"
FAILURE_GOAL_SNAP_FAILED = "goal_snap_failed"
FAILURE_NO_PATH = "no_path"


@dataclass(frozen=True)
class PlanRouteResult:
    route: Route | None
    diagnostics: PlanningDiagnostics
    failure: PlanningFailure | None = None


def movement_cost(a: GridCell, b: GridCell, resolution: float) -> float:
    return math.hypot(b.x - a.x, b.y - a.y) * resolution


def _neighbors_8_no_corner_cutting(costmap: Costmap, cell: GridCell) -> Iterable[GridCell]:
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbor = GridCell(cell.x + dx, cell.y + dy)
            if not costmap.is_traversable(neighbor):
                continue
            if dx != 0 and dy != 0:
                side_a = GridCell(cell.x + dx, cell.y)
                side_b = GridCell(cell.x, cell.y + dy)
                if not costmap.is_traversable(side_a) or not costmap.is_traversable(side_b):
                    continue
            yield neighbor


def _snap_to_traversable(costmap: Costmap, requested: GridCell, snap_radius_m: float) -> GridCell | None:
    snap_cells = int(math.ceil(max(0.0, snap_radius_m) / costmap.resolution))
    best = None
    best_distance_sq = None
    for dy in range(-snap_cells, snap_cells + 1):
        for dx in range(-snap_cells, snap_cells + 1):
            if dx * dx + dy * dy > snap_cells * snap_cells:
                continue
            cell = GridCell(requested.x + dx, requested.y + dy)
            if not costmap.is_traversable(cell):
                continue
            distance_sq = dx * dx + dy * dy
            if best is None or distance_sq < best_distance_sq or (
                distance_sq == best_distance_sq and cell < best
            ):
                best = cell
                best_distance_sq = distance_sq
    return best


def _reconstruct_path(came_from: Dict[GridCell, GridCell], current: GridCell) -> List[GridCell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _astar(costmap: Costmap, start: GridCell, goal: GridCell) -> Tuple[List[GridCell] | None, int]:
    queue: List[Tuple[float, int, GridCell]] = []
    heapq.heappush(queue, (0.0, 0, start))
    came_from: Dict[GridCell, GridCell] = {}
    g_score = {start: 0.0}
    tie_breaker = 0
    expanded = 0

    while queue:
        _priority, _tie, current = heapq.heappop(queue)
        expanded += 1
        if current == goal:
            return _reconstruct_path(came_from, current), expanded
        for neighbor in _neighbors_8_no_corner_cutting(costmap, current):
            tentative_g = g_score[current] + movement_cost(current, neighbor, costmap.resolution)
            if tentative_g >= g_score.get(neighbor, math.inf):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g
            heuristic = movement_cost(neighbor, goal, costmap.resolution)
            tie_breaker += 1
            heapq.heappush(queue, (tentative_g + heuristic, tie_breaker, neighbor))
    return None, expanded


def _route_from_cells(
    costmap: Costmap,
    cells: List[GridCell],
    requested_start: Pose2D,
    requested_goal: Pose2D,
) -> Route:
    points: List[RoutePoint] = []
    cumulative = 0.0
    previous = None
    for index, cell in enumerate(cells):
        if previous is None:
            segment = 0.0
        else:
            segment = movement_cost(previous, cell, costmap.resolution)
            cumulative += segment
        points.append(
            RoutePoint(
                index=index,
                cell=cell,
                pose=costmap.grid_to_world(cell),
                segment_length_m=segment,
                cumulative_length_m=cumulative,
            )
        )
        previous = cell
    return Route(
        points=tuple(points),
        requested_start=requested_start,
        requested_goal=requested_goal,
        snapped_start=points[0].pose,
        snapped_goal=points[-1].pose,
        length_m=cumulative,
    )


def _failure(
    reason: str,
    start_cell: GridCell,
    goal_cell: GridCell,
    snapped_start_cell: GridCell | None = None,
    snapped_goal_cell: GridCell | None = None,
    expanded_cells: int = 0,
) -> PlanRouteResult:
    diagnostics = PlanningDiagnostics(
        status=STATUS_FAILED,
        reason=reason,
        start_cell=start_cell,
        goal_cell=goal_cell,
        snapped_start_cell=snapped_start_cell,
        snapped_goal_cell=snapped_goal_cell,
        expanded_cells=expanded_cells,
    )
    return PlanRouteResult(
        route=None,
        diagnostics=diagnostics,
        failure=PlanningFailure(reason=reason, diagnostics=diagnostics),
    )


def plan_route(
    costmap: Costmap,
    start: Pose2D,
    goal: Pose2D,
    snap_radius_m: float = 0.30,
) -> PlanRouteResult:
    start_cell = costmap.world_to_grid(start)
    goal_cell = costmap.world_to_grid(goal)

    if not costmap.in_bounds(start_cell):
        return _failure(FAILURE_START_BLOCKED, start_cell, goal_cell)
    if not costmap.in_bounds(goal_cell):
        return _failure(FAILURE_GOAL_BLOCKED, start_cell, goal_cell)

    snapped_start = _snap_to_traversable(costmap, start_cell, snap_radius_m)
    if snapped_start is None:
        return _failure(FAILURE_START_SNAP_FAILED, start_cell, goal_cell)
    snapped_goal = _snap_to_traversable(costmap, goal_cell, snap_radius_m)
    if snapped_goal is None:
        return _failure(
            FAILURE_GOAL_SNAP_FAILED,
            start_cell,
            goal_cell,
            snapped_start_cell=snapped_start,
        )

    path, expanded = _astar(costmap, snapped_start, snapped_goal)
    if path is None:
        return _failure(
            FAILURE_NO_PATH,
            start_cell,
            goal_cell,
            snapped_start_cell=snapped_start,
            snapped_goal_cell=snapped_goal,
            expanded_cells=expanded,
        )

    route = _route_from_cells(costmap, path, requested_start=start, requested_goal=goal)
    diagnostics = PlanningDiagnostics(
        status=STATUS_OK,
        start_cell=start_cell,
        goal_cell=goal_cell,
        snapped_start_cell=snapped_start,
        snapped_goal_cell=snapped_goal,
        expanded_cells=expanded,
        path_cell_count=len(path),
        route_length_m=route.length_m,
    )
    return PlanRouteResult(route=route, diagnostics=diagnostics)

