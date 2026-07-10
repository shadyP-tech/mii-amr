"""Compatibility wrapper around Aufgabe 04 dry-run route planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.aufgabe04.navigation.costmap import Costmap
from scripts.aufgabe04.navigation.global_planner import PlanRouteResult, plan_route
from scripts.aufgabe04.navigation.map_io import load_occupancy_grid
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.stations.models import StationPose


@dataclass(frozen=True)
class AStarRouteRequest:
    map_yaml: Path
    start: StationPose
    goal: StationPose
    output_prefix: Path | None = None
    inflation_radius_m: float = 0.0
    snap_radius_m: float = 0.30


def plan_astar_route(request: AStarRouteRequest) -> PlanRouteResult:
    grid = load_occupancy_grid(request.map_yaml)
    costmap = Costmap.from_occupancy_grid(grid)
    if request.inflation_radius_m > 0.0:
        costmap = costmap.with_inflation(request.inflation_radius_m)
    return plan_route(
        costmap,
        Pose2D(request.start.x_m, request.start.y_m, request.start.yaw_rad),
        Pose2D(request.goal.x_m, request.goal.y_m, request.goal.yaw_rad),
        snap_radius_m=request.snap_radius_m,
    )

