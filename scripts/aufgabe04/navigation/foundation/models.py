"""Pure planning models for Aufgabe 04 dry-run navigation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Pose2D:
    x_m: float
    y_m: float
    yaw_rad: float = 0.0


@dataclass(frozen=True, order=True)
class GridCell:
    x: int
    y: int


@dataclass(frozen=True)
class RoutePoint:
    index: int
    cell: GridCell
    pose: Pose2D
    segment_length_m: float = 0.0
    cumulative_length_m: float = 0.0


@dataclass(frozen=True)
class Route:
    points: Tuple[RoutePoint, ...]
    requested_start: Pose2D
    requested_goal: Pose2D
    snapped_start: Pose2D
    snapped_goal: Pose2D
    length_m: float


@dataclass(frozen=True)
class PlanningDiagnostics:
    status: str
    reason: str = ""
    start_cell: GridCell | None = None
    goal_cell: GridCell | None = None
    snapped_start_cell: GridCell | None = None
    snapped_goal_cell: GridCell | None = None
    expanded_cells: int = 0
    path_cell_count: int = 0
    route_length_m: float = 0.0


@dataclass(frozen=True)
class PlanningFailure:
    reason: str
    diagnostics: PlanningDiagnostics

