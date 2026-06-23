"""Adapter boundary around Aufgabe 03 route planning."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from scripts.aufgabe04.stations.models import StationPose


@dataclass(frozen=True)
class AStarRouteRequest:
    map_yaml: Path
    start: StationPose
    goal: StationPose
    output_prefix: Path


def build_map_path_planner_args(request: AStarRouteRequest) -> Tuple[str, ...]:
    return (
        "--map",
        str(request.map_yaml),
        "--start",
        str(request.start.x_m),
        str(request.start.y_m),
        "--goal",
        str(request.goal.x_m),
        str(request.goal.y_m),
        "--output-prefix",
        str(request.output_prefix),
    )

