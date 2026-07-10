"""Adapt existing station visits into navigation route targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from scripts.aufgabe04.navigation.costmap import Costmap
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.stations.models import ApproachTarget, StationVisit


@dataclass(frozen=True)
class NavigationTarget:
    station_id: str
    pose: Pose2D
    source: str = "approach"


def pose_from_approach_target(target: ApproachTarget) -> Pose2D:
    return Pose2D(
        x_m=target.pose.x_m,
        y_m=target.pose.y_m,
        yaw_rad=target.pose.yaw_rad,
    )


def validate_navigation_target(costmap: Costmap, target: NavigationTarget) -> None:
    cell = costmap.world_to_grid(target.pose)
    if not costmap.in_bounds(cell):
        raise ValueError(f"station {target.station_id} target is outside map bounds: {cell}")
    if costmap.is_blocked(cell):
        raise ValueError(f"station {target.station_id} target is blocked: {cell}")


def navigation_targets_from_visits(
    visits: Iterable[StationVisit],
    costmap: Costmap,
) -> Tuple[NavigationTarget, ...]:
    targets = []
    for visit in visits:
        target = NavigationTarget(
            station_id=visit.station_id,
            pose=pose_from_approach_target(visit.target),
        )
        validate_navigation_target(costmap, target)
        targets.append(target)
    if not targets:
        raise ValueError("at least one station visit is required")
    return tuple(targets)

