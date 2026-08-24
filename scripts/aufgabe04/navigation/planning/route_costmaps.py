"""Reusable immutable costmaps for station and candidate route planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.map_io import OccupancyGrid
from scripts.aufgabe04.stations.models import Station


@dataclass(frozen=True)
class StationRouteCostmaps:
    """Static, planning, and target maps shared by route previews."""

    base_costmap: Costmap
    planning_costmap: Costmap
    target_costmap: Costmap


def build_station_route_costmaps(
    grid: OccupancyGrid,
    *,
    station_map: Mapping[str, Station],
    inflation_radius_m: float,
    transit_keepout_radius_m: float,
    arena_bounds: ArenaBounds,
) -> StationRouteCostmaps:
    """Build common maps once for routes sharing station geometry."""

    arena_bounds.validate()
    base_costmap = Costmap.from_occupancy_grid(grid).with_arena_bounds(
        arena_bounds
    )
    planning_costmap = base_costmap
    if inflation_radius_m > 0.0:
        planning_costmap = planning_costmap.with_inflation(
            inflation_radius_m
        )
    if transit_keepout_radius_m > 0.0:
        transit_keepouts = tuple(
            Station(
                station.station_id,
                station.pose,
                station.approach_offset_m,
                transit_keepout_radius_m,
            )
            for station in station_map.values()
        )
        planning_costmap = planning_costmap.with_station_keepouts(
            transit_keepouts
        )
    target_costmap = (
        base_costmap.with_inflation(inflation_radius_m)
        if inflation_radius_m > 0.0
        else base_costmap
    )
    return StationRouteCostmaps(
        base_costmap=base_costmap,
        planning_costmap=planning_costmap,
        target_costmap=target_costmap,
    )


__all__ = [
    "StationRouteCostmaps",
    "build_station_route_costmaps",
]
