"""Pure route-context assembly for Aufgabe 04 station dry runs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from scripts.aufgabe04.navigation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.costmap import Costmap
from scripts.aufgabe04.navigation.global_planner import PlanRouteResult, plan_route
from scripts.aufgabe04.navigation.map_io import (
    FrozenMapBundle,
    OccupancyGrid,
    load_occupancy_grid,
)
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.route_smoothing import smooth_plan_route_results
from scripts.aufgabe04.navigation.station_approach import NavigationTarget, navigation_targets_from_visits
from scripts.aufgabe04.stations.models import Station, StationVisit
from scripts.aufgabe04.stations.station_map import DEFAULT_STATIONS
from scripts.aufgabe04.stations.station_router import build_station_visits


@dataclass(frozen=True)
class StationRouteDryRun:
    grid: OccupancyGrid
    base_costmap: Costmap
    planning_costmap: Costmap
    station_map: Mapping[str, Station]
    visits: tuple[StationVisit, ...]
    targets: tuple[NavigationTarget, ...]
    results: tuple[PlanRouteResult, ...]
    arena_bounds: ArenaBounds
    metadata: Mapping[str, object]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_route_metadata(
    map_yaml: Path,
    grid: OccupancyGrid,
    station_ids: Iterable[str],
    *,
    station_layout_json: Path | None = None,
    arena_bounds: ArenaBounds | None = None,
    map_bundle: FrozenMapBundle | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "frame_id": "map",
        "map": str(map_yaml),
        "map_yaml": str(map_yaml),
        "map_image": str(grid.metadata.image_path),
        "map_yaml_sha256": (
            file_sha256(map_yaml)
            if map_bundle is None
            else map_bundle.yaml_sha256
        ),
        "map_image_sha256": (
            file_sha256(grid.metadata.image_path)
            if map_bundle is None
            else map_bundle.image_sha256
        ),
        "resolution": grid.metadata.resolution,
        "origin": grid.metadata.origin,
        "stations": list(station_ids),
        "coordinate_warning": (
            "Static saved-map coordinates only; this overlay does not validate "
            "live TF, AMCL, Nav2, sensors, or cmd_vel ownership."
        ),
    }
    if station_layout_json is not None:
        metadata["station_layout_json"] = str(station_layout_json)
    if map_bundle is not None:
        metadata["semantic_map_id"] = map_bundle.semantic_map_id
        metadata["map_bundle_sha256"] = map_bundle.bundle_sha256
    if arena_bounds is not None:
        metadata["arena_bounds"] = arena_bounds.to_metadata()
        metadata["arena_boundary_overlay"] = True
    return metadata


def build_station_route_dry_run(
    map_yaml: Path,
    station_ids: Iterable[str],
    *,
    station_map: Mapping[str, Station] | None = None,
    station_layout_json: Path | None = None,
    start: Pose2D | None = None,
    inflation_radius_m: float = 0.0,
    snap_radius_m: float = 0.30,
    transit_keepout_radius_m: float = 0.0,
    arena_bounds: ArenaBounds | None = None,
    occupancy_grid: OccupancyGrid | None = None,
    map_bundle: FrozenMapBundle | None = None,
    line_of_sight_optimization: bool = True,
) -> StationRouteDryRun:
    selected_arena_bounds = arena_bounds if arena_bounds is not None else ArenaBounds()
    selected_arena_bounds.validate()
    selected_station_ids = tuple(station_ids)
    if (occupancy_grid is None) != (map_bundle is None):
        raise ValueError("occupancy_grid and map_bundle must be supplied together")
    grid = occupancy_grid or load_occupancy_grid(map_yaml)
    if map_bundle is not None:
        if grid.width != map_bundle.width or grid.height != map_bundle.height:
            raise ValueError("occupancy grid dimensions do not match map bundle")
    # Saved maps may be padded beyond the measured arena and may not contain
    # the Gazebo/parkour walls at all.  Rasterize the physical boundary before
    # inflation so a route cannot leave the arena through nominally free map
    # cells.
    base_costmap = Costmap.from_occupancy_grid(grid).with_arena_bounds(
        selected_arena_bounds
    )
    selected_station_map = station_map if station_map is not None else DEFAULT_STATIONS
    visits = tuple(build_station_visits(selected_station_ids, selected_station_map))
    planning_costmap = base_costmap
    if inflation_radius_m > 0.0:
        planning_costmap = planning_costmap.with_inflation(inflation_radius_m)
    if transit_keepout_radius_m > 0.0:
        transit_keepouts = tuple(
            Station(
                station.station_id,
                station.pose,
                station.approach_offset_m,
                transit_keepout_radius_m,
            )
            for station in selected_station_map.values()
        )
        planning_costmap = planning_costmap.with_station_keepouts(transit_keepouts)

    target_costmap = base_costmap.with_inflation(inflation_radius_m) if inflation_radius_m > 0.0 else base_costmap
    targets = navigation_targets_from_visits(visits, target_costmap)
    current = start if start is not None else targets[0].pose
    results = []
    for target in targets:
        result = plan_route(planning_costmap, current, target.pose, snap_radius_m=snap_radius_m)
        results.append(result)
        if result.route is None:
            break
        current = target.pose
    smoothed_results = smooth_plan_route_results(
        tuple(results),
        costmap=planning_costmap,
        enabled=line_of_sight_optimization,
    )
    results = [item.result for item in smoothed_results]

    metadata = build_route_metadata(
        map_yaml,
        grid,
        selected_station_ids,
        station_layout_json=station_layout_json,
        arena_bounds=selected_arena_bounds,
        map_bundle=map_bundle,
    )
    metadata["inflation_radius_m"] = inflation_radius_m
    metadata["line_of_sight_route_optimization"] = {
        "enabled": line_of_sight_optimization,
        "legs": [item.summary.to_metadata() for item in smoothed_results],
        "input_point_count": sum(
            item.summary.input_point_count for item in smoothed_results
        ),
        "output_point_count": sum(
            item.summary.output_point_count for item in smoothed_results
        ),
        "optimized_leg_count": sum(
            1 for item in smoothed_results if item.summary.optimized
        ),
    }
    return StationRouteDryRun(
        grid=grid,
        base_costmap=base_costmap,
        planning_costmap=planning_costmap,
        station_map=selected_station_map,
        visits=visits,
        targets=targets,
        results=tuple(results),
        arena_bounds=selected_arena_bounds,
        metadata=metadata,
    )
