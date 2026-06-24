"""Pure route-context assembly for Aufgabe 04 station dry runs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from scripts.aufgabe04.navigation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.costmap import Costmap
from scripts.aufgabe04.navigation.global_planner import PlanRouteResult, plan_route
from scripts.aufgabe04.navigation.map_io import OccupancyGrid, load_occupancy_grid
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.station_approach import NavigationTarget, navigation_targets_from_visits
from scripts.aufgabe04.stations.models import Station, StationVisit
from scripts.aufgabe04.stations.station_map import DEFAULT_STATIONS
from scripts.aufgabe04.stations.station_router import build_station_visits


@dataclass(frozen=True)
class StationRouteDryRun:
    grid: OccupancyGrid
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
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "frame_id": "map",
        "map": str(map_yaml),
        "map_yaml": str(map_yaml),
        "map_image": str(grid.metadata.image_path),
        "map_image_sha256": file_sha256(grid.metadata.image_path),
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
    if arena_bounds is not None:
        metadata["arena_bounds"] = arena_bounds.to_metadata()
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
    arena_bounds: ArenaBounds | None = None,
) -> StationRouteDryRun:
    selected_arena_bounds = arena_bounds if arena_bounds is not None else ArenaBounds()
    selected_arena_bounds.validate()
    selected_station_ids = tuple(station_ids)
    grid = load_occupancy_grid(map_yaml)
    base_costmap = Costmap.from_occupancy_grid(grid)
    selected_station_map = station_map if station_map is not None else DEFAULT_STATIONS
    visits = tuple(build_station_visits(selected_station_ids, selected_station_map))
    planning_costmap = base_costmap.with_station_keepouts(selected_station_map.values())
    if inflation_radius_m > 0.0:
        planning_costmap = planning_costmap.with_inflation(inflation_radius_m)

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

    metadata = build_route_metadata(
        map_yaml,
        grid,
        selected_station_ids,
        station_layout_json=station_layout_json,
        arena_bounds=selected_arena_bounds,
    )
    return StationRouteDryRun(
        grid=grid,
        station_map=selected_station_map,
        visits=visits,
        targets=targets,
        results=tuple(results),
        arena_bounds=selected_arena_bounds,
        metadata=metadata,
    )
