"""Reproducible random station placement on Aufgabe 04 dry-run maps."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.map_io import OccupancyGrid, load_occupancy_grid
from scripts.aufgabe04.navigation.foundation.models import GridCell, Pose2D
from scripts.aufgabe04.stations.models import Station, StationPose
from scripts.aufgabe04.stations.station_map import build_station_map, normalize_station_id


YAW_MODE_TOWARD_CENTER = "toward-center"
YAW_MODE_RANDOM = "random"
YAW_MODES = (YAW_MODE_TOWARD_CENTER, YAW_MODE_RANDOM)


@dataclass(frozen=True)
class RandomStationLayoutConfig:
    map_yaml: Path
    station_ids: tuple[str, ...]
    seed: int
    clearance_radius_m: float = 0.10
    min_station_distance_m: float = 0.40
    start: Pose2D | None = None
    min_start_distance_m: float = 0.40
    yaw_mode: str = YAW_MODE_TOWARD_CENTER
    max_attempts: int = 5000
    approach_offset_m: float = 0.30
    keepout_radius_m: float = 0.20
    arena_bounds: ArenaBounds = ArenaBounds()


@dataclass(frozen=True)
class RandomStationLayoutResult:
    stations: Mapping[str, Station]
    metadata: Mapping[str, object]


def station_ids_from_count(count: int) -> tuple[str, ...]:
    if count <= 0:
        raise ValueError("station count must be positive")
    if count > 26:
        raise ValueError("station count must not exceed 26")
    return tuple(chr(ord("A") + index) for index in range(count))


def parse_station_ids(text: str) -> tuple[str, ...]:
    station_ids = tuple(normalize_station_id(part) for part in text.replace(",", " ").split())
    if not station_ids:
        raise ValueError("at least one station id is required")
    if len(set(station_ids)) != len(station_ids):
        raise ValueError("station ids must be unique")
    return station_ids


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _free_cell_center(costmap: Costmap) -> Pose2D:
    xs = []
    ys = []
    for y in range(costmap.height):
        for x in range(costmap.width):
            cell = GridCell(x, y)
            if costmap.is_traversable(cell):
                xs.append(x)
                ys.append(y)
    if not xs:
        raise ValueError("map has no traversable cells")
    center_cell = GridCell((min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2)
    return costmap.grid_to_world(center_cell)


def _candidate_cells(costmap: Costmap) -> tuple[GridCell, ...]:
    cells = [
        GridCell(x, y)
        for y in range(costmap.height)
        for x in range(costmap.width)
        if costmap.is_traversable(GridCell(x, y))
    ]
    if not cells:
        raise ValueError("map has no traversable candidate cells after clearance inflation")
    return tuple(cells)


def _distance(a: Pose2D | StationPose, b: Pose2D | StationPose) -> float:
    return math.hypot(a.x_m - b.x_m, a.y_m - b.y_m)


def _yaw_for_station(rng: random.Random, mode: str, pose: Pose2D, center: Pose2D) -> float:
    if mode == YAW_MODE_RANDOM:
        return rng.uniform(-math.pi, math.pi)
    if mode == YAW_MODE_TOWARD_CENTER:
        return math.atan2(center.y_m - pose.y_m, center.x_m - pose.x_m)
    raise ValueError(f"unsupported yaw mode: {mode}")


def _validate_config(config: RandomStationLayoutConfig, grid: OccupancyGrid) -> None:
    if not isinstance(config.seed, int) or isinstance(config.seed, bool):
        raise ValueError("seed must be an integer")
    if config.yaw_mode not in YAW_MODES:
        raise ValueError(f"yaw mode must be one of: {', '.join(YAW_MODES)}")
    if not config.station_ids:
        raise ValueError("at least one station id is required")
    if len(set(config.station_ids)) != len(config.station_ids):
        raise ValueError("station ids must be unique")
    if config.max_attempts <= 0:
        raise ValueError("max attempts must be positive")
    config.arena_bounds.validate()
    for name, value in [
        ("clearance radius", config.clearance_radius_m),
        ("minimum station distance", config.min_station_distance_m),
        ("minimum start distance", config.min_start_distance_m),
        ("approach offset", config.approach_offset_m),
        ("keepout radius", config.keepout_radius_m),
    ]:
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
    min_approach = config.keepout_radius_m + grid.metadata.resolution
    if config.approach_offset_m <= min_approach:
        raise ValueError(
            "approach offset must be greater than keepout radius plus map resolution"
        )


def _target_is_traversable(costmap: Costmap, station: Station) -> bool:
    pose = _qr_face_clearance_pose(station)
    cell = costmap.world_to_grid(pose)
    return costmap.in_bounds(cell) and costmap.is_traversable(cell)


def _qr_face_clearance_pose(station: Station) -> Pose2D:
    """Return the hidden QR-side clearance point (Gazebo local +x)."""

    return Pose2D(
        station.pose.x_m + math.cos(station.pose.yaw_rad) * station.approach_offset_m,
        station.pose.y_m + math.sin(station.pose.yaw_rad) * station.approach_offset_m,
        math.atan2(
            math.sin(station.pose.yaw_rad + math.pi),
            math.cos(station.pose.yaw_rad + math.pi),
        ),
    )


def _station_and_target_inside_arena(station: Station, bounds: ArenaBounds) -> bool:
    station_pose = Pose2D(station.pose.x_m, station.pose.y_m, station.pose.yaw_rad)
    target_pose = _qr_face_clearance_pose(station)
    return bounds.contains(station_pose) and bounds.contains(target_pose)


def _candidate_is_valid(
    station: Station,
    accepted: Sequence[Station],
    target_costmap: Costmap,
    final_base_costmap: Costmap,
    config: RandomStationLayoutConfig,
) -> bool:
    station_pose = Pose2D(station.pose.x_m, station.pose.y_m, station.pose.yaw_rad)
    if not _station_and_target_inside_arena(station, config.arena_bounds):
        return False
    station_cell = target_costmap.world_to_grid(station_pose)
    if not target_costmap.in_bounds(station_cell) or not target_costmap.is_traversable(station_cell):
        return False
    target_pose = _qr_face_clearance_pose(station)
    target_cell = target_costmap.world_to_grid(target_pose)
    if not target_costmap.in_bounds(target_cell) or not target_costmap.is_traversable(target_cell):
        return False
    if _distance(station.pose, target_pose) <= station.keepout_radius_m:
        return False
    if config.start is not None and _distance(station.pose, config.start) < config.min_start_distance_m:
        return False
    if any(_distance(station.pose, existing.pose) < config.min_station_distance_m for existing in accepted):
        return False

    trial_stations = tuple(accepted) + (station,)
    keepout_costmap = final_base_costmap.with_station_keepouts(trial_stations)
    inflated_keepout_costmap = keepout_costmap.with_inflation(config.clearance_radius_m)
    for item in trial_stations:
        if not _station_and_target_inside_arena(item, config.arena_bounds):
            return False
        if not _target_is_traversable(keepout_costmap, item):
            return False
        if not _target_is_traversable(inflated_keepout_costmap, item):
            return False
    return True


def _metadata_for_layout(
    grid: OccupancyGrid,
    config: RandomStationLayoutConfig,
) -> dict[str, object]:
    return {
        "seed": config.seed,
        "map_yaml": str(config.map_yaml),
        "map_resolution": grid.metadata.resolution,
        "map_origin": list(grid.metadata.origin),
        "map_width": grid.width,
        "map_height": grid.height,
        "map_yaml_sha256": _sha256(grid.metadata.yaml_path),
        "map_image_sha256": _sha256(grid.metadata.image_path),
        "station_ids": list(config.station_ids),
        "clearance_radius_m": config.clearance_radius_m,
        "min_station_distance_m": config.min_station_distance_m,
        "start": None if config.start is None else {
            "x_m": config.start.x_m,
            "y_m": config.start.y_m,
            "yaw_rad": config.start.yaw_rad,
        },
        "min_start_distance_m": config.min_start_distance_m,
        "yaw_mode": config.yaw_mode,
        "max_attempts": config.max_attempts,
        "approach_offset_m": config.approach_offset_m,
        "keepout_radius_m": config.keepout_radius_m,
        "arena_bounds": config.arena_bounds.to_metadata(),
    }


def generate_random_station_layout(config: RandomStationLayoutConfig) -> RandomStationLayoutResult:
    grid = load_occupancy_grid(config.map_yaml)
    normalized_config = RandomStationLayoutConfig(
        map_yaml=config.map_yaml,
        station_ids=tuple(normalize_station_id(station_id) for station_id in config.station_ids),
        seed=config.seed,
        clearance_radius_m=config.clearance_radius_m,
        min_station_distance_m=config.min_station_distance_m,
        start=config.start,
        min_start_distance_m=config.min_start_distance_m,
        yaw_mode=config.yaw_mode,
        max_attempts=config.max_attempts,
        approach_offset_m=config.approach_offset_m,
        keepout_radius_m=config.keepout_radius_m,
        arena_bounds=config.arena_bounds,
    )
    _validate_config(normalized_config, grid)

    base_costmap = Costmap.from_occupancy_grid(grid)
    clearance_costmap = base_costmap.with_inflation(normalized_config.clearance_radius_m)
    candidates = _candidate_cells(clearance_costmap)
    center = Pose2D(
        normalized_config.arena_bounds.center_x_m,
        normalized_config.arena_bounds.center_y_m,
        0.0,
    )
    rng = random.Random(normalized_config.seed)
    accepted: list[Station] = []
    attempts = 0

    while len(accepted) < len(normalized_config.station_ids) and attempts < normalized_config.max_attempts:
        attempts += 1
        cell = rng.choice(candidates)
        pose = clearance_costmap.grid_to_world(cell)
        yaw = _yaw_for_station(rng, normalized_config.yaw_mode, pose, center)
        station = Station(
            station_id=normalized_config.station_ids[len(accepted)],
            pose=StationPose(pose.x_m, pose.y_m, yaw),
            approach_offset_m=normalized_config.approach_offset_m,
            keepout_radius_m=normalized_config.keepout_radius_m,
        )
        if _candidate_is_valid(
            station,
            accepted,
            clearance_costmap,
            base_costmap,
            normalized_config,
        ):
            accepted.append(station)

    if len(accepted) != len(normalized_config.station_ids):
        raise ValueError(
            "could not generate requested station layout: "
            f"requested={len(normalized_config.station_ids)} accepted={len(accepted)} "
            f"max_attempts={normalized_config.max_attempts}"
        )

    return RandomStationLayoutResult(
        stations=build_station_map(accepted),
        metadata=_metadata_for_layout(grid, normalized_config),
    )
