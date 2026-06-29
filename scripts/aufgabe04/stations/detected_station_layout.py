"""Convert confirmed LiDAR stands into Aufgabe 04 station layout data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from scripts.aufgabe04.perception.stand_confirmation import ConfirmedStand
from scripts.aufgabe04.stations.models import Station, StationPose
from scripts.aufgabe04.stations.station_positioning import approach_target_for_station


@dataclass(frozen=True)
class DetectedStationLayoutConfig:
    station_id: str = "A"
    approach_offset_m: float = 0.30
    keepout_radius_m: float = 0.20
    stand_yaw_rad: float = 0.0
    arena_length_m: float = 3.90
    arena_width_m: float = 1.898
    arena_center_x_m: float = 0.0
    arena_center_y_m: float = 0.0
    arena_yaw_deg: float = 0.0
    arena_margin_m: float = 0.0


def station_from_confirmed_stand(
    stand: ConfirmedStand,
    *,
    config: DetectedStationLayoutConfig | None = None,
) -> Station:
    selected_config = config or DetectedStationLayoutConfig()
    _validate_arena(selected_config)
    if not _contains(selected_config, stand.x_m, stand.y_m):
        raise ValueError(f"confirmed stand {stand.stand_id} is outside arena bounds")
    if selected_config.approach_offset_m <= 0.0:
        raise ValueError("approach offset must be positive")
    if selected_config.keepout_radius_m < 0.0:
        raise ValueError("keepout radius must be non-negative")

    station = Station(
        station_id=selected_config.station_id,
        pose=StationPose(stand.x_m, stand.y_m, _normalize_angle(selected_config.stand_yaw_rad)),
        approach_offset_m=selected_config.approach_offset_m,
        keepout_radius_m=selected_config.keepout_radius_m,
    )
    target = approach_target_for_station(station)
    if not _contains(selected_config, target.pose.x_m, target.pose.y_m):
        raise ValueError(f"approach target for stand {stand.stand_id} is outside arena bounds")
    return station


def detected_station_metadata(
    stand: ConfirmedStand,
    *,
    source_observation_path: str,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "source": "lidar_detected_stand",
        "source_observation_path": source_observation_path,
        "stand_id": stand.stand_id,
        "stand_x_m": stand.x_m,
        "stand_y_m": stand.y_m,
        "stand_confidence": stand.confidence,
        "stand_hit_count": stand.hit_count,
        "stand_first_seen_sec": stand.first_seen_sec,
        "stand_first_confirmed_at_sec": stand.first_confirmed_at_sec,
        "stand_last_seen_sec": stand.last_seen_sec,
        "source_observation_ids": list(stand.source_observation_ids),
        "selected_observation_provenance": stand.provenance,
    }
    metadata.update(dict(extra or {}))
    return metadata


def _normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _validate_arena(config: DetectedStationLayoutConfig) -> None:
    if config.arena_length_m <= 0.0:
        raise ValueError("arena length must be positive")
    if config.arena_width_m <= 0.0:
        raise ValueError("arena width must be positive")
    if config.arena_margin_m < 0.0:
        raise ValueError("arena margin must be non-negative")
    if config.arena_margin_m * 2.0 >= config.arena_length_m:
        raise ValueError("arena margin leaves no usable arena length")
    if config.arena_margin_m * 2.0 >= config.arena_width_m:
        raise ValueError("arena margin leaves no usable arena width")


def _contains(config: DetectedStationLayoutConfig, x_m: float, y_m: float) -> bool:
    yaw = math.radians(config.arena_yaw_deg)
    dx = x_m - config.arena_center_x_m
    dy = y_m - config.arena_center_y_m
    local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    half_length = config.arena_length_m / 2.0 - config.arena_margin_m
    half_width = config.arena_width_m / 2.0 - config.arena_margin_m
    return abs(local_x) <= half_length and abs(local_y) <= half_width
