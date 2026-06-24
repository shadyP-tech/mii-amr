"""Station layout artifacts for Aufgabe 04 dry-run planning."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

from scripts.aufgabe04.stations.models import Station, StationPose
from scripts.aufgabe04.stations.station_map import build_station_map


SCHEMA_VERSION = 1


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _station_from_payload(payload: object, index: int) -> Station:
    item = _require_mapping(payload, f"stations[{index}]")
    raw_station_id = item.get("station_id")
    if not isinstance(raw_station_id, str):
        raise ValueError(f"stations[{index}].station_id must be a string")
    return Station(
        station_id=raw_station_id,
        pose=StationPose(
            x_m=_require_number(item.get("x_m"), f"stations[{index}].x_m"),
            y_m=_require_number(item.get("y_m"), f"stations[{index}].y_m"),
            yaw_rad=_require_number(item.get("yaw_rad"), f"stations[{index}].yaw_rad"),
        ),
        approach_offset_m=_require_number(
            item.get("approach_offset_m"),
            f"stations[{index}].approach_offset_m",
        ),
        keepout_radius_m=_require_number(
            item.get("keepout_radius_m"),
            f"stations[{index}].keepout_radius_m",
        ),
    )


def station_to_payload(station: Station) -> dict[str, object]:
    return {
        "station_id": station.station_id,
        "x_m": station.pose.x_m,
        "y_m": station.pose.y_m,
        "yaw_rad": station.pose.yaw_rad,
        "approach_offset_m": station.approach_offset_m,
        "keepout_radius_m": station.keepout_radius_m,
    }


def load_station_layout_json(path: Path) -> dict[str, Station]:
    payload = json.loads(Path(path).read_text())
    root = _require_mapping(payload, "layout")
    if root.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported station layout schema_version: {root.get('schema_version')!r}")
    stations_payload = root.get("stations")
    if not isinstance(stations_payload, list) or not stations_payload:
        raise ValueError("stations must be a non-empty list")
    stations = [
        _station_from_payload(station_payload, index)
        for index, station_payload in enumerate(stations_payload)
    ]
    return build_station_map(stations)


def station_layout_payload(
    stations: Iterable[Station],
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    station_list = sorted(stations, key=lambda station: station.station_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": dict(metadata or {}),
        "stations": [station_to_payload(station) for station in station_list],
    }


def write_station_layout_json(
    path: Path,
    stations: Iterable[Station],
    metadata: Mapping[str, object] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = station_layout_payload(stations, metadata)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_station_layout_csv(path: Path, stations: Iterable[Station]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "station_id",
                "x_m",
                "y_m",
                "yaw_rad",
                "approach_offset_m",
                "keepout_radius_m",
            ]
        )
        for station in sorted(stations, key=lambda item: item.station_id):
            writer.writerow(
                [
                    station.station_id,
                    station.pose.x_m,
                    station.pose.y_m,
                    station.pose.yaw_rad,
                    station.approach_offset_m,
                    station.keepout_radius_m,
                ]
            )
