"""Static Aufgabe 04 station map definitions.

Coordinates are placeholders until the real parkour station poses are measured.
Keep this module pure and explicit so route tests can validate every target.
"""

from typing import Dict, Iterable, Mapping

from .models import Station, StationPose


DEFAULT_STATIONS: Dict[str, Station] = {
    "A": Station("A", StationPose(0.0, 0.0, 0.0)),
    "B": Station("B", StationPose(1.0, 0.0, 0.0)),
    "C": Station("C", StationPose(1.0, 1.0, 0.0)),
}


def normalize_station_id(station_id: str) -> str:
    normalized = station_id.strip().upper()
    if not normalized:
        raise ValueError("station id must not be empty")
    return normalized


def build_station_map(stations: Iterable[Station]) -> Dict[str, Station]:
    station_map: Dict[str, Station] = {}
    for station in stations:
        station_id = normalize_station_id(station.station_id)
        if station_id in station_map:
            raise ValueError(f"duplicate station id: {station_id}")
        station_map[station_id] = Station(
            station_id=station_id,
            pose=station.pose,
            approach_offset_m=station.approach_offset_m,
            keepout_radius_m=station.keepout_radius_m,
        )
    return station_map


def get_station(station_map: Mapping[str, Station], station_id: str) -> Station:
    normalized = normalize_station_id(station_id)
    try:
        return station_map[normalized]
    except KeyError as exc:
        raise KeyError(f"unknown station id: {normalized}") from exc

