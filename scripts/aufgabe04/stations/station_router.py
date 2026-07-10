"""Build station visit requests from an ordered station list."""

from typing import Iterable, List, Mapping

from .models import Station, StationVisit
from .station_map import get_station
from .station_positioning import approach_target_for_station


def build_station_visits(
    station_ids: Iterable[str],
    station_map: Mapping[str, Station],
) -> List[StationVisit]:
    visits: List[StationVisit] = []
    for station_id in station_ids:
        station = get_station(station_map, station_id)
        visits.append(
            StationVisit(
                station_id=station.station_id,
                target=approach_target_for_station(station),
            )
        )
    if not visits:
        raise ValueError("at least one station visit is required")
    return visits

