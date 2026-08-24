"""Generate in-memory waypoint rows for per-station routes."""

from typing import Iterable, List, Tuple

from scripts.aufgabe04.stations.models import StationVisit


def station_visits_to_waypoint_rows(
    visits: Iterable[StationVisit],
) -> List[Tuple[int, float, float, str]]:
    rows: List[Tuple[int, float, float, str]] = []
    for index, visit in enumerate(visits):
        rows.append(
            (
                index,
                visit.target.pose.x_m,
                visit.target.pose.y_m,
                visit.station_id,
            )
        )
    if not rows:
        raise ValueError("at least one station visit is required")
    return rows

