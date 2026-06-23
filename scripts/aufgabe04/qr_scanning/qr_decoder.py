"""Pure QR payload parsing for Aufgabe 04.

ROS/OpenCV image acquisition belongs in ``onboard_camera_node.py``. This module
is intentionally pure so QR payload assumptions can be tested offline.
"""

from typing import Iterable, Optional, Set

from .models import QRDetection


def parse_station_payload(
    payload: str,
    *,
    known_stations: Optional[Iterable[str]] = None,
) -> QRDetection:
    text = payload.strip()
    if not text:
        raise ValueError("QR payload is empty")

    lowered = text.lower()
    for prefix in ("stations:", "station:", "route:"):
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip()
            break

    normalized = text.replace("->", ",").replace(";", ",").replace("|", ",")
    station_ids = tuple(part.strip().upper() for part in normalized.split(",") if part.strip())
    if not station_ids:
        raise ValueError("QR payload does not contain any station ids")
    if len(set(station_ids)) != len(station_ids):
        raise ValueError("QR payload contains duplicate station ids")

    allowed: Optional[Set[str]] = None
    if known_stations is not None:
        allowed = {station.strip().upper() for station in known_stations}
        unknown = [station for station in station_ids if station not in allowed]
        if unknown:
            raise ValueError(f"QR payload contains unknown station ids: {', '.join(unknown)}")

    return QRDetection(raw_text=payload, station_ids=station_ids)

