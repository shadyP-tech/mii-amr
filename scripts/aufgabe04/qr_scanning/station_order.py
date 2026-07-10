"""Convert a QR scan into the ordered station list used by the mission layer."""

from typing import Iterable, Optional

from .models import QRDetection, StationOrder
from .qr_decoder import parse_station_payload


def station_order_from_payload(
    payload: str,
    *,
    known_stations: Optional[Iterable[str]] = None,
) -> StationOrder:
    detection = parse_station_payload(payload, known_stations=known_stations)
    return station_order_from_detection(detection)


def station_order_from_detection(detection: QRDetection) -> StationOrder:
    if not detection.station_ids:
        raise ValueError("QR detection has no station ids")
    return StationOrder(station_ids=detection.station_ids, source_payload=detection.raw_text)

