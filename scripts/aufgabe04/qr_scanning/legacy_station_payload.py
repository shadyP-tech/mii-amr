"""Legacy QR station-order parsing.

The FastAPI-backed Aufgabe 04 flow scans one QR/station identifier only. This
module keeps the earlier station-order QR parser explicit so new code does not
accidentally treat QR payloads as mission plans.
"""

from __future__ import annotations

from typing import Iterable, Optional

from .models import QRDetection
from .qr_decoder import parse_station_payload


def parse_legacy_station_order_payload(
    payload: str,
    *,
    known_stations: Optional[Iterable[str]] = None,
) -> QRDetection:
    return parse_station_payload(payload, known_stations=known_stations)

