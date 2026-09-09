"""Pure QR identifier decoding for the FastAPI-backed Aufgabe 04 flow."""

from __future__ import annotations

from typing import Optional

from scripts.aufgabe04.stations.station_ids import canonical_qr_id

from .models import ScannedQR



def decode_qr_id(
    payload: str,
    *,
    confidence: float = 1.0,
    source: str = "unknown",
    timestamp_sec: Optional[float] = None,
) -> ScannedQR:
    qr_id = canonical_qr_id(payload)
    return ScannedQR(
        raw_text=payload,
        qr_id=qr_id,
        confidence=confidence,
        source=source,
        timestamp_sec=timestamp_sec,
    )

