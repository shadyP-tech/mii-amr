"""Pure QR identifier decoding for the FastAPI-backed Aufgabe 04 flow."""

from __future__ import annotations

import re
from typing import Optional

from .models import ScannedQR


_QR_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def decode_qr_id(
    payload: str,
    *,
    confidence: float = 1.0,
    source: str = "unknown",
    timestamp_sec: Optional[float] = None,
) -> ScannedQR:
    qr_id = payload.strip()
    if not qr_id:
        raise ValueError("QR payload is empty")
    if not _QR_ID_RE.fullmatch(qr_id):
        raise ValueError("QR payload must be a single station or QR identifier")
    return ScannedQR(
        raw_text=payload,
        qr_id=qr_id.upper(),
        confidence=confidence,
        source=source,
        timestamp_sec=timestamp_sec,
    )

