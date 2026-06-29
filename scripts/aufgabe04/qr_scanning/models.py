from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ScannedQR:
    raw_text: str
    qr_id: str
    confidence: float = 1.0
    source: str = "unknown"
    timestamp_sec: Optional[float] = None


@dataclass(frozen=True)
class QRDetection:
    raw_text: str
    station_ids: Tuple[str, ...]
    confidence: float = 1.0
    source: str = "unknown"
    timestamp_sec: Optional[float] = None


@dataclass(frozen=True)
class StationOrder:
    station_ids: Tuple[str, ...]
    source_payload: str


@dataclass(frozen=True)
class QRScanStatus:
    success: bool
    message: str
    detection: Optional[QRDetection] = None
