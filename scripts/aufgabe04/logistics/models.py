from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MissionState(str, Enum):
    IDLE = "idle"
    SCANNING_QR = "scanning_qr"
    ROUTING = "routing"
    NAVIGATING = "navigating"
    PICKING_UP = "picking_up"
    TRANSPORTING = "transporting"
    DROPPING_OFF = "dropping_off"
    COMPLETED = "completed"
    FAILED = "failed"


class PuckState(str, Enum):
    UNKNOWN = "unknown"
    NOT_HELD = "not_held"
    HELD = "held"
    DELIVERED = "delivered"


@dataclass(frozen=True)
class VisitResult:
    station_id: str
    status: str
    message: str = ""
    timestamp_sec: Optional[float] = None

