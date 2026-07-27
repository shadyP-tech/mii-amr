"""Fenced, pure state machine for puck ownership and custody.

A fencing token changes every time a puck is claimed.  Commands produced by a
previous owner (or by an owner before a release/reclaim cycle) therefore cannot
load, deliver, lose, or release the current claim.
"""

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, Optional


class CustodyState(str, Enum):
    AVAILABLE = "available"
    CLAIMED = "claimed"
    LOADED = "loaded"
    DELIVERED = "delivered"
    LOST = "lost"


@dataclass(frozen=True)
class PuckCustody:
    puck_id: str
    state: CustodyState = CustodyState.AVAILABLE
    owner_robot_id: str = ""
    fencing_token: int = 0
    source_station_id: str = ""
    target_station_id: str = ""
    updated_at_sec: Optional[float] = None
    detail: str = ""


@dataclass(frozen=True)
class PuckCustodyLedger:
    records: Dict[str, PuckCustody] = field(default_factory=dict)
    fencing_counters: Dict[str, int] = field(default_factory=dict)
    clock_watermark_sec: Optional[float] = None

    @classmethod
    def empty(cls) -> "PuckCustodyLedger":
        return cls()


def _require_identifier(name: str, value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _checked_time(ledger: PuckCustodyLedger, now_sec: float) -> float:
    if not math.isfinite(now_sec) or now_sec < 0.0:
        raise ValueError("now_sec must be finite and non-negative")
    if (
        ledger.clock_watermark_sec is not None
        and now_sec < ledger.clock_watermark_sec
    ):
        raise ValueError("custody clock moved backwards")
    return now_sec


def _replace_record(
    ledger: PuckCustodyLedger,
    record: PuckCustody,
    *,
    now_sec: float,
) -> PuckCustodyLedger:
    records = dict(ledger.records)
    records[record.puck_id] = record
    return replace(ledger, records=records, clock_watermark_sec=now_sec)


def register_puck(
    ledger: PuckCustodyLedger,
    puck_id: str,
    *,
    source_station_id: str = "",
    target_station_id: str = "",
    now_sec: float,
) -> PuckCustodyLedger:
    puck_id = _require_identifier("puck_id", puck_id)
    now_sec = _checked_time(ledger, now_sec)
    if puck_id in ledger.records:
        raise ValueError(f"puck {puck_id} is already registered")
    record = PuckCustody(
        puck_id=puck_id,
        source_station_id=source_station_id.strip(),
        target_station_id=target_station_id.strip(),
        updated_at_sec=now_sec,
    )
    return _replace_record(ledger, record, now_sec=now_sec)


def claim_puck(
    ledger: PuckCustodyLedger,
    puck_id: str,
    robot_id: str,
    *,
    now_sec: float,
) -> PuckCustodyLedger:
    puck_id = _require_identifier("puck_id", puck_id)
    robot_id = _require_identifier("robot_id", robot_id)
    now_sec = _checked_time(ledger, now_sec)
    current = ledger.records.get(puck_id)
    if current is None:
        raise ValueError(f"unknown puck {puck_id}")
    if current.state != CustodyState.AVAILABLE:
        raise ValueError(
            f"puck {puck_id} is {current.state.value} by {current.owner_robot_id or 'nobody'}"
        )
    token = max(
        ledger.fencing_counters.get(puck_id, 0), current.fencing_token
    ) + 1
    counters = dict(ledger.fencing_counters)
    counters[puck_id] = token
    claimed = replace(
        current,
        state=CustodyState.CLAIMED,
        owner_robot_id=robot_id,
        fencing_token=token,
        updated_at_sec=now_sec,
        detail="",
    )
    updated = replace(ledger, fencing_counters=counters)
    return _replace_record(updated, claimed, now_sec=now_sec)


def _owned_record(
    ledger: PuckCustodyLedger,
    puck_id: str,
    robot_id: str,
    fencing_token: int,
    *,
    now_sec: float,
) -> PuckCustody:
    _checked_time(ledger, now_sec)
    current = ledger.records.get(_require_identifier("puck_id", puck_id))
    robot_id = _require_identifier("robot_id", robot_id)
    if current is None:
        raise ValueError(f"unknown puck {puck_id}")
    if current.owner_robot_id != robot_id:
        raise ValueError(f"robot {robot_id} does not own puck {puck_id}")
    if fencing_token <= 0 or current.fencing_token != fencing_token:
        raise ValueError("stale puck custody fencing token")
    return current


def confirm_puck_loaded(
    ledger: PuckCustodyLedger,
    puck_id: str,
    robot_id: str,
    fencing_token: int,
    *,
    now_sec: float,
) -> PuckCustodyLedger:
    current = _owned_record(
        ledger, puck_id, robot_id, fencing_token, now_sec=now_sec
    )
    if current.state != CustodyState.CLAIMED:
        raise ValueError("only a claimed puck can be confirmed loaded")
    return _replace_record(
        ledger,
        replace(
            current,
            state=CustodyState.LOADED,
            updated_at_sec=now_sec,
            detail="",
        ),
        now_sec=now_sec,
    )


def confirm_puck_delivered(
    ledger: PuckCustodyLedger,
    puck_id: str,
    robot_id: str,
    fencing_token: int,
    *,
    now_sec: float,
) -> PuckCustodyLedger:
    current = _owned_record(
        ledger, puck_id, robot_id, fencing_token, now_sec=now_sec
    )
    if current.state != CustodyState.LOADED:
        raise ValueError("only a loaded puck can be confirmed delivered")
    return _replace_record(
        ledger,
        replace(
            current,
            state=CustodyState.DELIVERED,
            updated_at_sec=now_sec,
            detail="",
        ),
        now_sec=now_sec,
    )


def report_puck_lost(
    ledger: PuckCustodyLedger,
    puck_id: str,
    robot_id: str,
    fencing_token: int,
    *,
    reason: str,
    now_sec: float,
) -> PuckCustodyLedger:
    current = _owned_record(
        ledger, puck_id, robot_id, fencing_token, now_sec=now_sec
    )
    if current.state not in (CustodyState.CLAIMED, CustodyState.LOADED):
        raise ValueError("only a claimed or loaded puck can be reported lost")
    reason = reason.strip()
    if not reason:
        raise ValueError("lost-puck reason must not be empty")
    return _replace_record(
        ledger,
        replace(
            current,
            state=CustodyState.LOST,
            updated_at_sec=now_sec,
            detail=reason,
        ),
        now_sec=now_sec,
    )


def release_puck_claim(
    ledger: PuckCustodyLedger,
    puck_id: str,
    robot_id: str,
    fencing_token: int,
    *,
    now_sec: float,
) -> PuckCustodyLedger:
    current = _owned_record(
        ledger, puck_id, robot_id, fencing_token, now_sec=now_sec
    )
    if current.state != CustodyState.CLAIMED:
        raise ValueError("only an unloaded claim can be released")
    return _replace_record(
        ledger,
        replace(
            current,
            state=CustodyState.AVAILABLE,
            owner_robot_id="",
            updated_at_sec=now_sec,
            detail="",
        ),
        now_sec=now_sec,
    )
