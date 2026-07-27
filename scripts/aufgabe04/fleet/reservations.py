"""Exclusive, expiring and fenced permits for shared fleet resources."""

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Optional


@dataclass(frozen=True)
class ReservationPermit:
    resource_id: str
    robot_id: str
    fencing_token: int
    issued_at_sec: float
    expires_at_sec: float


@dataclass(frozen=True)
class ReservationTable:
    permits: Dict[str, ReservationPermit] = field(default_factory=dict)
    fencing_counters: Dict[str, int] = field(default_factory=dict)
    clock_watermark_sec: Optional[float] = None

    @classmethod
    def empty(cls) -> "ReservationTable":
        return cls()


def _identifier(name: str, value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _now(table: ReservationTable, now_sec: float) -> float:
    if not math.isfinite(now_sec) or now_sec < 0.0:
        raise ValueError("now_sec must be finite and non-negative")
    if table.clock_watermark_sec is not None and now_sec < table.clock_watermark_sec:
        raise ValueError("reservation clock moved backwards")
    return now_sec


def _ttl(ttl_sec: float) -> float:
    if not math.isfinite(ttl_sec) or ttl_sec <= 0.0:
        raise ValueError("ttl_sec must be finite and positive")
    return ttl_sec


def active_permit(
    table: ReservationTable, resource_id: str, *, now_sec: float
) -> Optional[ReservationPermit]:
    resource_id = _identifier("resource_id", resource_id)
    now_sec = _now(table, now_sec)
    permit = table.permits.get(resource_id)
    if permit is None or now_sec >= permit.expires_at_sec:
        return None
    return permit


def expire_permits(
    table: ReservationTable, *, now_sec: float
) -> ReservationTable:
    now_sec = _now(table, now_sec)
    permits = {
        resource_id: permit
        for resource_id, permit in table.permits.items()
        if now_sec < permit.expires_at_sec
    }
    return replace(table, permits=permits, clock_watermark_sec=now_sec)


def acquire_permit(
    table: ReservationTable,
    resource_id: str,
    robot_id: str,
    *,
    now_sec: float,
    ttl_sec: float,
) -> ReservationTable:
    resource_id = _identifier("resource_id", resource_id)
    robot_id = _identifier("robot_id", robot_id)
    now_sec = _now(table, now_sec)
    ttl_sec = _ttl(ttl_sec)
    current = table.permits.get(resource_id)
    active = current is not None and now_sec < current.expires_at_sec
    if active:
        raise ValueError(
            f"resource {resource_id} is already reserved by {current.robot_id}; "
            "renewal requires its fencing token"
        )
    counters = dict(table.fencing_counters)
    token = max(
        counters.get(resource_id, 0),
        current.fencing_token if current is not None else 0,
    ) + 1
    counters[resource_id] = token
    issued_at_sec = now_sec
    permits = dict(table.permits)
    permits[resource_id] = ReservationPermit(
        resource_id=resource_id,
        robot_id=robot_id,
        fencing_token=token,
        issued_at_sec=issued_at_sec,
        expires_at_sec=now_sec + ttl_sec,
    )
    return replace(
        table,
        permits=permits,
        fencing_counters=counters,
        clock_watermark_sec=now_sec,
    )


def validate_permit(
    table: ReservationTable,
    resource_id: str,
    robot_id: str,
    fencing_token: int,
    *,
    now_sec: float,
) -> ReservationPermit:
    permit = active_permit(table, resource_id, now_sec=now_sec)
    if permit is None:
        raise ValueError(f"resource {resource_id} has no active permit")
    if permit.robot_id != robot_id:
        raise ValueError(f"resource {resource_id} is reserved by {permit.robot_id}")
    if fencing_token <= 0 or permit.fencing_token != fencing_token:
        raise ValueError("stale reservation fencing token")
    return permit


def renew_permit(
    table: ReservationTable,
    resource_id: str,
    robot_id: str,
    fencing_token: int,
    *,
    now_sec: float,
    ttl_sec: float,
) -> ReservationTable:
    now_sec = _now(table, now_sec)
    ttl_sec = _ttl(ttl_sec)
    current = validate_permit(
        table, resource_id, robot_id, fencing_token, now_sec=now_sec
    )
    permits = dict(table.permits)
    permits[resource_id] = replace(current, expires_at_sec=now_sec + ttl_sec)
    return replace(table, permits=permits, clock_watermark_sec=now_sec)


def release_permit(
    table: ReservationTable,
    resource_id: str,
    robot_id: str,
    fencing_token: int,
    *,
    now_sec: float,
) -> ReservationTable:
    now_sec = _now(table, now_sec)
    validate_permit(table, resource_id, robot_id, fencing_token, now_sec=now_sec)
    permits = dict(table.permits)
    del permits[resource_id]
    return replace(table, permits=permits, clock_watermark_sec=now_sec)
