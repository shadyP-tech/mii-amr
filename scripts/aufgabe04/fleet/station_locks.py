"""Pure, fenced station occupancy leases.

The original acquire/release signatures remain valid.  Coordinated runtime
code should additionally pass ``now_sec`` and the returned fencing token so an
expired holder cannot affect a later lease, even if the same robot reacquires
the station.
"""

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Optional

from .models import RobotId, StationLease


@dataclass(frozen=True)
class StationLockTable:
    leases: Dict[str, StationLease]
    fencing_counters: Dict[str, int] = field(default_factory=dict)
    clock_watermark_sec: Optional[float] = None

    @classmethod
    def empty(cls) -> "StationLockTable":
        return cls(leases={})


def _require_identifier(name: str, value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _checked_now(table: StationLockTable, now_sec: Optional[float]) -> Optional[float]:
    if now_sec is None:
        return None
    if not math.isfinite(now_sec) or now_sec < 0.0:
        raise ValueError("now_sec must be finite and non-negative")
    if table.clock_watermark_sec is not None and now_sec < table.clock_watermark_sec:
        raise ValueError("station lease clock moved backwards")
    return now_sec


def _checked_expiry(
    expires_at_sec: Optional[float], now_sec: Optional[float]
) -> Optional[float]:
    if expires_at_sec is None:
        return None
    if not math.isfinite(expires_at_sec) or expires_at_sec < 0.0:
        raise ValueError("expires_at_sec must be finite and non-negative")
    if now_sec is not None and expires_at_sec <= now_sec:
        raise ValueError("station lease must expire after now_sec")
    return expires_at_sec


def lease_is_expired(lease: StationLease, *, now_sec: float) -> bool:
    if not math.isfinite(now_sec) or now_sec < 0.0:
        raise ValueError("now_sec must be finite and non-negative")
    return lease.expires_at_sec is not None and now_sec >= lease.expires_at_sec


def active_station_lease(
    table: StationLockTable,
    station_id: str,
    *,
    now_sec: Optional[float] = None,
) -> Optional[StationLease]:
    station_id = _require_identifier("station_id", station_id)
    now_sec = _checked_now(table, now_sec)
    lease = table.leases.get(station_id)
    if lease is not None and now_sec is not None and lease_is_expired(
        lease, now_sec=now_sec
    ):
        return None
    return lease


def expire_station_leases(
    table: StationLockTable, *, now_sec: float
) -> StationLockTable:
    now_sec = _checked_now(table, now_sec)
    assert now_sec is not None
    leases = {
        station_id: lease
        for station_id, lease in table.leases.items()
        if not lease_is_expired(lease, now_sec=now_sec)
    }
    return replace(table, leases=leases, clock_watermark_sec=now_sec)


def acquire_station(
    table: StationLockTable,
    station_id: str,
    robot_id: RobotId,
    *,
    expires_at_sec: Optional[float] = None,
    now_sec: Optional[float] = None,
) -> StationLockTable:
    station_id = _require_identifier("station_id", station_id)
    robot_id = _require_identifier("robot_id", robot_id)
    now_sec = _checked_now(table, now_sec)
    expires_at_sec = _checked_expiry(expires_at_sec, now_sec)
    current = table.leases.get(station_id)
    current_expired = (
        current is not None
        and now_sec is not None
        and lease_is_expired(current, now_sec=now_sec)
    )
    if current is not None and not current_expired and current.robot_id != robot_id:
        raise ValueError(f"station {station_id} is already leased by {current.robot_id}")

    counters = dict(table.fencing_counters)
    if current is not None and not current_expired and current.fencing_token > 0:
        token = current.fencing_token
        acquired_at_sec = current.acquired_at_sec
    else:
        token = max(
            counters.get(station_id, 0),
            current.fencing_token if current is not None else 0,
        ) + 1
        counters[station_id] = token
        acquired_at_sec = now_sec

    if current is not None and not current_expired and expires_at_sec is None:
        # A same-owner compatibility acquire must not accidentally convert a
        # finite lease into an infinite one.  Explicit renewal is the safe way
        # to extend its lifetime.
        expires_at_sec = current.expires_at_sec

    leases = dict(table.leases)
    leases[station_id] = StationLease(
        station_id=station_id,
        robot_id=robot_id,
        expires_at_sec=expires_at_sec,
        fencing_token=token,
        acquired_at_sec=acquired_at_sec,
    )
    return replace(
        table,
        leases=leases,
        fencing_counters=counters,
        clock_watermark_sec=(
            table.clock_watermark_sec if now_sec is None else now_sec
        ),
    )


def validate_station_lease(
    table: StationLockTable,
    station_id: str,
    robot_id: RobotId,
    fencing_token: int,
    *,
    now_sec: float,
) -> StationLease:
    lease = active_station_lease(table, station_id, now_sec=now_sec)
    if lease is None:
        raise ValueError(f"station {station_id} has no active lease")
    if lease.robot_id != robot_id:
        raise ValueError(f"station {station_id} is leased by {lease.robot_id}")
    if fencing_token <= 0 or lease.fencing_token != fencing_token:
        raise ValueError("stale station lease fencing token")
    return lease


def renew_station(
    table: StationLockTable,
    station_id: str,
    robot_id: RobotId,
    fencing_token: int,
    *,
    now_sec: float,
    expires_at_sec: float,
) -> StationLockTable:
    now_sec = _checked_now(table, now_sec)
    assert now_sec is not None
    expires_at_sec = _checked_expiry(expires_at_sec, now_sec)
    validate_station_lease(
        table,
        station_id,
        robot_id,
        fencing_token,
        now_sec=now_sec,
    )
    leases = dict(table.leases)
    leases[station_id] = replace(
        leases[station_id], expires_at_sec=expires_at_sec
    )
    return replace(table, leases=leases, clock_watermark_sec=now_sec)


def release_station(
    table: StationLockTable,
    station_id: str,
    robot_id: RobotId,
    *,
    fencing_token: Optional[int] = None,
    now_sec: Optional[float] = None,
) -> StationLockTable:
    """Release a station.

    Calls without a fencing token preserve the original API.  Runtime fleet
    code must pass both ``fencing_token`` and ``now_sec`` for stale-owner
    protection.
    """

    station_id = _require_identifier("station_id", station_id)
    robot_id = _require_identifier("robot_id", robot_id)
    now_sec = _checked_now(table, now_sec)
    current = table.leases.get(station_id)
    if current is None:
        return replace(
            table,
            clock_watermark_sec=(
                table.clock_watermark_sec if now_sec is None else now_sec
            ),
        )
    if now_sec is not None and lease_is_expired(current, now_sec=now_sec):
        raise ValueError("cannot release an expired station lease")
    if current.robot_id != robot_id:
        raise ValueError(f"robot {robot_id} cannot release lease held by {current.robot_id}")
    if fencing_token is not None and (
        fencing_token <= 0 or current.fencing_token != fencing_token
    ):
        raise ValueError("stale station lease fencing token")
    leases = dict(table.leases)
    del leases[station_id]
    return replace(
        table,
        leases=leases,
        clock_watermark_sec=(
            table.clock_watermark_sec if now_sec is None else now_sec
        ),
    )
