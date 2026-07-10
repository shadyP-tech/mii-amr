"""Pure station occupancy lock model."""

from dataclasses import dataclass, replace
from typing import Dict, Optional

from .models import RobotId, StationLease


@dataclass(frozen=True)
class StationLockTable:
    leases: Dict[str, StationLease]

    @classmethod
    def empty(cls) -> "StationLockTable":
        return cls(leases={})


def acquire_station(
    table: StationLockTable,
    station_id: str,
    robot_id: RobotId,
    *,
    expires_at_sec: Optional[float] = None,
) -> StationLockTable:
    current = table.leases.get(station_id)
    if current is not None and current.robot_id != robot_id:
        raise ValueError(f"station {station_id} is already leased by {current.robot_id}")
    leases = dict(table.leases)
    leases[station_id] = StationLease(station_id, robot_id, expires_at_sec)
    return replace(table, leases=leases)


def release_station(table: StationLockTable, station_id: str, robot_id: RobotId) -> StationLockTable:
    current = table.leases.get(station_id)
    if current is None:
        return table
    if current.robot_id != robot_id:
        raise ValueError(f"robot {robot_id} cannot release lease held by {current.robot_id}")
    leases = dict(table.leases)
    del leases[station_id]
    return replace(table, leases=leases)

