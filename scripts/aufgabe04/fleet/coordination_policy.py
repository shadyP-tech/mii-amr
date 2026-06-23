"""Pure right-before-left and station-waiting policy skeleton."""

from typing import Optional

from .models import PriorityDecision, RobotId
from .station_locks import StationLockTable


def station_entry_decision(
    table: StationLockTable,
    station_id: str,
    robot_id: RobotId,
) -> PriorityDecision:
    lease = table.leases.get(station_id)
    if lease is None or lease.robot_id == robot_id:
        return PriorityDecision.PROCEED
    return PriorityDecision.WAIT


def right_before_left_decision(
    *,
    robot_id: RobotId,
    robot_on_right_id: Optional[RobotId],
) -> PriorityDecision:
    if robot_on_right_id and robot_on_right_id != robot_id:
        return PriorityDecision.YIELD
    return PriorityDecision.PROCEED

