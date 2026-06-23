from dataclasses import dataclass
from enum import Enum
from typing import Optional


RobotId = str


class PriorityDecision(str, Enum):
    PROCEED = "proceed"
    WAIT = "wait"
    YIELD = "yield"


@dataclass(frozen=True)
class StationLease:
    station_id: str
    robot_id: RobotId
    expires_at_sec: Optional[float] = None


@dataclass(frozen=True)
class Conflict:
    first_robot_id: RobotId
    second_robot_id: RobotId
    reason: str

