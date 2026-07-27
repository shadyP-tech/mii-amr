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
    fencing_token: int = 0
    acquired_at_sec: Optional[float] = None


@dataclass(frozen=True)
class Conflict:
    first_robot_id: RobotId
    second_robot_id: RobotId
    reason: str
    fail_closed: bool = False
    closest_separation_m: Optional[float] = None
    required_separation_m: Optional[float] = None
