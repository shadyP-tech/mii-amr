"""Local robot status model for future ROS message adapters."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RobotStatus:
    robot_id: str
    station_id: str = ""
    x_m: Optional[float] = None
    y_m: Optional[float] = None
    yaw_rad: Optional[float] = None
    phase: str = "unknown"

