"""Shared-course conflict checks for two-robot dry runs."""

import math
from typing import Optional

from .models import Conflict
from .robot_status import RobotStatus


def detect_close_robot_conflict(
    first: RobotStatus,
    second: RobotStatus,
    *,
    min_separation_m: float,
) -> Optional[Conflict]:
    if first.x_m is None or first.y_m is None or second.x_m is None or second.y_m is None:
        return None
    distance = math.hypot(first.x_m - second.x_m, first.y_m - second.y_m)
    if distance < min_separation_m:
        return Conflict(first.robot_id, second.robot_id, f"separation {distance:.3f} m")
    return None

