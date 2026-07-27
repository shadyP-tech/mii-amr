"""Pure station-waiting and geometry-derived right-before-left policy."""

import math
from typing import Optional

from .models import PriorityDecision, RobotId
from .station_locks import StationLockTable, active_station_lease


def station_entry_decision(
    table: StationLockTable,
    station_id: str,
    robot_id: RobotId,
    *,
    now_sec: Optional[float] = None,
) -> PriorityDecision:
    lease = active_station_lease(table, station_id, now_sec=now_sec)
    if lease is None or lease.robot_id == robot_id:
        return PriorityDecision.PROCEED
    return PriorityDecision.WAIT


def right_before_left_decision(
    *,
    robot_id: RobotId,
    robot_on_right_id: Optional[RobotId],
) -> PriorityDecision:
    """Compatibility API for callers that already classified the right side."""

    if robot_on_right_id and robot_on_right_id != robot_id:
        return PriorityDecision.YIELD
    return PriorityDecision.PROCEED


def peer_is_on_right(
    *,
    robot_x_m: float,
    robot_y_m: float,
    robot_yaw_rad: float,
    peer_x_m: float,
    peer_y_m: float,
    lateral_epsilon_m: float = 1e-3,
) -> bool:
    """Classify peer position in the robot's local right half-plane."""

    values = (
        robot_x_m,
        robot_y_m,
        robot_yaw_rad,
        peer_x_m,
        peer_y_m,
        lateral_epsilon_m,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("right-before-left geometry must be finite")
    if lateral_epsilon_m < 0.0:
        raise ValueError("lateral_epsilon_m must be non-negative")
    delta_x = peer_x_m - robot_x_m
    delta_y = peer_y_m - robot_y_m
    # The right unit vector for heading (cos(yaw), sin(yaw)).
    right_component = delta_x * math.sin(robot_yaw_rad) - delta_y * math.cos(
        robot_yaw_rad
    )
    return right_component > lateral_epsilon_m


def geometry_right_before_left_decision(
    *,
    robot_id: RobotId,
    robot_x_m: float,
    robot_y_m: float,
    robot_yaw_rad: float,
    robot_requested_at_sec: float,
    peer_id: RobotId,
    peer_x_m: float,
    peer_y_m: float,
    peer_yaw_rad: float,
    peer_requested_at_sec: float,
    lateral_epsilon_m: float = 1e-3,
) -> PriorityDecision:
    """Return a deterministic bilateral right-before-left decision.

    Ambiguous geometry (both or neither robot on the other's right) is broken
    by request time and then robot ID.  Both callers therefore derive one
    winner without relying on message arrival order.
    """

    robot_id = robot_id.strip()
    peer_id = peer_id.strip()
    if not robot_id or not peer_id or robot_id == peer_id:
        raise ValueError("right-before-left requires two distinct robot IDs")
    if (
        not math.isfinite(robot_requested_at_sec)
        or robot_requested_at_sec < 0.0
        or not math.isfinite(peer_requested_at_sec)
        or peer_requested_at_sec < 0.0
    ):
        raise ValueError("right-before-left request times must be finite and non-negative")

    peer_on_right = peer_is_on_right(
        robot_x_m=robot_x_m,
        robot_y_m=robot_y_m,
        robot_yaw_rad=robot_yaw_rad,
        peer_x_m=peer_x_m,
        peer_y_m=peer_y_m,
        lateral_epsilon_m=lateral_epsilon_m,
    )
    robot_on_peer_right = peer_is_on_right(
        robot_x_m=peer_x_m,
        robot_y_m=peer_y_m,
        robot_yaw_rad=peer_yaw_rad,
        peer_x_m=robot_x_m,
        peer_y_m=robot_y_m,
        lateral_epsilon_m=lateral_epsilon_m,
    )
    if peer_on_right and not robot_on_peer_right:
        return PriorityDecision.YIELD
    if robot_on_peer_right and not peer_on_right:
        return PriorityDecision.PROCEED
    winner_id = min(
        (robot_requested_at_sec, robot_id),
        (peer_requested_at_sec, peer_id),
    )[1]
    return (
        PriorityDecision.PROCEED
        if winner_id == robot_id
        else PriorityDecision.YIELD
    )
