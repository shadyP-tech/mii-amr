"""Finite travel deadlines shared by preapproach planning and execution.

Each budget covers one incoming segment, its initial alignment, and any
certified corner alignment held on that target. Final-pose yaw has its own
runtime deadline and is deliberately excluded. Budgets do not renew with
progress and do not change the controller or any independent safety checks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import math
from numbers import Real

from scripts.aufgabe04.navigation.control.driving_behavior import CommandSmoothingConfig
from scripts.aufgabe04.navigation.control.waypoint_controller import (
    CertifiedCornerControlConfig,
    ControllerConfig,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D


MIN_WAYPOINT_TIMEOUT_SEC = 45.0
MAX_WAYPOINT_TIMEOUT_SEC = 120.0
MOTION_TIME_MARGIN_FACTOR = 1.25
SETTLING_ALLOWANCE_SEC = 5.0
_CORNER_THRESHOLD_RAD = CertifiedCornerControlConfig().turn_threshold_rad


class SegmentTimeBudgetError(ValueError):
    """A route or explicit deadline cannot satisfy the finite time policy."""


@dataclass(frozen=True)
class SegmentTimeBudget:
    target_index: int
    distance_m: float
    alignment_turn_rad: float
    corner_turn_rad: float
    nominal_motion_sec: float
    acceleration_allowance_sec: float
    required_timeout_sec: float
    timeout_sec: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _finite_number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SegmentTimeBudgetError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise SegmentTimeBudgetError(f"{name} must be finite") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise SegmentTimeBudgetError(f"{name} must be {qualifier}")
    return result


def _validate_pose(pose: Pose2D, name: str, *, allow_unconstrained_yaw: bool = False) -> None:
    if not isinstance(pose, Pose2D):
        raise SegmentTimeBudgetError(f"{name} must be a Pose2D")
    for field in ("x_m", "y_m", "yaw_rad"):
        # Route geometry uses NaN to mark an unconstrained waypoint heading.
        # Accept that sentinel only when this calculation cannot consume it.
        if (
            field == "yaw_rad"
            and allow_unconstrained_yaw
            and isinstance(pose.yaw_rad, float)
            and math.isnan(pose.yaw_rad)
        ):
            continue
        _finite_number(getattr(pose, field), f"{name}.{field}")


def _segment(start: Pose2D, end: Pose2D) -> tuple[float, float]:
    dx = _finite_number(float(end.x_m) - float(start.x_m), "segment dx")
    dy = _finite_number(float(end.y_m) - float(start.y_m), "segment dy")
    length = _finite_number(math.hypot(dx, dy), "segment distance")
    return length, math.atan2(dy, dx)


def _angle_difference(first: float, second: float) -> float:
    # Reduce before subtracting: even finite headings can overflow subtraction.
    return abs(math.remainder(
        math.remainder(first, math.tau) - math.remainder(second, math.tau),
        math.tau,
    ))


def _corner_turn(waypoints: Sequence[Pose2D], index: int) -> float:
    if not 0 < index < len(waypoints) - 1:
        return 0.0
    incoming_length, incoming = _segment(waypoints[index - 1], waypoints[index])
    outgoing_length, outgoing = _segment(waypoints[index], waypoints[index + 1])
    if min(incoming_length, outgoing_length) <= 1.0e-9:
        return 0.0
    turn = _angle_difference(outgoing, incoming)
    return turn if turn >= _CORNER_THRESHOLD_RAD else 0.0


def waypoint_time_budget(
    waypoints: Sequence[Pose2D],
    target_index: int,
    *,
    start_pose: Pose2D | None = None,
    controller: ControllerConfig,
    smoothing: CommandSmoothingConfig = CommandSmoothingConfig(),
    timeout_limit_sec: float | None = None,
) -> SegmentTimeBudget:
    """Calculate one fixed deadline, rejecting budgets above the hard cap.

    A supplied start pose is the actual admitted pose when the target timer
    starts. Without it, planning uses the preceding vertex and accounts for
    a previous certified corner's completed alignment, avoiding a second charge
    for the same turn. Sub-threshold bends still receive alignment allowance.
    An explicit timeout is accepted only when sufficient and within the cap.
    """

    if not isinstance(waypoints, Sequence) or not waypoints:
        raise SegmentTimeBudgetError("waypoints must be a nonempty sequence")
    if (
        isinstance(target_index, bool)
        or not isinstance(target_index, int)
        or not 0 <= target_index < len(waypoints)
    ):
        raise SegmentTimeBudgetError("target_index is outside the route")
    for index, pose in enumerate(waypoints):
        _validate_pose(
            pose,
            f"waypoints[{index}]",
            allow_unconstrained_yaw=(index > 0 or start_pose is not None or target_index > 1),
        )
        if index:
            _segment(waypoints[index - 1], pose)
    if start_pose is not None:
        _validate_pose(start_pose, "start_pose")
    if not isinstance(controller, ControllerConfig):
        raise SegmentTimeBudgetError("controller must be a ControllerConfig")
    linear_speed = _finite_number(controller.max_linear_mps, "max_linear_mps", positive=True)
    angular_speed = _finite_number(controller.max_angular_radps, "max_angular_radps", positive=True)
    if not isinstance(smoothing, CommandSmoothingConfig) or type(smoothing.enabled) is not bool:
        raise SegmentTimeBudgetError("smoothing must be a valid CommandSmoothingConfig")
    linear_accel = _finite_number(smoothing.max_linear_accel_mps2, "max_linear_accel_mps2", positive=True)
    angular_accel = _finite_number(smoothing.max_angular_accel_radps2, "max_angular_accel_radps2", positive=True)

    start = start_pose if start_pose is not None else waypoints[max(0, target_index - 1)]
    distance_m, bearing = _segment(start, waypoints[target_index])
    start_heading = float(start.yaw_rad)
    if start_pose is None and target_index > 1:
        _, start_heading = _segment(waypoints[target_index - 2], start)
        if _corner_turn(waypoints, target_index - 1):
            start_heading = bearing
    alignment_turn = _angle_difference(bearing, start_heading) if distance_m > 0.0 else 0.0
    corner_turn = _corner_turn(waypoints, target_index)
    nominal_motion = distance_m / linear_speed + (alignment_turn + corner_turn) / angular_speed
    acceleration_allowance = 0.0
    if smoothing.enabled:
        if distance_m > 0.0:
            acceleration_allowance += linear_speed / linear_accel
        # Initial alignment and corner alignment have independent zero starts.
        turn_phases = int(alignment_turn > 0.0) + int(corner_turn > 0.0)
        acceleration_allowance += turn_phases * angular_speed / angular_accel
    required = max(
        MIN_WAYPOINT_TIMEOUT_SEC,
        MOTION_TIME_MARGIN_FACTOR * (nominal_motion + acceleration_allowance)
        + SETTLING_ALLOWANCE_SEC,
    )
    _finite_number(nominal_motion, "nominal motion time")
    _finite_number(acceleration_allowance, "acceleration allowance")
    _finite_number(required, "required timeout")
    if required > MAX_WAYPOINT_TIMEOUT_SEC:
        raise SegmentTimeBudgetError(
            f"target {target_index} requires {required:.3f} s, exceeding "
            f"the {MAX_WAYPOINT_TIMEOUT_SEC:.3f} s waypoint timeout cap"
        )
    timeout = required
    if timeout_limit_sec is not None:
        timeout = _finite_number(timeout_limit_sec, "timeout_limit_sec", positive=True)
        if timeout > MAX_WAYPOINT_TIMEOUT_SEC:
            raise SegmentTimeBudgetError("explicit waypoint timeout exceeds the 120 s cap")
        if timeout < required:
            raise SegmentTimeBudgetError(
                f"target {target_index} requires {required:.3f} s; "
                f"explicit waypoint timeout {timeout:.3f} s is insufficient"
            )
    return SegmentTimeBudget(
        target_index=target_index,
        distance_m=distance_m,
        alignment_turn_rad=alignment_turn,
        corner_turn_rad=corner_turn,
        nominal_motion_sec=nominal_motion,
        acceleration_allowance_sec=acceleration_allowance,
        required_timeout_sec=required,
        timeout_sec=timeout,
    )


def route_time_budgets(
    waypoints: Sequence[Pose2D],
    *,
    controller: ControllerConfig,
    smoothing: CommandSmoothingConfig = CommandSmoothingConfig(),
    start_pose: Pose2D | None = None,
    timeout_limit_sec: float | None = None,
) -> tuple[SegmentTimeBudget, ...]:
    """Budget executable targets; an actual start applies to the first only."""

    if not isinstance(waypoints, Sequence) or not waypoints:
        raise SegmentTimeBudgetError("waypoints must be a nonempty sequence")
    first_target = 1 if len(waypoints) > 1 else 0
    return tuple(
        waypoint_time_budget(
            waypoints,
            index,
            start_pose=start_pose if index == first_target else None,
            controller=controller,
            smoothing=smoothing,
            timeout_limit_sec=timeout_limit_sec,
        )
        for index in range(first_target, len(waypoints))
    )
