"""Pure post-controller command admission decisions.

This module deliberately has no access to ROS, the follower node, trace I/O,
callback servicing, sleeps, or publishers.  The control loop remains the only
owner of when a decision is applied and when motion is revoked or emitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from scripts.aufgabe04.navigation.control.waypoint_controller import (
    VelocityCommand,
)
from scripts.aufgabe04.navigation.coverage.transient_blockage_policy import (
    LinearCommandFloorDecision,
    classify_linear_command,
    reachable_distance_progress_epsilon,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.commands import (
    finite_velocity_command,
)


@dataclass(frozen=True)
class CommandAdmissionDecision:
    """Pure clearance, motion-floor, and finite-command decision."""

    effective_command: VelocityCommand
    command_floor: LinearCommandFloorDecision
    clearance_limited_below_floor: bool
    finite: bool


@dataclass(frozen=True)
class PreparedCommandDecision:
    """Typed result of stateful preparation performed by the control loop."""

    raw_effective_command: VelocityCommand
    shaped_command: VelocityCommand | None
    shape_dt_sec: float | None
    stop_details: Mapping[str, object] | None = None
    trace_diagnostics: Mapping[str, object] | None = None


def command_admission_decision(
    nominal_command: VelocityCommand,
    *,
    front_clearance_scale: float,
    linear_motion_floor_mps: float,
    physical_route: bool,
) -> CommandAdmissionDecision:
    """Calculate the command that may proceed to watchdog and smoothing."""

    effective_command = VelocityCommand(
        nominal_command.linear_x_mps * front_clearance_scale,
        nominal_command.angular_z_radps,
    )
    command_floor = classify_linear_command(
        nominal_command.linear_x_mps,
        effective_command.linear_x_mps,
        linear_motion_floor_mps=linear_motion_floor_mps,
    )
    clearance_limited_below_floor = (
        physical_route
        and front_clearance_scale < 1.0 - 1.0e-12
        and command_floor.zero_hold_required
    )
    return CommandAdmissionDecision(
        effective_command=effective_command,
        command_floor=command_floor,
        clearance_limited_below_floor=clearance_limited_below_floor,
        finite=finite_velocity_command(
            effective_command.linear_x_mps,
            effective_command.angular_z_radps,
        ),
    )


def stuck_distance_progress_epsilon(
    configured_progress_epsilon_m: float,
    *,
    physical_route: bool,
    remaining_distance_m: float,
    waypoint_tolerance_m: float,
    effective_linear_x_mps: float,
    stuck_timeout_sec: float,
) -> float:
    """Return the reachable progress threshold used by the stuck watchdog."""

    if not physical_route:
        return configured_progress_epsilon_m
    bounded = reachable_distance_progress_epsilon(
        configured_progress_epsilon_m,
        remaining_distance_m=remaining_distance_m,
        waypoint_tolerance_m=waypoint_tolerance_m,
        expected_effective_travel_m=(
            abs(effective_linear_x_mps) * stuck_timeout_sec
        ),
    )
    if bounded < configured_progress_epsilon_m:
        # The comparison in the watchdog is strict.  Half of the reachable
        # headroom remains attainable before exact-vertex capture.
        return 0.5 * bounded
    return configured_progress_epsilon_m


def command_shape_interval_sec(
    *,
    loop_period_sec: float,
    now_monotonic: float,
    last_shape_at: float | None,
) -> float:
    """Calculate the smoother interval without mutating smoother state."""

    if last_shape_at is None:
        return loop_period_sec
    return now_monotonic - last_shape_at
