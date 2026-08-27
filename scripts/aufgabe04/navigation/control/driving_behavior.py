"""Pure driving-policy and control-loop cadence helpers.

This module deliberately contains no ROS imports and never publishes motion.
It centralizes the route-kind policy that configures the waypoint controller
and the deadline scheduler used by the sole ``/cmd_vel`` owner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from scripts.aufgabe04.navigation.approach.viewpoint_sampling_contract import (
    INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M,
)
from scripts.aufgabe04.navigation.control.waypoint_controller import (
    ControllerConfig,
    VelocityCommand,
)


INTERMEDIATE_ROUTE_KINDS = frozenset(
    {"axis_acquisition", "viewpoint_sampling"}
)
DYNAMIC_PHYSICAL_ROUTE_KINDS = frozenset(
    {"synchronized_face_approach", "synchronized_viewpoint"}
)
CATALOG_PHYSICAL_ROUTE_KINDS = frozenset({"catalog_face_approach"})
STATIC_PHYSICAL_ROUTE_KINDS = CATALOG_PHYSICAL_ROUTE_KINDS | frozenset(
    {"detected_stand_preapproach", "stand_discovery_corridor"}
)
STATIC_STARTUP_SEGMENT_JOIN_ROUTE_KINDS = frozenset(
    {"detected_stand_preapproach", "stand_discovery_corridor"}
)
PHYSICAL_ROUTE_KINDS = DYNAMIC_PHYSICAL_ROUTE_KINDS | STATIC_PHYSICAL_ROUTE_KINDS
# These routes carry a finite yaw as an endpoint inspection requirement.  It
# must not compete with the segment bearing while the robot is still in transit.
TERMINAL_HEADING_ONLY_PHYSICAL_ROUTE_KINDS = frozenset(
    {"detected_stand_preapproach", "stand_discovery_corridor"}
)
HEADING_CORRIDOR_ROUTE_KINDS = (
    PHYSICAL_ROUTE_KINDS - TERMINAL_HEADING_ONLY_PHYSICAL_ROUTE_KINDS
)
DYNAMIC_VIEWPOINT_ROUTE_KINDS = (
    INTERMEDIATE_ROUTE_KINDS | DYNAMIC_PHYSICAL_ROUTE_KINDS
)
# The camera observer associates a stand only within a three-degree optical-axis
# cone.  A detected-stand approach therefore must finish no looser than that
# association contract, while retaining terminal-heading-only control.
DETECTED_STAND_CAMERA_HEADING_TOLERANCE_RAD = math.radians(3.0)


def controller_config_for_route_kind(
    config: ControllerConfig,
    route_kind: str,
    *,
    reverse_staging: bool = False,
    viewpoint_sampling_goal_tolerance_m: float | None = None,
    viewpoint_sampling_heading_tolerance_rad: float | None = None,
    physical_waypoint_tolerance_m: float | None = None,
    physical_goal_tolerance_m: float | None = None,
) -> ControllerConfig:
    """Resolve translation and terminal-heading behavior for one route kind.

    Physical routes retain exact-vertex pursuit.  Smoothing is performed by
    the planner before certification; this policy never creates an unchecked
    pursuit chord at runtime.
    """

    if route_kind not in DYNAMIC_VIEWPOINT_ROUTE_KINDS | STATIC_PHYSICAL_ROUTE_KINDS:
        return config
    physical = route_kind in PHYSICAL_ROUTE_KINDS
    goal_tolerance = config.goal_tolerance_m
    intermediate = route_kind in INTERMEDIATE_ROUTE_KINDS
    if intermediate:
        goal_tolerance = min(
            goal_tolerance,
            INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M,
        )
        if viewpoint_sampling_goal_tolerance_m is not None:
            goal_tolerance = min(
                goal_tolerance,
                viewpoint_sampling_goal_tolerance_m,
            )
    terminal_goal_tolerance = config.terminal_goal_tolerance_m
    if intermediate and terminal_goal_tolerance is not None:
        terminal_goal_tolerance = min(
            terminal_goal_tolerance,
            goal_tolerance,
        )
    if physical and physical_waypoint_tolerance_m is not None:
        goal_tolerance = min(goal_tolerance, physical_waypoint_tolerance_m)
    if physical and physical_goal_tolerance_m is not None:
        terminal_goal_tolerance = min(
            config.goal_tolerance_m,
            physical_goal_tolerance_m,
        )
    heading_tolerance = config.heading_tolerance_rad
    if (
        route_kind in INTERMEDIATE_ROUTE_KINDS
        and viewpoint_sampling_heading_tolerance_rad is not None
    ):
        heading_tolerance = min(
            heading_tolerance,
            viewpoint_sampling_heading_tolerance_rad,
        )
    if route_kind == "detected_stand_preapproach":
        heading_tolerance = min(
            heading_tolerance,
            DETECTED_STAND_CAMERA_HEADING_TOLERANCE_RAD,
        )
    return replace(
        config,
        goal_tolerance_m=goal_tolerance,
        terminal_goal_tolerance_m=terminal_goal_tolerance,
        heading_tolerance_rad=heading_tolerance,
        enforce_heading_corridor=route_kind in HEADING_CORRIDOR_ROUTE_KINDS,
        reverse_staging=physical and reverse_staging,
        exact_vertex_pursuit=physical,
    )


@dataclass(frozen=True)
class ControlLoopTiming:
    """Pure deadline decision for one control-loop cycle."""

    sleep_sec: float
    next_deadline_sec: float
    skipped_deadline_count: int


@dataclass(frozen=True)
class CommandSmoothingConfig:
    """Acceleration limits for ordinary, nonzero controller commands.

    Any requested reduction is applied immediately. Retaining more velocity
    than the latest controller or clearance decision permits could violate a
    certified in-place alignment or obstacle slowdown.
    """

    enabled: bool = True
    max_linear_accel_mps2: float = 0.10
    max_angular_accel_radps2: float = 0.60

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        for name in (
            "max_linear_accel_mps2",
            "max_angular_accel_radps2",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


class CommandSmoother:
    """Limit acceleration without delaying any reduction or zero axis."""

    def __init__(self, config: CommandSmoothingConfig) -> None:
        if not isinstance(config, CommandSmoothingConfig):
            raise ValueError("config must be a CommandSmoothingConfig")
        self.config = config
        self._last = VelocityCommand(0.0, 0.0)
        self._has_last = False

    def reset(self) -> None:
        self._last = VelocityCommand(0.0, 0.0)
        self._has_last = False

    def apply(self, command: VelocityCommand, *, dt_sec: float) -> VelocityCommand:
        shaped = shape_velocity_command(
            command,
            self._last if self._has_last else None,
            dt_sec,
            self.config,
        )
        if shaped.linear_x_mps == 0.0 and shaped.angular_z_radps == 0.0:
            self.reset()
            return shaped
        self._last = shaped
        self._has_last = True
        return shaped


def shape_velocity_command(
    command: VelocityCommand,
    previous: VelocityCommand | None,
    dt_sec: float,
    config: CommandSmoothingConfig = CommandSmoothingConfig(),
) -> VelocityCommand:
    """Limit acceleration while preserving the latest safety envelope."""

    if not _finite_command(command):
        raise ValueError("velocity command must be finite")
    if command.linear_x_mps == 0.0 and command.angular_z_radps == 0.0:
        return VelocityCommand(0.0, 0.0)
    if not config.enabled:
        return command
    if not math.isfinite(dt_sec) or dt_sec < 0.0:
        raise ValueError("command smoothing dt must be finite and non-negative")
    previous_command = previous if previous is not None else VelocityCommand(0.0, 0.0)
    if not _finite_command(previous_command):
        previous_command = VelocityCommand(0.0, 0.0)
    return VelocityCommand(
        _shape_axis(
            previous_command.linear_x_mps,
            command.linear_x_mps,
            dt_sec=dt_sec,
            accel_limit=config.max_linear_accel_mps2,
        ),
        _shape_axis(
            previous_command.angular_z_radps,
            command.angular_z_radps,
            dt_sec=dt_sec,
            accel_limit=config.max_angular_accel_radps2,
        ),
    )


def next_control_loop_timing(
    *,
    previous_deadline_sec: float | None,
    now_sec: float,
    control_rate_hz: float,
) -> ControlLoopTiming:
    if not math.isfinite(control_rate_hz) or control_rate_hz <= 0.0:
        raise ValueError("control rate must be finite and positive")
    if not math.isfinite(now_sec):
        raise ValueError("current time must be finite")
    period_sec = 1.0 / control_rate_hz
    deadline = (
        now_sec + period_sec
        if previous_deadline_sec is None
        else float(previous_deadline_sec)
    )
    if not math.isfinite(deadline):
        raise ValueError("previous deadline must be finite")
    sleep_sec = max(0.0, deadline - now_sec)
    next_deadline = deadline + period_sec
    skipped = 0
    if next_deadline <= now_sec:
        skipped = int(math.floor((now_sec - next_deadline) / period_sec)) + 1
        next_deadline += skipped * period_sec
    return ControlLoopTiming(
        sleep_sec=sleep_sec,
        next_deadline_sec=next_deadline,
        skipped_deadline_count=skipped,
    )


def _finite_command(command: VelocityCommand) -> bool:
    return (
        isinstance(command, VelocityCommand)
        and math.isfinite(command.linear_x_mps)
        and math.isfinite(command.angular_z_radps)
    )


def _shape_axis(
    previous: float,
    target: float,
    *,
    dt_sec: float,
    accel_limit: float,
) -> float:
    if target == previous:
        return target
    if target == 0.0:
        return 0.0
    if previous != 0.0 and math.copysign(1.0, target) != math.copysign(
        1.0, previous
    ):
        # Never cross through zero in one shaped command. The next cycle may
        # accelerate from a known zero state in the new direction.
        return 0.0
    if abs(target) < abs(previous):
        # A limiter may not retain more motion than the raw safety-scaled
        # command currently permits.
        return target
    if dt_sec == 0.0:
        return previous
    max_delta = accel_limit * dt_sec
    delta = max(-max_delta, min(max_delta, target - previous))
    return previous + delta
