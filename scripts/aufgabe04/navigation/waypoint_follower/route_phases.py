"""Route phase, control-step, and bounded hold decision contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from scripts.aufgabe04.navigation.control.driving_behavior import (
    DYNAMIC_PHYSICAL_ROUTE_KINDS,
    DYNAMIC_VIEWPOINT_ROUTE_KINDS,
    INTERMEDIATE_ROUTE_KINDS,
)
from scripts.aufgabe04.navigation.control.waypoint_controller import (
    ControllerStep,
)
from scripts.aufgabe04.navigation.waypoint_follower.directives import (
    AcquisitionGoalAction,
    StringDirective,
)


class RouteCommandPhase(StringDirective):
    """Mutually exclusive route-command branches in precedence order."""

    DYNAMIC_JOIN = "dynamic_join"
    START_EGRESS = "start_egress"
    REVERSE_EGRESS_ALIGNMENT = "reverse_egress_alignment"
    CERTIFIED_ROUTE = "certified_route"


class WaypointLifecycleAction(StringDirective):
    """Control-loop effects after one controller step is evaluated."""

    PROCEED = "proceed"
    HOLD = "hold"
    COMPLETE = "complete"
    STOP = "stop"


class ControlStepAction(StringDirective):
    """Control-loop effect required by one controller-step resolution."""

    PROCEED = "proceed"
    ZERO_HOLD = "zero_hold"
    STOP = "stop"


class ControlStepStopKind(StringDirective):
    """Failure branch whose evidence and effects remain owned by ``run``."""

    NONE = ""
    DYNAMIC_JOIN = "dynamic_join"
    CERTIFIED_CORNER = "certified_corner"
    INTERMEDIATE_TERMINAL_HEADING = "intermediate_terminal_heading"


@dataclass(frozen=True)
class ControlStepResolution:
    """Typed command-phase outcome without publishing or loop control."""

    action: ControlStepAction
    command_phase: RouteCommandPhase
    step: ControllerStep | None = None
    corner_step: ControllerStep | None = None
    stop_kind: ControlStepStopKind = ControlStepStopKind.NONE
    stop_reason: str = ""
    stop_details: Mapping[str, object] | None = None


@dataclass(frozen=True)
class WaypointLifecycleDecision:
    """Stateful target, reached-goal, and timeout outcome for one step."""

    action: WaypointLifecycleAction
    stop_reason: str = ""
    stop_details: Mapping[str, object] | None = None
    evaluated_at: float | None = None


@dataclass(frozen=True)
class ViewpointSamplingDeadlineDecision:
    """Pure timeout outcome and evidence timing for one sampling cycle."""

    failure: str
    phase_elapsed_sec: float | None
    target_elapsed_sec: float | None


@dataclass(frozen=True)
class AcquisitionGoalDecision:
    """Pure acquisition-terminal action with its effective hold clock."""

    action: AcquisitionGoalAction
    hold_started_at: float
    hold_elapsed_sec: float


def route_command_phase(
    *,
    dynamic_join_pending: bool,
    start_egress_lock_index: int | None,
    start_egress_forward_alignment_index: int | None,
) -> RouteCommandPhase:
    """Select one command branch without touching follower runtime state.

    Join admission must dominate every egress state because its command is
    restricted to the certified route anchor.  The start-egress vertex lock
    then dominates reverse-to-forward alignment; normal route following is
    admitted only when none of those bounded handoffs remains active.
    """

    if dynamic_join_pending:
        return RouteCommandPhase.DYNAMIC_JOIN
    if start_egress_lock_index is not None:
        return RouteCommandPhase.START_EGRESS
    if start_egress_forward_alignment_index is not None:
        return RouteCommandPhase.REVERSE_EGRESS_ALIGNMENT
    return RouteCommandPhase.CERTIFIED_ROUTE


def dynamic_route_kind_transition_failure(
    current_route_kind: str, next_route_kind: str
) -> str:
    """Validate monotonic acquisition -> sampling -> physical handoffs."""

    if (
        current_route_kind == "stand_discovery_corridor"
        and next_route_kind == current_route_kind
    ):
        # A physical coverage blockage changes only the certified geometric
        # route.  The mission phase and committed inspection target stay the
        # same, so this is the only static-route hot handoff admitted here.
        return ""
    if not next_route_kind:
        return "missing dynamic route kind"
    if next_route_kind not in DYNAMIC_VIEWPOINT_ROUTE_KINDS:
        return f"unknown dynamic route kind: {next_route_kind}"
    if current_route_kind not in DYNAMIC_VIEWPOINT_ROUTE_KINDS:
        return f"unknown current dynamic route kind: {current_route_kind or '<missing>'}"
    if current_route_kind == next_route_kind:
        return ""
    if current_route_kind == "axis_acquisition" and next_route_kind in (
        {"viewpoint_sampling"} | DYNAMIC_PHYSICAL_ROUTE_KINDS
    ):
        return ""
    if (
        current_route_kind == "viewpoint_sampling"
        and next_route_kind in DYNAMIC_PHYSICAL_ROUTE_KINDS
    ):
        return ""
    if (
        current_route_kind in DYNAMIC_PHYSICAL_ROUTE_KINDS
        and next_route_kind in DYNAMIC_PHYSICAL_ROUTE_KINDS
    ):
        return ""
    return (
        "backward dynamic route phase transition: "
        f"{current_route_kind}->{next_route_kind}"
    )


def viewpoint_sampling_timeout_failure(
    *,
    route_kind: str,
    phase_started_at: float | None,
    now_monotonic: float,
    timeout_sec: float,
) -> str:
    if route_kind != "viewpoint_sampling":
        return ""
    if phase_started_at is None:
        return "viewpoint_sampling_clock_unavailable"
    if now_monotonic - phase_started_at >= timeout_sec:
        return "viewpoint_sampling_timeout"
    return ""


def viewpoint_sampling_target_timeout_failure(
    *,
    route_kind: str,
    target_started_at: float | None,
    now_monotonic: float,
    timeout_sec: float,
) -> str:
    failure = viewpoint_sampling_timeout_failure(
        route_kind=route_kind,
        phase_started_at=target_started_at,
        now_monotonic=now_monotonic,
        timeout_sec=timeout_sec,
    )
    return {
        "viewpoint_sampling_clock_unavailable": (
            "viewpoint_sampling_target_clock_unavailable"
        ),
        "viewpoint_sampling_timeout": "viewpoint_sampling_target_timeout",
    }.get(failure, failure)


def viewpoint_sampling_deadline_decision(
    *,
    route_kind: str,
    phase_started_at: float | None,
    target_started_at: float | None,
    now_monotonic: float,
    phase_timeout_sec: float,
    target_timeout_sec: float,
) -> ViewpointSamplingDeadlineDecision:
    """Evaluate total sampling time before the per-target deadline."""

    if route_kind != "viewpoint_sampling":
        return ViewpointSamplingDeadlineDecision("", None, None)
    phase_elapsed_sec = (
        None
        if phase_started_at is None
        else now_monotonic - phase_started_at
    )
    target_elapsed_sec = (
        None
        if target_started_at is None
        else now_monotonic - target_started_at
    )
    failure = viewpoint_sampling_timeout_failure(
        route_kind=route_kind,
        phase_started_at=phase_started_at,
        now_monotonic=now_monotonic,
        timeout_sec=phase_timeout_sec,
    )
    if not failure:
        failure = viewpoint_sampling_target_timeout_failure(
            route_kind=route_kind,
            target_started_at=target_started_at,
            now_monotonic=now_monotonic,
            timeout_sec=target_timeout_sec,
        )
    return ViewpointSamplingDeadlineDecision(
        failure,
        phase_elapsed_sec,
        target_elapsed_sec,
    )


def acquisition_goal_action(
    *,
    route_kind: str,
    provider_available: bool,
    hold_elapsed_sec: float,
    timeout_sec: float,
) -> AcquisitionGoalAction:
    """Decide whether a geometrically reached route is mission-terminal."""

    if route_kind not in INTERMEDIATE_ROUTE_KINDS:
        return AcquisitionGoalAction.COMPLETE
    if not provider_available:
        return AcquisitionGoalAction.MISSING_DYNAMIC_ROUTE_PROVIDER
    if route_kind == "axis_acquisition" and hold_elapsed_sec >= timeout_sec:
        return AcquisitionGoalAction.AXIS_ACQUISITION_TIMEOUT
    return AcquisitionGoalAction.HOLD_FOR_PHYSICAL_FACE


def acquisition_goal_decision(
    *,
    route_kind: str,
    provider_available: bool,
    hold_started_at: float | None,
    now_monotonic: float,
    timeout_sec: float,
) -> AcquisitionGoalDecision:
    """Initialize or reuse the hold clock and classify a reached goal."""

    effective_hold_started_at = (
        now_monotonic if hold_started_at is None else hold_started_at
    )
    hold_elapsed_sec = now_monotonic - effective_hold_started_at
    return AcquisitionGoalDecision(
        action=acquisition_goal_action(
            route_kind=route_kind,
            provider_available=provider_available,
            hold_elapsed_sec=hold_elapsed_sec,
            timeout_sec=timeout_sec,
        ),
        hold_started_at=effective_hold_started_at,
        hold_elapsed_sec=hold_elapsed_sec,
    )
