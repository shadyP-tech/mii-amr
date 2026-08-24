"""Route phase and bounded hold decisions for dynamic follower handoffs."""

from __future__ import annotations

from scripts.aufgabe04.navigation.control.driving_behavior import (
    DYNAMIC_PHYSICAL_ROUTE_KINDS,
    DYNAMIC_VIEWPOINT_ROUTE_KINDS,
    INTERMEDIATE_ROUTE_KINDS,
)


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


def acquisition_goal_action(
    *,
    route_kind: str,
    provider_available: bool,
    hold_elapsed_sec: float,
    timeout_sec: float,
) -> str:
    """Decide whether a geometrically reached route is mission-terminal."""

    if route_kind not in INTERMEDIATE_ROUTE_KINDS:
        return "complete"
    if not provider_available:
        return "missing_dynamic_route_provider"
    if route_kind == "axis_acquisition" and hold_elapsed_sec >= timeout_sec:
        return "axis_acquisition_timeout"
    return "hold_for_physical_face"
