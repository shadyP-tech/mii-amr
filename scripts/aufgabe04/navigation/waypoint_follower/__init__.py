"""Modular waypoint-follower contracts with a lazily loaded ROS runtime."""

from __future__ import annotations

from importlib import import_module

from scripts.aufgabe04.navigation.control.driving_behavior import (
    STATIC_PHYSICAL_ROUTE_KINDS,
    controller_config_for_route_kind,
)
from scripts.aufgabe04.navigation.control.follower_models import FollowerResult
from scripts.aufgabe04.navigation.waypoint_follower.config import FollowerConfig
from scripts.aufgabe04.navigation.waypoint_follower.pose_lookup import (
    PoseLookupResult,
    tf_lookup_failure_details,
)
from scripts.aufgabe04.navigation.waypoint_follower.route_admission import (
    certified_startup_join_action,
    dynamic_join_envelope_failure,
    stuck_progress_details,
)
from scripts.aufgabe04.navigation.waypoint_follower.route_phases import (
    acquisition_goal_action,
    dynamic_route_kind_transition_failure,
    viewpoint_sampling_target_timeout_failure,
    viewpoint_sampling_timeout_failure,
)
from scripts.aufgabe04.navigation.waypoint_follower.startup import (
    CertifiedStaticStartupDecision,
    CertifiedStartupRouteState,
    certified_startup_route_state,
    certified_static_startup_decision,
)
from scripts.aufgabe04.navigation.waypoint_follower.terminal_heading import (
    INTERMEDIATE_TERMINAL_HEADING_HOLD_EXCEEDED,
    IntermediateTerminalHeadingDecision,
    IntermediateTerminalHeadingLatch,
    compute_intermediate_terminal_heading_command,
    intermediate_terminal_heading_entry_tolerance_m,
    intermediate_terminal_heading_hold_diagnostics,
    reset_intermediate_terminal_heading_latch,
)
from scripts.aufgabe04.navigation.approach.viewpoint_sampling_contract import (
    INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M,
    INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M,
)


_RUNTIME_EXPORTS = frozenset(
    {
        "CALLBACK_SERVICE_BACKGROUND_EXECUTOR",
        "FOLLOWER_EXECUTOR_NUM_THREADS",
        "STALE_TF_RECOVERY_MAX_CALLBACKS",
        "STALE_TF_RECOVERY_MAX_DURATION_SEC",
        "STALE_TF_RECOVERY_SPIN_TIMEOUT_SEC",
        "SimpleWaypointFollowerNode",
        "TF_LISTENER_NODE_NAME",
        "run_simple_waypoint_follower",
    }
)


def __getattr__(name: str):
    if name not in _RUNTIME_EXPORTS:
        raise AttributeError(name)
    runtime = import_module(
        "scripts.aufgabe04.navigation.waypoint_follower.runtime"
    )
    value = getattr(runtime, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_RUNTIME_EXPORTS))


__all__ = [
    "CALLBACK_SERVICE_BACKGROUND_EXECUTOR",
    "FOLLOWER_EXECUTOR_NUM_THREADS",
    "FollowerConfig",
    "FollowerResult",
    "CertifiedStartupRouteState",
    "CertifiedStaticStartupDecision",
    "INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M",
    "INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M",
    "IntermediateTerminalHeadingLatch",
    "IntermediateTerminalHeadingDecision",
    "INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M",
    "INTERMEDIATE_TERMINAL_HEADING_HOLD_EXCEEDED",
    "INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M",
    "PoseLookupResult",
    "STALE_TF_RECOVERY_MAX_CALLBACKS",
    "STALE_TF_RECOVERY_MAX_DURATION_SEC",
    "STALE_TF_RECOVERY_SPIN_TIMEOUT_SEC",
    "STATIC_PHYSICAL_ROUTE_KINDS",
    "SimpleWaypointFollowerNode",
    "TF_LISTENER_NODE_NAME",
    "acquisition_goal_action",
    "certified_startup_join_action",
    "certified_startup_route_state",
    "certified_static_startup_decision",
    "compute_intermediate_terminal_heading_command",
    "controller_config_for_route_kind",
    "dynamic_join_envelope_failure",
    "dynamic_route_kind_transition_failure",
    "intermediate_terminal_heading_entry_tolerance_m",
    "intermediate_terminal_heading_hold_diagnostics",
    "reset_intermediate_terminal_heading_latch",
    "run_simple_waypoint_follower",
    "stuck_progress_details",
    "tf_lookup_failure_details",
    "viewpoint_sampling_target_timeout_failure",
    "viewpoint_sampling_timeout_failure",
]
