"""ROS-free result and stop-detail construction for the control loop.

The runtime loop owns ordering and side effects.  These helpers only assemble
immutable return values and diagnostic dictionaries so every stop path uses a
consistent contract without gaining access to the follower node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from scripts.aufgabe04.navigation.control.follower_models import FollowerResult
from scripts.aufgabe04.navigation.control.waypoint_controller import (
    ControllerStep,
)
from scripts.aufgabe04.navigation.execution.execution_route_certificate import (
    ExecutionRouteCheck,
)


@dataclass(frozen=True)
class CertifiedCornerStopEvidence:
    """Prepared corner-stop evidence without runtime side effects."""

    step: ControllerStep
    stop_details: Mapping[str, object]
    route_check: ExecutionRouteCheck | None = None


def noop_result(reason: str) -> FollowerResult:
    """Build the zero-duration result for a route with no motion segment."""

    return FollowerResult(
        status="noop",
        stop_reason=reason,
        duration_sec=0.0,
        distance_estimate_m=0.0,
        motion_published=False,
    )


def control_result(
    status: str,
    stop_reason: str,
    *,
    started_at: float,
    now_monotonic: float,
    distance_estimate_m: float,
    motion_published: bool,
    stop_details: Mapping[str, object] | None = None,
) -> FollowerResult:
    """Build one follower result from an explicit runtime snapshot."""

    return FollowerResult(
        status=status,
        stop_reason=stop_reason,
        duration_sec=now_monotonic - started_at,
        distance_estimate_m=distance_estimate_m,
        motion_published=motion_published,
        stop_details=stop_details,
    )


def initial_runtime_input_stop_details(
    upstream_details: Mapping[str, object] | None,
    *,
    reason: str,
    motion_published: bool,
) -> dict[str, object]:
    """Preserve upstream startup evidence and add missing phase markers."""

    details = dict(upstream_details or {})
    if not details:
        details = {
            "reason": reason,
            "source": "initial_runtime_input_wait",
            "fail_closed": True,
        }
    details.setdefault("execution_phase", "before_motion")
    details.setdefault("phase", "initial_runtime_input_wait")
    details.setdefault("motion_published", bool(motion_published))
    return details


def ros_shutdown_stop_details() -> dict[str, object]:
    """Build the explicit result contract for a stopped ROS context."""

    return {
        "reason": "ROS shutdown",
        "source": "rclpy",
        "phase": "control_loop",
        "fail_closed": True,
    }


def with_controller_trace_failure(
    primary_details: Mapping[str, object],
    trace_failure: str,
    *,
    fail_closed: bool | None = None,
) -> dict[str, object]:
    """Attach a secondary trace fault without replacing the primary stop."""

    details = {
        **dict(primary_details),
        "controller_trace_error": trace_failure,
        "controller_trace_fault_code": "controller_trace_write_failed",
    }
    if fail_closed is not None:
        details["fail_closed"] = fail_closed
    return details


def with_route_check_error(
    primary_details: Mapping[str, object],
    error: BaseException,
) -> dict[str, object]:
    """Attach a secondary route-certificate diagnostic exception."""

    return {
        **dict(primary_details),
        "route_check_error": str(error),
        "route_check_error_type": error.__class__.__name__,
    }


def viewpoint_sampling_timeout_stop_details(
    *,
    reason: str,
    route_kind: str,
    phase_elapsed_sec: float | None,
    target_elapsed_sec: float | None,
    phase_timeout_sec: float,
    target_timeout_sec: float,
) -> dict[str, object]:
    """Build evidence for total or per-target viewpoint-sampling timeout."""

    return {
        "reason": reason,
        "route_kind": route_kind,
        "phase_elapsed_sec": phase_elapsed_sec,
        "target_elapsed_sec": target_elapsed_sec,
        "phase_timeout_sec": phase_timeout_sec,
        "target_timeout_sec": target_timeout_sec,
        "fail_closed": True,
    }


def certified_static_start_stop_details(
    route_check_details: Mapping[str, object],
    *,
    certificate_reason: str,
) -> dict[str, object]:
    """Build evidence when the live pose misses the certified start segment."""

    return {
        **dict(route_check_details),
        "reason": "pose outside certified startup segment",
        "certificate_reason": certificate_reason,
        "startup_target_candidates": [0, 1],
        "source": "execution_route_certificate",
        "fail_closed": True,
    }


def certified_corner_stop_details(
    *,
    reason: str,
    route_kind: str,
    target_index: int,
    pursuit_index: int,
    distance_to_vertex_m: float,
    release_tolerance_m: float,
    hold_tolerance_m: float,
    tracking_tube_radius_m: float,
    reacquire_attempts: int,
    max_reacquire_attempts: int,
) -> dict[str, object]:
    """Build the certified-corner fail-closed evidence contract."""

    return {
        "reason": reason,
        "source": "execution_route_certificate",
        "route_kind": route_kind,
        "target_index": target_index,
        "pursuit_index": pursuit_index,
        "distance_to_vertex_m": distance_to_vertex_m,
        "release_tolerance_m": release_tolerance_m,
        "hold_tolerance_m": hold_tolerance_m,
        "tracking_tube_radius_m": tracking_tube_radius_m,
        "reacquire_attempts": reacquire_attempts,
        "max_reacquire_attempts": max_reacquire_attempts,
        "fail_closed": True,
    }


def intermediate_terminal_heading_stop_details(
    *,
    reason: str,
    route_kind: str,
    target_index: int,
    distance_to_target_m: float,
    entry_tolerance_m: float,
    hold_tolerance_m: float,
    distance_comparison_epsilon_m: float,
    hold_diagnostics: Mapping[str, object],
) -> dict[str, object]:
    """Build evidence for an intermediate terminal-heading hold failure."""

    return {
        "reason": reason,
        "fault_code": reason,
        "route_kind": route_kind,
        "target_index": target_index,
        "distance_to_target_m": distance_to_target_m,
        "entry_tolerance_m": entry_tolerance_m,
        "hold_tolerance_m": hold_tolerance_m,
        "distance_comparison_epsilon_m": distance_comparison_epsilon_m,
        "effective_hold_limit_m": (
            hold_tolerance_m + distance_comparison_epsilon_m
        ),
        **dict(hold_diagnostics),
        "fail_closed": True,
    }


def acquisition_goal_stop_details(
    *,
    reason: str,
    route_kind: str,
    hold_elapsed_sec: float,
    timeout_sec: float,
) -> dict[str, object]:
    """Build evidence when acquisition cannot transition to completion."""

    return {
        "reason": reason,
        "route_kind": route_kind,
        "hold_elapsed_sec": hold_elapsed_sec,
        "timeout_sec": timeout_sec,
        "fail_closed": True,
    }


def waypoint_timeout_stop_details(
    *,
    reason: str,
    route_kind: str,
    elapsed_sec: float,
    timeout_sec: float,
    target_index: int,
    pursuit_index: int,
    distance_to_target_m: float,
    progress_mode: str,
    axis_acquisition_target_revision: object,
    viewpoint_sampling_target_revision: object,
    robot_x_m: float,
    robot_y_m: float,
    robot_yaw_rad: float,
) -> dict[str, object]:
    """Build the evidence contract for a waypoint timeout."""

    return {
        "reason": reason,
        "route_kind": route_kind,
        "elapsed_sec": elapsed_sec,
        "timeout_sec": timeout_sec,
        "target_index": target_index,
        "pursuit_index": pursuit_index,
        "distance_to_target_m": distance_to_target_m,
        "progress_mode": progress_mode,
        "axis_acquisition_target_revision": axis_acquisition_target_revision,
        "viewpoint_sampling_target_revision": (
            viewpoint_sampling_target_revision
        ),
        "robot_pose": {
            "x_m": robot_x_m,
            "y_m": robot_y_m,
            "yaw_rad": robot_yaw_rad,
        },
        "fail_closed": True,
    }


def terminal_heading_timeout_stop_details(
    *,
    reason: str,
    route_kind: str,
    waypoint_elapsed_sec: float,
    waypoint_timeout_sec: float,
    terminal_heading_elapsed_sec: float,
    terminal_heading_timeout_sec: float,
    terminal_heading_entry_waypoint_elapsed_sec: float,
    target_index: int,
    pursuit_index: int,
    distance_to_target_m: float,
    progress_mode: str,
    controlled_heading_error_rad: float,
    robot_x_m: float,
    robot_y_m: float,
    robot_yaw_rad: float,
) -> dict[str, object]:
    """Build explicit evidence for the bounded final-heading deadline."""

    return {
        "reason": reason,
        "fault_code": "terminal_heading_timeout",
        "phase": "terminal_heading",
        "timeout_scope": "terminal_heading_phase",
        "route_kind": route_kind,
        "waypoint_elapsed_sec": waypoint_elapsed_sec,
        "waypoint_timeout_sec": waypoint_timeout_sec,
        "terminal_heading_elapsed_sec": terminal_heading_elapsed_sec,
        "terminal_heading_timeout_sec": terminal_heading_timeout_sec,
        "terminal_heading_entry_waypoint_elapsed_sec": (
            terminal_heading_entry_waypoint_elapsed_sec
        ),
        "target_index": target_index,
        "pursuit_index": pursuit_index,
        "distance_to_target_m": distance_to_target_m,
        "progress_mode": progress_mode,
        "controlled_heading_error_rad": controlled_heading_error_rad,
        "robot_pose": {
            "x_m": robot_x_m,
            "y_m": robot_y_m,
            "yaw_rad": robot_yaw_rad,
        },
        "fail_closed": True,
    }


def clearance_motion_floor_stop_details(
    *,
    reason: str,
    command_floor_details: Mapping[str, object],
    front_clearance_scale: float,
    front_clearance_details: Mapping[str, object] | None,
    target_index: int,
    pursuit_index: int,
    distance_to_target_m: float,
    progress_mode: str,
) -> dict[str, object]:
    """Build evidence for a clearance-scaled command requiring zero hold."""

    return {
        "reason": reason,
        "source": "linear_motion_floor",
        **dict(command_floor_details),
        "front_clearance_scale": front_clearance_scale,
        "front_clearance": dict(front_clearance_details or {}),
        "target_index": target_index,
        "pursuit_index": pursuit_index,
        "distance_to_target_m": distance_to_target_m,
        "progress_mode": progress_mode,
        "fail_closed": True,
    }


def nonfinite_velocity_stop_details(
    *,
    linear_x_mps: float,
    angular_z_radps: float,
) -> dict[str, object]:
    """Build the fail-closed evidence for a malformed controller command."""

    return {
        "reason": "controller produced a non-finite velocity command",
        "fault_code": "nonfinite_velocity_command",
        "linear_x_mps": linear_x_mps,
        "angular_z_radps": angular_z_radps,
        "fail_closed": True,
    }
