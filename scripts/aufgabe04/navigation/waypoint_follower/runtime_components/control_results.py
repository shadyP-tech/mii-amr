"""ROS-free result and stop-detail construction for the control loop.

The runtime loop owns ordering and side effects.  These helpers only assemble
immutable return values and diagnostic dictionaries so every stop path uses a
consistent contract without gaining access to the follower node.
"""

from __future__ import annotations

from typing import Mapping

from scripts.aufgabe04.navigation.control.follower_models import FollowerResult


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
