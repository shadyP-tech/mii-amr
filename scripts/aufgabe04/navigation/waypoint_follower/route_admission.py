"""Pure fail-closed route-admission diagnostics for the follower runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from scripts.aufgabe04.navigation.execution.execution_route_certificate import (
    ExecutionRouteCheck,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.waypoint_follower.directives import (
    StartupJoinAction,
    StringDirective,
)


class ExecutionRouteAdmissionStatus(StringDirective):
    """Whether route-tube checking was skipped, passed, or stopped."""

    SKIPPED = "skipped"
    ADMITTED = "admitted"
    STOP = "stop"


@dataclass(frozen=True)
class ExecutionRouteAdmissionDecision:
    """Typed route-tube result without runtime effects or trace I/O."""

    status: ExecutionRouteAdmissionStatus
    route_check: ExecutionRouteCheck | None = None
    stop_details: Mapping[str, object] | None = None


def dynamic_join_envelope_failure(
    pose: Pose2D,
    anchor: Pose2D,
    effective_join_limit_m: float | None,
) -> dict[str, object] | None:
    """Fail closed if a live pose leaves or cannot define the certified join disk."""

    if not all(math.isfinite(value) for value in (pose.x_m, pose.y_m, pose.yaw_rad)):
        return {
            "reason": "current robot pose is non-finite during dynamic-route join",
            "fault_code": "invalid_current_pose",
            "fail_closed": True,
        }
    if (
        effective_join_limit_m is None
        or not math.isfinite(effective_join_limit_m)
        or effective_join_limit_m <= 0.0
    ):
        return {
            "reason": "dynamic-route join envelope is invalid",
            "fault_code": "invalid_route_update",
            "fail_closed": True,
            "effective_join_limit_m": effective_join_limit_m,
        }
    join_distance = math.hypot(pose.x_m - anchor.x_m, pose.y_m - anchor.y_m)
    if not math.isfinite(join_distance):
        return {
            "reason": "dynamic-route join distance is non-finite",
            "fault_code": "invalid_current_pose",
            "fail_closed": True,
        }
    if join_distance > effective_join_limit_m:
        return {
            "reason": "robot left the certified dynamic-route join envelope",
            "fault_code": "join_envelope_exceeded",
            "fail_closed": True,
            "join_distance_m": join_distance,
            "effective_join_limit_m": effective_join_limit_m,
        }
    return None


def certified_startup_join_action(
    pose: Pose2D,
    anchor: Pose2D,
    effective_join_limit_m: float | None,
    join_tolerance_m: float,
) -> tuple[StartupJoinAction, dict[str, object] | None]:
    """Select only stop, anchor pursuit, or the anchor-complete zero cycle."""

    failure = dynamic_join_envelope_failure(
        pose,
        anchor,
        effective_join_limit_m,
    )
    if failure is not None:
        return StartupJoinAction.STOP, failure
    if not math.isfinite(join_tolerance_m) or join_tolerance_m <= 0.0:
        return StartupJoinAction.STOP, {
            "reason": "dynamic-route join tolerance is invalid",
            "fault_code": "invalid_route_update",
            "fail_closed": True,
        }
    distance_m = math.hypot(pose.x_m - anchor.x_m, pose.y_m - anchor.y_m)
    return (
        (StartupJoinAction.ZERO, None)
        if distance_m <= join_tolerance_m
        else (StartupJoinAction.ANCHOR, None)
    )


def stuck_progress_details(
    *,
    target_index: int,
    distance_to_target_m: float,
    last_progress_distance_m: float,
    elapsed_without_progress_sec: float,
    max_without_progress_sec: float,
    progress_epsilon_m: float,
    commanded_linear_x_mps: float,
    commanded_angular_z_radps: float,
    front_clearance_scale: float,
    effective_linear_x_mps: float,
    front_clearance_details: dict[str, object] | None = None,
    pursuit_index: int | None = None,
    controlled_heading_error_rad: float | None = None,
    last_progress_heading_error_rad: float | None = None,
    heading_progress_epsilon_rad: float | None = None,
    last_progress_target_index: int | None = None,
    last_progress_pursuit_index: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "stop_reason": "stuck no progress",
        "source": "progress_monitor",
        "target_index": target_index,
        "distance_to_target_m": distance_to_target_m,
        "last_progress_distance_m": last_progress_distance_m,
        "elapsed_without_progress_sec": elapsed_without_progress_sec,
        "max_without_progress_sec": max_without_progress_sec,
        "progress_epsilon_m": progress_epsilon_m,
        "commanded_linear_x_mps": commanded_linear_x_mps,
        "commanded_angular_z_radps": commanded_angular_z_radps,
        "front_clearance_scale": front_clearance_scale,
        "effective_linear_x_mps": effective_linear_x_mps,
        "pursuit_index": pursuit_index,
        "controlled_heading_error_rad": controlled_heading_error_rad,
        "last_progress_heading_error_rad": last_progress_heading_error_rad,
        "heading_progress_epsilon_rad": heading_progress_epsilon_rad,
        "last_progress_target_index": last_progress_target_index,
        "last_progress_pursuit_index": last_progress_pursuit_index,
    }
    if front_clearance_details is not None:
        payload["front_clearance"] = front_clearance_details
    return payload
