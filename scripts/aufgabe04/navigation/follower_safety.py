"""Pure safety decisions for Aufgabe 04 waypoint follower adapters."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from scripts.aufgabe04.navigation.models import Pose2D


NON_ALLOWLISTABLE_DIRECT_CMD_VEL_NODES = {
    "behavior_server",
    "controller_server",
    "velocity_smoother",
}


def is_non_allowlistable_direct_cmd_vel_publisher(identity: str) -> bool:
    return identity.rstrip("/").rsplit("/", 1)[-1] in NON_ALLOWLISTABLE_DIRECT_CMD_VEL_NODES


def message_freshness_failure(
    name: str,
    *,
    has_message: bool,
    receipt_age_sec: float | None,
    header_age_sec: float | None,
    max_age_sec: float,
) -> str:
    if not has_message or receipt_age_sec is None or header_age_sec is None:
        return f"missing {name}"
    if receipt_age_sec > max_age_sec or header_age_sec > max_age_sec:
        return f"stale {name}"
    return ""


def finite_positive_min(ranges: Iterable[float]) -> float | None:
    finite_ranges = [value for value in ranges if math.isfinite(value) and value > 0.0]
    if not finite_ranges:
        return None
    return min(finite_ranges)


def obstacle_failure(ranges: Sequence[float] | None, min_obstacle_distance_m: float) -> str:
    if not ranges:
        return ""
    nearest = finite_positive_min(ranges)
    if nearest is not None and nearest < min_obstacle_distance_m:
        return "obstacle too close"
    return ""


def initial_pose_failure(
    pose: Pose2D,
    first_waypoint: Pose2D,
    initial_distance_limit_m: float,
) -> str:
    distance = math.hypot(pose.x_m - first_waypoint.x_m, pose.y_m - first_waypoint.y_m)
    if distance > initial_distance_limit_m:
        return "initial pose too far from first waypoint"
    return ""


def waypoint_timeout_failure(elapsed_sec: float, timeout_sec: float) -> str:
    if elapsed_sec > timeout_sec:
        return "waypoint timeout"
    return ""


def startup_readiness_failure(
    *,
    scan_ready: bool,
    odom_ready: bool,
    pose_ready: bool,
) -> str:
    missing = [
        name
        for name, ready in (("scan", scan_ready), ("odom", odom_ready), ("pose", pose_ready))
        if not ready
    ]
    if missing:
        return f"startup timeout waiting for {', '.join(missing)}"
    return ""


def rotation_progress_failure(
    *,
    rotation_elapsed_sec: float,
    no_progress_elapsed_sec: float,
    max_rotation_sec: float,
    max_no_progress_sec: float,
) -> str:
    if rotation_elapsed_sec > max_rotation_sec:
        return "rotation timeout"
    if no_progress_elapsed_sec > max_no_progress_sec:
        return "rotation stalled: heading error not decreasing"
    return ""


def cmd_vel_ownership_failure(
    publisher_identities: Sequence[str],
    self_identity: str,
    allowed_external_identities: Sequence[str] = (),
) -> str:
    allowed = {
        identity
        for identity in allowed_external_identities
        if not is_non_allowlistable_direct_cmd_vel_publisher(identity)
    }
    external = sorted(
        {
            identity
            for identity in publisher_identities
            if identity != self_identity and identity not in allowed
        }
    )
    if external:
        return f"external cmd_vel publisher during run: {', '.join(external)}"
    return ""
