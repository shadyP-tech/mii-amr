"""Pure safety decisions for Aufgabe 04 waypoint follower adapters."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from scripts.aufgabe04.navigation.models import Pose2D


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


def cmd_vel_ownership_failure(
    publisher_identities: Sequence[str],
    self_identity: str,
) -> str:
    external = sorted({identity for identity in publisher_identities if identity != self_identity})
    if external:
        return f"external cmd_vel publisher during run: {', '.join(external)}"
    return ""
