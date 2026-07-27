"""Fail-closed shared-course checks with swept circular footprints."""

import math
from typing import Optional, Tuple

from .models import Conflict
from .robot_status import RobotStatus


def _finite(value: Optional[float]) -> bool:
    return value is not None and math.isfinite(value)


def status_safety_issue(
    status: RobotStatus,
    *,
    now_sec: Optional[float] = None,
    max_status_age_sec: Optional[float] = None,
    require_velocity: bool = False,
    future_tolerance_sec: float = 0.0,
) -> Optional[str]:
    """Return why a peer status is unsafe, or ``None`` when usable."""

    if not status.robot_id.strip():
        return "missing robot id"
    if not _finite(status.x_m) or not _finite(status.y_m):
        return "missing or invalid peer position"
    if require_velocity and (
        not _finite(status.velocity_x_mps) or not _finite(status.velocity_y_mps)
    ):
        return "missing or invalid peer velocity"
    if future_tolerance_sec < 0.0 or not math.isfinite(future_tolerance_sec):
        raise ValueError("future_tolerance_sec must be finite and non-negative")
    if (now_sec is None) != (max_status_age_sec is None):
        raise ValueError("now_sec and max_status_age_sec must be provided together")
    if now_sec is not None:
        if not math.isfinite(now_sec) or now_sec < 0.0:
            raise ValueError("now_sec must be finite and non-negative")
        if (
            max_status_age_sec is None
            or not math.isfinite(max_status_age_sec)
            or max_status_age_sec < 0.0
        ):
            raise ValueError("max_status_age_sec must be finite and non-negative")
        if not _finite(status.timestamp_sec):
            return "missing or invalid peer timestamp"
        assert status.timestamp_sec is not None
        if status.timestamp_sec > now_sec + future_tolerance_sec:
            return "peer timestamp is in the future"
        if now_sec - status.timestamp_sec > max_status_age_sec:
            return "stale peer status"
    return None


def _effective_radius(
    status: RobotStatus, *, default_footprint_radius_m: float
) -> Tuple[Optional[float], Optional[str]]:
    if status.payload_loaded:
        radius = status.loaded_footprint_radius_m
        if not _finite(radius) or radius is None or radius <= 0.0:
            return None, "loaded peer has no valid loaded footprint"
        if (
            status.footprint_radius_m is not None
            and (
                not math.isfinite(status.footprint_radius_m)
                or radius < status.footprint_radius_m
            )
        ):
            return None, "loaded footprint is smaller than base footprint"
        return radius, None
    radius = status.footprint_radius_m
    if radius is None:
        radius = default_footprint_radius_m
    if not math.isfinite(radius) or radius < 0.0:
        return None, "invalid peer footprint"
    return radius, None


def detect_close_robot_conflict(
    first: RobotStatus,
    second: RobotStatus,
    *,
    min_separation_m: float,
    now_sec: Optional[float] = None,
    max_status_age_sec: Optional[float] = None,
    prediction_horizon_sec: float = 0.0,
    default_footprint_radius_m: float = 0.0,
    future_tolerance_sec: float = 0.0,
) -> Optional[Conflict]:
    """Detect current or predicted overlap and fail closed on unusable state.

    ``min_separation_m`` is clearance in addition to both footprint radii.  A
    positive prediction horizon checks the minimum distance between the two
    synchronously swept footprint centers under constant velocity.
    """

    for name, value in (
        ("min_separation_m", min_separation_m),
        ("prediction_horizon_sec", prediction_horizon_sec),
        ("default_footprint_radius_m", default_footprint_radius_m),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if first.robot_id == second.robot_id:
        return Conflict(
            first.robot_id,
            second.robot_id,
            "duplicate robot id",
            fail_closed=True,
        )
    require_velocity = prediction_horizon_sec > 0.0
    for status in (first, second):
        issue = status_safety_issue(
            status,
            now_sec=now_sec,
            max_status_age_sec=max_status_age_sec,
            require_velocity=require_velocity,
            future_tolerance_sec=future_tolerance_sec,
        )
        if issue is not None:
            return Conflict(
                first.robot_id,
                second.robot_id,
                f"{status.robot_id or 'unknown peer'}: {issue}",
                fail_closed=True,
            )

    first_radius, first_radius_issue = _effective_radius(
        first, default_footprint_radius_m=default_footprint_radius_m
    )
    second_radius, second_radius_issue = _effective_radius(
        second, default_footprint_radius_m=default_footprint_radius_m
    )
    radius_issue = first_radius_issue or second_radius_issue
    if radius_issue is not None:
        return Conflict(
            first.robot_id,
            second.robot_id,
            radius_issue,
            fail_closed=True,
        )
    assert first.x_m is not None and first.y_m is not None
    assert second.x_m is not None and second.y_m is not None
    assert first_radius is not None and second_radius is not None
    relative_x = first.x_m - second.x_m
    relative_y = first.y_m - second.y_m
    closest_time_sec = 0.0
    if prediction_horizon_sec > 0.0:
        assert first.velocity_x_mps is not None and first.velocity_y_mps is not None
        assert second.velocity_x_mps is not None and second.velocity_y_mps is not None
        relative_vx = first.velocity_x_mps - second.velocity_x_mps
        relative_vy = first.velocity_y_mps - second.velocity_y_mps
        speed_sq = relative_vx * relative_vx + relative_vy * relative_vy
        if speed_sq > 1e-18:
            closest_time_sec = max(
                0.0,
                min(
                    prediction_horizon_sec,
                    -(relative_x * relative_vx + relative_y * relative_vy)
                    / speed_sq,
                ),
            )
        relative_x += relative_vx * closest_time_sec
        relative_y += relative_vy * closest_time_sec

    distance = math.hypot(relative_x, relative_y)
    required = min_separation_m + first_radius + second_radius
    if distance < required:
        return Conflict(
            first.robot_id,
            second.robot_id,
            (
                f"swept separation {distance:.3f} m below {required:.3f} m "
                f"at +{closest_time_sec:.3f} s"
            ),
            closest_separation_m=distance,
            required_separation_m=required,
        )
    return None
