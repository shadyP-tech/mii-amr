"""Pure safety decisions for Aufgabe 04 waypoint follower adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from scripts.aufgabe04.navigation.models import Pose2D


NO_VALID_SCAN_RANGES = "no valid scan ranges"
NO_VALID_FRONT_SECTOR_SCAN_RANGES = "no valid front-sector scan ranges"
OBSTACLE_TOO_CLOSE = "obstacle too close"


@dataclass(frozen=True)
class RangeSummary:
    nearest_valid_range_m: float | None
    valid_sample_count: int
    rejected_below_min_count: int
    rejected_above_max_count: int
    rejected_non_finite_count: int
    range_min_m: float | None
    range_max_m: float | None


@dataclass(frozen=True)
class ObstacleDecision:
    stop_reason: str
    nearest_valid_range_m: float | None
    valid_sample_count: int
    rejected_below_min_count: int
    rejected_above_max_count: int
    rejected_non_finite_count: int
    threshold_m: float
    range_min_m: float | None
    range_max_m: float | None
    source: str

    def to_log_dict(self) -> dict[str, object]:
        return {
            "stop_reason": self.stop_reason,
            "nearest_valid_range_m": self.nearest_valid_range_m,
            "valid_sample_count": self.valid_sample_count,
            "rejected_below_min_count": self.rejected_below_min_count,
            "rejected_above_max_count": self.rejected_above_max_count,
            "rejected_non_finite_count": self.rejected_non_finite_count,
            "threshold_m": self.threshold_m,
            "range_min_m": self.range_min_m,
            "range_max_m": self.range_max_m,
            "source": self.source,
        }


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


def _normal_bound(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value


def _scan_range_is_valid(
    value: float,
    range_min_m: float | None,
    range_max_m: float | None,
) -> bool:
    if not math.isfinite(value):
        return False
    lower = _normal_bound(range_min_m)
    upper = _normal_bound(range_max_m)
    if lower is not None:
        if value < lower:
            return False
    elif value <= 0.0:
        return False
    if upper is not None and value > upper:
        return False
    return True


def summarize_valid_ranges(
    ranges: Iterable[float],
    *,
    range_min_m: float | None = None,
    range_max_m: float | None = None,
) -> RangeSummary:
    lower = _normal_bound(range_min_m)
    upper = _normal_bound(range_max_m)
    valid: list[float] = []
    rejected_below_min_count = 0
    rejected_above_max_count = 0
    rejected_non_finite_count = 0

    for raw_value in ranges:
        value = float(raw_value)
        if not math.isfinite(value):
            rejected_non_finite_count += 1
            continue
        if lower is not None:
            if value < lower:
                rejected_below_min_count += 1
                continue
        elif value <= 0.0:
            rejected_below_min_count += 1
            continue
        if upper is not None and value > upper:
            rejected_above_max_count += 1
            continue
        valid.append(value)

    return RangeSummary(
        nearest_valid_range_m=min(valid) if valid else None,
        valid_sample_count=len(valid),
        rejected_below_min_count=rejected_below_min_count,
        rejected_above_max_count=rejected_above_max_count,
        rejected_non_finite_count=rejected_non_finite_count,
        range_min_m=lower,
        range_max_m=upper,
    )


def obstacle_decision(
    ranges: Sequence[float] | None,
    min_obstacle_distance_m: float,
    *,
    range_min_m: float | None = None,
    range_max_m: float | None = None,
    source: str = "global_scan",
) -> ObstacleDecision:
    summary = summarize_valid_ranges(
        ranges or (),
        range_min_m=range_min_m,
        range_max_m=range_max_m,
    )
    stop_reason = ""
    if summary.valid_sample_count == 0:
        stop_reason = NO_VALID_SCAN_RANGES
    elif (
        summary.nearest_valid_range_m is not None
        and summary.nearest_valid_range_m < min_obstacle_distance_m
    ):
        stop_reason = OBSTACLE_TOO_CLOSE
    return ObstacleDecision(
        stop_reason=stop_reason,
        nearest_valid_range_m=summary.nearest_valid_range_m,
        valid_sample_count=summary.valid_sample_count,
        rejected_below_min_count=summary.rejected_below_min_count,
        rejected_above_max_count=summary.rejected_above_max_count,
        rejected_non_finite_count=summary.rejected_non_finite_count,
        threshold_m=min_obstacle_distance_m,
        range_min_m=summary.range_min_m,
        range_max_m=summary.range_max_m,
        source=source,
    )


def obstacle_failure(
    ranges: Sequence[float] | None,
    min_obstacle_distance_m: float,
    *,
    range_min_m: float | None = None,
    range_max_m: float | None = None,
) -> str:
    if ranges is None:
        return ""
    return obstacle_decision(
        ranges,
        min_obstacle_distance_m,
        range_min_m=range_min_m,
        range_max_m=range_max_m,
    ).stop_reason


def sector_min_distance(
    ranges: Sequence[float] | None,
    angle_min_rad: float,
    angle_increment_rad: float,
    center_rad: float,
    half_width_rad: float,
    range_min_m: float | None = None,
    range_max_m: float | None = None,
) -> float | None:
    if not ranges or angle_increment_rad == 0.0:
        return None
    values: list[float] = []
    for index, value in enumerate(ranges):
        if not _scan_range_is_valid(value, range_min_m, range_max_m):
            continue
        angle = angle_min_rad + index * angle_increment_rad
        delta = math.atan2(math.sin(angle - center_rad), math.cos(angle - center_rad))
        if abs(delta) <= half_width_rad:
            values.append(value)
    if not values:
        return None
    return min(values)


def front_sector_decision(
    ranges: Sequence[float] | None,
    angle_min_rad: float,
    angle_increment_rad: float,
    center_rad: float,
    half_width_rad: float,
    stop_distance_m: float,
    *,
    range_min_m: float | None = None,
    range_max_m: float | None = None,
) -> ObstacleDecision:
    if not ranges or angle_increment_rad == 0.0:
        sector_ranges: list[float] = []
    else:
        sector_ranges = []
        for index, value in enumerate(ranges):
            angle = angle_min_rad + index * angle_increment_rad
            delta = math.atan2(math.sin(angle - center_rad), math.cos(angle - center_rad))
            if abs(delta) <= half_width_rad:
                sector_ranges.append(value)
    summary = summarize_valid_ranges(
        sector_ranges,
        range_min_m=range_min_m,
        range_max_m=range_max_m,
    )
    stop_reason = ""
    if summary.valid_sample_count == 0:
        stop_reason = NO_VALID_FRONT_SECTOR_SCAN_RANGES
    elif (
        summary.nearest_valid_range_m is not None
        and summary.nearest_valid_range_m <= stop_distance_m
    ):
        stop_reason = OBSTACLE_TOO_CLOSE
    return ObstacleDecision(
        stop_reason=stop_reason,
        nearest_valid_range_m=summary.nearest_valid_range_m,
        valid_sample_count=summary.valid_sample_count,
        rejected_below_min_count=summary.rejected_below_min_count,
        rejected_above_max_count=summary.rejected_above_max_count,
        rejected_non_finite_count=summary.rejected_non_finite_count,
        threshold_m=stop_distance_m,
        range_min_m=summary.range_min_m,
        range_max_m=summary.range_max_m,
        source="front_sector",
    )


def linear_scale_for_front_clearance(
    front_distance_m: float | None,
    stop_distance_m: float,
    slow_distance_m: float,
) -> float:
    if front_distance_m is None or slow_distance_m <= stop_distance_m:
        return 1.0
    if front_distance_m <= stop_distance_m:
        return 0.0
    if front_distance_m >= slow_distance_m:
        return 1.0
    return (front_distance_m - stop_distance_m) / (slow_distance_m - stop_distance_m)


def stuck_progress_failure(
    elapsed_without_progress_sec: float,
    max_without_progress_sec: float,
    forward_motion_commanded: bool,
) -> str:
    if forward_motion_commanded and elapsed_without_progress_sec > max_without_progress_sec:
        return "stuck no progress"
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
    allowed_external_publishers: Sequence[str] = (),
) -> str:
    allowed = set(allowed_external_publishers)
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
