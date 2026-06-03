from __future__ import annotations

import math

from .math_utils import (
    normalize_angle_rad,
    point_from_heading,
    valid_range,
    yaw_rad_from_pose,
)
from .models import (
    FAILURE_POSE_NOT_UNIQUE,
    FAILURE_WALL_SEPARATION_OUT_OF_TOLERANCE,
    ActiveExploreConfig,
    RawCandidate,
)


def candidate_range(candidate):
    value = getattr(candidate, "short_wall_candidate_range_m", None)
    if value is None or not math.isfinite(value) or value <= 0.0:
        return None
    return float(value)


def candidate_heater_score(candidate):
    value = getattr(candidate, "heater_profile_score", None)
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def candidate_profile_valid(candidate):
    if candidate is None:
        return False
    failed = getattr(candidate, "validity_failed_reason", None)
    features = getattr(candidate, "profile_features", None) or {}
    return failed is None and features.get("validity_failed_reason") is None


def short_wall_ranges(result):
    candidates = result.short_wall_candidates or {}
    negative = candidates.get("axis_negative")
    positive = candidates.get("axis_positive")
    return (
        negative,
        positive,
        candidate_range(negative),
        candidate_range(positive),
    )


def short_wall_range_sum_ok(negative_range, positive_range, config):
    if negative_range is None or positive_range is None:
        return False
    range_sum_error = negative_range + positive_range - config.arena_length_m
    return abs(range_sum_error) <= config.max_short_wall_range_sum_error_m


def geometry_is_recoverable(result, config: ActiveExploreConfig):
    if result.success:
        return False, "already_localized"
    if result.failure_reason not in {
        FAILURE_POSE_NOT_UNIQUE,
        FAILURE_WALL_SEPARATION_OUT_OF_TOLERANCE,
    }:
        return False, "not_recoverable_failure"
    long_fit = result.long_wall_fit
    if long_fit is None:
        return False, "invalid_long_wall_fit"
    if result.failure_reason == FAILURE_WALL_SEPARATION_OUT_OF_TOLERANCE:
        return True, "ok"
    if not getattr(long_fit, "ok", False) or long_fit.axis_angle_rad is None:
        return False, "invalid_long_wall_fit"
    _negative, _positive, negative_range, positive_range = short_wall_ranges(result)
    if negative_range is None or positive_range is None:
        return False, "missing_short_wall_ranges"
    if not short_wall_range_sum_ok(negative_range, positive_range, config):
        # Obstacle geometry can look like a short wall and corrupt the range
        # sum. Still allow scan-grid shadow exploration, but suppress
        # range-dependent recovery candidates below.
        return True, "ok"
    return True, "ok"


def generate_raw_candidates(
    result,
    scan,
    robot_pose,
    config: ActiveExploreConfig,
    origin_yaw_rad=0.0,
):
    ok, reason = geometry_is_recoverable(result, config)
    if not ok:
        return (), reason

    long_fit = result.long_wall_fit
    candidates = result.short_wall_candidates or {}
    negative = candidates.get("axis_negative")
    positive = candidates.get("axis_positive")
    negative_range = candidate_range(negative)
    positive_range = candidate_range(positive)
    ranges_trustworthy = short_wall_range_sum_ok(
        negative_range,
        positive_range,
        config,
    )
    axis_angle = getattr(long_fit, "axis_angle_rad", None)
    axis_heading = (
        yaw_rad_from_pose(robot_pose)
        if axis_angle is None
        else normalize_angle_rad(origin_yaw_rad + axis_angle)
    )
    normal_angle = getattr(long_fit, "normal_angle_rad", None)
    normal_heading = (
        None
        if normal_angle is None
        else normalize_angle_rad(origin_yaw_rad + normal_angle)
    )
    raw = []

    if (
        ranges_trustworthy
        and negative_range is not None
        and positive_range is not None
    ):
        if negative_range <= positive_range:
            nearest_side = "axis_negative"
            nearest_range = negative_range
            away_heading = axis_heading
        else:
            nearest_side = "axis_positive"
            nearest_range = positive_range
            away_heading = normalize_angle_rad(axis_heading + math.pi)
        center_step = config.target_nearest_short_wall_range_m - nearest_range
        if center_step >= config.center_min_step_m:
            distance = min(center_step, config.max_single_move_m)
            x, y = point_from_heading(robot_pose, away_heading, distance)
            raw.append(
                RawCandidate(
                    "provisional_center",
                    x,
                    y,
                    away_heading,
                    geometry_progress=distance / config.max_single_move_m,
                    metadata={
                        "nearest_side": nearest_side,
                        "nearest_range_m": nearest_range,
                        "requested_step_m": center_step,
                    },
                )
            )

    lateral_offset = getattr(long_fit, "lateral_offset_m", None)
    if (
        lateral_offset is not None
        and math.isfinite(lateral_offset)
        and abs(lateral_offset) > config.lateral_offset_threshold_m
        and normal_heading is not None
    ):
        lateral_step = min(
            max(0.0, abs(lateral_offset) - config.lateral_target_offset_m),
            config.max_single_move_m,
        )
        heading = (
            normal_heading
            if lateral_offset < 0.0
            else normalize_angle_rad(normal_heading + math.pi)
        )
        x, y = point_from_heading(robot_pose, heading, lateral_step)
        raw.append(
            RawCandidate(
                "lateral_recenter",
                x,
                y,
                heading,
                geometry_progress=0.8 * lateral_step / config.max_single_move_m,
                metadata={
                    "lateral_offset_m": lateral_offset,
                    "requested_step_m": lateral_step,
                },
            )
        )

    if (
        ranges_trustworthy
        and negative_range is not None
        and positive_range is not None
        and candidate_profile_valid(negative)
        and candidate_profile_valid(positive)
    ):
        negative_score = candidate_heater_score(negative)
        positive_score = candidate_heater_score(positive)
        if negative_score is not None and positive_score is not None:
            if negative_score >= positive_score:
                selected_side = "axis_negative"
                selected_range = negative_range
                selected_score = negative_score
                opposite_score = positive_score
                heading = normalize_angle_rad(axis_heading + math.pi)
            else:
                selected_side = "axis_positive"
                selected_range = positive_range
                selected_score = positive_score
                opposite_score = negative_score
                heading = axis_heading
            delta = selected_score - opposite_score
            step = selected_range - config.heater_approach_target_range_m
            if (
                step > 0.0
                and selected_score >= config.heater_approach_min_selected_score
                and opposite_score <= config.heater_approach_max_opposite_score
                and delta >= config.heater_approach_min_delta
            ):
                distance = min(step, config.max_single_move_m)
                x, y = point_from_heading(robot_pose, heading, distance)
                raw.append(
                    RawCandidate(
                        "suspected_heater_approach",
                        x,
                        y,
                        heading,
                        geometry_progress=0.7 * distance / config.max_single_move_m,
                        heater_potential=delta,
                        metadata={
                            "selected_side": selected_side,
                            "selected_score": selected_score,
                            "opposite_score": opposite_score,
                            "heater_delta": delta,
                            "requested_step_m": step,
                        },
                    )
                )

    yaw = yaw_rad_from_pose(robot_pose)
    for angle_deg in (-90, -60, -30, 0, 30, 60, 90):
        sector_min = min_scan_range_in_sector(
            scan,
            angle_deg - 8.0,
            angle_deg + 8.0,
        )
        if sector_min is None:
            continue
        usable_distance = sector_min - config.inflation_radius_m
        if usable_distance < config.center_min_step_m:
            continue
        distance = min(config.max_single_move_m, usable_distance)
        heading = normalize_angle_rad(yaw + math.radians(angle_deg))
        x, y = point_from_heading(robot_pose, heading, distance)
        raw.append(
            RawCandidate(
                "open_corridor",
                x,
                y,
                heading,
                geometry_progress=0.25 * distance / config.max_single_move_m,
                metadata={
                    "sector_center_deg": angle_deg,
                    "sector_min_range_m": sector_min,
                },
            )
        )

    return tuple(raw), "ok"


def min_scan_range_in_sector(scan, lower_deg, upper_deg):
    values = []
    for index, raw_range in enumerate(scan.ranges):
        if not valid_range(raw_range, scan.range_min, scan.range_max):
            continue
        angle = math.degrees(
            normalize_angle_rad(scan.angle_min + index * scan.angle_increment)
        )
        if lower_deg <= angle <= upper_deg:
            values.append(float(raw_range))
    return min(values) if values else None
