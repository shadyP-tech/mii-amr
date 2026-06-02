from __future__ import annotations

import math

from .diagnostics import effective_recovery_mode
from .math_utils import clamp, normalize_angle_rad
from .models import ArenaActiveSpinConfig, CenterRepositionAction, CenterRepositionStep


def opposite_axis_side(axis_side):
    if axis_side == "axis_negative":
        return "axis_positive"
    if axis_side == "axis_positive":
        return "axis_negative"
    return None


def candidate_range(candidate):
    if candidate is None:
        return None
    value = getattr(candidate, "short_wall_candidate_range_m", None)
    if value is None or not math.isfinite(value) or value <= 0.0:
        return None
    return float(value)


def candidate_heater_score(candidate):
    if candidate is None:
        return None
    value = getattr(candidate, "heater_profile_score", None)
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def candidate_profile_valid(candidate):
    if candidate is None:
        return False
    reason = getattr(candidate, "validity_failed_reason", None)
    if reason is not None:
        return False
    features = getattr(candidate, "profile_features", None) or {}
    return features.get("validity_failed_reason") is None


def short_wall_ranges_and_error(result, config: ArenaActiveSpinConfig):
    candidates = result.short_wall_candidates or {}
    negative = candidates.get("axis_negative")
    positive = candidates.get("axis_positive")
    negative_range = candidate_range(negative)
    positive_range = candidate_range(positive)
    if negative_range is None or positive_range is None:
        return None, None, None, None, None, None
    range_sum_error = negative_range + positive_range - config.arena_config.arena_length_m
    return candidates, negative, positive, negative_range, positive_range, range_sum_error


def choose_center_reposition_action(result, config: ArenaActiveSpinConfig, origin_yaw_rad=0.0):
    if effective_recovery_mode(config) != "legacy":
        return CenterRepositionAction(False, "center_reposition_disabled")
    if result.success or result.failure_reason != "pose_not_unique":
        return CenterRepositionAction(False, "center_reposition_not_pose_not_unique")
    long_fit = result.long_wall_fit
    if not getattr(long_fit, "ok", False) or long_fit.axis_angle_rad is None:
        return CenterRepositionAction(False, "center_reposition_invalid_long_wall_fit")

    (
        _candidates,
        negative,
        positive,
        negative_range,
        positive_range,
        range_sum_error,
    ) = short_wall_ranges_and_error(result, config)
    if negative_range is None or positive_range is None:
        return CenterRepositionAction(False, "center_reposition_missing_short_wall_ranges")

    if abs(range_sum_error) > config.arena_config.max_short_wall_range_sum_error_m:
        return CenterRepositionAction(
            False,
            "center_reposition_range_sum_invalid",
            range_sum_error_m=range_sum_error,
        )

    if negative_range <= positive_range:
        nearest_side = "axis_negative"
        nearest_range = negative_range
        far_range = positive_range
    else:
        nearest_side = "axis_positive"
        nearest_range = positive_range
        far_range = negative_range
    away_side = opposite_axis_side(nearest_side)

    heater_scores = {
        "axis_negative": getattr(negative, "heater_profile_score", None),
        "axis_positive": getattr(positive, "heater_profile_score", None),
    }
    steps = []

    raw_step = config.center_reposition_target_nearest_short_wall_range_m - nearest_range
    planned_distance = None
    local_heading = None
    odom_heading = None
    if raw_step >= config.center_reposition_min_step_m:
        planned_distance = clamp(
            raw_step,
            config.center_reposition_min_step_m,
            config.center_reposition_max_step_m,
        )
        local_heading = long_fit.axis_angle_rad
        if away_side == "axis_negative":
            local_heading += math.pi
        odom_heading = normalize_angle_rad(origin_yaw_rad + local_heading)
        steps.append(
            CenterRepositionStep(
                kind="longitudinal",
                reason="center_reposition_away_from_nearest_short_wall",
                planned_distance_m=planned_distance,
                local_heading_rad=normalize_angle_rad(local_heading),
                odom_heading_rad=odom_heading,
            )
        )

    lateral_offset = getattr(long_fit, "lateral_offset_m", None)
    lateral_target = config.center_reposition_lateral_target_offset_m
    lateral_planned_distance = None
    lateral_step_skipped = True
    lateral_skip_reason = "center_reposition_lateral_offset_unavailable"
    if lateral_offset is not None and math.isfinite(lateral_offset):
        lateral_error = abs(float(lateral_offset))
        if lateral_error <= config.center_reposition_lateral_offset_threshold_m:
            lateral_skip_reason = "center_reposition_lateral_offset_within_threshold"
        else:
            normal_angle = getattr(long_fit, "normal_angle_rad", None)
            if normal_angle is None or not math.isfinite(normal_angle):
                return CenterRepositionAction(
                    False,
                    "center_reposition_missing_lateral_normal",
                    nearest_axis_side=nearest_side,
                    away_axis_side=away_side,
                    nearest_short_wall_range_m=nearest_range,
                    far_short_wall_range_m=far_range,
                    target_nearest_short_wall_range_m=(
                        config.center_reposition_target_nearest_short_wall_range_m
                    ),
                    planned_distance_m=planned_distance if planned_distance is not None else max(0.0, raw_step),
                    range_sum_error_m=range_sum_error,
                    heater_scores=heater_scores,
                    lateral_offset_m=float(lateral_offset),
                    lateral_target_offset_m=lateral_target,
                    lateral_planned_distance_m=None,
                    lateral_step_skipped=True,
                    lateral_skip_reason="center_reposition_missing_lateral_normal",
                    steps=tuple(steps),
                )
            lateral_raw_step = max(0.0, lateral_error - lateral_target)
            lateral_planned_distance = clamp(
                lateral_raw_step,
                config.center_reposition_lateral_min_step_m,
                config.center_reposition_lateral_max_step_m,
            )
            lateral_heading = normal_angle if lateral_offset < 0.0 else normal_angle + math.pi
            lateral_odom_heading = normalize_angle_rad(origin_yaw_rad + lateral_heading)
            steps.append(
                CenterRepositionStep(
                    kind="lateral",
                    reason="center_reposition_reduce_lateral_offset_dynamic",
                    planned_distance_m=lateral_planned_distance,
                    local_heading_rad=normalize_angle_rad(lateral_heading),
                    odom_heading_rad=lateral_odom_heading,
                    dynamic_heading=True,
                    dynamic_heading_source="live_side_clearance",
                )
            )
            lateral_step_skipped = False
            lateral_skip_reason = None

    if not steps:
        return CenterRepositionAction(
            False,
            "center_reposition_not_useful_already_near_target",
            nearest_axis_side=nearest_side,
            away_axis_side=away_side,
            nearest_short_wall_range_m=nearest_range,
            far_short_wall_range_m=far_range,
            target_nearest_short_wall_range_m=(
                config.center_reposition_target_nearest_short_wall_range_m
            ),
            planned_distance_m=max(0.0, raw_step),
            range_sum_error_m=range_sum_error,
            heater_scores=heater_scores,
            lateral_offset_m=lateral_offset,
            lateral_target_offset_m=lateral_target,
            lateral_planned_distance_m=lateral_planned_distance,
            lateral_step_skipped=lateral_step_skipped,
            lateral_skip_reason=lateral_skip_reason,
        )

    if planned_distance is None:
        first_step = steps[0]
        planned_distance = first_step.planned_distance_m
        local_heading = first_step.local_heading_rad
        odom_heading = first_step.odom_heading_rad

    return CenterRepositionAction(
        True,
        "center_reposition_toward_arena_center",
        nearest_axis_side=nearest_side,
        away_axis_side=away_side,
        nearest_short_wall_range_m=nearest_range,
        far_short_wall_range_m=far_range,
        target_nearest_short_wall_range_m=(
            config.center_reposition_target_nearest_short_wall_range_m
        ),
        planned_distance_m=planned_distance,
        local_heading_rad=None if local_heading is None else normalize_angle_rad(local_heading),
        odom_heading_rad=odom_heading,
        range_sum_error_m=range_sum_error,
        heater_scores=heater_scores,
        lateral_offset_m=lateral_offset,
        lateral_target_offset_m=lateral_target,
        lateral_planned_distance_m=lateral_planned_distance,
        lateral_step_skipped=lateral_step_skipped,
        lateral_skip_reason=lateral_skip_reason,
        steps=tuple(steps),
    )


def choose_heater_approach_reposition_action(
    result,
    config: ArenaActiveSpinConfig,
    origin_yaw_rad=0.0,
):
    if effective_recovery_mode(config) != "legacy":
        return CenterRepositionAction(False, "heater_approach_reposition_disabled")
    if not config.center_reposition_enable_heater_approach:
        return CenterRepositionAction(False, "heater_approach_reposition_disabled")
    if result.success or result.failure_reason != "pose_not_unique":
        return CenterRepositionAction(False, "heater_approach_not_pose_not_unique")
    long_fit = result.long_wall_fit
    if not getattr(long_fit, "ok", False) or long_fit.axis_angle_rad is None:
        return CenterRepositionAction(False, "heater_approach_invalid_long_wall_fit")

    (
        _candidates,
        negative,
        positive,
        negative_range,
        positive_range,
        range_sum_error,
    ) = short_wall_ranges_and_error(result, config)
    if negative_range is None or positive_range is None:
        return CenterRepositionAction(False, "heater_approach_missing_short_wall_ranges")
    if abs(range_sum_error) > config.arena_config.max_short_wall_range_sum_error_m:
        return CenterRepositionAction(
            False,
            "heater_approach_range_sum_invalid",
            range_sum_error_m=range_sum_error,
        )
    if not candidate_profile_valid(negative) or not candidate_profile_valid(positive):
        return CenterRepositionAction(
            False,
            "heater_approach_profile_invalid",
            range_sum_error_m=range_sum_error,
        )

    negative_score = candidate_heater_score(negative)
    positive_score = candidate_heater_score(positive)
    if negative_score is None or positive_score is None:
        return CenterRepositionAction(
            False,
            "heater_approach_missing_heater_scores",
            range_sum_error_m=range_sum_error,
        )

    if negative_score >= positive_score:
        selected_side = "axis_negative"
        selected_range = negative_range
        selected_score = negative_score
        opposite_score = positive_score
    else:
        selected_side = "axis_positive"
        selected_range = positive_range
        selected_score = positive_score
        opposite_score = negative_score
    delta = selected_score - opposite_score
    heater_scores = {
        "axis_negative": negative_score,
        "axis_positive": positive_score,
    }
    common = {
        "suspected_heater_axis_side": selected_side,
        "suspected_heater_range_m": selected_range,
        "heater_approach_target_range_m": (
            config.center_reposition_heater_approach_target_range_m
        ),
        "range_sum_error_m": range_sum_error,
        "heater_scores": heater_scores,
        "selected_heater_score": selected_score,
        "opposite_heater_score": opposite_score,
        "heater_profile_delta": delta,
    }
    if selected_score < config.center_reposition_heater_approach_min_selected_score:
        return CenterRepositionAction(
            False,
            "heater_approach_selected_score_too_low",
            **common,
        )
    if opposite_score > config.center_reposition_heater_approach_max_opposite_score:
        return CenterRepositionAction(
            False,
            "heater_approach_opposite_score_too_high",
            **common,
        )
    if delta < config.center_reposition_heater_approach_min_delta:
        return CenterRepositionAction(
            False,
            "heater_approach_delta_too_low",
            **common,
        )

    raw_step = selected_range - config.center_reposition_heater_approach_target_range_m
    if raw_step < config.center_reposition_heater_approach_min_step_m:
        return CenterRepositionAction(
            False,
            "heater_approach_not_useful_already_near_target",
            planned_distance_m=max(0.0, raw_step),
            **common,
        )

    planned_distance = clamp(
        raw_step,
        config.center_reposition_heater_approach_min_step_m,
        config.center_reposition_heater_approach_max_step_m,
    )
    local_heading = long_fit.axis_angle_rad
    if selected_side == "axis_negative":
        local_heading += math.pi
    odom_heading = normalize_angle_rad(origin_yaw_rad + local_heading)
    step = CenterRepositionStep(
        kind="heater_approach",
        reason="heater_approach_toward_suspected_heater",
        planned_distance_m=planned_distance,
        local_heading_rad=normalize_angle_rad(local_heading),
        odom_heading_rad=odom_heading,
    )
    return CenterRepositionAction(
        True,
        "heater_approach_toward_suspected_heater",
        planned_distance_m=planned_distance,
        local_heading_rad=step.local_heading_rad,
        odom_heading_rad=odom_heading,
        steps=(step,),
        **common,
    )
