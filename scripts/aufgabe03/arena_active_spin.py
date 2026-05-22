#!/usr/bin/env python3
"""
Spin-only active arena localization helper.

This module owns the experimental live spin, scan/odom pairing, safety checks,
diagnostics, and call into the offline arena geometry localizer. It deliberately
does not publish /initialpose or interact with Nav2.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from arena_geometry_localizer import (
    ArenaGeometryConfig,
    Pose2D,
    ScanSample,
    analyze_scan_samples,
)


DEFAULT_STOP_COUNT = 10
DEFAULT_STOP_HZ = 10.0


@dataclass(frozen=True)
class PosePrior:
    x_m: float
    y_m: float
    yaw_rad: float
    covariance: list[float]


@dataclass
class ArenaActiveSpinResult:
    success: bool
    failure_reason: str | None
    pose_prior: PosePrior | None
    diagnostics: dict
    diagnostics_path: str | None = None


@dataclass(frozen=True)
class SectorClearance:
    ok: bool
    reason: str
    front_min_m: float | None
    left_min_m: float | None
    right_min_m: float | None
    rear_min_m: float | None


@dataclass(frozen=True)
class CenterRepositionStep:
    kind: str
    reason: str
    planned_distance_m: float
    local_heading_rad: float | None
    odom_heading_rad: float

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CenterRepositionAction:
    ok: bool
    reason: str
    nearest_axis_side: str | None = None
    away_axis_side: str | None = None
    suspected_heater_axis_side: str | None = None
    nearest_short_wall_range_m: float | None = None
    far_short_wall_range_m: float | None = None
    suspected_heater_range_m: float | None = None
    target_nearest_short_wall_range_m: float | None = None
    heater_approach_target_range_m: float | None = None
    planned_distance_m: float | None = None
    local_heading_rad: float | None = None
    odom_heading_rad: float | None = None
    range_sum_error_m: float | None = None
    heater_scores: dict[str, float] | None = None
    selected_heater_score: float | None = None
    opposite_heater_score: float | None = None
    heater_profile_delta: float | None = None
    lateral_offset_m: float | None = None
    lateral_target_offset_m: float | None = None
    lateral_planned_distance_m: float | None = None
    lateral_step_skipped: bool = True
    lateral_skip_reason: str | None = None
    steps: tuple[CenterRepositionStep, ...] = ()

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ArenaActiveSpinConfig:
    run_id: str
    diagnostics_path: Path
    cmd_vel_topic: str = "/cmd_vel"
    scan_topic: str = "/scan"
    odom_topic: str = "/odom"
    spin_direction: str = "ccw"
    angular_speed_rad_s: float = 0.25
    max_spin_sec: float = 30.0
    spin_complete_tolerance_deg: float = 5.0
    min_angular_progress_rad_s: float = 0.05
    progress_check_sec: float = 2.0
    min_scan_samples: int = 20
    max_odom_scan_age_sec: float = 0.20
    stop_settle_sec: float = 0.5
    min_front_clearance_m: float = 0.35
    min_side_clearance_m: float = 0.20
    min_rear_clearance_m: float = 0.20
    require_operator_confirmation: bool = True
    allow_extra_cmd_vel_publishers: bool = False
    on_failure: str = "abort"
    dry_run: bool = False
    range_stride: int = 6
    max_points: int = 3000
    control_rate_hz: float = 10.0
    enable_center_reposition: bool = False
    center_reposition_max_attempts: int = 1
    center_reposition_target_nearest_short_wall_range_m: float = 1.65
    center_reposition_min_step_m: float = 0.25
    center_reposition_max_step_m: float = 1.10
    center_reposition_linear_speed_mps: float = 0.08
    center_reposition_angular_speed_rad_s: float = 0.25
    center_reposition_heading_tolerance_deg: float = 8.0
    center_reposition_min_front_clearance_m: float = 0.45
    center_reposition_lateral_offset_threshold_m: float = 0.25
    center_reposition_lateral_target_offset_m: float = 0.10
    center_reposition_lateral_min_step_m: float = 0.15
    center_reposition_lateral_max_step_m: float = 0.55
    center_reposition_enable_heater_approach: bool = True
    center_reposition_heater_approach_max_attempts: int = 1
    center_reposition_heater_approach_target_range_m: float = 1.05
    center_reposition_heater_approach_min_selected_score: float = 0.50
    center_reposition_heater_approach_max_opposite_score: float = 0.30
    center_reposition_heater_approach_min_delta: float = 0.35
    center_reposition_heater_approach_min_step_m: float = 0.25
    center_reposition_heater_approach_max_step_m: float = 1.10
    arena_config: ArenaGeometryConfig = field(default_factory=ArenaGeometryConfig)


def normalize_angle_rad(angle_rad):
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def shortest_angle_delta_rad(start_rad, end_rad):
    return normalize_angle_rad(end_rad - start_rad)


def clamp(value, low, high):
    return max(low, min(high, value))


def opposite_axis_side(axis_side):
    if axis_side == "axis_negative":
        return "axis_positive"
    if axis_side == "axis_positive":
        return "axis_negative"
    return None


def spin_diagnostics_template():
    return {
        "target_rad": 2.0 * math.pi,
        "accumulated_rad": 0.0,
        "duration_sec": 0.0,
        "timeout": False,
    }


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
    if not config.enable_center_reposition:
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
            lateral_heading = normal_angle if lateral_offset > 0.0 else normal_angle + math.pi
            lateral_odom_heading = normalize_angle_rad(origin_yaw_rad + lateral_heading)
            steps.append(
                CenterRepositionStep(
                    kind="lateral",
                    reason="center_reposition_reduce_lateral_offset",
                    planned_distance_m=lateral_planned_distance,
                    local_heading_rad=normalize_angle_rad(lateral_heading),
                    odom_heading_rad=lateral_odom_heading,
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
    if not config.enable_center_reposition:
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


def yaw_from_quaternion(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def odom_pose_from_msg(msg):
    pose = msg.pose.pose
    orientation = pose.orientation
    return Pose2D(
        x=float(pose.position.x),
        y=float(pose.position.y),
        yaw_deg=math.degrees(
            yaw_from_quaternion(
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            )
        ),
    )


def scan_sample_from_msg(msg, odom_pose):
    return ScanSample(
        ranges=list(msg.ranges),
        angle_min=float(msg.angle_min),
        angle_increment=float(msg.angle_increment),
        range_min=float(msg.range_min),
        range_max=float(msg.range_max),
        odom_pose=odom_pose,
    )


def valid_range(value, range_min, range_max):
    return (
        value is not None
        and math.isfinite(value)
        and value >= range_min
        and value <= range_max
    )


def angle_in_sector(angle_deg, ranges):
    return any(lower <= angle_deg <= upper for lower, upper in ranges)


def min_sector_range(scan, sectors):
    values = []
    for index, raw_range in enumerate(scan.ranges):
        if not valid_range(raw_range, scan.range_min, scan.range_max):
            continue
        angle_rad = scan.angle_min + index * scan.angle_increment
        angle_deg = math.degrees(normalize_angle_rad(angle_rad))
        if angle_in_sector(angle_deg, sectors):
            values.append(float(raw_range))
    return min(values) if values else None


def evaluate_clearance(scan, config: ArenaActiveSpinConfig):
    front = min_sector_range(scan, [(-30.0, 30.0)])
    left = min_sector_range(scan, [(60.0, 120.0)])
    right = min_sector_range(scan, [(-120.0, -60.0)])
    rear = min_sector_range(scan, [(150.0, 180.0), (-180.0, -150.0)])
    checks = [
        ("front_clearance_missing", "front_clearance_below_limit", front, config.min_front_clearance_m),
        ("left_clearance_missing", "left_clearance_below_limit", left, config.min_side_clearance_m),
        ("right_clearance_missing", "right_clearance_below_limit", right, config.min_side_clearance_m),
        ("rear_clearance_missing", "rear_clearance_below_limit", rear, config.min_rear_clearance_m),
    ]
    for missing_reason, low_reason, value, limit in checks:
        if value is None:
            return SectorClearance(False, missing_reason, front, left, right, rear)
        if value < limit:
            return SectorClearance(False, low_reason, front, left, right, rear)
    return SectorClearance(True, "ok", front, left, right, rear)


def evaluate_reposition_clearance(scan, config: ArenaActiveSpinConfig):
    front = min_sector_range(scan, [(-30.0, 30.0)])
    left = min_sector_range(scan, [(60.0, 120.0)])
    right = min_sector_range(scan, [(-120.0, -60.0)])
    rear = min_sector_range(scan, [(150.0, 180.0), (-180.0, -150.0)])
    checks = [
        (
            "front_clearance_missing",
            "front_clearance_below_limit",
            front,
            config.center_reposition_min_front_clearance_m,
        ),
        ("left_clearance_missing", "left_clearance_below_limit", left, config.min_side_clearance_m),
        ("right_clearance_missing", "right_clearance_below_limit", right, config.min_side_clearance_m),
        ("rear_clearance_missing", "rear_clearance_below_limit", rear, config.min_rear_clearance_m),
    ]
    for missing_reason, low_reason, value, limit in checks:
        if value is None:
            return SectorClearance(False, missing_reason, front, left, right, rear)
        if value < limit:
            return SectorClearance(False, low_reason, front, left, right, rear)
    return SectorClearance(True, "ok", front, left, right, rear)


def covariance_list_from_localizer(covariance):
    values = [0.0] * 36
    values[0] = float(covariance["x_m2"])
    values[7] = float(covariance["y_m2"])
    values[35] = float(covariance["yaw_rad2"])
    return values


def pose_prior_from_localizer_result(result):
    pose = result.estimated_pose_prior
    covariance = result.estimated_covariance
    if pose is None or covariance is None:
        return None
    return PosePrior(
        x_m=float(pose.x),
        y_m=float(pose.y),
        yaw_rad=math.radians(float(pose.yaw_deg)),
        covariance=covariance_list_from_localizer(covariance),
    )


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    return value


def write_diagnostics_json(path: Path | str | None, diagnostics):
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(json_safe(diagnostics), file, indent=2, sort_keys=True)
        file.write("\n")
    return str(path)


def config_diagnostics(config: ArenaActiveSpinConfig):
    data = asdict(config)
    data["diagnostics_path"] = str(config.diagnostics_path)
    data["arena_config"] = asdict(config.arena_config)
    return data


def initial_diagnostics(config: ArenaActiveSpinConfig):
    return {
        "mode": "arena-active",
        "success": False,
        "failure_reason": "",
        "fallback_used": False,
        "config": config_diagnostics(config),
        "spin": spin_diagnostics_template(),
        "spin_attempts": [],
        "reposition": {
            "enabled": config.enable_center_reposition,
            "attempts": [],
        },
        "samples": {
            "scan_samples_collected": 0,
            "scan_samples_used": 0,
            "rejected_scan_samples": 0,
        },
        "safety": {
            "min_front_range_m": None,
            "min_left_range_m": None,
            "min_right_range_m": None,
            "min_rear_range_m": None,
        },
        "cmd_vel_publishers": {
            "count": None,
            "unexpected_count": None,
            "allowed": config.allow_extra_cmd_vel_publishers,
        },
        "localizer_result": None,
        "exception": None,
        "initialpose": {
            "published": False,
            "reason": "not_reached",
        },
    }


def update_safety_minima(diagnostics, clearance: SectorClearance):
    safety = diagnostics["safety"]
    for key, value in [
        ("min_front_range_m", clearance.front_min_m),
        ("min_left_range_m", clearance.left_min_m),
        ("min_right_range_m", clearance.right_min_m),
        ("min_rear_range_m", clearance.rear_min_m),
    ]:
        if value is None:
            continue
        current = safety.get(key)
        safety[key] = value if current is None else min(current, value)


def stop_repeatedly(
    publisher,
    twist_factory: Callable[[], object],
    sleep_fn: Callable[[float], None] = time.sleep,
    count=DEFAULT_STOP_COUNT,
    hz=DEFAULT_STOP_HZ,
):
    delay = 1.0 / hz
    for _ in range(count):
        publisher.publish(twist_factory())
        sleep_fn(delay)


class ArenaActiveSpinSession:
    def __init__(
        self,
        node,
        config: ArenaActiveSpinConfig,
        rclpy_module,
        twist_factory,
        scan_msg_type,
        odom_msg_type,
        qos_profile,
        input_fn=input,
        time_fn=time.time,
        sleep_fn=time.sleep,
        analyze_fn=analyze_scan_samples,
    ):
        self.node = node
        self.config = config
        self.rclpy = rclpy_module
        self.twist_factory = twist_factory
        self.input_fn = input_fn
        self.time_fn = time_fn
        self.sleep_fn = sleep_fn
        self.analyze_fn = analyze_fn
        self.latest_scan = None
        self.latest_scan_received_sec = None
        self.latest_odom_pose = None
        self.latest_odom_yaw_rad = None
        self.latest_odom_received_sec = None
        self.collecting = False
        self.samples = []
        self.rejected_samples = 0
        self.diagnostics = initial_diagnostics(config)
        self.scan_subscription = node.create_subscription(
            scan_msg_type,
            config.scan_topic,
            self.scan_callback,
            qos_profile,
        )
        self.odom_subscription = node.create_subscription(
            odom_msg_type,
            config.odom_topic,
            self.odom_callback,
            10,
        )

    def now(self):
        return self.time_fn()

    def scan_callback(self, msg):
        received_sec = self.now()
        self.latest_scan = msg
        self.latest_scan_received_sec = received_sec
        if not self.collecting:
            return
        if self.latest_odom_pose is None or self.latest_odom_received_sec is None:
            self.rejected_samples += 1
            return
        if received_sec - self.latest_odom_received_sec > self.config.max_odom_scan_age_sec:
            self.rejected_samples += 1
            return
        self.samples.append(scan_sample_from_msg(msg, self.latest_odom_pose))

    def odom_callback(self, msg):
        self.latest_odom_pose = odom_pose_from_msg(msg)
        self.latest_odom_yaw_rad = math.radians(self.latest_odom_pose.yaw_deg)
        self.latest_odom_received_sec = self.now()

    def fresh_scan_age_sec(self):
        if self.latest_scan_received_sec is None:
            return None
        return self.now() - self.latest_scan_received_sec

    def fresh_odom_age_sec(self):
        if self.latest_odom_received_sec is None:
            return None
        return self.now() - self.latest_odom_received_sec

    def wait_for_fresh_inputs(self):
        deadline = self.now() + min(5.0, self.config.max_spin_sec)
        while self.rclpy.ok() and self.now() <= deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
            scan_age = self.fresh_scan_age_sec()
            odom_age = self.fresh_odom_age_sec()
            if (
                self.latest_scan is not None
                and scan_age is not None
                and scan_age <= self.config.max_odom_scan_age_sec
                and self.latest_odom_pose is not None
                and odom_age is not None
                and odom_age <= self.config.max_odom_scan_age_sec
            ):
                return
        raise RuntimeError("fresh_scan_or_odom_unavailable")

    def refresh_fresh_inputs_after_prompt(self):
        deadline = self.now() + min(2.0, self.config.max_spin_sec)
        while self.rclpy.ok() and self.now() <= deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
            scan_age = self.fresh_scan_age_sec()
            odom_age = self.fresh_odom_age_sec()
            if (
                self.latest_scan is not None
                and scan_age is not None
                and scan_age <= self.config.max_odom_scan_age_sec
                and self.latest_odom_pose is not None
                and odom_age is not None
                and odom_age <= self.config.max_odom_scan_age_sec
            ):
                return
        raise RuntimeError("fresh_scan_or_odom_unavailable_after_prompt")

    def cmd_vel_publisher_check(self):
        count = None
        if hasattr(self.node, "count_publishers"):
            count = self.node.count_publishers(self.config.cmd_vel_topic)
        unexpected = None if count is None else max(0, int(count) - 1)
        self.diagnostics["cmd_vel_publishers"] = {
            "count": count,
            "unexpected_count": unexpected,
            "allowed": self.config.allow_extra_cmd_vel_publishers,
        }
        if (
            unexpected is not None
            and unexpected > 0
            and not self.config.allow_extra_cmd_vel_publishers
        ):
            raise RuntimeError("unexpected_cmd_vel_publishers")

    def print_operator_prompt(self):
        scan_age = self.fresh_scan_age_sec()
        odom_age = self.fresh_odom_age_sec()
        clearance = evaluate_clearance(self.latest_scan, self.config)
        update_safety_minima(self.diagnostics, clearance)
        print("\nArena-active spin-only startup")
        print(f"  angular speed: {self.config.angular_speed_rad_s:.3f} rad/s")
        print(f"  direction: {self.config.spin_direction}")
        print(f"  max spin time: {self.config.max_spin_sec:.1f} s")
        print(f"  front clearance: {clearance.front_min_m}")
        print(f"  left clearance: {clearance.left_min_m}")
        print(f"  right clearance: {clearance.right_min_m}")
        print(f"  rear clearance: {clearance.rear_min_m}")
        print(f"  latest scan age: {scan_age}")
        print(f"  latest odom age: {odom_age}")
        print(f"  cmd_vel publisher check: {self.diagnostics['cmd_vel_publishers']}")
        print("  expected action: rotate in place 360 degrees")
        if not clearance.ok:
            raise RuntimeError(f"scan_clearance_failed:{clearance.reason}")
        if self.config.require_operator_confirmation:
            self.input_fn("Press Enter to start arena-active spin, or Ctrl+C to abort: ")

    def publish_spin_command(self, publisher):
        command = self.twist_factory()
        sign = 1.0 if self.config.spin_direction == "ccw" else -1.0
        command.angular.z = sign * abs(self.config.angular_speed_rad_s)
        publisher.publish(command)

    def run_spin(self, publisher):
        self.wait_for_fresh_inputs()
        self.cmd_vel_publisher_check()
        self.print_operator_prompt()
        self.refresh_fresh_inputs_after_prompt()

        previous_yaw = self.latest_odom_yaw_rad
        if previous_yaw is None:
            raise RuntimeError("fresh_odom_unavailable")
        self.collecting = True
        accumulated = 0.0
        target = 2.0 * math.pi - math.radians(self.config.spin_complete_tolerance_deg)
        period = 1.0 / self.config.control_rate_hz
        start = self.now()
        last_progress_time = start
        last_progress_yaw = 0.0

        while self.rclpy.ok():
            if self.now() - start > self.config.max_spin_sec:
                self.diagnostics["spin"]["timeout"] = True
                raise RuntimeError("arena_active_spin_timeout")
            self.publish_spin_command(publisher)
            self.rclpy.spin_once(self.node, timeout_sec=period)
            now = self.now()
            scan_age = self.fresh_scan_age_sec()
            odom_age = self.fresh_odom_age_sec()
            if scan_age is None or scan_age > self.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_scan_during_spin")
            if odom_age is None or odom_age > self.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_odom_during_spin")

            clearance = evaluate_clearance(self.latest_scan, self.config)
            update_safety_minima(self.diagnostics, clearance)
            if not clearance.ok:
                raise RuntimeError(f"scan_clearance_failed:{clearance.reason}")

            current_yaw = self.latest_odom_yaw_rad
            delta = shortest_angle_delta_rad(previous_yaw, current_yaw)
            accumulated += delta
            previous_yaw = current_yaw
            self.diagnostics["spin"]["accumulated_rad"] = accumulated
            self.diagnostics["spin"]["duration_sec"] = now - start
            if abs(accumulated) >= target:
                return accumulated, now - start

            if now - last_progress_time >= self.config.progress_check_sec:
                progress_rate = abs(accumulated - last_progress_yaw) / (now - last_progress_time)
                if progress_rate < self.config.min_angular_progress_rad_s:
                    raise RuntimeError("insufficient_angular_progress")
                last_progress_time = now
                last_progress_yaw = accumulated

        raise RuntimeError("ros_shutdown_during_arena_active_spin")

    def reset_spin_collection(self, attempt_index):
        self.collecting = False
        self.samples = []
        self.rejected_samples = 0
        self.diagnostics["spin"] = {
            **spin_diagnostics_template(),
            "attempt_index": attempt_index,
        }

    def run_spin_attempt(self, publisher, attempt_index):
        self.reset_spin_collection(attempt_index)
        self.run_spin(publisher)
        self.collecting = False
        stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
        self.sleep_fn(self.config.stop_settle_sec)
        self.diagnostics["spin_attempts"].append(
            {
                **self.diagnostics["spin"],
                "scan_samples_collected": len(self.samples),
                "scan_samples_used": len(self.samples),
                "rejected_scan_samples": self.rejected_samples,
            }
        )

    def analyze_result(self):
        if len(self.samples) < self.config.min_scan_samples:
            raise RuntimeError(
                "insufficient_scan_samples:"
                f"{len(self.samples)}<{self.config.min_scan_samples}"
            )
        result = self.analyze_fn(
            self.samples,
            self.config.arena_config,
            range_stride=self.config.range_stride,
            max_points=self.config.max_points,
        )
        self.diagnostics["localizer_result"] = result.to_dict()
        return result

    def pose_prior_from_result_or_raise(self, result):
        if not result.success:
            raise RuntimeError(f"arena_localizer_failed:{result.failure_reason}")
        pose_prior = pose_prior_from_localizer_result(result)
        if pose_prior is None:
            raise RuntimeError("arena_localizer_missing_pose_prior")
        return pose_prior

    def first_sample_origin_yaw_rad(self):
        for sample in self.samples:
            if sample.odom_pose is not None:
                return math.radians(sample.odom_pose.yaw_deg)
        return 0.0

    def publish_turn_command(self, publisher, target_yaw_rad):
        if self.latest_odom_yaw_rad is None:
            raise RuntimeError("fresh_odom_unavailable_during_reposition_turn")
        command = self.twist_factory()
        delta = shortest_angle_delta_rad(self.latest_odom_yaw_rad, target_yaw_rad)
        command.angular.z = (
            1.0 if delta >= 0.0 else -1.0
        ) * abs(self.config.center_reposition_angular_speed_rad_s)
        publisher.publish(command)

    def turn_to_heading(self, publisher, target_yaw_rad):
        tolerance = math.radians(self.config.center_reposition_heading_tolerance_deg)
        deadline = self.now() + max(
            8.0,
            math.pi / max(0.01, abs(self.config.center_reposition_angular_speed_rad_s))
            + 3.0,
        )
        period = 1.0 / self.config.control_rate_hz
        while self.rclpy.ok() and self.now() <= deadline:
            self.rclpy.spin_once(self.node, timeout_sec=period)
            scan_age = self.fresh_scan_age_sec()
            odom_age = self.fresh_odom_age_sec()
            if scan_age is None or scan_age > self.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_scan_during_reposition_turn")
            if odom_age is None or odom_age > self.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_odom_during_reposition_turn")
            clearance = evaluate_clearance(self.latest_scan, self.config)
            update_safety_minima(self.diagnostics, clearance)
            if not clearance.ok:
                raise RuntimeError(f"reposition_turn_clearance_failed:{clearance.reason}")
            delta = shortest_angle_delta_rad(self.latest_odom_yaw_rad, target_yaw_rad)
            if abs(delta) <= tolerance:
                return
            self.publish_turn_command(publisher, target_yaw_rad)
        raise RuntimeError("center_reposition_turn_timeout")

    def publish_drive_command(self, publisher):
        command = self.twist_factory()
        command.linear.x = abs(self.config.center_reposition_linear_speed_mps)
        publisher.publish(command)

    def drive_forward(self, publisher, distance_m):
        if self.latest_odom_pose is None:
            raise RuntimeError("fresh_odom_unavailable_before_reposition_drive")
        start_x = self.latest_odom_pose.x
        start_y = self.latest_odom_pose.y
        deadline = self.now() + max(
            8.0,
            distance_m / max(0.01, abs(self.config.center_reposition_linear_speed_mps))
            + 3.0,
        )
        period = 1.0 / self.config.control_rate_hz
        while self.rclpy.ok() and self.now() <= deadline:
            self.rclpy.spin_once(self.node, timeout_sec=period)
            scan_age = self.fresh_scan_age_sec()
            odom_age = self.fresh_odom_age_sec()
            if scan_age is None or scan_age > self.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_scan_during_reposition_drive")
            if odom_age is None or odom_age > self.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_odom_during_reposition_drive")
            clearance = evaluate_reposition_clearance(self.latest_scan, self.config)
            update_safety_minima(self.diagnostics, clearance)
            if not clearance.ok:
                raise RuntimeError(f"reposition_drive_clearance_failed:{clearance.reason}")
            dx = self.latest_odom_pose.x - start_x
            dy = self.latest_odom_pose.y - start_y
            if math.hypot(dx, dy) >= distance_m:
                return math.hypot(dx, dy)
            self.publish_drive_command(publisher)
        raise RuntimeError("center_reposition_drive_timeout")

    def print_reposition_prompt(self, action: CenterRepositionAction):
        print("\nArena-active reposition recovery")
        print(f"  nearest short wall: {action.nearest_axis_side}")
        print(f"  away direction: {action.away_axis_side}")
        print(f"  nearest range: {action.nearest_short_wall_range_m}")
        print(f"  target nearest range: {action.target_nearest_short_wall_range_m}")
        print(f"  suspected heater wall: {action.suspected_heater_axis_side}")
        print(f"  suspected heater range: {action.suspected_heater_range_m}")
        print(f"  target heater range: {action.heater_approach_target_range_m}")
        print(f"  heater scores: {action.heater_scores}")
        print(f"  heater delta: {action.heater_profile_delta}")
        print(f"  lateral offset: {action.lateral_offset_m}")
        print(f"  target lateral offset: {action.lateral_target_offset_m}")
        steps = list(action.steps)
        if not steps and action.odom_heading_rad is not None and action.planned_distance_m is not None:
            steps = [
                CenterRepositionStep(
                    kind="legacy",
                    reason=action.reason,
                    planned_distance_m=action.planned_distance_m,
                    local_heading_rad=action.local_heading_rad,
                    odom_heading_rad=action.odom_heading_rad,
                )
            ]
        for index, step in enumerate(steps, start=1):
            print(
                f"  step {index} {step.kind}: "
                f"distance={step.planned_distance_m:.3f} m, "
                f"target odom heading={math.degrees(step.odom_heading_rad):.1f} deg"
            )
        if action.lateral_step_skipped:
            print(f"  lateral step: skipped ({action.lateral_skip_reason})")
        print("  expected action: turn, drive, optionally turn sideways, drive, then spin again")
        if self.config.require_operator_confirmation:
            self.input_fn("Press Enter to start center reposition, or Ctrl+C to abort: ")

    def execute_center_reposition(self, publisher, action: CenterRepositionAction):
        steps = list(action.steps)
        if not steps and action.odom_heading_rad is not None and action.planned_distance_m is not None:
            steps = [
                CenterRepositionStep(
                    kind="legacy",
                    reason=action.reason,
                    planned_distance_m=action.planned_distance_m,
                    local_heading_rad=action.local_heading_rad,
                    odom_heading_rad=action.odom_heading_rad,
                )
            ]
        if not action.ok or not steps:
            raise RuntimeError(action.reason)
        self.wait_for_fresh_inputs()
        self.print_reposition_prompt(action)
        self.refresh_fresh_inputs_after_prompt()
        clearance = evaluate_reposition_clearance(self.latest_scan, self.config)
        update_safety_minima(self.diagnostics, clearance)
        if not clearance.ok:
            raise RuntimeError(f"reposition_precheck_clearance_failed:{clearance.reason}")

        start = self.now()
        total_driven = 0.0
        step_records = []
        for index, step in enumerate(steps):
            if index > 0:
                self.wait_for_fresh_inputs()
            step_start = self.now()
            self.turn_to_heading(publisher, step.odom_heading_rad)
            stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
            self.wait_for_fresh_inputs()
            driven = self.drive_forward(publisher, step.planned_distance_m)
            stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
            total_driven += driven
            step_records.append(
                {
                    **step.to_dict(),
                    "driven_distance_m": driven,
                    "duration_sec": self.now() - step_start,
                }
            )
        record = action.to_dict()
        record["steps"] = step_records
        record["driven_distance_m"] = total_driven
        record["duration_sec"] = self.now() - start
        return record

    def finish_failure(self, reason, exception=None):
        self.diagnostics["success"] = False
        self.diagnostics["failure_reason"] = reason
        if exception is not None:
            self.diagnostics["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
            }
        self.diagnostics["samples"]["scan_samples_collected"] = len(self.samples)
        self.diagnostics["samples"]["scan_samples_used"] = len(self.samples)
        self.diagnostics["samples"]["rejected_scan_samples"] = self.rejected_samples
        path = write_diagnostics_json(self.config.diagnostics_path, self.diagnostics)
        return ArenaActiveSpinResult(False, reason, None, self.diagnostics, path)

    def finish_success(self, pose_prior):
        self.diagnostics["success"] = True
        self.diagnostics["failure_reason"] = ""
        self.diagnostics["samples"]["scan_samples_collected"] = len(self.samples)
        self.diagnostics["samples"]["scan_samples_used"] = len(self.samples)
        self.diagnostics["samples"]["rejected_scan_samples"] = self.rejected_samples
        self.diagnostics["initialpose"] = {
            "published": False,
            "reason": "dry_run" if self.config.dry_run else "pending_runner_publication",
        }
        path = write_diagnostics_json(self.config.diagnostics_path, self.diagnostics)
        return ArenaActiveSpinResult(True, None, pose_prior, self.diagnostics, path)

    def run(self, publisher):
        try:
            stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
            self.run_spin_attempt(publisher, attempt_index=0)
            result = self.analyze_result()
            if not result.success and self.config.enable_center_reposition:
                reposition_attempts = 0
                while (
                    not result.success
                    and reposition_attempts < self.config.center_reposition_max_attempts
                ):
                    origin_yaw = self.first_sample_origin_yaw_rad()
                    action = choose_center_reposition_action(result, self.config, origin_yaw)
                    attempt_record = {
                        "attempt_index": len(self.diagnostics["reposition"]["attempts"]),
                        "stage": "center",
                        "previous_failure_reason": result.failure_reason,
                        "previous_classifier_reason": result.short_wall_classification.reason,
                        "action": action.to_dict(),
                    }
                    self.diagnostics["reposition"]["attempts"].append(attempt_record)
                    if not action.ok:
                        break
                    self.diagnostics["fallback_used"] = True
                    motion_record = self.execute_center_reposition(publisher, action)
                    attempt_record["motion"] = motion_record
                    reposition_attempts += 1
                    self.run_spin_attempt(publisher, attempt_index=reposition_attempts)
                    result = self.analyze_result()
                    attempt_record["post_reposition_success"] = result.success
                    attempt_record["post_reposition_failure_reason"] = result.failure_reason
                    attempt_record["post_reposition_classifier_reason"] = (
                        result.short_wall_classification.reason
                    )
                heater_approach_attempts = 0
                while (
                    not result.success
                    and self.config.center_reposition_enable_heater_approach
                    and heater_approach_attempts
                    < self.config.center_reposition_heater_approach_max_attempts
                ):
                    origin_yaw = self.first_sample_origin_yaw_rad()
                    action = choose_heater_approach_reposition_action(
                        result,
                        self.config,
                        origin_yaw,
                    )
                    attempt_record = {
                        "attempt_index": len(self.diagnostics["reposition"]["attempts"]),
                        "stage": "heater_approach",
                        "previous_failure_reason": result.failure_reason,
                        "previous_classifier_reason": result.short_wall_classification.reason,
                        "action": action.to_dict(),
                    }
                    self.diagnostics["reposition"]["attempts"].append(attempt_record)
                    if not action.ok:
                        break
                    self.diagnostics["fallback_used"] = True
                    motion_record = self.execute_center_reposition(publisher, action)
                    attempt_record["motion"] = motion_record
                    heater_approach_attempts += 1
                    self.run_spin_attempt(
                        publisher,
                        attempt_index=len(self.diagnostics["spin_attempts"]),
                    )
                    result = self.analyze_result()
                    attempt_record["post_reposition_success"] = result.success
                    attempt_record["post_reposition_failure_reason"] = result.failure_reason
                    attempt_record["post_reposition_classifier_reason"] = (
                        result.short_wall_classification.reason
                    )
            pose_prior = self.pose_prior_from_result_or_raise(result)
            return self.finish_success(pose_prior)
        except KeyboardInterrupt:
            self.collecting = False
            stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
            return self.finish_failure("keyboard_interrupt")
        except Exception as exc:
            self.collecting = False
            stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
            return self.finish_failure(str(exc), exception=exc)


def run_arena_active_spin(
    node,
    publisher,
    config: ArenaActiveSpinConfig,
    rclpy_module,
    twist_factory,
    scan_msg_type,
    odom_msg_type,
    qos_profile,
    input_fn=input,
    time_fn=time.time,
    sleep_fn=time.sleep,
    analyze_fn=analyze_scan_samples,
):
    session = ArenaActiveSpinSession(
        node,
        config,
        rclpy_module,
        twist_factory,
        scan_msg_type,
        odom_msg_type,
        qos_profile,
        input_fn=input_fn,
        time_fn=time_fn,
        sleep_fn=sleep_fn,
        analyze_fn=analyze_fn,
    )
    return session.run(publisher)
