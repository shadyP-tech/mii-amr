"""Small pure waypoint-control helpers for Aufgabe 04."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from scripts.aufgabe04.navigation.models import Pose2D


@dataclass(frozen=True)
class VelocityCommand:
    linear_x_mps: float
    angular_z_radps: float


@dataclass(frozen=True)
class ControllerConfig:
    max_linear_mps: float = 0.055
    max_angular_radps: float = 0.18
    goal_tolerance_m: float = 0.08
    heading_tolerance_rad: float = 0.25
    rotate_gain: float = 1.2
    lookahead_distance_m: float = 0.18
    slow_heading_error_rad: float = 0.75
    stop_heading_error_rad: float = 1.25
    min_linear_speed_scale: float = 0.35
    max_progress_advance_m: float = 0.45


@dataclass(frozen=True)
class ControllerStep:
    command: VelocityCommand
    target_index: int
    reached_goal: bool
    distance_to_target_m: float
    pursuit_index: int = 0


def normalize_angle(angle_rad: float) -> float:
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


def distance(a: Pose2D, b: Pose2D) -> float:
    return math.hypot(a.x_m - b.x_m, a.y_m - b.y_m)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _closest_index_from(
    pose: Pose2D,
    waypoints: Sequence[Pose2D],
    start_index: int,
    max_advance_m: float,
) -> int:
    closest_index = start_index
    closest_distance = distance(pose, waypoints[start_index])
    cumulative = 0.0
    for index in range(start_index + 1, len(waypoints)):
        cumulative += distance(waypoints[index - 1], waypoints[index])
        if max_advance_m > 0.0 and cumulative > max_advance_m:
            break
        candidate_distance = distance(pose, waypoints[index])
        if candidate_distance < closest_distance:
            closest_index = index
            closest_distance = candidate_distance
    return closest_index


def _lookahead_index(
    waypoints: Sequence[Pose2D],
    start_index: int,
    lookahead_distance_m: float,
) -> int:
    if lookahead_distance_m <= 0.0:
        return start_index
    cumulative = 0.0
    for index in range(start_index + 1, len(waypoints)):
        cumulative += distance(waypoints[index - 1], waypoints[index])
        if cumulative >= lookahead_distance_m:
            return index
    return len(waypoints) - 1


def _linear_speed_for_heading(
    heading_error_abs: float,
    target_distance_m: float,
    config: ControllerConfig,
) -> float:
    if heading_error_abs >= config.stop_heading_error_rad:
        return 0.0

    min_scale = _clamp(config.min_linear_speed_scale, 0.0, 1.0)
    slow_heading = max(config.slow_heading_error_rad, 1e-6)
    stop_heading = max(config.stop_heading_error_rad, slow_heading + 1e-6)
    if heading_error_abs <= slow_heading:
        heading_fraction = heading_error_abs / slow_heading
        heading_scale = 1.0 - heading_fraction * (1.0 - min_scale)
    else:
        taper_fraction = (heading_error_abs - slow_heading) / (stop_heading - slow_heading)
        heading_scale = min_scale * (1.0 - taper_fraction)

    approach_distance_m = max(config.goal_tolerance_m * 2.0, 1e-6)
    approach_scale = _clamp(target_distance_m / approach_distance_m, 0.25, 1.0)
    return config.max_linear_mps * _clamp(heading_scale, 0.0, 1.0) * approach_scale


def compute_waypoint_command(
    pose: Pose2D,
    waypoints: Sequence[Pose2D],
    target_index: int,
    config: ControllerConfig,
) -> ControllerStep:
    if not waypoints:
        return ControllerStep(VelocityCommand(0.0, 0.0), 0, True, 0.0)
    index = min(max(target_index, 0), len(waypoints) - 1)
    index = _closest_index_from(pose, waypoints, index, config.max_progress_advance_m)
    target = waypoints[index]
    target_distance = distance(pose, target)
    while target_distance <= config.goal_tolerance_m and index < len(waypoints) - 1:
        index += 1
        target = waypoints[index]
        target_distance = distance(pose, target)

    reached_goal = index == len(waypoints) - 1 and target_distance <= config.goal_tolerance_m
    if reached_goal:
        return ControllerStep(VelocityCommand(0.0, 0.0), index, True, target_distance, index)

    pursuit_index = _lookahead_index(waypoints, index, config.lookahead_distance_m)
    pursuit = waypoints[pursuit_index]
    heading = math.atan2(pursuit.y_m - pose.y_m, pursuit.x_m - pose.x_m)
    heading_error = normalize_angle(heading - pose.yaw_rad)
    angular = _clamp(
        heading_error * config.rotate_gain,
        -config.max_angular_radps,
        config.max_angular_radps,
    )
    linear = _linear_speed_for_heading(abs(heading_error), target_distance, config)
    return ControllerStep(VelocityCommand(linear, angular), index, False, target_distance, pursuit_index)
