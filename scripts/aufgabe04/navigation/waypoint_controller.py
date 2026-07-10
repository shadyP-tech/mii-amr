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
    max_linear_mps: float = 0.05
    max_angular_radps: float = 0.15
    goal_tolerance_m: float = 0.08
    heading_tolerance_rad: float = 0.25
    rotate_gain: float = 1.2


@dataclass(frozen=True)
class ControllerStep:
    command: VelocityCommand
    target_index: int
    reached_goal: bool
    distance_to_target_m: float
    target_heading_rad: float
    heading_error_rad: float


def normalize_angle(angle_rad: float) -> float:
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


def distance(a: Pose2D, b: Pose2D) -> float:
    return math.hypot(a.x_m - b.x_m, a.y_m - b.y_m)


def forward_resume_target(pose: Pose2D, waypoints: Sequence[Pose2D]) -> tuple[int, float]:
    """Return the waypoint after the closest route point and route proximity."""

    if not waypoints:
        return 0, math.inf
    nearest_index = min(range(len(waypoints)), key=lambda index: distance(pose, waypoints[index]))
    nearest_distance = distance(pose, waypoints[nearest_index])
    return min(nearest_index + 1, len(waypoints) - 1), nearest_distance


def compute_waypoint_command(
    pose: Pose2D,
    waypoints: Sequence[Pose2D],
    target_index: int,
    config: ControllerConfig,
) -> ControllerStep:
    if not waypoints:
        return ControllerStep(VelocityCommand(0.0, 0.0), 0, True, 0.0, 0.0, 0.0)
    index = min(max(target_index, 0), len(waypoints) - 1)
    target = waypoints[index]
    target_distance = distance(pose, target)
    while target_distance <= config.goal_tolerance_m and index < len(waypoints) - 1:
        index += 1
        target = waypoints[index]
        target_distance = distance(pose, target)

    reached_goal = index == len(waypoints) - 1 and target_distance <= config.goal_tolerance_m
    if reached_goal:
        if math.isfinite(target.yaw_rad):
            final_heading_error = normalize_angle(target.yaw_rad - pose.yaw_rad)
            if abs(final_heading_error) > config.heading_tolerance_rad:
                angular = max(
                    -config.max_angular_radps,
                    min(config.max_angular_radps, final_heading_error * config.rotate_gain),
                )
                return ControllerStep(
                    VelocityCommand(0.0, angular), index, False, target_distance,
                    target.yaw_rad, final_heading_error,
                )
        return ControllerStep(
            VelocityCommand(0.0, 0.0),
            index,
            True,
            target_distance,
            pose.yaw_rad,
            0.0,
        )

    heading = math.atan2(target.y_m - pose.y_m, target.x_m - pose.x_m)
    heading_error = normalize_angle(heading - pose.yaw_rad)
    angular = max(
        -config.max_angular_radps,
        min(config.max_angular_radps, heading_error * config.rotate_gain),
    )
    linear = 0.0 if abs(heading_error) > config.heading_tolerance_rad else config.max_linear_mps
    return ControllerStep(
        VelocityCommand(linear, angular),
        index,
        False,
        target_distance,
        heading,
        heading_error,
    )
