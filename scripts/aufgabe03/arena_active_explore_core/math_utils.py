from __future__ import annotations

import math


def normalize_angle_rad(angle_rad):
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value, low, high):
    return max(low, min(high, value))


def valid_range(value, range_min, range_max):
    return (
        value is not None
        and math.isfinite(value)
        and value >= range_min
        and value <= range_max
    )


def yaw_rad_from_pose(pose):
    return math.radians(float(getattr(pose, "yaw_deg", 0.0)))


def point_from_heading(robot_pose, heading_rad, distance_m):
    return (
        float(robot_pose.x) + distance_m * math.cos(heading_rad),
        float(robot_pose.y) + distance_m * math.sin(heading_rad),
    )

