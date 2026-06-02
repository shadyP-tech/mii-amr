from __future__ import annotations

import math


def normalize_angle_rad(angle_rad):
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def shortest_angle_delta_rad(start_rad, end_rad):
    return normalize_angle_rad(end_rad - start_rad)


def clamp(value, low, high):
    return max(low, min(high, value))


def distance_2d(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def finite_point_2d(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x = value[0]
    y = value[1]
    try:
        x = float(x)
        y = float(y)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return [x, y]
