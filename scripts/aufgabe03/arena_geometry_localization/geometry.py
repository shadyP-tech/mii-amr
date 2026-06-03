from __future__ import annotations

import math
from typing import Sequence

from .models import LineFit


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def normalize_angle_rad(angle_rad):
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def normalize_undirected_angle_rad(angle_rad):
    angle = angle_rad % math.pi
    return angle if angle >= 0.0 else angle + math.pi


def undirected_angle_delta_rad(a, b):
    delta = abs(normalize_angle_rad(a - b))
    return min(delta, abs(math.pi - delta))


def percentile_sorted(ordered, percent):
    if not ordered:
        raise ValueError("percentile requires values")
    if len(ordered) == 1:
        return ordered[0]
    rank = (percent / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def percentile(values, percent):
    return percentile_sorted(sorted(values), percent)


def median(values):
    return percentile(values, 50.0)


def fit_line(points: Sequence[tuple[float, float]]):
    count = len(points)
    if count < 2:
        raise ValueError("line fit needs at least two points")

    sum_x = 0.0
    sum_y = 0.0
    for x, y in points:
        sum_x += x
        sum_y += y
    mean_x = sum_x / count
    mean_y = sum_y / count

    sxx = 0.0
    syy = 0.0
    sxy = 0.0
    for x, y in points:
        dx = x - mean_x
        dy = y - mean_y
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    sxx /= count
    syy /= count
    sxy /= count

    direction = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    direction = normalize_undirected_angle_rad(direction)
    normal_x = -math.sin(direction)
    normal_y = math.cos(direction)
    offset = normal_x * mean_x + normal_y * mean_y
    if offset < 0.0:
        normal_x = -normal_x
        normal_y = -normal_y
        offset = -offset

    residual_sum_sq = 0.0
    for x, y in points:
        residual = normal_x * x + normal_y * y - offset
        residual_sum_sq += residual * residual
    rmse = math.sqrt(residual_sum_sq / count)

    return LineFit(
        point_count=count,
        normal_x=normal_x,
        normal_y=normal_y,
        offset=offset,
        direction_angle_rad=direction,
        rmse_m=rmse,
    )


def vector_from_angle(angle_rad):
    return math.cos(angle_rad), math.sin(angle_rad)


def dot(point, vector):
    return point[0] * vector[0] + point[1] * vector[1]


def projection_clusters(points, normal, lower_center, upper_center, threshold):
    lower = []
    upper = []
    for point in points:
        projection = dot(point, normal)
        if abs(projection - lower_center) <= threshold:
            lower.append(point)
        if abs(projection - upper_center) <= threshold:
            upper.append(point)
    return lower, upper
