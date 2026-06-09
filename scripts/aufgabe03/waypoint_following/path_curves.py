from __future__ import annotations

import math

from .math_utils import clamp, distance_2d, normalize_angle_rad


def truncate_polyline_by_distance(points, max_distance_m):
    if len(points) < 2:
        raise RuntimeError("curve_path_too_short")
    if max_distance_m <= 0.0:
        raise RuntimeError("curve_distance_limit_exhausted")
    truncated = [points[0]]
    remaining = float(max_distance_m)
    previous = points[0]
    for point in points[1:]:
        segment = distance_2d(previous, point)
        if segment <= 1e-9:
            previous = point
            continue
        if segment <= remaining + 1e-9:
            truncated.append(point)
            remaining -= segment
            previous = point
            if remaining <= 1e-9:
                break
            continue
        ratio = remaining / segment
        truncated.append(
            (
                previous[0] + ratio * (point[0] - previous[0]),
                previous[1] + ratio * (point[1] - previous[1]),
            )
        )
        break
    if len(truncated) < 2:
        raise RuntimeError("curve_path_too_short")
    return tuple(truncated)


def select_curve_lookahead_target(path_points, current_point, lookahead_m):
    if not path_points:
        raise RuntimeError("curve_path_too_short")
    nearest_index = min(
        range(len(path_points)),
        key=lambda index: distance_2d(current_point, path_points[index]),
    )
    for point in path_points[nearest_index + 1 :]:
        if distance_2d(current_point, point) >= lookahead_m:
            return point
    return path_points[-1]


def _projection_on_segment(point, start, end):
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return distance_2d(point, start), 0.0, (float(start[0]), float(start[1]))
    ratio = (
        (float(point[0]) - float(start[0])) * dx
        + (float(point[1]) - float(start[1])) * dy
    ) / length_sq
    ratio = clamp(ratio, 0.0, 1.0)
    projected = (
        float(start[0]) + ratio * dx,
        float(start[1]) + ratio * dy,
    )
    return distance_2d(point, projected), ratio, projected


def polyline_lookahead_target(path_points, current_point, lookahead_m):
    points = [(float(x), float(y)) for x, y in path_points]
    if not points:
        raise RuntimeError("curve_path_too_short")
    if len(points) == 1:
        return points[0]

    best = None
    for index in range(len(points) - 1):
        distance_m, ratio, projected = _projection_on_segment(
            current_point,
            points[index],
            points[index + 1],
        )
        candidate = (distance_m, index, ratio, projected)
        if best is None or candidate < best:
            best = candidate

    _distance_m, segment_index, ratio, projected = best
    remaining = max(0.0, float(lookahead_m))
    previous = projected
    segment_end = points[segment_index + 1]
    segment_remaining = distance_2d(previous, segment_end)
    if segment_remaining >= remaining and segment_remaining > 1e-9:
        blend = remaining / segment_remaining
        return (
            previous[0] + blend * (segment_end[0] - previous[0]),
            previous[1] + blend * (segment_end[1] - previous[1]),
        )
    remaining -= segment_remaining
    previous = segment_end

    for point in points[segment_index + 2 :]:
        segment = distance_2d(previous, point)
        if segment <= 1e-9:
            previous = point
            continue
        if segment >= remaining:
            blend = remaining / segment
            return (
                previous[0] + blend * (point[0] - previous[0]),
                previous[1] + blend * (point[1] - previous[1]),
            )
        remaining -= segment
        previous = point
    return points[-1]


def pure_pursuit_curve_command(
    current_pose,
    target_point,
    lookahead_m,
    linear_speed_mps,
    max_angular_rad_s,
):
    dx = float(target_point[0]) - float(current_pose.x)
    dy = float(target_point[1]) - float(current_pose.y)
    target_heading = math.atan2(dy, dx)
    yaw = math.radians(float(current_pose.yaw_deg))
    alpha = normalize_angle_rad(target_heading - yaw)
    linear_scale = clamp(math.cos(abs(alpha)), 0.35, 1.0)
    linear_x = abs(linear_speed_mps) * linear_scale
    angular_z = clamp(
        2.0 * linear_x * math.sin(alpha) / max(0.01, lookahead_m),
        -abs(max_angular_rad_s),
        abs(max_angular_rad_s),
    )
    return linear_x, angular_z, alpha

