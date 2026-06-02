from __future__ import annotations

import math

from .math_utils import clamp, distance_2d, normalize_angle_rad


def active_explore_curve_execution_record(
    candidate,
    path_points,
    curve_samples,
    driven_distance_m,
    duration_sec,
    stop_reason,
    **extra,
):
    record = {
        "executor": "cmd_vel_curve",
        "executed": True,
        "candidate_kind": candidate.kind,
        "candidate_score": candidate.score,
        "path_length_m": candidate.path_length_m,
        "curve_path_world": [[float(x), float(y)] for x, y in path_points],
        "curve_samples": list(curve_samples),
        "driven_distance_m": float(driven_distance_m),
        "duration_sec": float(duration_sec),
        "stop_reason": stop_reason,
    }
    record.update(extra)
    return record


def truncate_polyline_by_distance(points, max_distance_m):
    if len(points) < 2:
        raise RuntimeError("active_explore_curve_path_too_short")
    if max_distance_m <= 0.0:
        raise RuntimeError("active_explore_distance_limit_exhausted")
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
        raise RuntimeError("active_explore_curve_path_too_short")
    return tuple(truncated)


def active_explore_curve_path(candidate, current_pose, max_distance_m):
    source = list(candidate.path_world)
    if len(source) < 2:
        source = list(candidate.simplified_path_world)
    if len(source) < 2:
        raise RuntimeError("active_explore_curve_path_too_short")
    start = (float(current_pose.x), float(current_pose.y))
    if distance_2d(start, source[0]) <= 0.10:
        points = [start, *source[1:]]
    else:
        points = [start, *source]
    return truncate_polyline_by_distance(points, max_distance_m)


def select_curve_lookahead_target(path_points, current_point, lookahead_m):
    if not path_points:
        raise RuntimeError("active_explore_curve_path_too_short")
    nearest_index = min(
        range(len(path_points)),
        key=lambda index: distance_2d(current_point, path_points[index]),
    )
    for point in path_points[nearest_index + 1 :]:
        if distance_2d(current_point, point) >= lookahead_m:
            return point
    return path_points[-1]


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
