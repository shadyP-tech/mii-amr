from __future__ import annotations

from .math_utils import distance_2d
from waypoint_following.path_curves import (
    pure_pursuit_curve_command,
    select_curve_lookahead_target as shared_select_curve_lookahead_target,
    truncate_polyline_by_distance as shared_truncate_polyline_by_distance,
)


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


def active_explore_curve_error(exc):
    message = str(exc).replace("curve_path_", "active_explore_curve_path_")
    return message.replace(
        "curve_distance_limit_exhausted",
        "active_explore_distance_limit_exhausted",
    )


def truncate_polyline_by_distance(points, max_distance_m):
    try:
        return shared_truncate_polyline_by_distance(points, max_distance_m)
    except RuntimeError as exc:
        raise RuntimeError(active_explore_curve_error(exc)) from exc


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
    try:
        return shared_select_curve_lookahead_target(
            path_points,
            current_point,
            lookahead_m,
        )
    except RuntimeError as exc:
        raise RuntimeError(active_explore_curve_error(exc)) from exc
