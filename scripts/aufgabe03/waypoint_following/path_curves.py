from __future__ import annotations

import math
from dataclasses import dataclass

from .math_utils import clamp, distance_2d, normalize_angle_rad, shortest_angle_delta_deg


ROUTE_HEADING_LOOKAHEAD_M = 0.16


@dataclass(frozen=True)
class RouteProjection:
    projected_point: tuple[float, float]
    segment_index: int
    segment_ratio: float
    route_progress_m: float
    route_heading_deg: float
    heading_error_to_route_deg: float
    cross_track_error_m: float
    signed_cross_track_error_m: float
    remaining_route_m: float
    projection_status: str = "locked"
    route_progress_delta_m: float | None = None
    route_progress_backward_delta_m: float = 0.0
    route_progress_forward_delta_m: float = 0.0
    raw_projection_progress_m: float | None = None
    raw_projection_segment_index: int | None = None
    effective_projection_progress_m: float | None = None
    anchor_progress_m: float | None = None
    anchor_segment_index: int | None = None
    rotate_anchor_backward_delta_m: float = 0.0
    rotate_anchor_forward_delta_m: float = 0.0
    local_cross_track_m: float | None = None
    preferred_branch_heading_deg: float | None = None
    selected_segment_heading_deg: float | None = None
    selected_branch_heading_error_deg: float | None = None
    rejected_wrong_heading_segment_count: int = 0
    branch_lock_stable_count: int = 0
    branch_lock_progress_span_m: float = 0.0
    branch_lock_release_required_span_m: float = 0.0


@dataclass(frozen=True)
class RouteHeading:
    heading_deg: float | None
    heading_error_deg: float | None
    source: str
    target_point: tuple[float, float] | None = None
    lookahead_m: float = 0.0


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


def route_cumulative_distances(points):
    normalized = [(float(x), float(y)) for x, y in points]
    if not normalized:
        return []
    cumulative = [0.0]
    previous = normalized[0]
    for point in normalized[1:]:
        cumulative.append(cumulative[-1] + distance_2d(previous, point))
        previous = point
    return cumulative


def route_point_at_progress(points, cumulative, progress_m):
    if not points:
        raise RuntimeError("route_projection_path_too_short")
    progress = clamp(float(progress_m), 0.0, cumulative[-1])
    for index in range(len(points) - 1):
        start_progress = cumulative[index]
        end_progress = cumulative[index + 1]
        segment_length = end_progress - start_progress
        if segment_length <= 1e-9:
            continue
        if progress <= end_progress + 1e-9:
            ratio = clamp((progress - start_progress) / segment_length, 0.0, 1.0)
            start = points[index]
            end = points[index + 1]
            return (
                start[0] + ratio * (end[0] - start[0]),
                start[1] + ratio * (end[1] - start[1]),
                index,
                ratio,
            )
    return points[-1][0], points[-1][1], max(0, len(points) - 2), 1.0


def _valid_segment_heading(route, start_index):
    if len(route) < 2:
        return None
    start_index = max(0, min(int(start_index), len(route) - 2))
    for index in range(start_index, len(route) - 1):
        start = route[index]
        end = route[index + 1]
        if distance_2d(start, end) > 1e-9:
            return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
    for index in range(start_index - 1, -1, -1):
        start = route[index]
        end = route[index + 1]
        if distance_2d(start, end) > 1e-9:
            return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
    return None


def route_heading_from_projection(
    points,
    projection,
    current_yaw_deg,
    heading_lookahead_m=ROUTE_HEADING_LOOKAHEAD_M,
):
    route = [(float(x), float(y)) for x, y in points]
    if len(route) < 2:
        return RouteHeading(None, None, "unavailable")
    cumulative = route_cumulative_distances(route)
    if not cumulative or cumulative[-1] <= 1e-9:
        return RouteHeading(None, None, "unavailable")

    heading_lookahead = max(0.0, float(heading_lookahead_m))
    projection_point = (
        float(projection.projected_point[0]),
        float(projection.projected_point[1]),
    )
    if projection.remaining_route_m >= heading_lookahead and heading_lookahead > 1e-9:
        ahead_x, ahead_y, _index, _ratio = route_point_at_progress(
            route,
            cumulative,
            projection.route_progress_m + heading_lookahead,
        )
        target = (ahead_x, ahead_y)
        if distance_2d(projection_point, target) > 1e-9:
            heading_deg = math.degrees(
                math.atan2(target[1] - projection_point[1], target[0] - projection_point[0])
            )
            return RouteHeading(
                heading_deg,
                shortest_angle_delta_deg(float(current_yaw_deg), heading_deg),
                "lookahead",
                target,
                heading_lookahead,
            )

    heading_deg = _valid_segment_heading(route, projection.segment_index)
    if heading_deg is None:
        return RouteHeading(None, None, "unavailable")
    return RouteHeading(
        heading_deg,
        shortest_angle_delta_deg(float(current_yaw_deg), heading_deg),
        "local_segment",
        None,
        0.0,
    )


def route_heading_at_progress(
    points,
    progress_m,
    heading_lookahead_m=ROUTE_HEADING_LOOKAHEAD_M,
):
    route = [(float(x), float(y)) for x, y in points]
    if len(route) < 2:
        return None
    cumulative = route_cumulative_distances(route)
    if not cumulative or cumulative[-1] <= 1e-9:
        return None
    progress = clamp(float(progress_m), 0.0, cumulative[-1])
    x, y, segment_index, _ratio = route_point_at_progress(route, cumulative, progress)
    projection = RouteProjection(
        projected_point=(x, y),
        segment_index=segment_index,
        segment_ratio=0.0,
        route_progress_m=progress,
        route_heading_deg=0.0,
        heading_error_to_route_deg=0.0,
        cross_track_error_m=0.0,
        signed_cross_track_error_m=0.0,
        remaining_route_m=max(0.0, cumulative[-1] - progress),
    )
    route_heading = route_heading_from_projection(
        route,
        projection,
        current_yaw_deg=0.0,
        heading_lookahead_m=heading_lookahead_m,
    )
    return route_heading.heading_deg


def project_point_to_route(
    points,
    current_pose,
    start_segment_index=0,
    previous_progress_m=None,
    max_forward_jump_m=None,
    backward_tolerance_m=0.03,
    allow_backward=False,
    projection_status="locked",
):
    route = [(float(x), float(y)) for x, y in points]
    if len(route) < 2:
        raise RuntimeError("route_projection_path_too_short")
    cumulative = route_cumulative_distances(route)
    if cumulative[-1] <= 1e-9:
        raise RuntimeError("route_projection_path_too_short")

    start_index = max(0, min(int(start_segment_index), len(route) - 2))
    max_progress = None
    if previous_progress_m is not None and max_forward_jump_m is not None:
        max_progress = float(previous_progress_m) + max(0.0, float(max_forward_jump_m))

    current = (float(current_pose.x), float(current_pose.y))
    best = None
    for index in range(start_index, len(route) - 1):
        start = route[index]
        end = route[index + 1]
        segment_length = cumulative[index + 1] - cumulative[index]
        if segment_length <= 1e-9:
            continue
        distance_m, ratio, projected = _projection_on_segment(current, start, end)
        progress_m = cumulative[index] + ratio * segment_length
        if max_progress is not None and progress_m > max_progress + 1e-9:
            continue
        candidate = (distance_m, progress_m, index, ratio, projected)
        if best is None or candidate < best:
            best = candidate

    if best is None:
        raise RuntimeError("route_projection_forward_jump")

    distance_m, progress_m, segment_index, ratio, projected = best
    route_progress_delta_m = None
    route_progress_backward_delta_m = 0.0
    route_progress_forward_delta_m = 0.0
    if previous_progress_m is not None and progress_m < float(previous_progress_m):
        backward_m = float(previous_progress_m) - progress_m
        route_progress_delta_m = progress_m - float(previous_progress_m)
        route_progress_backward_delta_m = backward_m
        if not allow_backward and backward_m > backward_tolerance_m:
            raise RuntimeError("route_projection_moved_backward")
        if not allow_backward:
            x, y, segment_index, ratio = route_point_at_progress(
                route,
                cumulative,
                previous_progress_m,
            )
            projected = (x, y)
            progress_m = float(previous_progress_m)
            distance_m = distance_2d(current, projected)
            route_progress_delta_m = 0.0
    elif previous_progress_m is not None:
        route_progress_delta_m = progress_m - float(previous_progress_m)
        route_progress_forward_delta_m = max(0.0, route_progress_delta_m)

    start = route[segment_index]
    end = route[segment_index + 1]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    route_heading_deg = math.degrees(math.atan2(dy, dx))
    heading_error = shortest_angle_delta_deg(float(current_pose.yaw_deg), route_heading_deg)
    signed_error = (
        dx * (current[1] - projected[1])
        - dy * (current[0] - projected[0])
    ) / max(1e-9, math.hypot(dx, dy))
    return RouteProjection(
        projected_point=projected,
        segment_index=segment_index,
        segment_ratio=ratio,
        route_progress_m=progress_m,
        route_heading_deg=route_heading_deg,
        heading_error_to_route_deg=heading_error,
        cross_track_error_m=distance_m,
        signed_cross_track_error_m=signed_error,
        remaining_route_m=max(0.0, cumulative[-1] - progress_m),
        projection_status=projection_status,
        route_progress_delta_m=route_progress_delta_m,
        route_progress_backward_delta_m=route_progress_backward_delta_m,
        route_progress_forward_delta_m=route_progress_forward_delta_m,
    )


def project_point_to_route_branch_window(
    points,
    current_pose,
    min_progress_m,
    max_progress_m,
    preferred_heading_deg,
    heading_tolerance_deg,
    previous_progress_m=None,
    heading_lookahead_m=ROUTE_HEADING_LOOKAHEAD_M,
    projection_status="post_rotate_branch_lock",
    stable_count=0,
    branch_lock_start_progress_m=None,
    branch_lock_release_required_span_m=0.0,
):
    route = [(float(x), float(y)) for x, y in points]
    if len(route) < 2:
        raise RuntimeError("route_projection_path_too_short")
    cumulative = route_cumulative_distances(route)
    if cumulative[-1] <= 1e-9:
        raise RuntimeError("route_projection_path_too_short")

    window_min = clamp(float(min_progress_m), 0.0, cumulative[-1])
    window_max = clamp(float(max_progress_m), 0.0, cumulative[-1])
    if window_max < window_min:
        raise RuntimeError("pure_pursuit_branch_ambiguous")

    current = (float(current_pose.x), float(current_pose.y))
    best = None
    rejected_wrong_heading_count = 0
    max_rejected_heading_error = 0.0
    preferred_heading = float(preferred_heading_deg)
    heading_tolerance = abs(float(heading_tolerance_deg))

    for index in range(len(route) - 1):
        start_progress = cumulative[index]
        end_progress = cumulative[index + 1]
        segment_length = end_progress - start_progress
        if segment_length <= 1e-9:
            continue
        overlap_start = max(start_progress, window_min)
        overlap_end = min(end_progress, window_max)
        if overlap_end < overlap_start - 1e-9:
            continue

        if overlap_end - overlap_start <= 1e-9:
            point_x, point_y, _segment_index, _ratio = route_point_at_progress(
                route,
                cumulative,
                overlap_start,
            )
            projected = (point_x, point_y)
            ratio = clamp(
                (overlap_start - start_progress) / segment_length,
                0.0,
                1.0,
            )
            distance_m = distance_2d(current, projected)
            progress_m = overlap_start
        else:
            start_ratio = clamp(
                (overlap_start - start_progress) / segment_length,
                0.0,
                1.0,
            )
            end_ratio = clamp(
                (overlap_end - start_progress) / segment_length,
                0.0,
                1.0,
            )
            start = route[index]
            end = route[index + 1]
            clipped_start = (
                start[0] + start_ratio * (end[0] - start[0]),
                start[1] + start_ratio * (end[1] - start[1]),
            )
            clipped_end = (
                start[0] + end_ratio * (end[0] - start[0]),
                start[1] + end_ratio * (end[1] - start[1]),
            )
            distance_m, clipped_ratio, projected = _projection_on_segment(
                current,
                clipped_start,
                clipped_end,
            )
            progress_m = overlap_start + clipped_ratio * (overlap_end - overlap_start)
            ratio = clamp(
                (progress_m - start_progress) / segment_length,
                0.0,
                1.0,
            )

        candidate_heading = route_heading_at_progress(
            route,
            progress_m,
            heading_lookahead_m,
        )
        if candidate_heading is None:
            continue
        heading_error = abs(shortest_angle_delta_deg(preferred_heading, candidate_heading))
        if heading_error > heading_tolerance:
            rejected_wrong_heading_count += 1
            max_rejected_heading_error = max(max_rejected_heading_error, heading_error)
            continue

        candidate = (distance_m, heading_error, -progress_m, index, ratio, projected, candidate_heading)
        if best is None or candidate < best:
            best = candidate

    if best is None:
        raise RuntimeError("pure_pursuit_branch_ambiguous")

    (
        distance_m,
        heading_error,
        negative_selected_progress_m,
        segment_index,
        selected_ratio,
        selected_projected,
        selected_heading,
    ) = best
    selected_progress_m = -negative_selected_progress_m
    effective_progress_m = selected_progress_m
    if previous_progress_m is not None:
        effective_progress_m = max(float(previous_progress_m), selected_progress_m)
    x, y, effective_segment_index, effective_ratio = route_point_at_progress(
        route,
        cumulative,
        effective_progress_m,
    )
    effective_projected = (x, y)
    effective_cross_track = distance_2d(current, effective_projected)
    route_progress_delta_m = None
    route_progress_backward_delta_m = 0.0
    route_progress_forward_delta_m = 0.0
    if previous_progress_m is not None:
        route_progress_delta_m = effective_progress_m - float(previous_progress_m)
        route_progress_backward_delta_m = max(0.0, -route_progress_delta_m)
        route_progress_forward_delta_m = max(0.0, route_progress_delta_m)

    start = route[effective_segment_index]
    end = route[effective_segment_index + 1]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    signed_error = (
        dx * (current[1] - effective_projected[1])
        - dy * (current[0] - effective_projected[0])
    ) / max(1e-9, math.hypot(dx, dy))
    selected_segment_heading = math.degrees(math.atan2(dy, dx))
    effective_heading = route_heading_at_progress(
        route,
        effective_progress_m,
        heading_lookahead_m,
    )
    if effective_heading is None:
        effective_heading = selected_heading
    effective_heading_error = abs(
        shortest_angle_delta_deg(preferred_heading, effective_heading)
    )
    branch_lock_progress_span_m = 0.0
    if branch_lock_start_progress_m is not None:
        branch_lock_progress_span_m = max(
            0.0,
            effective_progress_m - float(branch_lock_start_progress_m),
        )
    return RouteProjection(
        projected_point=effective_projected,
        segment_index=effective_segment_index,
        segment_ratio=effective_ratio,
        route_progress_m=effective_progress_m,
        route_heading_deg=effective_heading,
        heading_error_to_route_deg=shortest_angle_delta_deg(
            float(current_pose.yaw_deg),
            effective_heading,
        ),
        cross_track_error_m=effective_cross_track,
        signed_cross_track_error_m=signed_error,
        remaining_route_m=max(0.0, cumulative[-1] - effective_progress_m),
        projection_status=projection_status,
        route_progress_delta_m=route_progress_delta_m,
        route_progress_backward_delta_m=route_progress_backward_delta_m,
        route_progress_forward_delta_m=route_progress_forward_delta_m,
        raw_projection_progress_m=selected_progress_m,
        raw_projection_segment_index=segment_index,
        effective_projection_progress_m=effective_progress_m,
        preferred_branch_heading_deg=preferred_heading,
        selected_segment_heading_deg=selected_segment_heading,
        selected_branch_heading_error_deg=effective_heading_error,
        rejected_wrong_heading_segment_count=rejected_wrong_heading_count,
        branch_lock_stable_count=int(stable_count),
        branch_lock_progress_span_m=branch_lock_progress_span_m,
        branch_lock_release_required_span_m=float(branch_lock_release_required_span_m),
        local_cross_track_m=distance_m,
    )


def project_point_to_route_progress_window(
    points,
    current_pose,
    min_progress_m,
    max_progress_m,
    previous_progress_m=None,
    projection_status="rotate_anchor_raw",
):
    route = [(float(x), float(y)) for x, y in points]
    if len(route) < 2:
        raise RuntimeError("route_projection_path_too_short")
    cumulative = route_cumulative_distances(route)
    if cumulative[-1] <= 1e-9:
        raise RuntimeError("route_projection_path_too_short")

    window_min = clamp(float(min_progress_m), 0.0, cumulative[-1])
    window_max = clamp(float(max_progress_m), 0.0, cumulative[-1])
    if window_max < window_min:
        raise RuntimeError("route_projection_local_window_empty")

    current = (float(current_pose.x), float(current_pose.y))
    best = None
    for index in range(len(route) - 1):
        start_progress = cumulative[index]
        end_progress = cumulative[index + 1]
        segment_length = end_progress - start_progress
        if segment_length <= 1e-9:
            continue
        overlap_start = max(start_progress, window_min)
        overlap_end = min(end_progress, window_max)
        if overlap_end < overlap_start - 1e-9:
            continue

        if overlap_end - overlap_start <= 1e-9:
            point_x, point_y, _segment_index, _ratio = route_point_at_progress(
                route,
                cumulative,
                overlap_start,
            )
            projected = (point_x, point_y)
            ratio = clamp(
                (overlap_start - start_progress) / segment_length,
                0.0,
                1.0,
            )
            distance_m = distance_2d(current, projected)
            progress_m = overlap_start
        else:
            start_ratio = clamp(
                (overlap_start - start_progress) / segment_length,
                0.0,
                1.0,
            )
            end_ratio = clamp(
                (overlap_end - start_progress) / segment_length,
                0.0,
                1.0,
            )
            start = route[index]
            end = route[index + 1]
            clipped_start = (
                start[0] + start_ratio * (end[0] - start[0]),
                start[1] + start_ratio * (end[1] - start[1]),
            )
            clipped_end = (
                start[0] + end_ratio * (end[0] - start[0]),
                start[1] + end_ratio * (end[1] - start[1]),
            )
            distance_m, clipped_ratio, projected = _projection_on_segment(
                current,
                clipped_start,
                clipped_end,
            )
            progress_m = overlap_start + clipped_ratio * (overlap_end - overlap_start)
            ratio = clamp(
                (progress_m - start_progress) / segment_length,
                0.0,
                1.0,
            )

        candidate = (distance_m, progress_m, index, ratio, projected)
        if best is None or candidate < best:
            best = candidate

    if best is None:
        raise RuntimeError("route_projection_local_window_empty")

    distance_m, progress_m, segment_index, ratio, projected = best
    route_progress_delta_m = None
    route_progress_backward_delta_m = 0.0
    route_progress_forward_delta_m = 0.0
    if previous_progress_m is not None:
        route_progress_delta_m = progress_m - float(previous_progress_m)
        route_progress_backward_delta_m = max(0.0, -route_progress_delta_m)
        route_progress_forward_delta_m = max(0.0, route_progress_delta_m)

    start = route[segment_index]
    end = route[segment_index + 1]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    route_heading_deg = math.degrees(math.atan2(dy, dx))
    heading_error = shortest_angle_delta_deg(float(current_pose.yaw_deg), route_heading_deg)
    signed_error = (
        dx * (current[1] - projected[1])
        - dy * (current[0] - projected[0])
    ) / max(1e-9, math.hypot(dx, dy))
    return RouteProjection(
        projected_point=projected,
        segment_index=segment_index,
        segment_ratio=ratio,
        route_progress_m=progress_m,
        route_heading_deg=route_heading_deg,
        heading_error_to_route_deg=heading_error,
        cross_track_error_m=distance_m,
        signed_cross_track_error_m=signed_error,
        remaining_route_m=max(0.0, cumulative[-1] - progress_m),
        projection_status=projection_status,
        route_progress_delta_m=route_progress_delta_m,
        route_progress_backward_delta_m=route_progress_backward_delta_m,
        route_progress_forward_delta_m=route_progress_forward_delta_m,
    )


def route_points_from_projection(points, projection):
    route = [(float(x), float(y)) for x, y in points]
    if not route:
        return []
    index = max(0, min(int(projection.segment_index), len(route) - 1))
    result = [projection.projected_point]
    result.extend(route[index + 1 :])
    return result


def lookahead_target_from_route_anchor(path_points, lookahead_m):
    points = [(float(x), float(y)) for x, y in path_points]
    if not points:
        raise RuntimeError("curve_path_too_short")
    if len(points) == 1:
        return points[0]
    remaining = max(0.0, float(lookahead_m))
    previous = points[0]
    for point in points[1:]:
        segment = distance_2d(previous, point)
        if segment <= 1e-9:
            previous = point
            continue
        if segment >= remaining:
            ratio = 0.0 if segment <= 1e-9 else remaining / segment
            return (
                previous[0] + ratio * (point[0] - previous[0]),
                previous[1] + ratio * (point[1] - previous[1]),
            )
        remaining -= segment
        previous = point
    return points[-1]


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
    rotate_start_heading_error_deg=90.0,
):
    dx = float(target_point[0]) - float(current_pose.x)
    dy = float(target_point[1]) - float(current_pose.y)
    target_heading = math.atan2(dy, dx)
    yaw = math.radians(float(current_pose.yaw_deg))
    alpha = normalize_angle_rad(target_heading - yaw)
    rotate_start_rad = max(1e-6, math.radians(abs(rotate_start_heading_error_deg)))
    abs_alpha = abs(alpha)
    if abs_alpha >= rotate_start_rad:
        linear_scale = 0.0
    else:
        start_cos = math.cos(rotate_start_rad)
        denominator = max(1e-6, 1.0 - start_cos)
        linear_scale = clamp((math.cos(abs_alpha) - start_cos) / denominator, 0.0, 1.0)
    linear_x = abs(linear_speed_mps) * linear_scale
    angular_z = clamp(
        2.0 * linear_x * math.sin(alpha) / max(0.01, lookahead_m),
        -abs(max_angular_rad_s),
        abs(max_angular_rad_s),
    )
    return linear_x, angular_z, alpha
