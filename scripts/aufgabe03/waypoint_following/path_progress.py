from __future__ import annotations

import csv
import math
from pathlib import Path

from .math_utils import clamp, shortest_angle_delta_deg
from .models import (
    Pose2D,
    StartSelection,
    TargetState,
    TrackingPathValidation,
    Waypoint,
)


TRACKING_PATH_SMOOTHING_OFF = "off"
TRACKING_PATH_SMOOTHING_SHORTCUT = "shortcut"
TRACKING_PATH_SMOOTHING_MODES = (
    TRACKING_PATH_SMOOTHING_OFF,
    TRACKING_PATH_SMOOTHING_SHORTCUT,
)
DEFAULT_TRACKING_PATH_SMOOTHING = TRACKING_PATH_SMOOTHING_OFF
DEFAULT_TRACKING_PATH_SMOOTHING_SPACING_M = 0.05


def waypoint_distance(a, b):
    return math.hypot(b.x - a.x, b.y - a.y)


def heading_between(a, b):
    return math.degrees(math.atan2(b.y - a.y, b.x - a.x))


def target_state(current_pose, waypoint):
    dx = waypoint.x - current_pose.x
    dy = waypoint.y - current_pose.y
    heading = math.degrees(math.atan2(dy, dx))
    return TargetState(
        distance_m=math.hypot(dx, dy),
        heading_deg=heading,
        yaw_error_deg=shortest_angle_delta_deg(current_pose.yaw_deg, heading),
    )


def waypoint_reached(distance_m, is_final, waypoint_tolerance_m, goal_tolerance_m):
    tolerance = goal_tolerance_m if is_final else waypoint_tolerance_m
    return distance_m <= tolerance


def load_waypoints(path):
    path = Path(path)
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        required = {"index", "world_x_m", "world_y_m"}
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(
                f"{path} is missing required column(s): {', '.join(missing)}"
            )

        waypoints = []
        previous_xy = None
        for row in reader:
            waypoint = Waypoint(
                index=int(float(row["index"])),
                x=float(row["world_x_m"]),
                y=float(row["world_y_m"]),
            )
            xy = (waypoint.x, waypoint.y)
            if previous_xy is not None and xy == previous_xy:
                continue
            previous_xy = xy
            waypoints.append(waypoint)

    if not waypoints:
        raise ValueError(f"{path} does not contain any waypoints")
    return waypoints


def _read_waypoint_csv_rows(path, label):
    path = Path(path)
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        required = {"index", "world_x_m", "world_y_m"}
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(
                f"{path} is missing required column(s): {', '.join(missing)}"
            )

        points = []
        for row_number, row in enumerate(reader, start=2):
            try:
                point = Waypoint(
                    index=int(float(row["index"])),
                    x=float(row["world_x_m"]),
                    y=float(row["world_y_m"]),
                )
            except Exception as exc:
                raise ValueError(
                    f"{path} has malformed {label} row {row_number}: {exc}"
                ) from exc
            if not math.isfinite(point.x) or not math.isfinite(point.y):
                raise ValueError(
                    f"{path} has non-finite {label} coordinate at row {row_number}"
                )
            points.append(point)
    return points


def validate_tracking_point_structure(
    points,
    max_segment_m,
    label="tracking path",
    duplicate_tolerance_m=1e-9,
):
    points = list(points)
    if len(points) < 2:
        raise ValueError(f"{label} needs at least two tracking points")

    warnings = []
    duplicate_segments = 0
    segment_count = len(points) - 1
    for index in range(1, len(points)):
        segment_m = waypoint_distance(points[index - 1], points[index])
        if segment_m <= duplicate_tolerance_m:
            duplicate_segments += 1
            continue
        if segment_m > max_segment_m:
            raise ValueError(
                f"{label} segment jump exceeds limit at segment {index - 1}: "
                f"distance={segment_m:.3f} m, limit={max_segment_m:.3f} m"
            )

    if duplicate_segments:
        duplicate_ratio = duplicate_segments / segment_count
        if duplicate_ratio > 0.5:
            raise ValueError(
                f"{label} has too many duplicate segments: "
                f"{duplicate_segments}/{segment_count}"
            )
        warnings.append(
            f"{label} contains {duplicate_segments} duplicate segment(s)"
        )
    return tuple(warnings)


def load_tracking_path_csv(path, max_segment_m):
    points = _read_waypoint_csv_rows(path, "tracking path")
    warnings = validate_tracking_point_structure(
        points,
        max_segment_m=max_segment_m,
        label=f"{Path(path)} tracking path",
    )
    return points, warnings


def validate_tracking_path_geometry(
    route_waypoints,
    tracking_points,
    endpoint_tolerance_m,
    start_tolerance_m,
    allow_mismatch=False,
    current_pose=None,
    source="csv",
    structural_warnings=(),
):
    route_waypoints = list(route_waypoints)
    tracking_points = list(tracking_points)
    if not route_waypoints:
        raise ValueError("tracking path validation needs at least one route waypoint")
    if len(tracking_points) < 2:
        raise ValueError("tracking path needs at least two tracking points")

    start_error_m = waypoint_distance(tracking_points[0], route_waypoints[0])
    endpoint_error_m = waypoint_distance(tracking_points[-1], route_waypoints[-1])
    start_projection_error_m = None
    if current_pose is not None:
        start_projection_error_m, _segment_index, _projection = nearest_path_segment(
            current_pose,
            tracking_points,
        )

    start_ok = start_error_m <= start_tolerance_m
    if current_pose is not None and start_projection_error_m is not None:
        start_ok = start_ok or start_projection_error_m <= start_tolerance_m
    endpoint_ok = endpoint_error_m <= endpoint_tolerance_m

    warnings = list(structural_warnings)
    status = "ok"
    mismatch_reasons = []
    if not start_ok:
        mismatch_reasons.append(
            f"start_error={start_error_m:.3f} m"
            if start_projection_error_m is None
            else (
                f"start_error={start_error_m:.3f} m, "
                f"projection_error={start_projection_error_m:.3f} m"
            )
        )
    if not endpoint_ok:
        mismatch_reasons.append(f"endpoint_error={endpoint_error_m:.3f} m")

    if mismatch_reasons:
        message = (
            "tracking path geometric mismatch: "
            + "; ".join(mismatch_reasons)
            + (
                f"; start_tolerance={start_tolerance_m:.3f} m, "
                f"endpoint_tolerance={endpoint_tolerance_m:.3f} m"
            )
        )
        if not allow_mismatch:
            raise ValueError(message)
        status = "mismatch_allowed"
        warnings.append(message)

    return TrackingPathValidation(
        source=source,
        point_count=len(tracking_points),
        endpoint_error_m=endpoint_error_m,
        start_error_m=start_error_m,
        start_projection_error_m=start_projection_error_m,
        validation_status=status,
        warnings=tuple(warnings),
    )


def is_heading_change(previous_wp, current_wp, next_wp, tolerance_deg=1.0):
    incoming = heading_between(previous_wp, current_wp)
    outgoing = heading_between(current_wp, next_wp)
    return abs(shortest_angle_delta_deg(incoming, outgoing)) > tolerance_deg


def downsample_waypoints(waypoints, min_spacing_m):
    if len(waypoints) <= 2:
        return list(waypoints)

    selected = [waypoints[0]]
    for index in range(1, len(waypoints) - 1):
        current = waypoints[index]
        if is_heading_change(waypoints[index - 1], current, waypoints[index + 1]):
            selected.append(current)
            continue
        if waypoint_distance(selected[-1], current) >= min_spacing_m:
            selected.append(current)

    if selected[-1] != waypoints[-1]:
        selected.append(waypoints[-1])
    return selected


def prepare_executable_waypoints(waypoints, skip_first=True, min_spacing_m=0.0):
    executable = list(waypoints[1:] if skip_first else waypoints)
    if min_spacing_m > 0.0:
        executable = downsample_waypoints(executable, min_spacing_m)
    if len(executable) < 2:
        raise ValueError(
            "Waypoint CSV needs at least two executable waypoints after processing"
        )
    return executable


def distance_point_to_segment_m(point, segment_start, segment_end):
    dx = segment_end.x - segment_start.x
    dy = segment_end.y - segment_start.y
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return math.hypot(point.x - segment_start.x, point.y - segment_start.y), 0.0
    projection = (
        (point.x - segment_start.x) * dx + (point.y - segment_start.y) * dy
    ) / length_sq
    projection = clamp(projection, 0.0, 1.0)
    closest_x = segment_start.x + projection * dx
    closest_y = segment_start.y + projection * dy
    return math.hypot(point.x - closest_x, point.y - closest_y), projection


def nearest_path_segment(point, waypoints):
    if len(waypoints) < 2:
        raise ValueError("Need at least two waypoints for path-progress selection")

    best = None
    for segment_index in range(len(waypoints) - 1):
        distance_m, projection = distance_point_to_segment_m(
            point,
            waypoints[segment_index],
            waypoints[segment_index + 1],
        )
        candidate = (distance_m, segment_index, projection)
        if best is None or candidate < best:
            best = candidate
    return best


def select_path_progress_waypoints(
    waypoints,
    current_pose,
    start_on_path_tolerance_m,
    waypoint_tolerance_m,
    goal_tolerance_m,
    min_spacing_m=0.0,
):
    distance_to_path_m, segment_index, _projection = nearest_path_segment(
        current_pose,
        waypoints,
    )
    if distance_to_path_m > start_on_path_tolerance_m:
        raise ValueError(
            "Current pose is too far from the planned path: "
            f"distance={distance_to_path_m:.3f} m, "
            f"tolerance={start_on_path_tolerance_m:.3f} m"
        )

    next_index = min(segment_index + 1, len(waypoints) - 1)
    while next_index < len(waypoints) - 1:
        waypoint = waypoints[next_index]
        distance_m = math.hypot(waypoint.x - current_pose.x, waypoint.y - current_pose.y)
        if not waypoint_reached(
            distance_m,
            is_final=False,
            waypoint_tolerance_m=waypoint_tolerance_m,
            goal_tolerance_m=goal_tolerance_m,
        ):
            break
        next_index += 1

    selected = list(waypoints[next_index:])
    if min_spacing_m > 0.0 and len(selected) > 1:
        selected = downsample_waypoints(selected, min_spacing_m)
    if not selected:
        selected = [waypoints[-1]]

    return StartSelection(
        waypoints=selected,
        selected_segment_index=segment_index,
        selected_waypoint_index=selected[0].index,
        distance_to_path_m=distance_to_path_m,
    )


def select_executable_waypoints(
    waypoints,
    current_pose,
    start_selection,
    start_on_path_tolerance_m,
    waypoint_tolerance_m,
    goal_tolerance_m,
    min_spacing_m,
    skip_first=True,
):
    if start_selection == "fixed-skip":
        selected = prepare_executable_waypoints(
            waypoints,
            skip_first=skip_first,
            min_spacing_m=min_spacing_m,
        )
        return StartSelection(
            waypoints=selected,
            selected_segment_index=None,
            selected_waypoint_index=selected[0].index,
            distance_to_path_m=None,
        )
    if start_selection == "path-progress":
        return select_path_progress_waypoints(
            waypoints,
            current_pose,
            start_on_path_tolerance_m,
            waypoint_tolerance_m,
            goal_tolerance_m,
            min_spacing_m=min_spacing_m,
        )
    raise ValueError(f"unsupported start selection mode: {start_selection!r}")
