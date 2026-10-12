"""CSV routes, corner-preserving downsampling and path-progress selection."""

import csv
import math
from pathlib import Path

from .math_utils import clamp, shortest_angle_delta_deg
from .models import StartSelection, TargetState, Waypoint


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
            if not math.isfinite(waypoint.x) or not math.isfinite(waypoint.y):
                raise ValueError(f"{path} contains a non-finite waypoint")
            xy = (waypoint.x, waypoint.y)
            if previous_xy is not None and xy == previous_xy:
                continue
            previous_xy = xy
            waypoints.append(waypoint)

    if not waypoints:
        raise ValueError(f"{path} does not contain any waypoints")
    return waypoints


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
