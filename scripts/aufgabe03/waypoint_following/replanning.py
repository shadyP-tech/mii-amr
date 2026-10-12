"""Route validation and stopped replan acceptance with unchanged final goals."""

import math

import lidar_obstacle_map
import map_path_planner as planner
import replan_runtime
from .models import Waypoint
from .path_progress import select_path_progress_waypoints, waypoint_distance


def validate_route(occupancy_map, blocked, pose, waypoints):
    if not waypoints:
        raise RuntimeError("empty route")
    previous = (pose.x, pose.y)
    for waypoint in waypoints:
        if not all(math.isfinite(value) for value in (waypoint.x, waypoint.y)):
            raise RuntimeError("non-finite route coordinate")
        point = (waypoint.x, waypoint.y)
        if not planner.world_segment_is_clear(occupancy_map, blocked, previous, point):
            raise RuntimeError(f"route segment to waypoint {waypoint.index} intersects inflated obstacles or map boundary")
        previous = point


def select_remaining(waypoints, pose, args):
    if len(waypoints) == 1:
        if waypoint_distance(pose, waypoints[0]) > args.start_on_path_tolerance_m:
            raise RuntimeError("current pose too far from single-waypoint route")
        return list(waypoints)
    return select_path_progress_waypoints(
        waypoints, pose, args.start_on_path_tolerance_m,
        args.waypoint_tolerance_m, args.goal_tolerance_m, args.min_waypoint_spacing_m,
    ).waypoints


def accept_replan(result, pose, goal, previous_remaining, args):
    if not result.success:
        raise RuntimeError(f"LiDAR replan failed while stopped: {result.reason}")
    if result.run_local_map is None:
        raise RuntimeError("replan did not return a run-local map")
    route = [Waypoint(int(index), float(x), float(y)) for index, x, y in result.waypoints]
    if not route or any(not math.isfinite(value) for point in route for value in (point.x, point.y)):
        raise RuntimeError("replan returned an empty or non-finite route")
    if waypoint_distance(pose, route[0]) > args.max_start_snap_m:
        raise RuntimeError("replan start snap exceeds limit")
    if waypoint_distance(goal, route[-1]) > args.max_goal_snap_m:
        raise RuntimeError("replan goal snap exceeds limit")
    # Grid-cell snapping never silently changes the requested terminal goal.
    if waypoint_distance(goal, route[-1]) > 1e-9:
        route.append(goal)
    else:
        route[-1] = goal
    while len(route) > 1 and waypoint_distance(pose, route[0]) <= args.waypoint_tolerance_m:
        route.pop(0)
    occupancy_map = result.run_local_map.static_map
    static_blocked, _ = planner.inflate_blocked_cells(
        occupancy_map, args.robot_footprint_radius_m + args.run_local_map_clearance_margin_m,
    )
    blocked = static_blocked | set(result.inflated_obstacle_cells)
    # Check every segment after pruning, especially the new pose->first target
    # connector, which was not necessarily an edge of the original A* plan.
    validate_route(occupancy_map, blocked, pose, route)
    old_points = [(pose.x, pose.y), *((point.x, point.y) for point in previous_remaining)]
    new_points = [(pose.x, pose.y), *((point.x, point.y) for point in route)]
    old_length = sum(math.dist(a, b) for a, b in zip(old_points, old_points[1:]))
    new_length = sum(math.dist(a, b) for a, b in zip(new_points, new_points[1:]))
    if new_length > max(0.75, old_length) * args.max_replan_path_length_ratio:
        raise RuntimeError("accepted replan exceeds path-length bound")
    return route, blocked


def replan(node, pose, goal, remaining, initial=False):
    args = node.args
    node.stop_repeatedly()
    if not initial and node.replan_count >= args.max_replans:
        raise RuntimeError("maximum live replan count reached")
    if node.run_local_map is not None and node.run_local_map.update_count >= args.run_local_map_max_updates:
        raise RuntimeError("maximum run-local map updates reached")
    if not initial:
        node.replan_count += 1
    if initial:
        result = replan_runtime.perform_initial_run_local_replan(node, args, pose, goal, remaining)
    else:
        result = replan_runtime.perform_lidar_replan(node, args, pose, goal, remaining, node.replan_count)
    node.stop_repeatedly()
    # Never resume using pre-scan or pre-planning pose/localization state.
    pose = node.wait_for_health(args.health_recovery_time_sec)
    sparse_initial = (
        initial and not result.success and result.reason in {
            lidar_obstacle_map.RUN_LOCAL_FAILURE_TOO_FEW_SCAN_POINTS,
            lidar_obstacle_map.RUN_LOCAL_FAILURE_TOO_MANY_REJECTED_POINTS,
        } and result.run_local_map is not None
        and not result.run_local_map.confirmed_raw_cells
        and not result.run_local_map.inflated_obstacle_cells
    )
    if sparse_initial:
        # A healthy initial scan may contain only known walls. No dynamic map
        # change is accepted; the original static route still must be clear.
        validate_route(node.static_map, node.blocked_cells, pose, remaining)
        node.run_local_map = result.run_local_map
        return list(remaining)
    route, blocked = accept_replan(result, pose, goal, remaining, args)
    node.run_local_map = result.run_local_map
    node.blocked_cells = blocked
    return route
