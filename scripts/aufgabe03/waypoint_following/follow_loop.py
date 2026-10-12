"""One bounded stop-go control loop with stopped health recovery and replanning."""

import time

import map_path_planner as planner
from .controllers import StopGoController
from .models import RouteState
from .replanning import replan, select_remaining, validate_route
from .ros_runtime import BlockedError, HealthError


def follow_route(node, waypoints):
    args = node.args
    node.stop_repeatedly()
    pose = node.wait_for_health(args.startup_timeout_sec)
    node.static_map = planner.load_occupancy_map(args.static_map)
    node.blocked_cells, _ = planner.inflate_blocked_cells(
        node.static_map, args.robot_footprint_radius_m + args.run_local_map_clearance_margin_m,
    )
    goal = waypoints[-1]
    remaining = select_remaining(waypoints, pose, args)
    if args.run_local_obstacle_map:
        remaining = replan(node, pose, goal, remaining, initial=True)
        pose = node.wait_for_health(args.health_recovery_time_sec)
    validate_route(node.static_map, node.blocked_cells, pose, remaining)
    route = RouteState(remaining)
    controller = StopGoController(args)
    waypoint_started = time.monotonic()
    period = 1.0 / args.control_rate_hz
    while node.is_running() and not route.complete:
        cycle_started = time.monotonic()
        node.spin_once(0.0)
        if time.monotonic() - waypoint_started > args.max_waypoint_time_sec:
            raise RuntimeError("waypoint motion timeout")
        try:
            pose = node.check_health_or_raise()
            # Reaching a waypoint within tolerance changes the next connector.
            # Recheck it at the current pose, including after every recovery.
            validate_route(node.static_map, node.blocked_cells, pose, [route.current_waypoint()])
            step = controller.compute(pose, route)
            if step.reached:
                node.publish_velocity(0.0, 0.0)
                route.advance()
                waypoint_started = time.monotonic()
                continue
            node.publish_velocity(step.command.linear_x, step.command.angular_z)
        except HealthError:
            node.stop_repeatedly()
            node.wait_for_health(args.health_recovery_time_sec)
            controller = StopGoController(args)
            continue
        except BlockedError as exc:
            node.stop_repeatedly()
            if (not args.replan_on_obstacle or exc.mode != "forward"
                    or exc.safety.reason != "soft_stop"):
                raise
            pose = node.wait_for_health(args.health_recovery_time_sec)
            route = RouteState(replan(node, pose, goal, route.remaining()))
            controller = StopGoController(args)
            waypoint_started = time.monotonic()
            continue
        remaining_time = period - (time.monotonic() - cycle_started)
        if remaining_time > 0:
            time.sleep(remaining_time)
    if not route.complete:
        raise RuntimeError("ROS shutdown before route completion")
    node.stop_repeatedly()
    return True
