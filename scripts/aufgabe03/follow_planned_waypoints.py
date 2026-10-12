#!/usr/bin/env python3
"""Run the single conservative waypoint follower after AMCL localization."""

import sys

import map_path_planner
from waypoint_following.follower_cli import build_arg_parser, parse_args
from waypoint_following.path_progress import load_waypoints
from waypoint_following.replanning import validate_route
from waypoint_following.ros_runtime import WaypointFollower, rclpy


def main(argv=None):
    args = parse_args(argv)
    node = None
    initialized = False
    try:
        waypoints = load_waypoints(args.waypoints_csv)
        occupancy_map = map_path_planner.load_occupancy_map(args.static_map)
        if args.dry_run:
            blocked, _ = map_path_planner.inflate_blocked_cells(
                occupancy_map, args.robot_footprint_radius_m + args.run_local_map_clearance_margin_m,
            )
            validate_route(occupancy_map, blocked, waypoints[0], waypoints)
            print(f"Validated {len(waypoints)} waypoints and static collision clearance; no robot commands sent.")
            return 0
        if not args.yes:
            raise RuntimeError("live motion requires --yes after localization and exclusive /cmd_vel handoff")
        if rclpy is None:
            raise RuntimeError("ROS 2 dependencies unavailable; use --dry-run for offline validation")
        rclpy.init(args=[])
        initialized = True
        node = WaypointFollower(args)
        node.follow(waypoints)
        print("Waypoint route complete.")
        return 0
    except KeyboardInterrupt:
        print("Waypoint following interrupted; stopping.", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"follow_planned_waypoints.py: {exc}", file=sys.stderr)
        return 1
    finally:
        if node is not None:
            try:
                node.stop_repeatedly()
            finally:
                node.destroy_node()
        if initialized and rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
