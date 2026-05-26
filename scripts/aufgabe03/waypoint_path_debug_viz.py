#!/usr/bin/env python3
"""
Publish a waypoint CSV as RViz path and waypoint marker topics.

This node is visualization-only. It does not publish /cmd_vel and does not
interact with Nav2. Keep it running while setting up RViz so the planned-path
topics are visible in the "Add by topic" panel before the custom follower runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import rclpy
    from rclpy.node import Node
except ImportError:
    rclpy = None
    Node = object

import follow_planned_waypoints as follower


DEFAULT_WAYPOINTS_CSV = follower.DEFAULT_WAYPOINTS_CSV
DEFAULT_PATH_TOPIC = follower.DEFAULT_RVIZ_PATH_TOPIC
DEFAULT_WAYPOINT_MARKER_TOPIC = follower.DEFAULT_RVIZ_WAYPOINT_MARKER_TOPIC
DEFAULT_MAP_FRAME = "map"
DEFAULT_WATCH_PERIOD_SEC = 1.0


def load_display_waypoints(args):
    waypoints = follower.load_waypoints(args.waypoints)
    return follower.prepare_executable_waypoints(
        waypoints,
        skip_first=args.skip_first_waypoint,
        min_spacing_m=args.min_waypoint_spacing_m,
    )


def waypoint_file_mtime_ns(path):
    return Path(path).stat().st_mtime_ns


def build_route_messages(args, waypoints, stamp):
    return (
        follower.build_rviz_path_message(
            waypoints,
            args.map_frame,
            stamp,
            current_pose=None,
        ),
        follower.build_rviz_waypoint_markers(
            waypoints,
            args.map_frame,
            stamp,
            current_waypoint_index=args.current_waypoint_index,
        ),
    )


class WaypointPathDebugViz(Node):
    def __init__(self, args):
        if rclpy is None:
            raise RuntimeError("ROS2 Python packages are required to run this RViz publisher.")
        if not follower.rviz_messages_available():
            raise RuntimeError(
                "ROS RViz message types are unavailable. Source ROS 2 Humble before "
                "running the waypoint path debug publisher."
            )
        super().__init__("mii_amr_waypoint_path_debug_viz")
        self.args = args
        self.waypoints = []
        self.last_mtime_ns = None

        qos = follower.rviz_qos_profile()
        self.path_pub = self.create_publisher(follower.NavPath, args.path_topic, qos)
        self.marker_pub = self.create_publisher(
            follower.MarkerArray,
            args.waypoint_marker_topic,
            qos,
        )

        self.reload_waypoints(force=True)
        self.publish_route()
        self.timer = self.create_timer(args.watch_period_sec, self.timer_callback)
        self.get_logger().info(
            "Publishing waypoint path debug visualization: "
            f"path={args.path_topic}, waypoints={args.waypoint_marker_topic}, "
            f"source={args.waypoints}"
        )

    def reload_waypoints(self, force=False):
        mtime_ns = waypoint_file_mtime_ns(self.args.waypoints)
        if not force and self.last_mtime_ns == mtime_ns:
            return False
        self.waypoints = load_display_waypoints(self.args)
        self.last_mtime_ns = mtime_ns
        return True

    def publish_route(self):
        stamp = self.get_clock().now().to_msg()
        path_msg, marker_msg = build_route_messages(self.args, self.waypoints, stamp)
        self.path_pub.publish(path_msg)
        self.marker_pub.publish(marker_msg)

    def timer_callback(self):
        if not self.args.watch_file:
            return
        try:
            changed = self.reload_waypoints(force=False)
        except Exception as exc:
            self.get_logger().warn(f"Could not reload waypoint CSV: {exc}")
            return
        if changed:
            self.get_logger().info(f"Reloaded waypoint CSV: {self.args.waypoints}")
            self.publish_route()


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Publish a waypoint CSV as RViz Path and MarkerArray topics.",
    )
    parser.add_argument("--waypoints", default=DEFAULT_WAYPOINTS_CSV, type=Path)
    parser.add_argument("--path-topic", default=DEFAULT_PATH_TOPIC)
    parser.add_argument("--waypoint-marker-topic", default=DEFAULT_WAYPOINT_MARKER_TOPIC)
    parser.add_argument("--map-frame", default=DEFAULT_MAP_FRAME)
    parser.add_argument(
        "--skip-first-waypoint",
        action="store_true",
        help="Mirror the two-stage follower handoff by hiding waypoint 0.",
    )
    parser.add_argument("--min-waypoint-spacing-m", default=0.0, type=float)
    parser.add_argument("--current-waypoint-index", default=0, type=int)
    parser.add_argument("--watch-period-sec", default=DEFAULT_WATCH_PERIOD_SEC, type=float)
    parser.add_argument(
        "--no-watch",
        dest="watch_file",
        action="store_false",
        help="Do not reload the waypoint CSV if it changes while the node is running.",
    )
    parser.set_defaults(watch_file=True)
    return parser


def validate_args(parser, args):
    if args.min_waypoint_spacing_m < 0.0:
        parser.error("--min-waypoint-spacing-m must be >= 0")
    if args.current_waypoint_index < 0:
        parser.error("--current-waypoint-index must be >= 0")
    if args.watch_period_sec <= 0.0:
        parser.error("--watch-period-sec must be > 0")


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    if rclpy is None:
        raise SystemExit("ROS2 Python packages are required to run this RViz publisher.")

    rclpy.init(args=None)
    node = WaypointPathDebugViz(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
