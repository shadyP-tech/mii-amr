#!/usr/bin/env python3
"""
Publish the arena-active temporary exploration map for RViz.

This node is read-only: it subscribes to /scan and /odom, rebuilds the same
odom-frame LocalGrid used by active-explore localization recovery, and publishes
it as a nav_msgs/OccupancyGrid. It does not publish /cmd_vel or interact with
Nav2.
"""

from __future__ import annotations

import argparse
import time

try:
    import rclpy
    from nav_msgs.msg import OccupancyGrid, Odometry
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan
except ImportError:
    rclpy = None
    OccupancyGrid = None
    Odometry = object
    Node = object
    DurabilityPolicy = None
    QoSProfile = None
    qos_profile_sensor_data = None
    LaserScan = object

from arena_active_explore import (
    ActiveExploreConfig,
    build_local_grid_from_scan_samples,
    grid_cell_counts,
)
from arena_active_spin import (
    odom_pose_from_msg,
    scan_sample_from_msg,
    temporary_map_occupancy_data,
)
from two_stage_waypoint.model import (
    DEFAULT_ARENA_ACTIVE_TEMPORARY_MAP_FRAME,
    DEFAULT_ARENA_ACTIVE_TEMPORARY_MAP_PUBLISH_PERIOD_SEC,
    DEFAULT_ARENA_ACTIVE_TEMPORARY_MAP_TOPIC,
)


DEFAULT_SCAN_TOPIC = "/scan"
DEFAULT_ODOM_TOPIC = "/odom"
DEFAULT_MAX_ODOM_SCAN_AGE_SEC = 0.20
DEFAULT_MAP_MAX_SAMPLES = 240
DEFAULT_STATUS_PERIOD_SEC = 2.0


def temporary_map_config_from_args(args):
    return ActiveExploreConfig(
        grid_resolution_m=args.grid_resolution_m,
        grid_size_m=args.grid_size_m,
        inflation_radius_m=args.inflation_radius_m,
    )


def trim_scan_samples(samples, max_samples):
    max_samples = max(1, int(max_samples))
    if len(samples) <= max_samples:
        return samples
    return samples[-max_samples:]


def build_debug_grid(scan_samples, latest_odom_pose, config):
    if not scan_samples:
        raise RuntimeError("No scan samples are available for the temporary map")
    if latest_odom_pose is None:
        raise RuntimeError("No odom pose is available for the temporary map")
    return build_local_grid_from_scan_samples(scan_samples, latest_odom_pose, config)


def rviz_occupancy_grid_qos_profile():
    if QoSProfile is None or DurabilityPolicy is None:
        return 1
    qos = QoSProfile(depth=1)
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    return qos


def build_occupancy_grid_message(grid, frame_id, stamp):
    if OccupancyGrid is None:
        raise RuntimeError(
            "nav_msgs.msg.OccupancyGrid is unavailable; source ROS 2 Humble first"
        )
    msg = OccupancyGrid()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.info.resolution = float(grid.resolution_m)
    msg.info.width = int(grid.width)
    msg.info.height = int(grid.height)
    msg.info.origin.position.x = float(grid.origin_x)
    msg.info.origin.position.y = float(grid.origin_y)
    msg.info.origin.position.z = 0.0
    msg.info.origin.orientation.x = 0.0
    msg.info.origin.orientation.y = 0.0
    msg.info.origin.orientation.z = 0.0
    msg.info.origin.orientation.w = 1.0
    msg.data = temporary_map_occupancy_data(grid)
    return msg


class ArenaActiveTemporaryMapDebugViz(Node):
    def __init__(self, args):
        if rclpy is None:
            raise RuntimeError("ROS2 Python packages are required to run this RViz publisher.")
        if OccupancyGrid is None:
            raise RuntimeError(
                "nav_msgs.msg.OccupancyGrid is unavailable. Source ROS 2 Humble before "
                "running the temporary map debug publisher."
            )
        super().__init__("mii_amr_arena_active_temporary_map_debug_viz")
        self.args = args
        self.config = temporary_map_config_from_args(args)
        self.latest_odom_pose = None
        self.latest_odom_received_sec = None
        self.scan_samples = []
        self.rejected_scan_count = 0
        self.last_status_sec = 0.0

        self.map_pub = self.create_publisher(
            OccupancyGrid,
            args.map_topic,
            rviz_occupancy_grid_qos_profile(),
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            args.scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            args.odom_topic,
            self.odom_callback,
            10,
        )
        self.timer = self.create_timer(args.publish_period_sec, self.timer_callback)
        self.get_logger().info(
            "Publishing arena-active temporary map debug visualization: "
            f"map={args.map_topic}, frame={args.map_frame}, "
            f"scan={args.scan_topic}, odom={args.odom_topic}"
        )

    def now_sec(self):
        return time.time()

    def odom_callback(self, msg):
        self.latest_odom_pose = odom_pose_from_msg(msg)
        self.latest_odom_received_sec = self.now_sec()

    def scan_callback(self, msg):
        if self.latest_odom_pose is None or self.latest_odom_received_sec is None:
            self.rejected_scan_count += 1
            return
        if self.now_sec() - self.latest_odom_received_sec > self.args.max_odom_scan_age_sec:
            self.rejected_scan_count += 1
            return
        self.scan_samples.append(scan_sample_from_msg(msg, self.latest_odom_pose))
        self.scan_samples = trim_scan_samples(
            self.scan_samples,
            self.args.map_max_samples,
        )

    def timer_callback(self):
        if not self.scan_samples or self.latest_odom_pose is None:
            self.log_status_if_due("waiting_for_scan_and_odom")
            return
        try:
            grid = build_debug_grid(
                self.scan_samples,
                self.latest_odom_pose,
                self.config,
            )
            msg = build_occupancy_grid_message(
                grid,
                self.args.map_frame,
                self.get_clock().now().to_msg(),
            )
        except Exception as exc:
            self.get_logger().warn(f"Could not build temporary map: {exc}")
            return
        self.map_pub.publish(msg)
        counts = grid_cell_counts(grid)
        self.log_status_if_due(
            "published "
            f"samples={len(self.scan_samples)} "
            f"free={counts['free']} "
            f"occupied={counts['occupied']} "
            f"inflated={counts['inflated']} "
            f"unknown={counts['unknown']} "
            f"rejected_scans={self.rejected_scan_count}"
        )

    def log_status_if_due(self, text):
        now = self.now_sec()
        if now - self.last_status_sec < self.args.status_period_sec:
            return
        self.last_status_sec = now
        self.get_logger().info(text)


def build_arg_parser():
    defaults = ActiveExploreConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Publish a read-only RViz OccupancyGrid for the arena-active "
            "temporary odom-frame map."
        ),
    )
    parser.add_argument("--scan-topic", default=DEFAULT_SCAN_TOPIC)
    parser.add_argument("--odom-topic", default=DEFAULT_ODOM_TOPIC)
    parser.add_argument("--map-topic", default=DEFAULT_ARENA_ACTIVE_TEMPORARY_MAP_TOPIC)
    parser.add_argument("--map-frame", default=DEFAULT_ARENA_ACTIVE_TEMPORARY_MAP_FRAME)
    parser.add_argument(
        "--publish-period-sec",
        default=DEFAULT_ARENA_ACTIVE_TEMPORARY_MAP_PUBLISH_PERIOD_SEC,
        type=float,
    )
    parser.add_argument("--status-period-sec", default=DEFAULT_STATUS_PERIOD_SEC, type=float)
    parser.add_argument(
        "--max-odom-scan-age-sec",
        default=DEFAULT_MAX_ODOM_SCAN_AGE_SEC,
        type=float,
    )
    parser.add_argument("--map-max-samples", default=DEFAULT_MAP_MAX_SAMPLES, type=int)
    parser.add_argument(
        "--grid-resolution-m",
        default=defaults.grid_resolution_m,
        type=float,
    )
    parser.add_argument("--grid-size-m", default=defaults.grid_size_m, type=float)
    parser.add_argument(
        "--inflation-radius-m",
        default=defaults.inflation_radius_m,
        type=float,
    )
    return parser


def validate_args(parser, args):
    for field in [
        "publish_period_sec",
        "status_period_sec",
        "max_odom_scan_age_sec",
        "grid_resolution_m",
        "grid_size_m",
        "inflation_radius_m",
    ]:
        if getattr(args, field) <= 0.0:
            parser.error(f"--{field.replace('_', '-')} must be greater than zero")
    if args.map_max_samples < 1:
        parser.error("--map-max-samples must be >= 1")
    for field in ["scan_topic", "odom_topic", "map_topic", "map_frame"]:
        if not getattr(args, field):
            parser.error(f"--{field.replace('_', '-')} must not be empty")


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    if rclpy is None:
        raise SystemExit("ROS2 Python packages are required to run this RViz publisher.")

    rclpy.init(args=None)
    node = ArenaActiveTemporaryMapDebugViz(args)
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
