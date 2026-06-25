"""Minimal ROS2 waypoint follower for one Aufgabe 04 route segment."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence, Tuple

from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.ros_runtime_config import ResolvedRuntimeConfig
from scripts.aufgabe04.navigation.waypoint_controller import (
    ControllerConfig,
    compute_waypoint_command,
)

try:  # pragma: no cover - exercised on ROS hosts.
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer, TransformException, TransformListener
except ImportError:  # pragma: no cover - keeps offline tests ROS-free.
    rclpy = None
    Twist = None
    LaserScan = None
    Odometry = None
    Duration = None
    Node = object
    Time = None
    Buffer = None
    TransformException = Exception
    TransformListener = None


@dataclass(frozen=True)
class FollowerConfig:
    controller: ControllerConfig
    min_obstacle_distance_m: float = 0.20
    max_scan_age_sec: float = 1.0
    max_odom_age_sec: float = 1.0
    max_tf_age_sec: float = 1.0
    waypoint_timeout_sec: float = 45.0
    initial_distance_limit_m: float = 0.35
    control_rate_hz: float = 10.0
    allow_idle_nav2: bool = True
    allowed_cmd_vel_publishers: Tuple[str, ...] = ()


@dataclass(frozen=True)
class FollowerResult:
    status: str
    stop_reason: str
    duration_sec: float
    distance_estimate_m: float
    motion_published: bool


def _require_ros() -> None:
    if rclpy is None:
        raise RuntimeError("ROS2 Python packages are not available in this environment")


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class SimpleWaypointFollowerNode(Node):  # pragma: no cover - requires ROS runtime.
    def __init__(
        self,
        runtime_config: ResolvedRuntimeConfig,
        waypoints: Sequence[Pose2D],
        follower_config: FollowerConfig,
    ) -> None:
        super().__init__("aufgabe04_simple_waypoint_follower")
        self.runtime_config = runtime_config
        self.waypoints = tuple(waypoints)
        self.follower_config = follower_config
        self.target_index = 0
        self.latest_scan = None
        self.latest_scan_receipt = None
        self.latest_odom = None
        self.latest_odom_receipt = None
        self.motion_published = False
        self.distance_estimate_m = 0.0
        self.last_pose = None
        self.target_started_at = time.monotonic()

        self.cmd_vel_pub = self.create_publisher(Twist, runtime_config.cmd_vel_topic, 10)
        self.create_subscription(LaserScan, runtime_config.scan_topic, self._scan_callback, 10)
        self.create_subscription(Odometry, runtime_config.odom_topic, self._odom_callback, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _scan_callback(self, msg) -> None:
        self.latest_scan = msg
        self.latest_scan_receipt = self.get_clock().now()

    def _odom_callback(self, msg) -> None:
        self.latest_odom = msg
        self.latest_odom_receipt = self.get_clock().now()

    def publish_zero(self) -> None:
        self.cmd_vel_pub.publish(Twist())

    def publish_repeated_zero(self, count: int = 5) -> None:
        for _ in range(count):
            self.publish_zero()
            rclpy.spin_once(self, timeout_sec=0.02)

    def run(self) -> FollowerResult:
        if len(self.waypoints) < 2:
            return FollowerResult("noop", "fewer than two waypoints", 0.0, 0.0, False)
        started_at = time.monotonic()
        self.publish_repeated_zero()
        rate = self.create_rate(self.follower_config.control_rate_hz)
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.0)
                safety_failure = self._safety_failure()
                if safety_failure:
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "stopped",
                        safety_failure,
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                    )
                pose = self._current_pose()
                if pose is None:
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "stopped",
                        "map-to-base transform unavailable",
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                    )
                if self.last_pose is not None:
                    self.distance_estimate_m += math.hypot(
                        pose.x_m - self.last_pose.x_m,
                        pose.y_m - self.last_pose.y_m,
                    )
                self.last_pose = pose
                if self.target_index == 0:
                    first_distance = math.hypot(
                        pose.x_m - self.waypoints[0].x_m,
                        pose.y_m - self.waypoints[0].y_m,
                    )
                    if first_distance > self.follower_config.initial_distance_limit_m:
                        self.publish_repeated_zero()
                        return FollowerResult(
                            "stopped",
                            "initial pose too far from first waypoint",
                            time.monotonic() - started_at,
                            self.distance_estimate_m,
                            self.motion_published,
                        )
                step = compute_waypoint_command(
                    pose,
                    self.waypoints,
                    self.target_index,
                    self.follower_config.controller,
                )
                if step.target_index != self.target_index:
                    self.target_index = step.target_index
                    self.target_started_at = time.monotonic()
                if step.reached_goal:
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "completed",
                        "",
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                    )
                if time.monotonic() - self.target_started_at > self.follower_config.waypoint_timeout_sec:
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "stopped",
                        "waypoint timeout",
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                    )
                twist = Twist()
                twist.linear.x = step.command.linear_x_mps
                twist.angular.z = step.command.angular_z_radps
                self.cmd_vel_pub.publish(twist)
                self.motion_published = self.motion_published or abs(twist.linear.x) > 0.0 or abs(twist.angular.z) > 0.0
                rate.sleep()
        finally:
            self.publish_repeated_zero()

    def _safety_failure(self) -> str:
        scan_failure = self._freshness_failure("scan", self.latest_scan, self.latest_scan_receipt, self.follower_config.max_scan_age_sec)
        if scan_failure:
            return scan_failure
        odom_failure = self._freshness_failure("odom", self.latest_odom, self.latest_odom_receipt, self.follower_config.max_odom_age_sec)
        if odom_failure:
            return odom_failure
        obstacle_failure = self._obstacle_failure()
        if obstacle_failure:
            return obstacle_failure
        ownership_failure = self._cmd_vel_ownership_failure()
        if ownership_failure:
            return ownership_failure
        return ""

    def _freshness_failure(self, name: str, msg, receipt, max_age_sec: float) -> str:
        if msg is None or receipt is None:
            return f"missing {name}"
        now = self.get_clock().now()
        receipt_age = (now - receipt).nanoseconds / 1_000_000_000.0
        header_age = (now - Time.from_msg(msg.header.stamp)).nanoseconds / 1_000_000_000.0
        if receipt_age > max_age_sec or header_age > max_age_sec:
            return f"stale {name}"
        return ""

    def _obstacle_failure(self) -> str:
        ranges = getattr(self.latest_scan, "ranges", None)
        if not ranges:
            return ""
        finite_ranges = [value for value in ranges if math.isfinite(value) and value > 0.0]
        if finite_ranges and min(finite_ranges) < self.follower_config.min_obstacle_distance_m:
            return "obstacle too close"
        return ""

    def _cmd_vel_ownership_failure(self) -> str:
        publishers = self.get_publishers_info_by_topic(self.runtime_config.cmd_vel_topic)
        publisher_names = sorted({publisher.node_name for publisher in publishers})
        allowed = set(self.follower_config.allowed_cmd_vel_publishers)
        allowed.add(self.get_name())
        unknown = [
            name
            for name in publisher_names
            if name not in allowed
            and not (self.follower_config.allow_idle_nav2 and "nav" in name.lower())
        ]
        if unknown:
            return f"unapproved cmd_vel publisher during run: {', '.join(unknown)}"
        return ""

    def _current_pose(self) -> Pose2D | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.runtime_config.map_frame,
                self.runtime_config.base_frame,
                Time(),
                timeout=Duration(seconds=0.1),
            )
        except TransformException:
            return None
        age = (self.get_clock().now() - Time.from_msg(transform.header.stamp)).nanoseconds / 1_000_000_000.0
        if age > self.follower_config.max_tf_age_sec:
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return Pose2D(translation.x, translation.y, _yaw_from_quaternion(rotation))


def run_simple_waypoint_follower(
    runtime_config: ResolvedRuntimeConfig,
    waypoints: Sequence[Pose2D],
    follower_config: FollowerConfig,
) -> FollowerResult:
    _require_ros()
    rclpy.init(args=None)
    node = SimpleWaypointFollowerNode(runtime_config, waypoints, follower_config)
    try:
        return node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
