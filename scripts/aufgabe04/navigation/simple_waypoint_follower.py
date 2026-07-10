"""Minimal ROS2 waypoint follower for one Aufgabe 04 route segment."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

from scripts.aufgabe04.navigation.follower_models import FollowerResult
from scripts.aufgabe04.navigation.follower_safety import (
    cmd_vel_ownership_failure,
    message_freshness_failure,
    obstacle_failure,
    rotation_progress_failure,
    startup_readiness_failure,
    waypoint_timeout_failure,
)
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.ros_runtime_config import ResolvedRuntimeConfig
from scripts.aufgabe04.navigation.waypoint_controller import (
    ControllerConfig,
    compute_waypoint_command,
    forward_resume_target,
)

try:  # pragma: no cover - exercised on ROS hosts.
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.parameter import Parameter
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
    Parameter = None
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
    startup_timeout_sec: float = 3.0
    max_rotation_sec: float = 25.0
    max_rotation_no_progress_sec: float = 3.0
    min_heading_progress_rad: float = 0.03
    allowed_cmd_vel_publishers: tuple[str, ...] = ()


def _require_ros() -> None:
    if rclpy is None:
        raise RuntimeError("ROS2 Python packages are not available in this environment")


def _node_identity(endpoint) -> str:
    namespace = getattr(endpoint, "node_namespace", "") or ""
    name = getattr(endpoint, "node_name", "") or ""
    return _format_node_identity(namespace, name)


def _format_node_identity(namespace: str, name: str) -> str:
    if namespace in ("", "/"):
        return f"/{name}"
    return f"{namespace.rstrip('/')}/{name}"


def _frame_id(frame_id: str) -> str:
    return frame_id.strip("/")


def pose_from_odometry(msg, *, odom_frame: str, base_frame: str) -> Pose2D | None:
    """Return an odom-frame base pose only when message frames match exactly."""

    if msg is None:
        return None
    if _frame_id(msg.header.frame_id) != _frame_id(odom_frame):
        return None
    if _frame_id(msg.child_frame_id) != _frame_id(base_frame):
        return None
    pose = msg.pose.pose
    return Pose2D(pose.position.x, pose.position.y, _yaw_from_quaternion(pose.orientation))


def remaining_route_distance(
    pose: Pose2D | None,
    waypoints: Sequence[Pose2D],
    target_index: int,
) -> float:
    if pose is None or not waypoints:
        return 0.0
    index = min(max(target_index, 0), len(waypoints) - 1)
    remaining = math.hypot(
        waypoints[index].x_m - pose.x_m,
        waypoints[index].y_m - pose.y_m,
    )
    for start, end in zip(waypoints[index:], waypoints[index + 1 :]):
        remaining += math.hypot(end.x_m - start.x_m, end.y_m - start.y_m)
    return remaining


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
        super().__init__(
            "aufgabe04_simple_waypoint_follower",
            parameter_overrides=[
                Parameter(
                    "use_sim_time",
                    Parameter.Type.BOOL,
                    bool(runtime_config.use_sim_time),
                )
            ],
        )
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
        self.rotation_started_at = None
        self.last_heading_progress_at = None
        self.best_abs_heading_error_rad = None
        self.last_control_log_at = 0.0

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
        startup_failure = self._wait_for_initial_inputs()
        if startup_failure:
            self.publish_repeated_zero()
            return FollowerResult(
                "stopped",
                startup_failure,
                time.monotonic() - started_at,
                self.distance_estimate_m,
                self.motion_published,
            )
        initial_pose = self._current_pose()
        if initial_pose is None:
            self.publish_repeated_zero()
            return FollowerResult(
                "stopped", "map-to-base transform unavailable",
                time.monotonic() - started_at, self.distance_estimate_m,
                self.motion_published,
            )
        resume_index, route_proximity_m = forward_resume_target(initial_pose, self.waypoints)
        initial_failure = (
            "initial pose too far from route"
            if route_proximity_m > self.follower_config.initial_distance_limit_m
            else ""
        )
        if initial_failure:
            self.publish_repeated_zero()
            return FollowerResult(
                "stopped", initial_failure, time.monotonic() - started_at,
                self.distance_estimate_m, self.motion_published,
            )
        self.target_index = resume_index
        self.target_started_at = time.monotonic()
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
                step = compute_waypoint_command(
                    pose,
                    self.waypoints,
                    self.target_index,
                    self.follower_config.controller,
                )
                if step.target_index != self.target_index:
                    self.target_index = step.target_index
                    self.target_started_at = time.monotonic()
                    self._reset_rotation_progress()
                if step.reached_goal:
                    self.publish_repeated_zero()
                    return self._result("completed", "", started_at, pose)
                obstacle_failure_reason = self._obstacle_failure()
                if obstacle_failure_reason:
                    self.publish_repeated_zero()
                    return self._result(
                        "stopped",
                        obstacle_failure_reason,
                        started_at,
                        pose,
                    )
                timeout_failure = waypoint_timeout_failure(
                    time.monotonic() - self.target_started_at,
                    self.follower_config.waypoint_timeout_sec,
                )
                if timeout_failure:
                    self.publish_repeated_zero()
                    return self._result("stopped", timeout_failure, started_at, pose)
                rotation_failure = self._rotation_progress_failure(step)
                if rotation_failure:
                    self.publish_repeated_zero()
                    return self._result("stopped", rotation_failure, started_at, pose)
                self._log_control_sample(pose, step)
                twist = Twist()
                twist.linear.x = step.command.linear_x_mps
                twist.angular.z = step.command.angular_z_radps
                self.cmd_vel_pub.publish(twist)
                self.motion_published = self.motion_published or abs(twist.linear.x) > 0.0 or abs(twist.angular.z) > 0.0
                self._spin_control_period()
        finally:
            self.publish_repeated_zero()

    def _result(
        self,
        status: str,
        stop_reason: str,
        started_at: float,
        pose: Pose2D | None,
    ) -> FollowerResult:
        return FollowerResult(
            status,
            stop_reason,
            time.monotonic() - started_at,
            self.distance_estimate_m,
            self.motion_published,
            target_index=self.target_index,
            remaining_distance_m=remaining_route_distance(
                pose,
                self.waypoints,
                self.target_index,
            ),
            final_x_m=None if pose is None else pose.x_m,
            final_y_m=None if pose is None else pose.y_m,
            final_yaw_rad=None if pose is None else pose.yaw_rad,
        )

    def _spin_control_period(self) -> None:
        """Keep ROS callbacks moving while pacing the controller by wall time.

        ``rclpy.Rate.sleep()`` is unsafe here with simulated time: this node
        owns the only executor spin, so sleeping on the ROS clock prevents the
        /clock callback that would wake the rate.  A monotonic deadline keeps
        the safety loop bounded while continuing to service scan, odom, TF,
        and clock subscriptions.
        """

        period_sec = 1.0 / max(self.follower_config.control_rate_hz, 1e-6)
        deadline = time.monotonic() + period_sec
        while rclpy.ok():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            rclpy.spin_once(self, timeout_sec=min(0.02, remaining))

    def _reset_rotation_progress(self) -> None:
        self.rotation_started_at = None
        self.last_heading_progress_at = None
        self.best_abs_heading_error_rad = None

    def _rotation_progress_failure(self, step) -> str:
        rotating = (
            abs(step.command.linear_x_mps) <= 1e-9
            and abs(step.command.angular_z_radps) > 1e-9
        )
        if not rotating:
            self._reset_rotation_progress()
            return ""
        now = time.monotonic()
        abs_error = abs(step.heading_error_rad)
        if self.rotation_started_at is None:
            self.rotation_started_at = now
            self.last_heading_progress_at = now
            self.best_abs_heading_error_rad = abs_error
            return ""
        if (
            self.best_abs_heading_error_rad is None
            or abs_error
            <= self.best_abs_heading_error_rad - self.follower_config.min_heading_progress_rad
        ):
            self.best_abs_heading_error_rad = abs_error
            self.last_heading_progress_at = now
        return rotation_progress_failure(
            rotation_elapsed_sec=now - self.rotation_started_at,
            no_progress_elapsed_sec=now - (self.last_heading_progress_at or now),
            max_rotation_sec=self.follower_config.max_rotation_sec,
            max_no_progress_sec=self.follower_config.max_rotation_no_progress_sec,
        )

    def _log_control_sample(self, pose: Pose2D, step) -> None:
        now = time.monotonic()
        if now - self.last_control_log_at < 1.0:
            return
        self.last_control_log_at = now
        self.get_logger().info(
            "control "
            f"target={step.target_index} "
            f"pose=({pose.x_m:.3f},{pose.y_m:.3f},{pose.yaw_rad:.3f}) "
            f"target_heading={step.target_heading_rad:.3f} "
            f"heading_error={step.heading_error_rad:.3f} "
            f"cmd=({step.command.linear_x_mps:.3f},{step.command.angular_z_radps:.3f})"
        )

    def _wait_for_initial_inputs(self) -> str:
        deadline = time.monotonic() + self.follower_config.startup_timeout_sec
        scan_ready = False
        odom_ready = False
        pose_ready = False
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            scan_ready = not self._freshness_failure(
                "scan",
                self.latest_scan,
                self.latest_scan_receipt,
                self.follower_config.max_scan_age_sec,
            )
            odom_ready = not self._freshness_failure(
                "odom",
                self.latest_odom,
                self.latest_odom_receipt,
                self.follower_config.max_odom_age_sec,
            )
            pose_ready = self._current_pose() is not None
            self.publish_zero()
            if scan_ready and odom_ready and pose_ready:
                return ""
        return startup_readiness_failure(
            scan_ready=scan_ready,
            odom_ready=odom_ready,
            pose_ready=pose_ready,
        )

    def _safety_failure(self) -> str:
        scan_failure = self._freshness_failure("scan", self.latest_scan, self.latest_scan_receipt, self.follower_config.max_scan_age_sec)
        if scan_failure:
            return scan_failure
        odom_failure = self._freshness_failure("odom", self.latest_odom, self.latest_odom_receipt, self.follower_config.max_odom_age_sec)
        if odom_failure:
            return odom_failure
        ownership_failure = self._cmd_vel_ownership_failure()
        if ownership_failure:
            return ownership_failure
        return ""

    def _freshness_failure(self, name: str, msg, receipt, max_age_sec: float) -> str:
        if msg is None or receipt is None:
            return message_freshness_failure(
                name,
                has_message=False,
                receipt_age_sec=None,
                header_age_sec=None,
                max_age_sec=max_age_sec,
            )
        now = self.get_clock().now()
        receipt_age = (now - receipt).nanoseconds / 1_000_000_000.0
        header_age = (now - Time.from_msg(msg.header.stamp)).nanoseconds / 1_000_000_000.0
        return message_freshness_failure(
            name,
            has_message=True,
            receipt_age_sec=receipt_age,
            header_age_sec=header_age,
            max_age_sec=max_age_sec,
        )

    def _obstacle_failure(self) -> str:
        return obstacle_failure(
            getattr(self.latest_scan, "ranges", None),
            self.follower_config.min_obstacle_distance_m,
        )

    def _cmd_vel_ownership_failure(self) -> str:
        publishers = self.get_publishers_info_by_topic(self.runtime_config.cmd_vel_topic)
        publisher_identities = sorted({_node_identity(publisher) for publisher in publishers})
        self_identity = _format_node_identity(self.get_namespace(), self.get_name())
        return cmd_vel_ownership_failure(
            publisher_identities,
            self_identity,
            self.follower_config.allowed_cmd_vel_publishers,
        )

    def _current_pose(self) -> Pose2D | None:
        if _frame_id(self.runtime_config.map_frame) == _frame_id(
            self.runtime_config.odom_frame
        ):
            odom_pose = pose_from_odometry(
                self.latest_odom,
                odom_frame=self.runtime_config.odom_frame,
                base_frame=self.runtime_config.base_frame,
            )
            if odom_pose is not None:
                return odom_pose
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
