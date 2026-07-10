"""Minimal ROS2 waypoint follower for one Aufgabe 04 route segment."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

from scripts.aufgabe04.navigation.follower_models import FollowerResult
from scripts.aufgabe04.navigation.follower_safety import (
    NO_VALID_FRONT_SECTOR_SCAN_RANGES,
    OBSTACLE_TOO_CLOSE,
    cmd_vel_ownership_failure,
    front_sector_decision,
    initial_pose_failure,
    linear_scale_for_front_clearance,
    message_freshness_failure,
    obstacle_decision,
    stuck_progress_failure,
    waypoint_timeout_failure,
)
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
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
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
    qos_profile_sensor_data = None
    Time = None
    Buffer = None
    TransformException = Exception
    TransformListener = None


@dataclass(frozen=True)
class FollowerConfig:
    controller: ControllerConfig
    min_obstacle_distance_m: float = 0.20
    front_obstacle_slow_distance_m: float = 0.38
    front_obstacle_sector_rad: float = math.radians(35.0)
    max_scan_age_sec: float = 1.0
    max_odom_age_sec: float = 1.0
    max_tf_age_sec: float = 1.0
    initial_sensor_wait_sec: float = 2.0
    waypoint_timeout_sec: float = 45.0
    stuck_timeout_sec: float = 8.0
    stuck_progress_epsilon_m: float = 0.03
    initial_distance_limit_m: float = 0.35
    control_rate_hz: float = 10.0
    allowed_cmd_vel_publishers: Sequence[str] = ()


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


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

@dataclass(frozen=True)
class PoseLookupResult:
    pose: Pose2D | None
    details: dict[str, object] | None = None


def tf_lookup_failure_details(
    *,
    reason: str,
    target_frame: str,
    source_frame: str,
    max_age_sec: float,
    age_sec: float | None = None,
    exception: BaseException | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "stop_reason": "map-to-base transform unavailable",
        "source": "tf_lookup",
        "reason": reason,
        "target_frame": target_frame,
        "source_frame": source_frame,
        "max_age_sec": max_age_sec,
    }
    if age_sec is not None:
        payload["age_sec"] = age_sec
    if exception is not None:
        payload["exception_type"] = exception.__class__.__name__
        payload["exception"] = str(exception)
    return payload


def stuck_progress_details(
    *,
    target_index: int,
    distance_to_target_m: float,
    last_progress_distance_m: float,
    elapsed_without_progress_sec: float,
    max_without_progress_sec: float,
    progress_epsilon_m: float,
    commanded_linear_x_mps: float,
    commanded_angular_z_radps: float,
    front_clearance_scale: float,
    effective_linear_x_mps: float,
    front_clearance_details: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "stop_reason": "stuck no progress",
        "source": "progress_monitor",
        "target_index": target_index,
        "distance_to_target_m": distance_to_target_m,
        "last_progress_distance_m": last_progress_distance_m,
        "elapsed_without_progress_sec": elapsed_without_progress_sec,
        "max_without_progress_sec": max_without_progress_sec,
        "progress_epsilon_m": progress_epsilon_m,
        "commanded_linear_x_mps": commanded_linear_x_mps,
        "commanded_angular_z_radps": commanded_angular_z_radps,
        "front_clearance_scale": front_clearance_scale,
        "effective_linear_x_mps": effective_linear_x_mps,
    }
    if front_clearance_details is not None:
        payload["front_clearance"] = front_clearance_details
    return payload


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
        self.last_progress_distance_m = math.inf
        self.last_progress_at = time.monotonic()
        self.target_started_at = time.monotonic()
        self.latest_stop_details = None
        self.latest_front_clearance_details = None
        self._configure_sim_time(runtime_config.use_sim_time)

        self.cmd_vel_pub = self.create_publisher(Twist, runtime_config.cmd_vel_topic, 10)
        self.create_subscription(
            LaserScan,
            runtime_config.scan_topic,
            self._scan_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(Odometry, runtime_config.odom_topic, self._odom_callback, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _configure_sim_time(self, use_sim_time: bool) -> None:
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", use_sim_time)
            return
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, use_sim_time)])

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
        startup_failure = self._wait_for_initial_runtime_inputs(started_at)
        if startup_failure:
            self.publish_repeated_zero()
            return FollowerResult(
                "stopped",
                startup_failure,
                time.monotonic() - started_at,
                self.distance_estimate_m,
                self.motion_published,
            )
        loop_sleep_sec = 1.0 / max(self.follower_config.control_rate_hz, 1.0)
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
                        self.latest_stop_details,
                    )
                pose_lookup = self._current_pose_lookup()
                pose = pose_lookup.pose
                if pose is None:
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "stopped",
                        "map-to-base transform unavailable",
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        pose_lookup.details,
                    )
                if self.last_pose is not None:
                    self.distance_estimate_m += math.hypot(
                        pose.x_m - self.last_pose.x_m,
                        pose.y_m - self.last_pose.y_m,
                    )
                self.last_pose = pose
                if self.target_index == 0:
                    initial_failure = initial_pose_failure(
                        pose,
                        self.waypoints[0],
                        self.follower_config.initial_distance_limit_m,
                    )
                    if initial_failure:
                        self.publish_repeated_zero()
                        return FollowerResult(
                            "stopped",
                            initial_failure,
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
                    self.last_progress_distance_m = math.inf
                    self.last_progress_at = time.monotonic()
                if step.reached_goal:
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "completed",
                        "",
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                    )
                timeout_failure = waypoint_timeout_failure(
                    time.monotonic() - self.target_started_at,
                    self.follower_config.waypoint_timeout_sec,
                )
                if timeout_failure:
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "stopped",
                        timeout_failure,
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                    )
                now_monotonic = time.monotonic()
                front_clearance_scale = self._front_clearance_linear_scale()
                effective_linear_x_mps = step.command.linear_x_mps * front_clearance_scale
                progress_failure = self._progress_failure(
                    step.distance_to_target_m,
                    now_monotonic,
                    abs(step.command.linear_x_mps) > 0.0,
                )
                if progress_failure:
                    self.latest_stop_details = stuck_progress_details(
                        target_index=self.target_index,
                        distance_to_target_m=step.distance_to_target_m,
                        last_progress_distance_m=self.last_progress_distance_m,
                        elapsed_without_progress_sec=now_monotonic - self.last_progress_at,
                        max_without_progress_sec=self.follower_config.stuck_timeout_sec,
                        progress_epsilon_m=self.follower_config.stuck_progress_epsilon_m,
                        commanded_linear_x_mps=step.command.linear_x_mps,
                        commanded_angular_z_radps=step.command.angular_z_radps,
                        front_clearance_scale=front_clearance_scale,
                        effective_linear_x_mps=effective_linear_x_mps,
                        front_clearance_details=self.latest_front_clearance_details,
                    )
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "stopped",
                        progress_failure,
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        self.latest_stop_details,
                    )
                twist = Twist()
                twist.linear.x = effective_linear_x_mps
                twist.angular.z = step.command.angular_z_radps
                self.cmd_vel_pub.publish(twist)
                self.motion_published = self.motion_published or abs(twist.linear.x) > 0.0 or abs(twist.angular.z) > 0.0
                time.sleep(loop_sleep_sec)
        finally:
            self.publish_repeated_zero()

    def _wait_for_initial_runtime_inputs(self, started_at: float) -> str:
        deadline = started_at + self.follower_config.initial_sensor_wait_sec
        last_failure = "missing scan"
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            scan_failure = self._freshness_failure(
                "scan",
                self.latest_scan,
                self.latest_scan_receipt,
                self.follower_config.max_scan_age_sec,
            )
            if scan_failure:
                last_failure = scan_failure
            else:
                odom_failure = self._freshness_failure(
                    "odom",
                    self.latest_odom,
                    self.latest_odom_receipt,
                    self.follower_config.max_odom_age_sec,
                )
                if odom_failure:
                    last_failure = odom_failure
                else:
                    pose_lookup = self._current_pose_lookup()
                    if pose_lookup.pose is None:
                        self.latest_stop_details = pose_lookup.details
                        last_failure = "map-to-base transform unavailable"
                    else:
                        return ""
            if time.monotonic() >= deadline:
                return last_failure
            self.publish_zero()
        return "ROS shutdown"

    def _safety_failure(self) -> str:
        self.latest_stop_details = None
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

    def _scan_range_min(self) -> float | None:
        return float(getattr(self.latest_scan, "range_min")) if hasattr(self.latest_scan, "range_min") else None

    def _scan_range_max(self) -> float | None:
        return float(getattr(self.latest_scan, "range_max")) if hasattr(self.latest_scan, "range_max") else None

    def _obstacle_failure(self) -> str:
        decision = obstacle_decision(
            getattr(self.latest_scan, "ranges", None),
            self.follower_config.min_obstacle_distance_m,
            range_min_m=self._scan_range_min(),
            range_max_m=self._scan_range_max(),
        )
        if decision.stop_reason:
            self.latest_stop_details = decision.to_log_dict()
        return decision.stop_reason

    def _front_clearance_linear_scale(self) -> float:
        if self.latest_scan is None:
            return 1.0
        decision = front_sector_decision(
            getattr(self.latest_scan, "ranges", None),
            float(getattr(self.latest_scan, "angle_min", 0.0)),
            float(getattr(self.latest_scan, "angle_increment", 0.0)),
            0.0,
            self.follower_config.front_obstacle_sector_rad,
            self.follower_config.min_obstacle_distance_m,
            range_min_m=self._scan_range_min(),
            range_max_m=self._scan_range_max(),
        )
        self.latest_front_clearance_details = decision.to_log_dict()
        if decision.stop_reason in (NO_VALID_FRONT_SECTOR_SCAN_RANGES, OBSTACLE_TOO_CLOSE):
            return 0.0
        return linear_scale_for_front_clearance(
            decision.nearest_valid_range_m,
            self.follower_config.min_obstacle_distance_m,
            self.follower_config.front_obstacle_slow_distance_m,
        )

    def _progress_failure(
        self,
        distance_to_target_m: float,
        now_monotonic: float,
        forward_motion_commanded: bool,
    ) -> str:
        if distance_to_target_m + self.follower_config.stuck_progress_epsilon_m < self.last_progress_distance_m:
            self.last_progress_distance_m = distance_to_target_m
            self.last_progress_at = now_monotonic
            return ""
        return stuck_progress_failure(
            now_monotonic - self.last_progress_at,
            self.follower_config.stuck_timeout_sec,
            forward_motion_commanded,
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

    def _current_pose_lookup(self) -> PoseLookupResult:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.runtime_config.map_frame,
                self.runtime_config.base_frame,
                Time(),
                timeout=Duration(seconds=0.1),
            )
        except TransformException as exc:
            return PoseLookupResult(
                None,
                tf_lookup_failure_details(
                    reason="lookup_exception",
                    target_frame=self.runtime_config.map_frame,
                    source_frame=self.runtime_config.base_frame,
                    max_age_sec=self.follower_config.max_tf_age_sec,
                    exception=exc,
                ),
            )
        age = (self.get_clock().now() - Time.from_msg(transform.header.stamp)).nanoseconds / 1_000_000_000.0
        if age > self.follower_config.max_tf_age_sec:
            return PoseLookupResult(
                None,
                tf_lookup_failure_details(
                    reason="stale_transform",
                    target_frame=self.runtime_config.map_frame,
                    source_frame=self.runtime_config.base_frame,
                    max_age_sec=self.follower_config.max_tf_age_sec,
                    age_sec=age,
                ),
            )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return PoseLookupResult(Pose2D(translation.x, translation.y, _yaw_from_quaternion(rotation)))

    def _current_pose(self) -> Pose2D | None:
        return self._current_pose_lookup().pose


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
