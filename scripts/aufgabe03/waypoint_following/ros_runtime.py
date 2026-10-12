"""ROS subscriptions and the single safety gate for every nonzero command."""

import math
import time

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
    from sensor_msgs.msg import LaserScan
    import tf2_ros
except ImportError:
    rclpy = None
    Node = object

import replan_runtime
from .math_utils import quaternion_to_yaw_deg
from .models import AmclHealth, Pose2D
from .scan_safety import evaluate_scan_safety


class HealthError(RuntimeError):
    """A stopped retry may recover missing or stale sensor/localization data."""


class BlockedError(RuntimeError):
    def __init__(self, safety, mode):
        super().__init__(f"{mode} scan blocked: {safety.reason}")
        self.safety = safety
        self.mode = mode


def evaluate_amcl_health(covariance, age_sec, max_age_sec, max_var_x, max_var_y,
                         max_var_yaw, fail_on_bad_localization=True):
    """Localization failures always close the gate; warn-only behavior is retired."""
    warnings = []
    if age_sec is None or not math.isfinite(age_sec) or not 0 <= age_sec <= max_age_sec:
        warnings.append("stale_amcl")
    if len(covariance) != 36:
        return AmclHealth(False, [*warnings, "invalid_covariance"], None, None, None, age_sec)
    values = tuple(float(covariance[index]) for index in (0, 7, 35))
    for name, value, limit in zip(("x", "y", "yaw"), values, (max_var_x, max_var_y, max_var_yaw)):
        if not math.isfinite(value) or value < 0 or value > limit:
            warnings.append("invalid_cov_" + name)
    return AmclHealth(not warnings, warnings, *values, age_sec)


def fresh_age(now, stamp, limit, label, future_slack=0.25):
    age = None if stamp is None else now - stamp
    if stamp is None or stamp <= 0 or not math.isfinite(age) or not -future_slack <= age <= limit:
        raise HealthError(f"{label} missing or stale: age={age}, limit={limit}")
    return max(0.0, age)


class WaypointFollower(Node):
    def __init__(self, args):
        if rclpy is None:
            raise RuntimeError("ROS 2 Python dependencies unavailable; source the ROS environment")
        super().__init__("follow_planned_waypoints")
        self.args = args
        self.pub = self.create_publisher(Twist, args.cmd_vel_topic, 10)
        self.last_scan = None
        self.last_scan_received_sec = None
        self.last_amcl = None
        self.last_amcl_received_sec = None
        self.base_frame_used = args.base_frame
        self.run_local_map = None
        self.replan_count = 0

    def connect(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.scan_subscription = self.create_subscription(
            LaserScan, self.args.scan_topic, self.scan_callback, qos_profile_sensor_data,
        )
        self.amcl_subscription = self.create_subscription(
            PoseWithCovarianceStamped, self.args.amcl_topic, self.amcl_callback, 10,
        )

    def now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    def is_running(self):
        return rclpy.ok()

    def spin_once(self, timeout_sec):
        rclpy.spin_once(self, timeout_sec=timeout_sec)

    def scan_callback(self, message):
        self.last_scan = message
        self.last_scan_received_sec = time.time()

    def amcl_callback(self, message):
        self.last_amcl = message
        self.last_amcl_received_sec = time.time()

    def _publish(self, linear_x, angular_z):
        message = Twist()
        message.linear.x = float(linear_x)
        message.angular.z = float(angular_z)
        self.pub.publish(message)

    def publish_velocity(self, linear_x, angular_z):
        if linear_x != 0.0 or angular_z != 0.0:
            try:
                if (not math.isfinite(linear_x) or not math.isfinite(angular_z)
                        or not 0 <= linear_x <= self.args.linear_speed
                        or abs(angular_z) > self.args.max_angular_speed):
                    raise RuntimeError("invalid or out-of-bounds motion command")
                self.check_health_or_raise()
                mode = "forward" if linear_x > 0.0 else "rotate"
                safety = self.evaluate_current_scan_safety(mode)
                if not safety.safe:
                    raise BlockedError(safety, mode)
            except Exception:
                self._publish(0.0, 0.0)
                raise
        self._publish(linear_x, angular_z)

    def stop_repeatedly(self):
        # Keep attempting zeros even after ROS shutdown or a publish failure.
        for _ in range(10):
            try:
                self._publish(0.0, 0.0)
            except Exception:
                pass
            time.sleep(0.05)

    def lookup_pose(self):
        errors = []
        for frame in dict.fromkeys((self.args.base_frame, self.args.fallback_base_frame)):
            if not frame:
                continue
            try:
                transform = self.tf_buffer.lookup_transform(self.args.map_frame, frame, Time())
                position = transform.transform.translation
                q = transform.transform.rotation
                values = (position.x, position.y, q.x, q.y, q.z, q.w)
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("non-finite pose")
                if abs(sum(value * value for value in (q.x, q.y, q.z, q.w)) - 1.0) > 0.1:
                    raise ValueError("invalid pose quaternion")
                pose = Pose2D(float(position.x), float(position.y),
                              quaternion_to_yaw_deg(q.x, q.y, q.z, q.w),
                              replan_runtime.stamp_to_sec(transform.header.stamp), frame)
                fresh_age(self.now_sec(), pose.stamp_sec, self.args.max_pose_age_sec, "TF pose")
                self.base_frame_used = frame
                return pose
            except Exception as exc:
                errors.append(f"{frame}: {exc}")
        raise HealthError("TF unavailable: " + "; ".join(errors))

    def check_health_or_raise(self):
        pose = self.lookup_pose()
        now = self.now_sec()
        wall_now = time.time()
        for label, message, received, limit in (
            ("scan", self.last_scan, self.last_scan_received_sec, self.args.max_scan_age_sec),
            ("AMCL", self.last_amcl, self.last_amcl_received_sec, self.args.max_amcl_age_sec),
        ):
            if message is None:
                raise HealthError(f"missing {label}")
            fresh_age(wall_now, received, limit, label + " receipt", future_slack=0.0)
            fresh_age(now, replan_runtime.stamp_to_sec(message.header.stamp), limit, label + " timestamp")
        if self.last_amcl.header.frame_id.lstrip("/") != self.args.map_frame.lstrip("/"):
            raise HealthError("AMCL pose is not in the map frame")
        age = max(wall_now - self.last_amcl_received_sec,
                  now - replan_runtime.stamp_to_sec(self.last_amcl.header.stamp), 0.0)
        health = evaluate_amcl_health(self.last_amcl.pose.covariance, age,
                                     self.args.max_amcl_age_sec, self.args.max_amcl_var_x,
                                     self.args.max_amcl_var_y, self.args.max_amcl_var_yaw)
        if not health.ok:
            raise HealthError("AMCL unhealthy: " + ", ".join(health.warnings))
        scan = self.last_scan
        metadata = (scan.angle_min, scan.angle_increment, scan.range_min, scan.range_max)
        if (not scan.header.frame_id or not all(math.isfinite(value) for value in metadata)
                or scan.angle_increment == 0 or not 0 <= scan.range_min < scan.range_max):
            raise HealthError("invalid scan frame or geometry")
        self.scan_base_transform()
        return pose

    def wait_for_health(self, timeout_sec):
        deadline = time.monotonic() + timeout_sec
        error = "no sensor data"
        while self.is_running() and time.monotonic() < deadline:
            self.publish_velocity(0.0, 0.0)
            self.spin_once(0.05)
            try:
                return self.check_health_or_raise()
            except HealthError as exc:
                error = str(exc)
        raise RuntimeError(f"Health gate timed out while stopped: {error}")

    def scan_base_transform(self):
        scan = self.last_scan
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame_used, scan.header.frame_id,
                replan_runtime.time_from_stamp(scan.header.stamp),
            )
        except Exception as exc:
            raise HealthError(f"timestamped laser-to-base TF unavailable: {exc}") from exc
        try:
            replan_runtime.transform_point_2d(transform, 0.0, 0.0)
        except RuntimeError as exc:
            raise HealthError(str(exc)) from exc
        return transform

    def evaluate_current_scan_safety(self, mode):
        scan = self.last_scan
        transform = self.scan_base_transform()
        all_ranges, forward_ranges = [], []
        for point in replan_runtime.scan_frame_points(scan):
            x, y = replan_runtime.transform_point_2d(transform, point.x, point.y)
            distance = math.hypot(x, y)
            all_ranges.append(distance)
            if abs(math.degrees(math.atan2(y, x))) <= self.args.scan_half_angle_deg:
                forward_ranges.append(distance)
        # Hard clearance applies around the whole footprint, including forward
        # commands with a small angular correction.
        ranges = forward_ranges if mode == "forward" else all_ranges
        if all_ranges and min(all_ranges) < self.args.hard_stop_range_m:
            ranges = [min(all_ranges)]
        return evaluate_scan_safety(
            ranges, 0.0, 0.0, 0.0, math.inf, mode, self.args.scan_half_angle_deg,
            self.args.hard_stop_range_m, self.args.min_scan_range_m,
            self.args.rotation_stop_range_m,
        )

    def follow(self, waypoints):
        from .follow_loop import follow_route
        # Includes subscription/TF setup and startup, not only the control loop.
        try:
            self.connect()
            return follow_route(self, waypoints)
        finally:
            self.stop_repeatedly()
