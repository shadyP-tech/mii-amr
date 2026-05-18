#!/usr/bin/env python3
"""
Coordinate AMCL localization, Nav2 staging, and custom waypoint following.

This script intentionally keeps the existing waypoint follower separate. It
uses Nav2 only to reach the first waypoint as a staging pose, then hands off to
follow_planned_waypoints.py for the conservative path-following stage.
"""

import argparse
import csv
import math
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import rclpy
    from action_msgs.msg import GoalStatus
    from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    from std_srvs.srv import Empty
    import tf2_ros
except ImportError:
    rclpy = None
    GoalStatus = None
    NavigateToPose = None
    ActionClient = None
    Node = object
    qos_profile_sensor_data = None
    Time = None
    LaserScan = object
    Empty = None
    tf2_ros = None

    class _FallbackStamp:
        sec = 0
        nanosec = 0

    class _FallbackHeader:
        def __init__(self):
            self.frame_id = ""
            self.stamp = _FallbackStamp()

    class _FallbackPosition:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    class _FallbackOrientation:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
            self.w = 1.0

    class _FallbackPose:
        def __init__(self):
            self.position = _FallbackPosition()
            self.orientation = _FallbackOrientation()

    class _FallbackPoseWithCovariance:
        def __init__(self):
            self.pose = _FallbackPose()
            self.covariance = [0.0] * 36

    class PoseWithCovarianceStamped:
        def __init__(self):
            self.header = _FallbackHeader()
            self.pose = _FallbackPoseWithCovariance()

    class Twist:
        def __init__(self):
            self.linear = _FallbackPosition()
            self.angular = _FallbackPosition()


DEFAULT_WAYPOINTS_CSV = Path("results/aufgabe03/aufgabe03_waypoints.csv")
DEFAULT_RESULTS_CSV = Path("results/aufgabe03/aufgabe03_two_stage_runs.csv")
DEFAULT_FOLLOWER_SCRIPT = Path("scripts/aufgabe03/follow_planned_waypoints.py")

DEFAULT_LOCALIZATION_MODE = "global"
DEFAULT_LOCALIZATION_SPIN_DEG = 360.0
DEFAULT_LOCALIZATION_ANGULAR_SPEED_RADPS = 0.18
DEFAULT_AMCL_VALIDATION_TIMEOUT_SEC = 60.0
DEFAULT_KNOWN_START_VALIDATION_TIMEOUT_SEC = 30.0
DEFAULT_PREFLIGHT_TIMEOUT_SEC = 10.0
DEFAULT_NAV_TO_START_TIMEOUT_SEC = 180.0

DEFAULT_MAX_AMCL_AGE_SEC = 15.0
DEFAULT_MAX_AMCL_VAR_X = 0.05
DEFAULT_MAX_AMCL_VAR_Y = 0.05
DEFAULT_MAX_AMCL_VAR_YAW_RAD2 = 0.10
DEFAULT_STABLE_AMCL_SAMPLES = 5
DEFAULT_MAX_STABLE_POSE_JUMP_M = 0.05
DEFAULT_MAX_STABLE_YAW_JUMP_DEG = 10.0

DEFAULT_INITIAL_POSE_VAR_X = 0.05
DEFAULT_INITIAL_POSE_VAR_Y = 0.05
DEFAULT_INITIAL_POSE_VAR_YAW_RAD2 = 0.10

DEFAULT_SPIN_MIN_SCAN_RANGE_M = 0.18
DEFAULT_SPIN_MIN_VALID_SCAN_COUNT = 20
DEFAULT_MAX_SCAN_AGE_SEC = 8.0
DEFAULT_MAX_POSE_AGE_SEC = 10.0

DEFAULT_ARRIVAL_TOLERANCE_M = 0.15
DEFAULT_ARRIVAL_YAW_TOLERANCE_DEG = 45.0
DEFAULT_WAYPOINT_TOLERANCE_M = 0.12
DEFAULT_GOAL_TOLERANCE_M = 0.12
DEFAULT_MIN_WAYPOINT_SPACING_M = 0.12

DEFAULT_CONTROL_RATE_HZ = 10.0
STOP_PUBLISH_COUNT = 10
STOP_PUBLISH_HZ = 10.0

CSV_HEADER = [
    "timestamp",
    "start_wall_time",
    "end_wall_time",
    "duration_sec",
    "run_id",
    "waypoint_csv",
    "localization_mode",
    "status",
    "final_status_reason",
    "global_localization_service",
    "navigate_action",
    "initial_pose_topic",
    "amcl_topic",
    "cmd_vel_topic",
    "scan_topic",
    "map_frame",
    "selected_base_frame",
    "staging_x",
    "staging_y",
    "staging_yaw_deg",
    "localization_duration_sec",
    "nav2_duration_sec",
    "follower_duration_sec",
    "amcl_var_x",
    "amcl_var_y",
    "amcl_var_yaw_rad2",
    "stable_samples",
    "max_pose_jump_m",
    "max_yaw_jump_deg",
    "nav2_result_status",
    "tf_arrival_x",
    "tf_arrival_y",
    "tf_arrival_yaw_deg",
    "arrival_position_error_m",
    "arrival_yaw_error_deg",
    "follower_command",
    "follower_return_code",
    "final_tf_x",
    "final_tf_y",
    "final_tf_yaw_deg",
    "notes",
]

GOAL_STATUS_NAMES = {
    0: "UNKNOWN",
    1: "ACCEPTED",
    2: "EXECUTING",
    3: "CANCELING",
    4: "SUCCEEDED",
    5: "CANCELED",
    6: "ABORTED",
}


@dataclass(frozen=True)
class Waypoint:
    index: int
    x: float
    y: float


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw_deg: float
    stamp_sec: float | None = None
    frame_id: str = ""


@dataclass(frozen=True)
class StagingGoal:
    waypoint: Waypoint
    yaw_deg: float


@dataclass(frozen=True)
class ScanSafety:
    ok: bool
    reason: str
    valid_count: int
    min_range_m: float | None


@dataclass(frozen=True)
class PreflightRequirements:
    services: list[str]
    actions: list[str]
    topics: list[str]
    requires_tf_before_localization: bool


@dataclass(frozen=True)
class AmclCovariance:
    x: float
    y: float
    yaw_rad2: float


@dataclass(frozen=True)
class StabilityState:
    stable_count: int = 0
    previous_pose: Pose2D | None = None
    max_pose_jump_m: float = 0.0
    max_yaw_jump_deg: float = 0.0
    cov_x: float | None = None
    cov_y: float | None = None
    cov_yaw_rad2: float | None = None
    samples_seen: int = 0
    reason: str = "waiting_for_amcl"


@dataclass(frozen=True)
class ArrivalCheck:
    pose: Pose2D
    base_frame: str
    position_error_m: float
    yaw_error_deg: float


@dataclass
class RunDiagnostics:
    timestamp: str = ""
    start_wall_time: str = ""
    end_wall_time: str = ""
    duration_sec: float | None = None
    status: str = "failed"
    final_status_reason: str = ""
    selected_base_frame: str = ""
    localization_duration_sec: float | None = None
    nav2_duration_sec: float | None = None
    follower_duration_sec: float | None = None
    amcl_var_x: float | None = None
    amcl_var_y: float | None = None
    amcl_var_yaw_rad2: float | None = None
    stable_samples: int = 0
    max_pose_jump_m: float | None = None
    max_yaw_jump_deg: float | None = None
    nav2_result_status: str = ""
    tf_arrival_x: float | None = None
    tf_arrival_y: float | None = None
    tf_arrival_yaw_deg: float | None = None
    arrival_position_error_m: float | None = None
    arrival_yaw_error_deg: float | None = None
    follower_command: str = ""
    follower_return_code: int | None = None
    final_tf_x: float | None = None
    final_tf_y: float | None = None
    final_tf_yaw_deg: float | None = None
    notes: str = ""


def timestamp_now():
    return datetime.now().isoformat(timespec="seconds")


def empty_if_none(value):
    return "" if value is None else value


def shortest_angle_delta_deg(start_deg, end_deg):
    return (end_deg - start_deg + 180.0) % 360.0 - 180.0


def yaw_to_quaternion_values(yaw_deg):
    half = math.radians(yaw_deg) / 2.0
    return 0.0, 0.0, math.sin(half), math.cos(half)


def quaternion_to_yaw_deg(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def heading_between(a, b):
    return math.degrees(math.atan2(b.y - a.y, b.x - a.x))


def load_waypoints(path):
    path = Path(path)
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        required = {"index", "world_x_m", "world_y_m"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required column(s): {', '.join(sorted(missing))}")

        waypoints = []
        for row in reader:
            waypoint = Waypoint(
                index=int(row["index"]),
                x=float(row["world_x_m"]),
                y=float(row["world_y_m"]),
            )
            if waypoints and waypoint.x == waypoints[-1].x and waypoint.y == waypoints[-1].y:
                continue
            waypoints.append(waypoint)

    if len(waypoints) < 2:
        raise ValueError(f"{path} needs at least two waypoints for two-stage mode")
    return waypoints


def staging_goal_from_waypoints(waypoints):
    if len(waypoints) < 2:
        raise ValueError("two-stage mode needs at least two waypoints")
    return StagingGoal(waypoint=waypoints[0], yaw_deg=heading_between(waypoints[0], waypoints[1]))


def valid_scan_ranges(ranges, range_min=None, range_max=None):
    valid = []
    for value in ranges:
        if value is None or not math.isfinite(value):
            continue
        if range_min is not None and value < range_min:
            continue
        if range_max is not None and value > range_max:
            continue
        valid.append(float(value))
    return valid


def evaluate_spin_scan_safety(
    ranges,
    range_min=None,
    range_max=None,
    min_scan_range_m=DEFAULT_SPIN_MIN_SCAN_RANGE_M,
    min_valid_scan_count=DEFAULT_SPIN_MIN_VALID_SCAN_COUNT,
):
    valid = valid_scan_ranges(ranges, range_min=range_min, range_max=range_max)
    if len(valid) < min_valid_scan_count:
        return ScanSafety(False, "insufficient_valid_scan", len(valid), None)
    minimum = min(valid)
    if minimum < min_scan_range_m:
        return ScanSafety(False, "unsafe_proximity", len(valid), minimum)
    return ScanSafety(True, "ok", len(valid), minimum)


def amcl_covariances(covariance):
    if len(covariance) < 36:
        raise ValueError("AMCL covariance must have 36 entries")
    return AmclCovariance(
        x=float(covariance[0]),
        y=float(covariance[7]),
        yaw_rad2=float(covariance[35]),
    )


def pose_distance_m(a, b):
    return math.hypot(b.x - a.x, b.y - a.y)


def update_amcl_stability(
    state,
    pose,
    covariance,
    max_var_x,
    max_var_y,
    max_var_yaw_rad2,
    max_pose_jump_m,
    max_yaw_jump_deg,
):
    cov = amcl_covariances(covariance)
    samples_seen = state.samples_seen + 1
    if cov.x > max_var_x or cov.y > max_var_y or cov.yaw_rad2 > max_var_yaw_rad2:
        return StabilityState(
            stable_count=0,
            previous_pose=pose,
            max_pose_jump_m=state.max_pose_jump_m,
            max_yaw_jump_deg=state.max_yaw_jump_deg,
            cov_x=cov.x,
            cov_y=cov.y,
            cov_yaw_rad2=cov.yaw_rad2,
            samples_seen=samples_seen,
            reason="covariance_above_threshold",
        )

    pose_jump = 0.0
    yaw_jump = 0.0
    stable_count = state.stable_count + 1
    reason = "stable"
    if state.previous_pose is not None:
        pose_jump = pose_distance_m(state.previous_pose, pose)
        yaw_jump = abs(shortest_angle_delta_deg(state.previous_pose.yaw_deg, pose.yaw_deg))
        if pose_jump > max_pose_jump_m or yaw_jump > max_yaw_jump_deg:
            stable_count = 1
            reason = "pose_jump_above_threshold"

    return StabilityState(
        stable_count=stable_count,
        previous_pose=pose,
        max_pose_jump_m=max(state.max_pose_jump_m, pose_jump),
        max_yaw_jump_deg=max(state.max_yaw_jump_deg, yaw_jump),
        cov_x=cov.x,
        cov_y=cov.y,
        cov_yaw_rad2=cov.yaw_rad2,
        samples_seen=samples_seen,
        reason=reason,
    )


def amcl_validation_timed_out(start_sec, now_sec, timeout_sec):
    return now_sec - start_sec > timeout_sec


def build_initial_pose_message(
    x,
    y,
    yaw_deg,
    var_x,
    var_y,
    var_yaw_rad2,
    frame_id="map",
    stamp=None,
):
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    msg.pose.pose.position.x = float(x)
    msg.pose.pose.position.y = float(y)
    msg.pose.pose.position.z = 0.0
    qx, qy, qz, qw = yaw_to_quaternion_values(yaw_deg)
    msg.pose.pose.orientation.x = qx
    msg.pose.pose.orientation.y = qy
    msg.pose.pose.orientation.z = qz
    msg.pose.pose.orientation.w = qw
    covariance = [0.0] * 36
    covariance[0] = float(var_x)
    covariance[7] = float(var_y)
    covariance[35] = float(var_yaw_rad2)
    msg.pose.covariance = covariance
    return msg


def required_preflight_interfaces(args):
    services = []
    if args.localization_mode == "global":
        services.append(args.global_localization_service)
    return PreflightRequirements(
        services=services,
        actions=[args.navigate_action],
        topics=[args.scan_topic],
        requires_tf_before_localization=False,
    )


def build_follower_command(args):
    return [
        str(args.python_executable),
        str(args.follower_script),
        "--waypoints",
        str(args.waypoints),
        "--run-id",
        str(args.run_id),
        "--yes",
        "--start-selection",
        "path-progress",
        "--map-frame",
        args.map_frame,
        "--base-frame",
        args.base_frame,
        "--fallback-base-frame",
        args.fallback_base_frame,
        "--max-pose-age-sec",
        str(args.max_pose_age_sec),
        "--max-scan-age-sec",
        str(args.max_scan_age_sec),
        "--max-amcl-age-sec",
        str(args.max_amcl_age_sec),
        "--max-amcl-var-x",
        str(args.max_amcl_var_x),
        "--max-amcl-var-y",
        str(args.max_amcl_var_y),
        "--max-amcl-var-yaw",
        str(args.max_amcl_var_yaw_rad2),
        "--waypoint-tolerance-m",
        str(args.waypoint_tolerance_m),
        "--goal-tolerance-m",
        str(args.goal_tolerance_m),
        "--min-waypoint-spacing-m",
        str(args.min_waypoint_spacing_m),
        "--notes",
        f"{args.notes};two_stage_handoff",
    ]


def run_follower_command(command, runner=subprocess.run):
    return runner(command, check=False, shell=False)


def cleanup_motion(node):
    try:
        node.cancel_active_goal()
    finally:
        node.stop_repeatedly()


def goal_status_name(status):
    return GOAL_STATUS_NAMES.get(int(status), f"STATUS_{status}")


def pose2d_from_pose_msg(pose_msg, stamp_sec=None, frame_id=""):
    position = pose_msg.position
    orientation = pose_msg.orientation
    return Pose2D(
        x=float(position.x),
        y=float(position.y),
        yaw_deg=quaternion_to_yaw_deg(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ),
        stamp_sec=stamp_sec,
        frame_id=frame_id,
    )


def stamp_to_sec(stamp):
    if stamp is None:
        return None
    return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0


def transform_to_pose2d(transform, frame_id):
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    stamp_sec = stamp_to_sec(transform.header.stamp)
    return Pose2D(
        x=float(translation.x),
        y=float(translation.y),
        yaw_deg=quaternion_to_yaw_deg(rotation.x, rotation.y, rotation.z, rotation.w),
        stamp_sec=stamp_sec,
        frame_id=frame_id,
    )


def append_csv_row(path, header, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    if file_exists:
        with path.open(newline="") as file:
            existing_header = next(csv.reader(file), None)
        if existing_header != header:
            raise RuntimeError(f"{path} has an unrecognized schema. Move or migrate it first.")
    with path.open("a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def build_log_row(args, staging_goal, diagnostics):
    return [
        diagnostics.timestamp,
        diagnostics.start_wall_time,
        diagnostics.end_wall_time,
        empty_if_none(diagnostics.duration_sec),
        args.run_id,
        str(args.waypoints),
        args.localization_mode,
        diagnostics.status,
        diagnostics.final_status_reason,
        args.global_localization_service,
        args.navigate_action,
        args.initial_pose_topic,
        args.amcl_topic,
        args.cmd_vel_topic,
        args.scan_topic,
        args.map_frame,
        diagnostics.selected_base_frame,
        staging_goal.waypoint.x,
        staging_goal.waypoint.y,
        staging_goal.yaw_deg,
        empty_if_none(diagnostics.localization_duration_sec),
        empty_if_none(diagnostics.nav2_duration_sec),
        empty_if_none(diagnostics.follower_duration_sec),
        empty_if_none(diagnostics.amcl_var_x),
        empty_if_none(diagnostics.amcl_var_y),
        empty_if_none(diagnostics.amcl_var_yaw_rad2),
        diagnostics.stable_samples,
        empty_if_none(diagnostics.max_pose_jump_m),
        empty_if_none(diagnostics.max_yaw_jump_deg),
        diagnostics.nav2_result_status,
        empty_if_none(diagnostics.tf_arrival_x),
        empty_if_none(diagnostics.tf_arrival_y),
        empty_if_none(diagnostics.tf_arrival_yaw_deg),
        empty_if_none(diagnostics.arrival_position_error_m),
        empty_if_none(diagnostics.arrival_yaw_error_deg),
        diagnostics.follower_command,
        empty_if_none(diagnostics.follower_return_code),
        empty_if_none(diagnostics.final_tf_x),
        empty_if_none(diagnostics.final_tf_y),
        empty_if_none(diagnostics.final_tf_yaw_deg),
        diagnostics.notes,
    ]


class TwoStageCoordinator(Node):
    def __init__(self, args):
        if rclpy is None:
            raise RuntimeError("ROS 2 Python modules are unavailable. Source ROS 2 Humble first.")
        super().__init__("two_stage_waypoint_run")
        self.args = args
        self.last_scan = None
        self.last_scan_received_sec = None
        self.last_amcl = None
        self.last_amcl_received_sec = None
        self.active_goal_handle = None
        self.selected_base_frame = ""

        self.cmd_vel_pub = self.create_publisher(Twist, args.cmd_vel_topic, 10)
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            args.initial_pose_topic,
            10,
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            args.scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            args.amcl_topic,
            self.amcl_callback,
            10,
        )
        self.global_localization_client = self.create_client(
            Empty,
            args.global_localization_service,
        )
        self.navigate_client = ActionClient(self, NavigateToPose, args.navigate_action)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def scan_callback(self, msg):
        self.last_scan = msg
        self.last_scan_received_sec = time.time()

    def amcl_callback(self, msg):
        self.last_amcl = msg
        self.last_amcl_received_sec = time.time()

    def publish_stop(self):
        self.cmd_vel_pub.publish(Twist())

    def stop_repeatedly(self):
        delay = 1.0 / STOP_PUBLISH_HZ
        for _ in range(STOP_PUBLISH_COUNT):
            if rclpy.ok():
                self.publish_stop()
            time.sleep(delay)

    def wait_for_future(self, future, timeout_sec, description):
        deadline = time.time() + timeout_sec
        while rclpy.ok() and not future.done():
            if time.time() > deadline:
                raise RuntimeError(f"Timed out waiting for {description}")
            rclpy.spin_once(self, timeout_sec=0.1)
        if not future.done():
            raise RuntimeError(f"ROS shutdown while waiting for {description}")
        if future.exception() is not None:
            raise RuntimeError(f"{description} failed: {future.exception()}")
        return future.result()

    def wait_for_fresh_scan(self, timeout_sec):
        deadline = time.time() + timeout_sec
        while rclpy.ok() and time.time() <= deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.last_scan is None or self.last_scan_received_sec is None:
                continue
            if time.time() - self.last_scan_received_sec <= self.args.max_scan_age_sec:
                return self.last_scan
        raise RuntimeError(f"Timed out waiting for fresh {self.args.scan_topic}")

    def current_scan_safety(self):
        if self.last_scan is None:
            return ScanSafety(False, "missing_scan", 0, None)
        return evaluate_spin_scan_safety(
            self.last_scan.ranges,
            range_min=self.last_scan.range_min,
            range_max=self.last_scan.range_max,
            min_scan_range_m=self.args.spin_min_scan_range_m,
            min_valid_scan_count=self.args.spin_min_valid_scan_count,
        )

    def preflight_before_motion(self):
        requirements = required_preflight_interfaces(self.args)
        for service in requirements.services:
            if not self.global_localization_client.wait_for_service(
                timeout_sec=self.args.preflight_timeout_sec,
            ):
                raise RuntimeError(f"Required service is unavailable: {service}")
        if not self.navigate_client.wait_for_server(timeout_sec=self.args.preflight_timeout_sec):
            raise RuntimeError(f"Required action is unavailable: {self.args.navigate_action}")
        self.wait_for_fresh_scan(self.args.preflight_timeout_sec)
        safety = self.current_scan_safety()
        if not safety.ok:
            raise RuntimeError(f"Preflight scan safety failed: {safety.reason}")

    def call_global_localization(self):
        request = Empty.Request()
        future = self.global_localization_client.call_async(request)
        self.wait_for_future(
            future,
            self.args.preflight_timeout_sec,
            self.args.global_localization_service,
        )

    def perform_localization_spin(self):
        self.stop_repeatedly()
        angular_speed = abs(self.args.localization_angular_speed)
        direction = 1.0 if self.args.localization_spin_deg >= 0.0 else -1.0
        duration = math.radians(abs(self.args.localization_spin_deg)) / angular_speed
        period = 1.0 / self.args.control_rate_hz
        command = Twist()
        command.angular.z = direction * angular_speed
        start = time.time()
        while rclpy.ok() and time.time() - start < duration:
            rclpy.spin_once(self, timeout_sec=0.0)
            safety = self.current_scan_safety()
            if not safety.ok:
                raise RuntimeError(f"Localization spin scan safety failed: {safety.reason}")
            self.cmd_vel_pub.publish(command)
            time.sleep(period)
        self.stop_repeatedly()
        time.sleep(1.0)

    def publish_known_start_initial_pose(self):
        stamp = self.get_clock().now().to_msg()
        msg = build_initial_pose_message(
            self.args.initial_pose_x,
            self.args.initial_pose_y,
            self.args.initial_pose_yaw_deg,
            self.args.initial_pose_var_x,
            self.args.initial_pose_var_y,
            self.args.initial_pose_var_yaw_rad2,
            frame_id=self.args.map_frame,
            stamp=stamp,
        )
        for _ in range(3):
            self.initial_pose_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)

    def amcl_pose2d(self, msg):
        stamp_sec = stamp_to_sec(msg.header.stamp)
        return pose2d_from_pose_msg(msg.pose.pose, stamp_sec=stamp_sec, frame_id=msg.header.frame_id)

    def wait_for_amcl_validation(self, timeout_sec):
        start = time.time()
        state = StabilityState()
        processed_received_sec = None
        last_reason = state.reason
        while rclpy.ok():
            if amcl_validation_timed_out(start, time.time(), timeout_sec):
                raise RuntimeError(
                    "Timed out waiting for AMCL validation: "
                    f"reason={last_reason}, stable_samples={state.stable_count}"
                )
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.last_amcl is None or self.last_amcl_received_sec is None:
                continue
            if processed_received_sec == self.last_amcl_received_sec:
                continue
            processed_received_sec = self.last_amcl_received_sec
            age_sec = time.time() - self.last_amcl_received_sec
            if age_sec > self.args.max_amcl_age_sec:
                last_reason = "stale_amcl"
                continue
            pose = self.amcl_pose2d(self.last_amcl)
            state = update_amcl_stability(
                state,
                pose,
                self.last_amcl.pose.covariance,
                self.args.max_amcl_var_x,
                self.args.max_amcl_var_y,
                self.args.max_amcl_var_yaw_rad2,
                self.args.max_stable_pose_jump_m,
                self.args.max_stable_yaw_jump_deg,
            )
            last_reason = state.reason
            if state.stable_count >= self.args.stable_amcl_samples:
                return state
        raise RuntimeError("ROS shutdown while waiting for AMCL validation")

    def lookup_pose(self):
        errors = []
        for frame in [self.args.base_frame, self.args.fallback_base_frame]:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.args.map_frame,
                    frame,
                    Time(),
                )
                self.selected_base_frame = frame
                return transform_to_pose2d(transform, frame), frame
            except Exception as exc:
                errors.append(f"{frame}: {exc}")
        raise RuntimeError("Could not lookup TF pose: " + "; ".join(errors))

    def validate_post_localization_tf(self):
        pose, frame = self.lookup_pose()
        if pose.stamp_sec is not None and time.time() - pose.stamp_sec > self.args.max_pose_age_sec:
            raise RuntimeError(f"Post-localization TF is stale: frame={frame}")
        return pose, frame

    def navigate_to_staging(self, staging_goal):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = self.args.map_frame
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = staging_goal.waypoint.x
        goal_msg.pose.pose.position.y = staging_goal.waypoint.y
        goal_msg.pose.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quaternion_values(staging_goal.yaw_deg)
        goal_msg.pose.pose.orientation.x = qx
        goal_msg.pose.pose.orientation.y = qy
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        send_future = self.navigate_client.send_goal_async(goal_msg)
        goal_handle = self.wait_for_future(
            send_future,
            self.args.preflight_timeout_sec,
            "NavigateToPose goal acceptance",
        )
        if not goal_handle.accepted:
            raise RuntimeError("NavigateToPose goal was rejected")

        self.active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result = self.wait_for_future(
            result_future,
            self.args.nav_to_start_timeout_sec,
            "NavigateToPose result",
        )
        self.active_goal_handle = None
        status = int(result.status)
        status_name = goal_status_name(status)
        if status != 4:
            raise RuntimeError(f"NavigateToPose did not succeed: {status_name}")
        return status_name

    def cancel_active_goal(self):
        if self.active_goal_handle is None:
            return
        future = self.active_goal_handle.cancel_goal_async()
        try:
            self.wait_for_future(future, 2.0, "NavigateToPose cancellation")
        finally:
            self.active_goal_handle = None

    def verify_arrival(self, staging_goal):
        pose, frame = self.lookup_pose()
        position_error = math.hypot(
            pose.x - staging_goal.waypoint.x,
            pose.y - staging_goal.waypoint.y,
        )
        yaw_error = abs(shortest_angle_delta_deg(pose.yaw_deg, staging_goal.yaw_deg))
        if position_error > self.args.arrival_tolerance_m:
            raise RuntimeError(
                "Arrival position check failed: "
                f"error={position_error:.3f} m, "
                f"limit={self.args.arrival_tolerance_m:.3f} m"
            )
        if yaw_error > self.args.arrival_yaw_tolerance_deg:
            raise RuntimeError(
                "Arrival yaw check failed: "
                f"error={yaw_error:.1f} deg, "
                f"limit={self.args.arrival_yaw_tolerance_deg:.1f} deg"
            )
        return ArrivalCheck(pose, frame, position_error, yaw_error)


def require_motion_confirmation(args, staging_goal, follower_command):
    if args.yes:
        return True
    print("\nThis command may move the physical TurtleBot.")
    print("Safety requirements:")
    print("  - clear the arena and keep an operator near the robot")
    print("  - keep Ctrl+C and physical stop available")
    print("  - ensure no other controller is intentionally publishing /cmd_vel")
    print(f"Run ID: {args.run_id}")
    print(
        "Staging goal: "
        f"x={staging_goal.waypoint.x:.3f}, "
        f"y={staging_goal.waypoint.y:.3f}, "
        f"yaw={staging_goal.yaw_deg:.1f} deg"
    )
    print("Follower command:", shlex.join(follower_command))
    response = input("Type RUN to start two-stage waypoint run: ").strip()
    return response == "RUN"


def print_dry_run(args, waypoints, staging_goal, follower_command):
    print("Two-stage waypoint run dry run")
    print(f"Waypoint CSV: {args.waypoints}")
    print(f"Waypoints: {len(waypoints)}")
    print(
        "Selected waypoint 0: "
        f"x={staging_goal.waypoint.x:.3f}, y={staging_goal.waypoint.y:.3f}"
    )
    print(f"Computed staging yaw: {staging_goal.yaw_deg:.1f} deg")
    print(f"Localization mode: {args.localization_mode}")
    print("ROS interfaces:")
    print(f"  global localization service: {args.global_localization_service}")
    print(f"  navigate action: {args.navigate_action}")
    print(f"  initial pose topic: {args.initial_pose_topic}")
    print(f"  amcl topic: {args.amcl_topic}")
    print(f"  cmd_vel topic: {args.cmd_vel_topic}")
    print(f"  scan topic: {args.scan_topic}")
    print(f"Follower command: {shlex.join(follower_command)}")
    print(f"Log path: {args.results_csv}")
    print(f"ROS imports available: {'yes' if rclpy is not None else 'no'}")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Coordinate AMCL localization, Nav2 staging, and waypoint following.",
    )
    parser.add_argument("--waypoints", default=DEFAULT_WAYPOINTS_CSV, type=Path)
    parser.add_argument("--results-csv", default=DEFAULT_RESULTS_CSV, type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--notes", default="two_stage_waypoint_run")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--base-frame", default="base_footprint")
    parser.add_argument("--fallback-base-frame", default="base_link")

    parser.add_argument(
        "--localization-mode",
        default=DEFAULT_LOCALIZATION_MODE,
        choices=["global", "known-start"],
    )
    parser.add_argument("--localization-spin-deg", default=DEFAULT_LOCALIZATION_SPIN_DEG, type=float)
    parser.add_argument(
        "--localization-angular-speed",
        default=DEFAULT_LOCALIZATION_ANGULAR_SPEED_RADPS,
        type=float,
    )
    parser.add_argument(
        "--amcl-validation-timeout-sec",
        default=DEFAULT_AMCL_VALIDATION_TIMEOUT_SEC,
        type=float,
    )
    parser.add_argument(
        "--known-start-validation-timeout-sec",
        default=DEFAULT_KNOWN_START_VALIDATION_TIMEOUT_SEC,
        type=float,
    )
    parser.add_argument("--preflight-timeout-sec", default=DEFAULT_PREFLIGHT_TIMEOUT_SEC, type=float)
    parser.add_argument(
        "--nav-to-start-timeout-sec",
        default=DEFAULT_NAV_TO_START_TIMEOUT_SEC,
        type=float,
    )

    parser.add_argument("--initial-pose-x", type=float)
    parser.add_argument("--initial-pose-y", type=float)
    parser.add_argument("--initial-pose-yaw-deg", type=float)
    parser.add_argument("--initial-pose-var-x", default=DEFAULT_INITIAL_POSE_VAR_X, type=float)
    parser.add_argument("--initial-pose-var-y", default=DEFAULT_INITIAL_POSE_VAR_Y, type=float)
    parser.add_argument(
        "--initial-pose-var-yaw-rad2",
        default=DEFAULT_INITIAL_POSE_VAR_YAW_RAD2,
        type=float,
    )

    parser.add_argument("--global-localization-service", default="/reinitialize_global_localization")
    parser.add_argument("--navigate-action", default="/navigate_to_pose")
    parser.add_argument("--initial-pose-topic", default="/initialpose")
    parser.add_argument("--amcl-topic", default="/amcl_pose")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--follower-script", default=DEFAULT_FOLLOWER_SCRIPT, type=Path)
    parser.add_argument("--python-executable", default="python3")

    parser.add_argument("--max-pose-age-sec", default=DEFAULT_MAX_POSE_AGE_SEC, type=float)
    parser.add_argument("--max-scan-age-sec", default=DEFAULT_MAX_SCAN_AGE_SEC, type=float)
    parser.add_argument("--max-amcl-age-sec", default=DEFAULT_MAX_AMCL_AGE_SEC, type=float)
    parser.add_argument("--max-amcl-var-x", default=DEFAULT_MAX_AMCL_VAR_X, type=float)
    parser.add_argument("--max-amcl-var-y", default=DEFAULT_MAX_AMCL_VAR_Y, type=float)
    parser.add_argument(
        "--max-amcl-var-yaw-rad2",
        default=DEFAULT_MAX_AMCL_VAR_YAW_RAD2,
        type=float,
    )
    parser.add_argument("--stable-amcl-samples", default=DEFAULT_STABLE_AMCL_SAMPLES, type=int)
    parser.add_argument(
        "--max-stable-pose-jump-m",
        default=DEFAULT_MAX_STABLE_POSE_JUMP_M,
        type=float,
    )
    parser.add_argument(
        "--max-stable-yaw-jump-deg",
        default=DEFAULT_MAX_STABLE_YAW_JUMP_DEG,
        type=float,
    )
    parser.add_argument(
        "--spin-min-scan-range-m",
        default=DEFAULT_SPIN_MIN_SCAN_RANGE_M,
        type=float,
    )
    parser.add_argument(
        "--spin-min-valid-scan-count",
        default=DEFAULT_SPIN_MIN_VALID_SCAN_COUNT,
        type=int,
    )
    parser.add_argument("--arrival-tolerance-m", default=DEFAULT_ARRIVAL_TOLERANCE_M, type=float)
    parser.add_argument(
        "--arrival-yaw-tolerance-deg",
        default=DEFAULT_ARRIVAL_YAW_TOLERANCE_DEG,
        type=float,
    )
    parser.add_argument("--waypoint-tolerance-m", default=DEFAULT_WAYPOINT_TOLERANCE_M, type=float)
    parser.add_argument("--goal-tolerance-m", default=DEFAULT_GOAL_TOLERANCE_M, type=float)
    parser.add_argument(
        "--min-waypoint-spacing-m",
        default=DEFAULT_MIN_WAYPOINT_SPACING_M,
        type=float,
    )
    parser.add_argument("--control-rate-hz", default=DEFAULT_CONTROL_RATE_HZ, type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args(argv)

    if not args.run_id:
        args.run_id = datetime.now().strftime("waypoint_two_stage_%Y%m%d_%H%M%S")
    validate_args(parser, args)
    return args


def validate_args(parser, args):
    positive_float_fields = [
        "localization_angular_speed",
        "amcl_validation_timeout_sec",
        "known_start_validation_timeout_sec",
        "preflight_timeout_sec",
        "nav_to_start_timeout_sec",
        "initial_pose_var_x",
        "initial_pose_var_y",
        "initial_pose_var_yaw_rad2",
        "max_pose_age_sec",
        "max_scan_age_sec",
        "max_amcl_age_sec",
        "max_amcl_var_x",
        "max_amcl_var_y",
        "max_amcl_var_yaw_rad2",
        "max_stable_pose_jump_m",
        "max_stable_yaw_jump_deg",
        "spin_min_scan_range_m",
        "arrival_tolerance_m",
        "arrival_yaw_tolerance_deg",
        "waypoint_tolerance_m",
        "goal_tolerance_m",
        "control_rate_hz",
    ]
    for field in positive_float_fields:
        if getattr(args, field) <= 0.0:
            parser.error(f"--{field.replace('_', '-')} must be greater than zero")
    if args.localization_spin_deg == 0.0:
        parser.error("--localization-spin-deg must be non-zero")
    if args.stable_amcl_samples < 1:
        parser.error("--stable-amcl-samples must be >= 1")
    if args.spin_min_valid_scan_count < 1:
        parser.error("--spin-min-valid-scan-count must be >= 1")
    if args.min_waypoint_spacing_m < 0.0:
        parser.error("--min-waypoint-spacing-m must be non-negative")
    if args.localization_mode == "known-start":
        missing = [
            name
            for name in ["initial_pose_x", "initial_pose_y", "initial_pose_yaw_deg"]
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(
                "known-start mode requires "
                + ", ".join("--" + name.replace("_", "-") for name in missing)
            )


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        waypoints = load_waypoints(args.waypoints)
        staging_goal = staging_goal_from_waypoints(waypoints)
    except Exception as exc:
        print(f"two_stage_waypoint_run.py: error: {exc}", file=sys.stderr)
        return 2

    follower_command = build_follower_command(args)
    if args.dry_run:
        print_dry_run(args, waypoints, staging_goal, follower_command)
        return 0

    if not require_motion_confirmation(args, staging_goal, follower_command):
        print("Two-stage waypoint run cancelled.")
        return 130

    if rclpy is None:
        print("ROS 2 Python modules are unavailable. Source ROS 2 Humble first.", file=sys.stderr)
        return 2

    diagnostics = RunDiagnostics(
        timestamp=timestamp_now(),
        start_wall_time=timestamp_now(),
        follower_command=shlex.join(follower_command),
        notes=args.notes,
    )
    start_monotonic = time.time()
    node = None
    return_code = 1

    try:
        rclpy.init()
        node = TwoStageCoordinator(args)
        node.preflight_before_motion()

        phase_start = time.time()
        if args.localization_mode == "global":
            node.call_global_localization()
            node.perform_localization_spin()
            timeout = args.amcl_validation_timeout_sec
        else:
            node.publish_known_start_initial_pose()
            timeout = args.known_start_validation_timeout_sec
        stability = node.wait_for_amcl_validation(timeout)
        diagnostics.localization_duration_sec = time.time() - phase_start
        diagnostics.amcl_var_x = stability.cov_x
        diagnostics.amcl_var_y = stability.cov_y
        diagnostics.amcl_var_yaw_rad2 = stability.cov_yaw_rad2
        diagnostics.stable_samples = stability.stable_count
        diagnostics.max_pose_jump_m = stability.max_pose_jump_m
        diagnostics.max_yaw_jump_deg = stability.max_yaw_jump_deg

        _pose, frame = node.validate_post_localization_tf()
        diagnostics.selected_base_frame = frame

        phase_start = time.time()
        diagnostics.nav2_result_status = node.navigate_to_staging(staging_goal)
        diagnostics.nav2_duration_sec = time.time() - phase_start

        arrival = node.verify_arrival(staging_goal)
        diagnostics.selected_base_frame = arrival.base_frame
        diagnostics.tf_arrival_x = arrival.pose.x
        diagnostics.tf_arrival_y = arrival.pose.y
        diagnostics.tf_arrival_yaw_deg = arrival.pose.yaw_deg
        diagnostics.arrival_position_error_m = arrival.position_error_m
        diagnostics.arrival_yaw_error_deg = arrival.yaw_error_deg
        node.stop_repeatedly()

        phase_start = time.time()
        follower_result = run_follower_command(follower_command)
        diagnostics.follower_duration_sec = time.time() - phase_start
        diagnostics.follower_return_code = follower_result.returncode
        if follower_result.returncode != 0:
            raise RuntimeError(f"Follower exited with return code {follower_result.returncode}")

        final_pose, _frame = node.lookup_pose()
        diagnostics.final_tf_x = final_pose.x
        diagnostics.final_tf_y = final_pose.y
        diagnostics.final_tf_yaw_deg = final_pose.yaw_deg
        diagnostics.status = "completed"
        diagnostics.final_status_reason = "completed"
        return_code = 0

    except KeyboardInterrupt:
        diagnostics.status = "interrupted"
        diagnostics.final_status_reason = "keyboard_interrupt"
        print("Interrupted. Cancelling navigation and sending stop...")
        if node is not None:
            cleanup_motion(node)
        return_code = 130

    except Exception as exc:
        diagnostics.status = "failed"
        diagnostics.final_status_reason = str(exc)
        print(f"two_stage_waypoint_run.py: error: {exc}", file=sys.stderr)
        if node is not None:
            cleanup_motion(node)
        return_code = 1

    finally:
        diagnostics.end_wall_time = timestamp_now()
        diagnostics.duration_sec = time.time() - start_monotonic
        if node is not None:
            try:
                final_pose, _frame = node.lookup_pose()
                diagnostics.final_tf_x = final_pose.x
                diagnostics.final_tf_y = final_pose.y
                diagnostics.final_tf_yaw_deg = final_pose.yaw_deg
                diagnostics.selected_base_frame = node.selected_base_frame
            except Exception:
                pass
            try:
                node.destroy_node()
            finally:
                rclpy.shutdown()
        if not args.no_log:
            try:
                append_csv_row(
                    args.results_csv,
                    CSV_HEADER,
                    build_log_row(args, staging_goal, diagnostics),
                )
            except Exception as log_exc:
                print(f"Could not write two-stage run log: {log_exc}", file=sys.stderr)

    return return_code


if __name__ == "__main__":
    sys.exit(main())
