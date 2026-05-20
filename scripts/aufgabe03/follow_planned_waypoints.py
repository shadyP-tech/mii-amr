#!/usr/bin/env python3
"""
Follow simplified A* waypoints with a conservative TF-based controller.

This script executes a static waypoint CSV in the map frame. It assumes Nav2
localization/AMCL is already running, but it publishes /cmd_vel itself.
"""

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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
    qos_profile_sensor_data = None
    Time = None
    PoseWithCovarianceStamped = object
    Twist = None
    LaserScan = object
    tf2_ros = None


DEFAULT_WAYPOINTS_CSV = Path("results/aufgabe03/aufgabe03_waypoints.csv")
DEFAULT_RESULTS_CSV = Path("results/aufgabe03/aufgabe03_waypoint_follow_runs.csv")

DEFAULT_LINEAR_SPEED_MPS = 0.03
DEFAULT_MIN_LINEAR_SPEED_MPS = 0.01
DEFAULT_LINEAR_GAIN = 0.25
DEFAULT_MAX_ANGULAR_SPEED_RADPS = 0.12
DEFAULT_YAW_GAIN = 0.5
DEFAULT_WAYPOINT_TOLERANCE_M = 0.12
DEFAULT_GOAL_TOLERANCE_M = 0.12
DEFAULT_ROTATE_START_HEADING_ERROR_DEG = 20.0
DEFAULT_ROTATE_STOP_HEADING_ERROR_DEG = 4.0
DEFAULT_FORWARD_YAW_DEADBAND_DEG = 4.0
DEFAULT_FORWARD_STOP_HEADING_ERROR_DEG = 18.0
DEFAULT_MIN_WAYPOINT_SPACING_M = 0.12
DEFAULT_START_SELECTION = "path-progress"
DEFAULT_START_ON_PATH_TOLERANCE_M = 0.25
DEFAULT_SCAN_HALF_ANGLE_DEG = 35.0
DEFAULT_HARD_STOP_RANGE_M = 0.16
DEFAULT_MIN_SCAN_RANGE_M = 0.24
DEFAULT_ROTATION_STOP_RANGE_M = 0.18
DEFAULT_MAX_POSE_AGE_SEC = 10.0
DEFAULT_MAX_SCAN_AGE_SEC = 8.0
DEFAULT_MAX_AMCL_AGE_SEC = 15.0
DEFAULT_MAX_AMCL_VAR_X = 0.05
DEFAULT_MAX_AMCL_VAR_Y = 0.05
DEFAULT_MAX_AMCL_VAR_YAW = 0.10
DEFAULT_MAX_WAYPOINT_TIME_SEC = 180.0
DEFAULT_MAX_TF_UPDATE_GAP_SEC = 5.0
DEFAULT_TF_RECOVERY_TIME_SEC = 5.0
DEFAULT_LOCALIZATION_RECOVERY_TIME_SEC = 5.0
DEFAULT_CONTROL_RATE_HZ = 10.0
DEFAULT_SETTLE_SEC = 0.5

DEFAULT_STARTUP_TIMEOUT_SEC = 20.0
STOP_PUBLISH_COUNT = 10
STOP_PUBLISH_HZ = 10.0

BASE_CSV_HEADER = [
    "timestamp",
    "run_id",
    "waypoint_csv",
    "waypoint_count",
    "reached_count",
    "status",
    "blocked_waypoint_index",
    "blocked_waypoint_x",
    "blocked_waypoint_y",
    "timeout_waypoint_index",
    "base_frame_used",
    "start_x",
    "start_y",
    "start_yaw_deg",
    "final_x",
    "final_y",
    "final_yaw_deg",
    "min_scan_range_m",
    "p05_scan_range_m",
    "amcl_var_x",
    "amcl_var_y",
    "amcl_var_yaw",
    "linear_speed_mps",
    "min_linear_speed_mps",
    "linear_gain",
    "max_angular_speed_radps",
    "yaw_gain",
    "notes",
]

CSV_HEADER = BASE_CSV_HEADER + [
    "selected_start_segment_index",
    "selected_start_waypoint_index",
    "distance_to_path_m",
    "tf_pose_age_sec",
    "max_tf_update_gap_sec",
    "tf_stale_warning_count",
    "localization_warning_count",
    "recovery_pause_count",
    "max_abs_yaw_error_deg",
    "mean_abs_yaw_error_deg",
    "rotate_seconds",
    "forward_seconds",
    "final_status_reason",
]


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
class TargetState:
    distance_m: float
    heading_deg: float
    yaw_error_deg: float


@dataclass(frozen=True)
class ScanSafety:
    safe: bool
    reason: str
    valid_count: int
    min_range_m: float | None
    percentile_5_m: float | None


@dataclass(frozen=True)
class AmclHealth:
    ok: bool
    warnings: list[str]
    cov_x: float | None
    cov_y: float | None
    cov_yaw: float | None
    age_sec: float | None


@dataclass(frozen=True)
class StartSelection:
    waypoints: list[Waypoint]
    selected_segment_index: int | None
    selected_waypoint_index: int | None
    distance_to_path_m: float | None


@dataclass
class RuntimeDiagnostics:
    selected_start_segment_index: int | None = None
    selected_start_waypoint_index: int | None = None
    distance_to_path_m: float | None = None
    tf_pose_age_sec: float | None = None
    max_tf_update_gap_sec: float | None = None
    tf_stale_warning_count: int = 0
    localization_warning_count: int = 0
    recovery_pause_count: int = 0
    max_abs_yaw_error_deg: float = 0.0
    yaw_error_sum_deg: float = 0.0
    yaw_error_count: int = 0
    rotate_seconds: float = 0.0
    forward_seconds: float = 0.0
    final_status_reason: str = ""

    @property
    def mean_abs_yaw_error_deg(self):
        if self.yaw_error_count == 0:
            return 0.0
        return self.yaw_error_sum_deg / self.yaw_error_count


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def shortest_angle_delta_deg(start_deg, end_deg):
    return (end_deg - start_deg + 180.0) % 360.0 - 180.0


def normalize_angle_rad(angle_rad):
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_to_yaw_deg(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def waypoint_distance(a, b):
    return math.hypot(b.x - a.x, b.y - a.y)


def heading_between(a, b):
    return math.degrees(math.atan2(b.y - a.y, b.x - a.x))


def target_state(current_pose, waypoint):
    dx = waypoint.x - current_pose.x
    dy = waypoint.y - current_pose.y
    heading = math.degrees(math.atan2(dy, dx))
    return TargetState(
        distance_m=math.hypot(dx, dy),
        heading_deg=heading,
        yaw_error_deg=shortest_angle_delta_deg(current_pose.yaw_deg, heading),
    )


def waypoint_reached(distance_m, is_final, waypoint_tolerance_m, goal_tolerance_m):
    tolerance = goal_tolerance_m if is_final else waypoint_tolerance_m
    return distance_m <= tolerance


def should_rotate(current_mode, yaw_error_deg, start_threshold_deg, stop_threshold_deg):
    abs_error = abs(yaw_error_deg)
    if current_mode == "rotate":
        return abs_error > stop_threshold_deg
    return abs_error > start_threshold_deg


def velocity_command(
    distance_m,
    yaw_error_deg,
    rotate_mode,
    linear_speed_mps,
    min_linear_speed_mps,
    linear_gain,
    max_angular_speed_radps,
    yaw_gain,
    forward_yaw_deadband_deg=0.0,
    forward_stop_heading_error_deg=180.0,
):
    angular_z = clamp(
        math.radians(yaw_error_deg) * yaw_gain,
        -max_angular_speed_radps,
        max_angular_speed_radps,
    )
    abs_yaw_error = abs(yaw_error_deg)
    if rotate_mode or abs_yaw_error >= forward_stop_heading_error_deg:
        return 0.0, angular_z

    linear_x = clamp(
        distance_m * linear_gain,
        min_linear_speed_mps,
        linear_speed_mps,
    )
    if abs_yaw_error <= forward_yaw_deadband_deg:
        return linear_x, 0.0

    scale_span = forward_stop_heading_error_deg - forward_yaw_deadband_deg
    heading_scale = 1.0
    if scale_span > 0.0:
        heading_scale = 1.0 - (abs_yaw_error - forward_yaw_deadband_deg) / scale_span
    heading_scale = clamp(heading_scale, 0.0, 1.0)
    linear_x *= heading_scale
    if linear_x > 0.0:
        linear_x = max(min_linear_speed_mps, linear_x)
    return linear_x, angular_z


def load_waypoints(path):
    path = Path(path)
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        required = {"index", "world_x_m", "world_y_m"}
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(
                f"{path} is missing required column(s): {', '.join(missing)}"
            )

        waypoints = []
        previous_xy = None
        for row in reader:
            waypoint = Waypoint(
                index=int(float(row["index"])),
                x=float(row["world_x_m"]),
                y=float(row["world_y_m"]),
            )
            xy = (waypoint.x, waypoint.y)
            if previous_xy is not None and xy == previous_xy:
                continue
            previous_xy = xy
            waypoints.append(waypoint)

    if not waypoints:
        raise ValueError(f"{path} does not contain any waypoints")
    return waypoints


def is_heading_change(previous_wp, current_wp, next_wp, tolerance_deg=1.0):
    incoming = heading_between(previous_wp, current_wp)
    outgoing = heading_between(current_wp, next_wp)
    return abs(shortest_angle_delta_deg(incoming, outgoing)) > tolerance_deg


def downsample_waypoints(waypoints, min_spacing_m):
    if len(waypoints) <= 2:
        return list(waypoints)

    selected = [waypoints[0]]
    for index in range(1, len(waypoints) - 1):
        current = waypoints[index]
        if is_heading_change(waypoints[index - 1], current, waypoints[index + 1]):
            selected.append(current)
            continue
        if waypoint_distance(selected[-1], current) >= min_spacing_m:
            selected.append(current)

    if selected[-1] != waypoints[-1]:
        selected.append(waypoints[-1])
    return selected


def prepare_executable_waypoints(waypoints, skip_first=True, min_spacing_m=0.0):
    executable = list(waypoints[1:] if skip_first else waypoints)
    if min_spacing_m > 0.0:
        executable = downsample_waypoints(executable, min_spacing_m)
    if len(executable) < 2:
        raise ValueError(
            "Waypoint CSV needs at least two executable waypoints after processing"
        )
    return executable


def distance_point_to_segment_m(point, segment_start, segment_end):
    dx = segment_end.x - segment_start.x
    dy = segment_end.y - segment_start.y
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return math.hypot(point.x - segment_start.x, point.y - segment_start.y), 0.0
    projection = (
        (point.x - segment_start.x) * dx + (point.y - segment_start.y) * dy
    ) / length_sq
    projection = clamp(projection, 0.0, 1.0)
    closest_x = segment_start.x + projection * dx
    closest_y = segment_start.y + projection * dy
    return math.hypot(point.x - closest_x, point.y - closest_y), projection


def nearest_path_segment(point, waypoints):
    if len(waypoints) < 2:
        raise ValueError("Need at least two waypoints for path-progress selection")

    best = None
    for segment_index in range(len(waypoints) - 1):
        distance_m, projection = distance_point_to_segment_m(
            point,
            waypoints[segment_index],
            waypoints[segment_index + 1],
        )
        candidate = (distance_m, segment_index, projection)
        if best is None or candidate < best:
            best = candidate
    return best


def select_path_progress_waypoints(
    waypoints,
    current_pose,
    start_on_path_tolerance_m,
    waypoint_tolerance_m,
    goal_tolerance_m,
    min_spacing_m=0.0,
):
    distance_to_path_m, segment_index, _projection = nearest_path_segment(
        current_pose,
        waypoints,
    )
    if distance_to_path_m > start_on_path_tolerance_m:
        raise ValueError(
            "Current pose is too far from the planned path: "
            f"distance={distance_to_path_m:.3f} m, "
            f"tolerance={start_on_path_tolerance_m:.3f} m"
        )

    next_index = min(segment_index + 1, len(waypoints) - 1)
    while next_index < len(waypoints) - 1:
        waypoint = waypoints[next_index]
        distance_m = math.hypot(waypoint.x - current_pose.x, waypoint.y - current_pose.y)
        if not waypoint_reached(
            distance_m,
            is_final=False,
            waypoint_tolerance_m=waypoint_tolerance_m,
            goal_tolerance_m=goal_tolerance_m,
        ):
            break
        next_index += 1

    selected = list(waypoints[next_index:])
    if min_spacing_m > 0.0 and len(selected) > 1:
        selected = downsample_waypoints(selected, min_spacing_m)
    if not selected:
        selected = [waypoints[-1]]

    return StartSelection(
        waypoints=selected,
        selected_segment_index=segment_index,
        selected_waypoint_index=selected[0].index,
        distance_to_path_m=distance_to_path_m,
    )


def select_executable_waypoints(
    waypoints,
    current_pose,
    start_selection,
    start_on_path_tolerance_m,
    waypoint_tolerance_m,
    goal_tolerance_m,
    min_spacing_m,
    skip_first=True,
):
    if start_selection == "fixed-skip":
        selected = prepare_executable_waypoints(
            waypoints,
            skip_first=skip_first,
            min_spacing_m=min_spacing_m,
        )
        return StartSelection(
            waypoints=selected,
            selected_segment_index=None,
            selected_waypoint_index=selected[0].index,
            distance_to_path_m=None,
        )
    if start_selection == "path-progress":
        return select_path_progress_waypoints(
            waypoints,
            current_pose,
            start_on_path_tolerance_m,
            waypoint_tolerance_m,
            goal_tolerance_m,
            min_spacing_m=min_spacing_m,
        )
    raise ValueError(f"unsupported start selection mode: {start_selection!r}")


def percentile(values, percent):
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percent / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def valid_scan_ranges(
    ranges,
    angle_min,
    angle_increment,
    range_min,
    range_max,
    sector_half_angle_deg=None,
):
    selected = []
    half_angle_rad = (
        math.radians(sector_half_angle_deg)
        if sector_half_angle_deg is not None
        else None
    )
    for index, raw_range in enumerate(ranges):
        if not math.isfinite(raw_range):
            continue
        if raw_range < range_min or raw_range > range_max:
            continue
        if half_angle_rad is not None:
            angle = normalize_angle_rad(angle_min + index * angle_increment)
            if abs(angle) > half_angle_rad:
                continue
        selected.append(float(raw_range))
    return selected


def evaluate_scan_safety(
    ranges,
    angle_min,
    angle_increment,
    range_min,
    range_max,
    mode,
    scan_half_angle_deg,
    hard_stop_range_m,
    min_scan_range_m,
    rotation_stop_range_m,
):
    if mode not in {"forward", "rotate"}:
        raise ValueError(f"unsupported scan mode: {mode!r}")

    sector = scan_half_angle_deg if mode == "forward" else None
    selected = valid_scan_ranges(
        ranges,
        angle_min,
        angle_increment,
        range_min,
        range_max,
        sector_half_angle_deg=sector,
    )
    if not selected:
        return ScanSafety(False, "no_valid_scan_ranges", 0, None, None)

    min_range = min(selected)
    percentile_5 = percentile(selected, 5.0)
    soft_threshold = min_scan_range_m if mode == "forward" else rotation_stop_range_m

    if min_range < hard_stop_range_m:
        return ScanSafety(False, "hard_stop", len(selected), min_range, percentile_5)
    if percentile_5 < soft_threshold:
        return ScanSafety(False, "soft_stop", len(selected), min_range, percentile_5)
    return ScanSafety(True, "clear", len(selected), min_range, percentile_5)


def amcl_covariances(covariance):
    return float(covariance[0]), float(covariance[7]), float(covariance[35])


def evaluate_amcl_health(
    covariance,
    age_sec,
    max_age_sec,
    max_var_x,
    max_var_y,
    max_var_yaw,
    fail_on_bad_localization=False,
):
    cov_x, cov_y, cov_yaw = amcl_covariances(covariance)
    warnings = []
    if age_sec is None or age_sec > max_age_sec:
        warnings.append("stale_amcl")
    if cov_x > max_var_x:
        warnings.append("high_cov_x")
    if cov_y > max_var_y:
        warnings.append("high_cov_y")
    if cov_yaw > max_var_yaw:
        warnings.append("high_cov_yaw")
    return AmclHealth(
        ok=not warnings or not fail_on_bad_localization,
        warnings=warnings,
        cov_x=cov_x,
        cov_y=cov_y,
        cov_yaw=cov_yaw,
        age_sec=age_sec,
    )


def age_ok(age_sec, max_age_sec):
    return age_sec is not None and age_sec <= max_age_sec


def ordered_base_frames(base_frame, fallback_base_frame):
    frames = []
    for frame in [base_frame, fallback_base_frame]:
        if frame and frame not in frames:
            frames.append(frame)
    return frames


def build_log_row(
    args,
    waypoint_count,
    reached_count,
    status,
    notes,
    start_pose=None,
    final_pose=None,
    blocked_waypoint=None,
    timeout_waypoint=None,
    base_frame_used="",
    scan_safety=None,
    amcl_health=None,
    diagnostics=None,
):
    diagnostics = diagnostics or RuntimeDiagnostics()
    blocked = blocked_waypoint or Waypoint("", "", "")
    timeout = timeout_waypoint or Waypoint("", "", "")
    return [
        datetime.now().isoformat(timespec="seconds"),
        args.run_id,
        str(args.waypoints),
        waypoint_count,
        reached_count,
        status,
        blocked.index,
        blocked.x,
        blocked.y,
        timeout.index,
        base_frame_used,
        *(pose_fields(start_pose)),
        *(pose_fields(final_pose)),
        "" if scan_safety is None or scan_safety.min_range_m is None else scan_safety.min_range_m,
        "" if scan_safety is None or scan_safety.percentile_5_m is None else scan_safety.percentile_5_m,
        "" if amcl_health is None or amcl_health.cov_x is None else amcl_health.cov_x,
        "" if amcl_health is None or amcl_health.cov_y is None else amcl_health.cov_y,
        "" if amcl_health is None or amcl_health.cov_yaw is None else amcl_health.cov_yaw,
        args.linear_speed,
        args.min_linear_speed,
        args.linear_gain,
        args.max_angular_speed,
        args.yaw_gain,
        notes,
        "" if diagnostics.selected_start_segment_index is None else diagnostics.selected_start_segment_index,
        "" if diagnostics.selected_start_waypoint_index is None else diagnostics.selected_start_waypoint_index,
        "" if diagnostics.distance_to_path_m is None else diagnostics.distance_to_path_m,
        "" if diagnostics.tf_pose_age_sec is None else diagnostics.tf_pose_age_sec,
        "" if diagnostics.max_tf_update_gap_sec is None else diagnostics.max_tf_update_gap_sec,
        diagnostics.tf_stale_warning_count,
        diagnostics.localization_warning_count,
        diagnostics.recovery_pause_count,
        diagnostics.max_abs_yaw_error_deg,
        diagnostics.mean_abs_yaw_error_deg,
        diagnostics.rotate_seconds,
        diagnostics.forward_seconds,
        diagnostics.final_status_reason,
    ]


def pose_fields(pose):
    if pose is None:
        return ["", "", ""]
    return [pose.x, pose.y, pose.yaw_deg]


def append_csv_row(path, header, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    if file_exists:
        with path.open(newline="") as file:
            existing_header = next(csv.reader(file), None)
        if existing_header == header:
            pass
        elif existing_header and header[: len(existing_header)] == existing_header:
            migrate_csv_header(path, header)
        else:
            raise RuntimeError(
                f"{path} has an unrecognized schema. Move or migrate it first."
            )
    with path.open("a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def migrate_csv_header(path, header):
    path = Path(path)
    with path.open(newline="") as file:
        rows = list(csv.reader(file))

    migrated = [header]
    for row in rows[1:]:
        migrated.append(row + [""] * (len(header) - len(row)))

    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(migrated)


class WaypointFollower(Node):
    def __init__(self, args):
        if rclpy is None:
            raise RuntimeError(
                "ROS 2 Python modules are unavailable. Source ROS 2 Humble before "
                "running the waypoint follower."
            )
        super().__init__("follow_planned_waypoints")
        self.args = args
        self.last_scan = None
        self.last_scan_received_sec = None
        self.last_amcl = None
        self.last_amcl_received_sec = None
        self.base_frame_used = ""
        self.reached_count = 0
        self.start_pose = None
        self.final_pose = None
        self.last_scan_safety = None
        self.last_amcl_health = None
        self.diagnostics = RuntimeDiagnostics(
            max_tf_update_gap_sec=args.max_tf_update_gap_sec,
        )
        self.last_tf_stamp_sec = None
        self.last_tf_stamp_change_local_sec = None

        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self.amcl_callback,
            10,
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        time.sleep(1.0)

    def scan_callback(self, msg):
        self.last_scan = msg
        self.last_scan_received_sec = time.time()

    def amcl_callback(self, msg):
        self.last_amcl = msg
        self.last_amcl_received_sec = time.time()

    def publish_velocity(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.pub.publish(msg)

    def stop_repeatedly(self):
        msg = Twist()
        sleep_sec = 1.0 / STOP_PUBLISH_HZ
        for _ in range(STOP_PUBLISH_COUNT):
            if rclpy.ok():
                self.pub.publish(msg)
            time.sleep(sleep_sec)

    def wait_for_startup_gate(self, timeout_sec=None):
        if timeout_sec is None:
            timeout_sec = self.args.startup_timeout_sec
        require_amcl = (
            self.args.require_amcl_startup
            or self.args.fail_on_bad_localization
            or self.args.pause_on_bad_localization
        )
        start = time.time()
        while rclpy.ok():
            have_scan = self.last_scan is not None
            have_amcl = self.last_amcl is not None
            have_tf = False
            try:
                _pose, frame = self.lookup_pose()
                self.base_frame_used = frame
                have_tf = True
            except RuntimeError:
                have_tf = False

            if have_scan and have_tf and (have_amcl or not require_amcl):
                return
            if time.time() - start > timeout_sec:
                missing = []
                if not have_scan:
                    missing.append("/scan")
                if require_amcl and not have_amcl:
                    missing.append("/amcl_pose")
                if not have_tf:
                    missing.append(
                        f"TF {self.args.map_frame}->{self.args.base_frame}/"
                        f"{self.args.fallback_base_frame}"
                    )
                raise RuntimeError(
                    "Timed out waiting for startup data: " + ", ".join(missing)
                )
            rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError("ROS shutdown during startup gate.")

    def lookup_pose(self):
        errors = []
        for frame in ordered_base_frames(self.args.base_frame, self.args.fallback_base_frame):
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.args.map_frame,
                    frame,
                    Time(),
                )
                self.base_frame_used = frame
                return transform_to_pose2d(transform, frame), frame
            except Exception as exc:
                errors.append(f"{frame}: {exc}")
        raise RuntimeError("Could not lookup TF pose: " + "; ".join(errors))

    def update_tf_tracking(self, pose):
        if pose.stamp_sec is None:
            return None
        now = time.time()
        if self.last_tf_stamp_sec is None or pose.stamp_sec != self.last_tf_stamp_sec:
            self.last_tf_stamp_sec = pose.stamp_sec
            self.last_tf_stamp_change_local_sec = now
        if self.last_tf_stamp_change_local_sec is None:
            self.last_tf_stamp_change_local_sec = now
        return now - self.last_tf_stamp_change_local_sec

    def record_motion_sample(self, yaw_error_deg, linear_x, angular_z, sample_seconds):
        abs_error = abs(yaw_error_deg)
        self.diagnostics.max_abs_yaw_error_deg = max(
            self.diagnostics.max_abs_yaw_error_deg,
            abs_error,
        )
        self.diagnostics.yaw_error_sum_deg += abs_error
        self.diagnostics.yaw_error_count += 1
        if abs(linear_x) <= 1e-9 and abs(angular_z) > 1e-9:
            self.diagnostics.rotate_seconds += sample_seconds
        else:
            self.diagnostics.forward_seconds += sample_seconds

    def current_amcl_health(self):
        if self.last_amcl is None:
            return AmclHealth(
                ok=not self.args.fail_on_bad_localization,
                warnings=["missing_amcl"],
                cov_x=None,
                cov_y=None,
                cov_yaw=None,
                age_sec=None,
            )
        age_sec = (
            None if self.last_amcl_received_sec is None
            else time.time() - self.last_amcl_received_sec
        )
        covariance = self.last_amcl.pose.covariance
        return evaluate_amcl_health(
            covariance,
            age_sec,
            self.args.max_amcl_age_sec,
            self.args.max_amcl_var_x,
            self.args.max_amcl_var_y,
            self.args.max_amcl_var_yaw,
            fail_on_bad_localization=self.args.fail_on_bad_localization,
        )

    def check_health_or_raise(self):
        pose, frame = self.lookup_pose()
        if pose.stamp_sec is None:
            raise RuntimeError("TF pose has no usable timestamp.")
        pose_age = time.time() - pose.stamp_sec
        self.diagnostics.tf_pose_age_sec = pose_age
        if pose_age > self.args.max_pose_age_sec:
            self.diagnostics.tf_stale_warning_count += 1
            message = f"TF pose is stale: age={pose_age:.3f} sec"
            if self.args.fail_on_stale_tf:
                raise RuntimeError(message)
            self.get_logger().warn(message)

        tf_update_gap_sec = self.update_tf_tracking(pose)
        if (
            tf_update_gap_sec is not None
            and tf_update_gap_sec > self.args.max_tf_update_gap_sec
        ):
            raise RecoverableHealthError(
                "tf_update_gap",
                self.args.tf_recovery_time_sec,
                "TF transform stamp stopped updating: "
                f"gap={tf_update_gap_sec:.3f} sec, "
                f"limit={self.args.max_tf_update_gap_sec:.3f} sec",
            )

        scan_age = (
            None if self.last_scan_received_sec is None
            else time.time() - self.last_scan_received_sec
        )
        if not age_ok(scan_age, self.args.max_scan_age_sec):
            raise RuntimeError(f"/scan is stale: age={scan_age}")

        amcl_health = self.current_amcl_health()
        if amcl_health.warnings:
            self.diagnostics.localization_warning_count += 1
            message = "AMCL localization warning(s): " + ",".join(amcl_health.warnings)
            if not amcl_health.ok:
                raise RuntimeError(message)
            if self.args.pause_on_bad_localization:
                raise RecoverableHealthError(
                    "bad_localization",
                    self.args.localization_recovery_time_sec,
                    message,
                )
            self.get_logger().warn(message)

        return pose, frame, amcl_health

    def check_health_or_recover(self):
        while True:
            try:
                return self.check_health_or_raise()
            except RecoverableHealthError as exc:
                self.diagnostics.recovery_pause_count += 1
                self.stop_repeatedly()
                self.get_logger().warn(
                    f"{exc}; pausing for up to {exc.timeout_sec:.1f} sec"
                )
                deadline = time.time() + exc.timeout_sec
                last_message = str(exc)
                while time.time() < deadline and rclpy.ok():
                    rclpy.spin_once(self, timeout_sec=0.1)
                    time.sleep(0.1)
                    try:
                        return self.check_health_or_raise()
                    except RecoverableHealthError as retry_exc:
                        last_message = str(retry_exc)
                raise RuntimeError(
                    f"{exc.reason} did not recover within "
                    f"{exc.timeout_sec:.1f} sec: {last_message}"
                )

    def check_scan_or_raise(self, mode):
        if self.last_scan is None:
            raise RuntimeError("No /scan sample is available.")
        safety = evaluate_scan_safety(
            self.last_scan.ranges,
            self.last_scan.angle_min,
            self.last_scan.angle_increment,
            self.last_scan.range_min,
            self.last_scan.range_max,
            mode,
            self.args.scan_half_angle_deg,
            self.args.hard_stop_range_m,
            self.args.min_scan_range_m,
            self.args.rotation_stop_range_m,
        )
        if not safety.safe:
            raise BlockedByScanError(safety)
        return safety

    def follow_waypoints(self, waypoints):
        reached_count = 0
        start_pose, _frame, amcl_health = self.check_health_or_recover()
        final_pose = start_pose
        last_scan_safety = None
        self.start_pose = start_pose
        self.final_pose = final_pose
        self.last_amcl_health = amcl_health

        for waypoint_index, waypoint in enumerate(waypoints):
            self.get_logger().info(
                f"[{waypoint_index + 1}/{len(waypoints)}] "
                f"target waypoint {waypoint.index}: "
                f"x={waypoint.x:.3f}, y={waypoint.y:.3f}"
            )
            waypoint_start = time.time()
            mode = "forward"
            is_final = waypoint_index == len(waypoints) - 1

            while rclpy.ok():
                pose, _frame, amcl_health = self.check_health_or_recover()
                final_pose = pose
                self.final_pose = final_pose
                self.last_amcl_health = amcl_health
                state = target_state(pose, waypoint)

                if waypoint_reached(
                    state.distance_m,
                    is_final,
                    self.args.waypoint_tolerance_m,
                    self.args.goal_tolerance_m,
                ):
                    reached_count += 1
                    self.reached_count = reached_count
                    self.stop_repeatedly()
                    time.sleep(self.args.settle_sec)
                    break

                if time.time() - waypoint_start > self.args.max_waypoint_time_sec:
                    raise WaypointTimeoutError(waypoint)

                rotate_mode = should_rotate(
                    mode,
                    state.yaw_error_deg,
                    self.args.rotate_start_heading_error_deg,
                    self.args.rotate_stop_heading_error_deg,
                )
                mode = "rotate" if rotate_mode else "forward"
                try:
                    last_scan_safety = self.check_scan_or_raise(mode)
                    self.last_scan_safety = last_scan_safety
                except BlockedByScanError as exc:
                    raise BlockedByScanError(exc.scan_safety, waypoint) from exc
                linear_x, angular_z = velocity_command(
                    state.distance_m,
                    state.yaw_error_deg,
                    rotate_mode,
                    self.args.linear_speed,
                    self.args.min_linear_speed,
                    self.args.linear_gain,
                    self.args.max_angular_speed,
                    self.args.yaw_gain,
                    self.args.forward_yaw_deadband_deg,
                    self.args.forward_stop_heading_error_deg,
                )
                self.record_motion_sample(
                    state.yaw_error_deg,
                    linear_x,
                    angular_z,
                    1.0 / self.args.control_rate_hz,
                )
                self.publish_velocity(linear_x, angular_z)
                rclpy.spin_once(self, timeout_sec=1.0 / self.args.control_rate_hz)
                time.sleep(1.0 / self.args.control_rate_hz)

        return {
            "reached_count": reached_count,
            "start_pose": start_pose,
            "final_pose": final_pose,
            "scan_safety": last_scan_safety,
            "amcl_health": amcl_health,
            "base_frame_used": self.base_frame_used,
        }


class BlockedByScanError(RuntimeError):
    def __init__(self, scan_safety, waypoint=None):
        super().__init__(
            "Blocked by /scan safety: "
            f"reason={scan_safety.reason}, "
            f"min={scan_safety.min_range_m}, p05={scan_safety.percentile_5_m}"
        )
        self.scan_safety = scan_safety
        self.waypoint = waypoint


class WaypointTimeoutError(RuntimeError):
    def __init__(self, waypoint):
        super().__init__(f"Timed out trying to reach waypoint {waypoint.index}")
        self.waypoint = waypoint


class RecoverableHealthError(RuntimeError):
    def __init__(self, reason, timeout_sec, message):
        super().__init__(message)
        self.reason = reason
        self.timeout_sec = timeout_sec


def transform_to_pose2d(transform, frame_id):
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    stamp = transform.header.stamp
    stamp_sec = float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0
    return Pose2D(
        x=float(translation.x),
        y=float(translation.y),
        yaw_deg=quaternion_to_yaw_deg(
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        ),
        stamp_sec=stamp_sec,
        frame_id=frame_id,
    )


def require_motion_confirmation(args, waypoints):
    if args.yes:
        return True

    print("\nThis command will publish /cmd_vel to the physical TurtleBot.")
    print("Safety requirements:")
    print("  - Nav2 localization is running with the saved map")
    print("  - RViz pose estimate is set and /scan aligns with the map")
    print("  - no active Nav2 goal/controller is publishing /cmd_vel")
    print("  - clear the path and keep an operator near the robot")
    print("  - keep Ctrl+C and physical stop available")
    print(f"Run ID: {args.run_id}")
    print(f"Waypoints: {len(waypoints)} from {args.waypoints}")
    response = input("Type RUN to start waypoint following: ").strip()
    return response == "RUN"


def print_dry_run(args, raw_waypoints, executable_waypoints):
    print("Waypoint follower dry run")
    print(f"Waypoint CSV: {args.waypoints}")
    print(f"Raw waypoints: {len(raw_waypoints)}")
    print(f"Executable waypoints: {len(executable_waypoints)}")
    print(f"Map frame: {args.map_frame}")
    print(f"Base frame: {args.base_frame}, fallback: {args.fallback_base_frame}")
    print(f"Linear speed: {args.linear_speed:.3f} m/s")
    print(f"Max angular speed: {args.max_angular_speed:.3f} rad/s")
    print(f"Waypoint tolerance: {args.waypoint_tolerance_m:.3f} m")
    print(f"Goal tolerance: {args.goal_tolerance_m:.3f} m")
    print(f"Start selection: {args.start_selection}")
    if args.start_selection == "path-progress":
        print(
            "Runtime route selection uses live TF after startup; "
            "the route below is a fixed-skip preview."
        )
    print("Executable route:")
    for index, waypoint in enumerate(executable_waypoints, start=1):
        print(f"  {index}. source index {waypoint.index}: x={waypoint.x:.3f}, y={waypoint.y:.3f}")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Follow planned A* waypoints using TF pose and /cmd_vel.",
    )
    parser.add_argument("--waypoints", default=DEFAULT_WAYPOINTS_CSV, type=Path)
    parser.add_argument("--run-id", help="Run ID for logging.")
    parser.add_argument("--results-csv", default=DEFAULT_RESULTS_CSV, type=Path)
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--base-frame", default="base_footprint")
    parser.add_argument("--fallback-base-frame", default="base_link")
    parser.add_argument("--linear-speed", default=DEFAULT_LINEAR_SPEED_MPS, type=float)
    parser.add_argument("--min-linear-speed", default=DEFAULT_MIN_LINEAR_SPEED_MPS, type=float)
    parser.add_argument("--linear-gain", default=DEFAULT_LINEAR_GAIN, type=float)
    parser.add_argument("--max-angular-speed", default=DEFAULT_MAX_ANGULAR_SPEED_RADPS, type=float)
    parser.add_argument("--yaw-gain", default=DEFAULT_YAW_GAIN, type=float)
    parser.add_argument("--forward-yaw-deadband-deg", default=DEFAULT_FORWARD_YAW_DEADBAND_DEG, type=float)
    parser.add_argument("--forward-stop-heading-error-deg", default=DEFAULT_FORWARD_STOP_HEADING_ERROR_DEG, type=float)
    parser.add_argument("--waypoint-tolerance-m", default=DEFAULT_WAYPOINT_TOLERANCE_M, type=float)
    parser.add_argument("--goal-tolerance-m", default=DEFAULT_GOAL_TOLERANCE_M, type=float)
    parser.add_argument(
        "--rotate-start-heading-error-deg",
        default=DEFAULT_ROTATE_START_HEADING_ERROR_DEG,
        type=float,
    )
    parser.add_argument(
        "--rotate-stop-heading-error-deg",
        default=DEFAULT_ROTATE_STOP_HEADING_ERROR_DEG,
        type=float,
    )
    parser.add_argument("--min-waypoint-spacing-m", default=DEFAULT_MIN_WAYPOINT_SPACING_M, type=float)
    parser.add_argument(
        "--start-selection",
        default=DEFAULT_START_SELECTION,
        choices=["path-progress", "fixed-skip"],
    )
    parser.add_argument("--start-on-path-tolerance-m", default=DEFAULT_START_ON_PATH_TOLERANCE_M, type=float)
    parser.add_argument("--scan-half-angle-deg", default=DEFAULT_SCAN_HALF_ANGLE_DEG, type=float)
    parser.add_argument("--hard-stop-range-m", default=DEFAULT_HARD_STOP_RANGE_M, type=float)
    parser.add_argument("--min-scan-range-m", default=DEFAULT_MIN_SCAN_RANGE_M, type=float)
    parser.add_argument("--rotation-stop-range-m", default=DEFAULT_ROTATION_STOP_RANGE_M, type=float)
    parser.add_argument("--max-pose-age-sec", default=DEFAULT_MAX_POSE_AGE_SEC, type=float)
    parser.add_argument("--max-scan-age-sec", default=DEFAULT_MAX_SCAN_AGE_SEC, type=float)
    parser.add_argument("--max-amcl-age-sec", default=DEFAULT_MAX_AMCL_AGE_SEC, type=float)
    parser.add_argument("--max-amcl-var-x", default=DEFAULT_MAX_AMCL_VAR_X, type=float)
    parser.add_argument("--max-amcl-var-y", default=DEFAULT_MAX_AMCL_VAR_Y, type=float)
    parser.add_argument("--max-amcl-var-yaw", default=DEFAULT_MAX_AMCL_VAR_YAW, type=float)
    parser.add_argument("--max-waypoint-time-sec", default=DEFAULT_MAX_WAYPOINT_TIME_SEC, type=float)
    parser.add_argument("--max-tf-update-gap-sec", default=DEFAULT_MAX_TF_UPDATE_GAP_SEC, type=float)
    parser.add_argument("--tf-recovery-time-sec", default=DEFAULT_TF_RECOVERY_TIME_SEC, type=float)
    parser.add_argument(
        "--localization-recovery-time-sec",
        default=DEFAULT_LOCALIZATION_RECOVERY_TIME_SEC,
        type=float,
    )
    parser.add_argument("--control-rate-hz", default=DEFAULT_CONTROL_RATE_HZ, type=float)
    parser.add_argument("--settle-sec", default=DEFAULT_SETTLE_SEC, type=float)
    parser.add_argument("--startup-timeout-sec", default=DEFAULT_STARTUP_TIMEOUT_SEC, type=float)
    parser.add_argument("--notes", default="follow_planned_waypoints")
    parser.add_argument("--fail-on-bad-localization", action="store_true")
    parser.add_argument("--pause-on-bad-localization", action="store_true")
    parser.add_argument("--require-amcl-startup", action="store_true")
    parser.add_argument("--fail-on-stale-tf", action="store_true")
    parser.add_argument("--no-skip-first-waypoint", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args(argv)

    if not args.run_id:
        args.run_id = datetime.now().strftime("waypoint_follow_%Y%m%d_%H%M%S")
    validate_args(parser, args)
    return args


def validate_args(parser, args):
    positive_fields = [
        "linear_speed",
        "min_linear_speed",
        "linear_gain",
        "max_angular_speed",
        "yaw_gain",
        "waypoint_tolerance_m",
        "goal_tolerance_m",
        "rotate_start_heading_error_deg",
        "rotate_stop_heading_error_deg",
        "scan_half_angle_deg",
        "hard_stop_range_m",
        "min_scan_range_m",
        "rotation_stop_range_m",
        "start_on_path_tolerance_m",
        "max_pose_age_sec",
        "max_scan_age_sec",
        "max_amcl_age_sec",
        "max_amcl_var_x",
        "max_amcl_var_y",
        "max_amcl_var_yaw",
        "max_waypoint_time_sec",
        "max_tf_update_gap_sec",
        "tf_recovery_time_sec",
        "localization_recovery_time_sec",
        "control_rate_hz",
        "startup_timeout_sec",
    ]
    for field in positive_fields:
        if getattr(args, field) <= 0.0:
            parser.error(f"--{field.replace('_', '-')} must be greater than zero")
    if args.min_linear_speed > args.linear_speed:
        parser.error("--min-linear-speed must be <= --linear-speed")
    if args.rotate_stop_heading_error_deg >= args.rotate_start_heading_error_deg:
        parser.error("--rotate-stop-heading-error-deg must be < --rotate-start-heading-error-deg")
    if args.forward_yaw_deadband_deg < 0.0:
        parser.error("--forward-yaw-deadband-deg must be non-negative")
    if args.forward_yaw_deadband_deg >= args.forward_stop_heading_error_deg:
        parser.error("--forward-yaw-deadband-deg must be < --forward-stop-heading-error-deg")
    if args.forward_stop_heading_error_deg >= args.rotate_start_heading_error_deg:
        parser.error(
            "--forward-stop-heading-error-deg must be < "
            "--rotate-start-heading-error-deg"
        )
    if args.hard_stop_range_m >= args.min_scan_range_m:
        parser.error("--hard-stop-range-m must be < --min-scan-range-m")
    if args.hard_stop_range_m >= args.rotation_stop_range_m:
        parser.error("--hard-stop-range-m must be < --rotation-stop-range-m")
    if not (0.0 < args.scan_half_angle_deg <= 90.0):
        parser.error("--scan-half-angle-deg must be > 0 and <= 90")
    if args.settle_sec < 0.0:
        parser.error("--settle-sec must be non-negative")


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    try:
        raw_waypoints = load_waypoints(args.waypoints)
        preview_waypoints = prepare_executable_waypoints(
            raw_waypoints,
            skip_first=not args.no_skip_first_waypoint,
            min_spacing_m=args.min_waypoint_spacing_m,
        )
    except Exception as exc:
        print(f"follow_planned_waypoints.py: error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print_dry_run(args, raw_waypoints, preview_waypoints)
        return 0

    if not require_motion_confirmation(args, preview_waypoints):
        print("Waypoint following cancelled.")
        return 130

    if rclpy is None:
        print(
            "ROS 2 Python modules are unavailable. Source ROS 2 Humble before running.",
            file=sys.stderr,
        )
        return 2

    rclpy.init()
    node = WaypointFollower(args)
    status = "failed"
    notes = args.notes
    reached_count = 0
    executable_waypoints = preview_waypoints
    start_pose = None
    final_pose = None
    blocked_waypoint = None
    timeout_waypoint = None
    scan_safety = None
    amcl_health = None
    return_code = 1

    try:
        node.wait_for_startup_gate()
        start_pose, _frame, amcl_health = node.check_health_or_recover()
        start_selection = select_executable_waypoints(
            raw_waypoints,
            start_pose,
            args.start_selection,
            args.start_on_path_tolerance_m,
            args.waypoint_tolerance_m,
            args.goal_tolerance_m,
            args.min_waypoint_spacing_m,
            skip_first=not args.no_skip_first_waypoint,
        )
        executable_waypoints = start_selection.waypoints
        node.diagnostics.selected_start_segment_index = (
            start_selection.selected_segment_index
        )
        node.diagnostics.selected_start_waypoint_index = (
            start_selection.selected_waypoint_index
        )
        node.diagnostics.distance_to_path_m = start_selection.distance_to_path_m
        node.get_logger().info(
            "Selected executable route: "
            f"segment={start_selection.selected_segment_index}, "
            f"first_waypoint={start_selection.selected_waypoint_index}, "
            f"distance_to_path={start_selection.distance_to_path_m}"
        )
        result = node.follow_waypoints(executable_waypoints)
        reached_count = result["reached_count"]
        start_pose = result["start_pose"]
        final_pose = result["final_pose"]
        scan_safety = result["scan_safety"]
        amcl_health = result["amcl_health"]
        status = "completed"
        node.diagnostics.final_status_reason = "completed"
        return_code = 0

    except KeyboardInterrupt:
        status = "interrupted"
        notes = f"{args.notes};keyboard_interrupt"
        node.diagnostics.final_status_reason = "keyboard_interrupt"
        print("Interrupted. Sending stop command...")
        return_code = 130

    except BlockedByScanError as exc:
        status = "blocked"
        notes = f"{args.notes};{exc}"
        node.diagnostics.final_status_reason = str(exc)
        reached_count = node.reached_count
        start_pose = node.start_pose
        final_pose = node.final_pose
        amcl_health = node.last_amcl_health
        scan_safety = exc.scan_safety
        blocked_waypoint = exc.waypoint
        node.get_logger().error(str(exc))
        return_code = 1

    except WaypointTimeoutError as exc:
        status = "timeout"
        notes = f"{args.notes};{exc}"
        node.diagnostics.final_status_reason = str(exc)
        reached_count = node.reached_count
        start_pose = node.start_pose
        final_pose = node.final_pose
        scan_safety = node.last_scan_safety
        amcl_health = node.last_amcl_health
        timeout_waypoint = exc.waypoint
        node.get_logger().error(str(exc))
        return_code = 1

    except Exception as exc:
        status = "failed"
        notes = f"{args.notes};{exc}"
        node.diagnostics.final_status_reason = str(exc)
        reached_count = node.reached_count
        start_pose = node.start_pose
        final_pose = node.final_pose
        scan_safety = node.last_scan_safety
        amcl_health = node.last_amcl_health
        node.get_logger().error(str(exc))
        return_code = 1

    finally:
        try:
            node.stop_repeatedly()
            if final_pose is None:
                try:
                    final_pose, _frame = node.lookup_pose()
                except Exception:
                    final_pose = None
        finally:
            if not args.no_log:
                try:
                    row = build_log_row(
                        args,
                        len(executable_waypoints),
                        reached_count,
                        status,
                        notes,
                        start_pose=start_pose,
                        final_pose=final_pose,
                        blocked_waypoint=blocked_waypoint,
                        timeout_waypoint=timeout_waypoint,
                        base_frame_used=node.base_frame_used,
                        scan_safety=scan_safety,
                        amcl_health=amcl_health,
                        diagnostics=node.diagnostics,
                    )
                    append_csv_row(args.results_csv, CSV_HEADER, row)
                    node.get_logger().info(f"Saved run log to {args.results_csv}")
                except Exception as log_exc:
                    print(f"Could not write waypoint-follow log: {log_exc}", file=sys.stderr)
            node.destroy_node()
            rclpy.shutdown()

    return return_code


if __name__ == "__main__":
    sys.exit(main())
