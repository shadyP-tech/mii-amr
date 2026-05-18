#!/usr/bin/env python3
"""
Run a conservative coverage motion for Cartographer mapping in the lab arena.

This script is intentionally motion-only: start TurtleBot bringup and
Cartographer separately, then run this script to publish bounded /cmd_vel
commands while /odom terminates primitives and /scan provides an emergency stop.
"""

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
except ImportError:
    rclpy = None
    Node = object
    qos_profile_sensor_data = None
    Twist = None
    Odometry = object
    LaserScan = object


ARENA_LENGTH_M = 3.90
ARENA_WIDTH_M = 1.898
DEFAULT_SAFETY_MARGIN_M = 0.70
ROBOT_RADIUS_M = 0.105

DEFAULT_LINEAR_SPEED_MPS = 0.05
DEFAULT_ANGULAR_SPEED_RADPS = 0.25
DEFAULT_FORWARD_HALF_PASS_M = 1.20
DEFAULT_FORWARD_TOLERANCE_M = 0.02
DEFAULT_ROTATION_TOLERANCE_DEG = 2.0
DEFAULT_SETTLE_SEC = 0.5
DEFAULT_MIN_SCAN_RANGE_M = 0.28
DEFAULT_HARD_STOP_RANGE_M = 0.18
DEFAULT_SCAN_HALF_ANGLE_DEG = 35.0
DEFAULT_MAX_ACTION_TIME_SEC = 90.0

COMMAND_PERIOD_SEC = 0.05
TOPIC_TIMEOUT_SEC = 5.0
STOP_PUBLISH_COUNT = 10
STOP_PUBLISH_HZ = 10.0

DEFAULT_RESULTS_CSV = Path("results/aufgabe03/arena_coverage_runs.csv")

CSV_HEADER = [
    "timestamp",
    "run_id",
    "route_actions",
    "linear_speed_mps",
    "angular_speed_radps",
    "forward_half_pass_m",
    "forward_tolerance_m",
    "rotation_tolerance_deg",
    "settle_sec",
    "min_scan_range_m",
    "hard_stop_range_m",
    "scan_half_angle_deg",
    "max_action_time_sec",
    "odom_start_x",
    "odom_start_y",
    "odom_start_yaw_deg",
    "odom_final_x",
    "odom_final_y",
    "odom_final_yaw_deg",
    "status",
    "notes",
]


@dataclass(frozen=True)
class RouteAction:
    name: str
    kind: str
    value: float = 0.0


@dataclass(frozen=True)
class ScanSafety:
    safe: bool
    reason: str
    valid_count: int
    min_range_m: float | None
    percentile_5_m: float | None


def shortest_angle_delta_deg(start_deg, end_deg):
    return (end_deg - start_deg + 180.0) % 360.0 - 180.0


def normalize_angle_rad(angle_rad):
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def build_default_route(forward_half_pass_m=DEFAULT_FORWARD_HALF_PASS_M):
    return [
        RouteAction("SPIN_SCAN", "rotate", 360.0),
        RouteAction("FORWARD", "forward", forward_half_pass_m),
        RouteAction("SPIN_SCAN", "rotate", 360.0),
        RouteAction("TURN_AROUND", "rotate", 180.0),
        RouteAction("FORWARD", "forward", 2.0 * forward_half_pass_m),
        RouteAction("SPIN_SCAN", "rotate", 360.0),
        RouteAction("TURN_AROUND", "rotate", 180.0),
        RouteAction("FORWARD", "forward", forward_half_pass_m),
        RouteAction("STOP", "stop", 0.0),
    ]


def route_action_text(action):
    if action.kind == "forward":
        return f"{action.name} {action.value:.2f}m"
    if action.kind == "rotate":
        return f"{action.name} {action.value:.0f}deg"
    return action.name


def route_actions_text(route):
    return ";".join(route_action_text(action) for action in route)


def route_long_axis_positions(route):
    heading_deg = 0.0
    position_m = 0.0
    positions = [position_m]

    for action in route:
        if action.kind == "rotate":
            heading_deg += action.value
        elif action.kind == "forward":
            position_m += math.cos(math.radians(heading_deg)) * action.value
            positions.append(position_m)

    return positions


def route_long_axis_margin_m(route, arena_length_m=ARENA_LENGTH_M):
    max_abs_position = max(abs(position) for position in route_long_axis_positions(route))
    return arena_length_m / 2.0 - max_abs_position


def estimate_route_duration_sec(route, linear_speed_mps, angular_speed_radps, settle_sec):
    duration = 0.0
    moving_actions = 0
    for action in route:
        if action.kind == "forward":
            duration += action.value / linear_speed_mps
            moving_actions += 1
        elif action.kind == "rotate":
            duration += math.radians(abs(action.value)) / angular_speed_radps
            moving_actions += 1
    return duration + moving_actions * settle_sec


def projected_forward_progress_m(start_pose, current_pose):
    start_yaw_rad = math.radians(start_pose["yaw_deg"])
    dx = current_pose["x"] - start_pose["x"]
    dy = current_pose["y"] - start_pose["y"]
    return dx * math.cos(start_yaw_rad) + dy * math.sin(start_yaw_rad)


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

    lower_value = ordered[lower]
    upper_value = ordered[upper]
    weight = rank - lower
    return lower_value + (upper_value - lower_value) * weight


def valid_scan_ranges(
    ranges,
    angle_min,
    angle_increment,
    range_min,
    range_max,
    sector_half_angle_deg=None,
):
    selected = []
    sector_half_angle_rad = (
        math.radians(sector_half_angle_deg)
        if sector_half_angle_deg is not None
        else None
    )

    for index, raw_range in enumerate(ranges):
        if not math.isfinite(raw_range):
            continue
        if raw_range < range_min or raw_range > range_max:
            continue

        if sector_half_angle_rad is not None:
            angle = normalize_angle_rad(angle_min + index * angle_increment)
            if abs(angle) > sector_half_angle_rad:
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
    min_scan_range_m,
    hard_stop_range_m,
):
    if mode not in {"forward", "rotate"}:
        raise ValueError(f"unsupported scan safety mode: {mode!r}")

    sector_half_angle_deg = scan_half_angle_deg if mode == "forward" else None
    selected = valid_scan_ranges(
        ranges,
        angle_min,
        angle_increment,
        range_min,
        range_max,
        sector_half_angle_deg=sector_half_angle_deg,
    )

    if not selected:
        return ScanSafety(False, "no_valid_scan_ranges", 0, None, None)

    min_range = min(selected)
    percentile_5 = percentile(selected, 5.0)

    if min_range < hard_stop_range_m:
        return ScanSafety(False, "hard_stop", len(selected), min_range, percentile_5)
    if percentile_5 < min_scan_range_m:
        return ScanSafety(False, "soft_stop", len(selected), min_range, percentile_5)

    return ScanSafety(True, "clear", len(selected), min_range, percentile_5)


def validate_motion_config(args, route):
    errors = []
    if not (0.0 < args.linear_speed <= 0.15):
        errors.append("--linear-speed must be > 0 and <= 0.15")
    if not (0.0 < args.angular_speed <= 0.6):
        errors.append("--angular-speed must be > 0 and <= 0.6")
    if not (args.min_scan_range_m > ROBOT_RADIUS_M):
        errors.append("--min-scan-range-m must be greater than robot radius")
    if not (0.0 < args.hard_stop_range_m < args.min_scan_range_m):
        errors.append("--hard-stop-range-m must be > 0 and < --min-scan-range-m")
    if not (0.0 < args.scan_half_angle_deg <= 90.0):
        errors.append("--scan-half-angle-deg must be > 0 and <= 90")
    if not (0.0 < args.forward_tolerance_m < 0.20):
        errors.append("--forward-tolerance-m must be > 0 and < 0.20")
    if not (0.0 < args.rotation_tolerance_deg < 20.0):
        errors.append("--rotation-tolerance-deg must be > 0 and < 20")
    if args.settle_sec < 0.0:
        errors.append("--settle-sec must be non-negative")
    if args.max_action_time_sec <= 0.0:
        errors.append("--max-action-time-sec must be greater than zero")
    if args.forward_half_pass_m <= 0.0:
        errors.append("--forward-half-pass-m must be greater than zero")

    margin = route_long_axis_margin_m(route, ARENA_LENGTH_M)
    if margin < DEFAULT_SAFETY_MARGIN_M:
        errors.append(
            "route exceeds arena bounds: "
            f"long-axis margin {margin:.3f} m is below "
            f"{DEFAULT_SAFETY_MARGIN_M:.3f} m"
        )

    if errors:
        raise ValueError("; ".join(errors))


def odom_to_xy_yaw(msg):
    if msg is None:
        return None

    p = msg.pose.pose.position
    q = msg.pose.pose.orientation

    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return {
        "x": p.x,
        "y": p.y,
        "yaw_deg": math.degrees(yaw),
    }


def pose_fields(pose):
    if pose is None:
        return ["", "", ""]
    return [pose["x"], pose["y"], pose["yaw_deg"]]


def append_csv_row(path, header, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0

    if file_exists:
        with path.open(newline="") as file:
            reader = csv.reader(file)
            existing_header = next(reader, None)
        if existing_header != header:
            raise RuntimeError(
                f"{path} has an unrecognized schema. Move or migrate it before "
                "running arena coverage logging."
            )

    with path.open("a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def build_log_row(args, route, odom_start, odom_final, status, notes):
    return [
        datetime.now().isoformat(timespec="seconds"),
        args.run_id,
        route_actions_text(route),
        args.linear_speed,
        args.angular_speed,
        args.forward_half_pass_m,
        args.forward_tolerance_m,
        args.rotation_tolerance_deg,
        args.settle_sec,
        args.min_scan_range_m,
        args.hard_stop_range_m,
        args.scan_half_angle_deg,
        args.max_action_time_sec,
        *pose_fields(odom_start),
        *pose_fields(odom_final),
        status,
        notes,
    ]


class ArenaCoverageDrive(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError(
                "ROS 2 Python modules are unavailable. Source ROS 2 Humble before "
                "running arena coverage."
            )

        super().__init__("arena_coverage_drive")
        self.last_odom = None
        self.last_scan = None

        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10,
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            qos_profile_sensor_data,
        )
        time.sleep(1.0)

    def odom_callback(self, msg):
        self.last_odom = msg

    def scan_callback(self, msg):
        self.last_scan = msg

    def wait_for_topics(self, timeout_sec=TOPIC_TIMEOUT_SEC):
        start = time.time()
        while rclpy.ok():
            if self.last_odom is not None and self.last_scan is not None:
                return
            if time.time() - start > timeout_sec:
                missing = []
                if self.last_odom is None:
                    missing.append("/odom")
                if self.last_scan is None:
                    missing.append("/scan")
                raise RuntimeError(
                    "Timed out waiting for required topic(s): "
                    + ", ".join(missing)
                )
            rclpy.spin_once(self, timeout_sec=0.1)

        raise RuntimeError("ROS shutdown while waiting for /odom and /scan.")

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

    def check_scan_or_raise(self, mode, args):
        if self.last_scan is None:
            raise RuntimeError("No /scan sample is available for safety checking.")

        safety = evaluate_scan_safety(
            self.last_scan.ranges,
            self.last_scan.angle_min,
            self.last_scan.angle_increment,
            self.last_scan.range_min,
            self.last_scan.range_max,
            mode,
            args.scan_half_angle_deg,
            args.min_scan_range_m,
            args.hard_stop_range_m,
        )

        if not safety.safe:
            min_range = (
                "none"
                if safety.min_range_m is None
                else f"{safety.min_range_m:.3f} m"
            )
            percentile_5 = (
                "none"
                if safety.percentile_5_m is None
                else f"{safety.percentile_5_m:.3f} m"
            )
            raise RuntimeError(
                "Unsafe /scan during "
                f"{mode}: reason={safety.reason}, "
                f"valid_count={safety.valid_count}, "
                f"min={min_range}, p5={percentile_5}"
            )

    def wait_for_odom(self, timeout_sec=TOPIC_TIMEOUT_SEC):
        start = time.time()
        while rclpy.ok() and self.last_odom is None:
            if time.time() - start > timeout_sec:
                raise RuntimeError("Timed out waiting for /odom.")
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.last_odom is None:
            raise RuntimeError("ROS shutdown while waiting for /odom.")
        return self.last_odom

    def drive_forward(self, distance_m, args):
        start_msg = self.wait_for_odom()
        start_pose = odom_to_xy_yaw(start_msg)
        start_time = time.time()
        last_log_time = start_time

        while rclpy.ok():
            self.check_scan_or_raise("forward", args)

            if self.last_odom is not None:
                current_pose = odom_to_xy_yaw(self.last_odom)
                progress_m = projected_forward_progress_m(start_pose, current_pose)
                if progress_m >= distance_m - args.forward_tolerance_m:
                    self.get_logger().info(
                        "Reached forward target: "
                        f"{progress_m:.3f}/{distance_m:.3f} m"
                    )
                    self.stop_repeatedly()
                    return
            else:
                progress_m = None

            elapsed = time.time() - start_time
            if elapsed > args.max_action_time_sec:
                progress = "unknown" if progress_m is None else f"{progress_m:.3f} m"
                self.stop_repeatedly()
                raise RuntimeError(
                    "Timed out during forward action: "
                    f"progress={progress}, target={distance_m:.3f} m"
                )

            if time.time() - last_log_time >= 2.0:
                if progress_m is not None:
                    self.get_logger().info(
                        f"Forward progress {progress_m:.3f}/{distance_m:.3f} m"
                    )
                last_log_time = time.time()

            self.publish_velocity(args.linear_speed, 0.0)
            rclpy.spin_once(self, timeout_sec=COMMAND_PERIOD_SEC)
            time.sleep(COMMAND_PERIOD_SEC)

        self.stop_repeatedly()
        raise RuntimeError("ROS shutdown during forward action.")

    def rotate(self, angle_deg, args):
        start_msg = self.wait_for_odom()
        previous_yaw = odom_to_xy_yaw(start_msg)["yaw_deg"]
        accumulated_deg = 0.0
        sign = 1.0 if angle_deg >= 0.0 else -1.0
        target_abs_deg = abs(angle_deg)
        angular_z = sign * abs(args.angular_speed)
        start_time = time.time()
        last_log_time = start_time

        while rclpy.ok():
            self.check_scan_or_raise("rotate", args)

            if self.last_odom is not None:
                current_yaw = odom_to_xy_yaw(self.last_odom)["yaw_deg"]
                accumulated_deg += shortest_angle_delta_deg(previous_yaw, current_yaw)
                previous_yaw = current_yaw

                if abs(accumulated_deg) >= target_abs_deg - args.rotation_tolerance_deg:
                    self.get_logger().info(
                        "Reached rotation target: "
                        f"{accumulated_deg:.1f}/{angle_deg:.1f} deg"
                    )
                    self.stop_repeatedly()
                    return

            elapsed = time.time() - start_time
            if elapsed > args.max_action_time_sec:
                self.stop_repeatedly()
                raise RuntimeError(
                    "Timed out during rotation action: "
                    f"progress={accumulated_deg:.1f} deg, "
                    f"target={angle_deg:.1f} deg"
                )

            if time.time() - last_log_time >= 2.0:
                self.get_logger().info(
                    f"Rotation progress {accumulated_deg:.1f}/{angle_deg:.1f} deg"
                )
                last_log_time = time.time()

            self.publish_velocity(0.0, angular_z)
            rclpy.spin_once(self, timeout_sec=COMMAND_PERIOD_SEC)
            time.sleep(COMMAND_PERIOD_SEC)

        self.stop_repeatedly()
        raise RuntimeError("ROS shutdown during rotation action.")

    def execute_route(self, route, args):
        for index, action in enumerate(route, start=1):
            prefix = f"[{index}/{len(route)}] {route_action_text(action)}"
            self.get_logger().info(prefix)

            if action.kind == "forward":
                self.drive_forward(action.value, args)
            elif action.kind == "rotate":
                self.rotate(action.value, args)
            elif action.kind == "stop":
                self.stop_repeatedly()
            else:
                raise RuntimeError(f"Unsupported route action kind: {action.kind!r}")

            self.stop_repeatedly()
            time.sleep(args.settle_sec)


def print_dry_run(args, route):
    print("Arena coverage dry run")
    print(f"Arena: {ARENA_WIDTH_M:.3f} m x {ARENA_LENGTH_M:.3f} m")
    print(f"Assumed start: center, facing along the {ARENA_LENGTH_M:.2f} m axis")
    print("Assumptions:")
    print("  - Cartographer is already running")
    print("  - RViz shows /map, /scan, /tf, and robot pose updates")
    print("  - no large obstacle is inside the arena")
    print("  - operator is ready to stop the robot")
    print()
    print("Route:")
    for index, action in enumerate(route, start=1):
        print(f"  {index}. {route_action_text(action)}")

    positions = route_long_axis_positions(route)
    margin = route_long_axis_margin_m(route)
    duration = estimate_route_duration_sec(
        route,
        args.linear_speed,
        args.angular_speed,
        args.settle_sec,
    )
    print()
    print(f"Long-axis positions from start: {[round(p, 3) for p in positions]}")
    print(f"Minimum long-axis wall margin: {margin:.3f} m")
    print(f"Estimated duration: {duration:.1f} sec")
    print(f"Forward safety sector: +/-{args.scan_half_angle_deg:.1f} deg")
    print(f"Soft stop range: {args.min_scan_range_m:.3f} m")
    print(f"Hard stop range: {args.hard_stop_range_m:.3f} m")


def require_motion_confirmation(args, route):
    if args.yes:
        return True

    print("\nThis command will publish /cmd_vel to the physical TurtleBot.")
    print("Safety requirements:")
    print("  - place the robot near arena center")
    print(f"  - align the robot with the {ARENA_LENGTH_M:.2f} m arena axis")
    print("  - clear the arena of large obstacles")
    print("  - keep an operator near the TurtleBot")
    print("  - keep Ctrl+C and physical stop available")
    print("  - run Cartographer and verify RViz feedback first")
    print(f"Run ID: {args.run_id}")
    print(f"Route: {route_actions_text(route)}")
    response = input("Type RUN to start arena coverage motion: ").strip()
    return response == "RUN"


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run a conservative TurtleBot arena coverage motion.",
    )
    parser.add_argument("--run-id", help="Run ID for CSV logging.")
    parser.add_argument(
        "--linear-speed",
        default=DEFAULT_LINEAR_SPEED_MPS,
        type=float,
        help="Forward command speed in m/s.",
    )
    parser.add_argument(
        "--angular-speed",
        default=DEFAULT_ANGULAR_SPEED_RADPS,
        type=float,
        help="Absolute angular command speed in rad/s.",
    )
    parser.add_argument(
        "--forward-half-pass-m",
        default=DEFAULT_FORWARD_HALF_PASS_M,
        type=float,
        help="Distance from center to each long-axis scan point in meters.",
    )
    parser.add_argument(
        "--forward-tolerance-m",
        default=DEFAULT_FORWARD_TOLERANCE_M,
        type=float,
        help="Odometry tolerance for forward primitives.",
    )
    parser.add_argument(
        "--rotation-tolerance-deg",
        default=DEFAULT_ROTATION_TOLERANCE_DEG,
        type=float,
        help="Odometry tolerance for rotation primitives.",
    )
    parser.add_argument(
        "--settle-sec",
        default=DEFAULT_SETTLE_SEC,
        type=float,
        help="Pause after each primitive.",
    )
    parser.add_argument(
        "--min-scan-range-m",
        default=DEFAULT_MIN_SCAN_RANGE_M,
        type=float,
        help="Soft stop threshold for the 5th percentile scan range.",
    )
    parser.add_argument(
        "--hard-stop-range-m",
        default=DEFAULT_HARD_STOP_RANGE_M,
        type=float,
        help="Hard stop threshold for the minimum valid scan range.",
    )
    parser.add_argument(
        "--scan-half-angle-deg",
        default=DEFAULT_SCAN_HALF_ANGLE_DEG,
        type=float,
        help="Front-sector half angle for forward scan safety.",
    )
    parser.add_argument(
        "--max-action-time-sec",
        default=DEFAULT_MAX_ACTION_TIME_SEC,
        type=float,
        help="Maximum time allowed for a single primitive.",
    )
    parser.add_argument(
        "--results-csv",
        default=DEFAULT_RESULTS_CSV,
        type=Path,
        help="CSV file for arena coverage run logs.",
    )
    parser.add_argument(
        "--notes",
        default="arena_coverage_drive",
        help="Notes value written to the run log.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print route and safety summary without ROS or /cmd_vel.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive safety confirmation.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Do not append a row to results/aufgabe03/arena_coverage_runs.csv.",
    )
    args = parser.parse_args(argv)

    if not args.run_id:
        args.run_id = datetime.now().strftime("arena_coverage_%Y%m%d_%H%M%S")

    route = build_default_route(args.forward_half_pass_m)
    try:
        validate_motion_config(args, route)
    except ValueError as exc:
        parser.error(str(exc))

    return args


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    route = build_default_route(args.forward_half_pass_m)

    if args.dry_run:
        print_dry_run(args, route)
        return 0

    if not require_motion_confirmation(args, route):
        print("Arena coverage cancelled.")
        return 130

    if rclpy is None:
        print(
            "ROS 2 Python modules are unavailable. Source ROS 2 Humble before running.",
            file=sys.stderr,
        )
        return 2

    rclpy.init()
    node = ArenaCoverageDrive()
    status = "failed"
    notes = args.notes
    odom_start = None
    odom_final = None
    return_code = 1

    try:
        node.get_logger().info("Waiting for initial /odom and /scan...")
        node.wait_for_topics()
        odom_start = odom_to_xy_yaw(node.last_odom)
        node.get_logger().info(
            "Starting arena coverage route: " + route_actions_text(route)
        )
        node.execute_route(route, args)

        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.05)
        odom_final = odom_to_xy_yaw(node.last_odom)
        status = "completed"
        return_code = 0

    except KeyboardInterrupt:
        status = "interrupted"
        notes = f"{args.notes};keyboard_interrupt"
        print("Interrupted. Sending stop command...")
        return_code = 130

    except Exception as exc:
        status = "failed"
        notes = f"{args.notes};{exc}"
        node.get_logger().error(str(exc))
        return_code = 1

    finally:
        try:
            node.stop_repeatedly()
            for _ in range(5):
                rclpy.spin_once(node, timeout_sec=0.05)
            if odom_final is None:
                odom_final = odom_to_xy_yaw(node.last_odom)
        finally:
            if not args.no_log:
                try:
                    row = build_log_row(args, route, odom_start, odom_final, status, notes)
                    append_csv_row(args.results_csv, CSV_HEADER, row)
                    node.get_logger().info(f"Saved run log to {args.results_csv}")
                except Exception as log_exc:
                    print(f"Could not write arena coverage log: {log_exc}", file=sys.stderr)

            node.destroy_node()
            rclpy.shutdown()

    return return_code


if __name__ == "__main__":
    sys.exit(main())
