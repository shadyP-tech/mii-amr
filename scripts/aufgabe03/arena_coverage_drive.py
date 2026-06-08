#!/usr/bin/env python3
"""
Run a conservative coverage motion for Cartographer mapping in the lab arena.

This script is intentionally motion-only: start TurtleBot bringup and
Cartographer separately, then run this script to publish bounded /cmd_vel
commands while /odom terminates primitives and /scan provides an emergency stop.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, replace
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

from arena_active_spin_core.curve_following import (
    active_explore_curve_path,
    pure_pursuit_curve_command,
    select_curve_lookahead_target,
)
from arena_active_spin_core.math_utils import distance_2d
from arena_active_spin_core.scan_safety import odom_pose_from_msg, scan_sample_from_msg
from arena_shadow_coverage import (
    NO_SHADOW_REASONS,
    ShadowCoverageConfig,
    ShadowCoverageSummary,
    ShadowScanSample,
    plan_shadow_coverage_move,
    prune_shadow_samples,
)


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

DEFAULT_SHADOW_MAX_ATTEMPTS = 12
DEFAULT_SHADOW_MAX_SINGLE_MOVE_M = 0.80
DEFAULT_SHADOW_MAX_TOTAL_DISTANCE_M = 5.0
DEFAULT_SHADOW_MAX_CANDIDATE_PATH_M = 3.0
DEFAULT_SHADOW_GRID_SIZE_M = 5.0
DEFAULT_SHADOW_GRID_RESOLUTION_M = 0.05
DEFAULT_SHADOW_INFLATION_RADIUS_M = 0.12
DEFAULT_SHADOW_SOFT_CLEARANCE_RADIUS_M = 0.15
DEFAULT_SHADOW_SOFT_CLEARANCE_WEIGHT = 2.0
DEFAULT_SHADOW_MAX_PATH_SEGMENTS = 24
DEFAULT_SHADOW_MAX_SAMPLES = 1500
DEFAULT_SHADOW_MAX_SAMPLE_AGE_SEC = 30.0
DEFAULT_SHADOW_MAX_SAMPLE_TRAVEL_M = 1.25
DEFAULT_SHADOW_MAX_SAMPLE_YAW_SPAN_DEG = 420.0
DEFAULT_SHADOW_EMERGENCY_STOP_DISTANCE_M = 0.18
DEFAULT_SHADOW_SIDE_STOP_DISTANCE_M = 0.16
DEFAULT_SHADOW_MIN_VISIBLE_CELLS = 3
DEFAULT_SHADOW_MIN_MOVE_LENGTH_M = 0.12
DEFAULT_SHADOW_RECENT_TARGET_RADIUS_M = 0.25
DEFAULT_SHADOW_COMPLETION_CONFIRMATIONS = 2
DEFAULT_SHADOW_CURVE_LOOKAHEAD_M = 0.16
DEFAULT_SHADOW_CURVE_GOAL_TOLERANCE_M = 0.04
DEFAULT_SHADOW_CURVE_LINEAR_SPEED_MPS = 0.05
DEFAULT_SHADOW_CURVE_MAX_ANGULAR_RADPS = 0.40
DEFAULT_SHADOW_MAX_ODOM_SCAN_AGE_SEC = 0.5
DEFAULT_MAPPER_TOPIC = "/map"
DEFAULT_MAPPER_TOPIC_TIMEOUT_SEC = 5.0

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


@dataclass(frozen=True)
class CurveSafety:
    safe: bool
    reason: str
    front: ScanSafety
    left_min_m: float | None
    right_min_m: float | None


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


def valid_scan_ranges_in_sectors(
    ranges,
    angle_min,
    angle_increment,
    range_min,
    range_max,
    sectors_deg,
):
    selected = []
    for index, raw_range in enumerate(ranges):
        if not math.isfinite(raw_range):
            continue
        if raw_range < range_min or raw_range > range_max:
            continue
        angle = math.degrees(normalize_angle_rad(angle_min + index * angle_increment))
        if any(lower <= angle <= upper for lower, upper in sectors_deg):
            selected.append(float(raw_range))
    return selected


def min_scan_range_in_sectors(scan, sectors_deg):
    selected = valid_scan_ranges_in_sectors(
        scan.ranges,
        scan.angle_min,
        scan.angle_increment,
        scan.range_min,
        scan.range_max,
        sectors_deg,
    )
    return min(selected) if selected else None


def evaluate_shadow_curve_safety(scan, args):
    front = evaluate_scan_safety(
        scan.ranges,
        scan.angle_min,
        scan.angle_increment,
        scan.range_min,
        scan.range_max,
        "forward",
        args.scan_half_angle_deg,
        args.min_scan_range_m,
        args.shadow_emergency_stop_distance_m,
    )
    if not front.safe:
        return CurveSafety(False, f"front_{front.reason}", front, None, None)

    left = min_scan_range_in_sectors(scan, [(60.0, 120.0)])
    right = min_scan_range_in_sectors(scan, [(-120.0, -60.0)])
    if left is None:
        return CurveSafety(False, "left_clearance_missing", front, left, right)
    if right is None:
        return CurveSafety(False, "right_clearance_missing", front, left, right)
    if left < args.shadow_side_stop_distance_m:
        return CurveSafety(False, "left_clearance_below_limit", front, left, right)
    if right < args.shadow_side_stop_distance_m:
        return CurveSafety(False, "right_clearance_below_limit", front, left, right)
    return CurveSafety(True, "clear", front, left, right)


def shadow_config_from_args(args):
    return ShadowCoverageConfig(
        max_attempts=args.shadow_max_attempts,
        max_single_move_m=args.shadow_max_single_move_m,
        max_total_distance_m=args.shadow_max_total_distance_m,
        max_candidate_path_m=args.shadow_max_candidate_path_m,
        grid_resolution_m=args.shadow_grid_resolution_m,
        grid_size_m=args.shadow_grid_size_m,
        inflation_radius_m=args.shadow_inflation_radius_m,
        soft_clearance_radius_m=args.shadow_soft_clearance_radius_m,
        soft_clearance_weight=args.shadow_soft_clearance_weight,
        unknown_blocked=args.shadow_unknown_blocked,
        max_path_segments=args.shadow_max_path_segments,
        max_samples=args.shadow_max_samples,
        max_sample_age_sec=args.shadow_max_sample_age_sec,
        max_sample_travel_m=args.shadow_max_sample_travel_m,
        max_sample_yaw_span_deg=args.shadow_max_sample_yaw_span_deg,
        min_visible_shadow_cells=args.shadow_min_visible_cells,
        min_move_length_m=args.shadow_min_move_length_m,
        recent_target_radius_m=args.shadow_recent_target_radius_m,
        min_endpoint_clearance_m=args.shadow_side_stop_distance_m,
        completion_confirmations=args.shadow_completion_confirmations,
    )


def shadow_diagnostics_path_for_args(args):
    if args.no_shadow_diagnostics:
        return None
    if args.shadow_diagnostics_json is not None:
        return args.shadow_diagnostics_json
    return Path("results/aufgabe03") / f"{args.run_id}_shadow_coverage.json"


def shadow_notes_summary(diagnostics):
    summary = diagnostics.get("summary", {})
    if not summary:
        return "shadow_mode=experimental"
    return (
        "shadow_mode=experimental"
        f";shadow_attempts={summary.get('attempts', 0)}"
        f";shadow_moves={summary.get('moves_executed', 0)}"
        f";shadow_distance_m={float(summary.get('total_distance_m', 0.0)):.3f}"
        f";shadow_stop={summary.get('stop_reason', '')}"
        f";shadow_fallback={summary.get('fallback_used', False)}"
    )


def write_shadow_diagnostics(path, diagnostics):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")


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

    if getattr(args, "coverage_mode", "fixed") == "shadow":
        if args.shadow_max_attempts < 1:
            errors.append("--shadow-max-attempts must be >= 1")
        if args.shadow_completion_confirmations < 1:
            errors.append("--shadow-completion-confirmations must be >= 1")
        for attr, label in (
            ("shadow_max_single_move_m", "--shadow-max-single-move"),
            ("shadow_max_total_distance_m", "--shadow-max-total-distance"),
            ("shadow_grid_size_m", "--shadow-grid-size"),
            ("shadow_grid_resolution_m", "--shadow-grid-resolution"),
            ("shadow_max_sample_age_sec", "--shadow-max-sample-age-sec"),
            ("shadow_max_sample_travel_m", "--shadow-max-sample-travel"),
            ("shadow_max_sample_yaw_span_deg", "--shadow-max-sample-yaw-span-deg"),
            ("shadow_emergency_stop_distance_m", "--shadow-emergency-stop-distance"),
            ("shadow_side_stop_distance_m", "--shadow-side-stop-distance"),
            ("shadow_min_move_length_m", "--shadow-min-move-length"),
            ("shadow_recent_target_radius_m", "--shadow-recent-target-radius"),
            ("shadow_curve_lookahead_m", "--shadow-curve-lookahead"),
            ("shadow_curve_goal_tolerance_m", "--shadow-curve-goal-tolerance"),
            ("shadow_curve_linear_speed_mps", "--shadow-curve-linear-speed"),
            ("shadow_curve_max_angular_radps", "--shadow-curve-max-angular"),
            ("shadow_max_odom_scan_age_sec", "--shadow-max-odom-scan-age-sec"),
            ("mapper_topic_timeout_sec", "--mapper-topic-timeout-sec"),
        ):
            if getattr(args, attr) <= 0.0:
                errors.append(f"{label} must be greater than zero")
        if args.shadow_max_candidate_path_m is not None and args.shadow_max_candidate_path_m <= 0.0:
            errors.append("--shadow-max-candidate-path must be greater than zero")
        if args.shadow_inflation_radius_m < 0.0:
            errors.append("--shadow-inflation must be non-negative")
        if args.shadow_soft_clearance_radius_m < 0.0:
            errors.append("--shadow-soft-clearance-radius must be non-negative")
        if args.shadow_soft_clearance_weight < 0.0:
            errors.append("--shadow-soft-clearance-weight must be non-negative")
        if args.shadow_max_path_segments < 1:
            errors.append("--shadow-max-path-segments must be >= 1")
        if args.shadow_max_samples < 1:
            errors.append("--shadow-max-samples must be >= 1")
        if args.shadow_min_visible_cells < 1:
            errors.append("--shadow-min-visible-cells must be >= 1")
        if args.shadow_emergency_stop_distance_m < ROBOT_RADIUS_M:
            errors.append("--shadow-emergency-stop-distance must be >= robot radius")
        if args.shadow_side_stop_distance_m < ROBOT_RADIUS_M:
            errors.append("--shadow-side-stop-distance must be >= robot radius")
        if args.shadow_emergency_stop_distance_m >= args.min_scan_range_m:
            errors.append("--shadow-emergency-stop-distance must be below --min-scan-range-m")
        if args.shadow_curve_goal_tolerance_m > args.shadow_curve_lookahead_m:
            errors.append("--shadow-curve-goal-tolerance must be <= --shadow-curve-lookahead")
        if args.shadow_curve_linear_speed_mps > 0.10:
            errors.append("--shadow-curve-linear-speed must be <= 0.10")
        if args.shadow_curve_max_angular_radps > 0.80:
            errors.append("--shadow-curve-max-angular must be <= 0.80")

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
        self.last_scan_received_sec = None
        self.latest_odom_pose = None
        self.latest_odom_received_sec = None
        self.shadow_collecting = False
        self.shadow_segment_index = 0
        self.shadow_samples = []
        self.shadow_max_samples = DEFAULT_SHADOW_MAX_SAMPLES
        self.shadow_max_odom_scan_age_sec = DEFAULT_SHADOW_MAX_ODOM_SCAN_AGE_SEC
        self.shadow_sample_rejections = {
            "missing_odom": 0,
            "stale_odom": 0,
        }
        self.shadow_diagnostics = {}

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

    def configure_shadow_collection(self, args):
        self.shadow_collecting = args.coverage_mode == "shadow"
        self.shadow_max_samples = int(args.shadow_max_samples)
        self.shadow_max_odom_scan_age_sec = float(args.shadow_max_odom_scan_age_sec)
        self.shadow_segment_index = 0
        self.shadow_samples = []
        self.shadow_sample_rejections = {
            "missing_odom": 0,
            "stale_odom": 0,
        }

    def begin_shadow_segment(self):
        self.shadow_segment_index += 1

    def odom_callback(self, msg):
        self.last_odom = msg
        self.latest_odom_pose = odom_pose_from_msg(msg)
        self.latest_odom_received_sec = time.time()

    def scan_callback(self, msg):
        received_sec = time.time()
        self.last_scan = msg
        self.last_scan_received_sec = received_sec
        if not self.shadow_collecting:
            return
        if self.latest_odom_pose is None or self.latest_odom_received_sec is None:
            self.shadow_sample_rejections["missing_odom"] += 1
            return
        if received_sec - self.latest_odom_received_sec > self.shadow_max_odom_scan_age_sec:
            self.shadow_sample_rejections["stale_odom"] += 1
            return
        sample = scan_sample_from_msg(msg, self.latest_odom_pose)
        self.shadow_samples.append(
            ShadowScanSample.from_scan_sample(
                sample,
                stamp_sec=received_sec,
                segment_index=self.shadow_segment_index,
            )
        )
        if len(self.shadow_samples) > self.shadow_max_samples:
            del self.shadow_samples[: len(self.shadow_samples) - self.shadow_max_samples]

    def fresh_scan_age_sec(self):
        if self.last_scan_received_sec is None:
            return None
        return time.time() - self.last_scan_received_sec

    def fresh_odom_age_sec(self):
        if self.latest_odom_received_sec is None:
            return None
        return time.time() - self.latest_odom_received_sec

    def wait_for_fresh_shadow_inputs(self, args, timeout_sec=None):
        if timeout_sec is None:
            timeout_sec = max(TOPIC_TIMEOUT_SEC, args.shadow_max_odom_scan_age_sec + 1.0)
        start = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            scan_age = self.fresh_scan_age_sec()
            odom_age = self.fresh_odom_age_sec()
            if (
                scan_age is not None
                and odom_age is not None
                and scan_age <= args.shadow_max_odom_scan_age_sec
                and odom_age <= args.shadow_max_odom_scan_age_sec
            ):
                return
            if time.time() - start > timeout_sec:
                scan_text = "none" if scan_age is None else f"{scan_age:.3f}"
                odom_text = "none" if odom_age is None else f"{odom_age:.3f}"
                raise RuntimeError(
                    "timed_out_waiting_for_fresh_shadow_inputs:"
                    f"scan_age_sec={scan_text},odom_age_sec={odom_text},"
                    f"max_age_sec={args.shadow_max_odom_scan_age_sec:.3f}"
                )
        raise RuntimeError("ROS shutdown while waiting for fresh shadow inputs.")

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

    def wait_for_mapper_topic(self, args):
        start = time.time()
        while rclpy.ok() and time.time() - start <= args.mapper_topic_timeout_sec:
            topics = {name for name, _types in self.get_topic_names_and_types()}
            if args.mapper_topic in topics:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

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
                rclpy.spin_once(self, timeout_sec=0.0)
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

    def _shadow_curve_record(
        self,
        candidate,
        path_points,
        curve_samples,
        driven_distance_m,
        duration_sec,
        stop_reason,
        **extra,
    ):
        record = {
            "executor": "shadow_cmd_vel_curve",
            "executed": True,
            "candidate_kind": candidate.kind,
            "candidate_score": candidate.score,
            "target_x": float(candidate.target_x),
            "target_y": float(candidate.target_y),
            "path_length_m": candidate.path_length_m,
            "curve_path_world": [[float(x), float(y)] for x, y in path_points],
            "curve_samples": list(curve_samples),
            "driven_distance_m": float(driven_distance_m),
            "duration_sec": float(duration_sec),
            "stop_reason": stop_reason,
        }
        record.update(extra)
        return record

    def execute_shadow_curve(self, candidate, args, distance_limit_m):
        self.wait_for_fresh_shadow_inputs(args)
        if self.latest_odom_pose is None:
            raise RuntimeError("shadow_curve_missing_latest_odom_pose")
        move_limit = min(args.shadow_max_single_move_m, max(0.0, distance_limit_m))
        path_points = active_explore_curve_path(
            candidate,
            self.latest_odom_pose,
            move_limit,
        )
        start = time.time()
        deadline = start + max(
            8.0,
            move_limit / max(0.01, abs(args.shadow_curve_linear_speed_mps)) + 5.0,
        )
        final_target = path_points[-1]
        candidate_goal = (
            candidate.path_world[-1]
            if candidate.path_world
            else (
                candidate.simplified_path_world[-1]
                if candidate.simplified_path_world
                else (candidate.target_x, candidate.target_y)
            )
        )
        path_truncated = (
            distance_2d(final_target, candidate_goal)
            > args.shadow_curve_goal_tolerance_m
        )
        previous_point = (
            float(self.latest_odom_pose.x),
            float(self.latest_odom_pose.y),
        )
        total_driven = 0.0
        curve_samples = []

        try:
            while rclpy.ok() and time.time() <= deadline:
                rclpy.spin_once(self, timeout_sec=COMMAND_PERIOD_SEC)
                scan_age = self.fresh_scan_age_sec()
                odom_age = self.fresh_odom_age_sec()
                if scan_age is None or scan_age > args.shadow_max_odom_scan_age_sec:
                    raise RuntimeError("stale_scan_during_shadow_curve")
                if odom_age is None or odom_age > args.shadow_max_odom_scan_age_sec:
                    raise RuntimeError("stale_odom_during_shadow_curve")
                if self.latest_odom_pose is None:
                    raise RuntimeError("fresh_odom_unavailable_during_shadow_curve")

                current_point = (
                    float(self.latest_odom_pose.x),
                    float(self.latest_odom_pose.y),
                )
                delta = distance_2d(previous_point, current_point)
                if math.isfinite(delta):
                    total_driven += delta
                previous_point = current_point

                safety = evaluate_shadow_curve_safety(self.last_scan, args)
                if not safety.safe:
                    self.stop_repeatedly()
                    final_target_distance_m = distance_2d(current_point, final_target)
                    return self._shadow_curve_record(
                        candidate,
                        path_points,
                        curve_samples,
                        total_driven,
                        time.time() - start,
                        f"safety_stop:{safety.reason}",
                        safety_reason=safety.reason,
                        final_target_distance_m=final_target_distance_m,
                        goal_reached=(
                            final_target_distance_m
                            <= args.shadow_curve_goal_tolerance_m
                        ),
                        path_truncated=path_truncated,
                    )

                final_target_distance_m = distance_2d(current_point, final_target)
                if (
                    total_driven >= move_limit
                    or final_target_distance_m <= args.shadow_curve_goal_tolerance_m
                ):
                    self.stop_repeatedly()
                    return self._shadow_curve_record(
                        candidate,
                        path_points,
                        curve_samples,
                        total_driven,
                        time.time() - start,
                        "completed",
                        final_target_distance_m=final_target_distance_m,
                        goal_reached=(
                            final_target_distance_m
                            <= args.shadow_curve_goal_tolerance_m
                        ),
                        path_truncated=path_truncated,
                    )

                target = select_curve_lookahead_target(
                    path_points,
                    current_point,
                    args.shadow_curve_lookahead_m,
                )
                linear_x, angular_z, alpha = pure_pursuit_curve_command(
                    self.latest_odom_pose,
                    target,
                    args.shadow_curve_lookahead_m,
                    args.shadow_curve_linear_speed_mps,
                    args.shadow_curve_max_angular_radps,
                )
                remaining = max(0.0, move_limit - total_driven)
                linear_x = min(linear_x, remaining / max(COMMAND_PERIOD_SEC, 1e-6))
                curve_samples.append(
                    {
                        "odom_x": float(self.latest_odom_pose.x),
                        "odom_y": float(self.latest_odom_pose.y),
                        "odom_yaw_deg": float(self.latest_odom_pose.yaw_deg),
                        "target_x": float(target[0]),
                        "target_y": float(target[1]),
                        "alpha_rad": alpha,
                        "linear_x_mps": linear_x,
                        "angular_z_rad_s": angular_z,
                        "front_min_m": safety.front.min_range_m,
                        "left_min_m": safety.left_min_m,
                        "right_min_m": safety.right_min_m,
                    }
                )
                self.publish_velocity(linear_x, angular_z)
                time.sleep(COMMAND_PERIOD_SEC)

            final_target_distance_m = distance_2d(
                (
                    float(self.latest_odom_pose.x),
                    float(self.latest_odom_pose.y),
                ),
                final_target,
            )
            self.stop_repeatedly()
            return self._shadow_curve_record(
                candidate,
                path_points,
                curve_samples,
                total_driven,
                time.time() - start,
                "timeout_stop_after_progress",
                timeout_sec=deadline - start,
                final_target_distance_m=final_target_distance_m,
                goal_reached=(
                    final_target_distance_m <= args.shadow_curve_goal_tolerance_m
                ),
                path_truncated=path_truncated,
            )
        except Exception:
            self.stop_repeatedly()
            raise

    def _record_shadow_phase(self, diagnostics, phase, reason, plan=None):
        record = {
            "phase": phase,
            "reason": reason,
            "sample_count": len(self.shadow_samples),
        }
        if plan is not None:
            record["plan"] = plan.to_dict()
        diagnostics.setdefault("phases", []).append(record)

    def _plan_shadow_move(self, config, recent_attempts):
        pruned, _window = prune_shadow_samples(
            self.shadow_samples,
            config,
            now_sec=time.time(),
            current_segment=self.shadow_segment_index,
        )
        return plan_shadow_coverage_move(pruned, config, recent_attempts)

    def execute_shadow_coverage(self, args, fallback_route):
        config = shadow_config_from_args(args)
        diagnostics = {
            "coverage_mode": "shadow",
            "experimental": True,
            "config": config.__dict__,
            "mapper": {
                "topic": args.mapper_topic,
                "required": args.require_mapper_topic,
                "available": None,
            },
            "phases": [],
            "executions": [],
            "recent_attempts": [],
            "safety_events": [],
            "sample_rejections": self.shadow_sample_rejections,
        }
        self.shadow_diagnostics = diagnostics
        summary = ShadowCoverageSummary(final_phase="seed_scan")
        mapper_available = self.wait_for_mapper_topic(args)
        diagnostics["mapper"]["available"] = mapper_available
        if not mapper_available:
            message = f"Mapper topic {args.mapper_topic!r} was not seen before shadow coverage."
            if args.require_mapper_topic:
                summary = replace(summary, stop_reason="mapper_topic_missing", final_phase="failed")
                diagnostics["summary"] = summary.to_dict()
                raise RuntimeError(message)
            self.get_logger().warn(message)

        self._record_shadow_phase(diagnostics, "seed_scan", "initial_seed_spin")
        self.rotate(360.0, args)
        self.wait_for_fresh_shadow_inputs(args)
        summary = replace(summary, spin_count=summary.spin_count + 1)

        recent_attempts = []
        confirmations = 0
        shadow_motion_executed = False

        while rclpy.ok():
            if summary.attempts >= args.shadow_max_attempts:
                summary = replace(
                    summary,
                    stop_reason="shadow_motion_attempts_exhausted",
                    final_phase="failed",
                )
                diagnostics["summary"] = summary.to_dict()
                raise RuntimeError("shadow_motion_attempts_exhausted")
            if summary.total_distance_m >= args.shadow_max_total_distance_m:
                summary = replace(
                    summary,
                    stop_reason="shadow_total_distance_exhausted",
                    final_phase="failed",
                )
                diagnostics["summary"] = summary.to_dict()
                raise RuntimeError("shadow_total_distance_exhausted")

            self.wait_for_fresh_shadow_inputs(args)
            plan = self._plan_shadow_move(config, recent_attempts)
            self._record_shadow_phase(
                diagnostics,
                "shadow_mapping",
                plan.reason,
                plan=plan,
            )

            if plan.ok:
                confirmations = 0
                self.begin_shadow_segment()
                remaining_distance = args.shadow_max_total_distance_m - summary.total_distance_m
                record = self.execute_shadow_curve(
                    plan.selected,
                    args,
                    distance_limit_m=remaining_distance,
                )
                diagnostics["executions"].append(record)
                if record["stop_reason"] != "completed":
                    diagnostics["safety_events"].append(
                        {
                            "attempt": summary.attempts,
                            "reason": record["stop_reason"],
                            "target_x": record["target_x"],
                            "target_y": record["target_y"],
                        }
                    )
                    summary = replace(
                        summary,
                        attempts=summary.attempts + 1,
                        moves_executed=summary.moves_executed + 1,
                        total_distance_m=(
                            summary.total_distance_m
                            + float(record.get("driven_distance_m", 0.0))
                        ),
                        stop_reason=record["stop_reason"],
                        final_phase="failed",
                    )
                    diagnostics["summary"] = summary.to_dict()
                    raise RuntimeError(record["stop_reason"])
                shadow_motion_executed = True
                recent_attempt = {
                    "target_x": float(plan.selected.target_x),
                    "target_y": float(plan.selected.target_y),
                    "reason": record["stop_reason"],
                    "driven_distance_m": record.get("driven_distance_m", 0.0),
                }
                recent_attempts.append(recent_attempt)
                diagnostics["recent_attempts"].append(recent_attempt)
                summary = replace(
                    summary,
                    attempts=summary.attempts + 1,
                    moves_executed=summary.moves_executed + 1,
                    total_distance_m=(
                        summary.total_distance_m
                        + float(record.get("driven_distance_m", 0.0))
                    ),
                    final_phase="shadow_mapping",
                )
                time.sleep(args.settle_sec)
                continue

            if plan.reason in NO_SHADOW_REASONS:
                if (
                    not shadow_motion_executed
                    and not args.no_shadow_fallback_route
                ):
                    self._record_shadow_phase(
                        diagnostics,
                        "fixed_fallback",
                        "no_shadow_move_before_motion",
                    )
                    self.execute_route(fallback_route, args)
                    summary = replace(
                        summary,
                        fallback_used=True,
                        stop_reason="fixed_fallback_completed",
                        final_phase="complete",
                    )
                    diagnostics["summary"] = summary.to_dict()
                    return summary

                confirmations += 1
                self._record_shadow_phase(
                    diagnostics,
                    "shadow_confirm",
                    f"confirmation_{confirmations}",
                    plan=plan,
                )
                if confirmations < args.shadow_completion_confirmations:
                    continue

                self.begin_shadow_segment()
                self._record_shadow_phase(
                    diagnostics,
                    "final_verify_spin",
                    "shadow_exhausted_verify_spin",
                )
                self.rotate(360.0, args)
                self.wait_for_fresh_shadow_inputs(args)
                summary = replace(
                    summary,
                    spin_count=summary.spin_count + 1,
                    final_phase="final_verify_spin",
                )
                verify_plan = self._plan_shadow_move(config, recent_attempts)
                self._record_shadow_phase(
                    diagnostics,
                    "final_verify_plan",
                    verify_plan.reason,
                    plan=verify_plan,
                )
                if verify_plan.ok:
                    confirmations = 0
                    self._record_shadow_phase(
                        diagnostics,
                        "shadow_mapping",
                        "verification_found_new_shadow",
                        plan=verify_plan,
                    )
                    continue
                if verify_plan.reason in NO_SHADOW_REASONS:
                    summary = replace(
                        summary,
                        stop_reason="shadow_complete_verified",
                        final_phase="complete",
                    )
                    diagnostics["summary"] = summary.to_dict()
                    return summary
                summary = replace(
                    summary,
                    stop_reason=f"shadow_final_verify_failed:{verify_plan.reason}",
                    final_phase="failed",
                )
                diagnostics["summary"] = summary.to_dict()
                raise RuntimeError(summary.stop_reason)

            if not shadow_motion_executed and not args.no_shadow_fallback_route:
                self._record_shadow_phase(
                    diagnostics,
                    "fixed_fallback",
                    f"shadow_plan_failed_before_motion:{plan.reason}",
                    plan=plan,
                )
                self.execute_route(fallback_route, args)
                summary = replace(
                    summary,
                    fallback_used=True,
                    stop_reason="fixed_fallback_completed",
                    final_phase="complete",
                )
                diagnostics["summary"] = summary.to_dict()
                return summary

            summary = replace(
                summary,
                stop_reason=f"shadow_plan_failed:{plan.reason}",
                final_phase="failed",
            )
            diagnostics["summary"] = summary.to_dict()
            raise RuntimeError(summary.stop_reason)

        summary = replace(
            summary,
            stop_reason="ros_shutdown_during_shadow_coverage",
            final_phase="failed",
        )
        diagnostics["summary"] = summary.to_dict()
        raise RuntimeError("ROS shutdown during shadow coverage.")


def print_dry_run(args, route):
    print("Arena coverage dry run")
    print(f"Arena: {ARENA_WIDTH_M:.3f} m x {ARENA_LENGTH_M:.3f} m")
    print(f"Assumed start: center, facing along the {ARENA_LENGTH_M:.2f} m axis")
    print("Assumptions:")
    print("  - the external mapper is already running")
    print("  - RViz shows /map or mapper feedback, /scan, /tf, and robot pose updates")
    print("  - no large obstacle is inside the arena")
    print("  - operator is ready to stop the robot")
    print()
    print(f"Coverage mode: {args.coverage_mode}")
    if args.coverage_mode == "shadow":
        print("Experimental shadow coverage:")
        print("  - motion-only; the odom-frame shadow grid is not the saved map")
        print("  - initial seed spin, shadow-frontier curves, final verification spin")
        print(f"  - mapper topic: {args.mapper_topic} (strict={args.require_mapper_topic})")
        print(f"  - max attempts: {args.shadow_max_attempts}")
        print(f"  - max single move: {args.shadow_max_single_move_m:.3f} m")
        print(f"  - max total distance: {args.shadow_max_total_distance_m:.3f} m")
        print(f"  - sample window: current + previous segment, max {args.shadow_max_samples} samples")
        print(f"  - side stop distance: {args.shadow_side_stop_distance_m:.3f} m")
        print(f"  - fixed fallback before shadow motion: {not args.no_shadow_fallback_route}")
        print(f"  - diagnostics: {shadow_diagnostics_path_for_args(args)}")
    else:
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
    print("  - run the external mapper and verify RViz feedback first")
    print(f"Run ID: {args.run_id}")
    print(f"Coverage mode: {args.coverage_mode}")
    if args.coverage_mode == "shadow":
        print("Experimental shadow coverage is enabled.")
        print("  - the temporary odom-frame shadow grid is not the saved map")
        print("  - static map saving remains external")
        print(f"  - mapper topic check: {args.mapper_topic}")
        print(f"  - max shadow distance: {args.shadow_max_total_distance_m:.3f} m")
        print(f"  - side stop distance: {args.shadow_side_stop_distance_m:.3f} m")
    else:
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
        "--coverage-mode",
        default="fixed",
        choices=["fixed", "shadow"],
        help="Coverage motion mode. shadow is experimental and motion-only.",
    )
    parser.add_argument(
        "--shadow-max-attempts",
        default=DEFAULT_SHADOW_MAX_ATTEMPTS,
        type=int,
        help="Maximum experimental shadow curve motions.",
    )
    parser.add_argument(
        "--shadow-max-single-move",
        dest="shadow_max_single_move_m",
        default=DEFAULT_SHADOW_MAX_SINGLE_MOVE_M,
        type=float,
        help="Maximum distance for one shadow curve motion in meters.",
    )
    parser.add_argument(
        "--shadow-max-total-distance",
        dest="shadow_max_total_distance_m",
        default=DEFAULT_SHADOW_MAX_TOTAL_DISTANCE_M,
        type=float,
        help="Maximum total shadow curve distance in meters.",
    )
    parser.add_argument(
        "--shadow-max-candidate-path",
        dest="shadow_max_candidate_path_m",
        default=DEFAULT_SHADOW_MAX_CANDIDATE_PATH_M,
        type=float,
        help="Maximum planned candidate path length in meters.",
    )
    parser.add_argument(
        "--shadow-grid-size",
        dest="shadow_grid_size_m",
        default=DEFAULT_SHADOW_GRID_SIZE_M,
        type=float,
        help="Temporary odom-frame shadow grid size in meters.",
    )
    parser.add_argument(
        "--shadow-grid-resolution",
        dest="shadow_grid_resolution_m",
        default=DEFAULT_SHADOW_GRID_RESOLUTION_M,
        type=float,
        help="Temporary shadow grid resolution in meters.",
    )
    parser.add_argument(
        "--shadow-inflation",
        dest="shadow_inflation_radius_m",
        default=DEFAULT_SHADOW_INFLATION_RADIUS_M,
        type=float,
        help="Planning inflation radius for the temporary shadow grid.",
    )
    parser.add_argument(
        "--shadow-soft-clearance-radius",
        dest="shadow_soft_clearance_radius_m",
        default=DEFAULT_SHADOW_SOFT_CLEARANCE_RADIUS_M,
        type=float,
        help="Soft clearance penalty radius for shadow A*.",
    )
    parser.add_argument(
        "--shadow-soft-clearance-weight",
        default=DEFAULT_SHADOW_SOFT_CLEARANCE_WEIGHT,
        type=float,
        help="Soft clearance penalty weight for shadow A*.",
    )
    parser.add_argument(
        "--shadow-max-path-segments",
        default=DEFAULT_SHADOW_MAX_PATH_SEGMENTS,
        type=int,
        help="Maximum simplified A* path segments for a shadow candidate.",
    )
    parser.add_argument(
        "--shadow-max-samples",
        default=DEFAULT_SHADOW_MAX_SAMPLES,
        type=int,
        help="Maximum stored scan samples for shadow planning.",
    )
    parser.add_argument(
        "--shadow-max-sample-age-sec",
        default=DEFAULT_SHADOW_MAX_SAMPLE_AGE_SEC,
        type=float,
        help="Maximum shadow sample age in seconds.",
    )
    parser.add_argument(
        "--shadow-max-sample-travel",
        dest="shadow_max_sample_travel_m",
        default=DEFAULT_SHADOW_MAX_SAMPLE_TRAVEL_M,
        type=float,
        help="Maximum odom travel span for retained shadow samples.",
    )
    parser.add_argument(
        "--shadow-max-sample-yaw-span-deg",
        default=DEFAULT_SHADOW_MAX_SAMPLE_YAW_SPAN_DEG,
        type=float,
        help="Maximum accumulated odom yaw span for retained shadow samples.",
    )
    parser.add_argument(
        "--shadow-emergency-stop-distance",
        dest="shadow_emergency_stop_distance_m",
        default=DEFAULT_SHADOW_EMERGENCY_STOP_DISTANCE_M,
        type=float,
        help="Runtime front emergency stop distance for shadow curves.",
    )
    parser.add_argument(
        "--shadow-side-stop-distance",
        dest="shadow_side_stop_distance_m",
        default=DEFAULT_SHADOW_SIDE_STOP_DISTANCE_M,
        type=float,
        help="Runtime side stop distance for shadow curves.",
    )
    parser.add_argument(
        "--shadow-min-visible-cells",
        default=DEFAULT_SHADOW_MIN_VISIBLE_CELLS,
        type=int,
        help="Minimum visible shadow cells required for a candidate.",
    )
    parser.add_argument(
        "--shadow-min-move-length",
        dest="shadow_min_move_length_m",
        default=DEFAULT_SHADOW_MIN_MOVE_LENGTH_M,
        type=float,
        help="Minimum accepted shadow candidate path length.",
    )
    parser.add_argument(
        "--shadow-recent-target-radius",
        dest="shadow_recent_target_radius_m",
        default=DEFAULT_SHADOW_RECENT_TARGET_RADIUS_M,
        type=float,
        help="Radius for rejecting recently attempted shadow targets.",
    )
    parser.add_argument(
        "--shadow-completion-confirmations",
        default=DEFAULT_SHADOW_COMPLETION_CONFIRMATIONS,
        type=int,
        help="Consecutive no-shadow replans before final verification spin.",
    )
    parser.add_argument(
        "--shadow-curve-lookahead",
        dest="shadow_curve_lookahead_m",
        default=DEFAULT_SHADOW_CURVE_LOOKAHEAD_M,
        type=float,
        help="Pure-pursuit lookahead for shadow curves.",
    )
    parser.add_argument(
        "--shadow-curve-goal-tolerance",
        dest="shadow_curve_goal_tolerance_m",
        default=DEFAULT_SHADOW_CURVE_GOAL_TOLERANCE_M,
        type=float,
        help="Goal tolerance for shadow curve execution.",
    )
    parser.add_argument(
        "--shadow-curve-linear-speed",
        dest="shadow_curve_linear_speed_mps",
        default=DEFAULT_SHADOW_CURVE_LINEAR_SPEED_MPS,
        type=float,
        help="Linear speed for shadow curve execution.",
    )
    parser.add_argument(
        "--shadow-curve-max-angular",
        dest="shadow_curve_max_angular_radps",
        default=DEFAULT_SHADOW_CURVE_MAX_ANGULAR_RADPS,
        type=float,
        help="Maximum angular speed for shadow curve execution.",
    )
    parser.add_argument(
        "--shadow-max-odom-scan-age-sec",
        default=DEFAULT_SHADOW_MAX_ODOM_SCAN_AGE_SEC,
        type=float,
        help="Maximum odom age allowed when pairing scan samples.",
    )
    parser.add_argument(
        "--shadow-unknown-blocked",
        dest="shadow_unknown_blocked",
        action="store_true",
        default=True,
        help="Treat unknown grid cells as blocked for shadow planning.",
    )
    parser.add_argument(
        "--shadow-unknown-free",
        dest="shadow_unknown_blocked",
        action="store_false",
        help="Allow shadow planning through unknown grid cells.",
    )
    parser.add_argument(
        "--no-shadow-fallback-route",
        action="store_true",
        help="Disable fixed-route fallback before any shadow motion.",
    )
    parser.add_argument(
        "--mapper-topic",
        default=DEFAULT_MAPPER_TOPIC,
        help="Mapper output topic checked before experimental shadow motion.",
    )
    parser.add_argument(
        "--mapper-topic-timeout-sec",
        default=DEFAULT_MAPPER_TOPIC_TIMEOUT_SEC,
        type=float,
        help="Seconds to wait for the mapper topic before warning or failing.",
    )
    parser.add_argument(
        "--require-mapper-topic",
        action="store_true",
        help="Fail before motion if --mapper-topic is unavailable.",
    )
    parser.add_argument(
        "--shadow-diagnostics-json",
        type=Path,
        help="Path for experimental shadow diagnostics JSON.",
    )
    parser.add_argument(
        "--no-shadow-diagnostics",
        action="store_true",
        help="Do not write experimental shadow diagnostics JSON.",
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
    node.configure_shadow_collection(args)
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
            "Starting arena coverage mode: " + args.coverage_mode
        )
        if args.coverage_mode == "shadow":
            summary = node.execute_shadow_coverage(args, route)
            notes = f"{args.notes};{shadow_notes_summary(node.shadow_diagnostics)}"
            node.get_logger().info(
                "Completed experimental shadow coverage: "
                + json.dumps(summary.to_dict(), sort_keys=True)
            )
        else:
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
        if args.coverage_mode == "shadow":
            node.shadow_diagnostics["exception"] = "keyboard_interrupt"
            node.shadow_diagnostics.setdefault(
                "summary",
                ShadowCoverageSummary(
                    stop_reason="keyboard_interrupt",
                    final_phase="failed",
                ).to_dict(),
            )
        print("Interrupted. Sending stop command...")
        return_code = 130

    except Exception as exc:
        status = "failed"
        notes = f"{args.notes};{exc}"
        if args.coverage_mode == "shadow":
            node.shadow_diagnostics["exception"] = str(exc)
            node.shadow_diagnostics.setdefault(
                "summary",
                ShadowCoverageSummary(
                    stop_reason=str(exc),
                    final_phase="failed",
                ).to_dict(),
            )
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

            if args.coverage_mode == "shadow":
                try:
                    write_shadow_diagnostics(
                        shadow_diagnostics_path_for_args(args),
                        node.shadow_diagnostics,
                    )
                    if not args.no_shadow_diagnostics:
                        node.get_logger().info(
                            "Saved shadow diagnostics to "
                            f"{shadow_diagnostics_path_for_args(args)}"
                        )
                except Exception as diag_exc:
                    print(f"Could not write shadow diagnostics: {diag_exc}", file=sys.stderr)

            node.destroy_node()
            rclpy.shutdown()

    return return_code


if __name__ == "__main__":
    sys.exit(main())
