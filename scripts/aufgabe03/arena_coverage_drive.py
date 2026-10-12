#!/usr/bin/env python3
"""Explore obstacle shadows using fresh LiDAR and bounded odometry-frame curves.

Start TurtleBot bringup and a mapper separately. Each move is planned from a
rolling scan map, checked against live LiDAR, and stopped before replanning.
Completion requires a fresh verification spin with no remaining shadow frontier.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
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
    ShadowCoverageConfig,
    ShadowCoverageSummary,
    ShadowScanSample,
    heading_error_to_point_deg,
    plan_shadow_coverage_move,
    prune_shadow_samples,
)

COMMAND_PERIOD_SEC = 0.05
TOPIC_TIMEOUT_SEC = 5.0
STOP_PUBLISH_COUNT = 10
STOP_PUBLISH_HZ = 10.0


@dataclass(frozen=True)
class MotionConfig:
    angular_speed: float = 0.25
    rotation_tolerance_deg: float = 2.0
    min_angular_progress_rad_s: float = 0.05
    progress_check_sec: float = 2.0
    settle_sec: float = 0.5
    min_scan_range_m: float = 0.28
    hard_stop_range_m: float = 0.18
    scan_half_angle_deg: float = 35.0
    max_action_time_sec: float = 90.0
    max_mission_time_sec: float = 600.0
    shadow_emergency_stop_distance_m: float = 0.18
    shadow_side_stop_distance_m: float = 0.16
    shadow_curve_goal_tolerance_m: float = 0.04
    shadow_curve_linear_speed_mps: float = 0.05
    shadow_curve_max_angular_radps: float = 0.40
    shadow_max_odom_scan_age_sec: float = 0.50
    min_scan_samples: int = 20
    mapper_topic: str = "/map"
    mapper_topic_timeout_sec: float = 5.0
    require_mapper_topic: bool = False


# Keep CLI names explicit where they differ from planner configuration names.
SHADOW_OPTION_NAMES = {
    "min_visible_shadow_cells": "shadow_min_visible_cells",
    "min_endpoint_clearance_m": "shadow_side_stop_distance_m",
}


def shadow_config_from_args(args):
    values = {
        field.name: getattr(args, SHADOW_OPTION_NAMES.get(field.name, "shadow_" + field.name))
        for field in fields(ShadowCoverageConfig)
    }
    return ShadowCoverageConfig(**values)


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


def validate_motion_config(args):
    """Reject invalid budgets and thresholds before creating a ROS publisher."""
    motion = MotionConfig()
    shadow = shadow_config_from_args(args)
    nonnegative = {"settle_sec", "shadow_soft_clearance_weight", "shadow_recent_target_radius_m"}
    numeric = {field.name for field in fields(motion) if type(getattr(motion, field.name)) in (int, float)}
    numeric.update(
        SHADOW_OPTION_NAMES.get(field.name, "shadow_" + field.name)
        for field in fields(shadow)
        if type(getattr(shadow, field.name)) in (int, float)
    )
    for name in sorted(numeric):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (value == 0 and name not in nonnegative):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and {'>= 0' if name in nonnegative else '> 0'}")
    limits = {
        "angular_speed": 0.50,
        "shadow_curve_linear_speed_mps": 0.10,
        "shadow_curve_max_angular_radps": 0.50,
        "shadow_max_initial_heading_error_deg": 80.0,
        "rotation_tolerance_deg": 15.0,
        "scan_half_angle_deg": 90.0,
        "shadow_max_odom_scan_age_sec": 0.50,
    }
    for name, upper in limits.items():
        if getattr(args, name) > upper:
            raise ValueError(f"--{name.replace('_', '-')} must be <= {upper}")
    for name, lower in {
        "hard_stop_range_m": 0.18,
        "shadow_emergency_stop_distance_m": 0.18,
        "shadow_side_stop_distance_m": 0.16,
        "shadow_inflation_radius_m": 0.105,
    }.items():
        if getattr(args, name) < lower:
            raise ValueError(f"--{name.replace('_', '-')} must be >= {lower}")
    if args.min_scan_range_m < max(args.hard_stop_range_m, args.shadow_emergency_stop_distance_m):
        raise ValueError("--min-scan-range-m must cover both emergency stop thresholds")
    if shadow.grid_resolution_m >= shadow.grid_size_m:
        raise ValueError("shadow grid resolution must be smaller than grid size")
    if shadow.max_total_distance_m < shadow.max_single_move_m:
        raise ValueError("shadow total distance must be at least the single-move limit")
    if shadow.max_single_move_m < shadow.min_move_length_m:
        raise ValueError("shadow single-move limit must be at least the minimum move length")
    if args.shadow_curve_goal_tolerance_m >= shadow.min_move_length_m:
        raise ValueError("curve goal tolerance must be smaller than the minimum move length")
    if not shadow.unknown_blocked:
        raise ValueError("coverage requires unknown cells to remain blocked")
    if not args.mapper_topic.startswith("/"):
        raise ValueError("--mapper-topic must be an absolute ROS topic")


def terminal_shadow_plan(plan):
    """Unreachable, filtered, or unobserved shadows never establish completion."""
    return (
        not plan.ok
        and plan.reason == "no_shadow_frontier"
        and plan.grid is not None
        and plan.shadow_status.get("raw_shadow_unknown_cell_count") == 0
        and plan.shadow_status.get("moving_frontier_count") == 0
    )


class ArenaCoverageDrive(Node):
    def __init__(self):
        if rclpy is None:
            raise RuntimeError("Source ROS 2 Humble before running arena coverage.")
        super().__init__("arena_coverage_drive")
        self.last_scan = None
        self.last_scan_received_sec = None
        self.latest_odom_pose = None
        self.latest_odom_received_sec = None
        self.shadow_collecting = False
        self.shadow_segment_index = 0
        self.shadow_samples = []
        self.shadow_sample_rejections = {"missing_odom": 0, "stale_odom": 0}
        self.shadow_diagnostics = {}
        self.mission_deadline = None
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, qos_profile_sensor_data,
        )

    def now(self):
        return time.monotonic()

    def configure_shadow_collection(self, args):
        self.shadow_collecting = True
        self.shadow_max_samples = int(args.shadow_max_samples)
        self.shadow_max_odom_scan_age_sec = args.shadow_max_odom_scan_age_sec
        self.shadow_segment_index = 0
        self.shadow_samples = []

    def begin_shadow_segment(self):
        self.shadow_segment_index += 1

    def odom_callback(self, msg):
        self.latest_odom_pose = odom_pose_from_msg(msg)
        self.latest_odom_received_sec = self.now()

    def scan_callback(self, msg):
        received_sec = self.now()
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
        self.shadow_samples.append(ShadowScanSample.from_scan_sample(
            scan_sample_from_msg(msg, self.latest_odom_pose),
            stamp_sec=received_sec, segment_index=self.shadow_segment_index,
        ))
        if len(self.shadow_samples) > self.shadow_max_samples:
            del self.shadow_samples[:-self.shadow_max_samples]

    def fresh_scan_age_sec(self):
        return None if self.last_scan_received_sec is None else self.now() - self.last_scan_received_sec

    def fresh_odom_age_sec(self):
        return None if self.latest_odom_received_sec is None else self.now() - self.latest_odom_received_sec

    def check_deadline(self):
        if self.mission_deadline is not None and self.now() >= self.mission_deadline:
            raise RuntimeError("shadow_mission_timeout")
        if not rclpy.ok():
            raise RuntimeError("ros_shutdown_during_shadow_coverage")

    def check_fresh_inputs(self, args):
        self.check_deadline()
        for name, age in (("scan", self.fresh_scan_age_sec()), ("odom", self.fresh_odom_age_sec())):
            if age is None or age < 0 or age > args.shadow_max_odom_scan_age_sec:
                raise RuntimeError(f"stale_{name}_during_shadow_motion")
        if self.last_scan is None or self.latest_odom_pose is None:
            raise RuntimeError("shadow_sensor_data_missing")
        pose = self.latest_odom_pose
        if not all(math.isfinite(value) for value in (pose.x, pose.y, pose.yaw_deg)):
            raise RuntimeError("invalid_shadow_odom_pose")

    def wait_for_fresh_shadow_inputs(self, args, timeout_sec=TOPIC_TIMEOUT_SEC):
        deadline = self.now() + timeout_sec
        while self.now() < deadline:
            self.check_deadline()
            rclpy.spin_once(self, timeout_sec=COMMAND_PERIOD_SEC)
            scan_age, odom_age = self.fresh_scan_age_sec(), self.fresh_odom_age_sec()
            if all(age is not None and 0 <= age <= args.shadow_max_odom_scan_age_sec for age in (scan_age, odom_age)):
                self.check_fresh_inputs(args)
                return
        raise RuntimeError("timed_out_waiting_for_fresh_shadow_inputs")

    def wait_for_mapper_topic(self, args):
        deadline = self.now() + args.mapper_topic_timeout_sec
        while self.now() < deadline:
            self.check_deadline()
            if args.mapper_topic in {name for name, _ in self.get_topic_names_and_types()}:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def publish_velocity(self, linear_x, angular_z):
        command = Twist()
        command.linear.x = linear_x
        command.angular.z = angular_z
        self.pub.publish(command)

    def stop_repeatedly(self):
        for _ in range(STOP_PUBLISH_COUNT):
            if rclpy.ok():
                self.publish_velocity(0.0, 0.0)
                rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(1.0 / STOP_PUBLISH_HZ)

    def rotate(self, angle_deg, args):
        """Collect a full scan, failing on stale inputs, obstruction or no progress."""
        start = self.now()
        try:
            self.wait_for_fresh_shadow_inputs(args)
            previous_yaw = self.latest_odom_pose.yaw_deg
            accumulated = 0.0
            last_progress, last_progress_time = 0.0, self.now()
            direction = 1.0 if angle_deg >= 0 else -1.0
            while self.now() - start <= args.max_action_time_sec:
                rclpy.spin_once(self, timeout_sec=COMMAND_PERIOD_SEC)
                self.check_fresh_inputs(args)
                scan = self.last_scan
                safety = evaluate_scan_safety(
                    scan.ranges, scan.angle_min, scan.angle_increment,
                    scan.range_min, scan.range_max, "rotate", args.scan_half_angle_deg,
                    args.min_scan_range_m, args.hard_stop_range_m,
                )
                if not safety.safe:
                    raise RuntimeError(f"shadow_spin_clearance_failed:{safety.reason}")
                current_yaw = self.latest_odom_pose.yaw_deg
                accumulated += direction * shortest_angle_delta_deg(previous_yaw, current_yaw)
                previous_yaw = current_yaw
                if accumulated >= abs(angle_deg) - args.rotation_tolerance_deg:
                    if len(self.shadow_samples) < args.min_scan_samples:
                        raise RuntimeError("insufficient_shadow_scan_samples")
                    return
                elapsed = self.now() - last_progress_time
                if elapsed >= args.progress_check_sec:
                    progress_rate = math.radians(accumulated - last_progress) / elapsed
                    if progress_rate < args.min_angular_progress_rad_s:
                        raise RuntimeError("insufficient_angular_progress")
                    last_progress, last_progress_time = accumulated, self.now()
                self.publish_velocity(0.0, direction * args.angular_speed)
                time.sleep(COMMAND_PERIOD_SEC)
            raise RuntimeError("shadow_spin_timeout")
        finally:
            self.stop_repeatedly()

    def execute_shadow_curve(self, candidate, args, distance_limit_m):
        """Follow one bounded, heading-gated curve; every failure ends motion."""
        move_limit = min(args.shadow_max_single_move_m, distance_limit_m)
        if not math.isfinite(move_limit) or move_limit <= 0:
            raise RuntimeError("shadow_total_distance_exhausted")
        started = self.now()
        driven = 0.0
        record = {
            "target_x": float(candidate.target_x),
            "target_y": float(candidate.target_y),
            "driven_distance_m": 0.0,
            "duration_sec": 0.0,
            "stop_reason": "started",
        }
        self.shadow_diagnostics.setdefault("executions", []).append(record)
        try:
            self.wait_for_fresh_shadow_inputs(args)
            path = active_explore_curve_path(candidate, self.latest_odom_pose, move_limit)
            current = (self.latest_odom_pose.x, self.latest_odom_pose.y)
            initial_target = select_curve_lookahead_target(path, current, args.shadow_curve_lookahead_m)
            heading = heading_error_to_point_deg(self.latest_odom_pose, initial_target)
            if abs(heading) > args.shadow_max_initial_heading_error_deg:
                raise RuntimeError("initial_heading_error_too_large_execution")
            goal = candidate.path_world[-1] if candidate.path_world else (candidate.target_x, candidate.target_y)
            record["path_truncated"] = distance_2d(path[-1], goal) > args.shadow_curve_goal_tolerance_m
            deadline = started + min(
                args.max_action_time_sec,
                max(8.0, move_limit / args.shadow_curve_linear_speed_mps + 5.0),
            )
            previous = current
            while self.now() < deadline:
                rclpy.spin_once(self, timeout_sec=COMMAND_PERIOD_SEC)
                self.check_fresh_inputs(args)
                pose = self.latest_odom_pose
                current = (pose.x, pose.y)
                driven += distance_2d(previous, current)
                previous = current
                record["driven_distance_m"] = driven
                safety = evaluate_shadow_curve_safety(self.last_scan, args)
                if not safety.safe:
                    raise RuntimeError(f"shadow_curve_clearance_failed:{safety.reason}")
                if driven >= move_limit or distance_2d(current, path[-1]) <= args.shadow_curve_goal_tolerance_m:
                    record["stop_reason"] = "completed"
                    return record
                target = select_curve_lookahead_target(path, current, args.shadow_curve_lookahead_m)
                linear, angular, _ = pure_pursuit_curve_command(
                    pose, target, args.shadow_curve_lookahead_m,
                    args.shadow_curve_linear_speed_mps, args.shadow_curve_max_angular_radps,
                )
                remaining = max(0.0, move_limit - driven)
                linear = min(linear, remaining / COMMAND_PERIOD_SEC)
                self.publish_velocity(linear, angular)
                time.sleep(COMMAND_PERIOD_SEC)
            raise RuntimeError("shadow_curve_timeout")
        except BaseException as exc:
            record["stop_reason"] = "keyboard_interrupt" if isinstance(exc, KeyboardInterrupt) else str(exc)
            raise
        finally:
            record["duration_sec"] = self.now() - started
            self.stop_repeatedly()

    def _plan_shadow_move(self, config, recent_attempts):
        samples, _ = prune_shadow_samples(
            self.shadow_samples, config, now_sec=self.now(),
            current_segment=self.shadow_segment_index,
        )
        if len(samples) < 2:
            raise RuntimeError("insufficient_fresh_shadow_samples")
        plan = plan_shadow_coverage_move(samples, config, recent_attempts)
        self.check_deadline()
        self.shadow_diagnostics.setdefault("plans", []).append({
            "reason": plan.reason,
            "sample_window": plan.sample_window,
            "shadow_status": plan.shadow_status,
            "candidate_rejections": plan.candidate_rejections,
        })
        return plan

    def execute_shadow_coverage(self, args):
        config = shadow_config_from_args(args)
        summary = ShadowCoverageSummary(final_phase="seed_scan")
        self.mission_deadline = self.now() + args.max_mission_time_sec
        self.shadow_diagnostics = {
            "config": asdict(config), "motion": {f.name: getattr(args, f.name) for f in fields(MotionConfig)},
            "plans": [], "executions": [], "sample_rejections": self.shadow_sample_rejections,
        }
        recent_attempts = []
        confirmations = 0
        try:
            mapper_available = self.wait_for_mapper_topic(args)
            self.shadow_diagnostics["mapper_available"] = mapper_available
            if not mapper_available and args.require_mapper_topic:
                raise RuntimeError("mapper_topic_missing")
            if not mapper_available:
                self.get_logger().warn(f"Mapper topic {args.mapper_topic} not observed.")
            self.shadow_samples = []
            self.rotate(360.0, args)
            summary = replace(summary, spin_count=1)
            while True:
                self.wait_for_fresh_shadow_inputs(args)
                plan = self._plan_shadow_move(config, recent_attempts)
                if plan.ok:
                    confirmations = 0
                    if summary.attempts >= config.max_attempts:
                        raise RuntimeError("shadow_motion_attempts_exhausted")
                    remaining = config.max_total_distance_m - summary.total_distance_m
                    if remaining <= 0:
                        raise RuntimeError("shadow_total_distance_exhausted")
                    self.begin_shadow_segment()
                    summary = replace(summary, attempts=summary.attempts + 1, final_phase="shadow_mapping")
                    record = self.execute_shadow_curve(plan.selected, args, remaining)
                    summary = replace(
                        summary, moves_executed=summary.moves_executed + 1,
                        total_distance_m=summary.total_distance_m + record["driven_distance_m"],
                    )
                    if not record["path_truncated"]:
                        recent_attempts.append(record)
                    time.sleep(args.settle_sec)
                    continue
                if not terminal_shadow_plan(plan):
                    raise RuntimeError(f"shadow_plan_failed:{plan.reason}")
                confirmations += 1
                if confirmations < config.completion_confirmations:
                    continue
                # Rebuild evidence from the final spin; do not hide previously visited goals.
                summary = replace(summary, final_phase="final_verify_spin")
                self.begin_shadow_segment()
                self.shadow_samples = []
                self.rotate(360.0, args)
                summary = replace(summary, spin_count=summary.spin_count + 1)
                self.wait_for_fresh_shadow_inputs(args)
                verified = self._plan_shadow_move(config, ())
                if terminal_shadow_plan(verified):
                    summary = replace(summary, final_phase="complete", stop_reason="shadow_complete_verified")
                    return summary
                if not verified.ok:
                    raise RuntimeError(f"shadow_final_verify_failed:{verified.reason}")
                recent_attempts = []
                confirmations = 0
        except BaseException as exc:
            reason = "keyboard_interrupt" if isinstance(exc, KeyboardInterrupt) else str(exc)
            distance = sum(record["driven_distance_m"] for record in self.shadow_diagnostics["executions"])
            summary = replace(summary, total_distance_m=distance, final_phase="failed", stop_reason=reason)
            raise
        finally:
            self.shadow_diagnostics["summary"] = summary.to_dict()
            self.stop_repeatedly()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    motion = MotionConfig()
    for field in fields(motion):
        default = getattr(motion, field.name)
        option = "--" + field.name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(option, default=default, action=argparse.BooleanOptionalAction)
        else:
            parser.add_argument(option, default=default, type=type(default))
    shadow = ShadowCoverageConfig()
    for field in fields(shadow):
        name = SHADOW_OPTION_NAMES.get(field.name, "shadow_" + field.name)
        if name == "shadow_side_stop_distance_m":
            continue
        default = getattr(shadow, field.name)
        option = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(option, default=default, action=argparse.BooleanOptionalAction)
        else:
            parser.add_argument(option, default=default, type=type(default))
    parser.add_argument("--dry-run", action="store_true", help="Print configuration without publishing motion.")
    parser.add_argument("--yes", action="store_true", help="Start without the terminal confirmation prompt.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--shadow-diagnostics-json", type=Path)
    parser.add_argument("--no-shadow-diagnostics", action="store_true")
    args = parser.parse_args(argv)
    args.run_id = args.run_id or datetime.now().strftime("arena_coverage_%Y%m%d_%H%M%S")
    try:
        validate_motion_config(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def print_dry_run(args):
    print("Arena shadow coverage: seed spin -> bounded frontier curves -> verification spin")
    print(f"Motion budget: {args.shadow_max_attempts} moves, {args.shadow_max_total_distance_m:.2f} m, {args.max_mission_time_sec:.0f} s")
    print(f"Per move: {args.shadow_max_single_move_m:.2f} m at {args.shadow_curve_linear_speed_mps:.2f} m/s")
    print(f"Fresh scan/odom required within {args.shadow_max_odom_scan_age_sec:.2f} s")
    print("Requires TurtleBot bringup and external mapping; blocked or residual shadows fail closed.")


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        print_dry_run(args)
        return 0
    if rclpy is None:
        print("Source ROS 2 Humble before running arena coverage.", file=sys.stderr)
        return 2
    if not args.yes:
        print_dry_run(args)
        try:
            if input("Start robot motion? Type yes: ").strip().lower() != "yes":
                return 130
        except (KeyboardInterrupt, EOFError):
            return 130
    rclpy.init()
    node = None
    try:
        node = ArenaCoverageDrive()
        node.configure_shadow_collection(args)
        summary = node.execute_shadow_coverage(args)
        print(json.dumps(summary.to_dict(), sort_keys=True))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Arena coverage stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        if node is not None:
            try:
                node.stop_repeatedly()
            finally:
                if not args.no_shadow_diagnostics:
                    path = args.shadow_diagnostics_json or Path("results/aufgabe03") / f"{args.run_id}_shadow_coverage.json"
                    try:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(node.shadow_diagnostics, indent=2, sort_keys=True) + "\n")
                    except OSError as exc:
                        print(f"Could not save diagnostics: {exc}", file=sys.stderr)
                node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
