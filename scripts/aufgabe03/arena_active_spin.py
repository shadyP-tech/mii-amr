#!/usr/bin/env python3
"""
Spin-only active arena localization helper.

This module owns the experimental live spin, scan/odom pairing, safety checks,
diagnostics, and call into the offline arena geometry localizer. It deliberately
does not publish /initialpose or interact with Nav2.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from arena_geometry_localizer import (
    ArenaGeometryConfig,
    Pose2D,
    ScanSample,
    analyze_scan_samples,
)


DEFAULT_STOP_COUNT = 10
DEFAULT_STOP_HZ = 10.0


@dataclass(frozen=True)
class PosePrior:
    x_m: float
    y_m: float
    yaw_rad: float
    covariance: list[float]


@dataclass
class ArenaActiveSpinResult:
    success: bool
    failure_reason: str | None
    pose_prior: PosePrior | None
    diagnostics: dict
    diagnostics_path: str | None = None


@dataclass(frozen=True)
class SectorClearance:
    ok: bool
    reason: str
    front_min_m: float | None
    left_min_m: float | None
    right_min_m: float | None
    rear_min_m: float | None


@dataclass(frozen=True)
class ArenaActiveSpinConfig:
    run_id: str
    diagnostics_path: Path
    cmd_vel_topic: str = "/cmd_vel"
    scan_topic: str = "/scan"
    odom_topic: str = "/odom"
    spin_direction: str = "ccw"
    angular_speed_rad_s: float = 0.25
    max_spin_sec: float = 30.0
    spin_complete_tolerance_deg: float = 5.0
    min_angular_progress_rad_s: float = 0.05
    progress_check_sec: float = 2.0
    min_scan_samples: int = 20
    max_odom_scan_age_sec: float = 0.20
    stop_settle_sec: float = 0.5
    min_front_clearance_m: float = 0.35
    min_side_clearance_m: float = 0.20
    min_rear_clearance_m: float = 0.20
    require_operator_confirmation: bool = True
    allow_extra_cmd_vel_publishers: bool = False
    on_failure: str = "abort"
    dry_run: bool = False
    range_stride: int = 6
    max_points: int = 3000
    control_rate_hz: float = 10.0
    arena_config: ArenaGeometryConfig = field(default_factory=ArenaGeometryConfig)


def normalize_angle_rad(angle_rad):
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def shortest_angle_delta_rad(start_rad, end_rad):
    return normalize_angle_rad(end_rad - start_rad)


def yaw_from_quaternion(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def odom_pose_from_msg(msg):
    pose = msg.pose.pose
    orientation = pose.orientation
    return Pose2D(
        x=float(pose.position.x),
        y=float(pose.position.y),
        yaw_deg=math.degrees(
            yaw_from_quaternion(
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            )
        ),
    )


def scan_sample_from_msg(msg, odom_pose):
    return ScanSample(
        ranges=list(msg.ranges),
        angle_min=float(msg.angle_min),
        angle_increment=float(msg.angle_increment),
        range_min=float(msg.range_min),
        range_max=float(msg.range_max),
        odom_pose=odom_pose,
    )


def valid_range(value, range_min, range_max):
    return (
        value is not None
        and math.isfinite(value)
        and value >= range_min
        and value <= range_max
    )


def angle_in_sector(angle_deg, ranges):
    return any(lower <= angle_deg <= upper for lower, upper in ranges)


def min_sector_range(scan, sectors):
    values = []
    for index, raw_range in enumerate(scan.ranges):
        if not valid_range(raw_range, scan.range_min, scan.range_max):
            continue
        angle_rad = scan.angle_min + index * scan.angle_increment
        angle_deg = math.degrees(normalize_angle_rad(angle_rad))
        if angle_in_sector(angle_deg, sectors):
            values.append(float(raw_range))
    return min(values) if values else None


def evaluate_clearance(scan, config: ArenaActiveSpinConfig):
    front = min_sector_range(scan, [(-30.0, 30.0)])
    left = min_sector_range(scan, [(60.0, 120.0)])
    right = min_sector_range(scan, [(-120.0, -60.0)])
    rear = min_sector_range(scan, [(150.0, 180.0), (-180.0, -150.0)])
    checks = [
        ("front_clearance_missing", "front_clearance_below_limit", front, config.min_front_clearance_m),
        ("left_clearance_missing", "left_clearance_below_limit", left, config.min_side_clearance_m),
        ("right_clearance_missing", "right_clearance_below_limit", right, config.min_side_clearance_m),
        ("rear_clearance_missing", "rear_clearance_below_limit", rear, config.min_rear_clearance_m),
    ]
    for missing_reason, low_reason, value, limit in checks:
        if value is None:
            return SectorClearance(False, missing_reason, front, left, right, rear)
        if value < limit:
            return SectorClearance(False, low_reason, front, left, right, rear)
    return SectorClearance(True, "ok", front, left, right, rear)


def covariance_list_from_localizer(covariance):
    values = [0.0] * 36
    values[0] = float(covariance["x_m2"])
    values[7] = float(covariance["y_m2"])
    values[35] = float(covariance["yaw_rad2"])
    return values


def pose_prior_from_localizer_result(result):
    pose = result.estimated_pose_prior
    covariance = result.estimated_covariance
    if pose is None or covariance is None:
        return None
    return PosePrior(
        x_m=float(pose.x),
        y_m=float(pose.y),
        yaw_rad=math.radians(float(pose.yaw_deg)),
        covariance=covariance_list_from_localizer(covariance),
    )


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    return value


def write_diagnostics_json(path: Path | str | None, diagnostics):
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(json_safe(diagnostics), file, indent=2, sort_keys=True)
        file.write("\n")
    return str(path)


def config_diagnostics(config: ArenaActiveSpinConfig):
    data = asdict(config)
    data["diagnostics_path"] = str(config.diagnostics_path)
    data["arena_config"] = asdict(config.arena_config)
    return data


def initial_diagnostics(config: ArenaActiveSpinConfig):
    return {
        "mode": "arena-active",
        "success": False,
        "failure_reason": "",
        "fallback_used": False,
        "config": config_diagnostics(config),
        "spin": {
            "target_rad": 2.0 * math.pi,
            "accumulated_rad": 0.0,
            "duration_sec": 0.0,
            "timeout": False,
        },
        "samples": {
            "scan_samples_collected": 0,
            "scan_samples_used": 0,
            "rejected_scan_samples": 0,
        },
        "safety": {
            "min_front_range_m": None,
            "min_left_range_m": None,
            "min_right_range_m": None,
            "min_rear_range_m": None,
        },
        "cmd_vel_publishers": {
            "count": None,
            "unexpected_count": None,
            "allowed": config.allow_extra_cmd_vel_publishers,
        },
        "localizer_result": None,
        "exception": None,
        "initialpose": {
            "published": False,
            "reason": "not_reached",
        },
    }


def update_safety_minima(diagnostics, clearance: SectorClearance):
    safety = diagnostics["safety"]
    for key, value in [
        ("min_front_range_m", clearance.front_min_m),
        ("min_left_range_m", clearance.left_min_m),
        ("min_right_range_m", clearance.right_min_m),
        ("min_rear_range_m", clearance.rear_min_m),
    ]:
        if value is None:
            continue
        current = safety.get(key)
        safety[key] = value if current is None else min(current, value)


def stop_repeatedly(
    publisher,
    twist_factory: Callable[[], object],
    sleep_fn: Callable[[float], None] = time.sleep,
    count=DEFAULT_STOP_COUNT,
    hz=DEFAULT_STOP_HZ,
):
    delay = 1.0 / hz
    for _ in range(count):
        publisher.publish(twist_factory())
        sleep_fn(delay)


class ArenaActiveSpinSession:
    def __init__(
        self,
        node,
        config: ArenaActiveSpinConfig,
        rclpy_module,
        twist_factory,
        scan_msg_type,
        odom_msg_type,
        qos_profile,
        input_fn=input,
        time_fn=time.time,
        sleep_fn=time.sleep,
        analyze_fn=analyze_scan_samples,
    ):
        self.node = node
        self.config = config
        self.rclpy = rclpy_module
        self.twist_factory = twist_factory
        self.input_fn = input_fn
        self.time_fn = time_fn
        self.sleep_fn = sleep_fn
        self.analyze_fn = analyze_fn
        self.latest_scan = None
        self.latest_scan_received_sec = None
        self.latest_odom_pose = None
        self.latest_odom_yaw_rad = None
        self.latest_odom_received_sec = None
        self.collecting = False
        self.samples = []
        self.rejected_samples = 0
        self.diagnostics = initial_diagnostics(config)
        self.scan_subscription = node.create_subscription(
            scan_msg_type,
            config.scan_topic,
            self.scan_callback,
            qos_profile,
        )
        self.odom_subscription = node.create_subscription(
            odom_msg_type,
            config.odom_topic,
            self.odom_callback,
            10,
        )

    def now(self):
        return self.time_fn()

    def scan_callback(self, msg):
        received_sec = self.now()
        self.latest_scan = msg
        self.latest_scan_received_sec = received_sec
        if not self.collecting:
            return
        if self.latest_odom_pose is None or self.latest_odom_received_sec is None:
            self.rejected_samples += 1
            return
        if received_sec - self.latest_odom_received_sec > self.config.max_odom_scan_age_sec:
            self.rejected_samples += 1
            return
        self.samples.append(scan_sample_from_msg(msg, self.latest_odom_pose))

    def odom_callback(self, msg):
        self.latest_odom_pose = odom_pose_from_msg(msg)
        self.latest_odom_yaw_rad = math.radians(self.latest_odom_pose.yaw_deg)
        self.latest_odom_received_sec = self.now()

    def fresh_scan_age_sec(self):
        if self.latest_scan_received_sec is None:
            return None
        return self.now() - self.latest_scan_received_sec

    def fresh_odom_age_sec(self):
        if self.latest_odom_received_sec is None:
            return None
        return self.now() - self.latest_odom_received_sec

    def wait_for_fresh_inputs(self):
        deadline = self.now() + min(5.0, self.config.max_spin_sec)
        while self.rclpy.ok() and self.now() <= deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
            scan_age = self.fresh_scan_age_sec()
            odom_age = self.fresh_odom_age_sec()
            if (
                self.latest_scan is not None
                and scan_age is not None
                and scan_age <= self.config.max_odom_scan_age_sec
                and self.latest_odom_pose is not None
                and odom_age is not None
                and odom_age <= self.config.max_odom_scan_age_sec
            ):
                return
        raise RuntimeError("fresh_scan_or_odom_unavailable")

    def refresh_fresh_inputs_after_prompt(self):
        deadline = self.now() + min(2.0, self.config.max_spin_sec)
        while self.rclpy.ok() and self.now() <= deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
            scan_age = self.fresh_scan_age_sec()
            odom_age = self.fresh_odom_age_sec()
            if (
                self.latest_scan is not None
                and scan_age is not None
                and scan_age <= self.config.max_odom_scan_age_sec
                and self.latest_odom_pose is not None
                and odom_age is not None
                and odom_age <= self.config.max_odom_scan_age_sec
            ):
                return
        raise RuntimeError("fresh_scan_or_odom_unavailable_after_prompt")

    def cmd_vel_publisher_check(self):
        count = None
        if hasattr(self.node, "count_publishers"):
            count = self.node.count_publishers(self.config.cmd_vel_topic)
        unexpected = None if count is None else max(0, int(count) - 1)
        self.diagnostics["cmd_vel_publishers"] = {
            "count": count,
            "unexpected_count": unexpected,
            "allowed": self.config.allow_extra_cmd_vel_publishers,
        }
        if (
            unexpected is not None
            and unexpected > 0
            and not self.config.allow_extra_cmd_vel_publishers
        ):
            raise RuntimeError("unexpected_cmd_vel_publishers")

    def print_operator_prompt(self):
        scan_age = self.fresh_scan_age_sec()
        odom_age = self.fresh_odom_age_sec()
        clearance = evaluate_clearance(self.latest_scan, self.config)
        update_safety_minima(self.diagnostics, clearance)
        print("\nArena-active spin-only startup")
        print(f"  angular speed: {self.config.angular_speed_rad_s:.3f} rad/s")
        print(f"  direction: {self.config.spin_direction}")
        print(f"  max spin time: {self.config.max_spin_sec:.1f} s")
        print(f"  front clearance: {clearance.front_min_m}")
        print(f"  left clearance: {clearance.left_min_m}")
        print(f"  right clearance: {clearance.right_min_m}")
        print(f"  rear clearance: {clearance.rear_min_m}")
        print(f"  latest scan age: {scan_age}")
        print(f"  latest odom age: {odom_age}")
        print(f"  cmd_vel publisher check: {self.diagnostics['cmd_vel_publishers']}")
        print("  expected action: rotate in place 360 degrees")
        if not clearance.ok:
            raise RuntimeError(f"scan_clearance_failed:{clearance.reason}")
        if self.config.require_operator_confirmation:
            self.input_fn("Press Enter to start arena-active spin, or Ctrl+C to abort: ")

    def publish_spin_command(self, publisher):
        command = self.twist_factory()
        sign = 1.0 if self.config.spin_direction == "ccw" else -1.0
        command.angular.z = sign * abs(self.config.angular_speed_rad_s)
        publisher.publish(command)

    def run_spin(self, publisher):
        self.wait_for_fresh_inputs()
        self.cmd_vel_publisher_check()
        self.print_operator_prompt()
        self.refresh_fresh_inputs_after_prompt()

        previous_yaw = self.latest_odom_yaw_rad
        if previous_yaw is None:
            raise RuntimeError("fresh_odom_unavailable")
        self.collecting = True
        accumulated = 0.0
        target = 2.0 * math.pi - math.radians(self.config.spin_complete_tolerance_deg)
        period = 1.0 / self.config.control_rate_hz
        start = self.now()
        last_progress_time = start
        last_progress_yaw = 0.0

        while self.rclpy.ok():
            if self.now() - start > self.config.max_spin_sec:
                self.diagnostics["spin"]["timeout"] = True
                raise RuntimeError("arena_active_spin_timeout")
            self.publish_spin_command(publisher)
            self.rclpy.spin_once(self.node, timeout_sec=period)
            now = self.now()
            scan_age = self.fresh_scan_age_sec()
            odom_age = self.fresh_odom_age_sec()
            if scan_age is None or scan_age > self.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_scan_during_spin")
            if odom_age is None or odom_age > self.config.max_odom_scan_age_sec:
                raise RuntimeError("stale_odom_during_spin")

            clearance = evaluate_clearance(self.latest_scan, self.config)
            update_safety_minima(self.diagnostics, clearance)
            if not clearance.ok:
                raise RuntimeError(f"scan_clearance_failed:{clearance.reason}")

            current_yaw = self.latest_odom_yaw_rad
            delta = shortest_angle_delta_rad(previous_yaw, current_yaw)
            accumulated += delta
            previous_yaw = current_yaw
            self.diagnostics["spin"]["accumulated_rad"] = accumulated
            self.diagnostics["spin"]["duration_sec"] = now - start
            if abs(accumulated) >= target:
                return accumulated, now - start

            if now - last_progress_time >= self.config.progress_check_sec:
                progress_rate = abs(accumulated - last_progress_yaw) / (now - last_progress_time)
                if progress_rate < self.config.min_angular_progress_rad_s:
                    raise RuntimeError("insufficient_angular_progress")
                last_progress_time = now
                last_progress_yaw = accumulated

        raise RuntimeError("ros_shutdown_during_arena_active_spin")

    def analyze(self):
        if len(self.samples) < self.config.min_scan_samples:
            raise RuntimeError(
                "insufficient_scan_samples:"
                f"{len(self.samples)}<{self.config.min_scan_samples}"
            )
        result = self.analyze_fn(
            self.samples,
            self.config.arena_config,
            range_stride=self.config.range_stride,
            max_points=self.config.max_points,
        )
        self.diagnostics["localizer_result"] = result.to_dict()
        if not result.success:
            raise RuntimeError(f"arena_localizer_failed:{result.failure_reason}")
        pose_prior = pose_prior_from_localizer_result(result)
        if pose_prior is None:
            raise RuntimeError("arena_localizer_missing_pose_prior")
        return pose_prior

    def finish_failure(self, reason, exception=None):
        self.diagnostics["success"] = False
        self.diagnostics["failure_reason"] = reason
        if exception is not None:
            self.diagnostics["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
            }
        self.diagnostics["samples"]["scan_samples_collected"] = len(self.samples)
        self.diagnostics["samples"]["scan_samples_used"] = len(self.samples)
        self.diagnostics["samples"]["rejected_scan_samples"] = self.rejected_samples
        path = write_diagnostics_json(self.config.diagnostics_path, self.diagnostics)
        return ArenaActiveSpinResult(False, reason, None, self.diagnostics, path)

    def finish_success(self, pose_prior):
        self.diagnostics["success"] = True
        self.diagnostics["failure_reason"] = ""
        self.diagnostics["samples"]["scan_samples_collected"] = len(self.samples)
        self.diagnostics["samples"]["scan_samples_used"] = len(self.samples)
        self.diagnostics["samples"]["rejected_scan_samples"] = self.rejected_samples
        self.diagnostics["initialpose"] = {
            "published": False,
            "reason": "dry_run" if self.config.dry_run else "pending_runner_publication",
        }
        path = write_diagnostics_json(self.config.diagnostics_path, self.diagnostics)
        return ArenaActiveSpinResult(True, None, pose_prior, self.diagnostics, path)

    def run(self, publisher):
        try:
            stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
            self.run_spin(publisher)
            self.collecting = False
            stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
            self.sleep_fn(self.config.stop_settle_sec)
            pose_prior = self.analyze()
            return self.finish_success(pose_prior)
        except KeyboardInterrupt:
            self.collecting = False
            stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
            return self.finish_failure("keyboard_interrupt")
        except Exception as exc:
            self.collecting = False
            stop_repeatedly(publisher, self.twist_factory, self.sleep_fn)
            return self.finish_failure(str(exc), exception=exc)


def run_arena_active_spin(
    node,
    publisher,
    config: ArenaActiveSpinConfig,
    rclpy_module,
    twist_factory,
    scan_msg_type,
    odom_msg_type,
    qos_profile,
    input_fn=input,
    time_fn=time.time,
    sleep_fn=time.sleep,
    analyze_fn=analyze_scan_samples,
):
    session = ArenaActiveSpinSession(
        node,
        config,
        rclpy_module,
        twist_factory,
        scan_msg_type,
        odom_msg_type,
        qos_profile,
        input_fn=input_fn,
        time_fn=time_fn,
        sleep_fn=sleep_fn,
        analyze_fn=analyze_fn,
    )
    return session.run(publisher)
