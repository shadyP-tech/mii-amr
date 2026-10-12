#!/usr/bin/env python3

import math
import os
import re
import sys
import time
from datetime import datetime

if __package__:
    from .motion import (
        DriveNode, FORWARD_CSV_HEADER as CSV_HEADER, append_csv_row, close_motion_node,
        ensure_csv_schema, odom_to_xy_yaw, parse_float_env, pose_fields,
        rclpy, shortest_angle_delta_deg,
    )
else:
    from motion import (
        DriveNode, FORWARD_CSV_HEADER as CSV_HEADER, append_csv_row, close_motion_node,
        ensure_csv_schema, odom_to_xy_yaw, parse_float_env, pose_fields,
        rclpy, shortest_angle_delta_deg,
    )



DEFAULT_RUN_MODE = "linear-forward"
DEFAULT_LINEAR_SPEED_MPS = 0.10
DEFAULT_RUN_DISTANCE = "30cm"
ANGULAR_SPEED_RADPS = 0.0
RESULTS_CSV = "results/aufgabe02/scripted_drive_runs.csv"
DEFAULT_DRIVE_TIMEOUT_SEC = 120.0
DEFAULT_SIM_START_X = 0.5
DEFAULT_SIM_START_Y = 0.05
DEFAULT_SIM_START_YAW_DEG = 180.0
DEFAULT_START_POSITION_TOL_M = 0.02
DEFAULT_START_YAW_TOL_DEG = 3.0
DEFAULT_DISTANCE_TOL_M = 0.03
DEFAULT_LATERAL_TOL_M = 0.03
DEFAULT_YAW_DRIFT_TOL_DEG = 4.0

DISTANCE_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*"
    r"(m|meter|meters|cm|centimeter|centimeters|mm|millimeter|millimeters)?\s*$",
    re.IGNORECASE,
)


def parse_distance_m(value):
    match = DISTANCE_RE.match(str(value))
    if not match:
        raise ValueError(f"invalid distance value: {value!r}")

    amount = float(match.group(1))
    if not math.isfinite(amount):
        raise ValueError("RUN_DISTANCE must be finite")
    unit = (match.group(2) or "m").lower()

    if unit in {"m", "meter", "meters"}:
        return amount
    if unit in {"cm", "centimeter", "centimeters"}:
        return amount / 100.0
    if unit in {"mm", "millimeter", "millimeters"}:
        return amount / 1000.0

    raise ValueError(f"unsupported distance unit in {value!r}")


def configured_motion(env=os.environ):
    run_mode = env.get("RUN_MODE", DEFAULT_RUN_MODE)
    if run_mode != "linear-forward":
        raise ValueError(f"unsupported RUN_MODE={run_mode!r}; expected linear-forward")

    speed_mps = parse_float_env(env, "RUN_SPEED", DEFAULT_LINEAR_SPEED_MPS)
    distance_m = parse_distance_m(env.get("RUN_DISTANCE", DEFAULT_RUN_DISTANCE))

    if speed_mps <= 0.0:
        raise ValueError("RUN_SPEED must be greater than zero")
    if distance_m <= 0.0:
        raise ValueError("RUN_DISTANCE must be greater than zero")

    nominal_duration_sec = distance_m / speed_mps
    if not math.isfinite(nominal_duration_sec):
        raise ValueError("RUN_DISTANCE / RUN_SPEED must be finite")
    timeout_sec = parse_float_env(env, "RUN_TIMEOUT_SEC", DEFAULT_DRIVE_TIMEOUT_SEC)
    if timeout_sec <= 0.0:
        raise ValueError("RUN_TIMEOUT_SEC must be greater than zero")

    return {
        "run_mode": run_mode,
        "speed_mps": speed_mps,
        "distance_m": distance_m,
        "duration_sec": nominal_duration_sec,
        "timeout_sec": timeout_sec,
        "angular_speed_radps": ANGULAR_SPEED_RADPS,
    }


def validation_config(env=os.environ):
    defaults = {
        "start_x": DEFAULT_SIM_START_X,
        "start_y": DEFAULT_SIM_START_Y,
        "start_yaw_deg": DEFAULT_SIM_START_YAW_DEG,
        "start_position_tol_m": DEFAULT_START_POSITION_TOL_M,
        "start_yaw_tol_deg": DEFAULT_START_YAW_TOL_DEG,
        "distance_tol_m": DEFAULT_DISTANCE_TOL_M,
        "lateral_tol_m": DEFAULT_LATERAL_TOL_M,
        "yaw_drift_tol_deg": DEFAULT_YAW_DRIFT_TOL_DEG,
    }
    config = {name: parse_float_env(env, "SIM_" + name.upper(), value)
              for name, value in defaults.items()}
    for name, value in config.items():
        if "_tol_" in name and value < 0.0:
            raise ValueError(f"{name} must be non-negative")
    config["validate_start_pose"] = env.get("SIM_VALIDATE_START_POSE", "1").lower() not in {"0", "false", "no"}
    return config


def xy_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def motion_summary(start_pose, final_pose):
    dx = final_pose["x"] - start_pose["x"]
    dy = final_pose["y"] - start_pose["y"]
    start_yaw_rad = math.radians(start_pose["yaw_deg"])

    return {
        "dx_m": dx,
        "dy_m": dy,
        "distance_m": math.hypot(dx, dy),
        "forward_m": dx * math.cos(start_yaw_rad) + dy * math.sin(start_yaw_rad),
        "lateral_m": -dx * math.sin(start_yaw_rad) + dy * math.cos(start_yaw_rad),
        "yaw_change_deg": shortest_angle_delta_deg(
            start_pose["yaw_deg"],
            final_pose["yaw_deg"],
        ),
    }


def validate_start_pose(start_pose, config):
    if not config["validate_start_pose"]:
        return []

    position_error = xy_distance(
        config["start_x"],
        config["start_y"],
        start_pose["x"],
        start_pose["y"],
    )
    yaw_error = shortest_angle_delta_deg(
        config["start_yaw_deg"],
        start_pose["yaw_deg"],
    )
    errors = []

    if position_error > config["start_position_tol_m"]:
        errors.append(
            "start position error "
            f"{position_error:.3f} m exceeds {config['start_position_tol_m']:.3f} m"
        )
    if abs(yaw_error) > config["start_yaw_tol_deg"]:
        errors.append(
            "start yaw error "
            f"{yaw_error:.2f} deg exceeds {config['start_yaw_tol_deg']:.2f} deg"
        )

    return errors


def validate_motion(summary, expected_distance_m, config):
    errors = []

    if abs(summary["forward_m"] - expected_distance_m) > config["distance_tol_m"]:
        errors.append(
            "forward distance "
            f"{summary['forward_m']:.3f} m is outside "
            f"{expected_distance_m:.3f} +/- {config['distance_tol_m']:.3f} m"
        )
    if abs(summary["lateral_m"]) > config["lateral_tol_m"]:
        errors.append(
            "lateral drift "
            f"{summary['lateral_m']:.3f} m exceeds {config['lateral_tol_m']:.3f} m"
        )
    if abs(summary["yaw_change_deg"]) > config["yaw_drift_tol_deg"]:
        errors.append(
            "yaw drift "
            f"{summary['yaw_change_deg']:.2f} deg exceeds "
            f"{config['yaw_drift_tol_deg']:.2f} deg"
        )

    return errors


def ensure_result_file_schema(file_path):
    return ensure_csv_schema(file_path, CSV_HEADER)


class ScriptedDrive(DriveNode):
    def drive_until_forward_distance(
        self,
        start_pose,
        target_distance_m: float,
        linear_x: float,
        angular_z: float,
        timeout_sec: float,
    ):
        start_time = time.time()
        last_summary = None

        while rclpy.ok():
            if self.last_odom is not None:
                current_pose = odom_to_xy_yaw(self.last_odom)
                last_summary = motion_summary(start_pose, current_pose)

                if last_summary["forward_m"] >= target_distance_m:
                    self.stop()
                    return self.last_odom

            elapsed = time.time() - start_time
            if elapsed > timeout_sec:
                progress = (
                    f"{last_summary['forward_m']:.3f} m"
                    if last_summary is not None
                    else "unknown"
                )
                self.stop()
                raise RuntimeError(
                    "Timed out before reaching odometry target: "
                    f"progress={progress}, target={target_distance_m:.3f} m, "
                    f"timeout={timeout_sec:.1f} s"
                )

            self.publish_velocity(linear_x, angular_z)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.05)

        self.stop()
        raise RuntimeError("ROS shutdown before reaching odometry target.")

    def save_result(self, odom_start_msg, odom_final_msg):
        row = [datetime.now().isoformat(), self.run_id, *([""] * 6),
               *pose_fields(odom_to_xy_yaw(odom_start_msg)),
               *pose_fields(odom_to_xy_yaw(odom_final_msg)), "simulation"]
        append_csv_row(RESULTS_CSV, CSV_HEADER, row)
        self.get_logger().info(f"Saved result to {RESULTS_CSV}")


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else "manual_run"

    try:
        motion = configured_motion()
        checks = validation_config()
    except ValueError as exc:
        print(f"Invalid simulation configuration: {exc}", file=sys.stderr)
        return 2

    if rclpy is None:
        print(
            "ROS 2 Python modules are unavailable. Source ROS 2 Humble before running.",
            file=sys.stderr,
        )
        return 2

    rclpy.init()
    node = ScriptedDrive(run_id)

    try:
        node.get_logger().info(f"{run_id}: forward {motion['distance_m']:.3f}m at {motion['speed_mps']:.3f}m/s")
        if "RUN_DURATION_SEC" in os.environ:
            node.get_logger().warn(
                "RUN_DURATION_SEC is ignored; simulation now stops from odometry. "
                "Use RUN_TIMEOUT_SEC only to change the safety timeout."
            )

        node.get_logger().info("Waiting for initial odometry...")
        odom_start_msg = node.wait_for_odom()
        if odom_start_msg is None:
            raise RuntimeError("No initial /odom sample was received.")

        odom_start = odom_to_xy_yaw(odom_start_msg)
        start_errors = validate_start_pose(odom_start, checks)
        if start_errors:
            raise RuntimeError(
                "Simulation start pose validation failed: " + "; ".join(start_errors)
            )

        node.drive_until_forward_distance(
            odom_start,
            motion["distance_m"],
            motion["speed_mps"],
            motion["angular_speed_radps"],
            motion["timeout_sec"],
        )

        node.stop()

        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.05)

        odom_final_msg = node.last_odom
        if odom_final_msg is None:
            raise RuntimeError("No final /odom sample was received.")

        odom_final = odom_to_xy_yaw(odom_final_msg)
        summary = motion_summary(odom_start, odom_final)
        motion_errors = validate_motion(summary, motion["distance_m"], checks)
        if motion_errors:
            raise RuntimeError(
                "Simulation motion validation failed: "
                + "; ".join(motion_errors)
                + (
                    f"; measured distance={summary['distance_m']:.3f} m, "
                    f"forward={summary['forward_m']:.3f} m, "
                    f"lateral={summary['lateral_m']:.3f} m, "
                    f"yaw_change={summary['yaw_change_deg']:.2f} deg"
                )
            )

        node.save_result(odom_start_msg, odom_final_msg)

    except KeyboardInterrupt:
        print("Interrupted. Sending stop command...")
        return 130

    except RuntimeError as exc:
        node.get_logger().error(str(exc))
        return 1

    finally:
        close_motion_node(node, rclpy)

    return 0


if __name__ == "__main__":
    sys.exit(main())
