#!/usr/bin/env python3

import csv
import math
import os
import re
import sys
import time
from datetime import datetime

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
except ImportError:
    rclpy = None
    Node = object
    Twist = None
    Odometry = object


DEFAULT_RUN_MODE = "linear-forward"
DEFAULT_LINEAR_SPEED_MPS = 0.10
DEFAULT_RUN_DISTANCE = "30cm"
ANGULAR_SPEED_RADPS = 0.0
RESULTS_CSV = "results/scripted_drive_runs.csv"
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

LEGACY_CSV_HEADER = [
    "timestamp",
    "run_id",
    "x",
    "y",
    "z",
    "qx",
    "qy",
    "qz",
    "qw",
    "notes",
]

CSV_HEADER = [
    "timestamp",
    "run_id",

    "tracker_start_x",
    "tracker_start_y",
    "tracker_start_yaw_deg",

    "tracker_final_x",
    "tracker_final_y",
    "tracker_final_yaw_deg",

    "odom_start_x",
    "odom_start_y",
    "odom_start_yaw_deg",

    "odom_final_x",
    "odom_final_y",
    "odom_final_yaw_deg",

    "notes",
]


def quaternion_to_yaw_deg(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def parse_distance_m(value):
    match = DISTANCE_RE.match(str(value))
    if not match:
        raise ValueError(f"invalid distance value: {value!r}")

    amount = float(match.group(1))
    unit = (match.group(2) or "m").lower()

    if unit in {"m", "meter", "meters"}:
        return amount
    if unit in {"cm", "centimeter", "centimeters"}:
        return amount / 100.0
    if unit in {"mm", "millimeter", "millimeters"}:
        return amount / 1000.0

    raise ValueError(f"unsupported distance unit in {value!r}")


def parse_float_env(env, name, default):
    try:
        return float(env.get(name, default))
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc


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

    duration_sec = parse_float_env(env, "RUN_DURATION_SEC", distance_m / speed_mps)
    if duration_sec <= 0.0:
        raise ValueError("RUN_DURATION_SEC must be greater than zero")

    return {
        "run_mode": run_mode,
        "speed_mps": speed_mps,
        "distance_m": distance_m,
        "duration_sec": duration_sec,
        "angular_speed_radps": ANGULAR_SPEED_RADPS,
    }


def validation_config(env=os.environ):
    return {
        "validate_start_pose": env.get("SIM_VALIDATE_START_POSE", "1").lower()
        not in {"0", "false", "no"},
        "start_x": parse_float_env(env, "SIM_START_X", DEFAULT_SIM_START_X),
        "start_y": parse_float_env(env, "SIM_START_Y", DEFAULT_SIM_START_Y),
        "start_yaw_deg": parse_float_env(
            env,
            "SIM_START_YAW_DEG",
            DEFAULT_SIM_START_YAW_DEG,
        ),
        "start_position_tol_m": parse_float_env(
            env,
            "SIM_START_POSITION_TOL_M",
            DEFAULT_START_POSITION_TOL_M,
        ),
        "start_yaw_tol_deg": parse_float_env(
            env,
            "SIM_START_YAW_TOL_DEG",
            DEFAULT_START_YAW_TOL_DEG,
        ),
        "distance_tol_m": parse_float_env(
            env,
            "SIM_DISTANCE_TOL_M",
            DEFAULT_DISTANCE_TOL_M,
        ),
        "lateral_tol_m": parse_float_env(
            env,
            "SIM_LATERAL_TOL_M",
            DEFAULT_LATERAL_TOL_M,
        ),
        "yaw_drift_tol_deg": parse_float_env(
            env,
            "SIM_YAW_DRIFT_TOL_DEG",
            DEFAULT_YAW_DRIFT_TOL_DEG,
        ),
    }


def xy_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def shortest_angle_delta_deg(start_deg, end_deg):
    return (end_deg - start_deg + 180.0) % 360.0 - 180.0


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


def ensure_result_file_schema(file_path):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return False

    with open(file_path, newline="") as f:
        reader = csv.reader(f)
        existing_header = next(reader, None)
        existing_rows = list(reader)

    if existing_header == CSV_HEADER:
        return True

    if existing_header == LEGACY_CSV_HEADER:
        migrate_legacy_result_file(file_path, existing_rows)
        return True

    raise RuntimeError(
        f"{file_path} has an unrecognized schema. Move or migrate it before "
        "running the simulation script so new rows match real-run logs."
    )


def migrate_legacy_result_file(file_path, rows):
    tmp_path = f"{file_path}.tmp"
    with open(tmp_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        for row in rows:
            if len(row) != len(LEGACY_CSV_HEADER):
                continue

            legacy = dict(zip(LEGACY_CSV_HEADER, row))
            try:
                yaw_deg = quaternion_to_yaw_deg(
                    float(legacy["qx"]),
                    float(legacy["qy"]),
                    float(legacy["qz"]),
                    float(legacy["qw"]),
                )
            except ValueError:
                yaw_deg = ""

            writer.writerow([
                legacy["timestamp"],
                legacy["run_id"],
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                legacy["x"],
                legacy["y"],
                yaw_deg,
                legacy["notes"],
            ])

    os.replace(tmp_path, file_path)

    return True


class ScriptedDrive(Node):
    def __init__(self, run_id: str):
        if rclpy is None:
            raise RuntimeError(
                "ROS 2 Python modules are unavailable. Source ROS 2 Humble before "
                "running the simulation drive."
            )

        super().__init__("scripted_drive")

        self.run_id = run_id
        self.last_odom = None

        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10,
        )

        time.sleep(1.0)

    def odom_callback(self, msg: Odometry):
        self.last_odom = msg

    def wait_for_odom(self, timeout_sec=5.0):
        start = time.time()

        while rclpy.ok() and self.last_odom is None:
            if time.time() - start > timeout_sec:
                self.get_logger().warn("Timed out waiting for /odom")
                return None

            rclpy.spin_once(self, timeout_sec=0.1)

        return self.last_odom

    def publish_velocity(self, linear_x: float, angular_z: float):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.pub.publish(msg)

    def send_cmd(self, linear_x: float, angular_z: float, duration: float):
        start = time.time()

        while rclpy.ok() and time.time() - start < duration:
            self.publish_velocity(linear_x, angular_z)
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.1)

        self.stop()

    def stop(self):
        msg = Twist()
        for _ in range(10):
            if rclpy.ok():
                self.pub.publish(msg)
            time.sleep(0.05)

    def save_result(self, odom_start_msg, odom_final_msg):
        os.makedirs("results", exist_ok=True)

        file_exists = ensure_result_file_schema(RESULTS_CSV)

        odom_start = odom_to_xy_yaw(odom_start_msg)
        odom_final = odom_to_xy_yaw(odom_final_msg)

        with open(RESULTS_CSV, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(CSV_HEADER)

            writer.writerow([
                datetime.now().isoformat(),
                self.run_id,

                "",
                "",
                "",

                "",
                "",
                "",

                odom_start["x"] if odom_start else "",
                odom_start["y"] if odom_start else "",
                odom_start["yaw_deg"] if odom_start else "",

                odom_final["x"] if odom_final else "",
                odom_final["y"] if odom_final else "",
                odom_final["yaw_deg"] if odom_final else "",

                "simulation",
            ])

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
        node.get_logger().info(f"Starting run: {run_id}")
        node.get_logger().info(
            "Configured drive: "
            f"{motion['speed_mps']:.3f} m/s for {motion['duration_sec']:.3f} s "
            f"({motion['distance_m']:.3f} m target)"
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

        node.get_logger().info("Driving forward")
        node.send_cmd(
            motion["speed_mps"],
            motion["angular_speed_radps"],
            motion["duration_sec"],
        )

        node.get_logger().info("Done")
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
        try:
            node.stop()
        except Exception:
            pass
        return 130

    except RuntimeError as exc:
        node.get_logger().error(str(exc))
        try:
            node.stop()
        except Exception:
            pass
        return 1

    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
