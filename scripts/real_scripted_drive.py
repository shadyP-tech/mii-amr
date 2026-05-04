#!/usr/bin/env python3

import csv
import math
import os
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


DEFAULT_RUN_MODE = "rotate-in-place"
DEFAULT_LINEAR_SPEED_MPS = 0.10
DEFAULT_FORWARD_DURATION_SEC = 3.0
DEFAULT_ROTATION_ANGLE_DEG = -90.0
DEFAULT_ROTATION_ANGULAR_SPEED_RADPS = 0.30
FORWARD_RESULTS_CSV = "results/real_scripted_drive_runs.csv"
ROTATION_RESULTS_CSV = "results/real_rotation_runs.csv"

FORWARD_CSV_HEADER = [
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

ROTATION_CSV_HEADER = [
    "timestamp",
    "run_id",
    "run_mode",
    "command_angle_deg",
    "direction",
    "linear_x_mps",
    "angular_z_radps",
    "duration_sec",
    "tracker_start_x",
    "tracker_start_y",
    "tracker_start_yaw_deg",
    "tracker_final_x",
    "tracker_final_y",
    "tracker_final_yaw_deg",
    "tracker_yaw_change_deg",
    "tracker_yaw_error_deg",
    "tracker_dx",
    "tracker_dy",
    "tracker_position_drift_m",
    "odom_start_x",
    "odom_start_y",
    "odom_start_yaw_deg",
    "odom_final_x",
    "odom_final_y",
    "odom_final_yaw_deg",
    "odom_yaw_change_deg",
    "odom_yaw_error_deg",
    "odom_dx",
    "odom_dy",
    "odom_position_drift_m",
    "notes",
]


def parse_float_env(env, name, default):
    try:
        return float(env.get(name, default))
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc


def configured_motion(env=os.environ):
    run_mode = env.get("REAL_RUN_MODE", env.get("RUN_MODE", DEFAULT_RUN_MODE))

    if run_mode == "linear-forward":
        linear_x = parse_float_env(env, "RUN_SPEED", DEFAULT_LINEAR_SPEED_MPS)
        duration = parse_float_env(env, "RUN_DURATION_SEC", DEFAULT_FORWARD_DURATION_SEC)
        if linear_x <= 0.0:
            raise ValueError("RUN_SPEED must be greater than zero")
        if duration <= 0.0:
            raise ValueError("RUN_DURATION_SEC must be greater than zero")

        return {
            "run_mode": "linear-forward",
            "linear_x_mps": linear_x,
            "angular_z_radps": 0.0,
            "duration_sec": duration,
            "results_csv": FORWARD_RESULTS_CSV,
            "notes": "real",
        }

    if run_mode in {"rotate-in-place", "rotation", "rotate"}:
        angle_deg = parse_float_env(env, "RUN_ANGLE_DEG", DEFAULT_ROTATION_ANGLE_DEG)
        angular_speed = abs(
            parse_float_env(
                env,
                "RUN_ANGULAR_SPEED",
                DEFAULT_ROTATION_ANGULAR_SPEED_RADPS,
            )
        )

        if angle_deg == 0.0:
            raise ValueError("RUN_ANGLE_DEG must be non-zero")
        if angular_speed <= 0.0:
            raise ValueError("RUN_ANGULAR_SPEED must be greater than zero")

        angle_rad = math.radians(angle_deg)
        angular_z = math.copysign(angular_speed, angle_rad)

        return {
            "run_mode": "rotate-in-place",
            "command_angle_deg": angle_deg,
            "direction": "clockwise" if angle_deg < 0.0 else "counterclockwise",
            "linear_x_mps": 0.0,
            "angular_z_radps": angular_z,
            "duration_sec": abs(angle_rad) / angular_speed,
            "results_csv": ROTATION_RESULTS_CSV,
            "notes": "real_rotation",
        }

    raise ValueError(
        f"unsupported RUN_MODE={run_mode!r}; expected linear-forward or rotate-in-place"
    )


def shortest_angle_delta_deg(start_deg, end_deg):
    return (end_deg - start_deg + 180.0) % 360.0 - 180.0


def pose_delta(start_pose, final_pose, command_angle_deg=None):
    if start_pose is None or final_pose is None:
        result = {
            "dx": "",
            "dy": "",
            "position_drift_m": "",
            "yaw_change_deg": "",
        }
    else:
        dx = final_pose["x"] - start_pose["x"]
        dy = final_pose["y"] - start_pose["y"]
        result = {
            "dx": dx,
            "dy": dy,
            "position_drift_m": math.hypot(dx, dy),
            "yaw_change_deg": shortest_angle_delta_deg(
                start_pose["yaw_deg"],
                final_pose["yaw_deg"],
            ),
        }

    if command_angle_deg is not None and result["yaw_change_deg"] != "":
        result["yaw_error_deg"] = shortest_angle_delta_deg(
            command_angle_deg,
            result["yaw_change_deg"],
        )
    elif command_angle_deg is not None:
        result["yaw_error_deg"] = ""

    return result


def append_csv_row(file_path, header, row):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0

    if file_exists:
        with open(file_path, newline="") as f:
            existing_header = next(csv.reader(f), None)
        if existing_header != header:
            raise RuntimeError(
                f"{file_path} has an unrecognized schema. Move or migrate it "
                "before appending new real-run results."
            )

    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def read_tracker_pose(path="results/latest_tracker_pose.csv"):
    if not os.path.exists(path):
        return None

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None

    row = rows[-1]
    valid_pose = row.get("valid_pose")
    if valid_pose is not None and valid_pose.strip().lower() not in {"1", "true"}:
        return None

    try:
        num_detected = int(float(row.get("num_detected", 3) or 0))
        pose = {
            "timestamp": row.get("timestamp", ""),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "yaw_rad": float(row["yaw_rad"]),
            "yaw_deg": float(row["yaw_deg"]),
        }
    except (KeyError, TypeError, ValueError):
        return None

    if num_detected < 3:
        return None

    if not all(math.isfinite(pose[key]) for key in ("x", "y", "yaw_rad", "yaw_deg")):
        return None

    return pose


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
        "yaw_rad": yaw,
        "yaw_deg": math.degrees(yaw),
    }


def pose_fields(pose):
    if pose is None:
        return ["", "", ""]
    return [pose["x"], pose["y"], pose["yaw_deg"]]


def save_real_result(
    run_id,
    motion,
    tracker_start,
    tracker_final,
    odom_start_msg,
    odom_final_msg,
):
    odom_start = odom_to_xy_yaw(odom_start_msg)
    odom_final = odom_to_xy_yaw(odom_final_msg)

    if motion["run_mode"] == "rotate-in-place":
        save_rotation_result(
            run_id,
            motion,
            tracker_start,
            tracker_final,
            odom_start,
            odom_final,
        )
        return

    append_csv_row(
        motion["results_csv"],
        FORWARD_CSV_HEADER,
        [
            datetime.now().isoformat(),
            run_id,
            *pose_fields(tracker_start),
            *pose_fields(tracker_final),
            *pose_fields(odom_start),
            *pose_fields(odom_final),
            motion["notes"],
        ],
    )
    print(f"Saved result to {motion['results_csv']}")


def save_rotation_result(
    run_id,
    motion,
    tracker_start,
    tracker_final,
    odom_start,
    odom_final,
):
    tracker_delta = pose_delta(
        tracker_start,
        tracker_final,
        command_angle_deg=motion["command_angle_deg"],
    )
    odom_delta = pose_delta(
        odom_start,
        odom_final,
        command_angle_deg=motion["command_angle_deg"],
    )

    append_csv_row(
        motion["results_csv"],
        ROTATION_CSV_HEADER,
        [
            datetime.now().isoformat(),
            run_id,
            motion["run_mode"],
            motion["command_angle_deg"],
            motion["direction"],
            motion["linear_x_mps"],
            motion["angular_z_radps"],
            motion["duration_sec"],
            *pose_fields(tracker_start),
            *pose_fields(tracker_final),
            tracker_delta["yaw_change_deg"],
            tracker_delta["yaw_error_deg"],
            tracker_delta["dx"],
            tracker_delta["dy"],
            tracker_delta["position_drift_m"],
            *pose_fields(odom_start),
            *pose_fields(odom_final),
            odom_delta["yaw_change_deg"],
            odom_delta["yaw_error_deg"],
            odom_delta["dx"],
            odom_delta["dy"],
            odom_delta["position_drift_m"],
            motion["notes"],
        ],
    )
    print(f"Saved result to {motion['results_csv']}")


class ScriptedDrive(Node):
    def __init__(self, run_id: str):
        if rclpy is None:
            raise RuntimeError(
                "ROS 2 Python modules are unavailable. Source ROS 2 Humble before "
                "running the real robot drive."
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


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else "manual_run"

    try:
        motion = configured_motion()
    except ValueError as exc:
        print(f"Invalid real-run configuration: {exc}", file=sys.stderr)
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
            "Configured motion: "
            f"mode={motion['run_mode']}, "
            f"linear_x={motion['linear_x_mps']:.3f} m/s, "
            f"angular_z={motion['angular_z_radps']:.3f} rad/s, "
            f"duration={motion['duration_sec']:.3f} s"
        )
        if motion["run_mode"] == "rotate-in-place":
            node.get_logger().info(
                "Rotation command: "
                f"angle={motion['command_angle_deg']:.1f} deg, "
                f"direction={motion['direction']}"
            )

        node.get_logger().info("Waiting for initial odometry...")
        odom_start = node.wait_for_odom()

        tracker_start = read_tracker_pose()
        if tracker_start is None:
            node.get_logger().warn("No tracker start pose found.")

        node.get_logger().info("Executing configured motion")
        node.send_cmd(
            motion["linear_x_mps"],
            motion["angular_z_radps"],
            motion["duration_sec"],
        )

        node.get_logger().info("Done")
        node.stop()

        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.05)

        odom_final = node.last_odom
        tracker_final = read_tracker_pose()
        if tracker_final is None:
            node.get_logger().warn("No tracker final pose found.")

        save_real_result(
            run_id,
            motion,
            tracker_start,
            tracker_final,
            odom_start,
            odom_final,
        )

    except KeyboardInterrupt:
        print("Interrupted. Sending stop command...")
        try:
            node.stop()
        except Exception:
            pass

    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
