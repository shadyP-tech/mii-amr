#!/usr/bin/env python3

import csv
import math
import os
import sys
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


LINEAR_SPEED_MPS = 0.10
ANGULAR_SPEED_RADPS = 0.0
DRIVE_DURATION_SEC = 3.0
RESULTS_CSV = "results/scripted_drive_runs.csv"

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

    rclpy.init()
    node = ScriptedDrive(run_id)

    try:
        node.get_logger().info(f"Starting run: {run_id}")

        node.get_logger().info("Waiting for initial odometry...")
        odom_start = node.wait_for_odom()

        node.get_logger().info("Driving forward")
        node.send_cmd(LINEAR_SPEED_MPS, ANGULAR_SPEED_RADPS, DRIVE_DURATION_SEC)

        node.get_logger().info("Done")
        node.stop()

        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.05)

        odom_final = node.last_odom
        node.save_result(odom_start, odom_final)

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


if __name__ == "__main__":
    main()
