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


def read_tracker_pose(path="results/latest_tracker_pose.csv"):
    if not os.path.exists(path):
        return None

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None

    row = rows[-1]
    return {
        "timestamp": row.get("timestamp", ""),
        "x": float(row["x"]),
        "y": float(row["y"]),
        "yaw_rad": float(row["yaw_rad"]),
        "yaw_deg": float(row["yaw_deg"]),
    }


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


def save_real_result(run_id, tracker_start, tracker_final, odom_start_msg, odom_final_msg):
    os.makedirs("results", exist_ok=True)

    file_path = "results/real_scripted_drive_runs.csv"
    file_exists = os.path.exists(file_path)

    odom_start = odom_to_xy_yaw(odom_start_msg)
    odom_final = odom_to_xy_yaw(odom_final_msg)

    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
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
            ])

        writer.writerow([
            datetime.now().isoformat(),
            run_id,

            tracker_start["x"] if tracker_start else "",
            tracker_start["y"] if tracker_start else "",
            tracker_start["yaw_deg"] if tracker_start else "",

            tracker_final["x"] if tracker_final else "",
            tracker_final["y"] if tracker_final else "",
            tracker_final["yaw_deg"] if tracker_final else "",

            odom_start["x"] if odom_start else "",
            odom_start["y"] if odom_start else "",
            odom_start["yaw_deg"] if odom_start else "",

            odom_final["x"] if odom_final else "",
            odom_final["y"] if odom_final else "",
            odom_final["yaw_deg"] if odom_final else "",

            "real",
        ])

    print(f"Saved result to {file_path}")


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


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else "manual_run"

    rclpy.init()
    node = ScriptedDrive(run_id)

    try:
        node.get_logger().info(f"Starting run: {run_id}")

        node.get_logger().info("Waiting for initial odometry...")
        odom_start = node.wait_for_odom()

        tracker_start = read_tracker_pose()
        if tracker_start is None:
            node.get_logger().warn("No tracker start pose found.")

        node.get_logger().info("Driving forward")
        node.send_cmd(0.1, 0.0, 1.0)

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


if __name__ == "__main__":
    main()