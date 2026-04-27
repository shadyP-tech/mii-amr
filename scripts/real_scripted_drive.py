#!/usr/bin/env python3

import csv
import os
import sys
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


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

    def save_real_result(run_id, tracker_start, tracker_final, odom_start, odom_final):
        os.makedirs("results", exist_ok=True)
        file_path = "results/real_scripted_drive_runs.csv"
        file_exists = os.path.exists(file_path)

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
                    "odom_final_x",
                    "odom_final_y",

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

                odom_start.pose.pose.position.x if odom_start else "",
                odom_start.pose.pose.position.y if odom_start else "",
                odom_final.pose.pose.position.x if odom_final else "",
                odom_final.pose.pose.position.y if odom_final else "",

                "real",
            ])

        self.get_logger().info(f"Saved result to {file_path}")


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else "manual_run"

    rclpy.init()
    node = ScriptedDrive(run_id)

    try:
        node.get_logger().info(f"Starting run: {run_id}")

        node.get_logger().info("Driving forward")
        node.send_cmd(0.05, 0.0, 2.0)

        node.get_logger().info("Rotating")
        node.send_cmd(0.0, 0.3, 1.0)

        node.get_logger().info("Driving forward again")
        node.send_cmd(0.05, 0.0, 2.0)

        node.get_logger().info("Done")
        node.stop()

        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.05)

        node.save_result()

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