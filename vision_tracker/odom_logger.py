"""
odom_logger.py — Log ROS /odom to CSV for comparison with vision tracking.

Usage:
    python3 odom_logger.py

Writes:  data/odom_YYYYMMDD_HHMMSS.csv
Columns: timestamp, odom_x, odom_y, odom_yaw

Run this in parallel with main.py during experiments.  Both CSVs use
Unix timestamps so analyze_runs.py can align them later.

Requirements:
    - ROS environment sourced
    - rospy or rclpy available
"""

import csv
import math
import os
import sys
import time
from datetime import datetime

import config

# Try ROS 2 (rclpy) first, then fall back to ROS 1 (rospy)
ROS_VERSION = None

try:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry

    ROS_VERSION = 2
except ImportError:
    try:
        import rospy
        from nav_msgs.msg import Odometry

        ROS_VERSION = 1
    except ImportError:
        pass


def _quaternion_to_yaw(q):
    """Extract yaw from a quaternion (x, y, z, w)."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _make_csv_path():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(config.DATA_DIR, f"odom_{ts}.csv")


CSV_HEADER = ["timestamp", "odom_x", "odom_y", "odom_yaw"]


# ROS 2 implementation
class OdomLoggerROS2(Node):
    def __init__(self, csv_path):
        super().__init__("odom_logger")
        self._csvfile = open(csv_path, "w", newline="")
        self._writer = csv.writer(self._csvfile)
        self._writer.writerow(CSV_HEADER)

        self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.get_logger().info(f"Logging /odom to {csv_path}")

    def _odom_cb(self, msg):
        ts = time.time()
        pos = msg.pose.pose.position
        yaw = _quaternion_to_yaw(msg.pose.pose.orientation)
        self._writer.writerow(
            [
                f"{ts:.4f}",
                f"{pos.x:.5f}",
                f"{pos.y:.5f}",
                f"{yaw:.5f}",
            ]
        )

    def destroy_node(self):
        self._csvfile.close()
        super().destroy_node()


# ROS 1 implementation
def _run_ros1(csv_path):
    rospy.init_node("odom_logger", anonymous=True)

    csvfile = open(csv_path, "w", newline="")
    writer = csv.writer(csvfile)
    writer.writerow(CSV_HEADER)

    def _odom_cb(msg):
        ts = time.time()
        pos = msg.pose.pose.position
        yaw = _quaternion_to_yaw(msg.pose.pose.orientation)
        writer.writerow(
            [
                f"{ts:.4f}",
                f"{pos.x:.5f}",
                f"{pos.y:.5f}",
                f"{yaw:.5f}",
            ]
        )

    rospy.Subscriber("/odom", Odometry, _odom_cb)
    rospy.loginfo(f"Logging /odom to {csv_path}")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        csvfile.close()
        print(f"Odom data saved to {csv_path}")


# main
def main():
    if ROS_VERSION is None:
        print("ERROR: Neither rclpy (ROS 2) nor rospy (ROS 1) is available.")
        print("       Source your ROS workspace first.")
        sys.exit(1)

    csv_path = _make_csv_path()

    if ROS_VERSION == 2:
        rclpy.init()
        node = OdomLoggerROS2(csv_path)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
            print(f"Odom data saved to {csv_path}")
    else:
        _run_ros1(csv_path)


if __name__ == "__main__":
    main()
