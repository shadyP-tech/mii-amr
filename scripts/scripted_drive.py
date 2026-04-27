#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class ScriptedDrive(Node):
    def __init__(self):
        super().__init__("scripted_drive")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        time.sleep(1.0)

    def publish_velocity(self, linear_x: float, angular_z: float):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.pub.publish(msg)

    def send_cmd(self, linear_x: float, angular_z: float, duration: float):
        start = time.time()

        while rclpy.ok() and time.time() - start < duration:
            self.publish_velocity(linear_x, angular_z)
            time.sleep(0.1)

        self.stop()

    def stop(self):
        msg = Twist()
        for _ in range(10):
            if rclpy.ok():
                self.pub.publish(msg)
            time.sleep(0.05)


def main():
    rclpy.init()
    node = ScriptedDrive()

    try:
        node.get_logger().info("Driving forward")
        node.send_cmd(0.10, 0.0, 3.0)

        node.get_logger().info("Rotating")
        node.send_cmd(0.0, 0.5, 2.0)

        node.get_logger().info("Driving forward again")
        node.send_cmd(0.10, 0.0, 3.0)

        node.get_logger().info("Done")
        node.stop()

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