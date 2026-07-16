"""Passively capture timestamp-aligned ROS telemetry and camera frames in simulation."""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.perception.ros_image_adapter import raw_msg_to_bgr_frame  # noqa: E402


def quaternion_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def nearest_finite_range(ranges) -> float | None:
    valid = [float(value) for value in ranges if math.isfinite(float(value)) and float(value) > 0.0]
    return min(valid) if valid else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--sample-hz", type=float, default=5.0)
    parser.add_argument("--frame-fps", type=float, default=1.0)
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--camera-topic", default="/camera/image_raw")
    parser.add_argument("--overview-image-topic", default="")
    parser.add_argument("--model-states-topic", default="/gazebo/model_states")
    parser.add_argument("--model-name", default="burger")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--no-model-states", action="store_true")
    return parser


class SimulationDebugCapture:
    def __init__(self, args) -> None:
        try:
            import cv2
            import numpy
            import rclpy
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from rclpy.qos import QoSProfile, qos_profile_sensor_data
            from sensor_msgs.msg import Image, LaserScan
        except ImportError as exc:
            raise SystemExit("ROS 2, OpenCV, numpy and the standard ROS messages are required") from exc

        self.args = args
        self.cv2 = cv2
        self.numpy = numpy
        self.rclpy = rclpy
        self.bundle_dir = Path(args.bundle_dir)
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_file = (self.bundle_dir / "telemetry.jsonl").open("a", buffering=1)
        self.frame_index_file = (self.bundle_dir / "frames" / "frame_index.jsonl")
        self.frame_index_file.parent.mkdir(parents=True, exist_ok=True)
        self.frame_index = self.frame_index_file.open("a", buffering=1)
        self.node = rclpy.create_node("aufgabe04_simulation_debug_capture")
        sensor_qos = QoSProfile(
            reliability=qos_profile_sensor_data.reliability,
            durability=qos_profile_sensor_data.durability,
            history=qos_profile_sensor_data.history,
            depth=1,
        )
        self.pose = None
        self.command = {"linear_x": 0.0, "angular_z": 0.0}
        self.nearest_obstacle_m = None
        self.ground_truth_pose = None
        self.frame_sequence = 0
        self.last_frame_wall_time = {}

        self.node.create_subscription(Odometry, args.odom_topic, self._on_odom, sensor_qos)
        self.node.create_subscription(Twist, args.cmd_vel_topic, self._on_cmd_vel, sensor_qos)
        self.node.create_subscription(LaserScan, args.scan_topic, self._on_scan, sensor_qos)
        if not args.no_camera:
            self.node.create_subscription(
                Image,
                args.camera_topic,
                lambda message: self._on_image(message, "onboard"),
                sensor_qos,
            )
            if args.overview_image_topic:
                self.node.create_subscription(
                    Image,
                    args.overview_image_topic,
                    lambda message: self._on_image(message, "overview"),
                    sensor_qos,
                )
        if not args.no_model_states:
            try:
                from gazebo_msgs.msg import ModelStates
            except ImportError:
                self.node.get_logger().warning("gazebo_msgs unavailable; ground-truth capture disabled")
            else:
                self.node.create_subscription(
                    ModelStates, args.model_states_topic, self._on_model_states, sensor_qos
                )
        period = 1.0 / max(args.sample_hz, 0.1)
        self.node.create_timer(period, self._write_sample)

    def _pose_dict(self, pose) -> dict[str, float]:
        orientation = pose.orientation
        return {
            "x_m": float(pose.position.x),
            "y_m": float(pose.position.y),
            "yaw_rad": quaternion_yaw(
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            ),
        }

    def _on_odom(self, message) -> None:
        self.pose = self._pose_dict(message.pose.pose)

    def _on_cmd_vel(self, message) -> None:
        self.command = {
            "linear_x": float(message.linear.x),
            "angular_z": float(message.angular.z),
        }

    def _on_scan(self, message) -> None:
        self.nearest_obstacle_m = nearest_finite_range(message.ranges)

    def _on_model_states(self, message) -> None:
        try:
            index = list(message.name).index(self.args.model_name)
        except ValueError:
            return
        self.ground_truth_pose = self._pose_dict(message.pose[index])

    def _on_image(self, message, view: str) -> None:
        if self.args.frame_fps <= 0.0:
            return
        now = time.time()
        if now - self.last_frame_wall_time.get(view, 0.0) < 1.0 / self.args.frame_fps:
            return
        try:
            frame = raw_msg_to_bgr_frame(message, self.cv2, self.numpy)
        except ValueError as exc:
            self.node.get_logger().warning(str(exc))
            return
        self.last_frame_wall_time[view] = now
        self.frame_sequence += 1
        ros_time_sec = self.node.get_clock().now().nanoseconds / 1_000_000_000.0
        directory = self.bundle_dir / "frames" / view
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{self.frame_sequence:06d}_{now:.3f}.jpg"
        path = directory / filename
        if not self.cv2.imwrite(str(path), frame, [int(self.cv2.IMWRITE_JPEG_QUALITY), 85]):
            self.node.get_logger().warning(f"failed to write frame: {path}")
            return
        self.frame_index.write(
            json.dumps(
                {
                    "sequence": self.frame_sequence,
                    "view": view,
                    "wall_time_sec": now,
                    "ros_time_sec": ros_time_sec,
                    "path": str(path.relative_to(self.bundle_dir)),
                },
                sort_keys=True,
            )
            + "\n"
        )

    def _write_sample(self) -> None:
        payload = {
            "source": "telemetry",
            "wall_time_sec": time.time(),
            "ros_time_sec": self.node.get_clock().now().nanoseconds / 1_000_000_000.0,
            "pose": self.pose,
            "ground_truth_pose": self.ground_truth_pose,
            "command": self.command,
            "nearest_obstacle_m": self.nearest_obstacle_m,
        }
        self.telemetry_file.write(json.dumps(payload, sort_keys=True) + "\n")

    def close(self) -> None:
        self.telemetry_file.close()
        self.frame_index.close()
        self.node.destroy_node()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sample_hz <= 0.0:
        raise SystemExit("error: --sample-hz must be positive")
    if args.frame_fps < 0.0:
        raise SystemExit("error: --frame-fps must be non-negative")
    try:
        import rclpy
    except ImportError as exc:
        raise SystemExit("ROS 2 rclpy is required") from exc
    rclpy.init(args=None)
    capture = SimulationDebugCapture(args)

    def stop(_signum, _frame) -> None:
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(f"Simulation debug capture: {args.bundle_dir}", flush=True)
    try:
        rclpy.spin(capture.node)
    except KeyboardInterrupt:
        pass
    finally:
        capture.close()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
