"""Print current AMCL map pose as planner start arguments."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.ros_runtime_config import RuntimeConfig, resolve_runtime_config

try:  # pragma: no cover - exercised on ROS hosts.
    import rclpy
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from rclpy.node import Node
    from rclpy.time import Time
except ImportError:  # pragma: no cover - keeps offline tests ROS-free.
    rclpy = None
    PoseWithCovarianceStamped = None
    Node = object
    Time = None


@dataclass(frozen=True)
class CurrentAmclPose:
    x_m: float
    y_m: float
    yaw_rad: float
    frame_id: str
    topic: str
    header_stamp_sec: float
    receipt_age_sec: float
    header_age_sec: float


def _require_ros() -> None:
    if rclpy is None:
        raise RuntimeError("ROS2 Python packages are not available in this environment")


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0


def planner_args_from_pose(pose: CurrentAmclPose, *, precision: int = 6) -> str:
    return (
        f"--start-x {pose.x_m:.{precision}f} "
        f"--start-y {pose.y_m:.{precision}f} "
        f"--start-yaw {pose.yaw_rad:.{precision}f}"
    )


def validate_current_amcl_pose(
    pose: CurrentAmclPose,
    *,
    expected_frame: str,
    max_age_sec: float,
) -> None:
    if pose.frame_id.strip("/") != expected_frame.strip("/"):
        raise ValueError(f"AMCL frame mismatch: got {pose.frame_id!r}, expected {expected_frame!r}")
    if pose.header_age_sec > max_age_sec:
        raise ValueError(f"AMCL header age {pose.header_age_sec:.3f}s exceeds {max_age_sec:.3f}s")
    if pose.receipt_age_sec > max_age_sec:
        raise ValueError(f"AMCL receipt age {pose.receipt_age_sec:.3f}s exceeds {max_age_sec:.3f}s")


class CurrentAmclPoseReader(Node):  # pragma: no cover - requires ROS runtime.
    def __init__(self, *, topic: str) -> None:
        super().__init__("aufgabe04_current_amcl_pose_reader")
        self.topic = topic
        self.latest_msg = None
        self.latest_receipt = None
        self.create_subscription(PoseWithCovarianceStamped, topic, self._callback, 10)

    def _callback(self, msg) -> None:
        self.latest_msg = msg
        self.latest_receipt = self.get_clock().now()

    def current_pose(self) -> CurrentAmclPose | None:
        if self.latest_msg is None or self.latest_receipt is None:
            return None
        now = self.get_clock().now()
        msg = self.latest_msg
        pose = msg.pose.pose
        header_time = Time.from_msg(msg.header.stamp)
        return CurrentAmclPose(
            x_m=float(pose.position.x),
            y_m=float(pose.position.y),
            yaw_rad=yaw_from_quaternion(pose.orientation),
            frame_id=str(msg.header.frame_id),
            topic=self.topic,
            header_stamp_sec=stamp_to_sec(msg.header.stamp),
            receipt_age_sec=(now - self.latest_receipt).nanoseconds / 1_000_000_000.0,
            header_age_sec=(now - header_time).nanoseconds / 1_000_000_000.0,
        )


def read_current_amcl_pose(
    *,
    namespace: str,
    amcl_topic: str,
    map_frame: str,
    timeout_sec: float,
    max_age_sec: float,
) -> CurrentAmclPose:
    _require_ros()
    resolved = resolve_runtime_config(RuntimeConfig(namespace=namespace, amcl_topic=amcl_topic, map_frame=map_frame))
    rclpy.init(args=None)
    node = CurrentAmclPoseReader(topic=resolved.amcl_topic)
    deadline = node.get_clock().now().nanoseconds / 1_000_000_000.0 + timeout_sec
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            current = node.current_pose()
            if current is not None:
                validate_current_amcl_pose(current, expected_frame=resolved.map_frame, max_age_sec=max_age_sec)
                return current
            now_sec = node.get_clock().now().nanoseconds / 1_000_000_000.0
            if now_sec >= deadline:
                raise RuntimeError(f"timed out waiting for AMCL pose on {resolved.amcl_topic}")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise RuntimeError("ROS shutdown before AMCL pose was received")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="")
    parser.add_argument("--amcl-topic", default="amcl_pose")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--timeout-sec", type=float, default=3.0)
    parser.add_argument("--max-age-sec", type=float, default=2.0)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of planner CLI args")
    parser.add_argument("--precision", type=int, default=6)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        pose = read_current_amcl_pose(
            namespace=args.namespace,
            amcl_topic=args.amcl_topic,
            map_frame=args.map_frame,
            timeout_sec=args.timeout_sec,
            max_age_sec=args.max_age_sec,
        )
    except (RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

    if args.json:
        print(json.dumps(asdict(pose), indent=2, sort_keys=True))
    else:
        print(planner_args_from_pose(pose, precision=args.precision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
