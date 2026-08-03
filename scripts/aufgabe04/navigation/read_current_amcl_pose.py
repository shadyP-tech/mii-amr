"""Print current AMCL map pose as planner start arguments."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
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
    from std_srvs.srv import Empty
except ImportError:  # pragma: no cover - keeps offline tests ROS-free.
    rclpy = None
    PoseWithCovarianceStamped = None
    Node = object
    Time = None
    Empty = None


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
    def __init__(self, *, topic: str, nomotion_update_service: str | None) -> None:
        super().__init__("aufgabe04_current_amcl_pose_reader")
        self.topic = topic
        self.latest_msg = None
        self.latest_receipt = None
        self.create_subscription(PoseWithCovarianceStamped, topic, self._callback, 10)
        self.nomotion_client = (
            None
            if nomotion_update_service is None
            else self.create_client(Empty, nomotion_update_service)
        )
        self.nomotion_future = None

    def maybe_request_nomotion_update(self) -> bool:
        """Request one stationary AMCL publication after the subscriber exists."""

        if (
            self.nomotion_client is None
            or self.nomotion_future is not None
            or not self.nomotion_client.service_is_ready()
        ):
            return False
        self.nomotion_future = self.nomotion_client.call_async(Empty.Request())
        self.get_logger().info("requested stationary AMCL update")
        return True

    def nomotion_update_error(self) -> Exception | None:
        future = self.nomotion_future
        if future is None or not future.done():
            return None
        return future.exception()

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
    nomotion_update_service: str | None = "/request_nomotion_update",
) -> CurrentAmclPose:
    _require_ros()
    resolved = resolve_runtime_config(RuntimeConfig(namespace=namespace, amcl_topic=amcl_topic, map_frame=map_frame))
    rclpy.init(args=None)
    node = CurrentAmclPoseReader(
        topic=resolved.amcl_topic,
        nomotion_update_service=nomotion_update_service,
    )
    # The subscriber and service client must exist before requesting the
    # stationary update, otherwise AMCL's one publication can be missed.
    deadline = time.monotonic() + timeout_sec
    last_validation_error = None
    try:
        while rclpy.ok():
            node.maybe_request_nomotion_update()
            rclpy.spin_once(node, timeout_sec=0.05)
            current = node.current_pose()
            if current is not None:
                try:
                    validate_current_amcl_pose(
                        current,
                        expected_frame=resolved.map_frame,
                        max_age_sec=max_age_sec,
                    )
                except ValueError as exc:
                    # A retained or just-expired sample can arrive before the
                    # response to request_nomotion_update. Keep spinning for
                    # the fresh stationary publication instead of failing it.
                    last_validation_error = exc
                else:
                    return current
            service_error = node.nomotion_update_error()
            if service_error is not None:
                raise RuntimeError(
                    "stationary AMCL update service failed: "
                    f"{service_error}"
                )
            if time.monotonic() >= deadline:
                service_state = (
                    "disabled"
                    if nomotion_update_service is None
                    else "requested"
                    if node.nomotion_future is not None
                    else f"unavailable on {nomotion_update_service}"
                )
                raise RuntimeError(
                    f"timed out waiting for AMCL pose on {resolved.amcl_topic}; "
                    f"stationary update={service_state}"
                    + (
                        ""
                        if last_validation_error is None
                        else f"; last sample invalid: {last_validation_error}"
                    )
                )
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
    parser.add_argument(
        "--nomotion-update-service",
        default="/request_nomotion_update",
        help=(
            "AMCL service requested after the pose subscriber exists so a "
            "stationary robot publishes a fresh pose."
        ),
    )
    parser.add_argument(
        "--skip-nomotion-update",
        action="store_true",
        help="Wait for an AMCL publication without requesting a stationary update.",
    )
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
            nomotion_update_service=(
                None
                if args.skip_nomotion_update
                else args.nomotion_update_service
            ),
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
