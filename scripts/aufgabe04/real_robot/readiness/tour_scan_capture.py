"""Passive, bounded scan capture after the caller has stopped the robot."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import time

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.real_robot.readiness.tour_scan_contract import (
    HASH_FIELD, MAX_SCAN_AGE_SEC, MIN_SAMPLE_SEPARATION_SEC,
    capture_payload, finite, scan_evidence, stamp_seconds,
)

try:  # pragma: no cover - ROS adapter requires robot host.
    import rclpy
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer, TransformException, TransformListener
except ImportError:  # pragma: no cover
    rclpy = None
    Node = object


class TourScanCaptureError(RuntimeError):
    """No complete fresh stationary cohort was collected; motion stays stopped."""


class _TourScanNode(Node):  # pragma: no cover - ROS adapter.
    sample_count = 3
    payload_builder = staticmethod(capture_payload)
    metadata_profile = None

    def __init__(self, profile, tour_id, observation_not_before_sec=None):
        super().__init__("stored_pose_tour_scan_capture", namespace=profile.namespace)
        self.profile, self.tour_id = profile, tour_id
        self.observation_not_before_sec = observation_not_before_sec
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.pending = deque(maxlen=12)
        self.samples = deque(maxlen=self.sample_count)
        self.payload = None
        self.last_error = "no fresh scans received"
        self.subscription = self.create_subscription(
            LaserScan, profile.resolved_runtime().scan_topic,
            self._scan, qos_profile_sensor_data)

    def _scan(self, message):
        self.pending.append((message, time.time()))

    def poll(self):
        while self.pending:
            message, receipt = self.pending[0]
            try:
                stamp = stamp_seconds(message.header.stamp)
                if self.observation_not_before_sec is not None and stamp < self.observation_not_before_sec:
                    self.pending.popleft()
                    continue
                if time.time()-stamp > MAX_SCAN_AGE_SEC:
                    raise ValueError("source scan expired before exact-time TF")
                query = Time.from_msg(message.header.stamp)
                scan_tf = self.buffer.lookup_transform(self.profile.odom_frame,
                    self.profile.scan_frame, query, timeout=Duration(seconds=0))
                base_tf = self.buffer.lookup_transform(self.profile.odom_frame,
                    self.profile.base_frame, query, timeout=Duration(seconds=0))
                sample = scan_evidence(message, received_at_unix_sec=receipt,
                    scan_transform=scan_tf, base_transform=base_tf,
                    odom_frame=self.profile.odom_frame, base_frame=self.profile.base_frame,
                    scan_frame=self.profile.scan_frame)
                if self.metadata_profile is not None:
                    from scripts.aufgabe04.perception.lidar_scan_metadata import metadata_from_message
                    metadata = metadata_from_message(message, topology_profile=self.metadata_profile)
                    metadata.validate(sample["ranges"])
                    sample["scan_metadata"] = metadata.to_mapping()
            except TransformException as exc:
                self.last_error = f"exact scan-time transform unavailable: {exc}"
                return
            except (AttributeError, TypeError, ValueError) as exc:
                self.last_error = str(exc)
                self.pending.popleft()
                self.samples.clear()
                continue
            self.pending.popleft()
            if self.samples and stamp-float(self.samples[-1]["stamp_sec"]) < MIN_SAMPLE_SEPARATION_SEC-1e-6:
                continue
            self.samples.append(sample)
            if len(self.samples) == self.sample_count:
                try:
                    self.payload = self.payload_builder(tuple(self.samples), tour_id=self.tour_id,
                        odom_frame=self.profile.odom_frame, base_frame=self.profile.base_frame,
                        scan_frame=self.profile.scan_frame, captured_at_unix_sec=time.time())
                    return
                except ValueError as exc:
                    self.last_error = str(exc)


def _capture_stationary_scan(profile, *, tour_id: str, output_path: Path, node_factory,
                      hash_field, timeout_sec: float = 3.0,
                      observation_not_before_sec: float | None = None) -> Path:
    """Capture exact-time odom scan evidence without any velocity publisher.

    The caller owns exclusive-motion preflight and the stopped state. This node
    verifies that the cohort's source-stamped base poses remain stationary while it
    observes. A timeout or invalid cohort never produces an accepted artifact.
    """
    timeout = finite(timeout_sec, "timeout_sec")
    if observation_not_before_sec is not None:
        observation_not_before_sec = finite(observation_not_before_sec, "observation_not_before_sec")
    if not 0 < timeout <= 30:
        raise ValueError("capture timeout must be positive and at most 30 seconds")
    if rclpy is None:
        raise TourScanCaptureError("ROS2 scan capture dependencies are unavailable")
    path = Path(output_path)
    if path.exists() or path.is_symlink():
        raise ValueError("tour scan capture output must be fresh")
    owns_context = not rclpy.ok()
    node = None
    try:
        if owns_context:
            rclpy.init(args=None)
        node = node_factory(profile, tour_id, observation_not_before_sec)
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(.02, max(0., deadline-time.monotonic())))
            node.poll()
            if node.payload is not None:
                write_content_hashed_json(path, node.payload, hash_field=hash_field)
                return path
        raise TourScanCaptureError(f"stationary tour scan capture failed: {node.last_error}")
    finally:
        if node is not None:
            node.destroy_node()
        if owns_context and rclpy.ok():
            rclpy.shutdown()


def capture_tour_scan(profile, *, tour_id: str, output_path: Path,
                      timeout_sec: float = 3.0, observation_not_before_sec: float | None = None) -> Path:
    return _capture_stationary_scan(profile, tour_id=tour_id, output_path=output_path,
        node_factory=_TourScanNode, hash_field=HASH_FIELD, timeout_sec=timeout_sec,
        observation_not_before_sec=observation_not_before_sec)
