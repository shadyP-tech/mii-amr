"""ROS collection facade for pre-motion camera/LiDAR timing readiness.

The stable public API is re-exported from the ROS-free contract. This adapter
only subscribes to existing sensor topics and retains a bounded header window;
it never publishes, requests operator input, or starts subprocesses.
"""

from __future__ import annotations

from collections import deque
import time

from scripts.aufgabe04.real_robot.readiness.sensor_timing_contract import (
    DEFAULT_MAX_CAMERA_INFO_AGE_SEC,
    DEFAULT_MAX_CAMERA_INFO_IMAGE_SKEW_SEC,
    DEFAULT_MAX_FUTURE_TIMESTAMP_SEC,
    DEFAULT_MAX_IMAGE_SCAN_SKEW_SEC,
    DEFAULT_MAX_SENSOR_AGE_SEC,
    DEFAULT_SENSOR_TIMING_SAMPLE_CAPACITY,
    DEFAULT_SENSOR_TIMING_TIMEOUT_SEC,
    FAILURE_CAMERA_INFO_FRAME_EMPTY,
    FAILURE_CAMERA_INFO_FRAME_MISMATCH,
    FAILURE_CAMERA_INFO_IMAGE_SKEW,
    FAILURE_CAMERA_INFO_STAMP_FUTURE,
    FAILURE_CAMERA_INFO_STAMP_INVALID,
    FAILURE_CAMERA_INFO_STAMP_STALE,
    FAILURE_CAMERA_INFO_TIMEOUT,
    FAILURE_IMAGE_FRAME_EMPTY,
    FAILURE_IMAGE_FRAME_MISMATCH,
    FAILURE_IMAGE_SCAN_SKEW,
    FAILURE_IMAGE_STAMP_FUTURE,
    FAILURE_IMAGE_STAMP_INVALID,
    FAILURE_IMAGE_STAMP_STALE,
    FAILURE_IMAGE_TIMEOUT,
    FAILURE_FRESH_TUPLE_UNAVAILABLE,
    FAILURE_OBSERVATION_EFFECT,
    FAILURE_OBSERVER_CLOCK,
    FAILURE_ROS_UNAVAILABLE,
    FAILURE_SCAN_FRAME_EMPTY,
    FAILURE_SCAN_FRAME_MISMATCH,
    FAILURE_SCAN_STAMP_FUTURE,
    FAILURE_SCAN_STAMP_INVALID,
    FAILURE_SCAN_STAMP_STALE,
    FAILURE_SCAN_TIMEOUT,
    HeaderSample,
    SENSOR_TIMING_READINESS_SCHEMA_VERSION,
    SensorTimingEffect,
    SensorTimingEvidence,
    SensorTimingReadinessConfig,
    SensorTimingReadinessError,
    SensorTimingReadinessResult,
    evaluate_sensor_timing_readiness,
)


try:  # pragma: no cover - exercised only on a ROS2 host.
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, CompressedImage, LaserScan
except ImportError:  # pragma: no cover - keeps offline tests ROS-free.
    rclpy = None
    Node = object
    qos_profile_sensor_data = None
    CameraInfo = None
    CompressedImage = None
    LaserScan = None


def observe_sensor_timing_readiness(
    config: SensorTimingReadinessConfig,
    *,
    sensor_timing_effect: SensorTimingEffect | None = None,
) -> SensorTimingReadinessResult:
    """Collect and evaluate bounded pre-motion header timing evidence."""

    selected = config.validated()
    effect = sensor_timing_effect or _observe_with_ros
    try:
        evidence = effect(selected)
    except Exception as exc:
        evidence = SensorTimingEvidence(
            observed_at_ns=time.time_ns(),
            observer_failure_code=FAILURE_OBSERVATION_EFFECT,
            observer_error=f"{type(exc).__name__}: {exc}",
        )
    return evaluate_sensor_timing_readiness(selected, evidence)


def _stamp_to_ns(stamp: object) -> int:
    return int(getattr(stamp, "sec")) * 1_000_000_000 + int(
        getattr(stamp, "nanosec")
    )


def _header_sample(message: object, receipt_ns: int) -> HeaderSample:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    try:
        stamp_ns = _stamp_to_ns(stamp)
    except (AttributeError, TypeError, ValueError):
        stamp_ns = None
    return HeaderSample(
        stamp_ns=stamp_ns,
        frame_id=str(getattr(header, "frame_id", "")).strip("/"),
        receipt_ns=receipt_ns,
    )


class _SensorTimingReadinessNode(Node):  # pragma: no cover - ROS host only.
    def __init__(self, config: SensorTimingReadinessConfig) -> None:
        super().__init__("autonomous_sensor_timing_readiness")
        self.config = config
        self.images: deque[HeaderSample] = deque(maxlen=config.sample_capacity)
        self.camera_infos: deque[HeaderSample] = deque(
            maxlen=config.sample_capacity
        )
        self.scans: deque[HeaderSample] = deque(maxlen=config.sample_capacity)
        self.ready_evidence: SensorTimingEvidence | None = None
        self.create_subscription(
            CompressedImage,
            config.image_topic,
            self._on_image,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            config.camera_info_topic,
            self._on_camera_info,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            LaserScan,
            config.scan_topic,
            self._on_scan,
            qos_profile_sensor_data,
        )

    def _on_image(self, message: object) -> None:
        self.images.append(
            _header_sample(message, self.get_clock().now().nanoseconds)
        )
        self.poll()

    def _on_camera_info(self, message: object) -> None:
        self.camera_infos.append(
            _header_sample(message, self.get_clock().now().nanoseconds)
        )
        self.poll()

    def _on_scan(self, message: object) -> None:
        self.scans.append(
            _header_sample(message, self.get_clock().now().nanoseconds)
        )
        self.poll()

    def evidence(self, *, timed_out: bool = False) -> SensorTimingEvidence:
        return SensorTimingEvidence(
            observed_at_ns=self.get_clock().now().nanoseconds,
            image_samples=tuple(self.images),
            camera_info_samples=tuple(self.camera_infos),
            scan_samples=tuple(self.scans),
            timed_out=timed_out,
        )

    def poll(self) -> None:
        if self.ready_evidence is not None:
            return
        evidence = self.evidence()
        if evaluate_sensor_timing_readiness(self.config, evidence).ready:
            self.ready_evidence = evidence


def _observe_with_ros(
    config: SensorTimingReadinessConfig,
) -> SensorTimingEvidence:  # pragma: no cover - ROS host only.
    if (
        rclpy is None
        or CameraInfo is None
        or CompressedImage is None
        or LaserScan is None
    ):
        return SensorTimingEvidence(
            observed_at_ns=time.time_ns(),
            observer_failure_code=FAILURE_ROS_UNAVAILABLE,
            observer_error="ROS2 sensor message dependencies are unavailable",
        )

    owns_context = not rclpy.ok()
    if owns_context:
        rclpy.init(args=None)
    node = None
    try:
        node = _SensorTimingReadinessNode(config)
        deadline = time.monotonic() + config.timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if node.ready_evidence is not None:
                return node.ready_evidence
            remaining_sec = max(0.0, deadline - time.monotonic())
            rclpy.spin_once(
                node,
                timeout_sec=min(config.poll_interval_sec, remaining_sec),
            )
            node.poll()
        return node.evidence(timed_out=True)
    finally:
        if node is not None:
            node.destroy_node()
        if owns_context and rclpy.ok():
            rclpy.shutdown()


__all__ = [
    "DEFAULT_MAX_CAMERA_INFO_AGE_SEC",
    "DEFAULT_MAX_CAMERA_INFO_IMAGE_SKEW_SEC",
    "DEFAULT_MAX_FUTURE_TIMESTAMP_SEC",
    "DEFAULT_MAX_IMAGE_SCAN_SKEW_SEC",
    "DEFAULT_MAX_SENSOR_AGE_SEC",
    "DEFAULT_SENSOR_TIMING_SAMPLE_CAPACITY",
    "DEFAULT_SENSOR_TIMING_TIMEOUT_SEC",
    "FAILURE_CAMERA_INFO_FRAME_EMPTY",
    "FAILURE_CAMERA_INFO_FRAME_MISMATCH",
    "FAILURE_CAMERA_INFO_IMAGE_SKEW",
    "FAILURE_CAMERA_INFO_STAMP_FUTURE",
    "FAILURE_CAMERA_INFO_STAMP_INVALID",
    "FAILURE_CAMERA_INFO_STAMP_STALE",
    "FAILURE_CAMERA_INFO_TIMEOUT",
    "FAILURE_IMAGE_FRAME_EMPTY",
    "FAILURE_IMAGE_FRAME_MISMATCH",
    "FAILURE_IMAGE_SCAN_SKEW",
    "FAILURE_IMAGE_STAMP_FUTURE",
    "FAILURE_IMAGE_STAMP_INVALID",
    "FAILURE_IMAGE_STAMP_STALE",
    "FAILURE_IMAGE_TIMEOUT",
    "FAILURE_FRESH_TUPLE_UNAVAILABLE",
    "FAILURE_OBSERVATION_EFFECT",
    "FAILURE_OBSERVER_CLOCK",
    "FAILURE_ROS_UNAVAILABLE",
    "FAILURE_SCAN_FRAME_EMPTY",
    "FAILURE_SCAN_FRAME_MISMATCH",
    "FAILURE_SCAN_STAMP_FUTURE",
    "FAILURE_SCAN_STAMP_INVALID",
    "FAILURE_SCAN_STAMP_STALE",
    "FAILURE_SCAN_TIMEOUT",
    "HeaderSample",
    "SENSOR_TIMING_READINESS_SCHEMA_VERSION",
    "SensorTimingEffect",
    "SensorTimingEvidence",
    "SensorTimingReadinessConfig",
    "SensorTimingReadinessError",
    "SensorTimingReadinessResult",
    "evaluate_sensor_timing_readiness",
    "observe_sensor_timing_readiness",
]
