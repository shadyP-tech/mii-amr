"""ROS collection facade for exact-time LiDAR transform readiness.

The stable public API is re-exported from the ROS-free contract module.  This
facade adds only bounded evidence collection; it never publishes motion,
requests operator input, or starts subprocesses.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time

from scripts.aufgabe04.real_robot.autonomous_observation_tf_contract import (
    FAILURE_OBSERVATION_EFFECT,
    FAILURE_OBSERVER_CLOCK,
    FAILURE_ROS_UNAVAILABLE,
    FAILURE_SCAN_FRAME_EMPTY,
    FAILURE_SCAN_FRAME_MISMATCH,
    FAILURE_SCAN_STAMP_FUTURE,
    FAILURE_SCAN_STAMP_INVALID,
    FAILURE_SCAN_STAMP_STALE,
    FAILURE_SCAN_TIMEOUT,
    FAILURE_TRANSFORM_FRAME_MISMATCH,
    FAILURE_TRANSFORM_NOT_EXACT_TIME,
    FAILURE_TRANSFORM_PAYLOAD_INVALID,
    FAILURE_TRANSFORM_TIMING,
    FAILURE_TRANSFORM_UNAVAILABLE,
    OBSERVATION_TF_READINESS_SCHEMA_VERSION,
    ObservationEffect,
    ObservationTfEvidence,
    ObservationTfReadinessConfig,
    ObservationTfReadinessError,
    ObservationTfReadinessResult,
    evaluate_observation_tf_readiness,
)


try:  # pragma: no cover - exercised only on a ROS2 host.
    import rclpy
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer, TransformException, TransformListener
except ImportError:  # pragma: no cover - keeps offline tests ROS-free.
    rclpy = None
    Duration = None
    Node = object
    qos_profile_sensor_data = None
    Time = None
    LaserScan = None
    Buffer = None
    TransformException = Exception
    TransformListener = None


def observe_observation_tf_readiness(
    config: ObservationTfReadinessConfig,
    *,
    observation_effect: ObservationEffect | None = None,
) -> ObservationTfReadinessResult:
    """Collect and evaluate bounded readiness evidence.

    ``observation_effect`` is the test seam.  Production callers omit it and
    use the ROS2 subscriber/tf2 adapter below.
    """

    selected = config.validated()
    effect = observation_effect or _observe_with_ros
    try:
        evidence = effect(selected)
    except Exception as exc:  # Fail closed and preserve the collector error.
        evidence = ObservationTfEvidence(
            observed_at_ns=time.time_ns(),
            scan_received=False,
            observer_failure_code=FAILURE_OBSERVATION_EFFECT,
            observer_error=f"{type(exc).__name__}: {exc}",
        )
    return evaluate_observation_tf_readiness(selected, evidence)


@dataclass(frozen=True)
class _PendingScan:
    frame: str
    stamp_ns: int
    query_time: object


def _stamp_to_ns(stamp: object) -> int:
    return int(getattr(stamp, "sec")) * 1_000_000_000 + int(
        getattr(stamp, "nanosec")
    )


def _transform_pose_evidence(transform: object) -> dict[str, float]:
    """Extract audit fields without allowing malformed TF to look ready."""

    translation = getattr(getattr(transform, "transform"), "translation")
    rotation = getattr(getattr(transform, "transform"), "rotation")
    quaternion = tuple(
        float(value)
        for value in (rotation.x, rotation.y, rotation.z, rotation.w)
    )
    yaw_rad = math.atan2(
        2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
        1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
    )
    return {
        "transform_x_m": float(translation.x),
        "transform_y_m": float(translation.y),
        "transform_z_m": float(translation.z),
        "transform_yaw_rad": float(yaw_rad),
        "transform_quaternion_norm": math.sqrt(
            sum(value * value for value in quaternion)
        ),
    }


class _ObservationTfReadinessNode(Node):  # pragma: no cover - ROS host only.
    def __init__(self, config: ObservationTfReadinessConfig) -> None:
        super().__init__("autonomous_observation_tf_readiness")
        self.config = config
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pending: _PendingScan | None = None
        self.latest_evidence: ObservationTfEvidence | None = None
        self.ready_evidence: ObservationTfEvidence | None = None
        self.create_subscription(
            LaserScan,
            config.scan_topic,
            self._scan_callback,
            qos_profile_sensor_data,
        )

    def _scan_callback(self, message: object) -> None:
        header = getattr(message, "header", None)
        frame = str(getattr(header, "frame_id", ""))
        stamp = getattr(header, "stamp", None)
        try:
            stamp_ns = _stamp_to_ns(stamp)
        except (AttributeError, TypeError, ValueError):
            stamp_ns = 0
        self.latest_evidence = ObservationTfEvidence(
            observed_at_ns=self.get_clock().now().nanoseconds,
            scan_received=True,
            scan_frame=frame,
            scan_stamp_ns=stamp_ns,
        )
        if frame != self.config.expected_scan_frame or stamp_ns <= 0:
            self.pending = None
            return
        try:
            query_time = Time.from_msg(stamp)
        except (TypeError, ValueError) as exc:
            self.latest_evidence = replace(
                self.latest_evidence,
                transform_error=f"invalid exact-time query: {exc}",
            )
            self.pending = None
            return
        self.pending = _PendingScan(frame, stamp_ns, query_time)
        self.poll_transform()

    def poll_transform(self) -> None:
        pending = self.pending
        if pending is None or self.ready_evidence is not None:
            return
        now_ns = self.get_clock().now().nanoseconds
        base = ObservationTfEvidence(
            observed_at_ns=now_ns,
            scan_received=True,
            scan_frame=pending.frame,
            scan_stamp_ns=pending.stamp_ns,
            transform_checked=True,
            transform_target_frame=self.config.target_frame,
            transform_source_frame=pending.frame,
            transform_query_stamp_ns=pending.stamp_ns,
        )
        preliminary = evaluate_observation_tf_readiness(self.config, base)
        if preliminary.failure_code not in {FAILURE_TRANSFORM_UNAVAILABLE}:
            self.latest_evidence = base
            return
        zero_timeout = Duration(seconds=0.0)
        try:
            available = self.tf_buffer.can_transform(
                self.config.target_frame,
                pending.frame,
                pending.query_time,
                timeout=zero_timeout,
            )
            if not available:
                self.latest_evidence = base
                return
            transform = self.tf_buffer.lookup_transform(
                self.config.target_frame,
                pending.frame,
                pending.query_time,
                timeout=zero_timeout,
            )
        except TransformException as exc:
            self.latest_evidence = replace(base, transform_error=str(exc))
            return
        header = getattr(transform, "header", None)
        transform_stamp = getattr(header, "stamp", None)
        try:
            transform_stamp_ns = _stamp_to_ns(transform_stamp)
        except (AttributeError, TypeError, ValueError):
            transform_stamp_ns = None
        evidence = replace(
            base,
            transform_available=True,
            transform_target_frame=str(getattr(header, "frame_id", "")),
            transform_source_frame=str(
                getattr(transform, "child_frame_id", "")
            ),
            transform_stamp_ns=transform_stamp_ns,
            **_transform_pose_evidence(transform),
        )
        self.latest_evidence = evidence
        if evaluate_observation_tf_readiness(self.config, evidence).ready:
            self.ready_evidence = evidence


def _observe_with_ros(
    config: ObservationTfReadinessConfig,
) -> ObservationTfEvidence:  # pragma: no cover - ROS host only.
    if (
        rclpy is None
        or LaserScan is None
        or Buffer is None
        or TransformListener is None
    ):
        return ObservationTfEvidence(
            observed_at_ns=time.time_ns(),
            scan_received=False,
            observer_failure_code=FAILURE_ROS_UNAVAILABLE,
            observer_error="ROS2 LaserScan/tf2 dependencies are unavailable",
        )

    owns_context = not rclpy.ok()
    if owns_context:
        rclpy.init(args=None)
    node = None
    try:
        node = _ObservationTfReadinessNode(config)
        deadline = time.monotonic() + config.timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if node.ready_evidence is not None:
                return node.ready_evidence
            remaining_sec = max(0.0, deadline - time.monotonic())
            rclpy.spin_once(
                node,
                timeout_sec=min(config.poll_interval_sec, remaining_sec),
            )
            node.poll_transform()
        evidence = node.latest_evidence
        if evidence is None:
            evidence = ObservationTfEvidence(
                observed_at_ns=node.get_clock().now().nanoseconds,
                scan_received=False,
            )
        return replace(evidence, timed_out=True)
    finally:
        if node is not None:
            node.destroy_node()
        if owns_context and rclpy.ok():
            rclpy.shutdown()


__all__ = [
    "FAILURE_OBSERVATION_EFFECT",
    "FAILURE_OBSERVER_CLOCK",
    "FAILURE_ROS_UNAVAILABLE",
    "FAILURE_SCAN_FRAME_EMPTY",
    "FAILURE_SCAN_FRAME_MISMATCH",
    "FAILURE_SCAN_STAMP_FUTURE",
    "FAILURE_SCAN_STAMP_INVALID",
    "FAILURE_SCAN_STAMP_STALE",
    "FAILURE_SCAN_TIMEOUT",
    "FAILURE_TRANSFORM_FRAME_MISMATCH",
    "FAILURE_TRANSFORM_NOT_EXACT_TIME",
    "FAILURE_TRANSFORM_PAYLOAD_INVALID",
    "FAILURE_TRANSFORM_TIMING",
    "FAILURE_TRANSFORM_UNAVAILABLE",
    "OBSERVATION_TF_READINESS_SCHEMA_VERSION",
    "ObservationEffect",
    "ObservationTfEvidence",
    "ObservationTfReadinessConfig",
    "ObservationTfReadinessError",
    "ObservationTfReadinessResult",
    "evaluate_observation_tf_readiness",
    "observe_observation_tf_readiness",
]
