"""Observe Aufgabe 04 station stands from live LiDAR without publishing motion."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.map_io import freeze_map_bundle
from scripts.aufgabe04.navigation.odom_execution_certificate import (
    OdomExecutionCertificate,
    PlanarTransform2D,
    load_odom_execution_certificate,
    odom_execution_certificate_sha256,
)
from scripts.aufgabe04.navigation.ros_runtime_config import RuntimeConfig, resolve_runtime_config
from scripts.aufgabe04.perception.lidar_stand_detector import detect_stand_candidates_from_scan
from scripts.aufgabe04.perception.models import LidarStandDetectorConfig
from scripts.aufgabe04.perception.stand_confirmation import (
    StandConfirmationAccumulator,
    StandConfirmationConfig,
)
from scripts.aufgabe04.perception.stand_observation import (
    DEFAULT_OBSERVATION_TIMING_LIMITS,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_ID_SCOPE_RUNTIME_KEY,
    RUNTIME_TIMING_LIMITS_KEY,
    TF_LOOKUP_MODE_SCAN_TIME_EXACT,
    ObservationProvenance,
    ObservationTimingLimits,
    PlanarTransform,
    load_observation_jsonl,
    observation_timing_limits_from_runtime_config,
    observer_clock_name,
    observations_from_candidates,
    validated_observation_timing,
    validated_observation_stream_clock,
    validated_observation_id_scope,
    validated_scan_age_sec,
    write_observation_jsonl,
)

try:  # pragma: no cover - exercised on ROS hosts.
    import rclpy
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer, TransformException, TransformListener
except ImportError:  # pragma: no cover - keeps offline tests ROS-free.
    rclpy = None
    Duration = None
    Node = object
    Parameter = None
    qos_profile_sensor_data = None
    Time = None
    LaserScan = None
    Buffer = None
    TransformException = Exception
    TransformListener = None


OBSERVER_VERSION = "aufgabe04-stand-explorer-observe-only-v5-scan-time-tf"
FROZEN_ODOM_OBSERVER_VERSION = (
    "aufgabe04-stand-explorer-observe-only-v6-frozen-odom-scan-time-tf"
)
FROZEN_FRAME_EVIDENCE_KEY = "frozen_odom_observation_geometry"
DEFAULT_OUTPUT_JSONL = Path("results/aufgabe04/detected_stations/stand_observations.jsonl")


@dataclass(frozen=True)
class _PendingScan:
    message: object
    scan_frame: str
    scan_stamp_sec: float
    query_time: object
    deadline_monotonic_sec: float


@dataclass(frozen=True)
class FrozenObserverFrame:
    """Immutable evidence used to map odom-frame scan poses into ``map``.

    The certificate's convention is ``p_map = R * p_odom + translation``.
    Keeping this value outside the ROS node makes the observation geometry
    independently testable and prevents live ``map`` corrections from steering
    the stand-observation coordinates during an odom-certified leg.
    """

    certificate_path: Path
    certificate: OdomExecutionCertificate
    certificate_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.certificate, OdomExecutionCertificate):
            raise ValueError("certificate must be an OdomExecutionCertificate")
        expected_sha256 = odom_execution_certificate_sha256(self.certificate)
        if self.certificate_sha256 != expected_sha256:
            raise ValueError("odom execution certificate hash mismatch")
        object.__setattr__(self, "certificate_path", Path(self.certificate_path))

    def runtime_evidence(self) -> dict[str, object]:
        certificate = self.certificate
        return {
            "schema_version": 1,
            "mode": "frozen_map_from_odom",
            "odom_execution_certificate_path": str(self.certificate_path),
            "odom_execution_certificate_sha256": self.certificate_sha256,
            "source_frames": {
                "map_frame": certificate.map_frame,
                "odom_frame": certificate.odom_frame,
                "base_frame": certificate.base_frame,
            },
            "scan_tf_target_frame": certificate.odom_frame,
            "map_from_odom": {
                "x_m": certificate.map_from_odom.x_m,
                "y_m": certificate.map_from_odom.y_m,
                "yaw_rad": certificate.map_from_odom.yaw_rad,
            },
            "transform_stamp_sec": certificate.transform_stamp_sec,
            "transform_capture_time_sec": (
                certificate.transform_capture_time_sec
            ),
            "source_map_route_sha256": certificate.source_map_route_sha256,
            "transformed_odom_route_sha256": (
                certificate.transformed_odom_route_sha256
            ),
        }


def load_frozen_observer_frame(
    certificate_path: Path,
    *,
    map_frame: str,
    odom_frame: str,
    base_frame: str,
) -> FrozenObserverFrame:
    """Load one certificate and reject any configured-frame mismatch."""

    path = Path(certificate_path)
    try:
        certificate = load_odom_execution_certificate(path)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"invalid odom execution certificate {path}: {exc}"
        ) from exc
    expected_frames = {
        "map_frame": map_frame,
        "odom_frame": odom_frame,
        "base_frame": base_frame,
    }
    for field, expected in expected_frames.items():
        certified = getattr(certificate, field)
        if certified != expected:
            raise ValueError(
                f"odom execution certificate {field} mismatch: "
                f"certified={certified!r}, configured={expected!r}"
            )
    return FrozenObserverFrame(
        certificate_path=path.resolve(),
        certificate=certificate,
        certificate_sha256=odom_execution_certificate_sha256(certificate),
    )


def compose_frozen_scan_pose_in_map(
    *,
    odom_from_scan: PlanarTransform,
    map_from_odom: PlanarTransform2D,
) -> PlanarTransform:
    """Compose ``map<-odom`` with ``odom<-scan`` using planar SE(2)."""

    # Reconstruct both inputs through the certificate value type so forged,
    # non-finite, or non-normalized inputs cannot enter persisted geometry.
    odom_from_scan_value = PlanarTransform2D(
        odom_from_scan.x_m,
        odom_from_scan.y_m,
        odom_from_scan.yaw_rad,
    )
    map_from_odom_value = PlanarTransform2D(
        map_from_odom.x_m,
        map_from_odom.y_m,
        map_from_odom.yaw_rad,
    )
    cosine = math.cos(map_from_odom_value.yaw_rad)
    sine = math.sin(map_from_odom_value.yaw_rad)
    map_from_scan = PlanarTransform2D(
        x_m=(
            map_from_odom_value.x_m
            + cosine * odom_from_scan_value.x_m
            - sine * odom_from_scan_value.y_m
        ),
        y_m=(
            map_from_odom_value.y_m
            + sine * odom_from_scan_value.x_m
            + cosine * odom_from_scan_value.y_m
        ),
        yaw_rad=(
            map_from_odom_value.yaw_rad + odom_from_scan_value.yaw_rad
        ),
    )
    return PlanarTransform(
        x_m=map_from_scan.x_m,
        y_m=map_from_scan.y_m,
        yaw_rad=map_from_scan.yaw_rad,
    )


def _observation_tf_target_frame(
    runtime: RuntimeConfig,
    frozen_frame: FrozenObserverFrame | None,
) -> str:
    return runtime.map_frame if frozen_frame is None else runtime.odom_frame


def _observer_version(frozen_frame: FrozenObserverFrame | None) -> str:
    return OBSERVER_VERSION if frozen_frame is None else FROZEN_ODOM_OBSERVER_VERSION


def _validated_frozen_tf_frames(
    transform,
    *,
    expected_parent_frame: str,
    expected_child_frame: str,
) -> None:
    """Fail closed if tf2 returns a transform with unexpected frame labels."""

    parent_frame = getattr(getattr(transform, "header", None), "frame_id", None)
    child_frame = getattr(transform, "child_frame_id", None)
    if parent_frame != expected_parent_frame:
        raise ValueError(
            "exact-time TF parent frame mismatch: "
            f"expected={expected_parent_frame!r}, observed={parent_frame!r}"
        )
    if child_frame != expected_child_frame:
        raise ValueError(
            "exact-time TF child frame mismatch: "
            f"expected={expected_child_frame!r}, observed={child_frame!r}"
        )


def _validated_planar_pose_from_tf(transform) -> PlanarTransform:
    """Extract a finite planar pose from a normalized TF quaternion."""

    try:
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        x_m = float(translation.x)
        y_m = float(translation.y)
        quaternion = tuple(
            float(value)
            for value in (rotation.x, rotation.y, rotation.z, rotation.w)
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("exact-time TF payload is malformed") from exc
    if not all(math.isfinite(value) for value in (x_m, y_m, *quaternion)):
        raise ValueError("exact-time TF payload is non-finite")
    quaternion_norm_squared = sum(value * value for value in quaternion)
    if abs(quaternion_norm_squared - 1.0) > 1e-3:
        raise ValueError("exact-time TF quaternion is not normalized")
    value = PlanarTransform2D(x_m, y_m, _yaw_from_quaternion(rotation))
    return PlanarTransform(value.x_m, value.y_m, value.yaw_rad)


def _require_ros() -> None:
    if rclpy is None:
        raise RuntimeError("ROS2 Python packages are not available in this environment")


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0


def _transform_time_for_scan_stamp(stamp):
    """Convert the LaserScan header stamp into the exact tf2 query time."""

    if Time is None:
        _require_ros()
    return Time.from_msg(stamp)


def _node_parameter_overrides(allow_sim_time: bool):
    """Apply the CLI simulation-time switch to the actual ROS node clock."""

    if not allow_sim_time:
        return []
    if Parameter is None:
        _require_ros()
    return [Parameter("use_sim_time", Parameter.Type.BOOL, True)]


def _timing_limits_from_args(args) -> ObservationTimingLimits:
    return ObservationTimingLimits(
        max_scan_age_sec=args.max_scan_age_sec,
        max_future_timestamp_sec=args.max_future_timestamp_sec,
        max_tf_age_sec=args.max_tf_age_sec,
        max_tf_scan_skew_sec=args.max_tf_scan_skew_sec,
    ).validated()


def _validate_append_stream(
    path: Path,
    *,
    required_observer_clock: str,
    timing_limits: ObservationTimingLimits,
    frozen_frame: FrozenObserverFrame | None = None,
) -> None:
    """Fail before appending to a legacy or different-clock JSONL artifact."""

    if not path.exists():
        return
    observations = load_observation_jsonl(path)
    validated_observation_stream_clock(
        observations,
        required_observer_clock=required_observer_clock,
    )
    for observation in observations:
        existing_limits = observation_timing_limits_from_runtime_config(
            observation.provenance.runtime_config
        )
        if existing_limits != timing_limits:
            raise ValueError(
                "observation artifact uses incompatible producer timing limits"
            )
        existing_frozen_evidence = observation.provenance.runtime_config.get(
            FROZEN_FRAME_EVIDENCE_KEY
        )
        expected_frozen_evidence = (
            None if frozen_frame is None else frozen_frame.runtime_evidence()
        )
        if existing_frozen_evidence != expected_frozen_evidence:
            raise ValueError(
                "observation artifact uses incompatible observation geometry"
            )


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class StandExplorerNode(Node):  # pragma: no cover - requires ROS runtime.
    def __init__(self, args) -> None:
        super().__init__(
            "aufgabe04_stand_explorer_observer",
            parameter_overrides=_node_parameter_overrides(args.allow_sim_time),
        )
        self.args = args
        self.observation_id_scope = validated_observation_id_scope(
            getattr(args, "observation_id_scope", None)
        )
        self.timing_limits = _timing_limits_from_args(args)
        self.runtime = resolve_runtime_config(
            RuntimeConfig(
                namespace=args.namespace,
                scan_topic=args.scan_topic,
                odom_topic=args.odom_topic,
                cmd_vel_topic=args.cmd_vel_topic,
                amcl_topic=args.amcl_topic,
                map_frame=args.map_frame,
                odom_frame=args.odom_frame,
                base_frame=args.base_frame,
                localization_source=args.localization_source,
                use_sim_time=args.allow_sim_time,
            )
        )
        certificate_path = getattr(args, "odom_execution_certificate_json", None)
        self.frozen_observer_frame = (
            None
            if certificate_path is None
            else load_frozen_observer_frame(
                certificate_path,
                map_frame=self.runtime.map_frame,
                odom_frame=self.runtime.odom_frame,
                base_frame=self.runtime.base_frame,
            )
        )
        self.output_jsonl = args.output_jsonl
        self.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        _validate_append_stream(
            self.output_jsonl,
            required_observer_clock=observer_clock_name(
                use_sim_time=self.runtime.use_sim_time
            ),
            timing_limits=self.timing_limits,
            frozen_frame=self.frozen_observer_frame,
        )
        self.map_bundle = (
            None
            if args.map_yaml is None
            else freeze_map_bundle(
                args.map_yaml,
                semantic_map_id=args.semantic_map_id or args.map_yaml.stem,
                planning_frame=args.map_frame,
            )
        )
        self.detector_config = LidarStandDetectorConfig(
            min_range_m=args.min_range_m,
            max_range_m=args.max_range_m,
            max_cluster_gap_m=args.max_cluster_gap_m,
            min_cluster_points=args.min_cluster_points,
            min_width_m=args.min_width_m,
            max_width_m=args.max_width_m,
        )
        self.accumulator = StandConfirmationAccumulator(
            config=StandConfirmationConfig(
                merge_distance_m=args.merge_distance_m,
                min_hits=args.min_hits,
                max_age_sec=args.max_observation_age_sec,
                min_confidence=args.min_confidence,
                min_boundary_clearance_m=args.min_boundary_clearance_m,
            )
        )
        self.observation_count = 0
        self.processed_scan_count = 0
        self.detected_candidate_count = 0
        self.accepted_observation_count = 0
        self.last_confirmed_stand_count = 0
        self.started_unix_sec = time.time()
        self.last_processed_scan_stamp_sec: float | None = None
        self.last_scan_pose_map: dict[str, float] | None = None
        self.observation_enabled = True
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pending_scans: deque[_PendingScan] = deque()
        self.create_subscription(
            LaserScan,
            self.runtime.scan_topic,
            self._scan_callback,
            qos_profile_sensor_data,
        )
        self.create_timer(
            min(0.01, args.tf_timeout_sec),
            self._drain_pending_scans,
        )
        self.get_logger().info(
            "observe-only stand explorer listening on "
            f"{self.runtime.scan_topic}; output={self.output_jsonl}; "
            "geometry="
            + (
                "live_map_from_scan"
                if self.frozen_observer_frame is None
                else "frozen_map_from_odom_plus_exact_odom_from_scan"
            )
        )

    def set_observation_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self.observation_enabled:
            return
        self.observation_enabled = enabled
        if not enabled:
            self.pending_scans.clear()
        state = "enabled" if enabled else "paused for localization readiness"
        self.get_logger().info(f"stand observation {state}")

    def _scan_callback(self, msg) -> None:
        if not self.observation_enabled:
            return
        scan_frame = msg.header.frame_id
        if not scan_frame:
            self.get_logger().warn("dropping scan without header.frame_id")
            return
        scan_stamp_sec = _stamp_to_sec(msg.header.stamp)
        observer_clock_sec = _stamp_to_sec(self.get_clock().now().to_msg())
        try:
            # Reject zero, invalid, stale, or materially future-dated scans
            # before asking tf2 to resolve them.
            validated_scan_age_sec(
                observer_clock_sec=observer_clock_sec,
                scan_stamp_sec=scan_stamp_sec,
                max_scan_age_sec=self.timing_limits.max_scan_age_sec,
                max_future_timestamp_sec=(
                    self.timing_limits.max_future_timestamp_sec
                ),
            )
        except ValueError as exc:
            self.get_logger().warn(f"dropping scan: {exc}")
            return
        try:
            query_time = _transform_time_for_scan_stamp(msg.header.stamp)
        except (TypeError, ValueError) as exc:
            self.get_logger().warn(f"dropping scan with invalid timestamp: {exc}")
            return

        if len(self.pending_scans) >= self.args.pending_scan_limit:
            dropped = self.pending_scans.popleft()
            self.get_logger().warn(
                "dropping oldest pending scan because the exact-time TF queue "
                f"is full: stamp={dropped.scan_stamp_sec:.9f}"
            )
        self.pending_scans.append(
            _PendingScan(
                message=msg,
                scan_frame=scan_frame,
                scan_stamp_sec=scan_stamp_sec,
                query_time=query_time,
                deadline_monotonic_sec=time.monotonic() + self.args.tf_timeout_sec,
            )
        )
        # This retry is deliberately nonblocking. If TF has not arrived yet,
        # the timer retries after the executor has serviced the TF listener.
        self._drain_pending_scans()

    def _drain_pending_scans(self) -> None:
        pending_count = len(self.pending_scans)
        for _index in range(pending_count):
            pending = self.pending_scans.popleft()
            tf_target_frame = _observation_tf_target_frame(
                self.runtime,
                getattr(self, "frozen_observer_frame", None),
            )
            if time.monotonic() > pending.deadline_monotonic_sec:
                self.get_logger().warn(
                    f"dropping scan: exact-time {tf_target_frame}<-"
                    f"{pending.scan_frame} TF timed out for stamp "
                    f"{pending.scan_stamp_sec:.9f}"
                )
                continue

            zero_timeout = Duration(seconds=0.0)
            try:
                available = self.tf_buffer.can_transform(
                    tf_target_frame,
                    pending.scan_frame,
                    pending.query_time,
                    timeout=zero_timeout,
                )
            except TransformException as exc:
                self.get_logger().warn(
                    f"exact-time {tf_target_frame}<-"
                    f"{pending.scan_frame} TF check failed: {exc}"
                )
                self.pending_scans.append(pending)
                continue
            if not available:
                self.pending_scans.append(pending)
                continue

            try:
                transform = self.tf_buffer.lookup_transform(
                    tf_target_frame,
                    pending.scan_frame,
                    pending.query_time,
                    timeout=zero_timeout,
                )
            except TransformException as exc:
                # The buffer can change between can_transform and lookup.
                self.get_logger().warn(
                    f"exact-time {tf_target_frame}<-"
                    f"{pending.scan_frame} TF lookup raced: {exc}"
                )
                self.pending_scans.append(pending)
                continue
            self._process_scan_with_transform(pending, transform)

    def _process_scan_with_transform(self, pending: _PendingScan, transform) -> None:
        msg = pending.message
        scan_frame = pending.scan_frame
        scan_stamp_sec = pending.scan_stamp_sec
        frozen_frame = getattr(self, "frozen_observer_frame", None)

        observer_clock_sec = _stamp_to_sec(self.get_clock().now().to_msg())
        try:
            tf_target_frame = _observation_tf_target_frame(
                self.runtime,
                frozen_frame,
            )
            _validated_frozen_tf_frames(
                transform,
                expected_parent_frame=tf_target_frame,
                expected_child_frame=scan_frame,
            )
            tf_stamp_sec = _stamp_to_sec(transform.header.stamp)
        except (AttributeError, TypeError, ValueError) as exc:
            self.get_logger().warn(f"dropping scan: invalid exact-time TF: {exc}")
            return
        try:
            timing = validated_observation_timing(
                observer_clock_sec=observer_clock_sec,
                scan_stamp_sec=scan_stamp_sec,
                tf_stamp_sec=tf_stamp_sec,
                max_scan_age_sec=self.timing_limits.max_scan_age_sec,
                max_future_timestamp_sec=(
                    self.timing_limits.max_future_timestamp_sec
                ),
                max_tf_age_sec=self.timing_limits.max_tf_age_sec,
                max_tf_scan_skew_sec=(
                    self.timing_limits.max_tf_scan_skew_sec
                ),
            )
        except ValueError as exc:
            self.get_logger().warn(f"dropping scan: {exc}")
            return

        try:
            exact_time_tf_pose = _validated_planar_pose_from_tf(transform)
            if frozen_frame is None:
                scan_pose_in_map = exact_time_tf_pose
            else:
                scan_pose_in_map = compose_frozen_scan_pose_in_map(
                    odom_from_scan=exact_time_tf_pose,
                    map_from_odom=frozen_frame.certificate.map_from_odom,
                )
        except ValueError as exc:
            self.get_logger().warn(f"dropping scan: {exc}")
            return
        self.processed_scan_count += 1
        self.last_processed_scan_stamp_sec = scan_stamp_sec
        self.last_scan_pose_map = {
            "x_m": scan_pose_in_map.x_m,
            "y_m": scan_pose_in_map.y_m,
            "yaw_rad": scan_pose_in_map.yaw_rad,
        }
        candidates = detect_stand_candidates_from_scan(
            msg.ranges,
            angle_min_rad=msg.angle_min,
            angle_increment_rad=msg.angle_increment,
            config=self.detector_config,
        )
        self.detected_candidate_count += len(candidates)
        if not candidates:
            return

        runtime_config = dict(self.runtime.as_log_dict())
        runtime_config[RUNTIME_TIMING_LIMITS_KEY] = self.timing_limits.as_dict()
        if self.observation_id_scope is not None:
            runtime_config[OBSERVATION_ID_SCOPE_RUNTIME_KEY] = (
                self.observation_id_scope
            )
        if frozen_frame is not None:
            runtime_config[FROZEN_FRAME_EVIDENCE_KEY] = (
                frozen_frame.runtime_evidence()
            )
        provenance = ObservationProvenance(
            schema_version=OBSERVATION_SCHEMA_VERSION,
            observer_version=_observer_version(frozen_frame),
            resolved_scan_topic=self.runtime.scan_topic,
            scan_frame=scan_frame,
            map_frame=self.runtime.map_frame,
            base_frame=self.runtime.base_frame,
            localization_source=self.runtime.localization_source,
            scan_stamp_sec=scan_stamp_sec,
            tf_lookup_stamp_sec=tf_stamp_sec,
            tf_age_sec=timing.tf_age_sec,
            runtime_config=runtime_config,
            observer_clock=observer_clock_name(
                use_sim_time=self.runtime.use_sim_time
            ),
            observer_clock_sec=observer_clock_sec,
            scan_age_sec=timing.scan_age_sec,
            tf_scan_skew_sec=timing.tf_scan_skew_sec,
            tf_query_stamp_sec=scan_stamp_sec,
            tf_lookup_mode=TF_LOOKUP_MODE_SCAN_TIME_EXACT,
            map_yaml=str(self.args.map_yaml or ""),
            map_yaml_sha256=(
                "" if self.map_bundle is None else self.map_bundle.yaml_sha256
            ),
            map_image_sha256=(
                "" if self.map_bundle is None else self.map_bundle.image_sha256
            ),
            map_bundle_sha256=(
                "" if self.map_bundle is None else self.map_bundle.bundle_sha256
            ),
        )
        candidate_observations = observations_from_candidates(
            candidates,
            transform_scan_to_map=scan_pose_in_map,
            observed_at_sec=time.time(),
            provenance=provenance,
            start_index=self.observation_count + 1,
            observation_id_scope=self.observation_id_scope,
        )
        self.observation_count += len(candidate_observations)
        observations = tuple(
            observation
            for observation in candidate_observations
            if self.accumulator.accepts_observation(observation)
        )
        rejected_count = len(candidate_observations) - len(observations)
        if not observations:
            self.get_logger().info(
                f"rejected {rejected_count} candidate observations at perception gates"
            )
            return
        self.accepted_observation_count += len(observations)
        write_observation_jsonl(self.output_jsonl, observations)
        confirmed = self.accumulator.add_observations(observations)
        self.last_confirmed_stand_count = len(confirmed)
        self.get_logger().info(
            f"wrote {len(observations)} observations; rejected={rejected_count}; "
            f"confirmed_stands={len(confirmed)}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="")
    parser.add_argument("--scan-topic", default="scan")
    parser.add_argument("--odom-topic", default="odom")
    parser.add_argument("--cmd-vel-topic", default="cmd_vel")
    parser.add_argument("--amcl-topic", default="amcl_pose")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_footprint")
    parser.add_argument("--localization-source", default="amcl", choices=["amcl", "tf"])
    parser.add_argument(
        "--odom-execution-certificate-json",
        type=Path,
        default=None,
        help=(
            "Optional immutable odom execution certificate. When supplied, "
            "stand geometry uses exact-time odom<-scan TF composed with the "
            "certificate's frozen map<-odom transform; live map TF is never "
            "used for observation geometry."
        ),
    )
    parser.add_argument("--allow-sim-time", action="store_true")
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument(
        "--observation-id-scope",
        "--observation-id-prefix",
        dest="observation_id_scope",
        type=validated_observation_id_scope,
        default=None,
        help=(
            "Optional unique process/epoch scope embedded in every observation "
            "ID. Omit it to preserve legacy stand_observation_000001 IDs."
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help=(
            "Optional stopped-observation receipt. Written on clean shutdown even "
            "when no stand candidate was detected."
        ),
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="Wall-time observation duration; 0 keeps observing until Ctrl+C.",
    )
    parser.add_argument("--map-yaml", type=Path, default=None)
    parser.add_argument("--semantic-map-id", default="")
    parser.add_argument(
        "--tf-timeout-sec",
        type=float,
        default=0.50,
        help=(
            "Maximum monotonic wait in the bounded pending queue for the "
            "selected observation TF at the exact LaserScan timestamp."
        ),
    )
    parser.add_argument(
        "--pending-scan-limit",
        type=int,
        default=8,
        help="Maximum LaserScan messages retained while exact-time TF catches up.",
    )
    parser.add_argument(
        "--max-tf-age-sec",
        type=float,
        default=DEFAULT_OBSERVATION_TIMING_LIMITS.max_tf_age_sec,
    )
    parser.add_argument(
        "--max-scan-age-sec",
        type=float,
        default=DEFAULT_OBSERVATION_TIMING_LIMITS.max_scan_age_sec,
    )
    parser.add_argument(
        "--max-future-timestamp-sec",
        "--max-scan-future-skew-sec",
        dest="max_future_timestamp_sec",
        type=float,
        default=DEFAULT_OBSERVATION_TIMING_LIMITS.max_future_timestamp_sec,
    )
    parser.add_argument(
        "--max-tf-scan-skew-sec",
        type=float,
        default=DEFAULT_OBSERVATION_TIMING_LIMITS.max_tf_scan_skew_sec,
    )
    parser.add_argument("--min-range-m", type=float, default=0.08)
    parser.add_argument("--max-range-m", type=float, default=3.5)
    parser.add_argument("--max-cluster-gap-m", type=float, default=0.08)
    parser.add_argument("--min-cluster-points", type=int, default=2)
    parser.add_argument("--min-width-m", type=float, default=0.03)
    parser.add_argument("--max-width-m", type=float, default=0.45)
    parser.add_argument("--merge-distance-m", type=float, default=0.18)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--max-observation-age-sec", type=float, default=8.0)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--min-boundary-clearance-m", type=float, default=0.10)
    return parser


def observer_summary_payload(node: StandExplorerNode) -> dict[str, object]:
    """Return evidence that a scan epoch ran, including negative observations."""

    frozen_frame = getattr(node, "frozen_observer_frame", None)
    payload = {
        "schema_version": 1,
        "observer_version": _observer_version(frozen_frame),
        "motion_published": False,
        "started_unix_sec": node.started_unix_sec,
        "finished_unix_sec": time.time(),
        "output_jsonl": str(node.output_jsonl),
        "map_bundle_sha256": (
            "" if node.map_bundle is None else node.map_bundle.bundle_sha256
        ),
        "planning_frame": node.runtime.map_frame,
        "scan_frame_pose_in_planning_frame": node.last_scan_pose_map,
        "last_processed_scan_stamp_sec": node.last_processed_scan_stamp_sec,
        "processed_scan_count": node.processed_scan_count,
        "detected_candidate_count": node.detected_candidate_count,
        "accepted_observation_count": node.accepted_observation_count,
        "confirmed_stand_count": node.last_confirmed_stand_count,
        "runtime_config": node.runtime.as_log_dict(),
        "timing_limits": node.timing_limits.as_dict(),
    }
    if frozen_frame is not None:
        payload[FROZEN_FRAME_EVIDENCE_KEY] = frozen_frame.runtime_evidence()
    observation_id_scope = getattr(node, "observation_id_scope", None)
    if observation_id_scope is not None:
        payload[OBSERVATION_ID_SCOPE_RUNTIME_KEY] = observation_id_scope
    return payload


def write_observer_summary(path: Path, node: StandExplorerNode) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"refusing to overwrite observer summary: {path}")
    path.write_text(
        json.dumps(observer_summary_payload(node), indent=2, sort_keys=True) + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.tf_timeout_sec) or args.tf_timeout_sec <= 0.0:
        parser.error("--tf-timeout-sec must be finite and positive")
    if args.pending_scan_limit <= 0:
        parser.error("--pending-scan-limit must be positive")
    if not math.isfinite(args.duration_sec) or args.duration_sec < 0.0:
        parser.error("--duration-sec must be finite and non-negative")
    if args.summary_json is not None and args.summary_json.exists():
        parser.error(f"refusing to overwrite observer summary: {args.summary_json}")
    try:
        _timing_limits_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    _require_ros()
    rclpy.init(args=None)
    node = StandExplorerNode(args)
    try:
        if args.duration_sec > 0.0:
            deadline = time.monotonic() + args.duration_sec
            while rclpy.ok() and time.monotonic() < deadline:
                remaining_sec = max(0.0, deadline - time.monotonic())
                rclpy.spin_once(node, timeout_sec=min(0.1, remaining_sec))
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if args.summary_json is not None:
                write_observer_summary(args.summary_json, node)
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
