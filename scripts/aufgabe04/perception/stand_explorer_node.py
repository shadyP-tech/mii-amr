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
DEFAULT_OUTPUT_JSONL = Path("results/aufgabe04/detected_stations/stand_observations.jsonl")


@dataclass(frozen=True)
class _PendingScan:
    message: object
    scan_frame: str
    scan_stamp_sec: float
    query_time: object
    deadline_monotonic_sec: float


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
        self.output_jsonl = args.output_jsonl
        self.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        _validate_append_stream(
            self.output_jsonl,
            required_observer_clock=observer_clock_name(
                use_sim_time=self.runtime.use_sim_time
            ),
            timing_limits=self.timing_limits,
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
            f"{self.runtime.scan_topic}; output={self.output_jsonl}"
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
            if time.monotonic() > pending.deadline_monotonic_sec:
                self.get_logger().warn(
                    "dropping scan: exact-time map<-"
                    f"{pending.scan_frame} TF timed out for stamp "
                    f"{pending.scan_stamp_sec:.9f}"
                )
                continue

            zero_timeout = Duration(seconds=0.0)
            try:
                available = self.tf_buffer.can_transform(
                    self.runtime.map_frame,
                    pending.scan_frame,
                    pending.query_time,
                    timeout=zero_timeout,
                )
            except TransformException as exc:
                self.get_logger().warn(
                    "exact-time map<-"
                    f"{pending.scan_frame} TF check failed: {exc}"
                )
                self.pending_scans.append(pending)
                continue
            if not available:
                self.pending_scans.append(pending)
                continue

            try:
                transform = self.tf_buffer.lookup_transform(
                    self.runtime.map_frame,
                    pending.scan_frame,
                    pending.query_time,
                    timeout=zero_timeout,
                )
            except TransformException as exc:
                # The buffer can change between can_transform and lookup.
                self.get_logger().warn(
                    "exact-time map<-"
                    f"{pending.scan_frame} TF lookup raced: {exc}"
                )
                self.pending_scans.append(pending)
                continue
            self._process_scan_with_transform(pending, transform)

    def _process_scan_with_transform(self, pending: _PendingScan, transform) -> None:
        msg = pending.message
        scan_frame = pending.scan_frame
        scan_stamp_sec = pending.scan_stamp_sec

        observer_clock_sec = _stamp_to_sec(self.get_clock().now().to_msg())
        tf_stamp_sec = _stamp_to_sec(transform.header.stamp)
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

        self.processed_scan_count += 1
        self.last_processed_scan_stamp_sec = scan_stamp_sec
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        self.last_scan_pose_map = {
            "x_m": float(translation.x),
            "y_m": float(translation.y),
            "yaw_rad": _yaw_from_quaternion(rotation),
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
        provenance = ObservationProvenance(
            schema_version=OBSERVATION_SCHEMA_VERSION,
            observer_version=OBSERVER_VERSION,
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
            transform_scan_to_map=PlanarTransform(
                translation.x,
                translation.y,
                _yaw_from_quaternion(rotation),
            ),
            observed_at_sec=time.time(),
            provenance=provenance,
            start_index=self.observation_count + 1,
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
    parser.add_argument("--allow-sim-time", action="store_true")
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
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
            "Maximum monotonic wait in the bounded pending queue for map<-scan "
            "TF at the exact LaserScan timestamp."
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

    return {
        "schema_version": 1,
        "observer_version": OBSERVER_VERSION,
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
