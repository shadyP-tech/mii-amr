"""Observe Aufgabe 04 station stands from live LiDAR without publishing motion."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.route_context import file_sha256
from scripts.aufgabe04.navigation.ros_runtime_config import RuntimeConfig, resolve_runtime_config
from scripts.aufgabe04.perception.lidar_stand_detector import detect_stand_candidates_from_scan
from scripts.aufgabe04.perception.models import LidarStandDetectorConfig
from scripts.aufgabe04.perception.stand_confirmation import (
    StandConfirmationAccumulator,
    StandConfirmationConfig,
)
from scripts.aufgabe04.perception.stand_observation import (
    OBSERVATION_SCHEMA_VERSION,
    ObservationProvenance,
    PlanarTransform,
    observations_from_candidates,
    write_observation_jsonl,
)

try:  # pragma: no cover - exercised on ROS hosts.
    import rclpy
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer, TransformException, TransformListener
except ImportError:  # pragma: no cover - keeps offline tests ROS-free.
    rclpy = None
    Duration = None
    Node = object
    Time = None
    LaserScan = None
    Buffer = None
    TransformException = Exception
    TransformListener = None


OBSERVER_VERSION = "aufgabe04-stand-explorer-observe-only-v2-latest-tf"
DEFAULT_OUTPUT_JSONL = Path("results/aufgabe04/detected_stations/stand_observations.jsonl")


def _require_ros() -> None:
    if rclpy is None:
        raise RuntimeError("ROS2 Python packages are not available in this environment")


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0


def _latest_transform_time():
    """Return the zero ROS time, which asks tf2 for the latest transform.

    Looking up a transform at the exact LaserScan timestamp is fragile in
    simulation: Gazebo can publish a scan a few milliseconds before the
    corresponding odometry TF reaches this node.  The old exact-time lookup
    also blocked the single-threaded executor for the full scan period, which
    prevented the TransformListener from catching up.
    """

    if Time is None:
        _require_ros()
    return Time()


def _transform_age_sec(now_sec: float, transform_stamp_sec: float) -> float:
    """Return a non-negative TF age; timestamp-zero static TF is always valid."""

    if transform_stamp_sec <= 0.0:
        return 0.0
    return max(0.0, now_sec - transform_stamp_sec)


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class StandExplorerNode(Node):  # pragma: no cover - requires ROS runtime.
    def __init__(self, args) -> None:
        super().__init__("aufgabe04_stand_explorer_observer")
        self.args = args
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
            )
        )
        self.observation_count = 0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(LaserScan, self.runtime.scan_topic, self._scan_callback, 10)
        self.get_logger().info(
            "observe-only stand explorer listening on "
            f"{self.runtime.scan_topic}; output={self.output_jsonl}"
        )

    def _scan_callback(self, msg) -> None:
        scan_frame = msg.header.frame_id
        if not scan_frame:
            self.get_logger().warn("dropping scan without header.frame_id")
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                self.runtime.map_frame,
                scan_frame,
                _latest_transform_time(),
                timeout=Duration(seconds=self.args.tf_timeout_sec),
            )
        except TransformException as exc:
            self.get_logger().warn(f"dropping scan: map<-{scan_frame} TF unavailable: {exc}")
            return

        now = self.get_clock().now()
        tf_stamp_sec = _stamp_to_sec(transform.header.stamp)
        tf_age_sec = _transform_age_sec(
            _stamp_to_sec(now.to_msg()),
            tf_stamp_sec,
        )
        if tf_age_sec > self.args.max_tf_age_sec:
            self.get_logger().warn(f"dropping scan: TF age {tf_age_sec:.3f}s exceeds limit")
            return

        candidates = detect_stand_candidates_from_scan(
            msg.ranges,
            angle_min_rad=msg.angle_min,
            angle_increment_rad=msg.angle_increment,
            config=self.detector_config,
        )
        if not candidates:
            return

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        provenance = ObservationProvenance(
            schema_version=OBSERVATION_SCHEMA_VERSION,
            observer_version=OBSERVER_VERSION,
            resolved_scan_topic=self.runtime.scan_topic,
            scan_frame=scan_frame,
            map_frame=self.runtime.map_frame,
            base_frame=self.runtime.base_frame,
            localization_source=self.runtime.localization_source,
            scan_stamp_sec=_stamp_to_sec(msg.header.stamp),
            tf_lookup_stamp_sec=_stamp_to_sec(transform.header.stamp),
            tf_age_sec=tf_age_sec,
            runtime_config=self.runtime.as_log_dict(),
            map_yaml=str(self.args.map_yaml or ""),
            map_yaml_sha256=file_sha256(self.args.map_yaml) if self.args.map_yaml else "",
        )
        observations = observations_from_candidates(
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
        self.observation_count += len(observations)
        write_observation_jsonl(self.output_jsonl, observations)
        confirmed = self.accumulator.add_observations(observations)
        self.get_logger().info(
            f"wrote {len(observations)} observations; confirmed_stands={len(confirmed)}"
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
    parser.add_argument("--map-yaml", type=Path, default=None)
    parser.add_argument(
        "--tf-timeout-sec",
        type=float,
        default=0.05,
        help="Maximum wait for the latest available TF; keep below the scan period.",
    )
    parser.add_argument("--max-tf-age-sec", type=float, default=1.0)
    parser.add_argument("--min-range-m", type=float, default=0.08)
    parser.add_argument("--max-range-m", type=float, default=3.5)
    parser.add_argument("--max-cluster-gap-m", type=float, default=0.08)
    parser.add_argument("--min-cluster-points", type=int, default=3)
    parser.add_argument("--min-width-m", type=float, default=0.03)
    parser.add_argument("--max-width-m", type=float, default=0.45)
    parser.add_argument("--merge-distance-m", type=float, default=0.18)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--max-observation-age-sec", type=float, default=8.0)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_ros()
    rclpy.init(args=None)
    node = StandExplorerNode(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
