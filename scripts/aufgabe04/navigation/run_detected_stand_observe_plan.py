"""Wait for localization, observe LiDAR stands, and plan one candidate route.

This command is deliberately observe-and-plan only. It creates no velocity
publisher and never invokes a navigation or waypoint-following runner.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.dynamic_approach_planner import (
    DynamicApproachConfig,
    minimum_static_obstacle_inflation_m,
)
from scripts.aufgabe04.navigation.plan_detected_stand_exploration import (
    main as plan_detected_stand_exploration,
)
from scripts.aufgabe04.perception.stand_explorer_node import (
    StandExplorerNode,
    build_parser as build_observer_parser,
)

try:  # pragma: no cover - exercised on ROS hosts.
    import rclpy
    from rclpy.duration import Duration
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.time import Time
    from std_srvs.srv import Empty
    from tf2_ros import Buffer, TransformException, TransformListener
except ImportError:  # pragma: no cover - keeps offline tests ROS-free.
    rclpy = None
    Duration = None
    SingleThreadedExecutor = None
    Node = object
    Time = None
    Empty = None
    Buffer = None
    TransformException = Exception
    TransformListener = None


DEFAULT_MAP = Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--semantic-map-id", default="")
    parser.add_argument("--namespace", default="")
    parser.add_argument("--scan-topic", default="scan")
    parser.add_argument("--amcl-topic", default="amcl_pose")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_footprint")
    parser.add_argument(
        "--localization-source",
        choices=["amcl", "tf"],
        default="amcl",
    )
    parser.add_argument("--readiness-timeout-sec", type=float, default=30.0)
    parser.add_argument("--max-readiness-tf-age-sec", type=float, default=1.0)
    parser.add_argument("--observation-duration-sec", type=float, default=8.0)
    parser.add_argument(
        "--nomotion-update-service",
        default="/request_nomotion_update",
    )
    parser.add_argument(
        "--nomotion-refresh-sec",
        type=float,
        default=2.0,
        help=(
            "Request another stationary AMCL update at this interval while "
            "observing; ignored for TF localization or --skip-nomotion-update."
        ),
    )
    parser.add_argument(
        "--skip-nomotion-update",
        action="store_true",
        help="Do not request an AMCL update while stationary.",
    )
    parser.add_argument(
        "--order",
        choices=["nearest", "confidence", "stand-id"],
        default="confidence",
    )
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--merge-distance-m", type=float, default=0.18)
    parser.add_argument("--max-observation-age-sec", type=float, default=8.0)
    parser.add_argument("--stand-radius-m", type=float, default=0.06)
    parser.add_argument("--stand-position-uncertainty-m", type=float, default=0.02)
    parser.add_argument("--robot-radius-m", type=float, default=0.105)
    parser.add_argument("--collision-margin-m", type=float, default=0.02)
    parser.add_argument("--tracking-margin-m", type=float, default=0.03)
    parser.add_argument("--lidar-stop-distance-m", type=float, default=0.20)
    parser.add_argument("--scan-origin-to-base-offset-m", type=float, default=0.0)
    parser.add_argument("--lidar-clearance-margin-m", type=float, default=0.02)
    parser.add_argument("--stand-keepout-radius-m", type=float, default=0.26)
    parser.add_argument("--approach-offset-m", type=float, default=0.32)
    return parser


def _require_ros() -> None:
    if rclpy is None:
        raise RuntimeError("ROS2 Python packages are not available in this environment")


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class LocalizationReadinessNode(Node):  # pragma: no cover - requires ROS runtime.
    def __init__(
        self,
        *,
        target_frame: str,
        source_frame: str,
        nomotion_update_service: str | None,
    ) -> None:
        super().__init__("aufgabe04_stand_exploration_readiness")
        self.target_frame = target_frame
        self.source_frame = source_frame
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nomotion_client = (
            None
            if nomotion_update_service is None
            else self.create_client(Empty, nomotion_update_service)
        )
        self.nomotion_future = None

    def maybe_request_nomotion_update(self) -> None:
        if (
            self.nomotion_client is not None
            and self.nomotion_future is None
            and self.nomotion_client.service_is_ready()
        ):
            self.nomotion_future = self.nomotion_client.call_async(Empty.Request())
            self.get_logger().info("requested stationary AMCL update")

    def nomotion_update_complete(self) -> bool:
        if self.nomotion_client is None:
            return True
        if self.nomotion_future is None or not self.nomotion_future.done():
            return False
        return self.nomotion_future.exception() is None

    def refresh_nomotion_update(self) -> None:
        if self.nomotion_future is not None and self.nomotion_future.done():
            self.nomotion_future = None

    def fresh_pose(self, *, max_age_sec: float) -> Pose2D | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                Time(),
                timeout=Duration(seconds=0.0),
            )
        except TransformException:
            return None
        now_sec = _stamp_to_sec(self.get_clock().now().to_msg())
        stamp_sec = _stamp_to_sec(transform.header.stamp)
        age_sec = now_sec - stamp_sec
        if age_sec < -0.25 or age_sec > max_age_sec:
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return Pose2D(
            float(translation.x),
            float(translation.y),
            _yaw_from_quaternion(rotation),
        )


def _default_output_dir(now_sec: float | None = None) -> Path:
    stamp = time.strftime(
        "%Y%m%d_%H%M%S",
        time.gmtime(time.time() if now_sec is None else now_sec),
    )
    return Path("results/aufgabe04") / f"real_explore_{stamp}"


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "observations": output_dir / "stand_observations.jsonl",
        "state": output_dir / "exploration_state.json",
        "layout_json": output_dir / "layout.json",
        "layout_csv": output_dir / "layout.csv",
        "route_csv": output_dir / "route.csv",
        "diagnostics": output_dir / "route_diagnostics.json",
        "snapshot": output_dir / "candidate_snapshot.json",
        "summary": output_dir / "pipeline_summary.json",
    }


def _ensure_new_artifacts(paths: dict[str, Path]) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise ValueError(
            "refusing to append to or overwrite pipeline artifacts: "
            + ", ".join(existing)
        )


def _observer_args(args, paths: dict[str, Path]):
    argv = [
        "--namespace",
        args.namespace,
        "--scan-topic",
        args.scan_topic,
        "--amcl-topic",
        args.amcl_topic,
        "--map-frame",
        args.map_frame,
        "--odom-frame",
        args.odom_frame,
        "--base-frame",
        args.base_frame,
        "--localization-source",
        args.localization_source,
        "--map-yaml",
        str(args.map),
        "--semantic-map-id",
        args.semantic_map_id or args.map.stem,
        "--output-jsonl",
        str(paths["observations"]),
        "--min-hits",
        str(args.min_hits),
        "--min-confidence",
        str(args.min_confidence),
        "--merge-distance-m",
        str(args.merge_distance_m),
        "--max-observation-age-sec",
        str(args.max_observation_age_sec),
    ]
    return build_observer_parser().parse_args(argv)


def _planner_argv(args, paths: dict[str, Path], start: Pose2D) -> list[str]:
    clearance = _physical_clearance(args)
    return [
        "--observations-jsonl",
        str(paths["observations"]),
        "--map",
        str(args.map),
        "--start-x",
        str(start.x_m),
        "--start-y",
        str(start.y_m),
        "--start-yaw",
        str(start.yaw_rad),
        "--plan-mode",
        "next-candidate",
        "--max-stands",
        "1",
        "--order",
        args.order,
        "--min-hits",
        str(args.min_hits),
        "--min-confidence",
        str(args.min_confidence),
        "--merge-distance-m",
        str(args.merge_distance_m),
        "--max-observation-age-sec",
        str(args.max_observation_age_sec),
        "--approach-bearing-mode",
        "robot-to-stand",
        "--approach-offset-m",
        str(args.approach_offset_m),
        "--keepout-radius-m",
        str(args.stand_keepout_radius_m),
        "--candidate-transit-radius-m",
        str(clearance["minimum_candidate_transit_radius_m"]),
        "--inflation-radius-m",
        str(clearance["minimum_static_inflation_m"]),
        "--stand-radius-m",
        str(args.stand_radius_m),
        "--stand-position-uncertainty-m",
        str(args.stand_position_uncertainty_m),
        "--robot-radius-m",
        str(args.robot_radius_m),
        "--collision-margin-m",
        str(args.collision_margin_m),
        "--tracking-margin-m",
        str(args.tracking_margin_m),
        "--lidar-stop-distance-m",
        str(args.lidar_stop_distance_m),
        "--scan-origin-to-base-offset-m",
        str(args.scan_origin_to_base_offset_m),
        "--lidar-clearance-margin-m",
        str(args.lidar_clearance_margin_m),
        "--enforce-physical-clearance",
        "--required-map-frame",
        args.map_frame,
        "--required-base-frame",
        args.base_frame,
        "--required-localization-source",
        args.localization_source,
        "--semantic-map-id",
        args.semantic_map_id or args.map.stem,
        "--exploration-state-json",
        str(paths["state"]),
        "--layout-json",
        str(paths["layout_json"]),
        "--layout-csv",
        str(paths["layout_csv"]),
        "--route-csv",
        str(paths["route_csv"]),
        "--diagnostics-json",
        str(paths["diagnostics"]),
        "--candidate-snapshot-json",
        str(paths["snapshot"]),
    ]


def _physical_clearance(args) -> dict[str, float]:
    config = DynamicApproachConfig(
        stand_radius_m=args.stand_radius_m,
        stand_position_uncertainty_m=args.stand_position_uncertainty_m,
        robot_radius_m=args.robot_radius_m,
        collision_margin_m=args.collision_margin_m,
        tracking_margin_m=args.tracking_margin_m,
        standoff_distance_m=args.approach_offset_m,
        lidar_stop_distance_m=args.lidar_stop_distance_m,
        scan_origin_to_base_offset_m=args.scan_origin_to_base_offset_m,
        lidar_clearance_margin_m=args.lidar_clearance_margin_m,
        minimum_non_target_keepout_radius_m=args.stand_keepout_radius_m,
    )
    return {
        "minimum_static_inflation_m": minimum_static_obstacle_inflation_m(
            robot_radius_m=args.robot_radius_m,
            tracking_margin_m=args.tracking_margin_m,
            lidar_stop_distance_m=args.lidar_stop_distance_m,
            scan_origin_to_base_offset_m=args.scan_origin_to_base_offset_m,
            lidar_clearance_margin_m=args.lidar_clearance_margin_m,
        ),
        "minimum_active_standoff_m": config.minimum_lidar_standoff_m,
        "minimum_candidate_transit_radius_m": (
            config.non_target_stand_keepout_radius_m
        ),
    }


def _wait_for_ready_pose(
    *,
    executor,
    readiness_node: LocalizationReadinessNode,
    timeout_sec: float,
    max_tf_age_sec: float,
) -> Pose2D:
    deadline = time.monotonic() + timeout_sec
    while rclpy.ok() and time.monotonic() < deadline:
        readiness_node.maybe_request_nomotion_update()
        executor.spin_once(timeout_sec=0.05)
        pose = readiness_node.fresh_pose(max_age_sec=max_tf_age_sec)
        if pose is not None and readiness_node.nomotion_update_complete():
            return pose
    service_state = (
        "not requested"
        if readiness_node.nomotion_client is None
        else "unavailable or incomplete"
    )
    raise RuntimeError(
        "timed out waiting for fresh "
        f"{readiness_node.target_frame!r} -> {readiness_node.source_frame!r} "
        f"TF; stationary update={service_state}"
    )


def _observe(args, paths: dict[str, Path]) -> Pose2D:
    _require_ros()
    rclpy.init(args=None)
    observer = StandExplorerNode(_observer_args(args, paths))
    observer.set_observation_enabled(False)
    require_nomotion = (
        args.localization_source == "amcl" and not args.skip_nomotion_update
    )
    readiness = LocalizationReadinessNode(
        target_frame=args.map_frame,
        source_frame=args.base_frame,
        nomotion_update_service=(
            args.nomotion_update_service if require_nomotion else None
        ),
    )
    executor = SingleThreadedExecutor()
    executor.add_node(observer)
    executor.add_node(readiness)
    try:
        start = _wait_for_ready_pose(
            executor=executor,
            readiness_node=readiness,
            timeout_sec=args.readiness_timeout_sec,
            max_tf_age_sec=args.max_readiness_tf_age_sec,
        )
        print(
            "localization ready: "
            f"x={start.x_m:.3f} y={start.y_m:.3f} yaw={start.yaw_rad:.3f}"
        )
        observer.set_observation_enabled(True)
        deadline = time.monotonic() + args.observation_duration_sec
        next_nomotion_refresh = time.monotonic() + args.nomotion_refresh_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if require_nomotion and time.monotonic() >= next_nomotion_refresh:
                readiness.refresh_nomotion_update()
                readiness.maybe_request_nomotion_update()
                next_nomotion_refresh = time.monotonic() + args.nomotion_refresh_sec
            remaining_sec = max(0.0, deadline - time.monotonic())
            executor.spin_once(timeout_sec=min(0.1, remaining_sec))
        final_pose = readiness.fresh_pose(
            max_age_sec=args.max_readiness_tf_age_sec
        )
        if final_pose is None:
            readiness.refresh_nomotion_update()
            final_pose = _wait_for_ready_pose(
                executor=executor,
                readiness_node=readiness,
                timeout_sec=args.readiness_timeout_sec,
                max_tf_age_sec=args.max_readiness_tf_age_sec,
            )
        return final_pose
    finally:
        executor.remove_node(observer)
        executor.remove_node(readiness)
        observer.destroy_node()
        readiness.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _summary(paths: dict[str, Path], start: Pose2D) -> dict[str, object]:
    diagnostics = json.loads(paths["diagnostics"].read_text())
    snapshot = json.loads(paths["snapshot"].read_text())
    metadata = diagnostics["metadata"]
    selected_uid = metadata["selected_candidate_stand_id"]
    candidate = next(
        item
        for item in snapshot["candidates"]
        if item["candidate_uid"] == selected_uid
    )
    leg = diagnostics["legs"][0]
    return {
        "schema_version": 1,
        "status": "observe_and_plan_complete",
        "motion_published": False,
        "selected_candidate_uid": selected_uid,
        "selected_candidate_confidence": candidate["confidence"],
        "selected_candidate_hit_count": candidate["hit_count"],
        "selected_candidate_geometry": candidate["geometry"],
        "planning_start_pose": {
            "x_m": start.x_m,
            "y_m": start.y_m,
            "yaw_rad": start.yaw_rad,
        },
        "route_length_m": leg["route_length_m"],
        "route_point_count": leg["route_point_count"],
        "physical_clearance": metadata["physical_clearance"],
        "approach_bearing_mode": metadata["approach_bearing_mode"],
        "selected_approach_pose": metadata["selected_approach_pose"],
        "artifacts": {
            key: str(path)
            for key, path in paths.items()
            if key != "summary"
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in (
        "readiness_timeout_sec",
        "max_readiness_tf_age_sec",
        "observation_duration_sec",
        "nomotion_refresh_sec",
        "max_observation_age_sec",
        "stand_radius_m",
        "robot_radius_m",
        "lidar_stop_distance_m",
        "approach_offset_m",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in (
        "stand_position_uncertainty_m",
        "collision_margin_m",
        "tracking_margin_m",
        "lidar_clearance_margin_m",
        "stand_keepout_radius_m",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if not math.isfinite(args.scan_origin_to_base_offset_m):
        parser.error("--scan-origin-to-base-offset-m must be finite")
    output_dir = args.output_dir or _default_output_dir()
    paths = _artifact_paths(output_dir)
    try:
        _ensure_new_artifacts(paths)
        output_dir.mkdir(parents=True, exist_ok=True)
        print("observe-and-plan only: this command has no cmd_vel publisher")
        start = _observe(args, paths)
        if not paths["observations"].exists() or paths["observations"].stat().st_size == 0:
            raise ValueError("observer produced no stand observations")
        status = plan_detected_stand_exploration(_planner_argv(args, paths, start))
        if status != 0:
            return status
        summary = _summary(paths, start)
        paths["summary"].write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    except (OSError, RuntimeError, ValueError, KeyError, StopIteration) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
