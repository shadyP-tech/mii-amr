"""Plan a sequential route through all confirmed LiDAR-detected stand candidates."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.artifacts import write_diagnostics_json, write_route_csv
from scripts.aufgabe04.navigation.candidate_exploration_state import (
    STATUS_CONFIRMED,
    STATUS_REJECTED,
    load_candidate_exploration_state,
    write_candidate_exploration_state,
)
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.plan_first_detected_station import (
    validate_observation_provenance,
    validate_route_commitment_ready,
)
from scripts.aufgabe04.navigation.route_context import build_station_route_dry_run, file_sha256

try:  # pragma: no cover - exercised on ROS hosts.
    import rclpy
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformException, TransformListener
except ImportError:  # pragma: no cover - keeps offline tests ROS-free.
    rclpy = None
    Duration = None
    Node = object
    Parameter = None
    Time = None
    Buffer = None
    TransformException = Exception
    TransformListener = None
from scripts.aufgabe04.perception.stand_confirmation import (
    ConfirmedStand,
    StandConfirmationAccumulator,
    StandConfirmationConfig,
)
from scripts.aufgabe04.perception.stand_observation import load_observation_jsonl
from scripts.aufgabe04.stations.detected_station_layout import (
    DetectedStationLayoutConfig,
    detected_station_metadata,
    station_from_confirmed_stand,
)
from scripts.aufgabe04.stations.models import Station, StationPose
from scripts.aufgabe04.stations.station_layout_io import write_station_layout_csv, write_station_layout_json


DEFAULT_OBSERVATIONS_JSONL = Path("results/aufgabe04/detected_stations/stand_observations.jsonl")
DEFAULT_LAYOUT_JSON = Path("results/aufgabe04/detected_stations/detected_stand_exploration_layout.json")
DEFAULT_LAYOUT_CSV = Path("results/aufgabe04/detected_stations/detected_stand_exploration_layout.csv")
DEFAULT_ROUTE_CSV = Path("results/aufgabe04/routes/detected_stand_exploration_route.csv")
DEFAULT_DIAGNOSTICS_JSON = Path("results/aufgabe04/routes/detected_stand_exploration_route_diagnostics.json")
DEFAULT_EXPLORATION_STATE_JSON = Path("results/aufgabe04/detected_stations/detected_stand_exploration_state.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations-jsonl", type=Path, default=DEFAULT_OBSERVATIONS_JSONL)
    parser.add_argument("--map", required=True, type=Path, help="ROS map YAML path")
    parser.add_argument("--start-x", type=float)
    parser.add_argument("--start-y", type=float)
    parser.add_argument("--start-yaw", type=float, default=0.0)
    parser.add_argument(
        "--start-from-tf",
        action="store_true",
        help="Resolve start pose from a live TF transform instead of --start-x/--start-y.",
    )
    parser.add_argument("--start-tf-source-frame", default="base_footprint")
    parser.add_argument("--start-tf-target-frame", default="odom")
    parser.add_argument("--start-tf-timeout-sec", type=float, default=3.0)
    parser.add_argument("--start-tf-lookup-timeout-sec", type=float, default=0.2)
    parser.add_argument("--start-tf-use-sim-time", action="store_true")
    parser.add_argument("--station-prefix", default="D")
    parser.add_argument("--plan-mode", choices=["next-candidate", "full-route"], default="next-candidate")
    parser.add_argument("--exploration-state-json", type=Path, default=DEFAULT_EXPLORATION_STATE_JSON)
    parser.add_argument("--mark-confirmed-stand-id", action="append", default=[])
    parser.add_argument("--mark-rejected-stand-id", action="append", default=[])
    parser.add_argument("--stand-yaw-rad", type=float, default=0.0)
    parser.add_argument("--approach-offset-m", type=float, default=0.30)
    parser.add_argument("--keepout-radius-m", type=float, default=0.20)
    parser.add_argument("--candidate-transit-radius-m", type=float, default=0.15)
    parser.add_argument("--merge-distance-m", type=float, default=0.18)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--max-observation-age-sec", type=float, default=8.0)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--max-stands", type=int, default=0, help="0 means all confirmed stands")
    parser.add_argument(
        "--order",
        choices=["nearest", "confidence", "stand-id"],
        default="nearest",
        help="Candidate visit ordering policy.",
    )
    parser.add_argument("--max-tf-age-sec", type=float, default=1.0)
    parser.add_argument("--required-map-frame", default="map")
    parser.add_argument("--required-base-frame", default="base_footprint")
    parser.add_argument("--required-localization-source", default=None, choices=["amcl", "tf"])
    parser.add_argument("--inflation-radius-m", type=float, default=0.0)
    parser.add_argument("--snap-radius-m", type=float, default=0.30)
    parser.add_argument("--layout-json", type=Path, default=DEFAULT_LAYOUT_JSON)
    parser.add_argument("--layout-csv", type=Path, default=DEFAULT_LAYOUT_CSV)
    parser.add_argument("--route-csv", type=Path, default=DEFAULT_ROUTE_CSV)
    parser.add_argument("--diagnostics-json", type=Path, default=DEFAULT_DIAGNOSTICS_JSON)
    parser.add_argument("--arena-length-m", type=float, default=ArenaBounds.length_m)
    parser.add_argument("--arena-width-m", type=float, default=ArenaBounds.width_m)
    parser.add_argument("--arena-center-x-m", type=float, default=ArenaBounds.center_x_m)
    parser.add_argument("--arena-center-y-m", type=float, default=ArenaBounds.center_y_m)
    parser.add_argument("--arena-yaw-deg", type=float, default=ArenaBounds.yaw_deg)
    parser.add_argument("--arena-margin-m", type=float, default=ArenaBounds.margin_m)
    return parser


def _validated_confirmed_stands(args, arena_bounds: ArenaBounds) -> tuple[ConfirmedStand, ...]:
    observations = load_observation_jsonl(args.observations_jsonl)
    if not observations:
        raise ValueError("no stand observations found")
    for observation in observations:
        validate_observation_provenance(
            observation,
            map_yaml=args.map,
            required_map_frame=args.required_map_frame,
            required_base_frame=args.required_base_frame,
            required_localization_source=args.required_localization_source,
            max_tf_age_sec=args.max_tf_age_sec,
        )
    accumulator = StandConfirmationAccumulator(
        config=StandConfirmationConfig(
            merge_distance_m=args.merge_distance_m,
            min_hits=args.min_hits,
            max_age_sec=args.max_observation_age_sec,
            min_confidence=args.min_confidence,
        ),
        arena_bounds=arena_bounds,
    )
    stands = accumulator.add_observations(observations)
    if not stands:
        raise ValueError("no confirmed stand available")
    return stands


def _station_id(prefix: str, index: int) -> str:
    cleaned = "".join(ch for ch in prefix.strip().upper() if ch.isalnum()) or "D"
    return f"{cleaned}{index:02d}"


def _distance(a: Pose2D, b: Pose2D) -> float:
    return math.hypot(a.x_m - b.x_m, a.y_m - b.y_m)


def _require_ros() -> None:
    if rclpy is None:
        raise RuntimeError("ROS2 Python packages are not available in this environment")


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class CurrentTfPoseReader(Node):  # pragma: no cover - requires ROS runtime.
    def __init__(self, *, use_sim_time: bool) -> None:
        super().__init__("aufgabe04_current_tf_pose_reader")
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", use_sim_time)
        else:
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, use_sim_time)])
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def current_pose(
        self,
        *,
        target_frame: str,
        source_frame: str,
        lookup_timeout_sec: float,
    ) -> Pose2D:
        transform = self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            Time(),
            timeout=Duration(seconds=lookup_timeout_sec),
        )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return Pose2D(float(translation.x), float(translation.y), _yaw_from_quaternion(rotation))


def read_current_tf_pose(
    *,
    target_frame: str,
    source_frame: str,
    timeout_sec: float,
    lookup_timeout_sec: float,
    use_sim_time: bool,
) -> Pose2D:
    _require_ros()
    rclpy.init(args=None)
    node = CurrentTfPoseReader(use_sim_time=use_sim_time)
    deadline = node.get_clock().now().nanoseconds / 1_000_000_000.0 + timeout_sec
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            try:
                return node.current_pose(
                    target_frame=target_frame,
                    source_frame=source_frame,
                    lookup_timeout_sec=lookup_timeout_sec,
                )
            except TransformException:
                pass
            now_sec = node.get_clock().now().nanoseconds / 1_000_000_000.0
            if now_sec >= deadline:
                raise RuntimeError(
                    f"timed out waiting for TF {target_frame!r} -> {source_frame!r}"
                )
    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise RuntimeError("ROS shutdown before TF pose was received")


def start_pose_from_args(args) -> Pose2D:
    if args.start_from_tf:
        return read_current_tf_pose(
            target_frame=args.start_tf_target_frame,
            source_frame=args.start_tf_source_frame,
            timeout_sec=args.start_tf_timeout_sec,
            lookup_timeout_sec=args.start_tf_lookup_timeout_sec,
            use_sim_time=args.start_tf_use_sim_time,
        )
    if args.start_x is None or args.start_y is None:
        raise ValueError("provide --start-x and --start-y, or use --start-from-tf")
    return Pose2D(args.start_x, args.start_y, args.start_yaw)


def _ordered_stands(
    stands: tuple[ConfirmedStand, ...],
    *,
    start: Pose2D,
    order: str,
    max_stands: int,
    approach_offset_m: float,
    keepout_radius_m: float,
    stand_yaw_rad: float,
    arena_args,
) -> tuple[ConfirmedStand, ...]:
    candidates = list(stands)
    if order == "confidence":
        candidates.sort(key=lambda stand: (-stand.confidence, -stand.hit_count, stand.stand_id))
    elif order == "stand-id":
        candidates.sort(key=lambda stand: stand.stand_id)
    else:
        selected: list[ConfirmedStand] = []
        current = start
        while candidates:
            best_index = min(
                range(len(candidates)),
                key=lambda index: _distance(
                    current,
                    Pose2D(
                        candidates[index].x_m - math.cos(stand_yaw_rad) * approach_offset_m,
                        candidates[index].y_m - math.sin(stand_yaw_rad) * approach_offset_m,
                        stand_yaw_rad,
                    ),
                ),
            )
            stand = candidates.pop(best_index)
            selected.append(stand)
            current = Pose2D(
                stand.x_m - math.cos(stand_yaw_rad) * approach_offset_m,
                stand.y_m - math.sin(stand_yaw_rad) * approach_offset_m,
                stand_yaw_rad,
            )
        candidates = selected
    if max_stands > 0:
        candidates = candidates[:max_stands]
    return tuple(candidates)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    failed = False
    try:
        arena_bounds = ArenaBounds(
            length_m=args.arena_length_m,
            width_m=args.arena_width_m,
            center_x_m=args.arena_center_x_m,
            center_y_m=args.arena_center_y_m,
            yaw_deg=args.arena_yaw_deg,
            margin_m=args.arena_margin_m,
        )
        arena_bounds.validate()
        start = start_pose_from_args(args)
        stands = _validated_confirmed_stands(args, arena_bounds)
        state = load_candidate_exploration_state(args.exploration_state_json).with_decisions(
            stands,
            confirmed_stand_ids=args.mark_confirmed_stand_id,
            rejected_stand_ids=args.mark_rejected_stand_id,
        )
        if args.mark_confirmed_stand_id or args.mark_rejected_stand_id:
            write_candidate_exploration_state(args.exploration_state_json, state)
        pending = state.pending_stands(stands)
        ordered = _ordered_stands(
            pending if args.plan_mode == "next-candidate" else stands,
            start=start,
            order=args.order,
            max_stands=1 if args.plan_mode == "next-candidate" else args.max_stands,
            approach_offset_m=args.approach_offset_m,
            keepout_radius_m=args.keepout_radius_m,
            stand_yaw_rad=args.stand_yaw_rad,
            arena_args=args,
        )
        if not ordered:
            write_candidate_exploration_state(args.exploration_state_json, state)
            raise ValueError("no pending candidate available")

        stations = []
        stand_metadata = []
        for index, stand in enumerate(ordered, start=1):
            station = station_from_confirmed_stand(
                stand,
                config=DetectedStationLayoutConfig(
                    station_id=_station_id(args.station_prefix, index),
                    approach_offset_m=args.approach_offset_m,
                    keepout_radius_m=args.keepout_radius_m,
                    stand_yaw_rad=args.stand_yaw_rad,
                    arena_length_m=args.arena_length_m,
                    arena_width_m=args.arena_width_m,
                    arena_center_x_m=args.arena_center_x_m,
                    arena_center_y_m=args.arena_center_y_m,
                    arena_yaw_deg=args.arena_yaw_deg,
                    arena_margin_m=args.arena_margin_m,
                ),
            )
            stations.append(station)
            stand_metadata.append(
                detected_station_metadata(
                    stand,
                    source_observation_path=str(args.observations_jsonl),
                    extra={
                        "station_id": station.station_id,
                        "visit_index": index - 1,
                        "map_yaml": str(args.map),
                        "map_yaml_sha256": file_sha256(args.map),
                        "required_map_frame": args.required_map_frame,
                        "required_base_frame": args.required_base_frame,
                        "required_localization_source": args.required_localization_source,
                        "candidate_status": state.status_for(stand),
                    },
                )
            )

        transit_stations = []
        for stand in stands:
            if state.status_for(stand) == STATUS_REJECTED:
                continue
            transit_stations.append(
                Station(
                    stand.stand_id,
                    StationPose(stand.x_m, stand.y_m, args.stand_yaw_rad),
                    args.approach_offset_m,
                    args.candidate_transit_radius_m,
                )
            )

        station_map = {station.station_id: station for station in stations}
        planning_station_map = dict(station_map)
        planning_station_map.update({station.station_id: station for station in transit_stations})
        dry_run = build_station_route_dry_run(
            args.map,
            [station.station_id for station in stations],
            station_map=planning_station_map,
            station_layout_json=args.layout_json,
            start=start,
            inflation_radius_m=args.inflation_radius_m,
            snap_radius_m=args.snap_radius_m,
            transit_keepout_radius_m=args.candidate_transit_radius_m,
            arena_bounds=arena_bounds,
        )
        validate_route_commitment_ready(dry_run)
        write_station_layout_json(
            args.layout_json,
            stations,
            {
                "source": "lidar_detected_stand_exploration",
                "observations_jsonl": str(args.observations_jsonl),
                "order": args.order,
                "plan_mode": args.plan_mode,
                "stand_count": len(ordered),
                "pending_candidate_count": len(pending),
                "confirmed_candidate_count": state.count(STATUS_CONFIRMED, stands),
                "rejected_candidate_count": state.count(STATUS_REJECTED, stands),
                "candidate_transit_radius_m": args.candidate_transit_radius_m,
                "stands": stand_metadata,
            },
        )
        write_station_layout_csv(args.layout_csv, stations)
        route_metadata = dict(dry_run.metadata)
        route_metadata.update(
            {
                "source": "lidar_detected_stand_exploration",
                "observations_jsonl": str(args.observations_jsonl),
                "order": args.order,
                "plan_mode": args.plan_mode,
                "stand_count": len(ordered),
                "pending_candidate_count": len(pending),
                "confirmed_candidate_count": state.count(STATUS_CONFIRMED, stands),
                "rejected_candidate_count": state.count(STATUS_REJECTED, stands),
                "candidate_transit_radius_m": args.candidate_transit_radius_m,
                "exploration_state_json": str(args.exploration_state_json),
                "selected_candidate_stand_id": ordered[0].stand_id if args.plan_mode == "next-candidate" else "",
                "detected_stands": stand_metadata,
            }
        )
        write_route_csv(args.route_csv, dry_run.results)
        write_diagnostics_json(args.diagnostics_json, dry_run.results, metadata=route_metadata)
        write_candidate_exploration_state(args.exploration_state_json, state)
        failed = any(result.failure is not None for result in dry_run.results)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
