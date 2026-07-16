"""Plan a route to the latest simulation synchronized-viewpoint recommendation."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.costmap import Costmap
from scripts.aufgabe04.navigation.dynamic_approach_planner import (
    DynamicApproachConfig,
    face_normal_candidates as planner_face_candidates,
    plan_axis_acquisition,
    plan_dynamic_approach,
)
from scripts.aufgabe04.navigation.dynamic_replan_policy import (
    DynamicReplanPolicy,
    DynamicReplanState,
)
from scripts.aufgabe04.navigation.map_io import load_occupancy_grid
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.route_revision_store import (
    RouteRevisionError,
    RouteRevisionStore,
    read_committed_revision,
    read_route_revision,
)
from scripts.aufgabe04.navigation.viewpoint_recommendation import (
    MaterialTarget,
    angular_distance,
    load_recommendation,
    recommendation_to_dict,
)


_POINT_APPROACH_AXIS_STATES = frozenset(
    {"axis_acquisition", "viewpoint_sampling"}
)
_PHYSICAL_AXIS_STATES = frozenset({"target_committed", "resolved"})


def route_kind_for_axis_state(axis_state: str) -> str:
    if axis_state in _POINT_APPROACH_AXIS_STATES:
        return axis_state
    if axis_state in _PHYSICAL_AXIS_STATES:
        return "synchronized_face_approach"
    raise ValueError(f"unsupported synchronized viewpoint axis state: {axis_state}")


def load_recommended_payload(path: Path) -> tuple[Pose2D, Pose2D | None, dict]:
    payload = json.loads(Path(path).read_text())
    if payload.get("source") != "synchronized_lidar_camera_viewpoint":
        raise ValueError("recommended viewpoint source mismatch")
    pose = payload.get("pose")
    if not isinstance(pose, dict):
        raise ValueError("recommended viewpoint is missing pose")
    values = (pose.get("x_m"), pose.get("y_m"), pose.get("yaw_rad"))
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise ValueError("recommended viewpoint pose must be finite")
    target = Pose2D(float(values[0]), float(values[1]), float(values[2]))
    robot = payload.get("robot_pose")
    start = None
    if isinstance(robot, dict):
        robot_values = (robot.get("x_m"), robot.get("y_m"), robot.get("yaw_rad"))
        if all(isinstance(value, (int, float)) and math.isfinite(value) for value in robot_values):
            start = Pose2D(*(float(value) for value in robot_values))
    return target, start, payload


def load_recommended_pose(path: Path) -> Pose2D:
    return load_recommended_payload(path)[0]


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _raw_face_mapping(recommendation, config: DynamicApproachConfig) -> tuple[dict[int, str], int | None]:
    stable_faces = recommendation.face_candidates
    stand_axis = stable_faces[0].outward_normal_rad - math.pi / 2.0
    raw_faces = planner_face_candidates(recommendation.stand.center, stand_axis, config)
    direct = angular_distance(raw_faces[0].normal_rad, stable_faces[0].outward_normal_rad) + angular_distance(
        raw_faces[1].normal_rad, stable_faces[1].outward_normal_rad
    )
    swapped = angular_distance(raw_faces[0].normal_rad, stable_faces[1].outward_normal_rad) + angular_distance(
        raw_faces[1].normal_rad, stable_faces[0].outward_normal_rad
    )
    ordered = stable_faces if direct <= swapped else (stable_faces[1], stable_faces[0])
    face_id_by_raw = {0: ordered[0].face_id, 1: ordered[1].face_id}
    hard_raw = None
    evidence = recommendation.side_evidence
    if evidence.hard and evidence.valid:
        if evidence.face_id is None:
            raise ValueError("hard side evidence has no physical face ID")
        hard_face = next(face for face in stable_faces if face.face_id == evidence.face_id)
        if not hard_face.identity_resolved:
            raise ValueError("hard side evidence references unresolved face identity")
        hard_raw = next(index for index, face_id in face_id_by_raw.items() if face_id == evidence.face_id)
    return face_id_by_raw, hard_raw


def _route_csv_text(
    costmap,
    plan,
    *,
    stream_id: str,
    target_revision: int,
    route_kind: str = "synchronized_viewpoint",
) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        [
            "leg_index",
            "point_index",
            "grid_x",
            "grid_y",
            "world_x_m",
            "world_y_m",
            "yaw_rad",
            "segment_length_m",
            "cumulative_length_m",
            "protected",
            "corridor",
            "simulation_only",
            "route_kind",
            "stream_id",
            "route_revision",
            "target_revision",
            "manifest_path",
        ]
    )
    cumulative = 0.0
    previous = None
    for index, waypoint in enumerate(plan.waypoints):
        pose = waypoint.pose
        segment = 0.0 if previous is None else math.hypot(
            pose.x_m - previous.x_m, pose.y_m - previous.y_m
        )
        cumulative += segment
        cell = costmap.world_to_grid(pose)
        writer.writerow(
            [
                0,
                index,
                cell.x,
                cell.y,
                pose.x_m,
                pose.y_m,
                "" if not math.isfinite(pose.yaw_rad) else pose.yaw_rad,
                segment,
                cumulative,
                str(waypoint.protected).lower(),
                str(waypoint.corridor).lower(),
                "true",
                route_kind,
                stream_id,
                "",
                target_revision,
                "",
            ]
        )
        previous = pose
    return output.getvalue()


def _diagnostics_payload(plan_result, recommendation, target_revision: int) -> dict:
    plan = plan_result.plan
    return {
        "metadata": {
            "stage": "dynamic_synchronized_viewpoint_refinement",
            "simulation_only": True,
            "stream_id": recommendation.stream_id,
            "stand_id": recommendation.stand_id,
            "planning_frame": recommendation.planning_frame,
            "target_revision": target_revision,
            "source": recommendation.source,
            "approach_phase": recommendation.axis_state,
        },
        "legs": [
            {
                "diagnostics": {
                    "status": "ok" if plan is not None else "failed",
                    "reason": plan_result.diagnostics.failure_reason or "",
                    "route_length_m": 0.0 if plan is None else plan.length_m,
                    "dynamic_approach": asdict(plan_result.diagnostics),
                },
                "failure": (
                    None
                    if plan is not None
                    else {"reason": plan_result.diagnostics.failure_reason or "planning_failed"}
                ),
                "route_length_m": None if plan is None else plan.length_m,
                "route_point_count": 0 if plan is None else len(plan.waypoints),
            }
        ],
    }


def _publish_compatibility_aliases(args, committed) -> None:
    if committed.route_path is not None:
        _atomic_bytes(args.route_csv, committed.route_path.read_bytes())
    if committed.diagnostics_path is not None:
        _atomic_bytes(args.diagnostics_json, committed.diagnostics_path.read_bytes())


def _restart_state(existing) -> DynamicReplanState:
    """Restore the material target represented by an existing manifest."""

    if existing is None:
        return DynamicReplanState()
    revision = existing.target_revision
    target_payload = existing.manifest.get("target", {})
    if revision == 0 and not target_payload:
        return DynamicReplanState()
    if not isinstance(target_payload, dict):
        raise ValueError("existing route manifest target is malformed")
    try:
        pose = Pose2D(
            float(target_payload["x_m"]),
            float(target_payload["y_m"]),
            float(target_payload["yaw_rad"]),
        )
        face_id = str(target_payload["face_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("existing route manifest target is incomplete") from exc
    if not face_id or not all(math.isfinite(value) for value in asdict(pose).values()):
        raise ValueError("existing route manifest target is not finite")
    evidence_state = str(target_payload.get("evidence_state", ""))
    if not evidence_state:
        evidence = existing.manifest.get("evidence", {})
        evidence_state = (
            "hard_qr"
            if isinstance(evidence, dict)
            and evidence.get("hard") is True
            and evidence.get("valid") is True
            and evidence.get("face_id") == face_id
            else "ambiguous_axis"
        )
    current_target = MaterialTarget(face_id, pose, evidence_state)

    planned_start = None
    start_payload = existing.manifest.get("source_robot_pose", {})
    if isinstance(start_payload, dict) and start_payload:
        try:
            candidate = Pose2D(
                float(start_payload["x_m"]),
                float(start_payload["y_m"]),
                float(start_payload["yaw_rad"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("existing route manifest source_robot_pose is malformed") from exc
        if not all(math.isfinite(value) for value in asdict(candidate).values()):
            raise ValueError("existing route manifest source_robot_pose is not finite")
        planned_start = candidate

    active = existing.status == "active"
    return DynamicReplanState(
        target_revision=revision,
        current_target=current_target,
        last_observed_time_sec=float(existing.manifest["published_unix_sec"]),
        last_route_plan_time_sec=(
            float(existing.manifest["published_unix_sec"]) if active else None
        ),
        last_planned_start=planned_start if active else None,
        last_planned_target_revision=revision if active else 0,
    )


def _active_publication(existing) -> dict | None:
    """Load the immutable geometry retained for heartbeat-only revisions."""

    if existing is None or existing.status != "active":
        return None
    if existing.route_path is None or existing.diagnostics_path is None:
        raise ValueError("existing active route is missing immutable artifacts")
    manifest = existing.manifest
    required_mappings = {}
    for name in ("source_robot_pose", "target", "evidence", "safety_diagnostics"):
        value = manifest.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"existing route manifest {name} is malformed")
        required_mappings[name] = dict(value)
    diagnostics = json.loads(existing.diagnostics_path.read_text())
    if not isinstance(diagnostics, dict):
        raise ValueError("existing route diagnostics must be an object")
    return {
        "route_text": existing.route_path.read_text(),
        "diagnostics": diagnostics,
        **required_mappings,
        "route_length_m": float(manifest.get("new_route_length_m", 0.0)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--start-x", required=True, type=float)
    parser.add_argument("--start-y", required=True, type=float)
    parser.add_argument("--start-yaw", type=float, default=0.0)
    parser.add_argument("--recommended-pose-json", required=True, type=Path)
    parser.add_argument("--route-csv", required=True, type=Path)
    parser.add_argument("--diagnostics-json", required=True, type=Path)
    parser.add_argument("--route-manifest", type=Path, default=None)
    parser.add_argument("--stream-id", default="sim-stand-viewpoint")
    parser.add_argument("--writer-id", default="synchronized-viewpoint-planner")
    parser.add_argument("--writer-takeover", action="store_true")
    parser.add_argument("--map-frame", default="odom")
    parser.add_argument("--robot-radius-m", type=float, default=0.105)
    parser.add_argument("--tracking-margin-m", type=float, default=0.03)
    parser.add_argument("--collision-margin-m", type=float, default=0.02)
    parser.add_argument("--inflation-radius-m", type=float, default=None)
    parser.add_argument("--standoff-distance-m", type=float, default=None)
    parser.add_argument("--terminal-corridor-length-m", type=float, default=0.40)
    parser.add_argument("--corridor-sample-spacing-m", type=float, default=0.05)
    parser.add_argument("--lidar-stop-distance-m", type=float, default=0.18)
    parser.add_argument("--scan-origin-to-base-offset-m", type=float, default=0.0)
    parser.add_argument("--lidar-clearance-margin-m", type=float, default=0.02)
    parser.add_argument("--max-recommendation-age-sec", type=float, default=3.0)
    parser.add_argument("--target-position-threshold-m", type=float, default=0.06)
    parser.add_argument("--target-yaw-threshold-deg", type=float, default=10.0)
    parser.add_argument("--start-deviation-threshold-m", type=float, default=0.15)
    parser.add_argument(
        "--replan-on-start-deviation",
        action="store_true",
        help=(
            "Legacy behavior: replace route geometry when the moving robot leaves "
            "the last planned start. By default only material target changes replan."
        ),
    )
    parser.add_argument("--refresh-timeout-sec", type=float, default=4.0)
    parser.add_argument(
        "--terminal-route-lock-distance-m",
        type=float,
        default=0.42,
        help=(
            "Keep the installed collision-checked terminal corridor inside this "
            "target distance unless the material camera target changes."
        ),
    )
    parser.add_argument("--snap-radius-m", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--watch", action="store_true", help="Continuously replan from observer robot/target poses.")
    parser.add_argument("--replan-rate-hz", type=float, default=0.75)
    parser.add_argument("--max-replans", type=int, default=0, help="Stop after N plans; zero means run until interrupted.")
    args = parser.parse_args(argv)
    if args.snap_radius_m != 0.0:
        parser.error("dynamic corridor entry snapping is forbidden; --snap-radius-m must be 0")
    if args.max_replans < 0:
        parser.error("--max-replans must be non-negative")
    if (
        args.max_recommendation_age_sec <= 0.0
        or args.refresh_timeout_sec <= 0.0
        or args.terminal_route_lock_distance_m <= 0.0
    ):
        parser.error("freshness and refresh timeouts must be positive")
    args.route_manifest = args.route_manifest or args.route_csv.with_suffix(".manifest.json")
    try:
        costmap = Costmap.from_occupancy_grid(load_occupancy_grid(args.map))
        static_inflation = (
            args.robot_radius_m + args.tracking_margin_m
            if args.inflation_radius_m is None
            else args.inflation_radius_m
        )
        if static_inflation <= 0.0:
            raise ValueError("configuration-space inflation radius must be positive")
        costmap = costmap.with_inflation(static_inflation)
        store = RouteRevisionStore(
            args.route_manifest,
            stream_id=args.stream_id,
            writer_id=args.writer_id,
            # Use the planner's clock lookup rather than the store module's
            # import-time default so restart and test clock domains match.
            now_fn=lambda: time.time(),
        )
        existing = None
        if args.route_manifest.exists():
            existing = read_route_revision(
                args.route_manifest,
                expected_stream_id=args.stream_id,
            )
        policy = DynamicReplanPolicy(
            target_position_threshold_m=args.target_position_threshold_m,
            target_yaw_threshold_rad=math.radians(args.target_yaw_threshold_deg),
            start_deviation_threshold_m=args.start_deviation_threshold_m,
            refresh_timeout_sec=args.refresh_timeout_sec,
            terminal_route_lock_distance_m=args.terminal_route_lock_distance_m,
            replan_on_start_deviation=args.replan_on_start_deviation,
        )
        state = _restart_state(existing)
        active_publication = _active_publication(existing)
        plans = 0
        previous_route_length = (
            0.0
            if existing is None
            else float(existing.manifest.get("new_route_length_m", 0.0))
        )
        last_withdrawal_reason = None
        while True:
            now = time.time()
            try:
                recommendation = load_recommendation(
                    args.recommended_pose_json,
                    expected_frame=args.map_frame,
                    expected_source="synchronized_lidar_camera_viewpoint",
                    now_unix_sec=now,
                    max_age_sec=args.max_recommendation_age_sec,
                )
                if recommendation.stream_id != args.stream_id:
                    raise ValueError("viewpoint recommendation stream_id mismatch")
                if recommendation.axis_state.startswith("invalid_"):
                    raise ValueError(recommendation.axis_state)
                route_kind = route_kind_for_axis_state(recommendation.axis_state)
                start = (
                    recommendation.robot_pose
                    if args.watch
                    else Pose2D(args.start_x, args.start_y, args.start_yaw)
                )
                observed_standoff = math.hypot(
                    recommendation.material_target.pose.x_m - recommendation.stand.center.x_m,
                    recommendation.material_target.pose.y_m - recommendation.stand.center.y_m,
                )
                config = DynamicApproachConfig(
                    stand_radius_m=recommendation.stand.radius_m,
                    stand_position_uncertainty_m=recommendation.stand.uncertainty_m,
                    robot_radius_m=args.robot_radius_m,
                    collision_margin_m=args.collision_margin_m,
                    standoff_distance_m=(
                        observed_standoff
                        if args.standoff_distance_m is None
                        else args.standoff_distance_m
                    ),
                    terminal_corridor_length_m=args.terminal_corridor_length_m,
                    corridor_sample_spacing_m=args.corridor_sample_spacing_m,
                    lidar_stop_distance_m=args.lidar_stop_distance_m,
                    scan_origin_to_base_offset_m=args.scan_origin_to_base_offset_m,
                    lidar_clearance_margin_m=args.lidar_clearance_margin_m,
                )
                if recommendation.axis_state in _POINT_APPROACH_AXIS_STATES:
                    result = plan_axis_acquisition(
                        costmap,
                        start,
                        recommendation.stand.center,
                        recommendation.material_target.pose,
                        config=config,
                    )
                    effective_target = recommendation.material_target
                else:
                    face_id_by_raw, hard_raw_face = _raw_face_mapping(recommendation, config)
                    if hard_raw_face is None:
                        # A committed silhouette axis resolves the physical
                        # side currently facing the robot even when QR identity
                        # is still unknown. Never replace it with the opposite
                        # side merely because that route is shorter.
                        selected_physical = [
                            raw_id
                            for raw_id, stable_id in face_id_by_raw.items()
                            if stable_id == recommendation.material_target.face_id
                        ]
                        if selected_physical:
                            hard_raw_face = selected_physical[0]
                    if hard_raw_face is None and state.current_target is not None:
                        # Robot motion alone must not switch an ambiguous physical
                        # face merely because the other face became Euclidean-
                        # shorter. The observer/QR evidence owns face changes.
                        matching = [
                            raw_id
                            for raw_id, stable_id in face_id_by_raw.items()
                            if stable_id == state.current_target.face_id
                        ]
                        if matching:
                            hard_raw_face = matching[0]
                    stand_axis = recommendation.face_candidates[0].outward_normal_rad - math.pi / 2.0
                    result = plan_dynamic_approach(
                        costmap,
                        start,
                        recommendation.stand.center,
                        stand_axis,
                        hard_face_id=hard_raw_face,
                        config=config,
                    )
                    if result.plan is not None:
                        selected_face_id = face_id_by_raw[result.plan.selected_face_id]
                        effective_target = MaterialTarget(
                            face_id=selected_face_id,
                            pose=result.plan.target,
                            evidence_state=recommendation.material_target.evidence_state,
                        )
                if result.plan is None:
                    raise ValueError(result.diagnostics.failure_reason or "dynamic planning failed")
                # Revision the target the collision-aware planner actually
                # selected.  With ambiguous evidence the shortest valid face
                # may differ from the observer's provisional preference.
                state, decision = policy.evaluate(
                    state,
                    target=effective_target,
                    robot_pose=start,
                    now_sec=now,
                )
                if decision.fail_closed:
                    raise ValueError(";".join(decision.reasons))
                if not decision.should_replan:
                    if not args.watch:
                        return 0
                    time.sleep(1.0 / max(args.replan_rate_hz, 0.1))
                    continue
                heartbeat_only = (
                    active_publication is not None
                    and not decision.target_changed
                    and set(decision.reasons) == {"refresh_timeout"}
                )
                rec_payload = recommendation_to_dict(recommendation)
                if heartbeat_only:
                    committed = store.publish_active(
                        active_publication["route_text"],
                        active_publication["diagnostics"],
                        target_revision=decision.target_revision,
                        observation_unix_sec=recommendation.observation_unix_sec,
                        source_robot_pose=active_publication["source_robot_pose"],
                        target=active_publication["target"],
                        evidence=rec_payload["side_evidence"],
                        previous_route_length_m=active_publication["route_length_m"],
                        new_route_length_m=active_publication["route_length_m"],
                        safety_diagnostics=active_publication["safety_diagnostics"],
                        takeover=args.writer_takeover,
                    )
                    _publish_compatibility_aliases(args, committed)
                    state = policy.mark_route_planned(
                        state,
                        planned_start=Pose2D(
                            float(active_publication["source_robot_pose"]["x_m"]),
                            float(active_publication["source_robot_pose"]["y_m"]),
                            float(active_publication["source_robot_pose"]["yaw_rad"]),
                        ),
                        now_sec=now,
                        target_revision=decision.target_revision,
                    )
                    last_withdrawal_reason = None
                    time.sleep(1.0 / max(args.replan_rate_hz, 0.1))
                    continue
                route_text = _route_csv_text(
                    costmap,
                    result.plan,
                    stream_id=recommendation.stream_id,
                    target_revision=decision.target_revision,
                    route_kind=(
                        route_kind
                    ),
                )
                diagnostics = _diagnostics_payload(
                    result, recommendation, decision.target_revision
                )
                target_payload = {
                    **asdict(effective_target.pose),
                    "face_id": effective_target.face_id,
                    "evidence_state": effective_target.evidence_state,
                }
                safety_payload = {
                    **asdict(result.diagnostics),
                    "static_inflation_radius_m": static_inflation,
                }
                committed = store.publish_active(
                    route_text,
                    diagnostics,
                    target_revision=decision.target_revision,
                    observation_unix_sec=recommendation.observation_unix_sec,
                    source_robot_pose=asdict(start),
                    target=target_payload,
                    evidence=rec_payload["side_evidence"],
                    previous_route_length_m=previous_route_length,
                    new_route_length_m=result.plan.length_m,
                    safety_diagnostics=safety_payload,
                    takeover=args.writer_takeover,
                )
                _publish_compatibility_aliases(args, committed)
                previous_route_length = result.plan.length_m
                active_publication = {
                    "route_text": route_text,
                    "diagnostics": diagnostics,
                    "source_robot_pose": asdict(start),
                    "target": target_payload,
                    "evidence": rec_payload["side_evidence"],
                    "safety_diagnostics": safety_payload,
                    "route_length_m": result.plan.length_m,
                }
                state = policy.mark_route_planned(
                    state,
                    planned_start=start,
                    now_sec=now,
                    target_revision=decision.target_revision,
                )
                last_withdrawal_reason = None
                plans += 1
            except (OSError, ValueError, RouteRevisionError, json.JSONDecodeError) as exc:
                reason = str(exc)
                if reason != last_withdrawal_reason:
                    store.withdraw(
                        reason,
                        target_revision=state.target_revision,
                        observation_unix_sec=now,
                        takeover=args.writer_takeover,
                    )
                    last_withdrawal_reason = reason
                if not args.watch:
                    return 1
            if not args.watch or (args.max_replans > 0 and plans >= args.max_replans):
                return 0
            time.sleep(1.0 / max(args.replan_rate_hz, 0.1))
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, RouteRevisionError, json.JSONDecodeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
