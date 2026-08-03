#!/usr/bin/env python3
"""Run one fail-closed autonomous real-robot stand exploration mission.

The mission plans a single center rail, drives certified A* legs to stopped
inspection poses, fuses LiDAR candidates across those poses, visits every
stable candidate at a robot-facing pre-approach, and commits calibrated
camera/LiDAR QR-face poses.  Physical execution requires ``--execute`` and one
mission-level typed ``RUN``.  Every motion leg still passes the existing
route, ROS, obstacle, localization, and exclusive-velocity-owner gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.artifacts.content_store import (
    payload_sha256,
    write_content_hashed_json,
)
from scripts.aufgabe04.navigation.artifacts import (
    write_diagnostics_json,
    write_route_csv,
)
from scripts.aufgabe04.navigation.costmap import Costmap
from scripts.aufgabe04.navigation.detected_stand_preapproach import (
    CAMERA_AXIS_FACE_BEARING_MODE,
    ROBOT_TO_STAND_BEARING_MODE,
    seal_detected_stand_preapproach,
)
from scripts.aufgabe04.navigation.dynamic_approach_planner import (
    DynamicApproachConfig,
    minimum_static_obstacle_inflation_m,
)
from scripts.aufgabe04.navigation.global_planner import plan_route
from scripts.aufgabe04.navigation.map_io import load_occupancy_grid_with_bundle
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.plan_stand_coverage_survey import (
    main as plan_stand_coverage_survey,
)
from scripts.aufgabe04.navigation.read_current_amcl_pose import (
    read_current_amcl_pose,
)
from scripts.aufgabe04.navigation.record_stand_candidate_decision import (
    main as record_stand_candidate_decision,
)
from scripts.aufgabe04.navigation.record_stand_coverage_stop import (
    main as record_stand_coverage_stop,
)
from scripts.aufgabe04.navigation.route_context import build_station_route_dry_run
from scripts.aufgabe04.navigation.stand_coverage_survey import (
    STATUS_PENDING_CAMERA,
    CoverageSurveyPlan,
    StandSurveyRegistry,
    coverage_survey_plan_sha256,
    load_coverage_survey_plan,
    load_stand_survey_registry,
)
from scripts.aufgabe04.navigation.stand_discovery_route import (
    seal_stand_discovery_route,
)
from scripts.aufgabe04.navigation.viewpoint_recommendation import (
    load_recommendation,
    normalize_angle,
)
from scripts.aufgabe04.perception.stand_axis.model_profile import (
    load_stand_model,
)
from scripts.aufgabe04.real_robot.hardware_profile import (
    camera_calibration_sha256,
    load_camera_calibration,
    load_real_robot_profile,
)
from scripts.aufgabe04.real_robot.recommendation_builder import (
    REAL_VIEWPOINT_SOURCE,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateGeometry,
    CandidateSource,
    FrozenCandidate,
    candidate_snapshot_sha256,
    new_candidate_snapshot,
    write_candidate_snapshot,
)
from scripts.aufgabe04.stations.create_station_identity_registry import (
    create_registry,
)
from scripts.aufgabe04.stations.models import Station, StationPose
from scripts.aufgabe04.stations.station_identity_registry import (
    StationIdentity,
    station_identity_registry_sha256,
    write_station_identity_registry,
)


DEFAULT_MAP = Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml")
DEFAULT_OUTPUT_ROOT = Path("results/aufgabe04/real/autonomous_exploration")
DEFAULT_TRACKING_TUBE_RADIUS_M = 0.03
DEFAULT_COLLISION_MARGIN_M = 0.02
DEFAULT_LIDAR_STOP_DISTANCE_M = 0.20
DEFAULT_LIDAR_CLEARANCE_MARGIN_M = 0.02


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_session_id() -> str:
    return "stand_explore_" + time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _physical_clearance(profile, *, approach_offset_m: float) -> dict[str, float]:
    config = DynamicApproachConfig(
        stand_radius_m=0.06,
        stand_position_uncertainty_m=0.02,
        robot_radius_m=profile.robot_radius_m,
        collision_margin_m=DEFAULT_COLLISION_MARGIN_M,
        tracking_margin_m=DEFAULT_TRACKING_TUBE_RADIUS_M,
        standoff_distance_m=approach_offset_m,
        lidar_stop_distance_m=DEFAULT_LIDAR_STOP_DISTANCE_M,
        scan_origin_to_base_offset_m=profile.scan_origin_to_base_offset_m,
        lidar_clearance_margin_m=DEFAULT_LIDAR_CLEARANCE_MARGIN_M,
        minimum_non_target_keepout_radius_m=0.31,
    )
    return {
        "minimum_static_inflation_m": minimum_static_obstacle_inflation_m(
            robot_radius_m=profile.robot_radius_m,
            tracking_margin_m=DEFAULT_TRACKING_TUBE_RADIUS_M,
            lidar_stop_distance_m=DEFAULT_LIDAR_STOP_DISTANCE_M,
            scan_origin_to_base_offset_m=profile.scan_origin_to_base_offset_m,
            lidar_clearance_margin_m=DEFAULT_LIDAR_CLEARANCE_MARGIN_M,
        ),
        "minimum_active_standoff_m": config.minimum_lidar_standoff_m,
        "minimum_candidate_transit_radius_m": (
            config.non_target_stand_keepout_radius_m
        ),
    }


def candidate_snapshot_from_registry(
    registry: StandSurveyRegistry,
    plan: CoverageSurveyPlan,
    *,
    registry_path: Path,
    snapshot_id: str,
):
    """Freeze the persistent multi-viewpoint registry for route certification."""

    pending = tuple(
        candidate
        for candidate in registry.candidates
        if candidate.status == STATUS_PENDING_CAMERA
    )
    if not pending:
        raise ValueError("stand registry has no pending-camera candidates")
    detector_config_sha256 = payload_sha256(
        {
            "source": "stand_coverage_survey",
            "plan_sha256": coverage_survey_plan_sha256(plan),
            "config": {
                "candidate_merge_distance_m": (
                    plan.config.candidate_merge_distance_m
                ),
                "minimum_candidate_confidence": (
                    plan.config.minimum_candidate_confidence
                ),
                "minimum_distinct_viewpoints": (
                    plan.config.minimum_distinct_viewpoints
                ),
                "minimum_candidate_hits": plan.config.minimum_candidate_hits,
            },
        }
    )
    return new_candidate_snapshot(
        snapshot_id=snapshot_id,
        created_unix_sec=max(candidate.last_seen_sec for candidate in pending),
        planning_frame=plan.planning_frame,
        map_bundle_sha256=plan.map_bundle_sha256,
        candidates=(
            FrozenCandidate(
                candidate_uid=candidate.candidate_uid,
                geometry=CandidateGeometry(
                    x_m=candidate.x_m,
                    y_m=candidate.y_m,
                    radius_m=candidate.radius_m,
                    uncertainty_m=candidate.uncertainty_m,
                    keepout_radius_m=candidate.keepout_radius_m,
                ),
                source=CandidateSource(
                    source_kind="lidar/stand_coverage_survey",
                    source_artifact_sha256=_file_sha256(registry_path),
                    detector_config_sha256=detector_config_sha256,
                    observation_ids=candidate.source_observation_ids,
                ),
                confidence=candidate.confidence,
                hit_count=candidate.hit_count,
                first_seen_sec=candidate.first_seen_sec,
                last_seen_sec=candidate.last_seen_sec,
            )
            for candidate in pending
        ),
    )


def _nearest_candidate(snapshot, current_pose: Pose2D, unresolved: set[str]):
    options = [
        candidate
        for candidate in snapshot.candidates
        if candidate.candidate_uid in unresolved
    ]
    if not options:
        return None
    return min(
        options,
        key=lambda candidate: (
            math.hypot(
                current_pose.x_m - candidate.geometry.x_m,
                current_pose.y_m - candidate.geometry.y_m,
            ),
            candidate.candidate_uid,
        ),
    )


def plan_candidate_preapproach(
    *,
    map_yaml: Path,
    semantic_map_id: str,
    plan: CoverageSurveyPlan,
    snapshot,
    snapshot_path: Path,
    candidate_uid: str,
    start: Pose2D,
    output_dir: Path,
    approach_offset_m: float,
    inflation_radius_m: float,
    candidate_transit_radius_m: float,
    physical_clearance: dict[str, float],
    approach_normal_rad: float | None = None,
    axis_observation_path: Path | None = None,
) -> dict[str, str]:
    """Write and seal a robot-side or axis-selected stand inspection route."""

    candidate = snapshot.candidate_for(candidate_uid)
    if candidate is None:
        raise ValueError(f"unknown candidate {candidate_uid!r}")
    if (approach_normal_rad is None) != (axis_observation_path is None):
        raise ValueError(
            "axis-selected approach requires both normal and observation"
        )
    if approach_normal_rad is None:
        bearing = math.atan2(
            candidate.geometry.y_m - start.y_m,
            candidate.geometry.x_m - start.x_m,
        )
        bearing_mode = ROBOT_TO_STAND_BEARING_MODE
    else:
        if not math.isfinite(approach_normal_rad):
            raise ValueError("approach face normal must be finite")
        bearing = normalize_angle(approach_normal_rad + math.pi)
        bearing_mode = CAMERA_AXIS_FACE_BEARING_MODE
    stations = {}
    target_station_id = "D00"
    for index, item in enumerate(snapshot.candidates, start=1):
        yaw = bearing if item.candidate_uid == candidate_uid else 0.0
        station_id = (
            target_station_id
            if item.candidate_uid == candidate_uid
            else f"K{index:02d}"
        )
        stations[station_id] = Station(
            station_id,
            StationPose(item.geometry.x_m, item.geometry.y_m, yaw),
            approach_offset_m,
            candidate_transit_radius_m,
        )
    grid, map_bundle = load_occupancy_grid_with_bundle(
        map_yaml,
        semantic_map_id=semantic_map_id,
        planning_frame=plan.planning_frame,
    )
    if map_bundle.bundle_sha256 != snapshot.map_bundle_sha256:
        raise ValueError("candidate snapshot map differs from runtime map")
    dry_run = build_station_route_dry_run(
        map_yaml,
        [target_station_id],
        station_map=stations,
        start=start,
        inflation_radius_m=inflation_radius_m,
        snap_radius_m=plan.config.snap_radius_m,
        transit_keepout_radius_m=candidate_transit_radius_m,
        arena_bounds=plan.arena_bounds,
        occupancy_grid=grid,
        map_bundle=map_bundle,
    )
    result = dry_run.results[0]
    if result.route is None or result.failure is not None:
        reason = result.failure.reason if result.failure is not None else "no route"
        raise ValueError(f"candidate pre-approach A* failed: {reason}")
    endpoint = result.route.points[-1].pose
    terminal_yaw = math.atan2(
        candidate.geometry.y_m - endpoint.y_m,
        candidate.geometry.x_m - endpoint.x_m,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    route_csv = output_dir / "route.csv"
    diagnostics_json = output_dir / "route_diagnostics.json"
    pipeline_summary = output_dir / "pipeline_summary.json"
    local_snapshot = output_dir / "candidate_snapshot.json"
    shutil.copyfile(snapshot_path, local_snapshot)
    local_axis_observation = None
    if axis_observation_path is not None:
        local_axis_observation = output_dir / "axis_observation.json"
        shutil.copyfile(axis_observation_path, local_axis_observation)
    write_route_csv(route_csv, dry_run.results, final_yaw_by_leg={0: terminal_yaw})
    metadata = dict(dry_run.metadata)
    metadata.update(
        {
            "source": "lidar_detected_stand_exploration",
            "order": "nearest",
            "plan_mode": "next-candidate",
            "stand_count": 1,
            "candidate_transit_radius_m": candidate_transit_radius_m,
            "inflation_radius_m": inflation_radius_m,
            "approach_offset_m": approach_offset_m,
            "approach_bearing_mode": bearing_mode,
            "physical_clearance_enforced": True,
            "physical_clearance": physical_clearance,
            "candidate_snapshot_json": str(local_snapshot),
            "candidate_snapshot_sha256": candidate_snapshot_sha256(snapshot),
            "map_bundle_sha256": map_bundle.bundle_sha256,
            "planning_frame": plan.planning_frame,
            "selected_candidate_stand_id": candidate_uid,
            "selected_approach_pose": {
                "x_m": endpoint.x_m,
                "y_m": endpoint.y_m,
                "yaw_rad": terminal_yaw,
            },
        }
    )
    if local_axis_observation is not None:
        metadata.update(
            {
                "axis_observation_json": str(
                    local_axis_observation.resolve()
                ),
                "axis_observation_sha256": _file_sha256(
                    local_axis_observation
                ),
                "selected_face_normal_rad": normalize_angle(
                    float(approach_normal_rad)
                ),
            }
        )
    write_diagnostics_json(diagnostics_json, dry_run.results, metadata=metadata)
    _write_json(
        pipeline_summary,
        {
            "schema_version": 1,
            "status": "observe_and_plan_complete",
            "motion_published": False,
            "selected_candidate_uid": candidate_uid,
            "selected_approach_pose": metadata["selected_approach_pose"],
            "physical_clearance": physical_clearance,
        },
    )
    return seal_detected_stand_preapproach(pipeline_root=output_dir)


def _runner_command(
    *,
    profile,
    route_csv: Path,
    diagnostics_json: Path,
    certificate_json: Path,
    run_id: str,
    session_root: Path,
    coverage_plan: Path | None = None,
    candidate_snapshot: Path | None = None,
    dry_run: bool,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/aufgabe04/navigation/run_single_station_segment.py",
        "--route-csv",
        str(route_csv),
        "--diagnostics-json",
        str(diagnostics_json),
        "--route-certificate-json",
        str(certificate_json),
        "--leg-index",
        "0",
        "--run-id",
        run_id,
        "--robot-id",
        profile.robot_id,
        "--namespace",
        profile.namespace,
        "--scan-topic",
        profile.scan_topic,
        "--odom-topic",
        profile.odom_topic,
        "--cmd-vel-topic",
        profile.cmd_vel_topic,
        "--amcl-topic",
        profile.amcl_topic,
        "--map-frame",
        profile.map_frame,
        "--odom-frame",
        profile.odom_frame,
        "--base-frame",
        profile.base_frame,
        "--localization-source",
        profile.localization_source,
        "--max-linear-mps",
        str(profile.max_linear_speed_mps),
        "--max-angular-radps",
        str(profile.max_angular_speed_radps),
        "--min-obstacle-distance-m",
        str(DEFAULT_LIDAR_STOP_DISTANCE_M),
        "--certified-route-tube-radius-m",
        str(DEFAULT_TRACKING_TUBE_RADIUS_M),
        "--results-csv",
        str(session_root / "station_segment_runs.csv"),
        "--semantic-log",
        str(session_root / "run_events" / f"{run_id}.jsonl"),
        "--preflight-json",
        str(session_root / "preflight" / f"{run_id}.json"),
        "--operator-note",
        "UNLOADED autonomous stand exploration",
    ]
    if coverage_plan is not None:
        command.extend(["--coverage-plan", str(coverage_plan)])
    if candidate_snapshot is not None:
        command.extend(["--candidate-snapshot", str(candidate_snapshot)])
    if dry_run:
        command.append("--dry-run")
    return command


def _bundle_command(profile, run_id: str, runner: list[str]) -> list[str]:
    return [
        "scripts/common/run_with_bundle.sh",
        "--namespace",
        profile.namespace,
        "--cmd-vel-topic",
        profile.cmd_vel_topic,
        "--scan-topic",
        profile.scan_topic,
        "--odom-topic",
        profile.odom_topic,
        "--amcl-topic",
        profile.amcl_topic,
        "--map-frame",
        profile.map_frame,
        "--odom-frame",
        profile.odom_frame,
        "--base-frame",
        profile.base_frame,
        run_id,
        "--",
        *runner,
    ]


def _run_motion_leg(
    *,
    profile,
    sealed: dict[str, str],
    run_id: str,
    session_root: Path,
    execute: bool,
    coverage_plan: Path | None = None,
    candidate_snapshot: Path | None = None,
) -> None:
    common = {
        "profile": profile,
        "route_csv": Path(sealed["route_csv"]),
        "diagnostics_json": Path(sealed["diagnostics_json"]),
        "certificate_json": Path(sealed["route_certificate_json"]),
        "run_id": run_id,
        "session_root": session_root,
        "coverage_plan": coverage_plan,
        "candidate_snapshot": candidate_snapshot,
    }
    dry = _runner_command(**common, dry_run=True)
    if subprocess.run(dry, check=False).returncode != 0:
        raise RuntimeError(f"dry-run failed for {run_id}")
    if not execute:
        return
    runner = _runner_command(**common, dry_run=False)
    wrapped = _bundle_command(profile, run_id, runner)
    result = subprocess.run(
        wrapped,
        input="RUN\n",
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"physical route failed for {run_id}")


def _capture_lidar_epoch(
    *,
    profile,
    args,
    survey_root: Path,
    viewpoint_id: str,
) -> Path:
    epoch_root = survey_root / "raw_epochs" / viewpoint_id
    epoch_root.mkdir(parents=True, exist_ok=False)
    summary = epoch_root / "observer_summary.json"
    command = [
        sys.executable,
        "scripts/aufgabe04/perception/stand_explorer_node.py",
        "--namespace",
        profile.namespace,
        "--scan-topic",
        profile.scan_topic,
        "--odom-topic",
        profile.odom_topic,
        "--cmd-vel-topic",
        profile.cmd_vel_topic,
        "--amcl-topic",
        profile.amcl_topic,
        "--map-frame",
        profile.map_frame,
        "--odom-frame",
        profile.odom_frame,
        "--base-frame",
        profile.base_frame,
        "--localization-source",
        profile.localization_source,
        "--map-yaml",
        str(args.map),
        "--semantic-map-id",
        args.semantic_map_id,
        "--duration-sec",
        str(args.lidar_epoch_sec),
        "--output-jsonl",
        str(epoch_root / "observations.jsonl"),
        "--summary-json",
        str(summary),
    ]
    if subprocess.run(command, check=False).returncode != 0:
        raise RuntimeError(f"LiDAR epoch failed at {viewpoint_id}")
    payload = json.loads(summary.read_text())
    if int(payload.get("processed_scan_count", 0)) <= 0:
        raise RuntimeError(f"LiDAR epoch processed no scans at {viewpoint_id}")
    return summary


def _capture_camera_recommendation(
    *,
    profile,
    args,
    candidate,
    output_dir: Path,
) -> tuple[Path | None, str | None, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "observer_status.json"
    recommendation_path = output_dir / "recommendation.json"
    axis_observation_path = output_dir / "axis_observation.json"
    command = [
        sys.executable,
        "scripts/aufgabe04/real_robot/passive_viewpoint_node.py",
        "--robot-profile",
        str(args.robot_profile),
        "--camera-calibration",
        str(args.camera_calibration),
        "--stream-id",
        f"{args.session_id}_{candidate.candidate_uid}",
        "--stand-id",
        candidate.candidate_uid,
        "--expected-qr-id",
        "auto",
        "--stand-x",
        str(candidate.geometry.x_m),
        "--stand-y",
        str(candidate.geometry.y_m),
        "--stand-radius-m",
        str(candidate.geometry.radius_m),
        "--stand-uncertainty-m",
        str(candidate.geometry.uncertainty_m),
        "--target-distance-m",
        str(args.final_facing_offset_m),
        "--consensus-frames",
        str(args.axis_sample_count),
        "--status-json",
        str(status_path),
        "--recommended-pose-json",
        str(recommendation_path),
        "--axis-observation-json",
        str(axis_observation_path),
        "--debug-dir",
        str(output_dir / "perception_debug"),
        "--once",
    ]
    if args.stand_model_profile is not None:
        stand_model = load_stand_model(args.stand_model_profile)
        command.extend(
            [
                "--stand-model-profile",
                str(args.stand_model_profile),
                "--stand-face-size-m",
                str(stand_model.head_width_m),
            ]
        )
    process = subprocess.Popen(command)
    deadline = time.monotonic() + args.camera_timeout_sec
    try:
        while process.poll() is None and time.monotonic() < deadline:
            if recommendation_path.exists():
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.send_signal(signal.SIGINT)
                break
            time.sleep(0.1)
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=5.0)
    if not recommendation_path.exists():
        if axis_observation_path.exists():
            return None, None, axis_observation_path
        state = (
            json.loads(status_path.read_text()).get("state")
            if status_path.exists()
            else "no_status"
        )
        raise RuntimeError(
            f"camera/LiDAR observation timed out for "
            f"{candidate.candidate_uid} without a usable axis: {state}"
        )
    status = json.loads(status_path.read_text())
    qr_texts = tuple(status.get("qr_texts", ()))
    if len(qr_texts) != 1 or not str(qr_texts[0]).strip():
        raise RuntimeError("camera recommendation did not bind one QR identity")
    return recommendation_path, str(qr_texts[0]), (
        axis_observation_path if axis_observation_path.exists() else None
    )


def _opposite_face_normal(axis_observation_path: Path) -> float:
    """Select the axis-perpendicular face opposite the first camera view."""

    payload = json.loads(axis_observation_path.read_text())
    if payload.get("observation_kind") != "real_stand_axis_without_qr":
        raise ValueError("unexpected axis observation kind")
    axis_rad = float(payload["stand_axis_rad"])
    stand = payload["stand_center"]
    robot = payload["robot_pose"]
    robot_side = math.atan2(
        float(robot["y_m"]) - float(stand["y_m"]),
        float(robot["x_m"]) - float(stand["x_m"]),
    )
    normals = (
        normalize_angle(axis_rad + math.pi / 2.0),
        normalize_angle(axis_rad - math.pi / 2.0),
    )
    selected = min(
        normals,
        key=lambda normal: math.cos(normal - robot_side),
    )
    if math.cos(selected - robot_side) > -0.5:
        raise ValueError(
            "stand axis does not resolve a sufficiently opposite inspection face"
        )
    return selected


def _bounded_approach_offsets(
    requested_m: float,
    minimum_m: float,
    *,
    step_m: float = 0.05,
) -> tuple[float, ...]:
    """Return descending standoffs without crossing the physical minimum."""

    if not all(
        math.isfinite(value) and value > 0.0
        for value in (requested_m, minimum_m, step_m)
    ):
        raise ValueError("approach offsets and step must be finite and positive")
    if requested_m + 1.0e-9 < minimum_m:
        raise ValueError("requested approach offset is below physical minimum")
    values = []
    current = requested_m
    while current > minimum_m + 1.0e-9:
        values.append(round(current, 6))
        current -= step_m
    if not values or abs(values[-1] - minimum_m) > 1.0e-9:
        values.append(round(minimum_m, 6))
    return tuple(values)


def _is_approach_feasibility_failure(exc: ValueError) -> bool:
    message = str(exc)
    return (
        "candidate pre-approach A* failed" in message
        or "target is blocked" in message
    )


def _validate_facing_pose(
    *,
    args,
    profile,
    plan,
    snapshot,
    candidate,
    recommendation_path: Path,
    current_pose: Pose2D,
    output_dir: Path,
    inflation_radius_m: float,
) -> dict[str, object]:
    recommendation = load_recommendation(
        recommendation_path,
        expected_frame=profile.map_frame,
        expected_source=REAL_VIEWPOINT_SOURCE,
        expected_simulation_only=False,
    )
    target = recommendation.material_target.pose
    grid, map_bundle = load_occupancy_grid_with_bundle(
        args.map,
        semantic_map_id=args.semantic_map_id,
        planning_frame=profile.map_frame,
    )
    if map_bundle.bundle_sha256 != plan.map_bundle_sha256:
        raise ValueError("facing-pose validation map differs from survey")
    costmap = (
        Costmap.from_occupancy_grid(grid)
        .with_arena_bounds(plan.arena_bounds)
        .with_inflation(inflation_radius_m)
    )
    other_keepouts = tuple(
        Station(
            item.candidate_uid,
            StationPose(item.geometry.x_m, item.geometry.y_m, 0.0),
            0.0,
            item.geometry.keepout_radius_m,
        )
        for item in snapshot.candidates
        if item.candidate_uid != candidate.candidate_uid
    )
    if other_keepouts:
        costmap = costmap.with_station_keepouts(other_keepouts)
    route = plan_route(
        costmap,
        current_pose,
        target,
        snap_radius_m=plan.config.snap_radius_m,
    )
    if route.route is None or route.failure is not None:
        reason = route.failure.reason if route.failure is not None else "no route"
        raise ValueError(f"computed QR-facing pose is not A*-reachable: {reason}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_route_csv(
        output_dir / "facing_pose_validation_route.csv",
        (route,),
        final_yaw_by_leg={0: target.yaw_rad},
    )
    write_diagnostics_json(
        output_dir / "facing_pose_validation_diagnostics.json",
        (route,),
        metadata={
            "route_kind": "facing_pose_validation_only",
            "motion_authorized": False,
            "map_bundle_sha256": map_bundle.bundle_sha256,
            "candidate_uid": candidate.candidate_uid,
            "arena_boundary_overlay": True,
            "arena_bounds": plan.arena_bounds.to_metadata(),
            "inflation_radius_m": inflation_radius_m,
        },
    )
    qr_face = next(
        face
        for face in recommendation.face_candidates
        if face.face_id == recommendation.material_target.face_id
    )
    # The stand axis is axial. Canonicalize it to [0, pi).
    stand_axis = (qr_face.outward_normal_rad - math.pi / 2.0) % math.pi
    return {
        "candidate_uid": candidate.candidate_uid,
        "stand_center": {
            "x_m": candidate.geometry.x_m,
            "y_m": candidate.geometry.y_m,
        },
        "stand_axis_rad_axial": stand_axis,
        "qr_outward_normal_rad": normalize_angle(qr_face.outward_normal_rad),
        "facing_pose": {
            "x_m": target.x_m,
            "y_m": target.y_m,
            "yaw_rad": target.yaw_rad,
        },
        "axis_confidence": recommendation.axis_confidence,
        "axis_sample_count": recommendation.axis_sample_count,
        "recommendation_json": str(recommendation_path),
        "validation_route_csv": str(
            output_dir / "facing_pose_validation_route.csv"
        ),
        "motion_to_facing_pose_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-profile", required=True, type=Path)
    parser.add_argument("--camera-calibration", required=True, type=Path)
    parser.add_argument("--physical-site", required=True, type=Path)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--semantic-map-id", default="arena_1p898x3p9_auto")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-stand-count", type=int, default=3)
    parser.add_argument("--inspection-stop-spacing-m", type=float, default=0.70)
    parser.add_argument("--lidar-epoch-sec", type=float, default=8.0)
    parser.add_argument("--candidate-approach-offset-m", type=float, default=0.70)
    parser.add_argument("--final-facing-offset-m", type=float, default=0.35)
    parser.add_argument("--axis-sample-count", type=int, default=7)
    parser.add_argument("--camera-timeout-sec", type=float, default=90.0)
    parser.add_argument(
        "--stand-model-profile",
        type=Path,
        default=None,
        help="Optional content-hashed measured physical stand model.",
    )
    parser.add_argument(
        "--coverage-leg-limit",
        type=int,
        default=0,
        help=(
            "Real-test checkpoint: stop successfully after this many coverage "
            "legs; zero means run the complete mission."
        ),
    )
    parser.add_argument(
        "--stop-after-coverage",
        action="store_true",
        help=(
            "Real-test checkpoint: finish the center-corridor LiDAR survey and "
            "candidate snapshot, then stop before candidate approaches."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="After every dry-run passes, permit physical motion.",
    )
    return parser


def _validate_inputs(parser, args, profile, calibration) -> None:
    if args.expected_stand_count <= 0:
        parser.error("--expected-stand-count must be positive")
    if args.coverage_leg_limit < 0:
        parser.error("--coverage-leg-limit must be non-negative")
    for name in (
        "inspection_stop_spacing_m",
        "lidar_epoch_sec",
        "candidate_approach_offset_m",
        "final_facing_offset_m",
        "camera_timeout_sec",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.axis_sample_count < 7:
        parser.error("--axis-sample-count must be at least seven")
    if camera_calibration_sha256(calibration) != (
        profile.calibration_profile_sha256
    ):
        parser.error("camera calibration differs from robot profile")
    if (
        args.physical_site.stem != profile.physical_site_id
        or _file_sha256(args.physical_site) != profile.physical_site_sha256
    ):
        parser.error("physical site descriptor differs from robot profile")
    if profile.localization_source != "amcl":
        parser.error("autonomous real exploration requires AMCL localization")
    if args.stand_model_profile is not None:
        try:
            stand_model = load_stand_model(args.stand_model_profile)
        except (OSError, ValueError) as exc:
            parser.error(f"invalid stand model profile: {exc}")
        if not stand_model.committable:
            parser.error(
                "--stand-model-profile must have measurement_status=measured"
            )


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.session_id = args.session_id or _default_session_id()
    session_root = args.output_root / args.session_id
    if session_root.exists():
        parser.error(f"refusing to reuse existing session: {session_root}")
    try:
        profile = load_real_robot_profile(args.robot_profile)
        calibration = load_camera_calibration(args.camera_calibration)
        _validate_inputs(parser, args, profile, calibration)
        runtime = profile.resolved_runtime()
        clearance = _physical_clearance(
            profile,
            approach_offset_m=args.candidate_approach_offset_m,
        )
        if (
            args.candidate_approach_offset_m + 1.0e-9
            < clearance["minimum_active_standoff_m"]
        ):
            raise ValueError("candidate pre-approach is below physical minimum")
        if (
            args.final_facing_offset_m + 1.0e-9
            < clearance["minimum_active_standoff_m"]
        ):
            raise ValueError("final facing pose is below physical minimum")
        inflation_radius_m = max(
            0.25,
            clearance["minimum_static_inflation_m"],
        )
        candidate_keepout_radius_m = max(
            0.31,
            clearance["minimum_candidate_transit_radius_m"],
        )
        session_root.mkdir(parents=True, exist_ok=False)
        survey_root = session_root / "coverage"
        start = read_current_amcl_pose(
            namespace=profile.namespace,
            amcl_topic=profile.amcl_topic,
            map_frame=profile.map_frame,
            timeout_sec=5.0,
            max_age_sec=2.0,
        )
        planning_status = plan_stand_coverage_survey(
            [
                "--map",
                str(args.map),
                "--semantic-map-id",
                args.semantic_map_id,
                "--planning-frame",
                profile.map_frame,
                "--start-x",
                str(start.x_m),
                "--start-y",
                str(start.y_m),
                "--start-yaw",
                str(start.yaw_rad),
                "--survey-id",
                args.session_id,
                "--output-dir",
                str(survey_root),
                "--lane-count",
                "1",
                "--stop-spacing-m",
                str(args.inspection_stop_spacing_m),
                "--inflation-radius-m",
                str(inflation_radius_m),
                "--candidate-keepout-radius-m",
                str(candidate_keepout_radius_m),
                "--expected-stand-count",
                str(args.expected_stand_count),
            ]
        )
        if planning_status != 0:
            return planning_status
        plan_path = survey_root / "coverage_plan.json"
        plan = load_coverage_survey_plan(plan_path)

        if not args.execute:
            first_sealed = seal_stand_discovery_route(
                source_route_csv=survey_root / "legs/leg_000_route.csv",
                source_diagnostics_json=(
                    survey_root / "legs/leg_000_diagnostics.json"
                ),
                coverage_plan_path=plan_path,
                output_dir=session_root / "execution/coverage_leg_000",
            )
            _run_motion_leg(
                profile=profile,
                sealed=first_sealed,
                run_id=f"{args.session_id}_coverage_000_dry",
                session_root=session_root,
                execute=False,
                coverage_plan=plan_path,
            )
            _write_json(
                session_root / "mission_summary.json",
                {
                    "schema_version": 1,
                    "status": "first_leg_dry_run_ok",
                    "execute": False,
                    "motion_published": False,
                    "survey_root": str(survey_root),
                },
            )
            print(
                "First center-corridor leg passed the runner dry-run. "
                "Re-run with --execute for the physical mission."
            )
            return 0

        if args.coverage_leg_limit > 0:
            authorization_scope = (
                f"at most {args.coverage_leg_limit} center-corridor "
                "coverage leg(s)"
            )
        elif args.stop_after_coverage:
            authorization_scope = (
                "the complete center-corridor coverage pass, with no "
                "candidate-approach legs"
            )
        else:
            authorization_scope = (
                "the complete multi-leg stand exploration mission"
            )
        print(
            "Physical safety requirements: clear arena; unloaded robot; operator "
            "beside the robot; Ctrl+C and physical stop ready; separate exact-topic "
            f"zero Twist terminal ready. This RUN authorizes {authorization_scope}."
        )
        if input("Type RUN to authorize the autonomous exploration mission: ").strip() != "RUN":
            raise RuntimeError("operator did not authorize the mission")

        leg_index = 0
        while True:
            summary = json.loads((survey_root / "survey_summary.json").read_text())
            viewpoint_id = summary.get("next_viewpoint_id")
            if viewpoint_id is None:
                break
            source_route = survey_root / "legs" / f"leg_{leg_index:03d}_route.csv"
            source_diagnostics = (
                survey_root / "legs" / f"leg_{leg_index:03d}_diagnostics.json"
            )
            sealed = seal_stand_discovery_route(
                source_route_csv=source_route,
                source_diagnostics_json=source_diagnostics,
                coverage_plan_path=plan_path,
                output_dir=(
                    session_root / "execution" / f"coverage_leg_{leg_index:03d}"
                ),
            )
            _run_motion_leg(
                profile=profile,
                sealed=sealed,
                run_id=f"{args.session_id}_coverage_{leg_index:03d}",
                session_root=session_root,
                execute=True,
                coverage_plan=plan_path,
            )
            observer_summary = _capture_lidar_epoch(
                profile=profile,
                args=args,
                survey_root=survey_root,
                viewpoint_id=str(viewpoint_id),
            )
            status = record_stand_coverage_stop(
                [
                    "--survey-root",
                    str(survey_root),
                    "--map",
                    str(args.map),
                    "--semantic-map-id",
                    args.semantic_map_id,
                    "--viewpoint-id",
                    str(viewpoint_id),
                    "--observer-summary-json",
                    str(observer_summary),
                    "--scan-to-base-position-offset-m",
                    str(profile.scan_origin_to_base_offset_m),
                ]
            )
            if status != 0:
                raise RuntimeError(f"failed to fuse coverage stop {viewpoint_id}")
            leg_index += 1
            if (
                args.coverage_leg_limit > 0
                and leg_index >= args.coverage_leg_limit
            ):
                checkpoint_summary = json.loads(
                    (survey_root / "survey_summary.json").read_text()
                )
                result = {
                    "schema_version": 1,
                    "status": "coverage_leg_checkpoint_complete",
                    "motion_published": True,
                    "completed_coverage_legs": leg_index,
                    "next_viewpoint_id": checkpoint_summary.get(
                        "next_viewpoint_id"
                    ),
                    "survey_root": str(survey_root),
                }
                _write_json(session_root / "mission_summary.json", result)
                print(json.dumps(result, indent=2, sort_keys=True))
                return 0

        registry_path = survey_root / "stand_registry.json"
        registry = load_stand_survey_registry(registry_path, plan)
        pending = tuple(
            candidate
            for candidate in registry.candidates
            if candidate.status == STATUS_PENDING_CAMERA
        )
        if len(pending) != args.expected_stand_count:
            raise RuntimeError(
                "center-corridor survey did not resolve the expected stand count: "
                f"pending_camera={len(pending)} "
                f"expected={args.expected_stand_count}"
            )
        snapshot = candidate_snapshot_from_registry(
            registry,
            plan,
            registry_path=registry_path,
            snapshot_id=f"{args.session_id}_candidates",
        )
        snapshot_path = session_root / "candidate_snapshot.json"
        write_candidate_snapshot(snapshot_path, snapshot)

        if args.stop_after_coverage:
            result = {
                "schema_version": 1,
                "status": "coverage_complete",
                "motion_published": True,
                "stand_count": len(snapshot.candidates),
                "candidate_snapshot": str(snapshot_path),
                "candidate_snapshot_sha256": candidate_snapshot_sha256(
                    snapshot
                ),
                "survey_root": str(survey_root),
            }
            _write_json(session_root / "mission_summary.json", result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        unresolved = set(snapshot.candidate_uids)
        facing_records = []
        identities = []
        candidate_index = 0
        while unresolved:
            current = read_current_amcl_pose(
                namespace=profile.namespace,
                amcl_topic=profile.amcl_topic,
                map_frame=profile.map_frame,
                timeout_sec=5.0,
                max_age_sec=2.0,
            )
            candidate = _nearest_candidate(snapshot, current, unresolved)
            assert candidate is not None
            candidate_root = (
                session_root
                / "candidates"
                / f"{candidate_index:03d}_{candidate.candidate_uid}"
            )
            source_root = candidate_root / "preapproach_source"
            sealed = plan_candidate_preapproach(
                map_yaml=args.map,
                semantic_map_id=args.semantic_map_id,
                plan=plan,
                snapshot=snapshot,
                snapshot_path=snapshot_path,
                candidate_uid=candidate.candidate_uid,
                start=Pose2D(current.x_m, current.y_m, current.yaw_rad),
                output_dir=source_root,
                approach_offset_m=args.candidate_approach_offset_m,
                inflation_radius_m=inflation_radius_m,
                candidate_transit_radius_m=candidate_keepout_radius_m,
                physical_clearance=clearance,
            )
            _run_motion_leg(
                profile=profile,
                sealed=sealed,
                run_id=(
                    f"{args.session_id}_candidate_{candidate_index:03d}"
                ),
                session_root=session_root,
                execute=True,
                candidate_snapshot=source_root / "candidate_snapshot.json",
            )
            recommendation_path, qr_id, axis_observation_path = (
                _capture_camera_recommendation(
                    profile=profile,
                    args=args,
                    candidate=candidate,
                    output_dir=candidate_root / "camera_lidar_attempt_00",
                )
            )
            if recommendation_path is None:
                if axis_observation_path is None:
                    raise RuntimeError(
                        "observer returned neither QR recommendation nor axis"
                    )
                opposite_normal = _opposite_face_normal(
                    axis_observation_path
                )
                opposite_start = read_current_amcl_pose(
                    namespace=profile.namespace,
                    amcl_topic=profile.amcl_topic,
                    map_frame=profile.map_frame,
                    timeout_sec=5.0,
                    max_age_sec=2.0,
                )
                opposite_source = candidate_root / "opposite_face_source"
                opposite_sealed = None
                feasibility_failures = []
                for inspection_offset_m in _bounded_approach_offsets(
                    args.candidate_approach_offset_m,
                    clearance["minimum_active_standoff_m"],
                ):
                    try:
                        opposite_sealed = plan_candidate_preapproach(
                            map_yaml=args.map,
                            semantic_map_id=args.semantic_map_id,
                            plan=plan,
                            snapshot=snapshot,
                            snapshot_path=snapshot_path,
                            candidate_uid=candidate.candidate_uid,
                            start=Pose2D(
                                opposite_start.x_m,
                                opposite_start.y_m,
                                opposite_start.yaw_rad,
                            ),
                            output_dir=opposite_source,
                            approach_offset_m=inspection_offset_m,
                            inflation_radius_m=inflation_radius_m,
                            candidate_transit_radius_m=(
                                candidate_keepout_radius_m
                            ),
                            physical_clearance=clearance,
                            approach_normal_rad=opposite_normal,
                            axis_observation_path=axis_observation_path,
                        )
                        break
                    except ValueError as exc:
                        if not _is_approach_feasibility_failure(exc):
                            raise
                        feasibility_failures.append(
                            f"{inspection_offset_m:.3f} m: {exc}"
                        )
                if opposite_sealed is None:
                    raise RuntimeError(
                        "no physically allowed opposite-face approach was "
                        "A*-reachable: " + "; ".join(feasibility_failures)
                    )
                _run_motion_leg(
                    profile=profile,
                    sealed=opposite_sealed,
                    run_id=(
                        f"{args.session_id}_candidate_"
                        f"{candidate_index:03d}_opposite"
                    ),
                    session_root=session_root,
                    execute=True,
                    candidate_snapshot=(
                        opposite_source / "candidate_snapshot.json"
                    ),
                )
                recommendation_path, qr_id, _ = (
                    _capture_camera_recommendation(
                        profile=profile,
                        args=args,
                        candidate=candidate,
                        output_dir=(
                            candidate_root / "camera_lidar_attempt_01"
                        ),
                    )
                )
                if recommendation_path is None:
                    raise RuntimeError(
                        f"QR side remained unresolved after opposite-face "
                        f"inspection for {candidate.candidate_uid}"
                    )
            if qr_id is None:
                raise RuntimeError("camera recommendation has no QR identity")
            stopped_pose = read_current_amcl_pose(
                namespace=profile.namespace,
                amcl_topic=profile.amcl_topic,
                map_frame=profile.map_frame,
                timeout_sec=5.0,
                max_age_sec=2.0,
            )
            facing = _validate_facing_pose(
                args=args,
                profile=profile,
                plan=plan,
                snapshot=snapshot,
                candidate=candidate,
                recommendation_path=recommendation_path,
                current_pose=Pose2D(
                    stopped_pose.x_m,
                    stopped_pose.y_m,
                    stopped_pose.yaw_rad,
                ),
                output_dir=candidate_root,
                inflation_radius_m=inflation_radius_m,
            )
            facing["qr_id"] = qr_id
            facing_records.append(facing)
            identities.append(
                StationIdentity(
                    candidate.candidate_uid,
                    qr_id,
                    f"station_{qr_id}",
                )
            )
            receipt = candidate_root / "candidate_decision.json"
            _write_json(
                receipt,
                {
                    "schema_version": 1,
                    "survey_id": plan.survey_id,
                    "candidate_uid": candidate.candidate_uid,
                    "decision": "confirmed",
                    "decision_source": "camera_evidence",
                    "camera_evidence_path": str(recommendation_path),
                },
            )
            if record_stand_candidate_decision(
                [
                    "--survey-root",
                    str(survey_root),
                    "--decision-receipt-json",
                    str(receipt),
                ]
            ) != 0:
                raise RuntimeError("failed to commit camera candidate decision")
            unresolved.remove(candidate.candidate_uid)
            candidate_index += 1

        identity_registry, _source_sha = create_registry(
            candidate_snapshot=snapshot,
            mappings=identities,
            registry_id=f"{args.session_id}_identities",
            created_unix_sec=time.time(),
        )
        identity_path = session_root / "station_identity_registry.json"
        write_station_identity_registry(identity_path, identity_registry)
        catalog = {
            "schema_version": 1,
            "catalog_kind": "real_autonomous_stand_facing_poses",
            "session_id": args.session_id,
            "planning_frame": profile.map_frame,
            "map_bundle_sha256": plan.map_bundle_sha256,
            "coverage_plan_sha256": coverage_survey_plan_sha256(plan),
            "candidate_snapshot_sha256": candidate_snapshot_sha256(snapshot),
            "station_identity_registry_sha256": (
                station_identity_registry_sha256(identity_registry)
            ),
            "stand_count": len(facing_records),
            "records": sorted(
                facing_records,
                key=lambda item: str(item["candidate_uid"]),
            ),
        }
        catalog_path = session_root / "stand_facing_catalog.json"
        catalog_sha256 = write_content_hashed_json(
            catalog_path,
            catalog,
            hash_field="stand_facing_catalog_sha256",
        )
        final_summary = {
            "schema_version": 1,
            "status": "complete",
            "motion_published": True,
            "session_id": args.session_id,
            "stand_count": len(facing_records),
            "stand_facing_catalog": str(catalog_path),
            "stand_facing_catalog_sha256": catalog_sha256,
            "candidate_snapshot": str(snapshot_path),
            "station_identity_registry": str(identity_path),
            "survey_root": str(survey_root),
        }
        _write_json(session_root / "mission_summary.json", final_summary)
        print(json.dumps(final_summary, indent=2, sort_keys=True))
        return 0
    except (
        AssertionError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        if session_root.exists():
            _write_json(
                session_root / "mission_failure.json",
                {
                    "schema_version": 1,
                    "status": "failed_closed",
                    "reason": str(exc),
                    "motion_continues_authorized": False,
                },
            )
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
