#!/usr/bin/env python3
"""Run one fail-closed autonomous real-robot stand exploration mission.

The mission plans a single center rail, drives certified A* legs to stopped
inspection poses, fuses LiDAR candidates across those poses, visits every
stable candidate at a robot-facing pre-approach, and commits calibrated
camera/LiDAR QR-face poses.  Physical execution requires ``--execute`` and a
mission-level typed ``RUN``.  A route rebuilt after a pre-motion localization
mismatch requires another typed ``RUN``.  The mission authorization may cover
routine coverage and inspection children through exact one-use leg permits,
and a bounded same-leg, same-target post-motion global-localization reseal
through its narrower recovery permit.  Both paths require every fresh gate to
pass.  Every motion leg still passes the existing route, ROS, obstacle,
localization, and exclusive-velocity-owner gates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
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
from scripts.aufgabe04.navigation.coverage_candidate_admission import (
    coverage_candidate_admission_evidence,
    evaluate_coverage_candidate_admission,
)
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
from scripts.aufgabe04.navigation.mission_leg_motion_permit import (
    MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
    MISSION_LEG_RUN_CONFIRMATION,
    ROUTINE_MISSION_LEG_KINDS,
    MissionLegKind,
    MissionLegMotionAuthorization,
    MissionLegMotionPermit,
    load_mission_leg_motion_authorization,
    mission_leg_motion_authorization_sha256,
    write_mission_leg_motion_authorization,
    write_mission_leg_motion_permit,
)
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.plan_stand_coverage_survey import (
    main as plan_stand_coverage_survey,
)
from scripts.aufgabe04.navigation.read_current_amcl_pose import (
    read_current_amcl_pose,
)
from scripts.aufgabe04.navigation.ros_preflight import run_ros_preflight
from scripts.aufgabe04.navigation.ros_runtime_config import resolve_topic
from scripts.aufgabe04.navigation.runtime_localization_reseal import (
    evaluate_runtime_localization_reseal,
    evaluate_runtime_localization_reseal_budget,
)
from scripts.aufgabe04.navigation.runtime_motion_authorization import (
    MISSION_MOTION_AUTHORIZATION_SCOPE,
    MISSION_RUN_CONFIRMATION,
    RUNTIME_LOCALIZATION_RESEAL_RECOVERY_KIND,
    MissionMotionAuthorization,
    RuntimeLocalizationMotionPermit,
    file_sha256 as authorization_file_sha256,
    load_mission_motion_authorization,
    mission_motion_authorization_sha256,
    write_mission_motion_authorization,
    write_runtime_localization_motion_permit,
)
from scripts.aufgabe04.navigation.record_stand_candidate_decision import (
    main as record_stand_candidate_decision,
)
from scripts.aufgabe04.navigation.record_stand_coverage_stop import (
    record_stand_coverage_stop,
)
from scripts.aufgabe04.navigation.route_context import build_station_route_dry_run
from scripts.aufgabe04.navigation.stand_coverage_survey import (
    STATUS_PENDING_CAMERA,
    CoverageSurveyPlan,
    StandSurveyRegistry,
    coverage_survey_plan_sha256,
    load_coverage_survey_plan,
    load_survey_progress,
    load_stand_survey_registry,
    plan_next_survey_leg,
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
STATIONARY_AMCL_TIMEOUT_SEC = 15.0
DEFAULT_TRACKING_TUBE_RADIUS_M = 0.03
DEFAULT_COLLISION_MARGIN_M = 0.02
DEFAULT_LIDAR_STOP_DISTANCE_M = 0.20
DEFAULT_LIDAR_CLEARANCE_MARGIN_M = 0.02
DEFAULT_MAX_BLOCKAGE_REPLANS_PER_LEG = 3
DEFAULT_MAX_STARTUP_RESEALS_PER_LEG = 3
DEFAULT_MAX_RUNTIME_LOCALIZATION_RESEALS_PER_LEG = 1


@dataclass(frozen=True)
class MotionLegOutcome:
    run_id: str
    status: str
    stop_reason: str
    stop_details: dict[str, object]
    motion_published: bool
    returncode: int
    semantic_log_path: Path
    odom_execution_certificate_path: Path | None = None
    motion_authorization_permit_path: Path | None = None
    motion_authorization_permit_sha256: str = ""
    mission_leg_motion_permit_path: Path | None = None
    mission_leg_motion_permit_sha256: str = ""


@dataclass(frozen=True)
class RuntimeLocalizationPermitContext:
    """Exact mission scope needed to authorize one recovery child run."""

    mission_authorization_json: Path
    session_id: str
    leg_index: int
    target_viewpoint_id: str
    reseal_index: int
    max_runtime_reseals_per_leg: int
    rejected_run_id: str
    runtime_reseal_decision_evidence: dict[str, object]
    fresh_localization_evidence_path: Path
    permit_json_path: Path


@dataclass(frozen=True)
class MissionLegPermitContext:
    """Exact routine-leg identity authorized by the mission-level RUN."""

    mission_authorization_json: Path
    session_id: str
    semantic_map_id: str
    mission_leg_kind: MissionLegKind
    mission_leg_index: int
    target_id: str
    permit_json_path: Path


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


def _admit_preplanning_localization(
    runtime,
    session_root: Path,
    *,
    evidence_path: Path | None = None,
) -> Pose2D:
    """Bind route planning to one strictly admitted stationary map pose."""

    preflight = run_ros_preflight(
        runtime,
        max_localization_tf_future_sec=1.1,
        request_nomotion_update=True,
        nomotion_update_service=resolve_topic(
            "request_nomotion_update",
            runtime.namespace,
        ),
        nomotion_update_timeout_sec=STATIONARY_AMCL_TIMEOUT_SEC,
        max_stationary_amcl_position_spread_m=(
            0.5 * DEFAULT_TRACKING_TUBE_RADIUS_M
        ),
        max_stationary_amcl_yaw_spread_rad=0.03,
        max_stationary_amcl_position_std_m=(
            0.30
        ),
        max_stationary_amcl_yaw_std_rad=0.35,
    )
    evidence_path = (
        session_root / "preflight/preplanning_localization.json"
        if evidence_path is None
        else Path(evidence_path)
    )
    _write_json(evidence_path, preflight.to_json_dict())
    if not preflight.ok:
        raise RuntimeError(
            "preplanning localization admission failed: "
            + "; ".join(preflight.failures)
        )
    route_pose = preflight.route_pose
    if route_pose is None:
        raise RuntimeError(
            "preplanning localization admission returned no route pose"
        )
    try:
        return Pose2D(
            float(route_pose["x_m"]),
            float(route_pose["y_m"]),
            float(route_pose["yaw_rad"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"preplanning localization route pose is invalid: {exc}"
        ) from exc


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
    coverage_transient_replan: dict[str, object] | None = None,
    dry_run: bool,
    uncertainty_map_yaml: Path | None = None,
    localization_branch_proof_id: str = "",
    odom_execution_certificate_json: Path | None = None,
    uncertainty_budget_json: Path | None = None,
    mission_motion_authorization_json: Path | None = None,
    runtime_localization_motion_permit_json: Path | None = None,
    mission_leg_motion_authorization_json: Path | None = None,
    mission_leg_motion_permit_json: Path | None = None,
    mission_leg_kind: MissionLegKind | str | None = None,
    mission_leg_index: int | None = None,
    mission_leg_target_id: str = "",
    mission_leg_semantic_map_id: str = "",
    mission_leg_dry_preflight_json: Path | None = None,
    mission_leg_dry_odom_certificate_json: Path | None = None,
    mission_leg_dry_uncertainty_budget_json: Path | None = None,
    mission_session_id: str = "",
) -> list[str]:
    run_phase = "dry" if dry_run else "execute"
    odom_fields = (
        uncertainty_map_yaml,
        str(localization_branch_proof_id).strip(),
        odom_execution_certificate_json,
        uncertainty_budget_json,
    )
    odom_execution_requested = any(
        value is not None and value != "" for value in odom_fields
    )
    preflight_name = (
        f"{run_id}_{run_phase}.json"
        if odom_execution_requested
        else f"{run_id}.json"
    )
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
        str(session_root / "preflight" / preflight_name),
        "--operator-note",
        "UNLOADED autonomous stand exploration",
    ]
    if odom_execution_requested:
        if any(value is None or value == "" for value in odom_fields):
            raise ValueError(
                "uncertainty-aware odom execution arguments must be complete"
            )
        command.extend(
            [
                "--execution-pose-frame",
                "odom",
                "--odom-execution-certificate-json",
                str(odom_execution_certificate_json),
                "--uncertainty-budget-json",
                str(uncertainty_budget_json),
                "--uncertainty-map-yaml",
                str(uncertainty_map_yaml),
                "--localization-branch-proof-id",
                str(localization_branch_proof_id).strip(),
                "--uncertainty-robot-radius-m",
                str(profile.robot_radius_m),
                # Mean stability remains strict. Reported covariance is no
                # longer forced inside half of the 30 mm tracking tube; it is
                # charged against the route-specific clearance budget.
                "--max-stationary-amcl-position-std-m",
                "0.30",
                "--max-stationary-amcl-yaw-std-rad",
                "0.35",
            ]
        )
    if coverage_plan is not None:
        command.extend(["--coverage-plan", str(coverage_plan)])
    if candidate_snapshot is not None:
        command.extend(["--candidate-snapshot", str(candidate_snapshot)])
    if coverage_transient_replan is not None:
        command.extend(
            [
                "--coverage-transient-replan-survey-root",
                str(coverage_transient_replan["survey_root"]),
                "--coverage-transient-replan-session-root",
                str(coverage_transient_replan["session_root"]),
                "--coverage-transient-replan-map",
                str(coverage_transient_replan["map_yaml"]),
                "--coverage-transient-replan-semantic-map-id",
                str(coverage_transient_replan["semantic_map_id"]),
                "--coverage-transient-replan-target-viewpoint-id",
                str(coverage_transient_replan["target_viewpoint_id"]),
                "--coverage-transient-replan-robot-radius-m",
                str(coverage_transient_replan["robot_radius_m"]),
                "--coverage-transient-replan-max-count",
                str(coverage_transient_replan["max_replans"]),
                "--coverage-transient-replan-leg-index",
                str(coverage_transient_replan["leg_index"]),
                "--omnidirectional-hard-stop-distance-m",
                str(
                    float(coverage_transient_replan["robot_radius_m"])
                    + DEFAULT_COLLISION_MARGIN_M
                ),
            ]
        )
    authorization_fields = (
        mission_motion_authorization_json,
        runtime_localization_motion_permit_json,
    )
    if any(value is not None for value in authorization_fields):
        if any(value is None for value in authorization_fields):
            raise ValueError(
                "mission motion authorization and runtime localization "
                "permit must be supplied together"
            )
        if dry_run:
            raise ValueError(
                "runtime localization motion permits are live-run only"
            )
        if not str(mission_session_id).strip():
            raise ValueError(
                "runtime localization motion permit requires mission_session_id"
            )
        if coverage_transient_replan is None:
            raise ValueError(
                "runtime localization motion permit requires a coverage leg"
            )
        command.extend(
            [
                "--mission-motion-authorization-json",
                str(mission_motion_authorization_json),
                "--runtime-localization-motion-permit-json",
                str(runtime_localization_motion_permit_json),
                "--mission-session-id",
                str(mission_session_id).strip(),
            ]
        )
    mission_leg_fields = (
        mission_leg_motion_authorization_json,
        mission_leg_motion_permit_json,
        mission_leg_kind,
        mission_leg_index,
        mission_leg_target_id or None,
        mission_leg_semantic_map_id or None,
        mission_leg_dry_preflight_json,
        mission_leg_dry_odom_certificate_json,
        mission_leg_dry_uncertainty_budget_json,
    )
    if any(value is not None for value in mission_leg_fields):
        if any(value is None for value in mission_leg_fields):
            raise ValueError(
                "mission-leg authorization arguments must be supplied together"
            )
        if any(value is not None for value in authorization_fields):
            raise ValueError(
                "routine mission-leg and runtime-localization permits are "
                "mutually exclusive"
            )
        if dry_run:
            raise ValueError("mission-leg motion permits are live-run only")
        if not str(mission_session_id).strip():
            raise ValueError(
                "mission-leg motion permit requires mission_session_id"
            )
        kind = MissionLegKind(mission_leg_kind)
        if kind not in ROUTINE_MISSION_LEG_KINDS:
            raise ValueError("mission-leg permit requires a routine leg kind")
        assert mission_leg_index is not None
        command.extend(
            [
                "--mission-leg-motion-authorization-json",
                str(mission_leg_motion_authorization_json),
                "--mission-leg-motion-permit-json",
                str(mission_leg_motion_permit_json),
                "--mission-leg-kind",
                kind.value,
                "--mission-leg-index",
                str(mission_leg_index),
                "--mission-leg-target-id",
                str(mission_leg_target_id).strip(),
                "--mission-leg-semantic-map-id",
                str(mission_leg_semantic_map_id).strip(),
                "--mission-leg-dry-preflight-json",
                str(mission_leg_dry_preflight_json),
                "--mission-leg-dry-odom-certificate-json",
                str(mission_leg_dry_odom_certificate_json),
                "--mission-leg-dry-uncertainty-budget-json",
                str(mission_leg_dry_uncertainty_budget_json),
                "--mission-session-id",
                str(mission_session_id).strip(),
            ]
        )
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


def _motion_outcome_from_log(
    semantic_log_path: Path,
    *,
    run_id: str,
    returncode: int,
) -> MotionLegOutcome:
    try:
        events = [
            json.loads(line)
            for line in Path(semantic_log_path).read_text().splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid motion semantic log for {run_id}: {exc}") from exc
    terminal_events = [
        event
        for event in events
        if event.get("run_id") == run_id
        and event.get("event")
        in {"motion_completed", "safety_stop", "preflight_failed"}
    ]
    if not terminal_events:
        raise RuntimeError(f"motion runner produced no terminal motion event for {run_id}")
    event = terminal_events[-1]
    if event.get("event") == "preflight_failed":
        failures = event.get("failures", [])
        if not isinstance(failures, list) or not all(
            isinstance(failure, str) for failure in failures
        ):
            raise RuntimeError(f"preflight failures are invalid for {run_id}")
        status = "preflight_failed"
        stop_reason = "; ".join(failures) or "ROS preflight failed"
        details = {
            "failures": list(failures),
            "observations": event.get("observations", []),
            "runtime_config": event.get("runtime_config", {}),
            "fail_closed": True,
        }
        motion_published = False
    else:
        status = str(event.get("status", ""))
        stop_reason = str(event.get("stop_reason", ""))
        details = event.get("stop_details", {})
        motion_published = event.get("motion_published")
        if not isinstance(motion_published, bool):
            raise RuntimeError(
                f"motion runner returned non-boolean motion_published "
                f"for {run_id}"
            )
    if status not in {"completed", "stopped", "preflight_failed"}:
        raise RuntimeError(f"motion runner returned invalid status {status!r} for {run_id}")
    if (status == "completed") != (returncode == 0):
        raise RuntimeError(
            f"motion runner exit/status mismatch for {run_id}: "
            f"returncode={returncode} status={status}"
        )
    if not isinstance(details, dict):
        raise RuntimeError(f"motion runner stop details are invalid for {run_id}")
    return MotionLegOutcome(
        run_id=run_id,
        status=status,
        stop_reason=stop_reason,
        stop_details=dict(details),
        motion_published=motion_published,
        returncode=returncode,
        semantic_log_path=Path(semantic_log_path),
    )


def _issue_runtime_localization_motion_permit(
    *,
    context: RuntimeLocalizationPermitContext,
    run_id: str,
    route_csv: Path,
    diagnostics_json: Path,
    map_route_certificate_json: Path,
    dry_preflight_json: Path,
    dry_odom_certificate_json: Path,
    dry_uncertainty_budget_json: Path,
) -> tuple[Path, str]:
    """Seal one exact child-run permit after the replacement dry-run passes."""

    if run_id == context.rejected_run_id:
        raise ValueError("runtime localization permit run_id must be new")
    if context.reseal_index <= 0:
        raise ValueError("runtime localization permit reseal_index must be positive")
    if context.reseal_index > context.max_runtime_reseals_per_leg:
        raise ValueError("runtime localization permit reseal budget exhausted")
    master_path = Path(context.mission_authorization_json).absolute()
    master = load_mission_motion_authorization(master_path)
    decision_evidence = dict(context.runtime_reseal_decision_evidence)

    def sealed(path: Path) -> tuple[str, str]:
        canonical = Path(path).absolute()
        return str(canonical), authorization_file_sha256(canonical)

    fresh_path, fresh_sha256 = sealed(
        context.fresh_localization_evidence_path
    )
    route_path, route_sha256 = sealed(route_csv)
    diagnostics_path, diagnostics_sha256 = sealed(diagnostics_json)
    map_certificate_path, map_certificate_sha256 = sealed(
        map_route_certificate_json
    )
    dry_preflight_path, dry_preflight_sha256 = sealed(dry_preflight_json)
    dry_certificate_path, dry_certificate_sha256 = sealed(
        dry_odom_certificate_json
    )
    dry_budget_path, dry_budget_sha256 = sealed(
        dry_uncertainty_budget_json
    )
    permit = RuntimeLocalizationMotionPermit(
        master_authorization_sha256=mission_motion_authorization_sha256(master),
        master_authorization_path=str(master_path),
        run_id=run_id,
        leg_index=context.leg_index,
        target_viewpoint_id=context.target_viewpoint_id,
        reseal_index=context.reseal_index,
        max_runtime_reseals_per_leg=(
            context.max_runtime_reseals_per_leg
        ),
        rejected_run_id=context.rejected_run_id,
        runtime_reseal_decision_evidence=decision_evidence,
        runtime_reseal_decision_sha256=payload_sha256(decision_evidence),
        fresh_localization_evidence_path=fresh_path,
        fresh_localization_evidence_sha256=fresh_sha256,
        route_csv_path=route_path,
        route_csv_sha256=route_sha256,
        diagnostics_path=diagnostics_path,
        diagnostics_sha256=diagnostics_sha256,
        map_route_certificate_path=map_certificate_path,
        map_route_certificate_sha256=map_certificate_sha256,
        dry_preflight_path=dry_preflight_path,
        dry_preflight_sha256=dry_preflight_sha256,
        dry_odom_certificate_path=dry_certificate_path,
        dry_odom_certificate_sha256=dry_certificate_sha256,
        dry_uncertainty_budget_path=dry_budget_path,
        dry_uncertainty_budget_sha256=dry_budget_sha256,
        same_target_verified=True,
        dry_run_passed=True,
        additional_typed_run_required=False,
    )
    permit_path = Path(context.permit_json_path).absolute()
    permit_sha256 = write_runtime_localization_motion_permit(
        permit_path,
        permit,
    )
    return permit_path, permit_sha256


def _issue_mission_leg_motion_permit(
    *,
    context: MissionLegPermitContext,
    run_id: str,
    route_csv: Path,
    diagnostics_json: Path,
    map_route_certificate_json: Path,
    dry_preflight_json: Path,
    dry_odom_certificate_json: Path,
    dry_uncertainty_budget_json: Path,
) -> tuple[Path, str]:
    """Seal one exact routine child after its no-motion dry-run passes."""

    master_path = Path(context.mission_authorization_json).absolute()
    master = load_mission_leg_motion_authorization(master_path)

    def sealed(path: Path) -> tuple[str, str]:
        canonical = Path(path).absolute()
        return str(canonical), authorization_file_sha256(canonical)

    route_path, route_sha256 = sealed(route_csv)
    diagnostics_path, diagnostics_sha256 = sealed(diagnostics_json)
    map_certificate_path, map_certificate_sha256 = sealed(
        map_route_certificate_json
    )
    dry_preflight_path, dry_preflight_sha256 = sealed(dry_preflight_json)
    dry_certificate_path, dry_certificate_sha256 = sealed(
        dry_odom_certificate_json
    )
    dry_budget_path, dry_budget_sha256 = sealed(
        dry_uncertainty_budget_json
    )
    permit = MissionLegMotionPermit(
        master_authorization_sha256=(
            mission_leg_motion_authorization_sha256(master)
        ),
        master_authorization_path=str(master_path),
        session_id=context.session_id,
        robot_id=master.robot_id,
        namespace=master.namespace,
        cmd_vel_topic=master.cmd_vel_topic,
        semantic_map_id=context.semantic_map_id,
        localization_branch_proof_id=(
            master.localization_branch_proof_id
        ),
        run_id=run_id,
        mission_leg_kind=context.mission_leg_kind,
        mission_leg_index=context.mission_leg_index,
        target_id=context.target_id,
        route_csv_path=route_path,
        route_csv_sha256=route_sha256,
        diagnostics_path=diagnostics_path,
        diagnostics_sha256=diagnostics_sha256,
        map_route_certificate_path=map_certificate_path,
        map_route_certificate_sha256=map_certificate_sha256,
        dry_preflight_path=dry_preflight_path,
        dry_preflight_sha256=dry_preflight_sha256,
        dry_odom_certificate_path=dry_certificate_path,
        dry_odom_certificate_sha256=dry_certificate_sha256,
        dry_uncertainty_budget_path=dry_budget_path,
        dry_uncertainty_budget_sha256=dry_budget_sha256,
        dry_run_passed=True,
        additional_typed_run_required=False,
    )
    permit_path = Path(context.permit_json_path).absolute()
    permit_sha256 = write_mission_leg_motion_permit(permit_path, permit)
    return permit_path, permit_sha256


def _run_motion_leg(
    *,
    profile,
    sealed: dict[str, str],
    run_id: str,
    session_root: Path,
    execute: bool,
    coverage_plan: Path | None = None,
    candidate_snapshot: Path | None = None,
    coverage_transient_replan: dict[str, object] | None = None,
    require_fresh_confirmation: bool = False,
    fresh_confirmation_reason: str = "startup",
    fresh_localization_evidence_path: Path | None = None,
    uncertainty_map_yaml: Path | None = None,
    localization_branch_proof_id: str = "",
    runtime_localization_permit_context: (
        RuntimeLocalizationPermitContext | None
    ) = None,
    mission_leg_permit_context: MissionLegPermitContext | None = None,
) -> MotionLegOutcome:
    if require_fresh_confirmation and fresh_confirmation_reason not in {
        "startup",
        "runtime_localization",
    }:
        raise ValueError(
            "fresh_confirmation_reason must be startup or "
            "runtime_localization"
        )
    if runtime_localization_permit_context is not None and (
        not require_fresh_confirmation
        or fresh_confirmation_reason != "runtime_localization"
    ):
        raise ValueError(
            "runtime localization permit requires runtime fresh-confirmation context"
        )
    if (
        mission_leg_permit_context is not None
        and runtime_localization_permit_context is not None
    ):
        raise ValueError(
            "routine mission-leg and runtime-localization permits are "
            "mutually exclusive"
        )
    if mission_leg_permit_context is not None and require_fresh_confirmation:
        raise ValueError(
            "routine mission-leg authorization cannot cover a resealed route"
        )
    common = {
        "profile": profile,
        "route_csv": Path(sealed["route_csv"]),
        "diagnostics_json": Path(sealed["diagnostics_json"]),
        "certificate_json": Path(sealed["route_certificate_json"]),
        "run_id": run_id,
        "session_root": session_root,
        "coverage_plan": coverage_plan,
        "candidate_snapshot": candidate_snapshot,
        "coverage_transient_replan": coverage_transient_replan,
        "uncertainty_map_yaml": uncertainty_map_yaml,
        "localization_branch_proof_id": localization_branch_proof_id,
    }
    odom_root = session_root / "odom_execution"
    dry_certificate = (
        None
        if uncertainty_map_yaml is None
        else odom_root / f"{run_id}_dry_certificate.json"
    )
    dry_budget = (
        None
        if uncertainty_map_yaml is None
        else odom_root / f"{run_id}_dry_uncertainty_budget.json"
    )
    dry = _runner_command(
        **common,
        dry_run=True,
        odom_execution_certificate_json=dry_certificate,
        uncertainty_budget_json=dry_budget,
    )
    dry_result = subprocess.run(dry, check=False)
    if dry_result.returncode != 0:
        semantic_log = session_root / "run_events" / f"{run_id}.jsonl"
        try:
            outcome = _motion_outcome_from_log(
                semantic_log,
                run_id=run_id,
                returncode=dry_result.returncode,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"dry-run failed for {run_id}: {exc}") from exc
        if _is_resealable_startup_mismatch(outcome):
            return outcome
        raise RuntimeError(
            f"dry-run failed for {run_id}: {outcome.stop_reason}"
        )
    if not execute:
        return MotionLegOutcome(
            run_id=run_id,
            status="dry_run_ok",
            stop_reason="",
            stop_details={},
            motion_published=False,
            returncode=0,
            semantic_log_path=(
                session_root / "run_events" / f"{run_id}.jsonl"
            ),
            odom_execution_certificate_path=dry_certificate,
        )
    motion_permit_path = None
    motion_permit_sha256 = ""
    mission_leg_permit_path = None
    mission_leg_permit_sha256 = ""
    if runtime_localization_permit_context is not None:
        if dry_certificate is None or dry_budget is None:
            raise RuntimeError(
                "runtime localization permit requires uncertainty-aware dry evidence"
            )
        motion_permit_path, motion_permit_sha256 = (
            _issue_runtime_localization_motion_permit(
                context=runtime_localization_permit_context,
                run_id=run_id,
                route_csv=common["route_csv"],
                diagnostics_json=common["diagnostics_json"],
                map_route_certificate_json=common["certificate_json"],
                dry_preflight_json=(
                    session_root / "preflight" / f"{run_id}_dry.json"
                ),
                dry_odom_certificate_json=dry_certificate,
                dry_uncertainty_budget_json=dry_budget,
            )
        )
    if mission_leg_permit_context is not None:
        if dry_certificate is None or dry_budget is None:
            raise RuntimeError(
                "mission-leg permit requires uncertainty-aware dry evidence"
            )
        mission_leg_permit_path, mission_leg_permit_sha256 = (
            _issue_mission_leg_motion_permit(
                context=mission_leg_permit_context,
                run_id=run_id,
                route_csv=common["route_csv"],
                diagnostics_json=common["diagnostics_json"],
                map_route_certificate_json=common["certificate_json"],
                dry_preflight_json=(
                    session_root / "preflight" / f"{run_id}_dry.json"
                ),
                dry_odom_certificate_json=dry_certificate,
                dry_uncertainty_budget_json=dry_budget,
            )
        )
        _append_jsonl(
            session_root / "adaptive_replans.jsonl",
            {
                "schema_version": 1,
                "event": "mission_leg_motion_permit_issued",
                "timestamp": time.time(),
                "run_id": run_id,
                "mission_leg_kind": (
                    mission_leg_permit_context.mission_leg_kind.value
                ),
                "mission_leg_index": (
                    mission_leg_permit_context.mission_leg_index
                ),
                "target_id": mission_leg_permit_context.target_id,
                "mission_leg_motion_permit_json": str(
                    mission_leg_permit_path
                ),
                "mission_leg_motion_permit_sha256": (
                    mission_leg_permit_sha256
                ),
                "covered_by_initial_mission_run": True,
                "additional_typed_run_required": False,
            },
        )
    if require_fresh_confirmation:
        if fresh_confirmation_reason == "runtime_localization":
            print(
                "The prior route stopped after motion because the global "
                "localization consistency monitor required zero and reseal. "
                "A fresh stationary AMCL/TF admission, A* route, exact-start "
                "connector, dry-run, uncertainty budget, and certificate now "
                "match the newly admitted map pose."
            )
        else:
            print(
                "The prior route was rejected before motion because AMCL moved "
                "outside its certified startup segment. A new A* route, exact-start "
                "connector, dry-run, and certificate now match the rejected live pose."
            )
        print(f"Resealed route: {common['route_csv']}")
        print(f"Resealed map-route certificate: {common['certificate_json']}")
        if fresh_localization_evidence_path is not None:
            print(
                "Fresh stationary localization evidence: "
                f"{fresh_localization_evidence_path}"
            )
        if dry_certificate is not None:
            print(f"Dry odom-execution certificate: {dry_certificate}")
        if dry_budget is not None:
            print(f"Dry route-uncertainty budget: {dry_budget}")
        if runtime_localization_permit_context is not None:
            print(
                "No additional RUN is required: this exact same-target "
                "runtime-localization recovery is covered by the initial "
                "mission authorization."
            )
            print(f"Runtime localization motion permit: {motion_permit_path}")
            print(f"Runtime localization motion permit SHA-256: {motion_permit_sha256}")
        elif input(
            f"Type RUN to authorize the resealed route {run_id}: "
        ).strip() != "RUN":
            raise RuntimeError(
                f"operator did not authorize resealed route {run_id}"
            )
    live_certificate = (
        None
        if uncertainty_map_yaml is None
        else odom_root / f"{run_id}_execute_certificate.json"
    )
    live_budget = (
        None
        if uncertainty_map_yaml is None
        else odom_root / f"{run_id}_execute_uncertainty_budget.json"
    )
    runner = _runner_command(
        **common,
        dry_run=False,
        odom_execution_certificate_json=live_certificate,
        uncertainty_budget_json=live_budget,
        mission_motion_authorization_json=(
            None
            if runtime_localization_permit_context is None
            else runtime_localization_permit_context.mission_authorization_json
        ),
        runtime_localization_motion_permit_json=motion_permit_path,
        mission_leg_motion_authorization_json=(
            None
            if mission_leg_permit_context is None
            else mission_leg_permit_context.mission_authorization_json
        ),
        mission_leg_motion_permit_json=mission_leg_permit_path,
        mission_leg_kind=(
            None
            if mission_leg_permit_context is None
            else mission_leg_permit_context.mission_leg_kind
        ),
        mission_leg_index=(
            None
            if mission_leg_permit_context is None
            else mission_leg_permit_context.mission_leg_index
        ),
        mission_leg_target_id=(
            ""
            if mission_leg_permit_context is None
            else mission_leg_permit_context.target_id
        ),
        mission_leg_semantic_map_id=(
            ""
            if mission_leg_permit_context is None
            else mission_leg_permit_context.semantic_map_id
        ),
        mission_leg_dry_preflight_json=(
            None
            if mission_leg_permit_context is None
            else session_root / "preflight" / f"{run_id}_dry.json"
        ),
        mission_leg_dry_odom_certificate_json=(
            None if mission_leg_permit_context is None else dry_certificate
        ),
        mission_leg_dry_uncertainty_budget_json=(
            None if mission_leg_permit_context is None else dry_budget
        ),
        mission_session_id=(
            runtime_localization_permit_context.session_id
            if runtime_localization_permit_context is not None
            else (
                ""
                if mission_leg_permit_context is None
                else mission_leg_permit_context.session_id
            )
        ),
    )
    wrapped = _bundle_command(profile, run_id, runner)
    result = subprocess.run(
        wrapped,
        check=False,
    )
    return replace(
        _motion_outcome_from_log(
            session_root / "run_events" / f"{run_id}.jsonl",
            run_id=run_id,
            returncode=result.returncode,
        ),
        odom_execution_certificate_path=live_certificate,
        motion_authorization_permit_path=motion_permit_path,
        motion_authorization_permit_sha256=motion_permit_sha256,
        mission_leg_motion_permit_path=mission_leg_permit_path,
        mission_leg_motion_permit_sha256=mission_leg_permit_sha256,
    )


def _require_completed_motion(outcome: MotionLegOutcome) -> None:
    if outcome.status != "completed":
        raise RuntimeError(
            f"physical route failed for {outcome.run_id}: {outcome.stop_reason}"
        )


def _capture_lidar_epoch(
    *,
    profile,
    args,
    survey_root: Path,
    viewpoint_id: str,
    odom_execution_certificate_path: Path | None = None,
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
        "--observation-id-scope",
        str(viewpoint_id),
    ]
    if odom_execution_certificate_path is not None:
        certificate_path = Path(odom_execution_certificate_path)
        if not certificate_path.is_file():
            raise RuntimeError(
                "completed odom execution leg has no readable certificate: "
                f"{certificate_path}"
            )
        command.extend(
            [
                "--odom-execution-certificate-json",
                str(certificate_path),
            ]
        )
    if subprocess.run(command, check=False).returncode != 0:
        raise RuntimeError(f"LiDAR epoch failed at {viewpoint_id}")
    payload = json.loads(summary.read_text())
    if int(payload.get("processed_scan_count", 0)) <= 0:
        raise RuntimeError(f"LiDAR epoch processed no scans at {viewpoint_id}")
    return summary


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _is_resealable_startup_mismatch(outcome: MotionLegOutcome) -> bool:
    details = outcome.stop_details
    return (
        outcome.status == "stopped"
        and not outcome.motion_published
        and outcome.stop_reason == "pose outside certified startup segment"
        and details.get("source") == "execution_route_certificate"
        and details.get("phase") == "before_motion_confirmation"
        and isinstance(details.get("route_pose"), dict)
    )


def _is_runtime_localization_reseal_required(outcome: MotionLegOutcome) -> bool:
    return evaluate_runtime_localization_reseal(
        status=outcome.status,
        motion_published=outcome.motion_published,
        stop_details=outcome.stop_details,
    ).eligible


def _adopted_blockage_replans_for_run(
    adaptive_log: Path,
    *,
    run_id: str,
) -> list[dict[str, object]]:
    """Load adopted in-process blockage replans for one child run.

    A localization reseal starts a new child process.  Until the last dynamic
    overlay and its cumulative budget can be restored into that child, a run
    that already adopted such an overlay is not safe to resume.  Malformed
    evidence therefore fails closed instead of being treated as no overlay.
    """

    path = Path(adaptive_log)
    if not path.exists():
        return []
    try:
        payloads = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot validate prior blockage-replan state in {path}: {exc}"
        ) from exc
    if not all(isinstance(payload, dict) for payload in payloads):
        raise RuntimeError(
            f"cannot validate prior blockage-replan state in {path}: "
            "JSONL entries must be objects"
        )
    return [
        dict(payload)
        for payload in payloads
        if payload.get("event")
        == "transient_navigation_blockage_replanned"
        and payload.get("run_id") == run_id
    ]


def _startup_reseal_pose(outcome: MotionLegOutcome) -> Pose2D:
    if not _is_resealable_startup_mismatch(outcome):
        raise ValueError("outcome is not a resealable startup mismatch")
    raw = outcome.stop_details["route_pose"]
    assert isinstance(raw, dict)
    pose = Pose2D(
        float(raw["x_m"]),
        float(raw["y_m"]),
        float(raw["yaw_rad"]),
    )
    if not all(
        math.isfinite(value) for value in (pose.x_m, pose.y_m, pose.yaw_rad)
    ):
        raise ValueError("startup mismatch pose must be finite")
    return pose


def _coverage_reseal_suffix(
    *,
    startup_reseal_index: int,
    runtime_localization_reseal_index: int,
) -> str:
    if startup_reseal_index < 0 or runtime_localization_reseal_index < 0:
        raise ValueError("reseal indices must be non-negative")
    parts = []
    if startup_reseal_index:
        parts.append(f"startup_reseal_{startup_reseal_index:03d}")
    if runtime_localization_reseal_index:
        parts.append(
            "runtime_localization_reseal_"
            f"{runtime_localization_reseal_index:03d}"
        )
    return "" if not parts else "_" + "_".join(parts)


def _replan_coverage_source_from_pose(
    *,
    map_yaml: Path,
    semantic_map_id: str,
    survey_root: Path,
    plan_path: Path,
    expected_target_viewpoint_id: str,
    current_pose: Pose2D,
    rejected_outcome: MotionLegOutcome,
    reseal_index: int,
    output_dir: Path,
    reseal_kind: str,
    status: str,
) -> dict[str, str]:
    """Rebuild a complete motion-free A* coverage leg from a fresh map pose."""

    if reseal_index <= 0:
        raise ValueError(f"{reseal_kind} reseal index must be positive")
    plan = load_coverage_survey_plan(plan_path)
    progress = load_survey_progress(
        survey_root / "coverage_progress.json",
        plan,
    )
    registry = load_stand_survey_registry(
        survey_root / "stand_registry.json",
        plan,
    )
    grid, map_bundle = load_occupancy_grid_with_bundle(
        map_yaml,
        semantic_map_id=semantic_map_id,
        planning_frame=plan.planning_frame,
    )
    if map_bundle.bundle_sha256 != plan.map_bundle_sha256:
        raise ValueError(f"{reseal_kind} reseal map differs from coverage plan")
    next_leg = plan_next_survey_leg(
        grid,
        plan=plan,
        progress=progress,
        registry=registry,
        current_pose=current_pose,
    )
    if next_leg is None:
        raise ValueError(f"{reseal_kind} reseal found no remaining coverage leg")
    if next_leg.viewpoint.viewpoint_id != expected_target_viewpoint_id:
        raise ValueError(
            f"{reseal_kind} reseal changed the committed coverage target: "
            f"expected={expected_target_viewpoint_id} "
            f"selected={next_leg.viewpoint.viewpoint_id}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    route_path = output_dir / "route.csv"
    diagnostics_path = output_dir / "route_diagnostics.json"
    summary_path = output_dir / f"{reseal_kind}_reseal_summary.json"
    write_route_csv(
        route_path,
        (next_leg.route_result,),
        final_yaw_by_leg={0: next_leg.viewpoint.pose.yaw_rad},
    )
    write_diagnostics_json(
        diagnostics_path,
        (next_leg.route_result,),
        metadata={
            "schema_version": 1,
            "route_kind": "stand_coverage_survey",
            "motion_authorized": False,
            "survey_id": plan.survey_id,
            "plan_sha256": coverage_survey_plan_sha256(plan),
            "map_bundle_sha256": plan.map_bundle_sha256,
            "target_viewpoint_id": next_leg.viewpoint.viewpoint_id,
            "target_pose": {
                "x_m": next_leg.viewpoint.pose.x_m,
                "y_m": next_leg.viewpoint.pose.y_m,
                "yaw_rad": next_leg.viewpoint.pose.yaw_rad,
            },
            "candidate_keepout_count": sum(
                1
                for candidate in registry.candidates
                if candidate.status != "rejected"
            ),
            "unreachable_viewpoint_ids_before_target": list(
                next_leg.unreachable_viewpoint_ids
            ),
            "inflation_radius_m": plan.config.inflation_radius_m,
            "exact_start_connector": (
                next_leg.exact_start_connector.to_metadata()
            ),
            "arena_boundary_overlay": True,
            "arena_bounds": plan.arena_bounds.to_metadata(),
            "reseal_kind": reseal_kind,
            f"{reseal_kind}_reseal": True,
            f"{reseal_kind}_reseal_index": reseal_index,
            "rejected_run_id": rejected_outcome.run_id,
            "rejected_stop_details": rejected_outcome.stop_details,
        },
    )
    summary = {
        "schema_version": 1,
        "status": status,
        "motion_published": False,
        "reseal_kind": reseal_kind,
        f"{reseal_kind}_reseal_index": reseal_index,
        "rejected_run_id": rejected_outcome.run_id,
        "target_viewpoint_id": next_leg.viewpoint.viewpoint_id,
        "fresh_start_pose": {
            "x_m": current_pose.x_m,
            "y_m": current_pose.y_m,
            "yaw_rad": current_pose.yaw_rad,
        },
        "route_csv": str(route_path),
        "diagnostics_json": str(diagnostics_path),
    }
    _write_json(summary_path, summary)
    return {
        "route_csv": str(route_path),
        "diagnostics_json": str(diagnostics_path),
        "summary_json": str(summary_path),
    }


def _replan_startup_source(
    *,
    map_yaml: Path,
    semantic_map_id: str,
    survey_root: Path,
    plan_path: Path,
    expected_target_viewpoint_id: str,
    current_pose: Pose2D,
    rejected_outcome: MotionLegOutcome,
    reseal_index: int,
    output_dir: Path,
) -> dict[str, str]:
    """Rebuild a complete motion-free A* leg from a rejected live pose."""

    return _replan_coverage_source_from_pose(
        map_yaml=map_yaml,
        semantic_map_id=semantic_map_id,
        survey_root=survey_root,
        plan_path=plan_path,
        expected_target_viewpoint_id=expected_target_viewpoint_id,
        current_pose=current_pose,
        rejected_outcome=rejected_outcome,
        reseal_index=reseal_index,
        output_dir=output_dir,
        reseal_kind="startup",
        status="startup_route_replanned",
    )


def _replan_runtime_localization_source(
    *,
    map_yaml: Path,
    semantic_map_id: str,
    survey_root: Path,
    plan_path: Path,
    expected_target_viewpoint_id: str,
    current_pose: Pose2D,
    rejected_outcome: MotionLegOutcome,
    reseal_index: int,
    output_dir: Path,
) -> dict[str, str]:
    """Rebuild a complete A* leg after post-motion localization reseal."""

    return _replan_coverage_source_from_pose(
        map_yaml=map_yaml,
        semantic_map_id=semantic_map_id,
        survey_root=survey_root,
        plan_path=plan_path,
        expected_target_viewpoint_id=expected_target_viewpoint_id,
        current_pose=current_pose,
        rejected_outcome=rejected_outcome,
        reseal_index=reseal_index,
        output_dir=output_dir,
        reseal_kind="runtime_localization",
        status="runtime_localization_route_replanned",
    )


def _execute_coverage_leg_with_replans(
    *,
    profile,
    args,
    session_root: Path,
    survey_root: Path,
    plan_path: Path,
    leg_index: int,
    target_viewpoint_id: str,
    source_route: Path,
    source_diagnostics: Path,
    mission_motion_authorization_json: Path | None = None,
    mission_leg_motion_authorization_json: Path | None = None,
) -> MotionLegOutcome:
    """Run one coverage leg with in-process transient-overlay A* recovery."""

    localization_branch_proof_id = str(
        getattr(args, "localization_branch_proof_id", "")
    ).strip()
    startup_reseal_index = 0
    runtime_localization_reseal_index = 0
    fresh_confirmation_reason: str | None = None
    fresh_localization_evidence_path: Path | None = None
    pending_runtime_route_seal: dict[str, object] | None = None
    pending_runtime_permit_context: (
        RuntimeLocalizationPermitContext | None
    ) = None
    adaptive_log = session_root / "adaptive_replans.jsonl"
    while True:
        suffix = _coverage_reseal_suffix(
            startup_reseal_index=startup_reseal_index,
            runtime_localization_reseal_index=(
                runtime_localization_reseal_index
            ),
        )
        run_id = f"{args.session_id}_coverage_{leg_index:03d}{suffix}"
        execution_root = (
            session_root
            / "execution"
            / f"coverage_leg_{leg_index:03d}{suffix}"
        )
        try:
            sealed = seal_stand_discovery_route(
                source_route_csv=source_route,
                source_diagnostics_json=source_diagnostics,
                coverage_plan_path=plan_path,
                output_dir=execution_root,
            )
        except Exception as exc:
            if pending_runtime_route_seal is not None:
                _append_jsonl(
                    adaptive_log,
                    {
                        **pending_runtime_route_seal,
                        "schema_version": 1,
                        "event": "runtime_localization_reseal_failed",
                        "timestamp": time.time(),
                        "phase": "route_seal",
                        "failure": str(exc),
                        "motion_continues_authorized": False,
                    },
                )
            raise
        if pending_runtime_route_seal is not None:
            covered_by_initial_run = mission_motion_authorization_json is not None
            _append_jsonl(
                adaptive_log,
                {
                    **pending_runtime_route_seal,
                    "schema_version": 1,
                    "event": "runtime_localization_route_sealed",
                    "timestamp": time.time(),
                    "replacement_run_id": run_id,
                    "replacement_route_csv": sealed["route_csv"],
                    "replacement_diagnostics_json": sealed[
                        "diagnostics_json"
                    ],
                    "replacement_route_certificate_json": sealed[
                        "route_certificate_json"
                    ],
                    "expected_dry_odom_execution_certificate_json": str(
                        session_root
                        / "odom_execution"
                        / f"{run_id}_dry_certificate.json"
                    ),
                    "expected_dry_uncertainty_budget_json": str(
                        session_root
                        / "odom_execution"
                        / f"{run_id}_dry_uncertainty_budget.json"
                    ),
                    "fresh_typed_run_required": not covered_by_initial_run,
                    "covered_by_initial_mission_run": covered_by_initial_run,
                    "expected_runtime_localization_motion_permit_json": (
                        ""
                        if pending_runtime_permit_context is None
                        else str(pending_runtime_permit_context.permit_json_path)
                    ),
                    "motion_continues_authorized": False,
                },
            )
            pending_runtime_route_seal = None
        outcome = _run_motion_leg(
            profile=profile,
            sealed=sealed,
            run_id=run_id,
            session_root=session_root,
            execute=True,
            coverage_plan=plan_path,
            coverage_transient_replan={
                "survey_root": survey_root,
                "session_root": session_root,
                "map_yaml": args.map,
                "semantic_map_id": args.semantic_map_id,
                "target_viewpoint_id": target_viewpoint_id,
                "robot_radius_m": profile.robot_radius_m,
                "max_replans": args.max_blockage_replans_per_leg,
                "leg_index": leg_index,
            },
            require_fresh_confirmation=fresh_confirmation_reason is not None,
            fresh_confirmation_reason=(
                fresh_confirmation_reason or "startup"
            ),
            fresh_localization_evidence_path=fresh_localization_evidence_path,
            # Production execution always carries this proof (validated by
            # _validate_inputs). Keeping the empty-proof case map-native
            # preserves compatibility for no-motion/unit callers.
            uncertainty_map_yaml=(
                args.map if localization_branch_proof_id else None
            ),
            localization_branch_proof_id=localization_branch_proof_id,
            runtime_localization_permit_context=(
                pending_runtime_permit_context
            ),
            mission_leg_permit_context=(
                None
                if (
                    mission_leg_motion_authorization_json is None
                    or fresh_confirmation_reason is not None
                    or pending_runtime_permit_context is not None
                )
                else MissionLegPermitContext(
                    mission_authorization_json=Path(
                        mission_leg_motion_authorization_json
                    ).absolute(),
                    session_id=args.session_id,
                    semantic_map_id=args.semantic_map_id,
                    mission_leg_kind=MissionLegKind.COVERAGE,
                    mission_leg_index=leg_index,
                    target_id=target_viewpoint_id,
                    permit_json_path=(
                        session_root
                        / "motion_authorization"
                        / "mission_legs"
                        / f"{run_id}_permit.json"
                    ).absolute(),
                )
            ),
        )
        if outcome.motion_authorization_permit_path is not None:
            _append_jsonl(
                adaptive_log,
                {
                    "schema_version": 1,
                    "event": "runtime_localization_motion_permit_issued",
                    "timestamp": time.time(),
                    "leg_index": leg_index,
                    "run_id": outcome.run_id,
                    "runtime_localization_motion_permit_json": str(
                        outcome.motion_authorization_permit_path
                    ),
                    "runtime_localization_motion_permit_sha256": (
                        outcome.motion_authorization_permit_sha256
                    ),
                    "covered_by_initial_mission_run": True,
                    "additional_typed_run_required": False,
                },
            )
        pending_runtime_permit_context = None
        if outcome.status == "completed":
            return outcome
        if _is_resealable_startup_mismatch(outcome):
            if startup_reseal_index >= args.max_startup_reseals_per_leg:
                raise RuntimeError(
                    "startup reseal budget exhausted for coverage leg "
                    f"{leg_index}: {outcome.stop_reason}"
                )
            startup_reseal_index += 1
            rejected_pose = _startup_reseal_pose(outcome)
            reseal_root = (
                survey_root
                / "startup_reseals"
                / (
                    f"leg_{leg_index:03d}"
                    f"_startup_reseal_{startup_reseal_index:03d}"
                )
            )
            replanned = _replan_startup_source(
                map_yaml=args.map,
                semantic_map_id=args.semantic_map_id,
                survey_root=survey_root,
                plan_path=plan_path,
                expected_target_viewpoint_id=target_viewpoint_id,
                current_pose=rejected_pose,
                rejected_outcome=outcome,
                reseal_index=startup_reseal_index,
                output_dir=reseal_root,
            )
            _append_jsonl(
                adaptive_log,
                {
                    "schema_version": 1,
                    "event": "startup_pose_route_resealed",
                    "timestamp": time.time(),
                    "leg_index": leg_index,
                    "startup_reseal_index": startup_reseal_index,
                    "rejected_run_id": outcome.run_id,
                    "rejected_stop_details": outcome.stop_details,
                    "replacement_route_csv": replanned["route_csv"],
                    "replacement_diagnostics_json": replanned[
                        "diagnostics_json"
                    ],
                    "replacement_summary_json": replanned["summary_json"],
                    "dynamic_overlay_preserved": False,
                    "fresh_confirmation_required": True,
                },
            )
            source_route = Path(replanned["route_csv"])
            source_diagnostics = Path(replanned["diagnostics_json"])
            fresh_confirmation_reason = "startup"
            fresh_localization_evidence_path = None
            continue
        runtime_localization_decision = evaluate_runtime_localization_reseal(
            status=outcome.status,
            motion_published=outcome.motion_published,
            stop_details=outcome.stop_details,
        )
        if runtime_localization_decision.eligible:
            adopted_blockage_replans = _adopted_blockage_replans_for_run(
                adaptive_log,
                run_id=outcome.run_id,
            )
            if adopted_blockage_replans:
                _append_jsonl(
                    adaptive_log,
                    {
                        "schema_version": 1,
                        "event": "runtime_localization_reseal_rejected",
                        "timestamp": time.time(),
                        "leg_index": leg_index,
                        "rejected_run_id": outcome.run_id,
                        "reason": (
                            "adopted_transient_blockage_replan_not_replayable"
                        ),
                        "adopted_blockage_replan_count": len(
                            adopted_blockage_replans
                        ),
                        "runtime_localization_reseal_decision": (
                            runtime_localization_decision.to_evidence()
                        ),
                        "motion_continues_authorized": False,
                        "fail_closed": True,
                    },
                )
                raise RuntimeError(
                    "runtime localization reseal after an adopted transient "
                    "blockage replan is not supported yet; refusing to relaunch "
                    "a child runner with reset overlay state"
                )
            budget = evaluate_runtime_localization_reseal_budget(
                completed_reseal_count=runtime_localization_reseal_index,
                maximum_reseal_count=(
                    args.max_runtime_localization_reseals_per_leg
                ),
            )
            if not budget.allowed:
                _append_jsonl(
                    adaptive_log,
                    {
                        "schema_version": 1,
                        "event": "runtime_localization_reseal_rejected",
                        "timestamp": time.time(),
                        "leg_index": leg_index,
                        "rejected_run_id": outcome.run_id,
                        "reason": budget.reason,
                        "runtime_localization_reseal_decision": (
                            runtime_localization_decision.to_evidence()
                        ),
                        "runtime_localization_reseal_budget": (
                            budget.to_evidence()
                        ),
                        "motion_continues_authorized": False,
                        "fail_closed": True,
                    },
                )
                raise RuntimeError(
                    "runtime localization reseal budget exhausted for "
                    f"coverage leg {leg_index}: {outcome.stop_reason}"
                )
            assert budget.next_reseal_index is not None
            runtime_localization_reseal_index = budget.next_reseal_index
            runtime = (
                profile.resolved_runtime()
                if callable(getattr(profile, "resolved_runtime", None))
                else profile
            )
            fresh_localization_evidence_path = (
                session_root
                / "preflight"
                / "runtime_localization_reseals"
                / (
                    f"coverage_leg_{leg_index:03d}"
                    f"_runtime_localization_reseal_"
                    f"{runtime_localization_reseal_index:03d}.json"
                )
            )
            recovery_event_base = {
                "leg_index": leg_index,
                "runtime_localization_reseal_index": (
                    runtime_localization_reseal_index
                ),
                "rejected_run_id": outcome.run_id,
                "rejected_stop_details": outcome.stop_details,
                "fresh_localization_evidence_json": str(
                    fresh_localization_evidence_path
                ),
                "runtime_localization_reseal_decision": (
                    runtime_localization_decision.to_evidence()
                ),
                "runtime_localization_reseal_budget": budget.to_evidence(),
                "fresh_confirmation_required": (
                    mission_motion_authorization_json is None
                ),
                "covered_by_initial_mission_run": (
                    mission_motion_authorization_json is not None
                ),
                "additional_typed_run_required": (
                    mission_motion_authorization_json is None
                ),
                "motion_continues_authorized": False,
            }
            _append_jsonl(
                adaptive_log,
                {
                    **recovery_event_base,
                    "schema_version": 1,
                    "event": "runtime_localization_reseal_started",
                    "timestamp": time.time(),
                    "source_stop_requires_zero_cycle": True,
                },
            )
            try:
                admitted_pose = _admit_preplanning_localization(
                    runtime,
                    session_root,
                    evidence_path=fresh_localization_evidence_path,
                )
            except Exception as exc:
                _append_jsonl(
                    adaptive_log,
                    {
                        **recovery_event_base,
                        "schema_version": 1,
                        "event": "runtime_localization_reseal_failed",
                        "timestamp": time.time(),
                        "phase": "stationary_localization_admission",
                        "failure": str(exc),
                    },
                )
                raise
            fresh_start_pose = {
                "x_m": admitted_pose.x_m,
                "y_m": admitted_pose.y_m,
                "yaw_rad": admitted_pose.yaw_rad,
            }
            _append_jsonl(
                adaptive_log,
                {
                    **recovery_event_base,
                    "schema_version": 1,
                    "event": "runtime_localization_admitted",
                    "timestamp": time.time(),
                    "fresh_start_pose": fresh_start_pose,
                },
            )
            reseal_root = (
                survey_root
                / "runtime_localization_reseals"
                / (
                    f"leg_{leg_index:03d}"
                    f"_runtime_localization_reseal_"
                    f"{runtime_localization_reseal_index:03d}"
                )
            )
            try:
                replanned = _replan_runtime_localization_source(
                    map_yaml=args.map,
                    semantic_map_id=args.semantic_map_id,
                    survey_root=survey_root,
                    plan_path=plan_path,
                    expected_target_viewpoint_id=target_viewpoint_id,
                    current_pose=admitted_pose,
                    rejected_outcome=outcome,
                    reseal_index=runtime_localization_reseal_index,
                    output_dir=reseal_root,
                )
            except Exception as exc:
                _append_jsonl(
                    adaptive_log,
                    {
                        **recovery_event_base,
                        "schema_version": 1,
                        "event": "runtime_localization_reseal_failed",
                        "timestamp": time.time(),
                        "phase": "same_target_route_replan",
                        "failure": str(exc),
                    },
                )
                raise
            _append_jsonl(
                adaptive_log,
                {
                    **recovery_event_base,
                    "schema_version": 1,
                    "event": "runtime_localization_route_replanned",
                    "timestamp": time.time(),
                    "fresh_start_pose": fresh_start_pose,
                    "replacement_route_csv": replanned["route_csv"],
                    "replacement_diagnostics_json": replanned[
                        "diagnostics_json"
                    ],
                    "replacement_summary_json": replanned["summary_json"],
                    "dynamic_overlay_preserved": False,
                    "adopted_blockage_replan_count": 0,
                    "committed_target_viewpoint_id": target_viewpoint_id,
                },
            )
            pending_runtime_route_seal = {
                **recovery_event_base,
                "fresh_start_pose": fresh_start_pose,
                "committed_target_viewpoint_id": target_viewpoint_id,
                "replacement_source_route_csv": replanned["route_csv"],
                "replacement_source_diagnostics_json": replanned[
                    "diagnostics_json"
                ],
                "replacement_summary_json": replanned["summary_json"],
            }
            if mission_motion_authorization_json is not None:
                pending_runtime_permit_context = (
                    RuntimeLocalizationPermitContext(
                        mission_authorization_json=Path(
                            mission_motion_authorization_json
                        ).absolute(),
                        session_id=args.session_id,
                        leg_index=leg_index,
                        target_viewpoint_id=target_viewpoint_id,
                        reseal_index=runtime_localization_reseal_index,
                        max_runtime_reseals_per_leg=(
                            args.max_runtime_localization_reseals_per_leg
                        ),
                        rejected_run_id=outcome.run_id,
                        runtime_reseal_decision_evidence=(
                            runtime_localization_decision.to_evidence()
                        ),
                        fresh_localization_evidence_path=(
                            fresh_localization_evidence_path
                        ),
                        permit_json_path=(
                            session_root
                            / "motion_authorization"
                            / (
                                f"{args.session_id}_coverage_"
                                f"{leg_index:03d}_runtime_localization_"
                                f"reseal_{runtime_localization_reseal_index:03d}_"
                                "permit.json"
                            )
                        ).absolute(),
                    )
                )
            source_route = Path(replanned["route_csv"])
            source_diagnostics = Path(replanned["diagnostics_json"])
            fresh_confirmation_reason = "runtime_localization"
            continue
        _require_completed_motion(outcome)


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
    parser.add_argument(
        "--exact-inspection-point-count",
        type=int,
        choices=(2,),
        default=None,
        help=(
            "Select exactly two complementary centerline LiDAR inspection "
            "points while retaining the map-coverage and two-view candidate "
            "admission gates."
        ),
    )
    parser.add_argument("--lidar-epoch-sec", type=float, default=8.0)
    parser.add_argument("--candidate-approach-offset-m", type=float, default=0.70)
    parser.add_argument("--final-facing-offset-m", type=float, default=0.35)
    parser.add_argument("--axis-sample-count", type=int, default=7)
    parser.add_argument("--camera-timeout-sec", type=float, default=90.0)
    parser.add_argument(
        "--localization-branch-proof-id",
        default="",
        help=(
            "Operator evidence ID for a known physical start or an asymmetric "
            "landmark that resolves the saved map's symmetric pose branch. "
            "Required with --execute; covariance alone is insufficient."
        ),
    )
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
        "--max-blockage-replans-per-leg",
        type=int,
        default=DEFAULT_MAX_BLOCKAGE_REPLANS_PER_LEG,
        help=(
            "Maximum front-LiDAR transient-overlay A* recovery attempts for "
            "one coverage leg. Zero disables adaptive blockage recovery."
        ),
    )
    parser.add_argument(
        "--max-startup-reseals-per-leg",
        type=int,
        default=DEFAULT_MAX_STARTUP_RESEALS_PER_LEG,
        help=(
            "Maximum fresh-pose A* reseals after a route is rejected before "
            "motion because AMCL left its certified startup segment. Every "
            "coverage-leg reseal requires a new typed RUN."
        ),
    )
    parser.add_argument(
        "--max-runtime-localization-reseals-per-leg",
        type=int,
        default=DEFAULT_MAX_RUNTIME_LOCALIZATION_RESEALS_PER_LEG,
        help=(
            "Maximum fresh stationary AMCL/TF admissions and A* reseals after "
            "motion has stopped because the global localization consistency "
            "monitor invalidated the odom execution certificate. The initial "
            "mission RUN may cover these bounded same-leg, same-target retries "
            "after a fresh immutable motion permit is admitted."
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
    if args.max_blockage_replans_per_leg < 0:
        parser.error("--max-blockage-replans-per-leg must be non-negative")
    if args.max_startup_reseals_per_leg < 0:
        parser.error("--max-startup-reseals-per-leg must be non-negative")
    if args.max_runtime_localization_reseals_per_leg < 0:
        parser.error(
            "--max-runtime-localization-reseals-per-leg must be non-negative"
        )
    args.localization_branch_proof_id = str(
        args.localization_branch_proof_id
    ).strip()
    if args.execute and not args.localization_branch_proof_id:
        parser.error(
            "--execute requires --localization-branch-proof-id for a known "
            "physical start or asymmetric landmark"
        )
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
        start = _admit_preplanning_localization(
            runtime,
            session_root,
        )
        planning_command = [
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
        if args.exact_inspection_point_count is not None:
            planning_command.extend(
                [
                    "--exact-inspection-point-count",
                    str(args.exact_inspection_point_count),
                ]
            )
        planning_status = plan_stand_coverage_survey(planning_command)
        if planning_status != 0:
            return planning_status
        plan_path = survey_root / "coverage_plan.json"
        plan = load_coverage_survey_plan(plan_path)
        if (
            args.exact_inspection_point_count is not None
            and len(plan.viewpoints) != args.exact_inspection_point_count
        ):
            raise RuntimeError(
                "coverage planner did not preserve the exact inspection-point "
                "count before motion authorization"
            )

        if not args.execute:
            first_sealed = seal_stand_discovery_route(
                source_route_csv=survey_root / "legs/leg_000_route.csv",
                source_diagnostics_json=(
                    survey_root / "legs/leg_000_diagnostics.json"
                ),
                coverage_plan_path=plan_path,
                output_dir=session_root / "execution/coverage_leg_000",
            )
            first_dry_outcome = _run_motion_leg(
                profile=profile,
                sealed=first_sealed,
                run_id=f"{args.session_id}_coverage_000_dry",
                session_root=session_root,
                execute=False,
                coverage_plan=plan_path,
                uncertainty_map_yaml=args.map,
                localization_branch_proof_id=(
                    args.localization_branch_proof_id
                    or "dry_run_no_motion"
                ),
            )
            if first_dry_outcome.status != "dry_run_ok":
                raise RuntimeError(
                    "first center-corridor dry-run rejected the sealed route: "
                    f"{first_dry_outcome.stop_reason}"
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
            f"zero Twist terminal ready. This RUN authorizes {authorization_scope} "
            "and its separately sealed routine coverage, candidate, and "
            "opposite-face child legs, plus bounded scan-backed transient-"
            "obstacle A* replans "
            f"(maximum {args.max_blockage_replans_per_leg} per coverage leg). "
            "Each routine child must pass a fresh dry-run and all live gates, "
            "then atomically consume its exact one-leg permit; it will not ask "
            "for another RUN. "
            "A pre-motion AMCL/start mismatch never inherits this RUN: the "
            "route is rebuilt and a fresh typed RUN is required. On a coverage "
            "leg, an exact post-motion global-localization reseal also stops "
            "first, recollects stationary AMCL/TF evidence, rebuilds the route "
            "to the same coverage target, reruns every dry/live gate, and may "
            "reuse this RUN for at most "
            f"{args.max_runtime_localization_reseals_per_leg} reseal(s) per leg "
            "through an exact one-run permit. Route-tube, stale-TF, obstacle, "
            "ownership, target-change, malformed-evidence, and budget failures "
            "remain terminal."
        )
        if input("Type RUN to authorize the autonomous exploration mission: ").strip() != "RUN":
            raise RuntimeError("operator did not authorize the mission")

        mission_leg_motion_authorization_json = (
            session_root
            / "motion_authorization"
            / "mission_leg_motion_authorization.json"
        ).absolute()
        mission_leg_motion_authorization = MissionLegMotionAuthorization(
            session_id=args.session_id,
            robot_id=profile.robot_id,
            namespace=runtime.namespace,
            cmd_vel_topic=runtime.cmd_vel_topic,
            semantic_map_id=args.semantic_map_id,
            localization_branch_proof_id=(
                args.localization_branch_proof_id
            ),
            allowed_leg_kinds=ROUTINE_MISSION_LEG_KINDS,
            scope_text=MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
            operator_confirmation=MISSION_LEG_RUN_CONFIRMATION,
        )
        mission_leg_motion_authorization_hash = (
            write_mission_leg_motion_authorization(
                mission_leg_motion_authorization_json,
                mission_leg_motion_authorization,
            )
        )

        mission_motion_authorization_json = (
            session_root
            / "motion_authorization"
            / "mission_motion_authorization.json"
        ).absolute()
        mission_motion_authorization = MissionMotionAuthorization(
            session_id=args.session_id,
            robot_id=profile.robot_id,
            namespace=runtime.namespace,
            cmd_vel_topic=runtime.cmd_vel_topic,
            semantic_map_id=args.semantic_map_id,
            localization_branch_proof_id=(
                args.localization_branch_proof_id
            ),
            max_runtime_reseals_per_leg=(
                args.max_runtime_localization_reseals_per_leg
            ),
            scope_text=MISSION_MOTION_AUTHORIZATION_SCOPE,
            operator_confirmation=MISSION_RUN_CONFIRMATION,
            allowed_recovery_kind=(
                RUNTIME_LOCALIZATION_RESEAL_RECOVERY_KIND
            ),
        )
        mission_motion_authorization_hash = (
            write_mission_motion_authorization(
                mission_motion_authorization_json,
                mission_motion_authorization,
            )
        )
        _append_jsonl(
            session_root / "adaptive_replans.jsonl",
            {
                "schema_version": 1,
                "event": "mission_motion_authorization_issued",
                "timestamp": time.time(),
                "session_id": args.session_id,
                "authorization_scope": authorization_scope,
                "mission_motion_authorization_json": str(
                    mission_motion_authorization_json
                ),
                "mission_motion_authorization_sha256": (
                    mission_motion_authorization_hash
                ),
                "mission_leg_motion_authorization_json": str(
                    mission_leg_motion_authorization_json
                ),
                "mission_leg_motion_authorization_sha256": (
                    mission_leg_motion_authorization_hash
                ),
                "routine_leg_kinds": [
                    kind.value for kind in ROUTINE_MISSION_LEG_KINDS
                ],
                "routine_child_prompts_required": False,
                "startup_reseal_fresh_typed_run_required": True,
                "max_runtime_localization_reseals_per_leg": (
                    args.max_runtime_localization_reseals_per_leg
                ),
                "allowed_recovery_kind": (
                    RUNTIME_LOCALIZATION_RESEAL_RECOVERY_KIND
                ),
                "additional_typed_run_required_for_eligible_recovery": False,
            },
        )

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
            coverage_outcome = _execute_coverage_leg_with_replans(
                profile=profile,
                args=args,
                session_root=session_root,
                survey_root=survey_root,
                plan_path=plan_path,
                leg_index=leg_index,
                target_viewpoint_id=str(viewpoint_id),
                source_route=source_route,
                source_diagnostics=source_diagnostics,
                mission_motion_authorization_json=(
                    mission_motion_authorization_json
                ),
                mission_leg_motion_authorization_json=(
                    mission_leg_motion_authorization_json
                ),
            )
            observer_summary = _capture_lidar_epoch(
                profile=profile,
                args=args,
                survey_root=survey_root,
                viewpoint_id=str(viewpoint_id),
                odom_execution_certificate_path=(
                    coverage_outcome.odom_execution_certificate_path
                ),
            )
            record_stand_coverage_stop(
                survey_root=survey_root,
                map_yaml=args.map,
                semantic_map_id=args.semantic_map_id,
                viewpoint_id=str(viewpoint_id),
                observer_summary_json=observer_summary,
                scan_to_base_position_offset_m=(
                    profile.scan_origin_to_base_offset_m
                ),
            )
            leg_index += 1
            if (
                args.coverage_leg_limit > 0
                and leg_index >= args.coverage_leg_limit
            ):
                checkpoint_summary = json.loads(
                    (survey_root / "survey_summary.json").read_text()
                )
                # A checkpoint must never bypass the terminal coverage-to-
                # candidate admission gate.  When this was the final planned
                # viewpoint, leave the loop normally and validate the fused
                # registry before reporting success or starting approaches.
                if checkpoint_summary.get("next_viewpoint_id") is None:
                    continue
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
        coverage_candidate_admission = (
            evaluate_coverage_candidate_admission(
                plan,
                load_survey_progress(
                    survey_root / "coverage_progress.json",
                    plan,
                ),
                registry,
            )
        )
        coverage_candidate_admission_path = (
            session_root / "coverage_candidate_admission.json"
        )
        coverage_candidate_admission_sha256 = write_content_hashed_json(
            coverage_candidate_admission_path,
            coverage_candidate_admission_evidence(
                coverage_candidate_admission
            ),
            hash_field="coverage_candidate_admission_sha256",
        )
        if not coverage_candidate_admission.ready:
            raise RuntimeError(
                "coverage candidate admission rejected: "
                + ", ".join(coverage_candidate_admission.reasons)
            )
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
                "coverage_candidate_admission": str(
                    coverage_candidate_admission_path
                ),
                "coverage_candidate_admission_sha256": (
                    coverage_candidate_admission_sha256
                ),
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
                timeout_sec=STATIONARY_AMCL_TIMEOUT_SEC,
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
            candidate_run_id = (
                f"{args.session_id}_candidate_{candidate_index:03d}"
            )
            candidate_outcome = _run_motion_leg(
                profile=profile,
                sealed=sealed,
                run_id=candidate_run_id,
                session_root=session_root,
                execute=True,
                candidate_snapshot=source_root / "candidate_snapshot.json",
                uncertainty_map_yaml=args.map,
                localization_branch_proof_id=(
                    args.localization_branch_proof_id
                ),
                mission_leg_permit_context=MissionLegPermitContext(
                    mission_authorization_json=(
                        mission_leg_motion_authorization_json
                    ),
                    session_id=args.session_id,
                    semantic_map_id=args.semantic_map_id,
                    mission_leg_kind=(
                        MissionLegKind.CANDIDATE_PREAPPROACH
                    ),
                    mission_leg_index=candidate_index,
                    target_id=candidate.candidate_uid,
                    permit_json_path=(
                        session_root
                        / "motion_authorization"
                        / "mission_legs"
                        / f"{candidate_run_id}_permit.json"
                    ).absolute(),
                ),
            )
            _require_completed_motion(candidate_outcome)
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
                    timeout_sec=STATIONARY_AMCL_TIMEOUT_SEC,
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
                opposite_run_id = (
                    f"{args.session_id}_candidate_"
                    f"{candidate_index:03d}_opposite"
                )
                opposite_outcome = _run_motion_leg(
                    profile=profile,
                    sealed=opposite_sealed,
                    run_id=opposite_run_id,
                    session_root=session_root,
                    execute=True,
                    candidate_snapshot=(
                        opposite_source / "candidate_snapshot.json"
                    ),
                    uncertainty_map_yaml=args.map,
                    localization_branch_proof_id=(
                        args.localization_branch_proof_id
                    ),
                    mission_leg_permit_context=MissionLegPermitContext(
                        mission_authorization_json=(
                            mission_leg_motion_authorization_json
                        ),
                        session_id=args.session_id,
                        semantic_map_id=args.semantic_map_id,
                        mission_leg_kind=MissionLegKind.OPPOSITE_FACE,
                        mission_leg_index=candidate_index,
                        target_id=candidate.candidate_uid,
                        permit_json_path=(
                            session_root
                            / "motion_authorization"
                            / "mission_legs"
                            / f"{opposite_run_id}_permit.json"
                        ).absolute(),
                    ),
                )
                _require_completed_motion(opposite_outcome)
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
                timeout_sec=STATIONARY_AMCL_TIMEOUT_SEC,
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
            "coverage_candidate_admission": str(
                coverage_candidate_admission_path
            ),
            "coverage_candidate_admission_sha256": (
                coverage_candidate_admission_sha256
            ),
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
