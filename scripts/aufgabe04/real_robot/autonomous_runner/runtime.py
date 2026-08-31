#!/usr/bin/env python3
"""Run one fail-closed autonomous real-robot stand exploration mission.

The mission plans a single center rail, drives certified A* legs to stopped
inspection poses, fuses LiDAR candidates across those poses, visits every
stable candidate at a robot-facing pre-approach, and commits calibrated
camera/LiDAR QR-face poses.  Physical execution requires an explicit
``execute-*`` or ``resume-*`` run mode and a mission-level typed ``RUN``.  The
mission authorization may cover routine coverage and inspection children
through exact one-use leg permits. Bounded startup and post-motion
localization recovery use separate, evidence-bound recovery permits. Every
path requires all fresh gates to pass. Every motion leg still passes the
existing route, ROS, obstacle, localization, and exclusive-velocity-owner
gates.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.coverage.coverage_candidate_admission import (
    coverage_candidate_admission_evidence,
    evaluate_coverage_candidate_admission,
)
from scripts.aufgabe04.navigation.approach.dynamic_approach_planner import (
    DynamicApproachConfig,
    minimum_static_obstacle_inflation_m,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import (
    CandidatePlanningFrame,
)
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
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
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.missions.plan_stand_coverage_survey import (
    main as plan_stand_coverage_survey,
)
from scripts.aufgabe04.navigation.localization.read_current_amcl_pose import (
    read_current_pose2d_from_amcl,
)
from scripts.aufgabe04.navigation.localization.ros_preflight import (
    RosPreflightRequirements,
    run_ros_preflight,
)
from scripts.aufgabe04.navigation.foundation.ros_runtime_config import resolve_topic
from scripts.aufgabe04.navigation.execution.runtime_motion_authorization import (
    MISSION_MOTION_AUTHORIZATION_SCOPE,
    MISSION_RUN_CONFIRMATION,
    RUNTIME_LOCALIZATION_RESEAL_RECOVERY_KIND,
    MissionMotionAuthorization,
    file_sha256 as authorization_file_sha256,
    write_mission_motion_authorization,
)
from scripts.aufgabe04.navigation.coverage.record_stand_coverage_stop import (
    commit_stand_coverage_stop,
    plan_next_stand_coverage_leg,
)
from scripts.aufgabe04.navigation.coverage.exact_two_viewpoint_selection import (
    DEFAULT_EXACT_TWO_CANDIDATE_SPACING_M,
    DEFAULT_MINIMUM_EXACT_TWO_VIEWPOINT_BASELINE_M,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    STATUS_PENDING_CAMERA,
    CoverageSurveyConfig,
    CoverageSurveyPlan,
    StandSurveyRegistry,
    coverage_survey_plan_sha256,
    load_coverage_survey_plan,
    load_survey_progress,
    load_stand_survey_registry,
)
from scripts.aufgabe04.perception.lidar_stand_morphology import (
    stand_width_profile_from_radius,
)
from scripts.aufgabe04.navigation.coverage.stand_discovery_route import (
    seal_stand_discovery_route,
)
from scripts.aufgabe04.navigation.execution.startup_reseal_motion_authorization import (
    STARTUP_RESEAL_MOTION_AUTHORIZATION_SCOPE,
    STARTUP_RESEAL_RECOVERY_KIND,
    STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY,
    STARTUP_RESEAL_RECOVERY_SOURCE_KINDS,
    STARTUP_RESEAL_RUN_CONFIRMATION,
    StartupResealMotionAuthorization,
    write_startup_reseal_motion_authorization,
)
from scripts.aufgabe04.perception.stand_axis.model_profile import (
    load_measured_physical_stand_model,
)
from scripts.aufgabe04.real_robot.configuration.profile import (
    camera_calibration_sha256,
    load_camera_calibration,
    load_real_robot_profile,
)
from scripts.aufgabe04.real_robot.configuration.site_contract import (
    validate_physical_site_contract,
)
from scripts.aufgabe04.real_robot.readiness.candidate_planning_frame import (
    CANDIDATE_PLANNING_FRAME_PREFLIGHT_REQUIREMENTS,
    build_candidate_planning_frame,
)
from scripts.aufgabe04.real_robot.observer.diagnostics import (
    format_passive_observer_failure,
    is_candidate_local_observer_timeout,
    load_passive_observer_status,
)
from scripts.aufgabe04.real_robot.observer.process import (
    monitor_passive_observer_process,
)
from scripts.aufgabe04.real_robot.candidate.observation_deferral import (
    CandidateObservationUnavailableError,
)
from scripts.aufgabe04.real_robot.autonomous_runner.failure_reporting import (
    build_failed_closed_mission_summary,
)
from scripts.aufgabe04.real_robot.execution.artifact_paths import (
    resolve_child_artifact_paths,
    resolve_normal_artifact_path,
)
from scripts.aufgabe04.real_robot.execution.child_runner import (
    DEFAULT_COLLISION_MARGIN_M,
    DEFAULT_LIDAR_STOP_DISTANCE_M,
    DEFAULT_TRACKING_TUBE_RADIUS_M,
    DEFAULT_UNCERTAINTY_SIGMA_MULTIPLIER,
    MotionLegOutcome,
    build_bundle_command as _bundle_command,
    build_child_runner_command as _runner_command,
    parse_dry_run_outcome as _dry_motion_outcome_from_log,
    parse_motion_leg_outcome as _motion_outcome_from_log,
    semantic_log_size as _semantic_log_size,
)
from scripts.aufgabe04.real_robot.coverage_leg.execution import (
    CoverageLegConfig,
    CoverageLegEffects,
    MissionLegPermitContext,
    execute_coverage_leg_with_replans as execute_coverage_leg_state_machine,
)
from scripts.aufgabe04.real_robot.execution.localization_recovery import (
    RuntimeLocalizationPermitContext,
    issue_runtime_localization_motion_permit,
    resolved_runtime_localization_semantic_map_id,
)
from scripts.aufgabe04.real_robot.coverage_leg.replanning import (
    is_resealable_startup_mismatch,
)
from scripts.aufgabe04.real_robot.mission.coverage import (
    CompletedCoverageLeg,
    CoverageCheckpointComplete,
    CoverageCheckpointIdentity,
    CoverageCompletionPolicy,
    CoverageComplete,
    CoverageExactTwoCameraReady,
    CoverageLidarCheckpointComplete,
    CoverageMissionConfig,
    CoverageMissionEffects,
    PreparedCoverageLeg,
    PublishedCoverageCheckpoint,
    execute_coverage_mission,
)
from scripts.aufgabe04.real_robot.readiness.localization import (
    evaluate_localization_readiness_retry,
)
from scripts.aufgabe04.real_robot.readiness.initialpose_prompt import (
    InitialPosePromptConfig,
    prompt_for_initialpose_attempt,
)
from scripts.aufgabe04.real_robot.readiness.preauthorization import (
    PreauthorizationReadinessConfig,
    PreauthorizationReadinessEffects,
    admit_preauthorization_readiness,
)
from scripts.aufgabe04.real_robot.readiness.observation_tf_runtime import (
    ObservationTfReadinessConfig,
    ObservationTfReadinessError,
    observe_observation_tf_readiness,
)
from scripts.aufgabe04.real_robot.candidate.approach import (
    CandidateApproachConfig,
    CandidateApproachEffects,
    CandidateMotionLegRequest,
    CandidateObservation,
    CandidateObservationRequest,
    execute_candidate_approach_phase,
)
from scripts.aufgabe04.real_robot.candidate.startup_recovery import (
    CandidateStartupRecoveryAttempt,
)
from scripts.aufgabe04.real_robot.candidate.runtime_recovery import (
    CandidateRuntimeRecoveryAttempt,
)
from scripts.aufgabe04.real_robot.mission.checkpoint_resume import (
    admit_coverage_resume,
    restore_and_replan_coverage_resume,
)
from scripts.aufgabe04.real_robot.mission.modes import (
    AutonomousRunMode,
    resolve_autonomous_run_mode,
    validate_autonomous_viewpoint_scope,
    validate_session_id_mode_label,
)
from scripts.aufgabe04.real_robot.mission.reporting import (
    build_completed_camera_mission_summary as _completed_camera_mission_summary,
)
from scripts.aufgabe04.real_robot.readiness.post_observation import (
    PostObservationLocalizationConfig,
    PostObservationLocalizationEffects,
    admit_post_observation_localization,
)
from scripts.aufgabe04.real_robot.mission.session_manifest import (
    publish_coverage_checkpoint,
)
from scripts.aufgabe04.real_robot.readiness.startup_reseal import (
    StartupResealPermitContext,
    issue_startup_reseal_motion_permit,
    write_startup_reseal_permit_summary,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateGeometry,
    CandidateSource,
    FrozenCandidate,
    candidate_snapshot_sha256,
    new_candidate_snapshot,
    write_candidate_snapshot,
)


DEFAULT_MAP = Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml")
DEFAULT_OUTPUT_ROOT = Path("results/aufgabe04/real/autonomous_exploration")
STATIONARY_AMCL_TIMEOUT_SEC = 15.0
DEFAULT_LIDAR_CLEARANCE_MARGIN_M = 0.02
DEFAULT_MAX_BLOCKAGE_REPLANS_PER_LEG = 3
DEFAULT_MAX_STARTUP_RESEALS_PER_LEG = 3
DEFAULT_MAX_RUNTIME_LOCALIZATION_RESEALS_PER_LEG = 1
DEFAULT_MAX_LOCALIZATION_READINESS_RETRIES_PER_LEG = 2


from .mission_config import (
    _coverage_completion_policy,
    _plan_exact_inspection_point_count,
    _file_sha256,
    _checkpoint_config_sha256,
    _default_session_id,
    _write_json,
    _physical_clearance,
    candidate_snapshot_from_registry,
)












def _admit_observation_tf_readiness(
    profile,
    evidence_path: Path,
    *,
    phase: str,
    typed_run_already_issued: bool = False,
) -> tuple[Path, str]:
    """Persist one passive exact-scan TF gate or fail with typed evidence."""

    runtime = profile.resolved_runtime()
    result = observe_observation_tf_readiness(
        ObservationTfReadinessConfig(
            scan_topic=runtime.scan_topic,
            expected_scan_frame=profile.scan_frame,
            target_frame=runtime.odom_frame,
        )
    )
    path = Path(evidence_path)
    digest = write_content_hashed_json(
        path,
        {
            **result.to_dict(),
            "phase": phase,
            "typed_run_already_issued": typed_run_already_issued,
        },
        hash_field="observation_tf_readiness_sha256",
    )
    if not result.ready:
        raise ObservationTfReadinessError(
            result,
            evidence_path=str(path),
            evidence_sha256=digest,
            phase=phase,
            typed_run_already_issued=typed_run_already_issued,
        )
    return path, digest


def _admit_preplanning_localization(
    runtime,
    session_root: Path,
    *,
    evidence_path: Path | None = None,
) -> Pose2D:
    """Bind route planning to one strictly admitted stationary map pose."""

    preflight = _run_preplanning_localization_preflight(
        runtime,
        session_root,
        evidence_path=evidence_path,
    )
    return _pose_from_preplanning_preflight(preflight)


def _admit_candidate_planning_frame(
    runtime,
    session_root: Path,
    *,
    evidence_path: Path | None = None,
) -> CandidatePlanningFrame:
    """Admit one simultaneous stationary pose and ``map <- odom`` value."""

    candidate_evidence_path = (
        session_root / "preflight/candidate_planning_frame.json"
        if evidence_path is None
        else evidence_path
    )
    preflight = _run_preplanning_localization_preflight(
        runtime,
        session_root,
        evidence_path=candidate_evidence_path,
        preflight_requirements=(
            CANDIDATE_PLANNING_FRAME_PREFLIGHT_REQUIREMENTS
        ),
    )
    pose = _pose_from_preplanning_preflight(preflight)
    return build_candidate_planning_frame(
        preflight,
        current_pose=pose,
        map_frame=runtime.map_frame,
        odom_frame=runtime.odom_frame,
    )


def _run_preplanning_localization_preflight(
    runtime,
    session_root: Path,
    *,
    evidence_path: Path | None,
    preflight_requirements: RosPreflightRequirements | None = None,
):
    """Run and persist the shared stopped localization admission."""

    preflight_kwargs = {
        "max_localization_tf_future_sec": 1.1,
        "request_nomotion_update": True,
        "nomotion_update_service": resolve_topic(
            "request_nomotion_update",
            runtime.namespace,
        ),
        "nomotion_update_timeout_sec": STATIONARY_AMCL_TIMEOUT_SEC,
        "max_stationary_amcl_position_spread_m": (
            0.5 * DEFAULT_TRACKING_TUBE_RADIUS_M
        ),
        "max_stationary_amcl_yaw_spread_rad": 0.03,
        "max_stationary_amcl_position_std_m": (
            0.30
        ),
        "max_stationary_amcl_yaw_std_rad": 0.35,
    }
    if preflight_requirements is not None:
        preflight_kwargs["preflight_requirements"] = preflight_requirements
    preflight = run_ros_preflight(runtime, **preflight_kwargs)
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
    return preflight


def _pose_from_preplanning_preflight(preflight) -> Pose2D:
    """Extract the finite admitted route pose from a successful preflight."""

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

    master_path = resolve_normal_artifact_path(
        context.mission_authorization_json,
        label="mission leg authorization",
    )
    master = load_mission_leg_motion_authorization(master_path)

    def sealed(path: Path, label: str) -> tuple[str, str]:
        canonical = resolve_normal_artifact_path(path, label=label)
        return str(canonical), authorization_file_sha256(canonical)

    route_path, route_sha256 = sealed(route_csv, "mission leg route CSV")
    diagnostics_path, diagnostics_sha256 = sealed(
        diagnostics_json,
        "mission leg diagnostics JSON",
    )
    map_certificate_path, map_certificate_sha256 = sealed(
        map_route_certificate_json,
        "mission leg map-route certificate",
    )
    dry_preflight_path, dry_preflight_sha256 = sealed(
        dry_preflight_json,
        "mission leg dry preflight",
    )
    dry_certificate_path, dry_certificate_sha256 = sealed(
        dry_odom_certificate_json,
        "mission leg dry odom certificate",
    )
    dry_budget_path, dry_budget_sha256 = sealed(
        dry_uncertainty_budget_json,
        "mission leg dry uncertainty budget",
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
    permit_path = Path(context.permit_json_path).resolve(strict=False)
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
    uncertainty_sigma_multiplier: float = (
        DEFAULT_UNCERTAINTY_SIGMA_MULTIPLIER
    ),
    localization_branch_proof_id: str = "",
    runtime_localization_permit_context: (
        RuntimeLocalizationPermitContext | None
    ) = None,
    startup_reseal_permit_context: StartupResealPermitContext | None = None,
    mission_leg_permit_context: MissionLegPermitContext | None = None,
    observation_tf_evidence_path: Path | None = None,
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
    if startup_reseal_permit_context is not None and (
        not require_fresh_confirmation
        or fresh_confirmation_reason != "startup"
    ):
        raise ValueError(
            "startup reseal permit requires startup fresh-confirmation context"
        )
    permit_context_count = sum(
        context is not None
        for context in (
            runtime_localization_permit_context,
            startup_reseal_permit_context,
            mission_leg_permit_context,
        )
    )
    if permit_context_count > 1:
        raise ValueError(
            "routine-leg, startup-reseal, and runtime-localization permit "
            "contexts are mutually exclusive"
        )
    if mission_leg_permit_context is not None and require_fresh_confirmation:
        raise ValueError(
            "routine mission-leg authorization cannot cover a resealed route"
        )
    mission_leg_evidence_kind = None
    mission_leg_evidence_index = None
    mission_leg_evidence_target_id = ""
    if mission_leg_permit_context is not None:
        mission_leg_evidence_kind = mission_leg_permit_context.mission_leg_kind
        mission_leg_evidence_index = mission_leg_permit_context.mission_leg_index
        mission_leg_evidence_target_id = mission_leg_permit_context.target_id
    elif startup_reseal_permit_context is not None:
        mission_leg_evidence_kind = startup_reseal_permit_context.mission_leg_kind
        mission_leg_evidence_index = startup_reseal_permit_context.mission_leg_index
        mission_leg_evidence_target_id = startup_reseal_permit_context.target_id
    elif runtime_localization_permit_context is not None:
        mission_leg_evidence_kind = (
            runtime_localization_permit_context.mission_leg_kind
        )
        mission_leg_evidence_index = (
            runtime_localization_permit_context.mission_leg_index
        )
        mission_leg_evidence_target_id = (
            runtime_localization_permit_context.target_id
        )
    semantic_log = session_root / "run_events" / f"{run_id}.jsonl"
    if semantic_log.exists() or semantic_log.is_symlink():
        raise RuntimeError(
            "refusing to reuse an existing motion semantic log: "
            f"{semantic_log}"
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
        "uncertainty_sigma_multiplier": uncertainty_sigma_multiplier,
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
        mission_leg_evidence_kind=mission_leg_evidence_kind,
        mission_leg_evidence_index=mission_leg_evidence_index,
        mission_leg_evidence_target_id=mission_leg_evidence_target_id,
    )
    dry_result = subprocess.run(dry, check=False)
    if dry_result.returncode != 0:
        try:
            outcome = _motion_outcome_from_log(
                semantic_log,
                run_id=run_id,
                returncode=dry_result.returncode,
                start_offset=0,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"dry-run failed for {run_id}: {exc}") from exc
        if is_resealable_startup_mismatch(outcome):
            return outcome
        if evaluate_localization_readiness_retry(
            status=outcome.status,
            stop_reason=outcome.stop_reason,
            stop_details=outcome.stop_details,
            motion_published=outcome.motion_published,
        ).retryable:
            return outcome
        raise RuntimeError(
            f"dry-run failed for {run_id}: {outcome.stop_reason}"
        )
    try:
        dry_outcome = _dry_motion_outcome_from_log(
            semantic_log,
            run_id=run_id,
            returncode=dry_result.returncode,
            start_offset=0,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"dry-run success evidence is invalid for {run_id}: {exc}"
        ) from exc
    dry_preflight = (
        session_root
        / "preflight"
        / (
            f"{run_id}_dry.json"
            if uncertainty_map_yaml is not None
            else f"{run_id}.json"
        )
    )
    required_dry_artifacts = [
        ("dry semantic log", semantic_log),
        ("dry preflight", dry_preflight),
    ]
    if uncertainty_map_yaml is not None:
        if dry_certificate is None or dry_budget is None:
            raise RuntimeError(
                f"uncertainty-aware dry-run paths are missing for {run_id}"
            )
        required_dry_artifacts.extend(
            (
                ("dry odom execution certificate", dry_certificate),
                ("dry uncertainty budget", dry_budget),
            )
        )
    for label, path in required_dry_artifacts:
        try:
            resolve_normal_artifact_path(path, label=label)
        except ValueError as exc:
            raise RuntimeError(
                f"dry-run success artifact is invalid for {run_id}: {exc}"
            ) from exc
    dry_outcome = replace(
        dry_outcome,
        dry_preflight_path=dry_preflight,
        odom_execution_certificate_path=dry_certificate,
        dry_uncertainty_budget_path=dry_budget,
    )
    if permit_context_count:
        canonical_artifacts = resolve_child_artifact_paths(
            session_root=session_root,
            sealed=sealed,
        )
        session_root = canonical_artifacts.session_root
        semantic_log = session_root / "run_events" / f"{run_id}.jsonl"
        common.update(
            {
                "route_csv": canonical_artifacts.route_csv,
                "diagnostics_json": canonical_artifacts.diagnostics_json,
                "certificate_json": (
                    canonical_artifacts.route_certificate_json
                ),
                "session_root": session_root,
            }
        )
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
    if observation_tf_evidence_path is not None:
        _admit_observation_tf_readiness(
            profile,
            observation_tf_evidence_path,
            phase="coverage_leg_before_motion",
            typed_run_already_issued=execute,
        )
    if not execute:
        return dry_outcome
    motion_permit_path = None
    motion_permit_sha256 = ""
    mission_leg_permit_path = None
    mission_leg_permit_sha256 = ""
    startup_reseal_permit_path = None
    startup_reseal_permit_sha256 = ""
    if runtime_localization_permit_context is not None:
        if dry_certificate is None or dry_budget is None:
            raise RuntimeError(
                "runtime localization permit requires uncertainty-aware dry evidence"
            )
        motion_permit_path, motion_permit_sha256 = (
            issue_runtime_localization_motion_permit(
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
    if startup_reseal_permit_context is not None:
        if dry_certificate is None or dry_budget is None:
            raise RuntimeError(
                "startup reseal permit requires uncertainty-aware dry evidence"
            )
        startup_reseal_permit_path, startup_reseal_permit_sha256 = (
            issue_startup_reseal_motion_permit(
                context=startup_reseal_permit_context,
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
        elif (
            startup_reseal_permit_context is not None
            and startup_reseal_permit_context.recovery_source_kind
            == STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY
        ):
            print(
                "The prior child stopped before motion because its live "
                "map<-odom consistency evidence invalidated the frozen odom "
                "certificate. Fresh stationary AMCL/TF evidence, a new "
                "same-target A* route, exact-start connector, dry-run, "
                "uncertainty budget, and certificate now bind this recovery."
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
        elif startup_reseal_permit_context is not None:
            print(
                "No additional RUN is required: this exact same-target "
                "pre-motion startup recovery is covered by the initial "
                "mission authorization."
            )
            print(f"Startup reseal motion permit: {startup_reseal_permit_path}")
            print(
                "Startup reseal motion permit SHA-256: "
                f"{startup_reseal_permit_sha256}"
            )
        else:
            raise RuntimeError(
                f"resealed route {run_id} lacks an exact "
                f"{fresh_confirmation_reason} recovery permit; refusing "
                "to reprompt or launch motion"
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
    runtime_authorization_path = (
        None
        if runtime_localization_permit_context is None
        else resolve_normal_artifact_path(
            runtime_localization_permit_context.mission_authorization_json,
            label="runtime localization mission authorization",
        )
    )
    startup_authorization_path = (
        None
        if startup_reseal_permit_context is None
        else resolve_normal_artifact_path(
            startup_reseal_permit_context.mission_authorization_json,
            label="startup reseal motion authorization",
        )
    )
    mission_leg_authorization_path = (
        None
        if mission_leg_permit_context is None
        else resolve_normal_artifact_path(
            mission_leg_permit_context.mission_authorization_json,
            label="mission leg authorization",
        )
    )
    runner = _runner_command(
        **common,
        dry_run=False,
        odom_execution_certificate_json=live_certificate,
        uncertainty_budget_json=live_budget,
        mission_motion_authorization_json=runtime_authorization_path,
        runtime_localization_motion_permit_json=motion_permit_path,
        runtime_localization_mission_leg_kind=(
            None
            if runtime_localization_permit_context is None
            else runtime_localization_permit_context.mission_leg_kind
        ),
        runtime_localization_mission_leg_index=(
            None
            if runtime_localization_permit_context is None
            else runtime_localization_permit_context.mission_leg_index
        ),
        runtime_localization_target_id=(
            ""
            if runtime_localization_permit_context is None
            else runtime_localization_permit_context.target_id
        ),
        runtime_localization_target_viewpoint_id=(
            ""
            if runtime_localization_permit_context is None
            else (
                runtime_localization_permit_context.target_viewpoint_id
                if runtime_localization_permit_context.mission_leg_kind
                is MissionLegKind.COVERAGE
                else ""
            )
        ),
        runtime_localization_semantic_map_id=(
            ""
            if runtime_localization_permit_context is None
            else resolved_runtime_localization_semantic_map_id(
                runtime_localization_permit_context
            )
        ),
        startup_reseal_motion_authorization_json=(
            startup_authorization_path
        ),
        startup_reseal_motion_permit_json=startup_reseal_permit_path,
        startup_reseal_mission_leg_kind=(
            None
            if startup_reseal_permit_context is None
            else startup_reseal_permit_context.mission_leg_kind
        ),
        startup_reseal_mission_leg_index=(
            None
            if startup_reseal_permit_context is None
            else startup_reseal_permit_context.mission_leg_index
        ),
        startup_reseal_target_id=(
            ""
            if startup_reseal_permit_context is None
            else startup_reseal_permit_context.target_id
        ),
        startup_reseal_target_viewpoint_id=(
            ""
            if startup_reseal_permit_context is None
            else startup_reseal_permit_context.target_viewpoint_id
        ),
        startup_reseal_semantic_map_id=(
            ""
            if startup_reseal_permit_context is None
            else startup_reseal_permit_context.semantic_map_id
        ),
        mission_leg_motion_authorization_json=(
            mission_leg_authorization_path
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
        mission_leg_evidence_kind=mission_leg_evidence_kind,
        mission_leg_evidence_index=mission_leg_evidence_index,
        mission_leg_evidence_target_id=mission_leg_evidence_target_id,
        mission_session_id=(
            runtime_localization_permit_context.session_id
            if runtime_localization_permit_context is not None
            else (
                startup_reseal_permit_context.session_id
                if startup_reseal_permit_context is not None
                else (
                    ""
                    if mission_leg_permit_context is None
                    else mission_leg_permit_context.session_id
                )
            )
        ),
    )
    live_log_start_offset = _semantic_log_size(semantic_log)
    wrapped = _bundle_command(profile, run_id, runner)
    result = subprocess.run(
        wrapped,
        check=False,
    )
    return replace(
        _motion_outcome_from_log(
            semantic_log,
            run_id=run_id,
            returncode=result.returncode,
            start_offset=live_log_start_offset,
        ),
        odom_execution_certificate_path=live_certificate,
        motion_authorization_permit_path=motion_permit_path,
        motion_authorization_permit_sha256=motion_permit_sha256,
        mission_leg_motion_permit_path=mission_leg_permit_path,
        mission_leg_motion_permit_sha256=mission_leg_permit_sha256,
        startup_reseal_motion_permit_path=startup_reseal_permit_path,
        startup_reseal_motion_permit_sha256=startup_reseal_permit_sha256,
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
    observation_tf_evidence_path: Path | None = None,
    coverage_plan: CoverageSurveyPlan | None = None,
) -> Path:
    epoch_root = survey_root / "raw_epochs" / viewpoint_id
    epoch_root.mkdir(parents=True, exist_ok=False)
    if observation_tf_evidence_path is not None:
        _admit_observation_tf_readiness(
            profile,
            observation_tf_evidence_path,
            phase="coverage_lidar_epoch_before_observer",
            typed_run_already_issued=True,
        )
    selected_plan = coverage_plan or load_coverage_survey_plan(
        survey_root / "coverage_plan.json"
    )
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
        "--survey-candidate-radius-m",
        str(selected_plan.config.candidate_radius_m),
        "--visibility-receipts-jsonl",
        str(epoch_root / "visibility_receipts.jsonl"),
        "--visibility-survey-id",
        selected_plan.survey_id,
        "--visibility-viewpoint-id",
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
    startup_reseal_motion_authorization_json: Path | None = None,
) -> MotionLegOutcome:
    """Adapt parent runtime/effects to the extracted coverage state machine."""

    runtime = (
        profile.resolved_runtime()
        if callable(getattr(profile, "resolved_runtime", None))
        else profile
    )
    config = CoverageLegConfig(
        session_id=args.session_id,
        map_yaml=args.map,
        semantic_map_id=args.semantic_map_id,
        runtime=runtime,
        robot_radius_m=profile.robot_radius_m,
        max_blockage_replans_per_leg=args.max_blockage_replans_per_leg,
        max_startup_reseals_per_leg=int(
            getattr(
                args,
                "max_startup_reseals_per_leg",
                DEFAULT_MAX_STARTUP_RESEALS_PER_LEG,
            )
        ),
        max_runtime_localization_reseals_per_leg=int(
            getattr(
                args,
                "max_runtime_localization_reseals_per_leg",
                DEFAULT_MAX_RUNTIME_LOCALIZATION_RESEALS_PER_LEG,
            )
        ),
        max_localization_readiness_retries_per_leg=int(
            getattr(
                args,
                "max_localization_readiness_retries_per_leg",
                DEFAULT_MAX_LOCALIZATION_READINESS_RETRIES_PER_LEG,
            )
        ),
        localization_branch_proof_id=str(
            getattr(args, "localization_branch_proof_id", "")
        ).strip(),
        uncertainty_sigma_multiplier=float(
            getattr(
                args,
                "uncertainty_sigma_multiplier",
                DEFAULT_UNCERTAINTY_SIGMA_MULTIPLIER,
            )
        ),
    )
    effects = CoverageLegEffects(
        run_motion_leg=lambda **kwargs: _run_motion_leg(
            **kwargs,
            observation_tf_evidence_path=(
                session_root
                / "preflight"
                / f"{kwargs['run_id']}_observation_tf_before_motion.json"
            ),
        ),
        admit_preplanning_localization=_admit_preplanning_localization,
        seal_route=seal_stand_discovery_route,
        event_sink=_append_jsonl,
        clock=time.time,
    )
    return execute_coverage_leg_state_machine(
        profile=profile,
        config=config,
        effects=effects,
        session_root=session_root,
        survey_root=survey_root,
        plan_path=plan_path,
        leg_index=leg_index,
        target_viewpoint_id=target_viewpoint_id,
        source_route=source_route,
        source_diagnostics=source_diagnostics,
        mission_motion_authorization_json=(
            mission_motion_authorization_json
        ),
        mission_leg_motion_authorization_json=(
            mission_leg_motion_authorization_json
        ),
        startup_reseal_motion_authorization_json=(
            startup_reseal_motion_authorization_json
        ),
    )


def _capture_camera_recommendation(
    *,
    profile,
    args,
    candidate,
    output_dir: Path,
    observation_attempt_index: int = 0,
) -> tuple[Path | None, str | None, Path | None]:
    if args.stand_model_profile is None:
        raise RuntimeError(
            "camera exploration requires a measured physical stand model"
        )
    stand_model = load_measured_physical_stand_model(args.stand_model_profile)
    stand_head_center_height_m = stand_model.head_center_height_m
    if stand_head_center_height_m is None:
        raise RuntimeError(
            "camera exploration requires complete measured stand geometry"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "observer_status.json"
    status_events_path = output_dir / "observer_events.jsonl"
    process_evidence_path = output_dir / "observer_process.json"
    recommendation_path = output_dir / "recommendation.json"
    axis_observation_path = output_dir / "axis_observation.json"
    command = [
        sys.executable,
        "scripts/aufgabe04/real_robot/entrypoints/passive_viewpoint_node.py",
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
        "--stand-head-center-height-m",
        str(stand_head_center_height_m),
        "--target-distance-m",
        str(args.final_facing_offset_m),
        "--consensus-frames",
        str(args.axis_sample_count),
        "--status-json",
        str(status_path),
        "--status-events-jsonl",
        str(status_events_path),
        "--recommended-pose-json",
        str(recommendation_path),
        "--axis-observation-json",
        str(axis_observation_path),
        "--debug-dir",
        str(output_dir / "perception_debug"),
        "--once",
    ]
    command.extend(
        [
            "--stand-model-profile",
            str(args.stand_model_profile),
        ]
    )
    process = subprocess.Popen(command)
    process_evidence = monitor_passive_observer_process(
        process=process,
        recommendation_path=recommendation_path,
        axis_observation_path=axis_observation_path,
        timeout_sec=args.camera_timeout_sec,
    )
    write_content_hashed_json(
        process_evidence_path,
        process_evidence.to_dict(),
        hash_field="observer_process_evidence_sha256",
    )
    if process_evidence.artifact_kind == "axis_observation":
        return None, None, axis_observation_path
    if process_evidence.artifact_kind != "recommendation":
        status_evidence = load_passive_observer_status(status_path)
        reason = format_passive_observer_failure(
            candidate_uid=candidate.candidate_uid,
            process=process_evidence,
            status=status_evidence,
            process_evidence_path=process_evidence_path,
        )
        if is_candidate_local_observer_timeout(
            process=process_evidence,
            status=status_evidence,
        ):
            raise CandidateObservationUnavailableError(
                candidate_uid=candidate.candidate_uid,
                observation_attempt_index=observation_attempt_index,
                reason=reason,
                process_evidence={
                    **process_evidence.to_dict(),
                    "evidence_path": str(process_evidence_path),
                },
                status_evidence=status_evidence.to_dict(),
            )
        raise RuntimeError(reason)
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "camera recommendation committed but its terminal observer status "
            f"is unreadable: {type(exc).__name__}; observer_process_evidence="
            f"{process_evidence_path}"
        ) from exc
    if not isinstance(status, dict):
        raise RuntimeError(
            "camera recommendation terminal observer status must be a JSON "
            f"object; observer_process_evidence={process_evidence_path}"
        )
    qr_texts_value = status.get("qr_texts", ())
    if not isinstance(qr_texts_value, (list, tuple)):
        raise RuntimeError(
            "camera recommendation terminal observer status has invalid "
            f"qr_texts; observer_process_evidence={process_evidence_path}"
        )
    qr_texts = tuple(qr_texts_value)
    if len(qr_texts) != 1 or not str(qr_texts[0]).strip():
        raise RuntimeError(
            "camera recommendation did not bind one QR identity; "
            f"observer_process_evidence={process_evidence_path}"
        )
    return recommendation_path, str(qr_texts[0]), (
        axis_observation_path if axis_observation_path.exists() else None
    )


def _run_candidate_motion_leg(
    *,
    profile,
    request: CandidateMotionLegRequest,
) -> MotionLegOutcome:
    """Adapt one typed candidate request to the sole child motion edge."""

    return _run_motion_leg(
        profile=profile,
        sealed=request.sealed,
        run_id=request.run_id,
        session_root=request.session_root,
        execute=True,
        candidate_snapshot=request.candidate_snapshot_path,
        uncertainty_map_yaml=request.uncertainty_map_yaml,
        uncertainty_sigma_multiplier=request.uncertainty_sigma_multiplier,
        localization_branch_proof_id=request.localization_branch_proof_id,
        mission_leg_permit_context=MissionLegPermitContext(
            mission_authorization_json=request.mission_authorization_json,
            session_id=request.session_id,
            semantic_map_id=request.semantic_map_id,
            mission_leg_kind=request.mission_leg_kind,
            mission_leg_index=request.mission_leg_index,
            target_id=request.target_id,
            permit_json_path=request.permit_json_path,
        ),
    )


def _run_candidate_startup_reseal_motion_leg(
    *,
    profile,
    request: CandidateMotionLegRequest,
    attempt: CandidateStartupRecoveryAttempt,
    startup_reseal_motion_authorization_json: Path,
    max_startup_reseals_per_leg: int,
) -> MotionLegOutcome:
    """Run one exact candidate startup replacement under a one-use permit."""

    identity = attempt.identity
    if (
        identity.run_id != request.run_id
        or identity.routine_kind != request.mission_leg_kind.value
        or identity.routine_index != request.mission_leg_index
        or identity.target_id != request.target_id
        or identity.session_id != request.session_id
        or identity.semantic_map_id != request.semantic_map_id
    ):
        raise RuntimeError(
            "candidate startup replacement request changed its routine identity"
        )
    authorization_root = (
        request.session_root / "motion_authorization" / "startup_reseals"
    )
    summary_path = write_startup_reseal_permit_summary(
        authorization_root / f"{request.run_id}_sealed_summary.json",
        leg_index=request.mission_leg_index,
        target_viewpoint_id=request.target_id,
        mission_leg_kind=request.mission_leg_kind,
        mission_leg_index=request.mission_leg_index,
        target_id=request.target_id,
        reseal_index=attempt.reseal_index,
        rejected_run_id=attempt.rejected_outcome.run_id,
        fresh_start_x_m=attempt.fresh_start_pose.x_m,
        fresh_start_y_m=attempt.fresh_start_pose.y_m,
        fresh_start_yaw_rad=attempt.fresh_start_pose.yaw_rad,
        route_csv=Path(request.sealed["route_csv"]),
        diagnostics_json=Path(request.sealed["diagnostics_json"]),
        additional_typed_run_required=False,
        recovery_source_kind=attempt.recovery_source_kind,
    )
    permit_context = StartupResealPermitContext(
        mission_authorization_json=Path(
            startup_reseal_motion_authorization_json
        ).absolute(),
        session_id=request.session_id,
        semantic_map_id=request.semantic_map_id,
        leg_index=request.mission_leg_index,
        target_viewpoint_id=request.target_id,
        mission_leg_kind=request.mission_leg_kind,
        mission_leg_index=request.mission_leg_index,
        target_id=request.target_id,
        reseal_index=attempt.reseal_index,
        max_startup_reseals_per_leg=max_startup_reseals_per_leg,
        rejected_run_id=attempt.rejected_outcome.run_id,
        rejected_semantic_log_path=Path(
            attempt.rejected_outcome.semantic_log_path
        ).absolute(),
        startup_reseal_summary_path=summary_path,
        fresh_localization_evidence_path=(
            attempt.fresh_localization_evidence_path.absolute()
        ),
        permit_json_path=(
            authorization_root / f"{request.run_id}_permit.json"
        ).absolute(),
        recovery_source_kind=attempt.recovery_source_kind,
    )
    return _run_motion_leg(
        profile=profile,
        sealed=dict(request.sealed),
        run_id=request.run_id,
        session_root=request.session_root,
        execute=True,
        candidate_snapshot=request.candidate_snapshot_path,
        require_fresh_confirmation=True,
        fresh_confirmation_reason="startup",
        fresh_localization_evidence_path=(
            attempt.fresh_localization_evidence_path
        ),
        uncertainty_map_yaml=request.uncertainty_map_yaml,
        uncertainty_sigma_multiplier=request.uncertainty_sigma_multiplier,
        localization_branch_proof_id=request.localization_branch_proof_id,
        startup_reseal_permit_context=permit_context,
    )


def _run_candidate_runtime_localization_reseal_motion_leg(
    *,
    profile,
    request: CandidateMotionLegRequest,
    attempt: CandidateRuntimeRecoveryAttempt,
    mission_motion_authorization_json: Path,
    max_runtime_reseals_per_leg: int,
) -> MotionLegOutcome:
    """Run one exact post-motion candidate replacement under a one-use permit."""

    identity = attempt.identity
    if (
        identity.run_id != request.run_id
        or identity.routine_kind != request.mission_leg_kind.value
        or identity.routine_index != request.mission_leg_index
        or identity.target_id != request.target_id
        or identity.session_id != request.session_id
        or identity.semantic_map_id != request.semantic_map_id
    ):
        raise RuntimeError(
            "candidate runtime replacement request changed its routine identity"
        )
    authorization_root = (
        request.session_root
        / "motion_authorization"
        / "runtime_localization"
    )
    permit_context = RuntimeLocalizationPermitContext(
        mission_authorization_json=Path(
            mission_motion_authorization_json
        ).absolute(),
        session_id=request.session_id,
        leg_index=request.mission_leg_index,
        target_viewpoint_id=request.target_id,
        mission_leg_kind=request.mission_leg_kind,
        mission_leg_index=request.mission_leg_index,
        target_id=request.target_id,
        semantic_map_id=request.semantic_map_id,
        reseal_index=attempt.reseal_index,
        max_runtime_reseals_per_leg=max_runtime_reseals_per_leg,
        rejected_run_id=attempt.rejected_outcome.run_id,
        runtime_reseal_decision_evidence=(
            attempt.runtime_localization_decision.to_evidence()
        ),
        fresh_localization_evidence_path=(
            attempt.fresh_localization_evidence_path.absolute()
        ),
        permit_json_path=(
            authorization_root / f"{request.run_id}_permit.json"
        ).absolute(),
    )
    return _run_motion_leg(
        profile=profile,
        sealed=dict(request.sealed),
        run_id=request.run_id,
        session_root=request.session_root,
        execute=True,
        candidate_snapshot=request.candidate_snapshot_path,
        require_fresh_confirmation=True,
        fresh_confirmation_reason="runtime_localization",
        fresh_localization_evidence_path=(
            attempt.fresh_localization_evidence_path
        ),
        uncertainty_map_yaml=request.uncertainty_map_yaml,
        uncertainty_sigma_multiplier=request.uncertainty_sigma_multiplier,
        localization_branch_proof_id=request.localization_branch_proof_id,
        runtime_localization_permit_context=permit_context,
    )


def _capture_candidate_observation(
    *,
    profile,
    args,
    request: CandidateObservationRequest,
) -> CandidateObservation:
    """Adapt the passive observer process to the typed candidate boundary."""

    recommendation_path, qr_id, axis_observation_path = (
        _capture_camera_recommendation(
            profile=profile,
            args=args,
            candidate=request.candidate,
            output_dir=request.output_dir,
            observation_attempt_index=request.attempt_index,
        )
    )
    return CandidateObservation(
        recommendation_path=recommendation_path,
        qr_id=qr_id,
        axis_observation_path=axis_observation_path,
    )


from .cli import (
    build_parser,
)


def _validate_inputs(parser, args, profile, calibration) -> None:
    if (
        type(args.expected_stand_count) is not int
        or args.expected_stand_count <= 0
    ):
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
    if args.max_localization_readiness_retries_per_leg < 0:
        parser.error(
            "--max-localization-readiness-retries-per-leg must be non-negative"
        )
    if (
        not math.isfinite(args.initialpose_prompt_window_sec)
        or args.initialpose_prompt_window_sec <= 0.0
    ):
        parser.error("--initialpose-prompt-window-sec must be finite and positive")
    if args.max_camera_observation_attempts_per_candidate < 1:
        parser.error(
            "--max-camera-observation-attempts-per-candidate must be positive"
        )
    if (
        not math.isfinite(args.uncertainty_sigma_multiplier)
        or args.uncertainty_sigma_multiplier <= 0.0
    ):
        parser.error(
            "--uncertainty-sigma-multiplier must be finite and positive"
        )
    args.localization_branch_proof_id = str(
        args.localization_branch_proof_id
    ).strip()
    if args.execute and not args.localization_branch_proof_id:
        parser.error(
            "physical execution modes require --localization-branch-proof-id "
            "for a known physical start or asymmetric landmark"
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
            load_measured_physical_stand_model(args.stand_model_profile)
        except (OSError, ValueError) as exc:
            parser.error(f"invalid stand model profile: {exc}")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        resolved_run_mode = resolve_autonomous_run_mode(
            run_mode=args.run_mode,
            execute=args.execute,
            coverage_leg_limit=args.coverage_leg_limit,
            stop_after_coverage=args.stop_after_coverage,
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.run_mode = resolved_run_mode.mode.value
    args.execute = resolved_run_mode.execute
    args.coverage_leg_limit = resolved_run_mode.coverage_leg_limit
    args.stop_after_coverage = resolved_run_mode.stop_after_coverage
    try:
        validate_autonomous_viewpoint_scope(
            resolved_run_mode,
            exact_inspection_point_count=args.exact_inspection_point_count,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if (
        resolved_run_mode.camera_phase_enabled
        and args.stand_model_profile is None
    ):
        parser.error(
            f"--run-mode {resolved_run_mode.mode.value} requires "
            "--stand-model-profile with measured physical geometry"
        )
    resume_mode = (
        resolved_run_mode.mode
        is AutonomousRunMode.RESUME_NEXT_COVERAGE_LEG
    )
    if resume_mode and args.resume_checkpoint is None:
        parser.error(
            "--run-mode resume-next-coverage-leg requires --resume-checkpoint"
        )
    if not resume_mode and args.resume_checkpoint is not None:
        parser.error(
            "--resume-checkpoint requires --run-mode resume-next-coverage-leg"
        )
    if args.session_id:
        try:
            validate_session_id_mode_label(args.session_id, resolved_run_mode)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        args.session_id = _default_session_id(resolved_run_mode.mode)
    session_root = args.output_root / args.session_id
    if session_root.exists():
        parser.error(f"refusing to reuse existing session: {session_root}")
    try:
        profile = load_real_robot_profile(args.robot_profile)
        calibration = load_camera_calibration(args.camera_calibration)
        site_contract = validate_physical_site_contract(
            args.physical_site,
            profile=profile,
            requested_expected_stand_count=args.expected_stand_count,
            semantic_map_id=args.semantic_map_id,
            map_yaml=args.map,
            repository_root=ROOT,
        )
        args.expected_stand_count = site_contract.expected_stand_count
        args.physical_site = site_contract.physical_site_path
        args.map = site_contract.map_yaml_path
        _validate_inputs(parser, args, profile, calibration)
        stand_model = (
            None
            if args.stand_model_profile is None
            else load_measured_physical_stand_model(args.stand_model_profile)
        )
        runtime = profile.resolved_runtime()
        clearance = _physical_clearance(
            profile,
            approach_offset_m=args.candidate_approach_offset_m,
            stand_model_profile=stand_model,
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
        admitted_resume = None
        if resume_mode:
            admitted_resume = admit_coverage_resume(
                args.resume_checkpoint,
                new_session_id=args.session_id,
                robot_id=profile.robot_id,
                robot_profile_sha256=_file_sha256(args.robot_profile),
                calibration_profile_sha256=_file_sha256(
                    args.camera_calibration
                ),
                physical_site_sha256=_file_sha256(args.physical_site),
                map_bundle_sha256=site_contract.map_bundle.bundle_sha256,
                config_sha256=_checkpoint_config_sha256(args),
            )
        session_root.mkdir(parents=True, exist_ok=False)
        survey_root = session_root / "coverage"
        (
            preauthorization_observation_tf_path,
            preauthorization_observation_tf_sha256,
        ) = _admit_observation_tf_readiness(
            profile,
            session_root
            / "preflight"
            / "lidar_scan_tf_before_authorization.json",
            phase="preauthorization_observation_tf_readiness",
            typed_run_already_issued=False,
        )
        start = _admit_preplanning_localization(
            runtime,
            session_root,
        )
        resume_parent_checkpoint_path: Path | None = None
        if resume_mode:
            assert admitted_resume is not None
            restored_resume = restore_and_replan_coverage_resume(
                admitted_resume,
                survey_root=survey_root,
                map_yaml=args.map,
                semantic_map_id=args.semantic_map_id,
                current_pose=start,
            )
            plan_path = restored_resume.plan_path
            plan = restored_resume.plan
            initial_leg_index = restored_resume.leg_index
            resume_parent_checkpoint_path = (
                restored_resume.parent_checkpoint_path
            )
        else:
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
                        "--exact-two-candidate-spacing-m",
                        str(DEFAULT_EXACT_TWO_CANDIDATE_SPACING_M),
                        "--minimum-exact-two-viewpoint-baseline-m",
                        str(DEFAULT_MINIMUM_EXACT_TWO_VIEWPOINT_BASELINE_M),
                    ]
                )
            planning_status = plan_stand_coverage_survey(planning_command)
            if planning_status != 0:
                return planning_status
            plan_path = survey_root / "coverage_plan.json"
            plan = load_coverage_survey_plan(plan_path)
            initial_leg_index = 0
        if (
            args.exact_inspection_point_count is not None
            and len(plan.viewpoints) != args.exact_inspection_point_count
        ):
            raise RuntimeError(
                "coverage planner did not preserve the exact inspection-point "
                "count before motion authorization"
            )

        def run_preauthorization_dry_leg(request):
            return _run_motion_leg(
                profile=profile,
                sealed=request.sealed_route.to_mapping(),
                run_id=request.run_id,
                session_root=session_root,
                execute=False,
                coverage_plan=plan_path,
                uncertainty_map_yaml=args.map,
                uncertainty_sigma_multiplier=args.uncertainty_sigma_multiplier,
                localization_branch_proof_id=(
                    args.localization_branch_proof_id or "dry_run_no_motion"
                ),
            )

        initialpose_prompt_config = InitialPosePromptConfig(
            amcl_topic=getattr(profile, "amcl_topic", "amcl_pose"),
            observation_window_sec=args.initialpose_prompt_window_sec,
            maximum_retry_count=args.max_localization_readiness_retries_per_leg,
        )

        def prepare_initial_readiness_localization(request):
            if not args.prompt_for_initialpose:
                return
            prompt_for_initialpose_attempt(
                config=initialpose_prompt_config,
                attempt_index=request.attempt_index,
            )

        initial_admission = admit_preauthorization_readiness(
            PreauthorizationReadinessConfig(
                session_root=session_root,
                survey_root=survey_root,
                coverage_plan_path=plan_path,
                session_id=args.session_id,
                initial_leg_index=initial_leg_index,
                maximum_localization_readiness_retries=(
                    args.max_localization_readiness_retries_per_leg
                ),
                observation_tf_evidence_path=(
                    preauthorization_observation_tf_path
                ),
                observation_tf_evidence_sha256=(
                    preauthorization_observation_tf_sha256
                ),
            ),
            PreauthorizationReadinessEffects(
                seal_route=seal_stand_discovery_route,
                run_dry_motion_leg=run_preauthorization_dry_leg,
                append_event=_append_jsonl,
                publish_hashed_json=write_content_hashed_json,
                wall_clock=time.time,
                notify=print,
                prepare_localization_attempt=prepare_initial_readiness_localization,
            ),
        )
        initial_readiness = initial_admission.result
        initial_readiness_path = initial_admission.evidence_path
        initial_readiness_sha256 = initial_admission.evidence_sha256

        if not args.execute:
            _write_json(
                session_root / "mission_summary.json",
                {
                    "schema_version": 1,
                    "status": "first_leg_dry_run_ok",
                    "run_mode": args.run_mode,
                    "execute": False,
                    "motion_published": False,
                    "survey_root": str(survey_root),
                    "uncertainty_sigma_multiplier": (
                        args.uncertainty_sigma_multiplier
                    ),
                    "localization_readiness_retry_count": (
                        len(initial_readiness.attempts) - 1
                    ),
                    "initial_readiness_json": str(initial_readiness_path),
                    "initial_readiness_sha256": initial_readiness_sha256,
                },
            )
            print(
                "First center-corridor leg passed the runner dry-run. "
                "Use a new session with an explicit execute-* run mode for "
                "any separately authorized physical mission."
            )
            return 0

        # Freeze every checkpoint identity before asking the operator to RUN.
        # The immutable inputs are hashed once and reused across all coverage
        # legs; unreadable or inconsistent artifacts therefore fail before
        # any mission-level motion authorization is issued.
        checkpoint_identity = CoverageCheckpointIdentity(
            session_root=session_root,
            session_id=args.session_id,
            run_mode=args.run_mode,
            robot_id=profile.robot_id,
            robot_profile_sha256=_file_sha256(args.robot_profile),
            calibration_profile_sha256=_file_sha256(
                args.camera_calibration
            ),
            physical_site_sha256=_file_sha256(args.physical_site),
            map_bundle_sha256=plan.map_bundle_sha256,
            config_sha256=_checkpoint_config_sha256(args),
        )

        authorization_scope = resolved_run_mode.authorization_scope_text
        coverage_scoped_mode = not resolved_run_mode.camera_phase_enabled
        authorized_leg_kinds = (
            (MissionLegKind.COVERAGE,)
            if coverage_scoped_mode
            else ROUTINE_MISSION_LEG_KINDS
        )
        authorized_leg_description = (
            "coverage child legs"
            if coverage_scoped_mode
            else "coverage, candidate, and opposite-face child legs"
        )
        print(
            "Preauthorization readiness passed without motion: the exact "
            "first route cleared its AMCL uncertainty budget and the LiDAR "
            "scan-time TF chain was observable. This evidence is advisory; "
            "the authorized child will repeat every dry/live gate."
        )
        print(f"Initial readiness evidence: {initial_readiness_path}")
        print(f"Initial readiness SHA-256: {initial_readiness_sha256}")
        print(
            "Physical safety requirements: clear arena; unloaded robot; operator "
            "beside the robot; Ctrl+C and physical stop ready; separate exact-topic "
            f"zero Twist terminal ready. This RUN authorizes {authorization_scope} "
            f"and its separately sealed routine {authorized_leg_description}, "
            "plus bounded scan-backed transient-"
            "obstacle A* replans "
            f"(maximum {args.max_blockage_replans_per_leg} per coverage leg). "
            "Each routine child must pass a fresh dry-run and all live gates, "
            "then atomically consume its exact one-leg permit; it will not ask "
            "for another RUN. "
            "A same-target pre-motion AMCL/start mismatch or an exact "
            "zero-motion map<-odom consistency stop may reuse this RUN only "
            "after fresh stationary localization, route reconstruction, "
            "dry-run/certificates, and an exact one-use startup-recovery "
            "permit all pass. On an authorized routine leg, an exact "
            "post-motion global-localization reseal "
            "also stops first, recollects stationary AMCL/TF evidence, "
            "rebuilds the route to the same target, reruns every "
            "dry/live gate, and may reuse this RUN for at most "
            f"{args.max_runtime_localization_reseals_per_leg} reseal(s) per leg "
            "through an exact one-run permit. Route-tube, stale-TF, obstacle, "
            "ownership, target-change, malformed-evidence, and budget failures "
            "remain terminal. The AMCL envelope multiplier is "
            f"{args.uncertainty_sigma_multiplier:g}; it is charged to route "
            "clearance before motion and reused by the live map<-odom monitor."
            " A route-specific uncertainty-budget rejection before motion may "
            "trigger at most "
            f"{args.max_localization_readiness_retries_per_leg} fresh no-motion "
            "AMCL dry admission(s); no limit is relaxed and no permit is issued "
            "until one passes."
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
            allowed_leg_kinds=authorized_leg_kinds,
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
            allowed_mission_leg_kinds=authorized_leg_kinds,
        )
        mission_motion_authorization_hash = (
            write_mission_motion_authorization(
                mission_motion_authorization_json,
                mission_motion_authorization,
            )
        )
        startup_reseal_motion_authorization_json = (
            session_root
            / "motion_authorization"
            / "startup_reseal_motion_authorization.json"
        ).absolute()
        startup_reseal_motion_authorization = (
            StartupResealMotionAuthorization(
                session_id=args.session_id,
                robot_id=profile.robot_id,
                namespace=runtime.namespace,
                cmd_vel_topic=runtime.cmd_vel_topic,
                semantic_map_id=args.semantic_map_id,
                localization_branch_proof_id=(
                    args.localization_branch_proof_id
                ),
                max_startup_reseals_per_leg=(
                    args.max_startup_reseals_per_leg
                ),
                scope_text=STARTUP_RESEAL_MOTION_AUTHORIZATION_SCOPE,
                operator_confirmation=STARTUP_RESEAL_RUN_CONFIRMATION,
                allowed_recovery_kind=STARTUP_RESEAL_RECOVERY_KIND,
                allowed_mission_leg_kinds=authorized_leg_kinds,
            )
        )
        startup_reseal_motion_authorization_hash = (
            write_startup_reseal_motion_authorization(
                startup_reseal_motion_authorization_json,
                startup_reseal_motion_authorization,
            )
        )
        _append_jsonl(
            session_root / "adaptive_replans.jsonl",
            {
                "schema_version": 1,
                "event": "mission_motion_authorization_issued",
                "timestamp": time.time(),
                "session_id": args.session_id,
                "run_mode": args.run_mode,
                "authorization_scope": authorization_scope,
                "resume_parent_checkpoint": (
                    None
                    if resume_parent_checkpoint_path is None
                    else str(resume_parent_checkpoint_path)
                ),
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
                    kind.value for kind in authorized_leg_kinds
                ],
                "routine_child_prompts_required": False,
                "startup_reseal_fresh_typed_run_required": False,
                "startup_reseal_exact_recovery_permit_required": True,
                "startup_reseal_motion_authorization_json": str(
                    startup_reseal_motion_authorization_json
                ),
                "startup_reseal_motion_authorization_sha256": (
                    startup_reseal_motion_authorization_hash
                ),
                "max_startup_reseals_per_leg": (
                    args.max_startup_reseals_per_leg
                ),
                "max_runtime_localization_reseals_per_leg": (
                    args.max_runtime_localization_reseals_per_leg
                ),
                "max_localization_readiness_retries_per_leg": (
                    args.max_localization_readiness_retries_per_leg
                ),
                "uncertainty_sigma_multiplier": (
                    args.uncertainty_sigma_multiplier
                ),
                "allowed_recovery_kind": (
                    RUNTIME_LOCALIZATION_RESEAL_RECOVERY_KIND
                ),
                "allowed_startup_recovery_kind": (
                    STARTUP_RESEAL_RECOVERY_KIND
                ),
                "allowed_startup_recovery_source_kinds": sorted(
                    STARTUP_RESEAL_RECOVERY_SOURCE_KINDS
                ),
                "additional_typed_run_required_for_eligible_recovery": False,
                "initial_readiness_json": str(initial_readiness_path),
                "initial_readiness_sha256": initial_readiness_sha256,
                "initial_readiness_reusable_as_motion_permit": False,
                "preauthorization_observation_tf_json": str(
                    preauthorization_observation_tf_path
                ),
                "preauthorization_observation_tf_sha256": (
                    preauthorization_observation_tf_sha256
                ),
            },
        )

        def execute_completed_coverage_leg(request):
            outcome = _execute_coverage_leg_with_replans(
                profile=profile,
                args=args,
                session_root=session_root,
                survey_root=survey_root,
                plan_path=plan_path,
                leg_index=request.leg_index,
                target_viewpoint_id=request.target_viewpoint_id,
                source_route=request.source_route,
                source_diagnostics=request.source_diagnostics,
                mission_motion_authorization_json=(
                    mission_motion_authorization_json
                ),
                mission_leg_motion_authorization_json=(
                    mission_leg_motion_authorization_json
                ),
                startup_reseal_motion_authorization_json=(
                    startup_reseal_motion_authorization_json
                ),
            )
            if outcome.odom_execution_certificate_path is None:
                raise RuntimeError(
                    "completed coverage leg has no odom execution certificate"
                )
            return CompletedCoverageLeg(
                outcome.odom_execution_certificate_path
            )

        def capture_coverage_lidar_epoch(
            viewpoint_id,
            odom_execution_certificate_path,
        ):
            return _capture_lidar_epoch(
                profile=profile,
                args=args,
                survey_root=survey_root,
                viewpoint_id=viewpoint_id,
                odom_execution_certificate_path=(
                    odom_execution_certificate_path
                ),
                observation_tf_evidence_path=(
                    survey_root
                    / "raw_epochs"
                    / viewpoint_id
                    / "observation_tf_readiness.json"
                ),
                coverage_plan=plan,
            )

        def fuse_coverage_stop(viewpoint_id, observer_summary_path):
            return commit_stand_coverage_stop(
                survey_root=survey_root,
                map_yaml=args.map,
                semantic_map_id=args.semantic_map_id,
                viewpoint_id=viewpoint_id,
                observer_summary_json=observer_summary_path,
                scan_to_base_position_offset_m=(
                    profile.scan_origin_to_base_offset_m
                ),
            )

        def prepare_next_coverage_leg(request):
            post_observation = admit_post_observation_localization(
                PostObservationLocalizationConfig(
                    session_root=session_root,
                    session_id=args.session_id,
                    recorded_viewpoint_id=request.recorded_viewpoint_id,
                    maximum_retry_count=(
                        args.max_localization_readiness_retries_per_leg
                    ),
                ),
                PostObservationLocalizationEffects(
                    admit_localization=lambda evidence_path: (
                        _admit_preplanning_localization(
                            runtime,
                            session_root,
                            evidence_path=evidence_path,
                        )
                    ),
                    event_sink=lambda event: _append_jsonl(
                        session_root / "adaptive_replans.jsonl",
                        dict(event),
                    ),
                    clock=time.time,
                ),
            )
            planned = plan_next_stand_coverage_leg(
                survey_root=survey_root,
                map_yaml=args.map,
                semantic_map_id=args.semantic_map_id,
                expected_next_viewpoint_id=request.target_viewpoint_id,
                current_pose=post_observation.pose,
                localization_evidence_json=post_observation.evidence_path,
                localization_evidence_sha256=authorization_file_sha256(
                    post_observation.evidence_path
                ),
                checkpoint_manifest_json=request.checkpoint_manifest,
                checkpoint_manifest_sha256=request.checkpoint_manifest_sha256,
            )
            if planned.get("next_viewpoint_id") != request.target_viewpoint_id:
                raise RuntimeError(
                    "prepared coverage receipt changed the checkpointed target"
                )
            return PreparedCoverageLeg(
                leg_index=request.leg_index,
                target_viewpoint_id=request.target_viewpoint_id,
                source_route=resolve_normal_artifact_path(
                    planned["next_route_csv"],
                    label="prepared coverage route",
                ),
                source_diagnostics=resolve_normal_artifact_path(
                    planned["next_diagnostics_json"],
                    label="prepared coverage diagnostics",
                ),
            )

        def publish_coverage_checkpoint_effect(request):
            identity = request.identity
            published = publish_coverage_checkpoint(
                session_root=identity.session_root,
                session_id=identity.session_id,
                run_mode=identity.run_mode,
                robot_id=identity.robot_id,
                robot_profile_sha256=identity.robot_profile_sha256,
                calibration_profile_sha256=(
                    identity.calibration_profile_sha256
                ),
                physical_site_sha256=identity.physical_site_sha256,
                map_bundle_sha256=identity.map_bundle_sha256,
                config_sha256=identity.config_sha256,
                completed_coverage_legs=request.completed_coverage_legs,
                next_viewpoint_id=request.next_viewpoint_id,
                coverage_plan_path=request.coverage_plan_path,
                coverage_progress_path=request.coverage_progress_path,
                survey_summary_path=request.survey_summary_path,
                stand_registry_path=request.stand_registry_path,
                lidar_observer_summary_path=(
                    request.lidar_observer_summary_path
                ),
                parent_checkpoint_path=request.parent_checkpoint_path,
                status=request.checkpoint_status,
            )
            return PublishedCoverageCheckpoint(
                published.manifest_path,
                published.manifest_sha256,
            )

        def build_coverage_candidate_snapshot(
            registry,
            coverage_plan,
            registry_path,
            snapshot_id,
        ):
            return candidate_snapshot_from_registry(
                registry,
                coverage_plan,
                registry_path=registry_path,
                snapshot_id=snapshot_id,
            )

        coverage_phase = execute_coverage_mission(
            CoverageMissionConfig(
                survey_root=survey_root,
                plan=plan,
                coverage_plan_path=plan_path,
                checkpoint_identity=checkpoint_identity,
                expected_stand_count=args.expected_stand_count,
                initial_leg_index=initial_leg_index,
                coverage_leg_limit=args.coverage_leg_limit,
                parent_checkpoint_path=resume_parent_checkpoint_path,
                completion_policy=_coverage_completion_policy(
                    resolved_run_mode.mode,
                    exact_inspection_point_count=(
                        _plan_exact_inspection_point_count(
                            plan,
                            requested=args.exact_inspection_point_count,
                        )
                    ),
                ),
            ),
            CoverageMissionEffects(
                execute_completed_leg=execute_completed_coverage_leg,
                capture_lidar_epoch=capture_coverage_lidar_epoch,
                fuse_coverage_stop=fuse_coverage_stop,
                prepare_next_leg=prepare_next_coverage_leg,
                build_snapshot=build_coverage_candidate_snapshot,
                publish_checkpoint=publish_coverage_checkpoint_effect,
                load_progress=load_survey_progress,
                load_registry=load_stand_survey_registry,
                evaluate_admission=evaluate_coverage_candidate_admission,
                write_admission=lambda path, decision: (
                    write_content_hashed_json(
                        path,
                        coverage_candidate_admission_evidence(decision),
                        hash_field=(
                            "coverage_candidate_admission_sha256"
                        ),
                    )
                ),
                write_snapshot=write_candidate_snapshot,
                snapshot_sha256=candidate_snapshot_sha256,
            ),
        )

        if isinstance(
            coverage_phase,
            (CoverageCheckpointComplete, CoverageLidarCheckpointComplete),
        ):
            result = coverage_phase.to_mission_summary()
            _write_json(session_root / "mission_summary.json", result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        exact_two_camera_handoff_path: Path | None = None
        exact_two_camera_handoff_sha256: str | None = None
        exact_two_camera_summary: dict[str, object] | None = None
        if isinstance(coverage_phase, CoverageExactTwoCameraReady):
            if (
                resolved_run_mode.mode
                is not AutonomousRunMode.EXECUTE_EXACT_TWO_CAMERA
            ):
                raise RuntimeError(
                    "exact-two camera handoff returned outside its run mode"
                )
            snapshot = coverage_phase.candidate_snapshot
            snapshot_path = coverage_phase.candidate_snapshot_path
            snapshot_sha256 = coverage_phase.candidate_snapshot_sha256
            coverage_candidate_admission_path = (
                coverage_phase.camera_validation_admission_path
            )
            coverage_candidate_admission_sha256 = (
                coverage_phase.camera_validation_admission_sha256
            )
            exact_two_camera_handoff_path = coverage_phase.camera_handoff_path
            exact_two_camera_handoff_sha256 = (
                coverage_phase.camera_handoff_sha256
            )
            exact_two_camera_summary = coverage_phase.to_mission_summary()
        elif isinstance(coverage_phase, CoverageComplete):
            snapshot = coverage_phase.candidate_snapshot
            snapshot_path = coverage_phase.candidate_snapshot_path
            snapshot_sha256 = coverage_phase.candidate_snapshot_sha256
            coverage_candidate_admission_path = (
                coverage_phase.coverage_candidate_admission_path
            )
            coverage_candidate_admission_sha256 = (
                coverage_phase.coverage_candidate_admission_sha256
            )
        else:
            raise RuntimeError("coverage transaction returned an unknown outcome")

        if not resolved_run_mode.camera_phase_enabled:
            result = coverage_phase.to_mission_summary()
            _write_json(session_root / "mission_summary.json", result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        if mission_leg_motion_authorization_json is None:
            raise RuntimeError(
                "candidate phase requires mission-leg authorization evidence"
            )
        if startup_reseal_motion_authorization_json is None:
            raise RuntimeError(
                "candidate phase requires startup-reseal authorization evidence"
            )
        candidate_phase = execute_candidate_approach_phase(
            CandidateApproachConfig(
                session_root=session_root,
                survey_root=survey_root,
                session_id=args.session_id,
                semantic_map_id=args.semantic_map_id,
                planning_frame=profile.map_frame,
                map_yaml=args.map,
                plan=plan,
                snapshot=snapshot,
                snapshot_path=snapshot_path,
                approach_offset_m=args.candidate_approach_offset_m,
                inflation_radius_m=inflation_radius_m,
                candidate_transit_radius_m=candidate_keepout_radius_m,
                physical_clearance=clearance,
                uncertainty_sigma_multiplier=(
                    args.uncertainty_sigma_multiplier
                ),
                localization_branch_proof_id=(
                    args.localization_branch_proof_id
                ),
                mission_leg_motion_authorization_json=(
                    mission_leg_motion_authorization_json
                ),
                startup_reseal_motion_authorization_json=(
                    startup_reseal_motion_authorization_json
                ),
                max_startup_reseals_per_leg=(
                    args.max_startup_reseals_per_leg
                ),
                mission_motion_authorization_json=(
                    mission_motion_authorization_json
                ),
                max_runtime_localization_reseals_per_leg=(
                    args.max_runtime_localization_reseals_per_leg
                ),
                exact_two_camera_handoff_path=(
                    exact_two_camera_handoff_path
                ),
                exact_two_camera_handoff_sha256=(
                    exact_two_camera_handoff_sha256
                ),
                camera_selection_linear_speed_mps=(
                    profile.max_linear_speed_mps
                ),
                camera_selection_angular_speed_radps=(
                    profile.max_angular_speed_radps
                ),
                max_camera_observation_attempts_per_candidate=(
                    args.max_camera_observation_attempts_per_candidate
                ),
            ),
            CandidateApproachEffects(
                read_current_pose=lambda: read_current_pose2d_from_amcl(
                    namespace=profile.namespace,
                    amcl_topic=profile.amcl_topic,
                    map_frame=profile.map_frame,
                    timeout_sec=STATIONARY_AMCL_TIMEOUT_SEC,
                    max_age_sec=2.0,
                ),
                admit_planning_frame=lambda evidence_path: (
                    _admit_candidate_planning_frame(
                        runtime,
                        session_root,
                        evidence_path=evidence_path,
                    )
                ),
                run_motion_leg=lambda request: _run_candidate_motion_leg(
                    profile=profile,
                    request=request,
                ),
                admit_startup_localization=lambda evidence_path: (
                    _admit_preplanning_localization(
                        runtime,
                        session_root,
                        evidence_path=evidence_path,
                    )
                ),
                run_startup_reseal_motion_leg=(
                    lambda request, attempt: (
                        _run_candidate_startup_reseal_motion_leg(
                            profile=profile,
                            request=request,
                            attempt=attempt,
                            startup_reseal_motion_authorization_json=(
                                startup_reseal_motion_authorization_json
                            ),
                            max_startup_reseals_per_leg=(
                                args.max_startup_reseals_per_leg
                            ),
                        )
                    )
                ),
                admit_runtime_localization=lambda evidence_path: (
                    _admit_preplanning_localization(
                        runtime,
                        session_root,
                        evidence_path=evidence_path,
                    )
                ),
                run_runtime_localization_reseal_motion_leg=(
                    lambda request, attempt: (
                        _run_candidate_runtime_localization_reseal_motion_leg(
                            profile=profile,
                            request=request,
                            attempt=attempt,
                            mission_motion_authorization_json=(
                                mission_motion_authorization_json
                            ),
                            max_runtime_reseals_per_leg=(
                                args.max_runtime_localization_reseals_per_leg
                            ),
                        )
                    )
                ),
                event_sink=lambda path, payload: _append_jsonl(
                    path,
                    dict(payload),
                ),
                capture_observation=(
                    lambda request: _capture_candidate_observation(
                        profile=profile,
                        args=args,
                        request=request,
                    )
                ),
            ),
        )
        completed_stand_model = load_measured_physical_stand_model(
            args.stand_model_profile
        )
        result = _completed_camera_mission_summary(
            run_mode=args.run_mode,
            session_id=args.session_id,
            snapshot_path=snapshot_path,
            snapshot_sha256=snapshot_sha256,
            survey_root=survey_root,
            stand_model_profile=args.stand_model_profile,
            stand_model_profile_sha256=completed_stand_model.sha256,
            candidate_population_admission_path=(
                coverage_candidate_admission_path
            ),
            candidate_population_admission_sha256=(
                coverage_candidate_admission_sha256
            ),
            candidate_phase_fields=(
                candidate_phase.to_mission_summary_fields()
            ),
            exact_two_coverage_summary=exact_two_camera_summary,
            exact_two_camera_handoff_path=exact_two_camera_handoff_path,
            exact_two_camera_handoff_sha256=(
                exact_two_camera_handoff_sha256
            ),
        )
        _write_json(session_root / "mission_summary.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
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
                build_failed_closed_mission_summary(
                    run_mode=args.run_mode,
                    error=exc,
                ),
            )
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
