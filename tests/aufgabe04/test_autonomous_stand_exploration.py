import json
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.aufgabe04.real_robot.autonomous_runner import (
    runtime as autonomous_wrapper,
)
from scripts.aufgabe04.navigation.missions.plan_stand_coverage_survey import (
    main as plan_coverage,
)
from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.foundation.models import GridCell, Pose2D
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
    MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
    MISSION_LEG_RUN_CONFIRMATION,
    ROUTINE_MISSION_LEG_KINDS,
    MissionLegKind,
    MissionLegMotionAuthorization,
    load_mission_leg_motion_permit,
    write_mission_leg_motion_authorization,
)
from scripts.aufgabe04.navigation.localization.ros_preflight import (
    RosObservation,
    RosPreflightResult,
)
from scripts.aufgabe04.navigation.localization.runtime_localization_reseal import (
    evaluate_runtime_localization_reseal,
)
from scripts.aufgabe04.navigation.execution.runtime_motion_authorization import (
    MISSION_MOTION_AUTHORIZATION_SCOPE,
    MISSION_RUN_CONFIRMATION,
    RUNTIME_LOCALIZATION_RESEAL_RECOVERY_KIND,
    MissionMotionAuthorization,
    load_runtime_localization_motion_permit,
    write_mission_motion_authorization,
)
from scripts.aufgabe04.navigation.entrypoints.run_single_station_segment import (
    main as run_segment,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime import (
    STATIC_PHYSICAL_ROUTE_KINDS,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    STATUS_PENDING_CAMERA,
    CoverageSurveyConfig,
    CoverageSurveyPlan,
    SurveyCandidate,
    SurveyViewpoint,
    load_coverage_survey_plan,
    load_stand_survey_registry,
    stand_survey_registry_sha256,
    write_stand_survey_registry,
)
from scripts.aufgabe04.navigation.coverage.stand_discovery_route import (
    STAND_DISCOVERY_ROUTE_KIND,
    seal_stand_discovery_route,
    validate_stand_discovery_route_binding,
)
from scripts.aufgabe04.navigation.execution.startup_reseal_motion_authorization import (
    STARTUP_RESEAL_MOTION_AUTHORIZATION_SCOPE,
    STARTUP_RESEAL_RECOVERY_KIND,
    STARTUP_RESEAL_RUN_CONFIRMATION,
    StartupResealMotionAuthorization,
    load_startup_reseal_motion_permit,
    write_startup_reseal_motion_authorization,
)
from scripts.aufgabe04.navigation.coverage.transient_overlay_resume_state import (
    TransientOverlayResumeState,
)
from scripts.aufgabe04.navigation.planning.waypoint_csv import load_route_leg
from scripts.aufgabe04.perception.stand_axis.model_profile import (
    stand_model_from_payload,
    write_stand_model,
)
from scripts.aufgabe04.real_robot.coverage_leg import (
    replanning as autonomous_coverage_replanning,
)
from scripts.aufgabe04.real_robot.coverage_leg.replanning import (
    coverage_reseal_suffix,
    is_resealable_startup_mismatch,
    is_runtime_localization_reseal_required,
    replan_runtime_localization_source,
    replan_startup_source,
)
from scripts.aufgabe04.real_robot.autonomous_runner.runtime import (
    MotionLegOutcome,
    MissionLegPermitContext,
    RuntimeLocalizationPermitContext,
    StartupResealPermitContext,
    _admit_candidate_planning_frame,
    _admit_preplanning_localization,
    _capture_lidar_epoch,
    _execute_coverage_leg_with_replans,
    _motion_outcome_from_log,
    _runner_command,
    _run_candidate_runtime_localization_reseal_motion_leg,
    _run_motion_leg,
    build_parser,
    candidate_snapshot_from_registry,
)
from scripts.aufgabe04.real_robot.readiness.startup_reseal import (
    write_startup_reseal_permit_summary,
)
from scripts.aufgabe04.real_robot.candidate import (
    approach as autonomous_candidate_approach,
)
from scripts.aufgabe04.real_robot.candidate.approach import (
    CandidateApproachEffects,
    CandidateObservation,
    CameraCandidateInitialSelection,
    CandidateMotionLegRequest,
    bounded_approach_offsets,
    opposite_face_normal,
    plan_candidate_preapproach,
)
from scripts.aufgabe04.real_robot.candidate.observation_deferral import (
    CandidateObservationUnavailableError,
)
from scripts.aufgabe04.real_robot.candidate.runtime_recovery import (
    CandidateRuntimeRecoveryAttempt,
)
from scripts.aufgabe04.real_robot.candidate.startup_recovery import (
    CandidateRoutineIdentity,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateGeometry,
    CandidateSource,
    FrozenCandidate,
    new_candidate_snapshot,
    write_candidate_snapshot,
)


MAP = Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml")


def _append_child_event(
    session_root: Path,
    run_id: str,
    event: dict[str, object],
) -> None:
    event_path = session_root / "run_events" / f"{run_id}.jsonl"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def _materialize_dry_child_evidence(
    command: list[str],
    session_root: Path,
    run_id: str,
) -> None:
    for flag in (
        "--preflight-json",
        "--odom-execution-certificate-json",
        "--uncertainty-budget-json",
    ):
        if flag not in command:
            continue
        artifact = Path(command[command.index(flag) + 1])
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("{}\n", encoding="utf-8")
    _append_child_event(
        session_root,
        run_id,
        {
            "event": "dry_run_completed",
            "run_id": run_id,
            "status": "dry_run_ok",
            "motion_published": False,
        },
    )


def _runtime_localization_stop_details():
    return {
        "reason": "global localization consistency requires zero and reseal",
        "fault_code": "localization_reseal_required",
        "source": "global_consistency_monitor",
        "execution_pose_owner": "odom",
        "global_consistency_monitor": "amcl",
        "monitor_action": "FORCE_ZERO_RESEAL",
        "fail_closed": True,
        "continuity": {
            "accepted": False,
            "requires_zero_cycle": True,
            "requires_reseal": True,
            "decision": "force_zero_reseal",
            "reason": "map_from_odom_yaw_drift",
            "fail_closed": True,
        },
    }


def _stationary_localization_payload(
    x_m: float,
    y_m: float,
    yaw_rad: float,
) -> dict[str, object]:
    pose = {"x_m": x_m, "y_m": y_m, "yaw_rad": yaw_rad}
    return {
        "ok": True,
        "failures": [],
        "observations": [
            {
                "name": "stationary AMCL stability",
                "ok": True,
                "detail": "samples=2/2",
                "data": {
                    "sample_count": 2,
                    "required_sample_count": 2,
                    "service_request_count": 2,
                    "position_covariance_complete": True,
                    "yaw_covariance_complete": True,
                },
            }
        ],
        "runtime_config": {
            "localization_source": "amcl",
            "use_sim_time": False,
        },
        "route_pose": {
            "frame_id": "map",
            "child_frame_id": "base_footprint",
            **pose,
        },
        "odom_pose": None,
        "map_from_odom": None,
        "stationary_amcl_samples": [
            {**pose, "covariance": [0.0] * 36},
            {**pose, "covariance": [0.0] * 36},
        ],
        "stationary_map_from_odom_samples": [],
    }


class AutonomousStandExplorationTest(unittest.TestCase):
    def test_exact_two_checkpoint_selects_lidar_only_completion_policy(self):
        exact_plan = SimpleNamespace(
            config=SimpleNamespace(exact_inspection_point_count=2)
        )
        self.assertEqual(
            autonomous_wrapper._plan_exact_inspection_point_count(
                exact_plan,
                requested=None,
            ),
            2,
        )
        self.assertIs(
            autonomous_wrapper._coverage_completion_policy(
                autonomous_wrapper.AutonomousRunMode.EXECUTE_COVERAGE_CHECKPOINT,
                exact_inspection_point_count=2,
            ),
            autonomous_wrapper.CoverageCompletionPolicy.EXACT_TWO_LIDAR_CHECKPOINT,
        )
        self.assertIs(
            autonomous_wrapper._coverage_completion_policy(
                autonomous_wrapper.AutonomousRunMode.EXECUTE_FULL,
                exact_inspection_point_count=None,
            ),
            autonomous_wrapper.CoverageCompletionPolicy.CAMERA_READY,
        )
        self.assertIs(
            autonomous_wrapper._coverage_completion_policy(
                autonomous_wrapper.AutonomousRunMode.RESUME_NEXT_COVERAGE_LEG,
                exact_inspection_point_count=2,
            ),
            autonomous_wrapper.CoverageCompletionPolicy.EXACT_TWO_LIDAR_CHECKPOINT,
        )

    def test_exact_two_camera_mode_selects_camera_validation_policy(self):
        self.assertIs(
            autonomous_wrapper._coverage_completion_policy(
                autonomous_wrapper.AutonomousRunMode.EXECUTE_EXACT_TWO_CAMERA,
                exact_inspection_point_count=2,
            ),
            autonomous_wrapper.CoverageCompletionPolicy.EXACT_TWO_CAMERA_VALIDATION,
        )
        with self.assertRaisesRegex(ValueError, "requires exactly two"):
            autonomous_wrapper._coverage_completion_policy(
                autonomous_wrapper.AutonomousRunMode.EXECUTE_EXACT_TWO_CAMERA,
                exact_inspection_point_count=None,
            )

    def test_exact_two_camera_final_summary_separates_phase_completion(self):
        summary = autonomous_wrapper._completed_camera_mission_summary(
            run_mode="execute-exact-two-camera",
            session_id="exact_two_camera_001",
            snapshot_path=Path("session/candidate_snapshot.json"),
            snapshot_sha256="a" * 64,
            survey_root=Path("session/coverage"),
            stand_model_profile=Path("profiles/measured_stand.json"),
            stand_model_profile_sha256="9" * 64,
            candidate_population_admission_path=Path(
                "session/coverage_exact_two_camera_admission.json"
            ),
            candidate_population_admission_sha256="b" * 64,
            candidate_phase_fields={
                "stand_count": 5,
                "stand_facing_catalog": "session/catalog.json",
                "stand_facing_catalog_sha256": "c" * 64,
                "station_identity_registry": "session/identities.json",
                "station_identity_registry_sha256": "d" * 64,
                "motion_authorized": False,
            },
            exact_two_coverage_summary={
                "lidar_checkpoint_admission": "session/lidar.json",
                "lidar_checkpoint_admission_sha256": "e" * 64,
                "camera_validation_admission": "session/camera.json",
                "camera_validation_admission_sha256": "f" * 64,
                "camera_validation_candidate_uids": [
                    f"survey_candidate_{index:04d}" for index in range(1, 6)
                ],
                "active_lidar_registry_candidate_count": 6,
                "lidar_static_map_admitted_candidate_count": 5,
                "lidar_boundary_provisional_candidate_count": 1,
                "lidar_population_retained_candidate_count": 6,
                "camera_seed_candidate_count": 5,
                "camera_seed_selection_mode": (
                    "strict_exact_boundary_audit_only"
                ),
                "camera_seed_candidate_uids": [
                    f"survey_candidate_{index:04d}" for index in range(1, 6)
                ],
                "camera_seed_boundary_fill_candidate_uids": [],
                "camera_seed_boundary_audit_only_candidate_uids": [
                    "survey_candidate_0006"
                ],
                "camera_seed_excluded_candidate_uids": [
                    "survey_candidate_0006"
                ],
                "multi_view_candidate_uids": [
                    "survey_candidate_0001",
                    "survey_candidate_0002",
                ],
                "single_view_requires_camera_validation_candidate_uids": [
                    "survey_candidate_0003",
                    "survey_candidate_0004",
                    "survey_candidate_0005",
                ],
            },
            exact_two_camera_handoff_path=Path("session/handoff.json"),
            exact_two_camera_handoff_sha256="1" * 64,
        )

        self.assertTrue(summary["lidar_coverage_complete"])
        self.assertTrue(summary["lidar_checkpoint_complete"])
        self.assertTrue(summary["camera_validation_population_ready"])
        self.assertTrue(summary["candidate_snapshot_ready"])
        self.assertTrue(summary["camera_approach_executed"])
        self.assertTrue(summary["camera_validation_complete"])
        self.assertTrue(summary["camera_exploration_complete"])
        self.assertTrue(summary["exploration_complete"])
        self.assertEqual(summary["camera_estimator_mode"], "metric_model_only")
        self.assertEqual(summary["stand_model_profile_sha256"], "9" * 64)
        self.assertEqual(summary["camera_seed_candidate_count"], 5)
        self.assertEqual(summary["active_lidar_registry_candidate_count"], 6)
        self.assertEqual(
            summary["camera_seed_boundary_audit_only_candidate_uids"],
            ["survey_candidate_0006"],
        )
        self.assertEqual(
            len(
                summary[
                    "single_view_requires_camera_validation_candidate_uids"
                ]
            ),
            3,
        )
        self.assertFalse(summary["camera_approach_authorized"])
        self.assertFalse(summary["motion_authorized"])

    def _run_exact_two_camera_wrapper_retry_fixture(
        self,
        root: Path,
        *,
        max_attempts: int,
        candidate_a_failures: int,
        reject_candidate_sensor_timing: bool = False,
    ) -> dict[str, object]:
        """Run the real outer wrapper with ROS/motion replaced by typed effects."""

        session_id = "stand_explore_exact2_camera_wrapper_retry"
        output_root = root / "runs"
        session_root = output_root / session_id
        map_sha256 = "a" * 64
        fixture_paths = {
            name: root / name
            for name in (
                "robot.json",
                "camera.json",
                "site.json",
                "map.yaml",
                "stand_model.json",
            )
        }
        for name, path in fixture_paths.items():
            if name == "stand_model.json":
                continue
            path.write_text("{}\n", encoding="utf-8")
        write_stand_model(
            fixture_paths["stand_model.json"],
            stand_model_from_payload(
                {
                    "schema_version": 2,
                    "profile_id": "physical_test_v2",
                    "environment": "physical",
                    "measurement_status": "measured",
                    "head_width_m": 0.078,
                    "head_height_m": 0.078,
                    "head_depth_m": 0.006,
                    "qr_symbol_width_m": 0.062,
                    "qr_symbol_height_m": 0.062,
                    "qr_panel_width_m": 0.071,
                    "qr_panel_height_m": 0.071,
                    "qr_center_x_m": 0.0,
                    "qr_center_y_m": 0.0,
                    "head_top_height_m": 0.210,
                    "base_width_m": 0.153,
                    "base_depth_m": 0.153,
                    "tolerance_m": 0.002,
                    "source": "direct test metrology",
                }
            ),
        )

        def candidate(uid: str, x_m: float) -> FrozenCandidate:
            return FrozenCandidate(
                candidate_uid=uid,
                geometry=CandidateGeometry(
                    x_m=x_m,
                    y_m=0.0,
                    radius_m=0.06,
                    uncertainty_m=0.02,
                    keepout_radius_m=0.31,
                ),
                source=CandidateSource(
                    source_kind="lidar/stand_coverage_survey",
                    source_artifact_sha256="b" * 64,
                    detector_config_sha256="c" * 64,
                    observation_ids=(f"observation_{uid}",),
                ),
                confidence=0.9,
                hit_count=4,
                first_seen_sec=1.0,
                last_seen_sec=2.0,
            )

        candidates = (
            candidate("candidate_a", 0.2),
            candidate("candidate_b", 0.4),
            candidate("candidate_c", 0.6),
        )
        snapshot = new_candidate_snapshot(
            snapshot_id="exact_two_wrapper_snapshot",
            created_unix_sec=3.0,
            planning_frame="map",
            map_bundle_sha256=map_sha256,
            candidates=candidates,
        )
        snapshot_path = root / "candidate_snapshot.json"
        snapshot_sha256 = write_candidate_snapshot(snapshot_path, snapshot)
        first_cell = GridCell(0, 0)
        second_cell = GridCell(1, 0)
        plan = CoverageSurveyPlan(
            schema_version=1,
            survey_id=session_id,
            planning_frame="map",
            map_bundle_sha256=map_sha256,
            arena_bounds=ArenaBounds(),
            config=CoverageSurveyConfig(
                lane_count=1,
                expected_stand_count=len(candidates),
                exact_inspection_point_count=2,
            ),
            viewpoints=(
                SurveyViewpoint(
                    viewpoint_id="survey_vp_001",
                    pose=Pose2D(0.0, 0.0, 0.0),
                    cell=first_cell,
                    visible_cells=(first_cell, second_cell),
                ),
                SurveyViewpoint(
                    viewpoint_id="survey_vp_002",
                    pose=Pose2D(1.0, 0.0, math.pi),
                    cell=second_cell,
                    visible_cells=(first_cell, second_cell),
                ),
            ),
            surveyable_cells=(first_cell, second_cell),
            planned_covered_cells=(first_cell, second_cell),
            planned_coverage_ratio=1.0,
        )
        candidate_uids = list(snapshot.candidate_uids)

        class FakeExactTwoCameraReady:
            def __init__(self):
                self.candidate_snapshot = snapshot
                self.candidate_snapshot_path = snapshot_path
                self.candidate_snapshot_sha256 = snapshot_sha256
                self.camera_validation_admission_path = (
                    root / "camera_validation_admission.json"
                )
                self.camera_validation_admission_sha256 = "d" * 64
                self.camera_handoff_path = root / "camera_handoff.json"
                self.camera_handoff_sha256 = "e" * 64

            def to_mission_summary(self):
                return {
                    "lidar_checkpoint_admission": str(
                        root / "lidar_checkpoint_admission.json"
                    ),
                    "lidar_checkpoint_admission_sha256": "f" * 64,
                    "camera_validation_admission": str(
                        self.camera_validation_admission_path
                    ),
                    "camera_validation_admission_sha256": (
                        self.camera_validation_admission_sha256
                    ),
                    "camera_validation_candidate_uids": candidate_uids,
                    "active_lidar_registry_candidate_count": 3,
                    "lidar_static_map_admitted_candidate_count": 3,
                    "lidar_boundary_provisional_candidate_count": 0,
                    "lidar_population_retained_candidate_count": 3,
                    "camera_seed_candidate_count": 3,
                    "camera_seed_selection_mode": "strict_exact",
                    "camera_seed_candidate_uids": candidate_uids,
                    "camera_seed_boundary_fill_candidate_uids": [],
                    "camera_seed_boundary_audit_only_candidate_uids": [],
                    "camera_seed_excluded_candidate_uids": [],
                    "multi_view_candidate_uids": candidate_uids[:2],
                    "single_view_requires_camera_validation_candidate_uids": (
                        candidate_uids[2:]
                    ),
                }

        coverage_phase = FakeExactTwoCameraReady()
        profile = SimpleNamespace(
            robot_id="turtlebot1",
            robot_radius_m=0.105,
            namespace="",
            amcl_topic="amcl_pose",
            map_frame="map",
            scan_origin_to_base_offset_m=0.05,
            max_linear_speed_mps=0.055,
            max_angular_speed_radps=0.18,
            resolved_runtime=lambda: SimpleNamespace(
                namespace="",
                cmd_vel_topic="/cmd_vel",
            ),
        )
        initial_admission = SimpleNamespace(
            result=SimpleNamespace(attempts=(object(),)),
            evidence_path=root / "initial_readiness.json",
            evidence_sha256="1" * 64,
        )
        capture_count = {uid: 0 for uid in candidate_uids}
        capture_order: list[str] = []
        motion_targets: list[str] = []
        planning_commands: list[tuple[str, ...]] = []
        sensor_timing_phases: list[tuple[str, bool]] = []

        def record_planning_command(command):
            planning_commands.append(tuple(command))
            return 0

        def admit_sensor_timing(_profile, evidence_path, **kwargs):
            sensor_timing_phases.append(
                (
                    kwargs["phase"],
                    kwargs["typed_run_already_issued"],
                )
            )
            if (
                reject_candidate_sensor_timing
                and kwargs["phase"]
                == "candidate_route_sensor_timing_readiness"
            ):
                raise RuntimeError("candidate sensor timing rejected")
            return Path(evidence_path), "6" * 64

        def capture_observation(*, profile, args, request):
            del profile, args
            uid = request.candidate.candidate_uid
            capture_count[uid] += 1
            capture_order.append(uid)
            if uid == "candidate_a" and capture_count[uid] <= candidate_a_failures:
                raise CandidateObservationUnavailableError(
                    candidate_uid=uid,
                    observation_attempt_index=request.attempt_index,
                    reason="candidate-local LiDAR association deadline",
                    process_evidence={"completion_kind": "deadline"},
                    status_evidence={"state": "lidar_target_mismatch"},
                )
            request.output_dir.mkdir(parents=True, exist_ok=True)
            recommendation = request.output_dir / "recommendation.json"
            recommendation.write_text("{}\n", encoding="utf-8")
            return CandidateObservation(recommendation, f"QR_{uid}", None)

        def run_candidate_motion(*, profile, request):
            autonomous_wrapper._admit_sensor_timing_readiness(
                profile,
                request.session_root
                / "preflight"
                / (
                    f"{request.run_id}_camera_lidar_timing_before_motion.json"
                ),
                phase=(
                    autonomous_wrapper.CANDIDATE_ROUTE_SENSOR_TIMING_PHASE
                ),
                typed_run_already_issued=True,
            )
            motion_targets.append(request.target_id)
            return MotionLegOutcome(
                run_id=request.run_id,
                status="completed",
                stop_reason="",
                stop_details={},
                motion_published=True,
                returncode=0,
                semantic_log_path=root / f"{request.run_id}.jsonl",
            )

        def candidate_effects_factory(**effects):
            def select(request):
                uid = sorted(request.unresolved)[0]
                return CameraCandidateInitialSelection(
                    candidate_uid=uid,
                    prepared_plan=None,
                    evidence={
                        "selected_candidate_uid": uid,
                        "motion_authorized": False,
                    },
                )

            return CandidateApproachEffects(
                read_current_pose=effects["read_current_pose"],
                run_motion_leg=effects["run_motion_leg"],
                capture_observation=effects["capture_observation"],
                plan_preapproach=lambda _request: {"route_csv": "route.csv"},
                select_initial_preapproach=select,
                validate_facing=lambda request: {
                    "candidate_uid": request.candidate.candidate_uid
                },
                commit_decision=lambda _request: None,
                event_sink=effects["event_sink"],
                clock=lambda: 10.0,
            )

        with (
            patch.multiple(
                autonomous_wrapper,
                load_real_robot_profile=lambda *_args, **_kwargs: profile,
                load_camera_calibration=lambda *_args, **_kwargs: SimpleNamespace(),
                validate_physical_site_contract=lambda *_args, **_kwargs: (
                    SimpleNamespace(
                        expected_stand_count=len(candidates),
                        physical_site_path=fixture_paths["site.json"],
                        map_yaml_path=fixture_paths["map.yaml"],
                        map_bundle=SimpleNamespace(bundle_sha256=map_sha256),
                    )
                ),
                _validate_inputs=lambda *_args, **_kwargs: None,
                _physical_clearance=lambda *_args, **_kwargs: {
                    "minimum_active_standoff_m": 0.20,
                    "minimum_static_inflation_m": 0.25,
                    "minimum_candidate_transit_radius_m": 0.31,
                },
                _admit_observation_tf_readiness=(
                    lambda _profile, evidence_path, **_kwargs: (
                        Path(evidence_path),
                        "2" * 64,
                    )
                ),
                _admit_sensor_timing_readiness=admit_sensor_timing,
                _admit_preplanning_localization=lambda *_args, **_kwargs: Pose2D(
                    0.0, 0.0, 0.0
                ),
                plan_stand_coverage_survey=record_planning_command,
                load_coverage_survey_plan=lambda *_args, **_kwargs: plan,
                admit_preauthorization_readiness=(
                    lambda *_args, **_kwargs: initial_admission
                ),
                write_mission_leg_motion_authorization=(
                    lambda *_args, **_kwargs: "3" * 64
                ),
                write_mission_motion_authorization=(
                    lambda *_args, **_kwargs: "4" * 64
                ),
                write_startup_reseal_motion_authorization=(
                    lambda *_args, **_kwargs: "5" * 64
                ),
                execute_coverage_mission=lambda *_args, **_kwargs: coverage_phase,
                CoverageExactTwoCameraReady=FakeExactTwoCameraReady,
                CandidateApproachEffects=candidate_effects_factory,
                read_current_pose2d_from_amcl=lambda *_args, **_kwargs: Pose2D(
                    0.0, 0.0, 0.0
                ),
                _run_candidate_motion_leg=run_candidate_motion,
                _capture_candidate_observation=capture_observation,
            ),
            patch.object(
                autonomous_candidate_approach,
                "validate_candidate_approach_handoff",
                return_value={uid: "coverage_admitted" for uid in candidate_uids},
            ),
            patch("builtins.input", return_value="RUN") as run_prompt,
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()),
        ):
            try:
                exit_code = autonomous_wrapper.main(
                    [
                        "--robot-profile",
                        str(fixture_paths["robot.json"]),
                        "--camera-calibration",
                        str(fixture_paths["camera.json"]),
                        "--stand-model-profile",
                        str(fixture_paths["stand_model.json"]),
                        "--physical-site",
                        str(fixture_paths["site.json"]),
                        "--map",
                        str(fixture_paths["map.yaml"]),
                        "--session-id",
                        session_id,
                        "--output-root",
                        str(output_root),
                        "--expected-stand-count",
                        str(len(candidates)),
                        "--coverage-leg-limit",
                        "2",
                        "--exact-inspection-point-count",
                        "2",
                        "--max-runtime-localization-reseals-per-leg",
                        "0",
                        "--max-camera-observation-attempts-per-candidate",
                        str(max_attempts),
                        "--localization-branch-proof-id",
                        "known_start",
                        "--run-mode",
                        "execute-exact-two-camera",
                    ]
                )
            except SystemExit as exc:
                exit_code = exc.code

        return {
            "exit_code": exit_code,
            "session_root": session_root,
            "capture_order": capture_order,
            "motion_targets": motion_targets,
            "planning_command": planning_commands[0],
            "run_prompt_count": run_prompt.call_count,
            "sensor_timing_phases": sensor_timing_phases,
            "failure": (
                json.loads(
                    (session_root / "mission_failure.json").read_text(
                        encoding="utf-8"
                    )
                )
                if (session_root / "mission_failure.json").is_file()
                else None
            ),
        }

    def test_exact_two_wrapper_defers_local_timeout_and_completes_after_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_exact_two_camera_wrapper_retry_fixture(
                Path(tmp),
                max_attempts=2,
                candidate_a_failures=1,
            )
            session_root = result["session_root"]
            self.assertEqual(result["exit_code"], 0, result["failure"])
            self.assertEqual(
                result["capture_order"],
                ["candidate_a", "candidate_b", "candidate_c", "candidate_a"],
            )
            self.assertEqual(result["motion_targets"], result["capture_order"])
            planning_command = result["planning_command"]
            self.assertEqual(
                planning_command[planning_command.index("--lane-count") + 1],
                "1",
            )
            self.assertEqual(
                planning_command[
                    planning_command.index("--exact-two-candidate-spacing-m") + 1
                ],
                "0.4",
            )
            self.assertEqual(
                planning_command[
                    planning_command.index(
                        "--minimum-exact-two-viewpoint-baseline-m"
                    )
                    + 1
                ],
                "1.0",
            )
            self.assertEqual(
                planning_command[
                    planning_command.index(
                        "--startup-route-selection-robot-radius-m"
                    )
                    + 1
                ],
                "0.105",
            )
            self.assertEqual(
                planning_command[
                    planning_command.index(
                        "--startup-route-selection-tracking-tube-radius-m"
                    )
                    + 1
                ],
                "0.03",
            )
            self.assertEqual(
                planning_command[
                    planning_command.index(
                        "--startup-route-selection-sigma-multiplier"
                    )
                    + 1
                ],
                "2.0",
            )
            self.assertEqual(result["run_prompt_count"], 1)
            self.assertEqual(
                result["sensor_timing_phases"],
                [
                    ("preauthorization_sensor_timing_readiness", False),
                    ("candidate_route_sensor_timing_readiness", True),
                    ("candidate_route_sensor_timing_readiness", True),
                    ("candidate_route_sensor_timing_readiness", True),
                    ("candidate_route_sensor_timing_readiness", True),
                ],
            )
            events = [
                json.loads(line)
                for line in (session_root / "candidate_selection.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertTrue(
                any(
                    event.get("event")
                    == "camera_candidate_observation_deferred"
                    for event in events
                )
            )
            self.assertTrue(
                any(
                    event.get("event")
                    == "camera_candidate_observation_retry_pass"
                    for event in events
                )
            )
            summary = json.loads(
                (session_root / "mission_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["stand_count"], 3)
            self.assertTrue(summary["camera_validation_complete"])
            self.assertTrue(summary["exploration_complete"])
            self.assertFalse(summary["motion_authorized"])

    def test_exact_two_wrapper_fails_only_after_remaining_candidates_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_exact_two_camera_wrapper_retry_fixture(
                Path(tmp),
                max_attempts=1,
                candidate_a_failures=99,
            )
            session_root = result["session_root"]
            self.assertEqual(result["exit_code"], 2)
            self.assertEqual(
                result["capture_order"],
                ["candidate_a", "candidate_b", "candidate_c"],
            )
            self.assertEqual(result["motion_targets"], result["capture_order"])
            self.assertEqual(
                result["sensor_timing_phases"],
                [
                    ("preauthorization_sensor_timing_readiness", False),
                    ("candidate_route_sensor_timing_readiness", True),
                    ("candidate_route_sensor_timing_readiness", True),
                    ("candidate_route_sensor_timing_readiness", True),
                ],
            )
            failure = json.loads(
                (session_root / "mission_failure.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(failure["status"], "failed_closed")
            self.assertEqual(
                failure["failure_phase"], "candidate_approach_incomplete"
            )
            self.assertEqual(
                failure["resolved_candidate_uids"],
                ["candidate_b", "candidate_c"],
            )
            self.assertEqual(
                failure["unresolved_candidate_uids"], ["candidate_a"]
            )
            self.assertEqual(failure["max_candidate_observation_attempts"], 1)
            self.assertFalse(failure["motion_continues_authorized"])
            self.assertFalse((session_root / "mission_summary.json").exists())

    def test_candidate_timing_rejection_starts_no_candidate_motion_or_observer(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_exact_two_camera_wrapper_retry_fixture(
                Path(tmp),
                max_attempts=1,
                candidate_a_failures=0,
                reject_candidate_sensor_timing=True,
            )

            self.assertEqual(result["exit_code"], 2)
            self.assertEqual(result["motion_targets"], [])
            self.assertEqual(result["capture_order"], [])
            self.assertEqual(
                result["sensor_timing_phases"],
                [
                    ("preauthorization_sensor_timing_readiness", False),
                    ("candidate_route_sensor_timing_readiness", True),
                ],
            )
            self.assertIn(
                "candidate sensor timing rejected",
                result["failure"]["reason"],
            )

    def test_exact_two_final_summary_rejects_camera_seed_uid_mismatch(self):
        exact_summary = {
            "lidar_checkpoint_admission": "session/lidar.json",
            "lidar_checkpoint_admission_sha256": "e" * 64,
            "camera_validation_admission": "session/camera.json",
            "camera_validation_admission_sha256": "f" * 64,
            "camera_validation_candidate_uids": ["candidate_1"],
            "active_lidar_registry_candidate_count": 2,
            "lidar_static_map_admitted_candidate_count": 1,
            "lidar_boundary_provisional_candidate_count": 1,
            "lidar_population_retained_candidate_count": 2,
            "camera_seed_candidate_count": 1,
            "camera_seed_selection_mode": (
                "strict_exact_boundary_audit_only"
            ),
            "camera_seed_candidate_uids": ["candidate_2"],
            "camera_seed_boundary_fill_candidate_uids": [],
            "camera_seed_boundary_audit_only_candidate_uids": ["candidate_2"],
            "camera_seed_excluded_candidate_uids": ["candidate_2"],
            "multi_view_candidate_uids": ["candidate_1"],
            "single_view_requires_camera_validation_candidate_uids": [],
        }

        with self.assertRaisesRegex(ValueError, "differ from camera seeds"):
            autonomous_wrapper._completed_camera_mission_summary(
                run_mode="execute-exact-two-camera",
                session_id="exact_two_camera_mismatch",
                snapshot_path=Path("session/candidate_snapshot.json"),
                snapshot_sha256="a" * 64,
                survey_root=Path("session/coverage"),
                stand_model_profile=Path("profiles/measured_stand.json"),
                stand_model_profile_sha256="9" * 64,
                candidate_population_admission_path=Path(
                    "session/camera_admission.json"
                ),
                candidate_population_admission_sha256="b" * 64,
                candidate_phase_fields={
                    "stand_count": 1,
                    "motion_authorized": False,
                },
                exact_two_coverage_summary=exact_summary,
                exact_two_camera_handoff_path=Path("session/handoff.json"),
                exact_two_camera_handoff_sha256="1" * 64,
            )
    @staticmethod
    def _profile():
        return SimpleNamespace(
            robot_id="turtlebot1",
            namespace="",
            scan_topic="scan",
            odom_topic="odom",
            cmd_vel_topic="cmd_vel",
            amcl_topic="amcl_pose",
            map_frame="map",
            odom_frame="odom",
            base_frame="base_footprint",
            localization_source="amcl",
            max_linear_speed_mps=0.055,
            max_angular_speed_radps=0.18,
            robot_radius_m=0.105,
        )

    def test_candidate_motion_adapter_preserves_exact_permit_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = CandidateMotionLegRequest(
                sealed={"route_csv": str(root / "route.csv")},
                run_id="mission_candidate_003_opposite",
                session_root=root,
                candidate_snapshot_path=root / "candidate_snapshot.json",
                uncertainty_map_yaml=MAP,
                uncertainty_sigma_multiplier=2.0,
                localization_branch_proof_id="known_start",
                mission_authorization_json=root / "mission_authorization.json",
                session_id="mission",
                semantic_map_id="arena_1p898x3p9_auto",
                mission_leg_kind=MissionLegKind.OPPOSITE_FACE,
                mission_leg_index=3,
                target_id="candidate_7",
                permit_json_path=(root / "permit.json").absolute(),
            )
            expected = object()
            with patch.object(
                autonomous_wrapper,
                "_run_motion_leg",
                return_value=expected,
            ) as run:
                actual = autonomous_wrapper._run_candidate_motion_leg(
                    profile=self._profile(),
                    request=request,
                )

            self.assertIs(actual, expected)
            kwargs = run.call_args.kwargs
            self.assertTrue(kwargs["execute"])
            self.assertEqual(
                kwargs["sensor_timing_readiness_phase"],
                autonomous_wrapper.CANDIDATE_ROUTE_SENSOR_TIMING_PHASE,
            )
            self.assertEqual(kwargs["run_id"], request.run_id)
            self.assertEqual(
                kwargs["candidate_snapshot"],
                request.candidate_snapshot_path,
            )
            permit_context = kwargs["mission_leg_permit_context"]
            self.assertIsInstance(permit_context, MissionLegPermitContext)
            self.assertEqual(
                permit_context.mission_leg_kind,
                MissionLegKind.OPPOSITE_FACE,
            )
            self.assertEqual(permit_context.mission_leg_index, 3)
            self.assertEqual(permit_context.target_id, "candidate_7")
            self.assertEqual(
                permit_context.permit_json_path,
                request.permit_json_path,
            )

    def test_motion_outcome_preserves_preflight_failure_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            semantic_log = Path(tmp) / "events.jsonl"
            semantic_log.write_text(
                json.dumps(
                    {
                        "event": "preflight_failed",
                        "run_id": "coverage_000",
                        "failures": [
                            "stationary AMCL stability: position_std=0.0940m/0.0150m"
                        ],
                        "observations": [],
                        "runtime_config": {"localization_source": "amcl"},
                    }
                )
                + "\n"
            )

            outcome = _motion_outcome_from_log(
                semantic_log,
                run_id="coverage_000",
                returncode=1,
            )

        self.assertEqual(outcome.status, "preflight_failed")
        self.assertIn("position_std=0.0940m", outcome.stop_reason)
        self.assertFalse(outcome.motion_published)
        self.assertTrue(outcome.stop_details["fail_closed"])

    def test_motion_outcome_rejects_non_boolean_motion_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            semantic_log = Path(tmp) / "events.jsonl"
            semantic_log.write_text(
                json.dumps(
                    {
                        "event": "safety_stop",
                        "run_id": "coverage_000",
                        "status": "stopped",
                        "stop_reason": "zero and reseal",
                        "stop_details": _runtime_localization_stop_details(),
                        "motion_published": "false",
                    }
                )
                + "\n"
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "non-boolean motion_published",
            ):
                _motion_outcome_from_log(
                    semantic_log,
                    run_id="coverage_000",
                    returncode=1,
                )

    def test_motion_outcome_rejects_preflight_claiming_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            semantic_log = Path(tmp) / "events.jsonl"
            semantic_log.write_text(
                json.dumps(
                    {
                        "event": "preflight_failed",
                        "run_id": "coverage_000",
                        "failures": ["route uncertainty budget exhausted"],
                        "observations": [],
                        "runtime_config": {},
                        "motion_published": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "preflight_failed event must carry false motion_published",
            ):
                _motion_outcome_from_log(
                    semantic_log,
                    run_id="coverage_000",
                    returncode=1,
                )

    def test_motion_outcome_uses_only_the_current_invocation_log_slice(self):
        with tempfile.TemporaryDirectory() as tmp:
            semantic_log = Path(tmp) / "events.jsonl"
            stale = {
                "event": "safety_stop",
                "run_id": "coverage_000",
                "status": "stopped",
                "stop_reason": "stale stop",
                "stop_details": {"fail_closed": True},
                "motion_published": True,
            }
            current = {
                "event": "motion_completed",
                "run_id": "coverage_000",
                "status": "completed",
                "stop_reason": "",
                "stop_details": {},
                "motion_published": True,
            }
            stale_line = json.dumps(stale) + "\n"
            semantic_log.write_text(
                stale_line + json.dumps(current) + "\n",
                encoding="utf-8",
            )
            offset = len(stale_line.encode("utf-8"))

            outcome = _motion_outcome_from_log(
                semantic_log,
                run_id="coverage_000",
                returncode=0,
                start_offset=offset,
            )

        self.assertEqual(outcome.status, "completed")
        self.assertEqual(outcome.semantic_log_start_offset, offset)

    def test_motion_leg_refuses_a_preexisting_semantic_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "reused_child"
            semantic_log = root / "run_events" / f"{run_id}.jsonl"
            semantic_log.parent.mkdir(parents=True)
            semantic_log.write_text("{}\n", encoding="utf-8")
            sealed = {
                "route_csv": str(root / "route.csv"),
                "diagnostics_json": str(root / "diagnostics.json"),
                "route_certificate_json": str(root / "certificate.json"),
            }
            with patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "subprocess.run"
            ) as run, self.assertRaisesRegex(
                RuntimeError,
                "refusing to reuse an existing motion semantic log",
            ):
                _run_motion_leg(
                    profile=self._profile(),
                    sealed=sealed,
                    run_id=run_id,
                    session_root=root,
                    execute=True,
                )

            run.assert_not_called()

    def test_runtime_localization_wrapper_requires_complete_contract(self):
        complete = MotionLegOutcome(
            run_id="coverage_000",
            status="stopped",
            stop_reason="zero and reseal",
            stop_details=_runtime_localization_stop_details(),
            motion_published=True,
            returncode=1,
            semantic_log_path=Path("events.jsonl"),
        )
        self.assertTrue(is_runtime_localization_reseal_required(complete))

        for field in (
            "monitor_action",
            "continuity",
        ):
            with self.subTest(field=field):
                details = _runtime_localization_stop_details()
                details.pop(field)
                incomplete = replace(complete, stop_details=details)
                self.assertFalse(
                    is_runtime_localization_reseal_required(incomplete)
                )
        self.assertFalse(
            is_runtime_localization_reseal_required(
                replace(complete, motion_published=False)
            )
        )

    def test_coverage_reseal_suffix_keeps_retry_identities_distinct(self):
        self.assertEqual(
            coverage_reseal_suffix(
                startup_reseal_index=0,
                runtime_localization_reseal_index=0,
            ),
            "",
        )
        self.assertEqual(
            coverage_reseal_suffix(
                startup_reseal_index=2,
                runtime_localization_reseal_index=1,
            ),
            "_startup_reseal_002_runtime_localization_reseal_001",
        )

    def test_preplanning_localization_binds_start_to_admitted_pose(self):
        runtime = SimpleNamespace(namespace="")
        preflight = RosPreflightResult(
            ok=True,
            failures=[],
            observations=[],
            runtime_config={"localization_source": "amcl"},
            route_pose={
                "frame_id": "map",
                "child_frame_id": "base_footprint",
                "x_m": -0.4,
                "y_m": -0.6,
                "yaw_rad": 1.7,
            },
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
            "run_ros_preflight",
            return_value=preflight,
        ) as run_preflight:
            root = Path(tmp)
            start = _admit_preplanning_localization(runtime, root)
            evidence = json.loads(
                (root / "preflight/preplanning_localization.json").read_text()
            )

        self.assertEqual(start, Pose2D(-0.4, -0.6, 1.7))
        self.assertTrue(evidence["ok"])
        self.assertEqual(
            run_preflight.call_args.kwargs[
                "max_stationary_amcl_position_std_m"
            ],
            0.30,
        )
        self.assertEqual(
            run_preflight.call_args.kwargs[
                "max_stationary_amcl_position_spread_m"
            ],
            0.015,
        )
        self.assertEqual(
            run_preflight.call_args.kwargs[
                "max_localization_tf_future_sec"
            ],
            1.1,
        )
        self.assertNotIn(
            "preflight_requirements",
            run_preflight.call_args.kwargs,
        )
        self.assertNotIn(
            "execution_pose_owner",
            run_preflight.call_args.kwargs,
        )

    def test_candidate_planning_frame_binds_pose_and_map_from_odom(self):
        runtime = SimpleNamespace(
            namespace="",
            map_frame="map",
            odom_frame="odom",
        )
        preflight = RosPreflightResult(
            ok=True,
            failures=[],
            observations=[
                RosObservation(
                    "stationary map<-odom transform samples",
                    True,
                    "paired_samples=2/2",
                    {"required_pair_count": 2},
                ),
                RosObservation(
                    "tf map->odom",
                    True,
                    "age=0.010s",
                    {"target_frame": "map", "source_frame": "odom"},
                ),
            ],
            runtime_config={"localization_source": "amcl"},
            preflight_requirements={
                "stationary_map_from_odom_pairing_requested": True,
                "stationary_map_from_odom_pairing_required": True,
            },
            route_pose={
                "frame_id": "map",
                "child_frame_id": "base_footprint",
                "x_m": 1.2,
                "y_m": 0.3,
                "yaw_rad": 0.7,
            },
            map_from_odom={
                "target_frame": "map",
                "source_frame": "odom",
                "stamp_sec": 1.2e-8,
                "x_m": -1.638358757,
                "y_m": -0.437467937,
                "yaw_rad": -0.108542892,
            },
            stationary_map_from_odom_samples=[
                {
                    "source": "direct_dynamic_tf",
                    "target_frame": "map",
                    "source_frame": "odom",
                    "amcl_sample_index": 0,
                    "stamp_nanoseconds": 10,
                    "receipt_time_nanoseconds": 20,
                    "x_m": -1.638358757,
                    "y_m": -0.437467937,
                    "yaw_rad": -0.108542892,
                },
                {
                    "source": "direct_dynamic_tf",
                    "target_frame": "map",
                    "source_frame": "odom",
                    "amcl_sample_index": 1,
                    "stamp_nanoseconds": 11,
                    "receipt_time_nanoseconds": 21,
                    "x_m": -1.638358757,
                    "y_m": -0.437467937,
                    "yaw_rad": -0.108542892,
                },
            ],
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
            "run_ros_preflight",
            return_value=preflight,
        ) as run_preflight:
            admitted = _admit_candidate_planning_frame(
                runtime,
                Path(tmp),
            )
            candidate_evidence_written = (
                Path(tmp) / "preflight/candidate_planning_frame.json"
            ).is_file()

        self.assertEqual(admitted.current_pose, Pose2D(1.2, 0.3, 0.7))
        self.assertEqual(admitted.map_frame, "map")
        self.assertEqual(admitted.odom_frame, "odom")
        self.assertAlmostEqual(admitted.map_from_odom.x_m, -1.638358757)
        self.assertAlmostEqual(admitted.map_from_odom.yaw_rad, -0.108542892)
        self.assertTrue(candidate_evidence_written)
        requirements = run_preflight.call_args.kwargs[
            "preflight_requirements"
        ]
        self.assertTrue(
            requirements.require_stationary_map_from_odom_pairing
        )
        self.assertNotIn(
            "execution_pose_owner",
            run_preflight.call_args.kwargs,
        )

    def test_preplanning_localization_fails_before_route_planning(self):
        runtime = SimpleNamespace(namespace="")
        preflight = RosPreflightResult(
            ok=False,
            failures=["stationary AMCL stability: position_std=0.049m/0.015m"],
            observations=[],
            runtime_config={"localization_source": "amcl"},
            route_pose=None,
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
            "run_ros_preflight",
            return_value=preflight,
        ):
            root = Path(tmp)
            with self.assertRaisesRegex(
                RuntimeError,
                "preplanning localization admission failed",
            ):
                _admit_preplanning_localization(runtime, root)
            evidence = json.loads(
                (root / "preflight/preplanning_localization.json").read_text()
            )

        self.assertFalse(evidence["ok"])
        self.assertIn("position_std=0.049m", evidence["failures"][0])

    def _planned_center_corridor(self, root: Path) -> None:
        with redirect_stdout(StringIO()):
            status = plan_coverage(
                [
                    "--map",
                    str(MAP),
                    "--semantic-map-id",
                    "arena_1p898x3p9_auto",
                    "--planning-frame",
                    "map",
                    "--start-x",
                    "0",
                    "--start-y",
                    "0",
                    "--start-yaw",
                    "0",
                    "--survey-id",
                    "autonomous_test",
                    "--output-dir",
                    str(root),
                    "--lane-count",
                    "1",
                    "--stop-spacing-m",
                    "0.70",
                    "--expected-stand-count",
                    "1",
                ]
            )
        self.assertEqual(status, 0)

    def test_single_center_lane_is_plannable_and_meets_coverage_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "survey"
            self._planned_center_corridor(root)
            plan = load_coverage_survey_plan(root / "coverage_plan.json")
            self.assertEqual(plan.config.lane_count, 1)
            self.assertGreaterEqual(plan.planned_coverage_ratio, 0.95)
            self.assertTrue(
                all(abs(viewpoint.pose.y_m) <= 0.35 for viewpoint in plan.viewpoints)
            )

    def test_center_corridor_leg_is_sealed_as_a_certified_physical_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "survey"
            sealed_root = Path(tmp) / "sealed"
            self._planned_center_corridor(root)
            outputs = seal_stand_discovery_route(
                source_route_csv=root / "legs/leg_000_route.csv",
                source_diagnostics_json=root / "legs/leg_000_diagnostics.json",
                coverage_plan_path=root / "coverage_plan.json",
                output_dir=sealed_root,
            )
            leg = load_route_leg(Path(outputs["route_csv"]), 0)
            status = validate_stand_discovery_route_binding(
                Path(outputs["diagnostics_json"]),
                leg,
                coverage_plan_path=root / "coverage_plan.json",
            )
            self.assertTrue(status.ok, status.failures)
            self.assertEqual(leg.route_kind, STAND_DISCOVERY_ROUTE_KIND)
            self.assertFalse(leg.simulation_only)
            self.assertTrue(leg.raw_waypoints[-1].protected)
            self.assertIn(
                STAND_DISCOVERY_ROUTE_KIND,
                STATIC_PHYSICAL_ROUTE_KINDS,
            )
            diagnostics = json.loads(
                Path(outputs["diagnostics_json"]).read_text()
            )
            self.assertTrue(diagnostics["metadata"]["motion_authorized"])

            with patch(
                "scripts.aufgabe04.navigation.entrypoints.run_single_station_segment."
                "run_ros_preflight",
                return_value=RosPreflightResult(
                    ok=True,
                    failures=[],
                    observations=[],
                    runtime_config={},
                    route_pose={
                        "frame_id": "map",
                        "child_frame_id": "base_footprint",
                        "x_m": leg.raw_waypoints[0].pose.x_m,
                        "y_m": leg.raw_waypoints[0].pose.y_m,
                        "yaw_rad": 0.0,
                    },
                ),
            ), patch(
                "scripts.aufgabe04.navigation.entrypoints.run_single_station_segment."
                "run_simple_waypoint_follower"
            ) as follower, redirect_stdout(StringIO()):
                runner_status = run_segment(
                    [
                        "--route-csv",
                        outputs["route_csv"],
                        "--diagnostics-json",
                        outputs["diagnostics_json"],
                        "--route-certificate-json",
                        outputs["route_certificate_json"],
                        "--coverage-plan",
                        str(root / "coverage_plan.json"),
                        "--leg-index",
                        "0",
                        "--semantic-log",
                        str(Path(tmp) / "events.jsonl"),
                        "--results-csv",
                        str(Path(tmp) / "results.csv"),
                        "--preflight-json",
                        str(Path(tmp) / "preflight.json"),
                        "--dry-run",
                    ]
                )
            self.assertEqual(runner_status, 0)
            follower.assert_not_called()

    def test_center_corridor_rejects_nonfinal_heading_constraints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "survey"
            sealed_root = Path(tmp) / "sealed"
            self._planned_center_corridor(root)
            outputs = seal_stand_discovery_route(
                source_route_csv=root / "legs/leg_000_route.csv",
                source_diagnostics_json=root / "legs/leg_000_diagnostics.json",
                coverage_plan_path=root / "coverage_plan.json",
                output_dir=sealed_root,
            )
            leg = load_route_leg(Path(outputs["route_csv"]), 0)
            if len(leg.raw_waypoints) > 2:
                waypoint_index = 1
                raw_waypoints = leg.raw_waypoints
                intermediate = raw_waypoints[waypoint_index]
            else:
                waypoint_index = 1
                start = leg.raw_waypoints[0]
                final = leg.raw_waypoints[-1]
                intermediate = replace(
                    start,
                    pose=Pose2D(
                        (start.pose.x_m + final.pose.x_m) / 2.0,
                        (start.pose.y_m + final.pose.y_m) / 2.0,
                        float("nan"),
                    ),
                )
                raw_waypoints = (start, intermediate, final)
            malformed_waypoint = replace(
                intermediate,
                pose=Pose2D(
                    intermediate.pose.x_m,
                    intermediate.pose.y_m,
                    0.0,
                ),
            )
            malformed_leg = replace(
                leg,
                raw_waypoints=(
                    *raw_waypoints[:waypoint_index],
                    malformed_waypoint,
                    *raw_waypoints[waypoint_index + 1:],
                ),
            )

            status = validate_stand_discovery_route_binding(
                Path(outputs["diagnostics_json"]),
                malformed_leg,
                coverage_plan_path=root / "coverage_plan.json",
            )

        self.assertFalse(status.ok)
        self.assertIn(
            "stand discovery non-final waypoint yaw must be unconstrained",
            status.failures,
        )

    def test_discovery_dry_run_rejects_stale_start_before_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "survey"
            sealed_root = Path(tmp) / "sealed"
            self._planned_center_corridor(root)
            outputs = seal_stand_discovery_route(
                source_route_csv=root / "legs/leg_000_route.csv",
                source_diagnostics_json=root / "legs/leg_000_diagnostics.json",
                coverage_plan_path=root / "coverage_plan.json",
                output_dir=sealed_root,
            )
            leg = load_route_leg(Path(outputs["route_csv"]), 0)
            events = Path(tmp) / "events.jsonl"
            with patch(
                "scripts.aufgabe04.navigation.entrypoints.run_single_station_segment."
                "run_ros_preflight",
                return_value=RosPreflightResult(
                    ok=True,
                    failures=[],
                    observations=[],
                    runtime_config={},
                    route_pose={
                        "frame_id": "map",
                        "child_frame_id": "base_footprint",
                        "x_m": leg.raw_waypoints[0].pose.x_m - 0.10,
                        "y_m": leg.raw_waypoints[0].pose.y_m,
                        "yaw_rad": 0.0,
                    },
                ),
            ), patch(
                "scripts.aufgabe04.navigation.entrypoints.run_single_station_segment."
                "run_simple_waypoint_follower"
            ) as follower, redirect_stdout(StringIO()):
                status = run_segment(
                    [
                        "--route-csv",
                        outputs["route_csv"],
                        "--diagnostics-json",
                        outputs["diagnostics_json"],
                        "--route-certificate-json",
                        outputs["route_certificate_json"],
                        "--coverage-plan",
                        str(root / "coverage_plan.json"),
                        "--leg-index",
                        "0",
                        "--semantic-log",
                        str(events),
                        "--results-csv",
                        str(Path(tmp) / "results.csv"),
                        "--preflight-json",
                        str(Path(tmp) / "preflight.json"),
                        "--dry-run",
                    ]
                )
            self.assertEqual(status, 1)
            follower.assert_not_called()
            payloads = [
                json.loads(line)
                for line in events.read_text().splitlines()
                if line.strip()
            ]
            rejected = next(
                item for item in payloads
                if item["event"] == "startup_route_rejected"
            )
            self.assertFalse(rejected["motion_published"])
            self.assertEqual(
                rejected["stop_details"]["phase"],
                "before_motion_confirmation",
            )

    def test_pending_registry_freezes_to_candidate_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "survey"
            self._planned_center_corridor(root)
            plan = load_coverage_survey_plan(root / "coverage_plan.json")
            registry_path = root / "stand_registry.json"
            registry = load_stand_survey_registry(registry_path, plan)
            candidate = SurveyCandidate(
                candidate_uid="survey_candidate_0001",
                x_m=0.2,
                y_m=0.3,
                radius_m=0.06,
                uncertainty_m=0.02,
                keepout_radius_m=0.31,
                confidence=0.9,
                hit_count=5,
                first_seen_sec=10.0,
                last_seen_sec=12.0,
                source_observation_ids=("obs_1", "obs_2"),
                viewpoint_ids=("survey_vp_001", "survey_vp_002"),
                status=STATUS_PENDING_CAMERA,
            )
            registry = type(registry)(
                schema_version=registry.schema_version,
                survey_id=registry.survey_id,
                planning_frame=registry.planning_frame,
                map_bundle_sha256=registry.map_bundle_sha256,
                candidates=(candidate,),
            )
            write_stand_survey_registry(registry_path, registry, plan)
            snapshot = candidate_snapshot_from_registry(
                registry,
                plan,
                registry_path=registry_path,
                snapshot_id="autonomous_snapshot",
            )
            self.assertEqual(snapshot.candidate_uids, (candidate.candidate_uid,))
            self.assertAlmostEqual(
                snapshot.candidates[0].geometry.radius_m,
                candidate.radius_m,
            )
            self.assertEqual(
                snapshot.candidates[0].source.observation_ids,
                candidate.source_observation_ids,
            )
            self.assertEqual(
                snapshot.candidates[0].source.source_artifact_sha256,
                stand_survey_registry_sha256(registry),
            )

    def test_physical_clearance_uses_measured_stand_base_radius(self):
        stand_model = stand_model_from_payload(
            {
                "schema_version": 2,
                "profile_id": "measured_test_stand_v2",
                "environment": "physical",
                "measurement_status": "measured",
                "head_width_m": 0.078,
                "head_height_m": 0.078,
                "head_depth_m": 0.006,
                "qr_symbol_width_m": 0.062,
                "qr_symbol_height_m": 0.062,
                "qr_panel_width_m": 0.071,
                "qr_panel_height_m": 0.071,
                "qr_center_x_m": 0.0,
                "qr_center_y_m": 0.0,
                "head_top_height_m": 0.210,
                "base_width_m": 0.153,
                "base_depth_m": 0.153,
                "tolerance_m": 0.002,
                "source": "test metrology",
            }
        )
        profile = SimpleNamespace(
            robot_radius_m=0.105,
            scan_origin_to_base_offset_m=0.0,
        )

        clearance = autonomous_wrapper._physical_clearance(
            profile,
            approach_offset_m=0.70,
            stand_model_profile=stand_model,
        )

        expected_radius = math.hypot(0.153, 0.153) / 2.0 + 0.002
        self.assertAlmostEqual(
            clearance["stand_collision_radius_m"],
            expected_radius,
        )
        self.assertAlmostEqual(clearance["stand_lidar_radius_m"], 0.06)
        self.assertAlmostEqual(clearance["minimum_lidar_standoff_m"], 0.33)
        self.assertAlmostEqual(
            clearance["minimum_collision_standoff_m"],
            expected_radius + 0.02 + 0.105 + 0.02 + 0.03,
        )
        self.assertAlmostEqual(clearance["minimum_active_standoff_m"], 0.33)
        self.assertLess(clearance["minimum_active_standoff_m"], 0.35)

    def test_candidate_phase_uses_public_fresh_amcl_pose2d_reader(self):
        profile = SimpleNamespace(
            namespace="",
            amcl_topic="amcl_pose",
            map_frame="map",
        )

        with patch.object(
            autonomous_wrapper,
            "read_current_pose2d_from_amcl",
            return_value=Pose2D(-0.9284, -0.0379, 0.1740),
        ) as reader:
            pose_reader = autonomous_wrapper.CandidateApproachEffects(
                read_current_pose=lambda: autonomous_wrapper.read_current_pose2d_from_amcl(
                    namespace=profile.namespace,
                    amcl_topic=profile.amcl_topic,
                    map_frame=profile.map_frame,
                    timeout_sec=autonomous_wrapper.STATIONARY_AMCL_TIMEOUT_SEC,
                    max_age_sec=2.0,
                ),
                run_motion_leg=Mock(),
                capture_observation=Mock(),
            ).read_current_pose
            pose = pose_reader()

        self.assertIs(type(pose), Pose2D)
        self.assertEqual(pose, Pose2D(-0.9284, -0.0379, 0.1740))
        reader.assert_called_once_with(
            namespace="",
            amcl_topic="amcl_pose",
            map_frame="map",
            timeout_sec=autonomous_wrapper.STATIONARY_AMCL_TIMEOUT_SEC,
            max_age_sec=2.0,
        )

    def test_axis_only_observation_selects_opposite_face(self):
        with tempfile.TemporaryDirectory() as tmp:
            axis_path = Path(tmp) / "axis.json"
            axis_path.write_text(
                json.dumps(
                    {
                        "observation_kind": "real_stand_axis_without_qr",
                        "stand_axis_rad": 0.0,
                        "stand_center": {"x_m": 0.0, "y_m": 0.0},
                        "robot_pose": {
                            "x_m": 0.0,
                            "y_m": 0.7,
                            "yaw_rad": -math.pi / 2.0,
                        },
                    }
                )
            )
            selected = opposite_face_normal(axis_path)
            error = math.atan2(
                math.sin(selected + math.pi / 2.0),
                math.cos(selected + math.pi / 2.0),
            )
            self.assertAlmostEqual(error, 0.0, places=6)

    def test_real_hardware_checkpoint_options_are_explicit(self):
        args = build_parser().parse_args(
            [
                "--robot-profile",
                "robot.json",
                "--camera-calibration",
                "camera.json",
                "--physical-site",
                "site.json",
                "--coverage-leg-limit",
                "1",
                "--exact-inspection-point-count",
                "2",
                "--run-mode",
                "execute-coverage-checkpoint",
            ]
        )
        resolved = autonomous_wrapper.resolve_autonomous_run_mode(
            run_mode=args.run_mode,
            execute=args.execute,
            coverage_leg_limit=args.coverage_leg_limit,
            stop_after_coverage=args.stop_after_coverage,
        )

        self.assertEqual(resolved.coverage_leg_limit, 1)
        self.assertEqual(args.exact_inspection_point_count, 2)
        self.assertIsNone(args.expected_stand_count)
        self.assertEqual(args.max_startup_reseals_per_leg, 3)
        self.assertEqual(args.max_runtime_localization_reseals_per_leg, 1)
        self.assertEqual(args.max_localization_readiness_retries_per_leg, 2)
        self.assertEqual(args.uncertainty_sigma_multiplier, 2.0)
        self.assertFalse(args.prompt_for_initialpose)
        self.assertEqual(args.initialpose_prompt_window_sec, 2.0)
        self.assertFalse(resolved.stop_after_coverage)
        self.assertTrue(resolved.execute)

    def test_initialpose_prompt_option_is_explicit_and_non_authorizing(self):
        args = build_parser().parse_args(
            [
                "--robot-profile",
                "robot.json",
                "--camera-calibration",
                "camera.json",
                "--physical-site",
                "site.json",
                "--exact-inspection-point-count",
                "2",
                "--run-mode",
                "execute-exact-two-camera",
                "--prompt-for-initialpose",
                "--initialpose-prompt-window-sec",
                "3.0",
            ]
        )

        self.assertTrue(args.prompt_for_initialpose)
        self.assertEqual(args.initialpose_prompt_window_sec, 3.0)

    def test_exact_two_camera_options_enable_camera_phase(self):
        args = build_parser().parse_args(
            [
                "--robot-profile",
                "robot.json",
                "--camera-calibration",
                "camera.json",
                "--physical-site",
                "site.json",
                "--coverage-leg-limit",
                "2",
                "--exact-inspection-point-count",
                "2",
                "--run-mode",
                "execute-exact-two-camera",
            ]
        )
        resolved = autonomous_wrapper.resolve_autonomous_run_mode(
            run_mode=args.run_mode,
            execute=args.execute,
            coverage_leg_limit=args.coverage_leg_limit,
            stop_after_coverage=args.stop_after_coverage,
        )

        autonomous_wrapper.validate_autonomous_viewpoint_scope(
            resolved,
            exact_inspection_point_count=args.exact_inspection_point_count,
        )
        self.assertEqual(resolved.coverage_leg_limit, 2)
        self.assertTrue(resolved.camera_phase_enabled)
        self.assertTrue(resolved.execute)
        self.assertFalse(resolved.stop_after_coverage)

    def test_camera_modes_require_stand_model_before_loading_runtime_artifacts(self):
        cases = (
            (
                "execute-exact-two-camera",
                ["--exact-inspection-point-count", "2"],
            ),
            ("execute-full", []),
        )
        for run_mode, mode_args in cases:
            with (
                self.subTest(run_mode=run_mode),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                output_root = root / "runs"
                with (
                    patch.object(
                        autonomous_wrapper,
                        "load_real_robot_profile",
                    ) as load_profile,
                    patch("builtins.input") as run_prompt,
                    redirect_stderr(StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    autonomous_wrapper.main(
                        [
                            "--robot-profile", str(root / "robot.json"),
                            "--camera-calibration", str(root / "camera.json"),
                            "--physical-site", str(root / "site.json"),
                            "--output-root", str(output_root),
                            "--session-id", f"model_required_{run_mode}",
                            "--run-mode", run_mode,
                            *mode_args,
                        ]
                    )

                self.assertEqual(raised.exception.code, 2)
                load_profile.assert_not_called()
                run_prompt.assert_not_called()
                self.assertFalse(output_root.exists())

    def test_site_count_mismatch_fails_before_session_and_run_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "runs"
            with (
                patch.object(
                    autonomous_wrapper,
                    "load_real_robot_profile",
                    return_value=SimpleNamespace(),
                ),
                patch.object(
                    autonomous_wrapper,
                    "load_camera_calibration",
                    return_value=SimpleNamespace(),
                ),
                patch.object(
                    autonomous_wrapper,
                    "validate_physical_site_contract",
                    side_effect=ValueError(
                        "requested expected stand count differs from the "
                        "physical-site contract: requested=3 canonical=5"
                    ),
                ),
                patch("builtins.input") as run_prompt,
                redirect_stderr(StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                autonomous_wrapper.main(
                    [
                        "--robot-profile",
                        str(root / "robot.json"),
                        "--camera-calibration",
                        str(root / "camera.json"),
                        "--stand-model-profile",
                        str(root / "stand_model.json"),
                        "--physical-site",
                        str(root / "site.json"),
                        "--output-root",
                        str(output_root),
                        "--session-id",
                        "site_count_contract_execute_full",
                        "--expected-stand-count",
                        "3",
                        "--localization-branch-proof-id",
                        "known_start_marker_20260817",
                        "--execute",
                    ]
                )

            self.assertEqual(raised.exception.code, 2)
            run_prompt.assert_not_called()
            self.assertFalse(output_root.exists())

    def test_exact_two_full_mission_fails_before_loading_runtime_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "runs"
            with (
                patch.object(
                    autonomous_wrapper,
                    "load_real_robot_profile",
                ) as load_profile,
                patch("builtins.input") as run_prompt,
                redirect_stderr(StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                autonomous_wrapper.main(
                    [
                        "--robot-profile",
                        str(root / "robot.json"),
                        "--camera-calibration",
                        str(root / "camera.json"),
                        "--physical-site",
                        str(root / "site.json"),
                        "--output-root",
                        str(output_root),
                        "--session-id",
                        "five_stand_execute_full",
                        "--exact-inspection-point-count",
                        "2",
                        "--run-mode",
                        "execute-full",
                    ]
                )

            self.assertEqual(raised.exception.code, 2)
            load_profile.assert_not_called()
            run_prompt.assert_not_called()
            self.assertFalse(output_root.exists())

    def test_uncertainty_sigma_multiplier_must_be_finite_and_positive(self):
        for invalid in ("0", "-1", "nan", "inf"):
            with self.subTest(invalid=invalid):
                parser = build_parser()
                args = parser.parse_args(
                    [
                        "--robot-profile",
                        "robot.json",
                        "--camera-calibration",
                        "camera.json",
                        "--physical-site",
                        "site.json",
                        "--uncertainty-sigma-multiplier",
                        invalid,
                    ]
                )
                with redirect_stderr(StringIO()), self.assertRaises(
                    SystemExit
                ):
                    autonomous_wrapper._validate_inputs(
                        parser,
                        args,
                        None,
                        None,
                    )

    def _run_one_leg_wrapper_to_failure(
        self,
        root: Path,
        *,
        record_error=None,
        admission_error=None,
    ):
        session_id = "one_leg_final_checkpoint"
        output_root = root / "runs"
        session_root = output_root / session_id
        plan = SimpleNamespace(
            viewpoints=(SimpleNamespace(viewpoint_id="survey_vp_001"),),
            map_bundle_sha256="a" * 64,
        )
        registry = SimpleNamespace()
        progress = SimpleNamespace(visited_viewpoint_ids=())
        for fixture_name in ("robot.json", "camera.json", "site.json"):
            (root / fixture_name).write_text("{}\n")
        profile = SimpleNamespace(
            robot_id="turtlebot1",
            robot_radius_m=0.105,
            map_frame="map",
            scan_origin_to_base_offset_m=0.05,
            resolved_runtime=lambda: SimpleNamespace(
                namespace="",
                cmd_vel_topic="/cmd_vel",
            ),
        )

        def plan_one_leg(command):
            survey_root = Path(
                command[command.index("--output-dir") + 1]
            )
            survey_root.mkdir(parents=True, exist_ok=True)
            (survey_root / "survey_summary.json").write_text(
                json.dumps({"next_viewpoint_id": "survey_vp_001"}) + "\n"
            )
            return 0

        def record_final_stop(**kwargs):
            if record_error is not None:
                raise record_error
            survey_root = Path(kwargs["survey_root"])
            (survey_root / "survey_summary.json").write_text(
                json.dumps({"next_viewpoint_id": None}) + "\n"
            )
            return {"next_viewpoint_id": None}

        completed = MotionLegOutcome(
            run_id=f"{session_id}_coverage_000",
            status="completed",
            stop_reason="",
            stop_details={},
            motion_published=True,
            returncode=0,
            semantic_log_path=root / "events.jsonl",
            odom_execution_certificate_path=root / "odom_certificate.json",
        )
        with (
            patch.multiple(
                autonomous_wrapper,
                load_real_robot_profile=lambda *_args, **_kwargs: profile,
                load_camera_calibration=lambda *_args, **_kwargs: SimpleNamespace(),
                validate_physical_site_contract=lambda *_args, **_kwargs: SimpleNamespace(
                    expected_stand_count=1,
                    physical_site_path=(root / "site.json"),
                    map_yaml_path=(root / "map.yaml"),
                    map_bundle=SimpleNamespace(bundle_sha256="a" * 64),
                ),
                _validate_inputs=lambda *_args, **_kwargs: None,
                _physical_clearance=lambda *_args, **_kwargs: {
                    "minimum_active_standoff_m": 0.20,
                    "minimum_static_inflation_m": 0.25,
                    "minimum_candidate_transit_radius_m": 0.31,
                },
                _admit_preplanning_localization=lambda *_args, **_kwargs: Pose2D(
                    0.0, 0.0, 0.0
                ),
                _admit_observation_tf_readiness=lambda *_args, **_kwargs: (
                    session_root
                    / "preflight/lidar_scan_tf_before_authorization.json",
                    "f" * 64,
                ),
                admit_preauthorization_readiness=(
                    lambda *_args, **_kwargs: SimpleNamespace(
                        result=SimpleNamespace(attempts=(object(),)),
                        evidence_path=root / "initial_readiness.json",
                        evidence_sha256="e" * 64,
                    )
                ),
                plan_stand_coverage_survey=plan_one_leg,
                load_coverage_survey_plan=lambda *_args, **_kwargs: plan,
                write_mission_leg_motion_authorization=lambda *_args, **_kwargs: (
                    "mission-leg-authorization-sha256"
                ),
                write_mission_motion_authorization=lambda *_args, **_kwargs: (
                    "mission-authorization-sha256"
                ),
                _execute_coverage_leg_with_replans=lambda *_args, **_kwargs: completed,
                _capture_lidar_epoch=lambda *_args, **_kwargs: (
                    root / "observer_summary.json"
                ),
                load_stand_survey_registry=lambda *_args, **_kwargs: registry,
                load_survey_progress=lambda *_args, **_kwargs: progress,
            ),
            patch.object(
                autonomous_wrapper,
                "commit_stand_coverage_stop",
                side_effect=record_final_stop,
            ) as record_stop,
            patch.object(
                autonomous_wrapper,
                "publish_coverage_checkpoint",
                return_value=SimpleNamespace(
                    manifest_path=root / "terminal_checkpoint.json",
                    manifest_sha256="c" * 64,
                ),
            ) as publish_checkpoint,
            patch.object(
                autonomous_wrapper,
                "evaluate_coverage_candidate_admission",
                side_effect=admission_error,
            ) as evaluate_admission,
            patch("builtins.input", return_value="RUN") as run_prompt,
            patch("sys.stderr", new=StringIO()),
            redirect_stdout(StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            autonomous_wrapper.main(
                [
                    "--robot-profile",
                    str(root / "robot.json"),
                    "--camera-calibration",
                    str(root / "camera.json"),
                    "--physical-site",
                    str(root / "site.json"),
                    "--session-id",
                    session_id,
                    "--output-root",
                    str(output_root),
                    "--expected-stand-count",
                    "1",
                    "--coverage-leg-limit",
                    "1",
                    "--localization-branch-proof-id",
                    "known_start_marker_20260807",
                    "--execute",
                ]
            )
        run_prompt.assert_called_once_with(
            "Type RUN to authorize the autonomous exploration mission: "
        )
        if record_error is None:
            publish_checkpoint.assert_called_once()
        else:
            publish_checkpoint.assert_not_called()
        return session_root, raised.exception.code, record_stop, evaluate_admission

    def test_final_coverage_leg_limit_reaches_candidate_admission_gate(self):
        reason = "post-coverage candidate admission sentinel"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_root, exit_code, record_stop, evaluate_admission = (
                self._run_one_leg_wrapper_to_failure(
                    root,
                    admission_error=RuntimeError(reason),
                )
            )
            failure = json.loads(
                (session_root / "mission_failure.json").read_text()
            )

            self.assertEqual(exit_code, 2)
            record_stop.assert_called_once()
            fusion_call = record_stop.call_args.kwargs
            self.assertNotIn("next_leg_start_pose", fusion_call)
            self.assertNotIn("next_leg_localization_evidence_json", fusion_call)
            evaluate_admission.assert_called_once()
            self.assertEqual(failure["reason"], reason)
            self.assertFalse(failure["motion_continues_authorized"])
            self.assertFalse((session_root / "mission_summary.json").exists())

    def test_completed_observation_records_checkpoint_before_next_localization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_id = "post_observation_gap_checkpoint"
            output_root = root / "runs"
            session_root = output_root / session_id
            plan = SimpleNamespace(
                viewpoints=(
                    SimpleNamespace(viewpoint_id="survey_vp_001"),
                    SimpleNamespace(viewpoint_id="survey_vp_002"),
                ),
                map_bundle_sha256="a" * 64,
            )
            progress = SimpleNamespace(visited_viewpoint_ids=())
            for fixture_name in ("robot.json", "camera.json", "site.json"):
                (root / fixture_name).write_text("{}\n")
            profile = SimpleNamespace(
                robot_id="turtlebot1",
                robot_radius_m=0.105,
                map_frame="map",
                scan_origin_to_base_offset_m=0.05,
                resolved_runtime=lambda: SimpleNamespace(
                    namespace="",
                    cmd_vel_topic="/cmd_vel",
                ),
            )
            completed = MotionLegOutcome(
                run_id=f"{session_id}_coverage_000",
                status="completed",
                stop_reason="",
                stop_details={},
                motion_published=True,
                returncode=0,
                semantic_log_path=root / "events.jsonl",
                odom_execution_certificate_path=root / "odom_certificate.json",
            )
            localization_calls = []

            def admit_localization(*_args, **kwargs):
                localization_calls.append(kwargs.get("evidence_path"))
                return Pose2D(0.0, 0.0, 0.0)

            def plan_two_legs(command):
                survey_root = Path(command[command.index("--output-dir") + 1])
                survey_root.mkdir(parents=True, exist_ok=True)
                (survey_root / "survey_summary.json").write_text(
                    json.dumps({"next_viewpoint_id": "survey_vp_001"}) + "\n"
                )
                return 0

            def record_stop(**kwargs):
                self.assertNotIn("next_leg_start_pose", kwargs)
                self.assertNotIn("next_leg_localization_evidence_json", kwargs)
                survey_root = Path(kwargs["survey_root"])
                summary = {
                    "next_viewpoint_id": "survey_vp_002",
                    "recorded_viewpoint_id": "survey_vp_001",
                    "coverage_complete": False,
                    "visited_coverage_ratio": 0.5,
                    "visited_viewpoint_count": 1,
                    "total_viewpoint_count": 2,
                }
                (survey_root / "survey_summary.json").write_text(
                    json.dumps(summary) + "\n"
                )
                return summary

            with (
                patch.multiple(
                    autonomous_wrapper,
                    load_real_robot_profile=lambda *_args, **_kwargs: profile,
                    load_camera_calibration=lambda *_args, **_kwargs: SimpleNamespace(),
                    validate_physical_site_contract=lambda *_args, **_kwargs: SimpleNamespace(
                        expected_stand_count=1,
                        physical_site_path=(root / "site.json"),
                        map_yaml_path=(root / "map.yaml"),
                        map_bundle=SimpleNamespace(bundle_sha256="a" * 64),
                    ),
                    _validate_inputs=lambda *_args, **_kwargs: None,
                    _physical_clearance=lambda *_args, **_kwargs: {
                        "minimum_active_standoff_m": 0.20,
                        "minimum_static_inflation_m": 0.25,
                        "minimum_candidate_transit_radius_m": 0.31,
                    },
                    _admit_preplanning_localization=admit_localization,
                    _admit_observation_tf_readiness=lambda *_args, **_kwargs: (
                        session_root
                        / "preflight/lidar_scan_tf_before_authorization.json",
                        "f" * 64,
                    ),
                    admit_preauthorization_readiness=(
                        lambda *_args, **_kwargs: SimpleNamespace(
                            result=SimpleNamespace(attempts=(object(),)),
                            evidence_path=root / "initial_readiness.json",
                            evidence_sha256="e" * 64,
                        )
                    ),
                    plan_stand_coverage_survey=plan_two_legs,
                    load_coverage_survey_plan=lambda *_args, **_kwargs: plan,
                    write_mission_leg_motion_authorization=lambda *_args, **_kwargs: (
                        "mission-leg-authorization-sha256"
                    ),
                    write_mission_motion_authorization=lambda *_args, **_kwargs: (
                        "mission-authorization-sha256"
                    ),
                    _execute_coverage_leg_with_replans=lambda *_args, **_kwargs: completed,
                    _capture_lidar_epoch=lambda *_args, **_kwargs: (
                        root / "observer_summary.json"
                    ),
                    load_survey_progress=lambda *_args, **_kwargs: progress,
                ),
                patch.object(
                    autonomous_wrapper,
                    "commit_stand_coverage_stop",
                    side_effect=record_stop,
                ) as record_stop_mock,
                patch.object(
                    autonomous_wrapper,
                    "publish_coverage_checkpoint",
                    return_value=SimpleNamespace(
                        manifest_path=root / "checkpoint.json",
                        manifest_sha256="c" * 64,
                    ),
                ) as publish_checkpoint,
                patch("builtins.input", return_value="RUN") as run_prompt,
                patch("sys.stderr", new=StringIO()),
                redirect_stdout(StringIO()),
            ):
                exit_code = autonomous_wrapper.main(
                    [
                        "--robot-profile",
                        str(root / "robot.json"),
                        "--camera-calibration",
                        str(root / "camera.json"),
                        "--physical-site",
                        str(root / "site.json"),
                        "--session-id",
                        session_id,
                        "--output-root",
                        str(output_root),
                        "--expected-stand-count",
                        "1",
                        "--coverage-leg-limit",
                        "1",
                        "--localization-branch-proof-id",
                        "known_start_marker_20260807",
                        "--execute",
                    ]
                )

            self.assertEqual(exit_code, 0)
            run_prompt.assert_called_once()
            record_stop_mock.assert_called_once()
            publish_checkpoint.assert_called_once()
            self.assertEqual(
                localization_calls,
                [None],
            )
            self.assertFalse(
                any(
                    path is not None and "post_observation_localization" in str(path)
                    for path in localization_calls
                )
            )
            self.assertTrue((session_root / "mission_summary.json").exists())
            self.assertFalse((session_root / "mission_failure.json").exists())
            summary = json.loads(
                (session_root / "mission_summary.json").read_text()
            )
            self.assertEqual(summary["status"], "coverage_leg_checkpoint_complete")
            self.assertEqual(summary["next_viewpoint_id"], "survey_vp_002")

    def test_coverage_stop_exception_crosses_mission_failure_boundary(self):
        reason = "coverage stop fusion sentinel"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_root, exit_code, record_stop, evaluate_admission = (
                self._run_one_leg_wrapper_to_failure(
                    root,
                    record_error=ValueError(reason),
                )
            )
            failure = json.loads(
                (session_root / "mission_failure.json").read_text()
            )

            self.assertEqual(exit_code, 2)
            record_stop.assert_called_once()
            evaluate_admission.assert_not_called()
            self.assertEqual(failure["reason"], reason)
            self.assertFalse(failure["motion_continues_authorized"])

    def test_coverage_runner_keeps_route_and_mission_leg_indices_distinct(self):
        profile = self._profile()
        command = _runner_command(
            profile=profile,
            route_csv=Path("route.csv"),
            diagnostics_json=Path("diagnostics.json"),
            certificate_json=Path("certificate.json"),
            run_id="mission_coverage_002",
            session_root=Path("session"),
            coverage_plan=Path("survey/coverage_plan.json"),
            coverage_transient_replan={
                "survey_root": Path("survey"),
                "session_root": Path("session"),
                "map_yaml": MAP,
                "semantic_map_id": "arena_1p898x3p9_auto",
                "target_viewpoint_id": "survey_vp_003",
                "robot_radius_m": 0.105,
                "max_replans": 3,
                "leg_index": 2,
            },
            dry_run=False,
        )

        self.assertIn("--coverage-transient-replan-leg-index", command)
        self.assertEqual(
            command[command.index("--coverage-transient-replan-leg-index") + 1],
            "2",
        )
        self.assertEqual(
            command[command.index("--leg-index") + 1],
            "0",
        )
        self.assertEqual(
            float(
                command[
                    command.index("--omnidirectional-hard-stop-distance-m")
                    + 1
                ]
            ),
            0.125,
        )

    def test_runner_command_enables_complete_uncertainty_aware_odom_contract(self):
        command = _runner_command(
            profile=self._profile(),
            route_csv=Path("route.csv"),
            diagnostics_json=Path("diagnostics.json"),
            certificate_json=Path("map_certificate.json"),
            run_id="mission_coverage_000",
            session_root=Path("session"),
            dry_run=True,
            uncertainty_map_yaml=MAP,
            localization_branch_proof_id="known_start_marker_20260807",
            odom_execution_certificate_json=Path("odom_certificate.json"),
            uncertainty_budget_json=Path("uncertainty_budget.json"),
        )

        self.assertEqual(
            command[command.index("--execution-pose-frame") + 1], "odom"
        )
        self.assertEqual(
            command[
                command.index("--localization-branch-proof-id") + 1
            ],
            "known_start_marker_20260807",
        )
        self.assertEqual(
            command[command.index("--uncertainty-robot-radius-m") + 1],
            "0.105",
        )
        self.assertEqual(
            command[command.index("--uncertainty-sigma-multiplier") + 1],
            "2.0",
        )
        self.assertEqual(
            command[command.index("--max-stationary-amcl-position-std-m") + 1],
            "0.30",
        )
        self.assertEqual(
            command[command.index("--preflight-json") + 1],
            "session/preflight/mission_coverage_000_dry.json",
        )

    def test_runner_command_rejects_partial_odom_contract(self):
        with self.assertRaisesRegex(ValueError, "must be complete"):
            _runner_command(
                profile=self._profile(),
                route_csv=Path("route.csv"),
                diagnostics_json=Path("diagnostics.json"),
                certificate_json=Path("map_certificate.json"),
                run_id="mission_coverage_000",
                session_root=Path("session"),
                dry_run=True,
                uncertainty_map_yaml=MAP,
            )

    def test_runner_command_requires_complete_live_recovery_authorization(self):
        common = {
            "profile": self._profile(),
            "route_csv": Path("route.csv"),
            "diagnostics_json": Path("diagnostics.json"),
            "certificate_json": Path("map_certificate.json"),
            "run_id": "mission_coverage_000_runtime_localization_reseal_001",
            "session_root": Path("session"),
            "coverage_transient_replan": {
                "survey_root": Path("survey"),
                "session_root": Path("session"),
                "map_yaml": MAP,
                "semantic_map_id": "arena_1p898x3p9_auto",
                "target_viewpoint_id": "survey_vp_001",
                "robot_radius_m": 0.105,
                "max_replans": 3,
                "leg_index": 0,
            },
            "dry_run": False,
        }
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            _runner_command(
                **common,
                mission_motion_authorization_json=Path("mission.json"),
            )

        command = _runner_command(
            **common,
            mission_motion_authorization_json=Path("mission.json"),
            runtime_localization_motion_permit_json=Path("permit.json"),
            mission_session_id="mission",
        )
        self.assertEqual(
            command[
                command.index("--runtime-localization-motion-permit-json") + 1
            ],
            "permit.json",
        )
        self.assertEqual(
            command[command.index("--mission-session-id") + 1],
            "mission",
        )

    def test_lidar_epoch_uses_completed_odom_execution_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            certificate = root / "execute_certificate.json"
            certificate.write_text("{}\n")
            args = SimpleNamespace(
                map=MAP,
                semantic_map_id="arena_1p898x3p9_auto",
                lidar_epoch_sec=1.0,
            )

            def write_summary(command, **_kwargs):
                summary = Path(command[command.index("--summary-json") + 1])
                summary.write_text(
                    json.dumps({"processed_scan_count": 1}) + "\n"
                )
                return SimpleNamespace(returncode=0)

            with patch(
                "scripts.aufgabe04.real_robot."
                "autonomous_runner.runtime.subprocess.run",
                side_effect=write_summary,
            ) as run:
                summary = _capture_lidar_epoch(
                    profile=self._profile(),
                    args=args,
                    survey_root=root / "survey",
                    viewpoint_id="survey_vp_001",
                    odom_execution_certificate_path=certificate,
                    coverage_plan=SimpleNamespace(
                        survey_id="survey_01",
                        config=SimpleNamespace(candidate_radius_m=0.06),
                    ),
                )

        command = run.call_args.args[0]
        self.assertEqual(
            command[
                command.index("--odom-execution-certificate-json") + 1
            ],
            str(certificate),
        )
        self.assertEqual(
            command[command.index("--observation-id-scope") + 1],
            "survey_vp_001",
        )
        self.assertEqual(
            command[command.index("--survey-candidate-radius-m") + 1],
            "0.06",
        )
        self.assertEqual(
            command[command.index("--visibility-survey-id") + 1],
            "survey_01",
        )
        self.assertEqual(summary.name, "observer_summary.json")

    def test_coverage_leg_uses_one_runner_with_in_process_replanning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_route = root / "route.csv"
            source_diagnostics = root / "diagnostics.json"
            source_route.write_text("source\n")
            source_diagnostics.write_text("{}\n")
            completed = MotionLegOutcome(
                run_id="mission_coverage_000",
                status="completed",
                stop_reason="",
                stop_details={},
                motion_published=True,
                returncode=0,
                semantic_log_path=root / "completed.jsonl",
            )
            args = SimpleNamespace(
                session_id="mission",
                max_blockage_replans_per_leg=2,
                map=MAP,
                semantic_map_id="arena_1p898x3p9_auto",
            )
            profile = SimpleNamespace(robot_radius_m=0.105)
            sealed = {
                "route_csv": str(source_route),
                "diagnostics_json": str(source_diagnostics),
                "route_certificate_json": str(root / "certificate.json"),
            }
            with patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "seal_stand_discovery_route",
                return_value=sealed,
            ) as seal, patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "_run_motion_leg",
                return_value=completed,
            ) as run, patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "_capture_lidar_epoch",
            ) as observe:
                _execute_coverage_leg_with_replans(
                    profile=profile,
                    args=args,
                    session_root=root / "session",
                    survey_root=root / "survey",
                    plan_path=root / "coverage_plan.json",
                    leg_index=0,
                    target_viewpoint_id="survey_vp_001",
                    source_route=source_route,
                    source_diagnostics=source_diagnostics,
                )

            self.assertEqual(run.call_count, 1)
            self.assertEqual(seal.call_count, 1)
            observe.assert_not_called()
            recovery = run.call_args.kwargs["coverage_transient_replan"]
            self.assertEqual(recovery["target_viewpoint_id"], "survey_vp_001")
            self.assertEqual(recovery["max_replans"], 2)
            self.assertEqual(recovery["robot_radius_m"], 0.105)

    def test_startup_mismatch_replans_and_requires_fresh_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_route = root / "route.csv"
            source_diagnostics = root / "diagnostics.json"
            source_route.write_text("source\n")
            source_diagnostics.write_text("{}\n")
            replacement_route = root / "replacement_route.csv"
            replacement_diagnostics = root / "replacement_diagnostics.json"
            replacement_route.write_text("replacement\n")
            replacement_diagnostics.write_text("{}\n")
            rejected = MotionLegOutcome(
                run_id="mission_coverage_000",
                status="stopped",
                stop_reason="pose outside certified startup segment",
                stop_details={
                    "source": "execution_route_certificate",
                    "phase": "before_motion_confirmation",
                    "route_pose": {
                        "frame_id": "map",
                        "child_frame_id": "base_footprint",
                        "x_m": -0.5044625,
                        "y_m": -0.6255536,
                        "yaw_rad": 1.7007652,
                    },
                },
                motion_published=False,
                returncode=1,
                semantic_log_path=root / "rejected.jsonl",
            )
            completed = MotionLegOutcome(
                run_id="mission_coverage_000_startup_reseal_001",
                status="completed",
                stop_reason="",
                stop_details={},
                motion_published=True,
                returncode=0,
                semantic_log_path=root / "completed.jsonl",
            )
            args = SimpleNamespace(
                session_id="mission",
                max_blockage_replans_per_leg=2,
                max_startup_reseals_per_leg=2,
                map=MAP,
                semantic_map_id="arena_1p898x3p9_auto",
            )
            admitted_pose = Pose2D(-0.49, -0.61, 1.69)
            sealed = {
                "route_csv": str(source_route),
                "diagnostics_json": str(source_diagnostics),
                "route_certificate_json": str(root / "certificate.json"),
            }
            with patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "seal_stand_discovery_route",
                return_value=sealed,
            ) as seal, patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "_run_motion_leg",
                side_effect=(rejected, completed),
            ) as run, patch.object(
                autonomous_coverage_replanning,
                "replan_startup_source",
                return_value={
                    "route_csv": str(replacement_route),
                    "diagnostics_json": str(replacement_diagnostics),
                    "summary_json": str(root / "reseal_summary.json"),
                },
            ) as replan, patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "_admit_preplanning_localization",
                return_value=admitted_pose,
            ) as admit, patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "_capture_lidar_epoch"
            ) as observe:
                _execute_coverage_leg_with_replans(
                    profile=SimpleNamespace(robot_radius_m=0.105),
                    args=args,
                    session_root=root / "session",
                    survey_root=root / "survey",
                    plan_path=root / "coverage_plan.json",
                    leg_index=0,
                    target_viewpoint_id="survey_vp_001",
                    source_route=source_route,
                    source_diagnostics=source_diagnostics,
                )

            self.assertTrue(is_resealable_startup_mismatch(rejected))
            self.assertEqual(run.call_count, 2)
            self.assertFalse(
                run.call_args_list[0].kwargs["require_fresh_confirmation"]
            )
            self.assertTrue(
                run.call_args_list[1].kwargs["require_fresh_confirmation"]
            )
            self.assertEqual(seal.call_count, 2)
            self.assertEqual(
                seal.call_args_list[1].kwargs["source_route_csv"],
                replacement_route,
            )
            replan.assert_called_once()
            self.assertEqual(
                replan.call_args.kwargs["current_pose"], admitted_pose
            )
            admit.assert_called_once()
            observe.assert_not_called()
            adaptive = [
                json.loads(line)
                for line in (root / "session/adaptive_replans.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in adaptive],
                [
                    "startup_reseal_started",
                    "startup_localization_admitted",
                    "startup_pose_route_resealed",
                    "startup_reseal_route_sealed",
                ],
            )
            self.assertTrue(adaptive[2]["fresh_confirmation_required"])

    def test_runtime_localization_reseal_replans_from_fresh_admitted_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_route = root / "route.csv"
            source_diagnostics = root / "diagnostics.json"
            source_route.write_text("source\n")
            source_diagnostics.write_text("{}\n")
            replacement_route = root / "runtime_replacement_route.csv"
            replacement_diagnostics = root / "runtime_replacement_diagnostics.json"
            replacement_route.write_text("replacement\n")
            replacement_diagnostics.write_text("{}\n")
            stopped = MotionLegOutcome(
                run_id="mission_coverage_000",
                status="stopped",
                stop_reason="global localization consistency requires zero and reseal",
                stop_details=_runtime_localization_stop_details(),
                motion_published=True,
                returncode=1,
                semantic_log_path=root / "stopped.jsonl",
            )
            stopped.semantic_log_path.write_text("{}\n")
            completed = MotionLegOutcome(
                run_id=(
                    "mission_coverage_000_"
                    "runtime_localization_reseal_001"
                ),
                status="completed",
                stop_reason="",
                stop_details={},
                motion_published=True,
                returncode=0,
                semantic_log_path=root / "completed.jsonl",
            )
            args = SimpleNamespace(
                session_id="mission",
                max_blockage_replans_per_leg=2,
                max_startup_reseals_per_leg=2,
                max_runtime_localization_reseals_per_leg=1,
                map=MAP,
                semantic_map_id="arena_1p898x3p9_auto",
                localization_branch_proof_id="known_start_marker_20260807",
            )
            profile = SimpleNamespace(
                robot_radius_m=0.105,
                resolved_runtime=lambda: SimpleNamespace(namespace=""),
            )
            sealed = {
                "route_csv": str(source_route),
                "diagnostics_json": str(source_diagnostics),
                "route_certificate_json": str(root / "certificate.json"),
            }
            admitted_pose = Pose2D(-0.31, -0.47, 0.12)
            with patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "seal_stand_discovery_route",
                return_value=sealed,
            ) as seal, patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "_run_motion_leg",
                side_effect=(stopped, completed),
            ) as run, patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "_admit_preplanning_localization",
                return_value=admitted_pose,
            ) as admit, patch.object(
                autonomous_coverage_replanning,
                "replan_runtime_localization_source",
                return_value={
                    "route_csv": str(replacement_route),
                    "diagnostics_json": str(replacement_diagnostics),
                    "summary_json": str(root / "runtime_reseal_summary.json"),
                },
            ) as replan:
                outcome = _execute_coverage_leg_with_replans(
                    profile=profile,
                    args=args,
                    session_root=root / "session",
                    survey_root=root / "survey",
                    plan_path=root / "coverage_plan.json",
                    leg_index=0,
                    target_viewpoint_id="survey_vp_001",
                    source_route=source_route,
                    source_diagnostics=source_diagnostics,
                    mission_motion_authorization_json=(
                        root / "mission_authorization.json"
                    ),
                )

            self.assertEqual(outcome.status, "completed")
            self.assertTrue(is_runtime_localization_reseal_required(stopped))
            self.assertEqual(run.call_count, 2)
            self.assertTrue(
                run.call_args_list[1].kwargs["require_fresh_confirmation"]
            )
            self.assertEqual(
                run.call_args_list[1].kwargs["fresh_confirmation_reason"],
                "runtime_localization",
            )
            permit_context = run.call_args_list[1].kwargs[
                "runtime_localization_permit_context"
            ]
            self.assertIsInstance(
                permit_context,
                RuntimeLocalizationPermitContext,
            )
            self.assertEqual(
                permit_context.target_viewpoint_id,
                "survey_vp_001",
            )
            self.assertEqual(
                run.call_args_list[1].kwargs[
                    "fresh_localization_evidence_path"
                ],
                admit.call_args.kwargs["evidence_path"],
            )
            self.assertEqual(seal.call_count, 2)
            self.assertEqual(
                seal.call_args_list[1].kwargs["source_route_csv"],
                replacement_route,
            )
            admit.assert_called_once()
            self.assertIn(
                "runtime_localization_reseals",
                str(admit.call_args.kwargs["evidence_path"]),
            )
            replan.assert_called_once()
            self.assertEqual(
                replan.call_args.kwargs["current_pose"], admitted_pose
            )
            self.assertEqual(
                replan.call_args.kwargs["expected_target_viewpoint_id"],
                "survey_vp_001",
            )
            adaptive = [
                json.loads(line)
                for line in (root / "session/adaptive_replans.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in adaptive],
                [
                    "runtime_localization_reseal_started",
                    "runtime_localization_admitted",
                    "runtime_localization_route_replanned",
                    "runtime_localization_route_sealed",
                ],
            )
            self.assertFalse(adaptive[-1]["fresh_confirmation_required"])
            self.assertFalse(adaptive[-1]["fresh_typed_run_required"])
            self.assertTrue(adaptive[-1]["covered_by_initial_mission_run"])
            self.assertFalse(adaptive[-1]["motion_continues_authorized"])
            self.assertEqual(
                adaptive[-1]["committed_target_viewpoint_id"],
                "survey_vp_001",
            )
            self.assertIn(
                "runtime_localization_reseal_001_dry_certificate.json",
                adaptive[-1][
                    "expected_dry_odom_execution_certificate_json"
                ],
            )

    def test_runtime_localization_admission_failure_is_persisted_and_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_route = root / "route.csv"
            source_diagnostics = root / "diagnostics.json"
            source_route.write_text("route\n")
            source_diagnostics.write_text("{}\n")
            stopped = MotionLegOutcome(
                run_id="mission_coverage_000",
                status="stopped",
                stop_reason=(
                    "global localization consistency requires zero and reseal"
                ),
                stop_details=_runtime_localization_stop_details(),
                motion_published=True,
                returncode=1,
                semantic_log_path=root / "stopped.jsonl",
            )
            stopped.semantic_log_path.write_text("{}\n")
            args = SimpleNamespace(
                session_id="mission",
                max_blockage_replans_per_leg=2,
                max_startup_reseals_per_leg=2,
                max_runtime_localization_reseals_per_leg=1,
                map=MAP,
                semantic_map_id="arena_1p898x3p9_auto",
                localization_branch_proof_id="known_start_marker_20260807",
            )
            sealed = {
                "route_csv": str(source_route),
                "diagnostics_json": str(source_diagnostics),
                "route_certificate_json": str(root / "certificate.json"),
            }
            with patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "seal_stand_discovery_route",
                return_value=sealed,
            ), patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "_run_motion_leg",
                return_value=stopped,
            ) as run, patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "_admit_preplanning_localization",
                side_effect=RuntimeError("stationary AMCL did not converge"),
            ), patch.object(
                autonomous_coverage_replanning,
                "replan_runtime_localization_source",
            ) as replan:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "stationary AMCL did not converge",
                ):
                    _execute_coverage_leg_with_replans(
                        profile=SimpleNamespace(
                            robot_radius_m=0.105,
                            resolved_runtime=lambda: SimpleNamespace(
                                namespace=""
                            ),
                        ),
                        args=args,
                        session_root=root / "session",
                        survey_root=root / "survey",
                        plan_path=root / "coverage_plan.json",
                        leg_index=0,
                        target_viewpoint_id="survey_vp_001",
                        source_route=source_route,
                        source_diagnostics=source_diagnostics,
                    )

            self.assertEqual(run.call_count, 1)
            replan.assert_not_called()
            adaptive = [
                json.loads(line)
                for line in (root / "session/adaptive_replans.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in adaptive],
                [
                    "runtime_localization_reseal_started",
                    "runtime_localization_reseal_failed",
                ],
            )
            self.assertEqual(
                adaptive[-1]["phase"],
                "stationary_localization_admission",
            )
            self.assertFalse(adaptive[-1]["motion_continues_authorized"])

    def test_runtime_localization_reseal_budget_is_separate_and_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_route = root / "route.csv"
            source_diagnostics = root / "diagnostics.json"
            source_route.write_text("route\n")
            source_diagnostics.write_text("{}\n")
            stopped = MotionLegOutcome(
                run_id="mission_coverage_000",
                status="stopped",
                stop_reason="global localization consistency requires zero and reseal",
                stop_details=_runtime_localization_stop_details(),
                motion_published=True,
                returncode=1,
                semantic_log_path=root / "stopped.jsonl",
            )
            args = SimpleNamespace(
                session_id="mission",
                max_blockage_replans_per_leg=2,
                max_startup_reseals_per_leg=2,
                max_runtime_localization_reseals_per_leg=0,
                map=MAP,
                semantic_map_id="arena_1p898x3p9_auto",
                localization_branch_proof_id="known_start_marker_20260807",
            )
            sealed = {
                "route_csv": str(source_route),
                "diagnostics_json": str(source_diagnostics),
                "route_certificate_json": str(root / "certificate.json"),
            }
            with patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "seal_stand_discovery_route",
                return_value=sealed,
            ), patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "_run_motion_leg",
                return_value=stopped,
            ), patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "_admit_preplanning_localization",
            ) as admit:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "runtime localization reseal budget exhausted",
                ):
                    _execute_coverage_leg_with_replans(
                        profile=SimpleNamespace(robot_radius_m=0.105),
                        args=args,
                        session_root=root / "session",
                        survey_root=root / "survey",
                        plan_path=root / "coverage_plan.json",
                        leg_index=0,
                        target_viewpoint_id="survey_vp_001",
                        source_route=source_route,
                        source_diagnostics=source_diagnostics,
                    )

            admit.assert_not_called()

    def test_runtime_localization_reseal_preserves_adopted_blockage_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_route = root / "route.csv"
            source_diagnostics = root / "diagnostics.json"
            source_route.write_text("route\n")
            source_diagnostics.write_text("{}\n")
            replacement_route = root / "overlay_runtime_route.csv"
            replacement_diagnostics = root / "overlay_runtime_diagnostics.json"
            replacement_summary = root / "overlay_runtime_summary.json"
            replacement_route.write_text("replacement\n")
            replacement_diagnostics.write_text("{}\n")
            replacement_summary.write_text("{}\n")
            stopped = MotionLegOutcome(
                run_id="mission_coverage_000",
                status="stopped",
                stop_reason=(
                    "global localization consistency requires zero and reseal"
                ),
                stop_details=_runtime_localization_stop_details(),
                motion_published=True,
                returncode=1,
                semantic_log_path=root / "stopped.jsonl",
            )
            completed = MotionLegOutcome(
                run_id=(
                    "mission_coverage_000_"
                    "runtime_localization_reseal_001"
                ),
                status="completed",
                stop_reason="",
                stop_details={},
                motion_published=True,
                returncode=0,
                semantic_log_path=root / "completed.jsonl",
            )
            args = SimpleNamespace(
                session_id="mission",
                max_blockage_replans_per_leg=2,
                max_startup_reseals_per_leg=2,
                max_runtime_localization_reseals_per_leg=1,
                map=MAP,
                semantic_map_id="arena_1p898x3p9_auto",
                localization_branch_proof_id="known_start_marker_20260807",
            )
            sealed = {
                "route_csv": str(source_route),
                "diagnostics_json": str(source_diagnostics),
                "route_certificate_json": str(root / "certificate.json"),
            }
            resume_state_path = root / "transient_overlay_resume_state.json"
            resume_state_path.write_text("{}\n")
            resume_state = TransientOverlayResumeState(
                schema_version=1,
                coverage_plan_sha256="a" * 64,
                survey_id="mission",
                planning_frame="map",
                map_bundle_sha256="b" * 64,
                coverage_leg_index=0,
                target_viewpoint_id="survey_vp_001",
                completed_replan_count=1,
                max_replans=2,
                remaining_replans=1,
                transient_obstacle_overlay_path=str(root / "overlay.json"),
                transient_obstacle_overlay_sha256="c" * 64,
                overlay_candidate_ids=("transient_obstacle_0001",),
                adopted_route_paths=(str(root / "prior_route.csv"),),
                adopted_route_sha256s=("d" * 64,),
                source_run_ids=("mission_coverage_000",),
            )
            admitted_pose = Pose2D(-0.31, -0.47, 0.12)
            replanned = {
                "route_csv": str(replacement_route),
                "diagnostics_json": str(replacement_diagnostics),
                "summary_json": str(replacement_summary),
            }
            with patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "seal_stand_discovery_route",
                return_value=sealed,
            ) as seal, patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "_run_motion_leg",
                side_effect=(stopped, completed),
            ) as run, patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "_admit_preplanning_localization",
                return_value=admitted_pose,
            ) as admit, patch.object(
                autonomous_coverage_replanning,
                "advance_transient_overlay_resume_state",
                return_value=resume_state,
            ) as advance, patch.object(
                autonomous_coverage_replanning,
                "load_coverage_plan",
                return_value=SimpleNamespace(),
            ), patch.object(
                autonomous_coverage_replanning,
                "replan_source_preserving_transient_overlay",
                return_value=(
                    replanned,
                    resume_state,
                    resume_state_path,
                    "e" * 64,
                ),
            ) as replan:
                outcome = _execute_coverage_leg_with_replans(
                    profile=SimpleNamespace(
                        robot_radius_m=0.105,
                        resolved_runtime=lambda: SimpleNamespace(namespace=""),
                    ),
                    args=args,
                    session_root=root / "session",
                    survey_root=root / "survey",
                    plan_path=root / "coverage_plan.json",
                    leg_index=0,
                    target_viewpoint_id="survey_vp_001",
                    source_route=source_route,
                    source_diagnostics=source_diagnostics,
                    mission_motion_authorization_json=(
                        root / "mission_authorization.json"
                    ),
                )

            self.assertEqual(outcome.status, "completed")
            self.assertEqual(run.call_count, 2)
            advance.assert_called_once()
            self.assertEqual(
                advance.call_args.kwargs["outcome"],
                stopped,
            )
            self.assertEqual(
                advance.call_args.kwargs["artifact_root"],
                root / "session",
            )
            admit.assert_called_once()
            replan.assert_called_once()
            self.assertEqual(replan.call_args.kwargs["state"], resume_state)
            self.assertEqual(
                replan.call_args.kwargs["recovery_kind"],
                "runtime_localization",
            )
            self.assertEqual(
                replan.call_args.kwargs["artifact_root"],
                root / "session",
            )
            self.assertEqual(
                replan.call_args.kwargs["target_viewpoint_id"],
                "survey_vp_001",
            )
            self.assertEqual(seal.call_count, 2)
            self.assertEqual(
                seal.call_args_list[1].kwargs["source_route_csv"],
                replacement_route,
            )
            resumed_contract = run.call_args_list[1].kwargs[
                "coverage_transient_replan"
            ]
            self.assertEqual(
                resumed_contract["resume_state_json"],
                resume_state_path,
            )
            self.assertEqual(resumed_contract["max_replans"], 2)
            adaptive_log = root / "session/adaptive_replans.jsonl"
            adaptive = [
                json.loads(line)
                for line in adaptive_log.read_text().splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in adaptive],
                [
                    "runtime_localization_reseal_started",
                    "runtime_localization_admitted",
                    "runtime_localization_route_replanned",
                    "runtime_localization_route_sealed",
                ],
            )
            for event in adaptive:
                self.assertTrue(event["dynamic_overlay_preserved"])
                self.assertEqual(event["adopted_blockage_replan_count"], 1)
                self.assertEqual(event["remaining_blockage_replan_count"], 1)
            self.assertEqual(
                adaptive[-1]["transient_overlay_resume_state_json"],
                str(resume_state_path),
            )
            self.assertEqual(
                adaptive[-1]["transient_overlay_resume_state_sha256"],
                "e" * 64,
            )

    def test_runtime_blockage_failure_never_relaunches_the_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_route = root / "route.csv"
            source_diagnostics = root / "diagnostics.json"
            source_route.write_text("route\n")
            source_diagnostics.write_text("{}\n")
            blocked = MotionLegOutcome(
                run_id="mission_coverage_000",
                status="stopped",
                stop_reason="stuck no progress",
                stop_details={
                    "front_clearance": {
                        "nearest_valid_range_m": 0.248,
                        "nearest_valid_bearing_rad": 0.0,
                        "source": "front_sector",
                    },
                },
                motion_published=True,
                returncode=1,
                semantic_log_path=root / "blocked.jsonl",
            )
            args = SimpleNamespace(
                session_id="mission",
                max_blockage_replans_per_leg=2,
                max_startup_reseals_per_leg=2,
                map=MAP,
                semantic_map_id="arena_1p898x3p9_auto",
            )
            profile = SimpleNamespace(
                robot_radius_m=0.105,
                namespace="",
                amcl_topic="amcl_pose",
                map_frame="map",
            )
            sealed = {
                "route_csv": str(source_route),
                "diagnostics_json": str(source_diagnostics),
                "route_certificate_json": str(root / "certificate.json"),
            }
            with patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "seal_stand_discovery_route",
                return_value=sealed,
            ) as seal, patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "_run_motion_leg",
                return_value=blocked,
            ) as run:
                with self.assertRaisesRegex(RuntimeError, "stuck no progress"):
                    _execute_coverage_leg_with_replans(
                        profile=profile,
                        args=args,
                        session_root=root / "session",
                        survey_root=root / "survey",
                        plan_path=root / "coverage_plan.json",
                        leg_index=0,
                        target_viewpoint_id="survey_vp_001",
                        source_route=source_route,
                        source_diagnostics=source_diagnostics,
                    )

            self.assertEqual(run.call_count, 1)
            self.assertEqual(seal.call_count, 1)

    def test_dry_precheck_mismatch_is_returned_without_execution_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "startup_rejected"
            event_path = root / "run_events" / f"{run_id}.jsonl"
            event_payload = json.dumps(
                {
                    "event": "safety_stop",
                    "run_id": run_id,
                    "status": "stopped",
                    "stop_reason": "pose outside certified startup segment",
                    "stop_details": {
                        "source": "execution_route_certificate",
                        "phase": "before_motion_confirmation",
                        "route_pose": {
                            "frame_id": "map",
                            "child_frame_id": "base_footprint",
                            "x_m": -0.5044625,
                            "y_m": -0.6255536,
                            "yaw_rad": 1.7007652,
                        },
                    },
                    "motion_published": False,
                }
            )
            sealed = {
                "route_csv": str(root / "route.csv"),
                "diagnostics_json": str(root / "diagnostics.json"),
                "route_certificate_json": str(root / "certificate.json"),
            }

            def reject_dry(_command, **_kwargs):
                event_path.parent.mkdir(parents=True, exist_ok=True)
                with event_path.open("a", encoding="utf-8") as handle:
                    handle.write(event_payload + "\n")
                return SimpleNamespace(returncode=1)

            with patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "subprocess.run",
                side_effect=reject_dry,
            ) as run:
                outcome = _run_motion_leg(
                    profile=self._profile(),
                    sealed=sealed,
                    run_id=run_id,
                    session_root=root,
                    execute=True,
                    coverage_plan=root / "coverage_plan.json",
                )

            self.assertTrue(is_resealable_startup_mismatch(outcome))
            self.assertEqual(run.call_count, 1)

    def test_resealed_route_without_permit_stops_without_reprompting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sealed = {
                "route_csv": str(root / "route.csv"),
                "diagnostics_json": str(root / "diagnostics.json"),
                "route_certificate_json": str(root / "certificate.json"),
            }
            output = StringIO()
            localization_evidence = root / "runtime_localization.json"

            def run_dry(command, **_kwargs):
                _materialize_dry_child_evidence(command, root, "resealed")
                return SimpleNamespace(returncode=0)

            with patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "subprocess.run",
                side_effect=run_dry,
            ) as run, patch(
                "builtins.input",
                side_effect=AssertionError(
                    "autonomous reseal must not request another RUN"
                ),
            ), redirect_stdout(output):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "lacks an exact runtime_localization recovery permit",
                ):
                    _run_motion_leg(
                        profile=self._profile(),
                        sealed=sealed,
                        run_id="resealed",
                        session_root=root,
                        execute=True,
                        coverage_plan=root / "coverage_plan.json",
                        require_fresh_confirmation=True,
                        fresh_confirmation_reason="runtime_localization",
                        fresh_localization_evidence_path=(
                            localization_evidence
                        ),
                        uncertainty_map_yaml=MAP,
                        localization_branch_proof_id=(
                            "known_start_marker_20260807"
                        ),
                    )

            self.assertEqual(run.call_count, 1)
            rendered = output.getvalue()
            self.assertIn(str(localization_evidence), rendered)
            self.assertIn("resealed_dry_certificate.json", rendered)
            self.assertIn("resealed_dry_uncertainty_budget.json", rendered)

    def test_live_leg_keeps_child_runner_typed_run_interactive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "interactive_live_leg"
            sealed = {
                "route_csv": str(root / "route.csv"),
                "diagnostics_json": str(root / "diagnostics.json"),
                "route_certificate_json": str(root / "certificate.json"),
            }
            calls = 0

            def run_process(_command, **_kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    _materialize_dry_child_evidence(_command, root, run_id)
                else:
                    _append_child_event(
                        root,
                        run_id,
                        {
                            "event": "motion_completed",
                            "run_id": run_id,
                            "status": "completed",
                            "stop_reason": "",
                            "stop_details": {},
                            "motion_published": True,
                        },
                    )
                return SimpleNamespace(returncode=0)

            with patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "subprocess.run",
                side_effect=run_process,
            ) as run:
                outcome = _run_motion_leg(
                    profile=self._profile(),
                    sealed=sealed,
                    run_id=run_id,
                    session_root=root,
                    execute=True,
                )

        self.assertEqual(outcome.status, "completed")
        self.assertEqual(run.call_count, 2)
        self.assertNotIn("input", run.call_args_list[1].kwargs)

    def test_routine_leg_uses_one_time_mission_permit_without_parent_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_root = root / "session"
            run_id = "mission_coverage_001"
            artifact_root = session_root / "sealed_routes" / run_id
            artifact_root.mkdir(parents=True)
            route = artifact_root / "route.csv"
            diagnostics = artifact_root / "diagnostics.json"
            map_certificate = artifact_root / "map_certificate.json"
            for path, payload in (
                (route, "route\n"),
                (diagnostics, "{}\n"),
                (map_certificate, "{}\n"),
            ):
                path.write_text(payload)
            master_path = (
                session_root
                / "motion_authorization"
                / "mission_leg_motion_authorization.json"
            ).absolute()
            write_mission_leg_motion_authorization(
                master_path,
                MissionLegMotionAuthorization(
                    session_id="mission",
                    robot_id="turtlebot1",
                    namespace="",
                    cmd_vel_topic="/cmd_vel",
                    semantic_map_id="arena_1p898x3p9_auto",
                    localization_branch_proof_id=(
                        "known_start_marker_20260807"
                    ),
                    allowed_leg_kinds=ROUTINE_MISSION_LEG_KINDS,
                    scope_text=MISSION_LEG_MOTION_AUTHORIZATION_SCOPE,
                    operator_confirmation=MISSION_LEG_RUN_CONFIRMATION,
                ),
            )
            permit_path = (
                session_root
                / "motion_authorization"
                / "mission_legs"
                / f"{run_id}_permit.json"
            ).absolute()
            context = MissionLegPermitContext(
                mission_authorization_json=master_path,
                session_id="mission",
                semantic_map_id="arena_1p898x3p9_auto",
                mission_leg_kind=MissionLegKind.COVERAGE,
                mission_leg_index=1,
                target_id="survey_vp_002",
                permit_json_path=permit_path,
            )
            sealed = {
                "route_csv": str(route),
                "diagnostics_json": str(diagnostics),
                "route_certificate_json": str(map_certificate),
            }
            commands = []

            def run_process(command, **_kwargs):
                commands.append(command)
                if "--dry-run" in command:
                    _materialize_dry_child_evidence(
                        command,
                        session_root,
                        run_id,
                    )
                else:
                    _append_child_event(
                        session_root,
                        run_id,
                        {
                            "event": "motion_completed",
                            "run_id": run_id,
                            "status": "completed",
                            "stop_reason": "",
                            "stop_details": {},
                            "motion_published": True,
                        },
                    )
                return SimpleNamespace(returncode=0)

            with patch(
                "scripts.aufgabe04.real_robot."
                "autonomous_runner.runtime.subprocess.run",
                side_effect=run_process,
            ) as run, patch(
                "builtins.input",
                side_effect=AssertionError(
                    "routine leg permit must not prompt the parent"
                ),
            ):
                outcome = _run_motion_leg(
                    profile=self._profile(),
                    sealed=sealed,
                    run_id=run_id,
                    session_root=session_root,
                    execute=True,
                    coverage_plan=root / "coverage_plan.json",
                    uncertainty_map_yaml=MAP,
                    localization_branch_proof_id=(
                        "known_start_marker_20260807"
                    ),
                    mission_leg_permit_context=context,
                )
            canonical_permit_path = permit_path.resolve()
            permit = load_mission_leg_motion_permit(canonical_permit_path)

        self.assertEqual(outcome.status, "completed")
        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            outcome.mission_leg_motion_permit_path,
            canonical_permit_path,
        )
        self.assertTrue(outcome.mission_leg_motion_permit_sha256)
        self.assertEqual(permit.run_id, run_id)
        self.assertEqual(permit.mission_leg_kind, MissionLegKind.COVERAGE)
        self.assertFalse(permit.additional_typed_run_required)
        self.assertIn("--mission-leg-motion-permit-json", commands[1])
        self.assertNotIn("--runtime-localization-motion-permit-json", commands[1])
        self.assertEqual(
            commands[1][commands[1].index("--route-csv") + 1],
            str(route.resolve()),
        )
        self.assertEqual(
            commands[1][
                commands[1].index(
                    "--mission-leg-motion-authorization-json"
                )
                + 1
            ],
            str(master_path.resolve()),
        )

    def test_candidate_runtime_adapter_builds_generic_one_use_permit_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rejected_run_id = "mission_candidate_004_startup_reseal_001"
            replacement_run_id = (
                f"{rejected_run_id}_runtime_localization_reseal_001"
            )
            target_id = "survey_candidate_0005"
            identity = CandidateRoutineIdentity(
                session_id="mission",
                semantic_map_id="arena_1p898x3p9_auto",
                routine_kind=MissionLegKind.CANDIDATE_PREAPPROACH.value,
                routine_index=4,
                target_id=target_id,
                run_id=replacement_run_id,
            )
            rejected = MotionLegOutcome(
                run_id=rejected_run_id,
                status="stopped",
                stop_reason=(
                    "global localization consistency requires zero and reseal"
                ),
                stop_details=_runtime_localization_stop_details(),
                motion_published=True,
                returncode=2,
                semantic_log_path=root / "rejected.jsonl",
            )
            decision = evaluate_runtime_localization_reseal(
                status=rejected.status,
                motion_published=rejected.motion_published,
                stop_details=rejected.stop_details,
            )
            fresh_evidence = root / "fresh_localization.json"
            request = CandidateMotionLegRequest(
                sealed={
                    "route_csv": str(root / "route.csv"),
                    "diagnostics_json": str(root / "diagnostics.json"),
                    "route_certificate_json": str(root / "certificate.json"),
                },
                run_id=replacement_run_id,
                session_root=root / "session",
                candidate_snapshot_path=root / "candidate_snapshot.json",
                uncertainty_map_yaml=MAP,
                uncertainty_sigma_multiplier=2.0,
                localization_branch_proof_id="known_start_marker_20260807",
                mission_authorization_json=root / "mission_leg_auth.json",
                session_id="mission",
                semantic_map_id="arena_1p898x3p9_auto",
                mission_leg_kind=MissionLegKind.CANDIDATE_PREAPPROACH,
                mission_leg_index=4,
                target_id=target_id,
                permit_json_path=root / "routine_permit.json",
            )
            attempt = CandidateRuntimeRecoveryAttempt(
                identity=identity,
                reseal_index=1,
                rejected_outcome=rejected,
                runtime_localization_decision=decision,
                fresh_start_pose=Pose2D(0.1, 0.2, 0.3),
                attempt_root=root / "attempt",
                fresh_localization_evidence_path=fresh_evidence,
                source_root=root / "attempt" / "route_source",
            )
            sentinel = MotionLegOutcome(
                run_id=replacement_run_id,
                status="completed",
                stop_reason="",
                stop_details={},
                motion_published=True,
                returncode=0,
                semantic_log_path=root / "completed.jsonl",
            )

            with patch.object(
                autonomous_wrapper,
                "_run_motion_leg",
                return_value=sentinel,
            ) as run:
                result = (
                    _run_candidate_runtime_localization_reseal_motion_leg(
                        profile=self._profile(),
                        request=request,
                        attempt=attempt,
                        mission_motion_authorization_json=(
                            root / "mission_runtime_auth.json"
                        ),
                        max_runtime_reseals_per_leg=1,
                    )
                )

        self.assertIs(result, sentinel)
        self.assertEqual(
            run.call_args.kwargs["sensor_timing_readiness_phase"],
            autonomous_wrapper.CANDIDATE_ROUTE_SENSOR_TIMING_PHASE,
        )
        context = run.call_args.kwargs[
            "runtime_localization_permit_context"
        ]
        self.assertEqual(
            context.mission_leg_kind,
            MissionLegKind.CANDIDATE_PREAPPROACH,
        )
        self.assertEqual(context.mission_leg_index, 4)
        self.assertEqual(context.target_id, target_id)
        self.assertEqual(context.rejected_run_id, rejected_run_id)
        self.assertEqual(context.reseal_index, 1)
        self.assertTrue(run.call_args.kwargs["require_fresh_confirmation"])
        self.assertEqual(
            run.call_args.kwargs["fresh_confirmation_reason"],
            "runtime_localization",
        )
        self.assertIsNone(run.call_args.kwargs.get("coverage_transient_replan"))

    def test_runtime_localization_recovery_uses_scoped_permit_without_parent_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_root = root / "session"
            run_id = "mission_coverage_000_runtime_localization_reseal_001"
            artifact_root = session_root / "sealed_routes" / run_id
            artifact_root.mkdir(parents=True)
            route = artifact_root / "route.csv"
            diagnostics = artifact_root / "diagnostics.json"
            map_certificate = artifact_root / "map_certificate.json"
            fresh_localization = root / "fresh_localization.json"
            route.write_text(
                "leg_index,point_index,grid_x,grid_y,world_x_m,world_y_m,"
                "yaw_rad,segment_length_m,cumulative_length_m\n"
                "0,0,0,0,-0.48,-0.60,,0.0,0.0\n"
                "0,1,1,0,-0.38,-0.60,1.69,0.1,0.1\n"
            )
            diagnostics.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "planning_frame": "map",
                            "inflation_radius_m": 0.01,
                            "exact_start_connector": {
                                "required": True,
                                "validated": True,
                                "exact_start": {
                                    "x_m": -0.48,
                                    "y_m": -0.60,
                                    "yaw_rad": 1.69,
                                },
                                "anchor": {
                                    "x_m": -0.38,
                                    "y_m": -0.60,
                                    "yaw_rad": 1.69,
                                },
                                "connector_length_m": 0.1,
                                "required_clearance_m": 0.01,
                                "minimum_sampled_clearance_m": 0.5,
                                "minimum_continuous_clearance_m": 0.4975,
                                "minimum_margin_m": 0.4875,
                                "sample_spacing_m": 0.005,
                                "sample_count": 21,
                            },
                        }
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            map_certificate.write_text("{}\n")
            fresh_localization.write_text(
                json.dumps(
                    _stationary_localization_payload(-0.48, -0.60, 1.69)
                )
                + "\n"
            )
            master_path = (
                session_root
                / "motion_authorization"
                / "mission_motion_authorization.json"
            )
            write_mission_motion_authorization(
                master_path,
                MissionMotionAuthorization(
                    session_id="mission",
                    robot_id="turtlebot1",
                    namespace="",
                    cmd_vel_topic="/cmd_vel",
                    semantic_map_id="arena_1p898x3p9_auto",
                    localization_branch_proof_id="known_start_marker_20260807",
                    max_runtime_reseals_per_leg=1,
                    scope_text=MISSION_MOTION_AUTHORIZATION_SCOPE,
                    operator_confirmation=MISSION_RUN_CONFIRMATION,
                    allowed_recovery_kind=(
                        RUNTIME_LOCALIZATION_RESEAL_RECOVERY_KIND
                    ),
                ),
            )
            decision = evaluate_runtime_localization_reseal(
                status="stopped",
                motion_published=True,
                stop_details=_runtime_localization_stop_details(),
            )
            permit_path = (
                session_root
                / "motion_authorization"
                / "runtime_reseal_001_permit.json"
            )
            context = RuntimeLocalizationPermitContext(
                mission_authorization_json=master_path,
                session_id="mission",
                leg_index=0,
                target_viewpoint_id="survey_vp_001",
                reseal_index=1,
                max_runtime_reseals_per_leg=1,
                rejected_run_id="mission_coverage_000",
                runtime_reseal_decision_evidence=decision.to_evidence(),
                fresh_localization_evidence_path=fresh_localization,
                permit_json_path=permit_path,
            )
            sealed = {
                "route_csv": str(route),
                "diagnostics_json": str(diagnostics),
                "route_certificate_json": str(map_certificate),
            }
            commands = []

            def run_process(command, **_kwargs):
                commands.append(command)
                if "--dry-run" in command:
                    _materialize_dry_child_evidence(
                        command,
                        session_root,
                        run_id,
                    )
                else:
                    _append_child_event(
                        session_root,
                        run_id,
                        {
                            "event": "motion_completed",
                            "run_id": run_id,
                            "status": "completed",
                            "stop_reason": "",
                            "stop_details": {},
                            "motion_published": True,
                        },
                    )
                return SimpleNamespace(returncode=0)

            with patch(
                "scripts.aufgabe04.real_robot.autonomous_runner.runtime."
                "subprocess.run",
                side_effect=run_process,
            ) as run, patch(
                "builtins.input",
                side_effect=AssertionError(
                    "runtime localization permit must not prompt the parent"
                ),
            ):
                outcome = _run_motion_leg(
                    profile=self._profile(),
                    sealed=sealed,
                    run_id=run_id,
                    session_root=session_root,
                    execute=True,
                    coverage_plan=root / "coverage_plan.json",
                    coverage_transient_replan={
                        "survey_root": root / "survey",
                        "session_root": session_root,
                        "map_yaml": MAP,
                        "semantic_map_id": "arena_1p898x3p9_auto",
                        "target_viewpoint_id": "survey_vp_001",
                        "robot_radius_m": 0.105,
                        "max_replans": 3,
                        "leg_index": 0,
                    },
                    require_fresh_confirmation=True,
                    fresh_confirmation_reason="runtime_localization",
                    fresh_localization_evidence_path=fresh_localization,
                    uncertainty_map_yaml=MAP,
                    uncertainty_sigma_multiplier=2.25,
                    localization_branch_proof_id="known_start_marker_20260807",
                    runtime_localization_permit_context=context,
                )

            self.assertEqual(outcome.status, "completed")
            self.assertEqual(run.call_count, 2)
            canonical_permit_path = permit_path.resolve()
            self.assertEqual(
                outcome.motion_authorization_permit_path,
                canonical_permit_path,
            )
            self.assertTrue(outcome.motion_authorization_permit_sha256)
            permit = load_runtime_localization_motion_permit(
                canonical_permit_path
            )
            self.assertEqual(permit.run_id, run_id)
            self.assertFalse(permit.additional_typed_run_required)
            self.assertIn(
                "--runtime-localization-motion-permit-json",
                commands[1],
            )
            for command in commands:
                self.assertEqual(
                    command[
                        command.index("--uncertainty-sigma-multiplier") + 1
                    ],
                    "2.25",
                )
            self.assertNotIn("input", run.call_args_list[1].kwargs)
            self.assertEqual(
                commands[1][commands[1].index("--route-csv") + 1],
                str(route.resolve()),
            )
            self.assertEqual(
                commands[1][
                    commands[1].index(
                        "--mission-motion-authorization-json"
                    )
                    + 1
                ],
                str(master_path.resolve()),
            )

    def test_startup_reseal_uses_dedicated_permit_without_parent_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_root = root / "session"
            rejected_run_id = "mission_coverage_000"
            run_id = f"{rejected_run_id}_startup_reseal_001"
            artifact_root = session_root / "sealed_routes" / run_id
            artifact_root.mkdir(parents=True)
            route = artifact_root / "route.csv"
            diagnostics = artifact_root / "diagnostics.json"
            map_certificate = artifact_root / "map_certificate.json"
            fresh_localization = root / "fresh_localization.json"
            route.write_text(
                "leg_index,point_index,grid_x,grid_y,world_x_m,world_y_m,"
                "yaw_rad,segment_length_m,cumulative_length_m\n"
                "0,0,0,0,-0.48,-0.60,,0.0,0.0\n"
                "0,1,1,0,-0.38,-0.60,1.69,0.1,0.1\n"
            )
            diagnostics.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "planning_frame": "map",
                            "inflation_radius_m": 0.01,
                            "exact_start_connector": {
                                "required": True,
                                "validated": True,
                                "exact_start": {
                                    "x_m": -0.48,
                                    "y_m": -0.60,
                                    "yaw_rad": 1.69,
                                },
                                "anchor": {
                                    "x_m": -0.38,
                                    "y_m": -0.60,
                                    "yaw_rad": 1.69,
                                },
                                "connector_length_m": 0.1,
                                "required_clearance_m": 0.01,
                                "minimum_sampled_clearance_m": 0.5,
                                "minimum_continuous_clearance_m": 0.4975,
                                "minimum_margin_m": 0.4875,
                                "sample_spacing_m": 0.005,
                                "sample_count": 21,
                            },
                        }
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            map_certificate.write_text("{}\n")
            fresh_localization.write_text(
                json.dumps(
                    _stationary_localization_payload(-0.48, -0.60, 1.69)
                )
                + "\n"
            )
            rejected_log = root / "rejected.jsonl"
            rejected_log.write_text(
                json.dumps(
                    {
                        "event": "startup_route_rejected",
                        "run_id": rejected_run_id,
                        "leg_index": 0,
                        "coverage_leg_index": 0,
                        "target_viewpoint_id": "survey_vp_001",
                        "status": "stopped",
                        "stop_reason": (
                            "pose outside certified startup segment"
                        ),
                        "motion_published": False,
                        "stop_details": {
                            "reason": (
                                "pose outside certified startup segment"
                            ),
                            "source": "execution_route_certificate",
                            "phase": "before_motion_confirmation",
                            "fail_closed": True,
                        },
                    }
                )
                + "\n"
            )
            master_path = (
                session_root
                / "motion_authorization"
                / "startup_reseal_motion_authorization.json"
            ).absolute()
            write_startup_reseal_motion_authorization(
                master_path,
                StartupResealMotionAuthorization(
                    session_id="mission",
                    robot_id="turtlebot1",
                    namespace="",
                    cmd_vel_topic="/cmd_vel",
                    semantic_map_id="arena_1p898x3p9_auto",
                    localization_branch_proof_id=(
                        "known_start_marker_20260807"
                    ),
                    max_startup_reseals_per_leg=2,
                    scope_text=STARTUP_RESEAL_MOTION_AUTHORIZATION_SCOPE,
                    operator_confirmation=STARTUP_RESEAL_RUN_CONFIRMATION,
                    allowed_recovery_kind=STARTUP_RESEAL_RECOVERY_KIND,
                ),
            )
            summary_path = write_startup_reseal_permit_summary(
                root / "startup_summary.json",
                leg_index=0,
                target_viewpoint_id="survey_vp_001",
                reseal_index=1,
                rejected_run_id=rejected_run_id,
                fresh_start_x_m=-0.48,
                fresh_start_y_m=-0.60,
                fresh_start_yaw_rad=1.69,
                route_csv=route,
                diagnostics_json=diagnostics,
            )
            permit_path = (
                session_root
                / "motion_authorization"
                / "startup_reseals"
                / f"{run_id}_permit.json"
            ).absolute()
            context = StartupResealPermitContext(
                mission_authorization_json=master_path,
                session_id="mission",
                semantic_map_id="arena_1p898x3p9_auto",
                leg_index=0,
                target_viewpoint_id="survey_vp_001",
                reseal_index=1,
                max_startup_reseals_per_leg=2,
                rejected_run_id=rejected_run_id,
                rejected_semantic_log_path=rejected_log,
                startup_reseal_summary_path=summary_path,
                fresh_localization_evidence_path=fresh_localization,
                permit_json_path=permit_path,
            )
            sealed = {
                "route_csv": str(route),
                "diagnostics_json": str(diagnostics),
                "route_certificate_json": str(map_certificate),
            }
            commands = []

            def run_process(command, **_kwargs):
                commands.append(command)
                if "--dry-run" in command:
                    _materialize_dry_child_evidence(
                        command,
                        session_root,
                        run_id,
                    )
                else:
                    _append_child_event(
                        session_root,
                        run_id,
                        {
                            "event": "motion_completed",
                            "run_id": run_id,
                            "status": "completed",
                            "stop_reason": "",
                            "stop_details": {},
                            "motion_published": True,
                        },
                    )
                return SimpleNamespace(returncode=0)

            with patch(
                "scripts.aufgabe04.real_robot."
                "autonomous_runner.runtime.subprocess.run",
                side_effect=run_process,
            ) as run, patch(
                "builtins.input",
                side_effect=AssertionError(
                    "startup reseal permit must not prompt the parent"
                ),
            ):
                outcome = _run_motion_leg(
                    profile=self._profile(),
                    sealed=sealed,
                    run_id=run_id,
                    session_root=session_root,
                    execute=True,
                    coverage_plan=root / "coverage_plan.json",
                    coverage_transient_replan={
                        "survey_root": root / "survey",
                        "session_root": session_root,
                        "map_yaml": MAP,
                        "semantic_map_id": "arena_1p898x3p9_auto",
                        "target_viewpoint_id": "survey_vp_001",
                        "robot_radius_m": 0.105,
                        "max_replans": 3,
                        "leg_index": 0,
                    },
                    require_fresh_confirmation=True,
                    fresh_confirmation_reason="startup",
                    fresh_localization_evidence_path=fresh_localization,
                    uncertainty_map_yaml=MAP,
                    uncertainty_sigma_multiplier=2.0,
                    localization_branch_proof_id=(
                        "known_start_marker_20260807"
                    ),
                    startup_reseal_permit_context=context,
                )

            canonical_permit_path = permit_path.resolve()
            permit = load_startup_reseal_motion_permit(
                canonical_permit_path
            )
            self.assertEqual(outcome.status, "completed")
            self.assertEqual(run.call_count, 2)
            self.assertEqual(
                outcome.startup_reseal_motion_permit_path,
                canonical_permit_path,
            )
            self.assertTrue(outcome.startup_reseal_motion_permit_sha256)
            self.assertEqual(permit.run_id, run_id)
            self.assertEqual(permit.rejected_run_id, rejected_run_id)
            self.assertFalse(permit.additional_typed_run_required)
            self.assertIn(
                "--startup-reseal-motion-permit-json",
                commands[1],
            )
            self.assertNotIn("--mission-leg-motion-permit-json", commands[1])
            self.assertNotIn(
                "--runtime-localization-motion-permit-json",
                commands[1],
            )
            self.assertEqual(
                commands[1][commands[1].index("--route-csv") + 1],
                str(route.resolve()),
            )
            self.assertEqual(
                commands[1][
                    commands[1].index(
                        "--startup-reseal-motion-authorization-json"
                    )
                    + 1
                ],
                str(master_path.resolve()),
            )

    def test_startup_reseal_replans_full_a_star_leg_from_rejected_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            survey_root = root / "survey"
            self._planned_center_corridor(survey_root)
            plan_path = survey_root / "coverage_plan.json"
            summary = json.loads(
                (survey_root / "survey_summary.json").read_text()
            )
            current = Pose2D(0.04, 0.02, 0.1)
            rejected = MotionLegOutcome(
                run_id="rejected",
                status="stopped",
                stop_reason="pose outside certified startup segment",
                stop_details={
                    "source": "execution_route_certificate",
                    "phase": "before_motion_confirmation",
                    "route_pose": {
                        "frame_id": "map",
                        "child_frame_id": "base_footprint",
                        "x_m": current.x_m,
                        "y_m": current.y_m,
                        "yaw_rad": current.yaw_rad,
                    },
                },
                motion_published=False,
                returncode=1,
                semantic_log_path=root / "events.jsonl",
            )
            outputs = replan_startup_source(
                map_yaml=MAP,
                semantic_map_id="arena_1p898x3p9_auto",
                survey_root=survey_root,
                plan_path=plan_path,
                expected_target_viewpoint_id=summary["next_viewpoint_id"],
                current_pose=current,
                rejected_outcome=rejected,
                reseal_index=1,
                output_dir=root / "reseal_source",
            )
            leg = load_route_leg(Path(outputs["route_csv"]), 0)
            self.assertAlmostEqual(leg.raw_waypoints[0].pose.x_m, current.x_m)
            self.assertAlmostEqual(leg.raw_waypoints[0].pose.y_m, current.y_m)
            diagnostics = json.loads(
                Path(outputs["diagnostics_json"]).read_text()
            )
            self.assertTrue(diagnostics["metadata"]["startup_reseal"])
            self.assertTrue(
                diagnostics["metadata"]["exact_start_connector"]["validated"]
            )
            sealed = seal_stand_discovery_route(
                source_route_csv=Path(outputs["route_csv"]),
                source_diagnostics_json=Path(outputs["diagnostics_json"]),
                coverage_plan_path=plan_path,
                output_dir=root / "sealed_reseal",
            )
            self.assertTrue(Path(sealed["route_certificate_json"]).is_file())

    def test_runtime_localization_reseal_writes_same_target_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            survey_root = root / "survey"
            self._planned_center_corridor(survey_root)
            plan_path = survey_root / "coverage_plan.json"
            summary = json.loads(
                (survey_root / "survey_summary.json").read_text()
            )
            current = Pose2D(0.04, 0.02, 0.1)
            rejected = MotionLegOutcome(
                run_id="runtime_rejected",
                status="stopped",
                stop_reason=(
                    "global localization consistency requires zero and reseal"
                ),
                stop_details={
                    "fault_code": "localization_reseal_required",
                    "source": "global_consistency_monitor",
                    "fail_closed": True,
                },
                motion_published=True,
                returncode=1,
                semantic_log_path=root / "events.jsonl",
            )
            outputs = replan_runtime_localization_source(
                map_yaml=MAP,
                semantic_map_id="arena_1p898x3p9_auto",
                survey_root=survey_root,
                plan_path=plan_path,
                expected_target_viewpoint_id=summary["next_viewpoint_id"],
                current_pose=current,
                rejected_outcome=rejected,
                reseal_index=1,
                output_dir=root / "runtime_reseal_source",
            )
            diagnostics = json.loads(
                Path(outputs["diagnostics_json"]).read_text()
            )
            metadata = diagnostics["metadata"]
            self.assertEqual(metadata["reseal_kind"], "runtime_localization")
            self.assertTrue(metadata["runtime_localization_reseal"])
            self.assertEqual(metadata["runtime_localization_reseal_index"], 1)
            self.assertEqual(
                metadata["target_viewpoint_id"],
                summary["next_viewpoint_id"],
            )
            summary_payload = json.loads(
                Path(outputs["summary_json"]).read_text()
            )
            self.assertEqual(
                summary_payload["status"],
                "runtime_localization_route_replanned",
            )
            self.assertFalse(summary_payload["motion_published"])

    def test_opposite_inspection_offsets_never_cross_physical_minimum(self):
        offsets = bounded_approach_offsets(0.70, 0.32)
        self.assertEqual(offsets[0], 0.70)
        self.assertEqual(offsets[-1], 0.32)
        self.assertTrue(all(value >= 0.32 for value in offsets))
        self.assertEqual(tuple(sorted(offsets, reverse=True)), offsets)

    def test_opposite_face_route_is_bound_to_axis_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            survey_root = root / "survey"
            self._planned_center_corridor(survey_root)
            plan = load_coverage_survey_plan(
                survey_root / "coverage_plan.json"
            )
            registry_path = survey_root / "stand_registry.json"
            registry = load_stand_survey_registry(registry_path, plan)
            candidate = SurveyCandidate(
                candidate_uid="survey_candidate_0001",
                x_m=0.2,
                y_m=0.0,
                radius_m=0.06,
                uncertainty_m=0.02,
                keepout_radius_m=0.31,
                confidence=0.9,
                hit_count=5,
                first_seen_sec=10.0,
                last_seen_sec=12.0,
                source_observation_ids=("obs_1", "obs_2"),
                viewpoint_ids=("survey_vp_001", "survey_vp_002"),
                status=STATUS_PENDING_CAMERA,
            )
            registry = type(registry)(
                schema_version=registry.schema_version,
                survey_id=registry.survey_id,
                planning_frame=registry.planning_frame,
                map_bundle_sha256=registry.map_bundle_sha256,
                candidates=(candidate,),
            )
            write_stand_survey_registry(registry_path, registry, plan)
            snapshot = candidate_snapshot_from_registry(
                registry,
                plan,
                registry_path=registry_path,
                snapshot_id="opposite_route_snapshot",
            )
            snapshot_path = root / "candidate_snapshot.json"
            write_candidate_snapshot(snapshot_path, snapshot)
            axis_path = root / "axis_observation.json"
            axis_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "observation_kind": "real_stand_axis_without_qr",
                        "stand_id": candidate.candidate_uid,
                        "planning_frame": plan.planning_frame,
                        "stand_center": {"x_m": 0.2, "y_m": 0.0},
                        "robot_pose": {
                            "x_m": 0.2,
                            "y_m": 0.55,
                            "yaw_rad": -math.pi / 2.0,
                        },
                        "stand_axis_rad": 0.0,
                    }
                )
            )
            outputs = plan_candidate_preapproach(
                map_yaml=MAP,
                semantic_map_id="arena_1p898x3p9_auto",
                plan=plan,
                snapshot=snapshot,
                snapshot_path=snapshot_path,
                candidate_uid=candidate.candidate_uid,
                start=Pose2D(0.2, 0.55, -math.pi / 2.0),
                output_dir=root / "opposite_route",
                approach_offset_m=0.55,
                inflation_radius_m=0.25,
                candidate_transit_radius_m=0.31,
                physical_clearance={
                    "minimum_active_standoff_m": 0.26,
                    "minimum_candidate_transit_radius_m": 0.31,
                    "minimum_static_inflation_m": 0.25,
                },
                approach_normal_rad=-math.pi / 2.0,
                axis_observation_path=axis_path,
            )
            diagnostics = json.loads(
                Path(outputs["diagnostics_json"]).read_text()
            )
            self.assertEqual(
                diagnostics["metadata"]["approach_bearing_mode"],
                "camera-axis-face",
            )
            self.assertTrue(Path(outputs["route_certificate_json"]).exists())


if __name__ == "__main__":
    unittest.main()
