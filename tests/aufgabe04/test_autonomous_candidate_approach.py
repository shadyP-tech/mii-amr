import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import MissionLegKind
from scripts.aufgabe04.navigation.planning.map_io import load_occupancy_grid_with_bundle
from scripts.aufgabe04.navigation.foundation.models import (
    GridCell,
    PlanningDiagnostics,
    Pose2D,
    Route,
    RoutePoint,
)
from scripts.aufgabe04.navigation.planning.global_planner import PlanRouteResult
from scripts.aufgabe04.navigation.localization.read_current_amcl_pose import CurrentAmclPose
from scripts.aufgabe04.navigation.approach.camera_candidate_selection import (
    CameraCandidateRouteOption,
    CameraCandidateSelectionConfig,
    NoFeasibleCameraCandidateError,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_selection import (
    plan_and_select_camera_candidate,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_planning import (
    CandidatePreapproachUnreachableError,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import (
    CandidatePlanningFrame,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_reprojection import (
    CandidateFrameProvenance,
    CandidatePoint2D,
)
from scripts.aufgabe04.navigation.approach.backside_axis_frame_projection import (
    BacksideAxisFrameProjection,
    load_backside_axis_planning_observation,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
    STATUS_PENDING_CAMERA,
    CoverageSurveyConfig,
    CoverageSurveyPlan,
    StandSurveyRegistry,
    SurveyCandidate,
    SurveyViewpoint,
    stand_survey_registry_sha256,
    write_stand_survey_registry,
)
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D,
)
from scripts.aufgabe04.real_robot.candidate.approach import (
    CandidateApproachConfig,
    CandidateApproachEffects,
    CandidateApproachPoseError,
    CandidateObservation,
    CameraCandidateInitialSelection,
    CameraCandidateSelectionRequest,
    FacingValidationRequest,
    bounded_approach_offsets,
    execute_candidate_approach_phase,
    nearest_candidate,
    plan_candidate_preapproach,
    validate_facing_pose,
)
from scripts.aufgabe04.real_robot.execution.child_runner import MotionLegOutcome
from scripts.aufgabe04.real_robot.candidate.recovery_failure import (
    CandidateStartupRecoveryError,
)
from scripts.aufgabe04.real_robot.candidate.observation_deferral import (
    CandidateApproachIncompleteError,
    CandidateObservationUnavailableError,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateGeometry,
    CandidateSource,
    FrozenCandidate,
    new_candidate_snapshot,
    write_candidate_snapshot,
)
from scripts.aufgabe04.stations.station_identity_registry import (
    load_station_identity_registry,
)
from tests.aufgabe04.test_detected_station_exploration import write_free_map
from tests.aufgabe04.backside_axis_fixture import backside_axis_payload


class AutonomousCandidateApproachTest(unittest.TestCase):
    def _candidate(self, uid: str, x_m: float, y_m: float) -> FrozenCandidate:
        return FrozenCandidate(
            candidate_uid=uid,
            geometry=CandidateGeometry(
                x_m=x_m,
                y_m=y_m,
                radius_m=0.06,
                uncertainty_m=0.02,
                keepout_radius_m=0.31,
            ),
            source=CandidateSource(
                source_kind="lidar/stand_coverage_survey",
                source_artifact_sha256="a" * 64,
                detector_config_sha256="b" * 64,
                observation_ids=(f"observation_{uid}",),
            ),
            confidence=0.9,
            hit_count=4,
            first_seen_sec=1.0,
            last_seen_sec=2.0,
        )

    def _config(self, root: Path, candidates) -> CandidateApproachConfig:
        map_sha256 = "c" * 64
        snapshot = new_candidate_snapshot(
            snapshot_id="candidate_snapshot",
            created_unix_sec=3.0,
            planning_frame="map",
            map_bundle_sha256=map_sha256,
            candidates=tuple(candidates),
        )
        survey_cell = GridCell(0, 0)
        plan = CoverageSurveyPlan(
            schema_version=1,
            survey_id="survey",
            planning_frame="map",
            map_bundle_sha256=map_sha256,
            arena_bounds=ArenaBounds(),
            config=CoverageSurveyConfig(),
            viewpoints=(
                SurveyViewpoint(
                    viewpoint_id="survey_vp_001",
                    pose=Pose2D(0.0, 0.0, 0.0),
                    cell=survey_cell,
                    visible_cells=(survey_cell,),
                ),
            ),
            surveyable_cells=(survey_cell,),
            planned_covered_cells=(survey_cell,),
            planned_coverage_ratio=1.0,
        )
        return CandidateApproachConfig(
            session_root=root / "session",
            survey_root=root / "survey",
            session_id="mission",
            semantic_map_id="arena",
            planning_frame="map",
            map_yaml=root / "map.yaml",
            plan=plan,
            snapshot=snapshot,
            snapshot_path=root / "candidate_snapshot.json",
            approach_offset_m=0.70,
            inflation_radius_m=0.25,
            candidate_transit_radius_m=0.31,
            physical_clearance={
                "minimum_active_standoff_m": 0.32,
                "minimum_collision_standoff_m": 0.28,
                "minimum_candidate_transit_radius_m": 0.31,
                "minimum_static_inflation_m": 0.25,
            },
            uncertainty_sigma_multiplier=2.0,
            localization_branch_proof_id="known_start",
            mission_leg_motion_authorization_json=(
                root / "mission_leg_authorization.json"
            ),
        )

    def _write_frame_registry(
        self,
        config: CandidateApproachConfig,
        *,
        frozen_map_from_odom: PlanarTransform2D,
    ) -> CandidateApproachConfig:
        frozen_candidates = []
        for candidate in config.snapshot.candidates:
            frozen_candidates.append(
                SurveyCandidate(
                    candidate_uid=candidate.candidate_uid,
                    x_m=candidate.geometry.x_m,
                    y_m=candidate.geometry.y_m,
                    radius_m=candidate.geometry.radius_m,
                    uncertainty_m=candidate.geometry.uncertainty_m,
                    keepout_radius_m=candidate.geometry.keepout_radius_m,
                    confidence=candidate.confidence,
                    hit_count=candidate.hit_count,
                    first_seen_sec=candidate.first_seen_sec,
                    last_seen_sec=candidate.last_seen_sec,
                    source_observation_ids=candidate.source.observation_ids,
                    viewpoint_ids=("survey_vp_001",),
                    status=STATUS_PENDING_CAMERA,
                    frame_provenance=(
                        CandidateFrameProvenance.from_frozen_map_observation(
                            map_frame="map",
                            odom_frame="odom",
                            frozen_map_point=CandidatePoint2D(
                                candidate.geometry.x_m,
                                candidate.geometry.y_m,
                            ),
                            frozen_map_from_odom=frozen_map_from_odom,
                            source_evidence_id="frame_evidence",
                        )
                    ),
                )
            )
        registry = StandSurveyRegistry(
            schema_version=STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
            survey_id=config.plan.survey_id,
            planning_frame=config.planning_frame,
            map_bundle_sha256=config.plan.map_bundle_sha256,
            candidates=tuple(frozen_candidates),
        )
        write_stand_survey_registry(
            config.survey_root / "stand_registry.json",
            registry,
            config.plan,
        )
        registry_sha256 = stand_survey_registry_sha256(registry)
        snapshot = replace(
            config.snapshot,
            candidates=tuple(
                replace(
                    candidate,
                    source=replace(
                        candidate.source,
                        source_artifact_sha256=registry_sha256,
                    ),
                )
                for candidate in config.snapshot.candidates
            ),
        )
        write_candidate_snapshot(config.snapshot_path, snapshot)
        return replace(config, snapshot=snapshot)

    @staticmethod
    def _nearest_selection(
        request: CameraCandidateSelectionRequest,
    ) -> CameraCandidateInitialSelection:
        candidate = nearest_candidate(
            request.config.snapshot,
            request.current_pose,
            set(request.unresolved),
        )
        if candidate is None:
            raise RuntimeError("test selector found no unresolved candidate")
        return CameraCandidateInitialSelection(
            candidate_uid=candidate.candidate_uid,
            prepared_plan=None,
            evidence={
                "schema_version": 1,
                "selected_candidate_uid": candidate.candidate_uid,
                "selection_strategy": "test-nearest",
                "motion_authorized": False,
            },
        )

    @staticmethod
    def _completed(request) -> MotionLegOutcome:
        return MotionLegOutcome(
            run_id=request.run_id,
            status="completed",
            stop_reason="",
            stop_details={},
            motion_published=True,
            returncode=0,
            semantic_log_path=request.session_root / f"{request.run_id}.jsonl",
        )

    @staticmethod
    def _route_uncertainty_rejection(
        request,
        *,
        motion_published: bool = False,
        status: str = "preflight_failed",
        stop_reason: str | None = None,
        stop_detail_overrides: dict[str, object] | None = None,
        report_mission_leg_permit: bool = False,
    ) -> MotionLegOutcome:
        stop_reason = stop_reason or (
            "odom execution admission failed: route uncertainty budget "
            "exhausted: limiting_segment=segment:0002:0092 "
            "remaining_margin=-0.154957 m"
        )
        uncertainty_path = (
            request.session_root
            / "odom_execution"
            / f"{request.run_id}_dry_uncertainty_budget.json"
        )
        details = {
            "reason": stop_reason,
            "fault_code": "odom_execution_admission_failed",
            "execution_pose_owner": "odom",
            "global_consistency_monitor": "amcl",
            "motion_published": False,
            "fail_closed": True,
            "uncertainty_budget_accepted": False,
            "uncertainty_budget_json": str(uncertainty_path),
            "uncertainty_budget_sha256": "d" * 64,
            "route_uncertainty_limiting_segment_id": "segment:0002:0092",
            "route_uncertainty_remaining_margin_m": -0.154957,
        }
        details.update(stop_detail_overrides or {})
        return MotionLegOutcome(
            run_id=request.run_id,
            status=status,
            stop_reason=stop_reason,
            stop_details=details,
            motion_published=motion_published,
            returncode=1,
            semantic_log_path=(
                request.session_root / f"{request.run_id}.jsonl"
            ),
            dry_uncertainty_budget_path=uncertainty_path,
            mission_leg_motion_permit_path=(
                request.session_root / f"{request.run_id}_unexpected_permit.json"
                if report_mission_leg_permit
                else None
            ),
            mission_leg_motion_permit_sha256=(
                "e" * 64 if report_mission_leg_permit else ""
            ),
        )

    @staticmethod
    def _current_amcl_pose() -> CurrentAmclPose:
        return CurrentAmclPose(
            x_m=0.0,
            y_m=0.0,
            yaw_rad=0.0,
            frame_id="map",
            topic="/amcl_pose",
            header_stamp_sec=10.0,
            receipt_age_sec=0.1,
            header_age_sec=0.1,
        )

    def test_equal_distance_candidates_use_uid_tie_break(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(
                Path(tmp),
                (
                    self._candidate("candidate_b", -1.0, 0.0),
                    self._candidate("candidate_a", 1.0, 0.0),
                ),
            )

            selected = nearest_candidate(
                config.snapshot,
                Pose2D(0.0, 0.0, 0.0),
                set(config.snapshot.candidate_uids),
            )

            self.assertIsNotNone(selected)
            self.assertEqual(selected.candidate_uid, "candidate_a")

    def test_frame_bound_path_selects_and_observes_reprojected_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(
                root,
                (self._candidate("candidate_a", 2.0, 0.0),),
            )
            config = self._write_frame_registry(
                config,
                frozen_map_from_odom=PlanarTransform2D(1.0, 0.0, 0.0),
            )
            planning_frames = iter(
                (
                    CandidatePlanningFrame(
                        Pose2D(0.0, 0.0, 0.0),
                        PlanarTransform2D(0.0, 0.0, 0.0),
                    ),
                    CandidatePlanningFrame(
                        Pose2D(0.30, 0.0, 0.0),
                        PlanarTransform2D(0.0, 0.0, 0.0),
                    ),
                )
            )
            selected_geometry = []
            planned_geometry = []
            captured_geometry = []

            def select(request):
                selected_geometry.append(
                    request.config.snapshot.candidates[0].geometry.x_m
                )
                return self._nearest_selection(request)

            def plan(request):
                planned_geometry.append(
                    request.snapshot.candidates[0].geometry.x_m
                )
                self.assertIn(
                    "candidate_frame_projections",
                    str(request.snapshot_path),
                )
                return {"route_csv": "route.csv"}

            def capture(request):
                captured_geometry.append(request.candidate.geometry.x_m)
                return CandidateObservation(
                    request.output_dir / "recommendation.json",
                    "QR_A",
                    None,
                )

            outcome = execute_candidate_approach_phase(
                config,
                CandidateApproachEffects(
                    read_current_pose=lambda: Pose2D(0.30, 0.0, 0.0),
                    admit_planning_frame=lambda _path: next(planning_frames),
                    select_initial_preapproach=select,
                    plan_preapproach=plan,
                    run_motion_leg=self._completed,
                    capture_observation=capture,
                    validate_facing=lambda request: {
                        "candidate_uid": request.candidate.candidate_uid
                    },
                    commit_decision=lambda _request: None,
                    clock=lambda: 10.0,
                ),
            )

            self.assertEqual(outcome.visit_order, ("candidate_a",))
            self.assertEqual(selected_geometry, [1.0])
            self.assertEqual(planned_geometry, [1.0])
            self.assertEqual(captured_geometry, [1.0])
            self.assertTrue(
                (
                    config.session_root
                    / "candidates"
                    / "000_candidate_a"
                    / "candidate_arrival_admission.json"
                ).is_file()
            )

    def test_arrival_bearing_miss_rejects_before_camera_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = replace(
                self._config(
                    root,
                    (self._candidate("candidate_a", 2.0, 0.0),),
                ),
                max_camera_observation_attempts_per_candidate=1,
            )
            config = self._write_frame_registry(
                config,
                frozen_map_from_odom=PlanarTransform2D(1.0, 0.0, 0.0),
            )
            planning_frames = iter(
                (
                    CandidatePlanningFrame(
                        Pose2D(0.0, 0.0, 0.0),
                        PlanarTransform2D(0.0, 0.0, 0.0),
                    ),
                    CandidatePlanningFrame(
                        Pose2D(0.30, 0.0, math.radians(30.0)),
                        PlanarTransform2D(0.0, 0.0, 0.0),
                    ),
                )
            )
            capture = Mock()

            with self.assertRaises(CandidateApproachIncompleteError):
                execute_candidate_approach_phase(
                    config,
                    CandidateApproachEffects(
                        read_current_pose=lambda: Pose2D(0.30, 0.0, 0.0),
                        admit_planning_frame=lambda _path: next(
                            planning_frames
                        ),
                        select_initial_preapproach=self._nearest_selection,
                        plan_preapproach=lambda _request: {
                            "route_csv": "route.csv"
                        },
                        run_motion_leg=self._completed,
                        capture_observation=capture,
                        commit_decision=lambda _request: None,
                        clock=lambda: 10.0,
                    ),
                )

            capture.assert_not_called()
            admission_path = (
                config.session_root
                / "candidates"
                / "000_candidate_a"
                / "candidate_arrival_admission.json"
            )
            payload = json.loads(admission_path.read_text())
            self.assertFalse(payload["accepted"])
            self.assertIn(
                "bearing_error_above_maximum",
                payload["reasons"],
            )
            self.assertFalse(payload["motion_authorized"])

    def test_opposite_face_bearing_miss_rejects_second_camera_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = replace(
                self._config(
                    root,
                    (self._candidate("candidate_a", 2.0, 0.0),),
                ),
                max_camera_observation_attempts_per_candidate=1,
            )
            config = self._write_frame_registry(
                config,
                frozen_map_from_odom=PlanarTransform2D(1.0, 0.0, 0.0),
            )
            axis_path = root / "axis.json"
            axis_path.write_text(
                json.dumps(
                    backside_axis_payload(
                        stand_id="candidate_a",
                        stand_x_m=1.0,
                        robot_x_m=1.0,
                    )
                )
            )
            planning_frames = iter(
                (
                    CandidatePlanningFrame(
                        Pose2D(0.0, 0.0, 0.0),
                        PlanarTransform2D(0.0, 0.0, 0.0),
                    ),
                    CandidatePlanningFrame(
                        Pose2D(0.30, 0.0, 0.0),
                        PlanarTransform2D(0.0, 0.0, 0.0),
                    ),
                    CandidatePlanningFrame(
                        Pose2D(0.30, 0.20, 0.0),
                        PlanarTransform2D(0.0, 0.20, 0.0),
                    ),
                    CandidatePlanningFrame(
                        Pose2D(
                            1.0,
                            -0.50,
                            math.pi / 2.0 + math.radians(30.0),
                        ),
                        PlanarTransform2D(0.0, 0.20, 0.0),
                    ),
                )
            )
            capture_attempts = []
            planned_candidate_y_m = []
            planned_axis_evidence = []

            def capture(request):
                capture_attempts.append(request.attempt_index)
                if request.attempt_index == 0:
                    return CandidateObservation(None, None, axis_path)
                self.fail("second camera process started before arrival admission")

            def plan(request):
                planned_candidate_y_m.append(
                    request.snapshot.candidates[0].geometry.y_m
                )
                if request.axis_observation_path is not None:
                    planned_axis_evidence.append(
                        load_backside_axis_planning_observation(
                            request.axis_observation_path
                        )
                    )
                return {"route_csv": "route.csv"}

            with self.assertRaises(CandidateApproachIncompleteError):
                execute_candidate_approach_phase(
                    config,
                    CandidateApproachEffects(
                        read_current_pose=lambda: Pose2D(0.30, 0.0, 0.0),
                        admit_planning_frame=lambda _path: next(
                            planning_frames
                        ),
                        select_initial_preapproach=self._nearest_selection,
                        plan_preapproach=plan,
                        run_motion_leg=self._completed,
                        capture_observation=capture,
                        commit_decision=lambda _request: None,
                        clock=lambda: 10.0,
                    ),
                )

            self.assertEqual(capture_attempts, [0])
            self.assertEqual(planned_candidate_y_m, [0.0, 0.20])
            self.assertEqual(len(planned_axis_evidence), 1)
            projected_axis = planned_axis_evidence[0]
            self.assertIsInstance(projected_axis, BacksideAxisFrameProjection)
            self.assertAlmostEqual(projected_axis.stand_y_m, 0.20)
            self.assertAlmostEqual(projected_axis.robot_y_m, 0.90)
            admission_path = (
                config.session_root
                / "candidates"
                / "000_candidate_a"
                / "camera_attempt_01_arrival"
                / "admission.json"
            )
            payload = json.loads(admission_path.read_text())
            self.assertEqual(payload["observation_attempt_index"], 1)
            self.assertFalse(payload["accepted"])
            self.assertIn("bearing_error_above_maximum", payload["reasons"])
            self.assertFalse(payload["motion_authorized"])

    def test_opposite_face_startup_reseal_reprojects_axis_across_yaw_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = replace(
                self._config(
                    root,
                    (self._candidate("candidate_a", 2.0, 0.0),),
                ),
                max_startup_reseals_per_leg=1,
                startup_reseal_motion_authorization_json=(
                    root / "startup_authorization.json"
                ),
            )
            config = self._write_frame_registry(
                config,
                frozen_map_from_odom=PlanarTransform2D(1.0, 0.0, 0.0),
            )
            planning_frames = iter(
                (
                    # Selection and first camera capture use the same frame.
                    CandidatePlanningFrame(
                        Pose2D(0.0, 0.0, 0.0),
                        PlanarTransform2D(0.0, 0.0, 0.0),
                    ),
                    CandidatePlanningFrame(
                        Pose2D(0.30, 0.0, 0.0),
                        PlanarTransform2D(0.0, 0.0, 0.0),
                    ),
                    # Opposite planning first sees translation drift.
                    CandidatePlanningFrame(
                        Pose2D(0.30, 0.20, 0.0),
                        PlanarTransform2D(0.0, 0.20, 0.0),
                    ),
                    # Startup reseal then sees a 90-degree AMCL yaw change.
                    CandidatePlanningFrame(
                        Pose2D(0.30, 0.20, 0.0),
                        PlanarTransform2D(0.0, 0.0, math.pi / 2.0),
                    ),
                    CandidatePlanningFrame(
                        Pose2D(0.70, 1.0, math.pi),
                        PlanarTransform2D(0.0, 0.0, math.pi / 2.0),
                    ),
                )
            )
            axis_path = root / "axis.json"
            axis_path.write_text(
                json.dumps(
                    backside_axis_payload(
                        stand_id="candidate_a",
                        stand_x_m=1.0,
                        robot_x_m=1.0,
                        robot_y_m=0.70,
                    )
                )
            )
            axis_plans = []
            initial_motion_calls = []
            replacement_motion_calls = []

            def plan(request):
                if request.axis_observation_path is not None:
                    axis = load_backside_axis_planning_observation(
                        request.axis_observation_path
                    )
                    candidate = request.snapshot.candidates[0]
                    self.assertAlmostEqual(
                        axis.stand_x_m, candidate.geometry.x_m
                    )
                    self.assertAlmostEqual(
                        axis.stand_y_m, candidate.geometry.y_m
                    )
                    self.assertAlmostEqual(
                        axis.opposite_face_normal_rad,
                        request.approach_normal_rad,
                    )
                    axis_plans.append(axis)
                return {"route_csv": "route.csv"}

            def run_motion(request):
                initial_motion_calls.append(request)
                if request.mission_leg_kind != MissionLegKind.OPPOSITE_FACE:
                    return self._completed(request)
                return MotionLegOutcome(
                    run_id=request.run_id,
                    status="stopped",
                    stop_reason="pose outside certified startup segment",
                    stop_details={
                        "source": "execution_route_certificate",
                        "phase": "before_motion_confirmation",
                        "reason": "pose outside certified startup segment",
                        "fail_closed": True,
                        "route_pose": {
                            "x_m": 0.30,
                            "y_m": 0.20,
                            "yaw_rad": 0.0,
                        },
                    },
                    motion_published=False,
                    returncode=1,
                    semantic_log_path=root / "opposite_initial.jsonl",
                )

            def run_replacement(request, _attempt):
                replacement_motion_calls.append(request)
                return self._completed(request)

            def capture(request):
                if request.attempt_index == 0:
                    return CandidateObservation(None, None, axis_path)
                return CandidateObservation(
                    request.output_dir / "recommendation.json",
                    "QR_A",
                    None,
                )

            def admit_yaw_drift_frame(evidence_path):
                evidence_path.parent.mkdir(parents=True, exist_ok=True)
                evidence_path.write_text("{}\n", encoding="utf-8")
                return next(planning_frames)

            outcome = execute_candidate_approach_phase(
                config,
                CandidateApproachEffects(
                    read_current_pose=lambda: Pose2D(0.70, 1.0, math.pi),
                    admit_planning_frame=admit_yaw_drift_frame,
                    select_initial_preapproach=self._nearest_selection,
                    plan_preapproach=plan,
                    run_motion_leg=run_motion,
                    run_startup_reseal_motion_leg=run_replacement,
                    capture_observation=capture,
                    validate_facing=lambda request: {
                        "candidate_uid": request.candidate.candidate_uid
                    },
                    commit_decision=lambda _request: None,
                    clock=lambda: 10.0,
                ),
            )

            self.assertEqual(outcome.visit_order, ("candidate_a",))
            self.assertEqual(len(axis_plans), 2)
            self.assertTrue(
                all(
                    isinstance(axis, BacksideAxisFrameProjection)
                    for axis in axis_plans
                )
            )
            self.assertAlmostEqual(axis_plans[0].stand_y_m, 0.20)
            self.assertAlmostEqual(axis_plans[0].stand_axis_rad, 0.0)
            self.assertAlmostEqual(axis_plans[1].stand_x_m, 0.0)
            self.assertAlmostEqual(axis_plans[1].stand_y_m, 1.0)
            self.assertAlmostEqual(
                axis_plans[1].stand_axis_rad, math.pi / 2.0
            )
            self.assertAlmostEqual(
                axis_plans[1].opposite_face_normal_rad, 0.0
            )
            self.assertEqual(len(initial_motion_calls), 2)
            self.assertEqual(len(replacement_motion_calls), 1)
            identity = load_station_identity_registry(
                outcome.identity_registry_path,
                candidate_snapshot=config.snapshot,
            ).for_candidate("candidate_a")
            self.assertIsNotNone(identity)
            self.assertEqual(identity.qr_id, "QR_A")
            self.assertEqual(identity.server_station_id, "station_QR_A")

    def test_typed_observer_timeout_defers_then_retries_after_other_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(
                root,
                (
                    self._candidate("candidate_a", 0.2, 0.0),
                    self._candidate("candidate_b", 0.4, 0.0),
                ),
            )
            poses = iter(Pose2D(0.0, 0.0, 0.0) for _ in range(5))
            capture_count = {"candidate_a": 0, "candidate_b": 0}
            capture_roots = []
            motion_targets = []
            events = []

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

            def capture(request):
                uid = request.candidate.candidate_uid
                capture_count[uid] += 1
                capture_roots.append(request.output_dir.parent.name)
                if uid == "candidate_a" and capture_count[uid] == 1:
                    raise CandidateObservationUnavailableError(
                        candidate_uid=uid,
                        observation_attempt_index=request.attempt_index,
                        reason="candidate-local LiDAR association deadline",
                        process_evidence={"completion_kind": "deadline"},
                        status_evidence={"state": "lidar_target_mismatch"},
                    )
                return CandidateObservation(
                    request.output_dir / "recommendation.json",
                    f"QR_{uid}",
                    None,
                )

            def run_motion(request):
                motion_targets.append(request.target_id)
                return self._completed(request)

            outcome = execute_candidate_approach_phase(
                config,
                CandidateApproachEffects(
                    select_initial_preapproach=select,
                    read_current_pose=lambda: next(poses),
                    run_motion_leg=run_motion,
                    capture_observation=capture,
                    plan_preapproach=lambda _request: {"route_csv": "route.csv"},
                    validate_facing=lambda request: {
                        "candidate_uid": request.candidate.candidate_uid
                    },
                    commit_decision=lambda _request: None,
                    event_sink=lambda _path, payload: events.append(payload),
                    clock=lambda: 10.0,
                ),
            )

            self.assertEqual(motion_targets, ["candidate_a", "candidate_b", "candidate_a"])
            self.assertEqual(outcome.visit_order, ("candidate_b", "candidate_a"))
            self.assertEqual(
                capture_roots,
                ["000_candidate_a", "001_candidate_b", "002_candidate_a"],
            )
            self.assertEqual(capture_count, {"candidate_a": 2, "candidate_b": 1})
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

    def test_exhausted_typed_observer_timeout_fails_without_final_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = replace(
                self._config(
                    root,
                    (
                        self._candidate("candidate_a", 0.2, 0.0),
                        self._candidate("candidate_b", 0.4, 0.0),
                    ),
                ),
                max_camera_observation_attempts_per_candidate=1,
            )
            poses = iter(Pose2D(0.0, 0.0, 0.0) for _ in range(3))

            def select(request):
                uid = sorted(request.unresolved)[0]
                return CameraCandidateInitialSelection(
                    candidate_uid=uid,
                    prepared_plan=None,
                    evidence={"selected_candidate_uid": uid},
                )

            def capture(request):
                uid = request.candidate.candidate_uid
                if uid == "candidate_a":
                    raise CandidateObservationUnavailableError(
                        candidate_uid=uid,
                        observation_attempt_index=0,
                        reason="candidate-local timeout",
                        process_evidence={"completion_kind": "deadline"},
                        status_evidence={"state": "lidar_target_mismatch"},
                    )
                return CandidateObservation(
                    request.output_dir / "recommendation.json",
                    "QR_B",
                    None,
                )

            with self.assertRaises(CandidateApproachIncompleteError) as raised:
                execute_candidate_approach_phase(
                    config,
                    CandidateApproachEffects(
                        select_initial_preapproach=select,
                        read_current_pose=lambda: next(poses),
                        run_motion_leg=self._completed,
                        capture_observation=capture,
                        plan_preapproach=lambda _request: {
                            "route_csv": "route.csv"
                        },
                        validate_facing=lambda request: {
                            "candidate_uid": request.candidate.candidate_uid
                        },
                        commit_decision=lambda _request: None,
                    ),
                )

            fields = raised.exception.to_failure_fields()
            self.assertEqual(fields["resolved_candidate_uids"], ["candidate_b"])
            self.assertEqual(fields["unresolved_candidate_uids"], ["candidate_a"])
            self.assertFalse(
                (config.session_root / "station_identity_registry.json").exists()
            )
            self.assertFalse(
                (config.session_root / "stand_facing_catalog.json").exists()
            )

    def test_route_preview_selection_avoids_inside_standoff_large_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            map_yaml = write_free_map(root)
            _, map_bundle = load_occupancy_grid_with_bundle(
                map_yaml,
                semantic_map_id="arena",
                planning_frame="map",
            )
            near_inside = self._candidate("candidate_inside", 0.35, 0.45)
            forward = self._candidate("candidate_forward", 1.20, 0.0)
            config = self._config(root, (near_inside, forward))
            snapshot = new_candidate_snapshot(
                snapshot_id="candidate_snapshot",
                created_unix_sec=3.0,
                planning_frame="map",
                map_bundle_sha256=map_bundle.bundle_sha256,
                candidates=(near_inside, forward),
            )
            plan = replace(
                config.plan,
                map_bundle_sha256=map_bundle.bundle_sha256,
            )

            planned = plan_and_select_camera_candidate(
                map_yaml=map_yaml,
                semantic_map_id="arena",
                plan=plan,
                snapshot=snapshot,
                current_pose=Pose2D(0.0, 0.0, 0.0),
                unresolved=set(snapshot.candidate_uids),
                approach_offset_m=0.70,
                inflation_radius_m=0.25,
                candidate_transit_radius_m=0.31,
                physical_clearance=config.physical_clearance,
                selection_config=CameraCandidateSelectionConfig(
                    linear_speed_mps=0.055,
                    angular_speed_radps=0.18,
                ),
            )
            selection = planned.selection

            self.assertEqual(
                selection.selected_candidate_uid,
                "candidate_forward",
            )
            preview_by_uid = {
                row.candidate_uid: row.option
                for row in selection.ranked_candidates
            }
            self.assertTrue(
                preview_by_uid["candidate_inside"].inside_requested_standoff
            )
            self.assertGreater(
                preview_by_uid["candidate_inside"].initial_turn_rad,
                math.radians(100.0),
            )
            self.assertLess(
                preview_by_uid["candidate_forward"].initial_turn_rad,
                math.radians(30.0),
            )

    def test_selection_failure_blocks_planning_motion_and_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self._candidate("candidate_1", 0.2, 0.0)
            config = self._config(root, (candidate,))
            rejected = CameraCandidateRouteOption(
                candidate_uid=candidate.candidate_uid,
                feasible=False,
                failure_reason="no_path",
                route_length_m=None,
                turn_burden_rad=None,
                initial_turn_rad=None,
                inside_requested_standoff=True,
                support_class="coverage_admitted",
                confidence=candidate.confidence,
                hit_count=candidate.hit_count,
            )
            select = Mock(
                side_effect=NoFeasibleCameraCandidateError((rejected,))
            )
            plan_preapproach = Mock()
            run_motion = Mock()
            capture = Mock()
            events = []

            with self.assertRaises(NoFeasibleCameraCandidateError):
                execute_candidate_approach_phase(
                    config,
                    CandidateApproachEffects(
                        select_initial_preapproach=select,
                        read_current_pose=lambda: Pose2D(0.0, 0.0, 0.0),
                        run_motion_leg=run_motion,
                        capture_observation=capture,
                        plan_preapproach=plan_preapproach,
                        event_sink=lambda _path, payload: events.append(payload),
                    ),
                )

            select.assert_called_once()
            plan_preapproach.assert_not_called()
            run_motion.assert_not_called()
            capture.assert_not_called()
            self.assertEqual(
                events[0]["event"], "camera_candidate_selection_failed"
            )
            self.assertEqual(
                events[0]["error_code"], "no_feasible_camera_candidate"
            )
            self.assertFalse(events[0]["motion_authorized"])

    def test_preapproach_route_starts_at_exact_live_pose_and_binds_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            map_yaml = write_free_map(root)
            _, map_bundle = load_occupancy_grid_with_bundle(
                map_yaml,
                semantic_map_id="arena",
                planning_frame="map",
            )
            candidate = self._candidate("candidate_1", 0.50, 0.0)
            snapshot = new_candidate_snapshot(
                snapshot_id="snapshot",
                created_unix_sec=3.0,
                planning_frame="map",
                map_bundle_sha256=map_bundle.bundle_sha256,
                candidates=(candidate,),
            )
            snapshot_path = root / "candidate_snapshot_input.json"
            write_candidate_snapshot(snapshot_path, snapshot)
            survey_cell = GridCell(0, 0)
            plan = CoverageSurveyPlan(
                schema_version=1,
                survey_id="survey",
                planning_frame="map",
                map_bundle_sha256=map_bundle.bundle_sha256,
                arena_bounds=ArenaBounds(),
                config=CoverageSurveyConfig(snap_radius_m=0.30),
                viewpoints=(
                    SurveyViewpoint(
                        viewpoint_id="survey_vp_001",
                        pose=Pose2D(0.0, 0.0, 0.0),
                        cell=survey_cell,
                        visible_cells=(survey_cell,),
                    ),
                ),
                surveyable_cells=(survey_cell,),
                planned_covered_cells=(survey_cell,),
                planned_coverage_ratio=1.0,
            )
            live_start = Pose2D(-0.417, 0.013, 0.02)
            pipeline_root = root / "planned_candidate"

            outputs = plan_candidate_preapproach(
                map_yaml=map_yaml,
                semantic_map_id="arena",
                plan=plan,
                snapshot=snapshot,
                snapshot_path=snapshot_path,
                candidate_uid=candidate.candidate_uid,
                start=live_start,
                output_dir=pipeline_root,
                approach_offset_m=0.70,
                inflation_radius_m=0.25,
                candidate_transit_radius_m=0.31,
                physical_clearance={
                    "minimum_active_standoff_m": 0.32,
                    "minimum_candidate_transit_radius_m": 0.31,
                    "minimum_static_inflation_m": 0.25,
                },
            )

            with (pipeline_root / "route.csv").open(newline="") as handle:
                source_rows = list(csv.DictReader(handle))
            metadata = json.loads(
                (pipeline_root / "route_diagnostics.json").read_text()
            )["metadata"]
            self.assertAlmostEqual(float(source_rows[0]["world_x_m"]), live_start.x_m)
            self.assertAlmostEqual(float(source_rows[0]["world_y_m"]), live_start.y_m)
            self.assertTrue(metadata["exact_start_connector"]["validated"])
            self.assertEqual(
                metadata["route_start_pose_provenance"]["source"],
                "autonomous_candidate_current_pose",
            )
            self.assertEqual(
                metadata["route_start_pose_provenance"]["pose"]["x_m"],
                live_start.x_m,
            )
            self.assertTrue(Path(outputs["route_certificate_json"]).exists())

    def test_raw_current_amcl_pose_fails_before_candidate_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(
                Path(tmp),
                (self._candidate("candidate_1", 0.2, 0.0),),
            )
            plan = Mock()
            motion = Mock()
            capture = Mock()

            with self.assertRaises(CandidateApproachPoseError) as raised:
                execute_candidate_approach_phase(
                    config,
                    CandidateApproachEffects(
                        read_current_pose=self._current_amcl_pose,
                        run_motion_leg=motion,
                        capture_observation=capture,
                        plan_preapproach=plan,
                    ),
                )

            self.assertIn("initial_candidate_selection", str(raised.exception))
            self.assertIn("CurrentAmclPose", str(raised.exception))
            self.assertEqual(
                raised.exception.to_failure_fields()["failure_phase"],
                "candidate_approach_pose_contract",
            )
            plan.assert_not_called()
            motion.assert_not_called()
            capture.assert_not_called()

    def test_opposite_face_pose_contract_rejects_raw_amcl_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self._candidate("candidate_1", 0.2, 0.0)
            config = self._config(root, (candidate,))
            axis_path = root / "axis.json"
            axis_path.write_text(
                json.dumps(
                    backside_axis_payload(
                        stand_x_m=0.2,
                        robot_x_m=0.2,
                    )
                )
            )
            poses = iter((Pose2D(0.0, 0.0, 0.0), self._current_amcl_pose()))
            plan = Mock(return_value={"route_csv": "route.csv"})

            with self.assertRaisesRegex(
                CandidateApproachPoseError,
                "opposite_face_preapproach",
            ):
                execute_candidate_approach_phase(
                    config,
                    CandidateApproachEffects(
                        select_initial_preapproach=self._nearest_selection,
                        read_current_pose=lambda: next(poses),
                        run_motion_leg=self._completed,
                        capture_observation=lambda _request: CandidateObservation(
                            None,
                            None,
                            axis_path,
                        ),
                        plan_preapproach=plan,
                    ),
                )

            self.assertEqual(plan.call_count, 1)

    def test_stopped_pose_contract_rejects_nonfinite_navigation_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self._candidate("candidate_1", 0.2, 0.0)
            config = self._config(root, (candidate,))
            poses = iter(
                (
                    Pose2D(0.0, 0.0, 0.0),
                    Pose2D(math.nan, 0.0, 0.0),
                )
            )
            validate = Mock()
            commit = Mock()

            with self.assertRaisesRegex(
                CandidateApproachPoseError,
                "stopped_facing_validation",
            ):
                execute_candidate_approach_phase(
                    config,
                    CandidateApproachEffects(
                        select_initial_preapproach=self._nearest_selection,
                        read_current_pose=lambda: next(poses),
                        run_motion_leg=self._completed,
                        capture_observation=lambda request: CandidateObservation(
                            request.output_dir / "recommendation.json",
                            "QR_1",
                            None,
                        ),
                        plan_preapproach=lambda _request: {
                            "route_csv": "route.csv"
                        },
                        validate_facing=validate,
                        commit_decision=commit,
                    ),
                )

            validate.assert_not_called()
            commit.assert_not_called()

    def test_direct_qr_visits_nearest_first_and_preserves_exact_leg_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(
                root,
                (
                    self._candidate("candidate_far", 1.5, 0.0),
                    self._candidate("candidate_near", 0.2, 0.0),
                ),
            )
            pose_values = iter(
                (
                    Pose2D(0.0, 0.0, 0.0),
                    Pose2D(0.2, 0.0, 0.0),
                    Pose2D(0.2, 0.0, 0.0),
                    Pose2D(1.5, 0.0, 0.0),
                )
            )
            plan_requests = []
            motion_requests = []
            events = []

            def plan_preapproach(request):
                plan_requests.append(request)
                return {
                    "route_csv": str(request.output_dir / "route.csv"),
                    "diagnostics_json": str(
                        request.output_dir / "diagnostics.json"
                    ),
                    "route_certificate_json": str(
                        request.output_dir / "certificate.json"
                    ),
                }

            def run_motion(request):
                events.append(("motion", request.target_id))
                motion_requests.append(request)
                return self._completed(request)

            def capture(request):
                events.append(("observe", request.candidate.candidate_uid))
                return CandidateObservation(
                    recommendation_path=(
                        request.output_dir / "recommendation.json"
                    ),
                    qr_id=f"QR_{request.candidate.candidate_uid}",
                    axis_observation_path=None,
                )

            def validate(request):
                events.append(("validate", request.candidate.candidate_uid))
                return {"candidate_uid": request.candidate.candidate_uid}

            def commit(request):
                payload = json.loads(request.receipt_path.read_text())
                events.append(("commit", payload["candidate_uid"]))

            outcome = execute_candidate_approach_phase(
                config,
                CandidateApproachEffects(
                    select_initial_preapproach=self._nearest_selection,
                    read_current_pose=lambda: next(pose_values),
                    run_motion_leg=run_motion,
                    capture_observation=capture,
                    plan_preapproach=plan_preapproach,
                    validate_facing=validate,
                    commit_decision=commit,
                    clock=lambda: 10.0,
                ),
            )

            self.assertEqual(
                outcome.visit_order,
                ("candidate_near", "candidate_far"),
            )
            self.assertEqual(outcome.stand_count, 2)
            self.assertFalse(outcome.motion_authorized)
            self.assertTrue(outcome.identity_registry_path.is_file())
            self.assertTrue(outcome.stand_facing_catalog_path.is_file())
            self.assertEqual(
                [request.candidate_uid for request in plan_requests],
                list(outcome.visit_order),
            )
            self.assertEqual(
                [request.mission_leg_kind for request in motion_requests],
                [
                    MissionLegKind.CANDIDATE_PREAPPROACH,
                    MissionLegKind.CANDIDATE_PREAPPROACH,
                ],
            )
            for index, request in enumerate(motion_requests):
                self.assertEqual(request.mission_leg_index, index)
                self.assertEqual(request.target_id, outcome.visit_order[index])
                self.assertTrue(request.permit_json_path.is_absolute())
            for uid in outcome.visit_order:
                self.assertLess(
                    events.index(("validate", uid)),
                    events.index(("commit", uid)),
                )

    def test_direct_qr_route_preview_orders_before_motion_when_map_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            map_yaml = write_free_map(root)
            _, map_bundle = load_occupancy_grid_with_bundle(
                map_yaml,
                semantic_map_id="arena",
                planning_frame="map",
            )
            near_inside = self._candidate("candidate_inside", 0.35, 0.45)
            forward = self._candidate("candidate_forward", 1.20, 0.0)
            base = self._config(root, (near_inside, forward))
            snapshot = new_candidate_snapshot(
                snapshot_id="candidate_snapshot",
                created_unix_sec=3.0,
                planning_frame="map",
                map_bundle_sha256=map_bundle.bundle_sha256,
                candidates=(near_inside, forward),
            )
            config = replace(
                base,
                map_yaml=map_yaml,
                plan=replace(
                    base.plan,
                    map_bundle_sha256=map_bundle.bundle_sha256,
                ),
                snapshot=snapshot,
            )
            poses = iter(
                (
                    Pose2D(0.0, 0.0, 0.0),
                    Pose2D(0.50, 0.0, 0.0),
                    Pose2D(0.50, 0.0, 0.0),
                    Pose2D(-0.08, -0.10, 0.0),
                )
            )
            plan_requests = []
            selection_events = []

            nearest_without_route_preview = nearest_candidate(
                snapshot,
                Pose2D(0.0, 0.0, 0.0),
                set(snapshot.candidate_uids),
            )
            self.assertIsNotNone(nearest_without_route_preview)
            self.assertEqual(
                nearest_without_route_preview.candidate_uid,
                "candidate_inside",
            )

            def plan_preapproach(request):
                plan_requests.append(request)
                return {"route_csv": "route.csv"}

            outcome = execute_candidate_approach_phase(
                config,
                CandidateApproachEffects(
                    read_current_pose=lambda: next(poses),
                    run_motion_leg=self._completed,
                    capture_observation=lambda request: CandidateObservation(
                        request.output_dir / "recommendation.json",
                        f"QR_{request.candidate.candidate_uid}",
                        None,
                    ),
                    plan_preapproach=plan_preapproach,
                    validate_facing=lambda request: {
                        "candidate_uid": request.candidate.candidate_uid
                    },
                    commit_decision=lambda request: None,
                    event_sink=lambda _path, payload: selection_events.append(
                        payload
                    ),
                    clock=lambda: 10.0,
                ),
            )

            self.assertEqual(
                outcome.visit_order,
                ("candidate_forward", "candidate_inside"),
            )
            self.assertEqual(
                [request.candidate_uid for request in plan_requests],
                list(outcome.visit_order),
            )
            ranked_events = [
                event
                for event in selection_events
                if event["event"] == "camera_candidate_ranked"
            ]
            materialized_events = [
                event
                for event in selection_events
                if event["event"] == "camera_candidate_route_materialized"
            ]
            self.assertEqual(
                [event["selected_candidate_uid"] for event in ranked_events],
                list(outcome.visit_order),
            )
            self.assertEqual(
                [
                    event["materialized_candidate_uid"]
                    for event in materialized_events
                ],
                list(outcome.visit_order),
            )
            for uid, plan_request, ranked, materialized in zip(
                outcome.visit_order,
                plan_requests,
                ranked_events,
                materialized_events,
                strict=True,
            ):
                self.assertIn("selection_policy", ranked)
                self.assertEqual(ranked["selected_candidate_uid"], uid)
                self.assertFalse(ranked["route_materialized"])
                self.assertFalse(ranked["motion_authorized"])
                self.assertIsNotNone(plan_request.prepared_plan)
                self.assertEqual(plan_request.prepared_plan.candidate_uid, uid)
                self.assertEqual(
                    plan_request.selection_evidence["selected_candidate_uid"],
                    uid,
                )
                self.assertEqual(materialized["selected_candidate_uid"], uid)
                self.assertEqual(materialized["materialized_candidate_uid"], uid)
                self.assertTrue(
                    materialized["selected_route_reused_for_materialization"]
                )
                self.assertTrue(materialized["route_materialized"])
                self.assertFalse(materialized["motion_authorized"])

    def test_route_materialization_failure_is_logged_before_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self._candidate("candidate_1", 0.2, 0.0)
            config = self._config(root, (candidate,))
            materialize = Mock(
                side_effect=RuntimeError("prepared route binding mismatch")
            )
            run_motion = Mock()
            capture = Mock()
            events = []

            with self.assertRaisesRegex(
                RuntimeError,
                "prepared route binding mismatch",
            ):
                execute_candidate_approach_phase(
                    config,
                    CandidateApproachEffects(
                        select_initial_preapproach=self._nearest_selection,
                        read_current_pose=lambda: Pose2D(0.0, 0.0, 0.0),
                        run_motion_leg=run_motion,
                        capture_observation=capture,
                        plan_preapproach=materialize,
                        event_sink=lambda _path, payload: events.append(payload),
                        clock=lambda: 10.0,
                    ),
                )

            materialize.assert_called_once()
            run_motion.assert_not_called()
            capture.assert_not_called()
            ranked = next(
                event
                for event in events
                if event["event"] == "camera_candidate_ranked"
            )
            failed = next(
                event
                for event in events
                if event["event"]
                == "camera_candidate_route_materialization_failed"
            )
            self.assertEqual(
                ranked["selected_candidate_uid"],
                candidate.candidate_uid,
            )
            self.assertEqual(
                failed["selected_candidate_uid"],
                candidate.candidate_uid,
            )
            self.assertFalse(failed["route_materialized"])
            self.assertFalse(failed["motion_authorized"])
            self.assertEqual(failed["error_type"], "RuntimeError")
            self.assertEqual(
                failed["error_message"],
                "prepared route binding mismatch",
            )

    def test_axis_only_observation_uses_bounded_opposite_face_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self._candidate("candidate_1", 0.2, 0.0)
            config = self._config(root, (candidate,))
            axis_path = root / "axis.json"
            axis_path.write_text(
                json.dumps(
                    backside_axis_payload(
                        stand_x_m=0.2,
                        robot_x_m=0.2,
                    )
                )
            )
            poses = iter(
                (
                    Pose2D(0.0, 0.0, 0.0),
                    Pose2D(0.2, 0.7, 0.0),
                    Pose2D(0.2, -0.7, 0.0),
                )
            )
            planned_offsets = []
            motion_requests = []
            observation_requests = []

            def plan_preapproach(request):
                if request.approach_normal_rad is not None:
                    planned_offsets.append(request.approach_offset_m)
                    if len(planned_offsets) < 3:
                        raise CandidatePreapproachUnreachableError(
                            request.candidate_uid,
                            "target is blocked",
                        )
                return {"route_csv": "route.csv"}

            def run_motion(request):
                motion_requests.append(request)
                return self._completed(request)

            def capture(request):
                observation_requests.append(request)
                if request.attempt_index == 0:
                    return CandidateObservation(None, None, axis_path)
                return CandidateObservation(
                    request.output_dir / "recommendation.json",
                    "QR_1",
                    None,
                )

            outcome = execute_candidate_approach_phase(
                config,
                CandidateApproachEffects(
                    select_initial_preapproach=self._nearest_selection,
                    read_current_pose=lambda: next(poses),
                    run_motion_leg=run_motion,
                    capture_observation=capture,
                    plan_preapproach=plan_preapproach,
                    validate_facing=lambda request: {
                        "candidate_uid": request.candidate.candidate_uid
                    },
                    commit_decision=lambda request: None,
                    clock=lambda: 10.0,
                ),
            )

            self.assertEqual(planned_offsets, [0.70, 0.65, 0.60])
            self.assertTrue(all(value >= 0.32 for value in planned_offsets))
            self.assertEqual(
                [request.mission_leg_kind for request in motion_requests],
                [
                    MissionLegKind.CANDIDATE_PREAPPROACH,
                    MissionLegKind.OPPOSITE_FACE,
                ],
            )
            self.assertEqual(
                [request.mission_leg_index for request in motion_requests],
                [0, 0],
            )
            self.assertEqual(
                [request.target_id for request in motion_requests],
                [candidate.candidate_uid, candidate.candidate_uid],
            )
            self.assertEqual(
                [request.attempt_index for request in observation_requests],
                [0, 1],
            )
            self.assertEqual(outcome.visit_order, (candidate.candidate_uid,))

    def test_opposite_face_uncertainty_rejection_tries_smaller_standoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self._candidate("candidate_1", 0.2, 0.0)
            config = self._config(root, (candidate,))
            axis_path = root / "axis.json"
            axis_path.write_text(
                json.dumps(
                    backside_axis_payload(
                        stand_x_m=0.2,
                        robot_x_m=0.2,
                    )
                )
            )
            poses = iter(
                (
                    Pose2D(0.0, 0.0, 0.0),
                    Pose2D(0.2, 0.7, 0.0),
                    Pose2D(0.2, -0.55, 0.0),
                )
            )
            opposite_plan_requests = []
            opposite_motion_requests = []

            def plan_preapproach(request):
                if request.approach_normal_rad is None:
                    return {"route_csv": "primary.csv"}
                opposite_plan_requests.append(request)
                if request.approach_offset_m in (0.70, 0.65):
                    raise CandidatePreapproachUnreachableError(
                        request.candidate_uid,
                        "target is blocked",
                    )
                return {
                    "route_csv": (
                        f"opposite_{request.approach_offset_m:.2f}.csv"
                    ),
                    "test_approach_offset_m": (
                        f"{request.approach_offset_m:.2f}"
                    ),
                }

            def run_motion(request):
                if request.mission_leg_kind is not MissionLegKind.OPPOSITE_FACE:
                    return self._completed(request)
                opposite_motion_requests.append(request)
                if request.sealed["test_approach_offset_m"] == "0.60":
                    return self._route_uncertainty_rejection(request)
                return self._completed(request)

            def capture(request):
                if request.attempt_index == 0:
                    return CandidateObservation(None, None, axis_path)
                return CandidateObservation(
                    request.output_dir / "recommendation.json",
                    "QR_1",
                    None,
                )

            outcome = execute_candidate_approach_phase(
                config,
                CandidateApproachEffects(
                    select_initial_preapproach=self._nearest_selection,
                    read_current_pose=lambda: next(poses),
                    run_motion_leg=run_motion,
                    capture_observation=capture,
                    plan_preapproach=plan_preapproach,
                    validate_facing=lambda request: {
                        "candidate_uid": request.candidate.candidate_uid
                    },
                    commit_decision=lambda _request: None,
                    clock=lambda: 10.0,
                ),
            )

            self.assertEqual(
                [
                    request.approach_offset_m
                    for request in opposite_plan_requests[:4]
                ],
                [0.70, 0.65, 0.60, 0.55],
            )
            self.assertEqual(
                [
                    float(request.sealed["test_approach_offset_m"])
                    for request in opposite_motion_requests
                ],
                [0.60, 0.55],
            )
            self.assertEqual(
                len({request.output_dir for request in opposite_plan_requests}),
                len(opposite_plan_requests),
            )
            self.assertEqual(
                len({request.run_id for request in opposite_motion_requests}),
                2,
            )
            self.assertEqual(
                len(
                    {
                        request.permit_json_path
                        for request in opposite_motion_requests
                    }
                ),
                2,
            )
            self.assertEqual(outcome.visit_order, (candidate.candidate_uid,))

    def test_opposite_face_fallback_rejects_unsafe_or_inexact_outcome(self):
        cases = (
            ("motion_published", {"motion_published": True}),
            (
                "permit_reported",
                {"report_mission_leg_permit": True},
            ),
            (
                "wrong_fault",
                {
                    "stop_detail_overrides": {
                        "fault_code": "some_other_failure"
                    }
                },
            ),
            (
                "budget_claimed_accepted",
                {
                    "stop_detail_overrides": {
                        "uncertainty_budget_accepted": True
                    }
                },
            ),
        )
        for name, rejection_kwargs in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                candidate = self._candidate("candidate_1", 0.2, 0.0)
                config = self._config(root, (candidate,))
                axis_path = root / "axis.json"
                axis_path.write_text(
                    json.dumps(
                        backside_axis_payload(
                            stand_x_m=0.2,
                            robot_x_m=0.2,
                        )
                    )
                )
                poses = iter(
                    (
                        Pose2D(0.0, 0.0, 0.0),
                        Pose2D(0.2, 0.7, 0.0),
                    )
                )
                opposite_plan_requests = []
                capture = Mock(
                    return_value=CandidateObservation(None, None, axis_path)
                )

                def plan_preapproach(request):
                    if request.approach_normal_rad is None:
                        return {"route_csv": "primary.csv"}
                    opposite_plan_requests.append(request)
                    return {
                        "route_csv": "opposite.csv",
                        "test_approach_offset_m": (
                            f"{request.approach_offset_m:.2f}"
                        ),
                    }

                def run_motion(request):
                    if (
                        request.mission_leg_kind
                        is not MissionLegKind.OPPOSITE_FACE
                    ):
                        return self._completed(request)
                    return self._route_uncertainty_rejection(
                        request,
                        **rejection_kwargs,
                    )

                with self.assertRaises(RuntimeError):
                    execute_candidate_approach_phase(
                        config,
                        CandidateApproachEffects(
                            select_initial_preapproach=self._nearest_selection,
                            read_current_pose=lambda: next(poses),
                            run_motion_leg=run_motion,
                            capture_observation=capture,
                            plan_preapproach=plan_preapproach,
                            commit_decision=lambda _request: None,
                        ),
                    )

                self.assertEqual(
                    [
                        request.approach_offset_m
                        for request in opposite_plan_requests
                    ],
                    [0.70],
                )
                capture.assert_called_once()

    def test_opposite_face_uncertainty_fallback_exhaustion_is_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self._candidate("candidate_1", 0.2, 0.0)
            config = self._config(root, (candidate,))
            axis_path = root / "axis.json"
            axis_path.write_text(
                json.dumps(
                    backside_axis_payload(
                        stand_x_m=0.2,
                        robot_x_m=0.2,
                    )
                )
            )
            poses = iter(
                (
                    Pose2D(0.0, 0.0, 0.0),
                    Pose2D(0.2, 0.7, 0.0),
                )
            )
            opposite_plan_requests = []
            opposite_motion_requests = []
            capture = Mock(
                return_value=CandidateObservation(None, None, axis_path)
            )
            commit = Mock()

            def plan_preapproach(request):
                if request.approach_normal_rad is None:
                    return {"route_csv": "primary.csv"}
                opposite_plan_requests.append(request)
                return {
                    "route_csv": (
                        f"opposite_{request.approach_offset_m:.2f}.csv"
                    ),
                    "test_approach_offset_m": (
                        f"{request.approach_offset_m:.2f}"
                    ),
                }

            def run_motion(request):
                if request.mission_leg_kind is not MissionLegKind.OPPOSITE_FACE:
                    return self._completed(request)
                opposite_motion_requests.append(request)
                return self._route_uncertainty_rejection(request)

            with self.assertRaises(RuntimeError):
                execute_candidate_approach_phase(
                    config,
                    CandidateApproachEffects(
                        select_initial_preapproach=self._nearest_selection,
                        read_current_pose=lambda: next(poses),
                        run_motion_leg=run_motion,
                        capture_observation=capture,
                        plan_preapproach=plan_preapproach,
                        commit_decision=commit,
                    ),
                )

            expected_offsets = list(
                bounded_approach_offsets(
                    config.approach_offset_m,
                    config.physical_clearance["minimum_active_standoff_m"],
                )
            )
            self.assertEqual(
                [
                    float(request.sealed["test_approach_offset_m"])
                    for request in opposite_motion_requests
                ],
                expected_offsets,
            )
            self.assertEqual(
                len({request.output_dir for request in opposite_plan_requests}),
                len(expected_offsets),
            )
            self.assertEqual(
                len({request.run_id for request in opposite_motion_requests}),
                len(expected_offsets),
            )
            capture.assert_called_once()
            commit.assert_not_called()
            self.assertFalse(
                (config.session_root / "station_identity_registry.json").exists()
            )

    def test_direct_uncertainty_rejection_does_not_enter_standoff_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self._candidate("candidate_1", 0.2, 0.0)
            config = self._config(root, (candidate,))
            plan_requests = []
            capture = Mock()

            def plan_preapproach(request):
                plan_requests.append(request)
                return {"route_csv": "primary.csv"}

            with self.assertRaises(CandidateStartupRecoveryError):
                execute_candidate_approach_phase(
                    config,
                    CandidateApproachEffects(
                        select_initial_preapproach=self._nearest_selection,
                        read_current_pose=lambda: Pose2D(0.0, 0.0, 0.0),
                        run_motion_leg=self._route_uncertainty_rejection,
                        capture_observation=capture,
                        plan_preapproach=plan_preapproach,
                    ),
                )

            self.assertEqual(len(plan_requests), 1)
            self.assertIsNone(plan_requests[0].approach_normal_rad)
            capture.assert_not_called()

    def test_motion_outcome_identity_mismatch_blocks_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(
                root,
                (self._candidate("candidate_1", 0.2, 0.0),),
            )
            capture = Mock()

            def mismatched_outcome(request):
                return MotionLegOutcome(
                    run_id="another_run",
                    status="completed",
                    stop_reason="",
                    stop_details={},
                    motion_published=True,
                    returncode=0,
                    semantic_log_path=root / "events.jsonl",
                )

            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                execute_candidate_approach_phase(
                    config,
                    CandidateApproachEffects(
                        select_initial_preapproach=self._nearest_selection,
                        read_current_pose=lambda: Pose2D(0.0, 0.0, 0.0),
                        run_motion_leg=mismatched_outcome,
                        capture_observation=capture,
                        plan_preapproach=lambda request: {
                            "route_csv": "route.csv"
                        },
                    ),
                )

            capture.assert_not_called()
            self.assertFalse(
                (config.session_root / "station_identity_registry.json").exists()
            )

    def test_startup_mismatch_replans_same_candidate_without_new_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Pin the workstation regression: the original candidate route
            # started 33.1768 mm from AMCL, just outside the 30 mm route tube.
            rejected_start_offset_m = 0.03317680255718219
            candidate = self._candidate("candidate_1", 0.2, 0.0)
            config = replace(
                self._config(root, (candidate,)),
                max_startup_reseals_per_leg=1,
                startup_reseal_motion_authorization_json=(
                    root / "startup_authorization.json"
                ),
            )
            poses = iter(
                (Pose2D(0.0, 0.0, 0.0), Pose2D(0.2, 0.0, 0.0))
            )
            plan_requests = []
            replacement_attempts = []
            capture = Mock(
                side_effect=lambda request: CandidateObservation(
                    request.output_dir / "recommendation.json",
                    "QR_1",
                    None,
                )
            )

            def plan_preapproach(request):
                request.output_dir.mkdir(parents=True, exist_ok=False)
                plan_requests.append(request)
                return {
                    "route_csv": str(request.output_dir / "route.csv"),
                    "diagnostics_json": str(
                        request.output_dir / "route_diagnostics.json"
                    ),
                    "route_certificate_json": str(
                        request.output_dir / "route_certificate.json"
                    ),
                }

            def initial_motion(request):
                return MotionLegOutcome(
                    run_id=request.run_id,
                    status="stopped",
                    stop_reason="pose outside certified startup segment",
                    stop_details={
                        "source": "execution_route_certificate",
                        "phase": "before_motion_confirmation",
                        "reason": "pose outside certified startup segment",
                        "fail_closed": True,
                        "route_pose": {
                            "x_m": rejected_start_offset_m,
                            "y_m": 0.0,
                            "yaw_rad": 0.0,
                        },
                    },
                    motion_published=False,
                    returncode=1,
                    semantic_log_path=root / "initial_events.jsonl",
                    mission_leg_motion_permit_path=root / "initial_permit.json",
                    mission_leg_motion_permit_sha256="a" * 64,
                )

            def admit(evidence_path):
                evidence_path.parent.mkdir(parents=True, exist_ok=True)
                evidence_path.write_text("{}\n")
                return Pose2D(rejected_start_offset_m, 0.0, 0.0)

            def replacement_motion(request, attempt):
                replacement_attempts.append((request, attempt))
                return self._completed(request)

            with patch("builtins.input") as prompt:
                outcome = execute_candidate_approach_phase(
                    config,
                    CandidateApproachEffects(
                        select_initial_preapproach=self._nearest_selection,
                        read_current_pose=lambda: next(poses),
                        run_motion_leg=initial_motion,
                        run_startup_reseal_motion_leg=replacement_motion,
                        admit_startup_localization=admit,
                        capture_observation=capture,
                        plan_preapproach=plan_preapproach,
                        validate_facing=lambda request: {
                            "candidate_uid": request.candidate.candidate_uid
                        },
                        commit_decision=lambda request: None,
                        clock=lambda: 10.0,
                    ),
                )

            prompt.assert_not_called()
            self.assertEqual(outcome.visit_order, (candidate.candidate_uid,))
            self.assertEqual(len(plan_requests), 2)
            self.assertEqual(
                plan_requests[1].start,
                Pose2D(rejected_start_offset_m, 0.0, 0.0),
            )
            self.assertIn("startup_reseal_001", str(plan_requests[1].output_dir))
            self.assertEqual(len(replacement_attempts), 1)
            replacement_request, attempt = replacement_attempts[0]
            self.assertEqual(
                replacement_request.mission_leg_kind,
                MissionLegKind.CANDIDATE_PREAPPROACH,
            )
            self.assertEqual(replacement_request.target_id, candidate.candidate_uid)
            self.assertEqual(attempt.reseal_index, 1)
            self.assertTrue(replacement_request.run_id.endswith("startup_reseal_001"))
            capture.assert_called_once()

    def test_frame_bound_startup_reseal_reprojects_candidate_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self._candidate("candidate_1", 2.0, 0.0)
            config = replace(
                self._config(root, (candidate,)),
                max_startup_reseals_per_leg=1,
                startup_reseal_motion_authorization_json=(
                    root / "startup_authorization.json"
                ),
            )
            config = self._write_frame_registry(
                config,
                frozen_map_from_odom=PlanarTransform2D(1.0, 0.0, 0.0),
            )
            planning_frames = iter(
                (
                    CandidatePlanningFrame(
                        Pose2D(0.0, 0.0, 0.0),
                        PlanarTransform2D(0.0, 0.0, 0.0),
                    ),
                    CandidatePlanningFrame(
                        Pose2D(0.0332, 0.20, 0.0),
                        PlanarTransform2D(0.0, 0.20, 0.0),
                    ),
                    CandidatePlanningFrame(
                        Pose2D(0.30, 0.20, 0.0),
                        PlanarTransform2D(0.0, 0.20, 0.0),
                    ),
                )
            )
            plan_requests = []
            replacement_requests = []

            def plan(request):
                request.output_dir.mkdir(parents=True, exist_ok=True)
                plan_requests.append(request)
                return {"route_csv": str(request.output_dir / "route.csv")}

            def initial_motion(request):
                return MotionLegOutcome(
                    run_id=request.run_id,
                    status="stopped",
                    stop_reason="pose outside certified startup segment",
                    stop_details={
                        "source": "execution_route_certificate",
                        "phase": "before_motion_confirmation",
                        "reason": "pose outside certified startup segment",
                        "fail_closed": True,
                        "route_pose": {
                            "x_m": 0.0332,
                            "y_m": 0.20,
                            "yaw_rad": 0.0,
                        },
                    },
                    motion_published=False,
                    returncode=1,
                    semantic_log_path=root / "initial_events.jsonl",
                )

            def replacement_motion(request, _attempt):
                replacement_requests.append(request)
                return self._completed(request)

            def admit_planning_frame(evidence_path):
                evidence_path.parent.mkdir(parents=True, exist_ok=True)
                evidence_path.write_text("{}\n", encoding="utf-8")
                return next(planning_frames)

            outcome = execute_candidate_approach_phase(
                config,
                CandidateApproachEffects(
                    read_current_pose=lambda: Pose2D(0.30, 0.20, 0.0),
                    admit_planning_frame=admit_planning_frame,
                    select_initial_preapproach=self._nearest_selection,
                    plan_preapproach=plan,
                    run_motion_leg=initial_motion,
                    run_startup_reseal_motion_leg=replacement_motion,
                    capture_observation=lambda request: CandidateObservation(
                        request.output_dir / "recommendation.json",
                        "QR_1",
                        None,
                    ),
                    validate_facing=lambda request: {
                        "candidate_uid": request.candidate.candidate_uid
                    },
                    commit_decision=lambda _request: None,
                    clock=lambda: 10.0,
                ),
            )

            self.assertEqual(outcome.visit_order, (candidate.candidate_uid,))
            self.assertEqual(
                [
                    request.snapshot.candidates[0].geometry.y_m
                    for request in plan_requests
                ],
                [0.0, 0.20],
            )
            self.assertEqual(len(replacement_requests), 1)
            self.assertIn(
                "candidate_frame_projection",
                str(plan_requests[1].snapshot_path),
            )
            self.assertIn(
                "/route/",
                str(replacement_requests[0].candidate_snapshot_path),
            )

    def test_startup_replacement_runtime_reseal_continues_to_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self._candidate("candidate_1", 0.2, 0.0)
            mission_authorization = root / "mission_authorization.json"
            mission_authorization.write_text("{}\n", encoding="utf-8")
            config = replace(
                self._config(root, (candidate,)),
                max_startup_reseals_per_leg=1,
                startup_reseal_motion_authorization_json=(
                    root / "startup_authorization.json"
                ),
                mission_motion_authorization_json=mission_authorization,
                max_runtime_localization_reseals_per_leg=1,
            )
            poses = iter(
                (Pose2D(0.0, 0.0, 0.0), Pose2D(0.2, 0.0, 0.0))
            )
            plan_requests = []
            startup_attempts = []
            runtime_attempts = []
            events = []
            capture = Mock(
                side_effect=lambda request: CandidateObservation(
                    request.output_dir / "recommendation.json",
                    "QR_1",
                    None,
                )
            )

            def plan_preapproach(request):
                request.output_dir.mkdir(parents=True, exist_ok=False)
                plan_requests.append(request)
                return {
                    "route_csv": str(request.output_dir / "route.csv"),
                    "diagnostics_json": str(
                        request.output_dir / "route_diagnostics.json"
                    ),
                    "route_certificate_json": str(
                        request.output_dir / "route_certificate.json"
                    ),
                }

            def initial_motion(request):
                return MotionLegOutcome(
                    run_id=request.run_id,
                    status="stopped",
                    stop_reason="pose outside certified startup segment",
                    stop_details={
                        "source": "execution_route_certificate",
                        "phase": "before_motion_confirmation",
                        "reason": "pose outside certified startup segment",
                        "fail_closed": True,
                        "route_pose": {
                            "x_m": 0.0332,
                            "y_m": 0.0,
                            "yaw_rad": 0.0,
                        },
                    },
                    motion_published=False,
                    returncode=1,
                    semantic_log_path=root / "initial_events.jsonl",
                )

            def admit(evidence_path):
                evidence_path.parent.mkdir(parents=True, exist_ok=False)
                evidence_path.write_text("{}\n", encoding="utf-8")
                return Pose2D(0.0332, 0.0, 0.0)

            def startup_motion(request, attempt):
                startup_attempts.append((request, attempt))
                permit = root / "permits" / f"{request.run_id}_startup.json"
                permit.parent.mkdir(parents=True, exist_ok=True)
                permit.write_text("{}\n", encoding="utf-8")
                return MotionLegOutcome(
                    run_id=request.run_id,
                    status="stopped",
                    stop_reason=(
                        "global localization consistency requires zero and reseal"
                    ),
                    stop_details={
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
                            "reason": "map_from_odom_translation_drift",
                            "fail_closed": True,
                        },
                    },
                    motion_published=True,
                    returncode=2,
                    semantic_log_path=root / "startup_events.jsonl",
                    startup_reseal_motion_permit_path=permit.resolve(),
                    startup_reseal_motion_permit_sha256="a" * 64,
                )

            def runtime_motion(request, attempt):
                runtime_attempts.append((request, attempt))
                permit = root / "permits" / f"{request.run_id}_runtime.json"
                permit.parent.mkdir(parents=True, exist_ok=True)
                permit.write_text("{}\n", encoding="utf-8")
                return replace(
                    self._completed(request),
                    motion_authorization_permit_path=permit.resolve(),
                    motion_authorization_permit_sha256="b" * 64,
                )

            with patch("builtins.input") as prompt:
                outcome = execute_candidate_approach_phase(
                    config,
                    CandidateApproachEffects(
                        select_initial_preapproach=self._nearest_selection,
                        read_current_pose=lambda: next(poses),
                        run_motion_leg=initial_motion,
                        run_startup_reseal_motion_leg=startup_motion,
                        admit_startup_localization=admit,
                        run_runtime_localization_reseal_motion_leg=(
                            runtime_motion
                        ),
                        admit_runtime_localization=admit,
                        capture_observation=capture,
                        plan_preapproach=plan_preapproach,
                        validate_facing=lambda request: {
                            "candidate_uid": request.candidate.candidate_uid
                        },
                        commit_decision=lambda request: None,
                        event_sink=lambda _path, payload: events.append(payload),
                        clock=lambda: 10.0,
                    ),
                )

            prompt.assert_not_called()
            self.assertEqual(outcome.visit_order, (candidate.candidate_uid,))
            self.assertEqual(len(plan_requests), 3)
            self.assertEqual(len(startup_attempts), 1)
            self.assertEqual(len(runtime_attempts), 1)
            runtime_request, runtime_attempt = runtime_attempts[0]
            self.assertEqual(
                runtime_request.mission_leg_kind,
                MissionLegKind.CANDIDATE_PREAPPROACH,
            )
            self.assertEqual(runtime_request.mission_leg_index, 0)
            self.assertEqual(runtime_request.target_id, candidate.candidate_uid)
            self.assertEqual(
                runtime_attempt.identity.routine_kind,
                MissionLegKind.CANDIDATE_PREAPPROACH.value,
            )
            self.assertEqual(runtime_attempt.identity.routine_index, 0)
            self.assertEqual(
                runtime_attempt.identity.target_id,
                candidate.candidate_uid,
            )
            self.assertEqual(runtime_attempt.reseal_index, 1)
            self.assertIn(
                "startup_reseal_001_runtime_localization_reseal_001",
                runtime_request.run_id,
            )
            self.assertEqual(
                [
                    event["event"]
                    for event in events
                    if str(event.get("event", "")).startswith(
                        "candidate_runtime_localization"
                    )
                ],
                [
                    "candidate_runtime_localization_handoff_ready",
                    "candidate_runtime_localization_reseal_started",
                    "candidate_runtime_localization_admitted",
                    "candidate_runtime_localization_route_replanned",
                    "candidate_runtime_localization_permit_evidenced",
                    "candidate_runtime_localization_reseal_completed",
                ],
            )
            capture.assert_called_once()

    def test_exhausted_opposite_offsets_publish_no_identity_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self._candidate("candidate_1", 0.2, 0.0)
            config = self._config(root, (candidate,))
            axis_path = root / "axis.json"
            axis_path.write_text(
                json.dumps(
                    backside_axis_payload(
                        stand_x_m=0.2,
                        robot_x_m=0.2,
                    )
                )
            )
            poses = iter(
                (Pose2D(0.0, 0.0, 0.0), Pose2D(0.2, 0.7, 0.0))
            )
            attempted_offsets = []
            commit = Mock()

            def plan_preapproach(request):
                if request.approach_normal_rad is None:
                    return {"route_csv": "primary.csv"}
                attempted_offsets.append(request.approach_offset_m)
                raise ValueError(
                    "candidate pre-approach A* failed: target is blocked"
                )

            with self.assertRaisesRegex(
                RuntimeError,
                "no physically allowed opposite-face approach",
            ):
                execute_candidate_approach_phase(
                    config,
                    CandidateApproachEffects(
                        select_initial_preapproach=self._nearest_selection,
                        read_current_pose=lambda: next(poses),
                        run_motion_leg=self._completed,
                        capture_observation=lambda request: CandidateObservation(
                            None,
                            None,
                            axis_path,
                        ),
                        plan_preapproach=plan_preapproach,
                        commit_decision=commit,
                    ),
                )

            self.assertEqual(attempted_offsets[0], 0.70)
            self.assertEqual(attempted_offsets[-1], 0.32)
            self.assertTrue(all(value >= 0.32 for value in attempted_offsets))
            commit.assert_not_called()
            self.assertFalse(
                (config.session_root / "station_identity_registry.json").exists()
            )
            self.assertFalse(
                (config.session_root / "stand_facing_catalog.json").exists()
            )

    def test_facing_validation_failure_prevents_decision_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(
                root,
                (self._candidate("candidate_1", 0.2, 0.0),),
            )
            poses = iter(
                (Pose2D(0.0, 0.0, 0.0), Pose2D(0.2, 0.0, 0.0))
            )
            commit = Mock()

            with self.assertRaisesRegex(RuntimeError, "facing invalid"):
                execute_candidate_approach_phase(
                    config,
                    CandidateApproachEffects(
                        select_initial_preapproach=self._nearest_selection,
                        read_current_pose=lambda: next(poses),
                        run_motion_leg=self._completed,
                        capture_observation=lambda request: CandidateObservation(
                            request.output_dir / "recommendation.json",
                            "QR_1",
                            None,
                        ),
                        plan_preapproach=lambda request: {
                            "route_csv": "route.csv"
                        },
                        validate_facing=lambda request: (_ for _ in ()).throw(
                            RuntimeError("facing invalid")
                        ),
                        commit_decision=commit,
                    ),
                )

            commit.assert_not_called()
            self.assertFalse(
                (config.session_root / "station_identity_registry.json").exists()
            )
            self.assertFalse(
                (config.session_root / "stand_facing_catalog.json").exists()
            )

    def test_facing_validation_rejects_cross_candidate_recommendation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self._candidate("candidate_1", 0.2, 0.0)
            config = self._config(root, (candidate,))
            recommendation_path = root / "recommendation.json"

            with patch(
                "scripts.aufgabe04.real_robot.candidate."
                "approach.load_recommendation",
                return_value=SimpleNamespace(stand_id="candidate_2"),
            ):
                with self.assertRaisesRegex(ValueError, "stand_id mismatch"):
                    validate_facing_pose(
                        FacingValidationRequest(
                            config=config,
                            candidate=candidate,
                            recommendation_path=recommendation_path,
                            current_pose=Pose2D(0.0, 0.0, 0.0),
                            output_dir=root / "facing",
                        )
                    )

    def test_facing_validation_rejects_target_inside_active_standoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self._candidate("candidate_1", 0.0, 0.0)
            config = self._config(root, (candidate,))
            recommendation = SimpleNamespace(
                stand_id=candidate.candidate_uid,
                material_target=SimpleNamespace(
                    face_id="qr_face",
                    pose=Pose2D(0.31, 0.0, math.pi),
                ),
            )

            with (
                patch(
                    "scripts.aufgabe04.real_robot.candidate."
                    "approach.load_recommendation",
                    return_value=recommendation,
                ),
                patch(
                    "scripts.aufgabe04.real_robot.candidate."
                    "approach.load_occupancy_grid_with_bundle"
                ) as load_map,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "violates the active-stand standoff",
                ):
                    validate_facing_pose(
                        FacingValidationRequest(
                            config=config,
                            candidate=candidate,
                            recommendation_path=root / "recommendation.json",
                            current_pose=Pose2D(-0.7, 0.0, 0.0),
                            output_dir=root / "facing",
                        )
                    )

            load_map.assert_not_called()
            self.assertFalse((root / "facing").exists())

    def test_measured_model_facing_target_remains_reachable_at_035_m(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            map_yaml = write_free_map(
                root,
                width=80,
                height=40,
                resolution=0.05,
            )
            _, map_bundle = load_occupancy_grid_with_bundle(
                map_yaml,
                semantic_map_id="arena",
                planning_frame="map",
            )
            candidate = self._candidate("candidate_1", 0.0, 0.0)
            base_config = self._config(root, (candidate,))
            config = replace(
                base_config,
                map_yaml=map_yaml,
                plan=replace(
                    base_config.plan,
                    map_bundle_sha256=map_bundle.bundle_sha256,
                ),
                physical_clearance={
                    "minimum_active_standoff_m": 0.33,
                    "minimum_collision_standoff_m": 0.285,
                    "minimum_candidate_transit_radius_m": 0.34,
                    "minimum_static_inflation_m": 0.25,
                },
            )
            start = Pose2D(-0.70, 0.0, 0.0)
            target = Pose2D(-0.35, 0.0, 0.0)
            recommendation = SimpleNamespace(
                stand_id=candidate.candidate_uid,
                material_target=SimpleNamespace(
                    face_id="qr_face",
                    pose=target,
                ),
                face_candidates=(
                    SimpleNamespace(
                        face_id="qr_face",
                        outward_normal_rad=math.pi,
                    ),
                ),
                axis_confidence=0.95,
                axis_sample_count=7,
            )

            with patch(
                "scripts.aufgabe04.real_robot.candidate."
                "approach.load_recommendation",
                return_value=recommendation,
            ):
                result = validate_facing_pose(
                    FacingValidationRequest(
                        config=config,
                        candidate=candidate,
                        recommendation_path=root / "recommendation.json",
                        current_pose=start,
                        output_dir=root / "facing",
                    )
                )

            clearance = result["active_stand_clearance"]
            self.assertAlmostEqual(clearance["target_center_standoff_m"], 0.35)
            self.assertEqual(clearance["minimum_active_standoff_m"], 0.33)
            self.assertEqual(
                clearance["minimum_collision_standoff_m"],
                0.285,
            )
            self.assertGreaterEqual(
                clearance["route_centerline_minimum_standoff_m"] + 1.0e-9,
                0.285,
            )
            self.assertTrue(clearance["active_stand_in_planning_costmap"])
            self.assertTrue(clearance["continuous_centerline_validated"])
            self.assertTrue(Path(result["validation_route_csv"]).is_file())
            diagnostics_path = Path(result["validation_diagnostics_json"])
            diagnostics = json.loads(diagnostics_path.read_text())
            self.assertEqual(
                diagnostics["metadata"]["active_stand_clearance"],
                clearance,
            )

    def test_facing_validation_rejects_route_crossing_active_stand(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            map_yaml = write_free_map(root)
            _, map_bundle = load_occupancy_grid_with_bundle(
                map_yaml,
                semantic_map_id="arena",
                planning_frame="map",
            )
            candidate = self._candidate("candidate_1", 0.0, 0.0)
            config = self._config(root, (candidate,))
            config = replace(
                config,
                map_yaml=map_yaml,
                plan=replace(
                    config.plan,
                    map_bundle_sha256=map_bundle.bundle_sha256,
                ),
            )
            start = Pose2D(-0.7, 0.0, 0.0)
            target = Pose2D(0.40, 0.0, math.pi)
            route = Route(
                points=(
                    RoutePoint(0, GridCell(3, 10), start),
                    RoutePoint(
                        1,
                        GridCell(14, 10),
                        target,
                        segment_length_m=1.1,
                        cumulative_length_m=1.1,
                    ),
                ),
                requested_start=start,
                requested_goal=target,
                snapped_start=start,
                snapped_goal=target,
                length_m=1.1,
            )
            recommendation = SimpleNamespace(
                stand_id=candidate.candidate_uid,
                material_target=SimpleNamespace(
                    face_id="qr_face",
                    pose=target,
                ),
            )

            def return_crossing_route(costmap, *_args, **_kwargs):
                center_cell = costmap.world_to_grid(Pose2D(0.0, 0.0, 0.0))
                self.assertFalse(costmap.is_traversable(center_cell))
                return PlanRouteResult(
                    route=route,
                    diagnostics=PlanningDiagnostics(status="ok"),
                )

            with (
                patch(
                    "scripts.aufgabe04.real_robot.candidate."
                    "approach.load_recommendation",
                    return_value=recommendation,
                ),
                patch(
                    "scripts.aufgabe04.real_robot.candidate."
                    "approach.plan_route",
                    side_effect=return_crossing_route,
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "crosses the active-stand collision envelope",
                ):
                    validate_facing_pose(
                        FacingValidationRequest(
                            config=config,
                            candidate=candidate,
                            recommendation_path=root / "recommendation.json",
                            current_pose=start,
                            output_dir=root / "facing",
                        )
                    )

            self.assertFalse((root / "facing").exists())


if __name__ == "__main__":
    unittest.main()
