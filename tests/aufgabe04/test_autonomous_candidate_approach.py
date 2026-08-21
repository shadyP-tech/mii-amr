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

from scripts.aufgabe04.navigation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.mission_leg_motion_permit import MissionLegKind
from scripts.aufgabe04.navigation.models import GridCell, Pose2D
from scripts.aufgabe04.navigation.read_current_amcl_pose import CurrentAmclPose
from scripts.aufgabe04.navigation.stand_coverage_survey import (
    CoverageSurveyConfig,
    CoverageSurveyPlan,
    SurveyViewpoint,
)
from scripts.aufgabe04.real_robot.autonomous_candidate_approach import (
    CandidateApproachConfig,
    CandidateApproachEffects,
    CandidateApproachPoseError,
    CandidateObservation,
    FacingValidationRequest,
    execute_candidate_approach_phase,
    nearest_candidate,
    validate_facing_pose,
)
from scripts.aufgabe04.real_robot.autonomous_child_runner import MotionLegOutcome
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateGeometry,
    CandidateSource,
    FrozenCandidate,
    new_candidate_snapshot,
)


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
                "minimum_candidate_transit_radius_m": 0.31,
                "minimum_static_inflation_m": 0.25,
            },
            uncertainty_sigma_multiplier=2.0,
            localization_branch_proof_id="known_start",
            mission_leg_motion_authorization_json=(
                root / "mission_leg_authorization.json"
            ),
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
                    {
                        "observation_kind": "real_stand_axis_without_qr",
                        "stand_axis_rad": 0.0,
                        "stand_center": {"x_m": 0.2, "y_m": 0.0},
                        "robot_pose": {"x_m": 0.2, "y_m": 0.7},
                    }
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

    def test_axis_only_observation_uses_bounded_opposite_face_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self._candidate("candidate_1", 0.2, 0.0)
            config = self._config(root, (candidate,))
            axis_path = root / "axis.json"
            axis_path.write_text(
                json.dumps(
                    {
                        "observation_kind": "real_stand_axis_without_qr",
                        "stand_axis_rad": 0.0,
                        "stand_center": {"x_m": 0.2, "y_m": 0.0},
                        "robot_pose": {"x_m": 0.2, "y_m": 0.7},
                    }
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
                        raise ValueError(
                            "candidate pre-approach A* failed: target is blocked"
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

    def test_exhausted_opposite_offsets_publish_no_identity_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self._candidate("candidate_1", 0.2, 0.0)
            config = self._config(root, (candidate,))
            axis_path = root / "axis.json"
            axis_path.write_text(
                json.dumps(
                    {
                        "observation_kind": "real_stand_axis_without_qr",
                        "stand_axis_rad": 0.0,
                        "stand_center": {"x_m": 0.2, "y_m": 0.0},
                        "robot_pose": {"x_m": 0.2, "y_m": 0.7},
                    }
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
                "scripts.aufgabe04.real_robot."
                "autonomous_candidate_approach.load_recommendation",
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


if __name__ == "__main__":
    unittest.main()
