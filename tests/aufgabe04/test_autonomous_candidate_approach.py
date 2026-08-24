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
from scripts.aufgabe04.navigation.foundation.models import GridCell, Pose2D
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
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    CoverageSurveyConfig,
    CoverageSurveyPlan,
    SurveyViewpoint,
)
from scripts.aufgabe04.real_robot.autonomous_candidate_approach import (
    CandidateApproachConfig,
    CandidateApproachEffects,
    CandidateApproachPoseError,
    CandidateObservation,
    CameraCandidateInitialSelection,
    CameraCandidateSelectionRequest,
    FacingValidationRequest,
    execute_candidate_approach_phase,
    nearest_candidate,
    plan_candidate_preapproach,
    validate_facing_pose,
)
from scripts.aufgabe04.real_robot.autonomous_child_runner import MotionLegOutcome
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateGeometry,
    CandidateSource,
    FrozenCandidate,
    new_candidate_snapshot,
    write_candidate_snapshot,
)
from tests.aufgabe04.test_detected_station_exploration import write_free_map


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
