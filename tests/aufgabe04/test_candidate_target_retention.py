"""Corrected camera targets survive arrival and fresh centering frames."""

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from scripts.aufgabe04.navigation.approach.camera_head_alignment import make_camera_alignment
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import (
    CandidatePlanningFrame, project_candidate_snapshot_to_planning_frame,
)
from scripts.aufgabe04.navigation.approach.lidar_inspection_hint import LidarInspectionHint
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import load_stand_survey_registry
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.real_robot.candidate.approach import (
    CandidateApproachEffects, _CandidateObservationFrame, _admit_camera_arrival_geometry,
)
from scripts.aufgabe04.real_robot.candidate.observation_deferral import CandidateObservationUnavailableError
from scripts.aufgabe04.real_robot.candidate.retained_orientation import retain_orientation_after_arrival
from scripts.aufgabe04.real_robot.candidate.target_admission import require_frame_target
from scripts.aufgabe04.stations.candidate_snapshot import candidate_snapshot_sha256
from tests.aufgabe04 import test_autonomous_candidate_approach as runtime_fixtures
from tests.aufgabe04.test_candidate_route_uncertainty_selection import _uncertainty
from tests.aufgabe04.test_detected_station_exploration import write_free_map
from tests.aufgabe04.test_real_robot_pipeline import calibration


class CandidateTargetRetentionTests(unittest.TestCase):
    def fixture(self, root, *, translation_y=0.):
        write_free_map(root, width=100, height=100, resolution=.05)
        helper = runtime_fixtures.AutonomousCandidateApproachTest()
        config = helper._config(root, (helper._candidate("candidate_1", .5, .92),))
        config = helper._write_frame_registry(replace(config, camera_calibration=calibration()),
            frozen_map_from_odom=PlanarTransform2D(0., 0., 0.))
        candidate = config.snapshot.candidates[0]
        hint = LidarInspectionHint(candidate.candidate_uid, candidate_snapshot_sha256(config.snapshot),
            0., {"independent_view_requirement_met": True},
            center_x_m=.5, center_y_m=.86, center_uncertainty_m=.006,
            angle_uncertainty_rad=math.radians(3))
        alignment = make_camera_alignment(hint=hint, snapshot=config.snapshot,
            candidate_uid=candidate.candidate_uid, normal_rad=-math.pi/2, standoff_m=.5,
            calibration=config.camera_calibration,
            uncertainty={"localization_position_m": .005, "localization_yaw_rad": .01})
        source = _CandidateObservationFrame(config, candidate,
            CandidatePlanningFrame(Pose2D(.5, .36, math.pi/2), PlanarTransform2D(0., 0., 0.)),
            None, camera_alignment=alignment)
        current = CandidatePlanningFrame(Pose2D(.5, .36+translation_y, math.pi/2),
            PlanarTransform2D(0., translation_y, 0.))
        effects = CandidateApproachEffects(read_current_pose=Mock(), plan_preapproach=Mock(),
            run_motion_leg=Mock(), capture_observation=Mock(), validate_facing=Mock(),
            commit_decision=Mock(), admit_planning_frame=Mock(return_value=current),
            load_route_uncertainty_readiness=Mock(return_value=_uncertainty()))
        registry = load_stand_survey_registry(config.survey_root / "stand_registry.json")
        return source, current, effects, registry

    def arrive(self, source, effects, registry, root):
        return _admit_camera_arrival_geometry(source_config=source.config, effects=effects,
            source_registry=registry, candidate_uid=source.candidate.candidate_uid,
            candidate_root=root, observation_attempt_index=0, target_source_frame=source)

    def test_raw_blocked_fitted_clear_arrival_uses_bound_selected_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, current, effects, registry = self.fixture(root)
            arrival = self.arrive(source, effects, registry, root / "arrival")
            self.assertAlmostEqual(arrival.camera_target_geometry.y_m, .86)
            self.assertAlmostEqual(arrival.candidate.geometry.y_m, .92)
            receipt = json.loads((root / "arrival/candidate_arrival_admission.json").read_text())
            self.assertTrue(receipt["accepted"])
            self.assertAlmostEqual(receipt["target"]["y_m"], .86)
            self.assertFalse(receipt["head_alignment_verified"])
            self.assertFalse(receipt["camera_centered"])
            provenance = json.loads(arrival.camera_target_geometry_evidence_path.read_text())
            self.assertEqual(provenance["source_candidate_snapshot_sha256"], candidate_snapshot_sha256(source.config.snapshot))
            self.assertEqual(provenance["target_candidate_snapshot_sha256"], candidate_snapshot_sha256(arrival.config.snapshot))
            request = effects.load_route_uncertainty_readiness.call_args.args[0]
            self.assertEqual(request.preflight_json, root / "arrival/candidate_arrival_localization.json")
            self.assertEqual(request.expected_start, current.current_pose)
            effects.capture_observation.assert_not_called()
            effects.run_motion_leg.assert_not_called()

    def test_projected_fit_moving_into_wall_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _, effects, registry = self.fixture(root, translation_y=.12)
            with self.assertRaises(CandidateObservationUnavailableError) as raised:
                self.arrive(source, effects, registry, root / "arrival")
            self.assertEqual(raised.exception.reason, "candidate_target_ineligible")
            self.assertAlmostEqual(raised.exception.status_evidence["target"]["y_m"], .98)
            self.assertIn("target_static_map_incompatible", raised.exception.status_evidence["reasons"])
            effects.capture_observation.assert_not_called()
            effects.run_motion_leg.assert_not_called()

    def test_centering_refresh_rotates_and_translates_current_fit_not_old_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _, effects, registry = self.fixture(root)
            arrival = self.arrive(source, effects, registry, root / "arrival")
            # A fresh LiDAR fit supersedes the selected y=.86 alignment.
            arrival = replace(arrival, camera_target_geometry=replace(arrival.camera_target_geometry, y_m=.845))
            tf = PlanarTransform2D(-.4, -.2, .4)
            frame = CandidatePlanningFrame(Pose2D(0., 0., .4), tf)
            snapshot = project_candidate_snapshot_to_planning_frame(source.config.snapshot, registry, frame).projected_snapshot
            fresh = _CandidateObservationFrame(replace(source.config, snapshot=snapshot),
                snapshot.candidates[0], frame, None)
            retained = retain_orientation_after_arrival(arrival, fresh, root / "centering")
            expected_x = -.4+math.cos(.4)*.5-math.sin(.4)*.845
            expected_y = -.2+math.sin(.4)*.5+math.cos(.4)*.845
            self.assertAlmostEqual(retained.camera_target_geometry.x_m, expected_x)
            self.assertAlmostEqual(retained.camera_target_geometry.y_m, expected_y)
            self.assertIsNone(retained.camera_alignment)
            self.assertNotEqual(retained.camera_target_geometry, arrival.camera_target_geometry)
            self.assertTrue(require_frame_target(retained,
                evidence_path=root / "post_turn_admission.json", attempt_index=1).accepted)
            provenance = json.loads(retained.camera_target_geometry_evidence_path.read_text())
            self.assertEqual(provenance["source_kind"], "current_camera_target_geometry")
            self.assertFalse(provenance["head_alignment_verified"])
            self.assertFalse(provenance["motion_authorized"])

    def test_alignment_snapshot_mismatch_and_missing_fresh_bounds_fail_closed(self):
        for failure in ("snapshot", "readiness"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, _, effects, registry = self.fixture(root)
                if failure == "snapshot":
                    source = replace(source, camera_alignment={**source.camera_alignment,
                        "candidate_snapshot_sha256": "a" * 64})
                    expected, message = ValueError, "binding mismatch"
                else:
                    effects = replace(effects, load_route_uncertainty_readiness=None)
                    expected, message = RuntimeError, "fresh uncertainty"
                with self.assertRaisesRegex(expected, message):
                    self.arrive(source, effects, registry, root / "arrival")
                effects.capture_observation.assert_not_called()
                effects.run_motion_leg.assert_not_called()

    def test_geometry_refresh_requires_source_transform_and_snapshot_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _, effects, registry = self.fixture(root)
            arrival = self.arrive(source, effects, registry, root / "arrival")
            fresh = replace(arrival, camera_target_geometry=None, camera_alignment=None,
                camera_target_geometry_evidence_path=None)
            with self.assertRaisesRegex(ValueError, "matching admitted"):
                retain_orientation_after_arrival(replace(arrival, planning_frame=None), fresh, root / "invalid_frame")
            wrong_candidate = replace(arrival.candidate, confidence=.1)
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                retain_orientation_after_arrival(replace(arrival, candidate=wrong_candidate), fresh, root / "invalid_snapshot")
