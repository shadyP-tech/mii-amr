"""Recovery must refresh the calibration-aware route's exact stopped epoch."""
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.approach.candidate_inspection_view import (
    load_candidate_inspection_view, write_candidate_inspection_view,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_selection import plan_and_select_camera_candidate
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import MissionLegKind
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.candidate_planning_pose import admitted_candidate_planning_pose
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.real_robot.candidate.approach import (
    CandidateApproachEffects, CandidatePreapproachRequest, _execute_candidate_motion,
)
from scripts.aufgabe04.real_robot.candidate.route_uncertainty_readiness import load_candidate_route_uncertainty_readiness
from tests.aufgabe04 import test_autonomous_candidate_approach as runtime_fixtures
from tests.aufgabe04 import test_lidar_inspection_planning as planning_fixtures
from tests.aufgabe04.test_candidate_route_uncertainty_readiness import _preflight_payload

MODULE = "scripts.aufgabe04.real_robot.candidate.approach"


class CameraAlignmentRecoveryTest(unittest.TestCase):
    def exercise(self, root, *, runtime, frame_changed, missing_loader=False, missing_covariance=False):
        kwargs, snapshot_path = planning_fixtures.LidarInspectionPlanningTest().fixture(root)
        prepared = plan_and_select_camera_candidate(**kwargs).selected_plan
        config = runtime_fixtures.AutonomousCandidateApproachTest()._config(root, kwargs["snapshot"].candidates)
        config = replace(config, snapshot=kwargs["snapshot"], snapshot_path=snapshot_path,
                         robot_radius_m=.105, max_startup_reseals_per_leg=1)
        initial_view = root / "initial_view.json"
        write_candidate_inspection_view(initial_view, snapshot=kwargs["snapshot"], candidate_uid="candidate_1",
            start=prepared.start, view_normal_rad=prepared.camera_alignment["view_normal_rad"],
            purpose="lidar_axis_hint", view_index=0, camera_alignment=prepared.camera_alignment)
        request = CandidatePreapproachRequest(
            map_yaml=kwargs["map_yaml"], semantic_map_id="arena", plan=kwargs["plan"],
            snapshot=kwargs["snapshot"], snapshot_path=snapshot_path, candidate_uid="candidate_1",
            start=prepared.start, output_dir=root / "original", approach_offset_m=.5,
            inflation_radius_m=.25, candidate_transit_radius_m=.31,
            physical_clearance=kwargs["physical_clearance"], prepared_plan=prepared,
            inspection_view_path=initial_view)
        fresh_start = Pose2D(-.25, .02, 0.)
        original_frame = CandidatePlanningFrame(prepared.start, PlanarTransform2D(0., 0., 0.))
        target_transform = PlanarTransform2D(.1, .2, .25)
        fresh_frame = CandidatePlanningFrame(fresh_start, target_transform)
        target_snapshot = kwargs["snapshot"]
        if frame_changed:
            candidate = target_snapshot.candidates[0]
            x, y = candidate.geometry.x_m, candidate.geometry.y_m
            target_snapshot = replace(target_snapshot, candidates=(replace(candidate, geometry=replace(
                candidate.geometry, x_m=.1+math.cos(.25)*x-math.sin(.25)*y,
                y_m=.2+math.sin(.25)*x+math.cos(.25)*y)),))
        projected_config = replace(config, snapshot=target_snapshot)
        fresh_evidence = root / "new_stopped_epoch.json"
        payload = _preflight_payload(fresh_start)
        if frame_changed:
            dx, dy = fresh_start.x_m-.1, fresh_start.y_m-.2
            odom_x = math.cos(.25)*dx+math.sin(.25)*dy
            odom_y = -math.sin(.25)*dx+math.cos(.25)*dy
            payload["map_from_odom"].update(x_m=.1, y_m=.2, yaw_rad=.25)
            payload["odom_pose"].update(x_m=odom_x, y_m=odom_y, yaw_rad=-.25)
            payload["observations"][1]["data"].update(x_m=odom_x, y_m=odom_y, yaw_rad=-.25)
        for sample in payload["stationary_amcl_samples"]:
            sample["covariance"][0] = .000004
            sample["covariance"][7] = .000009
            sample["covariance"][35] = .0001
        if missing_covariance:
            payload.pop("stationary_amcl_samples")
        fresh_evidence.write_text(json.dumps(payload))
        fresh_start, _ = admitted_candidate_planning_pose(payload, map_frame="map", odom_frame="odom")
        fresh_frame = replace(fresh_frame, current_pose=fresh_start)
        planned = []
        def planner(replacement):
            planned.append(replacement)
            return {"route_csv": str(root / "replacement.csv")}
        loader = Mock(wraps=load_candidate_route_uncertainty_readiness)
        motion = Mock()
        effects = CandidateApproachEffects(
            read_current_pose=lambda: fresh_start, run_motion_leg=motion,
            capture_observation=Mock(), plan_preapproach=planner,
            admit_startup_localization=lambda _: fresh_start,
            admit_runtime_localization=lambda _: fresh_start,
            admit_planning_frame=(lambda _: fresh_frame) if frame_changed else None,
            load_route_uncertainty_readiness=None if missing_loader else loader,
            run_startup_reseal_motion_leg=motion, run_runtime_localization_reseal_motion_leg=motion)
        source_root = root / "replacement"
        source_root.mkdir()
        def recover(_initial, **arguments):
            callbacks = arguments["runtime_effects" if runtime else "startup_effects"]
            admitted = callbacks.admit_fresh_stationary_localization(fresh_evidence)
            attempt = SimpleNamespace(source_root=source_root, fresh_start_pose=admitted,
                fresh_localization_evidence_path=fresh_evidence,
                identity=SimpleNamespace(run_id="replacement_run"))
            return callbacks.replan_same_routine(attempt)
        projection = SimpleNamespace(config=projected_config)
        def run():
            with patch(MODULE+".execute_candidate_motion_with_recovery", side_effect=recover), patch(
                    MODULE+"._materialize_candidate_frame_projection", return_value=projection):
                return _execute_candidate_motion(config=config, effects=effects, candidate_root=root,
                    plan_request=request, initial_sealed={"route_csv": str(root / "original.csv")},
                    run_id="initial", leg_kind=MissionLegKind.CANDIDATE_PREAPPROACH,
                    candidate_index=1, target_id="candidate_1", frame_source_config=config,
                    source_registry=object(), plan_planning_frame=original_frame if frame_changed else None)
        if missing_loader or missing_covariance:
            with self.assertRaises((ValueError, RuntimeError)):
                run()
            self.assertEqual(planned, [])
            motion.assert_not_called()
            return
        run()
        self.assertEqual(len(planned), 1)
        self.assertIsNone(planned[0].prepared_plan)
        loader.assert_called_once()
        self.assertEqual(loader.call_args.args[0].preflight_json, fresh_evidence)
        self.assertEqual(loader.call_args.args[0].expected_start, fresh_start)
        updated = load_candidate_inspection_view(planned[0].inspection_view_path)["camera_alignment"]
        original = prepared.camera_alignment
        self.assertAlmostEqual(updated["localization_position_m"], config.uncertainty_sigma_multiplier*.003)
        self.assertAlmostEqual(updated["localization_yaw_rad"], config.uncertainty_sigma_multiplier*.01)
        self.assertEqual(updated["camera_calibration_sha256"], original["camera_calibration_sha256"])
        self.assertEqual(Path(updated["localization_source_evidence"]["source_preplanning_localization_json"]), fresh_evidence)
        x, y = original["center_x_m"], original["center_y_m"]
        self.assertAlmostEqual(updated["center_x_m"], .1+math.cos(.25)*x-math.sin(.25)*y if frame_changed else x)
        self.assertAlmostEqual(updated["center_y_m"], .2+math.sin(.25)*x+math.cos(.25)*y if frame_changed else y)
        self.assertAlmostEqual(math.remainder(updated["view_normal_rad"]-original["view_normal_rad"], 2*math.pi),
                               .25 if frame_changed else 0.)
        self.assertFalse(updated["head_alignment_verified"])
        motion.assert_not_called()

    def test_startup_and_runtime_refresh_exact_epoch_in_same_or_changed_frame(self):
        for runtime in (False, True):
            for frame_changed in (False, True):
                with self.subTest(runtime=runtime, frame_changed=frame_changed), tempfile.TemporaryDirectory() as tmp:
                    self.exercise(Path(tmp), runtime=runtime, frame_changed=frame_changed)

    def test_missing_readiness_cannot_replan_or_move_with_old_bounds(self):
        for runtime in (False, True):
            with self.subTest(runtime=runtime), tempfile.TemporaryDirectory() as tmp:
                self.exercise(Path(tmp), runtime=runtime, frame_changed=False, missing_loader=True)

    def test_fresh_artifact_without_covariance_cannot_reuse_old_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.exercise(Path(tmp), runtime=True, frame_changed=False, missing_covariance=True)
