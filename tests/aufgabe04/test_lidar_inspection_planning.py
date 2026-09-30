from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from scripts.aufgabe04.navigation.approach.camera_candidate_selection import CameraCandidateSelectionConfig
from scripts.aufgabe04.navigation.approach.candidate_inspection_view import write_candidate_inspection_view, load_candidate_inspection_view
from scripts.aufgabe04.navigation.approach.candidate_preapproach_compute import compute_candidate_preapproach_plan
from scripts.aufgabe04.navigation.approach.candidate_preapproach_materialization import materialize_candidate_preapproach_plan
from scripts.aufgabe04.navigation.approach.candidate_preapproach_models import CandidatePreapproachUnreachableError
from scripts.aufgabe04.navigation.approach.candidate_preapproach_selection import plan_and_select_camera_candidate
from scripts.aufgabe04.navigation.approach.candidate_route_uncertainty_selection import NoUncertaintyAdmittedCameraCandidateError
from scripts.aufgabe04.navigation.approach.lidar_inspection_hint import LidarInspectionHint
from scripts.aufgabe04.real_robot.candidate.approach import (
    CameraCandidateInitialSelection, CandidateApproachEffects, execute_candidate_approach_phase,
    CameraCandidateSelectionRequest, _select_initial_preapproach,
)
from scripts.aufgabe04.stations.candidate_snapshot import candidate_snapshot_sha256
from scripts.aufgabe04.stations.candidate_snapshot import new_candidate_snapshot, write_candidate_snapshot
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.planning.map_io import load_occupancy_grid_with_bundle
from tests.aufgabe04.test_detected_station_exploration import write_free_map
from tests.aufgabe04 import test_candidate_preapproach_planning as fixtures
from tests.aufgabe04 import test_autonomous_candidate_approach as runtime_fixtures
from tests.aufgabe04.test_real_robot_pipeline import calibration
from scripts.aufgabe04.real_robot.configuration.profile import RigidTransform
from scripts.aufgabe04.navigation.approach.camera_head_alignment import (
    camera_alignment_endpoint, requested_camera_base_pose, reproject_camera_alignment,
)


SELECTION_MODULE = "scripts.aufgabe04.navigation.approach.candidate_preapproach_selection"


class LidarInspectionPlanningTest(unittest.TestCase):
    def fixture(self, root):
        helper = fixtures.CandidatePreapproachPlanningTest()
        map_yaml = write_free_map(root, width=60, height=60, resolution=.05)
        _, bundle = load_occupancy_grid_with_bundle(map_yaml, semantic_map_id="arena", planning_frame="map")
        candidate = helper._candidate("candidate_1", .5, 0.)
        snapshot = new_candidate_snapshot(snapshot_id="snapshot", created_unix_sec=3.,
            planning_frame="map", map_bundle_sha256=bundle.bundle_sha256, candidates=(candidate,))
        path = root / "candidate_snapshot.json"
        write_candidate_snapshot(path, snapshot)
        hint = LidarInspectionHint(candidate.candidate_uid, candidate_snapshot_sha256(snapshot), 0.,
                                   {"stand_axis_authorized": False, "motion_authorized": False,
                                    "independent_view_requirement_met": True},
                                   center_x_m=.52, center_y_m=.015, center_uncertainty_m=.006,
                                   angle_uncertainty_rad=math.radians(3))
        kwargs = dict(map_yaml=root / "map.yaml", semantic_map_id="arena",
                      plan=helper._plan(snapshot.map_bundle_sha256), snapshot=snapshot,
                      current_pose=Pose2D(-.4, 0., 0.), unresolved={candidate.candidate_uid},
                      approach_offset_m=.5, inflation_radius_m=.25, candidate_transit_radius_m=.31,
                      physical_clearance=fixtures.PHYSICAL_CLEARANCE,
                      selection_config=CameraCandidateSelectionConfig(.055, .18),
                      lidar_inspection_hints={candidate.candidate_uid: hint},
                      camera_calibration=calibration(),
                      camera_alignment_uncertainty={"localization_position_m": .005,
                                                    "localization_yaw_rad": math.radians(1)})
        return kwargs, path

    def test_perpendicular_route_is_selected_and_sealed_with_advisory_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kwargs, path = self.fixture(root)
            selection = plan_and_select_camera_candidate(**kwargs)
            prepared = selection.selected_plan
            self.assertEqual(prepared.approach_bearing_mode, "candidate-inspection-view")
            candidate = kwargs["snapshot"].candidates[0]
            self.assertLess(abs(prepared.selected_approach_pose.x_m - candidate.geometry.x_m), .12)
            self.assertGreater(abs(prepared.selected_approach_pose.y_m - candidate.geometry.y_m), .4)
            view = root / "view.json"
            write_candidate_inspection_view(view, snapshot=kwargs["snapshot"], candidate_uid=candidate.candidate_uid,
                start=prepared.start, view_normal_rad=prepared.approach_bearing_rad-math.pi,
                purpose="lidar_axis_hint", view_index=0, camera_alignment=prepared.camera_alignment)
            sealed = materialize_candidate_preapproach_plan(prepared, snapshot=kwargs["snapshot"], snapshot_path=path,
                output_dir=root / "route", physical_clearance=fixtures.PHYSICAL_CLEARANCE,
                selection_evidence=selection.to_evidence(), inspection_view_path=view)
            self.assertTrue(Path(sealed["route_csv"]).exists())
            self.assertFalse(load_candidate_inspection_view(root / "route" / "inspection_view.json")["stand_axis_authorized"])
            with self.assertRaisesRegex(ValueError, "camera alignment|bearing mode"):
                materialize_candidate_preapproach_plan(prepared, snapshot=kwargs["snapshot"], snapshot_path=path,
                    output_dir=root / "unbound", physical_clearance=fixtures.PHYSICAL_CLEARANCE)

    def test_blocked_side_uses_other_side_and_both_blocked_fall_back(self):
        for both in (False, True):
            with self.subTest(both=both), tempfile.TemporaryDirectory() as tmp:
                kwargs, _ = self.fixture(Path(tmp))
                def preview(**arguments):
                    normal = arguments.get("inspection_view_normal_rad")
                    if normal is not None and (both or normal > 0):
                        raise CandidatePreapproachUnreachableError("candidate_1", "blocked test side")
                    return compute_candidate_preapproach_plan(**arguments)
                with patch(SELECTION_MODULE + ".compute_candidate_preapproach_plan", side_effect=preview):
                    selected = plan_and_select_camera_candidate(**kwargs)
                evidence = selected.to_evidence()["lidar_inspection_hints"]["candidate_views"]["candidate_1"]
                self.assertEqual(evidence["fallback"], both)
                self.assertEqual(selected.selected_plan.approach_bearing_mode,
                                 "robot-to-stand" if both else "candidate-inspection-view")
                if not both:
                    self.assertLess(selected.selected_plan.selected_approach_pose.y_m, 0)

    def test_uncertainty_rejection_of_first_side_does_not_discard_other_side(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs, _ = self.fixture(Path(tmp))
            calls = []
            def admit(**arguments):
                plan = arguments["plans_by_uid"]["candidate_1"]
                calls.append(plan)
                if plan.approach_bearing_rad < 0:
                    raise NoUncertaintyAdmittedCameraCandidateError({"motion_authorized": False})
                from types import SimpleNamespace
                from scripts.aufgabe04.navigation.approach.camera_candidate_selection import select_camera_candidate
                result = select_camera_candidate(arguments["options"], arguments["selection_config"])
                return SimpleNamespace(selection=result, selected_plan=plan,
                                       to_evidence=lambda: result.to_evidence())
            with patch(SELECTION_MODULE + ".select_uncertainty_admitted_camera_candidate", side_effect=admit):
                selected = plan_and_select_camera_candidate(**kwargs, route_uncertainty_context=object())
            self.assertEqual(len(calls), 3)  # Both sides, then final candidate ranking.
            self.assertGreater(selected.selected_plan.approach_bearing_rad, 0)
            evidence = selected.to_evidence()["lidar_inspection_hints"]["candidate_views"]["candidate_1"]
            self.assertEqual(evidence["views"][0]["reason"], "route_uncertainty_rejected")

    def test_no_hint_preserves_existing_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs, _ = self.fixture(Path(tmp))
            selected = plan_and_select_camera_candidate(**{**kwargs, "lidar_inspection_hints": {}})
            self.assertEqual(selected.selected_plan.approach_bearing_mode, "robot-to-stand")

    def test_missing_fit_calibration_or_localization_uses_explicit_unverified_fallback(self):
        for missing in ("fit", "independent_view", "calibration", "localization"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                kwargs, _ = self.fixture(Path(tmp))
                if missing == "fit":
                    hint = kwargs["lidar_inspection_hints"]["candidate_1"]
                    kwargs["lidar_inspection_hints"] = {"candidate_1": replace(hint, center_x_m=None)}
                elif missing == "independent_view":
                    hint = kwargs["lidar_inspection_hints"]["candidate_1"]
                    kwargs["lidar_inspection_hints"] = {"candidate_1": replace(hint,
                        evidence={**hint.evidence, "independent_view_requirement_met": False})}
                elif missing == "calibration":
                    kwargs["camera_calibration"] = None
                else:
                    kwargs["camera_alignment_uncertainty"] = None
                selected = plan_and_select_camera_candidate(**kwargs)
                self.assertEqual(selected.selected_plan.approach_bearing_mode, "robot-to-stand")
                evidence = selected.to_evidence()["lidar_inspection_hints"]["candidate_views"]["candidate_1"]
                self.assertTrue(evidence["fallback"])
                self.assertFalse(evidence["head_alignment_verified"])

    def test_camera_lever_arm_and_optical_yaw_are_used_after_quantization(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs, _ = self.fixture(Path(tmp))
            # Quaternion for camera optical frame, then a 17-degree base yaw.
            yaw = math.radians(17)
            qx, qy, qz, qw = kwargs["camera_calibration"].base_to_camera.rotation_xyzw
            c, s = math.cos(yaw/2), math.sin(yaw/2)
            rotation = (c*qx-s*qy, c*qy+s*qx, c*qz+s*qw, c*qw-s*qz)
            kwargs["camera_calibration"] = replace(kwargs["camera_calibration"],
                base_to_camera=RigidTransform((.07, .04, .1), rotation))
            selected = plan_and_select_camera_candidate(**kwargs).selected_plan
            alignment = selected.camera_alignment
            self.assertIsNotNone(alignment)
            self.assertAlmostEqual(alignment["camera_optical_yaw_rad"], yaw)
            endpoint = camera_alignment_endpoint(alignment, selected.selected_approach_pose)
            camera_heading = selected.terminal_yaw_rad+yaw
            bearing = math.atan2(alignment["center_y_m"]-endpoint["camera_y_m"],
                                 alignment["center_x_m"]-endpoint["camera_x_m"])
            self.assertAlmostEqual(math.remainder(camera_heading-bearing, 2*math.pi), 0.)
            self.assertTrue(endpoint["accepted"])
            requested = requested_camera_base_pose(alignment)
            self.assertAlmostEqual(selected.goal_cell_selection.requested_goal.x_m, requested.x_m)
            self.assertFalse(endpoint["head_alignment_verified"])

    def test_excessive_uncertainty_rejects_both_normal_previews(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs, _ = self.fixture(Path(tmp))
            kwargs["camera_alignment_uncertainty"]["localization_position_m"] = .2
            selected = plan_and_select_camera_candidate(**kwargs)
            self.assertEqual(selected.selected_plan.approach_bearing_mode, "robot-to-stand")
            evidence = selected.to_evidence()["lidar_inspection_hints"]["candidate_views"]["candidate_1"]
            self.assertEqual(len(evidence["views"]), 2)
            self.assertTrue(all("camera_alignment" in item["reason"] for item in evidence["views"]))

    def test_reprojection_moves_fitted_center_and_normal_and_refreshes_uncertainty(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs, _ = self.fixture(Path(tmp))
            alignment = plan_and_select_camera_candidate(**kwargs).selected_plan.camera_alignment
            source = Pose2D(1., 2., .2)
            target = Pose2D(-.5, .7, -.4)
            fresh = {"localization_position_m": .011, "localization_yaw_rad": .021}
            result = reproject_camera_alignment(alignment, snapshot=kwargs["snapshot"],
                candidate_uid="candidate_1", source_map_from_odom=source,
                target_map_from_odom=target, uncertainty=fresh)
            # Undo both map transforms and compare in their shared odom frame.
            def odom(point, frame):
                dx, dy = point[0]-frame.x_m, point[1]-frame.y_m
                return (math.cos(frame.yaw_rad)*dx+math.sin(frame.yaw_rad)*dy,
                        -math.sin(frame.yaw_rad)*dx+math.cos(frame.yaw_rad)*dy)
            before = odom((alignment["center_x_m"], alignment["center_y_m"]), source)
            after = odom((result["center_x_m"], result["center_y_m"]), target)
            for first, second in zip(before, after):
                self.assertAlmostEqual(first, second)
            self.assertAlmostEqual(math.remainder(result["view_normal_rad"]-alignment["view_normal_rad"],
                                                  2*math.pi), -.6)
            self.assertEqual(result["localization_position_m"], .011)
            self.assertEqual(result["camera_calibration_sha256"], alignment["camera_calibration_sha256"])
            with self.assertRaisesRegex(ValueError, "fresh localization"):
                reproject_camera_alignment(alignment, snapshot=kwargs["snapshot"], candidate_uid="candidate_1",
                    source_map_from_odom=None, target_map_from_odom=None, uncertainty=None)

    def test_alignment_cannot_claim_observed_front_or_back_at_sealing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kwargs, _ = self.fixture(root)
            prepared = plan_and_select_camera_candidate(**kwargs).selected_plan
            corrupted = {**prepared.camera_alignment, "head_alignment_verified": True}
            with self.assertRaisesRegex(ValueError, "unverified advisory"):
                write_candidate_inspection_view(root / "invalid.json", snapshot=kwargs["snapshot"],
                    candidate_uid="candidate_1", start=prepared.start,
                    view_normal_rad=prepared.approach_bearing_rad-math.pi,
                    purpose="lidar_axis_hint", view_index=0, camera_alignment=corrupted)

    def test_runtime_selection_consumes_hints_and_materializes_view_before_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kwargs, _ = self.fixture(root)
            helper = runtime_fixtures.AutonomousCandidateApproachTest()
            config = helper._config(root, kwargs["snapshot"].candidates)
            from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
            model = load_measured_physical_stand_model(Path("configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json"))
            config = replace(config, snapshot=kwargs["snapshot"],
                             plan=replace(kwargs["plan"], config=replace(kwargs["plan"].config, expected_stand_count=1)),
                             approach_offset_m=.5, measured_stand_model=model)
            loader = "scripts.aufgabe04.real_robot.candidate.lidar_inspection_hints.load_camera_lidar_hints"
            selection_module = "scripts.aufgabe04.real_robot.candidate.approach.plan_and_select_camera_candidate"
            def calibrated_selection(**arguments):
                arguments["camera_calibration"] = kwargs["camera_calibration"]
                arguments["camera_alignment_uncertainty"] = kwargs["camera_alignment_uncertainty"]
                return plan_and_select_camera_candidate(**arguments)
            with patch(loader, return_value=(kwargs["lidar_inspection_hints"], {})) as loaded, patch(
                    selection_module, side_effect=calibrated_selection):
                selection = _select_initial_preapproach(CameraCandidateSelectionRequest(
                    config, kwargs["current_pose"], frozenset(kwargs["unresolved"]), None))
            loaded.assert_called_once()
            self.assertEqual(selection.prepared_plan.approach_bearing_mode, "candidate-inspection-view")
            seen = []
            def stop_after_view(request):
                seen.append(load_candidate_inspection_view(request.inspection_view_path))
                self.assertIs(request.prepared_plan, selection.prepared_plan)
                raise RuntimeError("test stops before motion")
            move = Mock()
            with self.assertRaisesRegex(RuntimeError, "test stops before motion"):
                execute_candidate_approach_phase(config, CandidateApproachEffects(
                    read_current_pose=lambda: kwargs["current_pose"],
                    select_initial_preapproach=lambda _: selection,
                    plan_preapproach=stop_after_view, run_motion_leg=move, capture_observation=Mock(),
                ))
            move.assert_not_called()
            self.assertEqual(seen[0]["purpose"], "lidar_axis_hint")
            self.assertIsNotNone(seen[0]["source_observation"])


if __name__ == "__main__":
    unittest.main()
