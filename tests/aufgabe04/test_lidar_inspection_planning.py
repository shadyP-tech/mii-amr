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
                                   {"stand_axis_authorized": False, "motion_authorized": False})
        kwargs = dict(map_yaml=root / "map.yaml", semantic_map_id="arena",
                      plan=helper._plan(snapshot.map_bundle_sha256), snapshot=snapshot,
                      current_pose=Pose2D(-.4, 0., 0.), unresolved={candidate.candidate_uid},
                      approach_offset_m=.5, inflation_radius_m=.25, candidate_transit_radius_m=.31,
                      physical_clearance=fixtures.PHYSICAL_CLEARANCE,
                      selection_config=CameraCandidateSelectionConfig(.055, .18),
                      lidar_inspection_hints={candidate.candidate_uid: hint})
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
                purpose="lidar_axis_hint", view_index=0)
            sealed = materialize_candidate_preapproach_plan(prepared, snapshot=kwargs["snapshot"], snapshot_path=path,
                output_dir=root / "route", physical_clearance=fixtures.PHYSICAL_CLEARANCE,
                selection_evidence=selection.to_evidence(), inspection_view_path=view)
            self.assertTrue(Path(sealed["route_csv"]).exists())
            self.assertFalse(load_candidate_inspection_view(root / "route" / "inspection_view.json")["stand_axis_authorized"])
            with self.assertRaisesRegex(ValueError, "bearing mode"):
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

    def test_runtime_selection_consumes_hints_and_materializes_view_before_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kwargs, _ = self.fixture(root)
            helper = runtime_fixtures.AutonomousCandidateApproachTest()
            config = helper._config(root, kwargs["snapshot"].candidates)
            config = replace(config, snapshot=kwargs["snapshot"],
                             plan=replace(kwargs["plan"], config=replace(kwargs["plan"].config, expected_stand_count=1)),
                             approach_offset_m=.5)
            loader = "scripts.aufgabe04.real_robot.candidate.lidar_inspection_hints.load_camera_lidar_hints"
            with patch(loader, return_value=(kwargs["lidar_inspection_hints"], {})) as loaded:
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
