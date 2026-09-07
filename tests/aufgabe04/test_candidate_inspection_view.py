from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.navigation.approach.candidate_inspection_view import (
    INSPECTION_VIEW_BEARING_MODE, load_candidate_inspection_view,
    validate_candidate_inspection_view_binding, write_candidate_inspection_view,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_planning import (
    compute_candidate_preapproach_plan, materialize_candidate_preapproach_plan,
)
from scripts.aufgabe04.navigation.approach.detected_stand_preapproach import (
    validate_detected_stand_preapproach_binding,
)
from scripts.aufgabe04.navigation.planning.waypoint_csv import load_route_leg
from tests.aufgabe04 import test_candidate_preapproach_planning as fixtures


class CandidateInspectionViewTest(unittest.TestCase):
    def fixture(self, root):
        fixture = fixtures.CandidatePreapproachPlanningTest()
        candidate, snapshot, snapshot_path, old_plan = fixture._materialization_fixture(root)
        prepared = compute_candidate_preapproach_plan(
            map_yaml=root / "map.yaml", semantic_map_id="arena",
            plan=fixture._plan(snapshot.map_bundle_sha256), snapshot=snapshot,
            candidate_uid=candidate.candidate_uid, start=old_plan.start,
            approach_offset_m=0.70, inflation_radius_m=0.25,
            candidate_transit_radius_m=0.31, physical_clearance=fixtures.PHYSICAL_CLEARANCE,
            inspection_view_normal_rad=math.radians(51),
        )
        view_path = root / "view.json"
        write_candidate_inspection_view(
            view_path, snapshot=snapshot, candidate_uid=candidate.candidate_uid,
            start=prepared.start, view_normal_rad=math.radians(51),
            purpose="diverse_inspection", view_index=1,
        )
        return snapshot, snapshot_path, prepared, view_path

    def test_real_view_plan_seals_and_passes_runtime_binding_without_axis_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, snapshot_path, prepared, view_path = self.fixture(root)
            self.assertEqual(prepared.approach_bearing_mode, INSPECTION_VIEW_BEARING_MODE)
            outputs = materialize_candidate_preapproach_plan(
                prepared, snapshot=snapshot, snapshot_path=snapshot_path,
                output_dir=root / "route", physical_clearance=fixtures.PHYSICAL_CLEARANCE,
                inspection_view_path=view_path,
            )
            leg = load_route_leg(Path(outputs["route_csv"]), 0)
            status = validate_detected_stand_preapproach_binding(
                Path(outputs["diagnostics_json"]), leg, candidate_snapshot_path=snapshot_path,
            )
            self.assertTrue(status.ok, status.failures)
            self.assertTrue(Path(outputs["route_certificate_json"]).is_file())
            metadata = json.loads(Path(outputs["diagnostics_json"]).read_text())["metadata"]
            self.assertEqual(metadata["approach_bearing_mode"], INSPECTION_VIEW_BEARING_MODE)
            self.assertNotIn("axis_observation_json", metadata)
            self.assertGreater(leg.raw_waypoints[-1].pose.y_m, 0.50)

            copied = Path(metadata["inspection_view_json"])
            payload = json.loads(copied.read_text())
            payload["view_normal_rad"] = 0.0
            copied.write_text(json.dumps(payload))
            failed = validate_detected_stand_preapproach_binding(
                Path(outputs["diagnostics_json"]), leg, candidate_snapshot_path=snapshot_path,
            )
            self.assertFalse(failed.ok)
            self.assertTrue(any("inspection view" in failure for failure in failed.failures))

    def test_view_cannot_bind_another_snapshot_or_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, snapshot_path, prepared, view_path = self.fixture(root)
            evidence = load_candidate_inspection_view(view_path)
            with self.assertRaisesRegex(ValueError, "snapshot binding"):
                validate_candidate_inspection_view_binding(
                    evidence, snapshot=replace(snapshot, snapshot_id="different"),
                    candidate_uid=prepared.candidate_uid,
                )
            with self.assertRaisesRegex(ValueError, "start pose binding"):
                materialize_candidate_preapproach_plan(
                    replace(prepared, start=replace(prepared.start, yaw_rad=0.1)),
                    snapshot=snapshot, snapshot_path=snapshot_path,
                    output_dir=root / "bad", physical_clearance=fixtures.PHYSICAL_CLEARANCE,
                    inspection_view_path=view_path,
                )
            self.assertFalse((root / "bad").exists())

    def test_view_cannot_be_relabelled_as_certified_backside(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, snapshot_path, prepared, view_path = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "cannot replace certified"):
                materialize_candidate_preapproach_plan(
                    prepared, snapshot=snapshot, snapshot_path=snapshot_path,
                    output_dir=root / "bad", physical_clearance=fixtures.PHYSICAL_CLEARANCE,
                    inspection_view_path=view_path, approach_normal_rad=math.radians(51),
                    axis_observation_path=view_path,
                )
            self.assertFalse((root / "bad").exists())


if __name__ == "__main__":
    unittest.main()
