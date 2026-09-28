"""Actual planner artifacts preserve the full Start target across stopped legs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.navigation.approach.admitted_pose_route import (
    plan_admitted_pose_route,
    validate_admitted_pose_route_binding,
)
from scripts.aufgabe04.navigation.execution.execution_route_certificate import (
    file_sha256,
    load_execution_route_certificate,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.planning.waypoint_csv import load_route_leg
from tests.aufgabe04 import test_admitted_pose_route as route_fixtures
from tests.aufgabe04.test_admitted_return_uncertainty import _context


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


class AdmittedReturnStageBindingTest(unittest.TestCase):
    def _intermediate(self, root):
        args = route_fixtures.AdmittedPoseRouteTest()._fixture(root)
        args["route_uncertainty_context"] = _context(args["start"], heading_sigma_rad=0.4)
        result = plan_admitted_pose_route(**args)
        self.assertFalse(result["is_final_stage"])
        leg = load_route_leg(Path(result["route_csv"]), 0, thinning_min_spacing_m=0.0)
        payload = json.loads(Path(result["diagnostics_json"]).read_text())
        return args, result, leg, payload

    def _binding(self, result, leg, *, payload=None):
        return validate_admitted_pose_route_binding(
            Path(result["diagnostics_json"]),
            leg,
            candidate_snapshot_path=Path(result["candidate_snapshot"]),
            diagnostics_payload=payload,
        )

    def test_actual_sealed_prefix_preserves_exact_final_start_target(self):
        with tempfile.TemporaryDirectory() as directory:
            args, result, leg, payload = self._intermediate(Path(directory))
            self.assertTrue(self._binding(result, leg).ok)
            metadata = payload["metadata"]
            stage = metadata["return_to_start_stage"]
            full = json.loads(Path(result["full_return_route_json"]).read_text())
            target_evidence = json.loads(Path(result["target_evidence_json"]).read_text())
            selection = json.loads(Path(result["uncertainty_selection_json"]).read_text())
            self.assertEqual(stage["stage_index"], 0)
            self.assertIs(stage["final_stage"], False)
            self.assertEqual(stage["stage_target_pose"], result["stage_target_pose"])
            self.assertEqual(metadata["selected_approach_pose"], result["stage_target_pose"])
            self.assertEqual(asdict(leg.raw_waypoints[-1].pose), result["stage_target_pose"])
            self.assertNotEqual(result["stage_target_pose"], asdict(args["target"]))
            for stored in (
                metadata["stored_start_target_pose"],
                full["stored_start_target_pose"], full["poses"][-1],
                target_evidence["target_pose"],
            ):
                self.assertEqual(stored, asdict(args["target"]))
            self.assertEqual(full["poses"][0], asdict(args["start"]))
            self.assertIs(selection["motion_authorized"], False)
            self.assertTrue(selection["selected_admission"]["decision"]["decision"]["accepted"])
            self.assertTrue(leg.raw_waypoints[-1].protected)
            self.assertTrue(leg.raw_waypoints[-1].corridor)
            self.assertTrue(all(math.isnan(w.pose.yaw_rad) for w in leg.raw_waypoints[:-1]))
            certificate = load_execution_route_certificate(Path(result["route_certificate_json"]))
            self.assertEqual(certificate.route_sha256, file_sha256(Path(result["route_csv"])))
            self.assertTrue(certificate.exact_vertex_pursuit)

    def test_changed_stage_identity_or_endpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, result, leg, payload = self._intermediate(Path(directory))
            changes = (
                ("final_stage", True),
                ("stage_index", True),
                ("stage_index", 1),
                ("start_candidate_uid", "different_stand"),
                ("end_fraction", 0.25),
                ("stage_target_pose", {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}),
            )
            for key, value in changes:
                with self.subTest(field=key, value=value):
                    changed = deepcopy(payload)
                    changed["metadata"]["return_to_start_stage"][key] = value
                    self.assertFalse(self._binding(result, leg, payload=changed).ok)

    def test_changed_route_geometry_or_csv_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, result, leg, _ = self._intermediate(Path(directory))
            first = leg.raw_waypoints[0]
            changed_first = replace(first, pose=replace(first.pose, x_m=first.pose.x_m + 0.01))
            changed_leg = replace(leg, raw_waypoints=(changed_first, *leg.raw_waypoints[1:]))
            status = self._binding(result, changed_leg)
            self.assertFalse(status.ok)
            self.assertIn("prefix", status.failures[0])

            route_path = Path(result["route_csv"])
            route_path.write_text(route_path.read_text() + "\n")
            changed_leg = load_route_leg(route_path, 0, thinning_min_spacing_m=0.0)
            status = self._binding(result, changed_leg)
            self.assertFalse(status.ok)
            self.assertIn("route_csv_sha256", status.failures[0])

    def test_changed_full_route_file_is_rejected_even_if_file_hash_is_updated(self):
        with tempfile.TemporaryDirectory() as directory:
            _, result, leg, payload = self._intermediate(Path(directory))
            full_path = Path(result["full_return_route_json"])
            full = json.loads(full_path.read_text())
            full["poses"][-1]["yaw_rad"] += 0.1
            _write_json(full_path, full)
            status = self._binding(result, leg)
            self.assertFalse(status.ok)
            self.assertIn("full_return_route hash", status.failures[0])

            changed = deepcopy(payload)
            changed["metadata"]["return_to_start_stage"]["full_return_route_sha256"] = file_sha256(full_path)
            status = self._binding(result, leg, payload=changed)
            self.assertFalse(status.ok)
            self.assertIn("stored Start target", status.failures[0])

    def test_changed_uncertainty_artifact_cannot_reduce_reserves(self):
        with tempfile.TemporaryDirectory() as directory:
            _, result, leg, payload = self._intermediate(Path(directory))
            selection_path = Path(result["uncertainty_selection_json"])
            selection = json.loads(selection_path.read_text())
            selection["config"]["braking_latency_distance_m"] = 0.02
            _write_json(selection_path, selection)
            status = self._binding(result, leg)
            self.assertFalse(status.ok)
            self.assertIn("uncertainty_selection hash", status.failures[0])

            changed = deepcopy(payload)
            changed["metadata"]["return_to_start_stage"]["uncertainty_selection_sha256"] = file_sha256(selection_path)
            status = self._binding(result, leg, payload=changed)
            self.assertFalse(status.ok)
            self.assertIn("braking reserve", status.failures[0])

    def test_changed_stored_source_or_target_evidence_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, result, leg, _ = self._intermediate(Path(directory))
            source = Path(args["target_evidence"]["source_artifacts"][0]["path"])
            original_source = source.read_bytes()
            source.write_text('{"qr_id": "changed"}\n')
            status = self._binding(result, leg)
            self.assertFalse(status.ok)
            self.assertIn("source artifact hash", status.failures[0])
            source.write_bytes(original_source)
            self.assertTrue(self._binding(result, leg).ok)

            Path(result["target_evidence_json"]).write_text("{}\n")
            status = self._binding(result, leg)
            self.assertFalse(status.ok)
            self.assertIn("target evidence hash", status.failures[0])

    def test_nonzero_stage_index_requires_localization_context_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            args = route_fixtures.AdmittedPoseRouteTest()._fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "missing uncertainty context"):
                plan_admitted_pose_route(**args, return_stage_index=1)
            self.assertFalse(args["output_dir"].exists())

    def test_indexed_final_stationary_stage_has_exact_yaw_and_zero_translation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = route_fixtures.AdmittedPoseRouteTest()._fixture(
                Path(directory), target=Pose2D(-0.40, -0.20, -0.71),
            )
            result = plan_admitted_pose_route(
                **args, return_stage_index=1,
                route_uncertainty_context=_context(args["start"]),
            )
            leg = load_route_leg(Path(result["route_csv"]), 0, thinning_min_spacing_m=0.0)
            self.assertTrue(self._binding(result, leg).ok)
            self.assertTrue(result["is_final_stage"])
            self.assertTrue(leg.stationary_turn)
            self.assertEqual(leg.route_length_m, 0.0)
            self.assertEqual(len(leg.raw_waypoints), 2)
            self.assertEqual(leg.raw_waypoints[-1].pose, args["target"])
            self.assertEqual(
                (leg.raw_waypoints[0].pose.x_m, leg.raw_waypoints[0].pose.y_m),
                (args["target"].x_m, args["target"].y_m),
            )
            metadata = json.loads(Path(result["diagnostics_json"]).read_text())["metadata"]
            self.assertEqual(metadata["return_to_start_stage"]["stage_index"], 1)
            self.assertIs(metadata["return_to_start_stage"]["final_stage"], True)


if __name__ == "__main__":
    unittest.main()
