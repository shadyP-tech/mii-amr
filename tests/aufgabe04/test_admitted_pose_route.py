"""The Start return uses the stored robot pose without changing its goal."""

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.navigation.approach.admitted_pose_route import (
    ADMITTED_POSE_ROUTE_KIND, plan_admitted_pose_route,
    validate_admitted_pose_route_binding,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.control.driving_behavior import (
    STATIC_PHYSICAL_ROUTE_KINDS, TERMINAL_HEADING_ONLY_PHYSICAL_ROUTE_KINDS,
)
from scripts.aufgabe04.navigation.execution.execution_route_certificate import (
    check_execution_route_tube, file_sha256, load_execution_route_certificate,
    point_to_segment_distance_m,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.navigation.planning.map_io import load_occupancy_grid_with_bundle
from scripts.aufgabe04.navigation.planning.waypoint_csv import load_route_leg
from scripts.aufgabe04.navigation.waypoint_follower.startup import (
    certified_static_startup_decision,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    candidate_snapshot_sha256, new_candidate_snapshot, write_candidate_snapshot,
)
from tests.aufgabe04 import test_candidate_preapproach_planning as fixtures
from tests.aufgabe04.test_detected_station_exploration import write_free_map


class AdmittedPoseRouteTest(unittest.TestCase):
    def _fixture(self, root, *, extra_candidates=(), target=None, fine_grid=False):
        map_yaml = write_free_map(root, width=150, height=150, resolution=.02) if fine_grid else write_free_map(root)
        _, bundle = load_occupancy_grid_with_bundle(map_yaml, semantic_map_id="arena", planning_frame="map")
        candidate = fixtures.CandidatePreapproachPlanningTest._candidate("start_stand", 1.0, 0.)
        snapshot = new_candidate_snapshot(
            snapshot_id="full_pool", created_unix_sec=3., planning_frame="map",
            map_bundle_sha256=bundle.bundle_sha256,
            candidates=(candidate, *extra_candidates),
        )
        snapshot_path = root / "snapshot.json"
        write_candidate_snapshot(snapshot_path, snapshot)
        source = root / "stored_catalog.json"
        source.write_text('{"qr_id": "Start"}\n')
        target = target or Pose2D(.431, .213, -.71)
        frame = CandidatePlanningFrame(Pose2D(-.40, -.20, .3), PlanarTransform2D(0., 0., 0.))
        evidence = {
            "qr_id": "Start", "candidate_uid": "start_stand", "target_pose": asdict(target),
            "planning_frame": "map", "candidate_snapshot_sha256": candidate_snapshot_sha256(snapshot),
            "stored_pose": asdict(target), "source_planning_frame": frame.to_evidence(),
            "planning_frame_admission": frame.to_evidence(),
            "source_artifacts": [{"path": str(source), "sha256": file_sha256(source)}],
        }
        return {
            "map_yaml": map_yaml, "semantic_map_id": "arena",
            "plan": fixtures.CandidatePreapproachPlanningTest._plan(bundle.bundle_sha256),
            "snapshot": snapshot, "snapshot_path": snapshot_path, "candidate_uid": "start_stand",
            "start": frame.current_pose, "target": target, "output_dir": root / "sealed",
            "inflation_radius_m": .25, "physical_clearance": fixtures.PHYSICAL_CLEARANCE,
            "target_evidence": evidence,
        }

    def test_exact_pose_yaw_and_smoothed_route_are_sealed(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._fixture(Path(directory))
            result = plan_admitted_pose_route(**args)
            leg = load_route_leg(Path(result["route_csv"]), 0, thinning_min_spacing_m=0.)
            self.assertEqual(leg.route_kind, ADMITTED_POSE_ROUTE_KIND)
            self.assertEqual(leg.raw_waypoints[-1].pose, args["target"])
            self.assertEqual(len(leg.raw_waypoints), 2)
            self.assertTrue(leg.raw_waypoints[-1].protected)
            self.assertTrue(math.isnan(leg.raw_waypoints[0].pose.yaw_rad))
            cert = load_execution_route_certificate(Path(result["route_certificate_json"]))
            self.assertEqual(cert.route_kind, ADMITTED_POSE_ROUTE_KIND)
            self.assertTrue(cert.exact_vertex_pursuit)
            self.assertEqual(cert.candidate_snapshot_sha256, candidate_snapshot_sha256(args["snapshot"]))
            self.assertTrue(validate_admitted_pose_route_binding(Path(result["diagnostics_json"]), leg, candidate_snapshot_path=Path(result["candidate_snapshot"])).ok)
            self.assertIn(ADMITTED_POSE_ROUTE_KIND, STATIC_PHYSICAL_ROUTE_KINDS)
            self.assertIn(ADMITTED_POSE_ROUTE_KIND, TERMINAL_HEADING_ONLY_PHYSICAL_ROUTE_KINDS)

    def test_blocked_target_is_rejected_without_snapping_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._fixture(Path(directory), target=Pose2D(.90, .02, -.71))
            with self.assertRaisesRegex(ValueError, "exact stored target is blocked"):
                plan_admitted_pose_route(**args)
            self.assertFalse(args["output_dir"].exists())

    def test_non_target_candidate_stays_in_route_keepouts(self):
        with tempfile.TemporaryDirectory() as directory:
            obstacle = fixtures.CandidatePreapproachPlanningTest._candidate("unconfirmed_stand", .0, -.03)
            args = self._fixture(Path(directory), extra_candidates=(obstacle,))
            result = plan_admitted_pose_route(**args)
            leg = load_route_leg(Path(result["route_csv"]), 0, thinning_min_spacing_m=0.)
            self.assertGreater(len(leg.raw_waypoints), 2)
            poses = [w.pose for w in leg.raw_waypoints]
            for start, end in zip(poses, poses[1:]):
                self.assertGreaterEqual(point_to_segment_distance_m(Pose2D(.0, -.03), start, end), .31)

    def test_changed_source_artifact_rejected_by_child(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._fixture(Path(directory))
            result = plan_admitted_pose_route(**args)
            Path(args["target_evidence"]["source_artifacts"][0]["path"]).write_text("changed")
            leg = load_route_leg(Path(result["route_csv"]), 0, thinning_min_spacing_m=0.)
            status = validate_admitted_pose_route_binding(Path(result["diagnostics_json"]), leg, candidate_snapshot_path=Path(result["candidate_snapshot"]))
            self.assertFalse(status.ok)
            self.assertIn("source artifact hash", status.failures[0])

    def test_pose_swapping_and_wrong_qr_are_rejected(self):
        for field, value in (("qr_id", "start"), ("stored_pose", {"x_m": .44, "y_m": .213, "yaw_rad": -.71})):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                args = self._fixture(Path(directory))
                args["target_evidence"][field] = value
                with self.assertRaises(ValueError):
                    plan_admitted_pose_route(**args)
                self.assertFalse(args["output_dir"].exists())

    def test_stale_snapshot_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._fixture(Path(directory))
            changed = replace(args["snapshot"], created_unix_sec=4.)
            args["snapshot_path"].unlink()
            write_candidate_snapshot(args["snapshot_path"], changed)
            with self.assertRaisesRegex(ValueError, "snapshot file mismatch"):
                plan_admitted_pose_route(**args)
            self.assertFalse(args["output_dir"].exists())

    def test_measured_target_center_remains_an_obstacle(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._fixture(Path(directory))
            args["target_evidence"]["measured_target_center"] = {
                "x_m": .62, "y_m": .20, "uncertainty_m": .02,
            }
            with self.assertRaisesRegex(ValueError, "exact stored target is blocked"):
                plan_admitted_pose_route(**args)
            self.assertFalse(args["output_dir"].exists())

    def test_retained_facing_baseline_uncertainty_is_not_charged_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._fixture(Path(directory), target=Pose2D(.65, 0., -.71), fine_grid=True)
            args["physical_clearance"] = {
                **fixtures.PHYSICAL_CLEARANCE,
                "minimum_active_standoff_m": .35,
                "minimum_candidate_transit_radius_m": .34,
                "minimum_collision_standoff_m": .34,
            }
            args["target_evidence"]["measured_target_center"] = {
                "x_m": 1., "y_m": 0., "uncertainty_m": .02,
            }
            result = plan_admitted_pose_route(**args)
            leg = load_route_leg(Path(result["route_csv"]), 0)
            self.assertEqual(leg.raw_waypoints[-1].pose, args["target"])

    def test_malformed_measured_center_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._fixture(Path(directory))
            args["target_evidence"]["measured_target_center"] = {
                "x_m": 1., "y_m": 0., "uncertainty_m": -.1,
            }
            with self.assertRaisesRegex(ValueError, "measured target center"):
                plan_admitted_pose_route(**args)
            self.assertFalse(args["output_dir"].exists())

    def test_start_must_match_fresh_planning_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._fixture(Path(directory))
            args["start"] = Pose2D(-.45, -.20, .3)
            with self.assertRaisesRegex(ValueError, "route start differs"):
                plan_admitted_pose_route(**args)
            self.assertFalse(args["output_dir"].exists())

    def test_heading_only_return_preserves_position_without_detour(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._fixture(Path(directory), target=Pose2D(-.40, -.20, -.71))
            result = plan_admitted_pose_route(**args)
            leg = load_route_leg(Path(result["route_csv"]), 0)
            self.assertEqual(len(leg.raw_waypoints), 2)
            self.assertEqual(leg.route_length_m, 0.)
            self.assertEqual(leg.raw_waypoints[-1].pose, args["target"])
            self.assertEqual(leg.raw_waypoints[0].pose.x_m, args["target"].x_m)
            self.assertEqual(leg.raw_waypoints[0].pose.y_m, args["target"].y_m)
            metadata = json.loads(Path(result["diagnostics_json"]).read_text())["metadata"]
            self.assertIs(metadata["stationary_turn"], True)
            self.assertEqual(metadata["exact_start_connector"]["exact_start"], asdict(args["start"]))
            poses = tuple(w.pose for w in leg.raw_waypoints)
            self.assertEqual(certified_static_startup_decision(
                args["start"], poses, tracking_tube_radius_m=.03,
            ).target_index, 1)
            for offset, expected in ((.02, True), (.04, False)):
                observed = replace(args["start"], x_m=args["start"].x_m + offset)
                self.assertEqual(check_execution_route_tube(
                    observed, poses, target_index=1, pursuit_index=1,
                    tracking_tube_radius_m=.03,
                ).ok, expected)

    def test_terminal_yaw_or_target_evidence_changes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._fixture(Path(directory))
            result = plan_admitted_pose_route(**args)
            leg = load_route_leg(Path(result["route_csv"]), 0, thinning_min_spacing_m=0.)
            target = leg.raw_waypoints[-1]
            altered = replace(target, pose=replace(target.pose, yaw_rad=0.))
            changed = replace(leg, raw_waypoints=(*leg.raw_waypoints[:-1], altered))
            status = validate_admitted_pose_route_binding(Path(result["diagnostics_json"]), changed, candidate_snapshot_path=Path(result["candidate_snapshot"]))
            self.assertFalse(status.ok)
            Path(result["target_evidence_json"]).write_text("{}")
            status = validate_admitted_pose_route_binding(Path(result["diagnostics_json"]), leg, candidate_snapshot_path=Path(result["candidate_snapshot"]))
            self.assertFalse(status.ok)
            self.assertIn("evidence hash", status.failures[0])


if __name__ == "__main__":
    unittest.main()
