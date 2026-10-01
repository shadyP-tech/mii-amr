"""Replay recorded Start geometry without live localization or robot effects."""

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.navigation.approach.admitted_pose_route import (
    plan_admitted_pose_route, validate_admitted_pose_route_binding,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import CoverageSurveyConfig, CoverageSurveyPlan
from scripts.aufgabe04.navigation.execution.execution_route_certificate import point_to_segment_distance_m
from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import map_pose_to_odom, odom_pose_to_map
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.exact_start_connector import _segment_clearance_evidence
from scripts.aufgabe04.navigation.planning.map_io import load_occupancy_grid_with_bundle
from scripts.aufgabe04.navigation.planning.route_smoothing import segment_is_collision_free
from scripts.aufgabe04.navigation.planning.waypoint_csv import load_route_leg
from scripts.aufgabe04.stations.candidate_snapshot import candidate_snapshot_sha256, load_candidate_snapshot
from scripts.aufgabe04.stations.models import Station, StationPose


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "fixtures/start_return_20261001"


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_arguments(output):
    inputs_path = FIXTURE / "inputs.json"
    data = json.loads(inputs_path.read_text())
    snapshot_path = FIXTURE / "candidate_snapshot.json"
    snapshot = load_candidate_snapshot(snapshot_path)
    values = data["coverage_plan_consumed_fields"]
    plan = CoverageSurveyPlan(**{
        **values, "config": CoverageSurveyConfig(**values["config"]),
        "arena_bounds": ArenaBounds(**values["arena_bounds"]),
        "viewpoints": (), "surveyable_cells": (), "planned_covered_cells": (),
    })
    source = CandidatePlanningFrame.from_evidence(data["source_planning_frame"])
    current = CandidatePlanningFrame.from_evidence(data["current_planning_frame"])
    stored = Pose2D(**data["start_record"]["facing_pose"])
    target = odom_pose_to_map(map_pose_to_odom(stored, source.map_from_odom), current.map_from_odom)
    measured_stand = data["measured_target_source_stand"]
    old_measured = Pose2D(**measured_stand["center"])
    measured = odom_pose_to_map(map_pose_to_odom(old_measured, source.map_from_odom), current.map_from_odom)
    evidence = {
        "candidate_uid": data["start_record"]["candidate_uid"], "qr_id": "Start",
        "planning_frame": snapshot.planning_frame,
        "candidate_snapshot_sha256": candidate_snapshot_sha256(snapshot),
        "target_pose": asdict(target), "stored_pose": asdict(stored),
        "source_planning_frame": source.to_evidence(),
        "planning_frame_admission": current.to_evidence(),
        "source_artifacts": [{"path": str(p.resolve()), "sha256": file_hash(p)}
                             for p in (inputs_path, snapshot_path)],
        "measured_target_center": {"x_m": measured.x_m, "y_m": measured.y_m,
                                   "uncertainty_m": measured_stand["uncertainty_m"]},
    }
    return data, {
        "map_yaml": ROOT / data["map_yaml"], "semantic_map_id": data["semantic_map_id"],
        "plan": plan, "snapshot": snapshot, "snapshot_path": snapshot_path,
        "candidate_uid": data["start_record"]["candidate_uid"],
        "start": current.current_pose, "target": target, "output_dir": output,
        "inflation_radius_m": plan.config.inflation_radius_m,
        "physical_clearance": data["physical_clearance"], "target_evidence": evidence,
    }


def costmaps(args):
    grid, bundle = load_occupancy_grid_with_bundle(
        args["map_yaml"], semantic_map_id=args["semantic_map_id"], planning_frame="map")
    base = Costmap.from_occupancy_grid(grid).with_arena_bounds(args["plan"].arena_bounds)
    inflated = base.with_inflation(args["inflation_radius_m"])
    radius = args["physical_clearance"]["minimum_candidate_transit_radius_m"]
    planning = inflated.with_station_keepouts(tuple(
        Station(c.candidate_uid, StationPose(c.geometry.x_m, c.geometry.y_m, 0.), 0.,
                max(radius, c.geometry.keepout_radius_m)) for c in args["snapshot"].candidates))
    return bundle, base, inflated, planning


class RecordedStartReturnTests(unittest.TestCase):
    def test_source_snapshot_map_and_reprojected_physical_geometry_are_unchanged(self):
        data, args = recorded_arguments(Path("unused"))
        bundle, _, _, _ = costmaps(args)
        self.assertEqual(bundle.bundle_sha256, data["map_bundle_sha256"])
        self.assertEqual(file_hash(FIXTURE / "candidate_snapshot.json"),
                         data["source_files"][data["candidate_snapshot_source"]]["sha256"])
        for path in (data["map_yaml"], str(Path(data["map_yaml"]).with_suffix(".pgm"))):
            self.assertEqual(file_hash(ROOT / path), data["source_files"][path]["sha256"])
        for field, value in data["failure"]["start_target_pose"].items():
            self.assertAlmostEqual(getattr(args["target"], field), value, places=13)
        old = CandidatePlanningFrame.from_evidence(data["source_planning_frame"])
        current = CandidatePlanningFrame.from_evidence(data["current_planning_frame"])
        stored = Pose2D(**data["start_record"]["facing_pose"])
        self.assertEqual(len(args["snapshot"].candidates), 6)
        for candidate in args["snapshot"].candidates:
            with self.subTest(candidate=candidate.candidate_uid):
                before = data["source_candidate_geometries"][candidate.candidate_uid]
                old_center = Pose2D(before["x_m"], before["y_m"])
                new_center = odom_pose_to_map(map_pose_to_odom(old_center, old.map_from_odom), current.map_from_odom)
                self.assertAlmostEqual(new_center.x_m, candidate.geometry.x_m, places=12)
                self.assertAlmostEqual(new_center.y_m, candidate.geometry.y_m, places=12)
                old_distance = math.hypot(stored.x_m-old_center.x_m, stored.y_m-old_center.y_m)
                new_distance = math.hypot(args["target"].x_m-new_center.x_m, args["target"].y_m-new_center.y_m)
                self.assertAlmostEqual(old_distance, new_distance, places=12)
                self.assertEqual(candidate.geometry.keepout_radius_m, before["keepout_radius_m"])

    def test_recorded_raster_rejection_is_not_continuous_keepout_penetration(self):
        data, args = recorded_arguments(Path("unused"))
        _, _, inflated, planning = costmaps(args)
        self.assertEqual(data["failure"]["error"], "exact stored target is blocked; goal snapping is forbidden")
        self.assertFalse(data["failure"]["start_return_motion_published"])
        self.assertFalse(segment_is_collision_free(planning, args["target"], args["target"]))
        self.assertTrue(segment_is_collision_free(inflated, args["target"], args["target"]))
        active = args["snapshot"].candidate_for(args["candidate_uid"])
        distance = math.hypot(args["target"].x_m-active.geometry.x_m, args["target"].y_m-active.geometry.y_m)
        self.assertAlmostEqual(distance, .3417084174064008)
        self.assertAlmostEqual(distance-active.geometry.keepout_radius_m, .0017084174064008195)

    def test_corrected_full_route_preserves_exact_target_yaw_and_all_keepouts(self):
        with tempfile.TemporaryDirectory() as directory:
            data, args = recorded_arguments(Path(directory) / "return")
            original_snapshot_hash = candidate_snapshot_sha256(args["snapshot"])
            result = plan_admitted_pose_route(**args)
            self.assertTrue(result["is_final_stage"])
            leg = load_route_leg(Path(result["route_csv"]), 0, thinning_min_spacing_m=0.)
            status = validate_admitted_pose_route_binding(
                Path(result["diagnostics_json"]), leg,
                candidate_snapshot_path=Path(result["candidate_snapshot"]))
            self.assertTrue(status.ok, status.failures)
            poses = tuple(point.pose for point in leg.raw_waypoints)
            self.assertGreaterEqual(len(poses), 3)
            self.assertEqual(poses[-1], args["target"])
            self.assertEqual(result["stage_target_pose"], asdict(args["target"]))
            self.assertEqual(candidate_snapshot_sha256(load_candidate_snapshot(Path(result["candidate_snapshot"]))),
                             original_snapshot_hash)
            metadata = json.loads(Path(result["diagnostics_json"]).read_text())["metadata"]
            proof = metadata["exact_goal_connector"]
            self.assertEqual(proof["target"], asdict(args["target"]))
            self.assertEqual((proof["anchor"]["x_m"], proof["anchor"]["y_m"]),
                             (poses[-2].x_m, poses[-2].y_m))
            self.assertTrue(proof["candidate_keepouts_continuously_validated"])
            self.assertTrue(proof["static_clearance"]["validated"])
            self.assertFalse(proof["motion_authorized"])
            self.assertEqual(metadata["stored_start_target_pose"], asdict(args["target"]))
            _, base, inflated, planning = costmaps(args)
            self.assertFalse(segment_is_collision_free(planning, poses[-1], poses[-1]))
            # Recompute every segment against the unchanged full candidate
            # population, independent of any connector receipt's summaries.
            transit = data["physical_clearance"]["minimum_candidate_transit_radius_m"]
            for candidate in args["snapshot"].candidates:
                with self.subTest(candidate=candidate.candidate_uid):
                    center = Pose2D(candidate.geometry.x_m, candidate.geometry.y_m)
                    actual = min(point_to_segment_distance_m(center, a, b) for a, b in zip(poses, poses[1:]))
                    self.assertGreaterEqual(actual+1e-9, max(transit, candidate.geometry.keepout_radius_m))
            measured = args["target_evidence"]["measured_target_center"]
            center = Pose2D(measured["x_m"], measured["y_m"])
            active = args["snapshot"].candidate_for(args["candidate_uid"])
            required = data["physical_clearance"]["minimum_collision_standoff_m"] + max(
                0., measured["uncertainty_m"]-active.geometry.uncertainty_m)
            self.assertGreaterEqual(min(point_to_segment_distance_m(center, a, b)
                for a, b in zip(poses, poses[1:])), required)
            for index, (a, b) in enumerate(zip(poses, poses[1:])):
                with self.subTest(segment=index):
                    self.assertTrue(segment_is_collision_free(inflated, a, b))
                    # Intermediate CSV headings are intentionally unspecified
                    # (NaN); circular x/y clearance is rotation-independent.
                    clearance = _segment_clearance_evidence(
                        base, Pose2D(a.x_m, a.y_m, 0.), Pose2D(b.x_m, b.y_m, 0.),
                        required_clearance_m=args["inflation_radius_m"])
                    self.assertTrue(clearance.validated)


if __name__ == "__main__":
    unittest.main()
