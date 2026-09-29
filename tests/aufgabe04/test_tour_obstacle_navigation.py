"""Tour orchestration preserves target and never promotes a stopped outcome."""

from dataclasses import asdict, replace
import json
from pathlib import Path
from unittest.mock import Mock, patch
import unittest

from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.real_robot.mission.stored_start_pose import load_stored_admitted_poses
from scripts.aufgabe04.real_robot.mission.stored_pose_navigation import StoredPoseNavigationEffects
from scripts.aufgabe04.real_robot.mission import tour_obstacle_navigation as navigation
from tests.aufgabe04 import test_start_return as source_fixture


class TourObstacleNavigationTest(unittest.TestCase):
    def setUp(self):
        self.fixture = source_fixture.StartReturnTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.config, completed = self.fixture.completed("QR_002")
        self.stored = load_stored_admitted_poses(completed, self.config)["QR_002"]
        self.root = self.config.session_root.parent / "tour_visit"
        self.map = Mock(tour_id="tour1", map_bundle_sha256=self.config.plan.map_bundle_sha256)
        self.map.write_projection.side_effect = lambda path, frame: path
        self.scan = Mock(side_effect=lambda path: path)
        self.requests, self.plans = [], []
        self.writer = patch.object(navigation, "write_tour_terminal_evidence",
            side_effect=lambda path, **kwargs: (path, "e"*64)).start()
        self.addCleanup(patch.stopall)

    @staticmethod
    def frame(x=0., yaw=0., map_x=.4):
        return CandidatePlanningFrame(Pose2D(x, 0., yaw), PlanarTransform2D(map_x, 0., 0.))

    def effects(self, frames, outcomes, *, intermediate=False):
        frames, outcomes = iter(frames), iter(outcomes)
        def plan(**kwargs):
            self.plans.append(kwargs)
            final = not intermediate or kwargs["return_stage_index"] > 0
            return {"route_csv": "route.csv", "diagnostics_json": "diagnostics.json",
                "route_certificate_json": "certificate.json", "candidate_snapshot": str(kwargs["snapshot_path"]),
                "is_final_stage": final, "stage_target_pose": asdict(kwargs["target"] if final else Pose2D(.35, 0., 0.))}
        def move(request):
            self.assertFalse((self.root / "arrival.json").exists())
            self.requests.append(request)
            outcome = self.fixture.fixture.case.fixture._completed(request)
            mode = next(outcomes)
            if mode == "completed":
                return outcome
            return replace(outcome, status="stopped", returncode=1,
                stop_reason="obstacle too close" if mode == "blocked" else mode,
                stop_details={"source": "global_scan", "valid_sample_count": 20,
                              "nearest_valid_range_m": .18, "threshold_m": .20})
        return StoredPoseNavigationEffects(lambda path: next(frames), move, plan_route=plan,
            load_route_uncertainty_readiness=self.fixture.uncertainty)

    def execute(self, effects):
        return navigation.execute_tour_obstacle_navigation(self.stored, self.config, effects,
            tour_session_id="tour1", visit_index=2, output_root=self.root,
            obstacle_map=self.map, capture_scan=self.scan)

    def test_blocked_final_leg_replans_same_stage_and_exact_target_in_fresh_frame(self):
        result = self.execute(self.effects([
            self.frame(), self.frame(.1, map_x=.5), self.frame(.8, .1, map_x=.5),
        ], ["blocked", "completed"]))
        self.assertTrue(result["arrival_verified"])
        self.assertEqual(result["replan_count"], 1)
        self.assertEqual([r.mission_leg_index for r in self.requests], [12, 13])
        self.assertTrue(all(set(r.sealed) == {"route_csv", "diagnostics_json", "route_certificate_json"} for r in self.requests))
        self.assertEqual([p["return_stage_index"] for p in self.plans], [0, 0])
        self.assertAlmostEqual(self.plans[0]["target"].x_m, .7)
        self.assertAlmostEqual(self.plans[1]["target"].x_m, .8)
        evidence = [p["target_evidence"] for p in self.plans]
        self.assertTrue(all(e["qr_id"] == "QR_002" for e in evidence))
        self.assertEqual(evidence[1]["tour_navigation"]["replan_count"], 1)
        self.assertEqual(evidence[1]["tour_navigation"]["previous_terminal_sha256"], "e"*64)
        self.assertEqual(self.writer.call_args_list[0].kwargs["outcome"].status, "stopped")
        self.assertEqual(self.writer.call_args_list[0].kwargs["scan_capture_path"], self.root / "legs/001/scan_capture.json")
        self.assertEqual(self.map.update_from_capture.call_count, 2)
        self.assertFalse(result["legs"][0]["arrival_verified"])

    def test_successful_intermediate_stage_then_blockage_has_separate_counters(self):
        result = self.execute(self.effects([
            self.frame(), self.frame(.35), self.frame(.35), self.frame(.35), self.frame(.7, .1),
        ], ["completed", "blocked", "completed"], intermediate=True))
        self.assertEqual(result["leg_count"], 3)
        self.assertEqual([p["target_evidence"]["tour_navigation"]["stage_index"] for p in self.plans], [0, 1, 1])
        self.assertEqual([p["target_evidence"]["tour_navigation"]["replan_count"] for p in self.plans], [0, 0, 1])
        self.assertIsNotNone(self.writer.call_args_list[0].kwargs["arrival_path"])
        self.assertIsNone(self.writer.call_args_list[0].kwargs["scan_capture_path"])

    def test_two_replans_exhaust_budget_without_arrival(self):
        with self.assertRaisesRegex(RuntimeError, "replan budget exhausted"):
            self.execute(self.effects([self.frame()]*3, ["blocked"]*3))
        self.assertEqual(len(self.requests), 3)
        self.assertFalse((self.root / "arrival.json").exists())
        self.assertFalse(json.loads((self.root / "failure.json").read_text())["arrival_verified"])

    def test_non_geometric_fault_is_not_retried(self):
        with self.assertRaisesRegex(RuntimeError, "scan stale"):
            self.execute(self.effects([self.frame()], ["scan stale"]))
        self.assertEqual(len(self.requests), 1)
        self.writer.assert_not_called()

    def test_no_admissible_detour_keeps_blocked_visit_unreported(self):
        effects = self.effects([self.frame()]*2, ["blocked"])
        planner = effects.plan_route
        def plan(**kwargs):
            if kwargs["target_evidence"]["tour_navigation"]["replan_count"]:
                raise RuntimeError("no safe detour")
            return planner(**kwargs)
        with self.assertRaisesRegex(RuntimeError, "no safe detour"):
            self.execute(replace(effects, plan_route=plan))
        self.assertEqual(len(self.requests), 1)
        self.assertFalse((self.root / "arrival.json").exists())

    def test_failed_child_near_target_is_not_promoted_to_arrival(self):
        effects = self.effects([self.frame(), self.frame(.7, .1)], ["blocked", "scan stale"])
        with self.assertRaisesRegex(RuntimeError, "scan stale"):
            self.execute(effects)
        self.assertEqual(len(self.requests), 2)
        self.assertFalse((self.root / "arrival.json").exists())

    def test_already_at_target_still_checks_source_hashes(self):
        Path(self.stored.evidence["source_artifacts"][0]["path"]).write_text("changed")
        effects = self.effects([self.frame(.7, .1)], [])
        with self.assertRaisesRegex(ValueError, "source artifact hash mismatch"):
            self.execute(effects)
        self.assertEqual(self.requests, [])
        self.scan.assert_not_called()

    def test_successful_child_with_bad_arrival_fails_without_terminal_proof(self):
        with self.assertRaisesRegex(RuntimeError, "arrival tolerance"):
            self.execute(self.effects([self.frame(), self.frame(.4)], ["completed"]))
        self.writer.assert_not_called()
        self.assertFalse((self.root / "arrival.json").exists())


if __name__ == "__main__":
    unittest.main()
