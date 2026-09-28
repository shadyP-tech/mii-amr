"""Independent visits preserve QR identity and use fresh staged motion."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import MissionLegKind
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.real_robot.mission.stored_start_pose import load_stored_admitted_pose, load_stored_admitted_poses
from scripts.aufgabe04.real_robot.mission.stored_pose_navigation import StoredPoseNavigationEffects, execute_stored_pose_navigation
from tests.aufgabe04 import test_start_return as start_fixture


class StoredPoseNavigationTest(unittest.TestCase):
    def setUp(self):
        self.case = start_fixture.StartReturnTest()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)

    def source(self, *, facing=False):
        config, completed = self.case.completed("QR002", facing=facing)
        return config, load_stored_admitted_poses(completed, config)["QR002"]

    def test_both_catalog_kinds_load_their_actual_qr(self):
        for facing in (False, True):
            case = start_fixture.StartReturnTest()
            case.setUp()
            try:
                config, completed = case.completed("QR002", facing=facing)
                poses = load_stored_admitted_poses(completed, config)
                self.assertEqual(set(poses), {"QR002"})
                self.assertEqual(poses["QR002"].evidence["qr_id"], "QR002")
                self.assertEqual(poses["QR002"].evidence["pose_kind"],
                    "geometry_validated_facing_pose" if facing else "qr_verified_observation_pose")
                with self.assertRaises(ValueError):
                    load_stored_admitted_pose(completed, config, qr_id="QR003")
            finally:
                case.doCleanups()

    def test_two_stages_bind_tour_visit_and_reproject_actual_target(self):
        config, stored = self.source()
        output = config.session_root.parent / "tour" / "visits" / "002"
        frames = iter((
            CandidatePlanningFrame(Pose2D(0., 0., 0.), PlanarTransform2D(.4, 0., 0.)),
            CandidatePlanningFrame(Pose2D(.5, 0., 0.), PlanarTransform2D(.5, 0., 0.)),
            CandidatePlanningFrame(Pose2D(.9, 0., .1), PlanarTransform2D(.6, 0., 0.)),
        ))
        plans, requests = [], []
        def plan(**kwargs):
            plans.append(kwargs)
            final = kwargs["return_stage_index"] == 1
            return {"candidate_snapshot": str(kwargs["snapshot_path"]), "is_final_stage": final,
                "stage_target_pose": asdict(kwargs["target"] if final else Pose2D(.4, 0., 0.)),
                "route_csv": "route.csv", "diagnostics_json": "diagnostics.json",
                "route_certificate_json": "certificate.json", "target_evidence_json": "not_a_child_artifact.json"}
        def motion(request):
            requests.append(request)
            return self.case.fixture.case.fixture._completed(request)
        effects = StoredPoseNavigationEffects(lambda _: next(frames), motion, plan_route=plan,
            load_route_uncertainty_readiness=self.case.uncertainty)
        result = execute_stored_pose_navigation(stored, config, effects,
            tour_session_id="tour1", visit_index=2, output_root=output)
        self.assertTrue(result["arrival_verified"])
        self.assertTrue(result["target_pose_reached"])
        self.assertEqual(result["qr_id"], "QR002")
        self.assertEqual(result["leg_count"], 2)
        self.assertEqual([r.mission_leg_index for r in requests], [8, 9])
        self.assertTrue(all(r.mission_leg_kind is MissionLegKind.STORED_POSE_TOUR for r in requests))
        self.assertTrue(all(r.session_id == "tour1" and r.session_root == output for r in requests))
        self.assertTrue(all(set(r.sealed) == {"route_csv", "diagnostics_json", "route_certificate_json"} for r in requests))
        for index, planned in enumerate(plans):
            self.assertEqual(planned["purpose"], "stored_pose_tour")
            self.assertEqual(planned["target_evidence"]["qr_id"], "QR002")
            self.assertEqual(planned["target_evidence"]["tour_id"], "tour1")
            self.assertEqual(planned["target_evidence"]["visit_index"], 2)
            self.assertEqual(planned["snapshot"].candidate_uids, ("candidate_a", "unvisited"))
            self.assertAlmostEqual(planned["target"].x_m, .7 + .1 * index)
        self.assertFalse((config.session_root / "return_to_start").exists())
        arrival = json.loads((output / "arrival.json").read_text())
        self.assertEqual(arrival["qr_id"], "QR002")
        self.assertTrue(arrival["arrival_verified"])

    def test_changed_sources_fail_even_when_already_at_target(self):
        config, stored = self.source()
        Path(stored.evidence["catalog_path"]).write_text("changed")
        effects = StoredPoseNavigationEffects(Mock(), Mock())
        output = config.session_root.parent / "visit"
        with self.assertRaisesRegex(ValueError, "source artifact hash"):
            execute_stored_pose_navigation(stored, config, effects,
                tour_session_id="tour1", visit_index=0, output_root=output)
        effects.admit_planning_frame.assert_not_called()
        effects.run_motion_leg.assert_not_called()
        self.assertFalse(json.loads((output / "failure.json").read_text())["arrival_verified"])

    def test_invalid_tour_identity_rejected_before_effects(self):
        config, stored = self.source()
        effects = StoredPoseNavigationEffects(Mock(), Mock())
        for identity in ("tour..x", "../tour", ""):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                execute_stored_pose_navigation(stored, config, effects,
                    tour_session_id=identity, visit_index=0, output_root=config.session_root.parent / "visit")
        effects.admit_planning_frame.assert_not_called()


if __name__ == "__main__":
    unittest.main()
