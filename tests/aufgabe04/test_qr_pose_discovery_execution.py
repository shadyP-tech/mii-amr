"""The QR-only outcome must finish discovery without manufacturing geometry."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json
from scripts.aufgabe04.artifacts.qr_verified_observation_pose import (
    SOURCE_GATES, build_qr_verified_observation_pose,
)
from scripts.aufgabe04.real_robot.candidate.approach import (
    CandidateCameraCheckpoint, CandidateObservation, execute_candidate_approach_phase,
)
from scripts.aufgabe04.real_robot.mission.camera_goal_reporting import validate_completed_qr_goal
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.stations.candidate_snapshot import candidate_snapshot_sha256
from tests.aufgabe04 import test_candidate_qr_goal_execution as fixtures


class QrPoseDiscoveryExecutionTest(unittest.TestCase):
    def setUp(self):
        self.case = fixtures.CandidateQrGoalExecutionTest()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)

    def effects(self, config, *, qr_only, identities=None, mutate=None):
        base = self.case.effects(qr_by_uid=identities)
        capture_geometry = base.capture_observation

        def capture(request):
            uid = request.candidate.candidate_uid
            if uid not in qr_only:
                return capture_geometry(request)
            self.case.visited.append(uid)
            qr = (identities or {}).get(uid, f"QR_{uid}")
            cluster = {"associated": True, "eligible_cluster_count": 1,
                       "scan_stamp_sec": 9.9, "scan_frame_id": "base_scan",
                       "selected_cluster_source_indices": [0, 1, 2]}
            fields = dict(
                candidate_uid=uid, qr_id=qr, planning_frame="map",
                stream_id=f"{config.session_id}_{uid}", target_key=uid,
                stand_center={"x_m": request.candidate.geometry.x_m,
                              "y_m": request.candidate.geometry.y_m},
                robot_pose={"x_m": -.4, "y_m": .05, "yaw_rad": .1},
                sensor_stamp_sec=9.9, scan_stamp_sec=9.9, checked_at_sec=10.,
                robot_profile_sha256="a" * 64, calibration_profile_sha256="b" * 64,
                stand_model_profile_sha256="c" * 64,
                image_shape=[480, 640], qr_corners_px=[[10, 10], [30, 10], [30, 30], [10, 30]],
                qr_binding={"accepted": True, "reason": "decoded_qr_target_associated",
                            "symbol_count": 1, "qr_texts_for_evidence": [qr],
                            "camera_bearing_rad": 0., "association": cluster},
                motion_epoch=0, camera_signature=[400., 400., 320., 240.],
                source_gates={key: True for key in SOURCE_GATES},
                localization_provenance={"map_frame": "map", "base_frame": "base_footprint",
                    "scan_frame": "base_scan", "camera_frame": "camera",
                    "exact_image_transform_stamp_sec": 9.9, "exact_scan_transform_stamp_sec": 9.9},
            )
            if mutate is not None:
                mutate(fields)
            payload = build_qr_verified_observation_pose(**fields)
            path = request.output_dir / "qr_observation_pose.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
            return CandidateObservation(None, qr, None, qr_observation_pose_path=path)

        def facing(request):
            self.assertNotIn(request.candidate.candidate_uid, qr_only)
            return base.validate_facing(request)

        return replace(base, capture_observation=capture, validate_facing=facing)

    def test_all_fallbacks_complete_five_qr_goal_in_one_view_each(self):
        config = self.case.config()
        fallback_uids = {f"candidate_{i}" for i in range(6)}
        result = execute_candidate_approach_phase(config, self.effects(config, qr_only=fallback_uids))
        self.case.assert_bound_artifacts(config, result)
        self.assertEqual(len(self.case.visited), 5)
        self.assertFalse(self.case.committed)
        self.assertFalse(result.facing_records)
        self.assertEqual(len(result.qr_observation_records), 5)
        self.assertTrue(all(snapshot == config.snapshot for snapshot in self.case.planned_snapshots))
        facing = load_content_hashed_json(result.stand_facing_catalog_path,
                                          hash_field="stand_facing_catalog_sha256")
        self.assertFalse(facing["facing_complete"])
        self.assertEqual(facing["records"], [])
        observations = load_content_hashed_json(result.qr_observation_catalog_path,
                                                hash_field="qr_observation_pose_catalog_sha256")
        self.assertEqual(observations["stand_count"], 5)
        for record in observations["records"]:
            self.assertIsNone(record["stand_axis_rad"])
            self.assertFalse(record["facing_ready"])
            self.assertEqual(record["robot_observation_pose"], {"x_m": -.4, "y_m": .05, "yaw_rad": .1})
            candidate = config.snapshot.candidate_for(record["candidate_uid"])
            self.assertEqual(record["stand_center"]["x_m"], candidate.geometry.x_m)
            raw = json.loads(Path(record["qr_observation_pose_json"]).read_text())
            self.assertEqual(record["qr_verified_observation_pose_sha256"], raw["qr_verified_observation_pose_sha256"])
        summary = result.to_mission_summary_fields()
        self.assertTrue(summary["goal_completed"])
        self.assertFalse(summary["facing_complete"])
        self.assertEqual(summary["qr_only_stand_count"], 5)
        validate_completed_qr_goal(summary, snapshot_sha256=candidate_snapshot_sha256(config.snapshot), coverage=None)

    def test_mixed_discovery_retains_only_geometry_records_in_facing_catalog(self):
        config = self.case.config()
        result = execute_candidate_approach_phase(
            config, self.effects(config, qr_only={"candidate_1", "candidate_3"}))
        self.case.assert_bound_artifacts(config, result)
        self.assertEqual(len(result.facing_records), 3)
        self.assertEqual(len(result.qr_observation_records), 2)
        self.assertEqual(self.case.committed, ["candidate_0", "candidate_2", "candidate_4"])

    def test_duplicate_across_fallback_and_geometry_revokes_both_and_continues(self):
        config = self.case.config(pool_size=7)
        result = execute_candidate_approach_phase(config, self.effects(
            config, qr_only={"candidate_0", "candidate_3"},
            identities={"candidate_0": "QR_DUPLICATE", "candidate_1": "QR_DUPLICATE"}))
        self.case.assert_bound_artifacts(config, result)
        self.assertEqual(len(self.case.visited), 7)
        self.assertEqual({record["candidate_uid"] for record in result.qr_observation_records}, {"candidate_3"})
        self.assertNotIn("candidate_0", result.visit_order)
        self.assertNotIn("candidate_1", result.visit_order)

    def test_geometry_claim_then_duplicate_fallback_revokes_the_facing_record(self):
        config = self.case.config(pool_size=7)
        result = execute_candidate_approach_phase(config, self.effects(
            config, qr_only={"candidate_1", "candidate_3"},
            identities={"candidate_0": "QR_DUPLICATE", "candidate_1": "QR_DUPLICATE"}))
        self.case.assert_bound_artifacts(config, result)
        self.assertEqual(len(self.case.visited), 7)
        self.assertNotIn("candidate_0", {record["candidate_uid"] for record in result.facing_records})
        self.assertNotIn("candidate_1", {record["candidate_uid"] for record in result.qr_observation_records})

    def test_fallback_uses_projected_arrival_frame_after_drive(self):
        fixture = self.case.fixture
        config = fixture._config(self.case.root, (fixture._candidate("candidate_a", 2., 0.),))
        config = fixture._write_frame_registry(config, frozen_map_from_odom=PlanarTransform2D(1., 0., 0.))
        frames = iter((
            CandidatePlanningFrame(Pose2D(0., 0., 0.), PlanarTransform2D(0., 0., 0.)),
            CandidatePlanningFrame(Pose2D(.5, 0., 0.), PlanarTransform2D(.2, 0., 0.)),
        ))
        effects = self.effects(config, qr_only={"candidate_a"},
                               mutate=lambda data: data.update(robot_pose={"x_m": .5, "y_m": 0., "yaw_rad": 0.}))
        result = execute_candidate_approach_phase(config, replace(
            effects, admit_planning_frame=lambda _path: next(frames)))
        record, = result.qr_observation_records
        self.assertAlmostEqual(record["stand_center"]["x_m"], 1.2)
        self.assertEqual(record["robot_observation_pose"]["x_m"], .5)
        self.assertEqual(config.snapshot.candidate_for("candidate_a").geometry.x_m, 2.)
        self.assertTrue(Path(record["candidate_frame_projection_path"]).is_file())
        self.assertTrue(Path(record["camera_candidate_snapshot_path"]).is_file())

    def test_wrong_candidate_receipt_stops_before_counting_or_visiting_next(self):
        config = self.case.config()
        with self.assertRaisesRegex(RuntimeError, "candidate attempt"):
            execute_candidate_approach_phase(config, self.effects(
                config, qr_only={"candidate_0"}, mutate=lambda data: data.update(candidate_uid="wrong")))
        self.assertEqual(self.case.visited, ["candidate_0"])
        self.assertFalse((config.session_root / "qr_observation_pose_catalog.json").exists())

    def test_qr_fallback_honors_one_candidate_pilot_limit(self):
        config = replace(self.case.config(), stop_after_camera_candidates=1)
        result = execute_candidate_approach_phase(config, self.effects(config, qr_only={"candidate_0"}))
        self.assertIsInstance(result, CandidateCameraCheckpoint)
        self.assertEqual(self.case.visited, ["candidate_0"])
        checkpoint = load_content_hashed_json(result.checkpoint_path, hash_field="camera_candidate_checkpoint_sha256")
        self.assertEqual(len(checkpoint["qr_observation_records"]), 1)
        self.assertEqual(checkpoint["records"], [])
        self.assertFalse(result.to_mission_summary_fields()["exploration_complete"])

    def test_summary_cannot_claim_full_facing_completion_for_fallbacks(self):
        config = self.case.config()
        result = execute_candidate_approach_phase(config, self.effects(config, qr_only={"candidate_0"}))
        original = result.to_mission_summary_fields()
        for changes in ({"facing_complete": True}, {"qr_only_stand_count": 0},
                        {"qr_observation_pose_catalog_sha256": None}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                summary = {**deepcopy(original), **changes}
                validate_completed_qr_goal(summary, snapshot_sha256=candidate_snapshot_sha256(config.snapshot), coverage=None)


if __name__ == "__main__":
    unittest.main()
