"""Persistent current range misses defer a target without accepting measurements."""

from dataclasses import asdict
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.candidate_lidar_association import associate_candidate_lidar_target
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.real_robot.observer.node import PassiveRealViewpointNode
from scripts.aufgabe04.real_robot.observer import node as observer_node
from scripts.aufgabe04.real_robot.observer.process import PassiveObserverProcessEvidence
from scripts.aufgabe04.real_robot.observer.target_support_failure import validate_target_support_failure
from scripts.aufgabe04.real_robot.observer.target_support_handoff import load_target_support_failure_exit
from scripts.aufgabe04.real_robot.observer.target_support_runtime import (
    HASH_FIELD, target_support_binding,
)
from scripts.aufgabe04.stations.candidate_snapshot import load_candidate_snapshot
from tests.aufgabe04 import test_camera_observer_processing as processing_fixtures
from tests.aufgabe04 import test_target_reconciliation as reconciliation_fixtures


ROOT = reconciliation_fixtures.ROOT


class ObserverTargetSupportFailureTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.fixture = processing_fixtures.CameraObserverProcessingTest()
        self.adapter = self.fixture.make_adapter()
        args = self.adapter.args
        args.candidate_crop_snapshot = ROOT / "candidate_snapshot.json"
        args.stand_id = "survey_candidate_0005"
        geometry = load_candidate_snapshot(args.candidate_crop_snapshot).candidate_for(args.stand_id).geometry
        args.stand_x, args.stand_y = geometry.x_m, geometry.y_m
        args.status_json = self.output / "observer_status.json"
        args.recommended_pose_json = self.output / "recommendation.json"
        args.inspection_observation_json = self.output / "inspection.json"
        self.adapter._write_status = lambda state, **details: PassiveRealViewpointNode._write_status(
            self.adapter, state, **details)
        for function in ("real_robot_profile_sha256", "camera_calibration_sha256"):
            mocked = patch("scripts.aufgabe04.real_robot.observer.target_support_runtime." + function,
                           return_value="a" * 64)
            mocked.start()
            self.addCleanup(mocked.stop)

    def frame(self, stamp, *, age=.1, scan_skew=0., distance=1.05, pose=None,
              raw_association=True, associated=False, publish=True, proof=None, associated_head=False):
        self.fixture.clock_sec = stamp + age
        adapter = self.adapter
        scan = PlainLaserScan(ranges=(distance,) * 7,
            angle_min=-.15, angle_increment=.05, range_min=.01, range_max=10.,
            scan_stamp_sec=stamp + scan_skew, receipt_sec=stamp + scan_skew,
            scan_frame_id=adapter.profile.scan_frame)
        association = associate_candidate_lidar_target(scan, map_bearing_rad=0.,
            cone_half_angle_rad=math.radians(15), accepted_range_m=(.368, .588),
            now_sec=stamp + min(age, .1), max_scan_age_sec=.5)
        adapter._current_target_support = (dict(frame_stamp_sec=stamp,
            scan_stamp_sec=stamp + scan_skew, association=asdict(association),
            associated_head=associated_head) if raw_association else None)
        adapter._current_position_epoch_proof = proof
        update = adapter._record_observation_frame(
            robot_pose=pose or Pose2D(0., 0., 0.), image_stamp_sec=stamp,
            scan_stamp_sec=stamp + scan_skew, observed_at_sec=stamp + age,
            lidar_associated=associated, axis_yaw_rad=None, axis_source=None, qr_texts=())
        if publish:
            adapter._write_status("metric_model_measurement_unavailable", reason="head_unavailable")
        return update

    def collect(self, *, publish=True, **options):
        for index in range(7):
            self.frame(100. + index * .85, publish=publish, **options)

    def receipt(self):
        return load_content_hashed_json(self.output / "target_support_failure.json", hash_field=HASH_FIELD)

    def test_fresh_distinct_negative_window_exits_without_measurement_artifact(self):
        self.collect()
        status = json.loads(self.adapter.args.status_json.read_text())
        self.assertTrue(self.adapter.completed)
        self.assertEqual(status["state"], "target_reconciliation_required")
        payload = validate_target_support_failure(self.receipt(), target_binding=target_support_binding(self.adapter))
        self.assertEqual(payload["sample_count"], 7)
        self.assertGreaterEqual(payload["elapsed_sec"], 5.)
        self.assertFalse(payload["completion_authorized"])
        self.assertFalse(payload["motion_authorized"])
        self.assertEqual(status["observation_evidence"]["accepted_frame_count"], 0)
        self.assertEqual(status["observation_evidence"]["lidar_rejection_count"], 7)
        self.assertFalse(self.adapter.args.recommended_pose_json.exists())
        self.assertFalse(self.adapter.args.inspection_observation_json.exists())
        self.assertEqual(status["target_support_failure"]["path"], str(self.output / "target_support_failure.json"))
        candidate = load_candidate_snapshot(self.adapter.args.candidate_crop_snapshot).candidate_for(
            self.adapter.args.stand_id)
        parent_evidence = load_target_support_failure_exit(
            status_path=self.adapter.args.status_json,
            process=PassiveObserverProcessEvidence("child_exit", None, None, False, 0,
                                                  ("exit_observed",), ()),
            candidate=candidate, snapshot_path=self.adapter.args.candidate_crop_snapshot,
            planning_frame="map", stream_id=self.adapter.args.stream_id,
            robot_profile_sha256="a" * 64, calibration_profile_sha256="a" * 64,
            stand_model_profile_sha256=self.adapter.stand_model_profile.sha256)
        self.assertEqual(parent_evidence["target_support_failure"]["state"],
                         "target_reconciliation_required")

    def test_fewer_than_seven_or_short_opportunity_never_exits(self):
        for index in range(6):
            self.frame(100. + index)
        self.assertFalse(self.adapter.completed)
        self.frame(105.1)
        self.assertTrue(self.adapter.completed)

    def test_stale_sources_do_not_become_negative_evidence(self):
        self.collect(age=.6)
        self.assertFalse(self.adapter.completed)
        self.assertEqual(self.adapter._target_support_failure_window.samples, [])
        # Common evidence calls this a LiDAR miss; the separate source-age gate
        # must still distinguish stale processing from actual target absence.
        self.assertEqual(self.adapter.observation_evidence.snapshot().last_soft_miss_reason,
                         "lidar_target_not_associated")

    def test_missing_exact_tf_context_and_unsynchronized_tuples_cannot_defer(self):
        self.collect(raw_association=False)
        self.assertFalse(self.adapter.completed)
        self.collect(scan_skew=-.2)
        self.assertFalse(self.adapter.completed)

    def test_invalid_scan_rays_do_not_prove_target_absence(self):
        self.collect(distance=math.inf)
        self.assertFalse(self.adapter.completed)
        self.assertEqual(self.adapter._target_support_failure_window.samples, [])

    def test_positive_raw_or_independent_head_support_resets_negative_window(self):
        for positive in ({"distance": .5}, {"associated_head": True}, {"associated": True}):
            with self.subTest(positive=positive):
                for index in range(6):
                    self.frame(100. + index * .85)
                self.frame(105.1, **positive)
                self.assertFalse(self.adapter.completed)
                self.assertEqual(self.adapter._target_support_failure_window.samples, [])
                self.frame(105.95)
                self.assertFalse(self.adapter.completed)

    def test_valid_current_reconciliation_resets_window_but_ready_hint_does_not(self):
        fixture = reconciliation_fixtures.TargetReconciliationTest()
        fixture.setUp()
        proof = fixture.proof()
        stamp = proof["entries"][-1]["image_stamp_sec"]
        scan_stamp = proof["entries"][-1]["scan"]["scan_stamp_sec"]
        proof["target_key"] = self.adapter._target_evidence_key()
        for index in range(6):
            self.frame(stamp - (6-index) * .85)
        self.frame(stamp, scan_skew=scan_stamp-stamp, proof=proof)
        self.assertFalse(self.adapter.completed)
        self.assertEqual(self.adapter._target_support_failure_window.samples, [])
        for index in range(1, 8):
            self.frame(stamp + index * .85, proof={"ready": True})
        self.assertTrue(self.adapter.completed)

    def test_changed_pose_resets_window(self):
        for index in range(6):
            self.frame(100. + index * .85)
        self.frame(105.1, pose=Pose2D(.04, 0., 0.))
        self.assertFalse(self.adapter.completed)

    def test_contract_reset_cannot_reuse_negative_samples_in_new_epoch_zero(self):
        for index in range(6):
            self.frame(100. + index * .85)
        self.adapter._reset_observation_evidence()
        self.frame(105.1)
        self.assertFalse(self.adapter.completed)
        self.assertEqual(len(self.adapter._target_support_failure_window.samples), 1)

    def test_snapshot_missing_or_mismatched_never_creates_failure_receipt(self):
        for path in (None, self.output / "missing.json"):
            self.adapter.args.candidate_crop_snapshot = path
            self.collect()
            self.assertFalse(self.adapter.completed)
        self.adapter.args.candidate_crop_snapshot = ROOT / "candidate_snapshot.json"
        self.adapter.args.stand_x += .1
        self.adapter._reset_observation_evidence()
        self.collect()
        self.assertFalse(self.adapter.completed)

    def test_expired_publication_consumes_pending_failure_without_exit(self):
        self.collect(publish=False)
        self.fixture.clock_sec += 1.
        self.adapter._write_status("metric_model_measurement_unavailable")
        self.assertFalse(self.adapter.completed)
        self.assertFalse((self.output / "target_support_failure.json").exists())
        self.assertIsNone(self.adapter._target_support_failure)

    def test_receipt_expiring_during_serialization_does_not_exit(self):
        self.collect(publish=False)
        atomic_json = observer_node._atomic_json

        def slow_receipt(path, payload, **options):
            if path.name == "target_support_failure.json":
                self.fixture.clock_sec += .6
            return atomic_json(path, payload, **options)

        with patch.object(observer_node, "_atomic_json", side_effect=slow_receipt):
            self.adapter._write_status("metric_model_measurement_unavailable")
        self.assertFalse(self.adapter.completed)
        self.assertFalse((self.output / "target_support_failure.json").exists())
        self.assertEqual(self.adapter._camera_pipeline_counters["publication_rejections"], 1)

    def test_tf_or_other_status_cannot_publish_previous_processed_failure(self):
        self.collect(publish=False)
        self.adapter._write_status("tf_pending_exact_time")
        self.assertFalse(self.adapter.completed)
        self.adapter._write_status("metric_model_measurement_unavailable")
        self.assertFalse(self.adapter.completed)
        self.assertFalse((self.output / "target_support_failure.json").exists())

    def test_successful_stronger_commit_has_priority(self):
        self.collect(publish=False)

        def commit(_adapter):
            _adapter.completed = True
            return "qr_observation_pose_committed", {"qr_id": "QR_001"}

        with patch("scripts.aufgabe04.real_robot.observer.node.commit_qr_observation_pose", side_effect=commit):
            self.adapter._write_status("metric_model_measurement_unavailable")
        self.assertEqual(json.loads(self.adapter.args.status_json.read_text())["state"],
                         "qr_observation_pose_committed")
        self.assertFalse((self.output / "target_support_failure.json").exists())


if __name__ == "__main__":
    unittest.main()
