"""Front text/metric disagreement must not erase gated identity evidence."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.real_robot.observer.front_observation import (
    BACKSIDE_AXIS_SOURCES, front_observation_decision,
)
from scripts.aufgabe04.real_robot.observer.inspection_progress import (
    classify_inspection_progress,
)
from scripts.aufgabe04.real_robot.observer.node import PassiveRealViewpointNode


BACKSIDE = "model_backside_current_frame"
FRONT = "model_current_frame_refined"


class FrontObservationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.node = PassiveRealViewpointNode.__new__(PassiveRealViewpointNode)
        self.now_sec = 0.0
        clock = SimpleNamespace(now=lambda: SimpleNamespace(
            nanoseconds=round(self.now_sec * 1_000_000_000),
        ))
        self.node.node = SimpleNamespace(get_clock=lambda: clock)
        self.node.args = SimpleNamespace(
            inspection_observation_json=Path(self.temporary.name) / "inspection.json",
            stand_id="survey_candidate_0001", stream_id="run_candidate_0001",
            stand_x=-1.08, stand_y=-0.426, consensus_frames=7,
            consensus_max_deviation_deg=5, consensus_axis_ttl_sec=5,
            consensus_qr_ttl_sec=5, max_sensor_age_sec=.5,
            sync_tolerance_sec=.1, max_future_timestamp_sec=.05,
            stationary_translation_m=.02, stationary_rotation_deg=2,
        )
        self.node.profile = SimpleNamespace(map_frame="map")
        self.node.calibration = object()
        self.node.stand_model_profile = SimpleNamespace(sha256="c" * 64)
        self.node.observation_evidence = None
        self.node.completed = False
        self.node._reset_qr_marker_epoch()
        for name, value in (
            ("real_robot_profile_sha256", "a" * 64),
            ("camera_calibration_sha256", "b" * 64),
        ):
            patcher = patch(
                f"scripts.aufgabe04.real_robot.observer.node.{name}",
                return_value=value,
            )
            patcher.start()
            self.addCleanup(patcher.stop)

    def observe(self, stamp, *, texts=("Start",), associated=True,
                marker=False, source=BACKSIDE, pose=Pose2D(0, 0, 0),
                observed_at=None, scan_stamp=None):
        self.now_sec = stamp if observed_at is None else observed_at
        decision = front_observation_decision(
            qr_texts=texts, qr_marker_detected=marker, estimate_source=source,
            marker_seen_in_stationary_epoch=self.node._qr_marker_seen_in_stationary_epoch,
        )
        self.node._note_front_observation(decision, pose)
        fields = dict(
            robot_pose=pose, image_stamp_sec=stamp,
            scan_stamp_sec=stamp if scan_stamp is None else scan_stamp,
            observed_at_sec=stamp if observed_at is None else observed_at,
            lidar_associated=associated, qr_texts=texts,
        )
        if decision.withhold_backside_axis:
            update, metadata = self.node._record_front_seen_axis_unresolved(
                decision=decision, **fields,
            )
        else:
            update = self.node._record_observation_frame(
                axis_yaw_rad=None, axis_source=None, **fields,
            )
            metadata = decision.metadata(
                lidar_associated=associated, frame_accepted=update.frame_accepted,
            )
        payload = self.node._maybe_commit_inspection_progress(
            "evidence_not_committable", {
                "reason": "front_seen_axis_unresolved",
                "qr_texts": list(texts), "front_observation": metadata,
                "stand_axis_debug": {"advisory_camera_relative_yaw_rad": .4},
            },
        )
        return update, metadata, payload

    def test_recorded_seven_start_reads_with_no_association_remain_unbound(self):
        # Latest run: six decoded rows contradict a QR-free fallback source;
        # the seventh has a failed ordinary QR-model refinement. Every row
        # has preliminary LiDAR association false.
        sources = (BACKSIDE,) * 6 + (FRONT,)
        for index, source in enumerate(sources):
            update, metadata, payload = self.observe(
                10 + index, associated=False, source=source,
            )
            self.assertFalse(update.frame_accepted)
            self.assertEqual(update.reason, "lidar_target_not_associated")
            self.assertIsNone(update.resolved_qr_id)
            self.assertEqual(update.snapshot.current_qr_sample_count, 0)
            self.assertEqual(metadata["classification"], "front_readable")
            self.assertFalse(metadata["candidate_frame_accepted"])
            self.assertIsNone(payload)
        self.assertFalse(self.node.completed)
        self.assertFalse(self.node.args.inspection_observation_json.exists())

    def test_associated_front_text_latches_without_backside_axis_authority(self):
        updates = []
        for index in range(8):
            update, metadata, payload = self.observe(10 + index / 3)
            updates.append(update)
            self.assertFalse(update.axis_sample_accepted)
            self.assertIsNone(update.axis_consensus)
        self.assertIsNone(updates[0].resolved_qr_id)
        self.assertEqual(updates[1].resolved_qr_id, "Start")
        self.assertEqual(payload["qr_id"], "Start")
        self.assertEqual(payload["classification"], "front_readable")
        self.assertIsNone(payload["camera_relative_yaw_rad"])
        self.assertFalse(payload["completion_authorized"])
        self.assertFalse(payload["motion_authorized"])
        self.assertEqual(list(Path(self.temporary.name).iterdir()), [
            self.node.args.inspection_observation_json,
        ])

    def test_cpu_cadence_front_advisory_does_not_extend_qr_or_axis_ttl(self):
        for index in range(8):
            update, _, payload = self.observe(10 + 1.4 * index)
            if index < 7:
                self.assertIsNone(payload)
        # First tentative QR sample receives acquisition grace. The next
        # seven fresh tuples span 8.4 seconds in the measured CPU path.
        self.assertEqual(payload["sample_count"], 7)
        self.assertAlmostEqual(payload["sensor_stamp_sec"] - payload["first_sensor_stamp_sec"], 8.4)
        self.assertEqual(payload["qr_id"], "Start")
        self.assertEqual(payload["qr_sample_count"], 2)
        self.assertFalse(payload["completion_authorized"])
        self.assertFalse(payload["motion_authorized"])
        self.assertEqual(update.snapshot.qr_ttl_sec, 5)
        self.assertEqual(update.snapshot.axis_ttl_sec, 5)
        self.assertIsNone(update.axis_consensus)

    def test_long_front_advisory_history_cannot_revive_expired_qr(self):
        for index in range(8):
            update, _, payload = self.observe(
                10 + 1.4 * index, texts=("Start",) if index < 2 else (), marker=True,
            )
        self.assertIsNotNone(payload)
        self.assertIsNone(payload["qr_id"])
        self.assertEqual(payload["qr_sample_count"], 0)
        self.assertIsNone(update.resolved_qr_id)

    def test_long_advisory_window_does_not_admit_stale_or_skewed_tuples(self):
        for index in range(8):
            stamp = 10 + 1.4 * index
            options = {"observed_at": stamp + 1} if index % 2 else {"scan_stamp": stamp - 1}
            update, _, payload = self.observe(stamp, **options)
            self.assertFalse(update.frame_accepted)
            self.assertIsNone(payload)
        self.assertFalse(self.node.args.inspection_observation_json.exists())

    def test_motion_cannot_join_frames_across_the_long_advisory_window(self):
        for index in range(10):
            _update, _, payload = self.observe(
                10 + 1.4 * index,
                pose=Pose2D(0 if index < 6 else .1, 0, 0),
            )
            self.assertIsNone(payload)
        self.assertFalse(self.node.args.inspection_observation_json.exists())

    def test_conflict_after_qr_expiry_poisons_the_long_advisory_window(self):
        for index in range(10):
            texts = ("Start",) if index < 2 else (("QR_002",) if index >= 6 else ())
            update, _, payload = self.observe(10 + 1.4 * index, texts=texts, marker=True)
            self.assertIsNone(payload)
        self.assertTrue(update.snapshot.poisoned)
        self.assertFalse(self.node.args.inspection_observation_json.exists())

    def test_slow_axis_frames_still_expire_before_seven_frame_consensus(self):
        pose = Pose2D(0, 0, 0)
        for index in range(8):
            stamp = 10 + 1.4 * index
            self.now_sec = stamp
            update = self.node._record_observation_frame(
                robot_pose=pose, image_stamp_sec=stamp, scan_stamp_sec=stamp,
                observed_at_sec=stamp, lidar_associated=True,
                axis_yaw_rad=.1, axis_source=FRONT, qr_texts=("Start",),
            )
            self.assertIsNone(update.axis_consensus)
            self.assertLessEqual(update.snapshot.current_axis_sample_count, 4)

    def test_marker_misses_keep_qr_latch_and_remove_only_backside_buckets(self):
        window = self.node._ensure_observation_evidence(Pose2D(0, 0, 0))
        for index, source in enumerate((*BACKSIDE_AXIS_SOURCES, FRONT)):
            window.record_frame(
                target_key=self.node._target_evidence_key(),
                pose=self.node._evidence_pose(Pose2D(0, 0, 0)),
                frame_stamp_sec=10 + index / 10, lidar_stamp_sec=10 + index / 10,
                observed_at_sec=10 + index / 10, lidar_associated=True,
                axis_yaw_rad=.1, axis_source=source, qr_texts=("Start",),
            )
        update, _metadata, _payload = self.observe(11, texts=(), marker=True)
        self.assertEqual(update.resolved_qr_id, "Start")
        self.assertEqual(update.snapshot.current_axis_sample_count_by_source, {FRONT: 1})
        update, metadata, _payload = self.observe(11.3, texts=(), marker=False)
        self.assertEqual(update.resolved_qr_id, "Start")
        self.assertEqual(metadata["classification"], "front_unreadable")
        self.assertFalse(update.axis_sample_accepted)

    def test_seven_valid_front_axes_survive_intermittent_decoding(self):
        pose = Pose2D(0, 0, 0)
        for index in range(7):
            texts = ("Start",) if index < 2 else ()
            self.now_sec = 10 + index / 3
            decision = front_observation_decision(
                qr_texts=texts, qr_marker_detected=True, estimate_source=FRONT,
                marker_seen_in_stationary_epoch=self.node._qr_marker_seen_in_stationary_epoch,
            )
            self.node._note_front_observation(decision, pose)
            self.assertFalse(decision.withhold_backside_axis)
            update = self.node._record_observation_frame(
                robot_pose=pose, image_stamp_sec=10 + index / 3,
                scan_stamp_sec=10 + index / 3, observed_at_sec=10 + index / 3,
                lidar_associated=True, axis_yaw_rad=.1, axis_source=FRONT,
                qr_texts=texts,
            )
            if index < 6:
                self.assertIsNone(update.axis_consensus)
        self.assertEqual(update.axis_consensus.sample_count, 7)
        self.assertEqual(update.axis_consensus.source, FRONT)
        self.assertEqual(update.resolved_qr_id, "Start")

    def test_conflicting_or_multiple_associated_texts_remain_poisoned(self):
        self.observe(10)
        update, _, _ = self.observe(10.3, texts=("QR_002",))
        self.assertTrue(update.snapshot.poisoned)
        for index in range(8):
            update, _, payload = self.observe(11 + index / 3, texts=(), marker=True)
            self.assertTrue(update.snapshot.poisoned)
            self.assertIsNone(payload)
        self.assertFalse(self.node.completed)

    def test_multiple_unassociated_texts_neither_poison_nor_bind_identity(self):
        update, _, _ = self.observe(10, texts=("Start", "QR_002"), associated=False)
        self.assertFalse(update.snapshot.poisoned)
        self.assertIsNone(update.resolved_qr_id)
        update, _, _ = self.observe(10.3, texts=("Start", "QR_002"))
        self.assertTrue(update.snapshot.poisoned)
        self.assertEqual(update.snapshot.poison_reason, "multiple_qr_ids_in_associated_frame")

    def test_ttl_expiry_does_not_erase_stationary_identity_conflict_veto(self):
        self.observe(10)
        self.observe(10.3)
        update, _, _ = self.observe(16, texts=(), marker=True)
        self.assertIsNone(update.resolved_qr_id)
        update, _, payload = self.observe(16.3, texts=("QR_002",))
        self.assertTrue(update.snapshot.poisoned)
        self.assertIsNone(payload)

    def test_new_motion_epoch_can_recover_without_carrying_old_poison(self):
        self.observe(10)
        self.observe(10.3, texts=("QR_002",))
        # The operational node performs this reset after validating a real
        # pose change against the stationary marker's original anchor.
        self.node._reset_qr_marker_epoch()
        for index in range(8):
            update, _, payload = self.observe(
                11 + index / 3, texts=("QR_002",), pose=Pose2D(.1, 0, 0),
            )
            self.assertFalse(update.snapshot.poisoned)
        self.assertEqual(payload["qr_id"], "QR_002")
        self.assertFalse(payload["completion_authorized"])

    def test_stale_and_skewed_lidar_cannot_bind_front_text(self):
        for stamp, options in ((10, {"observed_at": 11}), (12, {"scan_stamp": 11})):
            update, _, payload = self.observe(stamp, **options)
            self.assertFalse(update.frame_accepted)
            self.assertIsNone(update.resolved_qr_id)
            self.assertIsNone(payload)

    def test_front_classification_never_uses_contradicted_backside_angle(self):
        value = classify_inspection_progress("evidence_not_committable", {
            "front_observation": {"axis_state": "unresolved", "classification": "front_readable"},
            "conditioning": {"reason": "oblique_silhouette"},
            "stand_axis_debug": {"advisory_camera_relative_yaw_rad": 1.5},
        })
        self.assertEqual(value.classification, "front_readable")
        self.assertIsNone(value.camera_relative_yaw_rad)


if __name__ == "__main__":
    unittest.main()
