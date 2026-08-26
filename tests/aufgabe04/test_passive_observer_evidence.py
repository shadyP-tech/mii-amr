import math
import unittest

from scripts.aufgabe04.real_robot.observer.evidence import (
    EvidencePose,
    PassiveObserverEvidence,
)


ANCHOR = EvidencePose(1.0, -0.5, 0.1)


def evidence(**overrides):
    kwargs = {
        "target_key": "candidate_05",
        "anchor_pose": ANCHOR,
        "required_axis_samples": 3,
        "max_axis_deviation_rad": math.radians(5.0),
    }
    kwargs.update(overrides)
    return PassiveObserverEvidence(**kwargs)


def frame(window, stamp, *, source="edges", yaw=0.01, qr=(), pose=ANCHOR, **overrides):
    kwargs = {
        "target_key": "candidate_05",
        "pose": pose,
        "frame_stamp_sec": stamp,
        "lidar_stamp_sec": stamp - 0.02,
        "observed_at_sec": stamp + 0.02,
        "lidar_associated": True,
        "axis_yaw_rad": yaw,
        "axis_source": source,
        "qr_texts": qr,
    }
    kwargs.update(overrides)
    return window.record_frame(**kwargs)


class PassiveObserverEvidenceTest(unittest.TestCase):
    def test_real_run_valid_burst_reaches_seven_inside_bounded_window(self):
        window = evidence(
            required_axis_samples=7,
            axis_ttl_sec=5.0,
        )
        stamps = (
            10.000,
            10.407,
            10.979,
            12.196,
            12.984,
            13.409,
            13.772,
        )
        result = None
        for index, stamp in enumerate(stamps):
            if index:
                window.note_soft_miss(
                    target_key="candidate_05",
                    pose=ANCHOR,
                    stamp_sec=stamp - 0.05,
                    reason="lidar_target_mismatch",
                )
            result = frame(
                window,
                stamp,
                yaw=0.01 + index * 0.001,
                qr=("QR_1",),
            )

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.axis_consensus)
        self.assertEqual(result.axis_consensus.sample_count, 7)
        self.assertEqual(result.resolved_qr_id, "QR_1")
        self.assertEqual(result.snapshot.soft_miss_count, 6)

    def test_soft_misses_and_source_changes_do_not_erase_fresh_axis_bucket(self):
        window = evidence()
        self.assertIsNone(frame(window, 10.0).axis_consensus)
        window.note_soft_miss(
            target_key="candidate_05",
            pose=ANCHOR,
            stamp_sec=10.5,
            reason="lidar_target_mismatch",
        )
        self.assertIsNone(frame(window, 11.0, source="stem", yaw=0.3).axis_consensus)
        self.assertIsNone(frame(window, 12.0, yaw=0.02).axis_consensus)
        result = frame(window, 13.0, yaw=0.00)
        self.assertIsNotNone(result.axis_consensus)
        self.assertEqual(result.axis_consensus.source, "edges")
        self.assertTrue(result.axis_only)
        self.assertEqual(
            result.snapshot.current_axis_sample_count_by_source,
            {"edges": 3, "stem": 1},
        )
        self.assertEqual(result.snapshot.peak_axis_sample_count_by_source["stem"], 1)

    def test_only_fresh_same_tuple_lidar_can_admit_evidence(self):
        window = evidence()
        mismatch = frame(window, 10.0, lidar_associated=False, qr=("QR_1",))
        stale = frame(
            window,
            11.0,
            lidar_stamp_sec=10.98,
            observed_at_sec=11.75,
            qr=("QR_1",),
        )
        skewed = frame(window, 12.0, lidar_stamp_sec=11.5, qr=("QR_1",))
        admitted = frame(window, 13.0, qr=("QR_1",))
        self.assertEqual(mismatch.reason, "lidar_target_not_associated")
        self.assertEqual(stale.reason, "lidar_tuple_stale")
        self.assertEqual(skewed.reason, "lidar_not_from_same_sensor_tuple")
        self.assertTrue(admitted.frame_accepted)
        self.assertEqual(admitted.snapshot.current_axis_sample_count, 1)
        self.assertEqual(admitted.snapshot.current_qr_sample_count, 1)
        self.assertEqual(admitted.snapshot.lidar_rejection_count, 3)
        self.assertEqual(admitted.snapshot.soft_miss_count, 3)

    def test_qr_requires_two_distinct_frames_and_empty_frame_does_not_reset(self):
        window = evidence(required_axis_samples=2)
        tentative = frame(window, 10.0, qr=("QR_1",))
        duplicate = frame(window, 10.0, qr=("QR_1",))
        empty = frame(window, 11.0, qr=())
        latched = frame(window, 12.0, qr=("QR_1",))
        self.assertEqual(tentative.snapshot.tentative_qr_id, "QR_1")
        self.assertIsNone(tentative.resolved_qr_id)
        self.assertEqual(duplicate.reason, "duplicate_frame_stamp")
        self.assertEqual(empty.snapshot.current_qr_sample_count, 1)
        self.assertEqual(latched.resolved_qr_id, "QR_1")
        self.assertEqual(latched.snapshot.current_qr_sample_count, 2)
        self.assertEqual(latched.snapshot.duplicate_frame_count, 1)

    def test_qr_channel_can_latch_without_axis_channel(self):
        window = evidence()
        first = frame(
            window,
            10.0,
            qr=("QR_1",),
            yaw=None,
            source=None,
        )
        second = frame(
            window,
            11.0,
            qr=("QR_1",),
            yaw=None,
            source=None,
        )
        self.assertIsNone(first.resolved_qr_id)
        self.assertEqual(second.resolved_qr_id, "QR_1")
        self.assertEqual(second.snapshot.current_axis_sample_count, 0)
        self.assertIsNone(second.axis_consensus)

    def test_tentative_qr_expires_and_axis_can_complete_without_identity(self):
        window = evidence(required_axis_samples=2)
        frame(window, 10.0, qr=("QR_1",))
        result = frame(window, 16.0, qr=(), yaw=0.02)
        self.assertIsNone(result.resolved_qr_id)
        self.assertIsNone(result.snapshot.tentative_qr_id)
        self.assertEqual(result.snapshot.current_qr_sample_count, 0)
        # The 10-second axis also expired, so one new sample cannot complete.
        self.assertIsNone(result.axis_consensus)
        completed = frame(window, 17.0, qr=(), yaw=0.01)
        self.assertIsNotNone(completed.axis_consensus)
        self.assertTrue(completed.axis_only)

    def test_conflicting_or_multiple_qr_identity_poisons_motion_epoch(self):
        conflict = evidence()
        frame(conflict, 10.0, qr=("QR_1",))
        poisoned = frame(conflict, 11.0, qr=("QR_2",))
        self.assertTrue(poisoned.snapshot.poisoned)
        self.assertEqual(poisoned.reason, "conflicting_qr_ids_in_motion_epoch")
        self.assertEqual(poisoned.snapshot.current_axis_sample_count, 0)
        blocked = frame(conflict, 12.0, qr=("QR_2",))
        self.assertFalse(blocked.frame_accepted)

        multiple = evidence()
        poisoned = frame(multiple, 10.0, qr=("QR_2", "QR_1"))
        self.assertTrue(poisoned.snapshot.poisoned)
        self.assertEqual(poisoned.reason, "multiple_qr_ids_in_associated_frame")

    def test_anchor_relative_motion_starts_new_epoch_and_clears_poison(self):
        window = evidence()
        frame(window, 10.0, qr=("QR_1",))
        frame(window, 11.0, qr=("QR_2",))
        moved_pose = EvidencePose(1.03, -0.5, 0.1)
        restarted = frame(window, 12.0, pose=moved_pose, qr=())
        self.assertTrue(restarted.motion_epoch_reset)
        self.assertEqual(restarted.snapshot.motion_epoch, 1)
        self.assertEqual(restarted.snapshot.motion_reset_count, 1)
        self.assertFalse(restarted.snapshot.poisoned)
        self.assertEqual(restarted.snapshot.current_axis_sample_count, 1)
        self.assertEqual(restarted.snapshot.peak_axis_sample_count, 1)
        self.assertEqual(restarted.snapshot.anchor_pose, moved_pose)

    def test_ttl_prunes_current_samples_but_preserves_peak_audit(self):
        window = evidence(required_axis_samples=2)
        frame(window, 10.0)
        completed = frame(window, 11.0)
        self.assertEqual(completed.snapshot.peak_axis_sample_count, 2)
        expired = window.note_soft_miss(
            target_key="candidate_05",
            pose=ANCHOR,
            stamp_sec=16.1,
            reason="silhouette_unavailable",
        )
        self.assertEqual(expired.snapshot.current_axis_sample_count, 0)
        self.assertEqual(expired.snapshot.peak_axis_sample_count, 2)
        self.assertEqual(expired.snapshot.soft_miss_count, 1)
        self.assertEqual(expired.snapshot.last_soft_miss_reason, "silhouette_unavailable")

    def test_target_is_fixed_for_lifetime_of_window(self):
        window = evidence()
        with self.assertRaisesRegex(ValueError, "fixed to 'candidate_05'"):
            window.note_soft_miss(
                target_key="candidate_06",
                pose=ANCHOR,
                stamp_sec=10.0,
                reason="wrong target",
            )


if __name__ == "__main__":
    unittest.main()
