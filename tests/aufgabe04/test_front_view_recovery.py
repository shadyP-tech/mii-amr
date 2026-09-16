"""A readable but incomplete front view gets one stopped recovery window."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.aufgabe04.real_robot.observer.front_view_recovery import (
    CORNER_REACQUISITION, FRONT_VIEW_RECOVERY_SEC, GEOMETRY_REACQUISITION,
    HEAD_GEOMETRY_REACQUISITION, FrontViewRecovery, front_view_failure_kind,
)
from tests.aufgabe04 import test_observer_inspection_progress as progress_fixtures


def details(reason="model_qr_text_without_geometry", **changes):
    return {"estimator_reason": reason, "qr_texts": ["QR_003"],
            "stand_axis_debug": {"estimator_usable": False, "estimator_reason": reason,
                                 "metric_model": {"qr_marker_verified": True,
                                                  "head_acquisition": {"candidate_associated": True},
                                                  "camera_target_registration": {"strict_retry_applied": True},
                                                  "model_pose_fit_source": (
                                                      None if reason == "model_qr_text_without_geometry"
                                                      else "joint_qr_head")}}, **changes}


class FrontRecoveryPolicyTests(unittest.TestCase):
    def observe(self, policy, now, **changes):
        values = dict(target_key="stream:candidate", now_sec=float(now), frame_stamp_sec=float(now),
                      robot_pose={"x_m": 0., "y_m": 0., "yaw_rad": 0.}, frame_accepted=True,
                      source_fresh=True, poisoned=False, motion_epoch_reset=False,
                      failure_kind=CORNER_REACQUISITION)
        return policy.observe(**{**values, **changes})

    def test_reasons_distinguish_corner_acquisition_from_incompatible_joint_geometry(self):
        state = "metric_model_measurement_unavailable"
        self.assertEqual(front_view_failure_kind(state, details()), CORNER_REACQUISITION)
        for reason in ("reprojection_error_too_high", "model_head_qr_geometry_mismatch"):
            self.assertEqual(front_view_failure_kind(state, details(reason)), GEOMETRY_REACQUISITION)
        self.assertIsNone(front_view_failure_kind("obsolete_detector_result", details()))
        self.assertIsNone(front_view_failure_kind(state, {"qr_texts": ["a", "b"], "estimator_reason": "model_qr_text_without_geometry"}))
        self.assertIsNone(front_view_failure_kind(state, {"estimator_reason": "model_qr_text_without_geometry"}))
        self.assertIsNone(front_view_failure_kind(state, {"qr_texts": ["QR_003"],
                                                        "stand_axis_debug": {"estimator_usable": True}}))

    def test_readable_text_requires_registered_head_or_own_associated_qr_ray(self):
        state = "metric_model_measurement_unavailable"
        for field, value in (("head_acquisition", {"candidate_associated": False}),
                             ("camera_target_registration", {"strict_retry_applied": False})):
            sample = details()
            sample["stand_axis_debug"]["metric_model"][field] = value
            self.assertIsNone(front_view_failure_kind(state, sample))
            sample["stand_axis_debug"]["decoded_qr_target_binding"] = {
                "accepted": True, "reason": "decoded_qr_target_associated",
            }
            self.assertEqual(front_view_failure_kind(state, sample), CORNER_REACQUISITION)

    def test_independent_rejected_head_has_its_own_bounded_recovery_phase(self):
        sample = details("head_model_yaw_uncertainty_too_high")
        sample["stand_axis_debug"]["estimator_source"] = "model_current_measured_head"
        kind = front_view_failure_kind("metric_model_measurement_unavailable", sample)
        self.assertEqual(kind, HEAD_GEOMETRY_REACQUISITION)
        policy = FrontViewRecovery()
        self.assertTrue(self.observe(policy, 10, failure_kind=kind))
        self.assertTrue(self.observe(policy, 20, failure_kind=CORNER_REACQUISITION))
        self.assertFalse(self.observe(policy, 40, failure_kind=kind))
        self.assertEqual(policy.metadata(now_sec=40)["deadline_monotonic_sec"], 40.)
        sample["stand_axis_debug"]["estimator_usable"] = True
        sample["stand_axis_debug"]["decoded_qr_target_binding"] = {
            "accepted": True, "reason": "decoded_qr_target_associated",
        }
        self.assertIsNone(front_view_failure_kind("metric_model_measurement_unavailable", sample))

    def test_good_head_still_requires_current_unbound_marker_reacquisition(self):
        sample = details("axis_estimated_current_measured_head")
        axis = sample["stand_axis_debug"]
        axis.update(estimator_source="model_current_measured_head", estimator_usable=True,
                    measured_head_lidar_admission={"accepted": True})
        axis["metric_model"]["head_acquisition"] = {}
        axis["metric_model"]["camera_target_registration"] = {}
        for state in ("collecting_consensus", "axis_observation_not_committable"):
            self.assertEqual(front_view_failure_kind(state, sample), CORNER_REACQUISITION)
        sample["qr_texts"] = []
        axis["metric_model"]["qr_marker_verified"] = False
        self.assertIsNone(front_view_failure_kind("axis_observation_not_committable", sample))

    def test_repeated_failures_and_changed_reason_never_renew_deadline(self):
        policy = FrontViewRecovery()
        self.assertTrue(self.observe(policy, 10))
        self.assertTrue(self.observe(policy, 25, failure_kind=GEOMETRY_REACQUISITION))
        self.assertTrue(self.observe(policy, 39.999))
        self.assertFalse(self.observe(policy, 40))
        for now in (41, 60, 89, 100):
            self.assertFalse(self.observe(policy, now))
        metadata = policy.metadata(now_sec=100.)
        self.assertEqual(metadata["started_monotonic_sec"], 10.)
        self.assertEqual(metadata["deadline_monotonic_sec"], 40.)
        self.assertTrue(metadata["budget_exhausted"])
        self.assertFalse(metadata["extends_parent_deadline"])
        self.assertFalse(metadata["motion_authorized"])

    def test_soft_misses_spend_existing_budget_without_starting_or_renewing_it(self):
        for changes in ({"source_fresh": False}, {"frame_accepted": False}, {"failure_kind": None}):
            with self.subTest(changes=changes):
                policy = FrontViewRecovery()
                self.assertFalse(self.observe(policy, 10, **changes))
                self.assertIsNone(policy.metadata(now_sec=10.)["deadline_monotonic_sec"])
                self.assertTrue(self.observe(policy, 11))
                self.assertTrue(self.observe(policy, 23, **changes))
                metadata = policy.metadata(now_sec=23.)
                self.assertEqual(metadata["remaining_sec"], 18.)
                self.assertEqual(metadata["deadline_monotonic_sec"], 41.)
                self.assertEqual(metadata["qualified_frame_count"], 1)
                self.assertEqual(metadata["last_qualified_source_stamp_sec"], 11.)
                self.assertTrue(metadata["current_advisory_deferred"])
                self.assertFalse(self.observe(policy, 41, **changes))
                self.assertFalse(self.observe(policy, 42))

    def test_duplicate_sensor_frame_cannot_start_or_recount_a_recovery(self):
        policy = FrontViewRecovery()
        self.assertFalse(self.observe(policy, 10, failure_kind=None))
        self.assertFalse(self.observe(policy, 11, frame_stamp_sec=10.))
        self.assertIsNone(policy.metadata(now_sec=11.)["deadline_monotonic_sec"])
        self.assertTrue(self.observe(policy, 12))
        self.assertTrue(self.observe(policy, 20, frame_stamp_sec=12.))
        self.assertTrue(self.observe(policy, 21, frame_stamp_sec=11.))
        self.assertEqual(policy.metadata(now_sec=21.)["qualified_frame_count"], 1)
        self.assertEqual(policy.metadata(now_sec=21.)["deadline_monotonic_sec"], 42.)
        self.assertFalse(self.observe(policy, 42, frame_stamp_sec=12.))

    def test_conflict_disables_same_epoch_even_when_later_frame_is_not_poisoned(self):
        policy = FrontViewRecovery()
        self.assertTrue(self.observe(policy, 10))
        self.assertFalse(self.observe(policy, 11, poisoned=True))
        self.assertFalse(self.observe(policy, 12))
        self.assertTrue(policy.metadata(now_sec=12.)["poisoned"])
        self.assertTrue(self.observe(policy, 13, motion_epoch_reset=True))
        self.assertEqual(policy.metadata(now_sec=13.)["deadline_monotonic_sec"], 43.)

    def test_true_target_and_stationary_pose_changes_get_new_bounded_epochs(self):
        policy = FrontViewRecovery()
        self.observe(policy, 10)
        self.assertFalse(self.observe(policy, 41))
        self.assertTrue(self.observe(policy, 42, target_key="other-target"))
        self.assertEqual(policy.metadata(now_sec=42.)["deadline_monotonic_sec"], 72.)
        self.assertTrue(self.observe(policy, 80, target_key="other-target",
                                     robot_pose={"x_m": .1, "y_m": 0., "yaw_rad": 0.}))
        self.assertEqual(policy.metadata(now_sec=80.)["deadline_monotonic_sec"], 110.)

    def test_missing_failure_kind_does_not_abandon_or_reset_prior_recovery_budget(self):
        policy = FrontViewRecovery()
        self.observe(policy, 10)
        self.assertTrue(self.observe(policy, 15, failure_kind=None))
        self.assertEqual(policy.metadata(now_sec=15.)["deadline_monotonic_sec"], 40.)
        self.assertTrue(self.observe(policy, 20))
        self.assertFalse(self.observe(policy, 40))

    def test_motion_or_target_change_drops_hold_until_new_qualified_observation(self):
        for change in ({"motion_epoch_reset": True}, {"target_key": "other-target"},
                       {"robot_pose": {"x_m": .1, "y_m": 0., "yaw_rad": 0.}},
                       {"robot_pose": {"x_m": 0., "y_m": 0., "yaw_rad": .1}}):
            with self.subTest(change=change):
                policy = FrontViewRecovery()
                self.assertTrue(self.observe(policy, 10))
                self.assertFalse(self.observe(policy, 15, failure_kind=None, **change))
                metadata = policy.metadata(now_sec=15.)
                self.assertIsNone(metadata["deadline_monotonic_sec"])
                self.assertEqual(metadata["qualified_frame_count"], 0)

    def test_duration_has_a_hard_thirty_second_cap(self):
        for value in (0, -1, 30.01, 90, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                FrontViewRecovery(duration_sec=value)
        self.assertEqual(FrontViewRecovery().duration_sec, FRONT_VIEW_RECOVERY_SEC)


class FrontRecoveryNodeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "inspection.json"
        self.fixture = progress_fixtures.InspectionObserverIntegrationTests()
        self.node = self.fixture.make_node(self.path)
        for name, value in (("real_robot_profile_sha256", "a" * 64),
                            ("camera_calibration_sha256", "b" * 64)):
            mock = patch(f"scripts.aufgabe04.real_robot.observer.node.{name}", return_value=value)
            mock.start()
            self.addCleanup(mock.stop)

    def feed(self, stamp, *, reason="model_qr_text_without_geometry", state="metric_model_measurement_unavailable",
             now=None, monotonic=None, associated_head=True,
             qr_sample_accepted=False, axis_sample_accepted=False, **frame_changes):
        self.fixture.set_current_frame(self.node, stamp)
        self.node._inspection_frame.update(frame_changes)
        self.node._inspection_frame.update(qr_sample_accepted=qr_sample_accepted,
                                           axis_sample_accepted=axis_sample_accepted)
        self.node._test_clock_sec = stamp if now is None else now
        payload = details(reason)
        if not associated_head:
            payload["stand_axis_debug"]["metric_model"]["head_acquisition"]["candidate_associated"] = False
            payload["stand_axis_debug"]["metric_model"]["camera_target_registration"]["strict_retry_applied"] = False
        with patch("scripts.aufgabe04.real_robot.observer.node.time.monotonic",
                   return_value=float(stamp if monotonic is None else monotonic)):
            return self.node._maybe_commit_inspection_progress(state, payload)

    def test_recorded_first_view_sequence_stays_stopped_then_exits_after_fixed_window(self):
        # All 12 accepted tuples from the original view, relative to the same
        # ROS-second origin. Five had ambiguous neutral heads despite nominal
        # LiDAR acceptance; these must not force an early advisory either.
        sequence = (
            (2.8589327, 3.2602165, True), (3.2588577, 3.6889920, True),
            (4.0587096, 4.4918559, True), (6.0583327, 6.3983119, True),
            (6.9248376, 7.3699656, False), (7.4580717, 7.8982704, True),
            (7.8579957, 8.3273289, False), (8.0912857, 8.5379241, False),
            (9.1910796, 9.7028840, False), (13.6569116, 14.1148667, True),
            (14.4567606, 14.9160671, True), (16.2230988, 16.5333927, False),
        )
        for index, (stamp, monotonic, associated) in enumerate(sequence):
            self.assertIsNone(self.feed(
                stamp, monotonic=monotonic, associated_head=associated,
                reason="reprojection_error_too_high" if index == 3 else "model_qr_text_without_geometry",
                current_qr_sample_count=1 if 3 <= index <= 8 else 0,
                qr_sample_accepted=index == 3,
            ))
        self.assertFalse(self.path.exists())
        self.assertFalse(self.node.completed)
        deadline = self.node._front_view_recovery.metadata(now_sec=16.2230988)["deadline_monotonic_sec"]
        self.assertAlmostEqual(deadline, 3.2602165 + 30.)
        # New repeated decoded text and QR latch expiry do not renew hold.
        for stamp in range(17, 34):
            self.assertIsNone(self.feed(float(stamp), current_qr_sample_count=0))
        result = self.feed(33.3)
        self.assertIsNotNone(result)
        self.assertTrue(self.node.completed)
        self.assertTrue(result["front_view_recovery"]["budget_exhausted"])
        self.assertEqual(result["front_view_recovery"]["phase"], CORNER_REACQUISITION)
        self.assertIsNone(result["qr_id"])
        self.assertFalse(result["completion_authorized"])

    def test_new_bound_qr_sample_does_not_renew_front_recovery_deadline(self):
        self.feed(10.)
        for stamp in range(11, 30):
            self.feed(float(stamp))
        self.assertIsNone(self.feed(30., reason="model_head_qr_geometry_mismatch",
                                    current_qr_sample_count=1, qr_sample_accepted=True))
        self.assertEqual(self.node._front_view_recovery.metadata(now_sec=30.)["deadline_monotonic_sec"], 40.)
        self.assertEqual(self.node._front_view_recovery.metadata(now_sec=30.)["phase"], GEOMETRY_REACQUISITION)

    def test_proposal_miss_cannot_exit_with_eighteen_seconds_of_recovery_remaining(self):
        # The latest third-candidate view had enough failed-view history for
        # an advisory, then one proposal/association miss lost its current
        # front marker while the stopped recovery still had 18 seconds left.
        for stamp in range(10, 22):
            self.assertIsNone(self.feed(float(stamp)))
        for stamp in range(22, 40):
            self.fixture.set_current_frame(self.node, float(stamp))
            with patch("scripts.aufgabe04.real_robot.observer.node.time.monotonic",
                       return_value=float(stamp)):
                self.assertIsNone(self.node._maybe_commit_inspection_progress(
                    "metric_model_measurement_unavailable", {},
                ))
            metadata = self.node._front_view_recovery.metadata(now_sec=float(stamp))
            self.assertEqual(metadata["remaining_sec"], 40. - stamp)
            self.assertEqual(metadata["qualified_frame_count"], 12)
            self.assertFalse(self.node.completed)
            self.assertFalse(self.path.exists())
        self.fixture.set_current_frame(self.node, 40.)
        with patch("scripts.aufgabe04.real_robot.observer.node.time.monotonic", return_value=40.):
            result = self.node._maybe_commit_inspection_progress(
                "metric_model_measurement_unavailable", {},
            )
        self.assertIsNotNone(result)
        self.assertTrue(result["front_view_recovery"]["budget_exhausted"])
        self.assertFalse(result["completion_authorized"])

    def test_good_measured_head_does_not_abandon_current_readable_unbound_qr_early(self):
        payload = details("axis_estimated_current_measured_head")
        payload["stand_axis_debug"].update(
            estimator_source="model_current_measured_head", estimator_usable=True,
            measured_head_lidar_admission={"accepted": True},
            advisory_camera_relative_yaw_rad=.66,
        )
        result = None
        for stamp in range(10, 41):
            self.fixture.set_current_frame(self.node, float(stamp))
            self.node._inspection_frame.update(axis_sample_accepted=True, qr_sample_accepted=False)
            state = "collecting_consensus" if stamp < 17 else "axis_observation_not_committable"
            with patch("scripts.aufgabe04.real_robot.observer.node.time.monotonic", return_value=float(stamp)):
                result = self.node._maybe_commit_inspection_progress(state, payload)
            if stamp < 40:
                self.assertIsNone(result)
        self.assertIsNotNone(result)
        self.assertTrue(result["front_view_recovery"]["budget_exhausted"])
        self.assertEqual(result["camera_relative_yaw_rad"], .66)
        self.assertIsNone(result["qr_id"])
        self.assertFalse(result["completion_authorized"])

    def test_corner_recovery_yields_immediately_to_axis_consensus_and_recommendation(self):
        self.feed(10.)
        for stamp in range(11, 18):
            self.assertIsNone(self.feed(float(stamp), state="collecting_consensus",
                                        axis_sample_accepted=True, current_qr_id="QR_003", current_qr_sample_count=2))
            self.assertFalse(self.node.completed)
        self.node.completed = True  # Existing stronger path commits before this hook.
        self.assertIsNone(self.feed(18., state="recommendation_committed"))
        self.assertFalse(self.path.exists())

    def test_unassociated_and_stale_frames_do_not_start_recovery(self):
        for stamp in range(10, 18):
            self.assertIsNone(self.feed(float(stamp), frame_accepted=False))
        self.assertIsNone(self.node._front_view_recovery.metadata(now_sec=18.)["deadline_monotonic_sec"])
        self.assertIsNone(self.feed(18., now=18.501))
        self.assertIsNone(self.node._front_view_recovery.metadata(now_sec=18.)["deadline_monotonic_sec"])
        self.assertFalse(self.path.exists())

    def test_ambiguous_head_with_nominal_association_cannot_start_recovery(self):
        self.fixture.set_current_frame(self.node, 10.)
        payload = details()
        payload["stand_axis_debug"]["metric_model"]["head_acquisition"] = {
            "candidate_associated": False, "reason": "head_proposal_candidate_association_rejected",
        }
        payload["stand_axis_debug"]["preliminary_candidate_lidar_association"] = {"associated": True}
        self.assertIsNone(self.node._maybe_commit_inspection_progress("metric_model_measurement_unavailable", payload))
        self.assertIsNone(getattr(self.node, "_front_view_recovery", None))

    def test_poison_clears_hold_and_true_motion_resets_epoch(self):
        self.feed(10.)
        self.assertIsNone(self.feed(11., poisoned=True))
        self.assertTrue(self.node._front_view_recovery.metadata(now_sec=11.)["poisoned"])
        self.assertIsNone(self.feed(12.))
        self.assertTrue(self.node._front_view_recovery.metadata(now_sec=12.)["poisoned"])
        self.assertIsNone(self.feed(13., motion_epoch_reset=True,
                                    robot_pose={"x_m": .1, "y_m": 0., "yaw_rad": 0.}))
        self.assertFalse(self.node._front_view_recovery.metadata(now_sec=13.)["poisoned"])
        self.assertEqual(self.node._front_view_recovery.metadata(now_sec=13.)["deadline_monotonic_sec"], 43.)
        self.assertFalse(self.path.exists())

    def test_expired_window_cannot_publish_an_obsolete_current_frame(self):
        for stamp in range(10, 40):
            self.assertIsNone(self.feed(float(stamp)))
        self.assertIsNone(self.feed(40., now=40.501))
        self.assertFalse(self.path.exists())
        self.assertFalse(self.node.completed)


if __name__ == "__main__":
    unittest.main()
