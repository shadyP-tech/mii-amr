from dataclasses import replace
from types import SimpleNamespace
import unittest

from scripts.aufgabe04.perception.debug.stand_axis_viewer import (
    _detector_result_is_obsolete,
    _unavailable_target_estimate,
)
from scripts.aufgabe04.perception.debug.viewer_frame_timing import ViewerFrameTiming
from scripts.aufgabe04.perception.stand_axis.models import StandAxisEdgeDebugArtifacts
from scripts.aufgabe04.perception.stand_axis.observation_freshness import observation_freshness
from scripts.aufgabe04.perception.stand_axis.pose_tracking import MetricPoseTracker
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import PlanarPoseHypothesis


class MetricObservationFreshnessTest(unittest.TestCase):
    def setUp(self):
        self.tracker = MetricPoseTracker(prediction_ttl_sec=0.25)
        self.profile = "a" * 64
        self.camera = (400.0, 400.0, 320.0, 240.0)
        self.pose = PlanarPoseHypothesis(
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.4), (0.0, 0.0, 1.0), 0.0, 0.1, True
        )
        self.estimate = replace(
            _unavailable_target_estimate("test"),
            usable=True, evidence_state="fresh_refined", model_profile_sha256=self.profile,
        )
        self.artifacts = StandAxisEdgeDebugArtifacts(
            edges=None, evidence_state="fresh_refined", model_pose=self.pose,
            model_profile_sha256=self.profile,
        )

    def update(self, observed=10.0, completed=10.1, **kwargs):
        return self.tracker.update_from_observation(
            kwargs.pop("estimate", self.estimate),
            kwargs.pop("artifacts", self.artifacts),
            observed_at_sec=observed, completed_at_sec=completed,
            profile_sha256=self.profile, camera_signature=self.camera, **kwargs,
        )

    def prediction(self, now):
        return self.tracker.prediction(
            now_sec=now, profile_sha256=self.profile, camera_signature=self.camera
        )

    def test_qr_only_failed_seed_does_not_replace_pose_or_extend_lifetime(self):
        self.assertTrue(self.update().accepted)
        failed = replace(self.estimate, usable=False, evidence_state="predicted_only")
        wrong = replace(self.artifacts, qr_detected=True, evidence_state="predicted_only",
                        model_pose=replace(self.pose, yaw_deg=45.0))
        result = self.update(10.15, 10.2, estimate=failed, artifacts=wrong)
        self.assertEqual(result.reason, "pose_not_verified")
        self.assertEqual(self.prediction(10.22).pose, self.pose)
        self.assertIsNone(self.prediction(10.26).pose)

    def test_success_is_dated_at_capture_not_processing_completion(self):
        self.assertTrue(self.update(10.0, 10.2).accepted)
        self.assertAlmostEqual(self.prediction(10.22).age_sec, 0.22)
        self.assertIsNone(self.prediction(10.26).pose)

    def test_stale_completion_cannot_enter_tracker_without_a_newer_frame(self):
        self.assertEqual(self.update(10.0, 10.926).reason, "pose_observation_stale")
        self.assertIsNone(self.prediction(10.93).pose)

    def test_external_result_rejection_preserves_previous_pose(self):
        self.assertTrue(self.update().accepted)
        self.assertFalse(self.update(10.1, 10.2, result_fresh=False).accepted)
        self.assertAlmostEqual(self.prediction(10.2).age_sec, 0.2)

    def test_duplicate_or_older_observations_do_not_renew_pose(self):
        self.assertTrue(self.update().accepted)
        for stamp in (10.0, 9.99):
            with self.subTest(stamp=stamp):
                self.assertEqual(self.update(stamp, 10.1).reason, "pose_observation_not_newer")

    def test_profile_mismatch_cannot_enter_tracker(self):
        wrong = replace(self.artifacts, model_profile_sha256="b" * 64)
        self.assertEqual(self.update(artifacts=wrong).reason, "pose_profile_mismatch")

    def test_missing_nonfinite_and_future_capture_timestamps_reject(self):
        for stamp in (None, float("nan"), float("inf"), 10.2):
            with self.subTest(stamp=stamp):
                self.assertFalse(self.update(stamp, 10.1).accepted)

    def test_receipt_age_is_absolute_independent_of_sequence(self):
        for newest in (8, 10, 12):
            with self.subTest(newest=newest):
                self.assertTrue(_detector_result_is_obsolete(
                    processed_sequence=10, newest_sequence=newest,
                    received_monotonic_sec=10.0, completed_monotonic_sec=10.926,
                    max_result_age_sec=0.18,
                ))

    def test_source_age_includes_transport_and_processing(self):
        timing = ViewerFrameTiming.from_read(SimpleNamespace(
            stamp_sec=1000.0, received_wall_sec=1000.2, received_monotonic_sec=10.0
        ))
        # Receipt budget passes at 100ms; absolute source budget fails at 300ms.
        result = timing.assess(now_sec=10.1, max_result_age_sec=0.18, max_frame_age_sec=0.25)
        self.assertFalse(result.accepted)
        self.assertAlmostEqual(timing.source_age_sec(10.1), 0.3)

    def test_recorded_age_includes_time_spent_detecting(self):
        timing = ViewerFrameTiming(10.0, 10.0 - 0.309)
        self.assertAlmostEqual(timing.source_age_sec(10.857), 1.166)

    def test_missing_receipt_timestamp_is_not_fresh_when_budget_enabled(self):
        self.assertFalse(observation_freshness(
            observed_at_sec=None, now_sec=1.0, max_age_sec=0.18
        ).accepted)

    def test_explicit_zero_budget_disables_age_guard_for_diagnostics(self):
        self.assertTrue(observation_freshness(
            observed_at_sec=1.0, now_sec=10.0, max_age_sec=0.0
        ).accepted)


if __name__ == "__main__":
    unittest.main()

class MetricSearchHintLifetimeTest(MetricObservationFreshnessTest):
    """A longer search lifetime never makes an old angle a current result."""

    def setUp(self):
        super().setUp()
        self.tracker = MetricPoseTracker(prediction_ttl_sec=.25,
            search_hint_ttl_sec=2., max_soft_misses=2)

    # Existing base-class tests deliberately require the legacy .25-second
    # search policy, so exercise this explicit opt-in independently below.
    def test_success_is_dated_at_capture_not_processing_completion(self):
        self.assertTrue(self.update().accepted)
        hint = self.prediction(10.8)
        self.assertEqual(hint.state, "predicted_only")
        self.assertAlmostEqual(hint.age_sec, .8)
        self.assertIsNone(self.prediction(12.01).pose)

    def test_qr_only_failed_seed_does_not_replace_pose_or_extend_lifetime(self):
        self.assertTrue(self.update().accepted)
        failed = replace(self.estimate, usable=False)
        for stamp in (10.4, 10.8):
            result = self.update(stamp, stamp+.05, estimate=failed)
            self.assertFalse(result.accepted)
            self.assertEqual(self.prediction(stamp+.1).pose, self.pose)
            self.assertAlmostEqual(self.prediction(stamp+.1).age_sec, stamp+.1-10.)
        self.assertIsNone(self.prediction(12.01).pose)

    def test_three_distinct_failed_observations_force_cold_search(self):
        self.assertTrue(self.update().accepted)
        for stamp in (10.4, 10.8, 11.2):
            self.update(stamp, stamp+.05, estimate=replace(self.estimate, usable=False))
        self.assertIsNone(self.prediction(11.3).pose)

    def test_repeated_failure_of_same_source_is_one_miss(self):
        self.assertTrue(self.update().accepted)
        for _ in range(4):
            self.update(10.4, 10.45, estimate=replace(self.estimate, usable=False))
        self.assertEqual(self.prediction(10.5).pose, self.pose)

    def test_long_search_lifetime_does_not_accept_stale_current_geometry(self):
        self.assertTrue(self.update().accepted)
        result = self.update(10.1, 10.6)
        self.assertEqual(result.reason, "pose_observation_stale")
        self.assertAlmostEqual(self.prediction(10.7).age_sec, .7)

    def test_changed_camera_context_invalidates_before_rejected_update(self):
        self.assertTrue(self.update().accepted)
        self.tracker.update_from_observation(None, None, observed_at_sec=10.1,
            completed_at_sec=10.2, profile_sha256=self.profile,
            camera_signature=(400., 400., 310., 240.))
        self.assertIsNone(self.prediction(10.25).pose)
