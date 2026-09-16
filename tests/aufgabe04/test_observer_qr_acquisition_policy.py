"""Current-pixel QR coverage and finite work on repeatedly empty back views."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from scripts.aufgabe04.perception.stand_axis.models import StandAxisEdgeDebugArtifacts
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation
from scripts.aufgabe04.real_robot.observer.qr_acquisition_policy import (
    MAX_FULL_QR_WORK_SEC, QrAcquisitionPolicy, evaluate_roi_with_qr_acquisition,
    merge_current_qr_observations,
)
from scripts.aufgabe04.real_robot.observer.qr_decode_cache import RoiQrDecodeCache


NOMINAL = (10, 10, 110, 110)
WIDE = (0, 0, 160, 160)
QUAD = ((20., 20.), (40., 20.), (40., 40.), (20., 40.))


class ObserverQrAcquisitionPolicyTests(unittest.TestCase):
    def frame_budget(self, policy=None, *, stamp=100., age=.08, now=10.):
        return (policy or QrAcquisitionPolicy()).begin_frame(
            target_key="candidate", image_stamp_sec=stamp,
            started_ros_sec=stamp + age, started_monotonic_sec=now,
            max_sensor_age_sec=.5,
        )

    def request(self, budget, *, roi=NOMINAL, source="nominal_projection", now=10.02,
                signal=False, bound=False):
        return budget.request(roi=roi, roi_source=source, now_monotonic_sec=now,
                              current_qr_signal=signal, identity_geometry_available=bound)

    def test_empty_search_alternates_crops_but_native_checks_every_current_crop(self):
        policy = QrAcquisitionPolicy()
        full_crops = []
        native_frames, fit_frames = [], []
        for index in range(18):
            frame = object()
            stamp, now = 100. + index * .2, 10. + index * .2
            budget, cache = self.frame_budget(policy, stamp=stamp, now=now), RoiQrDecodeCache()
            for roi, source in ((NOMINAL, "nominal_projection"), (WIDE, "expanded")):
                def native(current):
                    native_frames.append(current)
                    return ()
                def fit(observations):
                    self.assertEqual(observations, ())
                    fit_frames.append(frame)
                    return SimpleNamespace(), StandAxisEdgeDebugArtifacts(edges=None)
                def full(current, limit, diagnostics):
                    self.assertIs(current, frame)
                    self.assertLessEqual(limit, MAX_FULL_QR_WORK_SEC)
                    full_crops.append((index, roi))
                    return ()
                evaluate_roi_with_qr_acquisition(
                    frame=frame, roi=roi, roi_source=source, cache=cache, budget=budget,
                    native_decoder=native, full_decoder=full, estimate=fit, now=lambda: now + .02,
                )
        self.assertEqual(full_crops, [(0, NOMINAL), (5, WIDE), (10, NOMINAL), (15, WIDE)])
        self.assertEqual(len(native_frames), 36)
        self.assertEqual(native_frames, fit_frames)
        self.assertEqual(len({id(frame) for frame in native_frames}), 18)

    def test_current_marker_escalates_without_waiting_for_periodic_probe(self):
        policy = QrAcquisitionPolicy()
        self.assertTrue(self.request(self.frame_budget(policy)).allowed)
        budget = self.frame_budget(policy, stamp=100.2)
        native, full = Mock(return_value=()), Mock(return_value=(DecodedQrObservation("QR_003", QUAD, "full", 1.),))
        def fit(observations):
            return observations, StandAxisEdgeDebugArtifacts(edges=None, qr_detected=True,
                                                             qr_marker_verified=True)
        result, debug, observations, metadata = evaluate_roi_with_qr_acquisition(
            frame=object(), roi=NOMINAL, roi_source="nominal_projection",
            cache=RoiQrDecodeCache(), budget=budget, native_decoder=native,
            full_decoder=full, estimate=fit, now=lambda: 10.02,
        )
        self.assertIs(result, observations)
        self.assertIs(observations, full.return_value)
        self.assertTrue(debug.qr_marker_verified)
        self.assertTrue(metadata["current_image_geometry_refit"])
        self.assertEqual(metadata["acquisition"]["reason"], "current_qr_identity_recovery")
        native.assert_called_once()
        full.assert_called_once()

    def test_current_bound_identity_does_not_need_full_payload_recovery(self):
        budget = self.frame_budget()
        decision = self.request(budget, signal=True, bound=True)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "native_identity_geometry_available")

    def test_one_full_crop_and_exact_image_cache_reuse_do_not_reuse_geometry(self):
        budget, cache = self.frame_budget(), RoiQrDecodeCache()
        frame = object()
        native, full = Mock(return_value=()), Mock(return_value=())
        fit = Mock(return_value=(object(), StandAxisEdgeDebugArtifacts(edges=None, qr_detected=True)))
        for roi, source in ((NOMINAL, "nominal_projection"), (WIDE, "expanded"),
                            (NOMINAL, "nominal_projection")):
            evaluate_roi_with_qr_acquisition(
                frame=frame, roi=roi, roi_source=source, cache=cache, budget=budget,
                native_decoder=native, full_decoder=full, estimate=fit, now=lambda: 10.02,
            )
        self.assertEqual(native.call_count, 2)
        full.assert_called_once()
        self.assertEqual(fit.call_count, 3)
        self.assertTrue(budget.metadata()["decisions"][-1]["cache_only"])

    def test_insufficient_remaining_image_time_cannot_start_full_recovery(self):
        budget = self.frame_budget(age=.4)
        decision = self.request(budget, now=10.02, signal=True)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "image_processing_budget_exhausted")
        self.assertIsNone(budget._full_roi)

    def test_older_scan_work_deadline_bounds_qr_without_double_reserve(self):
        budget = QrAcquisitionPolicy().begin_frame(
            target_key="candidate", image_stamp_sec=100., started_ros_sec=100.1,
            started_monotonic_sec=10., max_sensor_age_sec=.5,
            work_deadline_monotonic_sec=10.08)
        decision = self.request(budget, now=10.02, signal=True)
        self.assertTrue(decision.allowed)
        self.assertAlmostEqual(decision.max_elapsed_sec, .06)
        self.assertFalse(self.request(budget, roi=WIDE, now=10.07, signal=True).allowed)

    def test_work_deadline_cannot_extend_image_freshness(self):
        budget = QrAcquisitionPolicy().begin_frame(
            target_key="candidate", image_stamp_sec=100., started_ros_sec=100.4,
            started_monotonic_sec=10., max_sensor_age_sec=.5,
            work_deadline_monotonic_sec=11.)
        self.assertFalse(self.request(budget, now=10.02, signal=True).allowed)

    def test_saturated_native_cache_cannot_repeat_the_one_allowed_full_decode(self):
        budget, cache = self.frame_budget(), RoiQrDecodeCache()
        frame = object()
        for roi in (NOMINAL, WIDE, (0, 0, 120, 120)):
            cache.decode(roi=roi, mode="native", frame=frame, decoder=lambda _: ())
        full = Mock(return_value=())
        fit = Mock(return_value=(object(), StandAxisEdgeDebugArtifacts(edges=None)))
        for _ in range(2):
            evaluate_roi_with_qr_acquisition(
                frame=frame, roi=NOMINAL, roi_source="nominal_projection", cache=cache,
                budget=budget, native_decoder=lambda _: (), full_decoder=full,
                estimate=fit, now=lambda: 10.02,
            )
        full.assert_called_once()
        self.assertEqual(fit.call_count, 2)

    def test_remaining_budget_and_publication_reserve_are_not_extended(self):
        budget = self.frame_budget(age=.30)
        decision = self.request(budget, now=10.06, signal=True)
        self.assertTrue(decision.allowed)
        self.assertAlmostEqual(decision.max_elapsed_sec, .09)
        self.assertAlmostEqual(budget.metadata()["deadline_monotonic_sec"], 10.2)

    def test_target_switch_restarts_schedule_without_reusing_any_observation(self):
        policy = QrAcquisitionPolicy()
        self.assertTrue(self.request(self.frame_budget(policy)).allowed)
        self.assertFalse(self.request(self.frame_budget(policy, stamp=100.1)).allowed)
        budget = policy.begin_frame(target_key="other", image_stamp_sec=100.1,
            started_ros_sec=100.2, started_monotonic_sec=10., max_sensor_age_sec=.5)
        self.assertTrue(self.request(budget).allowed)

    def test_native_text_and_conflicts_survive_empty_or_disagreeing_full_decode(self):
        native = (DecodedQrObservation("QR_003", None, "native", 1.),)
        self.assertIs(merge_current_qr_observations(native, ()), native)
        acquired = (DecodedQrObservation("QR_004", QUAD, "full", 1.),)
        self.assertEqual({item.text for item in merge_current_qr_observations(native, acquired)},
                         {"QR_003", "QR_004"})
        same = (DecodedQrObservation("QR_003", QUAD, "full", 1.),)
        self.assertIs(merge_current_qr_observations(native, same), same)
        multiple = same * 2
        self.assertEqual(len(merge_current_qr_observations(native, multiple)), 2)


if __name__ == "__main__":
    unittest.main()
