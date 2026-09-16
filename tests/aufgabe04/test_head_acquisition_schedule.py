"""Clipped projected crops cannot consume two acquisition passes per image."""

from types import SimpleNamespace
from contextlib import ExitStack
from unittest.mock import Mock, patch
import unittest

import numpy

from scripts.aufgabe04.real_robot.observer.head_acquisition_schedule import (
    HeadProcessingDeadline, select_cold_candidate_head,
)
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import HeadRoiAttempt
from scripts.aufgabe04.real_robot.observer.qr_acquisition_policy import QrAcquisitionDecision
from scripts.aufgabe04.real_robot.configuration.geometry import ImageRoi
from scripts.aufgabe04.perception.stand_axis.physical_head_pipeline import fit_physical_head_in_frame
from scripts.aufgabe04.real_robot.observer import node as observer_node
from tests.aufgabe04 import test_camera_observer_processing as processing_fixtures


class Frame:
    def __getitem__(self, slices):
        return self


class AcquisitionScheduleTests(unittest.TestCase):
    def setUp(self):
        self.profile = SimpleNamespace(sha256="a" * 64, measurement_status="measured")
        self.nominal = HeadRoiAttempt(ImageRoi(275, 183, 493, 402, 121.),
                                     "nominal_projection", 1.8, 384., 293., 121.)
        self.wide = HeadRoiAttempt(ImageRoi(111, 20, 657, 567, 121.),
                                  "target_centered_backside_reacquisition", 4.5,
                                  384., 293., 121.)

    def budget(self, **changes):
        args = dict(image_stamp_sec=100., scan_stamp_sec=100., started_ros_sec=100.27,
                    started_monotonic_sec=10., max_sensor_age_sec=.5)
        return HeadProcessingDeadline(**{**args, **changes})

    def test_deadline_uses_older_source_and_reserves_publication_time(self):
        budget = self.budget(scan_stamp_sec=99.9)
        self.assertAlmostEqual(budget.deadline_monotonic_sec, 10.08)
        self.assertTrue(budget.allow("locate", minimum_work_sec=.04, now=10.03))
        self.assertFalse(budget.allow("fit", minimum_work_sec=.01, now=10.075))

    def select(self, locate, diagnostics, budget):
        return select_cold_candidate_head((self.nominal, self.wide), frame=Frame(),
            model_profile=self.profile, acquire_registered=locate,
            diagnostics=diagnostics, budget=budget)

    def test_cold_search_starts_wide_and_returns_only_associated_strict_selection(self):
        result = object()
        locate = Mock(return_value=result)
        with patch("scripts.aufgabe04.real_robot.observer.head_acquisition_schedule.time.monotonic",
                   return_value=10.01):
            self.assertIs(self.select(locate, {}, self.budget()), result)
        locate.assert_called_once_with(self.wide, None)

    def test_failed_search_does_not_run_nominal_fit_or_invent_qr_absence(self):
        locate = Mock(return_value=None)
        with patch("scripts.aufgabe04.real_robot.observer.head_acquisition_schedule.time.monotonic",
                   return_value=10.01):
            selected = self.select(locate, {"reason": "head_proposal_ambiguous"}, self.budget())
        locate.assert_called_once()
        self.assertFalse(selected.registered)
        self.assertFalse(selected.selected.estimate.usable)
        self.assertIsNone(selected.selected.estimate.yaw_deg)
        self.assertIsNone(selected.selected.qr_observations)
        self.assertEqual(selected.selected.estimate.reason, "head_proposal_ambiguous")

    def test_no_new_search_when_source_budget_cannot_cover_it(self):
        locate = Mock()
        with patch("scripts.aufgabe04.real_robot.observer.head_acquisition_schedule.time.monotonic",
                   return_value=10.17):
            selected = self.select(locate, {}, self.budget())
        locate.assert_not_called()
        self.assertEqual(selected.selected.estimate.reason, "head_acquisition_deadline_exceeded")

    def test_expired_physical_head_stage_never_calls_acquisition_or_pose_fit(self):
        prefix = "scripts.aufgabe04.perception.stand_axis.physical_head_pipeline."
        with patch(prefix + "time.monotonic", return_value=10.), \
             patch(prefix + "fit_current_measured_head") as fit, \
             patch(prefix + "acquire_cold_head_proposal") as locate:
            estimate, debug, pose = fit_physical_head_in_frame(
                object(), Frame(), None, model_profile=self.profile, camera=object(),
                timing=Mock(), deadline_monotonic_sec=9.)
        fit.assert_not_called()
        locate.assert_not_called()
        self.assertFalse(estimate.usable)
        self.assertIsNone(pose)
        self.assertEqual(estimate.reason, "head_acquisition_deadline_exceeded")


class AcquisitionFailureObserverTests(unittest.TestCase):
    def test_head_miss_permits_bounded_identity_probe_but_expired_budget_skips_it(self):
        for failure in ("head_proposal_ambiguous", "head_acquisition_deadline_exceeded"):
            with self.subTest(failure=failure), ExitStack() as stack:
                fixture = processing_fixtures.CameraObserverProcessingTest()
                adapter = fixture.make_adapter()
                adapter.stand_model_profile.environment = "physical"
                adapter.stand_model_profile.committable = True
                frame = numpy.zeros((600, 800, 3), dtype=numpy.uint8)
                module = "scripts.aufgabe04.real_robot.observer.node."
                for name in ("camera_info_mismatches", "transform_mismatches"):
                    stack.enter_context(patch(module + name, return_value=()))
                stack.enter_context(patch(module + "compressed_msg_to_bgr_frame", return_value=frame))
                stack.enter_context(patch(module + "_rectify_bgr_frame", side_effect=lambda value, *_, **_kwargs: value))
                decoders = [stack.enter_context(patch(module + name, return_value=())) for name in (
                    "detect_qr_texts_bgr", "detect_qr_observations_bgr",
                    "detect_native_qr_observations_bgr")]
                fit = stack.enter_context(patch(module + "estimate_stand_axis_from_metric_model"))
                bind = stack.enter_context(patch(module + "bind_qr_observations_to_target",
                    wraps=observer_node.bind_qr_observations_to_target))

                def failed_proposal(*_args, diagnostics, **_kwargs):
                    diagnostics["reason"] = failure
                    return None

                acquire = stack.enter_context(patch(
                    module + "acquire_registered_head_measurement", side_effect=failed_proposal))
                if failure == "head_acquisition_deadline_exceeded":
                    stack.enter_context(patch(
                        "scripts.aufgabe04.real_robot.observer.head_acquisition_schedule."
                        "HeadProcessingDeadline.allow", return_value=False))
                    stack.enter_context(patch(
                        "scripts.aufgabe04.real_robot.observer.qr_acquisition_policy."
                        "QrFrameAcquisitionBudget.request",
                        return_value=QrAcquisitionDecision(
                            False, "image_processing_budget_exhausted", 0.)))
                adapter._process_latest()
                decoders[0].assert_not_called()  # Legacy unbounded text fallback.
                decoders[2].assert_not_called()
                if failure == "head_acquisition_deadline_exceeded":
                    decoders[1].assert_not_called()
                else:
                    decoders[1].assert_called_once()
                    self.assertLessEqual(decoders[1].call_args.kwargs["max_elapsed_sec"], .12)
                fit.assert_not_called()
                if failure == "head_acquisition_deadline_exceeded":
                    acquire.assert_not_called()
                else:
                    acquire.assert_called_once()
                self.assertFalse(adapter.completed)
                status = adapter._write_status.call_args
                self.assertEqual(status.args, ("metric_model_measurement_unavailable",))
                self.assertEqual(status.kwargs["estimator_reason"], failure)
                timing = status.kwargs["stand_axis_debug"]["metric_model"]["processing_timing"]
                metadata = timing["attempts"][0]["qr_decode"]
                self.assertEqual(metadata["performed"], failure != "head_acquisition_deadline_exceeded")
                # Skipped decoding remains unknown and supplies no identity or angle.
                self.assertIn(bind.call_args.args[0], (None, ()))
                self.assertIsNone(adapter._last_observation_update.resolved_qr_id)
                self.assertEqual(adapter._last_observation_update.snapshot.current_qr_sample_count, 0)
                self.assertEqual(adapter._last_observation_update.snapshot.current_axis_sample_count, 0)


if __name__ == "__main__":
    unittest.main()
