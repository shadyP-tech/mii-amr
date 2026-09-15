"""Exact current-image reuse must not reuse geometry or bypass acquisition."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

try:
    import numpy
except ImportError:  # pragma: no cover
    numpy = None

from scripts.aufgabe04.perception.stand_axis.geometry import _unusable
from scripts.aufgabe04.perception.stand_axis.model_input_cache import MetricModelInputCache
from scripts.aufgabe04.perception.stand_axis.model_pipeline import (
    estimate_stand_axis_from_metric_model,
)
from scripts.aufgabe04.perception.stand_axis.model_profile import load_stand_model
from scripts.aufgabe04.perception.stand_axis.models import StandAxisEdgeDebugArtifacts
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation


PIPELINE = "scripts.aufgabe04.perception.stand_axis.model_pipeline"
CACHE_MODULE = "scripts.aufgabe04.perception.stand_axis.model_input_cache"
ROI = (10, 20, 80, 90)


@unittest.skipIf(numpy is None, "NumPy is required for image view tests")
class ModelInputCacheTest(unittest.TestCase):
    def setUp(self):
        self.frame = numpy.zeros((100, 120, 3), dtype=numpy.uint8)
        self.cv2 = object()
        self.cache = MetricModelInputCache(self.frame)

    def begin(self, *, cache=None, frame=None, roi=ROI, **overrides):
        if frame is None:
            frame = self.frame[roi[1]:roi[3], roi[0]:roi[2]]
        settings = dict(
            cv2=self.cv2, edge_preprocess="channel_union", blur_kernel=5,
            canny_low=20, canny_high=60, pose_hint_present=False,
            qr_observations=(),
        )
        settings.update(overrides)
        return (self.cache if cache is None else cache).begin(frame, roi=roi, **settings)

    def test_same_current_pixels_reuse_edges_and_negative_qr_detection(self):
        edges = Mock(return_value=numpy.ones((70, 70), dtype=numpy.uint8))
        quad = Mock(return_value=None)
        for hit in (False, True):
            current = self.begin()
            numpy.testing.assert_array_equal(current.compute("edge_preprocessing", edges), 1)
            self.assertIsNone(current.compute("qr_detection", quad))
            self.assertEqual(self.cache.last_metadata["edge_preprocessing"]["cache_hit"], hit)
            self.assertEqual(self.cache.last_metadata["qr_detection"]["cache_hit"], hit)
        edges.assert_called_once_with()
        quad.assert_called_once_with()

    def test_crop_shape_or_pixel_mutation_cannot_alias(self):
        producer = Mock(return_value=None)
        self.begin().compute("qr_detection", producer)
        self.begin(roi=(11, 20, 81, 90)).compute("qr_detection", producer)
        self.frame[20, 10, 0] = 255
        self.begin().compute("qr_detection", producer)
        self.assertEqual(producer.call_count, 3)

    def test_another_image_or_wrong_crop_even_with_same_pixels_is_rejected(self):
        images = (
            self.frame.copy()[20:90, 10:80],
            self.frame[20:90, 11:81],
            self.frame[20:90, 10:80].copy(),
        )
        for crop in images:
            with self.subTest(crop=crop.shape), self.assertRaisesRegex(ValueError, "source-image ROI"):
                self.begin(frame=crop)

    def test_every_new_image_cache_recomputes(self):
        producer = Mock(return_value=None)
        for _ in range(2):
            self.begin(cache=MetricModelInputCache(self.frame)).compute("qr_detection", producer)
        self.assertEqual(producer.call_count, 2)

    def test_preprocessing_backend_mode_and_qr_evidence_have_separate_keys(self):
        changes = (
            {"edge_preprocess": "gray"}, {"blur_kernel": 3}, {"canny_low": 21},
            {"canny_high": 61}, {"pose_hint_present": True}, {"cv2": object()},
            {"qr_observations": None},
            {"qr_observations": (DecodedQrObservation("different", None, "wechat"),)},
        )
        for changed in changes:
            with self.subTest(changed=changed):
                cache = MetricModelInputCache(self.frame)
                producer = Mock(return_value=None)
                self.begin(cache=cache).compute("qr_detection", producer)
                self.begin(cache=cache, **changed).compute("qr_detection", producer)
                self.assertEqual(producer.call_count, 2)

    def test_qr_identity_geometry_and_provenance_cannot_alias(self):
        original = DecodedQrObservation(
            "stand", ((1., 1.), (9., 1.), (9., 9.), (1., 9.)), "native", 1.,
        )
        changes = (
            replace(original, text="neighbor"), replace(original, detector="wechat"),
            replace(original, scale=2.),
            replace(original, corners=((2., 1.), (9., 1.), (9., 9.), (2., 9.))),
        )
        for changed in changes:
            with self.subTest(changed=changed):
                cache = MetricModelInputCache(self.frame)
                producer = Mock(return_value=None)
                self.begin(cache=cache, qr_observations=(original,)).compute("qr_detection", producer)
                self.begin(cache=cache, qr_observations=(changed,)).compute("qr_detection", producer)
                self.assertEqual(producer.call_count, 2)

    def test_returned_edge_mutation_cannot_poison_another_evaluation(self):
        producer = Mock(return_value=numpy.ones((70, 70), dtype=numpy.uint8))
        first = self.begin().compute("edge_preprocessing", producer)
        first[:] = 0
        second = self.begin().compute("edge_preprocessing", producer)
        numpy.testing.assert_array_equal(second, 1)
        second[:] = 42
        numpy.testing.assert_array_equal(self.begin().compute("edge_preprocessing", producer), 1)

    def test_capacity_never_suppresses_work_for_a_new_key(self):
        producer = Mock(return_value=None)
        for x in range(3):
            self.begin(roi=(x, 0, x + 70, 70)).compute("qr_detection", producer)
        for _ in range(2):
            self.begin(roi=(4, 0, 74, 70)).compute("qr_detection", producer)
        self.begin(roi=(0, 0, 70, 70)).compute("qr_detection", producer)
        self.assertEqual(producer.call_count, 5)

    def test_failure_propagates_and_remains_retryable(self):
        producer = Mock(side_effect=(ValueError("detector failed"), None))
        with self.assertRaisesRegex(ValueError, "detector failed"):
            self.begin().compute("qr_detection", producer)
        self.assertIsNone(self.begin().compute("qr_detection", producer))
        self.assertEqual(producer.call_count, 2)

    def test_timing_keeps_reuse_cost_separate_from_original_work(self):
        first = self.begin()
        with patch(f"{CACHE_MODULE}.perf_counter", side_effect=(1., 1.025, 1.026)):
            first.compute("qr_detection", lambda: None)
        second = self.begin()
        with patch(f"{CACHE_MODULE}.perf_counter", side_effect=(2., 2.0001)):
            second.compute("qr_detection", lambda: self.fail("must reuse"))
        metadata = self.cache.last_metadata["qr_detection"]
        self.assertTrue(metadata["cache_hit"])
        self.assertAlmostEqual(metadata["producer_elapsed_ms"], 25.)
        self.assertAlmostEqual(metadata["elapsed_ms"], .1)

    def test_invalid_key_and_unsupported_stages_fail_closed(self):
        for roi in ((10.5, 20, 80, 90), (True, 20, 80, 90), (10, 20, 10, 90),
                    (-1, 20, 80, 90), (10, 20, 121, 90)):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                self.begin(roi=roi, frame=self.frame)
        for maximum in (0, 4, True):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                MetricModelInputCache(self.frame, max_entries=maximum)
        with self.assertRaisesRegex(ValueError, "invalid type"):
            self.begin(qr_observations=(object(),))
        producer = Mock()
        with self.assertRaisesRegex(ValueError, "only raw edges"):
            self.begin().compute("backside_pose", producer)
        producer.assert_not_called()

    def test_pipeline_reuses_only_inputs_and_reruns_current_head_geometry(self):
        profile = load_stand_model(Path(__file__).resolve().parents[2] /
                                   "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json")
        estimate = _unusable("test_head", source="model_current_measured_head")
        artifacts = StandAxisEdgeDebugArtifacts(edges=numpy.zeros((70, 70), dtype=numpy.uint8))
        common = dict(
            model_profile=profile, camera_fx_px=600., camera_fy_px=600.,
            camera_cx_px=60., camera_cy_px=50., expected_head_height_px=50.,
            expected_head_center_v_px=35., qr_observations=(),
            input_cache=self.cache, input_cache_roi=ROI,
        )
        with (
            patch(f"{PIPELINE}._canny_edges_from_frame", return_value=artifacts.edges) as edges,
            patch(f"{PIPELINE}.detect_qr_quad", return_value=None) as quad,
            patch(f"{PIPELINE}.fit_physical_head_in_frame",
                  side_effect=((estimate, artifacts, None),
                               (replace(estimate, reason="strict"), artifacts, None))) as head_fit,
        ):
            first, first_debug = estimate_stand_axis_from_metric_model(
                self.cv2, self.frame[20:90, 10:80], expected_head_center_u_px=45., **common)
            strict, strict_debug = estimate_stand_axis_from_metric_model(
                self.cv2, self.frame[20:90, 10:80], expected_head_center_u_px=35., **common)
        self.assertEqual(first.reason, "test_head")
        self.assertEqual(strict.reason, "strict")
        self.assertEqual(edges.call_count, 1)
        self.assertEqual(quad.call_count, 1)
        self.assertEqual(head_fit.call_count, 2)  # Geometry cannot be cached.
        self.assertEqual(head_fit.call_args_list[0].kwargs["expected_head_center_u_px"], 45.)
        self.assertEqual(head_fit.call_args_list[1].kwargs["expected_head_center_u_px"], 35.)
        for debug in (first_debug, strict_debug):
            self.assertIn("edge_preprocessing", debug.stage_timings_ms)
            self.assertIn("qr_detection", debug.stage_timings_ms)


if __name__ == "__main__":
    unittest.main()
