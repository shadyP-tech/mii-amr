"""Metric corridor filtering preserves full-image raw measurements exactly."""

import math
import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

from scripts.aufgabe04.perception.stand_axis import raw_support
from scripts.aufgabe04.perception.stand_axis.model_refinement import refine_projected_head_border
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint


@unittest.skipIf(cv2 is None or np is None, "OpenCV and NumPy required")
class RawMetricBackgroundFilterTests(unittest.TestCase):
    def scene(self, kind):
        raw = np.zeros((720, 1280), np.uint8)
        # Strong window/radiator lines dominate the image but not the finite
        # head-side corridors. These are untouched raw pixels, not a new Canny.
        raw[::9, :] = 255
        raw[:, ::11] = 255
        points = ((510., 300.), (630., 300.), (630., 420.), (510., 420.))
        if kind == "perspective":
            points = ((465., 240.), (590., 275.), (585., 440.), (460., 405.))
        elif kind == "image_edge":
            points = ((2., 3.), (122., 3.), (122., 123.), (2., 123.))
        corners = tuple(ImagePoint(x, y) for x, y in points)
        x0, y0 = (max(0, int(min(p[axis] for p in points)) - 35) for axis in (0, 1))
        x1, y1 = (int(max(p[axis] for p in points)) + 36 for axis in (0, 1))
        raw[y0:y1, x0:x1] = 0
        for index, (a, b) in enumerate(zip(points, points[1:] + points[:1])):
            if not (kind == "missing_border" and index == 0):
                cv2.line(raw, tuple(map(round, a)), tuple(map(round, b)), 255, 1)
        if kind == "extended_radiator":
            # Aligned distant continuations must neither grow endpoints nor
            # disappear from the full-coordinate current evidence contract.
            cv2.line(raw, (510, 10), (510, 710), 255, 1)
            cv2.line(raw, (630, 5), (630, 715), 255, 1)
            cv2.line(raw, (634, 30), (634, 695), 255, 1)
        return raw, corners

    def full_points(self, raw):
        found = cv2.findNonZero(raw)
        return np.empty((0, 2), np.float64) if found is None else found.reshape(-1, 2).astype(np.float64)

    def test_measurements_match_unfiltered_reference_for_background_and_failures(self):
        for kind in ("clutter", "perspective", "missing_border", "extended_radiator", "image_edge"):
            with self.subTest(kind=kind):
                raw, corners = self.scene(kind)
                original = raw.copy()
                # Bypass only the workload filter. Everything selecting raw
                # pixels, fitting rails and accepting corners remains real.
                with patch.object(raw_support, "_raw_edge_points_in_side_bounds",
                                  return_value=self.full_points(raw)):
                    full = refine_projected_head_border(cv2, raw, corners, corridor_half_width_px=6.)
                filtered = refine_projected_head_border(cv2, raw, corners, corridor_half_width_px=6.)
                for field in ("accepted", "reason", "corners", "support", "candidate_corners",
                              "corner_arm_support"):
                    self.assertEqual(getattr(filtered, field), getattr(full, field), field)
                np.testing.assert_array_equal(filtered.evidence_mask, full.evidence_mask)
                np.testing.assert_array_equal(raw, original)
                self.assertEqual(filtered.evidence_mask.shape, raw.shape)
                if kind == "missing_border":
                    self.assertFalse(filtered.accepted)
                else:
                    self.assertTrue(filtered.accepted, filtered.reason)

    def test_metric_extraction_scans_four_small_boxes_and_reuses_side_points(self):
        raw, corners = self.scene("clutter")
        full_count = len(self.full_points(raw))
        extraction_shapes, fitted_points = [], []
        original_find, original_fit = cv2.findNonZero, raw_support._fit_raw_edge_side_in_band

        def find(image):
            extraction_shapes.append(image.shape)
            return original_find(image)

        def fit(cv, points, *args, **kwargs):
            fitted_points.append(points)
            return original_fit(cv, points, *args, **kwargs)

        with patch.object(cv2, "findNonZero", side_effect=find), \
             patch.object(raw_support, "_fit_raw_edge_side_in_band", side_effect=fit):
            evidence, result = raw_support._raw_side_evidence_and_corners(
                cv2, raw, corners, prefer_prediction=True, recover_parallel_endpoints=False)
        self.assertIsNotNone(result)
        self.assertEqual(evidence.shape, raw.shape)
        self.assertEqual(len(extraction_shapes), 4)
        self.assertTrue(all(rows * cols < raw.size / 100 for rows, cols in extraction_shapes))
        self.assertEqual(len(fitted_points), 6)
        self.assertTrue(all(len(points) < full_count / 100 for points in fitted_points))
        self.assertIs(fitted_points[2], fitted_points[4])
        self.assertIs(fitted_points[3], fitted_points[5])

    def test_supplied_point_order_and_subpixel_coordinates_are_preserved(self):
        raw, corners = self.scene("perspective")
        points = self.full_points(raw)[::-1].copy()
        points += (.125, -.125)
        original = points.copy()
        options = dict(prefer_prediction=True, recover_parallel_endpoints=False,
                       maximum_band_px=5., edge_points=points)
        with patch.object(raw_support, "_raw_edge_points_in_side_bounds", return_value=points):
            full_mask, full_corners = raw_support._raw_side_evidence_and_corners(cv2, raw, corners, **options)
        filtered_mask, filtered_corners = raw_support._raw_side_evidence_and_corners(
            cv2, raw, corners, **options)
        self.assertIsNotNone(full_corners)
        self.assertEqual(filtered_corners, full_corners)
        np.testing.assert_array_equal(filtered_mask, full_mask)
        np.testing.assert_array_equal(points, original)

    def test_side_envelope_keeps_extended_disjoint_intervals_and_band_edges(self):
        start, end = ImagePoint(120.25, 90.75), ImagePoint(350.25, 170.75)
        dx, dy = end.u_px - start.u_px, end.v_px - start.v_px
        length = math.hypot(dx, dy)
        tangent, normal = np.array((dx, dy)) / length, np.array((-dy, dx)) / length
        band = 6.125
        intervals = ((-.25, .42), (.58, 1.35))
        points = np.array([
            np.array((start.u_px, start.v_px)) + t * length * tangent + offset * normal
            for t in np.linspace(-.5, 1.5, 401) for offset in (-band, 0., band)
        ])[::-1]
        # Supplied off-image points have always been valid inputs to the raw
        # side fitter; filtering must not silently clip that caller's array.
        raw = np.zeros((80, 80), np.uint8)
        local = raw_support._raw_edge_points_in_side_bounds(
            cv2, raw, start, end, band_px=band, intervals=intervals, edge_points=points)
        options = dict(band_px=band, intervals=intervals)
        full_line, full_evidence = raw_support._fit_raw_edge_side_in_band(
            cv2, points, start, end, **options)
        local_line, local_evidence = raw_support._fit_raw_edge_side_in_band(
            cv2, local, start, end, **options)
        self.assertIsNotNone(full_line)
        self.assertGreater(len(full_evidence), 100)
        self.assertEqual(local_line, full_line)
        np.testing.assert_array_equal(local_evidence, full_evidence)

    def test_legacy_endpoint_recovery_does_not_use_finite_metric_filter(self):
        raw, corners = self.scene("extended_radiator")
        with patch.object(raw_support, "_raw_edge_points_in_side_bounds",
                          side_effect=AssertionError("endpoint recovery needs its full search")):
            for prefer_prediction in (False, True):
                raw_support._raw_side_evidence_and_corners(
                    cv2, raw, corners, prefer_prediction=prefer_prediction,
                    fixed_parallel_side_direction=(0., 1.), recover_parallel_endpoints=True)

    def assert_support_matches_full_transform(self, raw, corners, tolerance=None):
        original = raw.copy()
        # The old implementation always transforms the complete mask. Force
        # only that path; keep its sampling, rounding and support policy real.
        with patch.object(raw_support, "_corners_inside_image", return_value=False):
            full = raw_support._quadrilateral_edge_support(cv2, raw, corners, tolerance_px=tolerance)
        cropped = raw_support._quadrilateral_edge_support(cv2, raw, corners, tolerance_px=tolerance)
        self.assertEqual(cropped, full)
        self.assertEqual(cropped.accepted, full.accepted)
        self.assertEqual(cropped.mean, full.mean)
        np.testing.assert_array_equal(raw, original)
        return cropped

    def test_bounded_distance_support_matches_full_transform_for_randomized_masks(self):
        rng = np.random.default_rng(410417)
        for index in range(100):
            height, width = (int(n) for n in rng.integers(90, 360, size=2))
            raw = (rng.random((height, width)) < (.001, .01, .1, .7)[index % 4]).astype(np.uint8) * 255
            u0, u1 = sorted(rng.uniform(0., width - .001, 2))
            v0, v1 = sorted(rng.uniform(0., height - .001, 2))
            # Skewed quadrilaterals and subpixel corners exercise all five
            # original sample intervals without pixel-aligned shortcuts.
            corners = (ImagePoint(u0, v0), ImagePoint(u1, v0 + .17 * (v1-v0)),
                       ImagePoint(u1 - .12 * (u1-u0), v1), ImagePoint(u0, v1 - .09 * (v1-v0)))
            tolerance = (None, 0., .5, .9549, .955, .9551, 1.3693, 2., 5., 8.125)[index % 10]
            with self.subTest(index=index, tolerance=tolerance):
                self.assert_support_matches_full_transform(raw, corners, tolerance)

    def test_bounded_distance_support_preserves_real_head_masks_and_stem_notch(self):
        for kind in ("clutter", "perspective", "missing_border", "extended_radiator", "image_edge"):
            raw, corners = self.scene(kind)
            for tolerance in (None, 0., 2., 3.5, 8.):
                with self.subTest(kind=kind, tolerance=tolerance):
                    self.assert_support_matches_full_transform(raw, corners, tolerance)
        raw, corners = self.scene("clutter")
        raw[418:423, 555:585] = 0  # A permitted central bottom/stem notch.
        support = self.assert_support_matches_full_transform(raw, corners)
        self.assertTrue(support.accepted)

    def test_distance_crop_keeps_tolerated_edges_beyond_corner_box(self):
        corners = tuple(ImagePoint(x, y) for x, y in ((80.5, 80.5), (160.5, 80.5),
                                                     (160.5, 160.5), (80.5, 160.5)))
        for separation in range(1, 8):
            raw = np.zeros((260, 300), np.uint8)
            cv2.rectangle(raw, (80-separation, 80-separation),
                          (161+separation, 161+separation), 255, 1)
            # Include thresholds on both sides of the actual chamfer distance.
            for tolerance in (.955 * separation - 1e-4, .955 * separation + 1e-4,
                              float(separation + 1)):
                with self.subTest(separation=separation, tolerance=tolerance):
                    self.assert_support_matches_full_transform(raw, corners, tolerance)

    def test_distance_crop_reduces_work_and_retains_full_fallback(self):
        raw, corners = self.scene("clutter")
        original_transform = cv2.distanceTransform
        transformed_shapes = []

        def transform(image, *args, **kwargs):
            transformed_shapes.append(image.shape)
            return original_transform(image, *args, **kwargs)

        with patch.object(cv2, "distanceTransform", side_effect=transform):
            raw_support._quadrilateral_edge_support(cv2, raw, corners)
            outside = (ImagePoint(-.1, 300.), *corners[1:])
            self.assert_support_matches_full_transform(raw, outside)
            for tolerance in (-1., math.inf):
                self.assert_support_matches_full_transform(raw, corners, tolerance)
        self.assertLess(math.prod(transformed_shapes[0]), raw.size / 20)
        self.assertTrue(all(shape == raw.shape for shape in transformed_shapes[1:]))


if __name__ == "__main__":
    unittest.main()
