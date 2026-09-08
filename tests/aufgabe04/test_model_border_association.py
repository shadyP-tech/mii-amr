"""Metric head refinement must associate complete, observed physical rails."""

from __future__ import annotations

import math
import unittest

try:
    import cv2
    import numpy
except ImportError:  # pragma: no cover
    cv2 = None
    numpy = None

from scripts.aufgabe04.perception.stand_axis.model_refinement import (
    refine_projected_head_border,
)
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint


def _corners(points):
    return tuple(ImagePoint(float(x), float(y)) for x, y in points)


def _draw_segment(edges, start, end, *, first=0.0, last=1.0, offset=0.0):
    dx, dy = end.u_px - start.u_px, end.v_px - start.v_px
    length = math.hypot(dx, dy)
    nx, ny = -dy / length, dx / length

    def pixel(fraction):
        return (
            round(start.u_px + fraction * dx + offset * nx),
            round(start.v_px + fraction * dy + offset * ny),
        )

    cv2.line(edges, pixel(first), pixel(last), 255, 1)


def _draw_head(edges, corners, *, missing_side=None):
    for side, (start, end) in enumerate(zip(corners, corners[1:] + corners[:1])):
        if side != missing_side:
            _draw_segment(edges, start, end)


def _draw_nearer_fragments(edges, predicted):
    # QR/background fragments are nearer the prediction but neither supplies
    # a complete rail. Independent per-position nearest selection can stitch
    # these fragments together with a different, displaced physical border.
    for start, end in zip(predicted, predicted[1:] + predicted[:1]):
        _draw_segment(edges, start, end, first=0.08, last=0.36, offset=0.0)
        _draw_segment(edges, start, end, first=0.62, last=0.90, offset=1.0)


@unittest.skipIf(cv2 is None or numpy is None, "OpenCV/numpy unavailable")
class ModelBorderAssociationTest(unittest.TestCase):
    def setUp(self):
        self.predicted = _corners(((80, 60), (200, 60), (200, 180), (80, 180)))

    def assertObservedCorners(self, result, expected, *, tolerance=1.75):
        self.assertTrue(result.accepted, result.reason)
        self.assertIsNotNone(result.corners)
        for actual, observed in zip(result.corners, expected):
            self.assertLessEqual(
                math.hypot(actual.u_px - observed.u_px, actual.v_px - observed.v_px),
                tolerance,
                f"fitted {actual} differs from observed physical corner {observed}",
            )

    def test_complete_shifted_head_wins_over_nearer_fragmented_rails(self):
        for shift_x, shift_y in ((6, 4), (-6, -4)):
            with self.subTest(shift=(shift_x, shift_y)):
                edges = numpy.zeros((260, 300), dtype=numpy.uint8)
                physical = _corners(tuple(
                    (point.u_px + shift_x, point.v_px + shift_y)
                    for point in self.predicted
                ))
                _draw_head(edges, physical)
                _draw_nearer_fragments(edges, self.predicted)
                original = edges.copy()

                result = refine_projected_head_border(
                    cv2, edges, self.predicted, corridor_half_width_px=8.0,
                )

                self.assertObservedCorners(result, physical)
                numpy.testing.assert_array_equal(edges, original)
                self.assertFalse(numpy.any((result.evidence_mask > 0) & (original == 0)))

    def test_blank_or_missing_side_does_not_turn_projection_into_measurement(self):
        for missing_side in ("all", 0, 1, 2, 3):
            with self.subTest(missing_side=missing_side):
                edges = numpy.zeros((260, 300), dtype=numpy.uint8)
                if missing_side != "all":
                    _draw_head(edges, self.predicted, missing_side=missing_side)

                result = refine_projected_head_border(
                    cv2, edges, self.predicted, corridor_half_width_px=8.0,
                )

                self.assertFalse(result.accepted)
                self.assertIsNone(result.corners)

    def test_complete_inner_qr_cannot_substitute_for_missing_outer_head(self):
        # All four QR rails are inside the maximum search corridor, but their
        # 84 px width/height cannot be the predicted 100 px physical head.
        predicted = _corners(((80, 60), (180, 60), (180, 160), (80, 160)))
        inner_qr = _corners(((88, 68), (172, 68), (172, 152), (88, 152)))
        edges = numpy.zeros((220, 260), dtype=numpy.uint8)
        _draw_head(edges, inner_qr)

        result = refine_projected_head_border(
            cv2, edges, predicted, corridor_half_width_px=8.0,
        )

        self.assertFalse(result.accepted)
        self.assertIsNone(result.corners)

    def test_background_continuation_does_not_extend_physical_head_corners(self):
        physical = _corners(((85, 64), (205, 64), (205, 184), (85, 184)))
        edges = numpy.zeros((260, 300), dtype=numpy.uint8)
        _draw_head(edges, physical)
        # A radiator rail aligns with the right side and continues beyond both
        # head corners; the observed cross-rails still define the actual head.
        _draw_segment(edges, physical[1], physical[2], first=-0.35, last=1.35)

        result = refine_projected_head_border(
            cv2, edges, self.predicted, corridor_half_width_px=8.0,
        )

        self.assertObservedCorners(result, physical)

    def test_rolled_perspective_head_keeps_observed_slopes_with_nearby_fragments(self):
        # Both side rails share a rolled direction while top/bottom retain
        # distinct perspective slopes. Axis-aligned rail selection is invalid.
        predicted = _corners(((90, 50), (210, 70), (186, 180), (63, 174)))
        physical = _corners(tuple(
            (point.u_px + 5, point.v_px + 3) for point in predicted
        ))
        edges = numpy.zeros((260, 300), dtype=numpy.uint8)
        _draw_head(edges, physical)
        _draw_nearer_fragments(edges, predicted)

        result = refine_projected_head_border(
            cv2, edges, predicted, corridor_half_width_px=8.0,
        )

        self.assertObservedCorners(result, physical, tolerance=2.0)

    def test_symmetric_competing_rails_never_create_unsupported_midpoint(self):
        edges = numpy.zeros((260, 300), dtype=numpy.uint8)
        outer = _corners(((74, 54), (206, 54), (206, 186), (74, 186)))
        inner = _corners(((86, 66), (194, 66), (194, 174), (86, 174)))
        _draw_head(edges, outer)
        _draw_head(edges, inner)

        result = refine_projected_head_border(
            cv2, edges, self.predicted, corridor_half_width_px=8.0,
        )

        # Either actual rail is plausible from this image alone. Rejection is
        # also valid; averaging the equally distant rails is not a measurement.
        if result.accepted:
            self.assertIsNotNone(result.corners)
            for actual, outer_corner, inner_corner in zip(result.corners, outer, inner):
                self.assertLessEqual(min(
                    abs(actual.u_px - outer_corner.u_px),
                    abs(actual.u_px - inner_corner.u_px),
                ), 1.75)
                self.assertLessEqual(min(
                    abs(actual.v_px - outer_corner.v_px),
                    abs(actual.v_px - inner_corner.v_px),
                ), 1.75)
        else:
            self.assertIsNone(result.corners)

    def test_head_outside_requested_corridor_remains_unavailable(self):
        for corridor, displacement in ((4.0, 5.0), (8.0, 9.0)):
            with self.subTest(corridor=corridor):
                edges = numpy.zeros((260, 300), dtype=numpy.uint8)
                outside = _corners(tuple(
                    (point.u_px + displacement, point.v_px)
                    for point in self.predicted
                ))
                _draw_head(edges, outside)

                result = refine_projected_head_border(
                    cv2, edges, self.predicted, corridor_half_width_px=corridor,
                )

                self.assertFalse(result.accepted)
                self.assertIsNone(result.corners)

    def test_disconnected_rails_cannot_invent_unsupported_corner_intersections(self):
        edges = numpy.zeros((260, 300), dtype=numpy.uint8)
        for start, end in zip(self.predicted, self.predicted[1:] + self.predicted[:1]):
            length = math.hypot(end.u_px - start.u_px, end.v_px - start.v_px)
            _draw_segment(edges, start, end, first=15.0 / length, last=1.0 - 15.0 / length)

        result = refine_projected_head_border(
            cv2, edges, self.predicted, corridor_half_width_px=8.0,
        )

        # Each side has a long, straight run, but both incident rails stop
        # 15 px before every proposed corner. Infinite-line intersections
        # must not fabricate the missing physical head-corner evidence.
        self.assertFalse(result.accepted)
        self.assertIsNone(result.corners)
        self.assertEqual(result.reason, "model_corner_evidence_insufficient")

    def test_small_corner_gaps_preserve_real_head_measurement(self):
        for predicted in (
            self.predicted,
            _corners(((90, 50), (210, 70), (186, 180), (63, 174))),
        ):
            with self.subTest(predicted=predicted):
                edges = numpy.zeros((260, 300), dtype=numpy.uint8)
                for start, end in zip(predicted, predicted[1:] + predicted[:1]):
                    length = math.hypot(end.u_px - start.u_px, end.v_px - start.v_px)
                    _draw_segment(
                        edges, start, end, first=3.0 / length, last=1.0 - 3.0 / length,
                    )

                result = refine_projected_head_border(
                    cv2, edges, predicted, corridor_half_width_px=8.0,
                )

                # Canny can leave a few pixels absent around rounded corners.
                # Both nearby incident rails remain independent measurements.
                self.assertObservedCorners(result, predicted, tolerance=2.0)


if __name__ == "__main__":
    unittest.main()
