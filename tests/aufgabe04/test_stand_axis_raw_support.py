from __future__ import annotations

import math
import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy
except ImportError:  # pragma: no cover - optional outside the project environment
    cv2 = None
    numpy = None

from scripts.aufgabe04.perception import stand_axis_image
from scripts.aufgabe04.perception.stand_axis_image import ImagePoint
from scripts.aufgabe04.perception.stand_axis_image import (
    _SilhouetteFaceCandidate,
    _level_camera_endpoint_perspective_consistent,
    _raw_side_evidence_and_corners,
    order_corners,
)


class _FakeImage:
    def __init__(self, shape):
        self.shape = shape

    def copy(self):
        return self


class StandAxisRawSupportTest(unittest.TestCase):
    @staticmethod
    def _rotate_quarter_turn(point: ImagePoint) -> ImagePoint:
        return ImagePoint(500.0 - point.v_px, 20.0 + point.u_px)

    def test_endpoint_check_uses_supplied_parallel_side_direction(self):
        accepted = (
            ImagePoint(297.0, 126.5),
            ImagePoint(297.0, 204.5),
            ImagePoint(339.0, 142.5),
            ImagePoint(339.0, 208.5),
        )
        rejected = (
            ImagePoint(297.0, 127.5),
            ImagePoint(297.0, 180.5),
            ImagePoint(339.0, 142.5),
            ImagePoint(339.0, 209.5),
        )

        self.assertTrue(
            _level_camera_endpoint_perspective_consistent(*accepted)
        )
        self.assertFalse(
            _level_camera_endpoint_perspective_consistent(*rejected)
        )

        rotated_accepted = tuple(
            self._rotate_quarter_turn(point) for point in accepted
        )
        rotated_rejected = tuple(
            self._rotate_quarter_turn(point) for point in rejected
        )
        for direction in ((-1.0, 0.0), (1.0, 0.0)):
            with self.subTest(direction=direction):
                self.assertTrue(
                    _level_camera_endpoint_perspective_consistent(
                        *rotated_accepted,
                        parallel_side_direction=direction,
                    )
                )
                self.assertFalse(
                    _level_camera_endpoint_perspective_consistent(
                        *rotated_rejected,
                        parallel_side_direction=direction,
                    )
                )

    def test_endpoint_check_rejects_invalid_directions(self):
        points = (
            ImagePoint(10.0, 10.0),
            ImagePoint(10.0, 30.0),
            ImagePoint(30.0, 10.0),
            ImagePoint(30.0, 30.0),
        )

        for direction in ((0.0, 0.0), (math.nan, 1.0)):
            with self.subTest(direction=direction):
                with self.assertRaisesRegex(
                    ValueError,
                    "finite non-zero 2D direction",
                ):
                    _level_camera_endpoint_perspective_consistent(
                        *points,
                        parallel_side_direction=direction,
                    )

    @unittest.skipIf(
        cv2 is None or numpy is None,
        "numpy and OpenCV are required for raw-side fitting",
    )
    def test_fixed_rolled_direction_preserves_observed_trapezoid_dimensions(self):
        center_u, center_v = 120.0, 100.0
        angle_rad = math.radians(24.0)
        cosine = math.cos(angle_rad)
        sine = math.sin(angle_rad)

        def rotate(point: ImagePoint) -> ImagePoint:
            relative_u = point.u_px - center_u
            relative_v = point.v_px - center_v
            return ImagePoint(
                center_u + cosine * relative_u - sine * relative_v,
                center_v + sine * relative_u + cosine * relative_v,
            )

        actual = tuple(
            rotate(point)
            for point in (
                ImagePoint(60.0, 35.0),
                ImagePoint(170.0, 48.0),
                ImagePoint(170.0, 148.0),
                ImagePoint(60.0, 135.0),
            )
        )
        rough = tuple(
            rotate(point)
            for point in (
                ImagePoint(64.0, 39.0),
                ImagePoint(166.0, 50.0),
                ImagePoint(166.0, 144.0),
                ImagePoint(64.0, 132.0),
            )
        )
        raw_edges = numpy.zeros((240, 270), dtype=numpy.uint8)
        polygon = numpy.array(
            [[(round(point.u_px), round(point.v_px)) for point in actual]],
            dtype=numpy.int32,
        )
        cv2.polylines(raw_edges, polygon, True, 255, thickness=2)

        direction = (-sine, cosine)
        evidence_mask, fitted = _raw_side_evidence_and_corners(
            cv2,
            raw_edges,
            rough,
            fixed_parallel_side_direction=direction,
        )

        self.assertIsNotNone(fitted)
        for observed, expected in zip(
            order_corners(fitted),
            order_corners(actual),
        ):
            self.assertAlmostEqual(observed.u_px, expected.u_px, delta=3.0)
            self.assertAlmostEqual(observed.v_px, expected.v_px, delta=3.0)
        fitted_top_left, fitted_top_right, fitted_bottom_right, fitted_bottom_left = (
            order_corners(fitted)
        )
        left_height = math.hypot(
            fitted_bottom_left.u_px - fitted_top_left.u_px,
            fitted_bottom_left.v_px - fitted_top_left.v_px,
        )
        right_height = math.hypot(
            fitted_bottom_right.u_px - fitted_top_right.u_px,
            fitted_bottom_right.v_px - fitted_top_right.v_px,
        )
        self.assertAlmostEqual(left_height, 100.0, delta=4.0)
        self.assertAlmostEqual(right_height, 100.0, delta=4.0)
        self.assertGreater(cv2.countNonZero(evidence_mask), 0)

    def test_silhouette_path_plumbs_direction_without_changing_failure_contract(self):
        frame = _FakeImage((80, 100, 3))
        raw_edges = _FakeImage((80, 100))
        topology_edges = _FakeImage((80, 100))
        face_mask = object()
        corners = (
            ImagePoint(20.0, 10.0),
            ImagePoint(70.0, 12.0),
            ImagePoint(68.0, 55.0),
            ImagePoint(22.0, 54.0),
        )
        candidate = _SilhouetteFaceCandidate(
            corners=corners,
            face_mask=face_mask,
            rectangle_fit_reliable=False,
            rectangle_fit_reason="head_rectangle_fit_unreliable",
        )
        direction = (0.25, 0.97)

        with (
            patch.object(
                stand_axis_image,
                "_canny_edges_from_frame",
                return_value=raw_edges,
            ),
            patch.object(
                stand_axis_image,
                "_topology_edges_from_frame",
                return_value=topology_edges,
            ),
            patch.object(
                stand_axis_image,
                "_edge_topology_hypotheses",
                return_value=[topology_edges],
            ),
            patch.object(
                stand_axis_image,
                "_largest_external_bounding_area",
                return_value=0.0,
            ),
            patch.object(
                stand_axis_image,
                "_plain_face_from_stem_cropped_edges",
                return_value=candidate,
            ) as detector,
        ):
            estimate, artifacts = (
                stand_axis_image.estimate_stand_axis_from_edges(
                    object(),
                    frame,
                    dilate_iterations=0,
                    min_area_px=0.0,
                    min_face_area_fraction=0.0,
                    parallel_side_direction=direction,
                    silhouette_only=True,
                )
            )

        self.assertEqual(
            detector.call_args.kwargs["_fixed_parallel_side_direction"],
            direction,
        )
        self.assertFalse(estimate.usable)
        self.assertEqual(estimate.reason, "head_rectangle_fit_unreliable")
        self.assertEqual(estimate.source, "edge_plain_face_stem_anchor")
        self.assertEqual(estimate.corners, corners)
        self.assertIs(artifacts.edges, topology_edges)
        self.assertIs(artifacts.raw_edges, raw_edges)
        self.assertIs(artifacts.face_mask, face_mask)


if __name__ == "__main__":
    unittest.main()
