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
from scripts.aufgabe04.perception.stand_axis import head_candidates
from scripts.aufgabe04.perception.stand_axis_image import ImagePoint
from scripts.aufgabe04.perception.stand_axis.head_candidates import (
    _head_first_face_from_edges,
    _short_centered_neck_support,
)
from scripts.aufgabe04.perception.stand_axis_image import (
    _SilhouetteFaceCandidate,
    _attach_structure_evidence,
    _level_camera_endpoint_perspective_consistent,
    _raw_side_evidence_and_corners,
    _validated_refitted_head_corners,
    order_corners,
)
from scripts.aufgabe04.perception.stand_structure_hypothesis import (
    StandStructureEvidence,
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

    def test_quadratic_head_gate_rejects_head_plus_stem_rectangle(self):
        square = tuple(
            ImagePoint(float(u), float(v))
            for u, v in ((20, 20), (120, 22), (118, 122), (22, 120))
        )
        elongated = tuple(
            ImagePoint(float(u), float(v))
            for u, v in ((20, 20), (120, 22), (118, 198), (22, 196))
        )

        self.assertIsNotNone(
            _validated_refitted_head_corners(
                square,
                square,
                image_shape=(240, 180),
                stem_center_x=70.0,
                stem_top_y=124.0,
                min_aspect_ratio=0.45,
                max_aspect_ratio=1.8,
            )
        )
        self.assertIsNone(
            _validated_refitted_head_corners(
                elongated,
                elongated,
                image_shape=(240, 180),
                stem_center_x=70.0,
                stem_top_y=204.0,
                min_aspect_ratio=0.45,
                max_aspect_ratio=1.8,
            )
        )

    def test_structure_evidence_cannot_promote_or_replace_head_corners(self):
        original = (
            ImagePoint(20.0, 20.0),
            ImagePoint(120.0, 22.0),
            ImagePoint(118.0, 122.0),
            ImagePoint(22.0, 120.0),
        )
        recovered = ((1.0, 1.0), (200.0, 1.0), (200.0, 220.0), (1.0, 220.0))
        evidence = StandStructureEvidence(
            accepted=True,
            tracking_supported=True,
            reason="structure_owned_head_supported",
            head_top_support=1.0,
            head_left_support=1.0,
            head_right_support=1.0,
            stem_left_support=1.0,
            stem_right_support=1.0,
            stem_span_px=80.0,
            base_support=1.0,
            base_span_px=180.0,
            base_center_offset_px=0.0,
            corners=recovered,
        )
        candidate = _SilhouetteFaceCandidate(
            corners=original,
            face_mask=object(),
            rectangle_fit_reliable=False,
        )
        with patch(
            "scripts.aufgabe04.perception.stand_axis.stem_candidates.evaluate_stand_structure",
            return_value=evidence,
        ):
            attached = _attach_structure_evidence(
                object(),
                candidate,
                measurement_edges=object(),
                stem_center_x=70.0,
                stem_top_y=124.0,
                min_aspect_ratio=0.45,
                max_aspect_ratio=1.8,
            )
        self.assertEqual(attached.corners, original)
        self.assertFalse(attached.rectangle_fit_reliable)
        self.assertIs(attached.face_mask, candidate.face_mask)

    @unittest.skipIf(
        cv2 is None or numpy is None,
        "numpy and OpenCV are required for raw-side fitting",
    )
    def test_head_first_candidate_uses_outer_square_with_split_bottom(self):
        edges = numpy.zeros((220, 260), dtype=numpy.uint8)
        # Outer head: the lower border is split by the centred stem exactly as
        # it is in the real camera recording.
        cv2.line(edges, (70, 30), (160, 30), 255, 2)
        cv2.line(edges, (70, 30), (70, 120), 255, 2)
        cv2.line(edges, (160, 30), (160, 120), 255, 2)
        cv2.line(edges, (70, 120), (104, 120), 255, 2)
        cv2.line(edges, (126, 120), (160, 120), 255, 2)
        cv2.line(edges, (110, 121), (110, 180), 255, 2)
        cv2.line(edges, (120, 121), (120, 180), 255, 2)
        # QR-like interior clutter and a nearby radiator-like rectangle must
        # not become the fitted head.
        cv2.rectangle(edges, (92, 52), (132, 96), 255, 1)
        cv2.rectangle(edges, (180, 55), (245, 130), 255, 2)

        candidate = _head_first_face_from_edges(
            cv2,
            edges,
            min_edge_height_px=8.0,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.8,
            fixed_parallel_side_direction=(0.0, 1.0),
        )

        self.assertIsNotNone(candidate)
        self.assertTrue(candidate.rectangle_fit_reliable)
        xs = [point.u_px for point in candidate.corners]
        ys = [point.v_px for point in candidate.corners]
        self.assertAlmostEqual(min(xs), 70.0, delta=4.0)
        self.assertAlmostEqual(max(xs), 160.0, delta=4.0)
        self.assertAlmostEqual(min(ys), 30.0, delta=4.0)
        self.assertAlmostEqual(max(ys), 120.0, delta=4.0)

    @unittest.skipIf(
        cv2 is None or numpy is None,
        "numpy and OpenCV are required for raw-side fitting",
    )
    def test_side_first_candidate_learns_rolled_parallel_rails(self):
        edges = numpy.zeros((220, 260), dtype=numpy.uint8)
        # The two outer rails have the same non-vertical image direction;
        # top and bottom are independently sloped by perspective.
        outer = numpy.array(
            [[(60, 30), (150, 42), (158, 132), (68, 120)]],
            dtype=numpy.int32,
        )
        cv2.polylines(edges, outer, True, 255, thickness=2)
        cv2.line(edges, (106, 121), (111, 180), 255, 2)
        cv2.line(edges, (121, 123), (126, 180), 255, 2)
        only_parallel_rails = numpy.array(
            [[[60, 30, 68, 120]], [[150, 42, 158, 132]]],
            dtype=numpy.int32,
        )

        with patch.object(cv2, "HoughLinesP", return_value=only_parallel_rails):
            candidate = _head_first_face_from_edges(
                cv2,
                edges,
                min_edge_height_px=8.0,
                min_aspect_ratio=0.45,
                max_aspect_ratio=1.8,
                fixed_parallel_side_direction=None,
            )

        self.assertIsNotNone(candidate)
        self.assertAlmostEqual(min(point.u_px for point in candidate.corners), 60.0, delta=4.0)
        self.assertAlmostEqual(max(point.u_px for point in candidate.corners), 158.0, delta=4.0)

    @unittest.skipIf(
        cv2 is None or numpy is None,
        "numpy and OpenCV are required for raw-side fitting",
    )
    def test_head_first_bounds_raw_verification_under_dense_hough_clutter(self):
        edges = numpy.zeros((240, 320), dtype=numpy.uint8)
        edges[0, 0] = 255
        # Deliberately provide many plausible rail and horizontal proposals,
        # similar to a radiator.  The bounded proposal stage must not invoke
        # the expensive raw four-side verifier for every pair.
        lines = []
        for x in range(20, 300, 14):
            lines.append([[x, 36, x, 156]])
        for y in range(30, 170, 14):
            lines.append([[70, y, 250, y]])
        dense_lines = numpy.array(lines, dtype=numpy.int32)
        with (
            patch.object(cv2, "HoughLinesP", return_value=dense_lines),
            patch.object(
                head_candidates,
                "_head_candidate_from_rough_corners",
                return_value=None,
            ) as verifier,
        ):
            candidate = _head_first_face_from_edges(
                cv2,
                edges,
                min_edge_height_px=8.0,
                min_aspect_ratio=0.45,
                max_aspect_ratio=1.8,
                fixed_parallel_side_direction=None,
            )

        self.assertIsNone(candidate)
        self.assertLessEqual(
            verifier.call_count,
            head_candidates._MAX_SIDE_FIRST_RAW_VERIFICATIONS
            + head_candidates._MAX_HORIZONTAL_RAW_VERIFICATIONS,
        )

    @unittest.skipIf(
        cv2 is None or numpy is None,
        "numpy and OpenCV are required for raw-side fitting",
    )
    def test_neck_validator_requires_two_post_rails(self):
        edges = numpy.zeros((200, 200), dtype=numpy.uint8)
        corners = tuple(
            ImagePoint(float(u), float(v))
            for u, v in ((60, 30), (140, 30), (140, 110), (60, 110))
        )
        # A single line/QR fragment below the head is insufficient.
        cv2.line(edges, (100, 111), (100, 150), 255, 2)
        self.assertFalse(_short_centered_neck_support(edges, corners))
        cv2.line(edges, (113, 111), (113, 150), 255, 2)
        self.assertTrue(_short_centered_neck_support(edges, corners))

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
