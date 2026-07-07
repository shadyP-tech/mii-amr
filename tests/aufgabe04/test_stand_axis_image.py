import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

try:
    import cv2
    import numpy
except ImportError:  # pragma: no cover - exercised only when local deps are missing
    cv2 = None
    numpy = None


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.perception.debug.stand_axis_viewer import build_parser  # noqa: E402
from scripts.aufgabe04.perception.stand_axis_image import (  # noqa: E402
    ImagePoint,
    _connected_border_mask_and_corners,
    _edge_pixels_inside_polygon,
    _expanded_head_edge_roi,
    _face_quadrilateral_from_silhouette,
    _plain_face_from_stem_cropped_edges,
    _scale_quadrilateral_about_center,
    _stem_anchored_face_from_edges,
    estimate_edge_on_axis_from_line,
    estimate_stand_axis_from_edges,
    estimate_stand_axis_from_corners,
    order_corners,
    quadrilateral_aspect_ratio,
    wide_row_band,
)


class StandAxisImageTest(unittest.TestCase):
    def make_corners(self, points):
        return tuple(ImagePoint(float(u), float(v)) for u, v in points)

    def test_order_corners_returns_expected_sequence(self):
        corners = order_corners(
            [
                ImagePoint(90, 80),
                ImagePoint(20, 20),
                ImagePoint(25, 85),
                ImagePoint(95, 25),
            ]
        )

        self.assertEqual([(point.u_px, point.v_px) for point in corners], [(20, 20), (95, 25), (90, 80), (25, 85)])

    def test_quadrilateral_aspect_ratio_uses_outer_edges(self):
        ratio = quadrilateral_aspect_ratio(
            self.make_corners([(40, 20), (100, 20), (100, 80), (40, 80)])
        )

        self.assertAlmostEqual(ratio, 1.0)

    def test_wide_row_band_selects_square_above_narrow_stem(self):
        row_widths = [4, 80, 82, 81, 79, 22, 20, 19]

        self.assertEqual(wide_row_band(row_widths, width_fraction=0.60), (1, 4))

    def test_scale_quadrilateral_about_center_uses_fixed_dimension_ratio(self):
        corners = _scale_quadrilateral_about_center(
            self.make_corners([(40, 30), (80, 30), (80, 70), (40, 70)]),
            1.5,
        )

        self.assertEqual(
            [(round(point.u_px), round(point.v_px)) for point in corners],
            [(30, 20), (90, 20), (90, 80), (30, 80)],
        )

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_silhouette_face_mask_shows_selected_head_outline_only(self):
        silhouette = numpy.zeros((150, 160), dtype=numpy.uint8)
        cv2.rectangle(silhouette, (30, 20), (110, 100), 255, thickness=cv2.FILLED)
        cv2.rectangle(silhouette, (62, 100), (78, 135), 255, thickness=cv2.FILLED)
        cv2.rectangle(silhouette, (45, 42), (55, 52), 0, thickness=cv2.FILLED)
        cv2.rectangle(silhouette, (82, 65), (92, 75), 0, thickness=cv2.FILLED)

        candidate = _face_quadrilateral_from_silhouette(
            cv2,
            silhouette,
            min_area_px=250.0,
            min_edge_height_px=8.0,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
            face_width_fraction=0.60,
        )

        self.assertIsNotNone(candidate)
        face_mask = candidate.face_mask
        self.assertEqual(face_mask.shape, silhouette.shape)
        self.assertEqual(int(face_mask[20, 30]), 255)
        self.assertEqual(int(face_mask[20, 110]), 255)
        self.assertEqual(int(face_mask[100, 30]), 255)
        self.assertEqual(int(face_mask[100, 110]), 255)
        self.assertEqual(int(face_mask[47, 50]), 0)
        self.assertEqual(int(face_mask[70, 87]), 0)
        self.assertEqual(int(face_mask[120, 70]), 0)
        self.assertEqual(int(face_mask[15, 70]), 0)
        self.assertEqual(candidate.corners, order_corners(candidate.corners))
        self.assertAlmostEqual(quadrilateral_aspect_ratio(candidate.corners), 1.0, delta=0.08)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_silhouette_prefers_stand_shape_over_larger_plain_square(self):
        silhouette = numpy.zeros((170, 260), dtype=numpy.uint8)
        cv2.rectangle(silhouette, (140, 30), (235, 125), 255, thickness=cv2.FILLED)
        cv2.rectangle(silhouette, (30, 25), (105, 100), 255, thickness=cv2.FILLED)
        cv2.rectangle(silhouette, (60, 100), (75, 150), 255, thickness=cv2.FILLED)

        candidate = _face_quadrilateral_from_silhouette(
            cv2,
            silhouette,
            min_area_px=250.0,
            min_edge_height_px=8.0,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
            face_width_fraction=0.60,
        )

        self.assertIsNotNone(candidate)
        x_values = [point.u_px for point in candidate.corners]
        self.assertLess(max(x_values), 120)
        self.assertGreater(min(x_values), 20)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_silhouette_expands_inner_face_to_detached_outer_edge(self):
        edges = numpy.zeros((150, 180), dtype=numpy.uint8)
        cv2.rectangle(edges, (45, 25), (115, 95), 255, thickness=cv2.FILLED)
        cv2.line(edges, (135, 22), (137, 98), 255, thickness=2)
        cv2.line(edges, (78, 95), (78, 135), 255, thickness=3)

        candidate = _face_quadrilateral_from_silhouette(
            cv2,
            edges,
            min_area_px=250.0,
            min_edge_height_px=8.0,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
            face_width_fraction=0.60,
        )

        self.assertIsNotNone(candidate)
        self.assertGreater(max(point.u_px for point in candidate.corners), 130)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_stem_anchored_face_uses_outer_box_around_qr_clutter(self):
        edges = numpy.zeros((180, 220), dtype=numpy.uint8)
        cv2.rectangle(edges, (55, 25), (150, 120), 255, thickness=2)
        cv2.rectangle(edges, (70, 42), (132, 105), 255, thickness=2)
        cv2.line(edges, (85, 55), (128, 92), 255, thickness=2)
        cv2.line(edges, (95, 120), (95, 165), 255, thickness=2)
        cv2.line(edges, (110, 120), (110, 165), 255, thickness=2)

        candidate = _stem_anchored_face_from_edges(
            cv2,
            edges,
            min_area_px=250.0,
            min_edge_height_px=8.0,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
        )

        self.assertIsNotNone(candidate)
        x_values = [point.u_px for point in candidate.corners]
        y_values = [point.v_px for point in candidate.corners]
        self.assertLessEqual(min(x_values), 58)
        self.assertGreaterEqual(max(x_values), 147)
        self.assertLessEqual(min(y_values), 28)
        self.assertGreaterEqual(max(y_values), 117)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_stem_anchored_face_preserves_slanted_outer_silhouette(self):
        edges = numpy.zeros((190, 230), dtype=numpy.uint8)
        outer = numpy.array([[(60, 30), (160, 20), (153, 122), (65, 115)]], dtype=numpy.int32)
        cv2.polylines(edges, outer, True, 255, thickness=2)
        cv2.rectangle(edges, (80, 45), (138, 104), 255, thickness=2)
        cv2.line(edges, (90, 58), (134, 92), 255, thickness=2)
        cv2.line(edges, (101, 118), (101, 172), 255, thickness=2)
        cv2.line(edges, (116, 118), (116, 172), 255, thickness=2)

        candidate = _stem_anchored_face_from_edges(
            cv2,
            edges,
            min_area_px=250.0,
            min_edge_height_px=8.0,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
        )

        self.assertIsNotNone(candidate)
        top_left, top_right, bottom_right, bottom_left = candidate.corners
        self.assertLess(top_left.u_px, bottom_left.u_px)
        self.assertGreater(top_right.u_px, bottom_right.u_px)
        self.assertLessEqual(top_left.u_px, 64)
        self.assertGreaterEqual(top_right.u_px, 156)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_plain_face_from_stem_ignores_internal_label(self):
        edges = numpy.zeros((190, 230), dtype=numpy.uint8)
        cv2.polylines(
            edges,
            numpy.array([[(62, 35), (158, 28), (158, 122), (58, 118)]], dtype=numpy.int32),
            True,
            255,
            thickness=2,
        )
        cv2.rectangle(edges, (95, 65), (125, 75), 255, thickness=2)
        cv2.line(edges, (100, 120), (100, 175), 255, thickness=2)
        cv2.line(edges, (115, 120), (115, 175), 255, thickness=2)

        candidate = _plain_face_from_stem_cropped_edges(
            cv2,
            edges,
            min_area_px=250.0,
            min_edge_height_px=8.0,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
        )

        self.assertIsNotNone(candidate)
        x_values = [point.u_px for point in candidate.corners]
        y_values = [point.v_px for point in candidate.corners]
        self.assertLessEqual(min(x_values), 65)
        self.assertGreaterEqual(max(x_values), 155)
        self.assertLessEqual(min(y_values), 38)
        self.assertGreaterEqual(max(y_values), 116)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_plain_face_from_stem_uses_backside_outer_head_contour(self):
        edges = numpy.zeros((210, 260), dtype=numpy.uint8)
        outer = numpy.array([[(84, 36), (186, 50), (176, 146), (96, 132)]], dtype=numpy.int32)
        cv2.polylines(edges, outer, True, 255, thickness=2)
        cv2.rectangle(edges, (122, 82), (154, 94), 255, thickness=2)
        cv2.line(edges, (126, 137), (126, 196), 255, thickness=2)
        cv2.line(edges, (143, 140), (143, 196), 255, thickness=2)

        candidate = _plain_face_from_stem_cropped_edges(
            cv2,
            edges,
            min_area_px=250.0,
            min_edge_height_px=8.0,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
        )

        self.assertIsNotNone(candidate)
        top_left, top_right, bottom_right, bottom_left = candidate.corners
        self.assertLessEqual(top_left.u_px, 88)
        self.assertGreaterEqual(top_right.u_px, 182)
        self.assertLessEqual(min(point.u_px for point in candidate.corners), 88)
        self.assertGreaterEqual(max(point.u_px for point in candidate.corners), 182)
        self.assertLessEqual(min(point.v_px for point in candidate.corners), 38)
        self.assertGreaterEqual(max(point.v_px for point in candidate.corners), 140)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_connected_border_uses_outer_points_not_internal_edges(self):
        cutout = numpy.zeros((180, 220), dtype=numpy.uint8)
        outer = numpy.array([[(62, 30), (162, 24), (154, 124), (70, 118)]], dtype=numpy.int32)
        cv2.polylines(cutout, outer, True, 255, thickness=2)
        cv2.line(cutout, (82, 45), (145, 108), 255, thickness=2)
        cv2.rectangle(cutout, (95, 65), (126, 78), 255, thickness=2)

        face_mask, corners = _connected_border_mask_and_corners(
            cv2,
            cutout,
            cutout,
            fallback_corners=self.make_corners([(62, 30), (162, 24), (154, 124), (70, 118)]),
            min_edge_height_px=8.0,
        )

        top_left, top_right, bottom_right, bottom_left = corners
        self.assertLessEqual(top_left.u_px, 66)
        self.assertGreaterEqual(top_right.u_px, 158)
        self.assertLessEqual(bottom_left.u_px, 72)
        self.assertGreaterEqual(max(point.u_px for point in corners), 158)
        self.assertEqual(face_mask.shape, cutout.shape)
        self.assertEqual(int(face_mask[66, 96]), 255)
        self.assertEqual(int(face_mask[30, 64]), 255)
        self.assertEqual(int(face_mask[10, 64]), 0)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_connected_border_debug_mask_keeps_visible_cuboid_edge(self):
        cutout = numpy.zeros((180, 220), dtype=numpy.uint8)
        cv2.polylines(
            cutout,
            numpy.array([[(70, 42), (152, 36), (160, 124), (76, 126)]], dtype=numpy.int32),
            True,
            255,
            thickness=2,
        )
        cv2.line(cutout, (58, 52), (70, 42), 255, thickness=2)
        cv2.line(cutout, (58, 52), (62, 132), 255, thickness=2)
        cv2.line(cutout, (62, 132), (76, 126), 255, thickness=2)

        face_mask, corners = _connected_border_mask_and_corners(
            cv2,
            cutout,
            cutout,
            fallback_corners=self.make_corners([(70, 42), (152, 36), (160, 124), (76, 126)]),
            min_edge_height_px=8.0,
        )

        self.assertEqual(int(face_mask[52, 58]), 255)
        self.assertEqual(int(face_mask[42, 70]), 255)
        self.assertEqual(face_mask.shape, cutout.shape)
        self.assertEqual(len(corners), 4)
        self.assertLessEqual(min(point.u_px for point in corners), 60)
        self.assertGreaterEqual(max(point.u_px for point in corners), 158)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_connected_border_fits_top_and_bottom_lines_around_stem_notch(self):
        cutout = numpy.zeros((180, 220), dtype=numpy.uint8)
        cv2.line(cutout, (55, 35), (165, 30), 255, thickness=2)
        cv2.line(cutout, (55, 35), (58, 122), 255, thickness=2)
        cv2.line(cutout, (165, 30), (168, 118), 255, thickness=2)
        cv2.line(cutout, (58, 122), (102, 121), 255, thickness=2)
        cv2.line(cutout, (126, 120), (168, 118), 255, thickness=2)
        cv2.line(cutout, (102, 121), (102, 144), 255, thickness=2)
        cv2.line(cutout, (126, 120), (126, 144), 255, thickness=2)

        face_mask, corners = _connected_border_mask_and_corners(
            cv2,
            cutout,
            cutout,
            fallback_corners=self.make_corners([(55, 35), (165, 30), (168, 118), (58, 122)]),
            min_edge_height_px=8.0,
        )

        top_left, top_right, bottom_right, bottom_left = corners
        self.assertLess(top_right.v_px, bottom_right.v_px)
        self.assertLess(top_left.v_px, bottom_left.v_px)
        self.assertLessEqual(abs(bottom_left.v_px - 122), 4)
        self.assertLessEqual(abs(bottom_right.v_px - 118), 4)
        self.assertEqual(int(face_mask[144, 102]), 255)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_edge_estimator_returns_rectangle_debug_mask(self):
        frame = numpy.full((210, 260, 3), 255, dtype=numpy.uint8)
        cv2.rectangle(frame, (84, 36), (186, 146), (0, 0, 120), thickness=cv2.FILLED)
        cv2.rectangle(frame, (122, 82), (154, 94), (240, 240, 240), thickness=cv2.FILLED)
        cv2.rectangle(frame, (126, 137), (143, 196), (0, 0, 120), thickness=cv2.FILLED)

        _estimate, artifacts = estimate_stand_axis_from_edges(
            cv2,
            frame,
            edge_preprocess="gray",
            min_area_px=250.0,
            min_edge_height_px=8.0,
            min_aspect_ratio=0.45,
            max_aspect_ratio=1.80,
        )

        self.assertIsNotNone(artifacts.face_mask)
        self.assertIsNotNone(artifacts.rectangle_mask)
        self.assertEqual(artifacts.rectangle_mask.shape, artifacts.edges.shape)
        self.assertGreater(cv2.countNonZero(artifacts.rectangle_mask), 0)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_edge_pixels_inside_polygon_uses_debug_margin_only(self):
        edges = numpy.zeros((100, 120), dtype=numpy.uint8)
        cv2.rectangle(edges, (30, 25), (90, 75), 255, thickness=1)
        cv2.line(edges, (27, 25), (27, 75), 255, thickness=1)
        corners = self.make_corners([(30, 25), (90, 25), (90, 75), (30, 75)])

        tight = _edge_pixels_inside_polygon(cv2, edges, corners, margin_px=0)
        expanded = _edge_pixels_inside_polygon(cv2, edges, corners, margin_px=4)

        self.assertEqual(int(tight[50, 27]), 0)
        self.assertEqual(int(expanded[50, 27]), 255)
        self.assertEqual(int(expanded[10, 27]), 0)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for silhouette tests")
    def test_expanded_head_roi_keeps_slanted_top_outside_rough_polygon(self):
        edges = numpy.zeros((140, 180), dtype=numpy.uint8)
        cv2.line(edges, (48, 18), (132, 34), 255, thickness=2)
        cv2.line(edges, (48, 18), (52, 105), 255, thickness=2)
        cv2.line(edges, (132, 34), (136, 112), 255, thickness=2)
        cv2.line(edges, (52, 105), (136, 112), 255, thickness=2)
        rough = self.make_corners([(54, 38), (128, 42), (132, 108), (56, 102)])

        edge_roi = _expanded_head_edge_roi(
            cv2,
            edges,
            rough,
            margin_px=10,
            stem_center_x=92.0,
            stem_top_y=108.0,
            min_edge_height_px=8.0,
        )

        self.assertEqual(int(edge_roi[18, 48]), 255)
        self.assertEqual(int(edge_roi[34, 132]), 255)
        self.assertEqual(int(edge_roi[10, 48]), 0)

    def test_front_facing_square_has_neutral_ratio(self):
        estimate = estimate_stand_axis_from_corners(
            self.make_corners([(40, 25), (100, 25), (100, 85), (40, 85)])
        )

        self.assertTrue(estimate.usable)
        self.assertEqual(estimate.mode, "face_visible")
        self.assertAlmostEqual(estimate.height_ratio, 1.0, delta=0.03)
        self.assertAlmostEqual(estimate.yaw_proxy, 0.0, delta=0.02)
        self.assertEqual(estimate.closer_side, "equal")

    def test_taller_left_edge_reports_left_as_closer(self):
        estimate = estimate_stand_axis_from_corners(
            self.make_corners([(35, 20), (105, 35), (100, 80), (35, 95)])
        )

        self.assertTrue(estimate.usable)
        self.assertEqual(estimate.mode, "face_visible")
        self.assertGreater(estimate.left_height_px, estimate.right_height_px)
        self.assertGreater(estimate.height_ratio, 1.0)
        self.assertGreater(estimate.yaw_proxy, 0.0)
        self.assertEqual(estimate.closer_side, "left")

    def test_edge_on_line_reports_side_on_without_ratio(self):
        estimate = estimate_edge_on_axis_from_line(
            ImagePoint(80, 20),
            ImagePoint(82, 92),
            min_edge_height_px=8.0,
        )

        self.assertTrue(estimate.usable)
        self.assertEqual(estimate.mode, "edge_on")
        self.assertEqual(estimate.reason, "edge_on_approx_90_deg")
        self.assertIsNone(estimate.height_ratio)
        self.assertIsNone(estimate.yaw_proxy)
        self.assertIsNone(estimate.yaw_deg)
        self.assertEqual(estimate.closer_side, "side_on")
        self.assertIsNotNone(estimate.axis_line)

    def test_short_edge_on_line_is_not_usable(self):
        estimate = estimate_edge_on_axis_from_line(
            ImagePoint(80, 20),
            ImagePoint(80, 24),
            min_edge_height_px=8.0,
        )

        self.assertFalse(estimate.usable)
        self.assertEqual(estimate.mode, "unavailable")
        self.assertEqual(estimate.reason, "edge_on_line_too_short")

    def test_optional_geometry_converts_ratio_to_yaw_degrees(self):
        estimate = estimate_stand_axis_from_corners(
            self.make_corners([(35, 20), (105, 35), (100, 80), (35, 95)]),
            stand_width_m=1.0,
            stand_distance_m=0.25,
        )

        self.assertTrue(estimate.usable)
        self.assertIsNotNone(estimate.yaw_deg)
        self.assertGreater(estimate.yaw_deg, 0.0)

    def test_calibrated_geometry_uses_lidar_distance_and_known_face_size(self):
        estimate = estimate_stand_axis_from_corners(
            self.make_corners([(40, 20), (79, 20), (79, 98), (40, 98)]),
            stand_width_m=0.078,
            stand_distance_m=1.0,
            camera_fx_px=1000.0,
        )

        self.assertTrue(estimate.usable)
        self.assertAlmostEqual(estimate.height_ratio, 1.0, delta=0.001)
        self.assertIsNotNone(estimate.yaw_deg)
        self.assertAlmostEqual(abs(estimate.yaw_deg), 60.0, delta=0.5)

    @unittest.skipIf(numpy is None or cv2 is None, "numpy and OpenCV are required for PnP tests")
    def test_square_pnp_uses_camera_intrinsics_for_plane_yaw(self):
        face_size_m = 0.078
        object_points = numpy.array(
            [
                [-face_size_m / 2.0, -face_size_m / 2.0, 0.0],
                [face_size_m / 2.0, -face_size_m / 2.0, 0.0],
                [face_size_m / 2.0, face_size_m / 2.0, 0.0],
                [-face_size_m / 2.0, face_size_m / 2.0, 0.0],
            ],
            dtype=numpy.float64,
        )
        camera_matrix = numpy.array([[530.0, 0.0, 320.0], [0.0, 530.0, 240.0], [0.0, 0.0, 1.0]])
        yaw_rad = numpy.deg2rad(25.0)
        rvec, _jacobian = cv2.Rodrigues(
            numpy.array(
                [
                    [numpy.cos(yaw_rad), 0.0, numpy.sin(yaw_rad)],
                    [0.0, 1.0, 0.0],
                    [-numpy.sin(yaw_rad), 0.0, numpy.cos(yaw_rad)],
                ],
                dtype=numpy.float64,
            )
        )
        tvec = numpy.array([[0.0], [0.0], [0.55]], dtype=numpy.float64)
        image_points, _jacobian = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            camera_matrix,
            numpy.zeros((4, 1), dtype=numpy.float64),
        )
        corners = self.make_corners([(float(point[0][0]), float(point[0][1])) for point in image_points])

        estimate = estimate_stand_axis_from_corners(
            corners,
            stand_width_m=face_size_m,
            camera_fx_px=530.0,
            camera_fy_px=530.0,
            camera_cx_px=320.0,
            camera_cy_px=240.0,
            cv2=cv2,
        )

        self.assertTrue(estimate.usable)
        self.assertIsNotNone(estimate.yaw_deg)
        self.assertAlmostEqual(abs(estimate.yaw_deg), 25.0, delta=1.0)

    def test_stand_axis_viewer_requires_ros_compressed_image_topic(self):
        parser = build_parser()

        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args([])

        args = parser.parse_args(["--compressed-image-topic", "/camera/image_raw/compressed"])
        self.assertEqual(args.compressed_image_topic, "/camera/image_raw/compressed")
        self.assertEqual(args.axis_source, "edges")

    def test_stand_axis_viewer_has_no_motion_arguments(self):
        parser = build_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }

        self.assertNotIn("--cmd-vel-topic", option_strings)
        self.assertNotIn("--run", option_strings)
        self.assertNotIn("--nav2-goal", option_strings)

    def test_stand_axis_viewer_exposes_edge_debug_options(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "--compressed-image-topic",
                "/camera/image_raw/compressed",
                "--axis-source",
                "edges",
                "--display-edges",
                "--canny-low",
                "40",
                "--canny-high",
                "120",
                "--edge-preprocess",
                "outer-border",
                "--hough-threshold",
                "18",
                "--hough-min-line-length-px",
                "10",
                "--hough-max-line-gap-px",
                "6",
                "--min-boundary-line-length-px",
                "42",
                "--face-width-fraction",
                "0.65",
                "--min-face-area-fraction",
                "0.35",
                "--display-face-mask",
            ]
        )

        self.assertEqual(args.axis_source, "edges")
        self.assertTrue(args.display_edges)
        self.assertTrue(args.display_face_mask)
        self.assertFalse(args.display_mask)
        self.assertEqual(args.canny_low, 40)
        self.assertEqual(args.canny_high, 120)
        self.assertEqual(args.edge_preprocess, "outer-border")
        self.assertEqual(args.hough_threshold, 18)
        self.assertEqual(args.hough_min_line_length_px, 10)
        self.assertEqual(args.hough_max_line_gap_px, 6)
        self.assertEqual(args.min_boundary_line_length_px, 42)
        self.assertAlmostEqual(args.face_width_fraction, 0.65)
        self.assertAlmostEqual(args.min_face_area_fraction, 0.35)

    def test_stand_axis_viewer_exposes_nonblocking_qr_options(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "--compressed-image-topic",
                "/camera/image_raw/compressed",
                "--qr-decode-fps",
                "1.5",
                "--qr-result-ttl-sec",
                "0.75",
                "--front-face-to-qr-width-ratio",
                "1.2",
                "--stand-face-size-m",
                "0.078",
                "--camera-fx-px",
                "530.0",
                "--camera-fy-px",
                "531.0",
                "--camera-cx-px",
                "320.0",
                "--camera-cy-px",
                "240.0",
                "--camera-fx-is-full-resolution",
                "--stand-head-depth-m",
                "0.007",
                "--stand-head-bottom-height-m",
                "0.135",
                "--scan-topic",
                "/scan",
                "--use-lidar-distance",
                "--lidar-bearing-rad",
                "0.1",
                "--lidar-cone-deg",
                "7.5",
                "--max-scan-age-sec",
                "0.8",
                "--no-qr-decode",
            ]
        )

        self.assertAlmostEqual(args.qr_decode_fps, 1.5)
        self.assertAlmostEqual(args.qr_result_ttl_sec, 0.75)
        self.assertAlmostEqual(args.front_face_to_qr_width_ratio, 1.2)
        self.assertAlmostEqual(args.stand_face_size_m, 0.078)
        self.assertAlmostEqual(args.camera_fx_px, 530.0)
        self.assertAlmostEqual(args.camera_fy_px, 531.0)
        self.assertAlmostEqual(args.camera_cx_px, 320.0)
        self.assertAlmostEqual(args.camera_cy_px, 240.0)
        self.assertTrue(args.camera_fx_is_full_resolution)
        self.assertAlmostEqual(args.stand_head_depth_m, 0.007)
        self.assertAlmostEqual(args.stand_head_bottom_height_m, 0.135)
        self.assertEqual(args.scan_topic, "/scan")
        self.assertTrue(args.use_lidar_distance)
        self.assertAlmostEqual(args.lidar_bearing_rad, 0.1)
        self.assertAlmostEqual(args.lidar_cone_deg, 7.5)
        self.assertAlmostEqual(args.max_scan_age_sec, 0.8)
        self.assertTrue(args.no_qr_decode)


if __name__ == "__main__":
    unittest.main()
