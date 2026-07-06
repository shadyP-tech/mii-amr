import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.perception.debug.stand_axis_viewer import build_parser  # noqa: E402
from scripts.aufgabe04.perception.stand_axis_image import (  # noqa: E402
    ImagePoint,
    estimate_edge_on_axis_from_line,
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
            ]
        )

        self.assertEqual(args.axis_source, "edges")
        self.assertTrue(args.display_edges)
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
                "--no-qr-decode",
            ]
        )

        self.assertAlmostEqual(args.qr_decode_fps, 1.5)
        self.assertAlmostEqual(args.qr_result_ttl_sec, 0.75)
        self.assertTrue(args.no_qr_decode)


if __name__ == "__main__":
    unittest.main()
