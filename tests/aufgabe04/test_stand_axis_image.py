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
    estimate_stand_axis_from_corners,
    order_corners,
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

    def test_front_facing_square_has_neutral_ratio(self):
        estimate = estimate_stand_axis_from_corners(
            self.make_corners([(40, 25), (100, 25), (100, 85), (40, 85)])
        )

        self.assertTrue(estimate.usable)
        self.assertAlmostEqual(estimate.height_ratio, 1.0, delta=0.03)
        self.assertAlmostEqual(estimate.yaw_proxy, 0.0, delta=0.02)
        self.assertEqual(estimate.closer_side, "equal")

    def test_taller_left_edge_reports_left_as_closer(self):
        estimate = estimate_stand_axis_from_corners(
            self.make_corners([(35, 20), (105, 35), (100, 80), (35, 95)])
        )

        self.assertTrue(estimate.usable)
        self.assertGreater(estimate.left_height_px, estimate.right_height_px)
        self.assertGreater(estimate.height_ratio, 1.0)
        self.assertGreater(estimate.yaw_proxy, 0.0)
        self.assertEqual(estimate.closer_side, "left")

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


if __name__ == "__main__":
    unittest.main()
