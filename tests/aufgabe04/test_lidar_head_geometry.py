import json
import math
from pathlib import Path
import unittest

from scripts.aufgabe04.navigation.approach.lidar_head_geometry import (
    HEAD_MODEL, adjacent_returns, combine_head_surfaces, fit_head_surface,
)


class LidarHeadGeometryTest(unittest.TestCase):
    def fit(self, left=-.039, right=.039, y=-.003):
        points = tuple((left + i * (right - left) / 4, y) for i in range(5))
        fit = fit_head_surface(points, sensor_position=(0., -.6))
        self.assertIsNotNone(fit)
        return fit

    def test_geometry_matches_measured_physical_profile(self):
        path = Path(__file__).parents[2] / "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json"
        profile = json.loads(path.read_text())
        self.assertEqual(HEAD_MODEL.width_m, profile["head_width_m"])
        self.assertEqual(HEAD_MODEL.depth_m, profile["head_depth_m"])
        self.assertEqual(HEAD_MODEL.tolerance_m, profile["tolerance_m"])

    def test_geometry_gap_scales_with_range_without_joining_depth_steps(self):
        increment = math.radians(1.67478)
        self.assertGreater(2 * 1.5 * math.sin(increment / 2), .04)
        self.assertTrue(adjacent_returns(1.5, 1.5, increment))
        self.assertFalse(adjacent_returns(.6, .65, increment))
        self.assertFalse(adjacent_returns(1.5, 1.55, increment))
        self.assertFalse(adjacent_returns(1.5, 1.5, 0.))

    def test_visible_face_offset_recovers_slab_center(self):
        fitted = self.fit()
        self.assertAlmostEqual(fitted.center_x_m, 0.)
        self.assertAlmostEqual(fitted.center_y_m, 0.)
        self.assertGreater(fitted.angle_uncertainty_rad, 0.)
        self.assertGreaterEqual(fitted.center_uncertainty_m, HEAD_MODEL.point_noise_m)

    def test_partial_visibility_preserves_center_ambiguity(self):
        fitted = self.fit(left=-.015)
        self.assertAlmostEqual(fitted.center_x_m, .012)
        self.assertGreaterEqual(fitted.along_uncertainty_m, .012)
        self.assertGreater(fitted.along_uncertainty_m, self.fit().along_uncertainty_m)

    def test_complementary_views_constrain_center_without_scan_count_precision(self):
        a, b = self.fit(right=.011), self.fit(left=-.011)
        combined = combine_head_surfaces((a, b), tangent_rad=0.)
        repeated = combine_head_surfaces((a, b) * 100, tangent_rad=0.)
        self.assertEqual(combined, repeated)
        self.assertAlmostEqual(combined.center_x_m, 0.)
        self.assertLess(combined.along_uncertainty_m, a.along_uncertainty_m)
        self.assertEqual(combined.angle_uncertainty_rad, a.angle_uncertainty_rad)

    def test_inconsistent_face_positions_have_no_joint_center(self):
        self.assertIsNone(combine_head_surfaces((self.fit(), self.fit(y=.03)), tangent_rad=0.))

    def test_sparse_or_base_sized_returns_cannot_fit_the_head(self):
        self.assertIsNone(fit_head_surface(((-.03, 0.), (0., 0.), (.03, 0.)), sensor_position=(0., -.6)))
        self.assertIsNone(fit_head_surface(tuple((i * .03, 0.) for i in range(6)), sensor_position=(0., -.6)))
        self.assertIsNone(fit_head_surface(tuple((i * .005, 0.) for i in range(5)), sensor_position=(0., -.6)))


if __name__ == "__main__":
    unittest.main()
