import math
import unittest

from scripts.aufgabe04.navigation.coverage.candidate_visibility_rays import (
    select_candidate_visibility_ray,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.lidar_visibility_evidence import (
    lidar_visibility_receipt_from_scan,
)


def _receipt(ranges, *, angle_min=-0.1, angle_increment=0.05):
    return lidar_visibility_receipt_from_scan(
        receipt_id="scan_01",
        survey_id="survey_01",
        viewpoint_id="viewpoint_02",
        planning_frame="map",
        scan_frame="base_scan",
        scan_topic="/scan",
        map_bundle_sha256="a" * 64,
        observer_config_sha256="b" * 64,
        scan_stamp_sec=1.0,
        pose_stamp_sec=1.0,
        observer_clock_sec=1.01,
        scan_pose_map=Pose2D(0.0, 0.0, 0.0),
        angle_min_rad=angle_min,
        angle_increment_rad=angle_increment,
        range_min_m=0.01,
        range_max_m=10.0,
        ranges_m=ranges,
    )


def _select(ranges, *, angle_min=-0.1, angle_increment=0.05, **overrides):
    kwargs = {
        "target_bearing_rad": 0.0,
        "candidate_distance_m": 1.0,
        "envelope_radius_m": 0.08,
        "far_edge_clearance_margin_m": 0.03,
        "matching_range_tolerance_m": 0.02,
    }
    kwargs.update(overrides)
    return select_candidate_visibility_ray(
        _receipt(ranges, angle_min=angle_min, angle_increment=angle_increment),
        **kwargs,
    )


class CandidateVisibilityRaySelectionTests(unittest.TestCase):
    def test_off_center_singleton_vetoes_clear_center(self):
        result = _select((3.0, 1.0, 3.0, 3.0, 3.0))
        self.assertEqual(result.classification, "matching")
        self.assertEqual(result.selected_ray_index, 1)
        self.assertEqual(result.supporting_ray_indices, (1,))
        self.assertEqual(result.intersecting_ray_indices, (1, 2, 3))

    def test_off_center_singleton_vetoes_invalid_center(self):
        result = _select((3.0, 1.0, math.inf, 3.0, 3.0))
        self.assertEqual(result.classification, "matching")
        self.assertEqual(result.selected_ray_index, 1)
        self.assertEqual(result.invalid_ray_indices, (2,))

    def test_off_center_nearer_return_vetoes_clear_center(self):
        result = _select((3.0, 0.5, 3.0, 3.0, 3.0))
        self.assertEqual(result.classification, "nearer")
        self.assertEqual(result.selected_ray_index, 1)
        self.assertEqual(result.occluding_ray_indices, (1,))

    def test_matching_precedes_nearer_return(self):
        result = _select((3.0, 1.0, 0.5, 3.0, 3.0))
        self.assertEqual(result.classification, "matching")
        self.assertEqual(result.selected_ray_index, 1)
        self.assertEqual(result.supporting_ray_indices, (1,))
        self.assertEqual(result.occluding_ray_indices, (2,))

    def test_genuinely_clear_scan_selects_center(self):
        result = _select((3.0,) * 5)
        self.assertEqual(result.classification, "clear")
        self.assertEqual(result.selected_ray_index, 2)
        self.assertEqual(result.supporting_ray_indices, ())
        self.assertEqual(result.occluding_ray_indices, ())

    def test_outside_disk_support_and_occlusion_do_not_veto(self):
        result = _select((1.0, 3.0, 3.0, 3.0, 0.5))
        self.assertEqual(result.classification, "clear")
        self.assertEqual(result.intersecting_ray_indices, (1, 2, 3))

    def test_ray_behind_sensor_does_not_intersect_forward_disk(self):
        result = _select((1.0, 3.0, 1.0), angle_min=-math.pi,
                         angle_increment=math.pi)
        self.assertEqual(result.classification, "clear")
        self.assertEqual(result.intersecting_ray_indices, (1,))

    def test_invalid_neighbor_preserves_center_dropout_policy(self):
        result = _select((3.0, math.inf, 3.0, math.nan, 3.0))
        self.assertEqual(result.classification, "clear")
        self.assertEqual(result.invalid_ray_indices, (1, 3))

    def test_invalid_center_without_finite_support_stays_invalid(self):
        result = _select((3.0, 3.0, math.inf, 3.0, 3.0))
        self.assertEqual(result.classification, "invalid")
        self.assertIsNone(result.selected_range_m)
        self.assertEqual(result.selected_ray_index, 2)

    def test_chord_geometry_distinguishes_nearer_off_axis_return(self):
        # A tangent-side ray enters the disk after 0.99 m.  A 0.95 m
        # return is therefore occluding, even though it exceeds d - r.
        result = _select((0.95, 3.0), angle_min=math.asin(0.0799),
                         angle_increment=0.1)
        self.assertEqual(result.classification, "nearer")
        self.assertEqual(result.occluding_ray_indices, (0,))

    def test_off_axis_background_beyond_chord_does_not_veto_center(self):
        result = _select((3.0, 1.09), angle_min=0.0,
                         angle_increment=math.asin(0.0799))
        self.assertEqual(result.classification, "clear")
        self.assertEqual(result.selected_ray_index, 0)

    def test_center_clearance_retains_conservative_radial_far_edge(self):
        result = _select((1.09,), angle_min=math.asin(0.0799))
        self.assertEqual(result.classification, "matching")
        self.assertEqual(result.supporting_ray_indices, (0,))

    def test_matching_margin_boundary_remains_conservative(self):
        result = _select((1.13,), angle_min=0.0)
        self.assertEqual(result.classification, "matching")
        result = _select((1.13001,), angle_min=0.0)
        self.assertEqual(result.classification, "clear")

    def test_no_ray_intersection_has_no_selected_evidence(self):
        result = _select((3.0, 3.0), angle_min=0.3)
        self.assertEqual(result.classification, "no_intersection")
        self.assertEqual(result.reason, "no_scan_ray_intersects_candidate_envelope")
        self.assertIsNone(result.selected_ray_index)
        self.assertIsNone(result.selected_ray_bearing_rad)
        self.assertIsNone(result.selected_ray_offset_rad)
        self.assertIsNone(result.selected_range_m)
        self.assertEqual(result.intersecting_ray_indices, ())

    def test_wraparound_selects_neighbor_across_pi(self):
        result = _select((1.0, 3.0, 3.0), angle_min=-math.pi - 0.05,
                         target_bearing_rad=math.pi)
        self.assertEqual(result.classification, "matching")
        self.assertEqual(result.selected_ray_index, 0)
        self.assertAlmostEqual(result.selected_ray_offset_rad, -0.05)

    def test_negative_increment_preserves_scan_indices(self):
        result = _select((3.0, 3.0, 3.0, 1.0, 3.0), angle_min=0.1,
                         angle_increment=-0.05)
        self.assertEqual(result.classification, "matching")
        self.assertEqual(result.selected_ray_index, 3)
        self.assertEqual(result.supporting_ray_indices, (3,))

    def test_equally_offset_support_uses_lower_scan_index(self):
        result = _select((1.0, 1.0), angle_min=-0.05, angle_increment=0.1)
        self.assertEqual(result.selected_ray_index, 0)
        self.assertEqual(result.supporting_ray_indices, (0, 1))

    def test_nonfinite_or_negative_geometry_is_rejected(self):
        for name, value in (
            ("target_bearing_rad", math.nan),
            ("candidate_distance_m", math.inf),
            ("envelope_radius_m", -0.1),
            ("far_edge_clearance_margin_m", -0.1),
            ("matching_range_tolerance_m", -0.1),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                _select((3.0,), **{name: value})


if __name__ == "__main__":
    unittest.main()
