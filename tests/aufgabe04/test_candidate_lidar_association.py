import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.perception.candidate_lidar_association import (  # noqa: E402
    associate_camera_registered_candidate_lidar_target,
    associate_candidate_lidar_target,
)
from scripts.aufgabe04.perception.stand_axis_lidar_roi import (  # noqa: E402
    PlainLaserScan,
    median_range_in_scan_cone,
)


class CandidateLidarAssociationTest(unittest.TestCase):
    def make_scan(
        self,
        ranges,
        *,
        angle_min_deg=-2.0,
        angle_increment_deg=2.0,
        receipt_sec=10.0,
    ) -> PlainLaserScan:
        return PlainLaserScan(
            ranges=tuple(ranges),
            angle_min=math.radians(angle_min_deg),
            angle_increment=math.radians(angle_increment_deg),
            range_min=0.10,
            range_max=3.50,
            scan_frame_id="base_scan",
            scan_stamp_sec=9.9,
            receipt_sec=receipt_sec,
        )

    def associate(self, scan, **overrides):
        arguments = {
            "map_bearing_rad": 0.0,
            "cone_half_angle_rad": math.radians(3.0),
            "accepted_range_m": (0.5249411022680418, 0.7449411022680419),
            "now_sec": 10.1,
            "max_scan_age_sec": 1.0,
        }
        arguments.update(overrides)
        return associate_candidate_lidar_target(scan, **arguments)

    def associate_registered(self, scan, **overrides):
        arguments = {
            "map_bearing_rad": 0.0,
            "observed_camera_bearing_rad": math.radians(9.2),
            "cone_half_angle_rad": math.radians(3.0),
            "accepted_range_m": (0.5249411022680418, 0.7449411022680419),
            "now_sec": 10.1,
            "max_scan_age_sec": 1.0,
        }
        arguments.update(overrides)
        return associate_camera_registered_candidate_lidar_target(
            scan,
            **arguments,
        )

    def test_regression_range_filters_0726_stand_before_0959_background_aggregation(self):
        scan = self.make_scan((0.726, 0.959))

        old_cone_query = median_range_in_scan_cone(
            scan,
            bearing_rad=0.0,
            cone_half_angle_rad=math.radians(3.0),
        )
        association = self.associate(scan)

        self.assertAlmostEqual(old_cone_query.distance_m, 0.8425)
        self.assertGreater(old_cone_query.distance_m, association.accepted_range_m[1])
        self.assertTrue(association.associated)
        self.assertAlmostEqual(association.distance_m, 0.726)
        self.assertEqual(association.cone_valid_sample_count, 2)
        self.assertEqual(association.in_range_sample_count, 1)
        self.assertEqual(association.selected_cluster_sample_count, 1)
        self.assertAlmostEqual(association.nearest_range_delta_m, 0.0)

    def test_contiguous_cluster_minimum_does_not_merge_across_background_return(self):
        scan = self.make_scan(
            (0.70, 0.72, 0.95, 0.69),
            angle_min_deg=-3.0,
            angle_increment_deg=2.0,
        )

        association = self.associate(
            scan,
            cone_half_angle_rad=math.radians(4.0),
            min_cluster_sample_count=2,
        )

        self.assertTrue(association.associated)
        self.assertEqual(association.in_range_sample_count, 3)
        self.assertEqual(association.candidate_cluster_count, 2)
        self.assertEqual(association.eligible_cluster_count, 1)
        self.assertEqual(association.selected_cluster_sample_count, 2)
        self.assertEqual(association.selected_cluster_start_index, 0)
        self.assertEqual(association.selected_cluster_end_index, 1)
        self.assertAlmostEqual(association.distance_m, 0.71)

    def test_camera_bearing_scores_clusters_without_expanding_map_cone(self):
        scan = self.make_scan(
            (0.70, 0.95, 0.71),
            angle_min_deg=-2.0,
            angle_increment_deg=2.0,
        )

        map_selected = self.associate(scan)
        camera_selected = self.associate(
            scan,
            observed_camera_bearing_rad=math.radians(2.0),
        )

        self.assertEqual(map_selected.selected_cluster_start_index, 0)
        self.assertEqual(camera_selected.selected_cluster_start_index, 2)
        self.assertEqual(camera_selected.selection_source, "camera_bearing")

    def test_camera_bearing_outside_unchanged_map_cone_fails_closed(self):
        scan = self.make_scan((0.70, 0.95, 0.71))

        association = self.associate(
            scan,
            observed_camera_bearing_rad=math.radians(20.0),
        )

        self.assertFalse(association.associated)
        self.assertEqual(association.rejection_reason, "camera_bearing_outside_map_cone")
        self.assertEqual(association.eligible_cluster_count, 0)

    def test_sample_outside_original_map_cone_is_never_admitted_by_camera_bearing(self):
        scan = self.make_scan(
            (0.95, 0.70),
            angle_min_deg=0.0,
            angle_increment_deg=10.0,
        )

        association = self.associate(
            scan,
            cone_half_angle_rad=math.radians(3.0),
        )

        self.assertFalse(association.associated)
        self.assertEqual(association.rejection_reason, "no_samples_in_accepted_range")
        self.assertEqual(association.cone_valid_sample_count, 1)

    def test_adjacent_indices_split_on_range_discontinuity(self):
        scan = self.make_scan((0.60, 0.70))

        association = self.associate(scan, min_cluster_sample_count=2)

        self.assertFalse(association.associated)
        self.assertEqual(
            association.rejection_reason,
            "no_contiguous_cluster_meets_minimum",
        )
        self.assertEqual(association.candidate_cluster_count, 2)

    def test_adjacent_indices_split_when_polar_points_are_too_far_apart(self):
        scan = self.make_scan(
            (0.70, 0.70),
            angle_min_deg=-2.0,
            angle_increment_deg=4.0,
        )

        association = self.associate(scan, min_cluster_sample_count=2)

        self.assertFalse(association.associated)
        self.assertEqual(
            association.rejection_reason,
            "no_contiguous_cluster_meets_minimum",
        )
        self.assertEqual(association.candidate_cluster_count, 2)

    def test_public_geometry_thresholds_can_admit_a_calibrated_wider_cluster(self):
        scan = self.make_scan(
            (0.60, 0.70),
            angle_min_deg=-1.0,
            angle_increment_deg=2.0,
        )

        association = self.associate(
            scan,
            min_cluster_sample_count=2,
            max_range_jump_m=0.11,
            max_point_gap_m=0.11,
        )

        self.assertTrue(association.associated)
        self.assertEqual(association.selected_cluster_sample_count, 2)
        self.assertAlmostEqual(association.max_range_jump_m, 0.11)
        self.assertAlmostEqual(association.max_point_gap_m, 0.11)

    def test_no_in_range_sample_reports_nearest_range_delta(self):
        scan = self.make_scan((0.959, 1.10))

        association = self.associate(scan)

        self.assertFalse(association.associated)
        self.assertEqual(association.rejection_reason, "no_samples_in_accepted_range")
        self.assertAlmostEqual(association.nearest_cone_distance_m, 0.959)
        self.assertAlmostEqual(
            association.nearest_range_delta_m,
            0.959 - association.accepted_range_m[1],
        )

    def test_singleton_fails_closed_when_two_contiguous_samples_are_required(self):
        association = self.associate(
            self.make_scan((0.726, 0.959)),
            min_cluster_sample_count=2,
        )

        self.assertFalse(association.associated)
        self.assertEqual(
            association.rejection_reason,
            "no_contiguous_cluster_meets_minimum",
        )
        self.assertEqual(association.eligible_cluster_count, 0)

    def test_missing_and_stale_scans_fail_closed(self):
        missing = self.associate(None)
        stale = self.associate(self.make_scan((0.70,), receipt_sec=1.0))

        self.assertFalse(missing.associated)
        self.assertEqual(missing.rejection_reason, "no_scan")
        self.assertFalse(stale.associated)
        self.assertEqual(stale.rejection_reason, "stale_scan")

    def test_invalid_scan_geometry_fails_closed(self):
        scan = PlainLaserScan(
            ranges=(0.70,),
            angle_min=0.0,
            angle_increment=0.0,
            range_min=0.10,
            range_max=3.50,
            scan_frame_id="base_scan",
        )

        association = self.associate(scan)

        self.assertFalse(association.associated)
        self.assertEqual(association.rejection_reason, "invalid_scan_geometry")

    def test_registered_camera_cone_recovers_bounded_nine_degree_shift(self):
        scan = self.make_scan(
            (2.0,) * 25 + (0.70,) + (2.0,) * 5,
            angle_min_deg=-16.0,
            angle_increment_deg=1.0,
        )

        legacy = self.associate(scan)
        registered = self.associate_registered(scan)

        self.assertFalse(legacy.associated)
        self.assertEqual(legacy.rejection_reason, "no_samples_in_accepted_range")
        self.assertTrue(registered.associated)
        self.assertAlmostEqual(registered.distance_m, 0.70)
        self.assertAlmostEqual(registered.map_bearing_rad, 0.0)
        self.assertAlmostEqual(
            registered.registered_search_bearing_rad,
            math.radians(9.2),
        )
        self.assertAlmostEqual(
            registered.camera_map_bearing_delta_rad,
            math.radians(9.2),
        )
        self.assertAlmostEqual(
            registered.max_camera_map_bearing_delta_rad,
            math.radians(12.0),
        )
        self.assertEqual(
            registered.search_bearing_source,
            "registered_camera_bearing",
        )
        self.assertIsNotNone(registered.search_association)
        self.assertAlmostEqual(
            registered.search_association.map_bearing_rad,
            math.radians(9.2),
        )
        self.assertEqual(
            registered.search_association.eligible_cluster_count,
            1,
        )

    def test_registered_camera_bearing_beyond_limit_fails_before_scan_search(self):
        registered = self.associate_registered(
            self.make_scan((0.70,)),
            observed_camera_bearing_rad=math.radians(12.01),
        )

        self.assertFalse(registered.associated)
        self.assertEqual(
            registered.rejection_reason,
            "camera_map_bearing_delta_exceeds_limit",
        )
        self.assertIsNone(registered.search_association)

    def test_registered_camera_map_delta_wraps_across_pi(self):
        scan = self.make_scan(
            (0.70,),
            angle_min_deg=-179.0,
            angle_increment_deg=1.0,
        )

        registered = self.associate_registered(
            scan,
            map_bearing_rad=math.radians(179.0),
            observed_camera_bearing_rad=math.radians(-179.0),
        )

        self.assertTrue(registered.associated)
        self.assertAlmostEqual(
            registered.camera_map_bearing_delta_rad,
            math.radians(2.0),
        )

    def test_registered_camera_cone_preserves_candidate_range_gate(self):
        scan = self.make_scan(
            (2.0,) * 25 + (0.90,) + (2.0,) * 5,
            angle_min_deg=-16.0,
            angle_increment_deg=1.0,
        )

        registered = self.associate_registered(scan)

        self.assertFalse(registered.associated)
        self.assertEqual(
            registered.rejection_reason,
            "no_samples_in_accepted_range",
        )
        self.assertIsNotNone(registered.search_association)
        self.assertAlmostEqual(
            registered.search_association.nearest_range_delta_m,
            0.90 - registered.search_association.accepted_range_m[1],
        )

    def test_registered_camera_cone_preserves_scan_freshness_gate(self):
        scan = self.make_scan(
            (2.0,) * 25 + (0.70,) + (2.0,) * 5,
            angle_min_deg=-16.0,
            angle_increment_deg=1.0,
            receipt_sec=1.0,
        )

        registered = self.associate_registered(scan)

        self.assertFalse(registered.associated)
        self.assertEqual(registered.rejection_reason, "stale_scan")
        self.assertIsNotNone(registered.search_association)
        self.assertEqual(
            registered.search_association.rejection_reason,
            "stale_scan",
        )
        self.assertAlmostEqual(registered.search_association.scan_age_sec, 9.1)

    def test_registered_camera_cone_rejects_multiple_eligible_clusters(self):
        ranges = [2.0] * 31
        ranges[24] = 0.68
        ranges[26] = 0.70
        scan = self.make_scan(
            ranges,
            angle_min_deg=-16.0,
            angle_increment_deg=1.0,
        )

        registered = self.associate_registered(scan)

        self.assertFalse(registered.associated)
        self.assertEqual(
            registered.rejection_reason,
            "ambiguous_registered_camera_clusters",
        )
        self.assertIsNotNone(registered.search_association)
        self.assertFalse(registered.search_association.associated)
        self.assertEqual(
            registered.search_association.eligible_cluster_count,
            2,
        )
        self.assertEqual(
            registered.search_association.selected_cluster_sample_count,
            0,
        )

    def test_registered_camera_inputs_must_be_finite(self):
        scan = self.make_scan((0.70,))

        with self.assertRaisesRegex(ValueError, "observed_camera_bearing_rad"):
            self.associate_registered(
                scan,
                observed_camera_bearing_rad=float("nan"),
            )
        with self.assertRaisesRegex(ValueError, "max_camera_map_bearing_delta_rad"):
            self.associate_registered(
                scan,
                max_camera_map_bearing_delta_rad=float("inf"),
            )
        with self.assertRaisesRegex(ValueError, "certified 12 degree"):
            self.associate_registered(
                scan,
                max_camera_map_bearing_delta_rad=math.radians(12.01),
            )
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            self.associate_registered(
                scan,
                max_camera_map_bearing_delta_rad=-0.001,
            )

    def test_registered_camera_exact_certified_delta_and_limit_are_admitted(self):
        ranges = [2.0] * 37
        ranges[28] = 0.70
        scan = self.make_scan(
            ranges,
            angle_min_deg=-16.0,
            angle_increment_deg=1.0,
        )

        registered = self.associate_registered(
            scan,
            observed_camera_bearing_rad=math.radians(12.0),
            max_camera_map_bearing_delta_rad=math.radians(12.0),
        )

        self.assertTrue(registered.associated)
        self.assertEqual(registered.search_association.eligible_cluster_count, 1)

    def test_registered_camera_default_degree_round_trip_keeps_certified_limit(self):
        ranges = [2.0] * 37
        ranges[28] = 0.70
        scan = self.make_scan(
            ranges,
            angle_min_deg=-16.0,
            angle_increment_deg=1.0,
        )
        round_tripped_limit = math.radians(math.degrees(math.radians(12.0)))

        registered = self.associate_registered(
            scan,
            observed_camera_bearing_rad=math.radians(12.0),
            max_camera_map_bearing_delta_rad=round_tripped_limit,
        )

        self.assertTrue(registered.associated)
        self.assertAlmostEqual(
            registered.max_camera_map_bearing_delta_rad,
            math.radians(12.0),
        )


if __name__ == "__main__":
    unittest.main()
