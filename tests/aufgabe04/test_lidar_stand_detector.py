import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.perception.lidar_stand_detector import (  # noqa: E402
    detect_stand_candidates,
    detect_stand_candidates_from_scan,
    scan_points_from_ranges,
)
from scripts.aufgabe04.perception.models import LidarStandDetectorConfig  # noqa: E402
from tests.aufgabe04.test_scan_topology import SCAN_GEOMETRIES


def scan_with_returns(return_indices, *, total=91, default_range=5.0, return_range=1.0):
    ranges = [default_range] * total
    for index in return_indices:
        ranges[index] = return_range
    return ranges


class LidarStandDetectorTest(unittest.TestCase):
    def setUp(self):
        self.angle_min = math.radians(-45.0)
        self.angle_increment = math.radians(1.0)
        self.config = LidarStandDetectorConfig(
            max_range_m=4.0,
            max_cluster_gap_m=0.08,
            min_cluster_points=3,
            min_width_m=0.03,
            max_width_m=0.25,
        )

    def detect(self, ranges):
        return detect_stand_candidates_from_scan(
            ranges,
            angle_min_rad=self.angle_min,
            angle_increment_rad=self.angle_increment,
            config=self.config,
        )

    def test_two_stands_are_clustered_separately_when_gap_exceeds_threshold(self):
        ranges = scan_with_returns([25, 26, 27, 63, 64, 65])

        candidates = self.detect(ranges)

        self.assertEqual(len(candidates), 2)
        self.assertLess(candidates[0].bearing_rad, 0.0)
        self.assertGreater(candidates[1].bearing_rad, 0.0)
        self.assertAlmostEqual(candidates[0].distance_m, 1.0, delta=0.02)
        self.assertGreater(candidates[0].approximate_width_m, 0.03)
        self.assertGreater(candidates[0].confidence, 0.45)

    def test_single_apparent_object_is_split_when_inter_cluster_gap_is_too_large(self):
        ranges = scan_with_returns([40, 41, 42, 50, 51, 52])

        candidates = self.detect(ranges)

        self.assertEqual(len(candidates), 2)
        self.assertLess(candidates[0].bearing_rad, candidates[1].bearing_rad)

    def test_noise_points_below_min_cluster_size_are_ignored(self):
        ranges = scan_with_returns([10, 30, 31, 57, 58, 59])

        candidates = self.detect(ranges)

        self.assertEqual(len(candidates), 1)
        self.assertGreater(candidates[0].bearing_rad, 0.0)
        self.assertEqual(candidates[0].point_count, 3)

    def test_invalid_scan_ranges_are_ignored_without_crashing(self):
        ranges = scan_with_returns([40, 41, 42])
        ranges[5] = float("inf")
        ranges[6] = float("nan")
        ranges[7] = 0.0
        ranges[8] = self.config.max_range_m

        candidates = self.detect(ranges)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].point_count, 3)

    def test_default_detector_accepts_two_beam_distant_stand_head(self):
        ranges = scan_with_returns([], total=360, default_range=5.0)
        ranges[27] = 1.65
        ranges[28] = 1.64

        candidates = detect_stand_candidates_from_scan(
            ranges,
            angle_min_rad=0.0,
            angle_increment_rad=0.01749303564429283,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].point_count, 2)
        self.assertGreater(candidates[0].confidence, 0.55)

    def test_full_rotation_seam_matches_interior_at_real_angular_resolutions(self):
        for count, minimum, increment in SCAN_GEOMETRIES:
            for rotation in (0.0, math.pi):
                with self.subTest(count=count, rotation=rotation):
                    common = dict(angle_min_rad=minimum + rotation,
                                  angle_increment_rad=increment,
                                  angle_max_rad=minimum + rotation + (count - 1) * increment,
                                  scan_topology_profile="full_rotation")
                    interior = scan_with_returns((20, 21), total=count, return_range=1.9)
                    seam = scan_with_returns((count - 1, 0), total=count, return_range=1.9)
                    ordinary = detect_stand_candidates_from_scan(interior, **common)
                    wrapped = detect_stand_candidates_from_scan(seam, **common)
                    self.assertEqual(len(ordinary), 1)
                    self.assertEqual(len(wrapped), 1)
                    self.assertEqual(wrapped[0].point_count, 2)
                    self.assertEqual(wrapped[0].source_indices, (count - 1, 0))
                    self.assertTrue(wrapped[0].wraps_scan_seam)
                    self.assertFalse(ordinary[0].wraps_scan_seam)
                    self.assertAlmostEqual(wrapped[0].distance_m, ordinary[0].distance_m, places=4)

    def test_bare_points_or_undeclared_scan_remain_linear(self):
        ranges = scan_with_returns((359, 0), total=360, return_range=1.9)
        common = dict(angle_min_rad=0.0, angle_increment_rad=math.radians(1))
        self.assertEqual(detect_stand_candidates_from_scan(ranges, **common), [])
        points = scan_points_from_ranges(ranges, **common)
        self.assertEqual(detect_stand_candidates(points), [])

    def test_seam_does_not_bridge_dropped_endpoints_or_missing_sector(self):
        for count, increment, indices in (
            (360, math.radians(1), (358, 0)),
            (360, math.radians(1), (359, 1)),
            (359, math.radians(1), (358, 0)),
            (180, math.radians(1), (179, 0)),
        ):
            with self.subTest(count=count, indices=indices):
                ranges = [float("nan")] * count
                for index in indices:
                    ranges[index] = 1.9
                self.assertEqual(detect_stand_candidates_from_scan(
                    ranges, angle_min_rad=0.0, angle_increment_rad=increment,
                    angle_max_rad=(count - 1) * increment, scan_topology_profile="full_rotation",
                ), [])

    def test_seam_preserves_gap_minimum_points_width_and_interior_dropout_gates(self):
        count, minimum, increment = SCAN_GEOMETRIES[1]
        common = dict(angle_min_rad=minimum, angle_increment_rad=increment,
                      angle_max_rad=minimum + (count - 1) * increment,
                      scan_topology_profile="full_rotation")
        ranges = scan_with_returns((count - 1, 0), total=count, return_range=1.9)
        for config in (LidarStandDetectorConfig(min_cluster_points=3),
                       LidarStandDetectorConfig(max_width_m=.04),
                       LidarStandDetectorConfig(max_cluster_gap_m=.04)):
            self.assertEqual(detect_stand_candidates_from_scan(ranges, config=config, **common), [])
        ranges[0] = 2.2
        self.assertEqual(detect_stand_candidates_from_scan(ranges, **common), [])
        ranges = scan_with_returns((count - 1, 0, 2, 3), total=count, return_range=1.9)
        candidates = detect_stand_candidates_from_scan(ranges, **common)
        self.assertEqual([candidate.source_indices for candidate in candidates], [(count - 1, 0), (2, 3)])


if __name__ == "__main__":
    unittest.main()
