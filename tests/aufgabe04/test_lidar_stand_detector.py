import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.perception.lidar_stand_detector import (  # noqa: E402
    detect_stand_candidates_from_scan,
)
from scripts.aufgabe04.perception.models import LidarStandDetectorConfig  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()

