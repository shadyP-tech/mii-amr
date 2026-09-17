"""Fast line filtering preserves distinct supporting rails and greedy order."""

import time
import unittest

from scripts.aufgabe04.perception.stand_axis.head_acquisition_budget import HeadAcquisitionDeadlineExceeded
from scripts.aufgabe04.perception.stand_axis.head_cold_acquisition import _distinct_rails


def rail(start, stop, offset, direction=0):
    a, b = (start, offset), (stop, offset)
    if direction:
        a, b = a[::-1], b[::-1]
    return (stop - start, a, b)


class HeadRailDuplicateTests(unittest.TestCase):
    def test_fragment_chain_cannot_merge_separate_parallel_borders(self):
        for direction in (0, 1):
            with self.subTest(direction=direction):
                first, bridge, last = [rail(0., 100., offset, direction)
                                       for offset in (0., 1.2, 2.4)]
                self.assertEqual(_distinct_rails([last, bridge, first], direction), [first, last])

    def test_longest_observed_fragment_remains_the_representative(self):
        for direction in (0, 1):
            long = rail(0., 120., 0., direction)
            short = rail(5., 110., .2, direction)
            self.assertEqual(_distinct_rails([short, long, long], direction), [long])

    def test_separation_and_overlap_boundaries_stay_distinct(self):
        first = rail(0., 100., 0.)
        for other in (rail(0., 100., 1.5), rail(50., 150., 0.), rail(101., 201., 0.)):
            with self.subTest(other=other):
                self.assertEqual(len(_distinct_rails([first, other], 0)), 2)

    def test_dense_repeated_grid_retains_every_distinct_rail(self):
        expected = [rail(0., 100., 4. * index) for index in range(100)]
        fragments = [rail(10., 90., 4. * index + .2) for index in range(100)]
        self.assertEqual(_distinct_rails(fragments + expected[::-1], 0), expected)

    def test_expired_comparison_never_returns_partial_rail_set(self):
        with self.assertRaises(HeadAcquisitionDeadlineExceeded):
            _distinct_rails([rail(0., 100., 0.)], 0,
                            deadline_monotonic_sec=time.monotonic() - 1.)


if __name__ == "__main__":
    unittest.main()
