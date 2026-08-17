from __future__ import annotations

import math
import unittest

from scripts.aufgabe04.navigation.spatial_assignment import (
    assign_spatial_points,
)


class SpatialAssignmentTest(unittest.TestCase):
    def test_global_assignment_recovers_match_lost_by_greedy_choice(self):
        assignments = assign_spatial_points(
            ((0.0, 0.0), (0.20, 0.0)),
            (
                (0.08, 0.0),  # Can match either left point; nearer to left 0.
                (0.09, -0.15),  # Can match only left 0 under the 0.18 m gate.
            ),
            maximum_distance_m=0.18,
        )

        self.assertEqual(
            tuple(
                (assignment.left_index, assignment.right_index)
                for assignment in assignments
            ),
            ((0, 1), (1, 0)),
        )

    def test_minimizes_total_distance_after_cardinality(self):
        assignments = assign_spatial_points(
            ((0.0, 0.0), (1.0, 0.0)),
            ((0.1, 0.0), (0.9, 0.0)),
            maximum_distance_m=1.0,
        )

        self.assertEqual(
            tuple(
                (assignment.left_index, assignment.right_index)
                for assignment in assignments
            ),
            ((0, 0), (1, 1)),
        )
        self.assertAlmostEqual(
            sum(item.distance_m for item in assignments),
            0.2,
        )

    def test_exact_cost_tie_uses_lexicographically_smallest_pairs(self):
        assignments = assign_spatial_points(
            ((0.0, 0.0), (0.0, 0.0)),
            ((0.0, 0.0), (0.0, 0.0)),
            maximum_distance_m=0.0,
        )

        self.assertEqual(
            tuple(
                (assignment.left_index, assignment.right_index)
                for assignment in assignments
            ),
            ((0, 0), (1, 1)),
        )

    def test_uses_smaller_input_side_without_changing_index_orientation(self):
        assignments = assign_spatial_points(
            ((0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (6.0, 0.0)),
            ((2.1, 0.0),),
            maximum_distance_m=0.2,
        )

        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].left_index, 1)
        self.assertEqual(assignments[0].right_index, 0)

    def test_empty_input_has_empty_assignment(self):
        self.assertEqual(
            assign_spatial_points((), ((0.0, 0.0),), maximum_distance_m=0.2),
            (),
        )

    def test_distance_gate_is_inclusive_without_tolerance_widening(self):
        self.assertEqual(
            len(
                assign_spatial_points(
                    ((0.0, 0.0),),
                    ((0.18, 0.0),),
                    maximum_distance_m=0.18,
                )
            ),
            1,
        )
        self.assertEqual(
            assign_spatial_points(
                ((0.0, 0.0),),
                ((math.nextafter(0.18, math.inf), 0.0),),
                maximum_distance_m=0.18,
            ),
            (),
        )

    def test_rejects_invalid_gate_and_coordinates(self):
        with self.assertRaisesRegex(ValueError, "maximum_distance_m"):
            assign_spatial_points((), (), maximum_distance_m=-0.1)
        with self.assertRaisesRegex(ValueError, "coordinates must be finite"):
            assign_spatial_points(
                ((math.nan, 0.0),),
                ((0.0, 0.0),),
                maximum_distance_m=0.2,
            )


if __name__ == "__main__":
    unittest.main()
