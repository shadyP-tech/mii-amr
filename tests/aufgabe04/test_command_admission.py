from __future__ import annotations

import math
import unittest

from scripts.aufgabe04.navigation.control.waypoint_controller import (
    VelocityCommand,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.command_admission import (
    command_admission_decision,
    command_shape_interval_sec,
    stuck_distance_progress_epsilon,
)


class CommandAdmissionTest(unittest.TestCase):
    def test_clearance_scaling_below_floor_requires_physical_zero_hold(self):
        decision = command_admission_decision(
            VelocityCommand(0.04, 0.1),
            front_clearance_scale=0.1,
            linear_motion_floor_mps=0.01,
            physical_route=True,
        )

        self.assertAlmostEqual(decision.effective_command.linear_x_mps, 0.004)
        self.assertEqual(decision.effective_command.angular_z_radps, 0.1)
        self.assertTrue(decision.command_floor.zero_hold_required)
        self.assertTrue(decision.clearance_limited_below_floor)
        self.assertTrue(decision.finite)

    def test_nonphysical_route_preserves_existing_no_floor_stop_policy(self):
        decision = command_admission_decision(
            VelocityCommand(0.04, 0.0),
            front_clearance_scale=0.1,
            linear_motion_floor_mps=0.01,
            physical_route=False,
        )

        self.assertTrue(decision.command_floor.zero_hold_required)
        self.assertFalse(decision.clearance_limited_below_floor)

    def test_nonfinite_angular_command_fails_admission(self):
        decision = command_admission_decision(
            VelocityCommand(0.0, math.nan),
            front_clearance_scale=1.0,
            linear_motion_floor_mps=0.01,
            physical_route=True,
        )

        self.assertFalse(decision.finite)

    def test_physical_stuck_threshold_is_bounded_by_reachable_distance(self):
        epsilon = stuck_distance_progress_epsilon(
            0.03,
            physical_route=True,
            remaining_distance_m=0.025,
            waypoint_tolerance_m=0.02,
            effective_linear_x_mps=0.01,
            stuck_timeout_sec=8.0,
        )

        self.assertLess(epsilon, 0.03)
        self.assertGreaterEqual(epsilon, 0.0)

    def test_nonphysical_stuck_threshold_remains_configured_value(self):
        self.assertEqual(
            stuck_distance_progress_epsilon(
                0.03,
                physical_route=False,
                remaining_distance_m=0.001,
                waypoint_tolerance_m=0.02,
                effective_linear_x_mps=0.0,
                stuck_timeout_sec=8.0,
            ),
            0.03,
        )

    def test_shape_interval_uses_period_for_first_command(self):
        self.assertEqual(
            command_shape_interval_sec(
                loop_period_sec=0.1,
                now_monotonic=12.0,
                last_shape_at=None,
            ),
            0.1,
        )
        self.assertAlmostEqual(
            command_shape_interval_sec(
                loop_period_sec=0.1,
                now_monotonic=12.3,
                last_shape_at=12.1,
            ),
            0.2,
        )


if __name__ == "__main__":
    unittest.main()
