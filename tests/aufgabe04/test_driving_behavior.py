import unittest

from scripts.aufgabe04.navigation.control.driving_behavior import (
    CommandSmoother,
    CommandSmoothingConfig,
    DETECTED_STAND_CAMERA_HEADING_TOLERANCE_RAD,
    controller_config_for_route_kind,
    next_control_loop_timing,
    shape_velocity_command,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime import FollowerConfig
from scripts.aufgabe04.navigation.control.waypoint_controller import ControllerConfig
from scripts.aufgabe04.navigation.control.waypoint_controller import VelocityCommand


class DrivingBehaviorTest(unittest.TestCase):
    def test_detected_stand_preapproach_uses_terminal_heading_only(self):
        configured = controller_config_for_route_kind(
            ControllerConfig(
                enforce_heading_corridor=True,
                heading_tolerance_rad=0.25,
            ),
            "detected_stand_preapproach",
        )

        self.assertFalse(configured.enforce_heading_corridor)
        self.assertTrue(configured.exact_vertex_pursuit)
        self.assertAlmostEqual(
            configured.heading_tolerance_rad,
            DETECTED_STAND_CAMERA_HEADING_TOLERANCE_RAD,
        )

    def test_detected_stand_preapproach_preserves_tighter_heading_tolerance(self):
        configured = controller_config_for_route_kind(
            ControllerConfig(heading_tolerance_rad=0.02),
            "detected_stand_preapproach",
        )

        self.assertAlmostEqual(configured.heading_tolerance_rad, 0.02)

    def test_discovery_and_ordinary_routes_do_not_inherit_camera_heading_cap(self):
        config = ControllerConfig(heading_tolerance_rad=0.25)

        discovery = controller_config_for_route_kind(
            config,
            "stand_discovery_corridor",
        )
        ordinary = controller_config_for_route_kind(config, "ordinary_waypoint")

        self.assertAlmostEqual(discovery.heading_tolerance_rad, 0.25)
        self.assertIs(ordinary, config)

    def test_command_smoother_ramps_from_zero_after_reset(self):
        smoother = CommandSmoother(
            CommandSmoothingConfig(
                max_linear_accel_mps2=0.10,
                max_angular_accel_radps2=0.60,
            )
        )

        first = smoother.apply(VelocityCommand(0.055, 0.18), dt_sec=0.1)
        self.assertAlmostEqual(first.linear_x_mps, 0.01)
        self.assertAlmostEqual(first.angular_z_radps, 0.06)

        smoother.reset()
        after_zero = smoother.apply(VelocityCommand(0.055, 0.18), dt_sec=0.1)
        self.assertAlmostEqual(after_zero.linear_x_mps, 0.01)
        self.assertAlmostEqual(after_zero.angular_z_radps, 0.06)

    def test_zero_command_bypasses_smoothing_and_resets(self):
        smoother = CommandSmoother(
            CommandSmoothingConfig(
                max_linear_accel_mps2=0.10,
                max_angular_accel_radps2=0.60,
            )
        )
        smoother.apply(VelocityCommand(0.055, 0.18), dt_sec=1.0)

        stopped = smoother.apply(VelocityCommand(0.0, 0.0), dt_sec=0.1)
        reverse = smoother.apply(VelocityCommand(-0.03, 0.0), dt_sec=0.1)

        self.assertEqual(stopped, VelocityCommand(0.0, 0.0))
        self.assertAlmostEqual(reverse.linear_x_mps, -0.01)
        self.assertAlmostEqual(reverse.angular_z_radps, 0.0)

    def test_each_zero_axis_is_immediate_during_in_place_alignment(self):
        smoother = CommandSmoother(CommandSmoothingConfig())
        smoother.apply(VelocityCommand(0.055, 0.18), dt_sec=1.0)

        rotate = smoother.apply(VelocityCommand(0.0, 0.12), dt_sec=0.1)
        straight = smoother.apply(VelocityCommand(0.03, 0.0), dt_sec=0.1)

        self.assertEqual(rotate.linear_x_mps, 0.0)
        self.assertLessEqual(abs(rotate.angular_z_radps), 0.12)
        self.assertEqual(straight.angular_z_radps, 0.0)
        self.assertLessEqual(abs(straight.linear_x_mps), 0.03)

    def test_reductions_are_immediate_and_sign_changes_cross_zero(self):
        smoother = CommandSmoother(CommandSmoothingConfig())
        smoother.apply(VelocityCommand(0.055, 0.18), dt_sec=1.0)

        reduced = smoother.apply(VelocityCommand(0.02, 0.04), dt_sec=0.1)
        reversed_command = smoother.apply(
            VelocityCommand(-0.02, -0.04),
            dt_sec=0.1,
        )

        self.assertEqual(reduced, VelocityCommand(0.02, 0.04))
        self.assertEqual(reversed_command, VelocityCommand(0.0, 0.0))

    def test_zero_dt_still_honors_safety_reductions(self):
        config = CommandSmoothingConfig()
        previous = VelocityCommand(0.05, 0.20)

        zero_axis = shape_velocity_command(
            VelocityCommand(0.0, 0.20),
            previous,
            0.0,
            config,
        )
        reduced = shape_velocity_command(
            VelocityCommand(0.02, 0.10),
            previous,
            0.0,
            config,
        )

        self.assertEqual(zero_axis, VelocityCommand(0.0, 0.20))
        self.assertEqual(reduced, VelocityCommand(0.02, 0.10))

    def test_disabled_smoothing_passes_commands_through(self):
        smoother = CommandSmoother(CommandSmoothingConfig(enabled=False))

        shaped = smoother.apply(VelocityCommand(0.055, 0.18), dt_sec=0.1)

        self.assertEqual(shaped, VelocityCommand(0.055, 0.18))

    def test_follower_rejects_acceleration_that_starts_below_motion_floor(self):
        with self.assertRaisesRegex(ValueError, "reach the motion floor"):
            FollowerConfig(
                controller=ControllerConfig(),
                command_smoothing=CommandSmoothingConfig(
                    max_linear_accel_mps2=0.05,
                ),
                control_rate_hz=10.0,
                linear_motion_floor_mps=0.01,
            )

    def test_next_control_loop_timing_preserves_deadline_cadence(self):
        first = next_control_loop_timing(
            previous_deadline_sec=None,
            now_sec=10.0,
            control_rate_hz=10.0,
        )
        self.assertAlmostEqual(first.sleep_sec, 0.1)
        self.assertAlmostEqual(first.next_deadline_sec, 10.2)

        overrun = next_control_loop_timing(
            previous_deadline_sec=10.2,
            now_sec=10.55,
            control_rate_hz=10.0,
        )
        self.assertEqual(overrun.sleep_sec, 0.0)
        self.assertEqual(overrun.skipped_deadline_count, 3)
        self.assertAlmostEqual(overrun.next_deadline_sec, 10.6)


if __name__ == "__main__":
    unittest.main()
