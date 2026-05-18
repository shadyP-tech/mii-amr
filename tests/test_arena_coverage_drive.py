import argparse
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "aufgabe03"))

import arena_coverage_drive as coverage  # noqa: E402


def default_args(**overrides):
    values = {
        "linear_speed": coverage.DEFAULT_LINEAR_SPEED_MPS,
        "angular_speed": coverage.DEFAULT_ANGULAR_SPEED_RADPS,
        "forward_half_pass_m": coverage.DEFAULT_FORWARD_HALF_PASS_M,
        "forward_tolerance_m": coverage.DEFAULT_FORWARD_TOLERANCE_M,
        "rotation_tolerance_deg": coverage.DEFAULT_ROTATION_TOLERANCE_DEG,
        "settle_sec": coverage.DEFAULT_SETTLE_SEC,
        "min_scan_range_m": coverage.DEFAULT_MIN_SCAN_RANGE_M,
        "hard_stop_range_m": coverage.DEFAULT_HARD_STOP_RANGE_M,
        "scan_half_angle_deg": coverage.DEFAULT_SCAN_HALF_ANGLE_DEG,
        "max_action_time_sec": coverage.DEFAULT_MAX_ACTION_TIME_SEC,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class ArenaCoverageDriveTest(unittest.TestCase):
    def test_default_route_names_distances_and_cumulative_positions(self):
        route = coverage.build_default_route()

        self.assertEqual(
            [action.name for action in route],
            [
                "SPIN_SCAN",
                "FORWARD",
                "SPIN_SCAN",
                "TURN_AROUND",
                "FORWARD",
                "SPIN_SCAN",
                "TURN_AROUND",
                "FORWARD",
                "STOP",
            ],
        )
        self.assertEqual(
            [action.kind for action in route],
            [
                "rotate",
                "forward",
                "rotate",
                "rotate",
                "forward",
                "rotate",
                "rotate",
                "forward",
                "stop",
            ],
        )
        self.assertAlmostEqual(route[1].value, 1.20)
        self.assertAlmostEqual(route[4].value, 2.40)
        self.assertAlmostEqual(route[7].value, 1.20)

        positions = coverage.route_long_axis_positions(route)
        self.assertEqual(len(positions), 4)
        self.assertAlmostEqual(positions[0], 0.0)
        self.assertAlmostEqual(positions[1], 1.20)
        self.assertAlmostEqual(positions[2], -1.20)
        self.assertAlmostEqual(positions[3], 0.0)

    def test_default_route_remains_inside_configured_arena_bounds(self):
        route = coverage.build_default_route()
        margin = coverage.route_long_axis_margin_m(route)

        self.assertGreaterEqual(margin, coverage.DEFAULT_SAFETY_MARGIN_M)
        self.assertAlmostEqual(margin, 0.75)
        coverage.validate_motion_config(default_args(), route)

    def test_route_geometry_validation_rejects_cumulative_bound_violation(self):
        route = coverage.build_default_route(forward_half_pass_m=1.30)

        with self.assertRaisesRegex(ValueError, "route exceeds arena bounds"):
            coverage.validate_motion_config(default_args(forward_half_pass_m=1.30), route)

    def test_validation_rejects_unsafe_cli_values(self):
        invalid_configs = [
            default_args(linear_speed=0.20),
            default_args(angular_speed=0.70),
            default_args(min_scan_range_m=coverage.ROBOT_RADIUS_M),
            default_args(hard_stop_range_m=0.30),
            default_args(scan_half_angle_deg=100.0),
            default_args(forward_tolerance_m=0.25),
            default_args(rotation_tolerance_deg=25.0),
            default_args(max_action_time_sec=0.0),
        ]

        for args in invalid_configs:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    coverage.validate_motion_config(
                        args,
                        coverage.build_default_route(args.forward_half_pass_m),
                    )

    def test_yaw_accumulation_handles_wraparound_and_full_rotation(self):
        yaw_samples = [170.0, 179.0, -172.0, -90.0, 0.0, 90.0, 170.0]
        accumulated = 0.0
        previous = yaw_samples[0]

        for current in yaw_samples[1:]:
            accumulated += coverage.shortest_angle_delta_deg(previous, current)
            previous = current

        self.assertAlmostEqual(accumulated, 360.0)

    def test_projected_forward_progress_uses_start_heading(self):
        start_pose = {"x": 1.0, "y": 2.0, "yaw_deg": 90.0}
        current_pose = {"x": 1.2, "y": 2.8, "yaw_deg": 92.0}

        self.assertAlmostEqual(
            coverage.projected_forward_progress_m(start_pose, current_pose),
            0.8,
        )

    def test_scan_filtering_ignores_invalid_ranges_and_selects_front_sector(self):
        ranges = [
            float("nan"),
            0.05,
            0.40,
            0.50,
            float("inf"),
            4.50,
            0.60,
        ]
        selected = coverage.valid_scan_ranges(
            ranges,
            angle_min=math.radians(-90.0),
            angle_increment=math.radians(30.0),
            range_min=0.10,
            range_max=4.00,
            sector_half_angle_deg=35.0,
        )

        self.assertEqual(selected, [0.4, 0.5])

    def test_no_valid_scan_ranges_is_unsafe(self):
        result = coverage.evaluate_scan_safety(
            [float("nan"), float("inf"), 0.05],
            angle_min=0.0,
            angle_increment=0.1,
            range_min=0.10,
            range_max=4.0,
            mode="forward",
            scan_half_angle_deg=35.0,
            min_scan_range_m=0.28,
            hard_stop_range_m=0.18,
        )

        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "no_valid_scan_ranges")

    def test_hard_stop_triggers_on_narrow_close_reading(self):
        result = coverage.evaluate_scan_safety(
            [0.17, 0.80, 0.90, 1.00],
            angle_min=math.radians(-15.0),
            angle_increment=math.radians(10.0),
            range_min=0.10,
            range_max=4.0,
            mode="forward",
            scan_half_angle_deg=35.0,
            min_scan_range_m=0.28,
            hard_stop_range_m=0.18,
        )

        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "hard_stop")

    def test_soft_percentile_stop_triggers_on_broad_near_obstacle(self):
        result = coverage.evaluate_scan_safety(
            [0.24, 0.25, 0.26, 0.27, 0.80, 0.90],
            angle_min=math.radians(-30.0),
            angle_increment=math.radians(10.0),
            range_min=0.10,
            range_max=4.0,
            mode="forward",
            scan_half_angle_deg=35.0,
            min_scan_range_m=0.28,
            hard_stop_range_m=0.18,
        )

        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "soft_stop")

    def test_rotation_scan_safety_uses_full_scan(self):
        result = coverage.evaluate_scan_safety(
            [0.50, 0.60, 0.20],
            angle_min=math.radians(120.0),
            angle_increment=math.radians(30.0),
            range_min=0.10,
            range_max=4.0,
            mode="rotate",
            scan_half_angle_deg=35.0,
            min_scan_range_m=0.28,
            hard_stop_range_m=0.18,
        )

        self.assertFalse(result.safe)
        self.assertEqual(result.reason, "soft_stop")


if __name__ == "__main__":
    unittest.main()
