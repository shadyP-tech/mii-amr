import math
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.control.follower_models import FollowerResult  # noqa: E402
from scripts.aufgabe04.navigation.waypoint_follower.runtime import (  # noqa: E402
    FollowerConfig,
    FollowerResult as SimpleFollowerResult,
    stuck_progress_details,
    tf_lookup_failure_details,
)
from scripts.aufgabe04.navigation.control.waypoint_controller import ControllerConfig  # noqa: E402
from scripts.aufgabe04.navigation.waypoint_follower.terminal_heading_budget import (  # noqa: E402
    DEFAULT_TERMINAL_HEADING_TIMEOUT_SEC,
)


class FollowerModelsTest(unittest.TestCase):
    def test_simple_follower_uses_shared_result_model(self):
        self.assertIs(SimpleFollowerResult, FollowerResult)

    def test_follower_config_waits_for_initial_runtime_inputs_by_default(self):
        config = FollowerConfig(controller=ControllerConfig())

        self.assertEqual(config.initial_sensor_wait_sec, 2.0)
        self.assertEqual(config.front_obstacle_slow_distance_m, 0.38)
        self.assertEqual(config.stuck_timeout_sec, 8.0)
        self.assertEqual(
            config.terminal_heading_timeout_sec,
            DEFAULT_TERMINAL_HEADING_TIMEOUT_SEC,
        )
        self.assertEqual(config.certified_corner_release_tolerance_m, 0.01)
        self.assertEqual(config.certified_corner_hold_tolerance_m, 0.025)
        self.assertEqual(config.certified_corner_max_reacquire_attempts, 2)

    def test_follower_timeouts_must_be_finite_and_positive(self):
        for field in ("waypoint_timeout_sec", "terminal_heading_timeout_sec"):
            for value in (0.0, -1.0, math.inf, math.nan):
                with self.subTest(field=field, value=value), self.assertRaises(
                    ValueError
                ):
                    FollowerConfig(
                        controller=ControllerConfig(),
                        **{field: value},
                    )

    def test_corner_hold_must_preserve_margin_inside_route_tube(self):
        with self.assertRaisesRegex(ValueError, "strictly inside"):
            FollowerConfig(
                controller=ControllerConfig(),
                certified_route_tube_radius_m=0.03,
                certified_corner_hold_tolerance_m=0.03,
            )

    def test_corner_reacquire_budget_must_be_non_negative_integer(self):
        for value in (-1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                FollowerConfig(
                    controller=ControllerConfig(),
                    certified_corner_max_reacquire_attempts=value,
                )

    def test_stuck_progress_details_include_command_and_clearance_context(self):
        details = stuck_progress_details(
            target_index=2,
            distance_to_target_m=0.42,
            last_progress_distance_m=0.43,
            elapsed_without_progress_sec=8.5,
            max_without_progress_sec=8.0,
            progress_epsilon_m=0.03,
            commanded_linear_x_mps=0.05,
            commanded_angular_z_radps=0.1,
            front_clearance_scale=0.0,
            effective_linear_x_mps=0.0,
            front_clearance_details={"source": "front_sector", "nearest_valid_range_m": 0.21},
            pursuit_index=3,
            controlled_heading_error_rad=0.72,
            last_progress_heading_error_rad=0.85,
            heading_progress_epsilon_rad=0.10,
            last_progress_target_index=2,
            last_progress_pursuit_index=3,
        )

        self.assertEqual(details["stop_reason"], "stuck no progress")
        self.assertEqual(details["source"], "progress_monitor")
        self.assertEqual(details["target_index"], 2)
        self.assertEqual(details["commanded_linear_x_mps"], 0.05)
        self.assertEqual(details["front_clearance_scale"], 0.0)
        self.assertEqual(details["effective_linear_x_mps"], 0.0)
        self.assertEqual(details["pursuit_index"], 3)
        self.assertEqual(details["controlled_heading_error_rad"], 0.72)
        self.assertEqual(details["heading_progress_epsilon_rad"], 0.10)
        self.assertEqual(details["front_clearance"]["source"], "front_sector")

    def test_tf_lookup_failure_details_distinguish_exception_and_stale(self):
        exception_details = tf_lookup_failure_details(
            reason="lookup_exception",
            target_frame="odom",
            source_frame="base_footprint",
            max_age_sec=2.0,
            exception=RuntimeError("missing transform"),
        )

        self.assertEqual(exception_details["stop_reason"], "map-to-base transform unavailable")
        self.assertEqual(exception_details["source"], "tf_lookup")
        self.assertEqual(exception_details["reason"], "lookup_exception")
        self.assertEqual(exception_details["target_frame"], "odom")
        self.assertEqual(exception_details["source_frame"], "base_footprint")
        self.assertEqual(exception_details["max_age_sec"], 2.0)
        self.assertEqual(exception_details["exception_type"], "RuntimeError")

        stale_details = tf_lookup_failure_details(
            reason="stale_transform",
            target_frame="odom",
            source_frame="base_footprint",
            max_age_sec=2.0,
            age_sec=2.5,
        )

        self.assertEqual(stale_details["reason"], "stale_transform")
        self.assertEqual(stale_details["age_sec"], 2.5)

    def test_result_model_is_frozen_and_adapter_neutral(self):
        result = FollowerResult("completed", "", 1.25, 0.5, True)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.distance_estimate_m, 0.5)
        with self.assertRaises(FrozenInstanceError):
            result.status = "stopped"


if __name__ == "__main__":
    unittest.main()
