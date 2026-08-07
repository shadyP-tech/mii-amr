import json
import math
import sys
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.models import Pose2D  # noqa: E402
from scripts.aufgabe04.navigation.transient_blockage_policy import (  # noqa: E402
    CLEARANCE_LIMITED_MOTION_FLOOR,
    LINEAR_COMMAND_BELOW_FLOOR,
    LINEAR_COMMAND_MOTION_CAPABLE,
    PersistentObstacleConfig,
    StationaryFrontSectorSample,
    classify_linear_command,
    confirm_persistent_obstacle,
    confirm_stationary_clearance,
    reachable_distance_progress_epsilon,
)


def sample(
    timestamp_sec: float,
    *,
    front_range_m: float = 0.234,
    front_bearing_rad: float = 0.0,
    map_x_m: float = 1.0,
    map_y_m: float = 2.0,
    map_yaw_rad: float = 0.0,
    odom_x_m: float = 0.4,
    odom_y_m: float = 0.7,
    odom_yaw_rad: float = 0.0,
) -> StationaryFrontSectorSample:
    return StationaryFrontSectorSample(
        timestamp_sec=timestamp_sec,
        front_range_m=front_range_m,
        front_bearing_rad=front_bearing_rad,
        map_pose=Pose2D(map_x_m, map_y_m, map_yaw_rad),
        odom_pose=Pose2D(odom_x_m, odom_y_m, odom_yaw_rad),
    )


class LinearCommandFloorTest(unittest.TestCase):
    def test_failed_run_scaled_command_requests_zero_hold(self):
        decision = classify_linear_command(
            0.0430346319,
            0.00812876317,
            linear_motion_floor_mps=0.01,
        )

        self.assertEqual(decision.nominal_class, LINEAR_COMMAND_MOTION_CAPABLE)
        self.assertEqual(decision.effective_class, LINEAR_COMMAND_BELOW_FLOOR)
        self.assertEqual(decision.output_linear_x_mps, 0.0)
        self.assertTrue(decision.zero_hold_required)
        self.assertTrue(decision.stationary_confirmation_required)
        self.assertTrue(decision.fail_closed)
        self.assertEqual(decision.reasons, (CLEARANCE_LIMITED_MOTION_FLOOR,))
        json.dumps(decision.to_log_dict(), allow_nan=False)

    def test_floor_is_inclusive_and_reverse_commands_use_magnitude(self):
        at_floor = classify_linear_command(0.02, 0.01)
        reverse = classify_linear_command(-0.03, -0.02)

        self.assertEqual(at_floor.effective_class, LINEAR_COMMAND_MOTION_CAPABLE)
        self.assertFalse(at_floor.zero_hold_required)
        self.assertEqual(at_floor.output_linear_x_mps, 0.01)
        self.assertEqual(reverse.effective_class, LINEAR_COMMAND_MOTION_CAPABLE)
        self.assertEqual(reverse.output_linear_x_mps, -0.02)

    def test_scaled_exact_zero_still_requires_stationary_confirmation(self):
        decision = classify_linear_command(0.04, 0.0)

        self.assertTrue(decision.zero_hold_required)
        self.assertTrue(decision.stationary_confirmation_required)
        self.assertEqual(
            decision.reasons, ("effective_linear_zero_during_nominal_motion",)
        )

    def test_floor_and_commands_require_finite_valid_values(self):
        for floor in (0.0, -0.1, float("inf"), float("nan")):
            with self.subTest(floor=floor), self.assertRaises(ValueError):
                classify_linear_command(0.04, 0.02, linear_motion_floor_mps=floor)
        with self.assertRaises(ValueError):
            classify_linear_command(float("nan"), 0.02)
        with self.assertRaises(ValueError):
            classify_linear_command(0.04, float("inf"))


class PersistentObstacleTest(unittest.TestCase):
    def setUp(self):
        self.config = PersistentObstacleConfig()

    def coherent_samples(self):
        return (
            sample(10.00, front_range_m=0.234, front_bearing_rad=-0.01),
            sample(
                10.10,
                front_range_m=0.232,
                front_bearing_rad=0.02,
                map_x_m=1.001,
                odom_x_m=0.401,
            ),
            sample(
                10.20,
                front_range_m=0.235,
                map_x_m=0.999,
                odom_x_m=0.399,
            ),
        )

    def test_single_0234_meter_ray_never_confirms(self):
        decision = confirm_persistent_obstacle(
            (sample(10.0, front_range_m=0.234),),
            now_sec=10.1,
            config=self.config,
        )

        self.assertFalse(decision.confirmed)
        self.assertTrue(decision.fail_closed)
        self.assertIn("insufficient_distinct_recent_samples", decision.reasons)
        self.assertEqual(decision.distinct_sample_count, 1)

    def test_coherent_distinct_recent_stationary_cluster_confirms(self):
        decision = confirm_persistent_obstacle(
            self.coherent_samples(), now_sec=10.25, config=self.config
        )

        self.assertTrue(decision.confirmed)
        self.assertFalse(decision.fail_closed)
        self.assertEqual(
            decision.reasons, ("coherent_persistent_obstacle_confirmed",)
        )
        self.assertEqual(decision.distinct_sample_count, 3)
        self.assertAlmostEqual(decision.median_front_range_m, 0.234)
        self.assertAlmostEqual(decision.median_front_bearing_rad, 0.0)
        self.assertLessEqual(
            decision.map_hit_spread_m, self.config.max_map_hit_spread_m
        )
        details = decision.to_log_dict()
        self.assertEqual(details["median_map_hit"]["x_m"], decision.median_map_hit_x_m)
        self.assertEqual(details["median_front_bearing_rad"], 0.0)
        self.assertEqual(details["thresholds"]["min_distinct_samples"], 3)
        json.dumps(details, allow_nan=False)

    def test_duplicate_or_too_close_timestamps_are_not_distinct(self):
        samples = (
            sample(10.00),
            sample(10.00),
            sample(10.03),
            sample(10.04),
        )

        decision = confirm_persistent_obstacle(
            samples, now_sec=10.1, config=self.config
        )

        self.assertFalse(decision.confirmed)
        self.assertEqual(decision.distinct_sample_count, 1)
        self.assertEqual(decision.duplicate_sample_count, 3)
        self.assertIn("insufficient_distinct_recent_samples", decision.reasons)

    def test_stale_and_overwide_windows_do_not_confirm(self):
        stale = confirm_persistent_obstacle(
            (sample(1.0), sample(1.1), sample(1.2)),
            now_sec=3.0,
            config=self.config,
        )
        overwide = confirm_persistent_obstacle(
            (sample(10.0), sample(10.4), sample(10.8)),
            now_sec=10.8,
            config=replace(
                self.config,
                max_sample_age_sec=2.0,
                max_sample_window_sec=0.5,
            ),
        )

        self.assertFalse(stale.confirmed)
        self.assertEqual(stale.reasons, ("no_recent_stationary_samples",))
        self.assertFalse(overwide.confirmed)
        self.assertIn("samples_exceed_max_sample_window", overwide.reasons)

    def test_motion_in_either_pose_frame_rejects_confirmation(self):
        map_motion = list(self.coherent_samples())
        map_motion[-1] = sample(10.20, map_x_m=1.08, odom_x_m=0.4)
        odom_motion = list(self.coherent_samples())
        odom_motion[-1] = sample(10.20, map_x_m=1.0, odom_x_m=0.75)

        map_decision = confirm_persistent_obstacle(
            map_motion, now_sec=10.25, config=self.config
        )
        odom_decision = confirm_persistent_obstacle(
            odom_motion, now_sec=10.25, config=self.config
        )

        self.assertFalse(map_decision.confirmed)
        self.assertIn("map_pose_not_stationary", map_decision.reasons)
        self.assertFalse(odom_decision.confirmed)
        self.assertIn("odom_pose_not_stationary", odom_decision.reasons)

    def test_map_odom_localization_divergence_is_an_explicit_failure(self):
        divergent = (
            sample(10.0),
            sample(10.1, map_x_m=1.01),
            sample(10.2, map_x_m=1.02),
        )
        config = replace(
            self.config,
            max_map_pose_translation_spread_m=0.05,
            max_map_odom_offset_spread_m=0.01,
        )

        decision = confirm_persistent_obstacle(
            divergent, now_sec=10.25, config=config
        )

        self.assertFalse(decision.confirmed)
        self.assertIn("map_odom_localization_divergence", decision.reasons)

    def test_incoherent_map_hits_and_nonfront_bearings_do_not_confirm(self):
        incoherent = (
            sample(10.0, front_bearing_rad=-0.25),
            sample(10.1, front_bearing_rad=0.0),
            sample(10.2, front_bearing_rad=0.25),
        )
        outside_front = (
            sample(10.0, front_bearing_rad=0.7),
            sample(10.1, front_bearing_rad=0.7),
            sample(10.2, front_bearing_rad=0.7),
        )

        cluster = confirm_persistent_obstacle(
            incoherent,
            now_sec=10.25,
            config=replace(self.config, max_map_hit_spread_m=0.02),
        )
        bearing = confirm_persistent_obstacle(
            outside_front, now_sec=10.25, config=self.config
        )

        self.assertFalse(cluster.confirmed)
        self.assertIn("map_hit_cluster_spread_exceeded", cluster.reasons)
        self.assertFalse(bearing.confirmed)
        self.assertIn("bearing_outside_front_sector", bearing.reasons)

    def test_future_timestamp_fails_closed(self):
        decision = confirm_persistent_obstacle(
            (*self.coherent_samples(), sample(11.0)),
            now_sec=10.25,
            config=self.config,
        )

        self.assertFalse(decision.confirmed)
        self.assertTrue(decision.fail_closed)
        self.assertEqual(decision.reasons, ("future_dated_stationary_sample",))

    def test_samples_and_config_are_frozen_and_finite(self):
        value = sample(10.0)
        with self.assertRaises(FrozenInstanceError):
            value.front_range_m = 0.3
        with self.assertRaises(ValueError):
            sample(10.0, front_range_m=float("nan"))
        with self.assertRaises(ValueError):
            sample(10.0, map_x_m=float("inf"))
        with self.assertRaises(ValueError):
            PersistentObstacleConfig(min_distinct_samples=1)
        with self.assertRaises(ValueError):
            PersistentObstacleConfig(min_sample_separation_sec=0.0)
        with self.assertRaises(ValueError):
            PersistentObstacleConfig(
                min_distinct_samples=4,
                min_sample_separation_sec=0.2,
                max_sample_window_sec=0.5,
            )
        with self.assertRaises(FrozenInstanceError):
            self.config.max_sample_age_sec = 1.0
        json.dumps(value.to_log_dict(), allow_nan=False)
        json.dumps(self.config.to_log_dict(), allow_nan=False)


class StationaryClearanceTest(unittest.TestCase):
    def setUp(self):
        self.config = PersistentObstacleConfig()

    def clear_samples(self):
        return (
            sample(20.0, front_range_m=0.31, front_bearing_rad=-0.02),
            sample(20.1, front_range_m=0.35, front_bearing_rad=0.01),
            sample(20.2, front_range_m=0.33, front_bearing_rad=0.0),
        )

    def test_all_and_median_ranges_above_threshold_confirm_clearance(self):
        decision = confirm_stationary_clearance(
            self.clear_samples(),
            now_sec=20.25,
            clearance_threshold_m=0.30,
            config=self.config,
        )

        self.assertTrue(decision.confirmed)
        self.assertFalse(decision.fail_closed)
        self.assertEqual(
            decision.reasons, ("stationary_front_clearance_confirmed",)
        )
        self.assertAlmostEqual(decision.minimum_front_range_m, 0.31)
        self.assertAlmostEqual(decision.median_front_range_m, 0.33)
        self.assertAlmostEqual(decision.median_front_bearing_rad, 0.0)
        json.dumps(decision.to_log_dict(), allow_nan=False)

    def test_one_range_at_or_below_threshold_fails_closed(self):
        one_blocked = (
            sample(20.0, front_range_m=0.30),
            sample(20.1, front_range_m=0.34),
            sample(20.2, front_range_m=0.35),
        )

        decision = confirm_stationary_clearance(
            one_blocked,
            now_sec=20.25,
            clearance_threshold_m=0.30,
            config=self.config,
        )

        self.assertFalse(decision.confirmed)
        self.assertTrue(decision.fail_closed)
        self.assertIn(
            "front_range_not_above_clearance_threshold", decision.reasons
        )
        self.assertNotIn(
            "median_front_range_not_above_clearance_threshold",
            decision.reasons,
        )

    def test_median_at_or_below_threshold_is_reported(self):
        median_blocked = (
            sample(20.0, front_range_m=0.29),
            sample(20.1, front_range_m=0.30),
            sample(20.2, front_range_m=0.35),
        )

        decision = confirm_stationary_clearance(
            median_blocked,
            now_sec=20.25,
            clearance_threshold_m=0.30,
            config=self.config,
        )

        self.assertFalse(decision.confirmed)
        self.assertIn(
            "median_front_range_not_above_clearance_threshold",
            decision.reasons,
        )

    def test_clearance_reuses_distinct_and_stationary_fail_closed_gates(self):
        duplicate = (sample(20.0), sample(20.0), sample(20.0))
        moving = list(self.clear_samples())
        moving[-1] = sample(20.2, front_range_m=0.33, odom_x_m=0.8)

        duplicate_decision = confirm_stationary_clearance(
            duplicate,
            now_sec=20.1,
            clearance_threshold_m=0.20,
            config=self.config,
        )
        moving_decision = confirm_stationary_clearance(
            moving,
            now_sec=20.25,
            clearance_threshold_m=0.30,
            config=self.config,
        )

        self.assertFalse(duplicate_decision.confirmed)
        self.assertIn(
            "insufficient_distinct_recent_samples",
            duplicate_decision.reasons,
        )
        self.assertFalse(moving_decision.confirmed)
        self.assertIn("odom_pose_not_stationary", moving_decision.reasons)

    def test_clearance_threshold_must_be_finite_and_positive(self):
        for threshold in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                confirm_stationary_clearance(
                    self.clear_samples(),
                    now_sec=20.25,
                    clearance_threshold_m=threshold,
                    config=self.config,
                )


class ReachableProgressEpsilonTest(unittest.TestCase):
    def test_epsilon_is_bounded_by_each_reachable_limit(self):
        self.assertAlmostEqual(
            reachable_distance_progress_epsilon(
                0.03,
                remaining_distance_m=0.20,
                waypoint_tolerance_m=0.03,
                expected_effective_travel_m=0.00812876317,
            ),
            0.00812876317,
        )
        self.assertAlmostEqual(
            reachable_distance_progress_epsilon(
                0.03,
                remaining_distance_m=0.04,
                waypoint_tolerance_m=0.03,
                expected_effective_travel_m=0.10,
            ),
            0.01,
        )
        self.assertAlmostEqual(
            reachable_distance_progress_epsilon(
                0.02,
                remaining_distance_m=0.20,
                waypoint_tolerance_m=0.03,
                expected_effective_travel_m=0.10,
            ),
            0.02,
        )

    def test_at_waypoint_or_zero_expected_travel_yields_zero(self):
        self.assertEqual(
            reachable_distance_progress_epsilon(
                0.03,
                remaining_distance_m=0.03,
                waypoint_tolerance_m=0.03,
                expected_effective_travel_m=0.10,
            ),
            0.0,
        )
        self.assertEqual(
            reachable_distance_progress_epsilon(
                0.03,
                remaining_distance_m=1.0,
                waypoint_tolerance_m=0.03,
                expected_effective_travel_m=0.0,
            ),
            0.0,
        )

    def test_invalid_progress_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            reachable_distance_progress_epsilon(
                0.03,
                remaining_distance_m=-0.1,
                waypoint_tolerance_m=0.03,
                expected_effective_travel_m=0.1,
            )
        with self.assertRaises(ValueError):
            reachable_distance_progress_epsilon(
                0.03,
                remaining_distance_m=0.1,
                waypoint_tolerance_m=0.03,
                expected_effective_travel_m=float("nan"),
            )


if __name__ == "__main__":
    unittest.main()
