import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.localization.ros_preflight import (  # noqa: E402
    StationaryAmclPoseSample,
    evaluate_latest_stationary_amcl_window,
    evaluate_stationary_amcl_stability,
)


def _covariance(*, x: float = 0.0001, y: float = 0.0001, yaw: float = 0.0004):
    values = [0.0] * 36
    values[0] = x
    values[7] = y
    values[35] = yaw
    return tuple(values)


class RosPreflightStationaryAmclStabilityTest(unittest.TestCase):
    def _evaluate(self, samples):
        return evaluate_stationary_amcl_stability(
            samples,
            required_sample_count=5,
            max_position_spread_m=0.015,
            max_yaw_spread_rad=0.03,
            max_position_std_m=0.015,
            max_yaw_std_rad=0.03,
        )

    def test_stable_samples_pass_and_report_covariance(self):
        samples = tuple(
            StationaryAmclPoseSample(
                x_m=0.001 * index,
                y_m=-0.0005 * index,
                yaw_rad=0.002 * index,
                covariance=_covariance(),
            )
            for index in range(5)
        )

        observation = self._evaluate(samples)

        self.assertTrue(observation.ok)
        self.assertLess(observation.data["maximum_position_spread_m"], 0.005)
        self.assertLess(observation.data["maximum_yaw_spread_rad"], 0.01)
        self.assertAlmostEqual(
            observation.data["maximum_reported_position_std_m"],
            0.01,
        )
        self.assertAlmostEqual(
            observation.data["maximum_reported_yaw_std_rad"],
            0.02,
        )

    def test_position_spread_above_limit_fails(self):
        samples = (
            StationaryAmclPoseSample(0.0, 0.0, 0.0),
            StationaryAmclPoseSample(0.016, 0.0, 0.0),
            StationaryAmclPoseSample(0.001, 0.0, 0.0),
            StationaryAmclPoseSample(0.002, 0.0, 0.0),
            StationaryAmclPoseSample(0.003, 0.0, 0.0),
        )

        observation = self._evaluate(samples)

        self.assertFalse(observation.ok)
        self.assertAlmostEqual(
            observation.data["maximum_position_spread_m"],
            0.016,
        )

    def test_yaw_spread_uses_wrapped_angle_distance(self):
        stable_across_wrap = tuple(
            StationaryAmclPoseSample(
                0.0,
                0.0,
                math.pi - 0.004 + 0.002 * index,
                _covariance(),
            )
            for index in range(5)
        )
        unstable = tuple(
            StationaryAmclPoseSample(
                0.0,
                0.0,
                0.01 * index,
                _covariance(),
            )
            for index in range(5)
        )

        self.assertTrue(self._evaluate(stable_across_wrap).ok)
        self.assertFalse(self._evaluate(unstable).ok)

    def test_insufficient_sample_count_fails_closed(self):
        observation = self._evaluate(
            (
                StationaryAmclPoseSample(0.0, 0.0, 0.0),
                StationaryAmclPoseSample(0.0, 0.0, 0.0),
            )
        )

        self.assertFalse(observation.ok)
        self.assertEqual(observation.data["sample_count"], 2)
        self.assertEqual(observation.data["required_sample_count"], 5)

    def test_stable_means_with_high_reported_covariance_fail(self):
        samples = tuple(
            StationaryAmclPoseSample(
                x_m=0.001 * index,
                y_m=0.0,
                yaw_rad=0.0,
                covariance=_covariance(x=0.008843, y=0.008843),
            )
            for index in range(5)
        )

        observation = self._evaluate(samples)

        self.assertFalse(observation.ok)
        self.assertAlmostEqual(
            observation.data["maximum_reported_position_std_m"],
            math.sqrt(0.008843),
        )
        self.assertEqual(observation.data["max_allowed_position_std_m"], 0.015)

    def test_missing_or_nonfinite_covariance_fails_closed(self):
        missing = tuple(
            StationaryAmclPoseSample(0.0, 0.0, 0.0)
            for _ in range(5)
        )
        nonfinite_covariance = list(_covariance())
        nonfinite_covariance[35] = math.nan
        nonfinite = tuple(
            StationaryAmclPoseSample(
                0.0,
                0.0,
                0.0,
                tuple(nonfinite_covariance),
            )
            for _ in range(5)
        )

        missing_observation = self._evaluate(missing)
        nonfinite_observation = self._evaluate(nonfinite)

        self.assertFalse(missing_observation.ok)
        self.assertFalse(missing_observation.data["position_covariance_complete"])
        self.assertFalse(missing_observation.data["yaw_covariance_complete"])
        self.assertFalse(nonfinite_observation.ok)
        self.assertTrue(nonfinite_observation.data["position_covariance_complete"])
        self.assertFalse(nonfinite_observation.data["yaw_covariance_complete"])

    def test_latest_window_can_converge_after_broad_initial_samples(self):
        settling = tuple(
            StationaryAmclPoseSample(
                x_m=0.05 * index,
                y_m=0.0,
                yaw_rad=0.01 * index,
                covariance=_covariance(x=0.09, y=0.01, yaw=0.04),
            )
            for index in range(3)
        )
        converged = tuple(
            StationaryAmclPoseSample(
                x_m=1.0 + 0.001 * index,
                y_m=2.0 - 0.0005 * index,
                yaw_rad=0.2 + 0.002 * index,
                covariance=_covariance(),
            )
            for index in range(5)
        )

        all_samples = settling + converged
        whole_history = self._evaluate(all_samples)
        latest_window = evaluate_latest_stationary_amcl_window(
            all_samples,
            required_sample_count=5,
            max_position_spread_m=0.015,
            max_yaw_spread_rad=0.03,
            max_position_std_m=0.015,
            max_yaw_std_rad=0.03,
        )

        self.assertFalse(whole_history.ok)
        self.assertTrue(latest_window.ok)
        self.assertEqual(latest_window.data["sample_count"], 5)
        self.assertEqual(latest_window.data["total_sample_count"], 8)
        self.assertEqual(latest_window.data["window_start_index"], 3)

    def test_latest_window_remains_fail_closed_until_complete(self):
        samples = tuple(
            StationaryAmclPoseSample(
                x_m=0.0,
                y_m=0.0,
                yaw_rad=0.0,
                covariance=_covariance(),
            )
            for _ in range(4)
        )

        observation = evaluate_latest_stationary_amcl_window(
            samples,
            required_sample_count=5,
            max_position_spread_m=0.015,
            max_yaw_spread_rad=0.03,
            max_position_std_m=0.015,
            max_yaw_std_rad=0.03,
        )

        self.assertFalse(observation.ok)
        self.assertEqual(observation.data["sample_count"], 4)
        self.assertEqual(observation.data["total_sample_count"], 4)


if __name__ == "__main__":
    unittest.main()
