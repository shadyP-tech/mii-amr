import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.ros_preflight import (  # noqa: E402
    StationaryAmclPoseSample,
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
            )
            for index in range(5)
        )
        unstable = tuple(
            StationaryAmclPoseSample(0.0, 0.0, 0.01 * index)
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


if __name__ == "__main__":
    unittest.main()
