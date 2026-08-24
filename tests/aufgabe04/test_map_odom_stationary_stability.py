import json
import math
import unittest

from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D,
)
from scripts.aufgabe04.navigation.localization.odom_route_adapter import (
    STATIONARY_STABILITY_ACCEPTED,
    STATIONARY_STABILITY_REJECTED,
    evaluate_map_odom_stationary_stability,
)


def _evaluate(
    samples: object | None,
    *,
    translation_limit_m: float = 0.05,
    yaw_limit_rad: float = 0.1,
):
    return evaluate_map_odom_stationary_stability(
        samples,
        max_translation_drift_m=translation_limit_m,
        max_yaw_drift_rad=yaw_limit_rad,
    )


class MapOdomStationaryStabilityTests(unittest.TestCase):
    def test_two_finite_samples_at_exact_thresholds_are_admitted(self) -> None:
        final_sample = PlanarTransform2D(0.0, 0.0, 0.0)
        samples = (
            PlanarTransform2D(0.03, 0.04, 0.1),
            final_sample,
        )

        result = _evaluate(samples)
        evidence = result.to_evidence()

        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "stationary_map_from_odom_stable")
        self.assertEqual(result.decision, STATIONARY_STABILITY_ACCEPTED)
        self.assertEqual(result.frozen_map_from_odom, final_sample)
        self.assertEqual(result.final_sample_index, 1)
        self.assertAlmostEqual(
            result.max_observed_translation_drift_m,
            0.05,
        )
        self.assertAlmostEqual(
            result.max_observed_absolute_yaw_drift_rad,
            0.1,
        )
        self.assertFalse(evidence["fail_closed"])
        self.assertEqual(evidence["sample_count"], 2)
        self.assertEqual(evidence["minimum_sample_count"], 2)
        self.assertEqual(
            evidence["threshold_semantics"],
            "accept_if_every_sample_is_less_than_or_equal_to_limit_"
            "from_final_sample",
        )
        self.assertEqual(evidence["frozen_map_from_odom"]["x_m"], 0.0)
        self.assertTrue(evidence["sample_comparisons"][0]["accepted"])
        final_comparison = evidence["sample_comparisons"][1]
        self.assertTrue(final_comparison["is_final_sample"])
        self.assertEqual(final_comparison["translation_drift_m"], 0.0)
        self.assertEqual(final_comparison["absolute_yaw_drift_rad"], 0.0)

    def test_every_sample_is_compared_directly_with_final_sample(self) -> None:
        samples = (
            PlanarTransform2D(0.0, 0.0, 0.0),
            PlanarTransform2D(0.04, 0.0, 0.0),
            PlanarTransform2D(0.08, 0.0, 0.0),
        )

        result = _evaluate(samples, translation_limit_m=0.05)
        evidence = result.to_evidence()

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "stationary_map_from_odom_unstable")
        self.assertEqual(result.decision, STATIONARY_STABILITY_REJECTED)
        self.assertEqual(result.unstable_sample_indices, (0,))
        self.assertAlmostEqual(
            result.sample_comparisons[0].translation_drift_m,
            0.08,
        )
        self.assertAlmostEqual(
            result.sample_comparisons[1].translation_drift_m,
            0.04,
        )
        self.assertIsNone(result.frozen_map_from_odom)
        self.assertIsNone(evidence["frozen_map_from_odom"])
        self.assertEqual(evidence["final_map_from_odom"]["x_m"], 0.08)
        self.assertTrue(evidence["fail_closed"])

    def test_relative_yaw_uses_continuity_wrap_semantics(self) -> None:
        samples = (
            PlanarTransform2D(0.0, 0.0, -math.pi + 0.02),
            PlanarTransform2D(0.0, 0.0, math.pi - 0.02),
        )

        result = _evaluate(samples, yaw_limit_rad=0.041)

        self.assertTrue(result.accepted)
        self.assertAlmostEqual(
            result.sample_comparisons[0].relative_yaw_rad,
            0.04,
        )
        self.assertAlmostEqual(
            result.sample_comparisons[0].absolute_yaw_drift_rad,
            0.04,
        )

    def test_missing_malformed_and_insufficient_evidence_fail_closed(self) -> None:
        valid = PlanarTransform2D(0.0, 0.0, 0.0)
        cases = (
            (
                "missing",
                None,
                "stationary_map_from_odom_samples_missing",
                0,
            ),
            (
                "mapping",
                {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
                "stationary_map_from_odom_samples_malformed",
                0,
            ),
            (
                "generator",
                (sample for sample in (valid, valid)),
                "stationary_map_from_odom_samples_malformed",
                0,
            ),
            (
                "empty",
                (),
                "stationary_map_from_odom_samples_insufficient",
                0,
            ),
            (
                "one sample",
                (valid,),
                "stationary_map_from_odom_samples_insufficient",
                1,
            ),
            (
                "wrong sample type",
                (valid, {"x_m": 0.0}),
                "stationary_map_from_odom_samples_malformed",
                2,
            ),
        )

        for name, samples, reason, sample_count in cases:
            with self.subTest(name=name):
                result = _evaluate(samples)
                evidence = result.to_evidence()

                self.assertFalse(result.accepted)
                self.assertEqual(result.reason, reason)
                self.assertEqual(result.decision, STATIONARY_STABILITY_REJECTED)
                self.assertEqual(result.sample_count, sample_count)
                self.assertIsNone(result.frozen_map_from_odom)
                self.assertTrue(evidence["fail_closed"])
                self.assertIsNotNone(evidence["validation_error"])
                self.assertEqual(evidence["sample_comparisons"], [])

    def test_corrupted_nonfinite_transform_is_rejected_deterministically(self) -> None:
        corrupt = object.__new__(PlanarTransform2D)
        object.__setattr__(corrupt, "x_m", math.nan)
        object.__setattr__(corrupt, "y_m", 0.0)
        object.__setattr__(corrupt, "yaw_rad", 0.0)
        samples = (PlanarTransform2D(0.0, 0.0, 0.0), corrupt)

        first = _evaluate(samples).to_evidence()
        second = _evaluate(samples).to_evidence()

        self.assertEqual(first, second)
        self.assertEqual(
            first["validation_error"],
            "samples[1] must be a finite PlanarTransform2D",
        )
        self.assertIsNone(first["final_map_from_odom"])
        json.dumps(first, allow_nan=False, sort_keys=True)

    def test_invalid_threshold_configuration_is_rejected_by_value_error(self) -> None:
        samples = (
            PlanarTransform2D(0.0, 0.0, 0.0),
            PlanarTransform2D(0.0, 0.0, 0.0),
        )

        for translation_limit in (-0.01, math.inf, True):
            with self.subTest(translation_limit=translation_limit):
                with self.assertRaises(ValueError):
                    _evaluate(
                        samples,
                        translation_limit_m=translation_limit,
                    )
        with self.assertRaisesRegex(ValueError, "<= pi"):
            _evaluate(samples, yaw_limit_rad=math.pi + 0.01)


if __name__ == "__main__":
    unittest.main()
