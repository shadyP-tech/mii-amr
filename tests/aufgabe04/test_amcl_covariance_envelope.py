import json
import math
import unittest

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import (
    PlanarCovariance,
)
from scripts.aufgabe04.navigation.localization.amcl_covariance_envelope import (
    conservative_amcl_covariance_envelope,
)


def amcl_sample(
    *,
    xx_m2=0.0,
    xy_m2=0.0,
    yx_m2=0.0,
    yy_m2=0.0,
    yaw_variance_rad2=0.0,
):
    covariance = [0.0] * 36
    covariance[0] = xx_m2
    covariance[1] = xy_m2
    covariance[6] = yx_m2
    covariance[7] = yy_m2
    covariance[35] = yaw_variance_rad2
    return {"covariance": covariance}


class AmclCovarianceEnvelopeTest(unittest.TestCase):
    def test_envelope_uses_maximum_position_eigenvalue_and_yaw_variance(self):
        samples = (
            amcl_sample(
                xx_m2=0.04,
                xy_m2=0.03,
                yx_m2=0.03,
                yy_m2=0.04,
                yaw_variance_rad2=0.0004,
            ),
            amcl_sample(
                xx_m2=0.06,
                yy_m2=0.01,
                yaw_variance_rad2=0.0025,
            ),
        )

        covariance, heading_sigma_rad, evidence = (
            conservative_amcl_covariance_envelope(samples)
        )

        self.assertEqual(covariance, PlanarCovariance(0.07, 0.0, 0.07))
        self.assertAlmostEqual(heading_sigma_rad, 0.05)
        self.assertEqual(
            evidence,
            {
                "envelope_kind": "isotropic_maximum_eigenvalue",
                "sample_count": 2,
                "samples": [
                    {
                        "sample_index": 0,
                        "xx_m2": 0.04,
                        "xy_m2": 0.03,
                        "yy_m2": 0.04,
                        "yaw_variance_rad2": 0.0004,
                        "largest_position_variance_m2": 0.07,
                    },
                    {
                        "sample_index": 1,
                        "xx_m2": 0.06,
                        "xy_m2": 0.0,
                        "yy_m2": 0.01,
                        "yaw_variance_rad2": 0.0025,
                        "largest_position_variance_m2": 0.06,
                    },
                ],
            },
        )

    def test_nearly_symmetric_cross_covariance_is_averaged(self):
        sample = amcl_sample(
            xx_m2=0.04,
            xy_m2=0.010000001,
            yx_m2=0.009999999,
            yy_m2=0.04,
        )

        covariance, _, evidence = conservative_amcl_covariance_envelope(
            (sample,)
        )

        self.assertAlmostEqual(covariance.xx_m2, 0.05)
        self.assertAlmostEqual(covariance.yy_m2, 0.05)
        self.assertEqual(covariance.xy_m2, 0.0)
        self.assertAlmostEqual(evidence["samples"][0]["xy_m2"], 0.01)

    def test_evidence_is_finite_json_and_deterministic(self):
        samples = (
            amcl_sample(
                xx_m2=0.03,
                xy_m2=-0.01,
                yx_m2=-0.01,
                yy_m2=0.02,
                yaw_variance_rad2=0.0009,
            ),
        )

        first = conservative_amcl_covariance_envelope(samples)
        second = conservative_amcl_covariance_envelope(samples)

        self.assertEqual(first, second)
        json.dumps(first[2], allow_nan=False, sort_keys=True)
        self.assertEqual(payload_sha256(first[2]), payload_sha256(second[2]))

    def test_empty_samples_fail_closed(self):
        with self.assertRaisesRegex(
            ValueError,
            "preflight has no accepted stationary AMCL samples",
        ):
            conservative_amcl_covariance_envelope(())

    def test_asymmetric_covariance_fails_closed(self):
        sample = amcl_sample(
            xx_m2=0.04,
            xy_m2=0.01,
            yx_m2=0.02,
            yy_m2=0.04,
        )

        with self.assertRaisesRegex(ValueError, "covariance is asymmetric"):
            conservative_amcl_covariance_envelope((sample,))

    def test_nonfinite_covariance_fails_closed(self):
        for index, value in ((0, math.nan), (5, math.inf), (35, -math.inf)):
            with self.subTest(index=index, value=value):
                sample = amcl_sample()
                sample["covariance"][index] = value

                with self.assertRaisesRegex(
                    ValueError,
                    "covariance is non-finite",
                ):
                    conservative_amcl_covariance_envelope((sample,))

    def test_incomplete_covariance_fails_closed(self):
        malformed_covariances = (
            None,
            (),
            [0.0] * 35,
            [0.0] * 37,
        )
        for covariance in malformed_covariances:
            with self.subTest(covariance=covariance):
                with self.assertRaisesRegex(
                    ValueError,
                    "covariance is incomplete",
                ):
                    conservative_amcl_covariance_envelope(
                        ({"covariance": covariance},)
                    )

    def test_negative_yaw_variance_fails_closed(self):
        sample = amcl_sample(yaw_variance_rad2=-1.0e-9)

        with self.assertRaisesRegex(
            ValueError,
            "yaw covariance is negative",
        ):
            conservative_amcl_covariance_envelope((sample,))


if __name__ == "__main__":
    unittest.main()
