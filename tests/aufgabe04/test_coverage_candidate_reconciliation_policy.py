from __future__ import annotations

import unittest

from scripts.aufgabe04.navigation.coverage.coverage_candidate_reconciliation_policy import (
    NegativeVisibilityRayPolicy,
    evaluate_negative_visibility_ray_policy,
)


class NegativeVisibilityRayPolicyTest(unittest.TestCase):
    def test_bounded_invalid_dropout_is_neutral(self):
        decision = evaluate_negative_visibility_ray_policy(
            ("clear",) * 64 + ("invalid",) * 16,
            distinct_clear_scan_count=64,
            policy=NegativeVisibilityRayPolicy(),
        )

        self.assertTrue(decision.rejection_supported)
        self.assertEqual(decision.reasons, ())
        self.assertAlmostEqual(decision.clear_ray_fraction, 0.8)
        self.assertAlmostEqual(decision.invalid_selected_ray_fraction, 0.2)

    def test_invalid_only_and_excess_dropout_retain(self):
        invalid_only = evaluate_negative_visibility_ray_policy(
            ("invalid",) * 10,
            distinct_clear_scan_count=0,
            policy=NegativeVisibilityRayPolicy(),
        )
        excessive = evaluate_negative_visibility_ray_policy(
            ("clear",) * 3 + ("invalid",) * 2,
            distinct_clear_scan_count=3,
            policy=NegativeVisibilityRayPolicy(),
        )

        self.assertFalse(invalid_only.rejection_supported)
        self.assertIn("selected_scan_ray_invalid", invalid_only.reasons)
        self.assertFalse(excessive.rejection_supported)
        self.assertIn("insufficient_clear_ray_fraction", excessive.reasons)

    def test_positive_or_occluding_return_is_always_a_hard_veto(self):
        for classification, reason in (
            ("matching", "matching_return_supports_candidate"),
            ("nearer", "nearer_return_occludes_candidate"),
        ):
            with self.subTest(classification=classification):
                decision = evaluate_negative_visibility_ray_policy(
                    ("clear",) * 79 + (classification,),
                    distinct_clear_scan_count=79,
                    policy=NegativeVisibilityRayPolicy(),
                )
                self.assertFalse(decision.rejection_supported)
                self.assertIn(reason, decision.reasons)


if __name__ == "__main__":
    unittest.main()
