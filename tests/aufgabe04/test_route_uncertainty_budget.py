import json
import math
import unittest

from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import (
    PlanarCovariance,
    RouteClearanceSegment,
    UNCERTAINTY_BUDGET_EXHAUSTED,
    evaluate_route_uncertainty_budget,
    evaluate_segment_uncertainty_budget,
    projected_sigma_m,
    radial_sigma_m,
    uncertainty_budget_evidence_sha256,
)


def segment(**overrides):
    values = {
        "segment_id": "s0",
        "raw_centerline_clearance_m": 1.0,
        "robot_radius_m": 0.2,
        "collision_margin_m": 0.03,
        "fixed_odom_tracking_bound_m": 0.02,
        "empirical_odom_drift_bound_m": 0.01,
        "braking_latency_distance_m": 0.04,
        "localization_sigma_multiplier": 2.0,
        "heading_contribution_m": 0.05,
        "covariance": PlanarCovariance(0.01, 0.0, 0.04),
        "segment_normal_x": 1.0,
        "segment_normal_y": 0.0,
        "is_corner": False,
    }
    values.update(overrides)
    return RouteClearanceSegment(**values)


class RouteUncertaintyBudgetTest(unittest.TestCase):
    def test_anisotropic_covariance_depends_on_segment_normal(self):
        covariance = PlanarCovariance(xx_m2=0.04, xy_m2=0.0, yy_m2=0.01)

        self.assertAlmostEqual(projected_sigma_m(covariance, 1.0, 0.0), 0.2)
        self.assertAlmostEqual(projected_sigma_m(covariance, 0.0, 1.0), 0.1)

    def test_rotated_covariance_projection_normalizes_arbitrary_normal(self):
        # Eigenvalues 0.04 and 0.01 with the major axis rotated by 45 degrees.
        covariance = PlanarCovariance(xx_m2=0.025, xy_m2=0.015, yy_m2=0.025)

        self.assertAlmostEqual(projected_sigma_m(covariance, 7.0, 7.0), 0.2)
        self.assertAlmostEqual(projected_sigma_m(covariance, -3.0, 3.0), 0.1)

    def test_corner_uses_worst_axis_radial_sigma(self):
        covariance = PlanarCovariance(xx_m2=0.025, xy_m2=0.015, yy_m2=0.025)
        corner = evaluate_segment_uncertainty_budget(
            segment(
                covariance=covariance,
                is_corner=True,
                segment_normal_x=-1.0,
                segment_normal_y=1.0,
            )
        )

        self.assertAlmostEqual(radial_sigma_m(covariance), 0.2)
        self.assertEqual(
            corner.evidence["geometry"]["projection_mode"],
            "corner_worst_axis",
        )
        self.assertAlmostEqual(corner.evidence["localization"]["sigma_m"], 0.2)

    def test_every_budget_term_is_explicit_and_deducted_once(self):
        base = segment(
            raw_centerline_clearance_m=2.0,
            robot_radius_m=0.0,
            collision_margin_m=0.0,
            fixed_odom_tracking_bound_m=0.0,
            empirical_odom_drift_bound_m=0.0,
            braking_latency_distance_m=0.0,
            localization_sigma_multiplier=0.0,
            heading_contribution_m=0.0,
            covariance=PlanarCovariance(0.01, 0.0, 0.01),
        )
        terms = (
            "robot_radius_m",
            "collision_margin_m",
            "fixed_odom_tracking_bound_m",
            "empirical_odom_drift_bound_m",
            "braking_latency_distance_m",
            "heading_contribution_m",
        )
        for term in terms:
            with self.subTest(term=term):
                decision = evaluate_segment_uncertainty_budget(
                    segment(**{**base.__dict__, term: 0.25})
                )
                self.assertAlmostEqual(decision.remaining_margin_m, 1.75)
                self.assertEqual(decision.evidence["budget_m"][term], 0.25)

        localization = evaluate_segment_uncertainty_budget(
            segment(**{**base.__dict__, "localization_sigma_multiplier": 2.5})
        )
        self.assertAlmostEqual(
            localization.evidence["budget_m"]["projected_localization_term_m"],
            0.25,
        )
        self.assertAlmostEqual(localization.remaining_margin_m, 1.75)

    def test_budget_sum_uses_raw_centerline_clearance(self):
        decision = evaluate_segment_uncertainty_budget(segment())

        # 0.20 radius + 0.03 collision + 0.02 tracking + 0.01 drift
        # + 0.04 braking + 2*0.10 projected sigma + 0.05 heading.
        self.assertAlmostEqual(
            decision.evidence["budget_m"]["required_clearance_m"], 0.55
        )
        self.assertAlmostEqual(decision.remaining_margin_m, 0.45)
        self.assertTrue(decision.accepted)

    def test_exact_boundary_rejects_and_positive_residual_accepts(self):
        exact = evaluate_segment_uncertainty_budget(
            segment(
                raw_centerline_clearance_m=0.5,
                robot_radius_m=0.5,
                collision_margin_m=0.0,
                fixed_odom_tracking_bound_m=0.0,
                empirical_odom_drift_bound_m=0.0,
                braking_latency_distance_m=0.0,
                localization_sigma_multiplier=0.0,
                heading_contribution_m=0.0,
                covariance=PlanarCovariance(0.0, 0.0, 0.0),
            )
        )
        positive = evaluate_segment_uncertainty_budget(
            segment(
                raw_centerline_clearance_m=math.nextafter(0.5, math.inf),
                robot_radius_m=0.5,
                collision_margin_m=0.0,
                fixed_odom_tracking_bound_m=0.0,
                empirical_odom_drift_bound_m=0.0,
                braking_latency_distance_m=0.0,
                localization_sigma_multiplier=0.0,
                heading_contribution_m=0.0,
                covariance=PlanarCovariance(0.0, 0.0, 0.0),
            )
        )

        self.assertEqual(exact.remaining_margin_m, 0.0)
        self.assertFalse(exact.accepted)
        self.assertEqual(exact.reason, UNCERTAINTY_BUDGET_EXHAUSTED)
        self.assertTrue(positive.accepted)
        self.assertGreater(positive.remaining_margin_m, 0.0)

    def test_missing_nonfinite_and_ambiguous_covariances_fail_closed(self):
        malformed = (
            None,
            {"xx_m2": 0.01, "xy_m2": 0.0},
            {"xx_m2": math.nan, "xy_m2": 0.0, "yy_m2": 0.01},
            {"xx_m2": 0.01, "xy_m2": 0.02, "yy_m2": 0.01},
            {"xx_m2": 0.0, "xy_m2": 1.0e-10, "yy_m2": 0.0},
            {
                "xx_m2": 0.01,
                "xy_m2": 0.0,
                "yy_m2": 0.01,
                "yx_m2": 0.0,
            },
        )
        for covariance in malformed:
            with self.subTest(covariance=covariance):
                decision = evaluate_segment_uncertainty_budget(
                    segment(covariance=covariance)
                )
                self.assertFalse(decision.accepted)
                self.assertEqual(decision.reason, UNCERTAINTY_BUDGET_EXHAUSTED)
                self.assertIsNone(decision.remaining_margin_m)
                self.assertFalse(decision.evidence["validation"]["ok"])
                json.dumps(decision.evidence, allow_nan=False)

    def test_nonfinite_term_and_missing_normal_fail_closed(self):
        for broken in (
            {"empirical_odom_drift_bound_m": math.inf},
            {
                "robot_radius_m": 1.0e308,
                "collision_margin_m": 1.0e308,
            },
            {"segment_normal_x": None},
            {"segment_normal_x": 0.0, "segment_normal_y": 0.0},
        ):
            with self.subTest(broken=broken):
                decision = evaluate_segment_uncertainty_budget(segment(**broken))
                self.assertFalse(decision.accepted)
                self.assertEqual(decision.reason, UNCERTAINTY_BUDGET_EXHAUSTED)
                self.assertIsNone(decision.remaining_margin_m)
                json.dumps(decision.evidence, allow_nan=False)

    def test_route_rejects_when_one_segment_exhausts_budget(self):
        decision = evaluate_route_uncertainty_budget(
            (
                segment(segment_id="safe", raw_centerline_clearance_m=1.0),
                segment(segment_id="tight", raw_centerline_clearance_m=0.4),
            )
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, UNCERTAINTY_BUDGET_EXHAUSTED)
        self.assertEqual(decision.limiting_segment_id, "tight")
        self.assertLess(decision.remaining_margin_m, 0.0)
        self.assertEqual(len(decision.segment_decisions), 2)

    def test_empty_route_and_duplicate_ids_fail_closed(self):
        empty = evaluate_route_uncertainty_budget(())
        duplicate = evaluate_route_uncertainty_budget(
            (segment(), segment())
        )

        self.assertFalse(empty.accepted)
        self.assertIsNone(empty.remaining_margin_m)
        self.assertFalse(duplicate.accepted)
        self.assertIn("duplicate_segment_id", duplicate.evidence["validation"]["errors"])

    def test_evidence_is_finite_json_and_content_hash_is_deterministic(self):
        first = evaluate_route_uncertainty_budget((segment(),))
        second = evaluate_route_uncertainty_budget((segment(),))

        encoded = json.dumps(
            first.evidence,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.assertIn('"probability_guarantee":false', encoded)
        self.assertEqual(
            uncertainty_budget_evidence_sha256(first),
            uncertainty_budget_evidence_sha256(second.evidence),
        )
        self.assertEqual(len(uncertainty_budget_evidence_sha256(first)), 64)


if __name__ == "__main__":
    unittest.main()
