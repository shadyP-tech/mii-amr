import unittest

from scripts.aufgabe04.real_robot.candidate.no_motion_route_rejection import (
    classify_no_motion_preflight_failure,
    classify_no_motion_route_uncertainty_rejection,
)


class NoMotionRouteRejectionTest(unittest.TestCase):
    @staticmethod
    def _route_details() -> tuple[str, dict[str, object]]:
        reason = (
            "odom execution admission failed: route uncertainty budget "
            "exhausted: limiting_segment=segment:0001:0072 "
            "remaining_margin=-0.026479 m"
        )
        return reason, {
            "reason": reason,
            "fault_code": "odom_execution_admission_failed",
            "execution_pose_owner": "odom",
            "global_consistency_monitor": "amcl",
            "motion_published": False,
            "fail_closed": True,
            "uncertainty_budget_accepted": False,
            "route_uncertainty_limiting_segment_id": "segment:0001:0072",
            "route_uncertainty_remaining_margin_m": -0.026479,
        }

    def test_runtime_classifies_any_structurally_valid_no_motion_preflight(self):
        reason = "camera and lidar timing preflight failed"
        decision = classify_no_motion_preflight_failure(
            status="preflight_failed",
            stop_reason=reason,
            stop_details={
                "reason": reason,
                "motion_published": False,
                "fail_closed": True,
            },
            motion_published=False,
            returncode=2,
        )

        self.assertTrue(decision.applies)
        self.assertTrue(decision.evidence_valid)
        self.assertEqual(
            decision.reason,
            "structured_no_motion_preflight_failure",
        )

    def test_completed_or_post_motion_outcome_still_requires_permit_gate(self):
        for status, motion_published in (
            ("completed", True),
            ("stopped", True),
            ("preflight_failed", True),
        ):
            with self.subTest(status=status):
                decision = classify_no_motion_preflight_failure(
                    status=status,
                    stop_reason="child outcome",
                    stop_details={},
                    motion_published=motion_published,
                )
                self.assertFalse(decision.applies)
                self.assertEqual(
                    decision.reason,
                    "runtime_permit_validation_required",
                )

    def test_no_motion_preflight_rejects_permit_and_unbound_evidence(self):
        reason = "preflight rejected"
        cases = (
            (
                "permit",
                {"issued_motion_permit_kinds": ("runtime_localization",)},
                "no_motion_preflight_reported_motion_permit",
            ),
            (
                "reason_binding",
                {
                    "stop_details": {
                        "reason": "different",
                        "motion_published": False,
                        "fail_closed": True,
                    }
                },
                "no_motion_preflight_reason_binding_mismatch",
            ),
        )
        for name, overrides, expected_reason in cases:
            with self.subTest(name=name):
                values = {
                    "status": "preflight_failed",
                    "stop_reason": reason,
                    "stop_details": {
                        "reason": reason,
                        "motion_published": False,
                        "fail_closed": True,
                    },
                    "motion_published": False,
                    "returncode": 2,
                    **overrides,
                }
                decision = classify_no_motion_preflight_failure(**values)
                self.assertTrue(decision.applies)
                self.assertFalse(decision.evidence_valid)
                self.assertEqual(decision.reason, expected_reason)

    def test_route_classifier_requires_exact_negative_margin_rejection(self):
        reason, details = self._route_details()
        accepted = classify_no_motion_route_uncertainty_rejection(
            status="preflight_failed",
            stop_reason=reason,
            stop_details=details,
            motion_published=False,
        )

        self.assertTrue(accepted.eligible)
        self.assertAlmostEqual(accepted.remaining_margin_m, -0.026479)
        self.assertEqual(accepted.limiting_segment_id, "segment:0001:0072")

        cases = (
            (
                "permit",
                details,
                {"issued_motion_permit_kinds": ("runtime_localization",)},
                "motion_permit_was_issued",
            ),
            (
                "nonnegative_margin",
                {
                    **details,
                    "route_uncertainty_remaining_margin_m": 0.0,
                },
                {},
                "negative_uncertainty_margin_missing",
            ),
            (
                "not_uncertainty_rejection",
                {
                    **details,
                    "fault_code": "other_failure",
                },
                {},
                "localization_readiness_fault_code_not_retryable",
            ),
        )
        for name, candidate_details, overrides, expected_reason in cases:
            with self.subTest(name=name):
                rejected = classify_no_motion_route_uncertainty_rejection(
                    status="preflight_failed",
                    stop_reason=reason,
                    stop_details=candidate_details,
                    motion_published=False,
                    **overrides,
                )
                self.assertFalse(rejected.eligible)
                self.assertEqual(rejected.reason, expected_reason)


if __name__ == "__main__":
    unittest.main()
