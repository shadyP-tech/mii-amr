import unittest

from scripts.aufgabe04.real_robot.readiness.localization import (
    evaluate_localization_readiness_retry,
    localization_readiness_suffix,
)


def retryable_details():
    return {
        "reason": "route uncertainty exhausted",
        "fault_code": "odom_execution_admission_failed",
        "execution_pose_owner": "odom",
        "global_consistency_monitor": "amcl",
        "motion_published": False,
        "fail_closed": True,
    }


class AutonomousLocalizationReadinessTest(unittest.TestCase):
    def test_accepts_only_fail_closed_no_motion_uncertainty_exhaustion(self):
        decision = evaluate_localization_readiness_retry(
            status="preflight_failed",
            stop_reason=(
                "odom execution admission failed: route uncertainty budget "
                "exhausted: limiting_segment=3 remaining_margin=-0.026910 m"
            ),
            stop_details=retryable_details(),
            motion_published=False,
        )

        self.assertTrue(decision.retryable)
        self.assertEqual(decision.reason, "fresh_no_motion_admission_allowed")

    def test_rejects_motion_and_unrelated_preflight_failures(self):
        base = {
            "status": "preflight_failed",
            "stop_reason": (
                "odom execution admission failed: route uncertainty budget "
                "exhausted: limiting_segment=3 remaining_margin=-0.01 m"
            ),
            "stop_details": retryable_details(),
            "motion_published": False,
        }
        mutations = (
            {"motion_published": True},
            {"status": "stopped"},
            {"stop_reason": "stationary AMCL stability failed"},
            {"stop_details": {**retryable_details(), "fail_closed": False}},
            {
                "stop_details": {
                    **retryable_details(),
                    "fault_code": "some_other_failure",
                }
            },
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assertFalse(
                    evaluate_localization_readiness_retry(
                        **{**base, **mutation}
                    ).retryable
                )

    def test_retry_suffix_is_stable_and_validated(self):
        self.assertEqual(localization_readiness_suffix(0), "")
        self.assertEqual(
            localization_readiness_suffix(2),
            "_localization_readiness_002",
        )
        for invalid in (-1, True, 1.2):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                localization_readiness_suffix(invalid)


if __name__ == "__main__":
    unittest.main()
