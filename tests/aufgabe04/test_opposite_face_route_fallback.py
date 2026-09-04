from dataclasses import replace
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.real_robot.candidate.opposite_face_route_fallback import (
    evaluate_opposite_face_route_fallback,
    opposite_face_route_attempt,
)
from scripts.aufgabe04.real_robot.candidate.recovery_failure import (
    CandidateStartupRecoveryError,
    RejectedChildFailure,
)
from scripts.aufgabe04.real_robot.execution.child_runner import MotionLegOutcome


class OppositeFaceRouteFallbackTest(unittest.TestCase):
    run_id = "mission_candidate_001_opposite"

    def _uncertainty_outcome(self) -> MotionLegOutcome:
        stop_reason = (
            "odom execution admission failed: route uncertainty budget "
            "exhausted: limiting_segment=segment:0002:0092 "
            "remaining_margin=-0.154957 m"
        )
        return MotionLegOutcome(
            run_id=self.run_id,
            status="preflight_failed",
            stop_reason=stop_reason,
            stop_details={
                "reason": stop_reason,
                "fault_code": "odom_execution_admission_failed",
                "execution_pose_owner": "odom",
                "global_consistency_monitor": "amcl",
                "motion_published": False,
                "fail_closed": True,
                "uncertainty_budget_accepted": False,
                "uncertainty_budget_json": "uncertainty.json",
                "uncertainty_budget_sha256": "d" * 64,
                "route_uncertainty_limiting_segment_id": (
                    "segment:0002:0092"
                ),
                "route_uncertainty_remaining_margin_m": -0.154957,
            },
            motion_published=False,
            returncode=1,
            semantic_log_path=Path("events.jsonl"),
            dry_uncertainty_budget_path=Path("uncertainty.json"),
        )

    @staticmethod
    def _error(
        outcome: MotionLegOutcome,
        *,
        phase: str = "outcome_rejection",
    ) -> CandidateStartupRecoveryError:
        rejected = RejectedChildFailure.from_outcome(
            outcome,
            policy_reason="outcome is not an eligible startup-segment mismatch",
            preserve_child_reason=False,
        )
        return CandidateStartupRecoveryError(
            rejected.rejection_message(),
            phase=phase,
            rejected_child=rejected,
        )

    def test_accepts_only_exact_initial_no_motion_no_permit_rejection(self):
        decision = evaluate_opposite_face_route_fallback(
            self._error(self._uncertainty_outcome()),
            expected_initial_run_id=self.run_id,
        )

        self.assertTrue(decision.eligible)
        self.assertEqual(
            decision.reason,
            "new_standoff_route_dry_preflight_allowed",
        )
        self.assertEqual(decision.rejected_run_id, self.run_id)
        self.assertAlmostEqual(decision.remaining_margin_m, -0.154957)
        self.assertEqual(decision.limiting_segment_id, "segment:0002:0092")
        self.assertFalse(decision.to_event_fields()["motion_permit_issued"])
        self.assertFalse(
            decision.to_event_fields()["motion_continues_authorized"]
        )

    def test_rejects_unsafe_or_inexact_rejection_evidence(self):
        base = self._uncertainty_outcome()
        cases = (
            (
                "wrong_recovery_phase",
                self._error(base, phase="budget_exhausted"),
                self.run_id,
            ),
            (
                "startup_reseal_child_is_not_initial_attempt",
                self._error(
                    replace(
                        base,
                        run_id=self.run_id + "_startup_reseal_001",
                    )
                ),
                self.run_id,
            ),
            (
                "motion_published",
                self._error(replace(base, motion_published=True)),
                self.run_id,
            ),
            (
                "permit_reported",
                self._error(
                    replace(
                        base,
                        mission_leg_motion_permit_path=Path("permit.json"),
                        mission_leg_motion_permit_sha256="e" * 64,
                    )
                ),
                self.run_id,
            ),
            (
                "nested_motion_not_false",
                self._error(
                    replace(
                        base,
                        stop_details={
                            **base.stop_details,
                            "motion_published": True,
                        },
                    )
                ),
                self.run_id,
            ),
            (
                "budget_not_rejected",
                self._error(
                    replace(
                        base,
                        stop_details={
                            **base.stop_details,
                            "uncertainty_budget_accepted": True,
                        },
                    )
                ),
                self.run_id,
            ),
            (
                "nonnegative_margin",
                self._error(
                    replace(
                        base,
                        stop_details={
                            **base.stop_details,
                            "route_uncertainty_remaining_margin_m": 0.0,
                        },
                    )
                ),
                self.run_id,
            ),
            (
                "nan_margin",
                self._error(
                    replace(
                        base,
                        stop_details={
                            **base.stop_details,
                            "route_uncertainty_remaining_margin_m": float("nan"),
                        },
                    )
                ),
                self.run_id,
            ),
            (
                "missing_limiting_segment",
                self._error(
                    replace(
                        base,
                        stop_details={
                            **base.stop_details,
                            "route_uncertainty_limiting_segment_id": "",
                        },
                    )
                ),
                self.run_id,
            ),
        )
        for name, error, expected_run_id in cases:
            with self.subTest(name=name):
                decision = evaluate_opposite_face_route_fallback(
                    error,
                    expected_initial_run_id=expected_run_id,
                )
                self.assertFalse(decision.eligible)

    def test_each_standoff_attempt_has_fresh_route_and_child_identity(self):
        first = opposite_face_route_attempt(
            base_run_id=self.run_id,
            base_source_root=Path("opposite_face_source"),
            attempt_index=0,
            approach_offset_m=0.60,
        )
        second = opposite_face_route_attempt(
            base_run_id=self.run_id,
            base_source_root=Path("opposite_face_source"),
            attempt_index=1,
            approach_offset_m=0.55,
        )

        self.assertNotEqual(first.run_id, second.run_id)
        self.assertNotEqual(first.source_root, second.source_root)
        self.assertEqual(first.approach_offset_m, 0.60)
        self.assertEqual(second.approach_offset_m, 0.55)
        self.assertFalse(first.to_event_fields()["motion_authorized"])
        self.assertTrue(first.to_event_fields()["route_limits_unchanged"])


if __name__ == "__main__":
    unittest.main()
