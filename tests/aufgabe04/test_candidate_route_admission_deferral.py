from dataclasses import replace
from pathlib import Path
import json
import unittest

from scripts.aufgabe04.real_robot.candidate.route_admission_deferral import (
    CandidateRouteAdmissionDeferralLedger,
    CandidateRouteAdmissionIncompleteError,
    evaluate_candidate_route_admission_deferral,
)
from scripts.aufgabe04.real_robot.candidate.recovery_failure import (
    CandidateStartupRecoveryError,
    RejectedChildFailure,
)
from scripts.aufgabe04.real_robot.execution.child_runner import MotionLegOutcome


RUN_ID = "mission_candidate_000"


def _uncertainty_outcome(*, run_id: str = RUN_ID) -> MotionLegOutcome:
    stop_reason = (
        "odom execution admission failed: route uncertainty budget "
        "exhausted: limiting_segment=segment:0002:0092 "
        "remaining_margin=-0.154957 m"
    )
    return MotionLegOutcome(
        run_id=run_id,
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
            "route_uncertainty_limiting_segment_id": "segment:0002:0092",
            "route_uncertainty_remaining_margin_m": -0.154957,
        },
        motion_published=False,
        returncode=1,
        semantic_log_path=Path("events.jsonl"),
        dry_uncertainty_budget_path=Path("uncertainty.json"),
    )


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


class CandidateRouteAdmissionDeferralDecisionTest(unittest.TestCase):
    def test_accepts_only_initial_no_motion_no_permit_route_uncertainty(self):
        decision = evaluate_candidate_route_admission_deferral(
            _error(_uncertainty_outcome()),
            expected_initial_run_id=RUN_ID,
        )

        self.assertTrue(decision.eligible)
        self.assertEqual(
            decision.reason,
            "next_candidate_route_dry_preflight_allowed",
        )
        self.assertEqual(decision.rejected_run_id, RUN_ID)
        self.assertAlmostEqual(decision.remaining_margin_m, -0.154957)
        self.assertEqual(decision.limiting_segment_id, "segment:0002:0092")
        self.assertFalse(decision.to_event_fields()["motion_authorized"])

    def test_rejects_post_motion_permit_or_replacement_child_evidence(self):
        base = _uncertainty_outcome()
        cases = (
            (
                "wrong_phase",
                _error(base, phase="budget_exhausted"),
                RUN_ID,
            ),
            (
                "replacement_child",
                _error(replace(base, run_id=f"{RUN_ID}_startup_reseal_001")),
                RUN_ID,
            ),
            (
                "motion_published",
                _error(replace(base, motion_published=True)),
                RUN_ID,
            ),
            (
                "permit_reported",
                _error(
                    replace(
                        base,
                        mission_leg_motion_permit_path=Path("permit.json"),
                        mission_leg_motion_permit_sha256="e" * 64,
                    )
                ),
                RUN_ID,
            ),
        )
        for name, error, expected_run_id in cases:
            with self.subTest(name=name):
                decision = evaluate_candidate_route_admission_deferral(
                    error,
                    expected_initial_run_id=expected_run_id,
                )
                self.assertFalse(decision.eligible)


class CandidateRouteAdmissionDeferralLedgerTest(unittest.TestCase):
    def test_rejected_candidate_is_excluded_until_next_route_pass(self):
        ledger = CandidateRouteAdmissionDeferralLedger(
            ["candidate_b", "candidate_a"],
            max_attempts_per_candidate=2,
        )
        error = _error(_uncertainty_outcome())
        decision = evaluate_candidate_route_admission_deferral(
            error,
            expected_initial_run_id=RUN_ID,
        )
        self.assertIsNotNone(error.rejected_child)

        attempt = ledger.mark_rejected(
            candidate_uid="candidate_a",
            rejected_child=error.rejected_child,
            decision=decision,
        )

        self.assertEqual(attempt.attempt_number, 1)
        state = ledger.selection_state(["candidate_a", "candidate_b"])
        self.assertEqual(state.eligible_candidate_uids, ("candidate_b",))
        self.assertEqual(state.excluded_candidate_uids, ("candidate_a",))
        json.dumps(attempt.to_dict())

        self.assertTrue(ledger.advance_pass(["candidate_a"]))
        retry_state = ledger.selection_state(["candidate_a"])
        self.assertEqual(retry_state.pass_index, 1)
        self.assertEqual(retry_state.eligible_candidate_uids, ("candidate_a",))

    def test_attempt_bound_ends_in_fail_closed_structured_error(self):
        ledger = CandidateRouteAdmissionDeferralLedger(
            ["candidate_a"],
            max_attempts_per_candidate=1,
        )
        error = _error(_uncertainty_outcome())
        decision = evaluate_candidate_route_admission_deferral(
            error,
            expected_initial_run_id=RUN_ID,
        )
        assert error.rejected_child is not None
        ledger.mark_rejected(
            candidate_uid="candidate_a",
            rejected_child=error.rejected_child,
            decision=decision,
        )

        state = ledger.selection_state(["candidate_a"])
        self.assertTrue(state.terminal_incomplete)
        self.assertFalse(ledger.advance_pass(["candidate_a"]))
        with self.assertRaises(CandidateRouteAdmissionIncompleteError) as raised:
            raise ledger.incomplete_error(["candidate_a"])
        fields = raised.exception.to_failure_fields()

        self.assertEqual(
            fields["failure_phase"],
            "candidate_route_admission_incomplete",
        )
        self.assertEqual(fields["unresolved_candidate_uids"], ["candidate_a"])
        self.assertFalse(fields["motion_continues_authorized"])
        self.assertTrue(fields["route_limits_unchanged"])
        self.assertTrue(fields["fail_closed"])
        json.dumps(fields)


if __name__ == "__main__":
    unittest.main()
