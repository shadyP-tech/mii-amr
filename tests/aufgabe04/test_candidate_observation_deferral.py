import json
from pathlib import Path
import unittest

from scripts.aufgabe04.real_robot.candidate.observation_deferral import (
    CandidateObservationDeferralLedger,
    CandidateObservationUnavailableError,
)


def _unavailable(
    candidate_uid: str,
    *,
    observer_attempt: int = 0,
    state: str = "lidar_target_mismatch",
) -> CandidateObservationUnavailableError:
    return CandidateObservationUnavailableError(
        candidate_uid=candidate_uid,
        observation_attempt_index=observer_attempt,
        reason=f"passive observer ended in {state}",
        process_evidence={
            "completion_kind": "deadline",
            "returncode": 130,
            "evidence_path": Path("candidate/observer_process.json"),
        },
        status_evidence={
            "state": state,
            "axis_consensus": {"sample_count": 0, "required_sample_count": 7},
        },
    )


class CandidateObservationUnavailableErrorTest(unittest.TestCase):
    def test_failure_fields_are_json_ready_and_motion_neutral(self):
        error = _unavailable("survey_candidate_0005")

        fields = error.to_failure_fields()

        self.assertEqual(fields["failure_phase"], "candidate_observation")
        self.assertEqual(fields["motion_capability"], "none")
        self.assertFalse(fields["motion_continues_authorized"])
        self.assertTrue(fields["fail_closed"])
        self.assertEqual(
            fields["observer_process_evidence"]["evidence_path"],
            "candidate/observer_process.json",
        )
        self.assertEqual(
            fields["observer_status_evidence"]["state"],
            "lidar_target_mismatch",
        )
        json.dumps(fields)

    def test_evidence_is_detached_from_mutable_caller_payload(self):
        process = {"completion_kind": "deadline"}
        status = {"state": "lidar_target_mismatch"}
        error = CandidateObservationUnavailableError(
            candidate_uid="candidate_a",
            observation_attempt_index=0,
            reason="target-local timeout",
            process_evidence=process,
            status_evidence=status,
        )

        process["completion_kind"] = "artifact"
        status["state"] = "recommended"

        self.assertEqual(
            error.process_evidence["completion_kind"],
            "deadline",
        )
        self.assertEqual(
            error.status_evidence["state"],
            "lidar_target_mismatch",
        )


class CandidateObservationDeferralLedgerTest(unittest.TestCase):
    def test_failure_is_excluded_until_other_candidates_finish_the_pass(self):
        ledger = CandidateObservationDeferralLedger(
            ["candidate_c", "candidate_a", "candidate_b"]
        )

        selection = ledger.select("candidate_c")
        self.assertEqual(selection.attempt_number, 1)
        ledger.mark_unavailable(_unavailable("candidate_c"))

        state = ledger.selection_state()
        self.assertEqual(
            state.eligible_candidate_uids,
            ("candidate_a", "candidate_b"),
        )
        self.assertEqual(state.excluded_candidate_uids, ("candidate_c",))
        with self.assertRaisesRegex(RuntimeError, "not eligible"):
            ledger.select("candidate_c")

        ledger.select("candidate_a")
        ledger.mark_resolved({"qr_id": "QR_001"})
        ledger.select("candidate_b")
        ledger.mark_resolved({"qr_id": "QR_002"})

        self.assertEqual(ledger.selection_state().eligible_candidate_uids, ())
        self.assertTrue(ledger.advance_pass())
        retry_state = ledger.selection_state()
        self.assertEqual(retry_state.pass_index, 1)
        self.assertEqual(retry_state.eligible_candidate_uids, ("candidate_c",))

    def test_default_two_attempt_bound_ends_in_structured_incomplete_error(self):
        ledger = CandidateObservationDeferralLedger(
            ["candidate_b", "candidate_a"]
        )

        ledger.select("candidate_b")
        ledger.mark_unavailable(_unavailable("candidate_b"))
        ledger.select("candidate_a")
        ledger.mark_resolved({"qr_id": "QR_001"})
        self.assertTrue(ledger.advance_pass())
        ledger.select("candidate_b")
        ledger.mark_unavailable(
            _unavailable("candidate_b", observer_attempt=1)
        )

        state = ledger.selection_state()
        self.assertTrue(state.terminal_incomplete)
        self.assertFalse(ledger.advance_pass())
        error = ledger.incomplete_error()
        fields = error.to_failure_fields()

        self.assertEqual(fields["resolved_candidate_uids"], ["candidate_a"])
        self.assertEqual(fields["unresolved_candidate_uids"], ["candidate_b"])
        self.assertEqual(fields["max_candidate_observation_attempts"], 2)
        self.assertEqual(fields["final_observation_pass_index"], 1)
        unavailable = [
            attempt
            for attempt in fields["candidate_observation_attempts"]
            if attempt["outcome"] == "unavailable"
        ]
        self.assertEqual(len(unavailable), 2)
        self.assertEqual(
            [item["candidate_observation_attempt_number"] for item in unavailable],
            [1, 2],
        )
        self.assertEqual(
            [item["observation_pass_index"] for item in unavailable],
            [0, 1],
        )
        self.assertEqual(fields["motion_capability"], "none")
        self.assertFalse(fields["motion_continues_authorized"])
        self.assertTrue(fields["fail_closed"])
        json.dumps(fields)

    def test_cannot_skip_eligible_candidates_or_defer_untyped_errors(self):
        ledger = CandidateObservationDeferralLedger(["candidate_a", "candidate_b"])

        with self.assertRaisesRegex(RuntimeError, "remain eligible"):
            ledger.advance_pass()
        ledger.select("candidate_a")
        with self.assertRaisesRegex(TypeError, "only CandidateObservationUnavailable"):
            ledger.mark_unavailable(  # type: ignore[arg-type]
                RuntimeError("generic failure")
            )
        with self.assertRaisesRegex(RuntimeError, "active selection"):
            ledger.advance_pass()

    def test_empty_ledger_is_complete_and_cannot_create_incomplete_error(self):
        ledger = CandidateObservationDeferralLedger([])

        state = ledger.selection_state()

        self.assertTrue(state.complete)
        self.assertFalse(state.terminal_incomplete)
        self.assertFalse(ledger.advance_pass())
        with self.assertRaisesRegex(RuntimeError, "ledger is complete"):
            ledger.incomplete_error()


if __name__ == "__main__":
    unittest.main()
