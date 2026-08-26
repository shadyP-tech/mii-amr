from __future__ import annotations

import unittest

from scripts.aufgabe04.real_robot.autonomous_runner.failure_reporting import (
    build_failed_closed_mission_summary,
)
from scripts.aufgabe04.real_robot.candidate.observation_deferral import (
    CandidateApproachIncompleteError,
)


class AutonomousFailureReportingTest(unittest.TestCase):
    def test_bounded_candidate_exhaustion_remains_structured_and_fail_closed(self):
        error = CandidateApproachIncompleteError(
            resolved_candidate_uids=("candidate_b",),
            unresolved_candidate_uids=("candidate_a",),
            attempt_evidence=(),
            max_attempts_per_candidate=2,
            final_pass_index=1,
        )

        failure = build_failed_closed_mission_summary(
            run_mode="execute-exact-two-camera",
            error=error,
        )

        self.assertEqual(failure["status"], "failed_closed")
        self.assertEqual(failure["failure_phase"], "candidate_approach_incomplete")
        self.assertEqual(failure["resolved_candidate_uids"], ["candidate_b"])
        self.assertEqual(failure["unresolved_candidate_uids"], ["candidate_a"])
        self.assertEqual(failure["max_candidate_observation_attempts"], 2)
        self.assertFalse(failure["motion_continues_authorized"])

    def test_structured_fields_cannot_override_terminal_invariants(self):
        class MisleadingError(RuntimeError):
            def to_failure_fields(self):
                return {
                    "schema_version": 99,
                    "status": "completed",
                    "run_mode": "execute-full",
                    "reason": "hidden",
                    "motion_continues_authorized": True,
                    "diagnostic": "retained",
                }

        error = MisleadingError("original failure")
        failure = build_failed_closed_mission_summary(
            run_mode="execute-exact-two-camera",
            error=error,
        )

        self.assertEqual(failure["schema_version"], 1)
        self.assertEqual(failure["status"], "failed_closed")
        self.assertEqual(failure["run_mode"], "execute-exact-two-camera")
        self.assertEqual(failure["reason"], "original failure")
        self.assertFalse(failure["motion_continues_authorized"])
        self.assertEqual(failure["diagnostic"], "retained")

    def test_broken_structured_reporter_does_not_hide_original_failure(self):
        class BrokenReporter(RuntimeError):
            def to_failure_fields(self):
                raise ValueError("secondary reporter failure")

        failure = build_failed_closed_mission_summary(
            run_mode="execute-exact-two-camera",
            error=BrokenReporter("primary failure"),
        )

        self.assertEqual(failure["reason"], "primary failure")
        self.assertEqual(failure["status"], "failed_closed")
        self.assertFalse(failure["motion_continues_authorized"])


if __name__ == "__main__":
    unittest.main()
