from pathlib import Path
import unittest

from scripts.aufgabe04.real_robot.candidate.recovery_failure import (
    CandidateStartupRecoveryError,
    RejectedChildFailure,
    issued_motion_permit_evidence,
    issued_motion_permit_kinds,
)
from scripts.aufgabe04.real_robot.execution.child_runner import MotionLegOutcome


def _stopped_outcome() -> MotionLegOutcome:
    return MotionLegOutcome(
        run_id="mission_candidate_000_startup_reseal_001",
        status="stopped",
        stop_reason="stuck no progress",
        stop_details={
            "source": "progress_monitor",
            "elapsed_without_progress_sec": 8.1,
        },
        motion_published=True,
        returncode=2,
        semantic_log_path=Path("run_events/candidate.jsonl"),
        mission_leg_motion_permit_sha256="a" * 64,
        startup_reseal_motion_permit_path=Path("permits/startup.json"),
        startup_reseal_motion_permit_sha256="b" * 64,
    )


class CandidateRecoveryFailureTest(unittest.TestCase):
    def test_permit_evidence_is_complete_and_json_ready(self):
        outcome = _stopped_outcome()

        evidence = issued_motion_permit_evidence(outcome)

        self.assertEqual(
            issued_motion_permit_kinds(outcome),
            ("routine_mission_leg", "startup_reseal"),
        )
        self.assertEqual(
            evidence["routine_mission_leg"],
            {"path": None, "sha256": "a" * 64},
        )
        self.assertEqual(
            evidence["startup_reseal"],
            {"path": "permits/startup.json", "sha256": "b" * 64},
        )

    def test_child_reason_is_primary_without_losing_fail_closed_policy(self):
        rejected = RejectedChildFailure.from_outcome(
            _stopped_outcome(),
            policy_reason="rejected candidate run published motion",
            preserve_child_reason=True,
        )

        self.assertEqual(rejected.reported_reason, "stuck no progress")
        self.assertEqual(
            rejected.rejection_message(),
            "candidate startup recovery rejected "
            "mission_candidate_000_startup_reseal_001: stuck no progress; "
            "fail-closed policy: rejected candidate run published motion",
        )
        self.assertEqual(
            rejected.to_event_fields()["rejection_policy_reason"],
            "rejected candidate run published motion",
        )

    def test_policy_reason_remains_primary_when_child_reason_is_not_preserved(self):
        rejected = RejectedChildFailure.from_outcome(
            _stopped_outcome(),
            policy_reason="motion outcome run identity mismatch",
            preserve_child_reason=False,
        )

        self.assertEqual(
            rejected.rejection_message(),
            "candidate startup recovery rejected "
            "mission_candidate_000_startup_reseal_001: "
            "motion outcome run identity mismatch",
        )

    def test_error_exposes_structured_mission_failure_fields(self):
        rejected = RejectedChildFailure.from_outcome(
            _stopped_outcome(),
            policy_reason="rejected candidate run published motion",
            preserve_child_reason=True,
        )
        error = CandidateStartupRecoveryError(
            rejected.rejection_message(),
            phase="outcome_rejection",
            rejected_child=rejected,
        )

        fields = error.to_failure_fields()

        self.assertEqual(fields["failure_phase"], "candidate_startup_recovery")
        self.assertEqual(fields["stop_reason"], "stuck no progress")
        self.assertEqual(fields["stop_details"]["source"], "progress_monitor")
        self.assertTrue(fields["motion_published"])
        self.assertFalse(fields["motion_continues_authorized"])
        self.assertTrue(fields["fail_closed"])


if __name__ == "__main__":
    unittest.main()
