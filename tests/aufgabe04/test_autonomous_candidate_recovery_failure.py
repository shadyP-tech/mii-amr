from pathlib import Path
from dataclasses import replace
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
    def test_stale_tf_terminal_message_explains_actual_run_without_rewriting_reason(self):
        stop_reason = "TF transform unavailable: map <- odom"
        outcome = replace(
            _stopped_outcome(), stop_reason=stop_reason, motion_published=False,
            stop_details={
                "source": "tf_lookup", "reason": "stale_transform",
                "age_sec": 1.664346432, "max_age_sec": 1.0,
                "initial_tf_acquisition": {
                    "denial_reason": "required_tf_edge_has_non_acquisition_failure",
                    "elapsed_sec": 3.065873957, "maximum_startup_wait_sec": 5.0,
                },
            },
        )
        rejected = RejectedChildFailure.from_outcome(
            outcome, policy_reason="invalid_initial_map_tf_stop",
            preserve_child_reason=True,
        )
        message = rejected.rejection_message()
        self.assertIn("stale_transform; age=1.664s; limit=1.000s", message)
        self.assertIn("required_tf_edge_has_non_acquisition_failure", message)
        self.assertIn("startup=3.066/5.000s", message)
        self.assertIn("fail-closed policy: invalid_initial_map_tf_stop", message)
        self.assertEqual(rejected.reported_reason, stop_reason)
        self.assertEqual(rejected.to_failure_fields()["stop_reason"], stop_reason)

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
