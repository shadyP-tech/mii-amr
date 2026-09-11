"""The mission adapter must preserve exact authority and canonical claim paths."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.aufgabe04.real_robot.candidate.startup_permit_retirement import retire_candidate_startup_permit
from scripts.aufgabe04.real_robot.candidate.startup_recovery import CandidateRoutineIdentity
from scripts.aufgabe04.real_robot.execution.child_runner import MotionLegOutcome
from tests.aufgabe04.test_startup_route_admission import recorded_stop_details


class CandidateStartupPermitAdapterTest(unittest.TestCase):
    def setUp(self):
        self.identity = CandidateRoutineIdentity("mission", "arena", "candidate_preapproach", 0,
                                                 "survey_candidate_0003", "candidate_000")
        details = recorded_stop_details()
        self.outcome = MotionLegOutcome(
            run_id=self.identity.run_id, status="preflight_failed", returncode=1,
            stop_reason=details["reason"], stop_details=details, motion_published=False,
            semantic_log_path=Path("/evidence/child.jsonl"),
        )
        self.root = Path("/evidence/recovery")
        self.helper = "scripts.aufgabe04.real_robot.candidate.startup_permit_retirement.retire_odom_startup_rejected_permit"

    def test_actual_issued_permit_uses_canonical_claim_slot_for_every_owner(self):
        for fields, kind in (
            ("mission_leg_motion_permit", "mission_leg"),
            ("startup_reseal_motion_permit", "startup_reseal"),
            ("motion_authorization_permit", "runtime_localization"),
        ):
            with self.subTest(kind=kind), patch(self.helper, return_value=Path("/claim/slot.json")) as helper:
                outcome = replace(self.outcome, **{fields+"_path": Path("/permit/old.json"),
                                                   fields+"_sha256": "a"*64})
                result = retire_candidate_startup_permit(outcome, self.identity, 2, self.root)
                args = helper.call_args.kwargs
                self.assertEqual(result, Path("/claim/slot.json"))
                self.assertEqual(args["permit_kind"], kind)
                self.assertEqual(args["expected_permit_sha256"], "a"*64)
                self.assertIsNone(args["disposition_path"])
                self.assertEqual(args["rejected_run_id"], self.identity.run_id)
                self.assertEqual(args["target_id"], self.identity.target_id)
                self.assertEqual(args["reseal_index"], 2)

    def test_no_permit_proof_uses_own_disposition_artifact(self):
        dry = replace(self.outcome, stop_details=recorded_stop_details(dry_run=True))
        with patch(self.helper, return_value=self.root / "rejected_permit_disposition.json") as helper:
            retire_candidate_startup_permit(dry, self.identity, 1, self.root)
        self.assertIsNone(helper.call_args.kwargs["permit_path"])
        self.assertIsNone(helper.call_args.kwargs["permit_kind"])
        self.assertEqual(helper.call_args.kwargs["expected_permit_sha256"], "")
        self.assertEqual(helper.call_args.kwargs["disposition_path"], self.root / "rejected_permit_disposition.json")

    def test_wrong_child_motion_or_conflicting_permits_never_reach_retirement(self):
        for outcome in (
            replace(self.outcome, run_id="other"),
            replace(self.outcome, motion_published=True),
            replace(self.outcome, mission_leg_motion_permit_path=Path("/p")),
            replace(self.outcome, mission_leg_motion_permit_path=Path("/p"),
                    mission_leg_motion_permit_sha256="a"*64,
                    startup_reseal_motion_permit_path=Path("/s"),
                    startup_reseal_motion_permit_sha256="b"*64),
        ):
            with self.subTest(outcome=outcome), patch(self.helper) as helper:
                with self.assertRaises(ValueError):
                    retire_candidate_startup_permit(outcome, self.identity, 1, self.root)
                helper.assert_not_called()
