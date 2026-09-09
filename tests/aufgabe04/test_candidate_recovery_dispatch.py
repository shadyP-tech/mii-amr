"""Real failure replay and bounded cross-phase candidate recovery tests."""

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.prestart_localization_reseal import evaluate_prestart_localization_reseal
from scripts.aufgabe04.navigation.localization.runtime_localization_reseal import evaluate_runtime_localization_reseal
from scripts.aufgabe04.real_robot.candidate.recovery_dispatch import execute_candidate_motion_with_recovery
from scripts.aufgabe04.real_robot.candidate.runtime_recovery import (
    CandidateRuntimeRecoveryConfig, CandidateRuntimeRecoveryEffects,
    CandidateRuntimeRecoveryError,
)
from scripts.aufgabe04.real_robot.candidate.startup_recovery import (
    CandidateRoutineIdentity, CandidateStartupRecoveryConfig,
    CandidateStartupRecoveryEffects, CandidateStartupRecoveryError,
)
from scripts.aufgabe04.real_robot.execution.child_runner import MotionLegOutcome


@dataclass(frozen=True)
class Request:
    identity: CandidateRoutineIdentity


class CandidateRecoveryDispatchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.identity = CandidateRoutineIdentity(
            session_id="mission", semantic_map_id="arena",
            routine_kind="candidate_preapproach", routine_index=1,
            target_id="survey_candidate_0001", run_id="mission_candidate_001",
        )
        self.saved = json.loads((Path(__file__).parent / "fixtures" /
            "candidate_runtime_prestart_stop_20260908.json").read_text())["stop_details"]
        self.calls, self.admissions, self.events = [], [], []
        self.mutate_outcome = lambda owner, outcome: outcome
        self.mutate_request = lambda owner, request: request

    def outcome(self, identity, phase, owner):
        reason = "global localization consistency requires zero and reseal"
        details = json.loads(json.dumps(self.saved))
        status, motion = "stopped", False
        if phase == "runtime":
            motion = True
            for key in ("phase", "execution_phase", "motion_published", "initial_tf_acquisition"):
                details.pop(key, None)
            details["monitor_warning"] = ""
            details["continuity"]["reason"] = "map_from_odom_translation_drift"
        elif phase == "mismatch":
            reason = "pose outside certified startup segment"
            details = {"source": "execution_route_certificate", "phase": "before_motion_confirmation",
                       "route_pose": {"x_m": 0.1, "y_m": 0.0, "yaw_rad": 0.0}}
        elif phase == "completed":
            reason, details, status, motion = "", {}, "completed", True
        elif phase == "unknown":
            reason, details = "stale transform sample", {"reason": "stale transform sample"}
        elif phase == "preflight":
            status, reason = "preflight_failed", "route uncertainty rejected"
            details = {"reason": reason, "motion_published": False, "fail_closed": True}
        elif phase != "prestart":
            raise AssertionError(phase)
        fields = {}
        if phase != "preflight":
            prefix = {"initial": "mission_leg_motion_permit", "startup": "startup_reseal_motion_permit",
                      "runtime": "motion_authorization_permit"}[owner]
            path = self.root / "permits" / f"{identity.run_id}.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps({"run_id": identity.run_id, "owner": owner}))
            fields = {prefix + "_path": path, prefix + "_sha256": sha256(path.read_bytes()).hexdigest()}
        log = self.root / f"{identity.run_id}.jsonl"
        log.write_text("{}\n")
        return MotionLegOutcome(run_id=identity.run_id, status=status, stop_reason=reason,
            stop_details=details, motion_published=motion, returncode=0 if status=="completed" else 2,
            semantic_log_path=log, **fields)

    def execute(self, phases, *, startup_budget=3, runtime_budget=2):
        phases = iter(phases)
        def run(owner, request, attempt=None):
            self.calls.append((owner, request.identity, attempt))
            expected_owner, phase = next(phases)
            self.assertEqual(owner, expected_owner)
            return self.mutate_outcome(owner, self.outcome(request.identity, phase, owner))
        def admit(path):
            self.admissions.append(path)
            path.parent.mkdir(parents=True, exist_ok=False)
            path.write_text('{"accepted":true}')
            return Pose2D(0.1, 0.0, 0.0)
        def replan(owner, attempt):
            attempt.source_root.mkdir()
            (attempt.source_root / "route.csv").write_text("x,y\n0.1,0.0\n")
            return self.mutate_request(owner, Request(attempt.identity))
        event = lambda path, payload: self.events.append(payload)
        startup = CandidateStartupRecoveryConfig(self.identity, self.root / "startup", self.root / "events.jsonl",
            startup_budget, allow_runtime_localization_handoff=bool(runtime_budget))
        runtime = CandidateRuntimeRecoveryConfig(self.identity, self.root / "runtime", self.root / "events.jsonl", runtime_budget)
        return execute_candidate_motion_with_recovery(Request(self.identity),
            startup_config=startup, runtime_config=runtime,
            startup_effects=CandidateStartupRecoveryEffects(
                run_initial=lambda request: run("initial", request),
                run_replacement=lambda request, attempt: run("startup", request, attempt),
                admit_fresh_stationary_localization=admit,
                replan_same_routine=lambda attempt: replan("startup", attempt),
                describe_request=lambda request: request.identity, event_sink=event),
            runtime_effects=CandidateRuntimeRecoveryEffects(
                run_replacement=lambda request, attempt: run("runtime", request, attempt),
                admit_fresh_stationary_localization=admit,
                replan_same_routine=lambda attempt: replan("runtime", attempt),
                describe_request=lambda request: request.identity, event_sink=event))

    def test_saved_runtime_replacement_prestart_stop_returns_to_startup(self):
        self.assertTrue(evaluate_prestart_localization_reseal(status="stopped", motion_published=False,
            stop_details=self.saved).eligible)
        self.assertEqual(evaluate_runtime_localization_reseal(status="stopped", motion_published=False,
            stop_details=self.saved).reason, "motion_not_published")
        result = self.execute([("initial", "runtime"), ("runtime", "prestart"), ("startup", "completed")], runtime_budget=1)
        self.assertEqual(result.status, "completed")
        self.assertEqual([call[0] for call in self.calls], ["initial", "runtime", "startup"])
        self.assertEqual(len(self.admissions), 2)
        self.assertEqual(self.calls[-1][2].rejected_outcome.run_id, self.calls[-2][1].run_id)
        self.assertIsNotNone(result.startup_reseal_motion_permit_path)
        self.assertIsNone(result.motion_authorization_permit_path)

    def test_alternating_recoveries_keep_both_counters_and_routine(self):
        result = self.execute([("initial", "mismatch"), ("startup", "runtime"),
            ("runtime", "prestart"), ("startup", "runtime"), ("runtime", "completed")])
        self.assertEqual(result.status, "completed")
        self.assertEqual([(kind, attempt.reseal_index) for kind, _, attempt in self.calls[1:]],
                         [("startup", 1), ("runtime", 1), ("startup", 2), ("runtime", 2)])
        self.assertEqual(len({identity.run_id for _, identity, _ in self.calls}), 5)
        self.assertEqual(len(set(self.admissions)), 4)
        for _, identity, _ in self.calls:
            self.assertEqual(replace(identity, run_id=self.identity.run_id), self.identity)

    def test_runtime_budget_does_not_reset_after_startup_handoff(self):
        with self.assertRaises(CandidateRuntimeRecoveryError) as caught:
            self.execute([("initial", "runtime"), ("runtime", "prestart"), ("startup", "runtime")], runtime_budget=1)
        self.assertEqual(caught.exception.phase, "budget_exhausted")
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(len(self.admissions), 2)

    def test_startup_budget_does_not_reset_after_runtime_handoff(self):
        with self.assertRaises(CandidateStartupRecoveryError) as caught:
            self.execute([("initial", "mismatch"), ("startup", "runtime"), ("runtime", "prestart")], startup_budget=1)
        self.assertEqual(caught.exception.phase, "budget_exhausted")
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(len(self.admissions), 2)

    def test_preflight_rejection_remains_terminal_before_permit_validation(self):
        with self.assertRaises(CandidateRuntimeRecoveryError) as caught:
            self.execute([("initial", "runtime"), ("runtime", "preflight")])
        self.assertEqual(caught.exception.phase, "replacement_preflight_failed")
        self.assertIn("route uncertainty rejected", str(caught.exception))
        self.assertEqual(len(self.calls), 2)

    def test_unknown_stopped_child_keeps_actual_cause_and_remains_terminal(self):
        with self.assertRaises(CandidateRuntimeRecoveryError) as caught:
            self.execute([("initial", "runtime"), ("runtime", "unknown")])
        self.assertIn("stale transform sample", str(caught.exception))
        fields = caught.exception.to_failure_fields()
        self.assertEqual(fields["reason"], "stale transform sample")
        self.assertIn("not an eligible", fields["rejection_policy_reason"])
        self.assertEqual(len(self.calls), 2)

    def test_wrong_initial_identity_and_motion_type_never_reach_recovery(self):
        for phase, change in (("completed", {"run_id": "wrong"}), ("prestart", {"run_id": "wrong"}),
                              ("completed", {"motion_published": 1})):
            with self.subTest(phase=phase, change=change):
                self.mutate_outcome = lambda owner, outcome: replace(outcome, **change)
                with self.assertRaises(CandidateStartupRecoveryError):
                    self.execute([("initial", phase)])
        self.assertFalse(self.admissions)

    def test_runtime_child_cannot_report_startup_permit(self):
        def mutate(owner, outcome):
            if owner != "runtime": return outcome
            return replace(outcome,
                startup_reseal_motion_permit_path=outcome.motion_authorization_permit_path,
                startup_reseal_motion_permit_sha256=outcome.motion_authorization_permit_sha256,
                motion_authorization_permit_path=None, motion_authorization_permit_sha256="")
        self.mutate_outcome = mutate
        with self.assertRaises(CandidateRuntimeRecoveryError):
            self.execute([("initial", "runtime"), ("runtime", "prestart")])
        self.assertEqual(len(self.calls), 2)

    def test_startup_child_cannot_report_runtime_permit(self):
        def mutate(owner, outcome):
            if owner != "startup": return outcome
            return replace(outcome,
                motion_authorization_permit_path=outcome.startup_reseal_motion_permit_path,
                motion_authorization_permit_sha256=outcome.startup_reseal_motion_permit_sha256,
                startup_reseal_motion_permit_path=None, startup_reseal_motion_permit_sha256="")
        self.mutate_outcome = mutate
        with self.assertRaises(CandidateStartupRecoveryError):
            self.execute([("initial", "runtime"), ("runtime", "prestart"), ("startup", "completed")])

    def test_cross_phase_permit_reuse_is_rejected(self):
        previous = []
        def mutate(owner, outcome):
            if owner == "runtime": previous.append(outcome)
            if owner == "startup":
                return replace(outcome, startup_reseal_motion_permit_path=previous[-1].motion_authorization_permit_path,
                    startup_reseal_motion_permit_sha256=previous[-1].motion_authorization_permit_sha256)
            return outcome
        self.mutate_outcome = mutate
        with self.assertRaisesRegex(CandidateStartupRecoveryError, "reused one-use"):
            self.execute([("initial", "runtime"), ("runtime", "prestart"), ("startup", "completed")])

    def test_resumed_startup_request_cannot_switch_target(self):
        self.mutate_request = lambda owner, request: (
            replace(request, identity=replace(request.identity, target_id="another_stand"))
            if owner == "startup" else request)
        with self.assertRaises(CandidateStartupRecoveryError) as caught:
            self.execute([("initial", "runtime"), ("runtime", "prestart")])
        self.assertEqual(caught.exception.phase, "same_routine_replan")
        self.assertEqual(len(self.calls), 2)

    def test_wrong_replacement_identity_cannot_handoff(self):
        self.mutate_outcome = lambda owner, outcome: replace(outcome, run_id="wrong") if owner == "runtime" else outcome
        with self.assertRaises(CandidateRuntimeRecoveryError):
            self.execute([("initial", "runtime"), ("runtime", "prestart")])
        self.assertEqual(len(self.calls), 2)

    def test_malformed_prestart_evidence_cannot_handoff(self):
        self.saved["motion_published"] = True
        with self.assertRaises(CandidateRuntimeRecoveryError):
            self.execute([("initial", "runtime"), ("runtime", "prestart")])
        self.assertEqual(len(self.calls), 2)


if __name__ == "__main__":
    unittest.main()
