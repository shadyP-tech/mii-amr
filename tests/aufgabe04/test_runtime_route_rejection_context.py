"""Replanning context never converts arbitrary runtime failures into retries."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.real_robot.candidate.recovery_dispatch import CandidateRecoveryState
from scripts.aufgabe04.real_robot.candidate.runtime_recovery import (
    CandidateRuntimeRecoveryConfig,
    CandidateRuntimeRecoveryEffects,
    CandidateRuntimeRecoveryError,
    execute_candidate_runtime_localization_recovery,
    validate_runtime_route_rejection_context,
)
from scripts.aufgabe04.real_robot.candidate.startup_recovery import CandidateRoutineIdentity
from tests.aufgabe04.test_autonomous_candidate_runtime_recovery import _outcome, _runtime_stop
from tests.aufgabe04 import test_candidate_recovery_dispatch as dispatch_fixture


def route_rejection_details():
    reason = (
        "odom execution admission failed: route uncertainty budget exhausted: "
        "limiting_segment=segment:0001:0072 remaining_margin=-0.0038 m"
    )
    return {
        "reason": reason,
        "fault_code": "odom_execution_admission_failed",
        "execution_pose_owner": "odom",
        "global_consistency_monitor": "amcl",
        "motion_published": False,
        "fail_closed": True,
        "uncertainty_budget_accepted": False,
        "route_uncertainty_limiting_segment_id": "segment:0001:0072",
        "route_uncertainty_remaining_margin_m": -0.0038,
    }


def capture_route_rejection(root, *, kind="opposite_face", transform=lambda value: value,
                            before_child=lambda attempt: None, source_transform=lambda value: value):
    identity = CandidateRoutineIdentity(
        session_id="mission", semantic_map_id="arena", routine_kind=kind,
        routine_index=1, target_id="survey_candidate_0001", run_id="mission_opposite_001",
    )
    state = CandidateRecoveryState()
    source = source_transform(_runtime_stop(root, identity.run_id))

    def admit(path):
        path.parent.mkdir(parents=True)
        path.write_text('{"accepted":true}')
        return Pose2D(0.1, 0.0, 0.0)

    def replan(attempt):
        attempt.source_root.mkdir()
        return attempt.identity

    def run(identity, attempt):
        before_child(attempt)
        details = route_rejection_details()
        return transform(_outcome(
            root, run_id=identity.run_id, status="preflight_failed",
            stop_reason=details["reason"], stop_details=details,
        ))

    try:
        execute_candidate_runtime_localization_recovery(
            source,
            config=CandidateRuntimeRecoveryConfig(identity, root / "runtime", root / "events.jsonl", 1),
            effects=CandidateRuntimeRecoveryEffects(
                admit_fresh_stationary_localization=admit,
                replan_same_routine=replan, describe_request=lambda request: request,
                run_replacement=run, event_sink=lambda path, event: None,
            ), recovery_state=state,
        )
    except CandidateRuntimeRecoveryError as error:
        return error, identity, state
    raise AssertionError("runtime route rejection unexpectedly returned")


class RuntimeRouteRejectionContextTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def valid(self):
        error, identity, state = capture_route_rejection(self.root)
        context = validate_runtime_route_rejection_context(error, expected_identity=identity)
        return error, identity, state, context

    def invalid(self, error, identity):
        with self.assertRaises(ValueError):
            validate_runtime_route_rejection_context(error, expected_identity=identity)

    def test_exact_recorded_negative_margin_produces_context_and_consumes_budget(self):
        error, identity, state, context = self.valid()
        self.assertEqual(error.phase, "replacement_preflight_failed")
        self.assertEqual(state.runtime_reseal_count, 1)
        self.assertEqual(context.original_identity, identity)
        self.assertEqual(context.attempt.fresh_start_pose, Pose2D(0.1, 0.0, 0.0))
        self.assertFalse(context.replacement_outcome.motion_published)
        self.assertFalse(error.to_failure_fields()["motion_continues_authorized"])

    def test_missing_context_and_wrong_error_or_phase_cannot_replan(self):
        error, identity, _, _ = self.valid()
        self.invalid(RuntimeError(str(error)), identity)
        self.invalid(CandidateRuntimeRecoveryError(str(error), phase=error.phase,
                     rejected_child=error.rejected_child), identity)
        error.phase = "budget_exhausted"
        self.invalid(error, identity)

    def test_context_cannot_be_replayed_for_another_routine_or_run(self):
        error, identity, _, _ = self.valid()
        for changed in (replace(identity, run_id="different"),
                        replace(identity, target_id="another"),
                        replace(identity, routine_index=2),
                        replace(identity, semantic_map_id="other_map")):
            with self.subTest(changed=changed):
                self.invalid(error, changed)

    def test_changed_child_details_or_return_code_cannot_replan(self):
        error, identity, _, context = self.valid()
        for outcome in (replace(context.replacement_outcome, returncode=0),
                        replace(context.replacement_outcome, run_id="unrelated"),
                        replace(context.replacement_outcome, motion_published=True),
                        replace(context.replacement_outcome, mission_leg_motion_permit_sha256="a" * 64)):
            with self.subTest(outcome=outcome):
                error.route_rejection_context = replace(context, replacement_outcome=outcome)
                self.invalid(error, identity)
        error.route_rejection_context = context
        error.rejected_child = replace(error.rejected_child, stop_reason="another failure")
        self.invalid(error, identity)

    def test_nested_outcome_mutation_cannot_replan(self):
        error, identity, _, context = self.valid()
        context.replacement_outcome.stop_details["route_uncertainty_remaining_margin_m"] = 0.0
        self.invalid(error, identity)

    def test_source_stop_mutation_cannot_replan(self):
        error, identity, _, context = self.valid()
        context.attempt.rejected_outcome.stop_details["monitor_action"] = "WARN"
        self.invalid(error, identity)

    def test_permit_mutation_or_disappearance_cannot_replan(self):
        error, identity, _, context = self.valid()
        context.source_permit_path.write_text('{"changed":true}')
        self.invalid(error, identity)
        context.source_permit_path.unlink()
        self.invalid(error, identity)

    def test_fresh_evidence_mutation_or_symlink_cannot_replan(self):
        error, identity, _, context = self.valid()
        path = context.attempt.fresh_localization_evidence_path
        path.write_text('{"accepted":false}')
        self.invalid(error, identity)
        path.unlink()
        elsewhere = self.root / "other.json"
        elsewhere.write_text('{"accepted":true}')
        path.symlink_to(elsewhere)
        self.invalid(error, identity)

    def test_forged_attempt_identity_pose_and_paths_are_rejected(self):
        error, identity, _, context = self.valid()
        for changed in (replace(context.attempt, fresh_start_pose=Pose2D(1.0, 0.0, 0.0)),
                        replace(context.attempt, reseal_index=2),
                        replace(context.attempt, source_root=self.root),
                        replace(context.attempt, identity=replace(context.attempt.identity, target_id="other"))):
            with self.subTest(attempt=changed):
                error.route_rejection_context = replace(context, attempt=changed)
                self.invalid(error, identity)

    def test_invalid_and_replayed_original_lineage_is_rejected(self):
        _, identity, _, context = self.valid()
        for run_id in (identity.run_id + "_unrelated",
                       identity.run_id + "_startup_reseal_000",
                       identity.run_id + "_startup_reseal_001_startup_reseal_001"):
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                CandidateRuntimeRecoveryConfig(
                    replace(identity, run_id=run_id), self.root, self.root / "events", 1,
                    original_identity=identity,
                )

    def test_other_stop_reason_and_permit_bearing_rejection_have_no_context(self):
        cases = (
            lambda outcome: replace(outcome, stop_reason="obstacle", stop_details={
                "reason": "obstacle", "motion_published": False, "fail_closed": True}),
            lambda outcome: replace(outcome, stop_details={**outcome.stop_details,
                "route_uncertainty_remaining_margin_m": 0.0}),
            lambda outcome: replace(outcome, startup_reseal_motion_permit_sha256="f" * 64),
        )
        for index, transform in enumerate(cases):
            with self.subTest(index=index):
                root = self.root / str(index)
                root.mkdir()
                error, identity, state = capture_route_rejection(root, transform=transform)
                self.assertIsNone(error.route_rejection_context)
                self.assertEqual(state.runtime_reseal_count, 1)
                self.invalid(error, identity)

    def test_other_source_stop_and_nonopposite_routine_have_no_context(self):
        error, identity, _ = capture_route_rejection(self.root / "other", source_transform=lambda value:
            replace(value, stop_details={"fault_code": "obstacle"}))
        self.assertIsNone(error.route_rejection_context)
        self.invalid(error, identity)
        error, identity, _ = capture_route_rejection(self.root / "nonopposite", kind="candidate_preapproach")
        self.assertIsNone(error.route_rejection_context)
        self.invalid(error, identity)

    def test_optional_nonjson_details_keep_existing_terminal_failure(self):
        for kind in ("opposite_face", "candidate_preapproach"):
            with self.subTest(kind=kind):
                error, _, state = capture_route_rejection(
                    self.root / kind, kind=kind, source_transform=lambda value: replace(
                        value, stop_details={**value.stop_details, "optional_diagnostic": object()},
                    ),
                )
                self.assertEqual(error.phase, "replacement_preflight_failed")
                self.assertIsNone(error.route_rejection_context)
                self.assertEqual(state.runtime_reseal_count, 1)

    def test_admitted_files_changed_during_child_do_not_get_context(self):
        for index, mutate in enumerate((
            lambda attempt: attempt.fresh_localization_evidence_path.write_text("changed"),
            lambda attempt: attempt.rejected_outcome.startup_reseal_motion_permit_path.write_text("changed"),
        )):
            with self.subTest(index=index):
                error, _, state = capture_route_rejection(self.root / str(index), before_child=mutate)
                self.assertIsNone(error.route_rejection_context)
                self.assertEqual(error.phase, "replacement_preflight_failed")
                self.assertEqual(state.runtime_reseal_count, 1)

    def test_dispatcher_keeps_original_identity_after_startup_changes_run_id(self):
        harness = dispatch_fixture.CandidateRecoveryDispatchTest(methodName="runTest")
        harness.setUp()
        self.addCleanup(harness.doCleanups)
        harness.identity = replace(harness.identity, routine_kind="opposite_face")

        def change_preflight(owner, outcome):
            if owner == "runtime":
                details = route_rejection_details()
                return replace(outcome, stop_reason=details["reason"], stop_details=details)
            return outcome

        harness.mutate_outcome = change_preflight
        with self.assertRaises(CandidateRuntimeRecoveryError) as caught:
            harness.execute([("initial", "mismatch"), ("startup", "runtime"), ("runtime", "preflight")])
        context = validate_runtime_route_rejection_context(caught.exception, expected_identity=harness.identity)
        self.assertEqual(context.original_identity, harness.identity)
        self.assertEqual(context.runtime_base_identity, harness.identity.replacement(1))
        self.assertEqual(context.attempt.rejected_outcome.run_id, harness.identity.replacement(1).run_id)


if __name__ == "__main__":
    unittest.main()
