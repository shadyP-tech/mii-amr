from dataclasses import dataclass, replace
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.real_robot.autonomous_candidate_runtime_recovery import (
    CandidateRuntimeRecoveryAttempt,
    CandidateRuntimeRecoveryConfig,
    CandidateRuntimeRecoveryEffects,
    CandidateRuntimeRecoveryError,
    execute_candidate_runtime_localization_recovery,
)
from scripts.aufgabe04.real_robot.autonomous_candidate_startup_recovery import (
    CandidateRoutineIdentity,
)
from scripts.aufgabe04.real_robot.autonomous_child_runner import MotionLegOutcome


@dataclass(frozen=True)
class _Request:
    identity: CandidateRoutineIdentity


def _stop_details(*, continuity_reason: str = "map_from_odom_translation_drift"):
    return {
        "fault_code": "localization_reseal_required",
        "source": "global_consistency_monitor",
        "execution_pose_owner": "odom",
        "global_consistency_monitor": "amcl",
        "monitor_action": "FORCE_ZERO_RESEAL",
        "fail_closed": True,
        "continuity": {
            "accepted": False,
            "requires_zero_cycle": True,
            "requires_reseal": True,
            "decision": "force_zero_reseal",
            "reason": continuity_reason,
            "fail_closed": True,
        },
    }


def _outcome(
    root: Path,
    *,
    run_id: str,
    status: str,
    stop_reason: str = "",
    stop_details: dict[str, object] | None = None,
    motion_published: bool = False,
    permit_name: str | None = None,
    permit_digest: str = "",
) -> MotionLegOutcome:
    semantic_log = root / "semantic" / f"{run_id}.jsonl"
    semantic_log.parent.mkdir(parents=True, exist_ok=True)
    semantic_log.write_text("{}\n", encoding="utf-8")
    permit_path = None
    if permit_name is not None:
        permit_path = (root / "permits" / permit_name).resolve()
        permit_path.parent.mkdir(parents=True, exist_ok=True)
        permit_path.write_text('{"one_use":true}\n', encoding="utf-8")
    return MotionLegOutcome(
        run_id=run_id,
        status=status,
        stop_reason=stop_reason,
        stop_details={} if stop_details is None else stop_details,
        motion_published=motion_published,
        returncode=0 if status == "completed" else 2,
        semantic_log_path=semantic_log,
        motion_authorization_permit_path=permit_path,
        motion_authorization_permit_sha256=permit_digest,
    )


def _runtime_stop(
    root: Path,
    run_id: str,
    *,
    permit_name: str | None = None,
    permit_digest: str = "",
    continuity_reason: str = "map_from_odom_translation_drift",
) -> MotionLegOutcome:
    reason = "global localization consistency requires zero and reseal"
    outcome = _outcome(
        root,
        run_id=run_id,
        status="stopped",
        stop_reason=reason,
        stop_details=_stop_details(continuity_reason=continuity_reason),
        motion_published=True,
        permit_name=permit_name,
        permit_digest=permit_digest,
    )
    if permit_name is not None:
        return outcome
    source_permit = (root / "permits" / f"{run_id}_startup.json").resolve()
    source_permit.parent.mkdir(parents=True, exist_ok=True)
    source_permit.write_text('{"one_use":true}\n', encoding="utf-8")
    return replace(
        outcome,
        startup_reseal_motion_permit_path=source_permit,
        startup_reseal_motion_permit_sha256="f" * 64,
    )


class CandidateRuntimeRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        # Mirrors the latest failure: startup recovery already produced this
        # child identity before its post-motion FORCE_ZERO_RESEAL stop.
        self.identity = CandidateRoutineIdentity(
            session_id="mission",
            semantic_map_id="arena",
            routine_kind="candidate_preapproach",
            routine_index=0,
            target_id="survey_candidate_0004",
            run_id="mission_candidate_000_startup_reseal_001",
        )
        self.events: list[dict[str, object]] = []
        self.admitted_paths: list[Path] = []
        self.replanned_attempts: list[CandidateRuntimeRecoveryAttempt] = []
        self.replacement_attempts: list[CandidateRuntimeRecoveryAttempt] = []

    def _identity(self, reseal_index: int) -> CandidateRoutineIdentity:
        return replace(
            self.identity,
            run_id=(
                f"{self.identity.run_id}_runtime_localization_reseal_"
                f"{reseal_index:03d}"
            ),
        )

    def _config(
        self,
        budget: int,
        *,
        recovery_name: str = "runtime_reseals",
    ) -> CandidateRuntimeRecoveryConfig:
        return CandidateRuntimeRecoveryConfig(
            initial_identity=self.identity,
            recovery_root=self.root / recovery_name,
            event_log_path=self.root / "events.jsonl",
            max_runtime_reseals=budget,
        )

    def _admit(self, path: Path) -> Pose2D:
        self.admitted_paths.append(path)
        path.parent.mkdir(parents=True, exist_ok=False)
        path.write_text('{"accepted":true}\n', encoding="utf-8")
        return Pose2D(-1.635, -0.511, 0.070)

    def _replan(self, attempt: CandidateRuntimeRecoveryAttempt) -> _Request:
        self.replanned_attempts.append(attempt)
        attempt.source_root.mkdir(parents=False, exist_ok=False)
        (attempt.source_root / "route.csv").write_text(
            "x,y\n-1.635,-0.511\n",
            encoding="utf-8",
        )
        return _Request(attempt.identity)

    def _event_sink(self, path: Path, payload: dict[str, object]) -> None:
        self.assertEqual(path, self.root / "events.jsonl")
        self.events.append(payload)

    def _effects(
        self,
        replacements: list[object],
        *,
        admit=None,
        replan=None,
        event_sink=None,
    ) -> CandidateRuntimeRecoveryEffects[_Request]:
        values = list(replacements)

        def run_replacement(
            request: _Request,
            attempt: CandidateRuntimeRecoveryAttempt,
        ) -> MotionLegOutcome:
            self.replacement_attempts.append(attempt)
            value = values.pop(0)
            if isinstance(value, Exception):
                raise value
            return value  # type: ignore[return-value]

        return CandidateRuntimeRecoveryEffects(
            admit_fresh_stationary_localization=(
                self._admit if admit is None else admit
            ),
            replan_same_routine=self._replan if replan is None else replan,
            describe_request=lambda request: request.identity,
            run_replacement=run_replacement,
            event_sink=(self._event_sink if event_sink is None else event_sink),
            clock=lambda: 123.0,
        )

    def test_completed_and_no_motion_initial_outcomes_are_unchanged(self):
        outcomes = (
            _outcome(
                self.root,
                run_id="already_completed_by_startup_coordinator",
                status="completed",
                motion_published=True,
            ),
            _outcome(
                self.root,
                run_id="no_motion_still_owned_by_startup_coordinator",
                status="preflight_failed",
                stop_reason="pose outside certified startup segment",
                motion_published=False,
            ),
        )
        for outcome in outcomes:
            with self.subTest(status=outcome.status):
                result = execute_candidate_runtime_localization_recovery(
                    outcome,
                    config=self._config(1),
                    effects=self._effects([]),
                )
                self.assertIs(result, outcome)

        self.assertFalse(self.events)
        self.assertFalse(self.admitted_paths)
        self.assertFalse(self.replacement_attempts)

    def test_latest_force_zero_reseal_recovers_with_fresh_exact_evidence(self):
        replacement_identity = self._identity(1)
        initial = _runtime_stop(self.root, self.identity.run_id)
        completed = _outcome(
            self.root,
            run_id=replacement_identity.run_id,
            status="completed",
            motion_published=True,
            permit_name="runtime_001.json",
            permit_digest="a" * 64,
        )

        result = execute_candidate_runtime_localization_recovery(
            initial,
            config=self._config(1),
            effects=self._effects([completed]),
        )

        self.assertIs(result, completed)
        self.assertEqual(len(self.replacement_attempts), 1)
        attempt = self.replacement_attempts[0]
        self.assertEqual(attempt.identity, replacement_identity)
        self.assertEqual(attempt.reseal_index, 1)
        self.assertEqual(attempt.rejected_outcome, initial)
        self.assertTrue(attempt.runtime_localization_decision.eligible)
        self.assertEqual(
            attempt.runtime_localization_decision.continuity_reason,
            "map_from_odom_translation_drift",
        )
        self.assertEqual(
            attempt.attempt_root,
            self.root / "runtime_reseals/runtime_localization_reseal_001",
        )
        self.assertTrue(attempt.fresh_localization_evidence_path.is_file())
        self.assertTrue(attempt.source_root.is_dir())
        self.assertEqual(
            [event["event"] for event in self.events],
            [
                "candidate_runtime_localization_reseal_started",
                "candidate_runtime_localization_admitted",
                "candidate_runtime_localization_route_replanned",
                "candidate_runtime_localization_permit_evidenced",
                "candidate_runtime_localization_reseal_completed",
            ],
        )
        self.assertTrue(self.events[-2]["one_use_runtime_permit_evidenced"])

    def test_post_motion_source_without_one_use_permit_is_terminal(self):
        unauthorized = _outcome(
            self.root,
            run_id=self.identity.run_id,
            status="stopped",
            stop_reason=(
                "global localization consistency requires zero and reseal"
            ),
            stop_details=_stop_details(),
            motion_published=True,
        )

        with self.assertRaisesRegex(
            CandidateRuntimeRecoveryError,
            "exactly one routine or startup one-use permit",
        ):
            execute_candidate_runtime_localization_recovery(
                unauthorized,
                config=self._config(1, recovery_name="unauthorized_source"),
                effects=self._effects([]),
            )

        self.assertFalse(self.admitted_paths)
        self.assertFalse(self.replacement_attempts)

    def test_repeated_eligible_reseals_use_distinct_deterministic_identities(self):
        first = self._runtime_replacement_stop(1, "a")
        second_identity = self._identity(2)
        second = _outcome(
            self.root,
            run_id=second_identity.run_id,
            status="completed",
            motion_published=True,
            permit_name="runtime_002.json",
            permit_digest="b" * 64,
        )

        result = execute_candidate_runtime_localization_recovery(
            _runtime_stop(self.root, self.identity.run_id),
            config=self._config(2),
            effects=self._effects([first, second]),
        )

        self.assertIs(result, second)
        self.assertEqual(
            [attempt.identity.run_id for attempt in self.replacement_attempts],
            [self._identity(1).run_id, self._identity(2).run_id],
        )
        self.assertEqual(
            [path.parent.name for path in self.admitted_paths],
            [
                "runtime_localization_reseal_001",
                "runtime_localization_reseal_002",
            ],
        )
        self.assertTrue(
            all(
                attempt.identity.target_id == self.identity.target_id
                and attempt.identity.routine_kind == self.identity.routine_kind
                and attempt.identity.routine_index == self.identity.routine_index
                for attempt in self.replacement_attempts
            )
        )

    def _runtime_replacement_stop(
        self,
        reseal_index: int,
        digest_character: str,
    ) -> MotionLegOutcome:
        return _runtime_stop(
            self.root,
            self._identity(reseal_index).run_id,
            permit_name=f"runtime_{reseal_index:03d}.json",
            permit_digest=digest_character * 64,
        )

    def test_runtime_budget_is_separate_bounded_and_never_launches_n_plus_one(self):
        first = self._runtime_replacement_stop(1, "a")

        with self.assertRaisesRegex(
            CandidateRuntimeRecoveryError,
            "budget exhausted",
        ) as caught:
            execute_candidate_runtime_localization_recovery(
                _runtime_stop(self.root, self.identity.run_id),
                config=self._config(1),
                effects=self._effects([first]),
            )

        self.assertEqual(len(self.replacement_attempts), 1)
        self.assertEqual(len(self.admitted_paths), 1)
        self.assertEqual(
            self.events[-1]["event"],
            "candidate_runtime_localization_reseal_exhausted",
        )
        failure = caught.exception.to_failure_fields()
        self.assertEqual(
            failure["failure_phase"],
            "candidate_runtime_localization_recovery",
        )
        self.assertEqual(
            failure["candidate_runtime_localization_recovery_phase"],
            "budget_exhausted",
        )
        self.assertFalse(failure["motion_continues_authorized"])
        self.assertTrue(failure["fail_closed"])

    def test_changed_request_or_outcome_identity_is_terminal(self):
        def changed_request(attempt: CandidateRuntimeRecoveryAttempt) -> _Request:
            attempt.source_root.mkdir(parents=False)
            return _Request(replace(attempt.identity, target_id="another_target"))

        cases = (
            (
                "request",
                changed_request,
                [],
                "committed routine identity",
            ),
            (
                "outcome",
                None,
                [
                    _outcome(
                        self.root,
                        run_id="wrong_run",
                        status="completed",
                        permit_name="wrong.json",
                        permit_digest="a" * 64,
                    )
                ],
                "identity mismatch",
            ),
        )
        for name, replan, replacements, message in cases:
            with self.subTest(name=name):
                self.events.clear()
                self.admitted_paths.clear()
                self.replanned_attempts.clear()
                self.replacement_attempts.clear()
                with self.assertRaisesRegex(CandidateRuntimeRecoveryError, message):
                    execute_candidate_runtime_localization_recovery(
                        _runtime_stop(self.root, self.identity.run_id),
                        config=self._config(1, recovery_name=f"identity_{name}"),
                        effects=self._effects(replacements, replan=replan),
                    )

    def test_noneligible_initial_and_malformed_replacement_fail_closed(self):
        noneligible = _outcome(
            self.root,
            run_id=self.identity.run_id,
            status="stopped",
            stop_reason="stuck no progress",
            stop_details={"fault_code": "no_progress"},
            motion_published=True,
        )
        with self.assertRaises(CandidateRuntimeRecoveryError) as initial_error:
            execute_candidate_runtime_localization_recovery(
                noneligible,
                config=self._config(2, recovery_name="noneligible_initial"),
                effects=self._effects([]),
            )
        self.assertEqual(
            initial_error.exception.rejected_child.decision_reason,
            "invalid_fault_code",
        )
        self.assertFalse(self.replacement_attempts)

        self.events.clear()
        malformed = _outcome(
            self.root,
            run_id=self._identity(1).run_id,
            status="stopped",
            stop_reason="partial localization stop",
            stop_details={"fault_code": "localization_reseal_required"},
            motion_published=True,
            permit_name="malformed.json",
            permit_digest="a" * 64,
        )
        with self.assertRaises(CandidateRuntimeRecoveryError) as child_error:
            execute_candidate_runtime_localization_recovery(
                _runtime_stop(self.root, self.identity.run_id),
                config=self._config(2, recovery_name="malformed_child"),
                effects=self._effects([malformed]),
            )
        self.assertEqual(
            child_error.exception.rejected_child.decision_reason,
            "invalid_source",
        )
        self.assertEqual(len(self.replacement_attempts), 1)
        self.assertEqual(
            self.events[-1]["event"],
            "candidate_runtime_localization_reseal_rejected",
        )

    def test_every_replacement_requires_new_runtime_permit_evidence(self):
        cases = (
            (None, "", "lacks complete"),
            ("invalid_digest.json", "not-a-digest", "lacks complete"),
        )
        for permit_name, digest, message in cases:
            with self.subTest(permit_name=permit_name):
                self.events.clear()
                self.admitted_paths.clear()
                self.replanned_attempts.clear()
                self.replacement_attempts.clear()
                completed = _outcome(
                    self.root,
                    run_id=self._identity(1).run_id,
                    status="completed",
                    motion_published=True,
                    permit_name=permit_name,
                    permit_digest=digest,
                )
                with self.assertRaisesRegex(
                    CandidateRuntimeRecoveryError,
                    message,
                ):
                    execute_candidate_runtime_localization_recovery(
                        _runtime_stop(self.root, self.identity.run_id),
                        config=self._config(
                            1,
                            recovery_name=(
                                "missing_permit"
                                if permit_name is None
                                else "invalid_permit"
                            ),
                        ),
                        effects=self._effects([completed]),
                    )

    def test_one_use_runtime_permit_cannot_be_reused_between_replacements(self):
        shared_path = (self.root / "permits/shared.json").resolve()
        first = self._runtime_replacement_stop(1, "a")
        first = replace(first, motion_authorization_permit_path=shared_path)
        shared_path.parent.mkdir(parents=True, exist_ok=True)
        shared_path.write_text('{"one_use":true}\n', encoding="utf-8")
        second = _outcome(
            self.root,
            run_id=self._identity(2).run_id,
            status="completed",
            motion_published=True,
            permit_name="other_path.json",
            permit_digest="a" * 64,
        )

        with self.assertRaisesRegex(
            CandidateRuntimeRecoveryError,
            "reused one-use permit",
        ):
            execute_candidate_runtime_localization_recovery(
                _runtime_stop(self.root, self.identity.run_id),
                config=self._config(2, recovery_name="permit_reuse"),
                effects=self._effects([first, second]),
            )

        self.assertEqual(len(self.replacement_attempts), 2)

    def test_callback_failures_are_terminal_and_evidenced(self):
        def failing_admit(_path: Path) -> Pose2D:
            raise RuntimeError("amcl unavailable")

        def invalid_pose(path: Path) -> Pose2D:
            path.parent.mkdir(parents=True)
            path.write_text("{}\n", encoding="utf-8")
            return Pose2D(float("nan"), 0.0, 0.0)

        def missing_evidence(_path: Path) -> Pose2D:
            return Pose2D(0.0, 0.0, 0.0)

        def failing_replan(attempt: CandidateRuntimeRecoveryAttempt) -> _Request:
            raise RuntimeError(f"cannot route {attempt.identity.target_id}")

        completed = _outcome(
            self.root,
            run_id=self._identity(1).run_id,
            status="completed",
            permit_name="callback.json",
            permit_digest="a" * 64,
        )
        cases = (
            ("stationary_localization_admission", failing_admit, None, []),
            ("stationary_localization_admission", invalid_pose, None, []),
            ("stationary_localization_admission", missing_evidence, None, []),
            ("same_routine_replan", None, failing_replan, []),
            ("replacement_run", None, None, [RuntimeError("child unavailable")]),
            ("replacement_outcome_contract", None, None, [object()]),
            (
                "replacement_outcome_contract",
                None,
                None,
                [replace(completed, motion_published=1)],
            ),
        )
        for index, (phase, admit, replan, replacements) in enumerate(cases):
            with self.subTest(phase=phase, index=index):
                self.events.clear()
                self.admitted_paths.clear()
                self.replanned_attempts.clear()
                self.replacement_attempts.clear()
                values = replacements or [completed]
                with self.assertRaises(CandidateRuntimeRecoveryError) as caught:
                    execute_candidate_runtime_localization_recovery(
                        _runtime_stop(self.root, self.identity.run_id),
                        config=self._config(
                            1,
                            recovery_name=f"callback_{index}",
                        ),
                        effects=self._effects(
                            values,
                            admit=admit,
                            replan=replan,
                        ),
                    )
                self.assertEqual(caught.exception.phase, phase)
                self.assertEqual(
                    self.events[-1]["event"],
                    "candidate_runtime_localization_reseal_failed",
                )

    def test_event_sink_failure_is_terminal(self):
        def failing_sink(_path: Path, _payload: dict[str, object]) -> None:
            raise OSError("event artifact unavailable")

        with self.assertRaisesRegex(
            CandidateRuntimeRecoveryError,
            "event_sink",
        ) as caught:
            execute_candidate_runtime_localization_recovery(
                _runtime_stop(self.root, self.identity.run_id),
                config=self._config(1, recovery_name="event_sink"),
                effects=self._effects([], event_sink=failing_sink),
            )

        self.assertEqual(caught.exception.phase, "event_sink")
        self.assertFalse(self.admitted_paths)
        self.assertFalse(self.replacement_attempts)

    def test_existing_attempt_and_source_paths_fail_before_motion(self):
        existing_root = (
            self.root
            / "existing_attempt/runtime_localization_reseal_001"
        )
        existing_root.mkdir(parents=True)
        with self.assertRaises(CandidateRuntimeRecoveryError) as attempt_error:
            execute_candidate_runtime_localization_recovery(
                _runtime_stop(self.root, self.identity.run_id),
                config=self._config(1, recovery_name="existing_attempt"),
                effects=self._effects([]),
            )
        self.assertEqual(attempt_error.exception.phase, "attempt_path_admission")
        self.assertFalse(self.replacement_attempts)

        self.events.clear()
        self.admitted_paths.clear()

        def precreated_source(path: Path) -> Pose2D:
            pose = self._admit(path)
            (path.parent / "route_source").mkdir()
            return pose

        with self.assertRaises(CandidateRuntimeRecoveryError) as source_error:
            execute_candidate_runtime_localization_recovery(
                _runtime_stop(self.root, self.identity.run_id),
                config=self._config(1, recovery_name="existing_source"),
                effects=self._effects([], admit=precreated_source),
            )
        self.assertEqual(source_error.exception.phase, "route_source_admission")
        self.assertFalse(self.replacement_attempts)


if __name__ == "__main__":
    unittest.main()
