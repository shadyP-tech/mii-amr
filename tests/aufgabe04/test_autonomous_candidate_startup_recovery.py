from dataclasses import dataclass, replace
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.odom_execution_certificate import (
    PlanarTransform2D,
)
from scripts.aufgabe04.navigation.odom_route_adapter import (
    OdomExecutionContext,
    evaluate_map_odom_continuity,
)
from scripts.aufgabe04.navigation.startup_reseal_motion_authorization import (
    STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY,
)
from scripts.aufgabe04.real_robot.autonomous_candidate_startup_recovery import (
    CandidateRoutineIdentity,
    CandidateStartupRecoveryAttempt,
    CandidateStartupRecoveryConfig,
    CandidateStartupRecoveryEffects,
    CandidateStartupRecoveryError,
    execute_candidate_motion_with_startup_recovery,
)
from scripts.aufgabe04.real_robot.autonomous_child_runner import MotionLegOutcome


@dataclass(frozen=True)
class _Request:
    identity: CandidateRoutineIdentity


def _outcome(
    root: Path,
    *,
    run_id: str,
    status: str,
    stop_reason: str = "",
    stop_details: dict[str, object] | None = None,
    motion_published: bool = False,
    **permit_fields: object,
) -> MotionLegOutcome:
    semantic_log = root / f"{run_id}.jsonl"
    semantic_log.parent.mkdir(parents=True, exist_ok=True)
    semantic_log.write_text("{}\n", encoding="utf-8")
    return MotionLegOutcome(
        run_id=run_id,
        status=status,
        stop_reason=stop_reason,
        stop_details={} if stop_details is None else stop_details,
        motion_published=motion_published,
        returncode=0 if status == "completed" else 2,
        semantic_log_path=semantic_log,
        **permit_fields,
    )


def _mismatch(root: Path, run_id: str, **kwargs: object) -> MotionLegOutcome:
    return _outcome(
        root,
        run_id=run_id,
        status="stopped",
        stop_reason="pose outside certified startup segment",
        stop_details={
            "source": "execution_route_certificate",
            "phase": "before_motion_confirmation",
            "route_pose": {"x_m": 0.98, "y_m": -0.07, "yaw_rad": 0.0},
        },
        **kwargs,
    )


def _prestart_continuity(root: Path, run_id: str) -> MotionLegOutcome:
    context = OdomExecutionContext(
        map_frame="map",
        odom_frame="odom",
        base_frame="base_footprint",
        frozen_map_from_odom=PlanarTransform2D(0.0, 0.0, 0.0),
        certificate_sha256="a" * 64,
        max_map_from_odom_translation_drift_m=0.03,
        max_map_from_odom_yaw_drift_rad=0.04,
    )
    continuity = evaluate_map_odom_continuity(
        context,
        PlanarTransform2D(0.08, 0.0, 0.0),
    )
    reason = "global localization consistency requires zero and reseal"
    return _outcome(
        root,
        run_id=run_id,
        status="stopped",
        stop_reason=reason,
        stop_details={
            "reason": reason,
            "fault_code": "localization_reseal_required",
            "source": "global_consistency_monitor",
            "execution_phase": "before_motion",
            "phase": "initial_runtime_input_wait",
            "execution_pose_owner": "odom",
            "global_consistency_monitor": "amcl",
            "monitor_action": "FORCE_ZERO_RESEAL",
            "monitor_reason": "reseal_required",
            "monitor_warning": "",
            "motion_published": False,
            "continuity": continuity.to_evidence(),
            "fail_closed": True,
        },
    )


class CandidateStartupRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = CandidateRoutineIdentity(
            session_id="mission",
            semantic_map_id="arena",
            routine_kind="candidate_preapproach",
            routine_index=4,
            target_id="survey_candidate_0005",
            run_id="mission_candidate_004",
        )
        self.events: list[dict[str, object]] = []
        self.admitted_paths: list[Path] = []
        self.replanned_attempts: list[CandidateStartupRecoveryAttempt] = []
        self.replacement_attempts: list[CandidateStartupRecoveryAttempt] = []

    def _config(self, budget: int) -> CandidateStartupRecoveryConfig:
        return CandidateStartupRecoveryConfig(
            initial_identity=self.identity,
            recovery_root=self.root / "startup_reseals",
            event_log_path=self.root / "events.jsonl",
            max_startup_reseals=budget,
        )

    def _admit(self, path: Path) -> Pose2D:
        self.admitted_paths.append(path)
        path.parent.mkdir(parents=True, exist_ok=False)
        path.write_text('{"accepted":true}\n', encoding="utf-8")
        return Pose2D(0.987, -0.070, 0.01)

    def _replan(self, attempt: CandidateStartupRecoveryAttempt) -> _Request:
        self.replanned_attempts.append(attempt)
        attempt.source_root.mkdir(parents=False, exist_ok=False)
        (attempt.source_root / "route.csv").write_text(
            "x,y\n0.987,-0.070\n", encoding="utf-8"
        )
        return _Request(attempt.identity)

    def _event_sink(self, path: Path, payload: dict[str, object]) -> None:
        self.assertEqual(path, self.root / "events.jsonl")
        self.events.append(payload)

    def _effects(
        self,
        *,
        initial: object,
        replacements: list[object],
        admit=None,
        replan=None,
    ) -> tuple[CandidateStartupRecoveryEffects[_Request], list[_Request]]:
        initial_calls: list[_Request] = []
        replacement_values = list(replacements)

        def run_initial(request: _Request) -> MotionLegOutcome:
            initial_calls.append(request)
            if isinstance(initial, Exception):
                raise initial
            return initial  # type: ignore[return-value]

        def run_replacement(
            request: _Request,
            attempt: CandidateStartupRecoveryAttempt,
        ) -> MotionLegOutcome:
            self.replacement_attempts.append(attempt)
            value = replacement_values.pop(0)
            if isinstance(value, Exception):
                raise value
            return value  # type: ignore[return-value]

        return (
            CandidateStartupRecoveryEffects(
                run_initial=run_initial,
                run_replacement=run_replacement,
                admit_fresh_stationary_localization=(
                    self._admit if admit is None else admit
                ),
                replan_same_routine=(
                    self._replan if replan is None else replan
                ),
                describe_request=lambda request: request.identity,
                event_sink=self._event_sink,
                clock=lambda: 123.0,
            ),
            initial_calls,
        )

    def test_success_after_one_retry_uses_deterministic_immutable_handoff(self):
        retry_identity = self.identity.replacement(1)
        effects, initial_calls = self._effects(
            initial=_mismatch(self.root, self.identity.run_id),
            replacements=[
                _outcome(
                    self.root,
                    run_id=retry_identity.run_id,
                    status="completed",
                    motion_published=True,
                )
            ],
        )

        result = execute_candidate_motion_with_startup_recovery(
            _Request(self.identity),
            config=self._config(2),
            effects=effects,
        )

        self.assertEqual(result.run_id, retry_identity.run_id)
        self.assertEqual(len(initial_calls), 1)
        self.assertEqual(len(self.replacement_attempts), 1)
        attempt = self.replacement_attempts[0]
        self.assertEqual(attempt.identity, retry_identity)
        self.assertEqual(attempt.reseal_index, 1)
        self.assertEqual(
            attempt.attempt_root,
            (self.root / "startup_reseals/startup_reseal_001").resolve(),
        )
        self.assertEqual(
            attempt.fresh_localization_evidence_path,
            attempt.attempt_root / "fresh_stationary_localization.json",
        )
        self.assertEqual(attempt.source_root, attempt.attempt_root / "route_source")
        self.assertTrue(attempt.fresh_localization_evidence_path.is_file())
        self.assertTrue(attempt.source_root.is_dir())
        self.assertEqual(
            [event["event"] for event in self.events],
            [
                "candidate_startup_recovery_started",
                "candidate_startup_localization_admitted",
                "candidate_startup_route_replanned",
                "candidate_startup_recovery_completed",
            ],
        )

    def test_multiple_retries_preserve_routine_identity_and_budget(self):
        retry_one = self.identity.replacement(1)
        retry_two = self.identity.replacement(2)
        effects, _calls = self._effects(
            initial=_mismatch(self.root, self.identity.run_id),
            replacements=[
                _mismatch(self.root, retry_one.run_id),
                _outcome(self.root, run_id=retry_two.run_id, status="completed"),
            ],
        )

        result = execute_candidate_motion_with_startup_recovery(
            _Request(self.identity),
            config=self._config(2),
            effects=effects,
        )

        self.assertEqual(result.run_id, retry_two.run_id)
        self.assertEqual(
            [attempt.reseal_index for attempt in self.replacement_attempts],
            [1, 2],
        )
        self.assertEqual(
            [attempt.identity.run_id for attempt in self.replacement_attempts],
            [retry_one.run_id, retry_two.run_id],
        )
        self.assertTrue(
            all(
                attempt.identity.target_id == self.identity.target_id
                and attempt.identity.routine_kind == self.identity.routine_kind
                for attempt in self.replacement_attempts
            )
        )

    def test_prestart_localization_continuity_is_distinguished(self):
        retry = self.identity.replacement(1)
        effects, _calls = self._effects(
            initial=_prestart_continuity(self.root, self.identity.run_id),
            replacements=[
                _outcome(self.root, run_id=retry.run_id, status="completed")
            ],
        )

        execute_candidate_motion_with_startup_recovery(
            _Request(self.identity),
            config=self._config(1),
            effects=effects,
        )

        self.assertEqual(
            self.replacement_attempts[0].recovery_source_kind,
            STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY,
        )

    def test_wrong_outcome_run_id_fails_before_recovery(self):
        effects, _calls = self._effects(
            initial=_mismatch(self.root, "another_run"),
            replacements=[],
        )

        with self.assertRaisesRegex(
            CandidateStartupRecoveryError,
            "identity mismatch",
        ):
            execute_candidate_motion_with_startup_recovery(
                _Request(self.identity),
                config=self._config(2),
                effects=effects,
            )

        self.assertFalse(self.admitted_paths)
        self.assertFalse(self.replacement_attempts)

    def test_replacement_request_cannot_change_routine_identity(self):
        def wrong_replan(attempt: CandidateStartupRecoveryAttempt) -> _Request:
            attempt.source_root.mkdir(parents=False)
            return _Request(replace(attempt.identity, target_id="another"))

        effects, _calls = self._effects(
            initial=_mismatch(self.root, self.identity.run_id),
            replacements=[],
            replan=wrong_replan,
        )

        with self.assertRaisesRegex(
            CandidateStartupRecoveryError,
            "committed routine identity",
        ):
            execute_candidate_motion_with_startup_recovery(
                _Request(self.identity),
                config=self._config(1),
                effects=effects,
            )

        self.assertFalse(self.replacement_attempts)

    def test_issued_permit_is_retained_as_consumed_rejection_evidence(self):
        permit_cases = (
            {"motion_authorization_permit_path": self.root / "runtime.json"},
            {"mission_leg_motion_permit_sha256": "a" * 64},
            {"startup_reseal_motion_permit_path": self.root / "startup.json"},
        )
        for permit_fields in permit_cases:
            with self.subTest(permit_fields=permit_fields):
                self.events.clear()
                effects, _calls = self._effects(
                    initial=_mismatch(
                        self.root,
                        self.identity.run_id,
                        **permit_fields,
                    ),
                    replacements=[
                        _outcome(
                            self.root,
                            run_id=self.identity.replacement(1).run_id,
                            status="completed",
                        )
                    ],
                )
                execute_candidate_motion_with_startup_recovery(
                    _Request(self.identity),
                    config=replace(
                        self._config(1),
                        recovery_root=(
                            self.root
                            / "permit_cases"
                            / str(len(self.admitted_paths))
                        ),
                    ),
                    effects=effects,
                )
                started = self.events[0]
                self.assertTrue(
                    started["source_rejection_issued_motion_permit"]
                )
                self.assertTrue(
                    started["source_rejection_issued_motion_permit_kinds"]
                )
                self.assertEqual(
                    started["source_rejection_stop_reason"],
                    "pose outside certified startup segment",
                )
                self.assertEqual(
                    started["source_rejection_stop_details"]["source"],
                    "execution_route_certificate",
                )

    def test_motion_published_replacement_preserves_child_failure_and_stops(self):
        retry_one = self.identity.replacement(1)
        retry_two = self.identity.replacement(2)
        stop_details = {
            "reason": "stuck no progress",
            "fault_code": "no_progress",
            "elapsed_sec": 11.4,
            "remaining_distance_m": 0.215,
            "controller_mode_reversals": 11,
        }
        effects, _calls = self._effects(
            initial=_mismatch(self.root, self.identity.run_id),
            replacements=[
                _outcome(
                    self.root,
                    run_id=retry_one.run_id,
                    status="stopped",
                    stop_reason="stuck no progress",
                    stop_details=stop_details,
                    motion_published=True,
                ),
                _outcome(
                    self.root,
                    run_id=retry_two.run_id,
                    status="completed",
                ),
            ],
        )

        with self.assertRaisesRegex(
            CandidateStartupRecoveryError,
            "stuck no progress",
        ) as caught:
            execute_candidate_motion_with_startup_recovery(
                _Request(self.identity),
                config=self._config(3),
                effects=effects,
            )

        self.assertIn(
            "fail-closed policy: rejected candidate run published motion",
            str(caught.exception),
        )
        self.assertEqual(len(self.admitted_paths), 1)
        self.assertEqual(len(self.replanned_attempts), 1)
        self.assertEqual(len(self.replacement_attempts), 1)
        self.assertEqual(
            [event["event"] for event in self.events],
            [
                "candidate_startup_recovery_started",
                "candidate_startup_localization_admitted",
                "candidate_startup_route_replanned",
                "candidate_startup_recovery_rejected",
            ],
        )
        self.assertFalse(
            self.events[0]["source_rejection_published_motion"]
        )

        rejected = self.events[-1]
        self.assertEqual(rejected["reason"], "stuck no progress")
        self.assertEqual(
            rejected["rejection_policy_reason"],
            "rejected candidate run published motion",
        )
        self.assertEqual(rejected["stop_details"], stop_details)
        self.assertEqual(rejected["rejected_stop_reason"], "stuck no progress")
        self.assertEqual(rejected["rejected_stop_details"], stop_details)
        self.assertTrue(rejected["motion_published"])

        failure = caught.exception.to_failure_fields()
        self.assertEqual(failure["failure_phase"], "candidate_startup_recovery")
        self.assertEqual(failure["stop_reason"], "stuck no progress")
        self.assertTrue(failure["motion_published"])
        self.assertFalse(failure["motion_continues_authorized"])
        self.assertTrue(failure["fail_closed"])

    def test_motion_and_noneligible_rejections_fail_closed(self):
        cases = (
            _mismatch(
                self.root,
                self.identity.run_id,
                motion_published=True,
            ),
            _outcome(
                self.root,
                run_id=self.identity.run_id,
                status="stopped",
                stop_reason="camera unavailable",
            ),
        )
        for outcome in cases:
            with self.subTest(stop_reason=outcome.stop_reason):
                self.events.clear()
                effects, _calls = self._effects(
                    initial=outcome,
                    replacements=[],
                )
                with self.assertRaises(CandidateStartupRecoveryError):
                    execute_candidate_motion_with_startup_recovery(
                        _Request(self.identity),
                        config=self._config(1),
                        effects=effects,
                    )
                self.assertEqual(
                    self.events[-1]["event"],
                    "candidate_startup_recovery_rejected",
                )

    def test_callback_failures_are_terminal_and_evidenced(self):
        def failing_admit(_path: Path) -> Pose2D:
            raise RuntimeError("amcl unavailable")

        def failing_replan(attempt: CandidateStartupRecoveryAttempt) -> _Request:
            raise RuntimeError(f"cannot route {attempt.identity.target_id}")

        cases = (
            ("initial_run", RuntimeError("child unavailable"), [], None, None),
            (
                "stationary_localization_admission",
                _mismatch(self.root, self.identity.run_id),
                [],
                failing_admit,
                None,
            ),
            (
                "same_routine_replan",
                _mismatch(self.root, self.identity.run_id),
                [],
                None,
                failing_replan,
            ),
            (
                "replacement_run",
                _mismatch(self.root, self.identity.run_id),
                [RuntimeError("replacement unavailable")],
                None,
                None,
            ),
        )
        for phase, initial, replacements, admit, replan in cases:
            with self.subTest(phase=phase):
                self.events.clear()
                self.admitted_paths.clear()
                self.replanned_attempts.clear()
                self.replacement_attempts.clear()
                recovery_root = self.root / f"callback_{phase}"
                config = replace(self._config(1), recovery_root=recovery_root)
                effects, _calls = self._effects(
                    initial=initial,
                    replacements=replacements,
                    admit=admit,
                    replan=replan,
                )
                with self.assertRaises(CandidateStartupRecoveryError):
                    execute_candidate_motion_with_startup_recovery(
                        _Request(self.identity),
                        config=config,
                        effects=effects,
                    )
                self.assertEqual(
                    self.events[-1]["event"],
                    "candidate_startup_recovery_failed",
                )
                self.assertEqual(self.events[-1]["phase"], phase)

    def test_budget_exhaustion_never_runs_attempt_n_plus_one(self):
        retry_one = self.identity.replacement(1)
        retry_two = self.identity.replacement(2)
        effects, initial_calls = self._effects(
            initial=_mismatch(self.root, self.identity.run_id),
            replacements=[
                _mismatch(self.root, retry_one.run_id),
                _mismatch(self.root, retry_two.run_id),
            ],
        )

        with self.assertRaisesRegex(
            CandidateStartupRecoveryError,
            "budget exhausted",
        ) as caught:
            execute_candidate_motion_with_startup_recovery(
                _Request(self.identity),
                config=self._config(2),
                effects=effects,
            )

        self.assertEqual(len(initial_calls), 1)
        self.assertEqual(len(self.admitted_paths), 2)
        self.assertEqual(len(self.replanned_attempts), 2)
        self.assertEqual(len(self.replacement_attempts), 2)
        self.assertEqual(
            self.events[-1]["event"],
            "candidate_startup_recovery_exhausted",
        )
        self.assertEqual(
            self.events[-1]["rejected_stop_reason"],
            "pose outside certified startup segment",
        )
        self.assertEqual(
            self.events[-1]["rejected_stop_details"]["source"],
            "execution_route_certificate",
        )
        failure = caught.exception.to_failure_fields()
        self.assertEqual(
            failure["candidate_startup_recovery_phase"],
            "budget_exhausted",
        )
        self.assertEqual(
            failure["stop_reason"],
            "pose outside certified startup segment",
        )
        self.assertEqual(
            failure["stop_details"]["source"],
            "execution_route_certificate",
        )
        self.assertFalse(failure["motion_published"])
        self.assertFalse(failure["motion_continues_authorized"])


if __name__ == "__main__":
    unittest.main()
