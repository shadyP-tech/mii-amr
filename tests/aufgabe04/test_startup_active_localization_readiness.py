from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.startup_active_localization import (
    StartupActiveLocalizationConfig,
)
from scripts.aufgabe04.navigation.missions.startup_route_uncertainty_selection import (
    StartupRouteUncertaintySelectionRejected,
)
from scripts.aufgabe04.real_robot.readiness.active_localization import (
    StartupActiveLocalizationPlanningConfig,
    StartupActiveLocalizationPlanningEffects,
    plan_with_optional_startup_active_localization,
)


def _motion_config(*, enabled: bool = True, attempts: int = 1):
    return StartupActiveLocalizationConfig(
        enabled=enabled,
        max_attempts=attempts,
        rotation_rad=0.35,
        angular_speed_radps=0.12,
        timeout_sec=8.0,
    )


def _completed_result():
    return {
        "status": "completed",
        "motion_published": True,
        "route_authorized": False,
        "mission_run_authorized": False,
        "requires_fresh_stationary_localization": True,
        "requires_separate_mission_run": True,
    }


class StartupActiveLocalizationReadinessTest(unittest.TestCase):
    def test_disabled_path_preserves_single_motion_free_planner_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_root = Path(tmp) / "session"
            attempts = []
            events = []

            outcome = plan_with_optional_startup_active_localization(
                StartupActiveLocalizationPlanningConfig(
                    session_root=session_root,
                    motion=_motion_config(enabled=False),
                ),
                StartupActiveLocalizationPlanningEffects(
                    plan_initial_route=lambda attempt: (
                        attempts.append(attempt) or 0
                    ),
                    run_active_localization=lambda *_: self.fail(
                        "active localization must remain disabled"
                    ),
                    admit_stationary_localization=lambda _: self.fail(
                        "disabled planning must not recollect localization"
                    ),
                    append_event=lambda path, event: events.append(
                        (path, event)
                    ),
                ),
                initial_start=Pose2D(1.0, 2.0, 0.1),
            )

        self.assertEqual(outcome.planning_status, 0)
        self.assertEqual(outcome.planning_attempt_count, 1)
        self.assertEqual(outcome.active_localization_attempt_count, 0)
        self.assertEqual(len(attempts), 1)
        self.assertFalse(attempts[0].propagate_route_selection_rejection)
        self.assertIsNone(attempts[0].selection_evidence_path)
        self.assertEqual(events, [])

    def test_exact_rejection_runs_localize_then_fresh_amcl_then_replans(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_root = Path(tmp) / "session"
            planning_attempts = []
            motion_calls = []
            admitted_paths = []
            events = []

            def plan(attempt):
                planning_attempts.append(attempt)
                if attempt.planning_attempt_index == 0:
                    raise StartupRouteUncertaintySelectionRejected(
                        evidence_path=attempt.selection_evidence_path,
                        evidence_sha256="a" * 64,
                        reason="no_accepted_route_options",
                    )
                return 0

            def run_motion(attempt_index, rejection):
                motion_calls.append((attempt_index, rejection))
                return _completed_result()

            def admit(path):
                admitted_paths.append(path)
                return Pose2D(1.1, 2.1, 0.2)

            outcome = plan_with_optional_startup_active_localization(
                StartupActiveLocalizationPlanningConfig(
                    session_root=session_root,
                    motion=_motion_config(),
                ),
                StartupActiveLocalizationPlanningEffects(
                    plan_initial_route=plan,
                    run_active_localization=run_motion,
                    admit_stationary_localization=admit,
                    append_event=lambda path, event: events.append(
                        (path, event)
                    ),
                    wall_clock=lambda: 123.0,
                ),
                initial_start=Pose2D(1.0, 2.0, 0.1),
            )

        attempt_root = session_root / "startup_active_localization"
        self.assertEqual(outcome.start, Pose2D(1.1, 2.1, 0.2))
        self.assertEqual(outcome.planning_attempt_count, 2)
        self.assertEqual(outcome.active_localization_attempt_count, 1)
        self.assertEqual([call[0] for call in motion_calls], [0])
        self.assertEqual(
            admitted_paths,
            [
                attempt_root
                / "attempt_000/post_motion_preplanning_localization.json"
            ],
        )
        self.assertEqual(
            planning_attempts[0].selection_evidence_path,
            attempt_root
            / "attempt_000/startup_route_uncertainty_selection.json",
        )
        self.assertEqual(
            planning_attempts[1].selection_evidence_path,
            attempt_root
            / "attempt_001/startup_route_uncertainty_selection.json",
        )
        self.assertEqual(
            [event[1]["event"] for event in events],
            [
                "startup_active_localization_enabled",
                "startup_active_localization_scheduled",
                "startup_active_localization_completed",
            ],
        )
        self.assertFalse(events[-1][1]["mission_run_authorized"])

    def test_budget_exhaustion_stops_before_second_localization_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_root = Path(tmp) / "session"
            motion_calls = []

            def plan(attempt):
                raise StartupRouteUncertaintySelectionRejected(
                    evidence_path=attempt.selection_evidence_path,
                    evidence_sha256="b" * 64,
                    reason="no_accepted_route_options",
                )

            effects = StartupActiveLocalizationPlanningEffects(
                plan_initial_route=plan,
                run_active_localization=lambda index, rejection: (
                    motion_calls.append(index) or _completed_result()
                ),
                admit_stationary_localization=lambda _: Pose2D(0.0, 0.0, 0.0),
                append_event=lambda *_: None,
            )
            with self.assertRaisesRegex(RuntimeError, "budget is exhausted"):
                plan_with_optional_startup_active_localization(
                    StartupActiveLocalizationPlanningConfig(
                        session_root=session_root,
                        motion=_motion_config(attempts=1),
                    ),
                    effects,
                    initial_start=Pose2D(0.0, 0.0, 0.0),
                )

        self.assertEqual(motion_calls, [0])

    def test_unrelated_planner_failure_is_never_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = StartupActiveLocalizationPlanningConfig(
                session_root=Path(tmp) / "session",
                motion=_motion_config(),
            )
            effects = StartupActiveLocalizationPlanningEffects(
                plan_initial_route=lambda _: (_ for _ in ()).throw(
                    ValueError("map is invalid")
                ),
                run_active_localization=lambda *_: self.fail(
                    "unrelated failures must not move"
                ),
                admit_stationary_localization=lambda _: self.fail(
                    "unrelated failures must not retry AMCL"
                ),
                append_event=lambda *_: None,
            )
            with self.assertRaisesRegex(ValueError, "map is invalid"):
                plan_with_optional_startup_active_localization(
                    config,
                    effects,
                    initial_start=Pose2D(0.0, 0.0, 0.0),
                )

    def test_rejection_from_wrong_evidence_path_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = StartupActiveLocalizationPlanningConfig(
                session_root=Path(tmp) / "session",
                motion=_motion_config(),
            )
            effects = StartupActiveLocalizationPlanningEffects(
                plan_initial_route=lambda _: (_ for _ in ()).throw(
                    StartupRouteUncertaintySelectionRejected(
                        evidence_path=Path(tmp) / "wrong.json",
                        evidence_sha256="c" * 64,
                        reason="no_accepted_route_options",
                    )
                ),
                run_active_localization=lambda *_: self.fail(
                    "wrong evidence must not authorize motion"
                ),
                admit_stationary_localization=lambda _: self.fail(
                    "wrong evidence must not trigger AMCL"
                ),
                append_event=lambda *_: None,
            )
            with self.assertRaisesRegex(RuntimeError, "not bound"):
                plan_with_optional_startup_active_localization(
                    config,
                    effects,
                    initial_start=Pose2D(0.0, 0.0, 0.0),
                )


if __name__ == "__main__":
    unittest.main()
