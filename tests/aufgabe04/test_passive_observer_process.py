from __future__ import annotations

import dataclasses
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest

from scripts.aufgabe04.real_robot.passive_observer_process import (
    PassiveObserverProcessEvidence,
    monitor_passive_observer_process,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 10.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.sleeps.append(duration)
        self.now += duration


class _Process:
    def __init__(
        self,
        *,
        returncode: int | None = None,
        wait_outcomes: tuple[int | str, ...] = (),
    ) -> None:
        self.returncode = returncode
        self.wait_outcomes = list(wait_outcomes)
        self.wait_timeouts: list[float | None] = []
        self.signals: list[int] = []
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if not self.wait_outcomes:
            raise AssertionError("test process has no configured wait outcome")
        outcome = self.wait_outcomes.pop(0)
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(["passive-observer"], timeout)
        self.returncode = int(outcome)
        return self.returncode

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1


class PassiveObserverProcessTests(unittest.TestCase):
    def test_recommendation_completion_reaps_child_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recommendation = root / "recommendation.json"
            recommendation.write_text("{}")
            process = _Process(wait_outcomes=(0,))

            evidence = monitor_passive_observer_process(
                process=process,
                recommendation_path=recommendation,
                axis_observation_path=root / "axis_observation.json",
                timeout_sec=90.0,
            )

        self.assertEqual(evidence.completion_kind, "artifact")
        self.assertEqual(evidence.artifact_kind, "recommendation")
        self.assertEqual(evidence.artifact_path, recommendation)
        self.assertFalse(evidence.deadline_expired)
        self.assertEqual(evidence.returncode, 0)
        self.assertEqual(evidence.cleanup_actions, ("graceful_wait",))
        self.assertEqual(evidence.signals_sent, ())
        self.assertEqual(process.wait_timeouts, [3.0])
        self.assertIsNotNone(process.returncode)

    def test_axis_only_artifact_is_an_immediate_terminal_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            axis_observation = root / "axis_observation.json"
            axis_observation.write_text("{}")
            process = _Process(wait_outcomes=(0,))
            clock = _Clock()

            evidence = monitor_passive_observer_process(
                process=process,
                recommendation_path=root / "recommendation.json",
                axis_observation_path=axis_observation,
                timeout_sec=90.0,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )

        self.assertEqual(evidence.completion_kind, "artifact")
        self.assertEqual(evidence.artifact_kind, "axis_observation")
        self.assertEqual(evidence.artifact_path, axis_observation)
        self.assertEqual(clock.sleeps, [])
        self.assertEqual(evidence.returncode, 0)
        self.assertIsNotNone(process.returncode)

    def test_early_nonzero_child_exit_is_not_reported_as_a_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = _Process(returncode=7)

            evidence = monitor_passive_observer_process(
                process=process,
                recommendation_path=root / "recommendation.json",
                axis_observation_path=root / "axis_observation.json",
                timeout_sec=90.0,
            )

        self.assertEqual(evidence.completion_kind, "child_exit")
        self.assertIsNone(evidence.artifact_kind)
        self.assertIsNone(evidence.artifact_path)
        self.assertFalse(evidence.deadline_expired)
        self.assertEqual(evidence.returncode, 7)
        self.assertEqual(evidence.cleanup_actions, ("exit_observed",))
        self.assertEqual(evidence.signals_sent, ())
        self.assertEqual(process.wait_timeouts, [])

    def test_deadline_escalates_through_sigint_terminate_and_kill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = _Process(wait_outcomes=("timeout", "timeout", -9))
            clock = _Clock()

            evidence = monitor_passive_observer_process(
                process=process,
                recommendation_path=root / "recommendation.json",
                axis_observation_path=root / "axis_observation.json",
                timeout_sec=0.25,
                poll_interval_sec=0.1,
                sigint_wait_timeout_sec=1.0,
                terminate_wait_timeout_sec=2.0,
                kill_wait_timeout_sec=3.0,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )

        self.assertEqual(evidence.completion_kind, "deadline")
        self.assertTrue(evidence.deadline_expired)
        self.assertIsNone(evidence.artifact_kind)
        self.assertEqual(evidence.returncode, -9)
        self.assertEqual(evidence.signals_sent, ("SIGINT", "SIGTERM", "SIGKILL"))
        self.assertEqual(
            evidence.cleanup_actions,
            (
                "send_sigint",
                "wait_after_sigint",
                "terminate",
                "wait_after_terminate",
                "kill",
                "wait_after_kill",
            ),
        )
        self.assertEqual(process.signals, [signal.SIGINT])
        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 1)
        self.assertEqual(process.wait_timeouts, [1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(clock.sleeps), 0.25)
        self.assertIsNotNone(process.returncode)

    def test_unexpected_graceful_wait_timeout_still_cleans_up_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recommendation = root / "recommendation.json"
            recommendation.write_text("{}")
            process = _Process(wait_outcomes=("timeout", "timeout", -15))

            evidence = monitor_passive_observer_process(
                process=process,
                recommendation_path=recommendation,
                axis_observation_path=root / "axis_observation.json",
                timeout_sec=90.0,
                graceful_wait_timeout_sec=0.5,
                sigint_wait_timeout_sec=0.75,
                terminate_wait_timeout_sec=1.0,
            )

        self.assertEqual(evidence.completion_kind, "artifact")
        self.assertEqual(evidence.artifact_kind, "recommendation")
        self.assertEqual(evidence.returncode, -15)
        self.assertEqual(evidence.signals_sent, ("SIGINT", "SIGTERM"))
        self.assertEqual(
            evidence.cleanup_actions,
            (
                "graceful_wait",
                "send_sigint",
                "wait_after_sigint",
                "terminate",
                "wait_after_terminate",
            ),
        )
        self.assertEqual(process.wait_timeouts, [0.5, 0.75, 1.0])
        self.assertEqual(process.signals, [signal.SIGINT])
        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 0)
        self.assertIsNotNone(process.returncode)

    def test_evidence_is_frozen(self) -> None:
        evidence = PassiveObserverProcessEvidence(
            completion_kind="child_exit",
            artifact_kind=None,
            artifact_path=None,
            deadline_expired=False,
            returncode=1,
            cleanup_actions=("exit_observed",),
            signals_sent=(),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            evidence.returncode = 0

        self.assertEqual(
            evidence.to_dict(),
            {
                "schema_version": 1,
                "completion_kind": "child_exit",
                "artifact_kind": None,
                "artifact_path": None,
                "deadline_expired": False,
                "returncode": 1,
                "cleanup_actions": ["exit_observed"],
                "signals_sent": [],
            },
        )


if __name__ == "__main__":
    unittest.main()
