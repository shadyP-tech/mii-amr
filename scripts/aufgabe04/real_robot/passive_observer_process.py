"""Bounded, motion-neutral lifecycle control for the passive observer child.

The autonomous runner starts the observer because it owns the command-line
configuration.  This module owns only the subsequent subprocess lifecycle:
wait for either terminal perception artifact, distinguish why waiting stopped,
and synchronously reap the child through a bounded escalation sequence.

There are deliberately no ROS imports or publishers here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import signal
import subprocess
import time
from typing import Callable, Literal, Protocol


ObserverArtifactKind = Literal["recommendation", "axis_observation"]
ObserverCompletionKind = Literal["artifact", "deadline", "child_exit"]
ObserverCleanupAction = Literal[
    "exit_observed",
    "graceful_wait",
    "send_sigint",
    "wait_after_sigint",
    "terminate",
    "wait_after_terminate",
    "kill",
    "wait_after_kill",
]
ObserverSignal = Literal["SIGINT", "SIGTERM", "SIGKILL"]


class ObserverProcess(Protocol):
    """Small subprocess surface required by the lifecycle monitor."""

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def send_signal(self, sig: int) -> None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


class PassiveObserverProcessError(RuntimeError):
    """Raised when the child cannot be synchronously reaped as required."""

    def __init__(
        self,
        message: str,
        *,
        cleanup_actions: tuple[ObserverCleanupAction, ...],
        signals_sent: tuple[ObserverSignal, ...],
    ) -> None:
        super().__init__(message)
        self.cleanup_actions = cleanup_actions
        self.signals_sent = signals_sent


@dataclass(frozen=True)
class PassiveObserverProcessEvidence:
    """Immutable evidence for one completely reaped observer child."""

    completion_kind: ObserverCompletionKind
    artifact_kind: ObserverArtifactKind | None
    artifact_path: Path | None
    deadline_expired: bool
    returncode: int
    cleanup_actions: tuple[ObserverCleanupAction, ...]
    signals_sent: tuple[ObserverSignal, ...]

    def __post_init__(self) -> None:
        artifact_completion = self.completion_kind == "artifact"
        if artifact_completion != (self.artifact_kind is not None):
            raise ValueError(
                "artifact completion must identify exactly one artifact kind"
            )
        if artifact_completion != (self.artifact_path is not None):
            raise ValueError(
                "artifact completion must identify exactly one artifact path"
            )
        if self.deadline_expired != (self.completion_kind == "deadline"):
            raise ValueError(
                "deadline_expired must match deadline completion"
            )

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-ready lifecycle evidence for the run bundle."""

        return {
            "schema_version": 1,
            "completion_kind": self.completion_kind,
            "artifact_kind": self.artifact_kind,
            "artifact_path": (
                str(self.artifact_path)
                if self.artifact_path is not None
                else None
            ),
            "deadline_expired": self.deadline_expired,
            "returncode": self.returncode,
            "cleanup_actions": list(self.cleanup_actions),
            "signals_sent": list(self.signals_sent),
        }


@dataclass(frozen=True)
class _DetectedArtifact:
    kind: ObserverArtifactKind
    path: Path


def _require_duration(value: float, *, field: str, positive: bool) -> float:
    try:
        duration = float(value)
    except (TypeError, ValueError) as exc:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be finite and {qualifier}") from exc
    minimum_ok = duration > 0.0 if positive else duration >= 0.0
    if not math.isfinite(duration) or not minimum_ok:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be finite and {qualifier}")
    return duration


def _detect_artifact(
    *,
    recommendation_path: Path,
    axis_observation_path: Path,
) -> _DetectedArtifact | None:
    # A recommendation is the richer terminal result when both files become
    # visible between polls.  The axis-only file remains a valid completion.
    if recommendation_path.exists():
        return _DetectedArtifact("recommendation", recommendation_path)
    if axis_observation_path.exists():
        return _DetectedArtifact("axis_observation", axis_observation_path)
    return None


def _wait_bounded(
    process: ObserverProcess,
    *,
    timeout_sec: float,
    action: ObserverCleanupAction,
    actions: list[ObserverCleanupAction],
) -> int | None:
    actions.append(action)
    try:
        return int(process.wait(timeout=timeout_sec))
    except subprocess.TimeoutExpired:
        return None


def _signal_and_wait(
    process: ObserverProcess,
    *,
    action: ObserverCleanupAction,
    wait_action: ObserverCleanupAction,
    signal_name: ObserverSignal,
    timeout_sec: float,
    actions: list[ObserverCleanupAction],
    signals_sent: list[ObserverSignal],
    send: Callable[[], None],
) -> int | None:
    returncode = process.poll()
    if returncode is not None:
        actions.append("exit_observed")
        return int(returncode)

    actions.append(action)
    try:
        send()
    except ProcessLookupError:
        # The child exited between poll() and signal delivery.  wait() still
        # performs the required synchronous reap.
        pass
    else:
        signals_sent.append(signal_name)
    return _wait_bounded(
        process,
        timeout_sec=timeout_sec,
        action=wait_action,
        actions=actions,
    )


def _reap_process(
    process: ObserverProcess,
    *,
    allow_graceful_wait: bool,
    graceful_wait_timeout_sec: float,
    sigint_wait_timeout_sec: float,
    terminate_wait_timeout_sec: float,
    kill_wait_timeout_sec: float,
) -> tuple[
    int,
    tuple[ObserverCleanupAction, ...],
    tuple[ObserverSignal, ...],
]:
    actions: list[ObserverCleanupAction] = []
    signals_sent: list[ObserverSignal] = []

    returncode = process.poll()
    if returncode is not None:
        actions.append("exit_observed")
        return int(returncode), tuple(actions), tuple(signals_sent)

    if allow_graceful_wait:
        returncode = _wait_bounded(
            process,
            timeout_sec=graceful_wait_timeout_sec,
            action="graceful_wait",
            actions=actions,
        )
        if returncode is not None:
            return returncode, tuple(actions), tuple(signals_sent)

    returncode = _signal_and_wait(
        process,
        action="send_sigint",
        wait_action="wait_after_sigint",
        signal_name="SIGINT",
        timeout_sec=sigint_wait_timeout_sec,
        actions=actions,
        signals_sent=signals_sent,
        send=lambda: process.send_signal(signal.SIGINT),
    )
    if returncode is not None:
        return returncode, tuple(actions), tuple(signals_sent)

    returncode = _signal_and_wait(
        process,
        action="terminate",
        wait_action="wait_after_terminate",
        signal_name="SIGTERM",
        timeout_sec=terminate_wait_timeout_sec,
        actions=actions,
        signals_sent=signals_sent,
        send=process.terminate,
    )
    if returncode is not None:
        return returncode, tuple(actions), tuple(signals_sent)

    returncode = _signal_and_wait(
        process,
        action="kill",
        wait_action="wait_after_kill",
        signal_name="SIGKILL",
        timeout_sec=kill_wait_timeout_sec,
        actions=actions,
        signals_sent=signals_sent,
        send=process.kill,
    )
    if returncode is not None:
        return returncode, tuple(actions), tuple(signals_sent)

    raise PassiveObserverProcessError(
        "passive observer remained alive after bounded SIGKILL cleanup",
        cleanup_actions=tuple(actions),
        signals_sent=tuple(signals_sent),
    )


def monitor_passive_observer_process(
    *,
    process: ObserverProcess,
    recommendation_path: Path,
    axis_observation_path: Path,
    timeout_sec: float,
    poll_interval_sec: float = 0.1,
    graceful_wait_timeout_sec: float = 3.0,
    sigint_wait_timeout_sec: float = 5.0,
    terminate_wait_timeout_sec: float = 5.0,
    kill_wait_timeout_sec: float = 5.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> PassiveObserverProcessEvidence:
    """Wait for an observer artifact and synchronously reap its child.

    Artifact presence wins a same-poll race with child exit or the deadline.
    Once the deadline expires, no extra graceful wait is added before SIGINT.
    A returned evidence object therefore guarantees that ``returncode`` is
    concrete and the child has been reaped.  Failure to reap even after the
    bounded kill stage raises :class:`PassiveObserverProcessError`.
    """

    timeout = _require_duration(timeout_sec, field="timeout_sec", positive=True)
    poll_interval = _require_duration(
        poll_interval_sec,
        field="poll_interval_sec",
        positive=True,
    )
    graceful_wait_timeout = _require_duration(
        graceful_wait_timeout_sec,
        field="graceful_wait_timeout_sec",
        positive=False,
    )
    sigint_wait_timeout = _require_duration(
        sigint_wait_timeout_sec,
        field="sigint_wait_timeout_sec",
        positive=False,
    )
    terminate_wait_timeout = _require_duration(
        terminate_wait_timeout_sec,
        field="terminate_wait_timeout_sec",
        positive=False,
    )
    kill_wait_timeout = _require_duration(
        kill_wait_timeout_sec,
        field="kill_wait_timeout_sec",
        positive=False,
    )

    recommendation = Path(recommendation_path)
    axis_observation = Path(axis_observation_path)
    started_at = float(monotonic())
    if not math.isfinite(started_at):
        raise ValueError("monotonic() must return a finite value")
    deadline = started_at + timeout

    artifact: _DetectedArtifact | None = None
    completion_kind: ObserverCompletionKind
    observed_returncode: int | None = None
    while True:
        artifact = _detect_artifact(
            recommendation_path=recommendation,
            axis_observation_path=axis_observation,
        )
        if artifact is not None:
            completion_kind = "artifact"
            break

        observed_returncode = process.poll()
        if observed_returncode is not None:
            # Close the small file-publication/process-exit race before
            # classifying the terminal condition as an early exit.
            artifact = _detect_artifact(
                recommendation_path=recommendation,
                axis_observation_path=axis_observation,
            )
            completion_kind = "artifact" if artifact is not None else "child_exit"
            break

        now = float(monotonic())
        if not math.isfinite(now):
            raise ValueError("monotonic() must return a finite value")
        if now >= deadline:
            artifact = _detect_artifact(
                recommendation_path=recommendation,
                axis_observation_path=axis_observation,
            )
            completion_kind = "artifact" if artifact is not None else "deadline"
            break
        sleep(min(poll_interval, deadline - now))

    if completion_kind == "child_exit":
        assert observed_returncode is not None
        returncode = int(observed_returncode)
        cleanup_actions: tuple[ObserverCleanupAction, ...] = ("exit_observed",)
        signals_sent: tuple[ObserverSignal, ...] = ()
    else:
        returncode, cleanup_actions, signals_sent = _reap_process(
            process,
            allow_graceful_wait=completion_kind == "artifact",
            graceful_wait_timeout_sec=graceful_wait_timeout,
            sigint_wait_timeout_sec=sigint_wait_timeout,
            terminate_wait_timeout_sec=terminate_wait_timeout,
            kill_wait_timeout_sec=kill_wait_timeout,
        )

    return PassiveObserverProcessEvidence(
        completion_kind=completion_kind,
        artifact_kind=artifact.kind if artifact is not None else None,
        artifact_path=artifact.path if artifact is not None else None,
        deadline_expired=completion_kind == "deadline",
        returncode=returncode,
        cleanup_actions=cleanup_actions,
        signals_sent=signals_sent,
    )


__all__ = [
    "ObserverArtifactKind",
    "ObserverCleanupAction",
    "ObserverCompletionKind",
    "ObserverProcess",
    "ObserverSignal",
    "PassiveObserverProcessError",
    "PassiveObserverProcessEvidence",
    "monitor_passive_observer_process",
]
