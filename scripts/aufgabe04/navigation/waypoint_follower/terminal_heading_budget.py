"""Pure lifecycle policy for the final terminal-heading time budget.

The ordinary waypoint deadline remains responsible for initial alignment and
translation.  Once the final controller target enters ``terminal_heading``, a
separate bounded clock owns convergence.  Keeping this state in a small value
object makes route replacement and target-change resets explicit and prevents
controller-mode chatter from silently extending the deadline.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


FINAL_TERMINAL_HEADING_PROGRESS_MODE = "terminal_heading"
TERMINAL_HEADING_TIMEOUT = "terminal heading timeout"

# A pi-radian correction needs about 17.46 s at the real TurtleBot ceiling of
# 0.18 rad/s.  The remaining 6.5 s covers angular acceleration, low-error
# proportional slowdown, and normal 10 Hz scheduling jitter without making
# the recovery phase unbounded.
DEFAULT_TERMINAL_HEADING_TIMEOUT_SEC = 24.0


@dataclass(frozen=True)
class TerminalHeadingBudgetState:
    """One target-bound, monotonic terminal-heading clock."""

    target_index: int | None = None
    started_at: float | None = None


@dataclass(frozen=True)
class TerminalHeadingBudgetDecision:
    """Updated immutable state plus the current phase outcome."""

    state: TerminalHeadingBudgetState
    active: bool
    elapsed_sec: float | None = None
    failure: str = ""


def reset_terminal_heading_budget(
    *,
    target_index: int | None = None,
) -> TerminalHeadingBudgetState:
    """Return a fresh, unarmed clock for a target or route revision."""

    return TerminalHeadingBudgetState(target_index=target_index)


def terminal_heading_budget_decision(
    state: TerminalHeadingBudgetState | None,
    *,
    target_index: int,
    final_target_index: int,
    progress_mode: str,
    now_monotonic: float,
    timeout_sec: float,
    entry_allowed: bool = True,
) -> TerminalHeadingBudgetDecision:
    """Arm once on final-heading entry and evaluate without chatter resets.

    A target change implicitly resets the prior state.  A material route
    replacement must call :func:`reset_terminal_heading_budget` explicitly,
    because a replacement can legitimately retain the same target index.
    Once armed, leaving and re-entering ``terminal_heading`` does not reset the
    clock.  Leaving the mode makes this phase inactive without discarding its
    timestamp, so the caller can enforce the ordinary waypoint deadline while
    outside final-heading control.  ``entry_allowed`` prevents a target that
    already exhausted that ordinary deadline from acquiring a fresh budget.
    """

    if not math.isfinite(now_monotonic):
        raise ValueError("now_monotonic must be finite")
    if not math.isfinite(timeout_sec) or timeout_sec <= 0.0:
        raise ValueError("timeout_sec must be finite and positive")

    current = state or reset_terminal_heading_budget(target_index=target_index)
    if current.target_index != target_index:
        current = reset_terminal_heading_budget(target_index=target_index)

    entering_final_heading = (
        target_index == final_target_index
        and progress_mode == FINAL_TERMINAL_HEADING_PROGRESS_MODE
    )
    if (
        current.started_at is None
        and entering_final_heading
        and entry_allowed
    ):
        current = TerminalHeadingBudgetState(
            target_index=target_index,
            started_at=now_monotonic,
        )

    if current.started_at is None or not entering_final_heading:
        return TerminalHeadingBudgetDecision(current, active=False)

    elapsed_sec = max(0.0, now_monotonic - current.started_at)
    failure = TERMINAL_HEADING_TIMEOUT if elapsed_sec > timeout_sec else ""
    return TerminalHeadingBudgetDecision(
        current,
        active=True,
        elapsed_sec=elapsed_sec,
        failure=failure,
    )


__all__ = [
    "DEFAULT_TERMINAL_HEADING_TIMEOUT_SEC",
    "FINAL_TERMINAL_HEADING_PROGRESS_MODE",
    "TERMINAL_HEADING_TIMEOUT",
    "TerminalHeadingBudgetDecision",
    "TerminalHeadingBudgetState",
    "reset_terminal_heading_budget",
    "terminal_heading_budget_decision",
]
