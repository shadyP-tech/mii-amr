"""Live motion admission and recovery for one follower control cycle.

This component receives an already route-admitted controller step and decides
whether clearance and progress checks allow it to reach command preparation.
It may revoke motion, record the clearance-floor trace, and perform the existing
bounded blockage recovery.  It never prepares or publishes a nonzero command,
services callbacks, refreshes routes, or controls the normal loop cadence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Mapping

from scripts.aufgabe04.navigation.control.waypoint_controller import (
    ControllerStep,
    VelocityCommand,
)
from scripts.aufgabe04.navigation.coverage.transient_blockage_policy import (
    CLEARANCE_LIMITED_MOTION_FLOOR,
)
from scripts.aufgabe04.navigation.execution.execution_route_certificate import (
    ExecutionRouteCheck,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.waypoint_follower.directives import (
    StringDirective,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.command_admission import (
    CommandAdmissionDecision,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.recovery_dispatch import (
    BlockageRecoveryTrigger,
    RecoveryLoopAction,
)


MonotonicClock = Callable[[], float]


class MotionCycleGuardAction(StringDirective):
    """Next loop action after live motion admission and recovery."""

    PROCEED = "proceed"
    RETRY = "retry"
    STOP = "stop"


@dataclass(frozen=True)
class MotionCycleGuardDecision:
    """An admitted command or the zero/terminal effect already applied."""

    action: MotionCycleGuardAction
    command_admission: CommandAdmissionDecision | None = None
    evaluated_at: float | None = None
    stop_reason: str = ""
    stop_details: Mapping[str, object] | None = None


class MotionCycleGuardRuntimeMixin:
    """Apply clearance, progress, and bounded recovery before publication."""

    def _motion_cycle_guard_decision(
        self,
        pose: Pose2D,
        step: ControllerStep,
        route_check: ExecutionRouteCheck | None,
        loop_period_sec: float,
        *,
        monotonic_fn: MonotonicClock = time.monotonic,
    ) -> MotionCycleGuardDecision:
        """Admit one step to command preparation or apply its zero outcome."""

        evaluated_at = monotonic_fn()
        motion_admission = self._motion_command_admission_decision(step)
        command_admission = motion_admission.command_admission
        front_clearance_scale = motion_admission.front_clearance_scale
        effective_linear_x_mps = (
            command_admission.effective_command.linear_x_mps
        )

        if motion_admission.stop_details is not None:
            self.latest_stop_details = motion_admission.stop_details
            self.publish_repeated_zero()
            trace_failure = self._append_controller_trace(
                event="motion_floor_zero_hold",
                pose=pose,
                step=step,
                route_check=route_check,
                nominal_command=step.command,
                effective_command=VelocityCommand(0.0, 0.0),
                reason=CLEARANCE_LIMITED_MOTION_FLOOR,
                fail_closed=False,
            )
            if trace_failure:
                return MotionCycleGuardDecision(
                    MotionCycleGuardAction.STOP,
                    stop_reason=trace_failure,
                    stop_details=self.latest_stop_details,
                )

            front_evidence = self.latest_front_clearance_details or {}
            recovery_disposition = self._blockage_recovery_outcome(
                trigger=BlockageRecoveryTrigger.CLEARANCE_FLOOR,
                pose=pose,
                stop_reason=CLEARANCE_LIMITED_MOTION_FLOOR,
                stop_details=self.latest_stop_details,
                front_evidence=front_evidence,
                nominal_linear_x_mps=step.command.linear_x_mps,
            )
            if (
                recovery_disposition.action
                == RecoveryLoopAction.ZERO_HOLD_AND_RETRY
            ):
                self.publish_zero()
                self._hold_zero_control_period(loop_period_sec)
                return MotionCycleGuardDecision(MotionCycleGuardAction.RETRY)
            return MotionCycleGuardDecision(
                MotionCycleGuardAction.STOP,
                stop_reason=recovery_disposition.stop_reason,
                stop_details=self.latest_stop_details,
            )

        progress_decision = self._progress_watchdog_decision(
            step,
            now_monotonic=evaluated_at,
            front_clearance_scale=front_clearance_scale,
            effective_linear_x_mps=effective_linear_x_mps,
        )
        if progress_decision.failure:
            self.latest_stop_details = progress_decision.stop_details
            self.publish_repeated_zero()
            front_evidence = self.latest_front_clearance_details or {}
            recovery_disposition = self._blockage_recovery_outcome(
                trigger=BlockageRecoveryTrigger.STUCK_WATCHDOG,
                pose=pose,
                stop_reason=progress_decision.failure,
                stop_details=self.latest_stop_details,
                front_evidence=front_evidence,
                nominal_linear_x_mps=step.command.linear_x_mps,
            )
            if recovery_disposition.action == RecoveryLoopAction.HOLD_AND_RETRY:
                self._hold_zero_control_period(loop_period_sec)
                return MotionCycleGuardDecision(MotionCycleGuardAction.RETRY)
            return MotionCycleGuardDecision(
                MotionCycleGuardAction.STOP,
                stop_reason=recovery_disposition.stop_reason,
                stop_details=self.latest_stop_details,
            )

        return MotionCycleGuardDecision(
            MotionCycleGuardAction.PROCEED,
            command_admission=command_admission,
            evaluated_at=evaluated_at,
        )
