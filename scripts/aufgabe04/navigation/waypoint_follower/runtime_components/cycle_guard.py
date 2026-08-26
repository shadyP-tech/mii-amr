"""Fail-closed admission for one follower control-loop cycle.

The guard preserves the runtime ordering that matters on the TurtleBot: drain
callbacks, evaluate LiDAR/runtime safety, evaluate global localization, and
only then acquire the control pose.  It may publish zero commands and record a
terminal trace, but it never publishes motion or constructs a final result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from scripts.aufgabe04.navigation.control.follower_safety import (
    OBSTACLE_TOO_CLOSE,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.waypoint_follower.directives import (
    StringDirective,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.control_results import (
    with_controller_trace_failure,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.recovery_dispatch import (
    BlockageRecoveryTrigger,
    RecoveryLoopAction,
    front_sector_recovery_evidence,
)


class ControlCycleGuardAction(StringDirective):
    """Next loop action after fail-closed cycle admission."""

    PROCEED = "proceed"
    RETRY = "retry"
    STOP = "stop"


@dataclass(frozen=True)
class ControlCycleGuardDecision:
    """Typed cycle-admission result returned to the orchestration loop."""

    action: ControlCycleGuardAction
    pose: Pose2D | None = None
    stop_reason: str = ""
    stop_details: Mapping[str, object] | None = None


class ControlCycleGuardRuntimeMixin:
    """Apply the ordered sensor, localization, and pose gates for one cycle."""

    def _control_cycle_guard_decision(
        self,
        loop_period_sec: float,
    ) -> ControlCycleGuardDecision:
        self._drain_runtime_callbacks()
        safety_failure = self._safety_failure()
        if safety_failure:
            front_evidence = front_sector_recovery_evidence(
                self.latest_stop_details
            )
            if (
                safety_failure == OBSTACLE_TOO_CLOSE
                and self.blockage_recovery_provider is not None
                and front_evidence is not None
            ):
                self.publish_repeated_zero()
                recovery_pose = (
                    self._current_pose_lookup_with_stale_recovery().pose
                )
                recovery_disposition = self._blockage_recovery_outcome(
                    trigger=BlockageRecoveryTrigger.OBSTACLE_SAFETY_STOP,
                    pose=recovery_pose,
                    stop_reason=safety_failure,
                    stop_details=self.latest_stop_details,
                    front_evidence=front_evidence,
                )
                if (
                    recovery_disposition.action
                    == RecoveryLoopAction.HOLD_AND_RETRY
                ):
                    self._hold_zero_control_period(loop_period_sec)
                    return ControlCycleGuardDecision(
                        ControlCycleGuardAction.RETRY
                    )
                if (
                    recovery_disposition.action
                    == RecoveryLoopAction.ZERO_HOLD_AND_RETRY
                ):
                    # A separately confirmed clear front sector may resume
                    # only after the next complete runtime-safety cycle.
                    self.publish_zero()
                    self._hold_zero_control_period(loop_period_sec)
                    return ControlCycleGuardDecision(
                        ControlCycleGuardAction.RETRY
                    )
                safety_failure = recovery_disposition.stop_reason
            self.publish_repeated_zero()
            return ControlCycleGuardDecision(
                ControlCycleGuardAction.STOP,
                stop_reason=safety_failure,
                stop_details=self.latest_stop_details,
            )

        localization_failure = self._global_consistency_monitor_failure()
        if localization_failure:
            # Ordinary safety has already run. Revoke the preceding command
            # before returning the terminal localization evidence.
            self.publish_repeated_zero()
            return ControlCycleGuardDecision(
                ControlCycleGuardAction.STOP,
                stop_reason=localization_failure,
                stop_details=self.latest_stop_details,
            )

        # Safety discovery can briefly delay TF listener callbacks. Recover
        # only that resulting stale-transform case while zero remains held.
        pose_lookup = self._current_pose_lookup_with_stale_recovery()
        pose = pose_lookup.pose
        if pose is None:
            self.publish_repeated_zero()
            stop_reason = str(
                (pose_lookup.details or {}).get(
                    "stop_reason",
                    "map-to-base transform unavailable",
                )
            )
            stop_details = dict(pose_lookup.details or {})
            if not stop_details.get("pose_lookup_trace_recorded"):
                trace_failure = self._append_controller_trace(
                    event="pose_lookup_stop",
                    reason=stop_reason,
                    fail_closed=True,
                    diagnostics=stop_details,
                )
                if trace_failure:
                    stop_details = with_controller_trace_failure(
                        stop_details,
                        trace_failure,
                    )
            return ControlCycleGuardDecision(
                ControlCycleGuardAction.STOP,
                stop_reason=stop_reason,
                stop_details=stop_details,
            )

        return ControlCycleGuardDecision(
            ControlCycleGuardAction.PROCEED,
            pose=pose,
        )
