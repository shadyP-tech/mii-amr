"""ROS-free blockage-recovery eligibility and control-loop disposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from scripts.aufgabe04.navigation.waypoint_follower.directives import (
    BlockageRecoveryAction,
    StringDirective,
)


class BlockageRecoveryTrigger(StringDirective):
    """Stop conditions that admit different recovery outcomes."""

    OBSTACLE_SAFETY_STOP = "obstacle_safety_stop"
    CLEARANCE_FLOOR = "clearance_floor"
    STUCK_WATCHDOG = "stuck_watchdog"


class RecoveryLoopAction(StringDirective):
    """Effects the control loop may apply after recovery returns."""

    ZERO_HOLD_AND_RETRY = "zero_hold_and_retry"
    HOLD_AND_RETRY = "hold_and_retry"
    STOP = "stop"


@dataclass(frozen=True)
class BlockageRecoveryDisposition:
    """Pure loop action and terminal fallback after a recovery attempt."""

    action: RecoveryLoopAction
    stop_reason: str = ""


def front_sector_recovery_evidence(
    stop_details: Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    """Extract only validated front-sector evidence from stop diagnostics."""

    evidence = (stop_details or {}).get("front_clearance")
    if not isinstance(evidence, Mapping):
        return None
    if evidence.get("source") != "front_sector":
        return None
    return evidence


def blockage_recovery_eligible(
    *,
    provider_available: bool,
    nominal_linear_x_mps: float,
    front_evidence: Mapping[str, object] | None,
) -> bool:
    """Admit replanning only for forward motion with front-sector evidence."""

    return (
        provider_available
        and nominal_linear_x_mps > 0.0
        and (front_evidence or {}).get("source") == "front_sector"
    )


def blockage_recovery_disposition(
    *,
    trigger: BlockageRecoveryTrigger,
    recovery_action: BlockageRecoveryAction | str,
    fallback_reason: str,
    latest_reason: object,
) -> BlockageRecoveryDisposition:
    """Map recovery outcomes without publishing, sleeping, or replanning."""

    if trigger == BlockageRecoveryTrigger.CLEARANCE_FLOOR:
        if recovery_action in {
            BlockageRecoveryAction.ADOPTED,
            BlockageRecoveryAction.CLEARED,
        }:
            return BlockageRecoveryDisposition(
                RecoveryLoopAction.ZERO_HOLD_AND_RETRY
            )
        if recovery_action == BlockageRecoveryAction.STOPPED:
            return BlockageRecoveryDisposition(
                RecoveryLoopAction.STOP,
                str(latest_reason),
            )
        return BlockageRecoveryDisposition(
            RecoveryLoopAction.STOP,
            fallback_reason,
        )

    if trigger == BlockageRecoveryTrigger.OBSTACLE_SAFETY_STOP:
        if recovery_action == BlockageRecoveryAction.ADOPTED:
            return BlockageRecoveryDisposition(
                RecoveryLoopAction.HOLD_AND_RETRY
            )
        if recovery_action == BlockageRecoveryAction.CLEARED:
            return BlockageRecoveryDisposition(
                RecoveryLoopAction.ZERO_HOLD_AND_RETRY
            )
        if recovery_action == BlockageRecoveryAction.STOPPED:
            return BlockageRecoveryDisposition(
                RecoveryLoopAction.STOP,
                str(latest_reason),
            )
        return BlockageRecoveryDisposition(
            RecoveryLoopAction.STOP,
            fallback_reason,
        )

    if trigger != BlockageRecoveryTrigger.STUCK_WATCHDOG:
        return BlockageRecoveryDisposition(
            RecoveryLoopAction.STOP,
            fallback_reason,
        )
    if recovery_action == BlockageRecoveryAction.ADOPTED:
        return BlockageRecoveryDisposition(RecoveryLoopAction.HOLD_AND_RETRY)
    if recovery_action in {
        BlockageRecoveryAction.CLEARED,
        BlockageRecoveryAction.STOPPED,
    }:
        return BlockageRecoveryDisposition(
            RecoveryLoopAction.STOP,
            str(latest_reason),
        )
    return BlockageRecoveryDisposition(
        RecoveryLoopAction.STOP,
        fallback_reason,
    )
