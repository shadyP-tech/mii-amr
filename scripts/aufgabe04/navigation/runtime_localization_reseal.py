"""Pure admission policy for restarting after a localization authority stop.

The waypoint follower owns the motion-side fail-closed action: it publishes
repeated zero commands and terminates the current odom execution certificate
when the live ``map <- odom`` correction leaves its reserved allowance.  This
module only classifies that persisted stop evidence and applies a bounded
retry budget.  It never plans a route, reads ROS, authorizes motion, or treats
the expired certificate as reusable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


RUNTIME_LOCALIZATION_RESEAL_SCHEMA_VERSION = 1
LOCALIZATION_RESEAL_FAULT_CODE = "localization_reseal_required"
GLOBAL_CONSISTENCY_MONITOR_SOURCE = "global_consistency_monitor"
FORCE_ZERO_RESEAL_ACTION = "FORCE_ZERO_RESEAL"
FORCE_ZERO_RESEAL_DECISION = "force_zero_reseal"


@dataclass(frozen=True)
class RuntimeLocalizationResealDecision:
    """Fail-closed classification of one terminal follower outcome."""

    eligible: bool
    reason: str
    execution_phase: str
    motion_published: bool | None
    continuity_reason: str

    @property
    def requires_fresh_localization(self) -> bool:
        return self.eligible

    @property
    def requires_new_route_certificate(self) -> bool:
        return self.eligible

    @property
    def requires_fresh_typed_run(self) -> bool:
        return self.eligible

    @property
    def automatic_motion_authorized(self) -> bool:
        return False

    def to_evidence(self) -> dict[str, Any]:
        return {
            "schema_version": RUNTIME_LOCALIZATION_RESEAL_SCHEMA_VERSION,
            "eligible": self.eligible,
            "reason": self.reason,
            "execution_phase": self.execution_phase,
            "motion_published": self.motion_published,
            "continuity_reason": self.continuity_reason,
            "requires_fresh_localization": self.requires_fresh_localization,
            "requires_new_route_certificate": (
                self.requires_new_route_certificate
            ),
            "requires_fresh_typed_run": self.requires_fresh_typed_run,
            "automatic_motion_authorized": self.automatic_motion_authorized,
        }


@dataclass(frozen=True)
class RuntimeLocalizationResealBudgetDecision:
    """Bound one localization reseal sequence for a coverage leg."""

    allowed: bool
    reason: str
    completed_reseal_count: int
    maximum_reseal_count: int
    next_reseal_index: int | None

    @property
    def automatic_motion_authorized(self) -> bool:
        return False

    def to_evidence(self) -> dict[str, Any]:
        return {
            "schema_version": RUNTIME_LOCALIZATION_RESEAL_SCHEMA_VERSION,
            "allowed": self.allowed,
            "reason": self.reason,
            "completed_reseal_count": self.completed_reseal_count,
            "maximum_reseal_count": self.maximum_reseal_count,
            "next_reseal_index": self.next_reseal_index,
            "automatic_motion_authorized": self.automatic_motion_authorized,
        }


def evaluate_runtime_localization_reseal(
    *,
    status: object,
    motion_published: object,
    stop_details: object,
) -> RuntimeLocalizationResealDecision:
    """Accept only the follower's complete zero-and-reseal evidence contract.

    Malformed or partial runtime evidence is an ordinary ineligible decision,
    not an exception.  The mission orchestrator must then retain its existing
    terminal failure behavior.
    """

    if status != "stopped":
        return _rejected("outcome_not_stopped", motion_published)
    if not isinstance(motion_published, bool):
        return _rejected("motion_published_not_boolean", None)
    if not motion_published:
        return _rejected("motion_not_published", motion_published)
    if not isinstance(stop_details, Mapping):
        return _rejected("stop_details_not_mapping", motion_published)

    required = (
        ("fault_code", LOCALIZATION_RESEAL_FAULT_CODE),
        ("source", GLOBAL_CONSISTENCY_MONITOR_SOURCE),
        ("execution_pose_owner", "odom"),
        ("global_consistency_monitor", "amcl"),
        ("monitor_action", FORCE_ZERO_RESEAL_ACTION),
        ("fail_closed", True),
    )
    for field, expected in required:
        if stop_details.get(field) != expected:
            return _rejected(
                f"invalid_{field}",
                motion_published,
            )

    continuity = stop_details.get("continuity")
    if not isinstance(continuity, Mapping):
        return _rejected("continuity_not_mapping", motion_published)
    continuity_required = (
        ("accepted", False),
        ("requires_zero_cycle", True),
        ("requires_reseal", True),
        ("decision", FORCE_ZERO_RESEAL_DECISION),
        ("fail_closed", True),
    )
    for field, expected in continuity_required:
        value = continuity.get(field)
        if isinstance(expected, bool):
            matches = value is expected
        else:
            matches = value == expected
        if not matches:
            return _rejected(
                f"invalid_continuity_{field}",
                motion_published,
            )

    continuity_reason = continuity.get("reason", "")
    if not isinstance(continuity_reason, str) or not continuity_reason.strip():
        return _rejected("invalid_continuity_reason", motion_published)
    return RuntimeLocalizationResealDecision(
        eligible=True,
        reason="runtime_localization_reseal_required",
        execution_phase="after_motion",
        motion_published=motion_published,
        continuity_reason=continuity_reason.strip(),
    )


def evaluate_runtime_localization_reseal_budget(
    *,
    completed_reseal_count: int,
    maximum_reseal_count: int,
) -> RuntimeLocalizationResealBudgetDecision:
    """Return the next one-based reseal index when budget remains."""

    completed = _nonnegative_int(
        completed_reseal_count,
        "completed_reseal_count",
    )
    maximum = _nonnegative_int(
        maximum_reseal_count,
        "maximum_reseal_count",
    )
    if completed >= maximum:
        return RuntimeLocalizationResealBudgetDecision(
            allowed=False,
            reason="runtime_localization_reseal_budget_exhausted",
            completed_reseal_count=completed,
            maximum_reseal_count=maximum,
            next_reseal_index=None,
        )
    return RuntimeLocalizationResealBudgetDecision(
        allowed=True,
        reason="runtime_localization_reseal_budget_available",
        completed_reseal_count=completed,
        maximum_reseal_count=maximum,
        next_reseal_index=completed + 1,
    )


def _rejected(
    reason: str,
    motion_published: object,
) -> RuntimeLocalizationResealDecision:
    return RuntimeLocalizationResealDecision(
        eligible=False,
        reason=reason,
        execution_phase="not_admitted",
        motion_published=(
            motion_published if isinstance(motion_published, bool) else None
        ),
        continuity_reason="",
    )


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value
