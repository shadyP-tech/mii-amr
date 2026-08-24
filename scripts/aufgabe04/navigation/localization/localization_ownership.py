"""Pure localization ownership and monitor decisions for Aufgabe 04."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence


LOCALIZATION_SOURCE_AMCL = "amcl"
LOCALIZATION_SOURCE_TF = "tf"
EXECUTION_POSE_OWNER_AMCL = LOCALIZATION_SOURCE_AMCL
EXECUTION_POSE_OWNER_TF = LOCALIZATION_SOURCE_TF
EXECUTION_POSE_OWNER_ODOM = "odom"
GLOBAL_CONSISTENCY_MONITOR_NONE = "none"
GLOBAL_CONSISTENCY_MONITOR_AMCL = "amcl"

MONITOR_ACTION_PASS = "PASS"
MONITOR_ACTION_LOG = "LOG"
MONITOR_ACTION_FORCE_ZERO_RESEAL = "FORCE_ZERO_RESEAL"
MONITOR_REASON_UNCERTAINTY_BUDGET_EXHAUSTED = "uncertainty_budget_exhausted"
MONITOR_REASON_RESEAL_REQUIRED = "reseal_required"

FAIL_UNSUPPORTED_SOURCE = "localization ownership: unsupported localization source"
FAIL_AMCL_STALE = "localization ownership: amcl data missing or stale"
FAIL_MAP_TO_ODOM = "localization ownership: dynamic map->odom unavailable"
FAIL_ROUTE_TRANSFORM = "localization ownership: route transform unavailable"
FAIL_AMCL_WITH_EXTERNAL_TF = "localization ownership: amcl conflicts with external tf owner"
FAIL_TF_WITH_AMCL = "localization ownership: tf conflicts with fresh amcl"
FAIL_AMBIGUOUS = "localization ownership: ambiguous owner evidence"
FAIL_UNSUPPORTED_EXECUTION_POSE_OWNER = (
    "localization ownership: unsupported execution pose owner"
)
FAIL_UNSUPPORTED_GLOBAL_CONSISTENCY_MONITOR = (
    "localization ownership: unsupported global consistency monitor"
)
FAIL_ODOM_TO_BASE = "localization ownership: odom->base data missing or stale"
FAIL_ODOM_ROUTE_FRAME = "localization ownership: odom execution requires an odom-frame route"
FAIL_FROZEN_MAP_TRANSFORM_CERTIFICATE = (
    "localization ownership: frozen map transform/certificate evidence missing"
)
FAIL_MONITOR_STALE = (
    "localization ownership: required global consistency monitor missing or stale"
)
FAIL_EXTERNAL_MAP_TO_ODOM_OWNER = (
    "localization ownership: external map->odom owner conflicts with declared roles"
)

EXECUTION_POSE_ACTION = "provide_execution_pose"
MONITOR_ONLY_ACTION = "pass_log_or_force_zero_reseal_only"
NO_MONITOR_ACTION = "none"


@dataclass(frozen=True)
class LocalizationOwnershipEvidence:
    localization_source: str
    amcl_fresh: bool
    map_to_odom_dynamic_fresh: bool
    route_transform_fresh: bool
    odom_to_base_fresh: bool = False
    route_uses_odom_frame: bool = False
    external_tf_owner_candidates: Sequence[str] = field(default_factory=tuple)
    ambiguous_owner_evidence: Sequence[str] = field(default_factory=tuple)
    execution_pose_owner: str = ""
    global_consistency_monitor: str = ""
    frozen_map_transform_certified: bool = False


@dataclass(frozen=True)
class LocalizationOwnershipDecision:
    ok: bool
    failure: str
    data: Dict[str, object]


@dataclass(frozen=True)
class LocalizationMonitorDecision:
    """A non-owning monitor result with no control or route-changing surface."""

    action: str
    reason: str = ""
    diagnostic_warning: str = ""

    def __post_init__(self) -> None:
        if self.action not in (
            MONITOR_ACTION_PASS,
            MONITOR_ACTION_LOG,
            MONITOR_ACTION_FORCE_ZERO_RESEAL,
        ):
            raise ValueError("unsupported localization monitor action")
        if self.action == MONITOR_ACTION_FORCE_ZERO_RESEAL:
            if self.reason not in (
                MONITOR_REASON_UNCERTAINTY_BUDGET_EXHAUSTED,
                MONITOR_REASON_RESEAL_REQUIRED,
            ):
                raise ValueError(
                    "force-zero/reseal monitor decision requires a stable reason"
                )
        elif self.reason:
            raise ValueError(
                "only force-zero/reseal monitor decisions may carry a reason"
            )


def evaluate_global_consistency_monitor(
    *,
    uncertainty_budget_exhausted: bool = False,
    reseal_required: bool = False,
    diagnostic_warning: str = "",
) -> LocalizationMonitorDecision:
    """Choose a bounded observation-only response for a consistency monitor.

    A monitor can pass, emit a diagnostic log, or demand a zero-command stop
    followed by resealing.  It cannot select poses, routes, or control output.
    Explicit reseal evidence takes precedence when both safety flags are set.
    """
    warning = str(diagnostic_warning).strip()
    if reseal_required:
        return LocalizationMonitorDecision(
            action=MONITOR_ACTION_FORCE_ZERO_RESEAL,
            reason=MONITOR_REASON_RESEAL_REQUIRED,
            diagnostic_warning=warning,
        )
    if uncertainty_budget_exhausted:
        return LocalizationMonitorDecision(
            action=MONITOR_ACTION_FORCE_ZERO_RESEAL,
            reason=MONITOR_REASON_UNCERTAINTY_BUDGET_EXHAUSTED,
            diagnostic_warning=warning,
        )
    if warning:
        return LocalizationMonitorDecision(
            action=MONITOR_ACTION_LOG,
            diagnostic_warning=warning,
        )
    return LocalizationMonitorDecision(action=MONITOR_ACTION_PASS)


def evaluate_localization_ownership(
    evidence: LocalizationOwnershipEvidence,
) -> LocalizationOwnershipDecision:
    """Return a stable pass/fail decision from ROS-free primitive evidence."""
    source = evidence.localization_source
    execution_pose_owner = evidence.execution_pose_owner or source
    global_consistency_monitor = (
        evidence.global_consistency_monitor or GLOBAL_CONSISTENCY_MONITOR_NONE
    )
    external_candidates = sorted(set(evidence.external_tf_owner_candidates))
    ambiguous_evidence = sorted(set(evidence.ambiguous_owner_evidence))

    data: Dict[str, object] = {
        "localization_source": source,
        "amcl_fresh": evidence.amcl_fresh,
        "map_to_odom_dynamic_fresh": evidence.map_to_odom_dynamic_fresh,
        "route_transform_fresh": evidence.route_transform_fresh,
        "odom_to_base_fresh": evidence.odom_to_base_fresh,
        "route_uses_odom_frame": evidence.route_uses_odom_frame,
        "execution_pose_owner": execution_pose_owner,
        "execution_pose_owner_action": EXECUTION_POSE_ACTION,
        "global_consistency_monitor": global_consistency_monitor,
        "global_consistency_monitor_action": (
            MONITOR_ONLY_ACTION
            if global_consistency_monitor != GLOBAL_CONSISTENCY_MONITOR_NONE
            else NO_MONITOR_ACTION
        ),
        "global_consistency_monitor_allowed_actions": (
            [
                MONITOR_ACTION_PASS,
                MONITOR_ACTION_LOG,
                MONITOR_ACTION_FORCE_ZERO_RESEAL,
            ]
            if global_consistency_monitor != GLOBAL_CONSISTENCY_MONITOR_NONE
            else []
        ),
        "frozen_map_transform_certified": evidence.frozen_map_transform_certified,
        "external_tf_owner_candidates": external_candidates,
        "ambiguous_owner_evidence": ambiguous_evidence,
    }

    failure = _localization_ownership_failure(
        source=source,
        execution_pose_owner=execution_pose_owner,
        global_consistency_monitor=global_consistency_monitor,
        amcl_fresh=evidence.amcl_fresh,
        map_to_odom_dynamic_fresh=evidence.map_to_odom_dynamic_fresh,
        route_transform_fresh=evidence.route_transform_fresh,
        odom_to_base_fresh=evidence.odom_to_base_fresh,
        route_uses_odom_frame=evidence.route_uses_odom_frame,
        frozen_map_transform_certified=evidence.frozen_map_transform_certified,
        external_tf_owner_candidates=external_candidates,
        ambiguous_owner_evidence=ambiguous_evidence,
    )
    return LocalizationOwnershipDecision(ok=not failure, failure=failure, data=data)


def _localization_ownership_failure(
    *,
    source: str,
    execution_pose_owner: str,
    global_consistency_monitor: str,
    amcl_fresh: bool,
    map_to_odom_dynamic_fresh: bool,
    route_transform_fresh: bool,
    odom_to_base_fresh: bool,
    route_uses_odom_frame: bool,
    frozen_map_transform_certified: bool,
    external_tf_owner_candidates: Sequence[str],
    ambiguous_owner_evidence: Sequence[str],
) -> str:
    if source not in (
        LOCALIZATION_SOURCE_AMCL,
        LOCALIZATION_SOURCE_TF,
    ):
        return FAIL_UNSUPPORTED_SOURCE
    if execution_pose_owner not in (
        EXECUTION_POSE_OWNER_AMCL,
        EXECUTION_POSE_OWNER_TF,
        EXECUTION_POSE_OWNER_ODOM,
    ):
        return FAIL_UNSUPPORTED_EXECUTION_POSE_OWNER
    if global_consistency_monitor not in (
        GLOBAL_CONSISTENCY_MONITOR_NONE,
        GLOBAL_CONSISTENCY_MONITOR_AMCL,
    ):
        return FAIL_UNSUPPORTED_GLOBAL_CONSISTENCY_MONITOR
    if ambiguous_owner_evidence:
        return FAIL_AMBIGUOUS
    if not route_transform_fresh:
        return FAIL_ROUTE_TRANSFORM
    if execution_pose_owner == EXECUTION_POSE_OWNER_ODOM:
        if not route_uses_odom_frame:
            return FAIL_ODOM_ROUTE_FRAME
        if not odom_to_base_fresh:
            return FAIL_ODOM_TO_BASE
        if not frozen_map_transform_certified:
            return FAIL_FROZEN_MAP_TRANSFORM_CERTIFICATE
        if external_tf_owner_candidates:
            return FAIL_EXTERNAL_MAP_TO_ODOM_OWNER
    elif not route_uses_odom_frame and not map_to_odom_dynamic_fresh:
        return FAIL_MAP_TO_ODOM
    if (
        global_consistency_monitor == GLOBAL_CONSISTENCY_MONITOR_AMCL
        and not amcl_fresh
    ):
        return FAIL_MONITOR_STALE
    if execution_pose_owner == EXECUTION_POSE_OWNER_AMCL:
        if not amcl_fresh:
            return FAIL_AMCL_STALE
        if external_tf_owner_candidates:
            return FAIL_AMCL_WITH_EXTERNAL_TF
        return ""
    if (
        global_consistency_monitor == GLOBAL_CONSISTENCY_MONITOR_AMCL
        and external_tf_owner_candidates
    ):
        return FAIL_EXTERNAL_MAP_TO_ODOM_OWNER
    if amcl_fresh and global_consistency_monitor != GLOBAL_CONSISTENCY_MONITOR_AMCL:
        return FAIL_TF_WITH_AMCL
    return ""
