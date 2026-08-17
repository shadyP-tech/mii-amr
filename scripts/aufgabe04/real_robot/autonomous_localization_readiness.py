"""Pure policy for bounded, no-motion AMCL readiness retries.

The policy deliberately recognizes only the route-specific uncertainty-budget
failure emitted by the certified odom-execution admission.  It never widens a
safety limit and never authorizes motion; it only permits a fresh dry preflight
for the same coverage target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


ODOM_EXECUTION_ADMISSION_FAILED = "odom_execution_admission_failed"
ROUTE_UNCERTAINTY_EXHAUSTED = "route uncertainty budget exhausted:"


@dataclass(frozen=True)
class LocalizationReadinessDecision:
    retryable: bool
    reason: str


def evaluate_localization_readiness_retry(
    *,
    status: str,
    stop_reason: str,
    stop_details: Mapping[str, object],
    motion_published: bool,
) -> LocalizationReadinessDecision:
    """Classify one failed dry admission without weakening fail-closed gates."""

    if motion_published:
        return LocalizationReadinessDecision(False, "motion_was_published")
    if status != "preflight_failed":
        return LocalizationReadinessDecision(False, "status_not_preflight_failed")
    if stop_details.get("fault_code") != ODOM_EXECUTION_ADMISSION_FAILED:
        return LocalizationReadinessDecision(False, "fault_code_not_retryable")
    if not stop_reason.startswith(
        "odom execution admission failed: " + ROUTE_UNCERTAINTY_EXHAUSTED
    ):
        return LocalizationReadinessDecision(
            False,
            "failure_not_route_uncertainty_exhaustion",
        )
    if stop_details.get("fail_closed") is not True:
        return LocalizationReadinessDecision(False, "failure_not_fail_closed")
    if stop_details.get("execution_pose_owner") != "odom":
        return LocalizationReadinessDecision(False, "execution_owner_not_odom")
    if stop_details.get("global_consistency_monitor") != "amcl":
        return LocalizationReadinessDecision(False, "monitor_not_amcl")
    return LocalizationReadinessDecision(True, "fresh_no_motion_admission_allowed")


def localization_readiness_suffix(retry_index: int) -> str:
    if not isinstance(retry_index, int) or isinstance(retry_index, bool):
        raise ValueError("localization readiness retry index must be an integer")
    if retry_index < 0:
        raise ValueError("localization readiness retry index must be non-negative")
    if retry_index == 0:
        return ""
    return f"_localization_readiness_{retry_index:03d}"


__all__ = [
    "LocalizationReadinessDecision",
    "evaluate_localization_readiness_retry",
    "localization_readiness_suffix",
]
