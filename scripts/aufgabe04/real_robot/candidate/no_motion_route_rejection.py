"""Pure classification of fail-closed, no-motion child rejections.

Runtime recovery and opposite-face standoff fallback consume the same child
evidence but have different authority boundaries.  Runtime recovery may
truthfully terminate on any structurally complete preflight rejection.  The
standoff fallback is narrower: it recognizes only the certified odom route-
uncertainty rejection with a finite negative margin.  Neither decision grants
motion authority, plans a route, reads ROS, or writes an artifact.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math

from scripts.aufgabe04.real_robot.readiness.localization import (
    evaluate_localization_readiness_retry,
)


@dataclass(frozen=True)
class NoMotionPreflightClassification:
    """Structural classification of one replacement child outcome."""

    applies: bool
    evidence_valid: bool
    reason: str
    issued_motion_permit_kinds: tuple[str, ...] = ()


@dataclass(frozen=True)
class NoMotionRouteUncertaintyRejection:
    """Exact route-uncertainty subset eligible for a new standoff route."""

    eligible: bool
    reason: str
    remaining_margin_m: float | None = None
    limiting_segment_id: str = ""


def _permit_evidence(
    issued_motion_permit_kinds: Iterable[str],
    motion_permit_evidence_present: bool | None,
) -> tuple[tuple[str, ...], bool]:
    kinds = tuple(issued_motion_permit_kinds)
    evidence_present = bool(kinds)
    if motion_permit_evidence_present is not None:
        evidence_present = evidence_present or bool(
            motion_permit_evidence_present
        )
    return kinds, evidence_present


def classify_no_motion_preflight_failure(
    *,
    status: object,
    stop_reason: object,
    stop_details: object,
    motion_published: object,
    issued_motion_permit_kinds: Iterable[str] = (),
    motion_permit_evidence_present: bool | None = None,
    returncode: object | None = None,
) -> NoMotionPreflightClassification:
    """Recognize a structurally bound preflight failure without authorizing it.

    ``applies=False`` deliberately routes completed or possibly post-motion
    outcomes to their normal permit validator.  ``evidence_valid=True`` means
    only that the no-motion terminal evidence is self-consistent.
    """

    if status != "preflight_failed" or motion_published is not False:
        return NoMotionPreflightClassification(
            applies=False,
            evidence_valid=False,
            reason="runtime_permit_validation_required",
        )

    permit_kinds, permit_evidence_present = _permit_evidence(
        issued_motion_permit_kinds,
        motion_permit_evidence_present,
    )
    if permit_evidence_present:
        return NoMotionPreflightClassification(
            applies=True,
            evidence_valid=False,
            reason="no_motion_preflight_reported_motion_permit",
            issued_motion_permit_kinds=permit_kinds,
        )
    if returncode is not None and type(returncode) is not int:
        return NoMotionPreflightClassification(
            applies=True,
            evidence_valid=False,
            reason="no_motion_preflight_returncode_not_integer",
        )
    if returncode == 0:
        return NoMotionPreflightClassification(
            applies=True,
            evidence_valid=False,
            reason="no_motion_preflight_returncode_was_zero",
        )
    if not isinstance(stop_reason, str) or not stop_reason.strip():
        return NoMotionPreflightClassification(
            applies=True,
            evidence_valid=False,
            reason="no_motion_preflight_stop_reason_missing",
        )
    if not isinstance(stop_details, Mapping):
        return NoMotionPreflightClassification(
            applies=True,
            evidence_valid=False,
            reason="no_motion_preflight_stop_details_not_mapping",
        )
    if stop_details.get("reason") != stop_reason:
        return NoMotionPreflightClassification(
            applies=True,
            evidence_valid=False,
            reason="no_motion_preflight_reason_binding_mismatch",
        )
    if stop_details.get("motion_published") is not False:
        return NoMotionPreflightClassification(
            applies=True,
            evidence_valid=False,
            reason="no_motion_preflight_nested_motion_evidence_not_false",
        )
    if stop_details.get("fail_closed") is not True:
        return NoMotionPreflightClassification(
            applies=True,
            evidence_valid=False,
            reason="no_motion_preflight_not_fail_closed",
        )
    return NoMotionPreflightClassification(
        applies=True,
        evidence_valid=True,
        reason="structured_no_motion_preflight_failure",
    )


def classify_no_motion_route_uncertainty_rejection(
    *,
    status: object,
    stop_reason: object,
    stop_details: object,
    motion_published: object,
    issued_motion_permit_kinds: Iterable[str] = (),
    motion_permit_evidence_present: bool | None = None,
) -> NoMotionRouteUncertaintyRejection:
    """Allow only a certified, no-permit route-uncertainty rejection."""

    if motion_published is not False:
        return NoMotionRouteUncertaintyRejection(
            False,
            "motion_was_published",
        )
    permit_kinds, permit_evidence_present = _permit_evidence(
        issued_motion_permit_kinds,
        motion_permit_evidence_present,
    )
    if permit_evidence_present:
        return NoMotionRouteUncertaintyRejection(
            False,
            "motion_permit_was_issued",
        )
    if not isinstance(stop_reason, str) or not isinstance(stop_details, Mapping):
        return NoMotionRouteUncertaintyRejection(
            False,
            "malformed_no_motion_preflight_evidence",
        )
    readiness = evaluate_localization_readiness_retry(
        status=str(status),
        stop_reason=stop_reason,
        stop_details=stop_details,
        motion_published=False,
    )
    if not readiness.retryable:
        return NoMotionRouteUncertaintyRejection(
            False,
            f"localization_readiness_{readiness.reason}",
        )
    preflight = classify_no_motion_preflight_failure(
        status=status,
        stop_reason=stop_reason,
        stop_details=stop_details,
        motion_published=motion_published,
        issued_motion_permit_kinds=permit_kinds,
        motion_permit_evidence_present=False,
    )
    structural_reasons = {
        "no_motion_preflight_reason_binding_mismatch": (
            "stop_reason_binding_mismatch"
        ),
        "no_motion_preflight_nested_motion_evidence_not_false": (
            "nested_motion_evidence_not_false"
        ),
    }
    if not preflight.evidence_valid:
        return NoMotionRouteUncertaintyRejection(
            False,
            structural_reasons.get(preflight.reason, preflight.reason),
        )
    if stop_details.get("uncertainty_budget_accepted") is not False:
        return NoMotionRouteUncertaintyRejection(
            False,
            "uncertainty_budget_rejection_missing",
        )
    margin = stop_details.get("route_uncertainty_remaining_margin_m")
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(float(margin))
        or float(margin) >= 0.0
    ):
        return NoMotionRouteUncertaintyRejection(
            False,
            "negative_uncertainty_margin_missing",
        )
    limiting_segment = stop_details.get(
        "route_uncertainty_limiting_segment_id"
    )
    if not isinstance(limiting_segment, str) or not limiting_segment.strip():
        return NoMotionRouteUncertaintyRejection(
            False,
            "limiting_segment_missing",
        )
    return NoMotionRouteUncertaintyRejection(
        True,
        "exact_no_motion_route_uncertainty_rejection",
        remaining_margin_m=float(margin),
        limiting_segment_id=limiting_segment.strip(),
    )


__all__ = [
    "NoMotionPreflightClassification",
    "NoMotionRouteUncertaintyRejection",
    "classify_no_motion_preflight_failure",
    "classify_no_motion_route_uncertainty_rejection",
]
