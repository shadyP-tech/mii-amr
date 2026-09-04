"""Pure policy for bounded opposite-face route standoff fallback.

The camera candidate coordinator may try a smaller physically allowed
standoff only when the *initial dry child* for the previous route proves that
the certified route-uncertainty budget was exhausted before motion and before
any motion permit was issued.  This module owns that exact classification and
the deterministic identity of each replacement route.  It never plans a
route, reads ROS, writes an artifact, issues a permit, or authorizes motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from scripts.aufgabe04.real_robot.candidate.no_motion_route_rejection import (
    classify_no_motion_route_uncertainty_rejection,
)
from scripts.aufgabe04.real_robot.candidate.recovery_failure import (
    CandidateStartupRecoveryError,
)


_STANDOFF_SUFFIX = "_standoff_{index:03d}"


def bounded_approach_offsets(
    requested_m: float,
    minimum_m: float,
    *,
    step_m: float = 0.05,
) -> tuple[float, ...]:
    """Return descending standoffs without crossing the physical minimum."""

    if not all(
        math.isfinite(value) and value > 0.0
        for value in (requested_m, minimum_m, step_m)
    ):
        raise ValueError("approach offsets and step must be finite and positive")
    if requested_m + 1.0e-9 < minimum_m:
        raise ValueError("requested approach offset is below physical minimum")
    values = []
    current = requested_m
    while current > minimum_m + 1.0e-9:
        values.append(round(current, 6))
        current -= step_m
    if not values or abs(values[-1] - minimum_m) > 1.0e-9:
        values.append(round(minimum_m, 6))
    return tuple(values)


@dataclass(frozen=True)
class OppositeFaceRouteAttempt:
    """Deterministic route and child identity for one admitted standoff."""

    attempt_index: int
    approach_offset_m: float
    run_id: str
    source_root: Path
    artifact_suffix: str

    def to_event_fields(self) -> dict[str, object]:
        return {
            "opposite_face_standoff_attempt_index": self.attempt_index,
            "approach_offset_m": self.approach_offset_m,
            "run_id": self.run_id,
            "route_source_root": str(self.source_root),
            "route_limits_unchanged": True,
            "motion_authorized": False,
        }


def opposite_face_route_attempt(
    *,
    base_run_id: str,
    base_source_root: Path,
    attempt_index: int,
    approach_offset_m: float,
) -> OppositeFaceRouteAttempt:
    """Bind one route attempt to fresh, non-reusable artifact names."""

    if not isinstance(base_run_id, str) or not base_run_id.strip():
        raise ValueError("opposite-face base run ID must be non-empty")
    if type(attempt_index) is not int or attempt_index < 0:
        raise ValueError("opposite-face attempt index must be non-negative")
    if (
        isinstance(approach_offset_m, bool)
        or not isinstance(approach_offset_m, (int, float))
        or not math.isfinite(float(approach_offset_m))
        or float(approach_offset_m) <= 0.0
    ):
        raise ValueError("opposite-face approach offset must be positive")
    suffix = (
        ""
        if attempt_index == 0
        else _STANDOFF_SUFFIX.format(index=attempt_index)
    )
    source_root = Path(base_source_root)
    if suffix:
        source_root = source_root.with_name(source_root.name + suffix)
    return OppositeFaceRouteAttempt(
        attempt_index=attempt_index,
        approach_offset_m=float(approach_offset_m),
        run_id=base_run_id + suffix,
        source_root=source_root,
        artifact_suffix=suffix,
    )


@dataclass(frozen=True)
class OppositeFaceRouteFallbackDecision:
    """Fail-closed classification of one completed route attempt."""

    eligible: bool
    reason: str
    rejected_run_id: str = ""
    remaining_margin_m: float | None = None
    limiting_segment_id: str = ""

    def to_event_fields(self) -> dict[str, object]:
        return {
            "fallback_eligible": self.eligible,
            "fallback_decision_reason": self.reason,
            "rejected_run_id": self.rejected_run_id,
            "route_uncertainty_remaining_margin_m": self.remaining_margin_m,
            "route_uncertainty_limiting_segment_id": self.limiting_segment_id,
            "motion_published": False,
            "motion_permit_issued": False,
            "motion_continues_authorized": False,
            "route_limits_unchanged": True,
        }


def _rejected(
    reason: str,
    *,
    rejected_run_id: str = "",
) -> OppositeFaceRouteFallbackDecision:
    return OppositeFaceRouteFallbackDecision(
        eligible=False,
        reason=reason,
        rejected_run_id=rejected_run_id,
    )


def evaluate_opposite_face_route_fallback(
    error: BaseException,
    *,
    expected_initial_run_id: str,
) -> OppositeFaceRouteFallbackDecision:
    """Allow only an exact no-motion, no-permit uncertainty rejection.

    Requiring the rejected run to equal ``expected_initial_run_id`` excludes
    failures after a startup reseal.  Such a path may already have consumed a
    permit earlier in the routine and therefore remains terminal here.
    """

    if not isinstance(error, CandidateStartupRecoveryError):
        return _rejected("error_not_candidate_startup_recovery")
    if error.phase != "outcome_rejection":
        return _rejected("recovery_phase_not_outcome_rejection")
    rejected = error.rejected_child
    if rejected is None:
        return _rejected("rejected_child_evidence_missing")
    if rejected.run_id != expected_initial_run_id:
        return _rejected(
            "rejected_child_not_initial_route_attempt",
            rejected_run_id=rejected.run_id,
        )
    route_rejection = classify_no_motion_route_uncertainty_rejection(
        status=rejected.status,
        stop_reason=rejected.stop_reason,
        stop_details=rejected.stop_details,
        motion_published=rejected.motion_published,
        issued_motion_permit_kinds=rejected.issued_motion_permit_kinds,
        motion_permit_evidence_present=bool(rejected.issued_motion_permits),
    )
    if not route_rejection.eligible:
        return _rejected(
            route_rejection.reason,
            rejected_run_id=rejected.run_id,
        )
    return OppositeFaceRouteFallbackDecision(
        eligible=True,
        reason="new_standoff_route_dry_preflight_allowed",
        rejected_run_id=rejected.run_id,
        remaining_margin_m=route_rejection.remaining_margin_m,
        limiting_segment_id=route_rejection.limiting_segment_id,
    )


__all__ = [
    "OppositeFaceRouteAttempt",
    "OppositeFaceRouteFallbackDecision",
    "bounded_approach_offsets",
    "evaluate_opposite_face_route_fallback",
    "opposite_face_route_attempt",
]
