"""Pure bounded deferral for candidate routes rejected before motion.

Only an exact route-uncertainty rejection that proves no motion and no motion
permit may enter this ledger.  The ledger does not authorize motion, plan a
route, read ROS, or write artifacts.  It only prevents one rejected candidate
from blocking other unresolved candidates in the same camera pass.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from pathlib import Path

from scripts.aufgabe04.real_robot.candidate.no_motion_route_rejection import (
    NoMotionRouteUncertaintyRejection,
    classify_no_motion_route_uncertainty_rejection,
)
from scripts.aufgabe04.real_robot.candidate.recovery_failure import (
    CandidateStartupRecoveryError,
    RejectedChildFailure,
)


def _candidate_uid(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("candidate_uid must be a non-empty string")
    return value.strip()


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _json_ready(value: object, *, field: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must not contain non-finite numbers")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key in sorted(value, key=lambda item: str(item)):
            if not isinstance(key, str) or not key:
                raise ValueError(f"{field} mapping keys must be non-empty strings")
            normalized[key] = _json_ready(value[key], field=f"{field}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _json_ready(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{field} contains unsupported evidence type {type(value).__name__}"
    )


def _evidence_mapping(value: Mapping[str, object], *, field: str) -> dict[str, object]:
    normalized = _json_ready(value, field=field)
    if not isinstance(normalized, dict):
        raise TypeError(f"{field} must be a mapping")
    return normalized


@dataclass(frozen=True)
class CandidateRouteAdmissionDeferralDecision:
    """Fail-closed classification for one candidate child rejection."""

    eligible: bool
    reason: str
    rejected_run_id: str = ""
    remaining_margin_m: float | None = None
    limiting_segment_id: str = ""

    def to_event_fields(self) -> dict[str, object]:
        return {
            "route_admission_deferral_eligible": self.eligible,
            "route_admission_deferral_reason": self.reason,
            "rejected_run_id": self.rejected_run_id,
            "route_uncertainty_remaining_margin_m": self.remaining_margin_m,
            "route_uncertainty_limiting_segment_id": self.limiting_segment_id,
            "motion_published": False,
            "motion_permit_issued": False,
            "motion_authorized": False,
            "motion_continues_authorized": False,
            "route_limits_unchanged": True,
        }


@dataclass(frozen=True)
class CandidateRouteAdmissionAttemptEvidence:
    """One deferred no-motion route-admission attempt."""

    candidate_uid: str
    pass_index: int
    attempt_number: int
    rejected_run_id: str
    reason: str
    remaining_margin_m: float
    limiting_segment_id: str
    stop_reason: str
    stop_details: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_uid": self.candidate_uid,
            "route_admission_pass_index": self.pass_index,
            "candidate_route_admission_attempt_number": self.attempt_number,
            "rejected_run_id": self.rejected_run_id,
            "reason": self.reason,
            "route_uncertainty_remaining_margin_m": self.remaining_margin_m,
            "route_uncertainty_limiting_segment_id": self.limiting_segment_id,
            "stop_reason": self.stop_reason,
            "stop_details": _evidence_mapping(
                self.stop_details,
                field="stop_details",
            ),
            "motion_published": False,
            "motion_authorized": False,
            "motion_continues_authorized": False,
            "route_limits_unchanged": True,
        }


@dataclass(frozen=True)
class CandidateRouteAdmissionSelectionState:
    """Current route-admission retry state for a filtered candidate set."""

    pass_index: int
    max_attempts_per_candidate: int
    eligible_candidate_uids: tuple[str, ...]
    excluded_candidate_uids: tuple[str, ...]
    exhausted_candidate_uids: tuple[str, ...]
    attempt_count_by_candidate: Mapping[str, int]
    terminal_incomplete: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "route_admission_pass_index": self.pass_index,
            "max_candidate_route_admission_attempts": (
                self.max_attempts_per_candidate
            ),
            "route_admission_eligible_candidate_uids": list(
                self.eligible_candidate_uids
            ),
            "route_admission_excluded_candidate_uids": list(
                self.excluded_candidate_uids
            ),
            "route_admission_exhausted_candidate_uids": list(
                self.exhausted_candidate_uids
            ),
            "candidate_route_admission_attempt_count_by_uid": {
                uid: int(self.attempt_count_by_candidate[uid])
                for uid in sorted(self.attempt_count_by_candidate)
            },
            "terminal_route_admission_incomplete": self.terminal_incomplete,
            "motion_authorized": False,
            "motion_continues_authorized": False,
            "route_limits_unchanged": True,
        }


class CandidateRouteAdmissionIncompleteError(RuntimeError):
    """Fail closed after all candidate route-admission attempts are exhausted."""

    def __init__(
        self,
        *,
        unresolved_candidate_uids: Iterable[str],
        attempt_evidence: Iterable[CandidateRouteAdmissionAttemptEvidence],
        max_attempts_per_candidate: int,
        final_pass_index: int,
    ) -> None:
        self.unresolved_candidate_uids = tuple(
            sorted(_candidate_uid(uid) for uid in unresolved_candidate_uids)
        )
        if not self.unresolved_candidate_uids:
            raise ValueError(
                "route admission incomplete error requires unresolved candidates"
            )
        self.attempt_evidence = tuple(attempt_evidence)
        for attempt in self.attempt_evidence:
            if not isinstance(attempt, CandidateRouteAdmissionAttemptEvidence):
                raise TypeError(
                    "attempt_evidence must contain "
                    "CandidateRouteAdmissionAttemptEvidence values"
                )
        self.max_attempts_per_candidate = _positive_int(
            max_attempts_per_candidate,
            field="max_attempts_per_candidate",
        )
        if type(final_pass_index) is not int or final_pass_index < 0:
            raise ValueError("final_pass_index must be a non-negative integer")
        self.final_pass_index = final_pass_index
        unresolved_text = ", ".join(self.unresolved_candidate_uids)
        super().__init__(
            "candidate route admission incomplete after bounded no-motion "
            f"route-uncertainty attempts; unresolved candidates: {unresolved_text}"
        )

    def to_failure_fields(self) -> dict[str, object]:
        return {
            "failure_phase": "candidate_route_admission_incomplete",
            "unresolved_candidate_uids": list(self.unresolved_candidate_uids),
            "max_candidate_route_admission_attempts": (
                self.max_attempts_per_candidate
            ),
            "final_route_admission_pass_index": self.final_pass_index,
            "candidate_route_admission_attempts": [
                attempt.to_dict() for attempt in self.attempt_evidence
            ],
            "motion_continues_authorized": False,
            "route_limits_unchanged": True,
            "fail_closed": True,
        }


class CandidateRouteAdmissionDeferralLedger:
    """Bounded retry state for no-motion route-uncertainty rejections."""

    def __init__(
        self,
        candidate_uids: Iterable[str],
        *,
        max_attempts_per_candidate: int = 2,
    ) -> None:
        normalized = tuple(_candidate_uid(uid) for uid in candidate_uids)
        if len(set(normalized)) != len(normalized):
            raise ValueError("candidate_uids must be unique")
        self.max_attempts_per_candidate = _positive_int(
            max_attempts_per_candidate,
            field="max_attempts_per_candidate",
        )
        self._candidate_uids = tuple(sorted(normalized))
        self._excluded: set[str] = set()
        self._attempt_counts = {uid: 0 for uid in self._candidate_uids}
        self._attempts: list[CandidateRouteAdmissionAttemptEvidence] = []
        self._pass_index = 0

    @property
    def attempts(self) -> tuple[CandidateRouteAdmissionAttemptEvidence, ...]:
        return tuple(self._attempts)

    def selection_state(
        self,
        candidate_uids: Iterable[str],
    ) -> CandidateRouteAdmissionSelectionState:
        filtered = tuple(sorted(_candidate_uid(uid) for uid in candidate_uids))
        unknown = sorted(set(filtered).difference(self._attempt_counts))
        if unknown:
            raise ValueError(
                "route admission state received unknown candidates: "
                + ", ".join(unknown)
            )
        eligible = tuple(
            uid
            for uid in filtered
            if uid not in self._excluded
            and self._attempt_counts[uid] < self.max_attempts_per_candidate
        )
        exhausted = tuple(
            uid
            for uid in filtered
            if self._attempt_counts[uid] >= self.max_attempts_per_candidate
        )
        terminal = bool(filtered) and not eligible and len(exhausted) == len(filtered)
        return CandidateRouteAdmissionSelectionState(
            pass_index=self._pass_index,
            max_attempts_per_candidate=self.max_attempts_per_candidate,
            eligible_candidate_uids=eligible,
            excluded_candidate_uids=tuple(sorted(self._excluded.intersection(filtered))),
            exhausted_candidate_uids=exhausted,
            attempt_count_by_candidate=dict(self._attempt_counts),
            terminal_incomplete=terminal,
        )

    def mark_rejected(
        self,
        *,
        candidate_uid: str,
        rejected_child: RejectedChildFailure,
        decision: CandidateRouteAdmissionDeferralDecision,
    ) -> CandidateRouteAdmissionAttemptEvidence:
        uid = _candidate_uid(candidate_uid)
        if uid not in self._attempt_counts:
            raise ValueError(f"unknown candidate_uid {uid}")
        if uid in self._excluded:
            raise RuntimeError(
                f"candidate {uid} is already excluded in route admission pass "
                f"{self._pass_index}"
            )
        if self._attempt_counts[uid] >= self.max_attempts_per_candidate:
            raise RuntimeError(f"candidate {uid} exhausted route admission attempts")
        if not isinstance(rejected_child, RejectedChildFailure):
            raise TypeError("rejected_child must be RejectedChildFailure")
        if not decision.eligible or decision.remaining_margin_m is None:
            raise ValueError("route admission decision is not deferrable")
        self._attempt_counts[uid] += 1
        evidence = CandidateRouteAdmissionAttemptEvidence(
            candidate_uid=uid,
            pass_index=self._pass_index,
            attempt_number=self._attempt_counts[uid],
            rejected_run_id=rejected_child.run_id,
            reason=decision.reason,
            remaining_margin_m=decision.remaining_margin_m,
            limiting_segment_id=decision.limiting_segment_id,
            stop_reason=rejected_child.stop_reason,
            stop_details=rejected_child.stop_details,
        )
        self._attempts.append(evidence)
        self._excluded.add(uid)
        return evidence

    def advance_pass(self, candidate_uids: Iterable[str]) -> bool:
        state = self.selection_state(candidate_uids)
        if state.eligible_candidate_uids:
            raise RuntimeError(
                "cannot advance route-admission pass while candidates remain eligible"
            )
        if state.terminal_incomplete:
            return False
        if not state.excluded_candidate_uids:
            return False
        self._pass_index += 1
        self._excluded.clear()
        return True

    def incomplete_error(
        self,
        candidate_uids: Iterable[str],
    ) -> CandidateRouteAdmissionIncompleteError:
        state = self.selection_state(candidate_uids)
        if state.eligible_candidate_uids:
            raise RuntimeError(
                "cannot finalize while route-admission candidates remain eligible"
            )
        if not state.terminal_incomplete:
            raise RuntimeError("candidate route-admission retry pass remains available")
        return CandidateRouteAdmissionIncompleteError(
            unresolved_candidate_uids=state.exhausted_candidate_uids,
            attempt_evidence=self._attempts,
            max_attempts_per_candidate=self.max_attempts_per_candidate,
            final_pass_index=state.pass_index,
        )


def evaluate_candidate_route_admission_deferral(
    error: BaseException,
    *,
    expected_initial_run_id: str,
) -> CandidateRouteAdmissionDeferralDecision:
    """Allow only exact initial no-motion route-uncertainty rejection."""

    if not isinstance(error, CandidateStartupRecoveryError):
        return CandidateRouteAdmissionDeferralDecision(
            False,
            "error_not_candidate_startup_recovery",
        )
    if error.phase != "outcome_rejection":
        return CandidateRouteAdmissionDeferralDecision(
            False,
            "recovery_phase_not_outcome_rejection",
        )
    rejected = error.rejected_child
    if rejected is None:
        return CandidateRouteAdmissionDeferralDecision(
            False,
            "rejected_child_evidence_missing",
        )
    if rejected.run_id != expected_initial_run_id:
        return CandidateRouteAdmissionDeferralDecision(
            False,
            "rejected_child_not_initial_route_attempt",
            rejected_run_id=rejected.run_id,
        )
    route_rejection: NoMotionRouteUncertaintyRejection = (
        classify_no_motion_route_uncertainty_rejection(
            status=rejected.status,
            stop_reason=rejected.stop_reason,
            stop_details=rejected.stop_details,
            motion_published=rejected.motion_published,
            issued_motion_permit_kinds=rejected.issued_motion_permit_kinds,
            motion_permit_evidence_present=bool(rejected.issued_motion_permits),
        )
    )
    if not route_rejection.eligible:
        return CandidateRouteAdmissionDeferralDecision(
            False,
            route_rejection.reason,
            rejected_run_id=rejected.run_id,
        )
    return CandidateRouteAdmissionDeferralDecision(
        True,
        "next_candidate_route_dry_preflight_allowed",
        rejected_run_id=rejected.run_id,
        remaining_margin_m=route_rejection.remaining_margin_m,
        limiting_segment_id=route_rejection.limiting_segment_id,
    )


__all__ = [
    "CandidateRouteAdmissionAttemptEvidence",
    "CandidateRouteAdmissionDeferralDecision",
    "CandidateRouteAdmissionDeferralLedger",
    "CandidateRouteAdmissionIncompleteError",
    "CandidateRouteAdmissionSelectionState",
    "evaluate_candidate_route_admission_deferral",
]
