"""Pure, bounded deferral contracts for candidate camera observations.

The autonomous candidate coordinator may continue with other candidates after
one passive observer attempt reports a *typed* target-local availability
failure.  This module owns that bookkeeping only.  It never connects to ROS,
starts a process, reads or writes an artifact, or grants motion authority.

An unavailable observation is excluded for the remainder of its current pass.
Only after every other eligible candidate has been handled may the coordinator
start another pass.  Each candidate has a strict attempt bound, and exhausting
that bound yields :class:`CandidateApproachIncompleteError`; unresolved
candidates can therefore never be reported as a successful camera phase.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal


AttemptOutcome = Literal["resolved", "unavailable"]


def _candidate_uid(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("candidate_uid must be a non-empty string")
    return value.strip()


def _nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _json_ready(value: object, *, field: str) -> object:
    """Return a detached, deterministic JSON-ready evidence value."""

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
            normalized[key] = _json_ready(
                value[key],
                field=f"{field}.{key}",
            )
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _json_ready(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{field} contains unsupported evidence type {type(value).__name__}"
    )


def _evidence_mapping(
    value: Mapping[str, object] | None,
    *,
    field: str,
) -> dict[str, object]:
    normalized = _json_ready({} if value is None else value, field=field)
    if not isinstance(normalized, dict):  # Defensive; input is type-narrowed.
        raise TypeError(f"{field} must be a mapping")
    return normalized


class CandidateObservationUnavailableError(RuntimeError):
    """Typed, target-local observer failure that may enter bounded deferral.

    Callers must use this type only after classifying the observer failure as
    local to this candidate attempt.  Systemic sensor, localization, TF, child
    lifecycle, or artifact-integrity failures remain terminal and must not be
    converted into this exception merely to keep the mission moving.
    """

    def __init__(
        self,
        *,
        candidate_uid: str,
        observation_attempt_index: int,
        reason: str,
        process_evidence: Mapping[str, object],
        status_evidence: Mapping[str, object],
    ) -> None:
        self.candidate_uid = _candidate_uid(candidate_uid)
        self.observation_attempt_index = _nonnegative_int(
            observation_attempt_index,
            field="observation_attempt_index",
        )
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("candidate observation reason must be non-empty")
        self.reason = reason.strip()
        self._process_evidence = _evidence_mapping(
            process_evidence,
            field="process_evidence",
        )
        self._status_evidence = _evidence_mapping(
            status_evidence,
            field="status_evidence",
        )
        super().__init__(
            "candidate observation unavailable for "
            f"{self.candidate_uid} on observer attempt "
            f"{self.observation_attempt_index}: {self.reason}"
        )

    @property
    def process_evidence(self) -> dict[str, object]:
        return _evidence_mapping(
            self._process_evidence,
            field="process_evidence",
        )

    @property
    def status_evidence(self) -> dict[str, object]:
        return _evidence_mapping(
            self._status_evidence,
            field="status_evidence",
        )

    def to_event_fields(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "event": "candidate_observation_unavailable",
            "candidate_uid": self.candidate_uid,
            "observation_attempt_index": self.observation_attempt_index,
            "candidate_observation_reason": self.reason,
            "observer_process_evidence": self.process_evidence,
            "observer_status_evidence": self.status_evidence,
            "motion_capability": "none",
            "motion_authorized": False,
            "motion_continues_authorized": False,
        }

    def to_failure_fields(self) -> dict[str, object]:
        return {
            "failure_phase": "candidate_observation",
            "candidate_uid": self.candidate_uid,
            "observation_attempt_index": self.observation_attempt_index,
            "candidate_observation_reason": self.reason,
            "observer_process_evidence": self.process_evidence,
            "observer_status_evidence": self.status_evidence,
            "motion_capability": "none",
            "motion_continues_authorized": False,
            "fail_closed": True,
        }


@dataclass(frozen=True)
class CandidateObservationSelection:
    """One selected candidate observation slot in a bounded pass."""

    candidate_uid: str
    pass_index: int
    attempt_number: int

    def to_event_fields(self) -> dict[str, object]:
        return {
            "candidate_uid": self.candidate_uid,
            "observation_pass_index": self.pass_index,
            "candidate_observation_attempt_number": self.attempt_number,
            "motion_capability": "none",
            "motion_authorized": False,
        }


@dataclass(frozen=True)
class CandidateObservationAttemptEvidence:
    """Immutable evidence for one completed ledger selection."""

    candidate_uid: str
    pass_index: int
    attempt_number: int
    outcome: AttemptOutcome
    observation_attempt_index: int | None
    reason: str | None
    process_evidence: Mapping[str, object]
    status_evidence: Mapping[str, object]
    result_evidence: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_uid": self.candidate_uid,
            "observation_pass_index": self.pass_index,
            "candidate_observation_attempt_number": self.attempt_number,
            "outcome": self.outcome,
            "observation_attempt_index": self.observation_attempt_index,
            "reason": self.reason,
            "observer_process_evidence": _evidence_mapping(
                self.process_evidence,
                field="process_evidence",
            ),
            "observer_status_evidence": _evidence_mapping(
                self.status_evidence,
                field="status_evidence",
            ),
            "result_evidence": _evidence_mapping(
                self.result_evidence,
                field="result_evidence",
            ),
            "motion_capability": "none",
            "motion_authorized": False,
        }


@dataclass(frozen=True)
class CandidateObservationSelectionState:
    """Deterministic snapshot used by the parent candidate selector."""

    pass_index: int
    max_attempts_per_candidate: int
    eligible_candidate_uids: tuple[str, ...]
    excluded_candidate_uids: tuple[str, ...]
    resolved_candidate_uids: tuple[str, ...]
    unresolved_candidate_uids: tuple[str, ...]
    selected: CandidateObservationSelection | None
    attempt_count_by_candidate: Mapping[str, int]
    complete: bool
    terminal_incomplete: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_pass_index": self.pass_index,
            "max_candidate_observation_attempts": (
                self.max_attempts_per_candidate
            ),
            "eligible_candidate_uids": list(self.eligible_candidate_uids),
            "excluded_candidate_uids": list(self.excluded_candidate_uids),
            "resolved_candidate_uids": list(self.resolved_candidate_uids),
            "unresolved_candidate_uids": list(self.unresolved_candidate_uids),
            "selected_candidate_observation": (
                None if self.selected is None else self.selected.to_event_fields()
            ),
            "candidate_observation_attempt_count_by_uid": {
                uid: int(self.attempt_count_by_candidate[uid])
                for uid in sorted(self.attempt_count_by_candidate)
            },
            "complete": self.complete,
            "terminal_incomplete": self.terminal_incomplete,
            "motion_capability": "none",
            "motion_authorized": False,
        }


class CandidateApproachIncompleteError(RuntimeError):
    """Fail-closed terminal result after bounded deferrals are exhausted."""

    def __init__(
        self,
        *,
        resolved_candidate_uids: Iterable[str],
        unresolved_candidate_uids: Iterable[str],
        attempt_evidence: Iterable[CandidateObservationAttemptEvidence],
        max_attempts_per_candidate: int,
        final_pass_index: int,
    ) -> None:
        self.resolved_candidate_uids = tuple(
            sorted(_candidate_uid(uid) for uid in resolved_candidate_uids)
        )
        self.unresolved_candidate_uids = tuple(
            sorted(_candidate_uid(uid) for uid in unresolved_candidate_uids)
        )
        if set(self.resolved_candidate_uids) & set(
            self.unresolved_candidate_uids
        ):
            raise ValueError("resolved and unresolved candidate sets overlap")
        if not self.unresolved_candidate_uids:
            raise ValueError(
                "candidate approach incomplete error requires unresolved candidates"
            )
        self.max_attempts_per_candidate = _positive_int(
            max_attempts_per_candidate,
            field="max_attempts_per_candidate",
        )
        self.final_pass_index = _nonnegative_int(
            final_pass_index,
            field="final_pass_index",
        )
        self.attempt_evidence = tuple(attempt_evidence)
        for attempt in self.attempt_evidence:
            if not isinstance(attempt, CandidateObservationAttemptEvidence):
                raise TypeError(
                    "attempt_evidence must contain "
                    "CandidateObservationAttemptEvidence values"
                )
        unresolved_text = ", ".join(self.unresolved_candidate_uids)
        super().__init__(
            "candidate approach incomplete after bounded camera observation "
            f"attempts; unresolved candidates: {unresolved_text}"
        )

    def to_failure_fields(self) -> dict[str, object]:
        return {
            "failure_phase": "candidate_approach_incomplete",
            "resolved_candidate_uids": list(self.resolved_candidate_uids),
            "unresolved_candidate_uids": list(self.unresolved_candidate_uids),
            "max_candidate_observation_attempts": (
                self.max_attempts_per_candidate
            ),
            "final_observation_pass_index": self.final_pass_index,
            "candidate_observation_attempts": [
                attempt.to_dict() for attempt in self.attempt_evidence
            ],
            "motion_capability": "none",
            "motion_continues_authorized": False,
            "fail_closed": True,
        }


class CandidateObservationDeferralLedger:
    """Deterministic bounded state for candidate-local observation retries."""

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
        self._resolved: set[str] = set()
        self._excluded: set[str] = set()
        self._attempt_counts = {uid: 0 for uid in self._candidate_uids}
        self._attempts: list[CandidateObservationAttemptEvidence] = []
        self._pass_index = 0
        self._selected: CandidateObservationSelection | None = None

    @property
    def attempts(self) -> tuple[CandidateObservationAttemptEvidence, ...]:
        return tuple(self._attempts)

    def _unresolved(self) -> tuple[str, ...]:
        return tuple(
            uid for uid in self._candidate_uids if uid not in self._resolved
        )

    def _eligible(self) -> tuple[str, ...]:
        if self._selected is not None:
            return ()
        return tuple(
            uid
            for uid in self._unresolved()
            if uid not in self._excluded
            and self._attempt_counts[uid] < self.max_attempts_per_candidate
        )

    def selection_state(self) -> CandidateObservationSelectionState:
        unresolved = self._unresolved()
        eligible = self._eligible()
        complete = not unresolved
        terminal_incomplete = (
            not complete
            and self._selected is None
            and not eligible
            and not any(
                self._attempt_counts[uid] < self.max_attempts_per_candidate
                for uid in unresolved
            )
        )
        return CandidateObservationSelectionState(
            pass_index=self._pass_index,
            max_attempts_per_candidate=self.max_attempts_per_candidate,
            eligible_candidate_uids=eligible,
            excluded_candidate_uids=tuple(sorted(self._excluded)),
            resolved_candidate_uids=tuple(sorted(self._resolved)),
            unresolved_candidate_uids=unresolved,
            selected=self._selected,
            attempt_count_by_candidate=dict(self._attempt_counts),
            complete=complete,
            terminal_incomplete=terminal_incomplete,
        )

    def select(self, candidate_uid: str) -> CandidateObservationSelection:
        uid = _candidate_uid(candidate_uid)
        if self._selected is not None:
            raise RuntimeError(
                "candidate observation selection already active for "
                f"{self._selected.candidate_uid}"
            )
        if uid not in self._attempt_counts:
            raise ValueError(f"unknown candidate_uid {uid}")
        if uid not in self._eligible():
            raise RuntimeError(
                f"candidate {uid} is not eligible in observation pass "
                f"{self._pass_index}"
            )
        self._attempt_counts[uid] += 1
        self._selected = CandidateObservationSelection(
            candidate_uid=uid,
            pass_index=self._pass_index,
            attempt_number=self._attempt_counts[uid],
        )
        return self._selected

    def mark_unavailable(
        self,
        error: CandidateObservationUnavailableError,
    ) -> CandidateObservationAttemptEvidence:
        selected = self._require_selection()
        if not isinstance(error, CandidateObservationUnavailableError):
            raise TypeError(
                "only CandidateObservationUnavailableError may be deferred"
            )
        if error.candidate_uid != selected.candidate_uid:
            raise ValueError(
                "unavailable observation candidate does not match active "
                f"selection {selected.candidate_uid}"
            )
        evidence = CandidateObservationAttemptEvidence(
            candidate_uid=selected.candidate_uid,
            pass_index=selected.pass_index,
            attempt_number=selected.attempt_number,
            outcome="unavailable",
            observation_attempt_index=error.observation_attempt_index,
            reason=error.reason,
            process_evidence=error.process_evidence,
            status_evidence=error.status_evidence,
            result_evidence={},
        )
        self._attempts.append(evidence)
        self._excluded.add(selected.candidate_uid)
        self._selected = None
        return evidence

    def mark_resolved(
        self,
        result_evidence: Mapping[str, object] | None = None,
    ) -> CandidateObservationAttemptEvidence:
        selected = self._require_selection()
        evidence = CandidateObservationAttemptEvidence(
            candidate_uid=selected.candidate_uid,
            pass_index=selected.pass_index,
            attempt_number=selected.attempt_number,
            outcome="resolved",
            observation_attempt_index=None,
            reason=None,
            process_evidence={},
            status_evidence={},
            result_evidence=_evidence_mapping(
                result_evidence,
                field="result_evidence",
            ),
        )
        self._attempts.append(evidence)
        self._resolved.add(selected.candidate_uid)
        self._excluded.discard(selected.candidate_uid)
        self._selected = None
        return evidence

    def advance_pass(self) -> bool:
        """Advance only after the current pass has no eligible candidate."""

        if self._selected is not None:
            raise RuntimeError("cannot advance with an active selection")
        if self._eligible():
            raise RuntimeError(
                "cannot advance observation pass while candidates remain eligible"
            )
        unresolved = self._unresolved()
        if not unresolved:
            return False
        if not any(
            self._attempt_counts[uid] < self.max_attempts_per_candidate
            for uid in unresolved
        ):
            return False
        self._pass_index += 1
        self._excluded.clear()
        return True

    def incomplete_error(self) -> CandidateApproachIncompleteError:
        state = self.selection_state()
        if state.complete:
            raise RuntimeError("candidate observation ledger is complete")
        if self._selected is not None:
            raise RuntimeError(
                "cannot finalize candidate observations with an active selection"
            )
        if state.eligible_candidate_uids:
            raise RuntimeError(
                "cannot finalize while candidate observations remain eligible"
            )
        if not state.terminal_incomplete:
            raise RuntimeError(
                "candidate observation retry pass remains available"
            )
        return CandidateApproachIncompleteError(
            resolved_candidate_uids=state.resolved_candidate_uids,
            unresolved_candidate_uids=state.unresolved_candidate_uids,
            attempt_evidence=self._attempts,
            max_attempts_per_candidate=self.max_attempts_per_candidate,
            final_pass_index=state.pass_index,
        )

    def _require_selection(self) -> CandidateObservationSelection:
        if self._selected is None:
            raise RuntimeError("no active candidate observation selection")
        return self._selected


__all__ = [
    "CandidateApproachIncompleteError",
    "CandidateObservationAttemptEvidence",
    "CandidateObservationDeferralLedger",
    "CandidateObservationSelection",
    "CandidateObservationSelectionState",
    "CandidateObservationUnavailableError",
]
