"""ROS-free admission of stand candidates after a coverage survey.

This module evaluates immutable survey snapshots only.  It does not publish
motion, choose a route, or authorize a camera-approach leg.  Malformed plan,
progress, or registry structures raise ``ValueError``; ordinary readiness
failures are returned as deterministic evidence with ``ready=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    STATUS_PENDING_CAMERA,
    STATUS_PROVISIONAL,
    STATUS_REJECTED,
    CoverageSurveyPlan,
    CoverageSurveyProgress,
    StandSurveyRegistry,
    SurveyCandidate,
    coverage_survey_plan_sha256,
    validate_stand_survey_registry,
    validate_survey_progress,
    visited_coverage_ratio,
)


COVERAGE_CANDIDATE_ADMISSION_SCHEMA_VERSION = 1
_COVERAGE_COMPARISON_EPSILON = 1.0e-12


@dataclass(frozen=True)
class CoverageCandidateEvidence:
    """Per-pending-candidate evidence for the post-coverage gate."""

    candidate_uid: str
    status: str
    confidence: float
    minimum_confidence: float
    confidence_met: bool
    hit_count: int
    minimum_hit_count: int
    hit_count_met: bool
    viewpoint_ids: tuple[str, ...]
    known_viewpoint_ids: tuple[str, ...]
    unknown_viewpoint_ids: tuple[str, ...]
    viewpoint_ids_distinct: bool
    distinct_known_viewpoint_count: int
    minimum_distinct_viewpoints: int
    distinct_known_viewpoints_met: bool
    required_exact_viewpoint_ids: tuple[str, ...]
    required_exact_viewpoints_met: bool
    admissible: bool
    reasons: tuple[str, ...]

    def to_evidence_dict(self) -> dict[str, object]:
        return {
            "candidate_uid": self.candidate_uid,
            "status": self.status,
            "admissible": self.admissible,
            "reasons": list(self.reasons),
            "confidence": {
                "value": self.confidence,
                "minimum": self.minimum_confidence,
                "met": self.confidence_met,
            },
            "hits": {
                "count": self.hit_count,
                "minimum": self.minimum_hit_count,
                "met": self.hit_count_met,
            },
            "viewpoints": {
                "ids": list(self.viewpoint_ids),
                "known_ids": list(self.known_viewpoint_ids),
                "unknown_ids": list(self.unknown_viewpoint_ids),
                "ids_distinct": self.viewpoint_ids_distinct,
                "distinct_known_count": self.distinct_known_viewpoint_count,
                "minimum_distinct_count": self.minimum_distinct_viewpoints,
                "minimum_distinct_met": self.distinct_known_viewpoints_met,
                "required_exact_ids": list(self.required_exact_viewpoint_ids),
                "required_exact_ids_met": self.required_exact_viewpoints_met,
            },
        }


@dataclass(frozen=True)
class CoverageCandidateAdmissionDecision:
    """Frozen decision and complete evidence for one survey snapshot."""

    schema_version: int
    survey_id: str
    planning_frame: str
    map_bundle_sha256: str
    plan_sha256: str
    progress_snapshot_sha256: str
    registry_snapshot_sha256: str
    ready: bool
    reasons: tuple[str, ...]
    planned_viewpoint_ids: tuple[str, ...]
    visited_viewpoint_ids: tuple[str, ...]
    unvisited_viewpoint_ids: tuple[str, ...]
    all_planned_viewpoints_visited: bool
    visited_coverage_ratio: float
    coverage_threshold: float
    coverage_threshold_met: bool
    expected_stand_count: int | None
    pending_candidate_uids: tuple[str, ...]
    provisional_candidate_uids: tuple[str, ...]
    other_non_rejected_candidate_uids: tuple[str, ...]
    rejected_candidate_uids: tuple[str, ...]
    pending_candidate_count_met: bool
    candidate_evidence: tuple[CoverageCandidateEvidence, ...]

    @property
    def admitted_candidate_uids(self) -> tuple[str, ...]:
        """Candidates admitted by the complete gate, or none on any failure."""

        if not self.ready:
            return ()
        return tuple(
            evidence.candidate_uid
            for evidence in self.candidate_evidence
            if evidence.admissible
        )

    def to_evidence_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "decision": "ready" if self.ready else "not_ready",
            "ready": self.ready,
            "reasons": list(self.reasons),
            "survey": {
                "survey_id": self.survey_id,
                "planning_frame": self.planning_frame,
                "map_bundle_sha256": self.map_bundle_sha256,
                "plan_sha256": self.plan_sha256,
                "progress_snapshot_sha256": self.progress_snapshot_sha256,
                "registry_snapshot_sha256": self.registry_snapshot_sha256,
            },
            "coverage": {
                "planned_viewpoint_ids": list(self.planned_viewpoint_ids),
                "visited_viewpoint_ids": list(self.visited_viewpoint_ids),
                "unvisited_viewpoint_ids": list(self.unvisited_viewpoint_ids),
                "all_planned_viewpoints_visited": (
                    self.all_planned_viewpoints_visited
                ),
                "visited_coverage_ratio": self.visited_coverage_ratio,
                "coverage_threshold": self.coverage_threshold,
                "coverage_threshold_met": self.coverage_threshold_met,
                "comparison_epsilon": _COVERAGE_COMPARISON_EPSILON,
            },
            "candidate_population": {
                "expected_stand_count": self.expected_stand_count,
                "pending_candidate_uids": list(self.pending_candidate_uids),
                "provisional_candidate_uids": list(
                    self.provisional_candidate_uids
                ),
                "other_non_rejected_candidate_uids": list(
                    self.other_non_rejected_candidate_uids
                ),
                "rejected_candidate_uids": list(self.rejected_candidate_uids),
                "pending_candidate_count_met": (
                    self.pending_candidate_count_met
                ),
                "admitted_candidate_uids": list(
                    self.admitted_candidate_uids
                ),
            },
            "candidate_evidence": [
                evidence.to_evidence_dict()
                for evidence in self.candidate_evidence
            ],
        }


def evaluate_coverage_candidate_admission(
    plan: CoverageSurveyPlan,
    progress: CoverageSurveyProgress,
    registry: StandSurveyRegistry,
) -> CoverageCandidateAdmissionDecision:
    """Evaluate the fail-closed post-coverage candidate gate.

    Structurally valid but incomplete or insufficient evidence always returns
    a decision.  The function raises ``ValueError`` only when one of the three
    supplied snapshots is malformed or their provenance is inconsistent.
    """

    _validate_input_structures(plan, progress, registry)

    planned_viewpoint_ids = plan.viewpoint_ids
    planned_viewpoint_set = set(planned_viewpoint_ids)
    visited_viewpoint_set = set(progress.visited_viewpoint_ids)
    visited_viewpoint_ids = tuple(
        viewpoint_id
        for viewpoint_id in planned_viewpoint_ids
        if viewpoint_id in visited_viewpoint_set
    )
    unvisited_viewpoint_ids = tuple(
        viewpoint_id
        for viewpoint_id in planned_viewpoint_ids
        if viewpoint_id not in visited_viewpoint_set
    )
    all_planned_viewpoints_visited = (
        visited_viewpoint_set == planned_viewpoint_set
    )
    coverage_ratio = visited_coverage_ratio(plan, progress)
    coverage_threshold_met = (
        coverage_ratio + _COVERAGE_COMPARISON_EPSILON
        >= plan.config.coverage_threshold
    )

    pending = tuple(
        candidate
        for candidate in registry.candidates
        if candidate.status == STATUS_PENDING_CAMERA
    )
    provisional = tuple(
        candidate
        for candidate in registry.candidates
        if candidate.status == STATUS_PROVISIONAL
    )
    rejected = tuple(
        candidate
        for candidate in registry.candidates
        if candidate.status == STATUS_REJECTED
    )
    other_non_rejected = tuple(
        candidate
        for candidate in registry.candidates
        if candidate.status
        not in {STATUS_PENDING_CAMERA, STATUS_PROVISIONAL, STATUS_REJECTED}
    )
    expected_stand_count = plan.config.expected_stand_count
    pending_candidate_count_met = (
        expected_stand_count is not None
        and len(pending) == expected_stand_count
    )
    candidate_evidence = tuple(
        _candidate_evidence(plan, candidate) for candidate in pending
    )

    reasons: list[str] = []
    if not all_planned_viewpoints_visited:
        reasons.append("planned_viewpoints_incomplete")
    if not coverage_threshold_met:
        reasons.append("visited_coverage_below_threshold")
    if expected_stand_count is None:
        reasons.append("expected_stand_count_unset")
    elif not pending_candidate_count_met:
        reasons.append("pending_candidate_count_mismatch")
    if provisional:
        reasons.append("provisional_candidates_present")
    if other_non_rejected:
        reasons.append("other_non_rejected_candidates_present")
    if any(not evidence.admissible for evidence in candidate_evidence):
        reasons.append("pending_candidate_requirements_not_met")

    return CoverageCandidateAdmissionDecision(
        schema_version=COVERAGE_CANDIDATE_ADMISSION_SCHEMA_VERSION,
        survey_id=plan.survey_id,
        planning_frame=plan.planning_frame,
        map_bundle_sha256=plan.map_bundle_sha256,
        plan_sha256=coverage_survey_plan_sha256(plan),
        progress_snapshot_sha256=_progress_snapshot_sha256(progress),
        registry_snapshot_sha256=_registry_snapshot_sha256(registry),
        ready=not reasons,
        reasons=tuple(reasons),
        planned_viewpoint_ids=planned_viewpoint_ids,
        visited_viewpoint_ids=visited_viewpoint_ids,
        unvisited_viewpoint_ids=unvisited_viewpoint_ids,
        all_planned_viewpoints_visited=all_planned_viewpoints_visited,
        visited_coverage_ratio=coverage_ratio,
        coverage_threshold=plan.config.coverage_threshold,
        coverage_threshold_met=coverage_threshold_met,
        expected_stand_count=expected_stand_count,
        pending_candidate_uids=tuple(
            candidate.candidate_uid for candidate in pending
        ),
        provisional_candidate_uids=tuple(
            candidate.candidate_uid for candidate in provisional
        ),
        other_non_rejected_candidate_uids=tuple(
            candidate.candidate_uid for candidate in other_non_rejected
        ),
        rejected_candidate_uids=tuple(
            candidate.candidate_uid for candidate in rejected
        ),
        pending_candidate_count_met=pending_candidate_count_met,
        candidate_evidence=candidate_evidence,
    )


def coverage_candidate_admission_evidence(
    decision: CoverageCandidateAdmissionDecision,
) -> dict[str, object]:
    """Return the canonical JSON-safe evidence payload for ``decision``."""

    if not isinstance(decision, CoverageCandidateAdmissionDecision):
        raise ValueError("decision must be a CoverageCandidateAdmissionDecision")
    return decision.to_evidence_dict()


def coverage_candidate_admission_evidence_sha256(
    value: CoverageCandidateAdmissionDecision | Mapping[str, object],
) -> str:
    """Hash admission evidence with the repository's canonical JSON codec."""

    if isinstance(value, CoverageCandidateAdmissionDecision):
        evidence = value.to_evidence_dict()
    elif isinstance(value, Mapping):
        evidence = value
    else:
        raise ValueError("admission evidence must be a decision or mapping")
    return payload_sha256(evidence)


def _candidate_evidence(
    plan: CoverageSurveyPlan,
    candidate: SurveyCandidate,
) -> CoverageCandidateEvidence:
    planned_ids = plan.viewpoint_ids
    planned_set = set(planned_ids)
    reported_ids = tuple(candidate.viewpoint_ids)
    reported_set = set(reported_ids)
    known_ids = tuple(
        viewpoint_id for viewpoint_id in planned_ids if viewpoint_id in reported_set
    )
    unknown_ids = tuple(sorted(reported_set - planned_set))
    viewpoint_ids_distinct = len(reported_ids) == len(reported_set)
    distinct_known_viewpoint_count = len(known_ids)
    confidence_met = (
        candidate.confidence + _COVERAGE_COMPARISON_EPSILON
        >= plan.config.minimum_candidate_confidence
    )
    hit_count_met = candidate.hit_count >= plan.config.minimum_candidate_hits
    distinct_known_viewpoints_met = (
        viewpoint_ids_distinct
        and not unknown_ids
        and distinct_known_viewpoint_count
        >= plan.config.minimum_distinct_viewpoints
    )
    required_exact_ids = (
        planned_ids
        if plan.config.exact_inspection_point_count == 2
        else ()
    )
    required_exact_viewpoints_met = (
        not required_exact_ids
        or set(required_exact_ids).issubset(reported_set)
    )

    reasons: list[str] = []
    if not confidence_met:
        reasons.append("confidence_below_minimum")
    if not hit_count_met:
        reasons.append("hit_count_below_minimum")
    if not viewpoint_ids_distinct:
        reasons.append("viewpoint_ids_replayed")
    if unknown_ids:
        reasons.append("unknown_viewpoint_ids")
    if distinct_known_viewpoint_count < plan.config.minimum_distinct_viewpoints:
        reasons.append("distinct_known_viewpoint_count_below_minimum")
    if not required_exact_viewpoints_met:
        reasons.append("exact_two_planned_viewpoints_missing")

    return CoverageCandidateEvidence(
        candidate_uid=candidate.candidate_uid,
        status=candidate.status,
        confidence=candidate.confidence,
        minimum_confidence=plan.config.minimum_candidate_confidence,
        confidence_met=confidence_met,
        hit_count=candidate.hit_count,
        minimum_hit_count=plan.config.minimum_candidate_hits,
        hit_count_met=hit_count_met,
        viewpoint_ids=tuple(sorted(reported_set)),
        known_viewpoint_ids=known_ids,
        unknown_viewpoint_ids=unknown_ids,
        viewpoint_ids_distinct=viewpoint_ids_distinct,
        distinct_known_viewpoint_count=distinct_known_viewpoint_count,
        minimum_distinct_viewpoints=plan.config.minimum_distinct_viewpoints,
        distinct_known_viewpoints_met=distinct_known_viewpoints_met,
        required_exact_viewpoint_ids=required_exact_ids,
        required_exact_viewpoints_met=required_exact_viewpoints_met,
        admissible=(
            confidence_met
            and hit_count_met
            and distinct_known_viewpoints_met
            and required_exact_viewpoints_met
        ),
        reasons=tuple(reasons),
    )


def _validate_input_structures(
    plan: object,
    progress: object,
    registry: object,
) -> None:
    if not isinstance(plan, CoverageSurveyPlan):
        raise ValueError("plan must be a CoverageSurveyPlan")
    if not isinstance(progress, CoverageSurveyProgress):
        raise ValueError("progress must be a CoverageSurveyProgress")
    if not isinstance(registry, StandSurveyRegistry):
        raise ValueError("registry must be a StandSurveyRegistry")
    try:
        validate_survey_progress(progress, plan)
        validate_stand_survey_registry(registry, plan)
    except ValueError:
        raise
    except (AttributeError, KeyError, TypeError, OverflowError) as exc:
        raise ValueError("malformed coverage candidate admission structure") from exc


def _progress_snapshot_sha256(progress: CoverageSurveyProgress) -> str:
    return payload_sha256(
        {
            "schema_version": progress.schema_version,
            "survey_id": progress.survey_id,
            "plan_sha256": progress.plan_sha256,
            "visited_viewpoint_ids": list(progress.visited_viewpoint_ids),
        }
    )


def _registry_snapshot_sha256(registry: StandSurveyRegistry) -> str:
    return payload_sha256(
        {
            "schema_version": registry.schema_version,
            "survey_id": registry.survey_id,
            "planning_frame": registry.planning_frame,
            "map_bundle_sha256": registry.map_bundle_sha256,
            "candidates": [
                _candidate_snapshot_payload(candidate)
                for candidate in registry.candidates
            ],
        }
    )


def _candidate_snapshot_payload(
    candidate: SurveyCandidate,
) -> dict[str, object]:
    return {
        "candidate_uid": candidate.candidate_uid,
        "x_m": candidate.x_m,
        "y_m": candidate.y_m,
        "radius_m": candidate.radius_m,
        "uncertainty_m": candidate.uncertainty_m,
        "keepout_radius_m": candidate.keepout_radius_m,
        "confidence": candidate.confidence,
        "hit_count": candidate.hit_count,
        "first_seen_sec": candidate.first_seen_sec,
        "last_seen_sec": candidate.last_seen_sec,
        "source_observation_ids": list(candidate.source_observation_ids),
        "viewpoint_ids": list(candidate.viewpoint_ids),
        "status": candidate.status,
    }


__all__ = [
    "COVERAGE_CANDIDATE_ADMISSION_SCHEMA_VERSION",
    "CoverageCandidateAdmissionDecision",
    "CoverageCandidateEvidence",
    "coverage_candidate_admission_evidence",
    "coverage_candidate_admission_evidence_sha256",
    "evaluate_coverage_candidate_admission",
]
