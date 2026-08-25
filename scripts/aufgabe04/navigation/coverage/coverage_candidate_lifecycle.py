"""Pure lifecycle evidence for coverage-survey LiDAR candidates.

The survey registry and the camera-approach queue represent different safety
boundaries.  A registry candidate was admitted by the stopped LiDAR/static-map
producer, while only ``pending_camera`` candidates belong to the existing
camera-validation queue.  This module keeps those facts separate and provides
an exact-two *coverage checkpoint* decision that never authorizes approach
motion or constructs a candidate snapshot.

Malformed snapshots raise ``ValueError``.  Structurally valid but incomplete
coverage or insufficient LiDAR evidence returns a deterministic decision with
``ready=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.coverage.stand_candidate_population_retention import (
    STATIC_MAP_DISPOSITION_ADMITTED,
    STATIC_MAP_DISPOSITION_BOUNDARY_PROVISIONAL,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    STATUS_CONFIRMED,
    STATUS_PENDING_CAMERA,
    STATUS_REJECTED,
    CoverageSurveyPlan,
    CoverageSurveyProgress,
    StandSurveyRegistry,
    SurveyCandidate,
    coverage_survey_plan_sha256,
    stand_survey_registry_sha256,
    validate_stand_survey_registry,
    validate_survey_progress,
    visited_coverage_ratio,
)


COVERAGE_CANDIDATE_LIFECYCLE_SCHEMA_VERSION = 2
EXACT_TWO_LIDAR_CHECKPOINT_SCHEMA_VERSION = 2
STATIC_MAP_ADMISSION_BASIS = "validated_survey_registry_membership"
BOUNDARY_PROVISIONAL_STATIC_MAP_BASIS = "boundary_provisional_static_map_shortfall"
_COVERAGE_COMPARISON_EPSILON = 1.0e-12


@dataclass(frozen=True)
class CoverageCandidateLifecycleEvidence:
    """One candidate's immutable LiDAR and camera lifecycle classes."""

    candidate_uid: str
    registry_status: str
    static_map_admission_basis: str
    static_map_disposition: str
    lidar_static_map_admitted: bool
    lidar_population_retained: bool
    active_lidar: bool
    confidence: float
    minimum_confidence: float
    confidence_supported: bool
    hit_count: int
    minimum_hit_count: int
    hit_count_supported: bool
    viewpoint_ids: tuple[str, ...]
    known_viewpoint_ids: tuple[str, ...]
    unknown_viewpoint_ids: tuple[str, ...]
    viewpoint_ids_distinct: bool
    basic_lidar_support: bool
    minimum_distinct_viewpoints: int
    distinct_known_viewpoint_count: int
    required_exact_viewpoint_ids: tuple[str, ...]
    required_exact_viewpoints_met: bool
    multi_view_supported: bool
    camera_validation_queued: bool
    camera_confirmed: bool
    camera_rejected: bool
    support_reasons: tuple[str, ...]

    def to_evidence_dict(self) -> dict[str, object]:
        return {
            "candidate_uid": self.candidate_uid,
            "registry_status": self.registry_status,
            "static_map_admission": {
                "admitted": self.lidar_static_map_admitted,
                "disposition": self.static_map_disposition,
                "population_retained": self.lidar_population_retained,
                "basis": self.static_map_admission_basis,
            },
            "lifecycle": {
                "active_lidar": self.active_lidar,
                "multi_view_supported": self.multi_view_supported,
                "camera_validation_queued": self.camera_validation_queued,
                "camera_confirmed": self.camera_confirmed,
                "camera_rejected": self.camera_rejected,
            },
            "basic_lidar_support": {
                "met": self.basic_lidar_support,
                "reasons": list(self.support_reasons),
                "confidence": {
                    "value": self.confidence,
                    "minimum": self.minimum_confidence,
                    "met": self.confidence_supported,
                },
                "hits": {
                    "count": self.hit_count,
                    "minimum": self.minimum_hit_count,
                    "met": self.hit_count_supported,
                },
                "viewpoints": {
                    "ids": list(self.viewpoint_ids),
                    "known_ids": list(self.known_viewpoint_ids),
                    "unknown_ids": list(self.unknown_viewpoint_ids),
                    "ids_distinct": self.viewpoint_ids_distinct,
                    "distinct_known_count": (
                        self.distinct_known_viewpoint_count
                    ),
                    "minimum_distinct_count": (
                        self.minimum_distinct_viewpoints
                    ),
                    "required_exact_ids": list(
                        self.required_exact_viewpoint_ids
                    ),
                    "required_exact_ids_met": (
                        self.required_exact_viewpoints_met
                    ),
                },
            },
        }


@dataclass(frozen=True)
class CoverageCandidatePopulation:
    """Deterministic lifecycle partition of one validated survey registry."""

    schema_version: int
    survey_id: str
    planning_frame: str
    map_bundle_sha256: str
    plan_sha256: str
    registry_snapshot_sha256: str
    candidates: tuple[CoverageCandidateLifecycleEvidence, ...]
    lidar_static_map_admitted_candidate_uids: tuple[str, ...]
    lidar_boundary_provisional_candidate_uids: tuple[str, ...]
    lidar_population_retained_candidate_uids: tuple[str, ...]
    active_lidar_candidate_uids: tuple[str, ...]
    basic_lidar_supported_candidate_uids: tuple[str, ...]
    multi_view_supported_candidate_uids: tuple[str, ...]
    camera_queue_candidate_uids: tuple[str, ...]
    camera_confirmed_candidate_uids: tuple[str, ...]
    rejected_candidate_uids: tuple[str, ...]

    def to_evidence_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "survey": {
                "survey_id": self.survey_id,
                "planning_frame": self.planning_frame,
                "map_bundle_sha256": self.map_bundle_sha256,
                "plan_sha256": self.plan_sha256,
                "registry_snapshot_sha256": self.registry_snapshot_sha256,
            },
            "candidate_uids": {
                "lidar_static_map_admitted": list(
                    self.lidar_static_map_admitted_candidate_uids
                ),
                "lidar_boundary_provisional": list(
                    self.lidar_boundary_provisional_candidate_uids
                ),
                "lidar_population_retained": list(
                    self.lidar_population_retained_candidate_uids
                ),
                "active_lidar": list(self.active_lidar_candidate_uids),
                "basic_lidar_supported": list(
                    self.basic_lidar_supported_candidate_uids
                ),
                "multi_view_supported": list(
                    self.multi_view_supported_candidate_uids
                ),
                "camera_validation_queue": list(
                    self.camera_queue_candidate_uids
                ),
                "camera_confirmed": list(
                    self.camera_confirmed_candidate_uids
                ),
                "rejected": list(self.rejected_candidate_uids),
            },
            "candidates": [
                candidate.to_evidence_dict() for candidate in self.candidates
            ],
        }


@dataclass(frozen=True)
class ExactTwoLidarCheckpointDecision:
    """LiDAR-only completion decision for an exact-two coverage checkpoint.

    ``ready`` means the stopped survey has enough evidence to complete the
    checkpoint.  It is deliberately not camera-candidate admission, and the
    decision can never authorize camera-approach motion.
    """

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
    active_lidar_candidate_count: int
    active_lidar_candidate_count_met: bool
    unsupported_active_lidar_candidate_uids: tuple[str, ...]
    active_lidar_candidate_support_met: bool
    camera_approach_authorized: bool
    population: CoverageCandidatePopulation

    @property
    def admitted_lidar_candidate_uids(self) -> tuple[str, ...]:
        """Return checkpoint-admitted UIDs only after every gate passes."""

        if not self.ready:
            return ()
        return self.population.active_lidar_candidate_uids

    def to_evidence_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "decision": "ready" if self.ready else "not_ready",
            "ready": self.ready,
            "reasons": list(self.reasons),
            "camera_approach_authorized": self.camera_approach_authorized,
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
                "unvisited_viewpoint_ids": list(
                    self.unvisited_viewpoint_ids
                ),
                "all_planned_viewpoints_visited": (
                    self.all_planned_viewpoints_visited
                ),
                "visited_coverage_ratio": self.visited_coverage_ratio,
                "coverage_threshold": self.coverage_threshold,
                "coverage_threshold_met": self.coverage_threshold_met,
                "comparison_epsilon": _COVERAGE_COMPARISON_EPSILON,
            },
            "lidar_candidate_gate": {
                "expected_stand_count": self.expected_stand_count,
                "active_lidar_candidate_count": (
                    self.active_lidar_candidate_count
                ),
                "active_lidar_candidate_count_met": (
                    self.active_lidar_candidate_count_met
                ),
                "unsupported_active_lidar_candidate_uids": list(
                    self.unsupported_active_lidar_candidate_uids
                ),
                "active_lidar_candidate_support_met": (
                    self.active_lidar_candidate_support_met
                ),
                "admitted_lidar_candidate_uids": list(
                    self.admitted_lidar_candidate_uids
                ),
            },
            "population": self.population.to_evidence_dict(),
        }


def classify_coverage_candidates(
    plan: CoverageSurveyPlan,
    registry: StandSurveyRegistry,
) -> CoverageCandidatePopulation:
    """Classify candidates without changing registry state or motion scope."""

    if not isinstance(plan, CoverageSurveyPlan):
        raise ValueError("plan must be a CoverageSurveyPlan")
    if not isinstance(registry, StandSurveyRegistry):
        raise ValueError("registry must be a StandSurveyRegistry")
    try:
        validate_stand_survey_registry(registry, plan)
    except ValueError:
        raise
    except (AttributeError, KeyError, TypeError, OverflowError) as exc:
        raise ValueError("malformed coverage candidate registry") from exc

    evidence = tuple(
        _classify_candidate(plan, candidate)
        for candidate in registry.candidates
    )
    return CoverageCandidatePopulation(
        schema_version=COVERAGE_CANDIDATE_LIFECYCLE_SCHEMA_VERSION,
        survey_id=plan.survey_id,
        planning_frame=plan.planning_frame,
        map_bundle_sha256=plan.map_bundle_sha256,
        plan_sha256=coverage_survey_plan_sha256(plan),
        registry_snapshot_sha256=_registry_snapshot_sha256(registry),
        candidates=evidence,
        lidar_static_map_admitted_candidate_uids=_selected_uids(
            evidence,
            "lidar_static_map_admitted",
        ),
        lidar_boundary_provisional_candidate_uids=tuple(
            item.candidate_uid
            for item in evidence
            if item.static_map_disposition
            == STATIC_MAP_DISPOSITION_BOUNDARY_PROVISIONAL
        ),
        lidar_population_retained_candidate_uids=_selected_uids(
            evidence,
            "lidar_population_retained",
        ),
        active_lidar_candidate_uids=_selected_uids(evidence, "active_lidar"),
        basic_lidar_supported_candidate_uids=tuple(
            item.candidate_uid
            for item in evidence
            if item.active_lidar and item.basic_lidar_support
        ),
        multi_view_supported_candidate_uids=_selected_uids(
            evidence,
            "multi_view_supported",
        ),
        camera_queue_candidate_uids=_selected_uids(
            evidence,
            "camera_validation_queued",
        ),
        camera_confirmed_candidate_uids=_selected_uids(
            evidence,
            "camera_confirmed",
        ),
        rejected_candidate_uids=_selected_uids(evidence, "camera_rejected"),
    )


def evaluate_exact_two_lidar_checkpoint(
    plan: CoverageSurveyPlan,
    progress: CoverageSurveyProgress,
    registry: StandSurveyRegistry,
) -> ExactTwoLidarCheckpointDecision:
    """Evaluate exact-two LiDAR checkpoint completion, never motion admission."""

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
        raise ValueError("malformed exact-two LiDAR checkpoint structure") from exc

    population = classify_coverage_candidates(plan, registry)
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
    expected_stand_count = plan.config.expected_stand_count
    active_candidate_count = len(population.active_lidar_candidate_uids)
    active_candidate_count_met = (
        expected_stand_count is not None
        and active_candidate_count == expected_stand_count
    )
    unsupported_active = tuple(
        item.candidate_uid
        for item in population.candidates
        if item.active_lidar and not item.basic_lidar_support
    )
    active_support_met = not unsupported_active

    reasons: list[str] = []
    if plan.config.exact_inspection_point_count != 2:
        reasons.append("exact_two_inspection_scope_required")
    if not all_planned_viewpoints_visited:
        reasons.append("planned_viewpoints_incomplete")
    if not coverage_threshold_met:
        reasons.append("visited_coverage_below_threshold")
    if expected_stand_count is None:
        reasons.append("expected_stand_count_unset")
    elif not active_candidate_count_met:
        reasons.append("active_lidar_candidate_count_mismatch")
    if not active_support_met:
        reasons.append("active_lidar_candidate_support_not_met")

    return ExactTwoLidarCheckpointDecision(
        schema_version=EXACT_TWO_LIDAR_CHECKPOINT_SCHEMA_VERSION,
        survey_id=plan.survey_id,
        planning_frame=plan.planning_frame,
        map_bundle_sha256=plan.map_bundle_sha256,
        plan_sha256=population.plan_sha256,
        progress_snapshot_sha256=_progress_snapshot_sha256(progress),
        registry_snapshot_sha256=population.registry_snapshot_sha256,
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
        active_lidar_candidate_count=active_candidate_count,
        active_lidar_candidate_count_met=active_candidate_count_met,
        unsupported_active_lidar_candidate_uids=unsupported_active,
        active_lidar_candidate_support_met=active_support_met,
        camera_approach_authorized=False,
        population=population,
    )


def coverage_candidate_population_evidence(
    population: CoverageCandidatePopulation,
) -> dict[str, object]:
    """Return the canonical JSON-safe population payload."""

    if not isinstance(population, CoverageCandidatePopulation):
        raise ValueError("population must be a CoverageCandidatePopulation")
    return population.to_evidence_dict()


def coverage_candidate_population_evidence_sha256(
    value: CoverageCandidatePopulation | Mapping[str, object],
) -> str:
    """Hash lifecycle evidence with the repository's canonical JSON codec."""

    if isinstance(value, CoverageCandidatePopulation):
        payload = value.to_evidence_dict()
    elif isinstance(value, Mapping):
        payload = value
    else:
        raise ValueError("population evidence must be a population or mapping")
    return payload_sha256(payload)


def exact_two_lidar_checkpoint_evidence(
    decision: ExactTwoLidarCheckpointDecision,
) -> dict[str, object]:
    """Return the canonical JSON-safe exact-two checkpoint payload."""

    if not isinstance(decision, ExactTwoLidarCheckpointDecision):
        raise ValueError("decision must be an ExactTwoLidarCheckpointDecision")
    return decision.to_evidence_dict()


def exact_two_lidar_checkpoint_evidence_sha256(
    value: ExactTwoLidarCheckpointDecision | Mapping[str, object],
) -> str:
    """Hash exact-two checkpoint evidence with the canonical JSON codec."""

    if isinstance(value, ExactTwoLidarCheckpointDecision):
        payload = value.to_evidence_dict()
    elif isinstance(value, Mapping):
        payload = value
    else:
        raise ValueError("checkpoint evidence must be a decision or mapping")
    return payload_sha256(payload)


def _classify_candidate(
    plan: CoverageSurveyPlan,
    candidate: SurveyCandidate,
) -> CoverageCandidateLifecycleEvidence:
    planned_ids = plan.viewpoint_ids
    planned_set = set(planned_ids)
    reported_ids = tuple(candidate.viewpoint_ids)
    reported_set = set(reported_ids)
    known_ids = tuple(
        viewpoint_id for viewpoint_id in planned_ids if viewpoint_id in reported_set
    )
    unknown_ids = tuple(sorted(reported_set - planned_set))
    ids_distinct = len(reported_ids) == len(reported_set)
    confidence_supported = (
        candidate.confidence + _COVERAGE_COMPARISON_EPSILON
        >= plan.config.minimum_candidate_confidence
    )
    hit_count_supported = (
        candidate.hit_count >= plan.config.minimum_candidate_hits
    )
    basic_lidar_support = (
        confidence_supported
        and hit_count_supported
        and ids_distinct
        and not unknown_ids
        and bool(known_ids)
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
    population_retained = candidate.static_map_disposition in {
        STATIC_MAP_DISPOSITION_ADMITTED,
        STATIC_MAP_DISPOSITION_BOUNDARY_PROVISIONAL,
    }
    active_lidar = (
        candidate.status != STATUS_REJECTED and population_retained
    )
    multi_view_supported = (
        active_lidar
        and basic_lidar_support
        and len(known_ids) >= plan.config.minimum_distinct_viewpoints
        and required_exact_viewpoints_met
    )

    support_reasons: list[str] = []
    if not confidence_supported:
        support_reasons.append("confidence_below_minimum")
    if not hit_count_supported:
        support_reasons.append("hit_count_below_minimum")
    if not ids_distinct:
        support_reasons.append("viewpoint_ids_replayed")
    if unknown_ids:
        support_reasons.append("unknown_viewpoint_ids")
    if not known_ids:
        support_reasons.append("known_viewpoint_evidence_missing")

    return CoverageCandidateLifecycleEvidence(
        candidate_uid=candidate.candidate_uid,
        registry_status=candidate.status,
        static_map_admission_basis=(
            BOUNDARY_PROVISIONAL_STATIC_MAP_BASIS
            if (
                candidate.static_map_disposition
                == STATIC_MAP_DISPOSITION_BOUNDARY_PROVISIONAL
            )
            else STATIC_MAP_ADMISSION_BASIS
        ),
        static_map_disposition=candidate.static_map_disposition,
        # The autonomous survey registry's producer admits candidates only
        # after the stopped static-map gate, except for explicitly retained
        # boundary-provisional candidates.  This records lineage; it does not
        # recompute geometry from coordinates alone.
        lidar_static_map_admitted=(
            candidate.static_map_disposition == STATIC_MAP_DISPOSITION_ADMITTED
        ),
        lidar_population_retained=population_retained,
        active_lidar=active_lidar,
        confidence=candidate.confidence,
        minimum_confidence=plan.config.minimum_candidate_confidence,
        confidence_supported=confidence_supported,
        hit_count=candidate.hit_count,
        minimum_hit_count=plan.config.minimum_candidate_hits,
        hit_count_supported=hit_count_supported,
        viewpoint_ids=tuple(sorted(reported_set)),
        known_viewpoint_ids=known_ids,
        unknown_viewpoint_ids=unknown_ids,
        viewpoint_ids_distinct=ids_distinct,
        basic_lidar_support=basic_lidar_support,
        minimum_distinct_viewpoints=plan.config.minimum_distinct_viewpoints,
        distinct_known_viewpoint_count=len(known_ids),
        required_exact_viewpoint_ids=required_exact_ids,
        required_exact_viewpoints_met=required_exact_viewpoints_met,
        multi_view_supported=multi_view_supported,
        # Queue membership remains the established status transition.  A
        # provisional candidate never becomes approach-authorized here.
        camera_validation_queued=(
            candidate.status == STATUS_PENDING_CAMERA
        ),
        camera_confirmed=(candidate.status == STATUS_CONFIRMED),
        camera_rejected=(candidate.status == STATUS_REJECTED),
        support_reasons=tuple(support_reasons),
    )


def _selected_uids(
    evidence: tuple[CoverageCandidateLifecycleEvidence, ...],
    attribute: str,
) -> tuple[str, ...]:
    return tuple(
        item.candidate_uid for item in evidence if bool(getattr(item, attribute))
    )


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
    return stand_survey_registry_sha256(registry)


__all__ = [
    "COVERAGE_CANDIDATE_LIFECYCLE_SCHEMA_VERSION",
    "EXACT_TWO_LIDAR_CHECKPOINT_SCHEMA_VERSION",
    "STATIC_MAP_ADMISSION_BASIS",
    "BOUNDARY_PROVISIONAL_STATIC_MAP_BASIS",
    "CoverageCandidateLifecycleEvidence",
    "CoverageCandidatePopulation",
    "ExactTwoLidarCheckpointDecision",
    "classify_coverage_candidates",
    "coverage_candidate_population_evidence",
    "coverage_candidate_population_evidence_sha256",
    "evaluate_exact_two_lidar_checkpoint",
    "exact_two_lidar_checkpoint_evidence",
    "exact_two_lidar_checkpoint_evidence_sha256",
]
