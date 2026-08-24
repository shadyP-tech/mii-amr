"""Typed, ROS-free contracts for exact-two LiDAR camera admission."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    STATUS_PENDING_CAMERA,
    STATUS_PROVISIONAL,
    StandSurveyRegistry,
    SurveyCandidate,
    validate_stand_survey_registry,
)


EXACT_TWO_CAMERA_ADMISSION_SCHEMA_VERSION = 1
EXACT_TWO_CAMERA_HANDOFF_SCHEMA_VERSION = 1

SUPPORT_CLASS_MULTI_VIEW = "multi_view"
SUPPORT_CLASS_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION = (
    "single_view_requires_camera_validation"
)
VALID_EXACT_TWO_CAMERA_SUPPORT_CLASSES = frozenset(
    {
        SUPPORT_CLASS_MULTI_VIEW,
        SUPPORT_CLASS_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION,
    }
)
SOURCE_KIND_MULTI_VIEW = "lidar/exact_two_multi_view"
SOURCE_KIND_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION = (
    "lidar/exact_two_single_view_requires_camera_validation"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_FRAME = re.compile(r"^[A-Za-z][A-Za-z0-9_/.-]{0,127}$")


class ExactTwoCameraAdmissionError(ValueError):
    """Validation/storage error with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ExactTwoCameraCandidateEvidence:
    """One active registry candidate's immutable camera support class."""

    candidate_uid: str
    registry_status: str
    active_lidar: bool
    static_map_admitted: bool
    basic_lidar_supported: bool
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
    distinct_known_viewpoint_count: int
    support_class: str | None
    source_kind: str | None
    admissible: bool
    reasons: tuple[str, ...]

    def to_evidence_dict(self) -> dict[str, object]:
        return {
            "candidate_uid": self.candidate_uid,
            "registry_status": self.registry_status,
            "active_lidar": self.active_lidar,
            "static_map_admitted": self.static_map_admitted,
            "basic_lidar_supported": self.basic_lidar_supported,
            "confidence": self.confidence,
            "minimum_confidence": self.minimum_confidence,
            "confidence_supported": self.confidence_supported,
            "hit_count": self.hit_count,
            "minimum_hit_count": self.minimum_hit_count,
            "hit_count_supported": self.hit_count_supported,
            "viewpoint_ids": list(self.viewpoint_ids),
            "known_viewpoint_ids": list(self.known_viewpoint_ids),
            "unknown_viewpoint_ids": list(self.unknown_viewpoint_ids),
            "viewpoint_ids_distinct": self.viewpoint_ids_distinct,
            "distinct_known_viewpoint_count": self.distinct_known_viewpoint_count,
            "support_class": self.support_class,
            "source_kind": self.source_kind,
            "admissible": self.admissible,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ExactTwoCameraAdmissionDecision:
    """Complete, motion-neutral camera-population decision."""

    schema_version: int
    survey_id: str
    planning_frame: str
    map_bundle_sha256: str
    plan_sha256: str
    progress_snapshot_sha256: str
    source_registry_sha256: str
    lidar_checkpoint_sha256: str
    ready: bool
    reasons: tuple[str, ...]
    camera_population_ready: bool
    motion_authorized: bool
    expected_stand_count: int | None
    active_candidate_count: int
    multi_view_candidate_uids: tuple[str, ...]
    single_view_candidate_uids: tuple[str, ...]
    blocked_candidate_uids: tuple[str, ...]
    candidate_evidence: tuple[ExactTwoCameraCandidateEvidence, ...]

    @property
    def admitted_candidate_uids(self) -> tuple[str, ...]:
        if not self.ready:
            return ()
        return tuple(
            sorted(self.multi_view_candidate_uids + self.single_view_candidate_uids)
        )

    def candidate_for(
        self, candidate_uid: str
    ) -> ExactTwoCameraCandidateEvidence | None:
        return next(
            (
                evidence
                for evidence in self.candidate_evidence
                if evidence.candidate_uid == candidate_uid
            ),
            None,
        )

    def to_evidence_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "survey_id": self.survey_id,
            "planning_frame": self.planning_frame,
            "map_bundle_sha256": self.map_bundle_sha256,
            "plan_sha256": self.plan_sha256,
            "progress_snapshot_sha256": self.progress_snapshot_sha256,
            "source_registry_sha256": self.source_registry_sha256,
            "lidar_checkpoint_sha256": self.lidar_checkpoint_sha256,
            "ready": self.ready,
            "reasons": list(self.reasons),
            "camera_population_ready": self.camera_population_ready,
            "motion_authorized": self.motion_authorized,
            "expected_stand_count": self.expected_stand_count,
            "active_candidate_count": self.active_candidate_count,
            "multi_view_candidate_uids": list(self.multi_view_candidate_uids),
            "single_view_candidate_uids": list(self.single_view_candidate_uids),
            "blocked_candidate_uids": list(self.blocked_candidate_uids),
            "candidate_evidence": [
                evidence.to_evidence_dict()
                for evidence in self.candidate_evidence
            ],
        }


@dataclass(frozen=True)
class ExactTwoCameraHandoffArtifact:
    """Content-hashed, motion-neutral binding consumed by camera approach."""

    schema_version: int
    handoff_id: str
    created_unix_sec: float
    survey_id: str
    planning_frame: str
    map_bundle_sha256: str
    plan_sha256: str
    progress_snapshot_sha256: str
    source_registry_sha256: str
    terminal_checkpoint_path: str
    terminal_checkpoint_sha256: str
    lidar_admission_path: str
    lidar_admission_sha256: str
    lidar_checkpoint_sha256: str
    camera_admission_path: str
    camera_admission_sha256: str
    candidate_snapshot_path: str
    candidate_snapshot_sha256: str
    candidate_snapshot_id: str
    camera_population_ready: bool
    motion_authorized: bool
    admission_decision: ExactTwoCameraAdmissionDecision

    @property
    def admitted_candidate_uids(self) -> tuple[str, ...]:
        return self.admission_decision.admitted_candidate_uids


def validate_exact_two_camera_admission(
    decision: ExactTwoCameraAdmissionDecision,
) -> None:
    if not isinstance(decision, ExactTwoCameraAdmissionDecision):
        raise ExactTwoCameraAdmissionError(
            "invalid_admission",
            "decision must be an ExactTwoCameraAdmissionDecision",
        )
    if (
        type(decision.schema_version) is not int
        or decision.schema_version != EXACT_TWO_CAMERA_ADMISSION_SCHEMA_VERSION
    ):
        raise ExactTwoCameraAdmissionError(
            "schema_mismatch", "unsupported exact-two camera admission schema"
        )
    validate_id(decision.survey_id, "survey_id")
    validate_frame(decision.planning_frame, "planning_frame")
    for field_name in (
        "map_bundle_sha256",
        "plan_sha256",
        "progress_snapshot_sha256",
        "source_registry_sha256",
        "lidar_checkpoint_sha256",
    ):
        validate_sha256(getattr(decision, field_name), field_name)
    boolean(decision.ready, "ready")
    boolean(decision.camera_population_ready, "camera_population_ready")
    boolean(decision.motion_authorized, "motion_authorized")
    if decision.motion_authorized:
        raise ExactTwoCameraAdmissionError(
            "motion_scope_violation",
            "camera admission must never authorize motion",
        )
    if decision.ready != decision.camera_population_ready:
        raise ExactTwoCameraAdmissionError(
            "invalid_admission", "ready and camera_population_ready must agree"
        )
    string_tuple(decision.reasons, "reasons", safe_ids=False)
    if decision.expected_stand_count is not None:
        nonnegative_integer(
            decision.expected_stand_count, "expected_stand_count"
        )
    nonnegative_integer(decision.active_candidate_count, "active_candidate_count")
    for field_name in (
        "multi_view_candidate_uids",
        "single_view_candidate_uids",
        "blocked_candidate_uids",
    ):
        sorted_unique_ids(getattr(decision, field_name), field_name)
    partitions = (
        set(decision.multi_view_candidate_uids),
        set(decision.single_view_candidate_uids),
        set(decision.blocked_candidate_uids),
    )
    if any(
        partitions[i].intersection(partitions[j])
        for i in range(3)
        for j in range(i + 1, 3)
    ):
        raise ExactTwoCameraAdmissionError(
            "invalid_admission", "candidate UID partitions must be disjoint"
        )
    if not isinstance(decision.candidate_evidence, tuple):
        raise ExactTwoCameraAdmissionError(
            "invalid_admission", "candidate_evidence must be a tuple"
        )
    evidence_uids = tuple(item.candidate_uid for item in decision.candidate_evidence)
    if evidence_uids != tuple(sorted(set(evidence_uids))):
        raise ExactTwoCameraAdmissionError(
            "invalid_admission", "candidate evidence must be sorted and unique"
        )
    for evidence in decision.candidate_evidence:
        validate_exact_two_camera_candidate_evidence(evidence)
    if len(evidence_uids) != decision.active_candidate_count:
        raise ExactTwoCameraAdmissionError(
            "invalid_admission", "active candidate count does not match evidence"
        )
    if set(evidence_uids) != set().union(*partitions):
        raise ExactTwoCameraAdmissionError(
            "invalid_admission", "candidate partitions do not cover active evidence"
        )
    for evidence in decision.candidate_evidence:
        expected_partition = (
            decision.multi_view_candidate_uids
            if evidence.support_class == SUPPORT_CLASS_MULTI_VIEW
            and evidence.admissible
            else decision.single_view_candidate_uids
            if evidence.support_class
            == SUPPORT_CLASS_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION
            and evidence.admissible
            else decision.blocked_candidate_uids
        )
        if evidence.candidate_uid not in expected_partition:
            raise ExactTwoCameraAdmissionError(
                "invalid_admission",
                f"support partition mismatch for {evidence.candidate_uid!r}",
            )
    if decision.ready:
        if decision.reasons:
            raise ExactTwoCameraAdmissionError(
                "invalid_admission", "ready admission must not contain reasons"
            )
        if decision.expected_stand_count is None:
            raise ExactTwoCameraAdmissionError(
                "invalid_admission", "ready admission requires expected count"
            )
        if decision.active_candidate_count != decision.expected_stand_count:
            raise ExactTwoCameraAdmissionError(
                "invalid_admission", "ready admission count does not match expected"
            )
        if decision.blocked_candidate_uids:
            raise ExactTwoCameraAdmissionError(
                "invalid_admission", "ready admission cannot contain blocked UIDs"
            )
        if len(decision.admitted_candidate_uids) != decision.active_candidate_count:
            raise ExactTwoCameraAdmissionError(
                "invalid_admission", "ready admission partition is incomplete"
            )
    elif not decision.reasons:
        raise ExactTwoCameraAdmissionError(
            "invalid_admission", "not-ready admission requires at least one reason"
        )


def validate_exact_two_camera_candidate_evidence(
    evidence: ExactTwoCameraCandidateEvidence,
) -> None:
    if not isinstance(evidence, ExactTwoCameraCandidateEvidence):
        raise ExactTwoCameraAdmissionError(
            "invalid_admission", "candidate evidence has the wrong type"
        )
    validate_id(evidence.candidate_uid, "candidate_uid")
    if not isinstance(evidence.registry_status, str) or not evidence.registry_status:
        raise ExactTwoCameraAdmissionError(
            "invalid_admission", "registry_status must be a non-empty string"
        )
    for field_name in (
        "active_lidar",
        "static_map_admitted",
        "basic_lidar_supported",
        "confidence_supported",
        "hit_count_supported",
        "viewpoint_ids_distinct",
        "admissible",
    ):
        boolean(getattr(evidence, field_name), field_name)
    confidence = finite_number(evidence.confidence, "confidence")
    minimum = finite_number(evidence.minimum_confidence, "minimum_confidence")
    if not (0.0 <= confidence <= 1.0 and 0.0 <= minimum <= 1.0):
        raise ExactTwoCameraAdmissionError(
            "invalid_admission", "candidate confidence values must be in [0, 1]"
        )
    nonnegative_integer(evidence.hit_count, "hit_count")
    nonnegative_integer(evidence.minimum_hit_count, "minimum_hit_count")
    sorted_unique_ids(evidence.viewpoint_ids, "viewpoint_ids")
    sorted_unique_ids(evidence.known_viewpoint_ids, "known_viewpoint_ids")
    sorted_unique_ids(evidence.unknown_viewpoint_ids, "unknown_viewpoint_ids")
    nonnegative_integer(
        evidence.distinct_known_viewpoint_count,
        "distinct_known_viewpoint_count",
    )
    if evidence.distinct_known_viewpoint_count != len(evidence.known_viewpoint_ids):
        raise ExactTwoCameraAdmissionError(
            "invalid_admission", "known viewpoint count mismatch"
        )
    if evidence.support_class is not None and (
        evidence.support_class not in VALID_EXACT_TWO_CAMERA_SUPPORT_CLASSES
    ):
        raise ExactTwoCameraAdmissionError(
            "invalid_support_class", "candidate has an unknown support class"
        )
    expected_source = (
        required_source_kind(evidence.support_class)
        if evidence.support_class is not None
        else None
    )
    if evidence.source_kind != expected_source:
        raise ExactTwoCameraAdmissionError(
            "invalid_admission", "source kind does not match support class"
        )
    string_tuple(evidence.reasons, "reasons", safe_ids=False)
    if not evidence.admissible:
        return
    if (
        not evidence.active_lidar
        or not evidence.static_map_admitted
        or not evidence.basic_lidar_supported
        or evidence.support_class is None
        or evidence.reasons
    ):
        raise ExactTwoCameraAdmissionError(
            "invalid_admission", "admissible candidate has unmet evidence"
        )
    if (
        evidence.support_class == SUPPORT_CLASS_MULTI_VIEW
        and evidence.registry_status != STATUS_PENDING_CAMERA
    ):
        raise ExactTwoCameraAdmissionError(
            "invalid_admission", "multi-view candidate is not pending_camera"
        )
    if (
        evidence.support_class
        == SUPPORT_CLASS_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION
        and (
            evidence.registry_status != STATUS_PROVISIONAL
            or evidence.distinct_known_viewpoint_count != 1
        )
    ):
        raise ExactTwoCameraAdmissionError(
            "invalid_admission", "single-view candidate evidence is inconsistent"
        )


def require_admitted_candidate_support(
    decision: ExactTwoCameraAdmissionDecision,
    candidate_uid: str,
    required_support_class: str | None = None,
) -> ExactTwoCameraCandidateEvidence:
    validate_exact_two_camera_admission(decision)
    validate_id(candidate_uid, "candidate_uid")
    if required_support_class is not None and (
        required_support_class not in VALID_EXACT_TWO_CAMERA_SUPPORT_CLASSES
    ):
        raise ExactTwoCameraAdmissionError(
            "invalid_support_class",
            f"unknown exact-two support class {required_support_class!r}",
        )
    evidence = decision.candidate_for(candidate_uid)
    if (
        not decision.ready
        or evidence is None
        or not evidence.admissible
        or candidate_uid not in decision.admitted_candidate_uids
    ):
        raise ExactTwoCameraAdmissionError(
            "candidate_not_admitted",
            f"candidate {candidate_uid!r} is not in the ready camera population",
        )
    if (
        required_support_class is not None
        and evidence.support_class != required_support_class
    ):
        raise ExactTwoCameraAdmissionError(
            "support_class_mismatch",
            f"candidate {candidate_uid!r} has support {evidence.support_class!r}, "
            f"not {required_support_class!r}",
        )
    return evidence


def required_source_kind(support_class: str | None) -> str:
    if support_class == SUPPORT_CLASS_MULTI_VIEW:
        return SOURCE_KIND_MULTI_VIEW
    if support_class == SUPPORT_CLASS_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION:
        return SOURCE_KIND_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION
    raise ExactTwoCameraAdmissionError(
        "invalid_support_class", f"unknown support class {support_class!r}"
    )


def stand_survey_registry_sha256(registry: StandSurveyRegistry) -> str:
    if not isinstance(registry, StandSurveyRegistry):
        raise ExactTwoCameraAdmissionError(
            "invalid_registry", "registry must be a StandSurveyRegistry"
        )
    try:
        validate_stand_survey_registry(registry)
    except ValueError as exc:
        raise ExactTwoCameraAdmissionError("invalid_registry", str(exc)) from exc
    return payload_sha256(
        {
            "schema_version": registry.schema_version,
            "survey_id": registry.survey_id,
            "planning_frame": registry.planning_frame,
            "map_bundle_sha256": registry.map_bundle_sha256,
            "candidates": [
                _survey_candidate_payload(candidate)
                for candidate in registry.candidates
            ],
        }
    )


def _survey_candidate_payload(candidate: SurveyCandidate) -> dict[str, object]:
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


def validate_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ExactTwoCameraAdmissionError(
            "invalid_identifier", f"{name} is not a safe identifier"
        )
    return value


def validate_frame(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_FRAME.fullmatch(value):
        raise ExactTwoCameraAdmissionError(
            "invalid_identifier", f"{name} is not a safe frame identifier"
        )
    return value


def validate_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ExactTwoCameraAdmissionError(
            "invalid_hash", f"{name} must be a lowercase SHA-256"
        )
    return value


def boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ExactTwoCameraAdmissionError(
            "invalid_admission", f"{name} must be a boolean"
        )
    return value


def finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExactTwoCameraAdmissionError(
            "invalid_admission", f"{name} must be a number"
        )
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ExactTwoCameraAdmissionError(
            "invalid_admission", f"{name} must be finite"
        )
    return parsed


def finite_nonnegative(value: object, name: str) -> float:
    parsed = finite_number(value, name)
    if parsed < 0.0:
        raise ExactTwoCameraAdmissionError(
            "invalid_admission", f"{name} must be non-negative"
        )
    return parsed


def nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExactTwoCameraAdmissionError(
            "invalid_admission", f"{name} must be a non-negative integer"
        )
    return value


def string_tuple(value: object, name: str, *, safe_ids: bool) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ExactTwoCameraAdmissionError(
            "invalid_admission", f"{name} must be a tuple"
        )
    for index, item in enumerate(value):
        if safe_ids:
            validate_id(item, f"{name}[{index}]")
        elif not isinstance(item, str) or not item:
            raise ExactTwoCameraAdmissionError(
                "invalid_admission", f"{name}[{index}] must be non-empty text"
            )
    return value


def sorted_unique_ids(value: object, name: str) -> tuple[str, ...]:
    values = string_tuple(value, name, safe_ids=True)
    if values != tuple(sorted(set(values))):
        raise ExactTwoCameraAdmissionError(
            "invalid_admission", f"{name} must be sorted and unique"
        )
    return values


__all__ = [name for name in globals() if not name.startswith("_")]
