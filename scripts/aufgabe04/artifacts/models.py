"""Immutable lifecycle models linking survey, mission plan, and execution."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable


ARTIFACT_MANIFEST_SCHEMA_VERSION = 1

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_TYPE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SAFE_FRAME = re.compile(r"^[A-Za-z][A-Za-z0-9_/.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ArtifactManifestError(ValueError):
    """Lifecycle-manifest validation error with a stable machine code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ArtifactReference:
    artifact_type: str
    artifact_id: str
    sha256: str


@dataclass(frozen=True)
class SurveyManifest:
    schema_version: int
    manifest_id: str
    created_unix_sec: float
    session_id: str
    environment: str
    planning_frame: str
    map_bundle: ArtifactReference
    candidate_snapshot: ArtifactReference
    environment_descriptor: ArtifactReference
    survey_config: ArtifactReference
    calibration_profile: ArtifactReference
    arrival_pose_catalog: ArtifactReference


@dataclass(frozen=True)
class MissionPlanManifest:
    schema_version: int
    manifest_id: str
    created_unix_sec: float
    robot_id: str
    parent_survey_manifest: ArtifactReference
    map_bundle: ArtifactReference
    candidate_snapshot: ArtifactReference
    station_identity_registry: ArtifactReference
    arrival_pose_catalog: ArtifactReference
    task_snapshot: ArtifactReference
    planner_config: ArtifactReference
    route_bundle: ArtifactReference
    required_station_order: tuple[str, ...]
    ordered_candidate_uids: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionEvidenceManifest:
    schema_version: int
    manifest_id: str
    created_unix_sec: float
    attempt_id: str
    robot_id: str
    parent_mission_plan_manifest: ArtifactReference
    controller_profile: ArtifactReference
    route_certificate: ArtifactReference
    event_log: ArtifactReference
    execution_summary: ArtifactReference
    started_unix_sec: float
    finished_unix_sec: float
    outcome: str


def artifact_reference(
    artifact_type: str, artifact_id: str, sha256: str
) -> ArtifactReference:
    reference = ArtifactReference(artifact_type, artifact_id, sha256)
    validate_artifact_reference(reference)
    return reference


def manifest_reference(
    manifest: SurveyManifest | MissionPlanManifest | ExecutionEvidenceManifest,
    sha256: str,
) -> ArtifactReference:
    if isinstance(manifest, SurveyManifest):
        artifact_type = "survey_manifest"
    elif isinstance(manifest, MissionPlanManifest):
        artifact_type = "mission_plan_manifest"
    elif isinstance(manifest, ExecutionEvidenceManifest):
        artifact_type = "execution_evidence_manifest"
    else:
        raise ArtifactManifestError(
            "invalid_reference", "unsupported manifest reference type"
        )
    return artifact_reference(artifact_type, manifest.manifest_id, sha256)


def validate_artifact_reference(
    reference: ArtifactReference, *, expected_type: str | tuple[str, ...] | None = None
) -> None:
    if not isinstance(reference, ArtifactReference):
        raise ArtifactManifestError(
            "invalid_reference", "artifact reference must be an ArtifactReference"
        )
    if not isinstance(reference.artifact_type, str) or not _SAFE_TYPE.fullmatch(
        reference.artifact_type
    ):
        raise ArtifactManifestError(
            "invalid_reference", "artifact_type is not a safe canonical type"
        )
    _validate_id(reference.artifact_id, "artifact_id")
    _validate_sha256(reference.sha256, "artifact sha256")
    if expected_type is not None:
        expected = (expected_type,) if isinstance(expected_type, str) else expected_type
        if reference.artifact_type not in expected:
            raise ArtifactManifestError(
                "invalid_reference",
                f"expected artifact type {expected}, got {reference.artifact_type!r}",
            )


def validate_survey_manifest(manifest: SurveyManifest) -> None:
    _validate_common(
        manifest.schema_version,
        manifest.manifest_id,
        manifest.created_unix_sec,
    )
    _validate_id(manifest.session_id, "session_id")
    if manifest.environment not in {"simulation", "real"}:
        raise ArtifactManifestError(
            "invalid_manifest", "environment must be 'simulation' or 'real'"
        )
    if not isinstance(manifest.planning_frame, str) or not _SAFE_FRAME.fullmatch(
        manifest.planning_frame
    ):
        raise ArtifactManifestError(
            "invalid_manifest", "planning_frame is not a valid frame identifier"
        )
    _validate_typed_references(
        (
            (manifest.map_bundle, "map_bundle"),
            (manifest.candidate_snapshot, "candidate_snapshot"),
            (
                manifest.environment_descriptor,
                ("simulation_world", "physical_site"),
            ),
            (manifest.survey_config, "survey_config"),
            (manifest.calibration_profile, "calibration_profile"),
            (manifest.arrival_pose_catalog, "arrival_pose_catalog"),
        )
    )
    expected_environment_type = (
        "simulation_world" if manifest.environment == "simulation" else "physical_site"
    )
    if manifest.environment_descriptor.artifact_type != expected_environment_type:
        raise ArtifactManifestError(
            "invalid_manifest",
            "environment descriptor type does not match manifest environment",
        )


def validate_mission_plan_manifest(manifest: MissionPlanManifest) -> None:
    _validate_common(
        manifest.schema_version,
        manifest.manifest_id,
        manifest.created_unix_sec,
    )
    _validate_id(manifest.robot_id, "robot_id")
    _validate_typed_references(
        (
            (manifest.parent_survey_manifest, "survey_manifest"),
            (manifest.map_bundle, "map_bundle"),
            (manifest.candidate_snapshot, "candidate_snapshot"),
            (manifest.station_identity_registry, "station_identity_registry"),
            (manifest.arrival_pose_catalog, "arrival_pose_catalog"),
            (manifest.task_snapshot, "task_snapshot"),
            (manifest.planner_config, "planner_config"),
            (manifest.route_bundle, "route_bundle"),
        )
    )
    required = _validated_id_sequence(
        manifest.required_station_order, "required_station_order"
    )
    candidates = _validated_id_sequence(
        manifest.ordered_candidate_uids, "ordered_candidate_uids"
    )
    if not required:
        raise ArtifactManifestError(
            "invalid_manifest", "required_station_order must not be empty"
        )
    if len(required) != len(candidates):
        raise ArtifactManifestError(
            "invalid_manifest",
            "required_station_order and ordered_candidate_uids must have equal length",
        )


def validate_execution_evidence_manifest(
    manifest: ExecutionEvidenceManifest,
) -> None:
    _validate_common(
        manifest.schema_version,
        manifest.manifest_id,
        manifest.created_unix_sec,
    )
    _validate_id(manifest.attempt_id, "attempt_id")
    _validate_id(manifest.robot_id, "robot_id")
    _validate_typed_references(
        (
            (manifest.parent_mission_plan_manifest, "mission_plan_manifest"),
            (manifest.controller_profile, "controller_profile"),
            (manifest.route_certificate, "route_certificate"),
            (manifest.event_log, "event_log"),
            (manifest.execution_summary, "execution_summary"),
        )
    )
    started = _finite_nonnegative(manifest.started_unix_sec, "started_unix_sec")
    finished = _finite_nonnegative(manifest.finished_unix_sec, "finished_unix_sec")
    created = _finite_nonnegative(manifest.created_unix_sec, "created_unix_sec")
    if finished < started:
        raise ArtifactManifestError(
            "invalid_manifest", "finished_unix_sec precedes started_unix_sec"
        )
    if created < finished:
        raise ArtifactManifestError(
            "invalid_manifest", "created_unix_sec precedes finished_unix_sec"
        )
    if manifest.outcome not in {"completed", "failed", "aborted"}:
        raise ArtifactManifestError(
            "invalid_manifest", "outcome must be completed, failed, or aborted"
        )


def validate_mission_plan_links(
    manifest: MissionPlanManifest,
    parent_survey: SurveyManifest,
    *,
    parent_sha256: str,
) -> None:
    """Verify the parent link and duplicated survey outputs fail closed."""

    validate_mission_plan_manifest(manifest)
    validate_survey_manifest(parent_survey)
    expected_parent = manifest_reference(parent_survey, parent_sha256)
    if manifest.parent_survey_manifest != expected_parent:
        raise ArtifactManifestError(
            "provenance_mismatch", "mission plan references another survey manifest"
        )
    if manifest.created_unix_sec < parent_survey.created_unix_sec:
        raise ArtifactManifestError(
            "provenance_mismatch", "mission plan predates its parent survey"
        )
    for name, actual, expected in (
        ("map_bundle", manifest.map_bundle, parent_survey.map_bundle),
        (
            "candidate_snapshot",
            manifest.candidate_snapshot,
            parent_survey.candidate_snapshot,
        ),
        (
            "arrival_pose_catalog",
            manifest.arrival_pose_catalog,
            parent_survey.arrival_pose_catalog,
        ),
    ):
        if actual != expected:
            raise ArtifactManifestError(
                "provenance_mismatch",
                f"mission plan {name} differs from its parent survey",
            )


def validate_execution_links(
    manifest: ExecutionEvidenceManifest,
    parent_mission: MissionPlanManifest,
    *,
    parent_sha256: str,
) -> None:
    """Verify execution evidence belongs to the exact planned robot mission."""

    validate_execution_evidence_manifest(manifest)
    validate_mission_plan_manifest(parent_mission)
    expected_parent = manifest_reference(parent_mission, parent_sha256)
    if manifest.parent_mission_plan_manifest != expected_parent:
        raise ArtifactManifestError(
            "provenance_mismatch",
            "execution evidence references another mission plan manifest",
        )
    if manifest.robot_id != parent_mission.robot_id:
        raise ArtifactManifestError(
            "provenance_mismatch",
            "execution robot_id differs from its parent mission plan",
        )
    if manifest.started_unix_sec < parent_mission.created_unix_sec:
        raise ArtifactManifestError(
            "provenance_mismatch", "execution started before its mission plan existed"
        )


def _validate_typed_references(
    values: Iterable[
        tuple[ArtifactReference, str | tuple[str, ...]]
    ],
) -> None:
    seen = set()
    for reference, expected_type in values:
        validate_artifact_reference(reference, expected_type=expected_type)
        identity = (reference.artifact_type, reference.artifact_id)
        if identity in seen:
            raise ArtifactManifestError(
                "invalid_manifest", f"duplicate artifact reference {identity!r}"
            )
        seen.add(identity)


def _validate_common(schema_version: int, manifest_id: str, created: float) -> None:
    if type(schema_version) is not int or (
        schema_version != ARTIFACT_MANIFEST_SCHEMA_VERSION
    ):
        raise ArtifactManifestError(
            "schema_mismatch", f"unsupported manifest schema {schema_version!r}"
        )
    _validate_id(manifest_id, "manifest_id")
    _finite_nonnegative(created, "created_unix_sec")


def _validated_id_sequence(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ArtifactManifestError("invalid_manifest", f"{name} must be a tuple")
    for index, value in enumerate(values):
        _validate_id(value, f"{name}[{index}]")
    return values


def _validate_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ArtifactManifestError(
            "invalid_manifest", f"{name} is not a safe identifier"
        )
    return value


def _validate_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ArtifactManifestError(
            "invalid_reference", f"{name} must be a lowercase SHA-256"
        )
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactManifestError("invalid_manifest", f"{name} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ArtifactManifestError(
            "invalid_manifest", f"{name} must be finite and non-negative"
        )
    return parsed
