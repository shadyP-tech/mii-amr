"""Strict content-hashed persistence for Aufgabe 04 lifecycle manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, TypeVar

from scripts.aufgabe04.artifacts.content_store import (
    ContentStoreError,
    load_content_hashed_json,
    payload_sha256,
    write_content_hashed_json,
)
from scripts.aufgabe04.artifacts.models import (
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    ArtifactManifestError,
    ArtifactReference,
    ExecutionEvidenceManifest,
    MissionPlanManifest,
    SurveyManifest,
    validate_execution_evidence_manifest,
    validate_execution_links,
    validate_mission_plan_manifest,
    validate_mission_plan_links,
    validate_survey_manifest,
)


_HASH_FIELD = "manifest_sha256"
_REFERENCE_FIELDS = frozenset({"artifact_type", "artifact_id", "sha256"})

_SURVEY_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_kind",
        "manifest_id",
        "created_unix_sec",
        "session_id",
        "environment",
        "planning_frame",
        "map_bundle",
        "candidate_snapshot",
        "environment_descriptor",
        "survey_config",
        "calibration_profile",
        "arrival_pose_catalog",
    }
)
_MISSION_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_kind",
        "manifest_id",
        "created_unix_sec",
        "robot_id",
        "parent_survey_manifest",
        "map_bundle",
        "candidate_snapshot",
        "station_identity_registry",
        "arrival_pose_catalog",
        "task_snapshot",
        "planner_config",
        "route_bundle",
        "required_station_order",
        "ordered_candidate_uids",
    }
)
_EXECUTION_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_kind",
        "manifest_id",
        "created_unix_sec",
        "attempt_id",
        "robot_id",
        "parent_mission_plan_manifest",
        "controller_profile",
        "route_certificate",
        "event_log",
        "execution_summary",
        "started_unix_sec",
        "finished_unix_sec",
        "outcome",
    }
)

_Manifest = TypeVar(
    "_Manifest", SurveyManifest, MissionPlanManifest, ExecutionEvidenceManifest
)


def survey_manifest_sha256(manifest: SurveyManifest) -> str:
    return payload_sha256(_survey_payload(manifest))


def mission_plan_manifest_sha256(manifest: MissionPlanManifest) -> str:
    return payload_sha256(_mission_payload(manifest))


def execution_evidence_manifest_sha256(
    manifest: ExecutionEvidenceManifest,
) -> str:
    return payload_sha256(_execution_payload(manifest))


def write_survey_manifest(path: Path, manifest: SurveyManifest) -> str:
    return _write(path, _survey_payload(manifest))


def write_mission_plan_manifest(path: Path, manifest: MissionPlanManifest) -> str:
    return _write(path, _mission_payload(manifest))


def write_execution_evidence_manifest(
    path: Path, manifest: ExecutionEvidenceManifest
) -> str:
    return _write(path, _execution_payload(manifest))


def load_survey_manifest(path: Path) -> SurveyManifest:
    return _load(path, _survey_from_payload)


def load_mission_plan_manifest(
    path: Path, *, parent_survey: SurveyManifest | None = None
) -> MissionPlanManifest:
    manifest = _load(path, _mission_from_payload)
    if parent_survey is not None:
        validate_mission_plan_links(
            manifest,
            parent_survey,
            parent_sha256=survey_manifest_sha256(parent_survey),
        )
    return manifest


def load_execution_evidence_manifest(
    path: Path, *, parent_mission: MissionPlanManifest | None = None
) -> ExecutionEvidenceManifest:
    manifest = _load(path, _execution_from_payload)
    if parent_mission is not None:
        validate_execution_links(
            manifest,
            parent_mission,
            parent_sha256=mission_plan_manifest_sha256(parent_mission),
        )
    return manifest


def _write(path: Path, payload: Mapping[str, object]) -> str:
    try:
        return write_content_hashed_json(path, payload, hash_field=_HASH_FIELD)
    except ContentStoreError as exc:
        raise ArtifactManifestError(exc.code, str(exc)) from exc


def _load(path: Path, parser: Callable[[Mapping[str, object]], _Manifest]) -> _Manifest:
    try:
        payload = load_content_hashed_json(path, hash_field=_HASH_FIELD)
    except ContentStoreError as exc:
        raise ArtifactManifestError(exc.code, str(exc)) from exc
    try:
        return parser(payload)
    except (KeyError, TypeError) as exc:
        raise ArtifactManifestError(
            "artifact_corrupt", "manifest has invalid field types"
        ) from exc


def _survey_payload(manifest: SurveyManifest) -> dict[str, object]:
    validate_survey_manifest(manifest)
    return {
        "schema_version": manifest.schema_version,
        "manifest_kind": "survey",
        "manifest_id": manifest.manifest_id,
        "created_unix_sec": manifest.created_unix_sec,
        "session_id": manifest.session_id,
        "environment": manifest.environment,
        "planning_frame": manifest.planning_frame,
        "map_bundle": _reference_payload(manifest.map_bundle),
        "candidate_snapshot": _reference_payload(manifest.candidate_snapshot),
        "environment_descriptor": _reference_payload(
            manifest.environment_descriptor
        ),
        "survey_config": _reference_payload(manifest.survey_config),
        "calibration_profile": _reference_payload(manifest.calibration_profile),
        "arrival_pose_catalog": _reference_payload(manifest.arrival_pose_catalog),
    }


def _mission_payload(manifest: MissionPlanManifest) -> dict[str, object]:
    validate_mission_plan_manifest(manifest)
    return {
        "schema_version": manifest.schema_version,
        "manifest_kind": "mission_plan",
        "manifest_id": manifest.manifest_id,
        "created_unix_sec": manifest.created_unix_sec,
        "robot_id": manifest.robot_id,
        "parent_survey_manifest": _reference_payload(
            manifest.parent_survey_manifest
        ),
        "map_bundle": _reference_payload(manifest.map_bundle),
        "candidate_snapshot": _reference_payload(manifest.candidate_snapshot),
        "station_identity_registry": _reference_payload(
            manifest.station_identity_registry
        ),
        "arrival_pose_catalog": _reference_payload(manifest.arrival_pose_catalog),
        "task_snapshot": _reference_payload(manifest.task_snapshot),
        "planner_config": _reference_payload(manifest.planner_config),
        "route_bundle": _reference_payload(manifest.route_bundle),
        "required_station_order": list(manifest.required_station_order),
        "ordered_candidate_uids": list(manifest.ordered_candidate_uids),
    }


def _execution_payload(manifest: ExecutionEvidenceManifest) -> dict[str, object]:
    validate_execution_evidence_manifest(manifest)
    return {
        "schema_version": manifest.schema_version,
        "manifest_kind": "execution_evidence",
        "manifest_id": manifest.manifest_id,
        "created_unix_sec": manifest.created_unix_sec,
        "attempt_id": manifest.attempt_id,
        "robot_id": manifest.robot_id,
        "parent_mission_plan_manifest": _reference_payload(
            manifest.parent_mission_plan_manifest
        ),
        "controller_profile": _reference_payload(manifest.controller_profile),
        "route_certificate": _reference_payload(manifest.route_certificate),
        "event_log": _reference_payload(manifest.event_log),
        "execution_summary": _reference_payload(manifest.execution_summary),
        "started_unix_sec": manifest.started_unix_sec,
        "finished_unix_sec": manifest.finished_unix_sec,
        "outcome": manifest.outcome,
    }


def _reference_payload(reference: ArtifactReference) -> dict[str, object]:
    return {
        "artifact_type": reference.artifact_type,
        "artifact_id": reference.artifact_id,
        "sha256": reference.sha256,
    }


def _survey_from_payload(payload: Mapping[str, object]) -> SurveyManifest:
    _require_fields(payload, _SURVEY_FIELDS, "survey manifest")
    _require_kind_and_schema(payload, "survey")
    manifest = SurveyManifest(
        schema_version=_integer(payload["schema_version"], "schema_version"),
        manifest_id=_string(payload["manifest_id"], "manifest_id"),
        created_unix_sec=_number(payload["created_unix_sec"], "created_unix_sec"),
        session_id=_string(payload["session_id"], "session_id"),
        environment=_string(payload["environment"], "environment"),
        planning_frame=_string(payload["planning_frame"], "planning_frame"),
        map_bundle=_reference(payload["map_bundle"], "map_bundle"),
        candidate_snapshot=_reference(
            payload["candidate_snapshot"], "candidate_snapshot"
        ),
        environment_descriptor=_reference(
            payload["environment_descriptor"], "environment_descriptor"
        ),
        survey_config=_reference(payload["survey_config"], "survey_config"),
        calibration_profile=_reference(
            payload["calibration_profile"], "calibration_profile"
        ),
        arrival_pose_catalog=_reference(
            payload["arrival_pose_catalog"], "arrival_pose_catalog"
        ),
    )
    validate_survey_manifest(manifest)
    return manifest


def _mission_from_payload(payload: Mapping[str, object]) -> MissionPlanManifest:
    _require_fields(payload, _MISSION_FIELDS, "mission plan manifest")
    _require_kind_and_schema(payload, "mission_plan")
    manifest = MissionPlanManifest(
        schema_version=_integer(payload["schema_version"], "schema_version"),
        manifest_id=_string(payload["manifest_id"], "manifest_id"),
        created_unix_sec=_number(payload["created_unix_sec"], "created_unix_sec"),
        robot_id=_string(payload["robot_id"], "robot_id"),
        parent_survey_manifest=_reference(
            payload["parent_survey_manifest"], "parent_survey_manifest"
        ),
        map_bundle=_reference(payload["map_bundle"], "map_bundle"),
        candidate_snapshot=_reference(
            payload["candidate_snapshot"], "candidate_snapshot"
        ),
        station_identity_registry=_reference(
            payload["station_identity_registry"], "station_identity_registry"
        ),
        arrival_pose_catalog=_reference(
            payload["arrival_pose_catalog"], "arrival_pose_catalog"
        ),
        task_snapshot=_reference(payload["task_snapshot"], "task_snapshot"),
        planner_config=_reference(payload["planner_config"], "planner_config"),
        route_bundle=_reference(payload["route_bundle"], "route_bundle"),
        required_station_order=_string_tuple(
            payload["required_station_order"], "required_station_order"
        ),
        ordered_candidate_uids=_string_tuple(
            payload["ordered_candidate_uids"], "ordered_candidate_uids"
        ),
    )
    validate_mission_plan_manifest(manifest)
    return manifest


def _execution_from_payload(
    payload: Mapping[str, object]
) -> ExecutionEvidenceManifest:
    _require_fields(payload, _EXECUTION_FIELDS, "execution evidence manifest")
    _require_kind_and_schema(payload, "execution_evidence")
    manifest = ExecutionEvidenceManifest(
        schema_version=_integer(payload["schema_version"], "schema_version"),
        manifest_id=_string(payload["manifest_id"], "manifest_id"),
        created_unix_sec=_number(payload["created_unix_sec"], "created_unix_sec"),
        attempt_id=_string(payload["attempt_id"], "attempt_id"),
        robot_id=_string(payload["robot_id"], "robot_id"),
        parent_mission_plan_manifest=_reference(
            payload["parent_mission_plan_manifest"],
            "parent_mission_plan_manifest",
        ),
        controller_profile=_reference(
            payload["controller_profile"], "controller_profile"
        ),
        route_certificate=_reference(
            payload["route_certificate"], "route_certificate"
        ),
        event_log=_reference(payload["event_log"], "event_log"),
        execution_summary=_reference(
            payload["execution_summary"], "execution_summary"
        ),
        started_unix_sec=_number(payload["started_unix_sec"], "started_unix_sec"),
        finished_unix_sec=_number(
            payload["finished_unix_sec"], "finished_unix_sec"
        ),
        outcome=_string(payload["outcome"], "outcome"),
    )
    validate_execution_evidence_manifest(manifest)
    return manifest


def _reference(value: object, name: str) -> ArtifactReference:
    item = _mapping(value, name)
    _require_fields(item, _REFERENCE_FIELDS, name)
    return ArtifactReference(
        artifact_type=_string(item["artifact_type"], f"{name}.artifact_type"),
        artifact_id=_string(item["artifact_id"], f"{name}.artifact_id"),
        sha256=_string(item["sha256"], f"{name}.sha256"),
    )


def _require_kind_and_schema(payload: Mapping[str, object], kind: str) -> None:
    if _string(payload["manifest_kind"], "manifest_kind") != kind:
        raise ArtifactManifestError(
            "artifact_corrupt", f"manifest_kind must be {kind!r}"
        )
    if _integer(payload["schema_version"], "schema_version") != (
        ARTIFACT_MANIFEST_SCHEMA_VERSION
    ):
        raise ArtifactManifestError(
            "schema_mismatch", f"unsupported manifest schema {payload['schema_version']!r}"
        )


def _require_fields(
    payload: Mapping[str, object], expected: frozenset[str], name: str
) -> None:
    actual = frozenset(payload)
    if actual != expected:
        raise ArtifactManifestError(
            "artifact_corrupt",
            f"{name} fields mismatch; "
            f"missing={sorted(expected - actual)} unknown={sorted(actual - expected)}",
        )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ArtifactManifestError("artifact_corrupt", f"{name} must be an object")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ArtifactManifestError("artifact_corrupt", f"{name} must be a string")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactManifestError("artifact_corrupt", f"{name} must be a number")
    return float(value)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactManifestError("artifact_corrupt", f"{name} must be an integer")
    return value


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ArtifactManifestError("artifact_corrupt", f"{name} must be an array")
    return tuple(_string(item, f"{name}[{index}]") for index, item in enumerate(value))
