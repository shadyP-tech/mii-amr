"""Strict, atomic persistence for the surveyed stand arrival-pose catalog.

The catalog is an immutable value.  Update functions return a new revision and
never silently overwrite a different estimate.  Exact retries return the same
object, while identity or geometry disagreements raise a stable coded error.

The JSON ``catalog_sha256`` covers the canonical root payload excluding the
hash field itself.  Writers use fsync plus atomic ``os.replace`` so readers see
either the previous complete snapshot or the next complete snapshot.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.aufgabe04.stations.arrival_pose_models import (
    ARRIVAL_POSE_CATALOG_SCHEMA_VERSION,
    ArrivalPoseCatalog,
    ArrivalPoseRecord,
    ArrivalPoseValidation,
    AxisEstimate,
    CandidateRejection,
    CatalogPose2D,
    CatalogProvenance,
    FaceSelection,
    StandEstimate,
)


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_FRAME = re.compile(r"^[A-Za-z][A-Za-z0-9_/.-]{0,127}$")
_SAFE_SOURCE = re.compile(r"^[A-Za-z0-9/][A-Za-z0-9_.:/+@-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GEOMETRY_TOLERANCE_M = 1.0e-6
_ANGLE_TOLERANCE_RAD = 1.0e-6

_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "catalog_id",
        "provenance",
        "revision",
        "frozen",
        "created_unix_sec",
        "updated_unix_sec",
        "frozen_unix_sec",
        "expected_candidate_uids",
        "records",
        "rejections",
        "catalog_sha256",
    }
)


class ArrivalPoseCatalogError(ValueError):
    """Catalog validation or update error with a stable machine code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def new_arrival_pose_catalog(
    *,
    catalog_id: str,
    provenance: CatalogProvenance,
    expected_candidate_uids: Iterable[str],
    created_unix_sec: float,
) -> ArrivalPoseCatalog:
    """Create revision zero of an open catalog."""

    expected = _normalized_candidate_uids(expected_candidate_uids)
    catalog = ArrivalPoseCatalog(
        schema_version=ARRIVAL_POSE_CATALOG_SCHEMA_VERSION,
        catalog_id=catalog_id,
        provenance=provenance,
        revision=0,
        frozen=False,
        created_unix_sec=created_unix_sec,
        updated_unix_sec=created_unix_sec,
        frozen_unix_sec=None,
        expected_candidate_uids=expected,
        records=(),
        rejections=(),
    )
    validate_arrival_pose_catalog(catalog)
    return catalog


def set_expected_candidate_uids(
    catalog: ArrivalPoseCatalog,
    candidate_uids: Iterable[str],
    *,
    updated_unix_sec: float,
) -> ArrivalPoseCatalog:
    """Seal or revise the expected candidate set before catalog freeze."""

    validate_arrival_pose_catalog(catalog)
    expected = _normalized_candidate_uids(candidate_uids)
    if expected == catalog.expected_candidate_uids:
        return catalog
    _require_mutable(catalog)
    if not catalog.resolved_candidate_uids.issubset(expected):
        missing = sorted(catalog.resolved_candidate_uids.difference(expected))
        raise ArrivalPoseCatalogError(
            "expected_candidates_conflict",
            f"expected candidates omit existing resolutions: {missing}",
        )
    timestamp = _next_timestamp(catalog, updated_unix_sec)
    updated = replace(
        catalog,
        revision=catalog.revision + 1,
        updated_unix_sec=timestamp,
        expected_candidate_uids=expected,
    )
    validate_arrival_pose_catalog(updated)
    return updated


def upsert_arrival_pose(
    catalog: ArrivalPoseCatalog,
    record: ArrivalPoseRecord,
    *,
    updated_unix_sec: float,
) -> ArrivalPoseCatalog:
    """Insert one committed record, or no-op an exact retry.

    A differing record for the same stable candidate UID is a conflict rather
    than an update.  This deliberately prevents a late observer process from
    silently replacing a pose already used to build a global mission route.
    """

    validate_arrival_pose_catalog(catalog)
    validate_arrival_pose_record(record, provenance=catalog.provenance)
    _require_expected(catalog, record.candidate_uid)

    existing = catalog.record_for(record.candidate_uid)
    if existing is not None:
        if arrival_pose_record_sha256(existing) == arrival_pose_record_sha256(record):
            return catalog
        raise ArrivalPoseCatalogError(
            "candidate_conflict",
            f"candidate {record.candidate_uid!r} already has a different arrival pose",
        )
    if record.candidate_uid in catalog.rejected_candidate_uids:
        raise ArrivalPoseCatalogError(
            "candidate_conflict",
            f"candidate {record.candidate_uid!r} is already rejected",
        )

    _require_mutable(catalog)
    _check_observation_identity_conflict(
        catalog,
        candidate_uid=record.candidate_uid,
        source_observation_ids=record.source_observation_ids,
    )
    timestamp = _next_timestamp(catalog, updated_unix_sec)
    records = tuple(sorted((*catalog.records, record), key=lambda item: item.candidate_uid))
    updated = replace(
        catalog,
        revision=catalog.revision + 1,
        updated_unix_sec=timestamp,
        records=records,
    )
    validate_arrival_pose_catalog(updated)
    return updated


def upsert_candidate_rejection(
    catalog: ArrivalPoseCatalog,
    rejection: CandidateRejection,
    *,
    updated_unix_sec: float,
) -> ArrivalPoseCatalog:
    """Store one explicit terminal rejection, or no-op an exact retry."""

    validate_arrival_pose_catalog(catalog)
    validate_candidate_rejection(rejection)
    _require_expected(catalog, rejection.candidate_uid)
    existing = next(
        (
            item
            for item in catalog.rejections
            if item.candidate_uid == rejection.candidate_uid
        ),
        None,
    )
    if existing is not None:
        if _candidate_rejection_payload(existing) == _candidate_rejection_payload(rejection):
            return catalog
        raise ArrivalPoseCatalogError(
            "candidate_conflict",
            f"candidate {rejection.candidate_uid!r} already has a different rejection",
        )
    if catalog.record_for(rejection.candidate_uid) is not None:
        raise ArrivalPoseCatalogError(
            "candidate_conflict",
            f"candidate {rejection.candidate_uid!r} already has an arrival pose",
        )

    _require_mutable(catalog)
    _check_observation_identity_conflict(
        catalog,
        candidate_uid=rejection.candidate_uid,
        source_observation_ids=rejection.source_observation_ids,
    )
    timestamp = _next_timestamp(catalog, updated_unix_sec)
    rejections = tuple(
        sorted((*catalog.rejections, rejection), key=lambda item: item.candidate_uid)
    )
    updated = replace(
        catalog,
        revision=catalog.revision + 1,
        updated_unix_sec=timestamp,
        rejections=rejections,
    )
    validate_arrival_pose_catalog(updated)
    return updated


def freeze_arrival_pose_catalog(
    catalog: ArrivalPoseCatalog,
    *,
    frozen_unix_sec: float,
) -> ArrivalPoseCatalog:
    """Freeze a complete catalog for deterministic route planning."""

    validate_arrival_pose_catalog(catalog)
    if catalog.frozen:
        return catalog
    if not catalog.complete:
        missing = sorted(
            set(catalog.expected_candidate_uids).difference(catalog.resolved_candidate_uids)
        )
        raise ArrivalPoseCatalogError(
            "catalog_incomplete",
            f"cannot freeze catalog; unresolved candidates: {missing}",
        )
    timestamp = _next_timestamp(catalog, frozen_unix_sec)
    frozen = replace(
        catalog,
        revision=catalog.revision + 1,
        frozen=True,
        updated_unix_sec=timestamp,
        frozen_unix_sec=timestamp,
    )
    validate_arrival_pose_catalog(frozen)
    return frozen


def validate_arrival_pose_catalog(
    catalog: ArrivalPoseCatalog,
    *,
    required_provenance: CatalogProvenance | None = None,
) -> None:
    """Validate a complete in-memory catalog snapshot."""

    if catalog.schema_version != ARRIVAL_POSE_CATALOG_SCHEMA_VERSION:
        raise ArrivalPoseCatalogError(
            "schema_mismatch",
            f"unsupported arrival-pose catalog schema {catalog.schema_version!r}",
        )
    _validate_safe_id(catalog.catalog_id, "catalog_id")
    validate_catalog_provenance(catalog.provenance)
    if required_provenance is not None:
        validate_catalog_provenance(required_provenance)
        if catalog.provenance != required_provenance:
            raise ArrivalPoseCatalogError(
                "provenance_mismatch",
                "arrival-pose catalog map/frame/world provenance does not match",
            )
    _validate_integer(catalog.revision, "revision", minimum=0)
    created = _finite_nonnegative(catalog.created_unix_sec, "created_unix_sec")
    updated = _finite_nonnegative(catalog.updated_unix_sec, "updated_unix_sec")
    if updated < created:
        raise ArrivalPoseCatalogError(
            "invalid_catalog", "updated_unix_sec precedes created_unix_sec"
        )
    if type(catalog.frozen) is not bool:
        raise ArrivalPoseCatalogError("invalid_catalog", "frozen must be boolean")
    if catalog.frozen:
        if catalog.frozen_unix_sec is None:
            raise ArrivalPoseCatalogError(
                "invalid_catalog", "frozen catalog lacks frozen_unix_sec"
            )
        frozen_at = _finite_nonnegative(catalog.frozen_unix_sec, "frozen_unix_sec")
        if frozen_at != updated:
            raise ArrivalPoseCatalogError(
                "invalid_catalog", "frozen_unix_sec must equal updated_unix_sec"
            )
    elif catalog.frozen_unix_sec is not None:
        raise ArrivalPoseCatalogError(
            "invalid_catalog", "open catalog must not have frozen_unix_sec"
        )

    expected = _normalized_candidate_uids(catalog.expected_candidate_uids)
    if expected != catalog.expected_candidate_uids:
        raise ArrivalPoseCatalogError(
            "invalid_catalog", "expected_candidate_uids must be sorted and unique"
        )

    record_ids: set[str] = set()
    observation_owners: dict[str, str] = {}
    for record in catalog.records:
        validate_arrival_pose_record(record, provenance=catalog.provenance)
        if record.candidate_uid in record_ids:
            raise ArrivalPoseCatalogError(
                "invalid_catalog", f"duplicate candidate record {record.candidate_uid!r}"
            )
        record_ids.add(record.candidate_uid)
        _claim_observations(observation_owners, record.candidate_uid, record.source_observation_ids)
    if tuple(sorted(catalog.records, key=lambda item: item.candidate_uid)) != catalog.records:
        raise ArrivalPoseCatalogError(
            "invalid_catalog", "records must be sorted by candidate_uid"
        )

    rejection_ids: set[str] = set()
    for rejection in catalog.rejections:
        validate_candidate_rejection(rejection)
        if rejection.candidate_uid in rejection_ids:
            raise ArrivalPoseCatalogError(
                "invalid_catalog", f"duplicate candidate rejection {rejection.candidate_uid!r}"
            )
        rejection_ids.add(rejection.candidate_uid)
        _claim_observations(
            observation_owners,
            rejection.candidate_uid,
            rejection.source_observation_ids,
        )
    if tuple(sorted(catalog.rejections, key=lambda item: item.candidate_uid)) != catalog.rejections:
        raise ArrivalPoseCatalogError(
            "invalid_catalog", "rejections must be sorted by candidate_uid"
        )
    overlap = record_ids.intersection(rejection_ids)
    if overlap:
        raise ArrivalPoseCatalogError(
            "invalid_catalog",
            f"candidates are both ready and rejected: {sorted(overlap)}",
        )
    resolved = record_ids | rejection_ids
    if expected and not resolved.issubset(expected):
        raise ArrivalPoseCatalogError(
            "invalid_catalog",
            f"catalog contains unexpected candidates: {sorted(resolved.difference(expected))}",
        )
    if catalog.frozen and not catalog.complete:
        raise ArrivalPoseCatalogError(
            "invalid_catalog", "frozen catalog must resolve every expected candidate"
        )


def validate_catalog_provenance(provenance: CatalogProvenance) -> None:
    _validate_frame(provenance.planning_frame, "provenance.planning_frame")
    _validate_sha256(provenance.map_yaml_sha256, "provenance.map_yaml_sha256")
    _validate_safe_id(provenance.world_id, "provenance.world_id")
    _validate_sha256(provenance.world_sha256, "provenance.world_sha256")
    _validate_safe_id(provenance.session_id, "provenance.session_id")
    if provenance.environment not in {"simulation", "real"}:
        raise ArrivalPoseCatalogError(
            "invalid_catalog", "provenance.environment must be 'simulation' or 'real'"
        )


def validate_arrival_pose_record(
    record: ArrivalPoseRecord,
    *,
    provenance: CatalogProvenance | None = None,
) -> None:
    """Validate record shape and the perpendicular arrival geometry."""

    _validate_safe_id(record.candidate_uid, "record.candidate_uid")
    _validate_safe_id(record.stand_id, "record.stand_id")
    stand_x = _finite_number(record.stand.x_m, "record.stand.x_m")
    stand_y = _finite_number(record.stand.y_m, "record.stand.y_m")
    if _finite_number(record.stand.radius_m, "record.stand.radius_m") <= 0.0:
        raise ArrivalPoseCatalogError("invalid_record", "record.stand.radius_m must be positive")
    _finite_nonnegative(record.stand.uncertainty_m, "record.stand.uncertainty_m")

    axis = _finite_number(record.axis.axis_rad, "record.axis.axis_rad")
    confidence = _finite_number(record.axis.confidence, "record.axis.confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ArrivalPoseCatalogError(
            "invalid_record", "record.axis.confidence must be in [0, 1]"
        )
    _validate_integer(record.axis.sample_count, "record.axis.sample_count", minimum=1)
    _validate_source(record.axis.estimator, "record.axis.estimator")
    _finite_nonnegative(record.axis.observation_unix_sec, "record.axis.observation_unix_sec")

    _validate_safe_id(record.face.face_id, "record.face.face_id")
    normal = _finite_number(
        record.face.outward_normal_rad, "record.face.outward_normal_rad"
    )
    for field, value in (
        ("identity_resolved", record.face.identity_resolved),
        ("evidence_hard", record.face.evidence_hard),
        ("evidence_valid", record.face.evidence_valid),
    ):
        if type(value) is not bool:
            raise ArrivalPoseCatalogError(
                "invalid_record", f"record.face.{field} must be boolean"
            )
    _validate_safe_id(record.face.evidence_kind, "record.face.evidence_kind")
    evidence_confidence = _finite_number(
        record.face.evidence_confidence, "record.face.evidence_confidence"
    )
    if not 0.0 <= evidence_confidence <= 1.0:
        raise ArrivalPoseCatalogError(
            "invalid_record", "record.face.evidence_confidence must be in [0, 1]"
        )
    _validate_source(record.face.evidence_provenance, "record.face.evidence_provenance")
    if not record.face.evidence_valid:
        raise ArrivalPoseCatalogError(
            "invalid_record", "selected arrival face must have valid evidence"
        )

    if abs(_axial_normal_error(axis, normal) - math.pi / 2.0) > _ANGLE_TOLERANCE_RAD:
        raise ArrivalPoseCatalogError(
            "geometry_mismatch", "selected face normal must be perpendicular to stand axis"
        )

    _validate_pose(record.arrival_pose, "record.arrival_pose")
    _validate_pose(record.corridor_entry_pose, "record.corridor_entry_pose")
    standoff = _finite_positive(record.standoff_m, "record.standoff_m")
    corridor_length = _finite_positive(
        record.corridor_length_m, "record.corridor_length_m"
    )
    _validate_ray_pose(
        pose=record.arrival_pose,
        stand_x=stand_x,
        stand_y=stand_y,
        normal_rad=normal,
        expected_distance_m=standoff,
        name="record.arrival_pose",
    )
    _validate_ray_pose(
        pose=record.corridor_entry_pose,
        stand_x=stand_x,
        stand_y=stand_y,
        normal_rad=normal,
        expected_distance_m=standoff + corridor_length,
        name="record.corridor_entry_pose",
    )

    validation = record.validation
    for field, value in (
        ("target_in_bounds", validation.target_in_bounds),
        ("target_collision_free", validation.target_collision_free),
        ("corridor_collision_free", validation.corridor_collision_free),
    ):
        if value is not True:
            raise ArrivalPoseCatalogError(
                "invalid_record", f"record.validation.{field} must be true"
            )
    _validate_sha256(
        validation.validated_map_yaml_sha256,
        "record.validation.validated_map_yaml_sha256",
    )
    if provenance is not None and (
        validation.validated_map_yaml_sha256 != provenance.map_yaml_sha256
    ):
        raise ArrivalPoseCatalogError(
            "provenance_mismatch", "record was validated against another occupancy map"
        )
    _finite_nonnegative(validation.validated_unix_sec, "record.validation.validated_unix_sec")

    observation_ids = _normalized_observation_ids(record.source_observation_ids)
    if observation_ids != record.source_observation_ids:
        raise ArrivalPoseCatalogError(
            "invalid_record", "source_observation_ids must be sorted and unique"
        )
    if not observation_ids:
        raise ArrivalPoseCatalogError(
            "invalid_record", "source_observation_ids must not be empty"
        )
    _finite_nonnegative(record.sensor_stamp_sec, "record.sensor_stamp_sec")
    _validate_source(record.source, "record.source")


def validate_candidate_rejection(rejection: CandidateRejection) -> None:
    _validate_safe_id(rejection.candidate_uid, "rejection.candidate_uid")
    if not isinstance(rejection.reason, str):
        raise ArrivalPoseCatalogError(
            "invalid_rejection", "rejection.reason must be a string"
        )
    reason = rejection.reason.strip()
    if not reason or len(reason) > 512:
        raise ArrivalPoseCatalogError(
            "invalid_rejection", "rejection.reason must contain 1..512 characters"
        )
    observation_ids = _normalized_observation_ids(rejection.source_observation_ids)
    if observation_ids != rejection.source_observation_ids:
        raise ArrivalPoseCatalogError(
            "invalid_rejection", "source_observation_ids must be sorted and unique"
        )
    _finite_nonnegative(rejection.rejected_unix_sec, "rejection.rejected_unix_sec")


def arrival_pose_record_sha256(record: ArrivalPoseRecord) -> str:
    validate_arrival_pose_record(record)
    return _sha256_payload(_arrival_pose_record_payload(record))


def arrival_pose_catalog_sha256(catalog: ArrivalPoseCatalog) -> str:
    validate_arrival_pose_catalog(catalog)
    return _sha256_payload(_catalog_payload_without_hash(catalog))


def arrival_pose_catalog_payload(catalog: ArrivalPoseCatalog) -> dict[str, object]:
    """Return the strict JSON payload including its content hash."""

    payload = _catalog_payload_without_hash(catalog)
    payload["catalog_sha256"] = _sha256_payload(payload)
    return payload


def write_arrival_pose_catalog(path: Path, catalog: ArrivalPoseCatalog) -> str:
    """Atomically publish a validated catalog and return its content hash."""

    payload = arrival_pose_catalog_payload(catalog)
    data = _pretty_json_bytes(payload)
    _atomic_replace(Path(path), data)
    return str(payload["catalog_sha256"])


def load_arrival_pose_catalog(
    path: Path,
    *,
    required_provenance: CatalogProvenance | None = None,
) -> ArrivalPoseCatalog:
    """Load, strictly parse, hash-check, and validate one snapshot."""

    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ArrivalPoseCatalogError(
            "catalog_unavailable", f"arrival-pose catalog is unavailable: {path}"
        ) from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArrivalPoseCatalogError(
            "catalog_corrupt", f"invalid arrival-pose catalog JSON: {path}"
        ) from exc
    root = _require_mapping(payload, "catalog")
    _require_exact_fields(root, _ROOT_FIELDS, "catalog")
    stored_hash = _require_string(root["catalog_sha256"], "catalog.catalog_sha256")
    _validate_sha256(stored_hash, "catalog.catalog_sha256")
    unhashed = dict(root)
    del unhashed["catalog_sha256"]
    actual_hash = _sha256_payload(unhashed)
    if stored_hash != actual_hash:
        raise ArrivalPoseCatalogError(
            "hash_mismatch",
            f"arrival-pose catalog hash mismatch: expected {stored_hash}, got {actual_hash}",
        )
    catalog = _catalog_from_payload(unhashed)
    validate_arrival_pose_catalog(catalog, required_provenance=required_provenance)
    return catalog


def _catalog_payload_without_hash(catalog: ArrivalPoseCatalog) -> dict[str, object]:
    validate_arrival_pose_catalog(catalog)
    return {
        "schema_version": catalog.schema_version,
        "catalog_id": catalog.catalog_id,
        "provenance": _provenance_payload(catalog.provenance),
        "revision": catalog.revision,
        "frozen": catalog.frozen,
        "created_unix_sec": catalog.created_unix_sec,
        "updated_unix_sec": catalog.updated_unix_sec,
        "frozen_unix_sec": catalog.frozen_unix_sec,
        "expected_candidate_uids": list(catalog.expected_candidate_uids),
        "records": [_arrival_pose_record_payload(record) for record in catalog.records],
        "rejections": [
            _candidate_rejection_payload(rejection) for rejection in catalog.rejections
        ],
    }


def _provenance_payload(provenance: CatalogProvenance) -> dict[str, object]:
    return {
        "planning_frame": provenance.planning_frame,
        "map_yaml_sha256": provenance.map_yaml_sha256,
        "world_id": provenance.world_id,
        "world_sha256": provenance.world_sha256,
        "session_id": provenance.session_id,
        "environment": provenance.environment,
    }


def _arrival_pose_record_payload(record: ArrivalPoseRecord) -> dict[str, object]:
    return {
        "candidate_uid": record.candidate_uid,
        "stand_id": record.stand_id,
        "stand": {
            "x_m": record.stand.x_m,
            "y_m": record.stand.y_m,
            "radius_m": record.stand.radius_m,
            "uncertainty_m": record.stand.uncertainty_m,
        },
        "axis": {
            "axis_rad": record.axis.axis_rad,
            "confidence": record.axis.confidence,
            "sample_count": record.axis.sample_count,
            "estimator": record.axis.estimator,
            "observation_unix_sec": record.axis.observation_unix_sec,
        },
        "face": {
            "face_id": record.face.face_id,
            "outward_normal_rad": record.face.outward_normal_rad,
            "identity_resolved": record.face.identity_resolved,
            "evidence_kind": record.face.evidence_kind,
            "evidence_confidence": record.face.evidence_confidence,
            "evidence_hard": record.face.evidence_hard,
            "evidence_valid": record.face.evidence_valid,
            "evidence_provenance": record.face.evidence_provenance,
        },
        "arrival_pose": _pose_payload(record.arrival_pose),
        "corridor_entry_pose": _pose_payload(record.corridor_entry_pose),
        "standoff_m": record.standoff_m,
        "corridor_length_m": record.corridor_length_m,
        "validation": {
            "target_in_bounds": record.validation.target_in_bounds,
            "target_collision_free": record.validation.target_collision_free,
            "corridor_collision_free": record.validation.corridor_collision_free,
            "validated_map_yaml_sha256": record.validation.validated_map_yaml_sha256,
            "validated_unix_sec": record.validation.validated_unix_sec,
        },
        "source_observation_ids": list(record.source_observation_ids),
        "sensor_stamp_sec": record.sensor_stamp_sec,
        "source": record.source,
    }


def _candidate_rejection_payload(rejection: CandidateRejection) -> dict[str, object]:
    return {
        "candidate_uid": rejection.candidate_uid,
        "reason": rejection.reason,
        "source_observation_ids": list(rejection.source_observation_ids),
        "rejected_unix_sec": rejection.rejected_unix_sec,
    }


def _pose_payload(pose: CatalogPose2D) -> dict[str, object]:
    return {"x_m": pose.x_m, "y_m": pose.y_m, "yaw_rad": pose.yaw_rad}


def _catalog_from_payload(payload: Mapping[str, object]) -> ArrivalPoseCatalog:
    if _parse_integer(payload["schema_version"], "schema_version", minimum=1) != (
        ARRIVAL_POSE_CATALOG_SCHEMA_VERSION
    ):
        raise ArrivalPoseCatalogError(
            "schema_mismatch",
            f"unsupported arrival-pose catalog schema {payload['schema_version']!r}",
        )
    expected_payload = _require_list(
        payload["expected_candidate_uids"], "expected_candidate_uids"
    )
    records_payload = _require_list(payload["records"], "records")
    rejections_payload = _require_list(payload["rejections"], "rejections")
    frozen_unix_sec = payload["frozen_unix_sec"]
    return ArrivalPoseCatalog(
        schema_version=ARRIVAL_POSE_CATALOG_SCHEMA_VERSION,
        catalog_id=_require_string(payload["catalog_id"], "catalog_id"),
        provenance=_provenance_from_payload(payload["provenance"]),
        revision=_parse_integer(payload["revision"], "revision", minimum=0),
        frozen=_require_bool(payload["frozen"], "frozen"),
        created_unix_sec=_parse_number(payload["created_unix_sec"], "created_unix_sec"),
        updated_unix_sec=_parse_number(payload["updated_unix_sec"], "updated_unix_sec"),
        frozen_unix_sec=(
            None
            if frozen_unix_sec is None
            else _parse_number(frozen_unix_sec, "frozen_unix_sec")
        ),
        expected_candidate_uids=tuple(
            _require_string(value, f"expected_candidate_uids[{index}]")
            for index, value in enumerate(expected_payload)
        ),
        records=tuple(
            _arrival_pose_record_from_payload(value, index)
            for index, value in enumerate(records_payload)
        ),
        rejections=tuple(
            _candidate_rejection_from_payload(value, index)
            for index, value in enumerate(rejections_payload)
        ),
    )


def _provenance_from_payload(payload: object) -> CatalogProvenance:
    item = _require_mapping(payload, "provenance")
    fields = frozenset(
        {
            "planning_frame",
            "map_yaml_sha256",
            "world_id",
            "world_sha256",
            "session_id",
            "environment",
        }
    )
    _require_exact_fields(item, fields, "provenance")
    return CatalogProvenance(
        planning_frame=_require_string(item["planning_frame"], "provenance.planning_frame"),
        map_yaml_sha256=_require_string(
            item["map_yaml_sha256"], "provenance.map_yaml_sha256"
        ),
        world_id=_require_string(item["world_id"], "provenance.world_id"),
        world_sha256=_require_string(item["world_sha256"], "provenance.world_sha256"),
        session_id=_require_string(item["session_id"], "provenance.session_id"),
        environment=_require_string(item["environment"], "provenance.environment"),
    )


def _arrival_pose_record_from_payload(payload: object, index: int) -> ArrivalPoseRecord:
    name = f"records[{index}]"
    item = _require_mapping(payload, name)
    _require_exact_fields(
        item,
        frozenset(
            {
                "candidate_uid",
                "stand_id",
                "stand",
                "axis",
                "face",
                "arrival_pose",
                "corridor_entry_pose",
                "standoff_m",
                "corridor_length_m",
                "validation",
                "source_observation_ids",
                "sensor_stamp_sec",
                "source",
            }
        ),
        name,
    )
    stand = _require_mapping(item["stand"], f"{name}.stand")
    _require_exact_fields(
        stand, frozenset({"x_m", "y_m", "radius_m", "uncertainty_m"}), f"{name}.stand"
    )
    axis = _require_mapping(item["axis"], f"{name}.axis")
    _require_exact_fields(
        axis,
        frozenset(
            {"axis_rad", "confidence", "sample_count", "estimator", "observation_unix_sec"}
        ),
        f"{name}.axis",
    )
    face = _require_mapping(item["face"], f"{name}.face")
    _require_exact_fields(
        face,
        frozenset(
            {
                "face_id",
                "outward_normal_rad",
                "identity_resolved",
                "evidence_kind",
                "evidence_confidence",
                "evidence_hard",
                "evidence_valid",
                "evidence_provenance",
            }
        ),
        f"{name}.face",
    )
    validation = _require_mapping(item["validation"], f"{name}.validation")
    _require_exact_fields(
        validation,
        frozenset(
            {
                "target_in_bounds",
                "target_collision_free",
                "corridor_collision_free",
                "validated_map_yaml_sha256",
                "validated_unix_sec",
            }
        ),
        f"{name}.validation",
    )
    observation_ids = _require_list(
        item["source_observation_ids"], f"{name}.source_observation_ids"
    )
    return ArrivalPoseRecord(
        candidate_uid=_require_string(item["candidate_uid"], f"{name}.candidate_uid"),
        stand_id=_require_string(item["stand_id"], f"{name}.stand_id"),
        stand=StandEstimate(
            x_m=_parse_number(stand["x_m"], f"{name}.stand.x_m"),
            y_m=_parse_number(stand["y_m"], f"{name}.stand.y_m"),
            radius_m=_parse_number(stand["radius_m"], f"{name}.stand.radius_m"),
            uncertainty_m=_parse_number(
                stand["uncertainty_m"], f"{name}.stand.uncertainty_m"
            ),
        ),
        axis=AxisEstimate(
            axis_rad=_parse_number(axis["axis_rad"], f"{name}.axis.axis_rad"),
            confidence=_parse_number(axis["confidence"], f"{name}.axis.confidence"),
            sample_count=_parse_integer(
                axis["sample_count"], f"{name}.axis.sample_count", minimum=1
            ),
            estimator=_require_string(axis["estimator"], f"{name}.axis.estimator"),
            observation_unix_sec=_parse_number(
                axis["observation_unix_sec"], f"{name}.axis.observation_unix_sec"
            ),
        ),
        face=FaceSelection(
            face_id=_require_string(face["face_id"], f"{name}.face.face_id"),
            outward_normal_rad=_parse_number(
                face["outward_normal_rad"], f"{name}.face.outward_normal_rad"
            ),
            identity_resolved=_require_bool(
                face["identity_resolved"], f"{name}.face.identity_resolved"
            ),
            evidence_kind=_require_string(
                face["evidence_kind"], f"{name}.face.evidence_kind"
            ),
            evidence_confidence=_parse_number(
                face["evidence_confidence"], f"{name}.face.evidence_confidence"
            ),
            evidence_hard=_require_bool(
                face["evidence_hard"], f"{name}.face.evidence_hard"
            ),
            evidence_valid=_require_bool(
                face["evidence_valid"], f"{name}.face.evidence_valid"
            ),
            evidence_provenance=_require_string(
                face["evidence_provenance"], f"{name}.face.evidence_provenance"
            ),
        ),
        arrival_pose=_pose_from_payload(item["arrival_pose"], f"{name}.arrival_pose"),
        corridor_entry_pose=_pose_from_payload(
            item["corridor_entry_pose"], f"{name}.corridor_entry_pose"
        ),
        standoff_m=_parse_number(item["standoff_m"], f"{name}.standoff_m"),
        corridor_length_m=_parse_number(
            item["corridor_length_m"], f"{name}.corridor_length_m"
        ),
        validation=ArrivalPoseValidation(
            target_in_bounds=_require_bool(
                validation["target_in_bounds"], f"{name}.validation.target_in_bounds"
            ),
            target_collision_free=_require_bool(
                validation["target_collision_free"],
                f"{name}.validation.target_collision_free",
            ),
            corridor_collision_free=_require_bool(
                validation["corridor_collision_free"],
                f"{name}.validation.corridor_collision_free",
            ),
            validated_map_yaml_sha256=_require_string(
                validation["validated_map_yaml_sha256"],
                f"{name}.validation.validated_map_yaml_sha256",
            ),
            validated_unix_sec=_parse_number(
                validation["validated_unix_sec"],
                f"{name}.validation.validated_unix_sec",
            ),
        ),
        source_observation_ids=tuple(
            _require_string(value, f"{name}.source_observation_ids[{source_index}]")
            for source_index, value in enumerate(observation_ids)
        ),
        sensor_stamp_sec=_parse_number(
            item["sensor_stamp_sec"], f"{name}.sensor_stamp_sec"
        ),
        source=_require_string(item["source"], f"{name}.source"),
    )


def _candidate_rejection_from_payload(payload: object, index: int) -> CandidateRejection:
    name = f"rejections[{index}]"
    item = _require_mapping(payload, name)
    _require_exact_fields(
        item,
        frozenset(
            {"candidate_uid", "reason", "source_observation_ids", "rejected_unix_sec"}
        ),
        name,
    )
    observation_ids = _require_list(
        item["source_observation_ids"], f"{name}.source_observation_ids"
    )
    return CandidateRejection(
        candidate_uid=_require_string(item["candidate_uid"], f"{name}.candidate_uid"),
        reason=_require_string(item["reason"], f"{name}.reason"),
        source_observation_ids=tuple(
            _require_string(value, f"{name}.source_observation_ids[{source_index}]")
            for source_index, value in enumerate(observation_ids)
        ),
        rejected_unix_sec=_parse_number(
            item["rejected_unix_sec"], f"{name}.rejected_unix_sec"
        ),
    )


def _pose_from_payload(payload: object, name: str) -> CatalogPose2D:
    item = _require_mapping(payload, name)
    _require_exact_fields(item, frozenset({"x_m", "y_m", "yaw_rad"}), name)
    return CatalogPose2D(
        x_m=_parse_number(item["x_m"], f"{name}.x_m"),
        y_m=_parse_number(item["y_m"], f"{name}.y_m"),
        yaw_rad=_parse_number(item["yaw_rad"], f"{name}.yaw_rad"),
    )


def _normalized_candidate_uids(values: Iterable[str]) -> tuple[str, ...]:
    raw_values = tuple(values)
    if any(not isinstance(value, str) for value in raw_values):
        raise ArrivalPoseCatalogError(
            "invalid_catalog", "candidate UIDs must be strings"
        )
    normalized = tuple(sorted(raw_values))
    for index, value in enumerate(normalized):
        _validate_safe_id(value, f"candidate_uids[{index}]")
    if len(set(normalized)) != len(normalized):
        raise ArrivalPoseCatalogError(
            "invalid_catalog", "candidate UIDs must be unique"
        )
    return normalized


def _normalized_observation_ids(values: Iterable[str]) -> tuple[str, ...]:
    raw_values = tuple(values)
    if any(not isinstance(value, str) for value in raw_values):
        raise ArrivalPoseCatalogError(
            "invalid_record", "source observation IDs must be strings"
        )
    normalized = tuple(sorted(raw_values))
    for index, value in enumerate(normalized):
        _validate_source(value, f"source_observation_ids[{index}]")
    if len(set(normalized)) != len(normalized):
        raise ArrivalPoseCatalogError(
            "invalid_record", "source observation IDs must be unique"
        )
    return normalized


def _check_observation_identity_conflict(
    catalog: ArrivalPoseCatalog,
    *,
    candidate_uid: str,
    source_observation_ids: Sequence[str],
) -> None:
    incoming = set(source_observation_ids)
    for record in catalog.records:
        if record.candidate_uid != candidate_uid and incoming.intersection(
            record.source_observation_ids
        ):
            raise ArrivalPoseCatalogError(
                "observation_identity_conflict",
                "source observation IDs are already owned by candidate "
                f"{record.candidate_uid!r}",
            )
    for rejection in catalog.rejections:
        if rejection.candidate_uid != candidate_uid and incoming.intersection(
            rejection.source_observation_ids
        ):
            raise ArrivalPoseCatalogError(
                "observation_identity_conflict",
                "source observation IDs are already owned by candidate "
                f"{rejection.candidate_uid!r}",
            )


def _claim_observations(
    owners: dict[str, str], candidate_uid: str, observation_ids: Sequence[str]
) -> None:
    for observation_id in observation_ids:
        owner = owners.get(observation_id)
        if owner is not None and owner != candidate_uid:
            raise ArrivalPoseCatalogError(
                "observation_identity_conflict",
                f"observation {observation_id!r} belongs to both {owner!r} and "
                f"{candidate_uid!r}",
            )
        owners[observation_id] = candidate_uid


def _require_expected(catalog: ArrivalPoseCatalog, candidate_uid: str) -> None:
    if catalog.expected_candidate_uids and candidate_uid not in catalog.expected_candidate_uids:
        raise ArrivalPoseCatalogError(
            "unexpected_candidate",
            f"candidate {candidate_uid!r} is not in the expected candidate set",
        )


def _require_mutable(catalog: ArrivalPoseCatalog) -> None:
    if catalog.frozen:
        raise ArrivalPoseCatalogError("catalog_frozen", "arrival-pose catalog is frozen")


def _next_timestamp(catalog: ArrivalPoseCatalog, value: float) -> float:
    timestamp = _finite_nonnegative(value, "updated_unix_sec")
    if timestamp < catalog.updated_unix_sec:
        raise ArrivalPoseCatalogError(
            "timestamp_rollback", "catalog update timestamp precedes the current revision"
        )
    return timestamp


def _validate_ray_pose(
    *,
    pose: CatalogPose2D,
    stand_x: float,
    stand_y: float,
    normal_rad: float,
    expected_distance_m: float,
    name: str,
) -> None:
    dx = pose.x_m - stand_x
    dy = pose.y_m - stand_y
    distance = math.hypot(dx, dy)
    if abs(distance - expected_distance_m) > _GEOMETRY_TOLERANCE_M:
        raise ArrivalPoseCatalogError(
            "geometry_mismatch",
            f"{name} must be {expected_distance_m:.6f} m from stand center",
        )
    radial_angle = math.atan2(dy, dx)
    if _angular_distance(radial_angle, normal_rad) > _ANGLE_TOLERANCE_RAD:
        raise ArrivalPoseCatalogError(
            "geometry_mismatch", f"{name} is not on selected face-normal ray"
        )
    expected_yaw = _normalize_angle(normal_rad + math.pi)
    if _angular_distance(pose.yaw_rad, expected_yaw) > _ANGLE_TOLERANCE_RAD:
        raise ArrivalPoseCatalogError(
            "geometry_mismatch", f"{name} yaw must face the stand center"
        )


def _axial_normal_error(axis_rad: float, normal_rad: float) -> float:
    delta = _angular_distance(axis_rad, normal_rad)
    return min(delta, abs(math.pi - delta))


def _normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _angular_distance(first: float, second: float) -> float:
    return abs(_normalize_angle(first - second))


def _validate_pose(pose: CatalogPose2D, name: str) -> None:
    _finite_number(pose.x_m, f"{name}.x_m")
    _finite_number(pose.y_m, f"{name}.y_m")
    _finite_number(pose.yaw_rad, f"{name}.yaw_rad")


def _validate_safe_id(value: object, name: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ArrivalPoseCatalogError(
            "invalid_catalog",
            f"{name} must start alphanumeric and contain only letters, digits, '.', '_' or '-'",
        )
    return value


def _validate_frame(value: object, name: str) -> str:
    if not isinstance(value, str) or _SAFE_FRAME.fullmatch(value) is None:
        raise ArrivalPoseCatalogError("invalid_catalog", f"{name} is not a valid frame")
    return value


def _validate_source(value: object, name: str) -> str:
    if not isinstance(value, str) or _SAFE_SOURCE.fullmatch(value) is None:
        raise ArrivalPoseCatalogError("invalid_record", f"{name} is not a valid source ID")
    return value


def _validate_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ArrivalPoseCatalogError(
            "invalid_catalog", f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArrivalPoseCatalogError("invalid_record", f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ArrivalPoseCatalogError("invalid_record", f"{name} must be finite")
    return parsed


def _finite_nonnegative(value: object, name: str) -> float:
    parsed = _finite_number(value, name)
    if parsed < 0.0:
        raise ArrivalPoseCatalogError("invalid_record", f"{name} must be non-negative")
    return parsed


def _finite_positive(value: object, name: str) -> float:
    parsed = _finite_number(value, name)
    if parsed <= 0.0:
        raise ArrivalPoseCatalogError("invalid_record", f"{name} must be positive")
    return parsed


def _validate_integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArrivalPoseCatalogError(
            "invalid_catalog", f"{name} must be an integer >= {minimum}"
        )
    return value


def _parse_number(value: object, name: str) -> float:
    return _finite_number(value, name)


def _parse_integer(value: object, name: str, *, minimum: int) -> int:
    return _validate_integer(value, name, minimum=minimum)


def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ArrivalPoseCatalogError("catalog_corrupt", f"{name} must be boolean")
    return value


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ArrivalPoseCatalogError("catalog_corrupt", f"{name} must be a string")
    return value


def _require_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ArrivalPoseCatalogError("catalog_corrupt", f"{name} must be an array")
    return value


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ArrivalPoseCatalogError("catalog_corrupt", f"{name} must be an object")
    return value


def _require_exact_fields(
    payload: Mapping[str, object], expected: frozenset[str], name: str
) -> None:
    actual = frozenset(payload)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        unknown = sorted(actual.difference(expected))
        raise ArrivalPoseCatalogError(
            "catalog_corrupt",
            f"{name} fields mismatch; missing={missing} unknown={unknown}",
        )


def _sha256_payload(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArrivalPoseCatalogError(
            "invalid_catalog", f"catalog is not finite JSON: {exc}"
        ) from exc


def _pretty_json_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArrivalPoseCatalogError(
            "invalid_catalog", f"catalog is not finite JSON: {exc}"
        ) from exc


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # Some test and in-memory filesystems do not support directory fsync.
        pass


def _atomic_replace(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
