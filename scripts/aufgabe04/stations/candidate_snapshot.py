"""Strict immutable snapshot of detector-produced stand candidates.

Candidate geometry and observation ancestry are hashed independently.  A
resume check can therefore reject reuse when a stable candidate UID moves or
when its detector evidence/configuration changes, even before comparing the
whole snapshot hash.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from scripts.aufgabe04.artifacts.content_store import (
    ContentStoreError,
    content_hashed_payload,
    load_content_hashed_json,
    payload_sha256,
    write_content_hashed_json,
)


CANDIDATE_SNAPSHOT_SCHEMA_VERSION = 1

_HASH_FIELD = "candidate_snapshot_sha256"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_FRAME = re.compile(r"^[A-Za-z][A-Za-z0-9_/.-]{0,127}$")
_SAFE_SOURCE = re.compile(r"^[A-Za-z0-9/][A-Za-z0-9_.:/+@-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "snapshot_id",
        "created_unix_sec",
        "planning_frame",
        "map_bundle_sha256",
        "candidates",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_uid",
        "geometry",
        "source",
        "confidence",
        "hit_count",
        "first_seen_sec",
        "last_seen_sec",
    }
)
_GEOMETRY_FIELDS = frozenset(
    {
        "x_m",
        "y_m",
        "radius_m",
        "uncertainty_m",
        "keepout_radius_m",
        "geometry_sha256",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "source_kind",
        "source_artifact_sha256",
        "detector_config_sha256",
        "observation_ids",
        "source_sha256",
    }
)


class CandidateSnapshotError(ValueError):
    """Candidate snapshot error with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CandidateGeometry:
    """Frozen stand envelope; keepout is a total robot-centre exclusion radius."""

    x_m: float
    y_m: float
    radius_m: float
    uncertainty_m: float
    keepout_radius_m: float


@dataclass(frozen=True)
class CandidateSource:
    source_kind: str
    source_artifact_sha256: str
    detector_config_sha256: str
    observation_ids: tuple[str, ...]


@dataclass(frozen=True)
class FrozenCandidate:
    candidate_uid: str
    geometry: CandidateGeometry
    source: CandidateSource
    confidence: float
    hit_count: int
    first_seen_sec: float
    last_seen_sec: float


@dataclass(frozen=True)
class CandidateSnapshot:
    schema_version: int
    snapshot_id: str
    created_unix_sec: float
    planning_frame: str
    map_bundle_sha256: str
    candidates: tuple[FrozenCandidate, ...]

    @property
    def candidate_uids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_uid for candidate in self.candidates)

    def candidate_for(self, candidate_uid: str) -> FrozenCandidate | None:
        return next(
            (
                candidate
                for candidate in self.candidates
                if candidate.candidate_uid == candidate_uid
            ),
            None,
        )


def new_candidate_snapshot(
    *,
    snapshot_id: str,
    created_unix_sec: float,
    planning_frame: str,
    map_bundle_sha256: str,
    candidates: Iterable[FrozenCandidate],
) -> CandidateSnapshot:
    snapshot = CandidateSnapshot(
        schema_version=CANDIDATE_SNAPSHOT_SCHEMA_VERSION,
        snapshot_id=snapshot_id,
        created_unix_sec=created_unix_sec,
        planning_frame=planning_frame,
        map_bundle_sha256=map_bundle_sha256,
        candidates=tuple(sorted(candidates, key=lambda item: item.candidate_uid)),
    )
    validate_candidate_snapshot(snapshot)
    return snapshot


def validate_candidate_snapshot(
    snapshot: CandidateSnapshot, *, required_map_bundle_sha256: str | None = None
) -> None:
    if type(snapshot.schema_version) is not int or (
        snapshot.schema_version != CANDIDATE_SNAPSHOT_SCHEMA_VERSION
    ):
        raise CandidateSnapshotError(
            "schema_mismatch",
            f"unsupported candidate snapshot schema {snapshot.schema_version!r}",
        )
    _validate_id(snapshot.snapshot_id, "snapshot_id")
    _finite_nonnegative(snapshot.created_unix_sec, "created_unix_sec")
    if not isinstance(snapshot.planning_frame, str) or not _SAFE_FRAME.fullmatch(
        snapshot.planning_frame
    ):
        raise CandidateSnapshotError(
            "invalid_snapshot", "planning_frame is not a valid frame identifier"
        )
    _validate_sha256(snapshot.map_bundle_sha256, "map_bundle_sha256")
    if required_map_bundle_sha256 is not None:
        _validate_sha256(required_map_bundle_sha256, "required_map_bundle_sha256")
        if snapshot.map_bundle_sha256 != required_map_bundle_sha256:
            raise CandidateSnapshotError(
                "provenance_mismatch",
                "candidate snapshot was created against another map bundle",
            )
    if not isinstance(snapshot.candidates, tuple) or not snapshot.candidates:
        raise CandidateSnapshotError(
            "invalid_snapshot", "candidates must be a non-empty tuple"
        )
    for candidate in snapshot.candidates:
        if not isinstance(candidate, FrozenCandidate):
            raise CandidateSnapshotError(
                "invalid_candidate", "candidates must contain FrozenCandidate values"
            )
    expected_order = tuple(
        sorted(snapshot.candidates, key=lambda item: item.candidate_uid)
    )
    if snapshot.candidates != expected_order:
        raise CandidateSnapshotError(
            "invalid_snapshot", "candidates must be sorted by candidate_uid"
        )

    candidate_uids = set()
    observation_owners: dict[str, str] = {}
    for candidate in snapshot.candidates:
        validate_frozen_candidate(candidate)
        if candidate.candidate_uid in candidate_uids:
            raise CandidateSnapshotError(
                "candidate_conflict",
                f"duplicate candidate UID {candidate.candidate_uid!r}",
            )
        candidate_uids.add(candidate.candidate_uid)
        for observation_id in candidate.source.observation_ids:
            owner = observation_owners.get(observation_id)
            if owner is not None and owner != candidate.candidate_uid:
                raise CandidateSnapshotError(
                    "observation_identity_conflict",
                    f"observation {observation_id!r} belongs to both {owner!r} "
                    f"and {candidate.candidate_uid!r}",
                )
            observation_owners[observation_id] = candidate.candidate_uid


def validate_frozen_candidate(candidate: FrozenCandidate) -> None:
    if not isinstance(candidate, FrozenCandidate):
        raise CandidateSnapshotError(
            "invalid_candidate", "candidate must be a FrozenCandidate"
        )
    if not isinstance(candidate.geometry, CandidateGeometry):
        raise CandidateSnapshotError(
            "invalid_geometry", "geometry must be a CandidateGeometry"
        )
    if not isinstance(candidate.source, CandidateSource):
        raise CandidateSnapshotError(
            "invalid_source", "source must be a CandidateSource"
        )
    _validate_id(candidate.candidate_uid, "candidate_uid")
    validate_candidate_geometry(candidate.geometry)
    validate_candidate_source(candidate.source)
    confidence = _finite_number(candidate.confidence, "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise CandidateSnapshotError(
            "invalid_candidate", "confidence must be in [0, 1]"
        )
    if (
        isinstance(candidate.hit_count, bool)
        or not isinstance(candidate.hit_count, int)
        or candidate.hit_count < 1
    ):
        raise CandidateSnapshotError(
            "invalid_candidate", "hit_count must be an integer >= 1"
        )
    first = _finite_nonnegative(candidate.first_seen_sec, "first_seen_sec")
    last = _finite_nonnegative(candidate.last_seen_sec, "last_seen_sec")
    if last < first:
        raise CandidateSnapshotError(
            "invalid_candidate", "last_seen_sec precedes first_seen_sec"
        )


def validate_candidate_geometry(geometry: CandidateGeometry) -> None:
    if not isinstance(geometry, CandidateGeometry):
        raise CandidateSnapshotError(
            "invalid_geometry", "geometry must be a CandidateGeometry"
        )
    _finite_number(geometry.x_m, "geometry.x_m")
    _finite_number(geometry.y_m, "geometry.y_m")
    radius = _finite_positive(geometry.radius_m, "geometry.radius_m")
    _finite_nonnegative(geometry.uncertainty_m, "geometry.uncertainty_m")
    keepout = _finite_positive(
        geometry.keepout_radius_m, "geometry.keepout_radius_m"
    )
    if keepout < radius:
        raise CandidateSnapshotError(
            "invalid_geometry", "keepout_radius_m must not be smaller than radius_m"
        )


def validate_candidate_source(source: CandidateSource) -> None:
    if not isinstance(source, CandidateSource):
        raise CandidateSnapshotError(
            "invalid_source", "source must be a CandidateSource"
        )
    if not isinstance(source.source_kind, str) or not _SAFE_SOURCE.fullmatch(
        source.source_kind
    ):
        raise CandidateSnapshotError(
            "invalid_source", "source_kind is not a safe source identifier"
        )
    _validate_sha256(source.source_artifact_sha256, "source_artifact_sha256")
    _validate_sha256(source.detector_config_sha256, "detector_config_sha256")
    if not isinstance(source.observation_ids, tuple) or not source.observation_ids:
        raise CandidateSnapshotError(
            "invalid_source", "observation_ids must be a non-empty tuple"
        )
    for index, observation_id in enumerate(source.observation_ids):
        _validate_id(observation_id, f"observation_ids[{index}]")
    normalized = tuple(sorted(set(source.observation_ids)))
    if normalized != source.observation_ids:
        raise CandidateSnapshotError(
            "invalid_source", "observation_ids must be sorted and unique"
        )


def candidate_geometry_sha256(geometry: CandidateGeometry) -> str:
    validate_candidate_geometry(geometry)
    return payload_sha256(_geometry_payload_without_hash(geometry))


def candidate_source_sha256(source: CandidateSource) -> str:
    validate_candidate_source(source)
    return payload_sha256(_source_payload_without_hash(source))


def candidate_snapshot_sha256(snapshot: CandidateSnapshot) -> str:
    return payload_sha256(_snapshot_payload_without_hash(snapshot))


def candidate_snapshot_payload(snapshot: CandidateSnapshot) -> dict[str, object]:
    return content_hashed_payload(
        _snapshot_payload_without_hash(snapshot), hash_field=_HASH_FIELD
    )


def write_candidate_snapshot(path: Path, snapshot: CandidateSnapshot) -> str:
    try:
        return write_content_hashed_json(
            path, _snapshot_payload_without_hash(snapshot), hash_field=_HASH_FIELD
        )
    except ContentStoreError as exc:
        raise CandidateSnapshotError(exc.code, str(exc)) from exc


def load_candidate_snapshot(
    path: Path, *, required_map_bundle_sha256: str | None = None
) -> CandidateSnapshot:
    try:
        payload = load_content_hashed_json(path, hash_field=_HASH_FIELD)
    except ContentStoreError as exc:
        raise CandidateSnapshotError(exc.code, str(exc)) from exc
    try:
        snapshot = _snapshot_from_payload(payload)
    except (KeyError, TypeError) as exc:
        raise CandidateSnapshotError(
            "artifact_corrupt", "candidate snapshot has invalid field types"
        ) from exc
    validate_candidate_snapshot(
        snapshot, required_map_bundle_sha256=required_map_bundle_sha256
    )
    return snapshot


def _snapshot_payload_without_hash(
    snapshot: CandidateSnapshot,
) -> dict[str, object]:
    validate_candidate_snapshot(snapshot)
    return {
        "schema_version": snapshot.schema_version,
        "snapshot_id": snapshot.snapshot_id,
        "created_unix_sec": snapshot.created_unix_sec,
        "planning_frame": snapshot.planning_frame,
        "map_bundle_sha256": snapshot.map_bundle_sha256,
        "candidates": [_candidate_payload(item) for item in snapshot.candidates],
    }


def _candidate_payload(candidate: FrozenCandidate) -> dict[str, object]:
    validate_frozen_candidate(candidate)
    geometry = _geometry_payload_without_hash(candidate.geometry)
    geometry["geometry_sha256"] = payload_sha256(geometry)
    source = _source_payload_without_hash(candidate.source)
    source["source_sha256"] = payload_sha256(source)
    return {
        "candidate_uid": candidate.candidate_uid,
        "geometry": geometry,
        "source": source,
        "confidence": candidate.confidence,
        "hit_count": candidate.hit_count,
        "first_seen_sec": candidate.first_seen_sec,
        "last_seen_sec": candidate.last_seen_sec,
    }


def _geometry_payload_without_hash(
    geometry: CandidateGeometry,
) -> dict[str, object]:
    return {
        "x_m": geometry.x_m,
        "y_m": geometry.y_m,
        "radius_m": geometry.radius_m,
        "uncertainty_m": geometry.uncertainty_m,
        "keepout_radius_m": geometry.keepout_radius_m,
    }


def _source_payload_without_hash(source: CandidateSource) -> dict[str, object]:
    return {
        "source_kind": source.source_kind,
        "source_artifact_sha256": source.source_artifact_sha256,
        "detector_config_sha256": source.detector_config_sha256,
        "observation_ids": list(source.observation_ids),
    }


def _snapshot_from_payload(payload: Mapping[str, object]) -> CandidateSnapshot:
    _require_fields(payload, _ROOT_FIELDS, "candidate snapshot")
    candidates = _list(payload["candidates"], "candidates")
    return CandidateSnapshot(
        schema_version=_integer(payload["schema_version"], "schema_version"),
        snapshot_id=_string(payload["snapshot_id"], "snapshot_id"),
        created_unix_sec=_number(payload["created_unix_sec"], "created_unix_sec"),
        planning_frame=_string(payload["planning_frame"], "planning_frame"),
        map_bundle_sha256=_string(
            payload["map_bundle_sha256"], "map_bundle_sha256"
        ),
        candidates=tuple(
            _candidate_from_payload(item, index)
            for index, item in enumerate(candidates)
        ),
    )


def _candidate_from_payload(value: object, index: int) -> FrozenCandidate:
    name = f"candidates[{index}]"
    item = _mapping(value, name)
    _require_fields(item, _CANDIDATE_FIELDS, name)
    return FrozenCandidate(
        candidate_uid=_string(item["candidate_uid"], f"{name}.candidate_uid"),
        geometry=_geometry_from_payload(item["geometry"], f"{name}.geometry"),
        source=_source_from_payload(item["source"], f"{name}.source"),
        confidence=_number(item["confidence"], f"{name}.confidence"),
        hit_count=_integer(item["hit_count"], f"{name}.hit_count"),
        first_seen_sec=_number(item["first_seen_sec"], f"{name}.first_seen_sec"),
        last_seen_sec=_number(item["last_seen_sec"], f"{name}.last_seen_sec"),
    )


def _geometry_from_payload(value: object, name: str) -> CandidateGeometry:
    item = _mapping(value, name)
    _require_fields(item, _GEOMETRY_FIELDS, name)
    stored = _string(item["geometry_sha256"], f"{name}.geometry_sha256")
    unhashed = dict(item)
    del unhashed["geometry_sha256"]
    _require_nested_hash(stored, unhashed, f"{name}.geometry_sha256")
    return CandidateGeometry(
        x_m=_number(item["x_m"], f"{name}.x_m"),
        y_m=_number(item["y_m"], f"{name}.y_m"),
        radius_m=_number(item["radius_m"], f"{name}.radius_m"),
        uncertainty_m=_number(item["uncertainty_m"], f"{name}.uncertainty_m"),
        keepout_radius_m=_number(
            item["keepout_radius_m"], f"{name}.keepout_radius_m"
        ),
    )


def _source_from_payload(value: object, name: str) -> CandidateSource:
    item = _mapping(value, name)
    _require_fields(item, _SOURCE_FIELDS, name)
    stored = _string(item["source_sha256"], f"{name}.source_sha256")
    unhashed = dict(item)
    del unhashed["source_sha256"]
    _require_nested_hash(stored, unhashed, f"{name}.source_sha256")
    observation_ids = _list(item["observation_ids"], f"{name}.observation_ids")
    return CandidateSource(
        source_kind=_string(item["source_kind"], f"{name}.source_kind"),
        source_artifact_sha256=_string(
            item["source_artifact_sha256"], f"{name}.source_artifact_sha256"
        ),
        detector_config_sha256=_string(
            item["detector_config_sha256"], f"{name}.detector_config_sha256"
        ),
        observation_ids=tuple(
            _string(value, f"{name}.observation_ids[{index}]")
            for index, value in enumerate(observation_ids)
        ),
    )


def _require_nested_hash(
    stored: str, payload: Mapping[str, object], name: str
) -> None:
    _validate_sha256(stored, name)
    actual = payload_sha256(payload)
    if stored != actual:
        raise CandidateSnapshotError(
            "hash_mismatch", f"{name} mismatch: expected {stored}, got {actual}"
        )


def _require_fields(
    payload: Mapping[str, object], expected: frozenset[str], name: str
) -> None:
    actual = frozenset(payload)
    if actual != expected:
        raise CandidateSnapshotError(
            "artifact_corrupt",
            f"{name} fields mismatch; "
            f"missing={sorted(expected - actual)} unknown={sorted(actual - expected)}",
        )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise CandidateSnapshotError("artifact_corrupt", f"{name} must be an object")
    return value


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise CandidateSnapshotError("artifact_corrupt", f"{name} must be an array")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise CandidateSnapshotError("artifact_corrupt", f"{name} must be a string")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateSnapshotError("artifact_corrupt", f"{name} must be a number")
    return float(value)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CandidateSnapshotError("artifact_corrupt", f"{name} must be an integer")
    return value


def _validate_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise CandidateSnapshotError(
            "invalid_candidate", f"{name} is not a safe identifier"
        )
    return value


def _validate_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise CandidateSnapshotError(
            "invalid_source", f"{name} must be a lowercase SHA-256"
        )
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateSnapshotError("invalid_candidate", f"{name} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise CandidateSnapshotError(
            "invalid_candidate", f"{name} must be finite"
        )
    return parsed


def _finite_nonnegative(value: object, name: str) -> float:
    parsed = _finite_number(value, name)
    if parsed < 0.0:
        raise CandidateSnapshotError(
            "invalid_candidate", f"{name} must be non-negative"
        )
    return parsed


def _finite_positive(value: object, name: str) -> float:
    parsed = _finite_number(value, name)
    if parsed <= 0.0:
        raise CandidateSnapshotError(
            "invalid_geometry", f"{name} must be positive"
        )
    return parsed
