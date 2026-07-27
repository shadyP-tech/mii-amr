"""One-to-one candidate, QR, and server-station identity registry."""

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
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateSnapshot,
    candidate_snapshot_sha256,
)


STATION_IDENTITY_REGISTRY_SCHEMA_VERSION = 1

_HASH_FIELD = "station_identity_registry_sha256"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "registry_id",
        "created_unix_sec",
        "candidate_snapshot_sha256",
        "source_artifact_sha256",
        "expected_candidate_uids",
        "mappings",
    }
)
_MAPPING_FIELDS = frozenset({"candidate_uid", "qr_id", "server_station_id"})


class StationIdentityRegistryError(ValueError):
    """Identity registry error with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class StationIdentity:
    candidate_uid: str
    qr_id: str
    server_station_id: str


@dataclass(frozen=True)
class StationIdentityRegistry:
    schema_version: int
    registry_id: str
    created_unix_sec: float
    candidate_snapshot_sha256: str
    source_artifact_sha256: str
    expected_candidate_uids: tuple[str, ...]
    mappings: tuple[StationIdentity, ...]

    def for_candidate(self, candidate_uid: str) -> StationIdentity | None:
        return next(
            (item for item in self.mappings if item.candidate_uid == candidate_uid),
            None,
        )

    def for_qr(self, qr_id: str) -> StationIdentity | None:
        return next((item for item in self.mappings if item.qr_id == qr_id), None)

    def for_server_station(self, server_station_id: str) -> StationIdentity | None:
        return next(
            (
                item
                for item in self.mappings
                if item.server_station_id == server_station_id
            ),
            None,
        )


def new_station_identity_registry(
    *,
    registry_id: str,
    created_unix_sec: float,
    candidate_snapshot_sha256: str,
    source_artifact_sha256: str,
    expected_candidate_uids: Iterable[str],
    mappings: Iterable[StationIdentity],
) -> StationIdentityRegistry:
    raw_expected = tuple(expected_candidate_uids)
    for index, candidate_uid in enumerate(raw_expected):
        _validate_id(candidate_uid, f"expected_candidate_uids[{index}]")
    raw_mappings = tuple(mappings)
    for mapping in raw_mappings:
        validate_station_identity(mapping)
    registry = StationIdentityRegistry(
        schema_version=STATION_IDENTITY_REGISTRY_SCHEMA_VERSION,
        registry_id=registry_id,
        created_unix_sec=created_unix_sec,
        candidate_snapshot_sha256=candidate_snapshot_sha256,
        source_artifact_sha256=source_artifact_sha256,
        expected_candidate_uids=tuple(sorted(raw_expected)),
        mappings=tuple(sorted(raw_mappings, key=lambda item: item.candidate_uid)),
    )
    validate_station_identity_registry(registry)
    return registry


def validate_station_identity_registry(
    registry: StationIdentityRegistry,
    *,
    candidate_snapshot: CandidateSnapshot | None = None,
) -> None:
    if type(registry.schema_version) is not int or (
        registry.schema_version != STATION_IDENTITY_REGISTRY_SCHEMA_VERSION
    ):
        raise StationIdentityRegistryError(
            "schema_mismatch",
            f"unsupported identity registry schema {registry.schema_version!r}",
        )
    _validate_id(registry.registry_id, "registry_id")
    _finite_nonnegative(registry.created_unix_sec, "created_unix_sec")
    _validate_sha256(
        registry.candidate_snapshot_sha256, "candidate_snapshot_sha256"
    )
    _validate_sha256(registry.source_artifact_sha256, "source_artifact_sha256")
    if not isinstance(registry.expected_candidate_uids, tuple):
        raise StationIdentityRegistryError(
            "invalid_registry", "expected_candidate_uids must be a tuple"
        )
    for index, candidate_uid in enumerate(registry.expected_candidate_uids):
        _validate_id(candidate_uid, f"expected_candidate_uids[{index}]")
    expected = tuple(sorted(set(registry.expected_candidate_uids)))
    if not expected or expected != registry.expected_candidate_uids:
        raise StationIdentityRegistryError(
            "invalid_registry",
            "expected_candidate_uids must be non-empty, sorted, and unique",
        )
    if not isinstance(registry.mappings, tuple):
        raise StationIdentityRegistryError(
            "invalid_registry", "mappings must be a tuple"
        )
    for mapping in registry.mappings:
        if not isinstance(mapping, StationIdentity):
            raise StationIdentityRegistryError(
                "invalid_registry", "mappings must contain StationIdentity values"
            )
    if tuple(sorted(registry.mappings, key=lambda item: item.candidate_uid)) != (
        registry.mappings
    ):
        raise StationIdentityRegistryError(
            "invalid_registry", "mappings must be sorted by candidate_uid"
        )

    candidate_ids = set()
    qr_ids = set()
    server_ids = set()
    for mapping in registry.mappings:
        validate_station_identity(mapping)
        if mapping.candidate_uid in candidate_ids:
            _raise_duplicate("candidate_uid", mapping.candidate_uid)
        if mapping.qr_id in qr_ids:
            _raise_duplicate("qr_id", mapping.qr_id)
        if mapping.server_station_id in server_ids:
            _raise_duplicate("server_station_id", mapping.server_station_id)
        candidate_ids.add(mapping.candidate_uid)
        qr_ids.add(mapping.qr_id)
        server_ids.add(mapping.server_station_id)
    if candidate_ids != set(expected):
        raise StationIdentityRegistryError(
            "incomplete_registry",
            "mappings must resolve every expected candidate exactly once; "
            f"missing={sorted(set(expected) - candidate_ids)} "
            f"unknown={sorted(candidate_ids - set(expected))}",
        )

    if candidate_snapshot is not None:
        actual_snapshot_sha256 = candidate_snapshot_sha256(candidate_snapshot)
        if registry.candidate_snapshot_sha256 != actual_snapshot_sha256:
            raise StationIdentityRegistryError(
                "provenance_mismatch",
                "identity registry references another candidate snapshot",
            )
        if registry.expected_candidate_uids != candidate_snapshot.candidate_uids:
            raise StationIdentityRegistryError(
                "provenance_mismatch",
                "identity registry candidate set differs from candidate snapshot",
            )


def validate_station_identity(identity: StationIdentity) -> None:
    if not isinstance(identity, StationIdentity):
        raise StationIdentityRegistryError(
            "invalid_registry", "identity must be a StationIdentity"
        )
    _validate_id(identity.candidate_uid, "candidate_uid")
    _validate_id(identity.qr_id, "qr_id")
    _validate_id(identity.server_station_id, "server_station_id")


def candidate_order_for_server_order(
    registry: StationIdentityRegistry, station_order: Iterable[str]
) -> tuple[str, ...]:
    """Resolve server order without sorting, optimizing, or deduplicating it."""

    validate_station_identity_registry(registry)
    result = []
    for index, station_id in enumerate(station_order):
        _validate_id(station_id, f"station_order[{index}]")
        identity = registry.for_server_station(station_id)
        if identity is None:
            raise StationIdentityRegistryError(
                "unknown_server_station",
                f"server station {station_id!r} is not associated with a candidate",
            )
        result.append(identity.candidate_uid)
    if not result:
        raise StationIdentityRegistryError(
            "invalid_station_order", "station_order must not be empty"
        )
    return tuple(result)


def station_identity_registry_sha256(registry: StationIdentityRegistry) -> str:
    return payload_sha256(_registry_payload_without_hash(registry))


def station_identity_registry_payload(
    registry: StationIdentityRegistry,
) -> dict[str, object]:
    return content_hashed_payload(
        _registry_payload_without_hash(registry), hash_field=_HASH_FIELD
    )


def write_station_identity_registry(
    path: Path, registry: StationIdentityRegistry
) -> str:
    try:
        return write_content_hashed_json(
            path, _registry_payload_without_hash(registry), hash_field=_HASH_FIELD
        )
    except ContentStoreError as exc:
        raise StationIdentityRegistryError(exc.code, str(exc)) from exc


def load_station_identity_registry(
    path: Path, *, candidate_snapshot: CandidateSnapshot | None = None
) -> StationIdentityRegistry:
    try:
        payload = load_content_hashed_json(path, hash_field=_HASH_FIELD)
    except ContentStoreError as exc:
        raise StationIdentityRegistryError(exc.code, str(exc)) from exc
    try:
        registry = _registry_from_payload(payload)
    except (KeyError, TypeError) as exc:
        raise StationIdentityRegistryError(
            "artifact_corrupt", "identity registry has invalid field types"
        ) from exc
    validate_station_identity_registry(
        registry, candidate_snapshot=candidate_snapshot
    )
    return registry


def _registry_payload_without_hash(
    registry: StationIdentityRegistry,
) -> dict[str, object]:
    validate_station_identity_registry(registry)
    return {
        "schema_version": registry.schema_version,
        "registry_id": registry.registry_id,
        "created_unix_sec": registry.created_unix_sec,
        "candidate_snapshot_sha256": registry.candidate_snapshot_sha256,
        "source_artifact_sha256": registry.source_artifact_sha256,
        "expected_candidate_uids": list(registry.expected_candidate_uids),
        "mappings": [
            {
                "candidate_uid": item.candidate_uid,
                "qr_id": item.qr_id,
                "server_station_id": item.server_station_id,
            }
            for item in registry.mappings
        ],
    }


def _registry_from_payload(
    payload: Mapping[str, object]
) -> StationIdentityRegistry:
    _require_fields(payload, _ROOT_FIELDS, "identity registry")
    expected = _list(payload["expected_candidate_uids"], "expected_candidate_uids")
    mappings = _list(payload["mappings"], "mappings")
    return StationIdentityRegistry(
        schema_version=_integer(payload["schema_version"], "schema_version"),
        registry_id=_string(payload["registry_id"], "registry_id"),
        created_unix_sec=_number(payload["created_unix_sec"], "created_unix_sec"),
        candidate_snapshot_sha256=_string(
            payload["candidate_snapshot_sha256"], "candidate_snapshot_sha256"
        ),
        source_artifact_sha256=_string(
            payload["source_artifact_sha256"], "source_artifact_sha256"
        ),
        expected_candidate_uids=tuple(
            _string(value, f"expected_candidate_uids[{index}]")
            for index, value in enumerate(expected)
        ),
        mappings=tuple(
            _mapping_from_payload(value, index)
            for index, value in enumerate(mappings)
        ),
    )


def _mapping_from_payload(value: object, index: int) -> StationIdentity:
    name = f"mappings[{index}]"
    item = _mapping(value, name)
    _require_fields(item, _MAPPING_FIELDS, name)
    return StationIdentity(
        candidate_uid=_string(item["candidate_uid"], f"{name}.candidate_uid"),
        qr_id=_string(item["qr_id"], f"{name}.qr_id"),
        server_station_id=_string(
            item["server_station_id"], f"{name}.server_station_id"
        ),
    )


def _raise_duplicate(field: str, value: str) -> None:
    raise StationIdentityRegistryError(
        "identity_conflict", f"duplicate {field} mapping for {value!r}"
    )


def _require_fields(
    payload: Mapping[str, object], expected: frozenset[str], name: str
) -> None:
    actual = frozenset(payload)
    if actual != expected:
        raise StationIdentityRegistryError(
            "artifact_corrupt",
            f"{name} fields mismatch; "
            f"missing={sorted(expected - actual)} unknown={sorted(actual - expected)}",
        )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise StationIdentityRegistryError(
            "artifact_corrupt", f"{name} must be an object"
        )
    return value


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise StationIdentityRegistryError(
            "artifact_corrupt", f"{name} must be an array"
        )
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise StationIdentityRegistryError(
            "artifact_corrupt", f"{name} must be a string"
        )
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StationIdentityRegistryError(
            "artifact_corrupt", f"{name} must be a number"
        )
    return float(value)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StationIdentityRegistryError(
            "artifact_corrupt", f"{name} must be an integer"
        )
    return value


def _validate_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise StationIdentityRegistryError(
            "invalid_registry", f"{name} is not a safe identifier"
        )
    return value


def _validate_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise StationIdentityRegistryError(
            "invalid_registry", f"{name} must be a lowercase SHA-256"
        )
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StationIdentityRegistryError(
            "invalid_registry", f"{name} must be a number"
        )
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise StationIdentityRegistryError(
            "invalid_registry", f"{name} must be finite and non-negative"
        )
    return parsed
