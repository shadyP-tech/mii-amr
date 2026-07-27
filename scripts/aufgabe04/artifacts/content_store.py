"""Canonical JSON hashing and immutable atomic publication.

The publisher never replaces an existing path.  It writes and fsyncs a
temporary file and then atomically links it into place.  Re-publishing the
same bytes is an idempotent retry; different bytes at the same path are a
stable conflict instead of an implicit artifact mutation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Mapping


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ContentStoreError(ValueError):
    """Hashed JSON storage error with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    """Return the single canonical representation used for content hashes."""

    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContentStoreError(
            "invalid_payload", f"payload is not finite JSON: {exc}"
        ) from exc


def payload_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def content_hashed_payload(
    payload: Mapping[str, object], *, hash_field: str
) -> dict[str, object]:
    if hash_field in payload:
        raise ContentStoreError(
            "invalid_payload", f"unhashed payload already contains {hash_field!r}"
        )
    result = dict(payload)
    result[hash_field] = payload_sha256(payload)
    return result


def write_content_hashed_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    hash_field: str,
) -> str:
    """Publish canonical content atomically without replacing prior content."""

    hashed = content_hashed_payload(payload, hash_field=hash_field)
    try:
        data = (
            json.dumps(
                hashed,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContentStoreError(
            "invalid_payload", f"payload is not finite JSON: {exc}"
        ) from exc
    _publish_immutable(Path(path), data)
    return str(hashed[hash_field])


def load_content_hashed_json(
    path: Path, *, hash_field: str
) -> dict[str, object]:
    """Strictly decode and hash-check a content-addressed JSON object."""

    path = Path(path)
    if path.is_symlink():
        raise ContentStoreError(
            "artifact_unavailable", f"artifact path must not be a symlink: {path}"
        )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ContentStoreError(
            "artifact_unavailable", f"artifact is unavailable: {path}"
        ) from exc
    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_strict_object_pairs
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ContentStoreError) as exc:
        if isinstance(exc, ContentStoreError):
            raise
        raise ContentStoreError(
            "artifact_corrupt", f"invalid artifact JSON: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ContentStoreError("artifact_corrupt", "artifact root must be an object")
    if hash_field not in payload:
        raise ContentStoreError(
            "artifact_corrupt", f"artifact is missing {hash_field!r}"
        )
    stored_hash = payload[hash_field]
    if not isinstance(stored_hash, str) or not _SHA256.fullmatch(stored_hash):
        raise ContentStoreError(
            "artifact_corrupt", f"{hash_field} must be a lowercase SHA-256"
        )
    unhashed = dict(payload)
    del unhashed[hash_field]
    actual_hash = payload_sha256(unhashed)
    if stored_hash != actual_hash:
        raise ContentStoreError(
            "hash_mismatch",
            f"artifact hash mismatch: expected {stored_hash}, got {actual_hash}",
        )
    return unhashed


def _strict_object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContentStoreError(
                "artifact_corrupt", f"duplicate JSON object key {key!r}"
            )
        result[key] = value
    return result


def _publish_immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ContentStoreError(
            "immutable_conflict", f"refusing artifact symlink target: {path}"
        )
    if path.exists():
        _require_identical_existing(path, data)
        return

    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require_identical_existing(path, data)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _require_identical_existing(path: Path, data: bytes) -> None:
    if path.is_symlink():
        raise ContentStoreError(
            "immutable_conflict", f"refusing artifact symlink target: {path}"
        )
    try:
        existing = path.read_bytes()
    except OSError as exc:
        raise ContentStoreError(
            "immutable_conflict", f"cannot verify existing artifact: {path}"
        ) from exc
    if existing != data:
        raise ContentStoreError(
            "immutable_conflict",
            f"refusing to replace immutable artifact with different content: {path}",
        )


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # Some in-memory and test filesystems do not support directory fsync.
        pass
