"""Immutable, simulation-only route revision publication and validation.

The JSON manifest is the sole hand-off pointer.  Route and diagnostics
artifacts are written with revision-specific names before the manifest is
atomically replaced, so a reader never has to consume a partially published
revision.

This module deliberately has no ROS imports.  It is shared by the simulation
planner and by the motion-side route hand-off.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


SCHEMA_VERSION = 1
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RouteRevisionError(RuntimeError):
    """A publication or validation failure with a stable machine code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class LoadedRouteRevision:
    """A validated snapshot of the authoritative route manifest."""

    manifest_path: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    route_path: Path | None
    diagnostics_path: Path | None
    duplicate: bool = False

    @property
    def route_revision(self) -> int:
        return int(self.manifest["route_revision"])

    @property
    def target_revision(self) -> int:
        return int(self.manifest["target_revision"])

    @property
    def status(self) -> str:
        return str(self.manifest["status"])

    @property
    def writer_id(self) -> str:
        return str(self.manifest["writer_id"])

    @property
    def writer_generation(self) -> int:
        return int(self.manifest["writer_generation"])

    @property
    def route_hash(self) -> str | None:
        descriptor = self.manifest.get("route")
        if isinstance(descriptor, Mapping):
            value = descriptor.get("sha256")
            return str(value) if value is not None else None
        return None

    @property
    def reason(self) -> str:
        return str(self.manifest.get("withdrawal_reason", ""))


def validate_safe_component(value: str, *, field: str) -> str:
    """Reject IDs that could escape or ambiguously address a revision tree."""

    value = str(value)
    if value in {".", ".."} or _SAFE_COMPONENT.fullmatch(value) is None:
        raise RouteRevisionError(
            "unsafe_component",
            f"{field} must be a safe path component containing only letters, "
            "digits, '.', '_' or '-'",
        )
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise RouteRevisionError("invalid_payload", f"payload is not finite JSON: {exc}") from exc


def _finite_timestamp(value: float, field: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise RouteRevisionError("invalid_manifest", f"{field} must be numeric") from exc
    if not math.isfinite(value):
        raise RouteRevisionError("invalid_manifest", f"{field} must be finite")
    return value


def _finite_nonnegative(value: float, field: str) -> float:
    value = _finite_timestamp(value, field)
    if value < 0.0:
        raise RouteRevisionError("invalid_manifest", f"{field} must be non-negative")
    return value


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise RouteRevisionError("invalid_manifest", f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RouteRevisionError("invalid_manifest", f"{field} must be an integer") from exc
    if parsed != value or parsed < minimum:
        raise RouteRevisionError(
            "invalid_manifest", f"{field} must be an integer >= {minimum}"
        )
    return parsed


def _exclusive_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RouteRevisionError(
            "immutable_revision_exists", f"refusing to overwrite immutable artifact {path}"
        ) from exc


def _fsync_directory(path: Path) -> None:
    try:
        directory_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        # Some test/in-memory filesystems do not support directory fsync.
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


def _read_stable_bytes(path: Path, *, attempts: int = 3) -> bytes:
    """Read the same manifest bytes twice, retrying across atomic updates."""

    if attempts < 1:
        raise ValueError("attempts must be positive")
    last_error: OSError | None = None
    for _ in range(attempts):
        try:
            first = path.read_bytes()
            second = path.read_bytes()
        except OSError as exc:
            last_error = exc
            continue
        if first == second:
            return first
    if last_error is not None and not path.exists():
        raise RouteRevisionError("manifest_unavailable", f"manifest is unavailable: {path}") from last_error
    raise RouteRevisionError("manifest_changed", "manifest changed while it was being read")


def _parse_manifest(raw: bytes, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RouteRevisionError("manifest_corrupt", f"invalid manifest JSON at {path}") from exc
    if not isinstance(payload, dict):
        raise RouteRevisionError("manifest_corrupt", "route manifest must contain a JSON object")
    return payload


def _validate_manifest_shape(payload: Mapping[str, Any]) -> None:
    if _integer(payload.get("schema_version"), "schema_version", minimum=1) != SCHEMA_VERSION:
        raise RouteRevisionError("schema_mismatch", "unsupported route manifest schema")
    if payload.get("simulation_only") is not True:
        raise RouteRevisionError("not_simulation_only", "route manifest is not simulation-only")
    validate_safe_component(payload.get("stream_id", ""), field="stream_id")
    validate_safe_component(payload.get("writer_id", ""), field="writer_id")
    _integer(payload.get("writer_generation"), "writer_generation", minimum=1)
    _integer(payload.get("target_revision"), "target_revision", minimum=0)
    _integer(payload.get("route_revision"), "route_revision", minimum=1)
    status = payload.get("status")
    if status not in {"active", "withdrawn"}:
        raise RouteRevisionError("invalid_manifest", "status must be active or withdrawn")
    _finite_timestamp(payload.get("published_unix_sec"), "published_unix_sec")
    _finite_timestamp(payload.get("observation_unix_sec"), "observation_unix_sec")
    if status == "withdrawn" and not str(payload.get("withdrawal_reason", "")).strip():
        raise RouteRevisionError("invalid_manifest", "withdrawn manifest lacks a reason")


def _check_age(
    payload: Mapping[str, Any],
    *,
    now_unix_sec: float,
    max_manifest_age_sec: float | None,
    max_observation_age_sec: float | None,
) -> None:
    now = _finite_timestamp(now_unix_sec, "now_unix_sec")
    limits = (
        ("manifest_stale", "published_unix_sec", max_manifest_age_sec),
        ("observation_stale", "observation_unix_sec", max_observation_age_sec),
    )
    for code, field, limit in limits:
        if limit is None:
            continue
        limit = _finite_nonnegative(limit, f"max_{field}_age_sec")
        age = now - float(payload[field])
        if age > limit:
            raise RouteRevisionError(code, f"{field} age {age:.3f}s exceeds {limit:.3f}s")
        if age < -1.0:
            raise RouteRevisionError("timestamp_in_future", f"{field} is {-age:.3f}s in the future")


def _contained_artifact(
    manifest_path: Path,
    revision_dir: Path,
    descriptor: Any,
    *,
    kind: str,
    route_revision: int,
) -> tuple[Path, str]:
    if not isinstance(descriptor, Mapping):
        raise RouteRevisionError("manifest_corrupt", f"{kind} descriptor is missing")
    relative_text = descriptor.get("relative_path")
    expected_hash = str(descriptor.get("sha256", ""))
    if not isinstance(relative_text, str) or not relative_text:
        raise RouteRevisionError("unsafe_path", f"{kind} relative_path is missing")
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise RouteRevisionError("unsafe_path", f"{kind} path must be a contained relative path")
    if _SHA256.fullmatch(expected_hash) is None:
        raise RouteRevisionError("manifest_corrupt", f"{kind} SHA-256 is invalid")

    expected_name = (
        f"route_{route_revision:06d}.csv"
        if kind == "route"
        else f"diagnostics_{route_revision:06d}.json"
    )
    candidate = manifest_path.parent / relative
    try:
        resolved = candidate.resolve(strict=True)
        root = revision_dir.resolve(strict=True)
    except OSError as exc:
        raise RouteRevisionError("artifact_unavailable", f"{kind} artifact is unavailable") from exc
    if candidate.is_symlink() or resolved.parent != root or resolved.name != expected_name:
        raise RouteRevisionError("unsafe_path", f"{kind} artifact escapes its revision directory")
    if not resolved.is_file():
        raise RouteRevisionError("artifact_unavailable", f"{kind} artifact is not a regular file")
    actual_hash = file_sha256(resolved)
    if actual_hash != expected_hash:
        raise RouteRevisionError("artifact_hash_mismatch", f"{kind} artifact hash mismatch")
    return resolved, actual_hash


def read_route_revision(
    manifest_path: Path,
    *,
    expected_stream_id: str | None = None,
    expected_writer_id: str | None = None,
    last_route_revision: int | None = None,
    last_manifest_sha256: str | None = None,
    require_contiguous_revision: bool = False,
    max_manifest_age_sec: float | None = None,
    max_observation_age_sec: float | None = None,
    now_unix_sec: float | None = None,
    verify_artifacts: bool = True,
) -> LoadedRouteRevision:
    """Load one stable manifest snapshot and validate its immutable artifacts.

    A repeated revision with the same manifest hash is returned with
    ``duplicate=True``.  Reusing a revision number for different bytes,
    rolling back, or (when requested) skipping a revision is rejected.
    """

    manifest_path = Path(manifest_path)
    raw = _read_stable_bytes(manifest_path)
    manifest_hash = _sha256_bytes(raw)
    payload = _parse_manifest(raw, manifest_path)
    _validate_manifest_shape(payload)

    stream_id = str(payload["stream_id"])
    writer_id = str(payload["writer_id"])
    if expected_stream_id is not None and stream_id != expected_stream_id:
        raise RouteRevisionError(
            "wrong_stream", f"expected stream {expected_stream_id!r}, got {stream_id!r}"
        )
    if expected_writer_id is not None and writer_id != expected_writer_id:
        raise RouteRevisionError(
            "wrong_writer", f"expected writer {expected_writer_id!r}, got {writer_id!r}"
        )

    revision = int(payload["route_revision"])
    duplicate = False
    if last_route_revision is not None:
        if revision < last_route_revision:
            raise RouteRevisionError(
                "revision_rollback", f"route revision rolled back from {last_route_revision} to {revision}"
            )
        if revision == last_route_revision:
            if last_manifest_sha256 is not None and manifest_hash != last_manifest_sha256:
                raise RouteRevisionError(
                    "duplicate_revision_conflict",
                    f"route revision {revision} was reused for different manifest bytes",
                )
            duplicate = True
        elif require_contiguous_revision and revision != last_route_revision + 1:
            raise RouteRevisionError(
                "revision_gap",
                f"expected route revision {last_route_revision + 1}, got {revision}",
            )

    _check_age(
        payload,
        now_unix_sec=time.time() if now_unix_sec is None else now_unix_sec,
        max_manifest_age_sec=max_manifest_age_sec,
        max_observation_age_sec=max_observation_age_sec,
    )

    route_path: Path | None = None
    diagnostics_path: Path | None = None
    if payload["status"] == "active" and verify_artifacts:
        revision_dir = manifest_path.parent / f"{manifest_path.stem}_revisions" / stream_id
        route_path, _ = _contained_artifact(
            manifest_path,
            revision_dir,
            payload.get("route"),
            kind="route",
            route_revision=revision,
        )
        diagnostics_path, _ = _contained_artifact(
            manifest_path,
            revision_dir,
            payload.get("diagnostics"),
            kind="diagnostics",
            route_revision=revision,
        )

    return LoadedRouteRevision(
        manifest_path=manifest_path,
        manifest=payload,
        manifest_sha256=manifest_hash,
        route_path=route_path,
        diagnostics_path=diagnostics_path,
        duplicate=duplicate,
    )


def read_committed_revision(
    manifest_path: Path,
    expected_stream_id: str | None = None,
    now_unix_sec: float | None = None,
    max_manifest_age_sec: float | None = None,
    max_observation_age_sec: float | None = None,
) -> LoadedRouteRevision:
    """Read an authoritative committed snapshot without reader history.

    This intentionally small public entry point is suitable for loading the
    initial simulation route before ROS preflight.  Active artifact paths and
    hashes are fully validated; a withdrawn snapshot has no route path and
    carries its reason in :attr:`LoadedRouteRevision.reason`.
    """

    return read_route_revision(
        manifest_path,
        expected_stream_id=expected_stream_id,
        now_unix_sec=now_unix_sec,
        max_manifest_age_sec=max_manifest_age_sec,
        max_observation_age_sec=max_observation_age_sec,
    )


class RouteRevisionStore:
    """Single-writer publisher for immutable route revisions."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        stream_id: str,
        writer_id: str,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.stream_id = validate_safe_component(stream_id, field="stream_id")
        self.writer_id = validate_safe_component(writer_id, field="writer_id")
        self.now_fn = now_fn
        self.revision_dir = (
            self.manifest_path.parent
            / f"{self.manifest_path.stem}_revisions"
            / self.stream_id
        )

    def _current(self) -> LoadedRouteRevision | None:
        if not self.manifest_path.exists():
            return None
        return read_route_revision(
            self.manifest_path,
            expected_stream_id=self.stream_id,
            verify_artifacts=False,
        )

    def _ownership(
        self,
        current: LoadedRouteRevision | None,
        *,
        takeover: bool,
    ) -> tuple[int, dict[str, Any] | None]:
        if current is None:
            if takeover:
                raise RouteRevisionError("invalid_takeover", "cannot take over a new route stream")
            return 1, None
        previous_id = current.writer_id
        previous_generation = current.writer_generation
        if previous_id == self.writer_id:
            return previous_generation, current.manifest.get("writer_takeover")
        if not takeover:
            raise RouteRevisionError(
                "writer_conflict",
                f"stream is owned by writer {previous_id!r}; explicit takeover is required",
            )
        return previous_generation + 1, {
            "previous_writer_id": previous_id,
            "previous_writer_generation": previous_generation,
            "takeover_unix_sec": _finite_timestamp(self.now_fn(), "takeover_unix_sec"),
        }

    def _next_revision(self, current: LoadedRouteRevision | None) -> int:
        highest = current.route_revision if current is not None else 0
        if self.revision_dir.exists():
            for path in self.revision_dir.iterdir():
                match = re.fullmatch(r"(?:route|diagnostics)_(\d{6})\.(?:csv|json)", path.name)
                if match:
                    highest = max(highest, int(match.group(1)))
        return highest + 1

    def _base_manifest(
        self,
        *,
        current: LoadedRouteRevision | None,
        takeover: bool,
        target_revision: int,
        observation_unix_sec: float,
    ) -> dict[str, Any]:
        writer_generation, takeover_record = self._ownership(current, takeover=takeover)
        target_revision = _integer(target_revision, "target_revision", minimum=0)
        if current is not None and target_revision < current.target_revision:
            raise RouteRevisionError(
                "target_revision_rollback",
                f"target revision rolled back from {current.target_revision} to {target_revision}",
            )
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "simulation_only": True,
            "stream_id": self.stream_id,
            "writer_id": self.writer_id,
            "writer_generation": writer_generation,
            "target_revision": target_revision,
            "route_revision": self._next_revision(current),
            "published_unix_sec": _finite_timestamp(self.now_fn(), "published_unix_sec"),
            "observation_unix_sec": _finite_timestamp(
                observation_unix_sec, "observation_unix_sec"
            ),
        }
        if takeover_record is not None:
            payload["writer_takeover"] = takeover_record
        return payload

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.manifest_path.parent).as_posix()

    def publish_active(
        self,
        route_csv_text: str,
        diagnostics: Mapping[str, Any],
        *,
        target_revision: int,
        observation_unix_sec: float,
        source_robot_pose: Mapping[str, Any],
        target: Mapping[str, Any],
        evidence: Mapping[str, Any],
        previous_route_length_m: float,
        new_route_length_m: float,
        safety_diagnostics: Mapping[str, Any],
        takeover: bool = False,
    ) -> LoadedRouteRevision:
        """Publish artifacts first and atomically activate their manifest last."""

        if not isinstance(route_csv_text, str) or not route_csv_text.strip():
            raise RouteRevisionError("invalid_payload", "route_csv_text must be non-empty")
        current = self._current()
        payload = self._base_manifest(
            current=current,
            takeover=takeover,
            target_revision=target_revision,
            observation_unix_sec=observation_unix_sec,
        )
        revision = int(payload["route_revision"])
        route_path = self.revision_dir / f"route_{revision:06d}.csv"
        diagnostics_path = self.revision_dir / f"diagnostics_{revision:06d}.json"
        route_bytes = route_csv_text.encode("utf-8")
        diagnostics_bytes = _json_bytes(dict(diagnostics))

        # These two exclusive writes intentionally precede the only
        # authoritative publication operation below.
        _exclusive_write(route_path, route_bytes)
        _exclusive_write(diagnostics_path, diagnostics_bytes)
        # Make both immutable directory entries durable before publishing the
        # manifest that points at them.
        _fsync_directory(self.revision_dir)
        _fsync_directory(self.revision_dir.parent)

        payload.update(
            {
                "status": "active",
                "source_robot_pose": dict(source_robot_pose),
                "target": dict(target),
                "evidence": dict(evidence),
                "previous_route_length_m": _finite_nonnegative(
                    previous_route_length_m, "previous_route_length_m"
                ),
                "new_route_length_m": _finite_nonnegative(
                    new_route_length_m, "new_route_length_m"
                ),
                "safety_diagnostics": dict(safety_diagnostics),
                "route": {
                    "relative_path": self._relative(route_path),
                    "sha256": _sha256_bytes(route_bytes),
                },
                "diagnostics": {
                    "relative_path": self._relative(diagnostics_path),
                    "sha256": _sha256_bytes(diagnostics_bytes),
                },
            }
        )
        _atomic_replace(self.manifest_path, _json_bytes(payload))
        return read_route_revision(
            self.manifest_path,
            expected_stream_id=self.stream_id,
            expected_writer_id=self.writer_id,
        )

    def withdraw(
        self,
        reason: str,
        *,
        target_revision: int | None = None,
        observation_unix_sec: float | None = None,
        takeover: bool = False,
    ) -> LoadedRouteRevision:
        """Publish a monotonic fail-closed withdrawal without route artifacts."""

        reason = str(reason).strip()
        if not reason:
            raise RouteRevisionError("invalid_payload", "withdrawal reason must be non-empty")
        current = self._current()
        if target_revision is None:
            target_revision = current.target_revision if current is not None else 0
        if observation_unix_sec is None:
            observation_unix_sec = (
                float(current.manifest["observation_unix_sec"])
                if current is not None
                else float(self.now_fn())
            )
        payload = self._base_manifest(
            current=current,
            takeover=takeover,
            target_revision=target_revision,
            observation_unix_sec=observation_unix_sec,
        )
        payload.update(
            {
                "status": "withdrawn",
                "withdrawal_reason": reason,
                "source_robot_pose": (
                    current.manifest.get("source_robot_pose", {}) if current is not None else {}
                ),
                "target": current.manifest.get("target", {}) if current is not None else {},
                "evidence": current.manifest.get("evidence", {}) if current is not None else {},
                "previous_route_length_m": (
                    current.manifest.get("new_route_length_m", 0.0)
                    if current is not None
                    else 0.0
                ),
                "new_route_length_m": 0.0,
                "safety_diagnostics": (
                    current.manifest.get("safety_diagnostics", {}) if current is not None else {}
                ),
            }
        )
        _atomic_replace(self.manifest_path, _json_bytes(payload))
        return read_route_revision(
            self.manifest_path,
            expected_stream_id=self.stream_id,
            expected_writer_id=self.writer_id,
            verify_artifacts=False,
        )
