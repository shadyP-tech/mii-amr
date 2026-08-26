"""Immutable evidence contract for autonomous coverage checkpoints.

The manifest binds a completed coverage-leg cursor to the exact files needed
to inspect or resume the survey later.  It is deliberately ROS-free and is
never a motion permit: loading or admitting one cannot authorize a robot run.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import Mapping

from scripts.aufgabe04.artifacts.content_store import (
    ContentStoreError,
    load_content_hashed_json,
    payload_sha256,
    write_content_hashed_json,
)
from scripts.aufgabe04.real_robot.mission.modes import AutonomousRunMode


AUTONOMOUS_SESSION_MANIFEST_SCHEMA_VERSION = 1
AUTONOMOUS_SESSION_MANIFEST_KIND = "autonomous_session_checkpoint"
AUTONOMOUS_SESSION_MANIFEST_HASH_FIELD = "manifest_sha256"
COVERAGE_LEG_CHECKPOINT_COMPLETE = "coverage_leg_checkpoint_complete"
COVERAGE_SURVEY_TERMINAL_CHECKPOINT = "coverage_survey_terminal_checkpoint"

AUTONOMOUS_CHECKPOINT_RUN_MODES = frozenset(
    {
        AutonomousRunMode.EXECUTE_COVERAGE_CHECKPOINT.value,
        AutonomousRunMode.EXECUTE_COVERAGE_ONLY.value,
        AutonomousRunMode.EXECUTE_EXACT_TWO_CAMERA.value,
        AutonomousRunMode.EXECUTE_FULL.value,
        AutonomousRunMode.RESUME_NEXT_COVERAGE_LEG.value,
    }
)
AUTONOMOUS_CHECKPOINT_STATUSES = frozenset(
    {
        COVERAGE_LEG_CHECKPOINT_COMPLETE,
        COVERAGE_SURVEY_TERMINAL_CHECKPOINT,
    }
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FILE_REFERENCE_FIELDS = frozenset({"path", "sha256"})
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_kind",
        "session_id",
        "run_mode",
        "status",
        "robot_id",
        "robot_profile_sha256",
        "calibration_profile_sha256",
        "physical_site_sha256",
        "map_bundle_sha256",
        "config_sha256",
        "completed_coverage_legs",
        "next_viewpoint_id",
        "coverage_plan",
        "coverage_progress",
        "survey_summary",
        "stand_registry",
        "lidar_observer_summary",
        "parent_checkpoint",
        "motion_authorized",
    }
)


class AutonomousSessionManifestError(ValueError):
    """Checkpoint-manifest error with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ArtifactFileReference:
    """Exact byte-level binding to one normal, non-symlink file."""

    path: str
    sha256: str


@dataclass(frozen=True)
class ParentCheckpointReference:
    """Content-hash binding to an earlier checkpoint manifest."""

    path: str
    sha256: str


@dataclass(frozen=True)
class AutonomousSessionManifest:
    """Evidence-only cursor after one or more completed coverage legs."""

    schema_version: int
    manifest_kind: str
    session_id: str
    run_mode: str
    status: str
    robot_id: str
    robot_profile_sha256: str
    calibration_profile_sha256: str
    physical_site_sha256: str
    map_bundle_sha256: str
    config_sha256: str
    completed_coverage_legs: int
    next_viewpoint_id: str | None
    coverage_plan: ArtifactFileReference
    coverage_progress: ArtifactFileReference
    survey_summary: ArtifactFileReference
    stand_registry: ArtifactFileReference
    lidar_observer_summary: ArtifactFileReference
    parent_checkpoint: ParentCheckpointReference | None = None
    motion_authorized: bool = False


@dataclass(frozen=True)
class PublishedAutonomousCheckpoint:
    """Result of publishing one immutable checkpoint snapshot."""

    manifest_path: Path
    manifest_sha256: str
    manifest: AutonomousSessionManifest


def artifact_file_reference(path: Path) -> ArtifactFileReference:
    """Create a canonical reference after checking and hashing ``path``."""

    source = _canonical_existing_file(Path(path), "artifact")
    return ArtifactFileReference(path=str(source), sha256=_file_sha256(source))


def parent_checkpoint_reference(path: Path) -> ParentCheckpointReference:
    """Create a reference to an already admitted parent checkpoint."""

    source = _canonical_existing_file(Path(path), "parent checkpoint")
    parent = admit_autonomous_session_manifest(source)
    return ParentCheckpointReference(
        path=str(source), sha256=autonomous_session_manifest_sha256(parent)
    )


def autonomous_session_manifest_payload(
    manifest: AutonomousSessionManifest,
) -> dict[str, object]:
    """Return the canonical unhashed payload after structural validation."""

    validate_autonomous_session_manifest(manifest)
    return {
        "schema_version": manifest.schema_version,
        "manifest_kind": manifest.manifest_kind,
        "session_id": manifest.session_id,
        "run_mode": manifest.run_mode,
        "status": manifest.status,
        "robot_id": manifest.robot_id,
        "robot_profile_sha256": manifest.robot_profile_sha256,
        "calibration_profile_sha256": manifest.calibration_profile_sha256,
        "physical_site_sha256": manifest.physical_site_sha256,
        "map_bundle_sha256": manifest.map_bundle_sha256,
        "config_sha256": manifest.config_sha256,
        "completed_coverage_legs": manifest.completed_coverage_legs,
        "next_viewpoint_id": manifest.next_viewpoint_id,
        "coverage_plan": _reference_payload(manifest.coverage_plan),
        "coverage_progress": _reference_payload(manifest.coverage_progress),
        "survey_summary": _reference_payload(manifest.survey_summary),
        "stand_registry": _reference_payload(manifest.stand_registry),
        "lidar_observer_summary": _reference_payload(
            manifest.lidar_observer_summary
        ),
        "parent_checkpoint": (
            None
            if manifest.parent_checkpoint is None
            else _reference_payload(manifest.parent_checkpoint)
        ),
        "motion_authorized": manifest.motion_authorized,
    }


def autonomous_session_manifest_sha256(
    manifest: AutonomousSessionManifest,
) -> str:
    """Hash the evidence payload without granting it motion authority."""

    return payload_sha256(autonomous_session_manifest_payload(manifest))


def validate_autonomous_session_manifest(
    manifest: AutonomousSessionManifest,
) -> None:
    """Validate the immutable structure without requiring live artifacts."""

    if not isinstance(manifest, AutonomousSessionManifest):
        raise AutonomousSessionManifestError(
            "invalid_manifest", "manifest must be an AutonomousSessionManifest"
        )
    if (
        type(manifest.schema_version) is not int
        or manifest.schema_version
        != AUTONOMOUS_SESSION_MANIFEST_SCHEMA_VERSION
    ):
        raise AutonomousSessionManifestError(
            "schema_mismatch", "unsupported autonomous session manifest schema"
        )
    if manifest.manifest_kind != AUTONOMOUS_SESSION_MANIFEST_KIND:
        raise AutonomousSessionManifestError(
            "invalid_manifest",
            f"manifest_kind must be {AUTONOMOUS_SESSION_MANIFEST_KIND!r}",
        )
    _require_safe_id(manifest.session_id, "session_id")
    _require_safe_id(manifest.robot_id, "robot_id")
    if manifest.run_mode not in AUTONOMOUS_CHECKPOINT_RUN_MODES:
        raise AutonomousSessionManifestError(
            "invalid_manifest",
            f"run_mode must be one of {sorted(AUTONOMOUS_CHECKPOINT_RUN_MODES)}",
        )
    if manifest.status not in AUTONOMOUS_CHECKPOINT_STATUSES:
        raise AutonomousSessionManifestError(
            "invalid_manifest",
            f"status must be one of {sorted(AUTONOMOUS_CHECKPOINT_STATUSES)}",
        )
    for name in (
        "robot_profile_sha256",
        "calibration_profile_sha256",
        "physical_site_sha256",
        "map_bundle_sha256",
        "config_sha256",
    ):
        _require_sha256(getattr(manifest, name), name)
    if (
        type(manifest.completed_coverage_legs) is not int
        or manifest.completed_coverage_legs < 1
    ):
        raise AutonomousSessionManifestError(
            "invalid_cursor",
            "coverage checkpoint requires at least one completed coverage leg",
        )
    _validate_checkpoint_cursor(
        status=manifest.status,
        next_viewpoint_id=manifest.next_viewpoint_id,
    )
    for name in (
        "coverage_plan",
        "coverage_progress",
        "survey_summary",
        "stand_registry",
        "lidar_observer_summary",
    ):
        _validate_file_reference(getattr(manifest, name), name)
    if manifest.parent_checkpoint is not None:
        _validate_parent_reference(manifest.parent_checkpoint)
    if manifest.motion_authorized is not False:
        raise AutonomousSessionManifestError(
            "invalid_manifest", "checkpoint evidence cannot authorize motion"
        )


def verify_autonomous_session_manifest_artifacts(
    manifest: AutonomousSessionManifest,
) -> None:
    """Live-rehash every artifact and recursively validate its parent chain."""

    _verify_manifest_artifacts(manifest, visited_parent_paths=set())


def write_autonomous_session_manifest(
    path: Path, manifest: AutonomousSessionManifest
) -> str:
    """Admit and atomically publish an immutable content-hashed checkpoint."""

    verify_autonomous_session_manifest_artifacts(manifest)
    try:
        return write_content_hashed_json(
            Path(path),
            autonomous_session_manifest_payload(manifest),
            hash_field=AUTONOMOUS_SESSION_MANIFEST_HASH_FIELD,
        )
    except ContentStoreError as exc:
        raise AutonomousSessionManifestError(exc.code, str(exc)) from exc


def load_autonomous_session_manifest(path: Path) -> AutonomousSessionManifest:
    """Hash-check and structurally load a checkpoint without live admission."""

    try:
        payload = load_content_hashed_json(
            Path(path), hash_field=AUTONOMOUS_SESSION_MANIFEST_HASH_FIELD
        )
    except ContentStoreError as exc:
        raise AutonomousSessionManifestError(exc.code, str(exc)) from exc
    return _manifest_from_payload(payload)


def admit_autonomous_session_manifest(path: Path) -> AutonomousSessionManifest:
    """Load a checkpoint and fail closed unless all bound bytes still exist."""

    source = _canonical_existing_file(Path(path), "checkpoint manifest")
    manifest = load_autonomous_session_manifest(source)
    _verify_manifest_artifacts(manifest, visited_parent_paths={source})
    return manifest


def publish_coverage_checkpoint(
    *,
    session_root: Path,
    session_id: str,
    run_mode: str,
    robot_id: str,
    robot_profile_sha256: str,
    calibration_profile_sha256: str,
    physical_site_sha256: str,
    map_bundle_sha256: str,
    config_sha256: str,
    completed_coverage_legs: int,
    next_viewpoint_id: str | None,
    coverage_plan_path: Path,
    coverage_progress_path: Path,
    survey_summary_path: Path,
    stand_registry_path: Path,
    lidar_observer_summary_path: Path,
    parent_checkpoint_path: Path | None = None,
    status: str = COVERAGE_LEG_CHECKPOINT_COMPLETE,
) -> PublishedAutonomousCheckpoint:
    """Snapshot mutable survey state and publish its evidence-only manifest."""

    if status not in AUTONOMOUS_CHECKPOINT_STATUSES:
        raise AutonomousSessionManifestError(
            "invalid_manifest",
            f"status must be one of {sorted(AUTONOMOUS_CHECKPOINT_STATUSES)}",
        )
    _validate_checkpoint_cursor(
        status=status,
        next_viewpoint_id=next_viewpoint_id,
    )
    root = Path(session_root).resolve(strict=True)
    checkpoint_root = (
        root
        / "checkpoints"
        / f"coverage_leg_{completed_coverage_legs:03d}"
    )
    try:
        checkpoint_root.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise AutonomousSessionManifestError(
            "immutable_conflict",
            f"refusing to reuse checkpoint directory: {checkpoint_root}",
        ) from exc
    sources = {
        "coverage_plan": Path(coverage_plan_path),
        "coverage_progress": Path(coverage_progress_path),
        "survey_summary": Path(survey_summary_path),
        "stand_registry": Path(stand_registry_path),
        "lidar_observer_summary": Path(lidar_observer_summary_path),
    }
    snapshots: dict[str, Path] = {}
    for name, source in sources.items():
        destination = checkpoint_root / f"{name}.json"
        _copy_immutable_snapshot(source, destination, name)
        snapshots[name] = destination

    parent = (
        None
        if parent_checkpoint_path is None
        else parent_checkpoint_reference(parent_checkpoint_path)
    )
    manifest = AutonomousSessionManifest(
        schema_version=AUTONOMOUS_SESSION_MANIFEST_SCHEMA_VERSION,
        manifest_kind=AUTONOMOUS_SESSION_MANIFEST_KIND,
        session_id=session_id,
        run_mode=run_mode,
        status=status,
        robot_id=robot_id,
        robot_profile_sha256=robot_profile_sha256,
        calibration_profile_sha256=calibration_profile_sha256,
        physical_site_sha256=physical_site_sha256,
        map_bundle_sha256=map_bundle_sha256,
        config_sha256=config_sha256,
        completed_coverage_legs=completed_coverage_legs,
        next_viewpoint_id=next_viewpoint_id,
        coverage_plan=artifact_file_reference(snapshots["coverage_plan"]),
        coverage_progress=artifact_file_reference(
            snapshots["coverage_progress"]
        ),
        survey_summary=artifact_file_reference(snapshots["survey_summary"]),
        stand_registry=artifact_file_reference(snapshots["stand_registry"]),
        lidar_observer_summary=artifact_file_reference(
            snapshots["lidar_observer_summary"]
        ),
        parent_checkpoint=parent,
        motion_authorized=False,
    )
    manifest_path = checkpoint_root / "manifest.json"
    digest = write_autonomous_session_manifest(manifest_path, manifest)
    return PublishedAutonomousCheckpoint(manifest_path, digest, manifest)


def _verify_manifest_artifacts(
    manifest: AutonomousSessionManifest,
    *,
    visited_parent_paths: set[Path],
) -> None:
    validate_autonomous_session_manifest(manifest)
    for name in (
        "coverage_plan",
        "coverage_progress",
        "survey_summary",
        "stand_registry",
        "lidar_observer_summary",
    ):
        reference = getattr(manifest, name)
        source = _canonical_existing_file(Path(reference.path), name)
        actual = _file_sha256(source)
        if actual != reference.sha256:
            raise AutonomousSessionManifestError(
                "hash_mismatch",
                f"{name} hash mismatch: expected {reference.sha256}, got {actual}",
            )

    parent_reference = manifest.parent_checkpoint
    if parent_reference is None:
        return
    parent_path = _canonical_existing_file(
        Path(parent_reference.path), "parent checkpoint"
    )
    if parent_path in visited_parent_paths:
        raise AutonomousSessionManifestError(
            "provenance_mismatch", "parent checkpoint chain contains a cycle"
        )
    parent = load_autonomous_session_manifest(parent_path)
    actual_parent_hash = autonomous_session_manifest_sha256(parent)
    if actual_parent_hash != parent_reference.sha256:
        raise AutonomousSessionManifestError(
            "provenance_mismatch",
            "parent checkpoint manifest hash does not match its reference",
        )
    _verify_manifest_artifacts(
        parent,
        visited_parent_paths=visited_parent_paths | {parent_path},
    )


def _manifest_from_payload(
    payload: Mapping[str, object],
) -> AutonomousSessionManifest:
    _require_exact_fields(payload, _MANIFEST_FIELDS, "checkpoint manifest")
    try:
        parent_payload = payload["parent_checkpoint"]
        manifest = AutonomousSessionManifest(
            schema_version=_integer(payload["schema_version"], "schema_version"),
            manifest_kind=_string(payload["manifest_kind"], "manifest_kind"),
            session_id=_string(payload["session_id"], "session_id"),
            run_mode=_string(payload["run_mode"], "run_mode"),
            status=_string(payload["status"], "status"),
            robot_id=_string(payload["robot_id"], "robot_id"),
            robot_profile_sha256=_string(
                payload["robot_profile_sha256"], "robot_profile_sha256"
            ),
            calibration_profile_sha256=_string(
                payload["calibration_profile_sha256"],
                "calibration_profile_sha256",
            ),
            physical_site_sha256=_string(
                payload["physical_site_sha256"], "physical_site_sha256"
            ),
            map_bundle_sha256=_string(
                payload["map_bundle_sha256"], "map_bundle_sha256"
            ),
            config_sha256=_string(payload["config_sha256"], "config_sha256"),
            completed_coverage_legs=_integer(
                payload["completed_coverage_legs"], "completed_coverage_legs"
            ),
            next_viewpoint_id=_optional_string(
                payload["next_viewpoint_id"], "next_viewpoint_id"
            ),
            coverage_plan=_file_reference(payload["coverage_plan"], "coverage_plan"),
            coverage_progress=_file_reference(
                payload["coverage_progress"], "coverage_progress"
            ),
            survey_summary=_file_reference(
                payload["survey_summary"], "survey_summary"
            ),
            stand_registry=_file_reference(
                payload["stand_registry"], "stand_registry"
            ),
            lidar_observer_summary=_file_reference(
                payload["lidar_observer_summary"], "lidar_observer_summary"
            ),
            parent_checkpoint=(
                None
                if parent_payload is None
                else _parent_reference(parent_payload)
            ),
            motion_authorized=_boolean(
                payload["motion_authorized"], "motion_authorized"
            ),
        )
    except KeyError as exc:
        raise AutonomousSessionManifestError(
            "artifact_corrupt", "checkpoint manifest is missing a field"
        ) from exc
    validate_autonomous_session_manifest(manifest)
    return manifest


def _file_reference(value: object, name: str) -> ArtifactFileReference:
    item = _mapping(value, name)
    _require_exact_fields(item, _FILE_REFERENCE_FIELDS, name)
    return ArtifactFileReference(
        path=_string(item["path"], f"{name}.path"),
        sha256=_string(item["sha256"], f"{name}.sha256"),
    )


def _parent_reference(value: object) -> ParentCheckpointReference:
    item = _mapping(value, "parent_checkpoint")
    _require_exact_fields(item, _FILE_REFERENCE_FIELDS, "parent_checkpoint")
    return ParentCheckpointReference(
        path=_string(item["path"], "parent_checkpoint.path"),
        sha256=_string(item["sha256"], "parent_checkpoint.sha256"),
    )


def _reference_payload(
    reference: ArtifactFileReference | ParentCheckpointReference,
) -> dict[str, object]:
    return {"path": reference.path, "sha256": reference.sha256}


def _validate_file_reference(
    reference: ArtifactFileReference, name: str
) -> None:
    if not isinstance(reference, ArtifactFileReference):
        raise AutonomousSessionManifestError(
            "invalid_manifest", f"{name} must be an ArtifactFileReference"
        )
    _require_stored_path(reference.path, f"{name}.path")
    _require_sha256(reference.sha256, f"{name}.sha256")


def _validate_parent_reference(reference: ParentCheckpointReference) -> None:
    if not isinstance(reference, ParentCheckpointReference):
        raise AutonomousSessionManifestError(
            "invalid_manifest",
            "parent_checkpoint must be a ParentCheckpointReference",
        )
    _require_stored_path(reference.path, "parent_checkpoint.path")
    _require_sha256(reference.sha256, "parent_checkpoint.sha256")


def _require_stored_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AutonomousSessionManifestError(
            "invalid_manifest", f"{name} must be a non-empty string"
        )
    path = Path(value)
    if (
        not path.is_absolute()
        or path != Path(os.path.normpath(str(path)))
        or path.is_symlink()
    ):
        raise AutonomousSessionManifestError(
            "invalid_manifest",
            f"{name} must be a canonical absolute non-symlink path",
        )
    return path


def _canonical_existing_file(path: Path, name: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise AutonomousSessionManifestError(
            "artifact_unavailable", f"{name} path must not be a symlink"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise AutonomousSessionManifestError(
            "artifact_unavailable", f"{name} is unavailable: {candidate}"
        ) from exc
    if not resolved.is_file():
        raise AutonomousSessionManifestError(
            "artifact_unavailable", f"{name} must be a normal file: {candidate}"
        )
    if candidate.is_absolute() and candidate != resolved:
        raise AutonomousSessionManifestError(
            "artifact_unavailable",
            f"{name} must not traverse symlinks or noncanonical components",
        )
    return resolved


def _file_sha256(path: Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise AutonomousSessionManifestError(
            "artifact_unavailable", f"artifact must be a normal file: {source}"
        )
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise AutonomousSessionManifestError(
            "artifact_unavailable", f"artifact is unavailable: {source}"
        ) from exc
    return digest.hexdigest()


def _copy_immutable_snapshot(source: Path, destination: Path, name: str) -> None:
    canonical_source = _canonical_existing_file(Path(source), name)
    try:
        data = canonical_source.read_bytes()
        with Path(destination).open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise AutonomousSessionManifestError(
            "artifact_unavailable",
            f"cannot publish immutable {name} snapshot: {exc}",
        ) from exc


def _require_safe_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise AutonomousSessionManifestError(
            "invalid_manifest", f"{name} is not a safe identifier"
        )
    return value


def _validate_checkpoint_cursor(
    *,
    status: str,
    next_viewpoint_id: object,
) -> None:
    if status == COVERAGE_LEG_CHECKPOINT_COMPLETE:
        if next_viewpoint_id is None:
            raise AutonomousSessionManifestError(
                "invalid_cursor",
                "resumable coverage-leg checkpoint requires a next viewpoint",
            )
        _require_safe_id(next_viewpoint_id, "next_viewpoint_id")
        return
    if status == COVERAGE_SURVEY_TERMINAL_CHECKPOINT:
        if next_viewpoint_id is not None:
            raise AutonomousSessionManifestError(
                "invalid_cursor",
                "terminal coverage-survey checkpoint must not have a next viewpoint",
            )
        return
    raise AutonomousSessionManifestError(
        "invalid_manifest",
        f"status must be one of {sorted(AUTONOMOUS_CHECKPOINT_STATUSES)}",
    )


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise AutonomousSessionManifestError(
            "invalid_manifest", f"{name} must be a lowercase SHA-256"
        )
    return value


def _require_exact_fields(
    payload: Mapping[str, object], expected: frozenset[str], name: str
) -> None:
    actual = frozenset(payload)
    if actual != expected:
        raise AutonomousSessionManifestError(
            "artifact_corrupt",
            f"{name} fields mismatch; "
            f"missing={sorted(expected - actual)} unknown={sorted(actual - expected)}",
        )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise AutonomousSessionManifestError(
            "artifact_corrupt", f"{name} must be an object"
        )
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise AutonomousSessionManifestError(
            "artifact_corrupt", f"{name} must be a string"
        )
    return value


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AutonomousSessionManifestError(
            "artifact_corrupt", f"{name} must be an integer"
        )
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise AutonomousSessionManifestError(
            "artifact_corrupt", f"{name} must be a boolean"
        )
    return value


__all__ = [
    "AUTONOMOUS_CHECKPOINT_RUN_MODES",
    "AUTONOMOUS_CHECKPOINT_STATUSES",
    "AUTONOMOUS_SESSION_MANIFEST_HASH_FIELD",
    "AUTONOMOUS_SESSION_MANIFEST_KIND",
    "AUTONOMOUS_SESSION_MANIFEST_SCHEMA_VERSION",
    "COVERAGE_LEG_CHECKPOINT_COMPLETE",
    "COVERAGE_SURVEY_TERMINAL_CHECKPOINT",
    "ArtifactFileReference",
    "AutonomousSessionManifest",
    "AutonomousSessionManifestError",
    "ParentCheckpointReference",
    "PublishedAutonomousCheckpoint",
    "admit_autonomous_session_manifest",
    "artifact_file_reference",
    "autonomous_session_manifest_payload",
    "autonomous_session_manifest_sha256",
    "load_autonomous_session_manifest",
    "parent_checkpoint_reference",
    "publish_coverage_checkpoint",
    "validate_autonomous_session_manifest",
    "verify_autonomous_session_manifest_artifacts",
    "write_autonomous_session_manifest",
]
