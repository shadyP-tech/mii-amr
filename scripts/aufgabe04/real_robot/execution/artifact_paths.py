"""Canonical filesystem boundary for autonomous child-route artifacts.

The route planner returns strings that may be relative to the workstation's
current directory.  Hashing, permit construction, and child argv must all use
one canonical identity for those files.  This module performs that conversion
without importing ROS or authorizing motion.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path


SEALED_ARTIFACT_FIELDS = frozenset(
    {"route_csv", "diagnostics_json", "route_certificate_json"}
)


class AutonomousArtifactPathError(ValueError):
    """A child artifact path cannot be admitted fail-closed."""


@dataclass(frozen=True)
class CanonicalChildArtifactPaths:
    """Canonical normal files admitted for one child process."""

    session_root: Path
    route_csv: Path
    diagnostics_json: Path
    route_certificate_json: Path


def resolve_child_artifact_paths(
    *,
    session_root: str | os.PathLike[str],
    sealed: Mapping[str, str | os.PathLike[str]],
) -> CanonicalChildArtifactPaths:
    """Resolve and admit sealed route artifacts for one autonomous child.

    Relative inputs are interpreted against the process working directory,
    matching the planner and CLI boundary.  ``Path.resolve(strict=True)`` is
    deliberately used for both the root and artifacts so platform aliases
    such as macOS ``/var`` -> ``/private/var`` converge on one identity.
    Final-component artifact symlinks are rejected before resolution.  The
    artifacts are not required to live below ``session_root`` because sealed
    recovery and test fixtures may bind externally produced files; exact
    content hashes and permit validation still bind the child to those bytes.
    """

    root_candidate = _path_value(session_root, "session_root")
    try:
        canonical_root = root_candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise AutonomousArtifactPathError(
            f"session_root is unavailable: {root_candidate}"
        ) from exc
    if not canonical_root.is_dir():
        raise AutonomousArtifactPathError(
            f"session_root must resolve to a directory: {root_candidate}"
        )

    if not isinstance(sealed, Mapping):
        raise AutonomousArtifactPathError("sealed must be a mapping")
    observed_fields = frozenset(sealed)
    if observed_fields != SEALED_ARTIFACT_FIELDS:
        missing = SEALED_ARTIFACT_FIELDS - observed_fields
        unexpected = observed_fields - SEALED_ARTIFACT_FIELDS
        details = []
        if missing:
            details.append(
                "missing=" + ",".join(sorted(str(field) for field in missing))
            )
        if unexpected:
            details.append(
                "unexpected="
                + ",".join(sorted(str(field) for field in unexpected))
            )
        raise AutonomousArtifactPathError(
            "sealed artifact fields mismatch: " + "; ".join(details)
        )

    resolved = {
        field: resolve_normal_artifact_path(
            sealed[field],
            label=field,
        )
        for field in SEALED_ARTIFACT_FIELDS
    }
    return CanonicalChildArtifactPaths(
        session_root=canonical_root,
        route_csv=resolved["route_csv"],
        diagnostics_json=resolved["diagnostics_json"],
        route_certificate_json=resolved["route_certificate_json"],
    )


def resolve_normal_artifact_path(
    path: str | os.PathLike[str],
    *,
    label: str = "artifact",
) -> Path:
    """Return one canonical existing normal, nonsymlink artifact path."""

    return _resolve_normal_file(_path_value(path, label), label)


def _path_value(value: object, name: str) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise AutonomousArtifactPathError(
            f"{name} must be a filesystem path"
        ) from exc
    if not isinstance(raw, str) or not raw:
        raise AutonomousArtifactPathError(
            f"{name} must be a non-empty text filesystem path"
        )
    return Path(raw)


def _resolve_normal_file(
    candidate: Path,
    name: str,
) -> Path:
    if candidate.is_symlink():
        raise AutonomousArtifactPathError(
            f"{name} must not be a symlink: {candidate}"
        )
    try:
        canonical = candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise AutonomousArtifactPathError(
            f"{name} is unavailable: {candidate}"
        ) from exc
    if not canonical.is_file():
        raise AutonomousArtifactPathError(
            f"{name} must resolve to a normal file: {canonical}"
        )
    return canonical
