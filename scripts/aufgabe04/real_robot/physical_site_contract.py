"""Fail-closed physical-site identity and stand-count validation.

The physical-site descriptor is the canonical source for the number of stands
in a real Aufgabe 04 arena.  This module keeps descriptor parsing and binding
checks independent of ROS so the complete contract can be admitted before a
motion prompt is shown.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Mapping

from scripts.aufgabe04.navigation.map_io import (
    FrozenMapBundle,
    freeze_map_bundle,
    read_map_metadata,
    validate_frozen_map_bundle,
)
from scripts.aufgabe04.real_robot.hardware_profile import (
    RealRobotProfile,
    validate_real_robot_profile,
)


PHYSICAL_SITE_SCHEMA_VERSION = 1
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_SITE_FIELDS = frozenset(
    {
        "schema_version",
        "physical_site_id",
        "description",
        "recorded_date",
        "map_measurement",
        "station_setup",
    }
)
_MAP_MEASUREMENT_FIELDS = frozenset(
    {
        "semantic_map_id",
        "map_yaml",
        "map_yaml_sha256",
        "map_image",
        "map_image_sha256",
        "map_bundle_sha256",
    }
)
_STATION_SETUP_FIELDS = frozenset(
    {
        "expected_stand_count",
        "stand_coordinates_supplied",
        "placement",
        "orientation",
    }
)


class PhysicalSiteContractError(ValueError):
    """Physical-site admission failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class MapMeasurement:
    semantic_map_id: str
    map_yaml: str
    map_yaml_sha256: str
    map_image: str
    map_image_sha256: str
    map_bundle_sha256: str


@dataclass(frozen=True)
class StationSetup:
    expected_stand_count: int
    stand_coordinates_supplied: bool
    placement: str
    orientation: str


@dataclass(frozen=True)
class PhysicalSiteContract:
    schema_version: int
    physical_site_id: str
    description: str
    recorded_date: str
    map_measurement: MapMeasurement
    station_setup: StationSetup


@dataclass(frozen=True)
class ValidatedPhysicalSiteContract:
    """Fully checked inputs safe to propagate into pre-motion planning."""

    site: PhysicalSiteContract
    physical_site_path: Path
    physical_site_sha256: str
    map_yaml_path: Path
    map_image_path: Path
    map_bundle: FrozenMapBundle
    expected_stand_count: int


def load_physical_site(path: Path) -> PhysicalSiteContract:
    """Strictly load a versioned physical-site descriptor.

    Structural loading does not touch the referenced map.  Use
    :func:`validate_physical_site_contract` to admit the complete binding.
    """

    source = Path(path)
    payload = _load_json_object(source)
    _require_exact_fields(payload, _SITE_FIELDS, "physical site")
    map_payload = _mapping(payload["map_measurement"], "map_measurement")
    station_payload = _mapping(payload["station_setup"], "station_setup")
    _require_exact_fields(
        map_payload, _MAP_MEASUREMENT_FIELDS, "map_measurement"
    )
    _require_exact_fields(
        station_payload, _STATION_SETUP_FIELDS, "station_setup"
    )

    site = PhysicalSiteContract(
        schema_version=_integer(payload["schema_version"], "schema_version"),
        physical_site_id=_identifier(
            payload["physical_site_id"], "physical_site_id"
        ),
        description=_nonempty_string(payload["description"], "description"),
        recorded_date=_iso_date(payload["recorded_date"], "recorded_date"),
        map_measurement=MapMeasurement(
            semantic_map_id=_identifier(
                map_payload["semantic_map_id"],
                "map_measurement.semantic_map_id",
            ),
            map_yaml=_repository_relative_path(
                map_payload["map_yaml"],
                "map_measurement.map_yaml",
                suffixes=(".yaml", ".yml"),
            ),
            map_yaml_sha256=_sha256(
                map_payload["map_yaml_sha256"],
                "map_measurement.map_yaml_sha256",
            ),
            map_image=_repository_relative_path(
                map_payload["map_image"],
                "map_measurement.map_image",
                suffixes=(".pgm",),
            ),
            map_image_sha256=_sha256(
                map_payload["map_image_sha256"],
                "map_measurement.map_image_sha256",
            ),
            map_bundle_sha256=_sha256(
                map_payload["map_bundle_sha256"],
                "map_measurement.map_bundle_sha256",
            ),
        ),
        station_setup=StationSetup(
            expected_stand_count=_positive_integer(
                station_payload["expected_stand_count"],
                "station_setup.expected_stand_count",
            ),
            stand_coordinates_supplied=_boolean(
                station_payload["stand_coordinates_supplied"],
                "station_setup.stand_coordinates_supplied",
            ),
            placement=_nonempty_string(
                station_payload["placement"], "station_setup.placement"
            ),
            orientation=_nonempty_string(
                station_payload["orientation"], "station_setup.orientation"
            ),
        ),
    )
    if site.schema_version != PHYSICAL_SITE_SCHEMA_VERSION:
        raise PhysicalSiteContractError(
            "schema_mismatch",
            f"unsupported physical-site schema {site.schema_version!r}",
        )
    if source.stem != site.physical_site_id:
        raise PhysicalSiteContractError(
            "site_id_mismatch",
            "physical-site filename stem must equal physical_site_id",
        )
    return site


def physical_site_sha256(path: Path) -> str:
    """Return the SHA-256 of exact descriptor bytes without following symlinks."""

    return _file_sha256(Path(path), "physical-site descriptor")


def resolve_expected_stand_count(
    site: PhysicalSiteContract,
    requested_expected_stand_count: int | None,
) -> int:
    """Resolve an optional CLI count without permitting a site override."""

    canonical = _positive_integer(
        site.station_setup.expected_stand_count,
        "station_setup.expected_stand_count",
    )
    if requested_expected_stand_count is None:
        return canonical
    if (
        type(requested_expected_stand_count) is not int
        or requested_expected_stand_count <= 0
    ):
        raise PhysicalSiteContractError(
            "invalid_request", "requested expected stand count must be positive"
        )
    requested = requested_expected_stand_count
    if requested != canonical:
        raise PhysicalSiteContractError(
            "stand_count_mismatch",
            "requested expected stand count differs from the physical-site "
            f"contract: requested={requested} canonical={canonical}",
        )
    return canonical


def validate_physical_site_contract(
    physical_site_path: Path,
    *,
    profile: RealRobotProfile,
    requested_expected_stand_count: int | None = None,
    semantic_map_id: str | None = None,
    map_yaml: Path | None = None,
    map_bundle: FrozenMapBundle | None = None,
    repository_root: Path | None = None,
) -> ValidatedPhysicalSiteContract:
    """Validate the site, profile, requested map, and frozen map as one unit.

    Repository paths stored in the descriptor are resolved below the supplied
    repository root (the checked-out repository by default), never relative to
    the process working directory.
    """

    site_path = Path(physical_site_path)
    site = load_physical_site(site_path)
    site_digest = physical_site_sha256(site_path)
    if not isinstance(profile, RealRobotProfile):
        raise PhysicalSiteContractError(
            "profile_invalid", "profile must be an immutable RealRobotProfile"
        )
    try:
        validate_real_robot_profile(profile)
    except (TypeError, ValueError) as exc:
        raise PhysicalSiteContractError(
            "profile_invalid", f"invalid real-robot profile: {exc}"
        ) from exc
    if profile.physical_site_id != site.physical_site_id:
        raise PhysicalSiteContractError(
            "profile_site_id_mismatch",
            "real-robot profile physical_site_id differs from the descriptor",
        )
    if profile.physical_site_sha256 != site_digest:
        raise PhysicalSiteContractError(
            "profile_site_hash_mismatch",
            "real-robot profile physical_site_sha256 differs from exact "
            "descriptor bytes",
        )

    expected_count = resolve_expected_stand_count(
        site, requested_expected_stand_count
    )
    measurement = site.map_measurement
    if semantic_map_id is not None:
        requested_map_id = _identifier(semantic_map_id, "semantic_map_id")
        if requested_map_id != measurement.semantic_map_id:
            raise PhysicalSiteContractError(
                "semantic_map_mismatch",
                "requested semantic map differs from the physical-site contract",
            )

    root = _resolved_repository_root(repository_root)
    canonical_yaml = _resolve_declared_path(root, measurement.map_yaml)
    canonical_image = _resolve_declared_path(root, measurement.map_image)
    if map_yaml is not None:
        requested_yaml = _resolve_requested_path(root, Path(map_yaml), "map_yaml")
        if requested_yaml != canonical_yaml:
            raise PhysicalSiteContractError(
                "map_yaml_mismatch",
                "requested map YAML path differs from the physical-site contract",
            )

    if _file_sha256(canonical_yaml, "map YAML") != measurement.map_yaml_sha256:
        raise PhysicalSiteContractError(
            "map_yaml_hash_mismatch",
            "map YAML bytes differ from the physical-site contract",
        )
    if _file_sha256(canonical_image, "map image") != measurement.map_image_sha256:
        raise PhysicalSiteContractError(
            "map_image_hash_mismatch",
            "map image bytes differ from the physical-site contract",
        )
    try:
        referenced_image = read_map_metadata(canonical_yaml).image_path.resolve(
            strict=True
        )
    except (OSError, TypeError, ValueError) as exc:
        raise PhysicalSiteContractError(
            "map_invalid", f"cannot resolve the map YAML image: {exc}"
        ) from exc
    if referenced_image != canonical_image:
        raise PhysicalSiteContractError(
            "map_image_path_mismatch",
            "map YAML references an image other than the physical-site map image",
        )

    try:
        actual_bundle = freeze_map_bundle(
            canonical_yaml,
            semantic_map_id=measurement.semantic_map_id,
            planning_frame=profile.map_frame,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise PhysicalSiteContractError(
            "map_invalid", f"cannot freeze the physical-site map: {exc}"
        ) from exc
    if actual_bundle.bundle_sha256 != measurement.map_bundle_sha256:
        raise PhysicalSiteContractError(
            "map_bundle_hash_mismatch",
            "frozen map bundle differs from the physical-site contract",
        )

    if map_bundle is not None:
        if not isinstance(map_bundle, FrozenMapBundle):
            raise PhysicalSiteContractError(
                "map_bundle_invalid",
                "requested map bundle must be an immutable FrozenMapBundle",
            )
        try:
            validate_frozen_map_bundle(map_bundle)
        except (TypeError, ValueError) as exc:
            raise PhysicalSiteContractError(
                "map_bundle_invalid", f"requested map bundle is invalid: {exc}"
            ) from exc
        if map_bundle != actual_bundle:
            raise PhysicalSiteContractError(
                "map_bundle_mismatch",
                "requested frozen map bundle differs from the physical-site contract",
            )

    return ValidatedPhysicalSiteContract(
        site=site,
        physical_site_path=site_path.resolve(strict=True),
        physical_site_sha256=site_digest,
        map_yaml_path=canonical_yaml,
        map_image_path=canonical_image,
        map_bundle=actual_bundle,
        expected_stand_count=expected_count,
    )


def _load_json_object(path: Path) -> Mapping[str, object]:
    if path.is_symlink():
        raise PhysicalSiteContractError(
            "site_unavailable", "physical-site descriptor must not be a symlink"
        )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PhysicalSiteContractError(
            "site_unavailable", f"cannot read physical-site descriptor: {path}"
        ) from exc
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhysicalSiteContractError(
            "site_corrupt", "physical-site descriptor is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise PhysicalSiteContractError(
            "site_corrupt", "physical-site descriptor root must be an object"
        )
    return payload


def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PhysicalSiteContractError(
                "site_corrupt", f"duplicate JSON object key {key!r}"
            )
        result[key] = value
    return result


def _require_exact_fields(
    payload: Mapping[str, object], expected: frozenset[str], name: str
) -> None:
    actual = frozenset(payload)
    if actual != expected:
        raise PhysicalSiteContractError(
            "site_corrupt",
            f"{name} fields mismatch; missing={sorted(expected - actual)} "
            f"unknown={sorted(actual - expected)}",
        )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise PhysicalSiteContractError("site_corrupt", f"{name} must be an object")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise PhysicalSiteContractError("site_corrupt", f"{name} must be an integer")
    return value


def _positive_integer(value: object, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise PhysicalSiteContractError(
            "site_corrupt", f"{name} must be positive"
        )
    return result


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise PhysicalSiteContractError("site_corrupt", f"{name} must be boolean")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise PhysicalSiteContractError(
            "site_corrupt", f"{name} must be a non-empty trimmed string"
        )
    return value


def _identifier(value: object, name: str) -> str:
    result = _nonempty_string(value, name)
    if not _SAFE_ID.fullmatch(result):
        raise PhysicalSiteContractError(
            "site_corrupt", f"{name} is not a safe identifier"
        )
    return result


def _iso_date(value: object, name: str) -> str:
    result = _nonempty_string(value, name)
    try:
        parsed = date.fromisoformat(result)
    except ValueError as exc:
        raise PhysicalSiteContractError(
            "site_corrupt", f"{name} must use YYYY-MM-DD"
        ) from exc
    if parsed.isoformat() != result:
        raise PhysicalSiteContractError(
            "site_corrupt", f"{name} must use canonical YYYY-MM-DD"
        )
    return result


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise PhysicalSiteContractError(
            "site_corrupt", f"{name} must be a lowercase SHA-256"
        )
    return value


def _repository_relative_path(
    value: object, name: str, *, suffixes: tuple[str, ...]
) -> str:
    result = _nonempty_string(value, name)
    if "\\" in result:
        raise PhysicalSiteContractError(
            "site_corrupt", f"{name} must use repository-relative POSIX syntax"
        )
    pure = PurePosixPath(result)
    if (
        pure.is_absolute()
        or result != pure.as_posix()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.suffix.lower() not in suffixes
    ):
        raise PhysicalSiteContractError(
            "site_corrupt", f"{name} is not a safe repository-relative path"
        )
    return result


def _resolved_repository_root(repository_root: Path | None) -> Path:
    candidate = _REPOSITORY_ROOT if repository_root is None else Path(repository_root)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise PhysicalSiteContractError(
            "repository_unavailable", f"repository root is unavailable: {candidate}"
        ) from exc
    if not resolved.is_dir():
        raise PhysicalSiteContractError(
            "repository_unavailable", "repository root must be a directory"
        )
    return resolved


def _resolve_declared_path(root: Path, declared: str) -> Path:
    return _resolve_below_root(root, root / Path(PurePosixPath(declared)), declared)


def _resolve_requested_path(root: Path, path: Path, name: str) -> Path:
    candidate = path if path.is_absolute() else root / path
    return _resolve_below_root(root, candidate, name)


def _resolve_below_root(root: Path, path: Path, name: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise PhysicalSiteContractError(
            "path_unavailable", f"{name} is unavailable below the repository root"
        ) from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise PhysicalSiteContractError(
            "path_unavailable", f"{name} must resolve to a regular file"
        )
    return resolved


def _file_sha256(path: Path, name: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise PhysicalSiteContractError(
            "artifact_unavailable", f"{name} must be an available regular file"
        )
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise PhysicalSiteContractError(
            "artifact_unavailable", f"cannot read {name}: {path}"
        ) from exc
    return digest.hexdigest()


__all__ = [
    "PHYSICAL_SITE_SCHEMA_VERSION",
    "MapMeasurement",
    "PhysicalSiteContract",
    "PhysicalSiteContractError",
    "StationSetup",
    "ValidatedPhysicalSiteContract",
    "load_physical_site",
    "physical_site_sha256",
    "resolve_expected_stand_count",
    "validate_physical_site_contract",
]
