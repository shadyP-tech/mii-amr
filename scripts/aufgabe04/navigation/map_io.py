"""Stdlib-only ROS trinary map and PGM parsing for Aufgabe 04 navigation."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from scripts.aufgabe04.artifacts.content_store import (
    ContentStoreError,
    load_content_hashed_json,
    write_content_hashed_json,
)


CELL_FREE = 0
CELL_OCCUPIED = 1
CELL_UNKNOWN = 2
FROZEN_MAP_BUNDLE_SCHEMA_VERSION = 1

_SAFE_MAP_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_FRAME = re.compile(r"^[A-Za-z][A-Za-z0-9_/.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAP_BUNDLE_HASH_FIELD = "map_bundle_sha256"
_MAP_BUNDLE_FIELDS = frozenset(
    {
        "schema_version",
        "semantic_map_id",
        "planning_frame",
        "yaml_sha256",
        "image_sha256",
        "resolution",
        "origin",
        "negate",
        "occupied_thresh",
        "free_thresh",
        "mode",
        "width",
        "height",
        "maxval",
    }
)


@dataclass(frozen=True)
class MapMetadata:
    yaml_path: Path
    image_path: Path
    resolution: float
    origin: Tuple[float, float, float]
    negate: int
    occupied_thresh: float
    free_thresh: float
    mode: str


@dataclass(frozen=True)
class PgmImage:
    width: int
    height: int
    maxval: int
    pixels: Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class OccupancyGrid:
    metadata: MapMetadata
    width: int
    height: int
    cells: Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class FrozenMapBundle:
    """Path-independent identity of the exact map bytes used for planning.

    Paths are deliberately absent.  A copied map has the same identity when
    its YAML and referenced PGM bytes, parsed geometry, semantic map ID, and
    planning frame are identical.
    """

    schema_version: int
    semantic_map_id: str
    planning_frame: str
    yaml_sha256: str
    image_sha256: str
    resolution: float
    origin: Tuple[float, float, float]
    negate: int
    occupied_thresh: float
    free_thresh: float
    mode: str
    width: int
    height: int
    maxval: int

    @property
    def bundle_sha256(self) -> str:
        return frozen_map_bundle_sha256(self)

    @property
    def content_sha256(self) -> str:
        return self.bundle_sha256


class FrozenMapBundleError(ValueError):
    """Frozen-map descriptor error with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index]
    return line


def parse_yaml_scalar(text: str):
    text = text.strip()
    if not text:
        return ""
    if text[0] in {"'", '"'} or text.startswith("["):
        return ast.literal_eval(text)
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(char in text for char in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return text


def read_simple_yaml(path: Path) -> Dict[str, object]:
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read map YAML: {path}") from exc
    return _read_simple_yaml_bytes(path, raw)


def _read_simple_yaml_bytes(path: Path, raw: bytes) -> Dict[str, object]:
    data: Dict[str, object] = {}
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not UTF-8 YAML") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = strip_inline_comment(line).strip()
        if not stripped:
            continue
        if ":" not in stripped:
            raise ValueError(f"{path}:{line_number}: expected 'key: value'")
        key, value = stripped.split(":", 1)
        key = key.strip()
        if key in data:
            raise ValueError(f"{path}:{line_number}: duplicate key {key!r}")
        data[key] = parse_yaml_scalar(value)
    return data


def read_map_metadata(path: Path) -> MapMetadata:
    yaml_path = Path(path)
    try:
        raw = yaml_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read map YAML: {yaml_path}") from exc
    return _read_map_metadata_bytes(yaml_path, raw)


def _read_map_metadata_bytes(yaml_path: Path, raw: bytes) -> MapMetadata:
    data = _read_simple_yaml_bytes(yaml_path, raw)
    required = [
        "image",
        "resolution",
        "origin",
        "negate",
        "occupied_thresh",
        "free_thresh",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{yaml_path} is missing required field(s): {', '.join(missing)}")

    mode = str(data.get("mode", "trinary")).lower()
    if mode != "trinary":
        raise ValueError("only trinary maps are supported")

    origin = data["origin"]
    if not isinstance(origin, list) or len(origin) != 3:
        raise ValueError(f"{yaml_path} origin must be [x, y, yaw]")
    origin_tuple = (float(origin[0]), float(origin[1]), float(origin[2]))
    if abs(origin_tuple[2]) > 1e-12:
        raise ValueError("only zero-yaw map origins are supported")

    image_path = Path(str(data["image"]))
    if not image_path.is_absolute():
        image_path = yaml_path.parent / image_path

    return MapMetadata(
        yaml_path=yaml_path,
        image_path=image_path,
        resolution=float(data["resolution"]),
        origin=origin_tuple,
        negate=int(data["negate"]),
        occupied_thresh=float(data["occupied_thresh"]),
        free_thresh=float(data["free_thresh"]),
        mode=mode,
    )


def _next_pgm_token(data: bytes, index: int) -> Tuple[str, int]:
    length = len(data)
    while index < length:
        byte = data[index]
        if byte == ord("#"):
            while index < length and data[index] not in b"\r\n":
                index += 1
            continue
        if chr(byte).isspace():
            index += 1
            continue
        break
    if index >= length:
        raise ValueError("unexpected end of PGM header")

    start = index
    while index < length:
        byte = data[index]
        if byte == ord("#") or chr(byte).isspace():
            break
        index += 1
    return data[start:index].decode("ascii"), index


def _skip_pgm_whitespace_and_comments(data: bytes, index: int) -> int:
    length = len(data)
    while index < length:
        byte = data[index]
        if byte == ord("#"):
            while index < length and data[index] not in b"\r\n":
                index += 1
            continue
        if chr(byte).isspace():
            index += 1
            continue
        break
    return index


def read_pgm(path: Path) -> PgmImage:
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read map image: {path}") from exc
    return _read_pgm_bytes(path, data)


def _read_pgm_bytes(path: Path, data: bytes) -> PgmImage:
    magic, index = _next_pgm_token(data, 0)
    if magic not in {"P2", "P5"}:
        raise ValueError(f"{path} is not a P2/P5 PGM image")

    width_text, index = _next_pgm_token(data, index)
    height_text, index = _next_pgm_token(data, index)
    maxval_text, index = _next_pgm_token(data, index)
    width = int(width_text)
    height = int(height_text)
    maxval = int(maxval_text)
    if width <= 0 or height <= 0:
        raise ValueError(f"{path} has invalid dimensions")
    if maxval <= 0 or maxval > 255:
        raise ValueError(f"{path} uses unsupported maxval {maxval}")

    if magic == "P2":
        values: List[int] = []
        while len(values) < width * height:
            token, index = _next_pgm_token(data, index)
            values.append(int(token))
        if any(value < 0 or value > maxval for value in values):
            raise ValueError(f"{path} contains a pixel outside 0..{maxval}")
    else:
        index = _skip_pgm_whitespace_and_comments(data, index)
        expected = width * height
        values = list(data[index:index + expected])
        if len(values) != expected:
            raise ValueError(f"{path} has incomplete binary pixel data")

    rows = tuple(
        tuple(values[row_start:row_start + width])
        for row_start in range(0, width * height, width)
    )
    return PgmImage(width=width, height=height, maxval=maxval, pixels=rows)


def image_to_grid(image_col: int, image_row: int, height: int) -> Tuple[int, int]:
    return image_col, height - 1 - image_row


def pixel_to_cell(pixel: int, metadata: MapMetadata, maxval: int = 255) -> int:
    if metadata.negate:
        probability = pixel / maxval
    else:
        probability = (maxval - pixel) / maxval
    if probability >= metadata.occupied_thresh:
        return CELL_OCCUPIED
    if probability <= metadata.free_thresh:
        return CELL_FREE
    return CELL_UNKNOWN


def build_occupancy_grid(metadata: MapMetadata, image: PgmImage) -> OccupancyGrid:
    rows = [[CELL_UNKNOWN for _ in range(image.width)] for _ in range(image.height)]
    for image_row in range(image.height):
        grid_y = image.height - 1 - image_row
        for image_col in range(image.width):
            rows[grid_y][image_col] = pixel_to_cell(
                image.pixels[image_row][image_col],
                metadata,
                maxval=image.maxval,
            )
    return OccupancyGrid(
        metadata=metadata,
        width=image.width,
        height=image.height,
        cells=tuple(tuple(row) for row in rows),
    )


def load_occupancy_grid(path: Path) -> OccupancyGrid:
    metadata = read_map_metadata(Path(path))
    image = read_pgm(metadata.image_path)
    return build_occupancy_grid(metadata, image)


def load_occupancy_grid_with_bundle(
    path: Path,
    *,
    semantic_map_id: str,
    planning_frame: str,
) -> tuple[OccupancyGrid, FrozenMapBundle]:
    """Load a grid and descriptor from the same immutable byte snapshots.

    This is the TOCTOU-safe entry point for future survey and route planning:
    both the grid and the returned hashes are derived from one YAML read and
    one image read.
    """

    yaml_path = Path(path)
    try:
        yaml_bytes = yaml_path.read_bytes()
    except OSError as exc:
        raise FrozenMapBundleError(
            "map_unavailable", f"cannot read map YAML: {yaml_path}"
        ) from exc
    try:
        metadata = _read_map_metadata_bytes(yaml_path, yaml_bytes)
        image_bytes = metadata.image_path.read_bytes()
        image = _read_pgm_bytes(metadata.image_path, image_bytes)
    except OSError as exc:
        raise FrozenMapBundleError(
            "map_unavailable", f"cannot read referenced map image: {metadata.image_path}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise FrozenMapBundleError("invalid_map", str(exc)) from exc

    bundle = FrozenMapBundle(
        schema_version=FROZEN_MAP_BUNDLE_SCHEMA_VERSION,
        semantic_map_id=semantic_map_id,
        planning_frame=planning_frame,
        yaml_sha256=hashlib.sha256(yaml_bytes).hexdigest(),
        image_sha256=hashlib.sha256(image_bytes).hexdigest(),
        resolution=metadata.resolution,
        origin=metadata.origin,
        negate=metadata.negate,
        occupied_thresh=metadata.occupied_thresh,
        free_thresh=metadata.free_thresh,
        mode=metadata.mode,
        width=image.width,
        height=image.height,
        maxval=image.maxval,
    )
    validate_frozen_map_bundle(bundle)
    return build_occupancy_grid(metadata, image), bundle


def freeze_map_bundle(
    path: Path, *, semantic_map_id: str, planning_frame: str
) -> FrozenMapBundle:
    """Return the content identity of a map snapshot without retaining paths."""

    _, bundle = load_occupancy_grid_with_bundle(
        path,
        semantic_map_id=semantic_map_id,
        planning_frame=planning_frame,
    )
    return bundle


def frozen_map_bundle_payload(bundle: FrozenMapBundle) -> dict[str, object]:
    validate_frozen_map_bundle(bundle)
    return {
        "schema_version": bundle.schema_version,
        "semantic_map_id": bundle.semantic_map_id,
        "planning_frame": bundle.planning_frame,
        "yaml_sha256": bundle.yaml_sha256,
        "image_sha256": bundle.image_sha256,
        "resolution": bundle.resolution,
        "origin": list(bundle.origin),
        "negate": bundle.negate,
        "occupied_thresh": bundle.occupied_thresh,
        "free_thresh": bundle.free_thresh,
        "mode": bundle.mode,
        "width": bundle.width,
        "height": bundle.height,
        "maxval": bundle.maxval,
    }


def frozen_map_bundle_sha256(bundle: FrozenMapBundle) -> str:
    payload = frozen_map_bundle_payload(bundle)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_frozen_map_bundle(path: Path, bundle: FrozenMapBundle) -> str:
    """Atomically publish an immutable map descriptor and return its hash."""

    try:
        return write_content_hashed_json(
            path,
            frozen_map_bundle_payload(bundle),
            hash_field=_MAP_BUNDLE_HASH_FIELD,
        )
    except ContentStoreError as exc:
        raise FrozenMapBundleError(exc.code, str(exc)) from exc


def load_frozen_map_bundle(
    path: Path,
    *,
    required_semantic_map_id: str | None = None,
    required_planning_frame: str | None = None,
) -> FrozenMapBundle:
    """Load, strictly parse, and content-hash-check a map descriptor."""

    try:
        payload = load_content_hashed_json(
            path, hash_field=_MAP_BUNDLE_HASH_FIELD
        )
    except ContentStoreError as exc:
        raise FrozenMapBundleError(exc.code, str(exc)) from exc
    actual_fields = frozenset(payload)
    if actual_fields != _MAP_BUNDLE_FIELDS:
        raise FrozenMapBundleError(
            "artifact_corrupt",
            "map bundle fields mismatch; "
            f"missing={sorted(_MAP_BUNDLE_FIELDS - actual_fields)} "
            f"unknown={sorted(actual_fields - _MAP_BUNDLE_FIELDS)}",
        )
    try:
        origin_value = payload["origin"]
        if not isinstance(origin_value, list) or len(origin_value) != 3:
            raise FrozenMapBundleError(
                "artifact_corrupt", "origin must be a three-element array"
            )
        bundle = FrozenMapBundle(
            schema_version=_strict_integer(payload["schema_version"], "schema_version"),
            semantic_map_id=_strict_string(
                payload["semantic_map_id"], "semantic_map_id"
            ),
            planning_frame=_strict_string(
                payload["planning_frame"], "planning_frame"
            ),
            yaml_sha256=_strict_string(payload["yaml_sha256"], "yaml_sha256"),
            image_sha256=_strict_string(payload["image_sha256"], "image_sha256"),
            resolution=_strict_number(payload["resolution"], "resolution"),
            origin=tuple(
                _strict_number(value, f"origin[{index}]")
                for index, value in enumerate(origin_value)
            ),
            negate=_strict_integer(payload["negate"], "negate"),
            occupied_thresh=_strict_number(
                payload["occupied_thresh"], "occupied_thresh"
            ),
            free_thresh=_strict_number(payload["free_thresh"], "free_thresh"),
            mode=_strict_string(payload["mode"], "mode"),
            width=_strict_integer(payload["width"], "width"),
            height=_strict_integer(payload["height"], "height"),
            maxval=_strict_integer(payload["maxval"], "maxval"),
        )
    except KeyError as exc:
        raise FrozenMapBundleError(
            "artifact_corrupt", "map bundle is missing a required field"
        ) from exc
    validate_frozen_map_bundle(bundle)
    if (
        required_semantic_map_id is not None
        and bundle.semantic_map_id != required_semantic_map_id
    ):
        raise FrozenMapBundleError(
            "provenance_mismatch", "map bundle semantic identity differs"
        )
    if (
        required_planning_frame is not None
        and bundle.planning_frame != required_planning_frame
    ):
        raise FrozenMapBundleError(
            "provenance_mismatch", "map bundle planning frame differs"
        )
    return bundle


def validate_frozen_map_bundle(bundle: FrozenMapBundle) -> None:
    if type(bundle.schema_version) is not int or (
        bundle.schema_version != FROZEN_MAP_BUNDLE_SCHEMA_VERSION
    ):
        raise FrozenMapBundleError(
            "schema_mismatch",
            f"unsupported frozen-map schema {bundle.schema_version!r}",
        )
    if not isinstance(bundle.semantic_map_id, str) or not _SAFE_MAP_ID.fullmatch(
        bundle.semantic_map_id
    ):
        raise FrozenMapBundleError(
            "invalid_map", "semantic_map_id is not a safe identifier"
        )
    if not isinstance(bundle.planning_frame, str) or not _SAFE_FRAME.fullmatch(
        bundle.planning_frame
    ):
        raise FrozenMapBundleError(
            "invalid_map", "planning_frame is not a valid frame identifier"
        )
    for name, value in (
        ("yaml_sha256", bundle.yaml_sha256),
        ("image_sha256", bundle.image_sha256),
    ):
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise FrozenMapBundleError(
                "invalid_map", f"{name} must be a lowercase SHA-256"
            )
    if (
        isinstance(bundle.resolution, bool)
        or not isinstance(bundle.resolution, (int, float))
        or not math.isfinite(bundle.resolution)
        or bundle.resolution <= 0.0
    ):
        raise FrozenMapBundleError(
            "invalid_map", "resolution must be finite and positive"
        )
    if (
        not isinstance(bundle.origin, tuple)
        or len(bundle.origin) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in bundle.origin
        )
    ):
        raise FrozenMapBundleError("invalid_map", "origin must contain three finite values")
    if abs(bundle.origin[2]) > 1.0e-12:
        raise FrozenMapBundleError(
            "invalid_map", "only zero-yaw map origins are supported"
        )
    if bundle.negate not in {0, 1} or isinstance(bundle.negate, bool):
        raise FrozenMapBundleError("invalid_map", "negate must be 0 or 1")
    if not (
        not isinstance(bundle.free_thresh, bool)
        and isinstance(bundle.free_thresh, (int, float))
        and not isinstance(bundle.occupied_thresh, bool)
        and isinstance(bundle.occupied_thresh, (int, float))
        and math.isfinite(bundle.free_thresh)
        and math.isfinite(bundle.occupied_thresh)
        and 0.0 <= bundle.free_thresh < bundle.occupied_thresh <= 1.0
    ):
        raise FrozenMapBundleError(
            "invalid_map", "map thresholds must satisfy 0 <= free < occupied <= 1"
        )
    if bundle.mode != "trinary":
        raise FrozenMapBundleError("invalid_map", "only trinary maps are supported")
    for name, value in (
        ("width", bundle.width),
        ("height", bundle.height),
        ("maxval", bundle.maxval),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise FrozenMapBundleError(
                "invalid_map", f"{name} must be a positive integer"
            )
    if bundle.maxval > 255:
        raise FrozenMapBundleError("invalid_map", "maxval must be at most 255")


def _strict_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise FrozenMapBundleError("artifact_corrupt", f"{name} must be a string")
    return value


def _strict_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrozenMapBundleError("artifact_corrupt", f"{name} must be a number")
    return float(value)


def _strict_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FrozenMapBundleError("artifact_corrupt", f"{name} must be an integer")
    return value
