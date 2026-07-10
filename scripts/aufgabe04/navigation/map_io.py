"""Stdlib-only ROS trinary map and PGM parsing for Aufgabe 04 navigation."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


CELL_FREE = 0
CELL_OCCUPIED = 1
CELL_UNKNOWN = 2


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
    data: Dict[str, object] = {}
    with Path(path).open() as file:
        for line_number, line in enumerate(file, start=1):
            stripped = strip_inline_comment(line).strip()
            if not stripped:
                continue
            if ":" not in stripped:
                raise ValueError(f"{path}:{line_number}: expected 'key: value'")
            key, value = stripped.split(":", 1)
            data[key.strip()] = parse_yaml_scalar(value)
    return data


def read_map_metadata(path: Path) -> MapMetadata:
    yaml_path = Path(path)
    data = read_simple_yaml(yaml_path)
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
    data = path.read_bytes()
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

