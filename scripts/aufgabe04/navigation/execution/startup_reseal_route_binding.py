"""Bind a startup-reseal fresh pose to immutable replacement-route artifacts."""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.navigation.execution.exact_start_route_binding import (
    validate_exact_start_route_binding,
)


FreshPose = tuple[float, float, float]


def validate_startup_reseal_route_binding(
    *,
    route_csv_path: Path,
    diagnostics_path: Path,
    fresh_pose: FreshPose,
    require_start_pose_provenance: bool,
) -> None:
    """Fail unless one persisted route starts at the admitted fresh pose.

    Route CSV writers intentionally leave non-terminal yaw cells blank.  The
    first waypoint therefore binds x/y, while the exact-start connector binds
    x/y/yaw.  Candidate routes additionally require start-pose provenance;
    current coverage routes may omit it, but it is validated whenever present.
    """

    expected_pose = _validated_pose(fresh_pose, "fresh stationary pose")
    route_xy, first_yaw = _load_single_route(Path(route_csv_path))
    metadata = _load_diagnostics_metadata(Path(diagnostics_path))

    validate_exact_start_route_binding(metadata, route_xy)
    connector = metadata["exact_start_connector"]
    assert isinstance(connector, Mapping)  # validated above
    exact_start = _mapping_pose(
        connector.get("exact_start"),
        "exact_start_connector.exact_start",
    )
    _require_same_pose(
        exact_start,
        expected_pose,
        "replacement exact start differs from fresh stationary pose",
    )

    if route_xy[0] != expected_pose[:2]:
        raise ValueError(
            "replacement route waypoint 0 differs from fresh stationary pose"
        )
    if first_yaw is not None and first_yaw != expected_pose[2]:
        raise ValueError(
            "replacement route waypoint 0 yaw differs from fresh stationary pose"
        )

    provenance = metadata.get("route_start_pose_provenance")
    if provenance is None:
        if require_start_pose_provenance:
            raise ValueError(
                "replacement route start-pose provenance is required"
            )
        return
    if not isinstance(provenance, Mapping):
        raise ValueError("replacement route start-pose provenance is malformed")
    source = provenance.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("replacement route start-pose provenance source is missing")
    planning_frame = metadata.get("planning_frame")
    if (
        not isinstance(planning_frame, str)
        or not planning_frame.strip()
        or provenance.get("planning_frame") != planning_frame
    ):
        raise ValueError(
            "replacement route start-pose provenance planning frame mismatch"
        )
    provenance_pose = _mapping_pose(
        provenance.get("pose"),
        "route_start_pose_provenance.pose",
    )
    _require_same_pose(
        provenance_pose,
        expected_pose,
        "replacement route start-pose provenance differs from fresh stationary pose",
    )
    _require_same_pose(
        provenance_pose,
        exact_start,
        "replacement route start-pose provenance differs from exact-start connector",
    )


def _load_single_route(
    path: Path,
) -> tuple[tuple[tuple[float, float], ...], float | None]:
    try:
        text = path.read_bytes().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        if reader.fieldnames is None:
            raise ValueError("replacement route CSV is missing a header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("replacement route CSV has duplicate columns")
        required = {"leg_index", "point_index", "world_x_m", "world_y_m"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "replacement route CSV missing columns: "
                + ", ".join(sorted(missing))
            )
        rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"invalid replacement route CSV: {path}") from exc
    if not rows:
        raise ValueError("replacement route CSV is empty")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("replacement route CSV row shape mismatch")

    route_xy: list[tuple[float, float]] = []
    leg_index: int | None = None
    first_yaw: float | None = None
    for expected_point_index, row in enumerate(rows):
        row_leg_index = _integer_text(row["leg_index"], "leg_index")
        if leg_index is None:
            leg_index = row_leg_index
        elif row_leg_index != leg_index:
            raise ValueError("replacement route CSV must contain exactly one leg")
        point_index = _integer_text(row["point_index"], "point_index")
        if point_index != expected_point_index:
            raise ValueError(
                "replacement route CSV point_index must be contiguous from 0"
            )
        route_xy.append(
            (
                _float_text(row["world_x_m"], "world_x_m"),
                _float_text(row["world_y_m"], "world_y_m"),
            )
        )
        if expected_point_index == 0:
            yaw_text = row.get("yaw_rad", "").strip()
            if yaw_text:
                first_yaw = _float_text(yaw_text, "yaw_rad")
    return tuple(route_xy), first_yaw


def _load_diagnostics_metadata(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object_pairs,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid replacement route diagnostics: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("replacement route diagnostics root must be an object")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("replacement route diagnostics are missing metadata")
    return metadata


def _validated_pose(value: object, name: str) -> FreshPose:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{name} must be an (x, y, yaw) tuple")
    return tuple(
        _finite_number(component, f"{name}.{field}")
        for field, component in zip(("x_m", "y_m", "yaw_rad"), value)
    )


def _mapping_pose(value: object, name: str) -> FreshPose:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return tuple(
        _finite_number(value.get(field), f"{name}.{field}")
        for field in ("x_m", "y_m", "yaw_rad")
    )


def _require_same_pose(observed: FreshPose, expected: FreshPose, message: str) -> None:
    if observed != expected:
        raise ValueError(message)


def _integer_text(value: str, name: str) -> int:
    try:
        selected = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"replacement route CSV {name} must be an integer") from exc
    if selected < 0:
        raise ValueError(f"replacement route CSV {name} must be non-negative")
    return selected


def _float_text(value: str, name: str) -> float:
    try:
        return _finite_number(float(value), f"replacement route CSV {name}")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"replacement route CSV {name} must be finite") from exc


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError(f"{name} must be finite")
    return selected


def _strict_object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


__all__ = ["validate_startup_reseal_route_binding"]
