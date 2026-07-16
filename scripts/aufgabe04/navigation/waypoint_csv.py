"""Load and validate Aufgabe 04 station-route CSV legs."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from scripts.aufgabe04.navigation.models import Pose2D


REQUIRED_ROUTE_COLUMNS = {
    "leg_index",
    "point_index",
    "world_x_m",
    "world_y_m",
    "cumulative_length_m",
}


@dataclass(frozen=True)
class RouteWaypoint:
    leg_index: int
    point_index: int
    pose: Pose2D
    cumulative_length_m: float
    protected: bool = False


@dataclass(frozen=True)
class SelectedRouteLeg:
    source_path: Path
    leg_index: int
    raw_waypoints: Tuple[RouteWaypoint, ...]
    executable_waypoints: Tuple[RouteWaypoint, ...]
    route_length_m: float
    thinning_min_spacing_m: float
    simulation_only: bool = False
    route_kind: str = ""
    stream_id: str = ""
    route_revision: int | None = None
    target_revision: int | None = None
    manifest_path: Path | None = None


def _parse_int(value: str, field: str, row_number: int) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {field} must be an integer") from exc


def _parse_finite_float(value: str, field: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"row {row_number}: {field} must be finite")
    return parsed


def _parse_optional_bool(value: str, field: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in ("", "false", "0", "no"):
        return False
    if normalized in ("true", "1", "yes"):
        return True
    raise ValueError(f"row {row_number}: {field} must be true or false")


def _parse_optional_int(value: str, field: str, row_number: int) -> int | None:
    if not value.strip():
        return None
    parsed = _parse_int(value, field, row_number)
    if parsed < 0:
        raise ValueError(f"row {row_number}: {field} must be non-negative")
    return parsed


def _distance(a: RouteWaypoint, b: RouteWaypoint) -> float:
    return math.hypot(a.pose.x_m - b.pose.x_m, a.pose.y_m - b.pose.y_m)


def thin_waypoints(
    waypoints: Sequence[RouteWaypoint],
    min_spacing_m: float,
) -> Tuple[RouteWaypoint, ...]:
    """Deterministically thin dense route points while preserving endpoints."""

    if min_spacing_m <= 0.0 or len(waypoints) <= 2:
        return tuple(waypoints)
    thinned: List[RouteWaypoint] = [waypoints[0]]
    for waypoint in waypoints[1:-1]:
        if waypoint.protected or _distance(thinned[-1], waypoint) >= min_spacing_m:
            thinned.append(waypoint)
    if thinned[-1] != waypoints[-1]:
        thinned.append(waypoints[-1])
    return tuple(thinned)


def load_route_leg(
    path: Path,
    leg_index: int,
    *,
    require_motion: bool = True,
    thinning_min_spacing_m: float = 0.0,
) -> SelectedRouteLeg:
    """Load one route leg from a station-route CSV artifact."""

    path = Path(path)
    if thinning_min_spacing_m < 0.0:
        raise ValueError("thinning_min_spacing_m must be non-negative")
    selected: List[RouteWaypoint] = []
    selected_metadata: List[tuple[bool, str, str, int | None, int | None, str]] = []
    seen_leg_indexes = set()
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("route CSV is missing a header")
        missing = REQUIRED_ROUTE_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"route CSV missing columns: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            row_leg_index = _parse_int(row["leg_index"], "leg_index", row_number)
            seen_leg_indexes.add(row_leg_index)
            if row_leg_index != leg_index:
                continue
            point_index = _parse_int(row["point_index"], "point_index", row_number)
            x_m = _parse_finite_float(row["world_x_m"], "world_x_m", row_number)
            y_m = _parse_finite_float(row["world_y_m"], "world_y_m", row_number)
            yaw_text = row.get("yaw_rad", "").strip()
            yaw_rad = (
                _parse_finite_float(yaw_text, "yaw_rad", row_number)
                if yaw_text
                else float("nan")
            )
            cumulative_length_m = _parse_finite_float(
                row["cumulative_length_m"],
                "cumulative_length_m",
                row_number,
            )
            protected = _parse_optional_bool(
                row.get("protected", ""), "protected", row_number
            )
            simulation_only = _parse_optional_bool(
                row.get("simulation_only", ""), "simulation_only", row_number
            )
            route_kind = row.get("route_kind", "").strip()
            stream_id = row.get("stream_id", "").strip()
            route_revision = _parse_optional_int(
                row.get("route_revision", ""), "route_revision", row_number
            )
            target_revision = _parse_optional_int(
                row.get("target_revision", ""), "target_revision", row_number
            )
            manifest_path = row.get("manifest_path", "").strip()
            selected.append(
                RouteWaypoint(
                    leg_index=row_leg_index,
                    point_index=point_index,
                    pose=Pose2D(x_m, y_m, yaw_rad),
                    cumulative_length_m=cumulative_length_m,
                    protected=protected,
                )
            )
            selected_metadata.append(
                (
                    simulation_only,
                    route_kind,
                    stream_id,
                    route_revision,
                    target_revision,
                    manifest_path,
                )
            )

    if not selected:
        available = ", ".join(str(value) for value in sorted(seen_leg_indexes))
        raise ValueError(f"leg_index {leg_index} not found; available legs: {available or 'none'}")

    selected.sort(key=lambda waypoint: waypoint.point_index)
    for expected, waypoint in enumerate(selected):
        if waypoint.point_index != expected:
            raise ValueError(
                f"leg_index {leg_index} point_index must be contiguous from 0; "
                f"expected {expected}, got {waypoint.point_index}"
            )

    route_length_m = selected[-1].cumulative_length_m
    unique_metadata = set(selected_metadata)
    if len(unique_metadata) != 1:
        raise ValueError(f"leg_index {leg_index} has inconsistent route provenance metadata")
    (
        simulation_only,
        route_kind,
        stream_id,
        route_revision,
        target_revision,
        manifest_text,
    ) = next(iter(unique_metadata))
    if require_motion:
        if len(selected) < 2:
            raise ValueError(f"leg_index {leg_index} has fewer than two points for motion")
        if route_length_m <= 0.0:
            raise ValueError(f"leg_index {leg_index} has non-positive route length")

    executable = thin_waypoints(selected, thinning_min_spacing_m)
    if require_motion and len(executable) < 2:
        raise ValueError(f"leg_index {leg_index} has fewer than two executable points")

    return SelectedRouteLeg(
        source_path=path,
        leg_index=leg_index,
        raw_waypoints=tuple(selected),
        executable_waypoints=executable,
        route_length_m=route_length_m,
        thinning_min_spacing_m=thinning_min_spacing_m,
        simulation_only=simulation_only,
        route_kind=route_kind,
        stream_id=stream_id,
        route_revision=route_revision,
        target_revision=target_revision,
        manifest_path=Path(manifest_text) if manifest_text else None,
    )


def poses_from_waypoints(waypoints: Iterable[RouteWaypoint]) -> Tuple[Pose2D, ...]:
    return tuple(waypoint.pose for waypoint in waypoints)
