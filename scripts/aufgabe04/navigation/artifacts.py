"""Dry-run route artifact writers for Aufgabe 04 navigation."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable, Mapping

from scripts.aufgabe04.navigation.global_planner import PlanRouteResult


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, frozenset):
        return list(value)
    raise TypeError(f"{type(value)!r} is not JSON serializable")


def write_route_csv(
    path: Path,
    leg_results: Iterable[PlanRouteResult],
    *,
    final_yaw_by_leg: Mapping[int, float] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "leg_index",
                "point_index",
                "grid_x",
                "grid_y",
                "world_x_m",
                "world_y_m",
                "yaw_rad",
                "segment_length_m",
                "cumulative_length_m",
            ]
        )
        for leg_index, result in enumerate(leg_results):
            if result.route is None:
                continue
            for point in result.route.points:
                final_yaw = None
                if final_yaw_by_leg and point.index == len(result.route.points) - 1:
                    final_yaw = final_yaw_by_leg.get(leg_index)
                writer.writerow(
                    [
                        leg_index,
                        point.index,
                        point.cell.x,
                        point.cell.y,
                        point.pose.x_m,
                        point.pose.y_m,
                        "" if final_yaw is None else final_yaw,
                        point.segment_length_m,
                        point.cumulative_length_m,
                    ]
                )


def write_diagnostics_json(
    path: Path,
    leg_results: Iterable[PlanRouteResult],
    metadata: Mapping[str, object] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": dict(metadata or {}),
        "legs": [
            {
                "diagnostics": result.diagnostics,
                "failure": result.failure,
                "route_length_m": result.route.length_m if result.route else None,
                "route_point_count": len(result.route.points) if result.route else 0,
            }
            for result in leg_results
        ],
    }
    path.write_text(json.dumps(payload, default=_json_default, indent=2, sort_keys=True) + "\n")


def write_overlay_svg(path: Path, svg_text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_text)


def write_overlay_metadata_json(path: Path, metadata: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(metadata), default=_json_default, indent=2, sort_keys=True) + "\n")
