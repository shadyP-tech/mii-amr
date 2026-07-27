"""Artifact serialization for a frozen arrival-catalog mission route."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from scripts.aufgabe04.navigation.arrival_route_graph import ArrivalRouteEdge
from scripts.aufgabe04.navigation.costmap import Costmap
from scripts.aufgabe04.navigation.full_route_optimizer import FullRoutePlan


ROUTE_KIND = "catalog_face_approach"


def write_arrival_route_csv(
    path: Path,
    costmap: Costmap,
    edges: Sequence[ArrivalRouteEdge],
    *,
    catalog_sha256: str,
    simulation_only: bool,
) -> None:
    """Write exact fixed targets and protected terminal corridors per leg."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            _write_arrival_route_csv_rows(
                handle,
                costmap,
                edges,
                catalog_sha256=catalog_sha256,
                simulation_only=simulation_only,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _write_arrival_route_csv_rows(
    handle,
    costmap: Costmap,
    edges: Sequence[ArrivalRouteEdge],
    *,
    catalog_sha256: str,
    simulation_only: bool,
) -> None:
    writer = csv.writer(handle)
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
                "protected",
                "corridor",
                "simulation_only",
                "route_kind",
                "stream_id",
                "route_revision",
                "target_revision",
                "manifest_path",
                "source_arrival_id",
                "target_arrival_id",
                "catalog_sha256",
            ]
        )
    for leg_index, edge in enumerate(edges):
        plan = edge.result.plan
        if plan is None:
            raise ValueError(
                f"cannot serialize unreachable edge {edge.source_id}->{edge.target_id}"
            )
        cumulative = 0.0
        previous = None
        for point_index, waypoint in enumerate(plan.waypoints):
            pose = waypoint.pose
            segment = (
                0.0
                if previous is None
                else math.hypot(
                    pose.x_m - previous.x_m,
                    pose.y_m - previous.y_m,
                )
            )
            cumulative += segment
            cell = costmap.world_to_grid(pose)
            writer.writerow(
                [
                    leg_index,
                    point_index,
                    cell.x,
                    cell.y,
                    pose.x_m,
                    pose.y_m,
                    "" if not math.isfinite(pose.yaw_rad) else pose.yaw_rad,
                    segment,
                    cumulative,
                    str(waypoint.protected).lower(),
                    str(waypoint.corridor).lower(),
                    str(simulation_only).lower(),
                    ROUTE_KIND,
                    "",
                    "",
                    "",
                    "",
                    edge.source_id,
                    edge.target_id,
                    catalog_sha256,
                ]
            )
            previous = pose


def arrival_route_diagnostics_payload(
    edges: Sequence[ArrivalRouteEdge],
    route_plan: FullRoutePlan,
    *,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    legs = []
    for edge in edges:
        plan = edge.result.plan
        if plan is None:
            raise ValueError("selected optimized edge is unreachable")
        legs.append(
            {
                "source_arrival_id": edge.source_id,
                "target_arrival_id": edge.target_id,
                "diagnostics": {
                    "status": "ok",
                    "reason": "",
                    "route_length_m": plan.length_m,
                    "fixed_arrival": asdict(edge.result.diagnostics),
                },
                "failure": None,
                "route_length_m": plan.length_m,
                "route_point_count": len(plan.waypoints),
                "exact_target": asdict(plan.target),
                "corridor_entry": asdict(plan.entry),
                "non_target_stand_clearances": [
                    asdict(clearance)
                    for clearance in edge.non_target_clearances
                ],
                "non_target_keepout_overlay": (
                    None
                    if edge.non_target_overlay is None
                    else asdict(edge.non_target_overlay)
                ),
            }
        )
    return {
        "metadata": {
            **dict(metadata),
            "route_kind": ROUTE_KIND,
            "optimization": {
                "algorithm": route_plan.algorithm,
                "optimal": route_plan.optimal,
                "fixed_station_order": route_plan.fixed_station_order,
                "total_cost_m": route_plan.total_cost,
                "station_order": list(route_plan.station_order),
                "arrival_order": list(route_plan.arrival_order),
            },
        },
        "legs": legs,
    }


def write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def pairwise_cost_payload(
    edges: Mapping[tuple[str, str], ArrivalRouteEdge],
) -> dict[str, object]:
    rows = []
    for (source_id, target_id), edge in sorted(edges.items()):
        rows.append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "reachable": edge.result.plan is not None,
                "cost_m": edge.cost_m,
                "failure_reason": edge.result.diagnostics.failure_reason,
                "non_target_stand_clearances": [
                    asdict(clearance)
                    for clearance in edge.non_target_clearances
                ],
                "non_target_keepout_overlay": (
                    None
                    if edge.non_target_overlay is None
                    else asdict(edge.non_target_overlay)
                ),
            }
        )
    return {"directed": True, "edges": rows}
