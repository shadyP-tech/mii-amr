#!/usr/bin/env python3
"""Freeze surveyed arrival poses and build one globally optimized full route."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.arrival_route_artifacts import (  # noqa: E402
    arrival_route_diagnostics_payload,
    pairwise_cost_payload,
    write_arrival_route_csv,
    write_json,
)
from scripts.aufgabe04.navigation.arrival_route_graph import (  # noqa: E402
    ArrivalRouteNode,
    build_arrival_route_graph,
    selected_edges,
)
from scripts.aufgabe04.navigation.costmap import Costmap  # noqa: E402
from scripts.aufgabe04.navigation.dynamic_approach_planner import (  # noqa: E402
    DynamicApproachConfig,
    FaceNormalCandidate,
)
from scripts.aufgabe04.navigation.full_route_optimizer import (  # noqa: E402
    optimize_full_route,
)
from scripts.aufgabe04.navigation.map_io import load_occupancy_grid  # noqa: E402
from scripts.aufgabe04.navigation.models import Pose2D  # noqa: E402
from scripts.aufgabe04.stations.arrival_pose_catalog import (  # noqa: E402
    arrival_pose_catalog_sha256,
    freeze_arrival_pose_catalog,
    load_arrival_pose_catalog,
    set_expected_candidate_uids,
    write_arrival_pose_catalog,
)
from scripts.aufgabe04.stations.arrival_pose_geometry import (  # noqa: E402
    angular_distance_rad,
    face_normal_rad,
)


DEFAULT_ROUTE = Path("results/aufgabe04/routes/optimized_arrival_route.csv")
DEFAULT_DIAGNOSTICS = Path(
    "results/aufgabe04/routes/optimized_arrival_route_diagnostics.json"
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument(
        "--world",
        type=Path,
        default=None,
        help="Simulation world artifact whose identity must match the catalog.",
    )
    parser.add_argument(
        "--session-id",
        default="",
        help="Simulation session identity whose catalog is being frozen.",
    )
    parser.add_argument("--map-frame", default="odom")
    parser.add_argument("--start-x", required=True, type=float)
    parser.add_argument("--start-y", required=True, type=float)
    parser.add_argument("--start-yaw", type=float, default=0.0)
    parser.add_argument("--route-csv", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--diagnostics-json", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--visit-order-json", type=Path, default=None)
    parser.add_argument("--pairwise-costs-json", type=Path, default=None)
    parser.add_argument("--catalog-snapshot-json", type=Path, default=None)
    parser.add_argument(
        "--expected-candidate-uid",
        action="append",
        default=[],
        help="Seal an open catalog with this expected candidate set before freezing.",
    )
    parser.add_argument(
        "--use-records-as-expected",
        action="store_true",
        help="Explicitly define the expected set as all current ready records.",
    )
    parser.add_argument(
        "--fixed-candidate-order",
        action="append",
        default=[],
        help="Preserve this candidate order instead of optimizing it; repeat in order.",
    )
    parser.add_argument("--allow-rejected-candidates", action="store_true")
    parser.add_argument("--exact-station-limit", type=int, default=12)
    parser.add_argument("--robot-radius-m", type=float, default=0.105)
    parser.add_argument("--tracking-margin-m", type=float, default=0.03)
    parser.add_argument("--collision-margin-m", type=float, default=0.02)
    parser.add_argument("--inflation-radius-m", type=float, default=None)
    parser.add_argument("--corridor-sample-spacing-m", type=float, default=0.05)
    parser.add_argument("--lidar-stop-distance-m", type=float, default=0.18)
    parser.add_argument("--scan-origin-to-base-offset-m", type=float, default=0.0)
    parser.add_argument("--lidar-clearance-margin-m", type=float, default=0.02)
    return parser


def _output_paths(args) -> tuple[Path, Path, Path]:
    stem = args.route_csv.with_suffix("")
    visit = args.visit_order_json or stem.with_name(stem.name + "_visit_order.json")
    costs = args.pairwise_costs_json or stem.with_name(
        stem.name + "_pairwise_costs.json"
    )
    snapshot = args.catalog_snapshot_json or stem.with_name(
        stem.name + "_catalog_snapshot.json"
    )
    return visit, costs, snapshot


def _route_node(record, args) -> ArrivalRouteNode:
    config = DynamicApproachConfig(
        stand_radius_m=record.stand.radius_m,
        stand_position_uncertainty_m=record.stand.uncertainty_m,
        robot_radius_m=args.robot_radius_m,
        collision_margin_m=args.collision_margin_m,
        standoff_distance_m=record.standoff_m,
        terminal_corridor_length_m=record.corridor_length_m,
        corridor_sample_spacing_m=args.corridor_sample_spacing_m,
        lidar_stop_distance_m=args.lidar_stop_distance_m,
        scan_origin_to_base_offset_m=args.scan_origin_to_base_offset_m,
        lidar_clearance_margin_m=args.lidar_clearance_margin_m,
    )
    canonical_normals = tuple(
        face_normal_rad(record.axis.axis_rad, face_id) for face_id in (0, 1)
    )
    face_id = min(
        range(2),
        key=lambda index: angular_distance_rad(
            record.face.outward_normal_rad,
            canonical_normals[index],
        ),
    )
    if angular_distance_rad(
        record.face.outward_normal_rad, canonical_normals[face_id]
    ) > 1.0e-6:
        raise ValueError("catalog face normal does not match its canonical stand axis")
    return ArrivalRouteNode(
        station_id=record.candidate_uid,
        arrival_id=f"{record.candidate_uid}::{record.face.face_id}",
        stand=Pose2D(record.stand.x_m, record.stand.y_m),
        face=FaceNormalCandidate(
            face_id=face_id,
            normal_rad=record.face.outward_normal_rad,
            target=Pose2D(
                record.arrival_pose.x_m,
                record.arrival_pose.y_m,
                record.arrival_pose.yaw_rad,
            ),
            entry=Pose2D(
                record.corridor_entry_pose.x_m,
                record.corridor_entry_pose.y_m,
                record.corridor_entry_pose.yaw_rad,
            ),
        ),
        config=config,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.expected_candidate_uid and args.use_records_as_expected:
        parser.error(
            "--expected-candidate-uid and --use-records-as-expected are mutually exclusive"
        )
    if args.exact_station_limit < 1:
        parser.error("--exact-station-limit must be positive")
    try:
        catalog = load_arrival_pose_catalog(args.catalog)
        map_sha256 = _file_sha256(args.map)
        if catalog.provenance.map_yaml_sha256 != map_sha256:
            raise ValueError("catalog occupancy-map SHA-256 does not match --map")
        if catalog.provenance.planning_frame != args.map_frame:
            raise ValueError("catalog planning frame does not match --map-frame")
        if catalog.provenance.environment == "simulation":
            if args.world is None or not args.session_id:
                raise ValueError(
                    "simulation catalog planning requires --world and --session-id"
                )
            if args.world.stem != catalog.provenance.world_id:
                raise ValueError("catalog world identity does not match --world")
            if _file_sha256(args.world) != catalog.provenance.world_sha256:
                raise ValueError("catalog world SHA-256 does not match --world")
            if args.session_id != catalog.provenance.session_id:
                raise ValueError("catalog session does not match --session-id")
        if catalog.rejections and not args.allow_rejected_candidates:
            rejected = ", ".join(item.candidate_uid for item in catalog.rejections)
            raise ValueError(
                "catalog contains rejected candidates; resurvey or pass "
                f"--allow-rejected-candidates: {rejected}"
            )
        if args.expected_candidate_uid:
            catalog = set_expected_candidate_uids(
                catalog,
                args.expected_candidate_uid,
                updated_unix_sec=max(time.time(), catalog.updated_unix_sec),
            )
        elif args.use_records_as_expected:
            catalog = set_expected_candidate_uids(
                catalog,
                (record.candidate_uid for record in catalog.records),
                updated_unix_sec=max(time.time(), catalog.updated_unix_sec),
            )
        catalog = freeze_arrival_pose_catalog(
            catalog,
            frozen_unix_sec=max(time.time(), catalog.updated_unix_sec),
        )
        catalog_sha256 = write_arrival_pose_catalog(args.catalog, catalog)
        if not catalog.records:
            raise ValueError("frozen catalog has no ready arrival poses")

        grid = load_occupancy_grid(args.map)
        costmap = Costmap.from_occupancy_grid(grid)
        inflation = (
            args.robot_radius_m + args.tracking_margin_m
            if args.inflation_radius_m is None
            else args.inflation_radius_m
        )
        if inflation <= 0.0:
            raise ValueError("configuration-space inflation radius must be positive")
        costmap = costmap.with_inflation(inflation)
        nodes = tuple(_route_node(record, args) for record in catalog.records)
        start = Pose2D(args.start_x, args.start_y, args.start_yaw)
        graph = build_arrival_route_graph(costmap, start, nodes)
        arrivals_by_candidate = {
            node.station_id: (node.arrival_id,) for node in nodes
        }
        fixed_order = tuple(args.fixed_candidate_order) or None
        route_plan = optimize_full_route(
            start_id=graph.start_id,
            arrivals_by_station=arrivals_by_candidate,
            directed_costs=graph.directed_costs,
            fixed_station_order=fixed_order,
            exact_station_limit=args.exact_station_limit,
        )
        edges = selected_edges(graph, route_plan.arrival_order)
        metadata = {
            "stage": "frozen_arrival_catalog_route",
            "simulation_only": catalog.provenance.environment == "simulation",
            "catalog_path": str(args.catalog.resolve()),
            "catalog_id": catalog.catalog_id,
            "catalog_revision": catalog.revision,
            "catalog_sha256": catalog_sha256,
            "map_yaml": str(args.map),
            "map_yaml_sha256": map_sha256,
            "planning_frame": catalog.provenance.planning_frame,
            "world_id": catalog.provenance.world_id,
            "world_sha256": catalog.provenance.world_sha256,
            "session_id": catalog.provenance.session_id,
            "static_inflation_radius_m": inflation,
            "non_target_stand_keepout_policy": (
                "max(body_uncertainty_collision,lidar_minimum_standoff)"
            ),
            "non_target_keepout_radius_by_arrival_m": {
                node.arrival_id: node.config.non_target_stand_keepout_radius_m
                for node in nodes
            },
        }
        write_arrival_route_csv(
            args.route_csv,
            costmap,
            edges,
            catalog_sha256=catalog_sha256,
            simulation_only=catalog.provenance.environment == "simulation",
        )
        metadata["route_csv_sha256"] = _file_sha256(args.route_csv)
        write_json(
            args.diagnostics_json,
            arrival_route_diagnostics_payload(edges, route_plan, metadata=metadata),
        )
        visit_path, costs_path, snapshot_path = _output_paths(args)
        write_json(
            visit_path,
            {
                **metadata,
                "algorithm": route_plan.algorithm,
                "optimal": route_plan.optimal,
                "total_cost_m": route_plan.total_cost,
                "candidate_order": list(route_plan.station_order),
                "arrival_order": list(route_plan.arrival_order),
            },
        )
        write_json(
            costs_path,
            {
                **metadata,
                **pairwise_cost_payload(graph.edges),
            },
        )
        write_arrival_pose_catalog(snapshot_path, catalog)
        print(
            json.dumps(
                {
                    "ok": True,
                    "optimal": route_plan.optimal,
                    "algorithm": route_plan.algorithm,
                    "total_cost_m": route_plan.total_cost,
                    "candidate_order": list(route_plan.station_order),
                    "route_csv": str(args.route_csv),
                    "diagnostics_json": str(args.diagnostics_json),
                    "catalog_sha256": arrival_pose_catalog_sha256(catalog),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
