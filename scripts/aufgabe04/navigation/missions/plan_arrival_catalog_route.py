#!/usr/bin/env python3
"""Freeze surveyed arrivals and build either a survey or task-ordered route."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.missions.arrival_route_artifacts import (  # noqa: E402
    arrival_route_diagnostics_payload,
    pairwise_cost_payload,
    write_arrival_route_csv,
    write_json,
)
from scripts.aufgabe04.navigation.missions.arrival_route_graph import (  # noqa: E402
    ArrivalRouteNode,
    build_arrival_route_graph,
    build_required_arrival_route_graph,
    resolve_station_arrival_order,
    selected_edges,
)
from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds  # noqa: E402
from scripts.aufgabe04.navigation.planning.costmap import Costmap  # noqa: E402
from scripts.aufgabe04.navigation.approach.dynamic_approach_planner import (  # noqa: E402
    DynamicApproachConfig,
    FaceNormalCandidate,
    minimum_static_obstacle_inflation_m,
)
from scripts.aufgabe04.navigation.planning.full_route_optimizer import (  # noqa: E402
    FullRoutePlan,
    OptimizedVisit,
    optimize_full_route,
)
from scripts.aufgabe04.navigation.execution.execution_route_certificate import (  # noqa: E402
    ExecutionRouteCertificate,
    write_execution_route_certificate,
)
from scripts.aufgabe04.navigation.planning.map_io import (  # noqa: E402
    load_frozen_map_bundle,
    load_occupancy_grid_with_bundle,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D  # noqa: E402
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
from scripts.aufgabe04.stations.candidate_snapshot import (  # noqa: E402
    candidate_snapshot_sha256,
    load_candidate_snapshot,
)
from scripts.aufgabe04.stations.station_identity_registry import (  # noqa: E402
    candidate_order_for_server_order,
    load_station_identity_registry,
    station_identity_registry_sha256,
)
from scripts.aufgabe04.logistics.server_validation.artifacts import (  # noqa: E402
    load_validated_task_snapshot,
    validated_task_snapshot_sha256,
)
from scripts.aufgabe04.artifacts import (  # noqa: E402
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    MissionPlanManifest,
    artifact_reference,
    load_survey_manifest,
    manifest_reference,
    survey_manifest_sha256,
    validate_mission_plan_links,
    write_mission_plan_manifest,
)
from scripts.aufgabe04.artifacts.content_store import (  # noqa: E402
    load_content_hashed_json,
    payload_sha256,
    write_content_hashed_json,
)


DEFAULT_ROUTE = Path("results/aufgabe04/routes/optimized_arrival_route.csv")
DEFAULT_DIAGNOSTICS = Path(
    "results/aufgabe04/routes/optimized_arrival_route_diagnostics.json"
)
DEFAULT_MAX_TASK_SNAPSHOT_AGE_SEC = 30.0
DEFAULT_MAX_TASK_FUTURE_SKEW_SEC = 2.0
ARTIFACT_DESCRIPTOR_SCHEMA_VERSION = 1
ARTIFACT_DESCRIPTOR_HASH_FIELD = "artifact_sha256"
SURVEY_CONFIG_HASH_FIELD = "survey_config_sha256"
_ARENA_BOUND_FIELDS = frozenset(
    {
        "length_m",
        "width_m",
        "center_x_m",
        "center_y_m",
        "yaw_deg",
        "margin_m",
    }
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_task_snapshot_freshness(
    task,
    *,
    now_sec: float,
    max_age_sec: float,
    max_future_skew_sec: float,
) -> None:
    """Fail closed when an immutable server decision is no longer current."""

    values = {
        "status_observed_at_sec": task.status_observed_at_sec,
        "plan_generated_at_sec": task.plan_generated_at_sec,
        "validated_at_sec": task.validated_at_sec,
    }
    if not math.isfinite(now_sec) or now_sec < 0.0:
        raise ValueError("task freshness clock must be finite and non-negative")
    if not math.isfinite(max_age_sec) or max_age_sec <= 0.0:
        raise ValueError("maximum task snapshot age must be finite and positive")
    if (
        not math.isfinite(max_future_skew_sec)
        or max_future_skew_sec < 0.0
    ):
        raise ValueError(
            "maximum task future skew must be finite and non-negative"
        )
    for name, value in values.items():
        if (
            value is None
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0.0
        ):
            raise ValueError(f"validated task {name} is invalid")
        if value > now_sec + max_future_skew_sec:
            raise ValueError(f"validated task {name} is in the future")

    validated_at_sec = float(task.validated_at_sec)
    if now_sec - validated_at_sec > max_age_sec:
        raise ValueError("validated task snapshot is stale")
    for source_name in ("status_observed_at_sec", "plan_generated_at_sec"):
        if float(values[source_name]) > validated_at_sec + max_future_skew_sec:
            raise ValueError(
                f"validated task {source_name} postdates task validation"
            )


def _require_catalog_unchanged(
    path: Path, *, expected_sha256: str, stage: str
) -> None:
    """Detect concurrent catalog replacement without ever writing it back."""

    current = load_arrival_pose_catalog(path)
    current_sha256 = arrival_pose_catalog_sha256(current)
    if current_sha256 != expected_sha256:
        raise ValueError(
            "arrival-pose catalog changed concurrently "
            f"during {stage}: expected {expected_sha256}, got {current_sha256}"
        )


def _arena_bounds_from_survey_config(
    payload: Mapping[str, object],
) -> ArenaBounds:
    raw_bounds = payload.get("arena_bounds")
    if (
        not isinstance(raw_bounds, Mapping)
        or frozenset(raw_bounds) != _ARENA_BOUND_FIELDS
    ):
        raise ValueError(
            "survey configuration arena_bounds must contain exactly "
            "length_m, width_m, center_x_m, center_y_m, yaw_deg, margin_m"
        )
    values: dict[str, float] = {}
    for field_name in _ARENA_BOUND_FIELDS:
        raw_value = raw_bounds.get(field_name)
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(
                f"survey configuration arena_bounds.{field_name} must be numeric"
            )
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(
                f"survey configuration arena_bounds.{field_name} must be finite"
            )
        values[field_name] = value
    arena_bounds = ArenaBounds(**values)
    arena_bounds.validate()
    return arena_bounds


def _load_bound_survey_arena_bounds(
    path: Path,
    *,
    manifest_sha256: str,
    catalog_sha256: str,
) -> ArenaBounds:
    payload = load_content_hashed_json(
        path,
        hash_field=SURVEY_CONFIG_HASH_FIELD,
    )
    actual_sha256 = payload_sha256(payload)
    if actual_sha256 != manifest_sha256:
        raise ValueError(
            "survey configuration artifact differs from survey manifest"
        )
    if catalog_sha256 and actual_sha256 != catalog_sha256:
        raise ValueError(
            "survey configuration artifact differs from catalog provenance"
        )
    return _arena_bounds_from_survey_config(payload)


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
    parser.add_argument("--semantic-map-id", default="")
    parser.add_argument("--arena-length-m", type=float, default=ArenaBounds.length_m)
    parser.add_argument("--arena-width-m", type=float, default=ArenaBounds.width_m)
    parser.add_argument(
        "--arena-center-x-m",
        type=float,
        default=ArenaBounds.center_x_m,
    )
    parser.add_argument(
        "--arena-center-y-m",
        type=float,
        default=ArenaBounds.center_y_m,
    )
    parser.add_argument("--arena-yaw-deg", type=float, default=ArenaBounds.yaw_deg)
    parser.add_argument("--arena-margin-m", type=float, default=ArenaBounds.margin_m)
    parser.add_argument("--map-bundle-json", type=Path, default=None)
    parser.add_argument("--candidate-snapshot", type=Path, default=None)
    parser.add_argument("--station-identity-registry", type=Path, default=None)
    parser.add_argument("--survey-manifest", type=Path, default=None)
    parser.add_argument(
        "--survey-config-json",
        type=Path,
        default=None,
        help=(
            "Content-hashed survey configuration referenced by --survey-manifest. "
            "Defaults to survey_config_<referenced SHA-256>.json beside the manifest."
        ),
    )
    parser.add_argument("--task-snapshot", type=Path, default=None)
    parser.add_argument("--robot-id", default="")
    parser.add_argument(
        "--max-task-snapshot-age-sec",
        type=float,
        default=DEFAULT_MAX_TASK_SNAPSHOT_AGE_SEC,
        help=(
            "Maximum age of validated_at_sec when admitting a logistics task "
            "snapshot."
        ),
    )
    parser.add_argument(
        "--max-task-future-skew-sec",
        type=float,
        default=DEFAULT_MAX_TASK_FUTURE_SKEW_SEC,
        help="Allowed positive clock skew for validated task timestamps.",
    )
    parser.add_argument("--mission-plan-manifest", type=Path, default=None)
    parser.add_argument(
        "--planner-config-json",
        type=Path,
        default=None,
        help=(
            "Immutable logistics planner configuration descriptor. The default "
            "filename contains the full payload hash."
        ),
    )
    parser.add_argument(
        "--route-bundle-json",
        type=Path,
        default=None,
        help=(
            "Immutable logistics route-bundle descriptor. The default filename "
            "contains the full payload hash."
        ),
    )
    parser.add_argument("--start-x", required=True, type=float)
    parser.add_argument("--start-y", required=True, type=float)
    parser.add_argument("--start-yaw", type=float, default=0.0)
    parser.add_argument("--route-csv", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--diagnostics-json", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--visit-order-json", type=Path, default=None)
    parser.add_argument("--pairwise-costs-json", type=Path, default=None)
    parser.add_argument("--catalog-snapshot-json", type=Path, default=None)
    parser.add_argument("--route-certificate-json", type=Path, default=None)
    parser.add_argument(
        "--command-owner",
        default="/aufgabe04_simple_waypoint_follower",
        help="Exclusive runtime cmd_vel owner bound into the route certificate.",
    )
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
    parser.add_argument(
        "--route-purpose",
        choices=("survey", "logistics"),
        default="survey",
        help=(
            "Survey routes may optimize stand order; logistics routes preserve "
            "the exact semantic station order supplied by the task server."
        ),
    )
    parser.add_argument(
        "--fixed-station-order",
        action="append",
        default=[],
        help=(
            "Optional assertion of the semantic task order; repeat in order. "
            "For logistics routes the immutable validated task snapshot is "
            "authoritative."
        ),
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


def _route_node(
    record,
    args,
    identity_registry=None,
    frozen_candidate=None,
) -> ArrivalRouteNode:
    stand_radius_m = record.stand.radius_m
    stand_uncertainty_m = record.stand.uncertainty_m
    frozen_keepout_radius_m = 0.0
    if frozen_candidate is not None:
        frozen = frozen_candidate.geometry
        center_delta_m = math.hypot(
            record.stand.x_m - frozen.x_m,
            record.stand.y_m - frozen.y_m,
        )
        stand_radius_m = max(stand_radius_m, frozen.radius_m)
        stand_uncertainty_m = max(
            stand_uncertainty_m,
            center_delta_m
            + frozen.radius_m
            + frozen.uncertainty_m
            - stand_radius_m,
            0.0,
        )
        frozen_keepout_radius_m = frozen.keepout_radius_m
    config = DynamicApproachConfig(
        stand_radius_m=stand_radius_m,
        stand_position_uncertainty_m=stand_uncertainty_m,
        robot_radius_m=args.robot_radius_m,
        collision_margin_m=args.collision_margin_m,
        tracking_margin_m=args.tracking_margin_m,
        standoff_distance_m=record.standoff_m,
        terminal_corridor_length_m=record.corridor_length_m,
        corridor_sample_spacing_m=args.corridor_sample_spacing_m,
        lidar_stop_distance_m=args.lidar_stop_distance_m,
        scan_origin_to_base_offset_m=args.scan_origin_to_base_offset_m,
        lidar_clearance_margin_m=args.lidar_clearance_margin_m,
        # Candidate snapshots store an actual robot-centre transit envelope.
        # DynamicApproachConfig expands it by tracking_margin_m exactly once.
        minimum_non_target_keepout_radius_m=frozen_keepout_radius_m,
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
    # Survey optimization is keyed by stable geometric candidate identity.
    # Only a validated semantic registry may promote it to a task station ID.
    station_id = record.candidate_uid
    if identity_registry is not None:
        identity = identity_registry.for_candidate(record.candidate_uid)
        if identity is None:
            raise ValueError(
                f"catalog candidate has no station identity: {record.candidate_uid}"
            )
        if record.stand_id not in (identity.qr_id, identity.server_station_id):
            raise ValueError(
                "catalog stand identity disagrees with station registry for "
                f"{record.candidate_uid}"
            )
        station_id = identity.server_station_id
    return ArrivalRouteNode(
        station_id=station_id,
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


def _fixed_task_route_plan(
    graph,
    station_order: tuple[str, ...],
    arrival_order: tuple[str, ...],
) -> FullRoutePlan:
    if len(station_order) != len(arrival_order):
        raise ValueError("task station and arrival orders must have equal length")
    edges = selected_edges(graph, arrival_order)
    visits = []
    total_cost = 0.0
    for station_id, arrival_id, edge in zip(
        station_order, arrival_order, edges
    ):
        if edge.cost_m is None:
            raise ValueError(
                f"required task edge is unreachable: {edge.source_id}->{edge.target_id}"
            )
        total_cost += edge.cost_m
        visits.append(
            OptimizedVisit(
                station_id=station_id,
                arrival_id=arrival_id,
                inbound_cost=edge.cost_m,
            )
        )
    return FullRoutePlan(
        start_id=graph.start_id,
        visits=tuple(visits),
        total_cost=total_cost,
        algorithm="fixed_task_order_a_star",
        optimal=True,
        fixed_station_order=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.expected_candidate_uid and args.use_records_as_expected:
        parser.error(
            "--expected-candidate-uid and --use-records-as-expected are mutually exclusive"
        )
    if args.fixed_candidate_order and args.fixed_station_order:
        parser.error(
            "--fixed-candidate-order and --fixed-station-order are mutually exclusive"
        )
    if args.route_purpose == "logistics" and args.fixed_candidate_order:
        parser.error(
            "logistics routes accept semantic --fixed-station-order only"
        )
    if args.route_purpose == "logistics" and (
        args.expected_candidate_uid or args.use_records_as_expected
    ):
        parser.error(
            "logistics routes require a sealed catalog and cannot revise its "
            "expected candidate set"
        )
    if args.route_purpose == "survey" and args.fixed_station_order:
        parser.error("survey routes accept --fixed-candidate-order only")
    if args.route_purpose == "survey" and (
        args.planner_config_json is not None
        or args.route_bundle_json is not None
    ):
        parser.error(
            "--planner-config-json and --route-bundle-json are logistics-only"
        )
    if args.route_purpose == "logistics" and (
        args.candidate_snapshot is None
        or args.station_identity_registry is None
        or args.map_bundle_json is None
    ):
        parser.error(
            "logistics routes require --map-bundle-json, --candidate-snapshot, "
            "and --station-identity-registry"
        )
    if args.route_purpose == "logistics" and (
        args.survey_manifest is None
        or args.task_snapshot is None
        or not args.robot_id
    ):
        parser.error(
            "logistics routes require --survey-manifest, --task-snapshot, and --robot-id"
        )
    if (args.candidate_snapshot is None) != (
        args.station_identity_registry is None
    ):
        parser.error(
            "--candidate-snapshot and --station-identity-registry must be paired"
        )
    if args.survey_config_json is not None and args.survey_manifest is None:
        parser.error("--survey-config-json requires --survey-manifest")
    if args.exact_station_limit < 1:
        parser.error("--exact-station-limit must be positive")
    if (
        not math.isfinite(args.max_task_snapshot_age_sec)
        or args.max_task_snapshot_age_sec <= 0.0
    ):
        parser.error("--max-task-snapshot-age-sec must be finite and positive")
    if (
        not math.isfinite(args.max_task_future_skew_sec)
        or args.max_task_future_skew_sec < 0.0
    ):
        parser.error(
            "--max-task-future-skew-sec must be finite and non-negative"
        )
    try:
        arena_bounds = ArenaBounds(
            length_m=args.arena_length_m,
            width_m=args.arena_width_m,
            center_x_m=args.arena_center_x_m,
            center_y_m=args.arena_center_y_m,
            yaw_deg=args.arena_yaw_deg,
            margin_m=args.arena_margin_m,
        )
        arena_bounds.validate()
        catalog = load_arrival_pose_catalog(args.catalog)
        input_catalog_sha256 = arrival_pose_catalog_sha256(catalog)
        input_catalog_was_frozen = catalog.frozen
        if args.route_purpose == "logistics" and not input_catalog_was_frozen:
            raise ValueError(
                "logistics arrival-pose catalog must already be frozen"
            )
        grid, map_bundle = load_occupancy_grid_with_bundle(
            args.map,
            semantic_map_id=args.semantic_map_id or args.map.stem,
            planning_frame=args.map_frame,
        )
        map_sha256 = map_bundle.yaml_sha256
        if args.map_bundle_json is not None:
            expected_map_bundle = load_frozen_map_bundle(
                args.map_bundle_json,
                required_semantic_map_id=map_bundle.semantic_map_id,
                required_planning_frame=args.map_frame,
            )
            if expected_map_bundle.bundle_sha256 != map_bundle.bundle_sha256:
                raise ValueError("planning map differs from frozen map bundle")
        candidate_snapshot = None
        identity_registry = None
        parent_survey_manifest = None
        validated_task = None
        task_checked_at_sec = None
        if args.survey_manifest is not None:
            parent_survey_manifest = load_survey_manifest(args.survey_manifest)
        if args.candidate_snapshot is not None:
            candidate_snapshot = load_candidate_snapshot(
                args.candidate_snapshot,
                required_map_bundle_sha256=map_bundle.bundle_sha256,
            )
            identity_registry = load_station_identity_registry(
                args.station_identity_registry,
                candidate_snapshot=candidate_snapshot,
            )
            if tuple(catalog.expected_candidate_uids) != tuple(
                candidate_snapshot.candidate_uids
            ):
                raise ValueError(
                    "catalog expected candidate set differs from frozen snapshot"
                )
        if args.route_purpose == "logistics":
            validated_task = load_validated_task_snapshot(args.task_snapshot)
            if validated_task.robot_id != args.robot_id:
                raise ValueError("validated task robot_id differs from --robot-id")
            if (
                not validated_task.ordered_station_ids
                or validated_task.ordered_station_ids[0]
                != validated_task.target_station
            ):
                raise ValueError(
                    "validated task order must begin with its current target"
                )
            if args.fixed_station_order and tuple(
                validated_task.ordered_station_ids
            ) != tuple(args.fixed_station_order):
                raise ValueError(
                    "--fixed-station-order differs from immutable server task order"
                )
            task_checked_at_sec = time.time()
            _validate_task_snapshot_freshness(
                validated_task,
                now_sec=task_checked_at_sec,
                max_age_sec=args.max_task_snapshot_age_sec,
                max_future_skew_sec=args.max_task_future_skew_sec,
            )
            required_catalog_bindings = {
                "map bundle": catalog.provenance.map_bundle_sha256,
                "candidate snapshot": (
                    catalog.provenance.candidate_snapshot_sha256
                ),
                "station identity registry": (
                    catalog.provenance.station_identity_registry_sha256
                ),
                "survey configuration": catalog.provenance.survey_config_sha256,
                "calibration profile": (
                    catalog.provenance.calibration_profile_sha256
                ),
                "survey input binding": (
                    catalog.provenance.survey_input_binding_sha256
                ),
            }
            missing_bindings = [
                name for name, digest in required_catalog_bindings.items() if not digest
            ]
            if missing_bindings:
                raise ValueError(
                    "logistics catalog lacks sealed survey provenance: "
                    + ", ".join(missing_bindings)
                )
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
        if args.route_purpose == "survey":
            catalog = freeze_arrival_pose_catalog(
                catalog,
                frozen_unix_sec=max(time.time(), catalog.updated_unix_sec),
            )
        # The source catalog is published only after every derived route
        # artifact succeeds.  A failed A* or serializer therefore cannot leave
        # an open survey catalog irreversibly frozen with no usable plan.
        catalog_sha256 = arrival_pose_catalog_sha256(catalog)
        if not catalog.records:
            raise ValueError("frozen catalog has no ready arrival poses")
        if parent_survey_manifest is not None:
            if (
                parent_survey_manifest.planning_frame
                != catalog.provenance.planning_frame
            ):
                raise ValueError(
                    "survey manifest planning frame differs from catalog"
                )
            if (
                parent_survey_manifest.environment
                != catalog.provenance.environment
            ):
                raise ValueError(
                    "survey manifest environment differs from catalog"
                )
            if parent_survey_manifest.session_id != catalog.provenance.session_id:
                raise ValueError("survey manifest session differs from catalog")
            if (
                parent_survey_manifest.environment_descriptor.artifact_id
                != catalog.provenance.world_id
                or parent_survey_manifest.environment_descriptor.sha256
                != catalog.provenance.world_sha256
            ):
                raise ValueError(
                    "survey manifest environment descriptor differs from catalog"
                )
            if parent_survey_manifest.map_bundle.artifact_id != (
                map_bundle.semantic_map_id
            ):
                raise ValueError(
                    "survey manifest semantic map identity differs from planner map"
                )
            if parent_survey_manifest.map_bundle.sha256 != map_bundle.bundle_sha256:
                raise ValueError("survey manifest map bundle differs from planner map")
            if candidate_snapshot is not None and (
                parent_survey_manifest.candidate_snapshot.sha256
                != candidate_snapshot_sha256(candidate_snapshot)
            ):
                raise ValueError(
                    "survey manifest candidate snapshot differs from planner input"
                )
            if (
                catalog.provenance.map_bundle_sha256
                and catalog.provenance.map_bundle_sha256
                != map_bundle.bundle_sha256
            ):
                raise ValueError("catalog frozen map bundle differs from planner map")
            if candidate_snapshot is not None and (
                catalog.provenance.candidate_snapshot_sha256
                != candidate_snapshot_sha256(candidate_snapshot)
            ):
                raise ValueError(
                    "catalog frozen candidate snapshot differs from planner input"
                )
            if identity_registry is not None and (
                catalog.provenance.station_identity_registry_sha256
                != station_identity_registry_sha256(identity_registry)
            ):
                raise ValueError(
                    "catalog station identity registry differs from planner input"
                )
            if (
                catalog.provenance.survey_config_sha256
                and (
                    catalog.provenance.survey_config_sha256
                    != parent_survey_manifest.survey_config.sha256
                    or catalog.provenance.calibration_profile_sha256
                    != parent_survey_manifest.calibration_profile.sha256
                )
            ):
                raise ValueError(
                    "survey manifest configuration/calibration differs from catalog"
                )
            if parent_survey_manifest.arrival_pose_catalog.artifact_id != (
                catalog.catalog_id
            ):
                raise ValueError(
                    "survey manifest catalog identity differs from planner catalog"
                )
            if (
                parent_survey_manifest.arrival_pose_catalog.sha256
                != catalog_sha256
            ):
                raise ValueError(
                    "survey manifest arrival catalog differs from planner catalog"
                )
            survey_config_path = (
                args.survey_config_json
                if args.survey_config_json is not None
                else args.survey_manifest.parent
                / (
                    "survey_config_"
                    f"{parent_survey_manifest.survey_config.sha256}.json"
                )
            )
            survey_arena_bounds = _load_bound_survey_arena_bounds(
                survey_config_path,
                manifest_sha256=parent_survey_manifest.survey_config.sha256,
                catalog_sha256=catalog.provenance.survey_config_sha256,
            )
            if survey_arena_bounds != arena_bounds:
                raise ValueError(
                    "planner arena bounds differ from bound survey configuration"
                )

        costmap = Costmap.from_occupancy_grid(grid).with_arena_bounds(
            arena_bounds
        )
        required_static_inflation = minimum_static_obstacle_inflation_m(
            robot_radius_m=args.robot_radius_m,
            tracking_margin_m=args.tracking_margin_m,
            lidar_stop_distance_m=args.lidar_stop_distance_m,
            scan_origin_to_base_offset_m=args.scan_origin_to_base_offset_m,
            lidar_clearance_margin_m=args.lidar_clearance_margin_m,
        )
        inflation = (
            required_static_inflation
            if args.inflation_radius_m is None
            else args.inflation_radius_m
        )
        if not math.isfinite(inflation) or inflation <= 0.0:
            raise ValueError(
                "configuration-space inflation radius must be finite and positive"
            )
        if inflation + 1.0e-12 < required_static_inflation:
            raise ValueError(
                "configuration-space inflation must cover both the robot body "
                "and live LiDAR stop distance along the certified tracking tube"
            )
        costmap = costmap.with_inflation(inflation)
        if candidate_snapshot is not None:
            for record in catalog.records:
                frozen_candidate = candidate_snapshot.candidate_for(
                    record.candidate_uid
                )
                if frozen_candidate is None:
                    raise ValueError(
                        f"catalog candidate is absent from snapshot: {record.candidate_uid}"
                    )
                center_delta_m = (
                    (record.stand.x_m - frozen_candidate.geometry.x_m) ** 2
                    + (record.stand.y_m - frozen_candidate.geometry.y_m) ** 2
                ) ** 0.5
                allowed_delta_m = (
                    record.stand.uncertainty_m
                    + frozen_candidate.geometry.uncertainty_m
                )
                if center_delta_m > allowed_delta_m + 1.0e-9:
                    raise ValueError(
                        "catalog stand estimate moved outside frozen candidate "
                        f"uncertainty for {record.candidate_uid}"
                    )
        nodes = tuple(
            _route_node(
                record,
                args,
                identity_registry if args.route_purpose == "logistics" else None,
                (
                    None
                    if candidate_snapshot is None
                    else candidate_snapshot.candidate_for(record.candidate_uid)
                ),
            )
            for record in catalog.records
        )
        start = Pose2D(args.start_x, args.start_y, args.start_yaw)
        candidate_to_station = {
            record.candidate_uid: record.candidate_uid for record in catalog.records
        }
        if args.route_purpose == "logistics":
            assert validated_task is not None
            fixed_order = tuple(validated_task.ordered_station_ids)
            required_arrivals = resolve_station_arrival_order(nodes, fixed_order)
            graph = build_required_arrival_route_graph(
                costmap,
                start,
                nodes,
                required_arrivals,
            )
            route_plan = _fixed_task_route_plan(
                graph,
                fixed_order,
                required_arrivals,
            )
        else:
            graph = build_arrival_route_graph(costmap, start, nodes)
            arrivals_by_station_lists: dict[str, list[str]] = {}
            for node in nodes:
                arrivals_by_station_lists.setdefault(node.station_id, []).append(
                    node.arrival_id
                )
            arrivals_by_station = {
                station_id: tuple(arrival_ids)
                for station_id, arrival_ids in arrivals_by_station_lists.items()
            }
            if args.fixed_candidate_order:
                try:
                    fixed_order = tuple(
                        candidate_to_station[candidate_uid]
                        for candidate_uid in args.fixed_candidate_order
                    )
                except KeyError as exc:
                    raise ValueError(
                        f"unknown fixed candidate UID: {exc.args[0]}"
                    ) from exc
            else:
                fixed_order = None
            route_plan = optimize_full_route(
                start_id=graph.start_id,
                arrivals_by_station=arrivals_by_station,
                directed_costs=graph.directed_costs,
                fixed_station_order=fixed_order,
                exact_station_limit=args.exact_station_limit,
            )
        edges = selected_edges(graph, route_plan.arrival_order)
        planned_candidate_uids = tuple(
            arrival_id.split("::", 1)[0]
            for arrival_id in route_plan.arrival_order
        )
        if validated_task is not None:
            task_checked_at_sec = time.time()
            _validate_task_snapshot_freshness(
                validated_task,
                now_sec=task_checked_at_sec,
                max_age_sec=args.max_task_snapshot_age_sec,
                max_future_skew_sec=args.max_task_future_skew_sec,
            )
        _require_catalog_unchanged(
            args.catalog,
            expected_sha256=input_catalog_sha256,
            stage="route planning",
        )
        metadata = {
            "stage": "frozen_arrival_catalog_route",
            "route_purpose": args.route_purpose,
            "task_order_preserved": args.route_purpose == "logistics",
            "station_order": list(route_plan.station_order),
            "candidate_order": list(planned_candidate_uids),
            "simulation_only": catalog.provenance.environment == "simulation",
            "catalog_path": str(args.catalog.resolve()),
            "catalog_id": catalog.catalog_id,
            "catalog_revision": catalog.revision,
            "catalog_sha256": catalog_sha256,
            "input_catalog_sha256": input_catalog_sha256,
            "input_catalog_was_frozen": input_catalog_was_frozen,
            "map_yaml": str(args.map),
            "map_yaml_sha256": map_sha256,
            "map_image_sha256": map_bundle.image_sha256,
            "map_bundle_sha256": map_bundle.bundle_sha256,
            "candidate_snapshot_sha256": (
                ""
                if candidate_snapshot is None
                else candidate_snapshot_sha256(candidate_snapshot)
            ),
            "station_identity_registry_sha256": (
                ""
                if identity_registry is None
                else station_identity_registry_sha256(identity_registry)
            ),
            "task_snapshot_sha256": (
                ""
                if validated_task is None
                else validated_task_snapshot_sha256(validated_task)
            ),
            "server_order_sha256": (
                "" if validated_task is None else validated_task.order_sha256
            ),
            "survey_manifest_sha256": (
                ""
                if parent_survey_manifest is None
                else survey_manifest_sha256(parent_survey_manifest)
            ),
            "survey_manifest_path": (
                ""
                if args.survey_manifest is None
                else str(args.survey_manifest.resolve())
            ),
            "task_snapshot_checked_at_sec": task_checked_at_sec,
            "max_task_snapshot_age_sec": args.max_task_snapshot_age_sec,
            "max_task_future_skew_sec": args.max_task_future_skew_sec,
            "planning_frame": catalog.provenance.planning_frame,
            "world_id": catalog.provenance.world_id,
            "world_sha256": catalog.provenance.world_sha256,
            "session_id": catalog.provenance.session_id,
            "arena_bounds": arena_bounds.to_metadata(),
            "arena_boundary_overlay": True,
            "static_inflation_radius_m": inflation,
            "required_static_inflation_radius_m": required_static_inflation,
            "non_target_stand_keepout_policy": (
                "max(body_uncertainty_collision,lidar_minimum_standoff,"
                "frozen_candidate_keepout)+certified_tracking_tube"
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
        certificate = ExecutionRouteCertificate(
            route_sha256=metadata["route_csv_sha256"],
            planning_frame=catalog.provenance.planning_frame,
            route_kind="catalog_face_approach",
            waypoint_count=sum(
                len(edge.result.plan.waypoints)
                for edge in edges
                if edge.result.plan is not None
            ),
            tracking_tube_radius_m=args.tracking_margin_m,
            exact_vertex_pursuit=True,
            command_owner=args.command_owner,
            map_bundle_sha256=map_bundle.bundle_sha256,
            candidate_snapshot_sha256=(
                ""
                if candidate_snapshot is None
                else candidate_snapshot_sha256(candidate_snapshot)
            ),
        )
        if args.route_certificate_json is None:
            args.route_certificate_json = args.route_csv.with_name(
                f"{args.route_csv.stem}_certificate_"
                f"{metadata['route_csv_sha256'][:16]}.json"
            )
        metadata["route_certificate_sha256"] = (
            write_execution_route_certificate(
                args.route_certificate_json,
                certificate,
            )
        )
        metadata["route_certificate_path"] = str(
            args.route_certificate_json.resolve()
        )
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
                "station_order": list(route_plan.station_order),
                "candidate_order": list(planned_candidate_uids),
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
        mission_plan_path = None
        mission_plan_sha256 = None
        planner_config_path = None
        planner_config_sha256 = None
        route_bundle_path = None
        route_bundle_sha256 = None
        if args.route_purpose == "logistics":
            assert parent_survey_manifest is not None
            assert validated_task is not None
            assert identity_registry is not None
            assert candidate_snapshot is not None
            ordered_candidate_uids = candidate_order_for_server_order(
                identity_registry,
                route_plan.station_order,
            )
            if ordered_candidate_uids != planned_candidate_uids:
                raise ValueError(
                    "planned arrivals differ from the server-ordered identity mapping"
                )
            planner_config_payload = {
                "schema_version": ARTIFACT_DESCRIPTOR_SCHEMA_VERSION,
                "artifact_kind": "planner_config",
                "route_purpose": args.route_purpose,
                "start_pose": {
                    "x_m": args.start_x,
                    "y_m": args.start_y,
                    "yaw_rad": args.start_yaw,
                },
                "robot_radius_m": args.robot_radius_m,
                "tracking_margin_m": args.tracking_margin_m,
                "collision_margin_m": args.collision_margin_m,
                "inflation_radius_m": inflation,
                "corridor_sample_spacing_m": args.corridor_sample_spacing_m,
                "lidar_stop_distance_m": args.lidar_stop_distance_m,
                "scan_origin_to_base_offset_m": (
                    args.scan_origin_to_base_offset_m
                ),
                "lidar_clearance_margin_m": args.lidar_clearance_margin_m,
                "arena_bounds": arena_bounds.to_metadata(),
                "arena_boundary_overlay": True,
                "command_owner": args.command_owner,
                "algorithm": route_plan.algorithm,
                "max_task_snapshot_age_sec": args.max_task_snapshot_age_sec,
                "max_task_future_skew_sec": args.max_task_future_skew_sec,
            }
            planner_config_sha256 = payload_sha256(planner_config_payload)
            planner_config_id = f"planner_config_{planner_config_sha256}"
            planner_config_path = (
                args.planner_config_json
                or args.route_csv.with_name(f"{planner_config_id}.json")
            )

            route_bundle_payload = {
                "schema_version": ARTIFACT_DESCRIPTOR_SCHEMA_VERSION,
                "artifact_kind": "route_bundle",
                "route_csv_sha256": _file_sha256(args.route_csv),
                "diagnostics_sha256": _file_sha256(args.diagnostics_json),
                "visit_order_sha256": _file_sha256(visit_path),
                "required_edge_costs_sha256": _file_sha256(costs_path),
                "catalog_snapshot_sha256": _file_sha256(snapshot_path),
                "route_certificate_sha256": metadata[
                    "route_certificate_sha256"
                ],
            }
            route_bundle_sha256 = payload_sha256(route_bundle_payload)
            route_bundle_id = f"route_bundle_{route_bundle_sha256}"
            route_bundle_path = (
                args.route_bundle_json
                or args.route_csv.with_name(f"{route_bundle_id}.json")
            )
            mission_plan = MissionPlanManifest(
                schema_version=ARTIFACT_MANIFEST_SCHEMA_VERSION,
                manifest_id=(
                    f"mission_{validated_task.mission_id}_"
                    f"{route_bundle_sha256[:16]}"
                ),
                created_unix_sec=max(
                    catalog.updated_unix_sec,
                    float(validated_task.validated_at_sec),
                    float(task_checked_at_sec),
                ),
                robot_id=args.robot_id,
                parent_survey_manifest=manifest_reference(
                    parent_survey_manifest,
                    survey_manifest_sha256(parent_survey_manifest),
                ),
                map_bundle=parent_survey_manifest.map_bundle,
                candidate_snapshot=parent_survey_manifest.candidate_snapshot,
                station_identity_registry=artifact_reference(
                    "station_identity_registry",
                    identity_registry.registry_id,
                    station_identity_registry_sha256(identity_registry),
                ),
                arrival_pose_catalog=parent_survey_manifest.arrival_pose_catalog,
                task_snapshot=artifact_reference(
                    "task_snapshot",
                    f"task_{validated_task.mission_id}",
                    validated_task_snapshot_sha256(validated_task),
                ),
                planner_config=artifact_reference(
                    "planner_config",
                    planner_config_id,
                    planner_config_sha256,
                ),
                route_bundle=artifact_reference(
                    "route_bundle",
                    route_bundle_id,
                    route_bundle_sha256,
                ),
                required_station_order=tuple(route_plan.station_order),
                ordered_candidate_uids=ordered_candidate_uids,
            )
            mission_plan_path = args.mission_plan_manifest or args.route_csv.with_name(
                f"{mission_plan.manifest_id}.json"
            )
            validate_mission_plan_links(
                mission_plan,
                parent_survey_manifest,
                parent_sha256=survey_manifest_sha256(parent_survey_manifest),
            )
            _require_catalog_unchanged(
                args.catalog,
                expected_sha256=input_catalog_sha256,
                stage="mission manifest publication",
            )
            written_planner_config_sha256 = write_content_hashed_json(
                planner_config_path,
                planner_config_payload,
                hash_field=ARTIFACT_DESCRIPTOR_HASH_FIELD,
            )
            if written_planner_config_sha256 != planner_config_sha256:
                raise ValueError("persisted planner configuration hash mismatch")
            written_route_bundle_sha256 = write_content_hashed_json(
                route_bundle_path,
                route_bundle_payload,
                hash_field=ARTIFACT_DESCRIPTOR_HASH_FIELD,
            )
            if written_route_bundle_sha256 != route_bundle_sha256:
                raise ValueError("persisted route-bundle hash mismatch")
            _require_catalog_unchanged(
                args.catalog,
                expected_sha256=input_catalog_sha256,
                stage="mission manifest publication",
            )
            mission_plan_sha256 = write_mission_plan_manifest(
                mission_plan_path,
                mission_plan,
            )
        _require_catalog_unchanged(
            args.catalog,
            expected_sha256=input_catalog_sha256,
            stage="final publication",
        )
        if args.route_purpose == "survey" and (
            catalog_sha256 != input_catalog_sha256
        ):
            write_arrival_pose_catalog(args.catalog, catalog)
        print(
            json.dumps(
                {
                    "ok": True,
                    "optimal": route_plan.optimal,
                    "algorithm": route_plan.algorithm,
                    "total_cost_m": route_plan.total_cost,
                    "station_order": list(route_plan.station_order),
                    "candidate_order": list(planned_candidate_uids),
                    "route_csv": str(args.route_csv),
                    "diagnostics_json": str(args.diagnostics_json),
                    "catalog_sha256": arrival_pose_catalog_sha256(catalog),
                    "mission_plan_manifest": (
                        "" if mission_plan_path is None else str(mission_plan_path)
                    ),
                    "mission_plan_manifest_sha256": mission_plan_sha256 or "",
                    "planner_config_json": (
                        "" if planner_config_path is None else str(planner_config_path)
                    ),
                    "planner_config_sha256": planner_config_sha256 or "",
                    "route_bundle_json": (
                        "" if route_bundle_path is None else str(route_bundle_path)
                    ),
                    "route_bundle_sha256": route_bundle_sha256 or "",
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
