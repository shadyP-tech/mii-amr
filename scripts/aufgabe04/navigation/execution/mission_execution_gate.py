"""Fail-closed admission of a task-ordered static route for execution."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.artifacts import (
    load_mission_plan_manifest,
    load_survey_manifest,
    mission_plan_manifest_sha256,
    survey_manifest_sha256,
)
from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json,
    payload_sha256,
)
from scripts.aufgabe04.navigation.execution.execution_route_certificate import (
    execution_route_certificate_sha256,
    load_execution_route_certificate,
)
from scripts.aufgabe04.navigation.planning.map_io import load_frozen_map_bundle
from scripts.aufgabe04.navigation.planning.waypoint_csv import SelectedRouteLeg
from scripts.aufgabe04.logistics.server_validation.artifacts import (
    load_validated_task_snapshot,
    validated_task_snapshot_sha256,
)
from scripts.aufgabe04.stations.arrival_pose_catalog import (
    arrival_pose_catalog_sha256,
    load_arrival_pose_catalog,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    candidate_snapshot_sha256,
    load_candidate_snapshot,
)
from scripts.aufgabe04.stations.station_identity_registry import (
    candidate_order_for_server_order,
    load_station_identity_registry,
    station_identity_registry_sha256,
)


ARTIFACT_DESCRIPTOR_HASH_FIELD = "artifact_sha256"
ARTIFACT_DESCRIPTOR_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROUTE_BUNDLE_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "route_csv_sha256",
        "diagnostics_sha256",
        "visit_order_sha256",
        "required_edge_costs_sha256",
        "catalog_snapshot_sha256",
        "route_certificate_sha256",
    }
)
_PLANNER_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "route_purpose",
        "start_pose",
        "robot_radius_m",
        "tracking_margin_m",
        "collision_margin_m",
        "inflation_radius_m",
        "corridor_sample_spacing_m",
        "lidar_stop_distance_m",
        "scan_origin_to_base_offset_m",
        "lidar_clearance_margin_m",
        "arena_bounds",
        "arena_boundary_overlay",
        "command_owner",
        "algorithm",
        "max_task_snapshot_age_sec",
        "max_task_future_skew_sec",
    }
)


@dataclass(frozen=True)
class MissionExecutionBinding:
    mission_plan_sha256: str
    route_bundle_sha256: str
    diagnostics_sha256: str
    expected_candidate_uid: str


@dataclass(frozen=True)
class DiagnosticsSnapshot:
    """One byte read used for both diagnostics parsing and hashing."""

    source_path: Path
    sha256: str
    raw_bytes: bytes
    require_metadata: bool = True

    @property
    def payload(self) -> Mapping[str, object]:
        # Return a fresh tree derived from the bound immutable bytes. A caller
        # cannot mutate the semantic view while leaving ``sha256`` unchanged.
        payload, _metadata = _decode_diagnostics_bytes(
            self.raw_bytes,
            source_path=self.source_path,
            require_metadata=self.require_metadata,
        )
        return payload

    @property
    def metadata(self) -> Mapping[str, object]:
        _payload, metadata = _decode_diagnostics_bytes(
            self.raw_bytes,
            source_path=self.source_path,
            require_metadata=True,
        )
        return metadata


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate diagnostics JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite diagnostics JSON value {value!r}")


def load_diagnostics_snapshot(
    path: Path,
    *,
    require_metadata: bool = True,
) -> DiagnosticsSnapshot:
    source_path = Path(path).resolve(strict=False)
    try:
        raw = source_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"route diagnostics are unavailable or invalid: {exc}") from exc
    _decode_diagnostics_bytes(
        raw,
        source_path=source_path,
        require_metadata=require_metadata,
    )
    return DiagnosticsSnapshot(
        source_path=source_path,
        sha256=hashlib.sha256(raw).hexdigest(),
        raw_bytes=raw,
        require_metadata=require_metadata,
    )


def _decode_diagnostics_bytes(
    raw: bytes,
    *,
    source_path: Path,
    require_metadata: bool,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"route diagnostics are unavailable or invalid at {source_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("route diagnostics root must be an object")
    metadata = payload.get("metadata")
    if require_metadata and not isinstance(metadata, dict):
        raise ValueError("route diagnostics metadata is missing")
    return payload, metadata if isinstance(metadata, dict) else {}


def load_diagnostics_metadata(path: Path) -> Mapping[str, object]:
    return load_diagnostics_snapshot(path).metadata


def route_purpose_from_diagnostics(path: Path) -> str:
    purpose = load_diagnostics_metadata(path).get("route_purpose")
    return purpose if isinstance(purpose, str) else ""


def validate_logistics_execution_bundle(
    *,
    route_leg: SelectedRouteLeg,
    diagnostics_path: Path,
    route_certificate_path: Path | None,
    mission_plan_path: Path,
    survey_manifest_path: Path,
    route_bundle_path: Path,
    planner_config_path: Path,
    runtime_map_bundle_path: Path,
    runtime_environment_path: Path,
    candidate_snapshot_path: Path,
    station_identity_registry_path: Path,
    arrival_pose_catalog_path: Path,
    task_snapshot_path: Path,
    robot_id: str,
    runtime_planning_frame: str,
    now_sec: float | None = None,
    diagnostics_snapshot: DiagnosticsSnapshot | None = None,
) -> MissionExecutionBinding:
    """Validate the mission commit root and every executable child identity."""

    survey = load_survey_manifest(survey_manifest_path)
    survey_sha256 = survey_manifest_sha256(survey)
    mission = load_mission_plan_manifest(
        mission_plan_path,
        parent_survey=survey,
    )
    mission_sha256 = mission_plan_manifest_sha256(mission)
    if mission.robot_id != robot_id:
        raise ValueError(
            "mission plan robot differs from execution robot: "
            f"mission={mission.robot_id!r}, runtime={robot_id!r}"
        )
    if mission.parent_survey_manifest.sha256 != survey_sha256:
        raise ValueError("mission plan references another survey manifest")

    route_bundle = load_content_hashed_json(
        route_bundle_path,
        hash_field=ARTIFACT_DESCRIPTOR_HASH_FIELD,
    )
    validate_route_bundle_descriptor(route_bundle)
    route_bundle_sha256 = payload_sha256(route_bundle)
    if mission.route_bundle.sha256 != route_bundle_sha256:
        raise ValueError("mission plan references another route bundle")
    if mission.route_bundle.artifact_id != f"route_bundle_{route_bundle_sha256}":
        raise ValueError("mission route-bundle artifact id is not content-addressed")

    planner_config = load_content_hashed_json(
        planner_config_path,
        hash_field=ARTIFACT_DESCRIPTOR_HASH_FIELD,
    )
    validate_planner_config_descriptor(planner_config)
    planner_config_sha256 = payload_sha256(planner_config)
    if mission.planner_config.sha256 != planner_config_sha256:
        raise ValueError("mission plan references another planner configuration")
    if mission.planner_config.artifact_id != (
        f"planner_config_{planner_config_sha256}"
    ):
        raise ValueError("mission planner-config artifact id is not content-addressed")
    if planner_config.get("route_purpose") != "logistics":
        raise ValueError("planner configuration is not a logistics plan")

    snapshot = diagnostics_snapshot or load_diagnostics_snapshot(diagnostics_path)
    if snapshot.source_path != Path(diagnostics_path).resolve(strict=False):
        raise ValueError("diagnostics snapshot was loaded from another path")
    if hashlib.sha256(snapshot.raw_bytes).hexdigest() != snapshot.sha256:
        raise ValueError("diagnostics snapshot digest does not match its bound bytes")
    diagnostics_sha256 = snapshot.sha256
    expected_hashes = {
        "route_csv_sha256": route_leg.source_sha256,
        "diagnostics_sha256": diagnostics_sha256,
    }
    for field, actual in expected_hashes.items():
        if route_bundle.get(field) != actual:
            raise ValueError(f"route bundle {field} mismatch")

    metadata = snapshot.metadata
    if metadata.get("route_purpose") != "logistics":
        raise ValueError("mission execution requires route_purpose=logistics")
    if metadata.get("planning_frame") != runtime_planning_frame:
        raise ValueError("mission route planning frame differs from runtime map frame")
    if metadata.get("arena_boundary_overlay") is not True:
        raise ValueError("mission route lacks the physical arena-boundary overlay")
    if metadata.get("arena_bounds") != planner_config.get("arena_bounds"):
        raise ValueError(
            "route diagnostics arena bounds differ from planner configuration"
        )

    recorded_certificate = metadata.get("route_certificate_path")
    certificate_path = route_certificate_path
    if certificate_path is None:
        if not isinstance(recorded_certificate, str) or not recorded_certificate:
            raise ValueError("logistics diagnostics do not reference a certificate")
        certificate_path = Path(recorded_certificate)
    certificate = load_execution_route_certificate(certificate_path)
    certificate_sha256 = execution_route_certificate_sha256(certificate)
    if route_bundle.get("route_certificate_sha256") != certificate_sha256:
        raise ValueError("route bundle certificate SHA-256 mismatch")
    if planner_config.get("command_owner") != certificate.command_owner:
        raise ValueError("planner command owner differs from route certificate")
    if planner_config.get("tracking_margin_m") != (
        certificate.tracking_tube_radius_m
    ):
        raise ValueError("planner tracking margin differs from route certificate")

    reference_checks = {
        "map_bundle_sha256": mission.map_bundle.sha256,
        "candidate_snapshot_sha256": mission.candidate_snapshot.sha256,
        "station_identity_registry_sha256": (
            mission.station_identity_registry.sha256
        ),
        "catalog_sha256": mission.arrival_pose_catalog.sha256,
        "task_snapshot_sha256": mission.task_snapshot.sha256,
        "survey_manifest_sha256": mission.parent_survey_manifest.sha256,
    }
    for field, expected in reference_checks.items():
        if metadata.get(field) != expected:
            raise ValueError(f"mission plan {field} differs from route diagnostics")

    station_order = metadata.get("station_order")
    candidate_order = metadata.get("candidate_order")
    if not isinstance(station_order, list) or tuple(station_order) != (
        mission.required_station_order
    ):
        raise ValueError("diagnostics station order differs from mission plan")
    if not isinstance(candidate_order, list) or tuple(candidate_order) != (
        mission.ordered_candidate_uids
    ):
        raise ValueError("diagnostics candidate order differs from mission plan")
    if not 0 <= route_leg.leg_index < len(mission.ordered_candidate_uids):
        raise ValueError("selected leg is outside the mission order")
    expected_candidate_uid = mission.ordered_candidate_uids[route_leg.leg_index]
    target_candidate_uid = route_leg.target_arrival_id.split("::", 1)[0]
    if target_candidate_uid != expected_candidate_uid:
        raise ValueError("selected route leg target differs from mission order")

    runtime_map_bundle = load_frozen_map_bundle(
        runtime_map_bundle_path,
        required_planning_frame=runtime_planning_frame,
    )
    if runtime_map_bundle.bundle_sha256 != mission.map_bundle.sha256:
        raise ValueError("runtime map bundle differs from mission plan")
    if runtime_map_bundle.semantic_map_id != mission.map_bundle.artifact_id:
        raise ValueError("runtime semantic map id differs from mission plan")

    candidate_snapshot = load_candidate_snapshot(
        candidate_snapshot_path,
        required_map_bundle_sha256=runtime_map_bundle.bundle_sha256,
    )
    if candidate_snapshot_sha256(candidate_snapshot) != (
        mission.candidate_snapshot.sha256
    ):
        raise ValueError("candidate snapshot differs from mission plan")
    if candidate_snapshot.snapshot_id != mission.candidate_snapshot.artifact_id:
        raise ValueError("candidate snapshot id differs from mission plan")
    identity_registry = load_station_identity_registry(
        station_identity_registry_path,
        candidate_snapshot=candidate_snapshot,
    )
    if station_identity_registry_sha256(identity_registry) != (
        mission.station_identity_registry.sha256
    ):
        raise ValueError("station identity registry differs from mission plan")
    if identity_registry.registry_id != mission.station_identity_registry.artifact_id:
        raise ValueError("station identity registry id differs from mission plan")
    catalog = load_arrival_pose_catalog(arrival_pose_catalog_path)
    if not catalog.frozen:
        raise ValueError("mission arrival-pose catalog is not frozen")
    if arrival_pose_catalog_sha256(catalog) != mission.arrival_pose_catalog.sha256:
        raise ValueError("arrival-pose catalog differs from mission plan")
    if catalog.catalog_id != mission.arrival_pose_catalog.artifact_id:
        raise ValueError("arrival-pose catalog id differs from mission plan")
    if tuple(catalog.expected_candidate_uids) != candidate_snapshot.candidate_uids:
        raise ValueError("catalog candidate set differs from mission snapshot")

    task = load_validated_task_snapshot(task_snapshot_path)
    if validated_task_snapshot_sha256(task) != mission.task_snapshot.sha256:
        raise ValueError("validated task snapshot differs from mission plan")
    if mission.task_snapshot.artifact_id != f"task_{task.mission_id}":
        raise ValueError("validated task artifact id differs from mission plan")
    if task.robot_id != mission.robot_id:
        raise ValueError("validated task robot differs from mission plan")
    if tuple(task.ordered_station_ids) != mission.required_station_order:
        raise ValueError("validated task order differs from mission plan")
    if task.order_sha256 != metadata.get("server_order_sha256"):
        raise ValueError("validated task order digest differs from route diagnostics")
    if candidate_order_for_server_order(
        identity_registry,
        task.ordered_station_ids,
    ) != mission.ordered_candidate_uids:
        raise ValueError("identity registry mapping differs from mission candidate order")

    if file_sha256(runtime_environment_path) != (
        survey.environment_descriptor.sha256
    ):
        raise ValueError("runtime environment descriptor differs from survey")
    if runtime_environment_path.stem != survey.environment_descriptor.artifact_id:
        raise ValueError("runtime environment id differs from survey")

    checked_at = _finite_number(
        metadata.get("task_snapshot_checked_at_sec"),
        "task_snapshot_checked_at_sec",
    )
    max_age = _positive_number(
        metadata.get("max_task_snapshot_age_sec"),
        "max_task_snapshot_age_sec",
    )
    max_future_skew = _nonnegative_number(
        metadata.get("max_task_future_skew_sec"),
        "max_task_future_skew_sec",
    )
    now = time.time() if now_sec is None else float(now_sec)
    if not math.isfinite(now) or now < 0.0:
        raise ValueError("execution clock must be finite and non-negative")
    if checked_at - now > max_future_skew:
        raise ValueError("task snapshot check time is too far in the future")
    if now - checked_at > max_age:
        raise ValueError("task-ordered mission plan is stale; revalidate and replan")
    validated_at = _finite_number(task.validated_at_sec, "validated_at_sec")
    if validated_at - now > max_future_skew:
        raise ValueError("validated task timestamp is too far in the future")
    if now - validated_at > max_age:
        raise ValueError("validated task snapshot is stale; refetch and replan")
    for name, value in (
        ("status_observed_at_sec", task.status_observed_at_sec),
        ("plan_generated_at_sec", task.plan_generated_at_sec),
    ):
        source_time = _finite_number(value, name)
        if source_time - now > max_future_skew:
            raise ValueError(f"validated task {name} is too far in the future")
        if source_time > validated_at + max_future_skew:
            raise ValueError(f"validated task {name} postdates validation")

    return MissionExecutionBinding(
        mission_plan_sha256=mission_sha256,
        route_bundle_sha256=route_bundle_sha256,
        diagnostics_sha256=diagnostics_sha256,
        expected_candidate_uid=expected_candidate_uid,
    )


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_number(value: object, name: str) -> float:
    result = _finite_number(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_number(value: object, name: str) -> float:
    result = _finite_number(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_exact_descriptor(
    payload: Mapping[str, object],
    *,
    name: str,
    expected_fields: frozenset[str],
    artifact_kind: str,
) -> None:
    actual_fields = frozenset(payload)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        unknown = sorted(actual_fields - expected_fields)
        raise ValueError(
            f"{name} descriptor fields differ from schema: "
            f"missing={missing}, unknown={unknown}"
        )
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != (
        ARTIFACT_DESCRIPTOR_SCHEMA_VERSION
    ):
        raise ValueError(f"{name} descriptor has unsupported schema_version")
    if payload.get("artifact_kind") != artifact_kind:
        raise ValueError(f"{name} descriptor has the wrong artifact kind")


def _validate_planner_config(payload: Mapping[str, object]) -> None:
    if payload.get("route_purpose") != "logistics":
        raise ValueError("planner configuration is not a logistics plan")
    start_pose = payload.get("start_pose")
    if not isinstance(start_pose, Mapping) or frozenset(start_pose) != frozenset(
        {"x_m", "y_m", "yaw_rad"}
    ):
        raise ValueError("planner start_pose must contain exactly x_m, y_m, yaw_rad")
    for field in ("x_m", "y_m", "yaw_rad"):
        _finite_number(start_pose.get(field), f"planner start_pose.{field}")
    for field in (
        "robot_radius_m",
        "inflation_radius_m",
        "corridor_sample_spacing_m",
        "lidar_stop_distance_m",
        "max_task_snapshot_age_sec",
    ):
        _positive_number(payload.get(field), f"planner {field}")
    for field in (
        "tracking_margin_m",
        "collision_margin_m",
        "lidar_clearance_margin_m",
        "max_task_future_skew_sec",
    ):
        _nonnegative_number(payload.get(field), f"planner {field}")
    _finite_number(
        payload.get("scan_origin_to_base_offset_m"),
        "planner scan_origin_to_base_offset_m",
    )
    arena_bounds = payload.get("arena_bounds")
    expected_arena_fields = frozenset(
        {
            "length_m",
            "width_m",
            "center_x_m",
            "center_y_m",
            "yaw_deg",
            "margin_m",
        }
    )
    if (
        not isinstance(arena_bounds, Mapping)
        or frozenset(arena_bounds) != expected_arena_fields
    ):
        raise ValueError(
            "planner arena_bounds must contain exactly length_m, width_m, "
            "center_x_m, center_y_m, yaw_deg, margin_m"
        )
    length_m = _positive_number(
        arena_bounds.get("length_m"),
        "planner arena_bounds.length_m",
    )
    width_m = _positive_number(
        arena_bounds.get("width_m"),
        "planner arena_bounds.width_m",
    )
    margin_m = _nonnegative_number(
        arena_bounds.get("margin_m"),
        "planner arena_bounds.margin_m",
    )
    for field in ("center_x_m", "center_y_m", "yaw_deg"):
        _finite_number(
            arena_bounds.get(field),
            f"planner arena_bounds.{field}",
        )
    if 2.0 * margin_m >= length_m or 2.0 * margin_m >= width_m:
        raise ValueError("planner arena margin leaves no usable arena")
    if payload.get("arena_boundary_overlay") is not True:
        raise ValueError("planner physical arena-boundary overlay must be enabled")
    minimum_inflation = float(payload["robot_radius_m"]) + float(
        payload["tracking_margin_m"]
    )
    if float(payload["inflation_radius_m"]) + 1.0e-12 < minimum_inflation:
        raise ValueError(
            "planner inflation radius is smaller than robot radius plus tracking margin"
        )
    command_owner = payload.get("command_owner")
    if (
        not isinstance(command_owner, str)
        or not command_owner.startswith("/")
        or command_owner == "/"
    ):
        raise ValueError("planner command_owner must be an absolute node identity")
    algorithm = payload.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm.strip():
        raise ValueError("planner algorithm must be non-empty")


def validate_route_bundle_descriptor(payload: Mapping[str, object]) -> None:
    _require_exact_descriptor(
        payload,
        name="route-bundle",
        expected_fields=_ROUTE_BUNDLE_FIELDS,
        artifact_kind="route_bundle",
    )
    for field in _ROUTE_BUNDLE_FIELDS:
        if field.endswith("_sha256"):
            _require_sha256(payload.get(field), f"route-bundle {field}")


def validate_planner_config_descriptor(payload: Mapping[str, object]) -> None:
    _require_exact_descriptor(
        payload,
        name="planner-config",
        expected_fields=_PLANNER_CONFIG_FIELDS,
        artifact_kind="planner_config",
    )
    _validate_planner_config(payload)
