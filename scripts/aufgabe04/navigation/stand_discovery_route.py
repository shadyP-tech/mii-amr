"""Seal and validate one real center-corridor stand-discovery route."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.navigation.dynamic_route_handoff import (
    validate_arena_boundary_evidence,
)
from scripts.aufgabe04.navigation.execution_route_certificate import (
    ExecutionRouteCertificate,
    file_sha256,
    write_execution_route_certificate,
)
from scripts.aufgabe04.navigation.safety_checks import PreflightStatus
from scripts.aufgabe04.navigation.stand_coverage_survey import (
    coverage_survey_plan_sha256,
    load_coverage_survey_plan,
)
from scripts.aufgabe04.navigation.waypoint_csv import SelectedRouteLeg, load_route_leg


STAND_DISCOVERY_ROUTE_KIND = "stand_discovery_corridor"
STAND_DISCOVERY_ROUTE_PURPOSE = "stand_discovery"
STAND_DISCOVERY_ROUTE_SOURCE = "map_based_center_corridor_exploration"
DEFAULT_TRACKING_TUBE_RADIUS_M = 0.03
DEFAULT_COMMAND_OWNER = "/aufgabe04_simple_waypoint_follower"


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _metadata(payload: Mapping[str, object]) -> Mapping[str, object]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("route diagnostics are missing metadata")
    return metadata


def validate_stand_discovery_route_binding(
    diagnostics_path: Path,
    leg: SelectedRouteLeg,
    *,
    coverage_plan_path: Path | None,
    diagnostics_payload: Mapping[str, object] | None = None,
    endpoint_tolerance_m: float = 0.08,
) -> PreflightStatus:
    """Validate the map, centerline plan, route bytes, and stopped endpoint."""

    failures: list[str] = []
    if leg.route_kind != STAND_DISCOVERY_ROUTE_KIND:
        return PreflightStatus(
            ok=False,
            failures=["stand discovery binding requires stand_discovery_corridor"],
        )
    if leg.simulation_only:
        failures.append("stand discovery route must not be simulation_only")
    if len(leg.raw_waypoints) < 2:
        failures.append("stand discovery route has fewer than two waypoints")
        return PreflightStatus(ok=False, failures=failures)
    final = leg.raw_waypoints[-1]
    if not final.protected or not final.corridor:
        failures.append("stand discovery final waypoint must be protected and corridor")
    if not math.isfinite(final.pose.yaw_rad):
        failures.append("stand discovery final waypoint yaw is unconstrained")

    try:
        payload = (
            _load_json(diagnostics_path)
            if diagnostics_payload is None
            else diagnostics_payload
        )
        metadata = _metadata(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        failures.append(f"invalid stand discovery diagnostics: {exc}")
        return PreflightStatus(ok=False, failures=failures)

    if metadata.get("route_kind") != STAND_DISCOVERY_ROUTE_KIND:
        failures.append("stand discovery route kind does not match diagnostics")
    if metadata.get("route_purpose") != STAND_DISCOVERY_ROUTE_PURPOSE:
        failures.append("stand discovery route_purpose must be stand_discovery")
    if metadata.get("source") != STAND_DISCOVERY_ROUTE_SOURCE:
        failures.append("stand discovery diagnostics have an unexpected source")
    if metadata.get("motion_authorized") is not True:
        failures.append("stand discovery route was not explicitly sealed for motion")
    if metadata.get("physical_clearance_enforced") is not True:
        failures.append("stand discovery physical clearance was not enforced")
    if metadata.get("route_csv_sha256") != leg.source_sha256:
        failures.append("stand discovery route CSV SHA-256 does not match diagnostics")
    try:
        validate_arena_boundary_evidence(metadata)
    except ValueError as exc:
        failures.append(f"arena boundary evidence is invalid: {exc}")

    if coverage_plan_path is None:
        failures.append("stand discovery route requires --coverage-plan")
        return PreflightStatus(ok=False, failures=failures)
    try:
        plan = load_coverage_survey_plan(coverage_plan_path)
    except (OSError, ValueError) as exc:
        failures.append(f"coverage plan validation failed: {exc}")
        return PreflightStatus(ok=False, failures=failures)
    if plan.config.lane_count != 1:
        failures.append("stand discovery physical route requires one center lane")
    if metadata.get("survey_id") != plan.survey_id:
        failures.append("stand discovery survey ID differs from coverage plan")
    if metadata.get("plan_sha256") != coverage_survey_plan_sha256(plan):
        failures.append("stand discovery plan SHA-256 differs from coverage plan")
    if metadata.get("map_bundle_sha256") != plan.map_bundle_sha256:
        failures.append("stand discovery map bundle differs from coverage plan")
    if metadata.get("planning_frame") != plan.planning_frame:
        failures.append("stand discovery planning frame differs from coverage plan")

    target_id = metadata.get("target_viewpoint_id")
    target = plan.viewpoint_for(str(target_id))
    if target is None:
        failures.append("stand discovery target viewpoint is absent from coverage plan")
    else:
        error_m = math.hypot(
            final.pose.x_m - target.pose.x_m,
            final.pose.y_m - target.pose.y_m,
        )
        if error_m > endpoint_tolerance_m:
            failures.append(
                "stand discovery route endpoint differs from planned viewpoint: "
                f"{error_m:.3f} m"
            )
    return PreflightStatus(ok=not failures, failures=failures)


def seal_stand_discovery_route(
    *,
    source_route_csv: Path,
    source_diagnostics_json: Path,
    coverage_plan_path: Path,
    output_dir: Path,
    command_owner: str = DEFAULT_COMMAND_OWNER,
    tracking_tube_radius_m: float = DEFAULT_TRACKING_TUBE_RADIUS_M,
) -> dict[str, str]:
    """Turn one motion-free survey leg into a certified real route."""

    source_route_csv = Path(source_route_csv)
    source_diagnostics_json = Path(source_diagnostics_json)
    coverage_plan_path = Path(coverage_plan_path)
    output_dir = Path(output_dir)
    outputs = {
        "route_csv": output_dir / "route.csv",
        "diagnostics_json": output_dir / "route_diagnostics.json",
        "route_certificate_json": output_dir / "route_certificate.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise ValueError(
            "refusing to overwrite sealed stand discovery artifacts: "
            + ", ".join(existing)
        )
    source_payload = _load_json(source_diagnostics_json)
    source_metadata = dict(_metadata(source_payload))
    if source_metadata.get("route_kind") != "stand_coverage_survey":
        raise ValueError("source route is not a stand coverage survey leg")
    if source_metadata.get("motion_authorized") is not False:
        raise ValueError("source survey leg must be explicitly motion-free")

    plan = load_coverage_survey_plan(coverage_plan_path)
    if plan.config.lane_count != 1:
        raise ValueError("real stand discovery requires a single center lane")
    if source_metadata.get("plan_sha256") != coverage_survey_plan_sha256(plan):
        raise ValueError("source survey leg belongs to another coverage plan")
    if source_metadata.get("map_bundle_sha256") != plan.map_bundle_sha256:
        raise ValueError("source survey leg map differs from coverage plan")
    target_id = str(source_metadata.get("target_viewpoint_id", ""))
    if plan.viewpoint_for(target_id) is None:
        raise ValueError("source survey leg target is absent from coverage plan")
    validate_arena_boundary_evidence(source_metadata)

    with source_route_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("source route CSV is missing a header")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    if len(rows) < 2:
        raise ValueError("source stand discovery route has fewer than two waypoints")
    for field in ("protected", "corridor", "simulation_only", "route_kind"):
        if field not in fieldnames:
            fieldnames.append(field)
    for index, row in enumerate(rows):
        row["protected"] = "true" if index == len(rows) - 1 else "false"
        row["corridor"] = "true" if index == len(rows) - 1 else "false"
        row["simulation_only"] = "false"
        row["route_kind"] = STAND_DISCOVERY_ROUTE_KIND

    output_dir.mkdir(parents=True, exist_ok=True)
    with outputs["route_csv"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    route_digest = file_sha256(outputs["route_csv"])
    certificate = ExecutionRouteCertificate(
        route_sha256=route_digest,
        planning_frame=plan.planning_frame,
        route_kind=STAND_DISCOVERY_ROUTE_KIND,
        waypoint_count=len(rows),
        tracking_tube_radius_m=tracking_tube_radius_m,
        exact_vertex_pursuit=True,
        command_owner=command_owner,
        map_bundle_sha256=plan.map_bundle_sha256,
    )
    certificate_digest = write_execution_route_certificate(
        outputs["route_certificate_json"],
        certificate,
    )
    source_metadata.update(
        {
            "route_kind": STAND_DISCOVERY_ROUTE_KIND,
            "route_purpose": STAND_DISCOVERY_ROUTE_PURPOSE,
            "source": STAND_DISCOVERY_ROUTE_SOURCE,
            "planning_frame": plan.planning_frame,
            "motion_authorized": True,
            "physical_clearance_enforced": True,
            "route_csv_sha256": route_digest,
            "route_certificate_path": str(
                outputs["route_certificate_json"].resolve()
            ),
            "route_certificate_sha256": certificate_digest,
            "coverage_plan_path": str(coverage_plan_path.resolve()),
            "source_route_sha256": file_sha256(source_route_csv),
            "source_diagnostics_sha256": file_sha256(source_diagnostics_json),
        }
    )
    sealed_payload = dict(source_payload)
    sealed_payload["metadata"] = source_metadata
    outputs["diagnostics_json"].write_text(
        json.dumps(sealed_payload, indent=2, sort_keys=True) + "\n"
    )

    leg = load_route_leg(outputs["route_csv"], 0, thinning_min_spacing_m=0.0)
    status = validate_stand_discovery_route_binding(
        outputs["diagnostics_json"],
        leg,
        coverage_plan_path=coverage_plan_path,
        diagnostics_payload=sealed_payload,
    )
    if not status.ok:
        raise ValueError("; ".join(status.failures))
    return {name: str(path) for name, path in outputs.items()}
