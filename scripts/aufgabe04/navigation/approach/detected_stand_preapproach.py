"""Seal and validate one LiDAR-detected stand pre-approach route."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.navigation.approach.backside_axis_frame_projection import (
    BacksideAxisFrameProjection,
    load_backside_axis_planning_observation,
)
from scripts.aufgabe04.navigation.execution.dynamic_route_handoff import (
    validate_arena_boundary_evidence,
)
from scripts.aufgabe04.navigation.execution.execution_route_certificate import (
    ExecutionRouteCertificate,
    file_sha256,
    point_to_segment_distance_m,
    write_execution_route_certificate,
)
from scripts.aufgabe04.navigation.execution.exact_start_route_binding import (
    validate_exact_start_route_binding,
)
from scripts.aufgabe04.navigation.control.safety_checks import PreflightStatus
from scripts.aufgabe04.navigation.planning.waypoint_csv import SelectedRouteLeg, load_route_leg
from scripts.aufgabe04.stations.candidate_snapshot import (
    candidate_snapshot_sha256,
    load_candidate_snapshot,
)


DETECTED_STAND_PREAPPROACH_ROUTE_KIND = "detected_stand_preapproach"
DETECTED_STAND_PREAPPROACH_ROUTE_PURPOSE = "pre_approach"
DEFAULT_TRACKING_TUBE_RADIUS_M = 0.03
DEFAULT_COMMAND_OWNER = "/aufgabe04_simple_waypoint_follower"
ROBOT_TO_STAND_BEARING_MODE = "robot-to-stand"
CAMERA_AXIS_FACE_BEARING_MODE = "camera-axis-face"


def _angle_error(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


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


def _minimum_route_clearance_m(leg: SelectedRouteLeg, x_m: float, y_m: float) -> float:
    from scripts.aufgabe04.navigation.foundation.models import Pose2D

    point = Pose2D(x_m, y_m, 0.0)
    poses = tuple(waypoint.pose for waypoint in leg.raw_waypoints)
    return min(
        point_to_segment_distance_m(point, start, end)
        for start, end in zip(poses, poses[1:])
    )


def _validate_route_start_pose_provenance(
    metadata: Mapping[str, object],
) -> None:
    provenance = metadata.get("route_start_pose_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("route start pose provenance is missing")
    source = provenance.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError("route start pose provenance source is missing")
    if provenance.get("planning_frame") != metadata.get("planning_frame"):
        raise ValueError("route start pose provenance has the wrong planning frame")
    pose = provenance.get("pose")
    connector = metadata.get("exact_start_connector")
    if not isinstance(pose, Mapping) or not isinstance(connector, Mapping):
        raise ValueError("route start pose provenance is incomplete")
    exact_start = connector.get("exact_start")
    if not isinstance(exact_start, Mapping):
        raise ValueError("exact-start connector start pose is missing")
    for field in ("x_m", "y_m", "yaw_rad"):
        recorded = _finite_number(pose.get(field), f"route start pose {field}")
        certified = _finite_number(
            exact_start.get(field),
            f"exact-start connector {field}",
        )
        if not math.isclose(recorded, certified, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(
                f"route start pose provenance {field} differs from connector"
            )


def validate_detected_stand_preapproach_binding(
    diagnostics_path: Path,
    leg: SelectedRouteLeg,
    *,
    candidate_snapshot_path: Path | None,
    diagnostics_payload: Mapping[str, object] | None = None,
    terminal_yaw_tolerance_rad: float = 0.15,
) -> PreflightStatus:
    """Validate the immutable detector, geometry, route, and clearance binding."""

    failures: list[str] = []
    if leg.route_kind != DETECTED_STAND_PREAPPROACH_ROUTE_KIND:
        return PreflightStatus(
            ok=False,
            failures=["detected stand binding requires detected_stand_preapproach"],
        )
    if leg.simulation_only:
        failures.append("detected stand pre-approach must not be simulation_only")
    if not leg.raw_waypoints:
        failures.append("detected stand pre-approach route is empty")
        return PreflightStatus(ok=False, failures=failures)
    final = leg.raw_waypoints[-1]
    if not final.protected or not final.corridor:
        failures.append("detected stand final waypoint must be protected and corridor")
    if not math.isfinite(final.pose.yaw_rad):
        failures.append("detected stand final waypoint yaw is unconstrained")

    try:
        payload = (
            _load_json(diagnostics_path)
            if diagnostics_payload is None
            else diagnostics_payload
        )
        metadata = _metadata(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        failures.append(f"invalid detected stand diagnostics: {exc}")
        return PreflightStatus(ok=False, failures=failures)

    if metadata.get("route_kind") != DETECTED_STAND_PREAPPROACH_ROUTE_KIND:
        failures.append("detected stand route kind does not match diagnostics")
    if metadata.get("route_purpose") != DETECTED_STAND_PREAPPROACH_ROUTE_PURPOSE:
        failures.append("detected stand diagnostics route_purpose must be pre_approach")
    if metadata.get("source") != "lidar_detected_stand_exploration":
        failures.append("detected stand diagnostics have an unexpected source")
    if metadata.get("plan_mode") != "next-candidate":
        failures.append("detected stand pre-approach requires next-candidate planning")
    bearing_mode = metadata.get("approach_bearing_mode")
    if bearing_mode not in {
        ROBOT_TO_STAND_BEARING_MODE,
        CAMERA_AXIS_FACE_BEARING_MODE,
    }:
        failures.append("detected stand approach bearing mode is unsupported")
    if metadata.get("physical_clearance_enforced") is not True:
        failures.append("detected stand physical clearance was not enforced")
    if metadata.get("route_csv_sha256") != leg.source_sha256:
        failures.append("detected stand route CSV SHA-256 does not match diagnostics")
    try:
        validate_exact_start_route_binding(
            metadata,
            tuple(
                (waypoint.pose.x_m, waypoint.pose.y_m)
                for waypoint in leg.raw_waypoints
            ),
        )
        _validate_route_start_pose_provenance(metadata)
    except ValueError as exc:
        failures.append(f"exact-start route binding is invalid: {exc}")

    if candidate_snapshot_path is None:
        failures.append("detected stand route requires --candidate-snapshot")
        return PreflightStatus(ok=False, failures=failures)
    try:
        snapshot = load_candidate_snapshot(candidate_snapshot_path)
    except (OSError, ValueError) as exc:
        failures.append(f"candidate snapshot validation failed: {exc}")
        return PreflightStatus(ok=False, failures=failures)
    snapshot_digest = candidate_snapshot_sha256(snapshot)
    if metadata.get("candidate_snapshot_sha256") != snapshot_digest:
        failures.append("candidate snapshot SHA-256 does not match diagnostics")
    if metadata.get("planning_frame") != snapshot.planning_frame:
        failures.append("candidate snapshot planning frame does not match diagnostics")
    if metadata.get("map_bundle_sha256") != snapshot.map_bundle_sha256:
        failures.append("candidate snapshot map bundle does not match diagnostics")

    selected_uid = metadata.get("selected_candidate_stand_id")
    if not isinstance(selected_uid, str) or not selected_uid:
        failures.append("selected candidate UID is missing")
        return PreflightStatus(ok=False, failures=failures)
    selected = snapshot.candidate_for(selected_uid)
    if selected is None:
        failures.append("selected candidate UID is absent from candidate snapshot")
        return PreflightStatus(ok=False, failures=failures)

    if bearing_mode == CAMERA_AXIS_FACE_BEARING_MODE:
        axis_path_value = metadata.get("axis_observation_json")
        if not isinstance(axis_path_value, str) or not axis_path_value:
            failures.append("camera-axis approach is missing axis observation")
        else:
            axis_path = Path(axis_path_value)
            try:
                axis_observation = load_backside_axis_planning_observation(
                    axis_path
                )
                axis_digest = file_sha256(axis_path)
                axis_rad = axis_observation.stand_axis_rad
                face_normal = _finite_number(
                    metadata.get("selected_face_normal_rad"),
                    "selected_face_normal_rad",
                )
                approach_offset = _finite_number(
                    metadata.get("approach_offset_m"), "approach_offset_m"
                )
                observed_normal = axis_observation.opposite_face_normal_rad
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                failures.append(f"axis observation validation failed: {exc}")
            else:
                if metadata.get("axis_observation_sha256") != axis_digest:
                    failures.append("axis observation SHA-256 does not match")
                if isinstance(
                    axis_observation, BacksideAxisFrameProjection
                ):
                    if metadata.get("axis_evidence_kind") != (
                        "backside_axis_frame_projection"
                    ):
                        failures.append(
                            "axis evidence kind does not identify its projection"
                        )
                    if metadata.get("source_axis_observation_json") != str(
                        axis_observation.source_axis_observation_path
                    ):
                        failures.append(
                            "source axis observation path does not match projection"
                        )
                    if metadata.get("source_axis_observation_sha256") != (
                        axis_observation.source_axis_observation_sha256
                    ):
                        failures.append(
                            "source axis observation SHA-256 does not match projection"
                        )
                    if metadata.get("axis_frame_projection_sha256") != (
                        axis_observation.projection_sha256
                    ):
                        failures.append(
                            "axis frame projection SHA-256 does not match"
                        )
                elif metadata.get("axis_evidence_kind") != (
                    "native_backside_axis_observation"
                ):
                    failures.append("axis evidence kind is invalid")
                if axis_observation.stand_id != selected_uid:
                    failures.append("axis observation stand ID does not match")
                if (
                    axis_observation.planning_frame
                    != snapshot.planning_frame
                ):
                    failures.append("axis observation planning frame does not match")
                if (
                    math.hypot(
                        axis_observation.stand_x_m - selected.geometry.x_m,
                        axis_observation.stand_y_m - selected.geometry.y_m,
                    )
                    > 1.0e-6
                ):
                    failures.append("axis observation stand center does not match")
                if _angle_error(face_normal, observed_normal) > 1.0e-9:
                    failures.append(
                        "selected face normal does not match axis observation"
                    )
                perpendicular_error = abs(
                    abs(
                        math.atan2(
                            math.sin(face_normal - axis_rad),
                            math.cos(face_normal - axis_rad),
                        )
                    )
                    - math.pi / 2.0
                )
                if perpendicular_error > 0.15:
                    failures.append("selected face normal is not perpendicular to axis")
                initial_side_angle = math.atan2(
                    axis_observation.robot_y_m - selected.geometry.y_m,
                    axis_observation.robot_x_m - selected.geometry.x_m,
                )
                if math.cos(face_normal - initial_side_angle) > -0.5:
                    failures.append(
                        "camera-axis approach does not inspect the opposite face"
                    )
                expected_x = (
                    selected.geometry.x_m
                    + approach_offset * math.cos(face_normal)
                )
                expected_y = (
                    selected.geometry.y_m
                    + approach_offset * math.sin(face_normal)
                )
                if math.hypot(
                    final.pose.x_m - expected_x,
                    final.pose.y_m - expected_y,
                ) > 0.06:
                    failures.append(
                        "camera-axis terminal position does not match selected face"
                    )

    clearance = metadata.get("physical_clearance")
    if not isinstance(clearance, Mapping):
        failures.append("physical_clearance metadata is missing")
        return PreflightStatus(ok=False, failures=failures)
    try:
        minimum_active = _finite_number(
            clearance.get("minimum_active_standoff_m"),
            "minimum_active_standoff_m",
        )
        minimum_transit = _finite_number(
            clearance.get("minimum_candidate_transit_radius_m"),
            "minimum_candidate_transit_radius_m",
        )
        minimum_inflation = _finite_number(
            clearance.get("minimum_static_inflation_m"),
            "minimum_static_inflation_m",
        )
        actual_inflation = _finite_number(
            metadata.get("inflation_radius_m"), "inflation_radius_m"
        )
        configured_transit = _finite_number(
            metadata.get("candidate_transit_radius_m"),
            "candidate_transit_radius_m",
        )
    except ValueError as exc:
        failures.append(str(exc))
        return PreflightStatus(ok=False, failures=failures)
    if actual_inflation + 1.0e-9 < minimum_inflation:
        failures.append("static inflation is below the recorded physical minimum")
    if configured_transit + 1.0e-9 < minimum_transit:
        failures.append("candidate transit radius is below the recorded minimum")

    target_distance = math.hypot(
        final.pose.x_m - selected.geometry.x_m,
        final.pose.y_m - selected.geometry.y_m,
    )
    if target_distance + 1.0e-9 < minimum_active:
        failures.append("terminal pose violates the selected stand LiDAR standoff")
    expected_yaw = math.atan2(
        selected.geometry.y_m - final.pose.y_m,
        selected.geometry.x_m - final.pose.x_m,
    )
    if (
        math.isfinite(final.pose.yaw_rad)
        and _angle_error(final.pose.yaw_rad, expected_yaw)
        > terminal_yaw_tolerance_rad
    ):
        failures.append("terminal yaw does not face the selected stand")

    for candidate in snapshot.candidates:
        if candidate.candidate_uid == selected_uid:
            continue
        required = max(candidate.geometry.keepout_radius_m, minimum_transit)
        measured = _minimum_route_clearance_m(
            leg, candidate.geometry.x_m, candidate.geometry.y_m
        )
        if measured + 1.0e-9 < required:
            failures.append(
                f"route clearance to {candidate.candidate_uid} is "
                f"{measured:.3f} m, below {required:.3f} m"
            )
    try:
        validate_arena_boundary_evidence(metadata)
    except ValueError as exc:
        failures.append(f"arena boundary evidence is invalid: {exc}")
    return PreflightStatus(ok=not failures, failures=failures)


def seal_detected_stand_preapproach(
    *,
    pipeline_root: Path,
    output_dir: Path | None = None,
    command_owner: str = DEFAULT_COMMAND_OWNER,
    tracking_tube_radius_m: float = DEFAULT_TRACKING_TUBE_RADIUS_M,
) -> dict[str, str]:
    """Create a typed route, certificate, and diagnostics without overwriting input."""

    pipeline_root = Path(pipeline_root)
    output_dir = (
        pipeline_root / "preapproach_execution"
        if output_dir is None
        else Path(output_dir)
    )
    source_route = pipeline_root / "route.csv"
    source_diagnostics = pipeline_root / "route_diagnostics.json"
    candidate_snapshot_path = pipeline_root / "candidate_snapshot.json"
    summary_path = pipeline_root / "pipeline_summary.json"
    outputs = {
        "route_csv": output_dir / "route.csv",
        "diagnostics_json": output_dir / "route_diagnostics.json",
        "route_certificate_json": output_dir / "route_certificate.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise ValueError("refusing to overwrite sealed artifacts: " + ", ".join(existing))
    summary = _load_json(summary_path)
    if summary.get("status") != "observe_and_plan_complete":
        raise ValueError("pipeline summary is not complete")
    if summary.get("motion_published") is not False:
        raise ValueError("pipeline summary does not prove observe-and-plan-only execution")
    source_payload = _load_json(source_diagnostics)
    source_metadata = dict(_metadata(source_payload))
    if source_metadata.get("approach_bearing_mode") not in {
        ROBOT_TO_STAND_BEARING_MODE,
        CAMERA_AXIS_FACE_BEARING_MODE,
    }:
        raise ValueError("source route has an unsupported approach bearing")
    if source_metadata.get("physical_clearance_enforced") is not True:
        raise ValueError("source route must enforce physical clearance")

    with source_route.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("source route CSV is missing a header")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    if len(rows) < 2:
        raise ValueError("source route has fewer than two waypoints")
    route_xy = []
    for index, row in enumerate(rows):
        try:
            point = (float(row["world_x_m"]), float(row["world_y_m"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"source route waypoint {index} coordinates must be numeric"
            ) from exc
        route_xy.append(point)
    validate_exact_start_route_binding(source_metadata, tuple(route_xy))
    _validate_route_start_pose_provenance(source_metadata)
    for field in ("protected", "corridor", "simulation_only", "route_kind"):
        if field not in fieldnames:
            fieldnames.append(field)
    for index, row in enumerate(rows):
        row["protected"] = "true" if index == len(rows) - 1 else "false"
        row["corridor"] = "true" if index == len(rows) - 1 else "false"
        row["simulation_only"] = "false"
        row["route_kind"] = DETECTED_STAND_PREAPPROACH_ROUTE_KIND

    output_dir.mkdir(parents=True, exist_ok=True)
    with outputs["route_csv"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    route_digest = file_sha256(outputs["route_csv"])
    snapshot = load_candidate_snapshot(candidate_snapshot_path)
    snapshot_digest = candidate_snapshot_sha256(snapshot)
    certificate = ExecutionRouteCertificate(
        route_sha256=route_digest,
        planning_frame=snapshot.planning_frame,
        route_kind=DETECTED_STAND_PREAPPROACH_ROUTE_KIND,
        waypoint_count=len(rows),
        tracking_tube_radius_m=tracking_tube_radius_m,
        exact_vertex_pursuit=True,
        command_owner=command_owner,
        map_bundle_sha256=snapshot.map_bundle_sha256,
        candidate_snapshot_sha256=snapshot_digest,
    )
    certificate_digest = write_execution_route_certificate(
        outputs["route_certificate_json"], certificate
    )
    source_metadata.update(
        {
            "route_kind": DETECTED_STAND_PREAPPROACH_ROUTE_KIND,
            "route_purpose": DETECTED_STAND_PREAPPROACH_ROUTE_PURPOSE,
            "planning_frame": snapshot.planning_frame,
            "route_csv_sha256": route_digest,
            "route_certificate_path": str(outputs["route_certificate_json"].resolve()),
            "route_certificate_sha256": certificate_digest,
            "candidate_snapshot_json": str(candidate_snapshot_path.resolve()),
            "candidate_snapshot_sha256": snapshot_digest,
            "source_route_sha256": file_sha256(source_route),
            "source_diagnostics_sha256": file_sha256(source_diagnostics),
            "source_pipeline_summary_sha256": file_sha256(summary_path),
        }
    )
    sealed_payload = dict(source_payload)
    sealed_payload["metadata"] = source_metadata
    outputs["diagnostics_json"].write_text(
        json.dumps(sealed_payload, indent=2, sort_keys=True) + "\n"
    )

    leg = load_route_leg(outputs["route_csv"], 0, thinning_min_spacing_m=0.0)
    status = validate_detected_stand_preapproach_binding(
        outputs["diagnostics_json"],
        leg,
        candidate_snapshot_path=candidate_snapshot_path,
        diagnostics_payload=sealed_payload,
    )
    if not status.ok:
        raise ValueError("; ".join(status.failures))
    return {name: str(path) for name, path in outputs.items()}
