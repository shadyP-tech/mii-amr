"""Plan and seal an exact stored Start pose after candidate exploration.

The saved robot pose is the goal, including its yaw.  It is never replaced by
a camera standoff or a nearby free goal cell.  All frozen candidates remain
obstacles, and collision-checked shortcuts reduce unnecessary driving turns.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import shutil
from typing import Mapping

from scripts.aufgabe04.navigation.approach.candidate_preapproach_compute import (
    validate_physical_clearance,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import (
    CandidatePlanningFrame,
)
from scripts.aufgabe04.navigation.control.safety_checks import PreflightStatus
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import CoverageSurveyPlan
from scripts.aufgabe04.navigation.execution.dynamic_route_handoff import (
    validate_arena_boundary_evidence,
)
from scripts.aufgabe04.navigation.execution.exact_start_route_binding import (
    validate_exact_start_route_binding,
)
from scripts.aufgabe04.navigation.execution.execution_route_certificate import (
    ExecutionRouteCertificate, file_sha256, point_to_segment_distance_m,
    write_execution_route_certificate,
)
from scripts.aufgabe04.navigation.execution.route_context import build_route_metadata
from scripts.aufgabe04.navigation.foundation.artifacts import (
    write_diagnostics_json, write_route_csv,
)
from scripts.aufgabe04.navigation.foundation.models import (
    PlanningDiagnostics, Pose2D, Route, RoutePoint,
)
from scripts.aufgabe04.navigation.planning.certified_exact_start_route import (
    certify_and_smooth_exact_start_route,
)
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.global_planner import PlanRouteResult, plan_route
from scripts.aufgabe04.navigation.planning.exact_start_connector import (
    prepend_certified_exact_start,
)
from scripts.aufgabe04.navigation.planning.map_io import load_occupancy_grid_with_bundle
from scripts.aufgabe04.navigation.planning.route_smoothing import (
    RouteSmoothingSummary, segment_is_collision_free,
)
from scripts.aufgabe04.navigation.planning.waypoint_csv import SelectedRouteLeg, load_route_leg
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateSnapshot, candidate_snapshot_sha256, load_candidate_snapshot,
)
from scripts.aufgabe04.stations.models import Station, StationPose


ADMITTED_POSE_ROUTE_KIND = "admitted_candidate_pose"
ADMITTED_POSE_ROUTE_PURPOSE = "return_to_start"
_SOURCE = "stored_admitted_start_pose"


def _pose(value: object, name: str) -> Pose2D:
    if not isinstance(value, Mapping) or set(value) != {"x_m", "y_m", "yaw_rad"}:
        raise ValueError(f"{name} must contain x_m, y_m and yaw_rad")
    if any(not _finite_number(v) for v in value.values()):
        raise ValueError(f"{name} must be finite and numeric")
    return Pose2D(**value)


def _same_pose(actual: Pose2D, expected: Pose2D) -> bool:
    yaw_error = actual.yaw_rad - expected.yaw_rad
    return (
        math.hypot(actual.x_m - expected.x_m, actual.y_m - expected.y_m) <= 1e-9
        and abs(math.atan2(math.sin(yaw_error), math.cos(yaw_error))) <= 1e-9
    )


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _validate_target_evidence(
    evidence: object, *, snapshot: CandidateSnapshot, candidate_uid: str,
    target: Pose2D, start: Pose2D,
) -> None:
    if not isinstance(evidence, Mapping):
        raise ValueError("stored target evidence must be an object")
    expected = {
        "qr_id": "Start", "candidate_uid": candidate_uid,
        "planning_frame": snapshot.planning_frame,
        "candidate_snapshot_sha256": candidate_snapshot_sha256(snapshot),
    }
    for key, value in expected.items():
        if evidence.get(key) != value:
            raise ValueError(f"stored target evidence {key} mismatch")
    if not _same_pose(_pose(evidence.get("target_pose"), "target_pose"), target):
        raise ValueError("stored target evidence pose differs from route goal")
    sources = evidence.get("source_artifacts")
    if not isinstance(sources, list) or not sources:
        raise ValueError("stored target requires source artifact hashes")
    for source in sources:
        if not isinstance(source, Mapping) or not isinstance(source.get("path"), str):
            raise ValueError("invalid stored target source artifact")
        if file_sha256(Path(source["path"])) != source.get("sha256"):
            raise ValueError("stored target source artifact hash mismatch")
    stored = _pose(evidence.get("stored_pose"), "stored_pose")
    old = CandidatePlanningFrame.from_evidence(evidence.get("source_planning_frame"))
    new = CandidatePlanningFrame.from_evidence(evidence.get("planning_frame_admission"))
    if not _same_pose(new.current_pose, start):
        raise ValueError("stored target route start differs from admitted planning frame")
    if (
        old.map_frame != new.map_frame
        or old.odom_frame != new.odom_frame
        or new.map_frame != snapshot.planning_frame
    ):
        raise ValueError("stored target projection frame mismatch")
    a, b = old.map_from_odom, new.map_from_odom
    dx, dy = stored.x_m - a.x_m, stored.y_m - a.y_m
    odom_x = math.cos(a.yaw_rad) * dx + math.sin(a.yaw_rad) * dy
    odom_y = -math.sin(a.yaw_rad) * dx + math.cos(a.yaw_rad) * dy
    projected = Pose2D(
        b.x_m + math.cos(b.yaw_rad) * odom_x - math.sin(b.yaw_rad) * odom_y,
        b.y_m + math.sin(b.yaw_rad) * odom_x + math.cos(b.yaw_rad) * odom_y,
        stored.yaw_rad - a.yaw_rad + b.yaw_rad,
    )
    if not _same_pose(projected, target):
        raise ValueError("stored target reprojection differs from route goal")


def _measured_center(evidence: Mapping[str, object]) -> tuple[Pose2D, float] | None:
    value = evidence.get("measured_target_center")
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"x_m", "y_m", "uncertainty_m"}:
        raise ValueError("measured target center requires x_m, y_m and uncertainty_m")
    if (
        any(not _finite_number(v) for v in value.values())
        or value["uncertainty_m"] < 0.
    ):
        raise ValueError(
            "measured target center must be finite with non-negative uncertainty"
        )
    return Pose2D(value["x_m"], value["y_m"]), value["uncertainty_m"]


def _validate_candidate_clearance(
    poses: tuple[Pose2D, ...], *, snapshot: CandidateSnapshot,
    candidate_uid: str, transit_radius: float, active_standoff: float,
    target_evidence: Mapping[str, object], collision_standoff: float,
) -> None:
    if (
        not _finite_number(collision_standoff) or collision_standoff < 0.
        or collision_standoff > active_standoff
    ):
        raise ValueError("invalid physical collision standoff")
    for candidate in snapshot.candidates:
        center = Pose2D(candidate.geometry.x_m, candidate.geometry.y_m)
        required = max(transit_radius, candidate.geometry.keepout_radius_m)
        measured = min(
            point_to_segment_distance_m(center, a, b)
            for a, b in zip(poses, poses[1:])
        )
        if measured + 1e-9 < required:
            raise ValueError(f"admitted pose route violates {candidate.candidate_uid} keepout")
        if candidate.candidate_uid == candidate_uid:
            standoff = math.hypot(
                poses[-1].x_m - center.x_m, poses[-1].y_m - center.y_m,
            )
            if standoff + 1e-9 < active_standoff:
                raise ValueError("stored target violates active stand standoff")
    measured_center = _measured_center(target_evidence)
    if measured_center is not None:
        center, uncertainty = measured_center
        candidate = snapshot.candidate_for(candidate_uid)
        required = collision_standoff + max(
            0., uncertainty - candidate.geometry.uncertainty_m,
        )
        measured = min(
            point_to_segment_distance_m(center, a, b)
            for a, b in zip(poses, poses[1:])
        )
        if measured + 1e-9 < required:
            raise ValueError("admitted pose route violates measured target keepout")
        standoff = math.hypot(
            poses[-1].x_m - center.x_m, poses[-1].y_m - center.y_m,
        )
        if standoff + 1e-9 < active_standoff:
            raise ValueError("stored target violates measured active stand standoff")


def plan_admitted_pose_route(
    *, map_yaml: Path, semantic_map_id: str, plan: CoverageSurveyPlan,
    snapshot: CandidateSnapshot, snapshot_path: Path, candidate_uid: str,
    start: Pose2D, target: Pose2D, output_dir: Path, inflation_radius_m: float,
    physical_clearance: Mapping[str, float], target_evidence: Mapping[str, object],
    candidate_transit_radius_m: float | None = None,
) -> dict[str, str]:
    """Create one independently sealed route; grant no live motion permission."""
    _pose(asdict(start), "start")
    _pose(asdict(target), "target")
    radius = (
        float(physical_clearance["minimum_candidate_transit_radius_m"])
        if candidate_transit_radius_m is None else candidate_transit_radius_m
    )
    if (
        not _finite_number(radius) or radius < 0
        or not _finite_number(inflation_radius_m) or inflation_radius_m < 0
    ):
        raise ValueError("route clearances must be finite and non-negative")
    active, _, _ = validate_physical_clearance(
        physical_clearance, inflation_radius_m=inflation_radius_m,
        candidate_transit_radius_m=radius,
    )
    collision = physical_clearance.get("minimum_collision_standoff_m", radius)
    if not _finite_number(collision) or collision < 0 or collision > active:
        raise ValueError("invalid physical collision standoff")
    if snapshot.candidate_for(candidate_uid) is None:
        raise ValueError("stored target candidate is absent from full snapshot")
    snapshot_hash = candidate_snapshot_sha256(snapshot)
    if candidate_snapshot_sha256(load_candidate_snapshot(snapshot_path)) != snapshot_hash:
        raise ValueError("stored target snapshot file mismatch")
    if (
        snapshot.planning_frame != plan.planning_frame
        or snapshot.map_bundle_sha256 != plan.map_bundle_sha256
    ):
        raise ValueError("stored target snapshot differs from coverage plan")
    _validate_target_evidence(
        target_evidence, snapshot=snapshot, candidate_uid=candidate_uid,
        target=target, start=start,
    )
    stationary_turn = (target.x_m, target.y_m) == (start.x_m, start.y_m)
    if stationary_turn and _same_pose(start, target):
        raise ValueError("stored Start pose already reached; no travel route required")
    grid, bundle = load_occupancy_grid_with_bundle(
        map_yaml, semantic_map_id=semantic_map_id, planning_frame=plan.planning_frame,
    )
    if bundle.bundle_sha256 != snapshot.map_bundle_sha256:
        raise ValueError("stored target map differs from runtime map")
    base = Costmap.from_occupancy_grid(grid).with_arena_bounds(plan.arena_bounds)
    planning = base.with_inflation(inflation_radius_m).with_station_keepouts(tuple(
        Station(
            c.candidate_uid, StationPose(c.geometry.x_m, c.geometry.y_m, 0.),
            0., max(radius, c.geometry.keepout_radius_m),
        )
        for c in snapshot.candidates
    ))
    measured_center = _measured_center(target_evidence)
    if measured_center is not None:
        center, uncertainty = measured_center
        measured_radius = collision + max(
            0., uncertainty - snapshot.candidate_for(candidate_uid).geometry.uncertainty_m,
        )
        planning = planning.with_station_keepouts((Station(
            "measured_target", StationPose(center.x_m, center.y_m, 0.),
            0., measured_radius,
        ),))
    # A blocked exact endpoint cannot be silently moved to a nearby goal cell.
    if not segment_is_collision_free(planning, target, target):
        raise ValueError("exact stored target is blocked; goal snapping is forbidden")
    if stationary_turn:
        # A stationary heading correction uses the actual saved point twice;
        # no grid-center excursion or fabricated translation is introduced.
        cell = planning.world_to_grid(target)
        result = PlanRouteResult(
            route=Route(
                points=(RoutePoint(0, cell, start), RoutePoint(1, cell, target)),
                requested_start=start, requested_goal=target,
                snapped_start=start, snapped_goal=target, length_m=0.,
            ),
            diagnostics=PlanningDiagnostics(
                status="ok", start_cell=cell, goal_cell=cell,
                snapped_start_cell=cell, snapped_goal_cell=cell, path_cell_count=2,
            ),
        )
    else:
        result = plan_route(
            planning, start, target, snap_radius_m=plan.config.snap_radius_m,
        )
    if result.route is None:
        raise ValueError(f"stored target route is unreachable: {result.diagnostics.reason}")
    route = result.route
    anchor = route.points[-1].pose
    if not segment_is_collision_free(planning, anchor, target):
        raise ValueError("exact stored target connector is blocked")
    distance = math.hypot(target.x_m - anchor.x_m, target.y_m - anchor.y_m)
    if distance > 1e-12:
        points = (*route.points, RoutePoint(
            len(route.points), planning.world_to_grid(target), target,
            distance, route.length_m + distance,
        ))
    else:
        points = (*route.points[:-1], replace(route.points[-1], pose=target))
    route = replace(route, points=points, snapped_goal=target, length_m=route.length_m + distance)
    result = replace(
        result, route=route,
        diagnostics=replace(result.diagnostics, route_length_m=route.length_m),
    )
    if stationary_turn:
        result, connector = prepend_certified_exact_start(
            result, base_costmap=base, start=start,
            required_clearance_m=inflation_radius_m,
        )
        smoothing = RouteSmoothingSummary(
            enabled=False, input_point_count=2, output_point_count=2,
            input_length_m=0., output_length_m=0., optimized=False,
            skipped_reason="stationary_turn_preserves_heading_handoff",
        )
    else:
        result, connector, smoothing = certify_and_smooth_exact_start_route(
            result, base_costmap=base, planning_costmap=planning,
            exact_start=start, required_clearance_m=inflation_radius_m,
        )
    assert result.route is not None
    if len(result.route.points) < 2:
        raise ValueError("stored Start pose already reached; no travel route required")
    poses = tuple(p.pose for p in result.route.points)
    _validate_candidate_clearance(
        poses, snapshot=snapshot, candidate_uid=candidate_uid,
        transit_radius=radius, active_standoff=active, target_evidence=target_evidence,
        collision_standoff=collision,
    )

    evidence_json = json.dumps(
        dict(target_evidence), indent=2, sort_keys=True, allow_nan=False,
    ) + "\n"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {key: output_dir / name for key, name in (
        ("route_csv", "route.csv"), ("diagnostics_json", "route_diagnostics.json"),
        ("route_certificate_json", "route_certificate.json"),
        ("candidate_snapshot", "candidate_snapshot.json"), ("target_evidence_json", "target_evidence.json"),
    )}
    shutil.copyfile(snapshot_path, paths["candidate_snapshot"])
    paths["target_evidence_json"].write_text(evidence_json)
    write_route_csv(paths["route_csv"], (result,), final_yaw_by_leg={0: target.yaw_rad})
    with paths["route_csv"].open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows, fields = list(reader), list(reader.fieldnames or ())
    fields.extend(("protected", "corridor", "simulation_only", "route_kind", "stationary_turn"))
    for index, row in enumerate(rows):
        row.update(
            protected=str(index == len(rows) - 1).lower(),
            corridor=str(index == len(rows) - 1).lower(),
            simulation_only="false", route_kind=ADMITTED_POSE_ROUTE_KIND,
            stationary_turn=str(stationary_turn).lower(),
        )
    with paths["route_csv"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    route_hash = file_sha256(paths["route_csv"])
    certificate_hash = write_execution_route_certificate(paths["route_certificate_json"], ExecutionRouteCertificate(
        route_sha256=route_hash, planning_frame=snapshot.planning_frame, route_kind=ADMITTED_POSE_ROUTE_KIND,
        waypoint_count=len(rows), tracking_tube_radius_m=0.03, exact_vertex_pursuit=True,
        command_owner="/aufgabe04_simple_waypoint_follower", map_bundle_sha256=bundle.bundle_sha256,
        candidate_snapshot_sha256=snapshot_hash,
    ))
    metadata = build_route_metadata(map_yaml, grid, (candidate_uid,), arena_bounds=plan.arena_bounds, map_bundle=bundle)
    metadata.update({
        "source": _SOURCE, "route_kind": ADMITTED_POSE_ROUTE_KIND,
        "stationary_turn": stationary_turn,
        "route_purpose": ADMITTED_POSE_ROUTE_PURPOSE, "motion_authorized": True,
        "planning_frame": snapshot.planning_frame, "physical_clearance_enforced": True,
        "physical_clearance": dict(physical_clearance), "inflation_radius_m": inflation_radius_m,
        "candidate_transit_radius_m": radius, "candidate_snapshot_json": str(paths["candidate_snapshot"]),
        "candidate_snapshot_sha256": snapshot_hash, "selected_candidate_stand_id": candidate_uid,
        "target_evidence_json": str(paths["target_evidence_json"]),
        "target_evidence_sha256": file_sha256(paths["target_evidence_json"]),
        "selected_approach_pose": asdict(target), "exact_start_connector": connector.to_metadata(),
        "route_start_pose_provenance": {"source": _SOURCE, "planning_frame": snapshot.planning_frame, "pose": asdict(start)},
        "line_of_sight_route_optimization": {"enabled": smoothing.enabled, "legs": [smoothing.to_metadata()]},
        "route_csv_sha256": route_hash, "route_certificate_path": str(paths["route_certificate_json"]),
        "route_certificate_sha256": certificate_hash,
    })
    write_diagnostics_json(paths["diagnostics_json"], (result,), metadata)
    status = validate_admitted_pose_route_binding(
        paths["diagnostics_json"],
        load_route_leg(paths["route_csv"], 0, thinning_min_spacing_m=0.),
        candidate_snapshot_path=paths["candidate_snapshot"],
    )
    if not status.ok:
        raise ValueError("; ".join(status.failures))
    return {key: str(path) for key, path in paths.items()}


def validate_admitted_pose_route_binding(
    diagnostics_path: Path, leg: SelectedRouteLeg, *, candidate_snapshot_path: Path | None,
    diagnostics_payload: Mapping[str, object] | None = None,
) -> PreflightStatus:
    """Fail closed when route, stored target, provenance or pool has changed."""
    try:
        payload = (
            json.loads(Path(diagnostics_path).read_text())
            if diagnostics_payload is None else diagnostics_payload
        )
        metadata = payload["metadata"]
        if not isinstance(metadata, Mapping):
            raise ValueError("missing admitted pose route metadata")
        for key, expected in (
            ("route_kind", ADMITTED_POSE_ROUTE_KIND),
            ("route_purpose", ADMITTED_POSE_ROUTE_PURPOSE), ("source", _SOURCE),
            ("motion_authorized", True), ("physical_clearance_enforced", True),
            ("route_csv_sha256", leg.source_sha256),
        ):
            if metadata.get(key) != expected:
                raise ValueError(f"admitted pose route {key} mismatch")
        if (
            leg.route_kind != ADMITTED_POSE_ROUTE_KIND or leg.simulation_only
            or len(leg.raw_waypoints) < 2
        ):
            raise ValueError("invalid physical admitted pose route")
        if candidate_snapshot_path is None:
            raise ValueError("admitted pose route requires --candidate-snapshot")
        snapshot = load_candidate_snapshot(candidate_snapshot_path)
        if (
            metadata.get("candidate_snapshot_sha256") != candidate_snapshot_sha256(snapshot)
            or metadata.get("map_bundle_sha256") != snapshot.map_bundle_sha256
            or metadata.get("planning_frame") != snapshot.planning_frame
        ):
            raise ValueError("admitted pose route snapshot binding mismatch")
        uid = metadata.get("selected_candidate_stand_id")
        if snapshot.candidate_for(uid) is None:
            raise ValueError("admitted pose candidate missing from snapshot")
        evidence_path = Path(metadata["target_evidence_json"])
        if file_sha256(evidence_path) != metadata.get("target_evidence_sha256"):
            raise ValueError("admitted target evidence hash mismatch")
        target = _pose(metadata.get("selected_approach_pose"), "selected_approach_pose")
        evidence = json.loads(evidence_path.read_text())
        start = _pose(metadata["exact_start_connector"]["exact_start"], "exact_start")
        _validate_target_evidence(
            evidence, snapshot=snapshot, candidate_uid=uid, target=target, start=start,
        )
        final = leg.raw_waypoints[-1]
        if not final.protected or not final.corridor or not _same_pose(final.pose, target):
            raise ValueError("route endpoint differs from exact stored admitted pose")
        if any(math.isfinite(w.pose.yaw_rad) for w in leg.raw_waypoints[:-1]):
            raise ValueError("admitted pose transit yaw must remain unconstrained")
        validate_arena_boundary_evidence(metadata)
        poses = tuple(w.pose for w in leg.raw_waypoints)
        stationary = metadata.get("stationary_turn")
        if not isinstance(stationary, bool):
            raise ValueError("admitted pose stationary-turn flag must be boolean")
        if getattr(leg, "stationary_turn", False) != stationary:
            raise ValueError("admitted pose stationary-turn CSV flag mismatch")
        zero_translation = all(
            (p.x_m, p.y_m) == (start.x_m, start.y_m) for p in poses
        )
        if stationary:
            if len(poses) != 2 or not zero_translation or leg.route_length_m != 0. or _same_pose(start, target):
                raise ValueError("stationary admitted turn has invalid geometry or heading")
        elif zero_translation or leg.route_length_m <= 0.:
            raise ValueError("admitted travel route must have positive translation")
        validate_exact_start_route_binding(metadata, tuple((p.x_m, p.y_m) for p in poses))
        active, _, _ = validate_physical_clearance(
            metadata["physical_clearance"],
            inflation_radius_m=metadata["inflation_radius_m"],
            candidate_transit_radius_m=metadata["candidate_transit_radius_m"],
        )
        _validate_candidate_clearance(
            poses, snapshot=snapshot, candidate_uid=uid,
            transit_radius=metadata["candidate_transit_radius_m"],
            active_standoff=active, target_evidence=evidence,
            collision_standoff=metadata["physical_clearance"].get(
                "minimum_collision_standoff_m", metadata["candidate_transit_radius_m"],
            ),
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return PreflightStatus(ok=False, failures=[f"admitted pose binding is invalid: {exc}"])
    return PreflightStatus(ok=True, failures=[])
