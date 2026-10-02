"""Plan and seal an exact stored Start pose after candidate exploration.

The saved robot pose is the goal, including its yaw.  It is never replaced by
a camera standoff or a nearby free goal cell.  All frozen candidates remain
obstacles, and collision-checked shortcuts reduce unnecessary driving turns.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Mapping

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.approach.admitted_return_uncertainty import (
    MINIMUM_REMAINING_ROUTE_M, MINIMUM_STAGE_DISPLACEMENT_M,
    build_return_prefix, evaluate_admitted_return_stage_uncertainty,
    executable_return_poses, return_route_geometry, select_admitted_return_prefix,
)
from scripts.aufgabe04.navigation.approach.stored_pose_route_alternatives import (
    AlternativeGeometryRejected, RouteAlternativesExhausted, StoredPoseRouteGeometry,
    select_stored_pose_route_alternative, validate_alternative_evidence,
)
from scripts.aufgabe04.navigation.approach.candidate_route_uncertainty_selection import (
    CandidateRouteUncertaintyContext,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_compute import (
    validate_physical_clearance,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import (
    CandidatePlanningFrame,
)
from scripts.aufgabe04.navigation.approach.exact_stored_goal_connector import (
    candidate_goal_anchors, certify_stored_goal_connector,
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
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
    MAX_RETURN_TO_START_LEGS, validate_stored_pose_tour_target_evidence,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_admission import RouteUncertaintyAdmissionConfig
from scripts.aufgabe04.navigation.execution.tour_replan_binding import validate_tour_navigation
from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import PlanarCovariance
from scripts.aufgabe04.navigation.foundation.artifacts import (
    write_diagnostics_json, write_route_csv,
)
from scripts.aufgabe04.navigation.foundation.models import (
    PlanningDiagnostics, Pose2D, Route, RoutePoint,
)
from scripts.aufgabe04.navigation.planning.certified_exact_start_route import (
    certify_and_smooth_exact_start_route,
)
from scripts.aufgabe04.navigation.planning.costmap import CELL_SOURCE_INFLATED, Costmap
from scripts.aufgabe04.navigation.planning.temporary_obstacle_overlay import apply_bound_temporary_obstacles
from scripts.aufgabe04.navigation.planning.global_planner import PlanRouteResult, plan_route
from scripts.aufgabe04.navigation.planning.exact_start_connector import (
    prepend_certified_exact_start, _segment_clearance_evidence,
)
from scripts.aufgabe04.navigation.planning.map_io import load_occupancy_grid_with_bundle
from scripts.aufgabe04.navigation.planning.route_smoothing import (
    RouteSmoothingSummary, segment_is_collision_free, supercover_segment_cells,
)
from scripts.aufgabe04.navigation.planning.waypoint_csv import SelectedRouteLeg, load_route_leg
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateSnapshot, candidate_snapshot_sha256, load_candidate_snapshot,
)
from scripts.aufgabe04.stations.models import Station, StationPose


ADMITTED_POSE_ROUTE_KIND = "admitted_candidate_pose"
ADMITTED_POSE_ROUTE_PURPOSE = "return_to_start"
STORED_POSE_TOUR_ROUTE_PURPOSE = "stored_pose_tour"
ADMITTED_POSE_ROUTE_PURPOSES = (ADMITTED_POSE_ROUTE_PURPOSE, STORED_POSE_TOUR_ROUTE_PURPOSE)
_SOURCE = "stored_admitted_start_pose"
_TOUR_SOURCE = "stored_admitted_pose_tour"


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
    target: Pose2D, start: Pose2D, purpose: str = ADMITTED_POSE_ROUTE_PURPOSE,
) -> None:
    if not isinstance(evidence, Mapping):
        raise ValueError("stored target evidence must be an object")
    expected = {
        "candidate_uid": candidate_uid,
        "planning_frame": snapshot.planning_frame,
        "candidate_snapshot_sha256": candidate_snapshot_sha256(snapshot),
    }
    if purpose == ADMITTED_POSE_ROUTE_PURPOSE:
        expected["qr_id"] = "Start"
    elif purpose == STORED_POSE_TOUR_ROUTE_PURPOSE:
        validate_stored_pose_tour_target_evidence(evidence)
    else:
        raise ValueError("unknown admitted pose route purpose")
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


def _stored_pose_costmaps(base, *, snapshot, radius, inflation, collision, candidate_uid, evidence):
    """Keep static obstacles separate from rasterized candidate provenance."""
    static = base.with_inflation(inflation)
    planning = static.with_station_keepouts(tuple(
        Station(c.candidate_uid, StationPose(c.geometry.x_m, c.geometry.y_m, 0.),
                0., max(radius, c.geometry.keepout_radius_m))
        for c in snapshot.candidates
    ))
    measured = _measured_center(evidence)
    if measured is not None:
        center, uncertainty = measured
        measured_radius = collision + max(
            0., uncertainty - snapshot.candidate_for(candidate_uid).geometry.uncertainty_m,
        )
        planning = planning.with_station_keepouts((Station(
            "measured_target", StationPose(center.x_m, center.y_m, 0.), 0., measured_radius,
        ),))
    return static, planning


def _append_exact_target(result, *, planning, target):
    route = result.route
    anchor = route.points[-1].pose
    distance = math.hypot(target.x_m - anchor.x_m, target.y_m - anchor.y_m)
    if distance > 1e-12:
        points = (*route.points, RoutePoint(
            len(route.points), planning.world_to_grid(target), target,
            distance, route.length_m + distance,
        ))
    else:
        points = (*route.points[:-1], replace(route.points[-1], pose=target))
    route = replace(route, points=points, requested_goal=target, snapped_goal=target,
                    length_m=route.length_m + distance)
    return replace(result, route=route, diagnostics=replace(
        result.diagnostics, route_length_m=route.length_m, path_cell_count=len(points),
        goal_cell=planning.world_to_grid(target), snapped_goal_cell=planning.world_to_grid(target),
    ))


def _plan_certified_stored_goal(*, base, static, planning, start, target, inflation,
                              snap_radius, validate_clearance):
    """Keep A* and smoothing outside keepouts; append one certified edge last."""
    if not segment_is_collision_free(static, target, target):
        raise ValueError("exact stored target is blocked by static inflation")
    try:
        validate_clearance((target, target))
    except ValueError as exc:
        raise ValueError(f"exact stored target is blocked by continuous clearance: {exc}") from exc
    for anchor in candidate_goal_anchors(planning, target):
        try:
            proof = certify_stored_goal_connector(
                base_costmap=base, static_costmap=static, planning_costmap=planning,
                anchor=anchor, target=target, inflation_radius_m=inflation,
                validate_candidate_clearance=validate_clearance,
            )
            prefix = plan_route(planning, start, anchor, snap_radius_m=snap_radius)
            if prefix.route is None:
                continue
            prefix, connector, smoothing = certify_and_smooth_exact_start_route(
                prefix, base_costmap=base, planning_costmap=planning,
                exact_start=start, required_clearance_m=inflation,
            )
            # No smoother may replace the proof's anchor or cross its final edge.
            if not _same_pose(prefix.route.points[-1].pose, anchor):
                continue
            result = _append_exact_target(prefix, planning=planning, target=target)
            validate_clearance(tuple(p.pose for p in result.route.points))
        except ValueError:
            continue
        return result, connector, smoothing, proof
    raise ValueError("exact stored target is blocked; no continuously certified terminal connector")


def _plan_admitted_pose_geometry(*, base, snapshot, radius, inflation, collision,
                                  candidate_uid, evidence, start, target, plan, active,
                                  purpose, stationary_turn):
    """Pure geometric candidate after map, source and frame authentication."""
    static, planning = _stored_pose_costmaps(
        base, snapshot=snapshot, radius=radius, inflation=inflation,
        collision=collision, candidate_uid=candidate_uid, evidence=evidence,
    )

    def validate_clearance(checked_poses):
        _validate_candidate_clearance(
            checked_poses, snapshot=snapshot, candidate_uid=candidate_uid,
            transit_radius=radius, active_standoff=active, target_evidence=evidence,
            collision_standoff=collision,
        )

    goal_connector = None
    target_blocked = not segment_is_collision_free(planning, target, target)
    if target_blocked:
        # Keep every planning cell intact. Only automatic Start return may
        # append a bounded, separately certified metric terminal segment.
        if stationary_turn or purpose != ADMITTED_POSE_ROUTE_PURPOSE:
            raise ValueError("exact stored target is blocked; goal snapping is forbidden")
        result, connector, smoothing, goal_connector = _plan_certified_stored_goal(
            base=base, static=static, planning=planning, start=start, target=target,
            inflation=inflation, snap_radius=plan.config.snap_radius_m,
            validate_clearance=validate_clearance,
        )
        goal_connector.update(map_bundle_sha256=snapshot.map_bundle_sha256,
                              candidate_snapshot_sha256=candidate_snapshot_sha256(snapshot))
    elif stationary_turn:
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
    if goal_connector is None:
        anchor = result.route.points[-1].pose
        if not segment_is_collision_free(planning, anchor, target):
            raise ValueError("exact stored target connector is blocked")
        result = _append_exact_target(result, planning=planning, target=target)
    if stationary_turn:
        result, connector = prepend_certified_exact_start(
            result, base_costmap=base, start=start,
            required_clearance_m=inflation,
        )
        smoothing = RouteSmoothingSummary(
            enabled=False, input_point_count=2, output_point_count=2,
            input_length_m=0., output_length_m=0., optimized=False,
            skipped_reason="stationary_turn_preserves_heading_handoff",
        )
    elif goal_connector is None:
        result, connector, smoothing = certify_and_smooth_exact_start_route(
            result, base_costmap=base, planning_costmap=planning,
            exact_start=start, required_clearance_m=inflation,
        )
    assert result.route is not None
    if len(result.route.points) < 2:
        raise ValueError("stored Start pose already reached; no travel route required")
    poses = tuple(p.pose for p in result.route.points)
    validate_clearance(poses)
    full_poses = (start, *poses[1:-1], target)
    return StoredPoseRouteGeometry(result, planning, connector, smoothing, full_poses), goal_connector



def plan_admitted_pose_route(
    *, map_yaml: Path, semantic_map_id: str, plan: CoverageSurveyPlan,
    snapshot: CandidateSnapshot, snapshot_path: Path, candidate_uid: str,
    start: Pose2D, target: Pose2D, output_dir: Path, inflation_radius_m: float,
    physical_clearance: Mapping[str, float], target_evidence: Mapping[str, object],
    candidate_transit_radius_m: float | None = None,
    route_uncertainty_context: CandidateRouteUncertaintyContext | None = None,
    return_stage_index: int = 0,
    purpose: str = ADMITTED_POSE_ROUTE_PURPOSE,
    temporary_obstacle_overlay_path: Path | None = None,
) -> dict[str, object]:
    """Create one independently sealed route; grant no live motion permission."""
    _pose(asdict(start), "start")
    _pose(asdict(target), "target")
    if purpose not in ADMITTED_POSE_ROUTE_PURPOSES:
        raise ValueError("unknown admitted pose route purpose")
    source = _TOUR_SOURCE if purpose == STORED_POSE_TOUR_ROUTE_PURPOSE else _SOURCE
    tour_identity = (
        {key: target_evidence.get(key) for key in ("tour_id", "visit_index", "qr_id")}
        if purpose == STORED_POSE_TOUR_ROUTE_PURPOSE else {}
    )
    navigation = target_evidence.get("tour_navigation")
    if navigation is not None:
        if purpose != STORED_POSE_TOUR_ROUTE_PURPOSE or not isinstance(navigation, Mapping):
            raise ValueError("tour navigation evidence requires stored pose tour purpose")
        tour_identity["tour_navigation"] = validate_tour_navigation(navigation)
        if navigation["stage_index"] != return_stage_index:
            raise ValueError("tour navigation stage differs from planning stage")
    overlay_binding = {}
    if temporary_obstacle_overlay_path is not None:
        overlay_path = Path(temporary_obstacle_overlay_path).resolve(strict=True)
        overlay_binding = {"temporary_obstacle_overlay_json": str(overlay_path),
                           "temporary_obstacle_overlay_sha256": file_sha256(overlay_path)}
    elif navigation is not None:
        raise ValueError("dynamic tour requires a temporary obstacle overlay")
    if (
        isinstance(return_stage_index, bool) or not isinstance(return_stage_index, int)
        or not 0 <= return_stage_index < MAX_RETURN_TO_START_LEGS
        or (route_uncertainty_context is None and return_stage_index != 0)
    ):
        raise ValueError("invalid return stage index or missing uncertainty context")
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
        target=target, start=start, purpose=purpose,
    )
    _measured_center(target_evidence)  # Malformed input is never a retryable geometry failure.
    stationary_turn = (target.x_m, target.y_m) == (start.x_m, start.y_m)
    if stationary_turn and _same_pose(start, target):
        raise ValueError("stored Start pose already reached; no travel route required")
    grid, bundle = load_occupancy_grid_with_bundle(
        map_yaml, semantic_map_id=semantic_map_id, planning_frame=plan.planning_frame,
    )
    if bundle.bundle_sha256 != snapshot.map_bundle_sha256:
        raise ValueError("stored target map differs from runtime map")
    base = Costmap.from_occupancy_grid(grid).with_arena_bounds(plan.arena_bounds)
    base = apply_bound_temporary_obstacles(base, {
        "route_purpose": purpose, **tour_identity, **overlay_binding,
        "map_bundle_sha256": bundle.bundle_sha256, "planning_frame": plan.planning_frame,
        "planning_frame_admission": target_evidence["planning_frame_admission"],
    })
    evidence_json = json.dumps(
        dict(target_evidence), indent=2, sort_keys=True, allow_nan=False,
    ) + "\n"
    target_sha256 = hashlib.sha256(evidence_json.encode()).hexdigest()
    alternative_evidence = None
    goal_connector = None
    geometry_args = dict(base=base, snapshot=snapshot, radius=radius, collision=collision,
        candidate_uid=candidate_uid, evidence=target_evidence, start=start, target=target,
        plan=plan, active=active, purpose=purpose, stationary_turn=stationary_turn)
    use_alternatives = navigation is not None and not stationary_turn and route_uncertainty_context is not None
    if use_alternatives:
        def build_candidate(inflation):
            # Only this pure geometry boundary may yield a retryable rejection.
            # Source files, frame admission and overlay replay were checked above.
            try:
                geometry, _ = _plan_admitted_pose_geometry(**geometry_args, inflation=inflation)
            except ValueError as exc:
                raise AlternativeGeometryRejected(str(exc)) from exc
            return geometry
        try:
            geometry, stage, inflation_radius_m, alternative_evidence = select_stored_pose_route_alternative(
                build_candidate, base_radius_m=inflation_radius_m, base_costmap=base,
                uncertainty=route_uncertainty_context, target_evidence_sha256=target_sha256,
                identity={"map_bundle_sha256": bundle.bundle_sha256, "candidate_snapshot_sha256": snapshot_hash,
                    "candidate_uid": candidate_uid, **tour_identity,
                    "temporary_obstacle_overlay_sha256": overlay_binding["temporary_obstacle_overlay_sha256"]},
            )
        except RouteAlternativesExhausted as exc:
            failure_path = Path(output_dir).with_name(Path(output_dir).name + "_alternatives_failure.json")
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            with failure_path.open("x") as handle:
                json.dump(exc.evidence, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
            raise
    else:
        geometry, goal_connector = _plan_admitted_pose_geometry(**geometry_args, inflation=inflation_radius_m)
        stage = (None if route_uncertainty_context is None else select_admitted_return_prefix(
            full_poses=geometry.full_poses, base_costmap=base,
            uncertainty=route_uncertainty_context, target_evidence_sha256=target_sha256,
            minimum_prefix_vertex_index=1 if geometry.connector.required else 0,
        ))
    result, planning, connector, smoothing = (geometry.result, geometry.planning_costmap,
                                             geometry.connector, geometry.smoothing)
    full_poses = geometry.full_poses
    poses, stage_target = full_poses, target
    if stage is not None:
        if not stage.is_final_stage and return_stage_index == MAX_RETURN_TO_START_LEGS - 1:
            raise ValueError("return stage limit exhausted before exact Start target")
        if goal_connector is not None and not stage.is_final_stage and stage.end_segment_index >= len(full_poses) - 2:
            raise ValueError("return prefix cannot stop inside the exact goal connector")
        poses, stage_target = stage.poses, stage.stage_target_pose
        cumulative = 0.
        stage_points = []
        for index, pose in enumerate(poses):
            distance = 0. if not index else math.hypot(pose.x_m - poses[index-1].x_m, pose.y_m - poses[index-1].y_m)
            cumulative += distance
            stage_points.append(RoutePoint(index, planning.world_to_grid(pose), pose, distance, cumulative))
        result = replace(result, route=replace(
            result.route, points=tuple(stage_points), requested_goal=stage_target,
            snapped_goal=stage_target, length_m=cumulative,
        ), diagnostics=replace(
            result.diagnostics, route_length_m=cumulative, path_cell_count=len(stage_points),
            goal_cell=stage_points[-1].cell, snapped_goal_cell=stage_points[-1].cell,
        ))
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {key: output_dir / name for key, name in (
        ("route_csv", "route.csv"), ("diagnostics_json", "route_diagnostics.json"),
        ("route_certificate_json", "route_certificate.json"),
        ("candidate_snapshot", "candidate_snapshot.json"), ("target_evidence_json", "target_evidence.json"),
    )}
    shutil.copyfile(snapshot_path, paths["candidate_snapshot"])
    paths["target_evidence_json"].write_text(evidence_json)
    if overlay_binding:
        paths["temporary_obstacle_overlay_json"] = output_dir / "temporary_obstacle_overlay.json"
        shutil.copyfile(overlay_binding["temporary_obstacle_overlay_json"], paths["temporary_obstacle_overlay_json"])
        overlay_binding["temporary_obstacle_overlay_json"] = str(paths["temporary_obstacle_overlay_json"])
    alternative_binding = {}
    if alternative_evidence is not None:
        paths["route_alternatives_json"] = output_dir / "route_alternatives.json"
        paths["route_alternatives_json"].write_text(json.dumps(alternative_evidence, indent=2, sort_keys=True, allow_nan=False) + "\n")
        alternative_binding = {"route_alternatives_json": str(paths["route_alternatives_json"]),
                               "route_alternatives_sha256": file_sha256(paths["route_alternatives_json"])}
    stage_metadata = None
    if stage is not None:
        paths["full_return_route_json"] = output_dir / "full_return_route.json"
        paths["uncertainty_selection_json"] = output_dir / "return_uncertainty_selection.json"
        full_evidence = {
            "schema_version": 1, "stage_index": return_stage_index,
            "start_candidate_uid": candidate_uid, "planning_frame": snapshot.planning_frame,
            "map_bundle_sha256": bundle.bundle_sha256, "candidate_snapshot_sha256": snapshot_hash,
            "target_evidence_sha256": file_sha256(paths["target_evidence_json"]),
            "stored_start_target_pose": asdict(target), "start_pose": asdict(start),
            "exact_start_connector": connector.to_metadata(),
            "exact_goal_connector": goal_connector,
            **return_route_geometry(full_poses),
            **({"route_purpose": purpose, **tour_identity} if tour_identity else {}),
            **overlay_binding, **alternative_binding,
        }
        paths["full_return_route_json"].write_text(json.dumps(full_evidence, indent=2, sort_keys=True, allow_nan=False) + "\n")
        paths["uncertainty_selection_json"].write_text(json.dumps(stage.evidence, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stage_metadata = {
            "stage_index": return_stage_index, "final_stage": stage.is_final_stage,
            "start_candidate_uid": candidate_uid, "stage_target_pose": asdict(stage_target),
            "end_segment_index": stage.end_segment_index, "end_fraction": stage.end_fraction,
            "full_return_route_json": str(paths["full_return_route_json"]),
            "full_return_route_sha256": file_sha256(paths["full_return_route_json"]),
            "uncertainty_selection_json": str(paths["uncertainty_selection_json"]),
            "uncertainty_selection_sha256": file_sha256(paths["uncertainty_selection_json"]),
        }
    write_route_csv(paths["route_csv"], (result,), final_yaw_by_leg={0: stage_target.yaw_rad})
    with paths["route_csv"].open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows, fields = list(reader), list(reader.fieldnames or ())
    fields.extend(("protected", "corridor", "simulation_only", "route_kind", "stationary_turn"))
    for index, row in enumerate(rows):
        terminal_anchor = (goal_connector is not None and (stage is None or stage.is_final_stage)
                           and index == len(rows) - 2)
        row.update(
            protected=str(index == len(rows) - 1 or terminal_anchor).lower(),
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
        "source": source, "route_kind": ADMITTED_POSE_ROUTE_KIND,
        "stationary_turn": stationary_turn,
        "route_purpose": purpose, "motion_authorized": True,
        **tour_identity,
        **overlay_binding, **alternative_binding,
        "planning_frame": snapshot.planning_frame, "physical_clearance_enforced": True,
        "physical_clearance": dict(physical_clearance), "inflation_radius_m": inflation_radius_m,
        "candidate_transit_radius_m": radius, "candidate_snapshot_json": str(paths["candidate_snapshot"]),
        "candidate_snapshot_sha256": snapshot_hash, "selected_candidate_stand_id": candidate_uid,
        "target_evidence_json": str(paths["target_evidence_json"]),
        "target_evidence_sha256": file_sha256(paths["target_evidence_json"]),
        "selected_approach_pose": asdict(stage_target), "stored_start_target_pose": asdict(target),
        "exact_start_connector": connector.to_metadata(),
        "exact_goal_connector": goal_connector,
        "route_start_pose_provenance": {"source": source, "planning_frame": snapshot.planning_frame, "pose": asdict(start)},
        "line_of_sight_route_optimization": {"enabled": smoothing.enabled, "legs": [smoothing.to_metadata()]},
        "route_csv_sha256": route_hash, "route_certificate_path": str(paths["route_certificate_json"]),
        "route_certificate_sha256": certificate_hash,
    })
    if stage_metadata is not None:
        metadata["return_to_start_stage"] = stage_metadata
    write_diagnostics_json(paths["diagnostics_json"], (result,), metadata)
    status = validate_admitted_pose_route_binding(
        paths["diagnostics_json"],
        load_route_leg(paths["route_csv"], 0, thinning_min_spacing_m=0.),
        candidate_snapshot_path=paths["candidate_snapshot"],
    )
    if not status.ok:
        raise ValueError("; ".join(status.failures))
    return {
        **{key: str(path) for key, path in paths.items()},
        "is_final_stage": stage is None or stage.is_final_stage,
        "stage_target_pose": asdict(stage_target),
    }


def _validate_return_stage(
    metadata: Mapping[str, object], poses: tuple[Pose2D, ...], *,
    start: Pose2D, stored_target: Pose2D, target: Pose2D,
) -> tuple[Pose2D, ...]:
    stage = metadata.get("return_to_start_stage")
    if stage is None:
        if not _same_pose(target, stored_target):
            raise ValueError("unstaged return must end at exact stored Start")
        return poses
    if not isinstance(stage, Mapping):
        raise ValueError("invalid return stage metadata")
    index, final = stage.get("stage_index"), stage.get("final_stage")
    if (
        isinstance(index, bool) or not isinstance(index, int)
        or not 0 <= index < MAX_RETURN_TO_START_LEGS or not isinstance(final, bool)
        or (index == MAX_RETURN_TO_START_LEGS - 1 and not final)
        or stage.get("start_candidate_uid") != metadata.get("selected_candidate_stand_id")
        or not _same_pose(_pose(stage.get("stage_target_pose"), "stage_target_pose"), target)
    ):
        raise ValueError("return stage index, identity or endpoint mismatch")
    artifacts = {}
    for name in ("full_return_route", "uncertainty_selection"):
        path = Path(stage[f"{name}_json"])
        if file_sha256(path) != stage.get(f"{name}_sha256"):
            raise ValueError(f"return {name} hash mismatch")
        artifacts[name] = json.loads(path.read_text())
        if not isinstance(artifacts[name], Mapping):
            raise ValueError(f"return {name} must be an object")
    full, selection = artifacts["full_return_route"], artifacts["uncertainty_selection"]
    if full.get("exact_goal_connector") != metadata.get("exact_goal_connector"):
        raise ValueError("full return exact goal connector mismatch")
    if metadata.get("route_purpose") == STORED_POSE_TOUR_ROUTE_PURPOSE:
        for key in ("route_purpose", "tour_id", "visit_index", "qr_id"):
            if full.get(key) != metadata.get(key):
                raise ValueError(f"full stored-pose tour {key} mismatch")
        for key in ("tour_navigation", "temporary_obstacle_overlay_json", "temporary_obstacle_overlay_sha256",
                    "route_alternatives_json", "route_alternatives_sha256"):
            if full.get(key) != metadata.get(key):
                raise ValueError(f"full stored-pose tour {key} mismatch")
    for key, expected in (
        ("schema_version", 1), ("stage_index", index),
        ("start_candidate_uid", stage["start_candidate_uid"]),
        ("planning_frame", metadata["planning_frame"]),
        ("candidate_snapshot_sha256", metadata["candidate_snapshot_sha256"]),
        ("map_bundle_sha256", metadata["map_bundle_sha256"]),
        ("target_evidence_sha256", metadata["target_evidence_sha256"]),
        ("exact_start_connector", metadata["exact_start_connector"]),
    ):
        if full.get(key) != expected:
            raise ValueError(f"full return {key} mismatch")
    full_poses = tuple(_pose(p, "full return pose") for p in full["poses"])
    if (
        len(full_poses) < 2 or not _same_pose(full_poses[0], start)
        or not _same_pose(full_poses[-1], stored_target)
        or not _same_pose(_pose(full.get("start_pose"), "full start"), start)
        or not _same_pose(_pose(full.get("stored_start_target_pose"), "full target"), stored_target)
    ):
        raise ValueError("full return differs from admitted start or stored Start target")
    end_index, fraction = stage.get("end_segment_index"), stage.get("end_fraction")
    prefix = build_return_prefix(full_poses, end_index, fraction)
    if metadata.get("exact_goal_connector") is not None and not final and end_index >= len(full_poses) - 2:
        raise ValueError("return prefix cannot stop inside the exact goal connector")
    if final != (end_index == len(full_poses) - 2 and fraction == 1.):
        raise ValueError("return final-stage flag differs from full-route endpoint")
    if len(prefix) != len(poses) or any(
        math.hypot(a.x_m - b.x_m, a.y_m - b.y_m) > 1e-9 for a, b in zip(prefix, poses)
    ) or not _same_pose(prefix[-1], target):
        raise ValueError("return route is not the bound full-route prefix")
    if not final:
        remaining = sum(math.hypot(b.x_m - a.x_m, b.y_m - a.y_m) for a, b in zip(full_poses[end_index+1:], full_poses[end_index+2:]))
        remaining += math.hypot(full_poses[end_index+1].x_m - target.x_m, full_poses[end_index+1].y_m - target.y_m)
        if (
            math.hypot(target.x_m - start.x_m, target.y_m - start.y_m) < MINIMUM_STAGE_DISPLACEMENT_M
            or remaining < MINIMUM_REMAINING_ROUTE_M
        ):
            raise ValueError("return prefix does not make meaningful progress")
    expected_selection = {
        "schema_version": 1, "motion_authorized": False, "is_final_stage": final,
        "full_route_geometry_sha256": payload_sha256(return_route_geometry(full_poses)),
        "selected_route_geometry_sha256": payload_sha256(return_route_geometry(prefix)),
        "target_evidence_sha256": metadata["target_evidence_sha256"],
        "end_segment_index": end_index, "end_fraction": fraction,
        "minimum_prefix_vertex_index": 1 if metadata["exact_start_connector"]["required"] else 0,
    }
    for key, expected in expected_selection.items():
        if selection.get(key) != expected:
            raise ValueError(f"return uncertainty selection {key} mismatch")
    config = RouteUncertaintyAdmissionConfig(**selection["config"])
    if (
        config.heading_reference_x_m != start.x_m or config.heading_reference_y_m != start.y_m
        or config.braking_latency_distance_m < .075 - 1e-12
    ):
        raise ValueError("return selection anchor or braking reserve mismatch")
    grid, bundle = load_occupancy_grid_with_bundle(
        Path(metadata["map_yaml"]), semantic_map_id=metadata["semantic_map_id"],
        planning_frame=metadata["planning_frame"],
    )
    if bundle.bundle_sha256 != metadata["map_bundle_sha256"]:
        raise ValueError("return selection map bundle mismatch")
    base = Costmap.from_occupancy_grid(grid).with_arena_bounds(validate_arena_boundary_evidence(metadata))
    base = apply_bound_temporary_obstacles(base, metadata)
    admission = evaluate_admitted_return_stage_uncertainty(
        base, executable_return_poses(prefix), PlanarCovariance(**selection["covariance"]), config,
        start_pose=start, target_evidence_sha256=metadata["target_evidence_sha256"], is_final_stage=final,
    )
    if not admission.decision.accepted or payload_sha256(admission.evidence) != payload_sha256(selection["selected_admission"]):
        raise ValueError("return selected uncertainty evidence mismatch or rejected")
    if metadata.get("tour_navigation") is not None and not metadata.get("stationary_turn"):
        path = Path(metadata["route_alternatives_json"])
        if file_sha256(path) != metadata.get("route_alternatives_sha256"):
            raise ValueError("route alternatives hash mismatch")
        alternatives = json.loads(path.read_text())
        validate_alternative_evidence(alternatives, selected_radius_m=metadata["inflation_radius_m"],
            full_poses=full_poses, selection=selection, target_sha256=metadata["target_evidence_sha256"],
            identity={"map_bundle_sha256": metadata["map_bundle_sha256"],
                "candidate_snapshot_sha256": metadata["candidate_snapshot_sha256"],
                "candidate_uid": metadata["selected_candidate_stand_id"],
                **{key: metadata[key] for key in ("tour_id", "visit_index", "qr_id", "tour_navigation")},
                "temporary_obstacle_overlay_sha256": metadata["temporary_obstacle_overlay_sha256"]})
        if (alternatives["map_resolution_m"] != base.resolution
            or alternatives["robot_radius_m"] != config.robot_radius_m):
            raise ValueError("route alternatives map resolution or robot footprint mismatch")
        validate_physical_clearance(metadata["physical_clearance"],
            inflation_radius_m=alternatives["base_inflation_radius_m"],
            candidate_transit_radius_m=metadata["candidate_transit_radius_m"])
    elif any(key in metadata for key in ("route_alternatives_json", "route_alternatives_sha256")):
        raise ValueError("route alternatives require a travelling dynamic tour")
    validate_exact_start_route_binding(metadata, tuple((p.x_m, p.y_m) for p in full_poses))
    return full_poses


def _validate_exact_goal_connector(metadata, full_poses, *, snapshot, evidence,
                                   candidate_uid, active, stored_target, leg):
    """Recompute the terminal proof from bound source maps, never flags alone."""
    grid, bundle = load_occupancy_grid_with_bundle(
        Path(metadata["map_yaml"]), semantic_map_id=metadata["semantic_map_id"],
        planning_frame=metadata["planning_frame"],
    )
    if bundle.bundle_sha256 != metadata["map_bundle_sha256"]:
        raise ValueError("exact goal connector map bundle mismatch")
    base = Costmap.from_occupancy_grid(grid).with_arena_bounds(validate_arena_boundary_evidence(metadata))
    base = apply_bound_temporary_obstacles(base, metadata)
    radius = metadata["candidate_transit_radius_m"]
    collision = metadata["physical_clearance"].get("minimum_collision_standoff_m", radius)
    static, planning = _stored_pose_costmaps(
        base, snapshot=snapshot, radius=radius, inflation=metadata["inflation_radius_m"],
        collision=collision, candidate_uid=candidate_uid, evidence=evidence,
    )
    if metadata.get("route_alternatives_json") is not None:
        connector = metadata["exact_start_connector"]
        start = _pose(connector["exact_start"], "exact start")
        anchor = _pose(connector["anchor"], "exact start anchor")
        recomputed = _segment_clearance_evidence(base, start, anchor,
            required_clearance_m=metadata["inflation_radius_m"])
        if payload_sha256(recomputed.to_metadata()) != payload_sha256(connector):
            raise ValueError("route alternatives exact-start clearance proof mismatch")
        for index, (first, second) in enumerate(zip(full_poses, full_poses[1:])):
            if index == 0 and connector["required"]:
                if any(not planning.is_traversable(cell) and planning.cell_sources.get(cell) != CELL_SOURCE_INFLATED
                       for cell in supercover_segment_cells(planning, first, second)):
                    raise ValueError("route alternatives connector intersects live keepout")
            elif not segment_is_collision_free(planning, first, second):
                raise ValueError("route alternatives geometry violates selected inflation")
    recorded = metadata.get("exact_goal_connector")
    if recorded is None:
        if not segment_is_collision_free(planning, stored_target, stored_target):
            raise ValueError("blocked exact stored target lacks an exact goal connector proof")
        return
    if metadata["route_purpose"] != ADMITTED_POSE_ROUTE_PURPOSE or metadata["stationary_turn"]:
        raise ValueError("exact goal connector is restricted to travelling automatic Start return")

    def validate_clearance(poses):
        _validate_candidate_clearance(
            poses, snapshot=snapshot, candidate_uid=candidate_uid,
            transit_radius=radius, active_standoff=active, target_evidence=evidence,
            collision_standoff=collision,
        )

    recomputed = certify_stored_goal_connector(
        base_costmap=base, static_costmap=static, planning_costmap=planning,
        anchor=full_poses[-2], target=stored_target,
        inflation_radius_m=metadata["inflation_radius_m"],
        validate_candidate_clearance=validate_clearance,
    )
    recomputed.update(map_bundle_sha256=bundle.bundle_sha256,
                      candidate_snapshot_sha256=candidate_snapshot_sha256(snapshot))
    if payload_sha256(recorded) != payload_sha256(recomputed):
        raise ValueError("exact goal connector evidence differs from recomputed route clearance")
    stage = metadata.get("return_to_start_stage")
    if (stage is None or stage["final_stage"]) and not leg.raw_waypoints[-2].protected:
        raise ValueError("exact goal connector anchor must remain protected from thinning")


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
        purpose = metadata.get("route_purpose")
        if purpose not in ADMITTED_POSE_ROUTE_PURPOSES:
            raise ValueError("admitted pose route route_purpose mismatch")
        for key, expected in (
            ("route_kind", ADMITTED_POSE_ROUTE_KIND),
            ("source", _TOUR_SOURCE if purpose == STORED_POSE_TOUR_ROUTE_PURPOSE else _SOURCE),
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
        stored_target = _pose(metadata.get("stored_start_target_pose", metadata.get("selected_approach_pose")), "stored_start_target_pose")
        evidence = json.loads(evidence_path.read_text())
        start = _pose(metadata["exact_start_connector"]["exact_start"], "exact_start")
        _validate_target_evidence(
            evidence, snapshot=snapshot, candidate_uid=uid, target=stored_target, start=start, purpose=purpose,
        )
        if purpose == STORED_POSE_TOUR_ROUTE_PURPOSE:
            for key in ("tour_id", "visit_index", "qr_id"):
                if (
                    metadata.get(key) != evidence.get(key)
                    or (key == "visit_index" and type(metadata.get(key)) is not int)
                ):
                    raise ValueError(f"stored-pose tour {key} differs from target evidence")
            if metadata.get("tour_navigation") != evidence.get("tour_navigation"):
                raise ValueError("stored-pose tour navigation differs from target evidence")
            navigation = metadata.get("tour_navigation")
            if navigation is not None:
                navigation = validate_tour_navigation(navigation)
                if navigation["stage_index"] != metadata.get("return_to_start_stage", {}).get("stage_index"):
                    raise ValueError("stored-pose tour navigation stage differs from selected route")
        # This also covers legacy unstaged tooling routes: no artifact may
        # bypass scope, source replay or exact projection-frame validation.
        if any(key in metadata for key in (
            "temporary_obstacle_overlay_json", "temporary_obstacle_overlay_sha256", "tour_navigation",
        )):
            grid, bundle = load_occupancy_grid_with_bundle(
                Path(metadata["map_yaml"]), semantic_map_id=metadata["semantic_map_id"],
                planning_frame=metadata["planning_frame"])
            if bundle.bundle_sha256 != metadata["map_bundle_sha256"]:
                raise ValueError("temporary occupancy route map bundle mismatch")
            apply_bound_temporary_obstacles(Costmap.from_occupancy_grid(grid), metadata)
        final = leg.raw_waypoints[-1]
        if not final.protected or not final.corridor or not _same_pose(final.pose, target):
            raise ValueError("route endpoint differs from exact stored admitted pose")
        if any(math.isfinite(w.pose.yaw_rad) for w in leg.raw_waypoints[:-1]):
            raise ValueError("admitted pose transit yaw must remain unconstrained")
        validate_arena_boundary_evidence(metadata)
        poses = tuple(w.pose for w in leg.raw_waypoints)
        full_poses = _validate_return_stage(metadata, poses, start=start, stored_target=stored_target, target=target)
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
        _validate_exact_goal_connector(
            metadata, full_poses, snapshot=snapshot, evidence=evidence,
            candidate_uid=uid, active=active, stored_target=stored_target, leg=leg,
        )
        _validate_candidate_clearance(
            full_poses, snapshot=snapshot, candidate_uid=uid,
            transit_radius=metadata["candidate_transit_radius_m"],
            active_standoff=active, target_evidence=evidence,
            collision_standoff=metadata["physical_clearance"].get(
                "minimum_collision_standoff_m", metadata["candidate_transit_radius_m"],
            ),
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return PreflightStatus(ok=False, failures=[f"admitted pose binding is invalid: {exc}"])
    return PreflightStatus(ok=True, failures=[])
