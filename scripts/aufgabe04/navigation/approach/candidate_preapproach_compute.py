"""ROS-free, no-write computation for camera candidate pre-approaches.

The selector previews every LiDAR-admitted candidate through this module.  A
preview loads or reuses immutable map inputs, computes and certifies the exact
route, and returns metrics without creating artifacts or authorizing motion.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.navigation.approach.candidate_goal_cell_selection import (
    NoSafetyRankedGoalRouteError,
    plan_safety_ranked_quantized_goal,
)
from scripts.aufgabe04.navigation.approach.candidate_preapproach_models import (
    CandidatePlanningContext,
    CandidatePreapproachPlan,
    CandidatePreapproachUnreachableError,
)
from scripts.aufgabe04.navigation.approach.detected_stand_preapproach import (
    CAMERA_AXIS_FACE_BEARING_MODE,
    ROBOT_TO_STAND_BEARING_MODE,
)
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import (
    normalize_angle,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    CoverageSurveyPlan,
)
from scripts.aufgabe04.navigation.execution.execution_route_certificate import (
    point_to_segment_distance_m,
)
from scripts.aufgabe04.navigation.execution.route_context import (
    StationRouteDryRun,
    build_route_metadata,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D, Route
from scripts.aufgabe04.navigation.planning.certified_exact_start_route import (
    certify_and_smooth_exact_start_route,
)
from scripts.aufgabe04.navigation.planning.global_planner import plan_route
from scripts.aufgabe04.navigation.planning.map_io import (
    load_occupancy_grid_with_bundle,
)
from scripts.aufgabe04.navigation.planning.route_costmaps import (
    build_station_route_costmaps,
)
from scripts.aufgabe04.navigation.planning.route_smoothing import (
    smooth_plan_route_results,
)
from scripts.aufgabe04.navigation.planning.station_approach import (
    navigation_targets_from_visits,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateSnapshot,
    candidate_snapshot_sha256,
)
from scripts.aufgabe04.stations.models import Station, StationPose
from scripts.aufgabe04.stations.station_router import build_station_visits


def load_candidate_planning_context(
    map_yaml: Path,
    *,
    semantic_map_id: str,
    plan: CoverageSurveyPlan,
    snapshot: CandidateSnapshot,
    inflation_radius_m: float,
    candidate_transit_radius_m: float,
    physical_clearance: Mapping[str, float],
) -> CandidatePlanningContext:
    """Load immutable map inputs once and validate every evidence binding."""

    minimum_active, minimum_transit, minimum_inflation = (
        validate_physical_clearance(
            physical_clearance,
            inflation_radius_m=inflation_radius_m,
            candidate_transit_radius_m=candidate_transit_radius_m,
        )
    )
    grid, map_bundle = load_occupancy_grid_with_bundle(
        map_yaml,
        semantic_map_id=semantic_map_id,
        planning_frame=plan.planning_frame,
    )
    if map_bundle.bundle_sha256 != snapshot.map_bundle_sha256:
        raise ValueError("candidate snapshot map differs from runtime map")
    if snapshot.map_bundle_sha256 != plan.map_bundle_sha256:
        raise ValueError("candidate snapshot map differs from coverage plan")
    keepouts = {
        candidate.candidate_uid: Station(
            candidate.candidate_uid,
            StationPose(
                candidate.geometry.x_m,
                candidate.geometry.y_m,
                0.0,
            ),
            0.0,
            candidate_transit_radius_m,
        )
        for candidate in snapshot.candidates
    }
    costmaps = build_station_route_costmaps(
        grid,
        station_map=keepouts,
        inflation_radius_m=inflation_radius_m,
        transit_keepout_radius_m=candidate_transit_radius_m,
        arena_bounds=plan.arena_bounds,
    )
    return CandidatePlanningContext(
        grid=grid,
        map_bundle=map_bundle,
        costmaps=costmaps,
        candidate_snapshot_sha256=candidate_snapshot_sha256(snapshot),
        inflation_radius_m=inflation_radius_m,
        candidate_transit_radius_m=candidate_transit_radius_m,
        minimum_active_standoff_m=minimum_active,
        minimum_candidate_transit_radius_m=minimum_transit,
        minimum_static_inflation_m=minimum_inflation,
    )


def compute_candidate_preapproach_plan(
    *,
    map_yaml: Path,
    semantic_map_id: str,
    plan: CoverageSurveyPlan,
    snapshot: CandidateSnapshot,
    candidate_uid: str,
    start: Pose2D,
    approach_offset_m: float,
    inflation_radius_m: float,
    candidate_transit_radius_m: float,
    physical_clearance: Mapping[str, float],
    approach_normal_rad: float | None = None,
    planning_context: CandidatePlanningContext | None = None,
) -> CandidatePreapproachPlan:
    """Compute the exact route used for both candidate scoring and sealing."""

    _validate_finite_pose(start)
    for name, value, allow_zero in (
        ("approach_offset_m", approach_offset_m, False),
        ("inflation_radius_m", inflation_radius_m, True),
        ("candidate_transit_radius_m", candidate_transit_radius_m, True),
    ):
        if not math.isfinite(value) or value < 0.0 or (
            not allow_zero and value == 0.0
        ):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {qualifier}")

    candidate = snapshot.candidate_for(candidate_uid)
    if candidate is None:
        raise ValueError(f"unknown candidate {candidate_uid!r}")
    if snapshot.map_bundle_sha256 != plan.map_bundle_sha256:
        raise ValueError("candidate snapshot map differs from coverage plan")

    if approach_normal_rad is None:
        bearing = math.atan2(
            candidate.geometry.y_m - start.y_m,
            candidate.geometry.x_m - start.x_m,
        )
        bearing_mode = ROBOT_TO_STAND_BEARING_MODE
    else:
        if not math.isfinite(approach_normal_rad):
            raise ValueError("approach face normal must be finite")
        bearing = normalize_angle(approach_normal_rad + math.pi)
        bearing_mode = CAMERA_AXIS_FACE_BEARING_MODE

    context = planning_context or load_candidate_planning_context(
        map_yaml,
        semantic_map_id=semantic_map_id,
        plan=plan,
        snapshot=snapshot,
        inflation_radius_m=inflation_radius_m,
        candidate_transit_radius_m=candidate_transit_radius_m,
        physical_clearance=physical_clearance,
    )
    _validate_context_binding(
        context,
        snapshot=snapshot,
        physical_clearance=physical_clearance,
        inflation_radius_m=inflation_radius_m,
        candidate_transit_radius_m=candidate_transit_radius_m,
    )
    validate_approach_outside_transit_keepout(
        approach_offset_m=approach_offset_m,
        candidate_transit_radius_m=candidate_transit_radius_m,
        map_resolution_m=context.grid.metadata.resolution,
    )

    stations = _candidate_station_map(
        snapshot=snapshot,
        target_uid=candidate_uid,
        target_bearing_rad=bearing,
        approach_offset_m=approach_offset_m,
        candidate_transit_radius_m=candidate_transit_radius_m,
    )
    try:
        visits = tuple(build_station_visits(("D00",), stations))
        targets = navigation_targets_from_visits(
            visits,
            context.costmaps.target_costmap,
        )
    except ValueError as exc:
        if not _is_candidate_target_failure(str(exc)):
            raise
        raise CandidatePreapproachUnreachableError(
            candidate_uid,
            str(exc),
        ) from exc

    goal_cell_selection = None
    requested_goal = targets[0].pose
    if approach_normal_rad is None:
        result = plan_route(
            context.costmaps.planning_costmap,
            start,
            requested_goal,
            snap_radius_m=plan.config.snap_radius_m,
        )
        unsmoothed = smooth_plan_route_results(
            (result,),
            costmap=context.costmaps.planning_costmap,
            enabled=False,
        )[0]
        result = unsmoothed.result
        dry_run_smoothing = unsmoothed.summary
        if result.route is None or result.failure is not None:
            reason = (
                result.failure.reason
                if result.failure is not None
                else "no route"
            )
            raise CandidatePreapproachUnreachableError(candidate_uid, reason)
        try:
            result, connector, smoothing = certify_and_smooth_exact_start_route(
                result,
                base_costmap=context.costmaps.base_costmap,
                planning_costmap=context.costmaps.planning_costmap,
                exact_start=start,
                required_clearance_m=inflation_radius_m,
            )
        except ValueError as exc:
            raise CandidatePreapproachUnreachableError(
                candidate_uid,
                str(exc),
            ) from exc
    else:
        try:
            selected_goal = plan_safety_ranked_quantized_goal(
                base_costmap=context.costmaps.base_costmap,
                planning_costmap=context.costmaps.planning_costmap,
                start=start,
                requested_goal=requested_goal,
                stand=Pose2D(
                    candidate.geometry.x_m,
                    candidate.geometry.y_m,
                    0.0,
                ),
                minimum_standoff_m=context.minimum_active_standoff_m,
                snap_radius_m=plan.config.snap_radius_m,
                required_start_clearance_m=inflation_radius_m,
                route_rejection_reason=lambda route: (
                    _candidate_route_clearance_failure(
                        candidate_uid=candidate_uid,
                        route=route,
                        snapshot=snapshot,
                        minimum_candidate_transit_radius_m=(
                            context.minimum_candidate_transit_radius_m
                        ),
                    )
                ),
            )
        except NoSafetyRankedGoalRouteError as exc:
            raise CandidatePreapproachUnreachableError(
                candidate_uid,
                str(exc),
            ) from exc
        result = selected_goal.result
        connector = selected_goal.connector
        smoothing = selected_goal.smoothing
        goal_cell_selection = selected_goal.evidence
        dry_run_smoothing = smoothing

    metadata = build_route_metadata(
        map_yaml,
        context.grid,
        ("D00",),
        arena_bounds=plan.arena_bounds,
        map_bundle=context.map_bundle,
    )
    metadata["inflation_radius_m"] = inflation_radius_m
    metadata["line_of_sight_route_optimization"] = {
        "enabled": dry_run_smoothing.enabled,
        "legs": [dry_run_smoothing.to_metadata()],
        "input_point_count": dry_run_smoothing.input_point_count,
        "output_point_count": dry_run_smoothing.output_point_count,
        "optimized_leg_count": int(dry_run_smoothing.optimized),
    }
    dry_run = StationRouteDryRun(
        grid=context.grid,
        base_costmap=context.costmaps.base_costmap,
        planning_costmap=context.costmaps.planning_costmap,
        station_map=stations,
        visits=visits,
        targets=targets,
        results=(result,),
        arena_bounds=plan.arena_bounds,
        metadata=metadata,
    )

    assert result.route is not None
    endpoint = result.route.points[-1].pose
    terminal_yaw = math.atan2(
        candidate.geometry.y_m - endpoint.y_m,
        candidate.geometry.x_m - endpoint.x_m,
    )
    initial_turn, turn_burden = route_turn_metrics(
        result.route,
        start_yaw_rad=start.yaw_rad,
        terminal_yaw_rad=terminal_yaw,
    )
    distance_to_stand = math.hypot(
        start.x_m - candidate.geometry.x_m,
        start.y_m - candidate.geometry.y_m,
    )
    endpoint_standoff = math.hypot(
        endpoint.x_m - candidate.geometry.x_m,
        endpoint.y_m - candidate.geometry.y_m,
    )
    _validate_candidate_route_clearance(
        candidate_uid=candidate_uid,
        route=result.route,
        endpoint_standoff_m=endpoint_standoff,
        snapshot=snapshot,
        minimum_active_standoff_m=context.minimum_active_standoff_m,
        minimum_candidate_transit_radius_m=(
            context.minimum_candidate_transit_radius_m
        ),
    )
    return CandidatePreapproachPlan(
        candidate_uid=candidate_uid,
        candidate_snapshot_sha256=context.candidate_snapshot_sha256,
        start=start,
        approach_offset_m=approach_offset_m,
        inflation_radius_m=inflation_radius_m,
        candidate_transit_radius_m=candidate_transit_radius_m,
        approach_bearing_rad=bearing,
        approach_bearing_mode=bearing_mode,
        dry_run=dry_run,
        result=result,
        connector=connector,
        smoothing=smoothing,
        selected_approach_pose=Pose2D(endpoint.x_m, endpoint.y_m, terminal_yaw),
        terminal_yaw_rad=terminal_yaw,
        route_length_m=result.route.length_m,
        initial_turn_rad=initial_turn,
        turn_burden_rad=turn_burden,
        distance_to_stand_m=distance_to_stand,
        endpoint_standoff_m=endpoint_standoff,
        inside_requested_standoff=(
            distance_to_stand + 1.0e-9 < approach_offset_m
        ),
        minimum_active_standoff_m=context.minimum_active_standoff_m,
        minimum_candidate_transit_radius_m=(
            context.minimum_candidate_transit_radius_m
        ),
        minimum_static_inflation_m=context.minimum_static_inflation_m,
        goal_cell_selection=goal_cell_selection,
    )


def route_turn_metrics(
    route: Route,
    *,
    start_yaw_rad: float,
    terminal_yaw_rad: float,
) -> tuple[float, float]:
    """Return initial turn and total absolute heading change for a route."""

    if not math.isfinite(start_yaw_rad) or not math.isfinite(terminal_yaw_rad):
        raise ValueError("route headings must be finite")
    headings: list[float] = []
    for previous, current in zip(route.points, route.points[1:]):
        dx = current.pose.x_m - previous.pose.x_m
        dy = current.pose.y_m - previous.pose.y_m
        if math.hypot(dx, dy) > 1.0e-9:
            headings.append(math.atan2(dy, dx))
    if not headings:
        turn = abs(normalize_angle(terminal_yaw_rad - start_yaw_rad))
        return turn, turn
    initial = abs(normalize_angle(headings[0] - start_yaw_rad))
    total = initial
    total += sum(
        abs(normalize_angle(current - previous))
        for previous, current in zip(headings, headings[1:])
    )
    total += abs(normalize_angle(terminal_yaw_rad - headings[-1]))
    return initial, total


def validate_physical_clearance(
    physical_clearance: Mapping[str, float],
    *,
    inflation_radius_m: float,
    candidate_transit_radius_m: float,
) -> tuple[float, float, float]:
    """Validate configured physical minima and return normalized values."""

    values = []
    for name in (
        "minimum_active_standoff_m",
        "minimum_candidate_transit_radius_m",
        "minimum_static_inflation_m",
    ):
        try:
            value = float(physical_clearance[name])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"physical clearance {name} must be numeric") from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"physical clearance {name} must be finite and non-negative"
            )
        values.append(value)
    minimum_active, minimum_transit, minimum_inflation = values
    if inflation_radius_m + 1.0e-9 < minimum_inflation:
        raise ValueError("static inflation is below the physical minimum")
    if candidate_transit_radius_m + 1.0e-9 < minimum_transit:
        raise ValueError("candidate transit radius is below the physical minimum")
    return minimum_active, minimum_transit, minimum_inflation


def _validate_context_binding(
    context: CandidatePlanningContext,
    *,
    snapshot: CandidateSnapshot,
    physical_clearance: Mapping[str, float],
    inflation_radius_m: float,
    candidate_transit_radius_m: float,
) -> None:
    if context.map_bundle.bundle_sha256 != snapshot.map_bundle_sha256:
        raise ValueError("candidate planning context has the wrong map bundle")
    if context.candidate_snapshot_sha256 != candidate_snapshot_sha256(snapshot):
        raise ValueError("candidate planning context has the wrong snapshot")
    if not math.isclose(
        context.inflation_radius_m,
        inflation_radius_m,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ) or not math.isclose(
        context.candidate_transit_radius_m,
        candidate_transit_radius_m,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("candidate planning context has the wrong clearance")
    expected_clearance = validate_physical_clearance(
        physical_clearance,
        inflation_radius_m=inflation_radius_m,
        candidate_transit_radius_m=candidate_transit_radius_m,
    )
    context_clearance = (
        context.minimum_active_standoff_m,
        context.minimum_candidate_transit_radius_m,
        context.minimum_static_inflation_m,
    )
    if context_clearance != expected_clearance:
        raise ValueError("candidate planning context has the wrong physical minimums")


def _candidate_station_map(
    *,
    snapshot: CandidateSnapshot,
    target_uid: str,
    target_bearing_rad: float,
    approach_offset_m: float,
    candidate_transit_radius_m: float,
) -> Mapping[str, Station]:
    stations: dict[str, Station] = {}
    for index, item in enumerate(snapshot.candidates, start=1):
        station_id = "D00" if item.candidate_uid == target_uid else f"K{index:02d}"
        stations[station_id] = Station(
            station_id,
            StationPose(
                item.geometry.x_m,
                item.geometry.y_m,
                target_bearing_rad if item.candidate_uid == target_uid else 0.0,
            ),
            approach_offset_m,
            candidate_transit_radius_m,
        )
    return stations


def _validate_finite_pose(pose: Pose2D) -> None:
    if not isinstance(pose, Pose2D) or not all(
        math.isfinite(float(value))
        for value in (pose.x_m, pose.y_m, pose.yaw_rad)
    ):
        raise ValueError("candidate route start must be a finite Pose2D")


def validate_approach_outside_transit_keepout(
    *,
    approach_offset_m: float,
    candidate_transit_radius_m: float,
    map_resolution_m: float,
) -> None:
    if not math.isfinite(map_resolution_m) or map_resolution_m <= 0.0:
        raise ValueError("map resolution must be finite and positive")
    # The requested world point is rasterized to a cell center before A*.  A
    # half-cell diagonal is the maximum quantization shift, so retaining that
    # margin prevents a nominally outside target from landing in self-keepout.
    raster_margin_m = map_resolution_m / math.sqrt(2.0)
    minimum_offset_m = candidate_transit_radius_m + raster_margin_m
    if approach_offset_m <= minimum_offset_m + 1.0e-9:
        raise ValueError(
            "candidate approach offset must remain outside its transit "
            "keepout after map rasterization: "
            f"offset={approach_offset_m:.3f} m, minimum>{minimum_offset_m:.3f} m"
        )


def _validate_candidate_route_clearance(
    *,
    candidate_uid: str,
    route: Route,
    endpoint_standoff_m: float,
    snapshot: CandidateSnapshot,
    minimum_active_standoff_m: float,
    minimum_candidate_transit_radius_m: float,
) -> None:
    if endpoint_standoff_m + 1.0e-9 < minimum_active_standoff_m:
        raise CandidatePreapproachUnreachableError(
            candidate_uid,
            "terminal pose violates the selected stand LiDAR standoff",
        )
    reason = _candidate_route_clearance_failure(
        candidate_uid=candidate_uid,
        route=route,
        snapshot=snapshot,
        minimum_candidate_transit_radius_m=(
            minimum_candidate_transit_radius_m
        ),
    )
    if reason is not None:
        raise CandidatePreapproachUnreachableError(candidate_uid, reason)


def _candidate_route_clearance_failure(
    *,
    candidate_uid: str,
    route: Route,
    snapshot: CandidateSnapshot,
    minimum_candidate_transit_radius_m: float,
) -> str | None:
    """Return the first non-target keepout violation on a candidate route."""

    for candidate in snapshot.candidates:
        if candidate.candidate_uid == candidate_uid:
            continue
        required = max(
            candidate.geometry.keepout_radius_m,
            minimum_candidate_transit_radius_m,
        )
        measured = _minimum_route_clearance_m(
            route,
            candidate.geometry.x_m,
            candidate.geometry.y_m,
        )
        if measured + 1.0e-9 < required:
            return (
                "route clearance to "
                f"{candidate.candidate_uid} is {measured:.3f} m, "
                f"below {required:.3f} m"
            )
    return None


def _minimum_route_clearance_m(
    route: Route,
    point_x_m: float,
    point_y_m: float,
) -> float:
    poses = tuple(point.pose for point in route.points)
    if not poses:
        raise ValueError("candidate route is empty")
    point = Pose2D(point_x_m, point_y_m, 0.0)
    if len(poses) < 2:
        return math.hypot(poses[0].x_m - point_x_m, poses[0].y_m - point_y_m)
    return min(
        point_to_segment_distance_m(point, start, end)
        for start, end in zip(poses, poses[1:])
    )


def _is_candidate_target_failure(message: str) -> bool:
    return (
        " target is blocked:" in message
        or " target is outside map bounds:" in message
    )


__all__ = [
    "compute_candidate_preapproach_plan",
    "load_candidate_planning_context",
    "route_turn_metrics",
    "validate_approach_outside_transit_keepout",
    "validate_physical_clearance",
]
