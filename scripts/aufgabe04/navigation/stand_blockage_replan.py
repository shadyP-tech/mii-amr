"""Stopped stand-observation fusion and certified A* blockage replanning.

This module never publishes motion.  It consumes a stationary observation
receipt, adds confirmed stand candidates to the persistent survey registry,
and plans a replacement route to the still-unvisited coverage viewpoint.  If
the robot is inside the conservative transit keepout, the replacement begins
with a short, continuously checked segment that moves away from the blocker.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from scripts.aufgabe04.navigation.artifacts import (
    write_diagnostics_json,
    write_route_csv,
)
from scripts.aufgabe04.navigation.costmap import Costmap
from scripts.aufgabe04.navigation.dynamic_approach_planner import (
    segment_is_collision_free,
)
from scripts.aufgabe04.navigation.global_planner import PlanRouteResult, plan_route
from scripts.aufgabe04.navigation.map_io import (
    OccupancyGrid,
    load_occupancy_grid_with_bundle,
)
from scripts.aufgabe04.navigation.models import (
    GridCell,
    Pose2D,
    Route,
    RoutePoint,
)
from scripts.aufgabe04.navigation.record_stand_coverage_stop import (
    _epoch_stands,
    _load_summary,
    _observations_from_epoch,
    _summary_scan_pose,
)
from scripts.aufgabe04.navigation.stand_coverage_survey import (
    CoverageSurveyPlan,
    StandSurveyRegistry,
    SurveyCandidate,
    coverage_survey_plan_sha256,
    fuse_confirmed_stands,
    load_coverage_survey_plan,
    load_stand_survey_registry,
    load_survey_progress,
    write_stand_survey_registry,
)
from scripts.aufgabe04.perception.stand_confirmation import ConfirmedStand
from scripts.aufgabe04.stations.models import Station, StationPose


BLOCKAGE_REPLAN_SCHEMA_VERSION = 1
DEFAULT_COLLISION_MARGIN_M = 0.02
DEFAULT_TRACKING_TUBE_RADIUS_M = 0.03
DEFAULT_BLOCKER_MAX_RANGE_M = 0.70
DEFAULT_BLOCKER_HALF_ANGLE_RAD = math.radians(70.0)
_EPSILON_M = 1.0e-9


@dataclass(frozen=True)
class BlockageRoutePlan:
    route_result: PlanRouteResult
    target_viewpoint_id: str
    target_pose: Pose2D
    blocker_candidate_uids: tuple[str, ...]
    start_pose: Pose2D
    egress_anchor: Pose2D
    egress_distance_m: float
    minimum_egress_hard_clearance_m: float


def _angle_error(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _candidate_distance(candidate: SurveyCandidate, pose: Pose2D) -> float:
    return math.hypot(candidate.x_m - pose.x_m, candidate.y_m - pose.y_m)


def _hard_exclusion_radius_m(
    candidate: SurveyCandidate,
    *,
    robot_radius_m: float,
    collision_margin_m: float,
    tracking_tube_radius_m: float,
) -> float:
    return (
        candidate.radius_m
        + candidate.uncertainty_m
        + robot_radius_m
        + collision_margin_m
        + tracking_tube_radius_m
    )


def _point_to_segment_distance_m(
    point: Pose2D,
    start: Pose2D,
    end: Pose2D,
) -> float:
    dx = end.x_m - start.x_m
    dy = end.y_m - start.y_m
    denominator = dx * dx + dy * dy
    if denominator <= 1.0e-18:
        return math.hypot(point.x_m - start.x_m, point.y_m - start.y_m)
    fraction = max(
        0.0,
        min(
            1.0,
            (
                (point.x_m - start.x_m) * dx
                + (point.y_m - start.y_m) * dy
            )
            / denominator,
        ),
    )
    nearest_x = start.x_m + fraction * dx
    nearest_y = start.y_m + fraction * dy
    return math.hypot(point.x_m - nearest_x, point.y_m - nearest_y)


def blocker_candidate_uids(
    registry: StandSurveyRegistry,
    stands: Sequence[ConfirmedStand],
    start: Pose2D,
    *,
    max_range_m: float = DEFAULT_BLOCKER_MAX_RANGE_M,
    half_angle_rad: float = DEFAULT_BLOCKER_HALF_ANGLE_RAD,
) -> tuple[str, ...]:
    """Bind newly confirmed observations to plausible frontal blockers."""

    observation_ids = {
        observation_id
        for stand in stands
        for observation_id in stand.source_observation_ids
    }
    selected = []
    for candidate in registry.candidates:
        if not observation_ids.intersection(candidate.source_observation_ids):
            continue
        distance_m = _candidate_distance(candidate, start)
        bearing = math.atan2(candidate.y_m - start.y_m, candidate.x_m - start.x_m)
        if distance_m > max_range_m + _EPSILON_M:
            continue
        if abs(_angle_error(bearing - start.yaw_rad)) > half_angle_rad + _EPSILON_M:
            continue
        selected.append((distance_m, candidate.candidate_uid))
    return tuple(uid for _distance, uid in sorted(selected))


def _known_candidates(registry: StandSurveyRegistry) -> tuple[SurveyCandidate, ...]:
    return tuple(
        candidate
        for candidate in registry.candidates
        if candidate.status != "rejected"
    )


def _static_costmap(
    occupancy_grid: OccupancyGrid,
    plan: CoverageSurveyPlan,
) -> Costmap:
    return (
        Costmap.from_occupancy_grid(occupancy_grid)
        .with_arena_bounds(plan.arena_bounds)
        .with_inflation(plan.config.inflation_radius_m)
    )


def _planning_costmap(
    static_costmap: Costmap,
    candidates: Sequence[SurveyCandidate],
) -> Costmap:
    # A half-cell diagonal closes the continuous gap between circular geometry
    # and centre-sampled grid rasterization.
    raster_margin_m = math.sqrt(2.0) * static_costmap.resolution / 2.0
    return static_costmap.with_station_keepouts(
        Station(
            station_id=candidate.candidate_uid,
            pose=StationPose(candidate.x_m, candidate.y_m, 0.0),
            approach_offset_m=0.0,
            keepout_radius_m=candidate.keepout_radius_m + raster_margin_m,
        )
        for candidate in candidates
    )


def _minimum_hard_segment_clearance_m(
    start: Pose2D,
    end: Pose2D,
    candidates: Sequence[SurveyCandidate],
    *,
    robot_radius_m: float,
    collision_margin_m: float,
    tracking_tube_radius_m: float,
) -> float:
    if not candidates:
        return math.inf
    return min(
        _point_to_segment_distance_m(
            Pose2D(candidate.x_m, candidate.y_m, 0.0),
            start,
            end,
        )
        - _hard_exclusion_radius_m(
            candidate,
            robot_radius_m=robot_radius_m,
            collision_margin_m=collision_margin_m,
            tracking_tube_radius_m=tracking_tube_radius_m,
        )
        for candidate in candidates
    )


def _candidate_anchor_cells(
    costmap: Costmap,
    start: Pose2D,
    *,
    search_radius_cells: int,
) -> Iterable[tuple[float, GridCell, Pose2D]]:
    start_cell = costmap.world_to_grid(start)
    candidates = []
    for dy in range(-search_radius_cells, search_radius_cells + 1):
        for dx in range(-search_radius_cells, search_radius_cells + 1):
            cell = GridCell(start_cell.x + dx, start_cell.y + dy)
            if not costmap.is_traversable(cell):
                continue
            anchor = costmap.grid_to_world(cell)
            distance_m = math.hypot(anchor.x_m - start.x_m, anchor.y_m - start.y_m)
            if distance_m <= _EPSILON_M:
                continue
            candidates.append((distance_m, cell, anchor))
    return tuple(sorted(candidates, key=lambda item: (item[0], item[1])))


def _find_safe_egress_anchor(
    static_costmap: Costmap,
    planning_costmap: Costmap,
    start: Pose2D,
    candidates: Sequence[SurveyCandidate],
    blockers: Sequence[SurveyCandidate],
    *,
    robot_radius_m: float,
    collision_margin_m: float,
    tracking_tube_radius_m: float,
    search_radius_m: float,
) -> tuple[Pose2D, float]:
    for candidate in candidates:
        start_clearance_m = (
            _candidate_distance(candidate, start)
            - _hard_exclusion_radius_m(
                candidate,
                robot_radius_m=robot_radius_m,
                collision_margin_m=collision_margin_m,
                tracking_tube_radius_m=tracking_tube_radius_m,
            )
        )
        if start_clearance_m <= _EPSILON_M:
            raise ValueError(
                "blockage pose is inside the hard stand exclusion envelope: "
                f"candidate={candidate.candidate_uid} "
                f"clearance={start_clearance_m:.6f} m"
            )

    search_radius_cells = max(
        1,
        int(math.ceil(search_radius_m / static_costmap.resolution)),
    )
    for _distance_m, _cell, anchor in _candidate_anchor_cells(
        planning_costmap,
        start,
        search_radius_cells=search_radius_cells,
    ):
        if not segment_is_collision_free(static_costmap, start, anchor):
            continue
        # When starting inside a conservative transit keepout, the connector
        # must make non-negative progress away from every blocking stand.
        moves_away = all(
            (
                (anchor.x_m - start.x_m) * (start.x_m - blocker.x_m)
                + (anchor.y_m - start.y_m) * (start.y_m - blocker.y_m)
            )
            > _EPSILON_M
            for blocker in blockers
        )
        if blockers and not moves_away:
            continue
        minimum_clearance_m = _minimum_hard_segment_clearance_m(
            start,
            anchor,
            candidates,
            robot_radius_m=robot_radius_m,
            collision_margin_m=collision_margin_m,
            tracking_tube_radius_m=tracking_tube_radius_m,
        )
        if minimum_clearance_m <= _EPSILON_M:
            continue
        return anchor, minimum_clearance_m
    raise ValueError("no continuously safe stand-blockage egress anchor")


def _prepend_exact_egress(
    result: PlanRouteResult,
    *,
    start: Pose2D,
    anchor: Pose2D,
    costmap: Costmap,
) -> PlanRouteResult:
    if result.route is None:
        return result
    original = list(result.route.points)
    if not original:
        raise ValueError("A* blockage replan returned no route points")
    if math.hypot(
        original[0].pose.x_m - anchor.x_m,
        original[0].pose.y_m - anchor.y_m,
    ) > 1.0e-8:
        raise ValueError("A* blockage replan lost its certified egress anchor")
    poses_and_cells = [(start, costmap.world_to_grid(start))]
    poses_and_cells.extend((point.pose, point.cell) for point in original)
    points = []
    cumulative_m = 0.0
    previous = None
    for index, (pose, cell) in enumerate(poses_and_cells):
        segment_m = (
            0.0
            if previous is None
            else math.hypot(pose.x_m - previous.x_m, pose.y_m - previous.y_m)
        )
        cumulative_m += segment_m
        points.append(
            RoutePoint(
                index=index,
                cell=cell,
                pose=pose,
                segment_length_m=segment_m,
                cumulative_length_m=cumulative_m,
            )
        )
        previous = pose
    route = Route(
        points=tuple(points),
        requested_start=start,
        requested_goal=result.route.requested_goal,
        snapped_start=anchor,
        snapped_goal=result.route.snapped_goal,
        length_m=cumulative_m,
    )
    diagnostics = replace(
        result.diagnostics,
        start_cell=costmap.world_to_grid(start),
        route_length_m=route.length_m,
    )
    return PlanRouteResult(route=route, diagnostics=diagnostics)


def _validate_route_keepout_clearance(
    route: Route,
    candidates: Sequence[SurveyCandidate],
    *,
    egress_end_index: int,
    robot_radius_m: float,
    collision_margin_m: float,
    tracking_tube_radius_m: float,
) -> None:
    for index, (start_point, end_point) in enumerate(
        zip(route.points, route.points[1:]),
        start=1,
    ):
        for candidate in candidates:
            distance_m = _point_to_segment_distance_m(
                Pose2D(candidate.x_m, candidate.y_m, 0.0),
                start_point.pose,
                end_point.pose,
            )
            minimum_m = (
                _hard_exclusion_radius_m(
                    candidate,
                    robot_radius_m=robot_radius_m,
                    collision_margin_m=collision_margin_m,
                    tracking_tube_radius_m=tracking_tube_radius_m,
                )
                if index <= egress_end_index
                else candidate.keepout_radius_m
            )
            if distance_m + _EPSILON_M < minimum_m:
                raise ValueError(
                    "replacement route violates stand clearance: "
                    f"segment={index - 1}->{index} "
                    f"candidate={candidate.candidate_uid} "
                    f"distance={distance_m:.6f} m minimum={minimum_m:.6f} m"
                )


def plan_blockage_route_to_viewpoint(
    occupancy_grid: OccupancyGrid,
    *,
    plan: CoverageSurveyPlan,
    registry: StandSurveyRegistry,
    start: Pose2D,
    target_viewpoint_id: str,
    blocker_uids: Sequence[str],
    robot_radius_m: float,
    collision_margin_m: float = DEFAULT_COLLISION_MARGIN_M,
    tracking_tube_radius_m: float = DEFAULT_TRACKING_TUBE_RADIUS_M,
    egress_search_radius_m: float = 0.60,
) -> BlockageRoutePlan:
    candidates = _known_candidates(registry)
    by_uid = {candidate.candidate_uid: candidate for candidate in candidates}
    blockers = tuple(by_uid[uid] for uid in blocker_uids if uid in by_uid)
    if not blockers:
        raise ValueError("blockage replan has no bound blocking stand candidate")
    target = plan.viewpoint_for(target_viewpoint_id)
    if target is None:
        raise ValueError(f"unknown blockage target viewpoint {target_viewpoint_id!r}")
    static_costmap = _static_costmap(occupancy_grid, plan)
    planning_costmap = _planning_costmap(static_costmap, candidates)
    anchor, minimum_clearance_m = _find_safe_egress_anchor(
        static_costmap,
        planning_costmap,
        start,
        candidates,
        blockers,
        robot_radius_m=robot_radius_m,
        collision_margin_m=collision_margin_m,
        tracking_tube_radius_m=tracking_tube_radius_m,
        search_radius_m=egress_search_radius_m,
    )
    planned = plan_route(
        planning_costmap,
        anchor,
        target.pose,
        snap_radius_m=plan.config.snap_radius_m,
    )
    if planned.route is None:
        reason = planned.failure.reason if planned.failure is not None else "no_path"
        raise ValueError(f"stand-blockage A* failed: {reason}")
    replacement = _prepend_exact_egress(
        planned,
        start=start,
        anchor=anchor,
        costmap=static_costmap,
    )
    assert replacement.route is not None
    _validate_route_keepout_clearance(
        replacement.route,
        candidates,
        egress_end_index=1,
        robot_radius_m=robot_radius_m,
        collision_margin_m=collision_margin_m,
        tracking_tube_radius_m=tracking_tube_radius_m,
    )
    return BlockageRoutePlan(
        route_result=replacement,
        target_viewpoint_id=target.viewpoint_id,
        target_pose=target.pose,
        blocker_candidate_uids=tuple(candidate.candidate_uid for candidate in blockers),
        start_pose=start,
        egress_anchor=anchor,
        egress_distance_m=math.hypot(anchor.x_m - start.x_m, anchor.y_m - start.y_m),
        minimum_egress_hard_clearance_m=minimum_clearance_m,
    )


def record_blockage_replan(
    *,
    survey_root: Path,
    map_yaml: Path,
    semantic_map_id: str,
    target_viewpoint_id: str,
    blockage_id: str,
    observer_summary_path: Path,
    output_dir: Path,
    robot_radius_m: float,
) -> dict[str, str]:
    """Validate one stationary blockage epoch and atomically expose a replan."""

    survey_root = Path(survey_root)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite blockage replan: {output_dir}")
    plan_path = survey_root / "coverage_plan.json"
    progress_path = survey_root / "coverage_progress.json"
    registry_path = survey_root / "stand_registry.json"
    plan = load_coverage_survey_plan(plan_path)
    progress = load_survey_progress(progress_path, plan)
    if target_viewpoint_id in progress.visited_viewpoint_ids:
        raise ValueError("cannot replan to an already visited viewpoint")
    registry = load_stand_survey_registry(registry_path, plan)
    grid, map_bundle = load_occupancy_grid_with_bundle(
        map_yaml,
        semantic_map_id=semantic_map_id,
        planning_frame=plan.planning_frame,
    )
    if map_bundle.bundle_sha256 != plan.map_bundle_sha256:
        raise ValueError("runtime map bundle differs from coverage plan")
    observer_summary = _load_summary(observer_summary_path)
    if observer_summary.get("map_bundle_sha256") != plan.map_bundle_sha256:
        raise ValueError("blockage observer map differs from coverage plan")
    if observer_summary.get("planning_frame") != plan.planning_frame:
        raise ValueError("blockage observer planning frame differs from plan")
    start = _summary_scan_pose(observer_summary)
    observations_path = Path(str(observer_summary.get("output_jsonl", "")))
    observations = _observations_from_epoch(
        summary=observer_summary,
        observations_path=observations_path,
        map_yaml=map_yaml,
        map_bundle=map_bundle,
        plan=plan,
    )
    stands = _epoch_stands(observations, plan)
    if not stands:
        raise ValueError("blocking LiDAR epoch confirmed no stand")
    updated_registry = fuse_confirmed_stands(
        registry,
        stands,
        viewpoint_id=blockage_id,
        config=plan.config,
    )
    blocker_uids = blocker_candidate_uids(updated_registry, stands, start)
    if not blocker_uids:
        raise ValueError("confirmed epoch has no near frontal blocking stand")
    blockage_plan = plan_blockage_route_to_viewpoint(
        grid,
        plan=plan,
        registry=updated_registry,
        start=start,
        target_viewpoint_id=target_viewpoint_id,
        blocker_uids=blocker_uids,
        robot_radius_m=robot_radius_m,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    route_path = output_dir / "route.csv"
    diagnostics_path = output_dir / "route_diagnostics.json"
    epoch_path = output_dir / "blockage_epoch.json"
    summary_path = output_dir / "blockage_replan_summary.json"
    write_route_csv(
        route_path,
        (blockage_plan.route_result,),
        final_yaw_by_leg={0: blockage_plan.target_pose.yaw_rad},
    )
    write_diagnostics_json(
        diagnostics_path,
        (blockage_plan.route_result,),
        metadata={
            "schema_version": 1,
            "route_kind": "stand_coverage_survey",
            "motion_authorized": False,
            "adaptive_blockage_replan": True,
            "survey_id": plan.survey_id,
            "plan_sha256": coverage_survey_plan_sha256(plan),
            "map_bundle_sha256": plan.map_bundle_sha256,
            "target_viewpoint_id": target_viewpoint_id,
            "target_pose": {
                "x_m": blockage_plan.target_pose.x_m,
                "y_m": blockage_plan.target_pose.y_m,
                "yaw_rad": blockage_plan.target_pose.yaw_rad,
            },
            "candidate_keepout_count": len(_known_candidates(updated_registry)),
            "blocker_candidate_uids": list(blocker_uids),
            "blockage_id": blockage_id,
            "blockage_observer_summary": str(observer_summary_path),
            "egress_anchor": {
                "x_m": blockage_plan.egress_anchor.x_m,
                "y_m": blockage_plan.egress_anchor.y_m,
            },
            "egress_distance_m": blockage_plan.egress_distance_m,
            "minimum_egress_hard_clearance_m": (
                blockage_plan.minimum_egress_hard_clearance_m
            ),
            "inflation_radius_m": plan.config.inflation_radius_m,
            "arena_boundary_overlay": True,
            "arena_bounds": plan.arena_bounds.to_metadata(),
        },
    )
    epoch_path.write_text(
        json.dumps(
            {
                "schema_version": BLOCKAGE_REPLAN_SCHEMA_VERSION,
                "blockage_id": blockage_id,
                "target_viewpoint_id": target_viewpoint_id,
                "observer_summary_json": str(observer_summary_path),
                "observations_jsonl": str(observations_path),
                "processed_scan_count": observer_summary["processed_scan_count"],
                "accepted_observation_count": len(observations),
                "confirmed_epoch_candidate_count": len(stands),
                "blocker_candidate_uids": list(blocker_uids),
                "scan_pose": {
                    "x_m": start.x_m,
                    "y_m": start.y_m,
                    "yaw_rad": start.yaw_rad,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    # Publish the mutable registry only after every replacement artifact and
    # clearance gate has succeeded.
    write_stand_survey_registry(registry_path, updated_registry, plan)
    summary = {
        "schema_version": BLOCKAGE_REPLAN_SCHEMA_VERSION,
        "status": "stand_blockage_replan_ready",
        "motion_published": False,
        "blockage_id": blockage_id,
        "target_viewpoint_id": target_viewpoint_id,
        "blocker_candidate_uids": list(blocker_uids),
        "route_csv": str(route_path),
        "diagnostics_json": str(diagnostics_path),
        "blockage_epoch_json": str(epoch_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {
        "route_csv": str(route_path),
        "diagnostics_json": str(diagnostics_path),
        "summary_json": str(summary_path),
        "blockage_epoch_json": str(epoch_path),
    }
