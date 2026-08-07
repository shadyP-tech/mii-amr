"""Certified A* replanning around stopped transit obstacles.

This module never publishes motion.  The autonomous explorer uses the
transient path below while travelling to a survey viewpoint: it converts the
front-clearance sample that already stopped the follower into a run-local
keepout and replans without performing a semantic stand-observation epoch or
changing the persistent survey registry.  If the robot is inside the
conservative transit keepout, the replacement begins with a short,
continuously checked connector that moves away from the blocker.  Forward
motion is preferred within the controller's translation-heading envelope; a
reverse recovery instead keeps one straight prefix through a farther,
rotation-safe forward-transition anchor.

The older stationary stand-observation entrypoint remains available for
artifact compatibility, but the autonomous coverage orchestrator must not use
it before reaching an inspection viewpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Sequence

from scripts.aufgabe04.navigation.artifacts import (
    write_diagnostics_json,
    write_route_csv,
)
from scripts.aufgabe04.navigation.costmap import Costmap
from scripts.aufgabe04.navigation.coverage_escape_geometry import (
    DEFAULT_FORWARD_TRANSLATION_HEADING_LIMIT_RAD,
    DEFAULT_REVERSE_CONNECTOR_ALIGNMENT_TOLERANCE_RAD,
    EGRESS_MODE_FORWARD,
    CircularEscapeKeepout,
    ExecutableEscapeGeometry,
    choose_egress_connectors,
    find_reverse_transition_anchors,
    validate_executable_escape_route,
)
from scripts.aufgabe04.navigation.dynamic_approach_planner import (
    greedy_line_of_sight_shortcut,
)
from scripts.aufgabe04.navigation.global_planner import PlanRouteResult, plan_route
from scripts.aufgabe04.navigation.map_io import (
    OccupancyGrid,
    load_occupancy_grid_with_bundle,
)
from scripts.aufgabe04.navigation.models import (
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
from scripts.aufgabe04.navigation.transient_blockage_policy import (
    CLEARANCE_LIMITED_MOTION_FLOOR,
)
from scripts.aufgabe04.perception.stand_confirmation import ConfirmedStand
from scripts.aufgabe04.stations.models import Station, StationPose


BLOCKAGE_REPLAN_SCHEMA_VERSION = 1
TRANSIENT_OBSTACLE_OVERLAY_SCHEMA_VERSION = 1
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
    egress_mode: str
    egress_transition_anchor: Pose2D
    egress_transition_waypoint_index: int
    egress_forward_waypoint_index: int | None
    egress_connector_heading_error_rad: float
    forward_translation_heading_limit_rad: float
    reverse_connector_alignment_tolerance_rad: float
    reverse_connector_heading_error_rad: float | None
    minimum_transition_keepout_tube_clearance_m: float | None
    tracking_tube_radius_m: float
    astar_tail_raw_point_count: int
    astar_tail_smoothed_point_count: int


@dataclass(frozen=True)
class TransientObstacleOverlay:
    """Run-local navigation obstacles that are never survey evidence."""

    schema_version: int
    survey_id: str
    planning_frame: str
    map_bundle_sha256: str
    candidates: tuple[SurveyCandidate, ...]


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


def _front_clearance_m(stop_details: dict[str, object]) -> float:
    front = stop_details.get("front_clearance")
    if not isinstance(front, dict) or front.get("source") != "front_sector":
        raise ValueError("transient blockage requires front-sector LiDAR evidence")
    try:
        clearance_m = float(front["nearest_valid_range_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("transient blockage has no finite front clearance") from exc
    if not math.isfinite(clearance_m) or clearance_m <= 0.0:
        raise ValueError("transient blockage has no finite front clearance")
    return clearance_m


def _front_bearing_rad(stop_details: dict[str, object]) -> float:
    front = stop_details.get("front_clearance")
    if not isinstance(front, dict) or front.get("source") != "front_sector":
        raise ValueError("transient blockage requires front-sector LiDAR evidence")
    try:
        bearing_rad = float(front["nearest_valid_bearing_rad"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("transient blockage has no finite obstacle bearing") from exc
    if not math.isfinite(bearing_rad):
        raise ValueError("transient blockage has no finite obstacle bearing")
    return bearing_rad


def _transient_candidate_payload(candidate: SurveyCandidate) -> dict[str, object]:
    return {
        "candidate_uid": candidate.candidate_uid,
        "x_m": candidate.x_m,
        "y_m": candidate.y_m,
        "radius_m": candidate.radius_m,
        "uncertainty_m": candidate.uncertainty_m,
        "keepout_radius_m": candidate.keepout_radius_m,
        "source_observation_ids": list(candidate.source_observation_ids),
    }


def _transient_candidate_from_payload(payload: dict[str, object]) -> SurveyCandidate:
    try:
        candidate = SurveyCandidate(
            candidate_uid=str(payload["candidate_uid"]),
            x_m=float(payload["x_m"]),
            y_m=float(payload["y_m"]),
            radius_m=float(payload["radius_m"]),
            uncertainty_m=float(payload["uncertainty_m"]),
            keepout_radius_m=float(payload["keepout_radius_m"]),
            confidence=1.0,
            hit_count=1,
            first_seen_sec=0.0,
            last_seen_sec=0.0,
            source_observation_ids=tuple(
                str(item) for item in payload.get("source_observation_ids", ())
            ),
            viewpoint_ids=(),
            status="provisional",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid transient obstacle candidate") from exc
    if not all(
        math.isfinite(value)
        for value in (
            candidate.x_m,
            candidate.y_m,
            candidate.radius_m,
            candidate.uncertainty_m,
            candidate.keepout_radius_m,
        )
    ):
        raise ValueError("transient obstacle candidate values must be finite")
    if (
        candidate.radius_m < 0.0
        or candidate.uncertainty_m < 0.0
        or candidate.keepout_radius_m <= 0.0
    ):
        raise ValueError("transient obstacle candidate geometry is invalid")
    return candidate


def load_transient_obstacle_overlay(
    path: Path,
    *,
    plan: CoverageSurveyPlan,
) -> TransientObstacleOverlay:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read transient obstacle overlay: {path}") from exc
    if payload.get("schema_version") != TRANSIENT_OBSTACLE_OVERLAY_SCHEMA_VERSION:
        raise ValueError("unsupported transient obstacle overlay schema")
    if payload.get("survey_id") != plan.survey_id:
        raise ValueError("transient obstacle overlay survey differs from plan")
    if payload.get("planning_frame") != plan.planning_frame:
        raise ValueError("transient obstacle overlay frame differs from plan")
    if payload.get("map_bundle_sha256") != plan.map_bundle_sha256:
        raise ValueError("transient obstacle overlay map differs from plan")
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("transient obstacle overlay candidates must be a list")
    candidates = tuple(
        _transient_candidate_from_payload(item)
        for item in raw_candidates
        if isinstance(item, dict)
    )
    if len(candidates) != len(raw_candidates):
        raise ValueError("transient obstacle overlay candidate must be an object")
    if not candidates:
        raise ValueError("transient obstacle overlay is empty")
    if len({item.candidate_uid for item in candidates}) != len(candidates):
        raise ValueError("transient obstacle overlay candidate IDs are not unique")
    return TransientObstacleOverlay(
        schema_version=TRANSIENT_OBSTACLE_OVERLAY_SCHEMA_VERSION,
        survey_id=plan.survey_id,
        planning_frame=plan.planning_frame,
        map_bundle_sha256=plan.map_bundle_sha256,
        candidates=candidates,
    )


def write_transient_obstacle_overlay(
    path: Path,
    overlay: TransientObstacleOverlay,
    *,
    source: dict[str, object],
) -> None:
    path = Path(path)
    if path.exists():
        raise ValueError(f"refusing to overwrite transient obstacle overlay: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": overlay.schema_version,
                "purpose": "transient_navigation_obstacle",
                "semantic_survey_evidence": False,
                "motion_published": False,
                "survey_id": overlay.survey_id,
                "planning_frame": overlay.planning_frame,
                "map_bundle_sha256": overlay.map_bundle_sha256,
                "source": source,
                "candidates": [
                    _transient_candidate_payload(item)
                    for item in overlay.candidates
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _new_transient_candidate(
    *,
    plan: CoverageSurveyPlan,
    start: Pose2D,
    clearance_m: float,
    bearing_rad: float,
    blockage_id: str,
    candidate_index: int,
) -> SurveyCandidate:
    # The LiDAR return is on the visible obstacle surface. Move the nominal
    # centre one configured obstacle radius beyond it so the hard robot-body
    # envelope remains physically meaningful at the already-safe stop pose.
    centre_range_m = clearance_m + plan.config.candidate_radius_m
    map_bearing_rad = start.yaw_rad + bearing_rad
    return SurveyCandidate(
        candidate_uid=f"transient_obstacle_{candidate_index:04d}",
        x_m=start.x_m + centre_range_m * math.cos(map_bearing_rad),
        y_m=start.y_m + centre_range_m * math.sin(map_bearing_rad),
        radius_m=plan.config.candidate_radius_m,
        uncertainty_m=plan.config.candidate_uncertainty_m,
        keepout_radius_m=plan.config.candidate_keepout_radius_m,
        confidence=1.0,
        hit_count=1,
        first_seen_sec=0.0,
        last_seen_sec=0.0,
        source_observation_ids=(blockage_id,),
        viewpoint_ids=(),
        status="provisional",
    )


def _merge_transient_candidate(
    candidates: Sequence[SurveyCandidate],
    candidate: SurveyCandidate,
    *,
    merge_distance_m: float,
) -> tuple[tuple[SurveyCandidate, ...], str]:
    for existing in candidates:
        if math.hypot(existing.x_m - candidate.x_m, existing.y_m - candidate.y_m) <= (
            merge_distance_m + _EPSILON_M
        ):
            return tuple(candidates), existing.candidate_uid
    return tuple(candidates) + (candidate,), candidate.candidate_uid


def _combined_registry(
    persistent: StandSurveyRegistry,
    transient: TransientObstacleOverlay,
) -> StandSurveyRegistry:
    persistent_ids = {item.candidate_uid for item in persistent.candidates}
    if any(item.candidate_uid in persistent_ids for item in transient.candidates):
        raise ValueError("transient obstacle ID collides with stand registry")
    return StandSurveyRegistry(
        schema_version=persistent.schema_version,
        survey_id=persistent.survey_id,
        planning_frame=persistent.planning_frame,
        map_bundle_sha256=persistent.map_bundle_sha256,
        candidates=persistent.candidates + transient.candidates,
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


def _escape_keepouts(
    candidates: Sequence[SurveyCandidate],
    *,
    robot_radius_m: float,
    collision_margin_m: float,
    tracking_tube_radius_m: float,
) -> tuple[CircularEscapeKeepout, ...]:
    return tuple(
        CircularEscapeKeepout(
            candidate_uid=candidate.candidate_uid,
            center=Pose2D(candidate.x_m, candidate.y_m, 0.0),
            hard_exclusion_radius_m=_hard_exclusion_radius_m(
                candidate,
                robot_radius_m=robot_radius_m,
                collision_margin_m=collision_margin_m,
                tracking_tube_radius_m=tracking_tube_radius_m,
            ),
            route_keepout_radius_m=candidate.keepout_radius_m,
        )
        for candidate in candidates
    )


def _same_position(a: Pose2D, b: Pose2D) -> bool:
    return math.hypot(a.x_m - b.x_m, a.y_m - b.y_m) <= 1.0e-8


def _simplified_astar_tail(
    result: PlanRouteResult,
    *,
    costmap: Costmap,
) -> tuple[Pose2D, ...]:
    if result.route is None:
        raise ValueError("cannot simplify a failed blockage A* route")
    poses = tuple(point.pose for point in result.route.points)
    if not poses:
        raise ValueError("A* blockage replan returned no route points")
    # Only the A* tail is eligible for smoothing.  The exact stopped pose and
    # its exceptional connector are prepended afterwards and can never be
    # removed by a line-of-sight shortcut.
    return greedy_line_of_sight_shortcut(costmap, poses)


def _rebuild_escape_route(
    result: PlanRouteResult,
    *,
    start: Pose2D,
    connector_anchor: Pose2D,
    simplified_tail: Sequence[Pose2D],
    costmap: Costmap,
) -> PlanRouteResult:
    if result.route is None:
        return result
    if not simplified_tail:
        raise ValueError("simplified blockage A* tail is empty")
    poses = [start, connector_anchor]
    for pose in simplified_tail:
        if not _same_position(poses[-1], pose):
            poses.append(pose)
    if len(poses) < 2:
        raise ValueError("reconstructed blockage route has fewer than two points")
    points = []
    cumulative_m = 0.0
    previous = None
    for index, pose in enumerate(poses):
        segment_m = (
            0.0
            if previous is None
            else math.hypot(pose.x_m - previous.x_m, pose.y_m - previous.y_m)
        )
        cumulative_m += segment_m
        points.append(
            RoutePoint(
                index=index,
                cell=costmap.world_to_grid(pose),
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
        # Preserve the raw A* start.  On a reverse escape this is the farther
        # transition cell, while waypoint 1 remains the separately certified
        # exact-start connector anchor used by the sealer.
        snapped_start=result.route.snapped_start,
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
    forward_translation_heading_limit_rad: float = (
        DEFAULT_FORWARD_TRANSLATION_HEADING_LIMIT_RAD
    ),
    reverse_connector_alignment_tolerance_rad: float = (
        DEFAULT_REVERSE_CONNECTOR_ALIGNMENT_TOLERANCE_RAD
    ),
) -> BlockageRoutePlan:
    if not math.isfinite(robot_radius_m) or robot_radius_m <= 0.0:
        raise ValueError("robot radius must be finite and positive")
    if not math.isfinite(collision_margin_m) or collision_margin_m < 0.0:
        raise ValueError("collision margin must be finite and non-negative")
    if (
        not math.isfinite(tracking_tube_radius_m)
        or tracking_tube_radius_m <= 0.0
    ):
        raise ValueError("tracking tube radius must be finite and positive")
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
    keepouts = _escape_keepouts(
        candidates,
        robot_radius_m=robot_radius_m,
        collision_margin_m=collision_margin_m,
        tracking_tube_radius_m=tracking_tube_radius_m,
    )
    connector_choices = choose_egress_connectors(
        static_costmap,
        planning_costmap,
        start,
        keepouts,
        blocker_candidate_uids=(
            candidate.candidate_uid for candidate in blockers
        ),
        search_radius_m=egress_search_radius_m,
        forward_translation_heading_limit_rad=(
            forward_translation_heading_limit_rad
        ),
        reverse_alignment_tolerance_rad=(
            reverse_connector_alignment_tolerance_rad
        ),
    )
    if not connector_choices:
        raise ValueError(
            "no kinematically executable stand-blockage egress connector"
        )

    replacement: PlanRouteResult | None = None
    geometry: ExecutableEscapeGeometry | None = None
    selected_raw_tail_count = 0
    selected_smoothed_tail_count = 0
    rejection_reasons: list[str] = []
    for connector in connector_choices:
        if connector.mode == EGRESS_MODE_FORWARD:
            planned = plan_route(
                planning_costmap,
                connector.anchor,
                target.pose,
                snap_radius_m=plan.config.snap_radius_m,
            )
            if planned.route is None:
                reason = (
                    planned.failure.reason
                    if planned.failure is not None
                    else "no_path"
                )
                rejection_reasons.append(f"forward_astar:{reason}")
                continue
            try:
                tail = _simplified_astar_tail(
                    planned,
                    costmap=planning_costmap,
                )
                candidate_result = _rebuild_escape_route(
                    planned,
                    start=start,
                    connector_anchor=connector.anchor,
                    simplified_tail=tail,
                    costmap=planning_costmap,
                )
                assert candidate_result.route is not None
                candidate_geometry = validate_executable_escape_route(
                    static_costmap,
                    planning_costmap,
                    start,
                    connector,
                    tuple(
                        point.pose for point in candidate_result.route.points
                    ),
                    keepouts,
                    transition_waypoint_index=1,
                    tracking_tube_radius_m=tracking_tube_radius_m,
                    forward_translation_heading_limit_rad=(
                        forward_translation_heading_limit_rad
                    ),
                    reverse_alignment_tolerance_rad=(
                        reverse_connector_alignment_tolerance_rad
                    ),
                )
            except ValueError as exc:
                rejection_reasons.append(f"forward_geometry:{exc}")
                continue
            replacement = candidate_result
            geometry = candidate_geometry
            selected_raw_tail_count = len(planned.route.points)
            selected_smoothed_tail_count = len(tail)
            break

        transitions = find_reverse_transition_anchors(
            planning_costmap,
            start,
            connector,
            keepouts,
            tracking_tube_radius_m=tracking_tube_radius_m,
            search_radius_m=egress_search_radius_m,
            reverse_alignment_tolerance_rad=(
                reverse_connector_alignment_tolerance_rad
            ),
        )
        if not transitions:
            rejection_reasons.append("reverse_transition:no_straight_anchor")
            continue
        for transition in transitions:
            planned = plan_route(
                planning_costmap,
                transition.anchor,
                target.pose,
                snap_radius_m=plan.config.snap_radius_m,
            )
            if planned.route is None:
                reason = (
                    planned.failure.reason
                    if planned.failure is not None
                    else "no_path"
                )
                rejection_reasons.append(f"reverse_astar:{reason}")
                continue
            try:
                tail = _simplified_astar_tail(
                    planned,
                    costmap=planning_costmap,
                )
                candidate_result = _rebuild_escape_route(
                    planned,
                    start=start,
                    connector_anchor=connector.anchor,
                    simplified_tail=tail,
                    costmap=planning_costmap,
                )
                assert candidate_result.route is not None
                candidate_geometry = validate_executable_escape_route(
                    static_costmap,
                    planning_costmap,
                    start,
                    connector,
                    tuple(
                        point.pose for point in candidate_result.route.points
                    ),
                    keepouts,
                    transition_waypoint_index=2,
                    tracking_tube_radius_m=tracking_tube_radius_m,
                    forward_translation_heading_limit_rad=(
                        forward_translation_heading_limit_rad
                    ),
                    reverse_alignment_tolerance_rad=(
                        reverse_connector_alignment_tolerance_rad
                    ),
                )
            except ValueError as exc:
                rejection_reasons.append(f"reverse_geometry:{exc}")
                continue
            replacement = candidate_result
            geometry = candidate_geometry
            selected_raw_tail_count = len(planned.route.points)
            selected_smoothed_tail_count = len(tail)
            break
        if replacement is not None:
            break

    if replacement is None or geometry is None:
        detail = rejection_reasons[-1] if rejection_reasons else "no_candidate"
        raise ValueError(
            "no executable stand-blockage replacement route: " + detail
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
        egress_anchor=geometry.connector_anchor,
        egress_distance_m=math.hypot(
            geometry.connector_anchor.x_m - start.x_m,
            geometry.connector_anchor.y_m - start.y_m,
        ),
        minimum_egress_hard_clearance_m=(
            geometry.minimum_connector_hard_clearance_m
        ),
        egress_mode=geometry.mode,
        egress_transition_anchor=geometry.transition_anchor,
        egress_transition_waypoint_index=(
            geometry.transition_waypoint_index
        ),
        egress_forward_waypoint_index=geometry.forward_waypoint_index,
        egress_connector_heading_error_rad=(
            geometry.connector_heading_error_rad
        ),
        forward_translation_heading_limit_rad=(
            geometry.forward_translation_heading_limit_rad
        ),
        reverse_connector_alignment_tolerance_rad=(
            geometry.reverse_alignment_tolerance_rad
        ),
        reverse_connector_heading_error_rad=(
            geometry.connector_heading_error_rad
            if geometry.mode != EGRESS_MODE_FORWARD
            else None
        ),
        minimum_transition_keepout_tube_clearance_m=(
            geometry.minimum_transition_keepout_tube_clearance_m
        ),
        tracking_tube_radius_m=tracking_tube_radius_m,
        astar_tail_raw_point_count=selected_raw_tail_count,
        astar_tail_smoothed_point_count=selected_smoothed_tail_count,
    )


def _egress_metadata(route_plan: BlockageRoutePlan) -> dict[str, object]:
    return {
        "egress_anchor": {
            "x_m": route_plan.egress_anchor.x_m,
            "y_m": route_plan.egress_anchor.y_m,
        },
        "egress_distance_m": route_plan.egress_distance_m,
        "minimum_egress_hard_clearance_m": (
            route_plan.minimum_egress_hard_clearance_m
        ),
        "egress_mode": route_plan.egress_mode,
        "egress_transition_anchor": {
            "x_m": route_plan.egress_transition_anchor.x_m,
            "y_m": route_plan.egress_transition_anchor.y_m,
        },
        "egress_transition_waypoint_index": (
            route_plan.egress_transition_waypoint_index
        ),
        "egress_forward_waypoint_index": (
            route_plan.egress_forward_waypoint_index
        ),
        "egress_connector_heading_error_rad": (
            route_plan.egress_connector_heading_error_rad
        ),
        "forward_translation_heading_limit_rad": (
            route_plan.forward_translation_heading_limit_rad
        ),
        "reverse_connector_alignment_tolerance_rad": (
            route_plan.reverse_connector_alignment_tolerance_rad
        ),
        "reverse_connector_heading_error_rad": (
            route_plan.reverse_connector_heading_error_rad
        ),
        "minimum_transition_keepout_tube_clearance_m": (
            route_plan.minimum_transition_keepout_tube_clearance_m
        ),
        "tracking_tube_radius_m": route_plan.tracking_tube_radius_m,
        "raw_astar_path_cell_count": (
            route_plan.route_result.diagnostics.path_cell_count
        ),
        "astar_tail_raw_point_count": route_plan.astar_tail_raw_point_count,
        "astar_tail_smoothed_point_count": (
            route_plan.astar_tail_smoothed_point_count
        ),
        "executable_route_point_count": (
            0
            if route_plan.route_result.route is None
            else len(route_plan.route_result.route.points)
        ),
    }


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
    tracking_tube_radius_m: float = DEFAULT_TRACKING_TUBE_RADIUS_M,
    forward_translation_heading_limit_rad: float = (
        DEFAULT_FORWARD_TRANSLATION_HEADING_LIMIT_RAD
    ),
    reverse_connector_alignment_tolerance_rad: float = (
        DEFAULT_REVERSE_CONNECTOR_ALIGNMENT_TOLERANCE_RAD
    ),
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
        tracking_tube_radius_m=tracking_tube_radius_m,
        forward_translation_heading_limit_rad=(
            forward_translation_heading_limit_rad
        ),
        reverse_connector_alignment_tolerance_rad=(
            reverse_connector_alignment_tolerance_rad
        ),
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
            **_egress_metadata(blockage_plan),
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


def _write_transient_replan_artifacts(
    *,
    plan: CoverageSurveyPlan,
    route_plan: BlockageRoutePlan,
    overlay: TransientObstacleOverlay,
    output_dir: Path,
    source: dict[str, object],
    status: str,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite transient replan: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    route_path = output_dir / "route.csv"
    diagnostics_path = output_dir / "route_diagnostics.json"
    overlay_path = output_dir / "transient_obstacle_overlay.json"
    summary_path = output_dir / "transient_replan_summary.json"
    write_route_csv(
        route_path,
        (route_plan.route_result,),
        final_yaw_by_leg={0: route_plan.target_pose.yaw_rad},
    )
    write_diagnostics_json(
        diagnostics_path,
        (route_plan.route_result,),
        metadata={
            "schema_version": 1,
            "route_kind": "stand_coverage_survey",
            "motion_authorized": False,
            "adaptive_blockage_replan": True,
            "transient_obstacle_overlay": True,
            "semantic_stand_observation": False,
            "survey_id": plan.survey_id,
            "plan_sha256": coverage_survey_plan_sha256(plan),
            "map_bundle_sha256": plan.map_bundle_sha256,
            "target_viewpoint_id": route_plan.target_viewpoint_id,
            "target_pose": {
                "x_m": route_plan.target_pose.x_m,
                "y_m": route_plan.target_pose.y_m,
                "yaw_rad": route_plan.target_pose.yaw_rad,
            },
            "transient_obstacle_count": len(overlay.candidates),
            "blocker_candidate_uids": list(
                route_plan.blocker_candidate_uids
            ),
            **_egress_metadata(route_plan),
            "inflation_radius_m": plan.config.inflation_radius_m,
            "arena_boundary_overlay": True,
            "arena_bounds": plan.arena_bounds.to_metadata(),
        },
    )
    write_transient_obstacle_overlay(
        overlay_path,
        overlay,
        source=source,
    )
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": TRANSIENT_OBSTACLE_OVERLAY_SCHEMA_VERSION,
                "status": status,
                "motion_published": False,
                "semantic_survey_evidence": False,
                "target_viewpoint_id": route_plan.target_viewpoint_id,
                "blocker_candidate_uids": list(
                    route_plan.blocker_candidate_uids
                ),
                "route_csv": str(route_path),
                "diagnostics_json": str(diagnostics_path),
                "transient_obstacle_overlay_json": str(overlay_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {
        "route_csv": str(route_path),
        "diagnostics_json": str(diagnostics_path),
        "summary_json": str(summary_path),
        "transient_obstacle_overlay_json": str(overlay_path),
    }


def record_transient_blockage_replan(
    *,
    survey_root: Path,
    map_yaml: Path,
    semantic_map_id: str,
    target_viewpoint_id: str,
    blockage_id: str,
    stop_pose: Pose2D,
    stop_reason: str,
    stop_details: dict[str, object],
    output_dir: Path,
    robot_radius_m: float,
    existing_overlay_path: Path | None = None,
    tracking_tube_radius_m: float = DEFAULT_TRACKING_TUBE_RADIUS_M,
    forward_translation_heading_limit_rad: float = (
        DEFAULT_FORWARD_TRANSLATION_HEADING_LIMIT_RAD
    ),
    reverse_connector_alignment_tolerance_rad: float = (
        DEFAULT_REVERSE_CONNECTOR_ALIGNMENT_TOLERANCE_RAD
    ),
) -> dict[str, str]:
    """Create a run-local A* keepout from the follower's existing scan sample.

    This function deliberately does not call the stand detector and does not
    write ``stand_registry.json`` or coverage progress.
    """

    if stop_reason not in {
        "stuck no progress",
        "obstacle too close",
        CLEARANCE_LIMITED_MOTION_FLOOR,
    }:
        raise ValueError(
            "only a front-sector blockage stop can create an overlay"
        )
    clearance_m = _front_clearance_m(stop_details)
    bearing_rad = _front_bearing_rad(stop_details)
    survey_root = Path(survey_root)
    plan = load_coverage_survey_plan(survey_root / "coverage_plan.json")
    progress = load_survey_progress(survey_root / "coverage_progress.json", plan)
    if target_viewpoint_id in progress.visited_viewpoint_ids:
        raise ValueError("cannot replan to an already visited viewpoint")
    persistent = load_stand_survey_registry(
        survey_root / "stand_registry.json",
        plan,
    )
    grid, map_bundle = load_occupancy_grid_with_bundle(
        map_yaml,
        semantic_map_id=semantic_map_id,
        planning_frame=plan.planning_frame,
    )
    if map_bundle.bundle_sha256 != plan.map_bundle_sha256:
        raise ValueError("transient blockage map differs from coverage plan")
    previous = (
        ()
        if existing_overlay_path is None
        else load_transient_obstacle_overlay(
            existing_overlay_path,
            plan=plan,
        ).candidates
    )
    proposed = _new_transient_candidate(
        plan=plan,
        start=stop_pose,
        clearance_m=clearance_m,
        bearing_rad=bearing_rad,
        blockage_id=blockage_id,
        candidate_index=len(previous) + 1,
    )
    candidates, blocker_uid = _merge_transient_candidate(
        previous,
        proposed,
        merge_distance_m=plan.config.candidate_merge_distance_m,
    )
    overlay = TransientObstacleOverlay(
        schema_version=TRANSIENT_OBSTACLE_OVERLAY_SCHEMA_VERSION,
        survey_id=plan.survey_id,
        planning_frame=plan.planning_frame,
        map_bundle_sha256=plan.map_bundle_sha256,
        candidates=candidates,
    )
    route_plan = plan_blockage_route_to_viewpoint(
        grid,
        plan=plan,
        registry=_combined_registry(persistent, overlay),
        start=stop_pose,
        target_viewpoint_id=target_viewpoint_id,
        blocker_uids=(blocker_uid,),
        robot_radius_m=robot_radius_m,
        tracking_tube_radius_m=tracking_tube_radius_m,
        forward_translation_heading_limit_rad=(
            forward_translation_heading_limit_rad
        ),
        reverse_connector_alignment_tolerance_rad=(
            reverse_connector_alignment_tolerance_rad
        ),
    )
    return _write_transient_replan_artifacts(
        plan=plan,
        route_plan=route_plan,
        overlay=overlay,
        output_dir=output_dir,
        source={
            "event": "transit_front_blockage",
            "blockage_id": blockage_id,
            "stop_reason": stop_reason,
            "stop_details": stop_details,
            "stop_pose": {
                "x_m": stop_pose.x_m,
                "y_m": stop_pose.y_m,
                "yaw_rad": stop_pose.yaw_rad,
            },
            "front_clearance_m": clearance_m,
            "front_bearing_rad": bearing_rad,
        },
        status="transient_blockage_replan_ready",
    )


def replan_transient_blockage_from_overlay(
    *,
    survey_root: Path,
    map_yaml: Path,
    semantic_map_id: str,
    target_viewpoint_id: str,
    current_pose: Pose2D,
    overlay_path: Path,
    output_dir: Path,
    robot_radius_m: float,
    rejected_run_id: str,
    rejected_stop_details: dict[str, object],
    tracking_tube_radius_m: float = DEFAULT_TRACKING_TUBE_RADIUS_M,
    forward_translation_heading_limit_rad: float = (
        DEFAULT_FORWARD_TRANSLATION_HEADING_LIMIT_RAD
    ),
    reverse_connector_alignment_tolerance_rad: float = (
        DEFAULT_REVERSE_CONNECTOR_ALIGNMENT_TOLERANCE_RAD
    ),
) -> dict[str, str]:
    """Rebuild from fresh AMCL while preserving the dynamic obstacle overlay."""

    survey_root = Path(survey_root)
    plan = load_coverage_survey_plan(survey_root / "coverage_plan.json")
    persistent = load_stand_survey_registry(
        survey_root / "stand_registry.json",
        plan,
    )
    overlay = load_transient_obstacle_overlay(overlay_path, plan=plan)
    grid, map_bundle = load_occupancy_grid_with_bundle(
        map_yaml,
        semantic_map_id=semantic_map_id,
        planning_frame=plan.planning_frame,
    )
    if map_bundle.bundle_sha256 != plan.map_bundle_sha256:
        raise ValueError("transient startup replan map differs from coverage plan")
    # Bind the closest overlay obstacle as the egress blocker. Other transient
    # and persistent obstacles remain active keepouts in the same costmap.
    blocker = min(
        overlay.candidates,
        key=lambda item: _candidate_distance(item, current_pose),
    )
    route_plan = plan_blockage_route_to_viewpoint(
        grid,
        plan=plan,
        registry=_combined_registry(persistent, overlay),
        start=current_pose,
        target_viewpoint_id=target_viewpoint_id,
        blocker_uids=(blocker.candidate_uid,),
        robot_radius_m=robot_radius_m,
        tracking_tube_radius_m=tracking_tube_radius_m,
        forward_translation_heading_limit_rad=(
            forward_translation_heading_limit_rad
        ),
        reverse_connector_alignment_tolerance_rad=(
            reverse_connector_alignment_tolerance_rad
        ),
    )
    return _write_transient_replan_artifacts(
        plan=plan,
        route_plan=route_plan,
        overlay=overlay,
        output_dir=output_dir,
        source={
            "event": "transient_overlay_startup_replan",
            "rejected_run_id": rejected_run_id,
            "rejected_stop_details": rejected_stop_details,
            "fresh_start_pose": {
                "x_m": current_pose.x_m,
                "y_m": current_pose.y_m,
                "yaw_rad": current_pose.yaw_rad,
            },
            "previous_overlay_json": str(overlay_path),
        },
        status="transient_overlay_startup_replan_ready",
    )
