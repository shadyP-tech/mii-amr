"""Pure coverage-survey planning and persistent stand-candidate fusion.

The existing LiDAR confirmation accumulator intentionally operates over a
short observation window.  This module adds the longer-lived state needed for
multi-viewpoint arena coverage without publishing motion or importing ROS.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.approach.candidate_frame_reprojection import (
    CandidateFrameProvenance,
    candidate_frame_provenance_from_mapping,
)
from scripts.aufgabe04.navigation.coverage.candidate_frame_registry import (
    candidate_spatial_match_points,
    frame_provenance_from_confirmed_stand,
    merge_candidate_frame_provenance,
)
from scripts.aufgabe04.navigation.coverage.exact_two_viewpoint_selection import (
    ExactTwoViewpointCandidate,
    select_exact_two_viewpoint_cells,
)
from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.planning.certified_exact_start_route import (
    certify_and_smooth_exact_start_route,
)
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.coverage.coverage_candidate_reporting import (
    coverage_phase_completion_fields,
)
from scripts.aufgabe04.navigation.coverage.stand_candidate_population_retention import (
    RETAINED_STATIC_MAP_DISPOSITIONS,
    STATIC_MAP_DISPOSITION_ADMITTED,
    STATIC_MAP_DISPOSITION_BOUNDARY_PROVISIONAL,
    merge_retained_static_map_dispositions,
    validate_retained_static_map_disposition,
)
from scripts.aufgabe04.navigation.planning.exact_start_connector import (
    ExactStartConnectorEvidence,
)
from scripts.aufgabe04.navigation.planning.global_planner import PlanRouteResult, plan_route
from scripts.aufgabe04.navigation.planning.map_io import OccupancyGrid
from scripts.aufgabe04.navigation.foundation.models import GridCell, Pose2D
from scripts.aufgabe04.navigation.planning.route_smoothing import RouteSmoothingSummary
from scripts.aufgabe04.navigation.planning.spatial_assignment import assign_spatial_points
from scripts.aufgabe04.perception.stand_confirmation import ConfirmedStand
from scripts.aufgabe04.stations.models import Station, StationPose


SURVEY_PLAN_SCHEMA_VERSION = 1
SURVEY_PROGRESS_SCHEMA_VERSION = 1
LEGACY_STAND_SURVEY_REGISTRY_SCHEMA_VERSION = 1
STATIC_MAP_STAND_SURVEY_REGISTRY_SCHEMA_VERSION = 2
FRAME_PROVENANCE_STAND_SURVEY_REGISTRY_SCHEMA_VERSION = 3
STAND_SURVEY_REGISTRY_SCHEMA_VERSION = 4

STATUS_PROVISIONAL = "provisional"
STATUS_PENDING_CAMERA = "pending_camera"
STATUS_CONFIRMED = "confirmed"
STATUS_REJECTED = "rejected"
REJECTION_BASIS_CAMERA = "camera_observation"
REJECTION_BASIS_NEGATIVE_VISIBILITY = "lidar_negative_visibility"
VALID_REJECTION_BASES = frozenset(
    {REJECTION_BASIS_CAMERA, REJECTION_BASIS_NEGATIVE_VISIBILITY}
)
VALID_CANDIDATE_STATUSES = frozenset(
    {
        STATUS_PROVISIONAL,
        STATUS_PENDING_CAMERA,
        STATUS_CONFIRMED,
        STATUS_REJECTED,
    }
)


@dataclass(frozen=True)
class CoverageSurveyConfig:
    """Geometry and completion gates for a deterministic rail survey."""

    lane_count: int = 2
    stop_spacing_m: float = 0.90
    visibility_radius_m: float = 1.35
    inflation_radius_m: float = 0.25
    snap_radius_m: float = 0.30
    minimum_boundary_clearance_m: float = 0.10
    coverage_threshold: float = 0.95
    candidate_merge_distance_m: float = 0.18
    observation_epoch_max_age_sec: float = 8.0
    minimum_candidate_confidence: float = 0.55
    minimum_distinct_viewpoints: int = 2
    minimum_candidate_hits: int = 3
    candidate_radius_m: float = 0.06
    candidate_uncertainty_m: float = 0.02
    candidate_keepout_radius_m: float = 0.31
    expected_stand_count: int | None = None
    exact_inspection_point_count: int | None = None
    exact_two_candidate_spacing_m: float | None = None
    minimum_exact_two_viewpoint_baseline_m: float | None = None

    def validated(self) -> "CoverageSurveyConfig":
        positive = {
            "stop_spacing_m": self.stop_spacing_m,
            "visibility_radius_m": self.visibility_radius_m,
            "snap_radius_m": self.snap_radius_m,
            "candidate_merge_distance_m": self.candidate_merge_distance_m,
            "observation_epoch_max_age_sec": self.observation_epoch_max_age_sec,
            "candidate_radius_m": self.candidate_radius_m,
            "candidate_keepout_radius_m": self.candidate_keepout_radius_m,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        nonnegative = {
            "inflation_radius_m": self.inflation_radius_m,
            "minimum_boundary_clearance_m": self.minimum_boundary_clearance_m,
            "candidate_uncertainty_m": self.candidate_uncertainty_m,
        }
        for name, value in nonnegative.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if type(self.lane_count) is not int or self.lane_count < 1:
            raise ValueError("lane_count must be a positive integer")
        if self.exact_inspection_point_count is not None:
            if (
                type(self.exact_inspection_point_count) is not int
                or self.exact_inspection_point_count != 2
            ):
                raise ValueError(
                    "exact_inspection_point_count must be exactly 2 when set"
                )
            if self.lane_count != 1:
                raise ValueError(
                    "exact_inspection_point_count requires lane_count=1 "
                    "for center-corridor inspection"
                )
            if (self.exact_two_candidate_spacing_m is None) != (
                self.minimum_exact_two_viewpoint_baseline_m is None
            ):
                raise ValueError(
                    "new exact-two geometry fields must either both be set "
                    "or both be absent for a legacy plan"
                )
        elif (
            self.exact_two_candidate_spacing_m is not None
            or self.minimum_exact_two_viewpoint_baseline_m is not None
        ):
            raise ValueError(
                "exact-two geometry fields require "
                "exact_inspection_point_count=2"
            )
        for name, value in (
            ("exact_two_candidate_spacing_m", self.exact_two_candidate_spacing_m),
            (
                "minimum_exact_two_viewpoint_baseline_m",
                self.minimum_exact_two_viewpoint_baseline_m,
            ),
        ):
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if (
            type(self.minimum_distinct_viewpoints) is not int
            or self.minimum_distinct_viewpoints < 1
        ):
            raise ValueError(
                "minimum_distinct_viewpoints must be a positive integer"
            )
        if (
            type(self.minimum_candidate_hits) is not int
            or self.minimum_candidate_hits < 1
        ):
            raise ValueError("minimum_candidate_hits must be a positive integer")
        if not math.isfinite(self.coverage_threshold) or not (
            0.0 < self.coverage_threshold <= 1.0
        ):
            raise ValueError("coverage_threshold must be in (0, 1]")
        if not math.isfinite(self.minimum_candidate_confidence) or not (
            0.0 <= self.minimum_candidate_confidence <= 1.0
        ):
            raise ValueError("minimum_candidate_confidence must be in [0, 1]")
        if self.expected_stand_count is not None and (
            type(self.expected_stand_count) is not int
            or self.expected_stand_count < 0
        ):
            raise ValueError("expected_stand_count must be a non-negative integer")
        if self.candidate_keepout_radius_m + 1.0e-12 < (
            self.candidate_radius_m + self.candidate_uncertainty_m
        ):
            raise ValueError(
                "candidate_keepout_radius_m must cover radius plus uncertainty"
            )
        return self


@dataclass(frozen=True)
class SurveyViewpoint:
    viewpoint_id: str
    pose: Pose2D
    cell: GridCell
    visible_cells: tuple[GridCell, ...]


@dataclass(frozen=True)
class CoverageSurveyPlan:
    schema_version: int
    survey_id: str
    planning_frame: str
    map_bundle_sha256: str
    arena_bounds: ArenaBounds
    config: CoverageSurveyConfig
    viewpoints: tuple[SurveyViewpoint, ...]
    surveyable_cells: tuple[GridCell, ...]
    planned_covered_cells: tuple[GridCell, ...]
    planned_coverage_ratio: float

    @property
    def viewpoint_ids(self) -> tuple[str, ...]:
        return tuple(item.viewpoint_id for item in self.viewpoints)

    def viewpoint_for(self, viewpoint_id: str) -> SurveyViewpoint | None:
        return next(
            (
                item
                for item in self.viewpoints
                if item.viewpoint_id == viewpoint_id
            ),
            None,
        )


@dataclass(frozen=True)
class CoverageSurveyProgress:
    schema_version: int
    survey_id: str
    plan_sha256: str
    visited_viewpoint_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SurveyCandidate:
    candidate_uid: str
    x_m: float
    y_m: float
    radius_m: float
    uncertainty_m: float
    keepout_radius_m: float
    confidence: float
    hit_count: int
    first_seen_sec: float
    last_seen_sec: float
    source_observation_ids: tuple[str, ...]
    viewpoint_ids: tuple[str, ...]
    status: str
    static_map_disposition: str = STATIC_MAP_DISPOSITION_ADMITTED
    frame_provenance: CandidateFrameProvenance | None = None
    rejection_basis: str | None = None


@dataclass(frozen=True)
class StandSurveyRegistry:
    schema_version: int
    survey_id: str
    planning_frame: str
    map_bundle_sha256: str
    candidates: tuple[SurveyCandidate, ...] = ()

    def candidate_for(self, candidate_uid: str) -> SurveyCandidate | None:
        return next(
            (
                candidate
                for candidate in self.candidates
                if candidate.candidate_uid == candidate_uid
            ),
            None,
        )


@dataclass(frozen=True)
class NextSurveyLeg:
    viewpoint: SurveyViewpoint
    route_result: PlanRouteResult
    unreachable_viewpoint_ids: tuple[str, ...]
    exact_start_connector: ExactStartConnectorEvidence
    route_smoothing: RouteSmoothingSummary


def build_coverage_survey_plan(
    occupancy_grid: OccupancyGrid,
    *,
    map_bundle_sha256: str,
    start: Pose2D,
    survey_id: str,
    planning_frame: str = "map",
    arena_bounds: ArenaBounds | None = None,
    config: CoverageSurveyConfig | None = None,
) -> CoverageSurveyPlan:
    """Generate a map-snapped boustrophedon survey with visibility evidence."""

    selected_config = (config or CoverageSurveyConfig()).validated()
    if (
        selected_config.exact_inspection_point_count == 2
        and (
            selected_config.exact_two_candidate_spacing_m is None
            or selected_config.minimum_exact_two_viewpoint_baseline_m is None
        )
    ):
        raise ValueError(
            "legacy lane_count=1 exact-two plans may be loaded or resumed, "
            "but new exact-two planning requires explicit dense candidate "
            "spacing and a minimum world-space viewpoint baseline"
        )
    selected_arena = arena_bounds or ArenaBounds()
    selected_arena.validate()
    _validate_nonempty_id(survey_id, "survey_id")
    _validate_nonempty_id(planning_frame, "planning_frame")
    _validate_sha256(map_bundle_sha256, "map_bundle_sha256")
    _validate_pose(start, "start")

    visibility_costmap = Costmap.from_occupancy_grid(
        occupancy_grid
    ).with_arena_bounds(selected_arena)
    planning_costmap = visibility_costmap.with_inflation(
        selected_config.inflation_radius_m
    )
    sequence_config = selected_config
    if selected_config.exact_inspection_point_count == 2:
        candidate_spacing_m = selected_config.exact_two_candidate_spacing_m
        if candidate_spacing_m is None:  # defensive; rejected above
            raise ValueError("new exact-two planning requires candidate spacing")
        sequence_config = replace(
            selected_config,
            stop_spacing_m=candidate_spacing_m,
        )
    requested_sequences = _requested_boustrophedon_sequences(
        selected_arena,
        sequence_config,
    )
    snapped_sequences = tuple(
        _snap_sequence(planning_costmap, sequence, selected_config.snap_radius_m)
        for sequence in requested_sequences
    )
    snapped_sequences = tuple(sequence for sequence in snapped_sequences if sequence)
    if not snapped_sequences:
        raise ValueError("no traversable coverage viewpoints could be generated")
    cells = min(
        snapped_sequences,
        key=lambda sequence: (
            _cell_distance_m(planning_costmap, start, sequence[0]),
            tuple((cell.x, cell.y) for cell in sequence),
        ),
    )

    surveyable_cells = tuple(
        sorted(
            cell
            for y in range(visibility_costmap.height)
            for x in range(visibility_costmap.width)
            if visibility_costmap.is_traversable(cell := GridCell(x, y))
            and selected_arena.boundary_clearance_m(
                visibility_costmap.grid_to_world(cell)
            )
            + 1.0e-12
            >= selected_config.minimum_boundary_clearance_m
        )
    )
    if not surveyable_cells:
        raise ValueError("coverage survey has no surveyable cells")

    visible_by_cell = {
        cell: _visible_cells(
            visibility_costmap,
            cell,
            surveyable_cells,
            selected_config.visibility_radius_m,
        )
        for cell in cells
    }
    if selected_config.exact_inspection_point_count == 2:
        minimum_viewpoint_baseline_m = (
            selected_config.minimum_exact_two_viewpoint_baseline_m
        )
        if minimum_viewpoint_baseline_m is None:  # defensive; rejected above
            raise ValueError(
                "new exact-two planning requires a minimum viewpoint baseline"
            )
        surveyable_world_xy = {
            cell: (
                visibility_costmap.grid_to_world(cell).x_m,
                visibility_costmap.grid_to_world(cell).y_m,
            )
            for cell in surveyable_cells
        }
        cells = select_exact_two_viewpoint_cells(
            tuple(
                ExactTwoViewpointCandidate(
                    cell=cell,
                    x_m=planning_costmap.grid_to_world(cell).x_m,
                    y_m=planning_costmap.grid_to_world(cell).y_m,
                    visible_cells=visible_by_cell[cell],
                )
                for cell in cells
            ),
            surveyable_world_xy=surveyable_world_xy,
            coverage_threshold=selected_config.coverage_threshold,
            minimum_viewpoint_baseline_m=minimum_viewpoint_baseline_m,
            start_x_m=start.x_m,
            start_y_m=start.y_m,
        )
    viewpoints = []
    for index, cell in enumerate(cells):
        if index + 1 < len(cells):
            next_cell = cells[index + 1]
            yaw_rad = math.atan2(next_cell.y - cell.y, next_cell.x - cell.x)
        elif index > 0:
            previous = cells[index - 1]
            yaw_rad = math.atan2(cell.y - previous.y, cell.x - previous.x)
        else:
            yaw_rad = start.yaw_rad
        viewpoints.append(
            SurveyViewpoint(
                viewpoint_id=f"survey_vp_{index + 1:03d}",
                pose=planning_costmap.grid_to_world(cell, yaw_rad=yaw_rad),
                cell=cell,
                visible_cells=visible_by_cell[cell],
            )
        )
    planned_covered = tuple(
        sorted(
            {
                cell
                for viewpoint in viewpoints
                for cell in viewpoint.visible_cells
            }
        )
    )
    planned_ratio = len(planned_covered) / len(surveyable_cells)
    if planned_ratio + 1.0e-12 < selected_config.coverage_threshold:
        raise ValueError(
            "generated survey does not meet coverage threshold: "
            f"{planned_ratio:.3f} < {selected_config.coverage_threshold:.3f}"
        )
    plan = CoverageSurveyPlan(
        schema_version=SURVEY_PLAN_SCHEMA_VERSION,
        survey_id=survey_id,
        planning_frame=planning_frame,
        map_bundle_sha256=map_bundle_sha256,
        arena_bounds=selected_arena,
        config=selected_config,
        viewpoints=tuple(viewpoints),
        surveyable_cells=surveyable_cells,
        planned_covered_cells=planned_covered,
        planned_coverage_ratio=planned_ratio,
    )
    validate_coverage_survey_plan(plan)
    return plan


def new_survey_progress(plan: CoverageSurveyPlan) -> CoverageSurveyProgress:
    validate_coverage_survey_plan(plan)
    return CoverageSurveyProgress(
        schema_version=SURVEY_PROGRESS_SCHEMA_VERSION,
        survey_id=plan.survey_id,
        plan_sha256=coverage_survey_plan_sha256(plan),
    )


def new_stand_survey_registry(plan: CoverageSurveyPlan) -> StandSurveyRegistry:
    validate_coverage_survey_plan(plan)
    return StandSurveyRegistry(
        schema_version=STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
        survey_id=plan.survey_id,
        planning_frame=plan.planning_frame,
        map_bundle_sha256=plan.map_bundle_sha256,
    )


def mark_viewpoint_visited(
    plan: CoverageSurveyPlan,
    progress: CoverageSurveyProgress,
    viewpoint_id: str,
) -> CoverageSurveyProgress:
    validate_survey_progress(progress, plan)
    if plan.viewpoint_for(viewpoint_id) is None:
        raise ValueError(f"unknown survey viewpoint {viewpoint_id!r}")
    visited = tuple(
        viewpoint.viewpoint_id
        for viewpoint in plan.viewpoints
        if viewpoint.viewpoint_id
        in {*progress.visited_viewpoint_ids, viewpoint_id}
    )
    return replace(progress, visited_viewpoint_ids=visited)


def visited_coverage_ratio(
    plan: CoverageSurveyPlan,
    progress: CoverageSurveyProgress,
) -> float:
    validate_survey_progress(progress, plan)
    visible = {
        cell
        for viewpoint in plan.viewpoints
        if viewpoint.viewpoint_id in progress.visited_viewpoint_ids
        for cell in viewpoint.visible_cells
    }
    return len(visible) / len(plan.surveyable_cells)


def fuse_confirmed_stands(
    registry: StandSurveyRegistry,
    stands: Iterable[ConfirmedStand],
    *,
    viewpoint_id: str,
    config: CoverageSurveyConfig,
    static_map_disposition_by_stand_id: Mapping[str, str] | None = None,
) -> StandSurveyRegistry:
    """Fuse one stopped observation epoch into stable spatial candidate IDs."""

    selected_config = config.validated()
    validate_stand_survey_registry(registry)
    _validate_nonempty_id(viewpoint_id, "viewpoint_id")
    candidates = list(registry.candidates)
    ordered_stands = tuple(
        sorted(stands, key=lambda item: (item.x_m, item.y_m, item.stand_id))
    )
    for stand in ordered_stands:
        _validate_confirmed_stand(stand)
    stand_frame_provenance = tuple(
        frame_provenance_from_confirmed_stand(
            stand,
            expected_map_frame=registry.planning_frame,
            expected_map_bundle_sha256=registry.map_bundle_sha256,
        )
        for stand in ordered_stands
    )
    requested_disposition_by_stand_id = (
        _validated_static_map_disposition_by_stand_id(
            static_map_disposition_by_stand_id,
            stand_ids={stand.stand_id for stand in ordered_stands},
        )
    )

    matches, replayed_stand_indices = _candidate_matches_for_epoch(
        candidates,
        ordered_stands,
        selected_config.candidate_merge_distance_m,
        stand_frame_provenance=stand_frame_provenance,
    )
    for stand_index, stand in enumerate(ordered_stands):
        if stand_index in replayed_stand_indices:
            continue
        incoming_disposition = requested_disposition_by_stand_id.get(
            stand.stand_id,
            STATIC_MAP_DISPOSITION_ADMITTED,
        )
        match_index = matches.get(stand_index)
        incoming_frame_provenance = stand_frame_provenance[stand_index]
        if match_index is None:
            candidate = SurveyCandidate(
                candidate_uid=_next_candidate_uid(candidates),
                x_m=stand.x_m,
                y_m=stand.y_m,
                radius_m=selected_config.candidate_radius_m,
                uncertainty_m=selected_config.candidate_uncertainty_m,
                keepout_radius_m=selected_config.candidate_keepout_radius_m,
                confidence=stand.confidence,
                hit_count=stand.hit_count,
                first_seen_sec=stand.first_seen_sec,
                last_seen_sec=stand.last_seen_sec,
                source_observation_ids=tuple(
                    sorted(set(stand.source_observation_ids))
                ),
                viewpoint_ids=(viewpoint_id,),
                status=STATUS_PROVISIONAL,
                static_map_disposition=incoming_disposition,
                frame_provenance=incoming_frame_provenance,
            )
            candidates.append(
                _advance_candidate_status(candidate, selected_config)
            )
            continue
        candidates[match_index] = _merge_candidate(
            candidates[match_index],
            stand,
            viewpoint_id=viewpoint_id,
            config=selected_config,
            incoming_static_map_disposition=incoming_disposition,
            incoming_frame_provenance=incoming_frame_provenance,
        )
    updated = replace(
        registry,
        candidates=tuple(sorted(candidates, key=lambda item: item.candidate_uid)),
    )
    validate_stand_survey_registry(updated)
    return updated


def decide_candidate(
    registry: StandSurveyRegistry,
    candidate_uid: str,
    *,
    status: str,
) -> StandSurveyRegistry:
    """Persist a stopped camera decision without changing candidate geometry."""

    validate_stand_survey_registry(registry)
    if status not in {STATUS_CONFIRMED, STATUS_REJECTED}:
        raise ValueError("candidate decision must be confirmed or rejected")
    found = False
    candidates = []
    for candidate in registry.candidates:
        if candidate.candidate_uid == candidate_uid:
            found = True
            candidates.append(
                replace(
                    candidate,
                    status=status,
                    rejection_basis=(
                        REJECTION_BASIS_CAMERA
                        if status == STATUS_REJECTED
                        else None
                    ),
                )
            )
        else:
            candidates.append(candidate)
    if not found:
        raise ValueError(f"unknown survey candidate {candidate_uid!r}")
    updated = replace(registry, candidates=tuple(candidates))
    validate_stand_survey_registry(updated)
    return updated


def survey_status(
    plan: CoverageSurveyPlan,
    progress: CoverageSurveyProgress,
    registry: StandSurveyRegistry,
) -> dict[str, object]:
    validate_survey_progress(progress, plan)
    validate_stand_survey_registry(registry, plan)
    ratio = visited_coverage_ratio(plan, progress)
    counts = {
        status: sum(
            1 for candidate in registry.candidates if candidate.status == status
        )
        for status in sorted(VALID_CANDIDATE_STATUSES)
    }
    static_map_disposition_counts = {
        disposition: sum(
            1
            for candidate in registry.candidates
            if candidate.static_map_disposition == disposition
        )
        for disposition in sorted(RETAINED_STATIC_MAP_DISPOSITIONS)
    }
    coverage_complete = ratio + 1.0e-12 >= plan.config.coverage_threshold
    unresolved = counts[STATUS_PROVISIONAL] + counts[STATUS_PENDING_CAMERA]
    expected_count_met = (
        True
        if plan.config.expected_stand_count is None
        else counts[STATUS_CONFIRMED] == plan.config.expected_stand_count
    )
    # An untouched registry has no unresolved candidates only vacuously.  Do
    # not expose that as camera-resolution completion until candidate
    # resolution is meaningful: either LiDAR coverage has finished without
    # finding a candidate, or at least one candidate has entered the registry.
    camera_resolution_evaluable = (
        coverage_complete or bool(registry.candidates)
    )
    camera_resolution_complete = (
        camera_resolution_evaluable and unresolved == 0
    )
    completion_fields = coverage_phase_completion_fields(
        lidar_coverage_complete=coverage_complete,
        camera_candidate_resolution_complete=camera_resolution_complete,
        camera_expected_stand_count_met=expected_count_met,
    )
    return {
        "survey_id": plan.survey_id,
        "visited_viewpoint_count": len(progress.visited_viewpoint_ids),
        "total_viewpoint_count": len(plan.viewpoints),
        "visited_coverage_ratio": ratio,
        "planned_coverage_ratio": plan.planned_coverage_ratio,
        "coverage_threshold": plan.config.coverage_threshold,
        "coverage_complete": coverage_complete,
        # Phase-scoped names keep LiDAR completion distinct from the later
        # camera-confirmation lifecycle.  The two legacy keys below remain for
        # schema-v1 readers, but no new mission summary should rely on them
        # without also reporting these explicit camera fields.
        **completion_fields,
        "candidate_counts": counts,
        "static_map_disposition_counts": static_map_disposition_counts,
        "boundary_provisional_candidate_count": (
            static_map_disposition_counts[
                STATIC_MAP_DISPOSITION_BOUNDARY_PROVISIONAL
            ]
        ),
        "unresolved_candidate_count": unresolved,
        "expected_stand_count": plan.config.expected_stand_count,
        "expected_stand_count_met": expected_count_met,
        "camera_confirmed_stand_count": counts[STATUS_CONFIRMED],
        "camera_candidate_resolution_evaluable": camera_resolution_evaluable,
    }


def plan_next_survey_leg(
    occupancy_grid: OccupancyGrid,
    *,
    plan: CoverageSurveyPlan,
    progress: CoverageSurveyProgress,
    registry: StandSurveyRegistry,
    current_pose: Pose2D,
) -> NextSurveyLeg | None:
    """Replan to the next reachable unvisited viewpoint with live keepouts."""

    validate_survey_progress(progress, plan)
    validate_stand_survey_registry(registry, plan)
    _validate_pose(current_pose, "current_pose")
    base_costmap = Costmap.from_occupancy_grid(
        occupancy_grid
    ).with_arena_bounds(plan.arena_bounds)
    costmap = base_costmap.with_inflation(plan.config.inflation_radius_m)
    keepouts = tuple(
        Station(
            station_id=candidate.candidate_uid,
            pose=StationPose(candidate.x_m, candidate.y_m, 0.0),
            approach_offset_m=0.0,
            keepout_radius_m=candidate.keepout_radius_m,
        )
        for candidate in registry.candidates
        if candidate.status != STATUS_REJECTED
    )
    if keepouts:
        costmap = costmap.with_station_keepouts(keepouts)

    unreachable = []
    for viewpoint in plan.viewpoints:
        if viewpoint.viewpoint_id in progress.visited_viewpoint_ids:
            continue
        result = plan_route(
            costmap,
            current_pose,
            viewpoint.pose,
            snap_radius_m=plan.config.snap_radius_m,
        )
        if result.route is not None:
            route_result, connector, smoothing_summary = (
                certify_and_smooth_exact_start_route(
                    result,
                    base_costmap=base_costmap,
                    planning_costmap=costmap,
                    exact_start=current_pose,
                    required_clearance_m=plan.config.inflation_radius_m,
                )
            )
            return NextSurveyLeg(
                viewpoint=viewpoint,
                route_result=route_result,
                unreachable_viewpoint_ids=tuple(unreachable),
                exact_start_connector=connector,
                route_smoothing=smoothing_summary,
            )
        unreachable.append(viewpoint.viewpoint_id)
    if unreachable:
        raise ValueError(
            "no unvisited survey viewpoint is reachable with current keepouts: "
            + ", ".join(unreachable)
        )
    return None


def validate_coverage_survey_plan(plan: CoverageSurveyPlan) -> None:
    if (
        type(plan.schema_version) is not int
        or plan.schema_version != SURVEY_PLAN_SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported survey plan schema {plan.schema_version!r}")
    _validate_nonempty_id(plan.survey_id, "survey_id")
    _validate_nonempty_id(plan.planning_frame, "planning_frame")
    _validate_sha256(plan.map_bundle_sha256, "map_bundle_sha256")
    plan.arena_bounds.validate()
    plan.config.validated()
    if not plan.viewpoints:
        raise ValueError("survey plan must contain viewpoints")
    if not plan.surveyable_cells:
        raise ValueError("survey plan must contain surveyable cells")
    ids = tuple(item.viewpoint_id for item in plan.viewpoints)
    if len(ids) != len(set(ids)):
        raise ValueError("survey viewpoint IDs must be unique")
    if tuple(sorted(set(plan.surveyable_cells))) != plan.surveyable_cells:
        raise ValueError("surveyable cells must be sorted and unique")
    if tuple(sorted(set(plan.planned_covered_cells))) != (
        plan.planned_covered_cells
    ):
        raise ValueError("planned covered cells must be sorted and unique")
    surveyable = set(plan.surveyable_cells)
    if not set(plan.planned_covered_cells).issubset(surveyable):
        raise ValueError("planned coverage contains a non-surveyable cell")
    expected_ratio = len(plan.planned_covered_cells) / len(plan.surveyable_cells)
    if not math.isclose(
        plan.planned_coverage_ratio,
        expected_ratio,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("planned coverage ratio does not match covered cells")
    if (
        plan.config.exact_inspection_point_count == 2
        and plan.planned_coverage_ratio + 1.0e-12
        < plan.config.coverage_threshold
    ):
        raise ValueError(
            "exact-two survey does not meet coverage threshold: "
            f"{plan.planned_coverage_ratio:.3f} < "
            f"{plan.config.coverage_threshold:.3f}"
        )
    for viewpoint in plan.viewpoints:
        _validate_nonempty_id(viewpoint.viewpoint_id, "viewpoint_id")
        _validate_pose(viewpoint.pose, "viewpoint.pose")
        if tuple(sorted(set(viewpoint.visible_cells))) != viewpoint.visible_cells:
            raise ValueError("viewpoint visible cells must be sorted and unique")
        if not set(viewpoint.visible_cells).issubset(surveyable):
            raise ValueError("viewpoint visibility contains a non-surveyable cell")
    if plan.config.exact_inspection_point_count == 2:
        if len(plan.viewpoints) != 2:
            raise ValueError(
                "exact-two survey plan must contain exactly two viewpoints"
            )
        if plan.viewpoints[0].cell == plan.viewpoints[1].cell:
            raise ValueError(
                "exact-two survey viewpoints must use distinct snapped cells"
            )
        shared_visibility = set(plan.viewpoints[0].visible_cells).intersection(
            plan.viewpoints[1].visible_cells
        )
        if not shared_visibility:
            raise ValueError(
                "exact-two survey viewpoints must have shared visibility"
            )
        minimum_viewpoint_baseline_m = (
            plan.config.minimum_exact_two_viewpoint_baseline_m
        )
        if minimum_viewpoint_baseline_m is not None:
            actual_baseline_m = math.hypot(
                plan.viewpoints[1].pose.x_m - plan.viewpoints[0].pose.x_m,
                plan.viewpoints[1].pose.y_m - plan.viewpoints[0].pose.y_m,
            )
            if (
                actual_baseline_m + 1.0e-12
                < minimum_viewpoint_baseline_m
            ):
                raise ValueError(
                    "exact-two survey viewpoints violate the persisted "
                    "minimum world-space viewpoint baseline: "
                    f"{actual_baseline_m:.3f} < "
                    f"{minimum_viewpoint_baseline_m:.3f} m"
                )


def validate_survey_progress(
    progress: CoverageSurveyProgress,
    plan: CoverageSurveyPlan,
) -> None:
    validate_coverage_survey_plan(plan)
    if (
        type(progress.schema_version) is not int
        or progress.schema_version != SURVEY_PROGRESS_SCHEMA_VERSION
    ):
        raise ValueError(
            f"unsupported survey progress schema {progress.schema_version!r}"
        )
    if progress.survey_id != plan.survey_id:
        raise ValueError("survey progress belongs to another survey")
    if progress.plan_sha256 != coverage_survey_plan_sha256(plan):
        raise ValueError("survey progress references another plan")
    if len(progress.visited_viewpoint_ids) != len(
        set(progress.visited_viewpoint_ids)
    ):
        raise ValueError("visited viewpoint IDs must be unique")
    if not set(progress.visited_viewpoint_ids).issubset(plan.viewpoint_ids):
        raise ValueError("survey progress references an unknown viewpoint")


def validate_stand_survey_registry(
    registry: StandSurveyRegistry,
    plan: CoverageSurveyPlan | None = None,
) -> None:
    if (
        type(registry.schema_version) is not int
        or registry.schema_version != STAND_SURVEY_REGISTRY_SCHEMA_VERSION
    ):
        raise ValueError(
            f"unsupported stand survey registry schema {registry.schema_version!r}"
        )
    _validate_nonempty_id(registry.survey_id, "survey_id")
    _validate_nonempty_id(registry.planning_frame, "planning_frame")
    _validate_sha256(registry.map_bundle_sha256, "map_bundle_sha256")
    if plan is not None and (
        registry.survey_id != plan.survey_id
        or registry.planning_frame != plan.planning_frame
        or registry.map_bundle_sha256 != plan.map_bundle_sha256
    ):
        raise ValueError("stand survey registry provenance differs from plan")
    ids = []
    observation_owners: dict[str, str] = {}
    for candidate in registry.candidates:
        _validate_survey_candidate(candidate)
        ids.append(candidate.candidate_uid)
        for observation_id in candidate.source_observation_ids:
            owner = observation_owners.get(observation_id)
            if owner is not None and owner != candidate.candidate_uid:
                raise ValueError(
                    f"observation {observation_id!r} belongs to multiple candidates"
                )
            observation_owners[observation_id] = candidate.candidate_uid
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("survey candidates must be sorted with unique IDs")


def coverage_survey_plan_payload(
    plan: CoverageSurveyPlan,
    *,
    include_hash: bool = True,
) -> dict[str, object]:
    payload = {
        "schema_version": plan.schema_version,
        "survey_id": plan.survey_id,
        "planning_frame": plan.planning_frame,
        "map_bundle_sha256": plan.map_bundle_sha256,
        "arena_bounds": plan.arena_bounds.to_metadata(),
        "config": _config_payload(plan.config),
        "viewpoints": [_viewpoint_payload(item) for item in plan.viewpoints],
        "surveyable_cells": [_cell_payload(cell) for cell in plan.surveyable_cells],
        "planned_covered_cells": [
            _cell_payload(cell) for cell in plan.planned_covered_cells
        ],
        "planned_coverage_ratio": plan.planned_coverage_ratio,
    }
    if include_hash:
        payload["plan_sha256"] = payload_sha256(payload)
    return payload


def coverage_survey_plan_sha256(plan: CoverageSurveyPlan) -> str:
    validate_coverage_survey_plan(plan)
    return payload_sha256(coverage_survey_plan_payload(plan, include_hash=False))


def write_coverage_survey_plan(path: Path, plan: CoverageSurveyPlan) -> str:
    validate_coverage_survey_plan(plan)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = coverage_survey_plan_payload(plan)
    _write_immutable_json(path, payload)
    return str(payload["plan_sha256"])


def load_coverage_survey_plan(path: Path) -> CoverageSurveyPlan:
    payload = _load_json_object(path)
    supplied_hash = str(payload.pop("plan_sha256", ""))
    if supplied_hash != payload_sha256(payload):
        raise ValueError("survey plan hash mismatch")
    plan = CoverageSurveyPlan(
        schema_version=int(payload["schema_version"]),
        survey_id=str(payload["survey_id"]),
        planning_frame=str(payload["planning_frame"]),
        map_bundle_sha256=str(payload["map_bundle_sha256"]),
        arena_bounds=ArenaBounds(**_float_mapping(payload["arena_bounds"])),
        config=_config_from_payload(payload["config"]),
        viewpoints=tuple(
            _viewpoint_from_payload(item) for item in _list(payload["viewpoints"])
        ),
        surveyable_cells=tuple(
            _cell_from_payload(item) for item in _list(payload["surveyable_cells"])
        ),
        planned_covered_cells=tuple(
            _cell_from_payload(item)
            for item in _list(payload["planned_covered_cells"])
        ),
        planned_coverage_ratio=float(payload["planned_coverage_ratio"]),
    )
    validate_coverage_survey_plan(plan)
    return plan


def write_survey_progress(
    path: Path,
    progress: CoverageSurveyProgress,
    plan: CoverageSurveyPlan,
) -> None:
    validate_survey_progress(progress, plan)
    _write_mutable_json(
        path,
        {
            "schema_version": progress.schema_version,
            "survey_id": progress.survey_id,
            "plan_sha256": progress.plan_sha256,
            "visited_viewpoint_ids": list(progress.visited_viewpoint_ids),
        },
    )


def load_survey_progress(
    path: Path,
    plan: CoverageSurveyPlan,
) -> CoverageSurveyProgress:
    payload = _load_json_object(path)
    progress = CoverageSurveyProgress(
        schema_version=int(payload["schema_version"]),
        survey_id=str(payload["survey_id"]),
        plan_sha256=str(payload["plan_sha256"]),
        visited_viewpoint_ids=tuple(
            str(value) for value in _list(payload["visited_viewpoint_ids"])
        ),
    )
    validate_survey_progress(progress, plan)
    return progress


def write_stand_survey_registry(
    path: Path,
    registry: StandSurveyRegistry,
    plan: CoverageSurveyPlan | None = None,
) -> None:
    validate_stand_survey_registry(registry, plan)
    _write_mutable_json(path, stand_survey_registry_payload(registry))


def stand_survey_registry_payload(
    registry: StandSurveyRegistry,
) -> dict[str, object]:
    """Return the canonical full registry payload used by every hash edge."""

    validate_stand_survey_registry(registry)
    return {
        "schema_version": registry.schema_version,
        "survey_id": registry.survey_id,
        "planning_frame": registry.planning_frame,
        "map_bundle_sha256": registry.map_bundle_sha256,
        "candidates": [
            survey_candidate_payload(candidate)
            for candidate in registry.candidates
        ],
    }


def stand_survey_registry_sha256(registry: StandSurveyRegistry) -> str:
    """Hash canonical geometry, disposition, and observation-frame provenance."""

    return payload_sha256(stand_survey_registry_payload(registry))


def load_stand_survey_registry(
    path: Path,
    plan: CoverageSurveyPlan | None = None,
) -> StandSurveyRegistry:
    payload = _load_json_object(path)
    source_schema_version = int(payload["schema_version"])
    if source_schema_version not in {
        LEGACY_STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
        STATIC_MAP_STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
        FRAME_PROVENANCE_STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
        STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
    }:
        raise ValueError(
            "unsupported stand survey registry schema "
            f"{source_schema_version!r}"
        )
    registry = StandSurveyRegistry(
        # Schema v1 predates boundary retention, v1/v2 predate frozen
        # observation-frame provenance, and v1-v3 predate explicit rejection
        # provenance. Upgrade those lineages before any canonical v4 hash.
        schema_version=STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
        survey_id=str(payload["survey_id"]),
        planning_frame=str(payload["planning_frame"]),
        map_bundle_sha256=str(payload["map_bundle_sha256"]),
        candidates=tuple(
            _candidate_from_payload(
                item,
                source_registry_schema_version=source_schema_version,
            )
            for item in _list(payload.get("candidates", []))
        ),
    )
    validate_stand_survey_registry(registry, plan)
    return registry


def _requested_boustrophedon_sequences(
    arena: ArenaBounds,
    config: CoverageSurveyConfig,
) -> tuple[tuple[Pose2D, ...], ...]:
    edge_clearance = (
        arena.margin_m + config.inflation_radius_m + config.minimum_boundary_clearance_m
    )
    usable_length = arena.length_m - 2.0 * edge_clearance
    usable_width = arena.width_m - 2.0 * edge_clearance
    if usable_length <= 0.0 or usable_width <= 0.0:
        raise ValueError("inflation and margins leave no survey corridor")
    stop_count = max(2, int(math.ceil(usable_length / config.stop_spacing_m)) + 1)
    x_values = tuple(
        -usable_length / 2.0 + usable_length * index / (stop_count - 1)
        for index in range(stop_count)
    )
    lane_values = tuple(
        -usable_width / 2.0
        + usable_width * (index + 0.5) / config.lane_count
        for index in range(config.lane_count)
    )

    def sequence(
        lane_order: tuple[float, ...],
        *,
        reverse_first: bool,
    ) -> tuple[Pose2D, ...]:
        requested = []
        for lane_index, local_y in enumerate(lane_order):
            reverse = reverse_first if lane_index % 2 == 0 else not reverse_first
            lane_x = tuple(reversed(x_values)) if reverse else x_values
            requested.extend(
                _arena_local_pose(arena, local_x, local_y)
                for local_x in lane_x
            )
        return tuple(requested)

    forward_lanes = lane_values
    reverse_lanes = tuple(reversed(lane_values))
    return (
        sequence(forward_lanes, reverse_first=False),
        sequence(forward_lanes, reverse_first=True),
        sequence(reverse_lanes, reverse_first=False),
        sequence(reverse_lanes, reverse_first=True),
    )


def _arena_local_pose(arena: ArenaBounds, local_x: float, local_y: float) -> Pose2D:
    yaw = math.radians(arena.yaw_deg)
    return Pose2D(
        arena.center_x_m + math.cos(yaw) * local_x - math.sin(yaw) * local_y,
        arena.center_y_m + math.sin(yaw) * local_x + math.cos(yaw) * local_y,
        yaw,
    )


def _snap_sequence(
    costmap: Costmap,
    requested: Iterable[Pose2D],
    snap_radius_m: float,
) -> tuple[GridCell, ...]:
    cells = []
    for pose in requested:
        snapped = _nearest_traversable_cell(
            costmap,
            costmap.world_to_grid(pose),
            snap_radius_m,
        )
        if snapped is not None and (not cells or cells[-1] != snapped):
            cells.append(snapped)
    return tuple(cells)


def _nearest_traversable_cell(
    costmap: Costmap,
    requested: GridCell,
    radius_m: float,
) -> GridCell | None:
    radius_cells = int(math.ceil(radius_m / costmap.resolution))
    options = []
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy > radius_cells * radius_cells:
                continue
            cell = GridCell(requested.x + dx, requested.y + dy)
            if costmap.is_traversable(cell):
                options.append((dx * dx + dy * dy, cell))
    return min(options)[1] if options else None


def _cell_distance_m(costmap: Costmap, pose: Pose2D, cell: GridCell) -> float:
    world = costmap.grid_to_world(cell)
    return math.hypot(pose.x_m - world.x_m, pose.y_m - world.y_m)


def _visible_cells(
    costmap: Costmap,
    origin: GridCell,
    surveyable_cells: tuple[GridCell, ...],
    radius_m: float,
) -> tuple[GridCell, ...]:
    radius_cells = radius_m / costmap.resolution
    visible = []
    for target in surveyable_cells:
        if math.hypot(target.x - origin.x, target.y - origin.y) > (
            radius_cells + 1.0e-12
        ):
            continue
        line = _bresenham_cells(origin, target)
        if any(costmap.is_blocked(cell) for cell in line[1:-1]):
            continue
        visible.append(target)
    return tuple(sorted(visible))


def _bresenham_cells(start: GridCell, end: GridCell) -> tuple[GridCell, ...]:
    x0, y0 = start.x, start.y
    x1, y1 = end.x, end.y
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    cells = []
    while True:
        cells.append(GridCell(x0, y0))
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy
    return tuple(cells)


def _candidate_matches_for_epoch(
    candidates: list[SurveyCandidate],
    stands: tuple[ConfirmedStand, ...],
    match_radius_m: float,
    *,
    stand_frame_provenance: tuple[CandidateFrameProvenance | None, ...],
) -> tuple[dict[int, int], frozenset[int]]:
    """Match a stopped epoch without order-dependent nearest-neighbor claims.

    Previously consumed observation IDs retain their candidate identity even
    if the candidate centroid has since moved. A complete replay is a no-op
    and does not consume a slot in the new epoch's one-to-one assignment.
    """

    observation_owners = {
        observation_id: candidate_index
        for candidate_index, candidate in enumerate(candidates)
        for observation_id in candidate.source_observation_ids
    }
    matches: dict[int, int] = {}
    replayed_stand_indices: set[int] = set()
    identity_claimed_candidates: set[int] = set()
    for stand_index, stand in enumerate(stands):
        owners = {
            observation_owners[observation_id]
            for observation_id in stand.source_observation_ids
            if observation_id in observation_owners
        }
        if not owners:
            continue
        if len(owners) != 1:
            raise ValueError(
                "confirmed stand observations belong to multiple survey candidates"
            )
        owner = next(iter(owners))
        new_observation_ids = set(stand.source_observation_ids).difference(
            observation_owners
        )
        if not new_observation_ids:
            replayed_stand_indices.add(stand_index)
            continue
        if owner in identity_claimed_candidates:
            raise ValueError(
                "multiple incoming stands claim the same survey candidate identity"
            )
        matches[stand_index] = owner
        identity_claimed_candidates.add(owner)

    available_candidate_indices = tuple(
        index
        for index in range(len(candidates))
        if index not in identity_claimed_candidates
    )
    available_stand_indices = tuple(
        index
        for index in range(len(stands))
        if index not in matches and index not in replayed_stand_indices
    )
    candidate_points, stand_points = candidate_spatial_match_points(
        tuple(
            (candidates[index].x_m, candidates[index].y_m)
            for index in available_candidate_indices
        ),
        tuple(
            (stands[index].x_m, stands[index].y_m)
            for index in available_stand_indices
        ),
        candidate_frames=tuple(
            candidates[index].frame_provenance
            for index in available_candidate_indices
        ),
        stand_frames=tuple(
            stand_frame_provenance[index]
            for index in available_stand_indices
        ),
    )
    spatial_matches = assign_spatial_points(
        candidate_points,
        stand_points,
        maximum_distance_m=match_radius_m,
    )
    for assignment in spatial_matches:
        matches[available_stand_indices[assignment.right_index]] = (
            available_candidate_indices[assignment.left_index]
        )
    return matches, frozenset(replayed_stand_indices)


def _merge_candidate(
    candidate: SurveyCandidate,
    stand: ConfirmedStand,
    *,
    viewpoint_id: str,
    config: CoverageSurveyConfig,
    incoming_static_map_disposition: str = STATIC_MAP_DISPOSITION_ADMITTED,
    incoming_frame_provenance: CandidateFrameProvenance | None = None,
) -> SurveyCandidate:
    existing_observations = set(candidate.source_observation_ids)
    new_observation_ids = tuple(
        item
        for item in stand.source_observation_ids
        if item not in existing_observations
    )
    if not new_observation_ids:
        return candidate
    incoming_weight = max(stand.hit_count, len(new_observation_ids))
    total_hits = candidate.hit_count + incoming_weight
    merged_frame_provenance = merge_candidate_frame_provenance(
        candidate.frame_provenance,
        incoming_frame_provenance,
        existing_weight=candidate.hit_count,
        incoming_weight=incoming_weight,
    )
    if merged_frame_provenance is None:
        merged_x_m = (
            candidate.x_m * candidate.hit_count + stand.x_m * incoming_weight
        ) / total_hits
        merged_y_m = (
            candidate.y_m * candidate.hit_count + stand.y_m * incoming_weight
        ) / total_hits
    else:
        current_reference = merged_frame_provenance.frozen_map_point
        if current_reference is None:
            raise ValueError(
                "merged candidate frame provenance lacks its map reference"
            )
        merged_x_m = current_reference.x_m
        merged_y_m = current_reference.y_m
    merged = replace(
        candidate,
        x_m=merged_x_m,
        y_m=merged_y_m,
        confidence=(
            candidate.confidence * candidate.hit_count
            + stand.confidence * incoming_weight
        )
        / total_hits,
        hit_count=total_hits,
        first_seen_sec=min(candidate.first_seen_sec, stand.first_seen_sec),
        last_seen_sec=max(candidate.last_seen_sec, stand.last_seen_sec),
        source_observation_ids=tuple(
            sorted({*candidate.source_observation_ids, *new_observation_ids})
        ),
        viewpoint_ids=tuple(sorted({*candidate.viewpoint_ids, viewpoint_id})),
        static_map_disposition=merge_retained_static_map_dispositions(
            candidate.static_map_disposition,
            incoming_static_map_disposition,
        ),
        frame_provenance=merged_frame_provenance,
    )
    return _advance_candidate_status(merged, config)


def _advance_candidate_status(
    candidate: SurveyCandidate,
    config: CoverageSurveyConfig,
) -> SurveyCandidate:
    if candidate.status in {STATUS_CONFIRMED, STATUS_REJECTED}:
        return candidate
    ready = (
        candidate.hit_count >= config.minimum_candidate_hits
        and len(candidate.viewpoint_ids) >= config.minimum_distinct_viewpoints
    )
    return replace(
        candidate,
        status=STATUS_PENDING_CAMERA if ready else STATUS_PROVISIONAL,
    )


def _validated_static_map_disposition_by_stand_id(
    disposition_by_stand_id: Mapping[str, str] | None,
    *,
    stand_ids: set[str],
) -> dict[str, str]:
    if disposition_by_stand_id is None:
        return {}
    if not isinstance(disposition_by_stand_id, Mapping):
        raise ValueError("static-map disposition map must be a mapping")
    result: dict[str, str] = {}
    for stand_id, disposition in disposition_by_stand_id.items():
        parsed_id = str(stand_id)
        _validate_nonempty_id(parsed_id, "stand_id")
        result[parsed_id] = validate_retained_static_map_disposition(
            str(disposition)
        )
    supplied_ids = set(result)
    if supplied_ids != stand_ids:
        raise ValueError(
            "static-map disposition map must cover exactly the fused stands: "
            f"missing={sorted(stand_ids - supplied_ids)}, "
            f"unknown={sorted(supplied_ids - stand_ids)}"
        )
    return result


def _next_candidate_uid(candidates: Iterable[SurveyCandidate]) -> str:
    used = {candidate.candidate_uid for candidate in candidates}
    index = 1
    while f"survey_candidate_{index:04d}" in used:
        index += 1
    return f"survey_candidate_{index:04d}"


def _validate_survey_candidate(candidate: SurveyCandidate) -> None:
    _validate_nonempty_id(candidate.candidate_uid, "candidate_uid")
    for name, value in {
        "x_m": candidate.x_m,
        "y_m": candidate.y_m,
        "radius_m": candidate.radius_m,
        "uncertainty_m": candidate.uncertainty_m,
        "keepout_radius_m": candidate.keepout_radius_m,
        "confidence": candidate.confidence,
        "first_seen_sec": candidate.first_seen_sec,
        "last_seen_sec": candidate.last_seen_sec,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"candidate {name} must be finite")
    if candidate.radius_m <= 0.0 or candidate.keepout_radius_m <= 0.0:
        raise ValueError("candidate radius and keepout must be positive")
    if candidate.uncertainty_m < 0.0:
        raise ValueError("candidate uncertainty must be non-negative")
    if not 0.0 <= candidate.confidence <= 1.0:
        raise ValueError("candidate confidence must be in [0, 1]")
    if type(candidate.hit_count) is not int or candidate.hit_count < 1:
        raise ValueError("candidate hit_count must be a positive integer")
    if candidate.last_seen_sec < candidate.first_seen_sec:
        raise ValueError("candidate last_seen_sec precedes first_seen_sec")
    if candidate.status not in VALID_CANDIDATE_STATUSES:
        raise ValueError(f"invalid candidate status {candidate.status!r}")
    if (
        candidate.rejection_basis is not None
        and candidate.rejection_basis not in VALID_REJECTION_BASES
    ):
        raise ValueError(
            f"invalid candidate rejection basis {candidate.rejection_basis!r}"
        )
    if (
        candidate.rejection_basis is not None
        and candidate.status != STATUS_REJECTED
    ):
        raise ValueError(
            "candidate rejection basis requires rejected lifecycle status"
        )
    if candidate.status == STATUS_REJECTED and candidate.rejection_basis is None:
        raise ValueError("rejected candidate must record its rejection basis")
    validate_retained_static_map_disposition(
        candidate.static_map_disposition
    )
    if candidate.frame_provenance is not None:
        if not isinstance(candidate.frame_provenance, CandidateFrameProvenance):
            raise ValueError(
                "candidate frame_provenance must be CandidateFrameProvenance"
            )
        # Reconstructing through the strict mapping path validates both the
        # canonical odom point and any retained frozen transform reference.
        candidate_frame_provenance_from_mapping(
            candidate.frame_provenance.to_mapping()
        )
    if not candidate.source_observation_ids:
        raise ValueError("candidate must retain source observations")
    if not candidate.viewpoint_ids:
        raise ValueError("candidate must retain source viewpoints")
    if len(candidate.source_observation_ids) != len(
        set(candidate.source_observation_ids)
    ):
        raise ValueError("candidate observation IDs must be unique")
    if len(candidate.viewpoint_ids) != len(set(candidate.viewpoint_ids)):
        raise ValueError("candidate viewpoint IDs must be unique")


def _validate_confirmed_stand(stand: ConfirmedStand) -> None:
    for name, value in {
        "x_m": stand.x_m,
        "y_m": stand.y_m,
        "confidence": stand.confidence,
        "first_seen_sec": stand.first_seen_sec,
        "last_seen_sec": stand.last_seen_sec,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"confirmed stand {name} must be finite")
    if type(stand.hit_count) is not int or stand.hit_count < 1:
        raise ValueError("confirmed stand hit_count must be positive")
    if not stand.source_observation_ids:
        raise ValueError("confirmed stand must retain observation IDs")


def _validate_pose(pose: Pose2D, name: str) -> None:
    if not all(math.isfinite(value) for value in (pose.x_m, pose.y_m, pose.yaw_rad)):
        raise ValueError(f"{name} must be finite")


def _validate_nonempty_id(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _config_payload(config: CoverageSurveyConfig) -> dict[str, object]:
    payload: dict[str, object] = {
        "lane_count": config.lane_count,
        "stop_spacing_m": config.stop_spacing_m,
        "visibility_radius_m": config.visibility_radius_m,
        "inflation_radius_m": config.inflation_radius_m,
        "snap_radius_m": config.snap_radius_m,
        "minimum_boundary_clearance_m": config.minimum_boundary_clearance_m,
        "coverage_threshold": config.coverage_threshold,
        "candidate_merge_distance_m": config.candidate_merge_distance_m,
        "observation_epoch_max_age_sec": config.observation_epoch_max_age_sec,
        "minimum_candidate_confidence": config.minimum_candidate_confidence,
        "minimum_distinct_viewpoints": config.minimum_distinct_viewpoints,
        "minimum_candidate_hits": config.minimum_candidate_hits,
        "candidate_radius_m": config.candidate_radius_m,
        "candidate_uncertainty_m": config.candidate_uncertainty_m,
        "candidate_keepout_radius_m": config.candidate_keepout_radius_m,
        "expected_stand_count": config.expected_stand_count,
    }
    if config.exact_inspection_point_count is not None:
        payload["exact_inspection_point_count"] = (
            config.exact_inspection_point_count
        )
    if config.exact_two_candidate_spacing_m is not None:
        payload["exact_two_candidate_spacing_m"] = (
            config.exact_two_candidate_spacing_m
        )
    if config.minimum_exact_two_viewpoint_baseline_m is not None:
        payload["minimum_exact_two_viewpoint_baseline_m"] = (
            config.minimum_exact_two_viewpoint_baseline_m
        )
    return payload


def _config_from_payload(payload: object) -> CoverageSurveyConfig:
    item = _mapping(payload)
    return CoverageSurveyConfig(
        lane_count=int(item["lane_count"]),
        stop_spacing_m=float(item["stop_spacing_m"]),
        exact_inspection_point_count=item.get("exact_inspection_point_count"),
        exact_two_candidate_spacing_m=item.get("exact_two_candidate_spacing_m"),
        minimum_exact_two_viewpoint_baseline_m=item.get(
            "minimum_exact_two_viewpoint_baseline_m"
        ),
        visibility_radius_m=float(item["visibility_radius_m"]),
        inflation_radius_m=float(item["inflation_radius_m"]),
        snap_radius_m=float(item["snap_radius_m"]),
        minimum_boundary_clearance_m=float(
            item["minimum_boundary_clearance_m"]
        ),
        coverage_threshold=float(item["coverage_threshold"]),
        candidate_merge_distance_m=float(item["candidate_merge_distance_m"]),
        observation_epoch_max_age_sec=float(
            item["observation_epoch_max_age_sec"]
        ),
        minimum_candidate_confidence=float(
            item["minimum_candidate_confidence"]
        ),
        minimum_distinct_viewpoints=int(item["minimum_distinct_viewpoints"]),
        minimum_candidate_hits=int(item["minimum_candidate_hits"]),
        candidate_radius_m=float(item["candidate_radius_m"]),
        candidate_uncertainty_m=float(item["candidate_uncertainty_m"]),
        candidate_keepout_radius_m=float(item["candidate_keepout_radius_m"]),
        expected_stand_count=(
            None
            if item.get("expected_stand_count") is None
            else int(item["expected_stand_count"])
        ),
    ).validated()


def _cell_payload(cell: GridCell) -> list[int]:
    return [cell.x, cell.y]


def _cell_from_payload(payload: object) -> GridCell:
    values = _list(payload)
    if len(values) != 2:
        raise ValueError("grid cell payload must contain x and y")
    return GridCell(int(values[0]), int(values[1]))


def _viewpoint_payload(viewpoint: SurveyViewpoint) -> dict[str, object]:
    return {
        "viewpoint_id": viewpoint.viewpoint_id,
        "pose": {
            "x_m": viewpoint.pose.x_m,
            "y_m": viewpoint.pose.y_m,
            "yaw_rad": viewpoint.pose.yaw_rad,
        },
        "cell": _cell_payload(viewpoint.cell),
        "visible_cells": [
            _cell_payload(cell) for cell in viewpoint.visible_cells
        ],
    }


def _viewpoint_from_payload(payload: object) -> SurveyViewpoint:
    item = _mapping(payload)
    pose = _mapping(item["pose"])
    return SurveyViewpoint(
        viewpoint_id=str(item["viewpoint_id"]),
        pose=Pose2D(
            float(pose["x_m"]),
            float(pose["y_m"]),
            float(pose["yaw_rad"]),
        ),
        cell=_cell_from_payload(item["cell"]),
        visible_cells=tuple(
            _cell_from_payload(value)
            for value in _list(item["visible_cells"])
        ),
    )


def survey_candidate_payload(candidate: SurveyCandidate) -> dict[str, object]:
    """Return the canonical registry/hash payload for one candidate."""

    _validate_survey_candidate(candidate)
    return {
        "candidate_uid": candidate.candidate_uid,
        "x_m": candidate.x_m,
        "y_m": candidate.y_m,
        "radius_m": candidate.radius_m,
        "uncertainty_m": candidate.uncertainty_m,
        "keepout_radius_m": candidate.keepout_radius_m,
        "confidence": candidate.confidence,
        "hit_count": candidate.hit_count,
        "first_seen_sec": candidate.first_seen_sec,
        "last_seen_sec": candidate.last_seen_sec,
        "source_observation_ids": list(candidate.source_observation_ids),
        "viewpoint_ids": list(candidate.viewpoint_ids),
        "status": candidate.status,
        "static_map_disposition": candidate.static_map_disposition,
        "frame_provenance": (
            None
            if candidate.frame_provenance is None
            else candidate.frame_provenance.to_mapping()
        ),
        "rejection_basis": candidate.rejection_basis,
    }


def _candidate_from_payload(
    payload: object,
    *,
    source_registry_schema_version: int,
) -> SurveyCandidate:
    item = _mapping(payload)
    status = str(item["status"])
    if source_registry_schema_version == LEGACY_STAND_SURVEY_REGISTRY_SCHEMA_VERSION:
        if "static_map_disposition" in item:
            raise ValueError(
                "stand survey registry schema 1 candidate unexpectedly "
                "contains static_map_disposition"
            )
        static_map_disposition = STATIC_MAP_DISPOSITION_ADMITTED
        frame_provenance = None
        rejection_basis = (
            REJECTION_BASIS_CAMERA if status == STATUS_REJECTED else None
        )
    elif source_registry_schema_version == STATIC_MAP_STAND_SURVEY_REGISTRY_SCHEMA_VERSION:
        if "static_map_disposition" not in item:
            raise ValueError(
                "stand survey registry schema 2 candidate is missing "
                "static_map_disposition"
            )
        static_map_disposition = str(item["static_map_disposition"])
        if "frame_provenance" in item:
            raise ValueError(
                "stand survey registry schema 2 candidate unexpectedly "
                "contains frame_provenance"
            )
        frame_provenance = None
        rejection_basis = (
            REJECTION_BASIS_CAMERA if status == STATUS_REJECTED else None
        )
    elif source_registry_schema_version in {
        FRAME_PROVENANCE_STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
        STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
    }:
        if "static_map_disposition" not in item:
            raise ValueError(
                "stand survey registry candidate is missing "
                "static_map_disposition"
            )
        if "frame_provenance" not in item:
            raise ValueError(
                "stand survey registry candidate is missing "
                "frame_provenance"
            )
        static_map_disposition = str(item["static_map_disposition"])
        raw_frame_provenance = item["frame_provenance"]
        if raw_frame_provenance is None:
            frame_provenance = None
        elif isinstance(raw_frame_provenance, Mapping):
            frame_provenance = candidate_frame_provenance_from_mapping(
                raw_frame_provenance
            )
        else:
            raise ValueError("candidate frame_provenance must be an object or null")
        if (
            source_registry_schema_version
            == FRAME_PROVENANCE_STAND_SURVEY_REGISTRY_SCHEMA_VERSION
        ):
            if "rejection_basis" in item:
                raise ValueError(
                    "stand survey registry schema 3 candidate unexpectedly "
                    "contains rejection_basis"
                )
            rejection_basis = (
                REJECTION_BASIS_CAMERA if status == STATUS_REJECTED else None
            )
        else:
            if "rejection_basis" not in item:
                raise ValueError(
                    "stand survey registry schema 4 candidate is missing "
                    "rejection_basis"
                )
            raw_rejection_basis = item["rejection_basis"]
            rejection_basis = (
                None
                if raw_rejection_basis is None
                else str(raw_rejection_basis)
            )
    else:
        raise ValueError(
            "unsupported stand survey registry schema "
            f"{source_registry_schema_version!r}"
        )
    return SurveyCandidate(
        candidate_uid=str(item["candidate_uid"]),
        x_m=float(item["x_m"]),
        y_m=float(item["y_m"]),
        radius_m=float(item["radius_m"]),
        uncertainty_m=float(item["uncertainty_m"]),
        keepout_radius_m=float(item["keepout_radius_m"]),
        confidence=float(item["confidence"]),
        hit_count=int(item["hit_count"]),
        first_seen_sec=float(item["first_seen_sec"]),
        last_seen_sec=float(item["last_seen_sec"]),
        source_observation_ids=tuple(
            str(value) for value in _list(item["source_observation_ids"])
        ),
        viewpoint_ids=tuple(
            str(value) for value in _list(item["viewpoint_ids"])
        ),
        status=status,
        static_map_disposition=static_map_disposition,
        frame_provenance=frame_provenance,
        rejection_basis=rejection_basis,
    )


def _mapping(payload: object) -> Mapping[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    return payload


def _float_mapping(payload: object) -> dict[str, float]:
    return {str(key): float(value) for key, value in _mapping(payload).items()}


def _list(payload: object) -> list[object]:
    if not isinstance(payload, list):
        raise ValueError("expected a JSON list")
    return payload


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load survey JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"survey JSON {path} must contain an object")
    return payload


def _write_immutable_json(path: Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"refusing to overwrite immutable survey artifact: {path}")
        return
    path.write_text(text)


def _write_mutable_json(path: Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
