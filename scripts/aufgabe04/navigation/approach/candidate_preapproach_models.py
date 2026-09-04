"""Immutable data contracts for camera candidate pre-approach planning.

These models carry a no-write route preview from computation into artifact
materialization.  Keeping them free of orchestration and filesystem behavior
makes the candidate-selection boundary explicit and testable.
"""

from __future__ import annotations

from dataclasses import dataclass

from scripts.aufgabe04.navigation.approach.candidate_goal_cell_selection import (
    GoalCellSelectionEvidence,
)
from scripts.aufgabe04.navigation.execution.route_context import (
    StationRouteDryRun,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.planning.exact_start_connector import (
    ExactStartConnectorEvidence,
)
from scripts.aufgabe04.navigation.planning.global_planner import PlanRouteResult
from scripts.aufgabe04.navigation.planning.map_io import (
    FrozenMapBundle,
    OccupancyGrid,
)
from scripts.aufgabe04.navigation.planning.route_costmaps import (
    StationRouteCostmaps,
)
from scripts.aufgabe04.navigation.planning.route_smoothing import (
    RouteSmoothingSummary,
)


@dataclass(frozen=True)
class CandidatePlanningContext:
    """Frozen map inputs shared by all candidate previews at one live pose."""

    grid: OccupancyGrid
    map_bundle: FrozenMapBundle
    costmaps: StationRouteCostmaps
    candidate_snapshot_sha256: str
    inflation_radius_m: float
    candidate_transit_radius_m: float
    minimum_active_standoff_m: float
    minimum_candidate_transit_radius_m: float
    minimum_static_inflation_m: float


@dataclass(frozen=True)
class CandidatePreapproachPlan:
    """One fully computed, not-yet-materialized candidate route."""

    candidate_uid: str
    candidate_snapshot_sha256: str
    start: Pose2D
    approach_offset_m: float
    inflation_radius_m: float
    candidate_transit_radius_m: float
    approach_bearing_rad: float
    approach_bearing_mode: str
    dry_run: StationRouteDryRun
    result: PlanRouteResult
    connector: ExactStartConnectorEvidence
    smoothing: RouteSmoothingSummary
    selected_approach_pose: Pose2D
    terminal_yaw_rad: float
    route_length_m: float
    initial_turn_rad: float
    turn_burden_rad: float
    distance_to_stand_m: float
    endpoint_standoff_m: float
    inside_requested_standoff: bool
    minimum_active_standoff_m: float
    minimum_candidate_transit_radius_m: float
    minimum_static_inflation_m: float
    goal_cell_selection: GoalCellSelectionEvidence | None

    @property
    def map_bundle_sha256(self) -> str:
        value = self.dry_run.metadata.get("map_bundle_sha256")
        if not isinstance(value, str) or not value:
            raise RuntimeError("candidate route has no frozen map binding")
        return value


class CandidatePreapproachUnreachableError(ValueError):
    """A candidate-specific route failure that may be compared safely."""

    def __init__(self, candidate_uid: str, reason: str) -> None:
        self.candidate_uid = candidate_uid
        self.reason = reason
        super().__init__(
            f"candidate {candidate_uid!r} pre-approach is unreachable: {reason}"
        )


__all__ = [
    "CandidatePlanningContext",
    "CandidatePreapproachPlan",
    "CandidatePreapproachUnreachableError",
]
