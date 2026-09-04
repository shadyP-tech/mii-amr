"""Safety-ranked rasterization for a continuous candidate approach target.

Candidate geometry produces a continuous world-space target, while A* must
terminate at a grid-cell centre.  A target close to a cell boundary can move
to a different centre after a very small frame projection, even though an
adjacent centre represents the same target within the map's ordinary
half-cell-diagonal quantization envelope.  This module compares only those
equivalent centres, preserves an already-satisfied no-motion route, and
otherwise prefers materially greater continuous static-map clearance.

The policy is advisory planning, not motion admission.  Every option still
uses the existing inflated costmap, exact-start connector, candidate-clearance
callback, and downstream route-uncertainty preflight.  This module is ROS-free
and never writes artifacts or authorizes motion.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

from scripts.aufgabe04.navigation.approach.dynamic_approach_planner import (
    point_clearance_to_blocked_m,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_defaults import (
    DEFAULT_UNCERTAINTY_CLEARANCE_SAMPLE_SPACING_M,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_sampling import (
    sample_route_for_uncertainty_admission,
)
from scripts.aufgabe04.navigation.foundation.models import GridCell, Pose2D, Route
from scripts.aufgabe04.navigation.planning.certified_exact_start_route import (
    certify_and_smooth_exact_start_route,
)
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.exact_start_connector import (
    ExactStartConnectorEvidence,
)
from scripts.aufgabe04.navigation.planning.global_planner import (
    PlanRouteResult,
    plan_route,
)
from scripts.aufgabe04.navigation.planning.route_smoothing import (
    RouteSmoothingSummary,
)


GOAL_CELL_SELECTION_POLICY = (
    "no-motion-preserving-static-clearance-within-rasterization-envelope"
)
# Clearance is derived from continuous geometry and can vary at floating-point
# scale across equivalent formulations.  Treat improvements up to one
# millimetre as a tie; this is two orders of magnitude smaller than the 5 cm
# wall-clearance improvement that motivated the policy.
GOAL_CELL_CLEARANCE_RANKING_TOLERANCE_M = 0.001
GOAL_CELL_RANKING_RATIONALE = (
    "accepted_no_motion_route_first",
    "maximum_route_raw_clearance_lower_bound_m_outside_tolerance",
    "minimum_continuous_target_error_m_within_clearance_tolerance",
    "minimum_route_length_m",
    "grid_cell_x_then_y",
)

_EPSILON_M = 1.0e-9


@dataclass(frozen=True)
class GoalCellRouteOptionEvidence:
    """Outcome of fully evaluating one quantization-equivalent goal cell."""

    cell: GridCell
    goal: Pose2D
    continuous_target_error_m: float
    endpoint_standoff_m: float
    accepted: bool
    rejection_reason: str | None
    route_raw_clearance_lower_bound_m: float | None = None
    route_length_m: float | None = None

    def to_metadata(self) -> dict[str, object]:
        return {
            "cell": {"x": self.cell.x, "y": self.cell.y},
            "goal": _pose_metadata(self.goal),
            "continuous_target_error_m": self.continuous_target_error_m,
            "endpoint_standoff_m": self.endpoint_standoff_m,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "route_raw_clearance_lower_bound_m": (
                self.route_raw_clearance_lower_bound_m
            ),
            "route_length_m": self.route_length_m,
        }


@dataclass(frozen=True)
class GoalCellSelectionEvidence:
    """Durable explanation of one deterministic goal-cell choice."""

    requested_goal: Pose2D
    requested_goal_cell: GridCell
    selected_goal: Pose2D
    selected_goal_cell: GridCell
    quantization_envelope_m: float
    clearance_sampling_spacing_m: float
    clearance_ranking_tolerance_m: float
    selected_route_raw_clearance_lower_bound_m: float
    selected_continuous_target_error_m: float
    selected_route_length_m: float
    options: tuple[GoalCellRouteOptionEvidence, ...]
    policy: str = GOAL_CELL_SELECTION_POLICY
    ranking_rationale: tuple[str, ...] = GOAL_CELL_RANKING_RATIONALE

    def to_metadata(self) -> dict[str, object]:
        clearance_score_kind = (
            "static_point_clearance_m"
            if self.selected_route_length_m <= _EPSILON_M
            else "continuous_route_clearance_lower_bound_m"
        )
        return {
            "schema_version": 2,
            "policy": self.policy,
            "advisory_only": True,
            "final_route_uncertainty_preflight_authoritative": True,
            "requested_goal": _pose_metadata(self.requested_goal),
            "requested_goal_cell": {
                "x": self.requested_goal_cell.x,
                "y": self.requested_goal_cell.y,
            },
            "selected_goal": _pose_metadata(self.selected_goal),
            "selected_goal_cell": {
                "x": self.selected_goal_cell.x,
                "y": self.selected_goal_cell.y,
            },
            "quantization_envelope_m": self.quantization_envelope_m,
            "clearance_sampling_spacing_m": (
                self.clearance_sampling_spacing_m
            ),
            "clearance_ranking_tolerance_m": (
                self.clearance_ranking_tolerance_m
            ),
            "clearance_ranking_tolerance_semantics": (
                "inclusive; differences at or below the tolerance use the "
                "remaining deterministic ranking criteria"
            ),
            "selected_route_raw_clearance_lower_bound_m": (
                self.selected_route_raw_clearance_lower_bound_m
            ),
            "selected_route_clearance_score_kind": clearance_score_kind,
            "selected_continuous_target_error_m": (
                self.selected_continuous_target_error_m
            ),
            "selected_route_length_m": self.selected_route_length_m,
            "ranking_rationale": list(self.ranking_rationale),
            # Unselected routes are not retained in CandidatePreapproachPlan,
            # so their derived scores cannot be independently rebound during
            # materialization.  They remain ephemeral computation evidence;
            # only selected fields that are rechecked against the retained
            # route are persisted.
            "unselected_option_evidence_persisted": False,
        }


@dataclass(frozen=True)
class SafetyRankedGoalPlan:
    """Selected full route plus its no-motion selection evidence."""

    result: PlanRouteResult
    connector: ExactStartConnectorEvidence
    smoothing: RouteSmoothingSummary
    evidence: GoalCellSelectionEvidence


@dataclass(frozen=True)
class _AcceptedGoalPlan:
    result: PlanRouteResult
    connector: ExactStartConnectorEvidence
    smoothing: RouteSmoothingSummary
    option: GoalCellRouteOptionEvidence


class NoSafetyRankedGoalRouteError(ValueError):
    """No rasterization-equivalent cell produced an acceptable full route."""

    def __init__(self, options: tuple[GoalCellRouteOptionEvidence, ...]) -> None:
        self.options = options
        detail = "; ".join(
            f"({option.cell.x},{option.cell.y}):{option.rejection_reason}"
            for option in options
        )
        super().__init__(
            "no certified rasterization-equivalent goal route"
            + (f": {detail}" if detail else "")
        )


RouteRejectionReason = Callable[[Route], str | None]


def rasterization_equivalent_goal_cells(
    costmap: Costmap,
    requested_goal: Pose2D,
) -> tuple[GridCell, ...]:
    """Return cells whose centres fit the ordinary half-diagonal error bound."""

    _validate_pose(requested_goal, "requested goal")
    _validate_resolution(costmap.resolution)
    requested_cell = costmap.world_to_grid(requested_goal)
    envelope_m = costmap.resolution / math.sqrt(2.0)
    cells = []
    for y in range(requested_cell.y - 1, requested_cell.y + 2):
        for x in range(requested_cell.x - 1, requested_cell.x + 2):
            cell = GridCell(x, y)
            if not costmap.in_bounds(cell):
                continue
            centre = costmap.grid_to_world(cell)
            if _distance(centre, requested_goal) <= envelope_m + _EPSILON_M:
                cells.append(cell)
    return tuple(sorted(cells))


def continuous_route_clearance_lower_bound_m(
    base_costmap: Costmap,
    route: Route,
    *,
    sampling_spacing_m: float = DEFAULT_UNCERTAINTY_CLEARANCE_SAMPLE_SPACING_M,
) -> float:
    """Return a conservative static-clearance score for route ranking.

    A one-point route represents an already-satisfied goal and has no segment
    to sample.  Its score is the exact point's conservative static clearance,
    so this advisory planner does not manufacture a move to an adjacent cell.
    This special case does not alter the downstream route-uncertainty
    admission contract; that final preflight remains authoritative whenever
    motion is proposed.
    """

    if not math.isfinite(sampling_spacing_m) or sampling_spacing_m <= 0.0:
        raise ValueError("clearance sampling spacing must be finite and positive")
    if not isinstance(route, Route) or not route.points:
        raise ValueError("route must contain at least one point")
    if len(route.points) == 1:
        result = point_clearance_to_blocked_m(
            base_costmap,
            route.points[0].pose,
        )
        if not math.isfinite(result) or result < 0.0:
            raise ValueError("route point clearance is invalid")
        return result
    profile, error = sample_route_for_uncertainty_admission(
        base_costmap,
        tuple(point.pose for point in route.points),
        sampling_spacing_m,
    )
    if error is not None or profile is None or not profile.segments:
        raise ValueError(error or "route has no clearance-bearing segments")
    result = min(segment.clearance_lower_bound_m for segment in profile.segments)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("route clearance lower bound is invalid")
    return result


def select_deterministic_goal_cell_option(
    options: tuple[GoalCellRouteOptionEvidence, ...],
    *,
    clearance_tolerance_m: float = (
        GOAL_CELL_CLEARANCE_RANKING_TOLERANCE_M
    ),
) -> GoalCellRouteOptionEvidence:
    """Return the deterministic winner among accepted goal-cell options.

    An already-satisfied zero-length route is preferred because it exposes the
    robot to no route motion.  Otherwise, clearance differences greater than
    ``clearance_tolerance_m`` remain decisive.  Values inside the inclusive
    tie band fall back to target fidelity, route length, and cell order, so
    micrometre-scale numeric noise cannot flip the chosen side.
    """

    if not math.isfinite(clearance_tolerance_m) or clearance_tolerance_m < 0.0:
        raise ValueError(
            "clearance ranking tolerance must be finite and non-negative"
        )
    accepted = []
    for option in options:
        if not isinstance(option, GoalCellRouteOptionEvidence):
            raise TypeError("goal-cell option evidence has the wrong type")
        if not option.accepted:
            continue
        clearance_m = option.route_raw_clearance_lower_bound_m
        route_length_m = option.route_length_m
        if (
            clearance_m is None
            or not math.isfinite(clearance_m)
            or clearance_m < 0.0
        ):
            raise ValueError("accepted goal-cell option has invalid clearance")
        if (
            route_length_m is None
            or not math.isfinite(route_length_m)
            or route_length_m < 0.0
        ):
            raise ValueError("accepted goal-cell option has invalid route length")
        if (
            not math.isfinite(option.continuous_target_error_m)
            or option.continuous_target_error_m < 0.0
        ):
            raise ValueError("accepted goal-cell option has invalid target error")
        accepted.append(option)
    if not accepted:
        raise ValueError("goal-cell selection has no accepted option")

    no_motion = tuple(
        option
        for option in accepted
        if float(option.route_length_m) <= _EPSILON_M
    )
    eligible = no_motion or tuple(accepted)
    maximum_clearance_m = max(
        float(option.route_raw_clearance_lower_bound_m)
        for option in eligible
    )
    clearance_tied = tuple(
        option
        for option in eligible
        if maximum_clearance_m
        - float(option.route_raw_clearance_lower_bound_m)
        <= clearance_tolerance_m + _EPSILON_M
    )
    return min(
        clearance_tied,
        key=lambda option: (
            option.continuous_target_error_m,
            float(option.route_length_m),
            option.cell.x,
            option.cell.y,
        ),
    )


def plan_safety_ranked_quantized_goal(
    *,
    base_costmap: Costmap,
    planning_costmap: Costmap,
    start: Pose2D,
    requested_goal: Pose2D,
    stand: Pose2D,
    minimum_standoff_m: float,
    snap_radius_m: float,
    required_start_clearance_m: float,
    route_rejection_reason: RouteRejectionReason,
    sampling_spacing_m: float = DEFAULT_UNCERTAINTY_CLEARANCE_SAMPLE_SPACING_M,
) -> SafetyRankedGoalPlan:
    """Plan every bounded equivalent goal and retain maximum raw clearance.

    ``route_rejection_reason`` applies caller-owned semantic checks (for
    example non-target candidate keepouts) to every fully generated route
    before it is eligible for ranking.
    """

    _validate_compatible_costmaps(base_costmap, planning_costmap)
    _validate_pose(start, "route start")
    _validate_pose(requested_goal, "requested goal")
    _validate_pose(stand, "stand")
    for name, value in (
        ("minimum_standoff_m", minimum_standoff_m),
        ("snap_radius_m", snap_radius_m),
        ("required_start_clearance_m", required_start_clearance_m),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not callable(route_rejection_reason):
        raise TypeError("route_rejection_reason must be callable")
    if not math.isfinite(sampling_spacing_m) or sampling_spacing_m <= 0.0:
        raise ValueError("sampling_spacing_m must be finite and positive")

    requested_cell = planning_costmap.world_to_grid(requested_goal)
    envelope_m = planning_costmap.resolution / math.sqrt(2.0)
    option_evidence: list[GoalCellRouteOptionEvidence] = []
    accepted: list[_AcceptedGoalPlan] = []
    for cell in rasterization_equivalent_goal_cells(
        planning_costmap,
        requested_goal,
    ):
        goal = planning_costmap.grid_to_world(
            cell,
            yaw_rad=requested_goal.yaw_rad,
        )
        target_error_m = _distance(goal, requested_goal)
        endpoint_standoff_m = _distance(goal, stand)
        rejection = None
        result = None
        connector = None
        smoothing = None
        clearance_m = None
        route_length_m = None

        if endpoint_standoff_m + _EPSILON_M < minimum_standoff_m:
            rejection = "endpoint_below_minimum_standoff"
        elif not planning_costmap.is_traversable(cell):
            rejection = "goal_cell_not_traversable"
        else:
            result = plan_route(
                planning_costmap,
                start,
                goal,
                snap_radius_m=snap_radius_m,
            )
            if result.route is None:
                rejection = f"astar_failed:{result.diagnostics.reason}"
            elif (
                result.diagnostics.snapped_goal_cell != cell
                or result.route.points[-1].cell != cell
            ):
                rejection = "astar_goal_was_snapped"
            else:
                try:
                    result, connector, smoothing = (
                        certify_and_smooth_exact_start_route(
                            result,
                            base_costmap=base_costmap,
                            planning_costmap=planning_costmap,
                            exact_start=start,
                            required_clearance_m=required_start_clearance_m,
                        )
                    )
                except ValueError as exc:
                    rejection = f"exact_start_or_smoothing_failed:{exc}"

        if rejection is None:
            assert result is not None and result.route is not None
            semantic_rejection = route_rejection_reason(result.route)
            if semantic_rejection is not None:
                if not isinstance(semantic_rejection, str) or not semantic_rejection:
                    raise TypeError(
                        "route_rejection_reason must return non-empty text or None"
                    )
                rejection = semantic_rejection
            else:
                try:
                    clearance_m = continuous_route_clearance_lower_bound_m(
                        base_costmap,
                        result.route,
                        sampling_spacing_m=sampling_spacing_m,
                    )
                except ValueError as exc:
                    rejection = f"route_clearance_scoring_failed:{exc}"
                else:
                    route_length_m = result.route.length_m

        option = GoalCellRouteOptionEvidence(
            cell=cell,
            goal=goal,
            continuous_target_error_m=target_error_m,
            endpoint_standoff_m=endpoint_standoff_m,
            accepted=rejection is None,
            rejection_reason=rejection,
            route_raw_clearance_lower_bound_m=clearance_m,
            route_length_m=route_length_m,
        )
        option_evidence.append(option)
        if rejection is None:
            assert result is not None
            assert connector is not None
            assert smoothing is not None
            accepted.append(_AcceptedGoalPlan(result, connector, smoothing, option))

    options = tuple(option_evidence)
    if not accepted:
        raise NoSafetyRankedGoalRouteError(options)
    selected_option = select_deterministic_goal_cell_option(
        options,
        clearance_tolerance_m=GOAL_CELL_CLEARANCE_RANKING_TOLERANCE_M,
    )
    selected = next(
        item for item in accepted if item.option is selected_option
    )
    assert selected_option.route_raw_clearance_lower_bound_m is not None
    assert selected_option.route_length_m is not None
    evidence = GoalCellSelectionEvidence(
        requested_goal=requested_goal,
        requested_goal_cell=requested_cell,
        selected_goal=selected_option.goal,
        selected_goal_cell=selected_option.cell,
        quantization_envelope_m=envelope_m,
        clearance_sampling_spacing_m=sampling_spacing_m,
        clearance_ranking_tolerance_m=(
            GOAL_CELL_CLEARANCE_RANKING_TOLERANCE_M
        ),
        selected_route_raw_clearance_lower_bound_m=(
            selected_option.route_raw_clearance_lower_bound_m
        ),
        selected_continuous_target_error_m=(
            selected_option.continuous_target_error_m
        ),
        selected_route_length_m=selected_option.route_length_m,
        options=options,
    )
    return SafetyRankedGoalPlan(
        result=selected.result,
        connector=selected.connector,
        smoothing=selected.smoothing,
        evidence=evidence,
    )


def validate_goal_cell_selection_binding(
    evidence: GoalCellSelectionEvidence,
    *,
    base_costmap: Costmap,
    planning_costmap: Costmap,
    result: PlanRouteResult,
    expected_requested_goal: Pose2D,
    stand: Pose2D,
    minimum_standoff_m: float,
) -> None:
    """Reject altered selection evidence before artifact materialization."""

    if not isinstance(evidence, GoalCellSelectionEvidence):
        raise TypeError("goal-cell selection evidence has the wrong type")
    if evidence.policy != GOAL_CELL_SELECTION_POLICY:
        raise ValueError("goal-cell selection policy mismatch")
    if evidence.ranking_rationale != GOAL_CELL_RANKING_RATIONALE:
        raise ValueError("goal-cell selection ranking rationale mismatch")
    if not math.isclose(
        evidence.clearance_ranking_tolerance_m,
        GOAL_CELL_CLEARANCE_RANKING_TOLERANCE_M,
        rel_tol=0.0,
        abs_tol=_EPSILON_M,
    ):
        raise ValueError("goal-cell selection clearance tolerance mismatch")
    _validate_compatible_costmaps(base_costmap, planning_costmap)
    _validate_pose(stand, "stand")
    _validate_pose(evidence.selected_goal, "selected goal")
    _validate_pose(expected_requested_goal, "expected requested goal")
    if not math.isfinite(minimum_standoff_m) or minimum_standoff_m < 0.0:
        raise ValueError("minimum standoff must be finite and non-negative")
    if _pose_distance_with_yaw(evidence.requested_goal, expected_requested_goal) > (
        _EPSILON_M
    ):
        raise ValueError("goal-cell selection requested target binding mismatch")
    # Bind the cell to the exact stored target, not a trigonometrically
    # reconstructed equivalent.  At an exact grid boundary, harmless
    # sub-nanometre sine/cosine roundoff can otherwise change ``floor`` even
    # though the continuous target binding above is satisfied.
    expected_requested_cell = planning_costmap.world_to_grid(
        evidence.requested_goal
    )
    if evidence.requested_goal_cell != expected_requested_cell:
        raise ValueError("goal-cell selection requested cell binding mismatch")
    expected_envelope_m = planning_costmap.resolution / math.sqrt(2.0)
    if not math.isclose(
        evidence.quantization_envelope_m,
        expected_envelope_m,
        rel_tol=0.0,
        abs_tol=_EPSILON_M,
    ):
        raise ValueError("goal-cell selection quantization envelope mismatch")

    if not isinstance(evidence.options, tuple):
        raise TypeError("goal-cell selection options must be a tuple")
    expected_cells = rasterization_equivalent_goal_cells(
        planning_costmap,
        evidence.requested_goal,
    )
    if tuple(option.cell for option in evidence.options) != expected_cells:
        raise ValueError("goal-cell selection option-cell evidence mismatch")
    for option in evidence.options:
        _validate_goal_cell_option_binding(
            option,
            planning_costmap=planning_costmap,
            requested_goal=evidence.requested_goal,
            stand=stand,
            minimum_standoff_m=minimum_standoff_m,
        )
    deterministic_winner = select_deterministic_goal_cell_option(
        evidence.options,
        clearance_tolerance_m=evidence.clearance_ranking_tolerance_m,
    )
    if deterministic_winner.cell != evidence.selected_goal_cell:
        raise ValueError(
            "goal-cell selection selected option is not deterministic winner"
        )
    if _pose_distance_with_yaw(
        deterministic_winner.goal,
        evidence.selected_goal,
    ) > _EPSILON_M:
        raise ValueError("goal-cell selection winner-goal evidence mismatch")

    if result.route is None or not result.route.points:
        raise ValueError("goal-cell selection requires a successful route")
    endpoint = result.route.points[-1]
    if evidence.selected_goal_cell != endpoint.cell:
        raise ValueError("goal-cell selection differs from route endpoint cell")
    if _distance(evidence.selected_goal, endpoint.pose) > _EPSILON_M:
        raise ValueError("goal-cell selection differs from route endpoint pose")
    if _distance(evidence.selected_goal, expected_requested_goal) > (
        expected_envelope_m + _EPSILON_M
    ):
        raise ValueError("selected goal lies outside the quantization envelope")
    if _distance(evidence.selected_goal, stand) + _EPSILON_M < minimum_standoff_m:
        raise ValueError("selected goal violates minimum stand standoff")
    selected_options = tuple(
        option
        for option in evidence.options
        if option.cell == evidence.selected_goal_cell and option.accepted
    )
    if len(selected_options) != 1:
        raise ValueError("goal-cell selection lacks one accepted selected option")
    selected_option = selected_options[0]
    if selected_option != deterministic_winner:
        raise ValueError("goal-cell selection winner evidence mismatch")
    if not math.isclose(
        selected_option.continuous_target_error_m,
        evidence.selected_continuous_target_error_m,
        rel_tol=0.0,
        abs_tol=_EPSILON_M,
    ):
        raise ValueError("selected goal target-error evidence mismatch")
    if not math.isclose(
        float(selected_option.route_length_m),
        evidence.selected_route_length_m,
        rel_tol=0.0,
        abs_tol=_EPSILON_M,
    ) or not math.isclose(
        result.route.length_m,
        evidence.selected_route_length_m,
        rel_tol=0.0,
        abs_tol=_EPSILON_M,
    ):
        raise ValueError("selected goal route-length evidence mismatch")
    observed_clearance_m = continuous_route_clearance_lower_bound_m(
        base_costmap,
        result.route,
        sampling_spacing_m=evidence.clearance_sampling_spacing_m,
    )
    if not math.isclose(
        observed_clearance_m,
        evidence.selected_route_raw_clearance_lower_bound_m,
        rel_tol=0.0,
        abs_tol=_EPSILON_M,
    ) or not math.isclose(
        observed_clearance_m,
        float(selected_option.route_raw_clearance_lower_bound_m),
        rel_tol=0.0,
        abs_tol=_EPSILON_M,
    ):
        raise ValueError("selected goal route-clearance evidence mismatch")


def _validate_goal_cell_option_binding(
    option: GoalCellRouteOptionEvidence,
    *,
    planning_costmap: Costmap,
    requested_goal: Pose2D,
    stand: Pose2D,
    minimum_standoff_m: float,
) -> None:
    """Bind every ephemeral option field before selecting durable metadata."""

    if not isinstance(option, GoalCellRouteOptionEvidence):
        raise TypeError("goal-cell option evidence has the wrong type")
    if not isinstance(option.cell, GridCell):
        raise TypeError("goal-cell option cell has the wrong type")
    if type(option.accepted) is not bool:
        raise TypeError("goal-cell option accepted flag must be boolean")
    _validate_pose(option.goal, "goal-cell option goal")
    expected_goal = planning_costmap.grid_to_world(
        option.cell,
        yaw_rad=requested_goal.yaw_rad,
    )
    if _pose_distance_with_yaw(option.goal, expected_goal) > _EPSILON_M:
        raise ValueError("goal-cell option goal evidence mismatch")
    expected_target_error_m = _distance(expected_goal, requested_goal)
    if not math.isclose(
        option.continuous_target_error_m,
        expected_target_error_m,
        rel_tol=0.0,
        abs_tol=_EPSILON_M,
    ):
        raise ValueError("goal-cell option target-error evidence mismatch")
    expected_standoff_m = _distance(expected_goal, stand)
    if not math.isclose(
        option.endpoint_standoff_m,
        expected_standoff_m,
        rel_tol=0.0,
        abs_tol=_EPSILON_M,
    ):
        raise ValueError("goal-cell option standoff evidence mismatch")

    if option.accepted:
        if option.rejection_reason is not None:
            raise ValueError("accepted goal-cell option has a rejection reason")
        if expected_standoff_m + _EPSILON_M < minimum_standoff_m:
            raise ValueError("accepted goal-cell option violates minimum standoff")
        if not planning_costmap.is_traversable(option.cell):
            raise ValueError("accepted goal-cell option is not traversable")
        # The shared selector performs strict numeric validation for every
        # accepted option and is reused for deterministic-winner validation.
        select_deterministic_goal_cell_option((option,))
        return

    if not isinstance(option.rejection_reason, str) or not option.rejection_reason:
        raise ValueError("rejected goal-cell option lacks a rejection reason")
    if (
        option.route_raw_clearance_lower_bound_m is not None
        or option.route_length_m is not None
    ):
        raise ValueError("rejected goal-cell option carries accepted-route metrics")


def _validate_compatible_costmaps(left: Costmap, right: Costmap) -> None:
    left_geometry = (
        left.width,
        left.height,
        left.metadata.resolution,
        left.metadata.origin,
    )
    right_geometry = (
        right.width,
        right.height,
        right.metadata.resolution,
        right.metadata.origin,
    )
    if left_geometry != right_geometry:
        raise ValueError("base and planning costmap geometry must match")
    _validate_resolution(left.resolution)


def _validate_resolution(value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("costmap resolution must be finite and positive")


def _validate_pose(value: Pose2D, label: str) -> None:
    if not isinstance(value, Pose2D) or not all(
        math.isfinite(component)
        for component in (value.x_m, value.y_m, value.yaw_rad)
    ):
        raise ValueError(f"{label} must be a finite Pose2D")


def _distance(left: Pose2D, right: Pose2D) -> float:
    return math.hypot(left.x_m - right.x_m, left.y_m - right.y_m)


def _pose_distance_with_yaw(left: Pose2D, right: Pose2D) -> float:
    return max(
        _distance(left, right),
        abs(math.atan2(
            math.sin(left.yaw_rad - right.yaw_rad),
            math.cos(left.yaw_rad - right.yaw_rad),
        )),
    )


def _pose_metadata(pose: Pose2D) -> dict[str, float]:
    return {
        "x_m": pose.x_m,
        "y_m": pose.y_m,
        "yaw_rad": pose.yaw_rad,
    }


__all__ = [
    "GOAL_CELL_CLEARANCE_RANKING_TOLERANCE_M",
    "GOAL_CELL_RANKING_RATIONALE",
    "GOAL_CELL_SELECTION_POLICY",
    "GoalCellRouteOptionEvidence",
    "GoalCellSelectionEvidence",
    "NoSafetyRankedGoalRouteError",
    "SafetyRankedGoalPlan",
    "continuous_route_clearance_lower_bound_m",
    "plan_safety_ranked_quantized_goal",
    "rasterization_equivalent_goal_cells",
    "select_deterministic_goal_cell_option",
    "validate_goal_cell_selection_binding",
]
