"""Certify a short exact-goal join across candidate raster rounding only.

The unchanged planning costmap still governs the grid route. This proof binds
one final metric segment to the stored pose; it never removes a keepout or
authorizes motion. The caller supplies the full candidate-clearance validator.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Callable

from scripts.aufgabe04.navigation.approach.admitted_return_uncertainty import MINIMUM_REMAINING_ROUTE_M
from scripts.aufgabe04.navigation.foundation.models import GridCell, Pose2D
from scripts.aufgabe04.navigation.planning.costmap import CELL_SOURCE_STATION_KEEPOUT, Costmap
from scripts.aufgabe04.navigation.planning.exact_start_connector import _segment_clearance_evidence
from scripts.aufgabe04.navigation.planning.route_smoothing import (
    segment_is_collision_free, supercover_segment_cells,
)


EXACT_STORED_GOAL_CONNECTOR_POLICY = "candidate_raster_exact_stored_goal_connector"
MAXIMUM_CONNECTOR_LENGTH_CELLS = 3.0
_EPSILON_M = 1.0e-9


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _pose(value, name, *, require_yaw=True):
    if not isinstance(value, Pose2D) or not all(_finite(v) for v in (
        value.x_m, value.y_m, *((value.yaw_rad,) if require_yaw else ()),
    )):
        raise ValueError(f"stored-goal connector {name} must be a finite pose")


def _maximum_length(costmap):
    if not isinstance(costmap, Costmap) or not _finite(costmap.resolution) or costmap.resolution <= 0:
        raise ValueError("stored-goal connector requires a positive costmap resolution")
    # A stopped uncertainty prefix must leave at least this much route. Keeping
    # the connector no longer prevents a prefix from stopping inside it.
    return min(MAXIMUM_CONNECTOR_LENGTH_CELLS * costmap.resolution, MINIMUM_REMAINING_ROUTE_M)


def candidate_goal_anchors(planning_costmap: Costmap, target: Pose2D) -> tuple[Pose2D, ...]:
    """Enumerate nearby free grid centres without replacing the exact goal."""
    _pose(target, "target")
    maximum = _maximum_length(planning_costmap)
    target_cell = planning_costmap.world_to_grid(target)
    if not planning_costmap.in_bounds(target_cell):
        return ()
    radius = int(math.ceil(maximum / planning_costmap.resolution))
    candidates = []
    for y in range(target_cell.y - radius, target_cell.y + radius + 1):
        for x in range(target_cell.x - radius, target_cell.x + radius + 1):
            cell = GridCell(x, y)
            if not planning_costmap.is_traversable(cell):
                continue
            anchor = planning_costmap.grid_to_world(cell)
            length = math.hypot(anchor.x_m - target.x_m, anchor.y_m - target.y_m)
            if _EPSILON_M < length <= maximum + _EPSILON_M:
                candidates.append((length, x, y, anchor))
    return tuple(value[-1] for value in sorted(candidates, key=lambda value: value[:3]))


def certify_stored_goal_connector(
    *, base_costmap: Costmap, static_costmap: Costmap, planning_costmap: Costmap,
    anchor: Pose2D, target: Pose2D, inflation_radius_m: float,
    validate_candidate_clearance: Callable[[tuple[Pose2D, Pose2D]], None],
) -> dict[str, object]:
    """Prove static and full continuous candidate clearance for one suffix.

    ``base_costmap`` includes arena bounds and any temporary obstacles;
    ``static_costmap`` is its inflated version before candidate rasterization.
    The callback must validate both original and measured candidate keepouts
    along the complete segment, plus the exact target's active standoff.
    """
    _pose(anchor, "anchor", require_yaw=False)
    _pose(target, "target")
    maximum = _maximum_length(planning_costmap)
    for costmap in (base_costmap, static_costmap):
        _maximum_length(costmap)
        if (costmap.metadata, costmap.width, costmap.height, costmap.cells) != (
            planning_costmap.metadata, planning_costmap.width, planning_costmap.height, planning_costmap.cells,
        ):
            raise ValueError("stored-goal connector costmaps differ")
    if not _finite(inflation_radius_m) or inflation_radius_m < 0:
        raise ValueError("stored-goal connector inflation must be finite and nonnegative")
    if static_costmap.blocked_cells != base_costmap.with_inflation(inflation_radius_m).blocked_cells:
        raise ValueError("stored-goal connector static map differs from required inflation")
    if not static_costmap.blocked_cells.issubset(planning_costmap.blocked_cells):
        raise ValueError("stored-goal connector planning map removed static obstacles")
    # Transit yaw is unconstrained in CSV; the terminal stored yaw is exact.
    anchor = Pose2D(anchor.x_m, anchor.y_m, 0.)
    anchor_cell = planning_costmap.world_to_grid(anchor)
    center = planning_costmap.grid_to_world(anchor_cell)
    if math.hypot(anchor.x_m - center.x_m, anchor.y_m - center.y_m) > _EPSILON_M:
        raise ValueError("stored-goal connector anchor must be a grid cell centre")
    if not segment_is_collision_free(planning_costmap, anchor, anchor):
        raise ValueError("stored-goal connector anchor is blocked")
    length = math.hypot(target.x_m - anchor.x_m, target.y_m - anchor.y_m)
    if not _EPSILON_M < length <= maximum + _EPSILON_M:
        raise ValueError("stored-goal connector length exceeds its local bound or is zero")
    if not segment_is_collision_free(static_costmap, anchor, target):
        raise ValueError("stored-goal connector intersects static inflation or temporary obstacles")
    target_cells = supercover_segment_cells(planning_costmap, target, target)
    blocked_target = tuple(cell for cell in target_cells if planning_costmap.is_blocked(cell))
    if not blocked_target or any(
        planning_costmap.cell_sources.get(cell) != CELL_SOURCE_STATION_KEEPOUT for cell in blocked_target
    ):
        raise ValueError("stored-goal connector target is not blocked only by candidate raster")
    if any(
        planning_costmap.is_blocked(cell)
        and planning_costmap.cell_sources.get(cell) != CELL_SOURCE_STATION_KEEPOUT
        for cell in supercover_segment_cells(planning_costmap, anchor, target)
    ):
        raise ValueError("stored-goal connector intersects an unsupported planning overlay")
    static_proof = _segment_clearance_evidence(
        base_costmap, anchor, target, required_clearance_m=inflation_radius_m,
    )
    if not callable(validate_candidate_clearance):
        raise ValueError("stored-goal connector requires a candidate-clearance validator")
    if validate_candidate_clearance((anchor, target)) is not None:
        raise ValueError("stored-goal connector candidate validator must raise on failure")
    return {
        "schema_version": 1, "policy": EXACT_STORED_GOAL_CONNECTOR_POLICY,
        "anchor": asdict(anchor), "target": asdict(target),
        "connector_length_m": length, "maximum_connector_length_m": maximum,
        "static_clearance": static_proof.to_metadata(),
        "candidate_keepouts_continuously_validated": True,
        "static_inflated_connector_grid_validated": True,
        "target_blocked_only_by_candidate_raster": True,
        "motion_authorized": False,
    }
