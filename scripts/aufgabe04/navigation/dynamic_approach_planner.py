"""Pure collision-safe planning for a dynamically observed stand approach.

The static ``Costmap`` passed to this module is assumed to already contain the
one and only static-obstacle inflation required by the caller.  This module
only adds a run-local, configuration-space keepout for the detected stand.
It deliberately has no ROS, Gazebo, camera, or simulation imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from scripts.aufgabe04.navigation.costmap import CELL_SOURCE_RUN_LOCAL, Costmap
from scripts.aufgabe04.navigation.global_planner import plan_route
from scripts.aufgabe04.navigation.models import GridCell, Pose2D


_EPSILON = 1.0e-10


@dataclass(frozen=True)
class DynamicApproachConfig:
    """Physical and geometric constraints for one stand approach."""

    stand_radius_m: float = 0.06
    stand_position_uncertainty_m: float = 0.02
    robot_radius_m: float = 0.105
    collision_margin_m: float = 0.02
    standoff_distance_m: float = 0.32
    terminal_corridor_length_m: float = 0.40
    corridor_sample_spacing_m: float = 0.05
    lidar_stop_distance_m: float = 0.18
    scan_origin_to_base_offset_m: float = 0.0
    lidar_clearance_margin_m: float = 0.02

    def __post_init__(self) -> None:
        finite_values = {
            "stand_radius_m": self.stand_radius_m,
            "stand_position_uncertainty_m": self.stand_position_uncertainty_m,
            "robot_radius_m": self.robot_radius_m,
            "collision_margin_m": self.collision_margin_m,
            "standoff_distance_m": self.standoff_distance_m,
            "terminal_corridor_length_m": self.terminal_corridor_length_m,
            "corridor_sample_spacing_m": self.corridor_sample_spacing_m,
            "lidar_stop_distance_m": self.lidar_stop_distance_m,
            "scan_origin_to_base_offset_m": self.scan_origin_to_base_offset_m,
            "lidar_clearance_margin_m": self.lidar_clearance_margin_m,
        }
        for name, value in finite_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in (
            "stand_radius_m",
            "robot_radius_m",
            "standoff_distance_m",
            "terminal_corridor_length_m",
            "corridor_sample_spacing_m",
            "lidar_stop_distance_m",
        ):
            if finite_values[name] <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "stand_position_uncertainty_m",
            "collision_margin_m",
            "lidar_clearance_margin_m",
        ):
            if finite_values[name] < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.30 <= self.terminal_corridor_length_m <= 0.50:
            raise ValueError("terminal_corridor_length_m must be in [0.30, 0.50]")

    @property
    def stand_keepout_radius_m(self) -> float:
        return (
            self.stand_radius_m
            + self.stand_position_uncertainty_m
            + self.robot_radius_m
            + self.collision_margin_m
        )

    @property
    def minimum_lidar_standoff_m(self) -> float:
        # The sign of a sensor mounting offset is not always available in
        # recommendation payloads.  Its magnitude is therefore used here as
        # the conservative reduction in stand clearance.
        return (
            self.stand_radius_m
            + self.lidar_stop_distance_m
            + abs(self.scan_origin_to_base_offset_m)
            + self.lidar_clearance_margin_m
        )


@dataclass(frozen=True)
class FaceNormalCandidate:
    """A stable stand face and its inner/outer approach poses."""

    face_id: int
    normal_rad: float
    target: Pose2D
    entry: Pose2D


@dataclass(frozen=True)
class DynamicApproachWaypoint:
    pose: Pose2D
    protected: bool
    corridor: bool


@dataclass(frozen=True)
class CandidateDiagnostics:
    face_id: int
    target: Pose2D
    entry: Pose2D
    valid: bool
    rejection_reasons: tuple[str, ...]
    astar_cell_count: int = 0
    astar_expanded_cells: int = 0
    smoothed_waypoint_count: int = 0
    route_length_m: float | None = None


@dataclass(frozen=True)
class DynamicApproachDiagnostics:
    keepout_radius_m: float
    keepout_cell_count: int
    minimum_lidar_standoff_m: float
    start_join_clearance_m: float
    requested_hard_face_id: int | None
    selected_face_id: int | None
    failure_reason: str | None
    candidates: tuple[CandidateDiagnostics, ...]


@dataclass(frozen=True)
class DynamicApproachPlan:
    waypoints: tuple[DynamicApproachWaypoint, ...]
    selected_face_id: int
    stand: Pose2D
    target: Pose2D
    entry: Pose2D
    length_m: float
    diagnostics: DynamicApproachDiagnostics


@dataclass(frozen=True)
class DynamicApproachPlanResult:
    plan: DynamicApproachPlan | None
    diagnostics: DynamicApproachDiagnostics


@dataclass(frozen=True)
class _ValidCandidate:
    candidate: FaceNormalCandidate
    waypoints: tuple[DynamicApproachWaypoint, ...]
    length_m: float
    diagnostics: CandidateDiagnostics


def _finite_pose(pose: Pose2D, *, name: str) -> None:
    if not math.isfinite(pose.x_m) or not math.isfinite(pose.y_m):
        raise ValueError(f"{name} position must be finite")


def _canonical_axial_angle(axis_rad: float) -> float:
    """Map a 180-degree-symmetric axis to a stable half-open interval."""

    if not math.isfinite(axis_rad):
        raise ValueError("stand axis must be finite")
    canonical = (axis_rad + math.pi / 2.0) % math.pi - math.pi / 2.0
    # Prevent floating-point representations of +pi/2 from changing face IDs.
    return -math.pi / 2.0 if canonical >= math.pi / 2.0 - _EPSILON else canonical


def _normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def face_normal_candidates(
    stand: Pose2D,
    stand_axis_rad: float,
    config: DynamicApproachConfig = DynamicApproachConfig(),
) -> tuple[FaceNormalCandidate, FaceNormalCandidate]:
    """Return stable face 0/1 candidates for an axial stand orientation.

    ``stand_axis_rad`` and ``stand_axis_rad + pi`` produce identical face IDs.
    Face 0 uses the positive normal of the canonical axis; face 1 is opposite.
    """

    _finite_pose(stand, name="stand")
    axis = _canonical_axial_angle(stand_axis_rad)
    normals = (axis + math.pi / 2.0, axis - math.pi / 2.0)
    candidates = []
    for face_id, normal in enumerate(normals):
        normal = _normalize_angle(normal)
        yaw = _normalize_angle(normal + math.pi)
        candidates.append(
            FaceNormalCandidate(
                face_id=face_id,
                normal_rad=normal,
                target=Pose2D(
                    stand.x_m + config.standoff_distance_m * math.cos(normal),
                    stand.y_m + config.standoff_distance_m * math.sin(normal),
                    yaw,
                ),
                entry=Pose2D(
                    stand.x_m
                    + (config.standoff_distance_m + config.terminal_corridor_length_m)
                    * math.cos(normal),
                    stand.y_m
                    + (config.standoff_distance_m + config.terminal_corridor_length_m)
                    * math.sin(normal),
                    yaw,
                ),
            )
        )
    return candidates[0], candidates[1]


def _distance_point_to_cell_square(costmap: Costmap, pose: Pose2D, cell: GridCell) -> float:
    origin_x, origin_y, _ = costmap.metadata.origin
    x0 = origin_x + cell.x * costmap.resolution
    y0 = origin_y + cell.y * costmap.resolution
    x1 = x0 + costmap.resolution
    y1 = y0 + costmap.resolution
    dx = max(x0 - pose.x_m, 0.0, pose.x_m - x1)
    dy = max(y0 - pose.y_m, 0.0, pose.y_m - y1)
    return math.hypot(dx, dy)


def circular_keepout_cells(
    costmap: Costmap,
    center: Pose2D,
    radius_m: float,
) -> frozenset[GridCell]:
    """Rasterize every cell square touched by a closed world-space disk."""

    _finite_pose(center, name="keepout center")
    if not math.isfinite(radius_m) or radius_m < 0.0:
        raise ValueError("keepout radius must be finite and non-negative")
    origin_x, origin_y, _ = costmap.metadata.origin
    resolution = costmap.resolution
    min_x = math.floor((center.x_m - radius_m - origin_x) / resolution) - 1
    max_x = math.floor((center.x_m + radius_m - origin_x) / resolution) + 1
    min_y = math.floor((center.y_m - radius_m - origin_y) / resolution) - 1
    max_y = math.floor((center.y_m + radius_m - origin_y) / resolution) + 1
    touched = set()
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            cell = GridCell(x, y)
            if not costmap.in_bounds(cell):
                continue
            if _distance_point_to_cell_square(costmap, center, cell) <= radius_m + _EPSILON:
                touched.add(cell)
    return frozenset(touched)


def point_clearance_to_blocked_m(costmap: Costmap, pose: Pose2D) -> float:
    """Return a conservative free-disk radius around an exact world pose."""

    _finite_pose(pose, name="clearance pose")
    containing = costmap.world_to_grid(pose)
    if not costmap.is_traversable(containing):
        return 0.0
    origin_x, origin_y, _ = costmap.metadata.origin
    map_max_x = origin_x + costmap.width * costmap.resolution
    map_max_y = origin_y + costmap.height * costmap.resolution
    clearance = min(
        pose.x_m - origin_x,
        map_max_x - pose.x_m,
        pose.y_m - origin_y,
        map_max_y - pose.y_m,
    )
    for cell in costmap.blocked_cells:
        clearance = min(clearance, _distance_point_to_cell_square(costmap, pose, cell))
        if clearance <= 0.0:
            return 0.0
    # Segment/cell checks use closed squares.  Stay strictly inside the free
    # disk so a join at the advertised limit cannot touch a blocked boundary.
    return max(0.0, clearance - 1.0e-6)


def with_dynamic_stand_keepout(
    costmap: Costmap,
    stand: Pose2D,
    config: DynamicApproachConfig = DynamicApproachConfig(),
) -> tuple[Costmap, frozenset[GridCell]]:
    """Overlay only the detected stand keepout; never reinflate static cells."""

    cells = circular_keepout_cells(costmap, stand, config.stand_keepout_radius_m)
    return costmap.with_blocked_cells(cells, source=CELL_SOURCE_RUN_LOCAL), cells


def _segment_intersects_closed_cell(
    costmap: Costmap,
    start: Pose2D,
    end: Pose2D,
    cell: GridCell,
) -> bool:
    origin_x, origin_y, _ = costmap.metadata.origin
    x_min = origin_x + cell.x * costmap.resolution
    y_min = origin_y + cell.y * costmap.resolution
    x_max = x_min + costmap.resolution
    y_max = y_min + costmap.resolution
    dx = end.x_m - start.x_m
    dy = end.y_m - start.y_m
    t_min = 0.0
    t_max = 1.0
    for coordinate, delta, lower, upper in (
        (start.x_m, dx, x_min, x_max),
        (start.y_m, dy, y_min, y_max),
    ):
        if abs(delta) <= _EPSILON:
            if coordinate < lower - _EPSILON or coordinate > upper + _EPSILON:
                return False
            continue
        near = (lower - coordinate) / delta
        far = (upper - coordinate) / delta
        if near > far:
            near, far = far, near
        t_min = max(t_min, near)
        t_max = min(t_max, far)
        if t_min > t_max + _EPSILON:
            return False
    return True


def supercover_segment_cells(
    costmap: Costmap,
    start: Pose2D,
    end: Pose2D,
) -> tuple[GridCell, ...]:
    """Return all grid squares touched by a closed world-space segment.

    Closed-square intersection intentionally includes both sides of a grid
    boundary and all four cells at a corner.  Consequently a diagonal cannot
    graze or cut the corner of an occupied cell unnoticed.
    """

    _finite_pose(start, name="segment start")
    _finite_pose(end, name="segment end")
    origin_x, origin_y, _ = costmap.metadata.origin
    resolution = costmap.resolution
    min_x = math.floor((min(start.x_m, end.x_m) - origin_x) / resolution) - 1
    max_x = math.floor((max(start.x_m, end.x_m) - origin_x) / resolution) + 1
    min_y = math.floor((min(start.y_m, end.y_m) - origin_y) / resolution) - 1
    max_y = math.floor((max(start.y_m, end.y_m) - origin_y) / resolution) + 1
    touched = []
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            cell = GridCell(x, y)
            if _segment_intersects_closed_cell(costmap, start, end, cell):
                touched.append(cell)
    return tuple(sorted(set(touched)))


def segment_is_collision_free(costmap: Costmap, start: Pose2D, end: Pose2D) -> bool:
    """Validate a segment against an already configuration-space costmap."""

    return all(costmap.is_traversable(cell) for cell in supercover_segment_cells(costmap, start, end))


def greedy_line_of_sight_shortcut(
    costmap: Costmap,
    poses: Sequence[Pose2D],
) -> tuple[Pose2D, ...]:
    """Greedily remove grid turns only when the full shortcut is collision-free."""

    if not poses:
        return ()
    if len(poses) == 1:
        return (poses[0],)
    shortened = [poses[0]]
    anchor = 0
    while anchor < len(poses) - 1:
        selected = None
        for candidate in range(len(poses) - 1, anchor, -1):
            if segment_is_collision_free(costmap, poses[anchor], poses[candidate]):
                selected = candidate
                break
        if selected is None:
            raise ValueError("input polyline contains a colliding segment")
        shortened.append(poses[selected])
        anchor = selected
    return tuple(shortened)


def _polyline_length(poses: Iterable[Pose2D]) -> float:
    total = 0.0
    previous = None
    for pose in poses:
        if previous is not None:
            total += math.hypot(pose.x_m - previous.x_m, pose.y_m - previous.y_m)
        previous = pose
    return total


def _same_position(a: Pose2D, b: Pose2D) -> bool:
    return math.hypot(a.x_m - b.x_m, a.y_m - b.y_m) <= _EPSILON


def _nan_pose(pose: Pose2D) -> Pose2D:
    return Pose2D(pose.x_m, pose.y_m, math.nan)


def _candidate_failure(
    candidate: FaceNormalCandidate,
    reasons: Sequence[str],
    *,
    astar_cell_count: int = 0,
    astar_expanded_cells: int = 0,
) -> CandidateDiagnostics:
    return CandidateDiagnostics(
        face_id=candidate.face_id,
        target=candidate.target,
        entry=candidate.entry,
        valid=False,
        rejection_reasons=tuple(dict.fromkeys(reasons)),
        astar_cell_count=astar_cell_count,
        astar_expanded_cells=astar_expanded_cells,
    )


def _terminal_corridor_poses(
    candidate: FaceNormalCandidate,
    config: DynamicApproachConfig,
) -> tuple[Pose2D, ...]:
    samples = max(
        1,
        int(math.ceil(config.terminal_corridor_length_m / config.corridor_sample_spacing_m)),
    )
    poses = []
    for index in range(samples + 1):
        fraction = index / samples
        poses.append(
            Pose2D(
                candidate.entry.x_m
                + fraction * (candidate.target.x_m - candidate.entry.x_m),
                candidate.entry.y_m
                + fraction * (candidate.target.y_m - candidate.entry.y_m),
                candidate.target.yaw_rad,
            )
        )
    return tuple(poses)


def _validate_candidate(
    costmap: Costmap,
    start: Pose2D,
    candidate: FaceNormalCandidate,
    config: DynamicApproachConfig,
    global_reasons: Sequence[str],
) -> tuple[_ValidCandidate | None, CandidateDiagnostics]:
    reasons = list(global_reasons)
    target_cells = supercover_segment_cells(costmap, candidate.target, candidate.target)
    if not target_cells or any(not costmap.is_traversable(cell) for cell in target_cells):
        reasons.append("target_not_traversable")
    entry_cells = supercover_segment_cells(costmap, candidate.entry, candidate.entry)
    if not entry_cells or any(not costmap.is_traversable(cell) for cell in entry_cells):
        reasons.append("entry_not_traversable")
    if not segment_is_collision_free(costmap, candidate.entry, candidate.target):
        reasons.append("terminal_corridor_blocked")
    if reasons:
        diagnostics = _candidate_failure(candidate, reasons)
        return None, diagnostics

    route_result = plan_route(costmap, start, candidate.entry, snap_radius_m=0.0)
    expanded = route_result.diagnostics.expanded_cells
    if route_result.route is None:
        diagnostics = _candidate_failure(
            candidate,
            (f"astar_failed:{route_result.diagnostics.reason}",),
            astar_expanded_cells=expanded,
        )
        return None, diagnostics
    route = route_result.route
    entry_cell = costmap.world_to_grid(candidate.entry)
    if (
        route_result.diagnostics.snapped_goal_cell != entry_cell
        or route.points[-1].cell != entry_cell
    ):
        diagnostics = _candidate_failure(
            candidate,
            ("astar_goal_was_snapped",),
            astar_cell_count=len(route.points),
            astar_expanded_cells=expanded,
        )
        return None, diagnostics

    goal_center = route.points[-1].pose
    if not segment_is_collision_free(costmap, goal_center, candidate.entry):
        diagnostics = _candidate_failure(
            candidate,
            ("entry_connector_blocked",),
            astar_cell_count=len(route.points),
            astar_expanded_cells=expanded,
        )
        return None, diagnostics
    first_center = route.points[0].pose
    if not segment_is_collision_free(costmap, start, first_center):
        diagnostics = _candidate_failure(
            candidate,
            ("start_connector_blocked",),
            astar_cell_count=len(route.points),
            astar_expanded_cells=expanded,
        )
        return None, diagnostics

    # Include the exact world-space entry in line-of-sight smoothing.  Leaving
    # it until after smoothing preserved an unnecessary final grid-cell-center
    # kink; on reverse staging that tiny kink could demand a large body turn
    # immediately before the protected corridor.
    grid_poses = [start]
    grid_poses.extend(point.pose for point in route.points)
    grid_poses.append(candidate.entry)
    deduplicated_grid = [grid_poses[0]]
    for pose in grid_poses[1:]:
        if not _same_position(deduplicated_grid[-1], pose):
            deduplicated_grid.append(pose)
    smoothed = list(greedy_line_of_sight_shortcut(costmap, deduplicated_grid))
    if not _same_position(smoothed[-1], candidate.entry):
        smoothed.append(candidate.entry)

    waypoints = [
        DynamicApproachWaypoint(_nan_pose(pose), protected=False, corridor=False)
        for pose in smoothed
    ]
    corridor = _terminal_corridor_poses(candidate, config)
    for pose in corridor:
        if _same_position(waypoints[-1].pose, pose):
            # Preserve a zero-length semantic handoff at the exact entry: the
            # first copy terminates forward/reverse staging with unconstrained
            # yaw, and the protected copy switches to forward corridor motion.
            # Its zero geometric length does not alter route cost or safety.
            waypoints.append(
                DynamicApproachWaypoint(pose, protected=True, corridor=True)
            )
        else:
            waypoints.append(
                DynamicApproachWaypoint(pose, protected=True, corridor=True)
            )
    waypoint_tuple = tuple(waypoints)
    length = _polyline_length(waypoint.pose for waypoint in waypoint_tuple)
    diagnostics = CandidateDiagnostics(
        face_id=candidate.face_id,
        target=candidate.target,
        entry=candidate.entry,
        valid=True,
        rejection_reasons=(),
        astar_cell_count=len(route.points),
        astar_expanded_cells=expanded,
        smoothed_waypoint_count=len(smoothed),
        route_length_m=length,
    )
    return (
        _ValidCandidate(candidate, waypoint_tuple, length, diagnostics),
        diagnostics,
    )


def plan_axis_acquisition(
    costmap: Costmap,
    start: Pose2D,
    stand: Pose2D,
    target: Pose2D,
    *,
    config: DynamicApproachConfig = DynamicApproachConfig(),
) -> DynamicApproachPlanResult:
    """Plan to one fixed observation pose while the stand axis is unknown.

    The supplied target is produced once from the initial robot-to-stand ray.
    It is deliberately not recomputed from the moving robot.  No terminal
    face corridor is created until silhouette consensus commits a stand face.
    """

    _finite_pose(start, name="start")
    _finite_pose(stand, name="stand")
    _finite_pose(target, name="axis acquisition target")
    planning_costmap, keepout_cells = with_dynamic_stand_keepout(costmap, stand, config)
    start_join_clearance = point_clearance_to_blocked_m(planning_costmap, start)
    standoff = math.hypot(target.x_m - stand.x_m, target.y_m - stand.y_m)
    reasons = []
    if standoff <= config.stand_keepout_radius_m + _EPSILON:
        reasons.append("acquisition_target_inside_stand_keepout")
    if standoff < config.minimum_lidar_standoff_m - _EPSILON:
        reasons.append("acquisition_target_incompatible_with_lidar_stop")
    target_cells = supercover_segment_cells(planning_costmap, target, target)
    if not target_cells or any(not planning_costmap.is_traversable(cell) for cell in target_cells):
        reasons.append("acquisition_target_not_traversable")

    route = None
    expanded = 0
    if not reasons:
        route_result = plan_route(planning_costmap, start, target, snap_radius_m=0.0)
        expanded = route_result.diagnostics.expanded_cells
        route = route_result.route
        if route is None:
            reasons.append(f"astar_failed:{route_result.diagnostics.reason}")
        elif route_result.diagnostics.snapped_goal_cell != planning_costmap.world_to_grid(target):
            reasons.append("astar_goal_was_snapped")

    waypoints: tuple[DynamicApproachWaypoint, ...] = ()
    length = None
    astar_count = 0 if route is None else len(route.points)
    if route is not None and not reasons:
        poses = [start, *(point.pose for point in route.points), target]
        deduplicated = [poses[0]]
        for pose in poses[1:]:
            if not _same_position(deduplicated[-1], pose):
                deduplicated.append(pose)
        if len(deduplicated) > 1 and not segment_is_collision_free(
            planning_costmap, start, deduplicated[1]
        ):
            reasons.append("start_connector_blocked")
        else:
            smoothed = list(greedy_line_of_sight_shortcut(planning_costmap, deduplicated))
            if not reasons:
                waypoints = tuple(
                    DynamicApproachWaypoint(
                        target if index == len(smoothed) - 1 else _nan_pose(pose),
                        protected=False,
                        corridor=False,
                    )
                    for index, pose in enumerate(smoothed)
                )
                length = _polyline_length(waypoint.pose for waypoint in waypoints)

    candidate_diagnostics = CandidateDiagnostics(
        face_id=-1,
        target=target,
        entry=target,
        valid=not reasons,
        rejection_reasons=tuple(dict.fromkeys(reasons)),
        astar_cell_count=astar_count,
        astar_expanded_cells=expanded,
        smoothed_waypoint_count=len(waypoints),
        route_length_m=length,
    )
    diagnostics = DynamicApproachDiagnostics(
        keepout_radius_m=config.stand_keepout_radius_m,
        keepout_cell_count=len(keepout_cells),
        minimum_lidar_standoff_m=config.minimum_lidar_standoff_m,
        start_join_clearance_m=start_join_clearance,
        requested_hard_face_id=None,
        selected_face_id=-1 if not reasons else None,
        failure_reason=(None if not reasons else reasons[0]),
        candidates=(candidate_diagnostics,),
    )
    if reasons or length is None:
        return DynamicApproachPlanResult(None, diagnostics)
    return DynamicApproachPlanResult(
        DynamicApproachPlan(
            waypoints=waypoints,
            selected_face_id=-1,
            stand=stand,
            target=target,
            entry=target,
            length_m=length,
            diagnostics=diagnostics,
        ),
        diagnostics,
    )


def plan_dynamic_approach(
    costmap: Costmap,
    start: Pose2D,
    stand: Pose2D,
    stand_axis_rad: float,
    *,
    hard_face_id: int | None = None,
    config: DynamicApproachConfig = DynamicApproachConfig(),
) -> DynamicApproachPlanResult:
    """Plan to the safest valid stand face, or explicitly withdraw the target.

    With ambiguous side evidence (``hard_face_id is None``), both faces are
    evaluated and the shortest valid plan wins; face ID is the deterministic
    tie-break.  Hard evidence never falls back to the opposite face.
    """

    _finite_pose(start, name="start")
    _finite_pose(stand, name="stand")
    if hard_face_id not in (None, 0, 1):
        raise ValueError("hard_face_id must be None, 0, or 1")
    candidates = face_normal_candidates(stand, stand_axis_rad, config)
    planning_costmap, keepout_cells = with_dynamic_stand_keepout(costmap, stand, config)
    start_join_clearance = point_clearance_to_blocked_m(planning_costmap, start)
    global_reasons = []
    if config.standoff_distance_m <= config.stand_keepout_radius_m + _EPSILON:
        global_reasons.append("standoff_inside_stand_keepout")
    if config.standoff_distance_m < config.minimum_lidar_standoff_m - _EPSILON:
        global_reasons.append("standoff_incompatible_with_lidar_stop")

    valid_by_face: dict[int, _ValidCandidate] = {}
    diagnostics_by_face = []
    for candidate in candidates:
        valid, diagnostics = _validate_candidate(
            planning_costmap,
            start,
            candidate,
            config,
            global_reasons,
        )
        diagnostics_by_face.append(diagnostics)
        if valid is not None:
            valid_by_face[candidate.face_id] = valid

    selected = None
    failure_reason = None
    if hard_face_id is not None:
        selected = valid_by_face.get(hard_face_id)
        if selected is None:
            failure_reason = f"hard_face_{hard_face_id}_invalid"
    elif valid_by_face:
        selected = min(valid_by_face.values(), key=lambda item: (item.length_m, item.candidate.face_id))
    else:
        failure_reason = "no_valid_face_candidate"

    diagnostics = DynamicApproachDiagnostics(
        keepout_radius_m=config.stand_keepout_radius_m,
        keepout_cell_count=len(keepout_cells),
        minimum_lidar_standoff_m=config.minimum_lidar_standoff_m,
        start_join_clearance_m=start_join_clearance,
        requested_hard_face_id=hard_face_id,
        selected_face_id=None if selected is None else selected.candidate.face_id,
        failure_reason=failure_reason,
        candidates=tuple(diagnostics_by_face),
    )
    if selected is None:
        return DynamicApproachPlanResult(plan=None, diagnostics=diagnostics)
    plan = DynamicApproachPlan(
        waypoints=selected.waypoints,
        selected_face_id=selected.candidate.face_id,
        stand=stand,
        target=selected.candidate.target,
        entry=selected.candidate.entry,
        length_m=selected.length_m,
        diagnostics=diagnostics,
    )
    return DynamicApproachPlanResult(plan=plan, diagnostics=diagnostics)
