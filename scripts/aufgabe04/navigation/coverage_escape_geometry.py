"""Pure geometry gates for executable coverage-blockage escapes.

The blockage planner may start inside a newly introduced transit keepout even
though the stopped robot pose is still outside the smaller hard body envelope.
That exceptional first connector is useful only when the waypoint controller
can execute it without inventing additional reverse corners.  This module keeps
that kinematic contract independent of ROS and of the replan artifact writer.

Forward connectors are preferred.  A reverse connector is accepted only when
it is already aligned with the robot body, continues as one straight geometric
prefix to a rotation-safe transition anchor, and has a normal forward segment
after that anchor with the complete keepout plus execution-tube clearance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from scripts.aufgabe04.navigation.costmap import Costmap
from scripts.aufgabe04.navigation.dynamic_approach_planner import (
    segment_is_collision_free,
)
from scripts.aufgabe04.navigation.models import GridCell, Pose2D


EGRESS_MODE_FORWARD = "forward"
EGRESS_MODE_STRAIGHT_REVERSE = "straight_reverse"
DEFAULT_FORWARD_TRANSLATION_HEADING_LIMIT_RAD = 1.25
DEFAULT_REVERSE_CONNECTOR_ALIGNMENT_TOLERANCE_RAD = 0.10
_EPSILON_M = 1.0e-9
_POSITION_TOLERANCE_M = 1.0e-8


@dataclass(frozen=True)
class CircularEscapeKeepout:
    """One obstacle's hard connector and ordinary route envelopes."""

    candidate_uid: str
    center: Pose2D
    hard_exclusion_radius_m: float
    route_keepout_radius_m: float

    def __post_init__(self) -> None:
        if not str(self.candidate_uid).strip():
            raise ValueError("escape keepout candidate_uid must be non-empty")
        values = (
            self.center.x_m,
            self.center.y_m,
            self.hard_exclusion_radius_m,
            self.route_keepout_radius_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("escape keepout geometry must be finite")
        if self.hard_exclusion_radius_m <= 0.0:
            raise ValueError("hard exclusion radius must be positive")
        if self.route_keepout_radius_m <= 0.0:
            raise ValueError("route keepout radius must be positive")


@dataclass(frozen=True)
class EgressConnectorChoice:
    """A continuously safe exact-start connector with an executable mode."""

    anchor: Pose2D
    mode: str
    connector_distance_m: float
    connector_heading_error_rad: float
    minimum_hard_clearance_m: float

    def __post_init__(self) -> None:
        _finite_pose(self.anchor, name="egress connector anchor")
        if self.mode not in {
            EGRESS_MODE_FORWARD,
            EGRESS_MODE_STRAIGHT_REVERSE,
        }:
            raise ValueError(f"unknown egress connector mode {self.mode!r}")
        if not all(
            math.isfinite(value)
            for value in (
                self.connector_distance_m,
                self.connector_heading_error_rad,
            )
        ) or math.isnan(self.minimum_hard_clearance_m):
            raise ValueError("egress connector evidence must be numeric")
        if self.connector_distance_m <= 0.0:
            raise ValueError("egress connector distance must be positive")
        if self.minimum_hard_clearance_m <= 0.0:
            raise ValueError("egress connector hard clearance must be positive")


@dataclass(frozen=True)
class ReverseTransitionChoice:
    """A farther collinear anchor at which reverse motion may end."""

    anchor: Pose2D
    distance_from_start_m: float
    reverse_heading_error_rad: float
    minimum_keepout_tube_clearance_m: float

    def __post_init__(self) -> None:
        _finite_pose(self.anchor, name="reverse transition anchor")
        if not all(
            math.isfinite(value)
            for value in (
                self.distance_from_start_m,
                self.reverse_heading_error_rad,
            )
        ) or math.isnan(self.minimum_keepout_tube_clearance_m):
            raise ValueError("reverse transition evidence must be numeric")
        if self.distance_from_start_m <= 0.0:
            raise ValueError("reverse transition distance must be positive")
        if self.minimum_keepout_tube_clearance_m <= 0.0:
            raise ValueError(
                "reverse transition keepout-tube clearance must be positive"
            )


@dataclass(frozen=True)
class ExecutableEscapeGeometry:
    """Metadata binding route vertices to the controller handoff contract."""

    mode: str
    connector_anchor: Pose2D
    transition_anchor: Pose2D
    transition_waypoint_index: int
    forward_waypoint_index: int | None
    connector_heading_error_rad: float
    forward_translation_heading_limit_rad: float
    reverse_alignment_tolerance_rad: float
    minimum_connector_hard_clearance_m: float
    minimum_transition_keepout_tube_clearance_m: float | None


def _finite_pose(pose: Pose2D, *, name: str) -> None:
    if not all(
        math.isfinite(value)
        for value in (pose.x_m, pose.y_m, pose.yaw_rad)
    ):
        raise ValueError(f"{name} must be finite")


def _normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _same_position(a: Pose2D, b: Pose2D) -> bool:
    return math.hypot(a.x_m - b.x_m, a.y_m - b.y_m) <= (
        _POSITION_TOLERANCE_M
    )


def _segment_heading(start: Pose2D, end: Pose2D) -> float:
    dx = end.x_m - start.x_m
    dy = end.y_m - start.y_m
    if math.hypot(dx, dy) <= _EPSILON_M:
        raise ValueError("escape route contains a zero-length geometric segment")
    return math.atan2(dy, dx)


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


def _minimum_segment_margin_m(
    start: Pose2D,
    end: Pose2D,
    keepouts: Sequence[CircularEscapeKeepout],
    *,
    use_hard_envelope: bool,
    tracking_tube_radius_m: float = 0.0,
) -> float:
    if not keepouts:
        return math.inf
    return min(
        _point_to_segment_distance_m(keepout.center, start, end)
        - (
            keepout.hard_exclusion_radius_m
            if use_hard_envelope
            else keepout.route_keepout_radius_m + tracking_tube_radius_m
        )
        for keepout in keepouts
    )


def _candidate_anchor_poses(
    costmap: Costmap,
    start: Pose2D,
    *,
    search_radius_m: float,
) -> tuple[tuple[float, GridCell, Pose2D], ...]:
    if not math.isfinite(search_radius_m) or search_radius_m <= 0.0:
        raise ValueError("egress search radius must be finite and positive")
    radius_cells = max(1, int(math.ceil(search_radius_m / costmap.resolution)))
    start_cell = costmap.world_to_grid(start)
    anchors = []
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            cell = GridCell(start_cell.x + dx, start_cell.y + dy)
            if not costmap.is_traversable(cell):
                continue
            anchor = costmap.grid_to_world(cell)
            distance_m = math.hypot(
                anchor.x_m - start.x_m,
                anchor.y_m - start.y_m,
            )
            if (
                distance_m <= _EPSILON_M
                or distance_m > search_radius_m + costmap.resolution * 1.0e-6
            ):
                continue
            anchors.append((distance_m, cell, anchor))
    return tuple(sorted(anchors, key=lambda item: (item[0], item[1])))


def choose_egress_connectors(
    static_costmap: Costmap,
    planning_costmap: Costmap,
    start: Pose2D,
    keepouts: Sequence[CircularEscapeKeepout],
    *,
    blocker_candidate_uids: Iterable[str],
    search_radius_m: float,
    forward_translation_heading_limit_rad: float = (
        DEFAULT_FORWARD_TRANSLATION_HEADING_LIMIT_RAD
    ),
    reverse_alignment_tolerance_rad: float = (
        DEFAULT_REVERSE_CONNECTOR_ALIGNMENT_TOLERANCE_RAD
    ),
) -> tuple[EgressConnectorChoice, ...]:
    """Return safe connector choices in forward-first execution order."""

    _finite_pose(start, name="blockage start pose")
    if (
        not math.isfinite(forward_translation_heading_limit_rad)
        or forward_translation_heading_limit_rad <= 0.0
        or forward_translation_heading_limit_rad > math.pi / 2.0
    ):
        raise ValueError(
            "forward translation heading limit must be finite and in (0, pi/2]"
        )
    if (
        not math.isfinite(reverse_alignment_tolerance_rad)
        or reverse_alignment_tolerance_rad <= 0.0
        or reverse_alignment_tolerance_rad > math.pi / 2.0
    ):
        raise ValueError(
            "reverse alignment tolerance must be finite and in (0, pi/2]"
        )
    by_uid = {keepout.candidate_uid: keepout for keepout in keepouts}
    blocker_ids = tuple(dict.fromkeys(str(uid) for uid in blocker_candidate_uids))
    blockers = tuple(by_uid[uid] for uid in blocker_ids if uid in by_uid)
    if blocker_ids and len(blockers) != len(blocker_ids):
        raise ValueError("blockage connector references an unknown blocker")

    for keepout in keepouts:
        start_margin_m = (
            math.hypot(
                start.x_m - keepout.center.x_m,
                start.y_m - keepout.center.y_m,
            )
            - keepout.hard_exclusion_radius_m
        )
        if start_margin_m <= _EPSILON_M:
            raise ValueError(
                "blockage pose is inside the hard stand exclusion envelope: "
                f"candidate={keepout.candidate_uid} "
                f"clearance={start_margin_m:.6f} m"
            )

    choices: list[tuple[int, float, GridCell, EgressConnectorChoice]] = []
    for distance_m, cell, anchor in _candidate_anchor_poses(
        planning_costmap,
        start,
        search_radius_m=search_radius_m,
    ):
        if not segment_is_collision_free(static_costmap, start, anchor):
            continue
        if blockers and not all(
            (
                (anchor.x_m - start.x_m)
                * (start.x_m - blocker.center.x_m)
                + (anchor.y_m - start.y_m)
                * (start.y_m - blocker.center.y_m)
            )
            > _EPSILON_M
            for blocker in blockers
        ):
            continue
        minimum_hard_clearance_m = _minimum_segment_margin_m(
            start,
            anchor,
            keepouts,
            use_hard_envelope=True,
        )
        if minimum_hard_clearance_m <= _EPSILON_M:
            continue

        heading_rad = _segment_heading(start, anchor)
        forward_error_rad = _normalize_angle(heading_rad - start.yaw_rad)
        reverse_error_rad = _normalize_angle(
            heading_rad + math.pi - start.yaw_rad
        )
        if (
            abs(forward_error_rad)
            <= forward_translation_heading_limit_rad + 1.0e-12
        ):
            mode = EGRESS_MODE_FORWARD
            heading_error_rad = forward_error_rad
            mode_rank = 0
        elif abs(reverse_error_rad) <= reverse_alignment_tolerance_rad + 1.0e-12:
            mode = EGRESS_MODE_STRAIGHT_REVERSE
            heading_error_rad = reverse_error_rad
            mode_rank = 1
        else:
            continue
        choice = EgressConnectorChoice(
            anchor=anchor,
            mode=mode,
            connector_distance_m=distance_m,
            connector_heading_error_rad=heading_error_rad,
            minimum_hard_clearance_m=minimum_hard_clearance_m,
        )
        choices.append((mode_rank, distance_m, cell, choice))

    return tuple(item[-1] for item in sorted(choices, key=lambda item: item[:3]))


def find_reverse_transition_anchors(
    planning_costmap: Costmap,
    start: Pose2D,
    connector: EgressConnectorChoice,
    keepouts: Sequence[CircularEscapeKeepout],
    *,
    tracking_tube_radius_m: float,
    search_radius_m: float,
    reverse_alignment_tolerance_rad: float = (
        DEFAULT_REVERSE_CONNECTOR_ALIGNMENT_TOLERANCE_RAD
    ),
) -> tuple[ReverseTransitionChoice, ...]:
    """Find farther anchors that extend, rather than turn, a reverse escape."""

    if connector.mode != EGRESS_MODE_STRAIGHT_REVERSE:
        raise ValueError("reverse transition search requires a reverse connector")
    if (
        not math.isfinite(tracking_tube_radius_m)
        or tracking_tube_radius_m <= 0.0
    ):
        raise ValueError("tracking tube radius must be finite and positive")
    if (
        not math.isfinite(reverse_alignment_tolerance_rad)
        or reverse_alignment_tolerance_rad <= 0.0
        or reverse_alignment_tolerance_rad > math.pi / 2.0
    ):
        raise ValueError(
            "reverse alignment tolerance must be finite and in (0, pi/2]"
        )

    first_heading_rad = _segment_heading(start, connector.anchor)
    transitions: list[tuple[float, GridCell, ReverseTransitionChoice]] = []
    for distance_m, cell, anchor in _candidate_anchor_poses(
        planning_costmap,
        start,
        search_radius_m=search_radius_m,
    ):
        if distance_m <= connector.connector_distance_m + _EPSILON_M:
            continue
        start_heading_rad = _segment_heading(start, anchor)
        continuation_heading_rad = _segment_heading(connector.anchor, anchor)
        reverse_error_rad = _normalize_angle(
            start_heading_rad + math.pi - start.yaw_rad
        )
        continuation_reverse_error_rad = _normalize_angle(
            continuation_heading_rad + math.pi - start.yaw_rad
        )
        if (
            abs(reverse_error_rad) > reverse_alignment_tolerance_rad + 1.0e-12
            or abs(continuation_reverse_error_rad)
            > reverse_alignment_tolerance_rad + 1.0e-12
            or abs(_normalize_angle(continuation_heading_rad - first_heading_rad))
            > reverse_alignment_tolerance_rad + 1.0e-12
        ):
            continue
        if not segment_is_collision_free(
            planning_costmap,
            connector.anchor,
            anchor,
        ):
            continue
        minimum_margin_m = _minimum_segment_margin_m(
            connector.anchor,
            anchor,
            keepouts,
            use_hard_envelope=False,
            tracking_tube_radius_m=tracking_tube_radius_m,
        )
        if minimum_margin_m <= _EPSILON_M:
            continue
        transitions.append(
            (
                distance_m,
                cell,
                ReverseTransitionChoice(
                    anchor=anchor,
                    distance_from_start_m=distance_m,
                    reverse_heading_error_rad=reverse_error_rad,
                    minimum_keepout_tube_clearance_m=minimum_margin_m,
                ),
            )
        )
    return tuple(
        item[-1] for item in sorted(transitions, key=lambda item: item[:2])
    )


def validate_executable_escape_route(
    static_costmap: Costmap,
    planning_costmap: Costmap,
    start: Pose2D,
    connector: EgressConnectorChoice,
    route_poses: Sequence[Pose2D],
    keepouts: Sequence[CircularEscapeKeepout],
    *,
    transition_waypoint_index: int,
    tracking_tube_radius_m: float,
    forward_translation_heading_limit_rad: float = (
        DEFAULT_FORWARD_TRANSLATION_HEADING_LIMIT_RAD
    ),
    reverse_alignment_tolerance_rad: float = (
        DEFAULT_REVERSE_CONNECTOR_ALIGNMENT_TOLERANCE_RAD
    ),
) -> ExecutableEscapeGeometry:
    """Validate the exact prefix and reject reverse routes with early corners."""

    poses = tuple(route_poses)
    if (
        not math.isfinite(tracking_tube_radius_m)
        or tracking_tube_radius_m <= 0.0
    ):
        raise ValueError("tracking tube radius must be finite and positive")
    if (
        not math.isfinite(reverse_alignment_tolerance_rad)
        or reverse_alignment_tolerance_rad <= 0.0
        or reverse_alignment_tolerance_rad > math.pi / 2.0
    ):
        raise ValueError(
            "reverse alignment tolerance must be finite and in (0, pi/2]"
        )
    if len(poses) < 2:
        raise ValueError("blockage escape route has fewer than two waypoints")
    if not _same_position(poses[0], start):
        raise ValueError("blockage escape route lost its exact start pose")
    if not _same_position(poses[1], connector.anchor):
        raise ValueError("blockage escape route lost its safe connector anchor")
    if not segment_is_collision_free(static_costmap, poses[0], poses[1]):
        raise ValueError("blockage escape exact-start connector is not collision-free")
    hard_margin_m = _minimum_segment_margin_m(
        poses[0],
        poses[1],
        keepouts,
        use_hard_envelope=True,
    )
    if hard_margin_m <= _EPSILON_M:
        raise ValueError("blockage escape exact-start connector lacks hard clearance")
    if not 1 <= transition_waypoint_index < len(poses):
        raise ValueError("escape transition waypoint index is outside the route")
    if (
        not math.isfinite(forward_translation_heading_limit_rad)
        or forward_translation_heading_limit_rad <= 0.0
        or forward_translation_heading_limit_rad > math.pi / 2.0
    ):
        raise ValueError(
            "forward translation heading limit must be finite and in (0, pi/2]"
        )

    if connector.mode == EGRESS_MODE_FORWARD:
        if transition_waypoint_index != 1:
            raise ValueError("forward escape transition must be the connector anchor")
        if (
            abs(connector.connector_heading_error_rad)
            > forward_translation_heading_limit_rad + 1.0e-12
        ):
            raise ValueError(
                "forward connector exceeds the translation heading limit"
            )
        return ExecutableEscapeGeometry(
            mode=EGRESS_MODE_FORWARD,
            connector_anchor=poses[1],
            transition_anchor=poses[1],
            transition_waypoint_index=1,
            forward_waypoint_index=2 if len(poses) > 2 else None,
            connector_heading_error_rad=connector.connector_heading_error_rad,
            forward_translation_heading_limit_rad=(
                forward_translation_heading_limit_rad
            ),
            reverse_alignment_tolerance_rad=reverse_alignment_tolerance_rad,
            minimum_connector_hard_clearance_m=hard_margin_m,
            minimum_transition_keepout_tube_clearance_m=None,
        )
    if connector.mode != EGRESS_MODE_STRAIGHT_REVERSE:
        raise ValueError(f"unknown blockage egress mode {connector.mode!r}")
    if transition_waypoint_index < 2:
        raise ValueError(
            "reverse escape requires a farther straight transition anchor"
        )
    if transition_waypoint_index >= len(poses) - 1:
        raise ValueError("reverse escape transition has no outgoing forward segment")

    first_heading_rad = _segment_heading(poses[0], poses[1])
    transition_margin_m = math.inf
    for segment_index in range(transition_waypoint_index):
        segment_start = poses[segment_index]
        segment_end = poses[segment_index + 1]
        segment_heading_rad = _segment_heading(segment_start, segment_end)
        reverse_error_rad = _normalize_angle(
            segment_heading_rad + math.pi - start.yaw_rad
        )
        straight_error_rad = _normalize_angle(
            segment_heading_rad - first_heading_rad
        )
        if (
            abs(reverse_error_rad) > reverse_alignment_tolerance_rad + 1.0e-12
            or abs(straight_error_rad)
            > reverse_alignment_tolerance_rad + 1.0e-12
        ):
            raise ValueError(
                "replacement requires a material multi-corner reverse chain"
            )
        if segment_index == 0:
            continue
        if not segment_is_collision_free(
            planning_costmap,
            segment_start,
            segment_end,
        ):
            raise ValueError("straight reverse transition segment is not collision-free")
        segment_margin_m = _minimum_segment_margin_m(
            segment_start,
            segment_end,
            keepouts,
            use_hard_envelope=False,
            tracking_tube_radius_m=tracking_tube_radius_m,
        )
        if segment_margin_m <= _EPSILON_M:
            raise ValueError(
                "straight reverse transition lacks keepout plus execution-tube clearance"
            )
        transition_margin_m = min(transition_margin_m, segment_margin_m)

    outgoing_start = poses[transition_waypoint_index]
    outgoing_end = poses[transition_waypoint_index + 1]
    if not segment_is_collision_free(
        planning_costmap,
        outgoing_start,
        outgoing_end,
    ):
        raise ValueError("reverse transition outgoing segment is not collision-free")
    outgoing_margin_m = _minimum_segment_margin_m(
        outgoing_start,
        outgoing_end,
        keepouts,
        use_hard_envelope=False,
        tracking_tube_radius_m=tracking_tube_radius_m,
    )
    if outgoing_margin_m <= _EPSILON_M:
        raise ValueError(
            "reverse transition outgoing segment lacks keepout plus "
            "execution-tube clearance"
        )
    transition_margin_m = min(transition_margin_m, outgoing_margin_m)
    return ExecutableEscapeGeometry(
        mode=EGRESS_MODE_STRAIGHT_REVERSE,
        connector_anchor=poses[1],
        transition_anchor=poses[transition_waypoint_index],
        transition_waypoint_index=transition_waypoint_index,
        forward_waypoint_index=transition_waypoint_index + 1,
        connector_heading_error_rad=connector.connector_heading_error_rad,
        forward_translation_heading_limit_rad=(
            forward_translation_heading_limit_rad
        ),
        reverse_alignment_tolerance_rad=reverse_alignment_tolerance_rad,
        minimum_connector_hard_clearance_m=hard_margin_m,
        minimum_transition_keepout_tube_clearance_m=transition_margin_m,
    )
