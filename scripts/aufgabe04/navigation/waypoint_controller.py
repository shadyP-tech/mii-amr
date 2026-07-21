"""Small pure waypoint-control helpers for Aufgabe 04."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

from scripts.aufgabe04.navigation.models import Pose2D


@dataclass(frozen=True)
class VelocityCommand:
    linear_x_mps: float
    angular_z_radps: float


@dataclass(frozen=True)
class ControllerConfig:
    max_linear_mps: float = 0.055
    max_angular_radps: float = 0.18
    goal_tolerance_m: float = 0.08
    heading_tolerance_rad: float = 0.25
    rotate_gain: float = 1.2
    lookahead_distance_m: float = 0.18
    slow_heading_error_rad: float = 0.75
    stop_heading_error_rad: float = 1.25
    min_linear_speed_scale: float = 0.35
    max_progress_advance_m: float = 0.45
    enforce_heading_corridor: bool = False
    reverse_staging: bool = False


@dataclass(frozen=True)
class StartEgressControlConfig:
    """Conservative controls for leaving a certified keepout start cell.

    The ordinary controller deliberately blends translation into fairly large
    heading errors so it can follow smooth routes.  A start-egress segment is
    different: its clearance certificate applies to the exact segment leading
    to the locked first vertex.  Translating before the robot is aligned with
    that segment can consume the certificate's small clearance margin.
    """

    alignment_tolerance_rad: float = 0.10
    max_linear_mps: float = 0.03

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.alignment_tolerance_rad)
            or self.alignment_tolerance_rad <= 0.0
            or self.alignment_tolerance_rad > math.pi / 2.0
        ):
            raise ValueError(
                "alignment_tolerance_rad must be finite and in (0, pi/2]"
            )
        if not math.isfinite(self.max_linear_mps) or self.max_linear_mps <= 0.0:
            raise ValueError("max_linear_mps must be finite and positive")


@dataclass(frozen=True)
class ControllerStep:
    command: VelocityCommand
    target_index: int
    reached_goal: bool
    distance_to_target_m: float
    pursuit_index: int = 0
    controlled_heading_error_rad: float = math.nan


def normalize_angle(angle_rad: float) -> float:
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


def distance(a: Pose2D, b: Pose2D) -> float:
    return math.hypot(a.x_m - b.x_m, a.y_m - b.y_m)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def first_heading_corridor_index(waypoints: Sequence[Pose2D]) -> int | None:
    """Return the first finite-yaw waypoint of a protected approach."""

    return next(
        (index for index, waypoint in enumerate(waypoints) if math.isfinite(waypoint.yaw_rad)),
        None,
    )


def reverse_staging_is_preferred(
    pose: Pose2D, waypoints: Sequence[Pose2D]
) -> bool:
    """Choose one stable travel direction for the pre-corridor segment.

    A camera approach can commit while the robot is already inside the outer
    corridor radius.  Driving forward to the outer entry then points the body
    almost opposite the required inward corridor yaw.  Reversing that staging
    segment keeps the camera facing the stand and minimizes the heading change
    at the forward-only protected corridor handoff.
    """

    corridor_index = first_heading_corridor_index(waypoints)
    if corridor_index is None or corridor_index <= 0:
        return False
    entry = waypoints[corridor_index]
    dx = entry.x_m - pose.x_m
    dy = entry.y_m - pose.y_m
    if math.hypot(dx, dy) <= 1.0e-9:
        previous = waypoints[corridor_index - 1]
        dx = entry.x_m - previous.x_m
        dy = entry.y_m - previous.y_m
    if math.hypot(dx, dy) <= 1.0e-9:
        return False
    forward_heading = math.atan2(dy, dx)
    reverse_heading = normalize_angle(forward_heading + math.pi)
    forward_error = abs(normalize_angle(entry.yaw_rad - forward_heading))
    reverse_error = abs(normalize_angle(entry.yaw_rad - reverse_heading))
    return reverse_error + 1.0e-9 < forward_error


def _closest_index_from(
    pose: Pose2D,
    waypoints: Sequence[Pose2D],
    start_index: int,
    max_advance_m: float,
    enforce_heading_corridor: bool,
) -> int:
    closest_index = start_index
    closest_distance = distance(pose, waypoints[start_index])
    cumulative = 0.0
    for index in range(start_index + 1, len(waypoints)):
        # Finite-yaw runs are protected approach corridors.  Never advance a
        # progress cursor across their entry (or across a new constrained
        # heading) merely because a later point is Euclidean-nearer.
        if enforce_heading_corridor:
            previous_yaw = waypoints[index - 1].yaw_rad
            candidate_yaw = waypoints[index].yaw_rad
            if math.isfinite(previous_yaw) != math.isfinite(candidate_yaw):
                break
            if (
                math.isfinite(previous_yaw)
                and abs(normalize_angle(candidate_yaw - previous_yaw)) > 1.0e-6
            ):
                break
        cumulative += distance(waypoints[index - 1], waypoints[index])
        advance_limit_exceeded = (
            max_advance_m > 0.0 and cumulative > max_advance_m
        )
        is_immediate_successor = index == start_index + 1
        if advance_limit_exceeded and not is_immediate_successor:
            break
        candidate_distance = distance(pose, waypoints[index])
        if candidate_distance < closest_distance:
            closest_index = index
            closest_distance = candidate_distance
        if advance_limit_exceeded:
            # ``max_advance_m`` is an anti-skip window, not a maximum legal
            # route-segment length.  The immediate successor must remain
            # eligible even when a sparse, collision-certified segment is
            # longer than the window, but no waypoint beyond it may be used.
            break
    return closest_index


def _lookahead_index(
    pose: Pose2D,
    waypoints: Sequence[Pose2D],
    start_index: int,
    lookahead_distance_m: float,
    enforce_heading_corridor: bool,
) -> int:
    if lookahead_distance_m <= 0.0:
        return start_index
    # Include the distance from the robot to the current target.  Starting at
    # zero let lookahead skip a target that was still far away.
    cumulative = distance(pose, waypoints[start_index])
    if cumulative >= lookahead_distance_m:
        return start_index
    for index in range(start_index + 1, len(waypoints)):
        if enforce_heading_corridor:
            previous_yaw = waypoints[index - 1].yaw_rad
            candidate_yaw = waypoints[index].yaw_rad
            if math.isfinite(previous_yaw) != math.isfinite(candidate_yaw):
                return index - 1
            if (
                math.isfinite(previous_yaw)
                and abs(normalize_angle(candidate_yaw - previous_yaw)) > 1.0e-6
            ):
                return index - 1
        cumulative += distance(waypoints[index - 1], waypoints[index])
        if cumulative >= lookahead_distance_m:
            return index
    return len(waypoints) - 1


def _should_latch_pursuit_progress(
    waypoints: Sequence[Pose2D],
    target_index: int,
    pursuit_index: int,
) -> bool:
    """Make ordinary pure-pursuit progress monotonic.

    The pursuit waypoint is the point the controller already commands toward.
    Persisting it as progress ensures the next cycle cannot revert to an older
    waypoint merely because the robot left that waypoint's lookahead circle.
    This is essential when a segment is longer than twice the lookahead radius:
    the radius boundary is crossed before the Euclidean-nearest midpoint.

    Free/constrained and changed-heading boundaries remain explicit handoffs
    and are never latched across, independent of the caller's corridor mode.
    """

    if pursuit_index <= target_index:
        return False
    for index in range(target_index + 1, pursuit_index + 1):
        previous_yaw = waypoints[index - 1].yaw_rad
        candidate_yaw = waypoints[index].yaw_rad
        previous_has_heading = math.isfinite(previous_yaw)
        candidate_has_heading = math.isfinite(candidate_yaw)
        if previous_has_heading != candidate_has_heading:
            return False
        if (
            previous_has_heading
            and abs(normalize_angle(candidate_yaw - previous_yaw)) > 1.0e-6
        ):
            return False
    return True


def _linear_speed_for_heading(
    heading_error_abs: float,
    target_distance_m: float,
    config: ControllerConfig,
) -> float:
    if heading_error_abs >= config.stop_heading_error_rad:
        return 0.0

    min_scale = _clamp(config.min_linear_speed_scale, 0.0, 1.0)
    slow_heading = max(config.slow_heading_error_rad, 1e-6)
    stop_heading = max(config.stop_heading_error_rad, slow_heading + 1e-6)
    if heading_error_abs <= slow_heading:
        heading_fraction = heading_error_abs / slow_heading
        heading_scale = 1.0 - heading_fraction * (1.0 - min_scale)
    else:
        taper_fraction = (heading_error_abs - slow_heading) / (stop_heading - slow_heading)
        heading_scale = min_scale * (1.0 - taper_fraction)

    approach_distance_m = max(config.goal_tolerance_m * 2.0, 1e-6)
    approach_scale = _clamp(target_distance_m / approach_distance_m, 0.25, 1.0)
    return config.max_linear_mps * _clamp(heading_scale, 0.0, 1.0) * approach_scale


def compute_waypoint_command(
    pose: Pose2D,
    waypoints: Sequence[Pose2D],
    target_index: int,
    config: ControllerConfig,
    *,
    locked_pursuit_index: int | None = None,
) -> ControllerStep:
    if not waypoints:
        return ControllerStep(VelocityCommand(0.0, 0.0), 0, True, 0.0)
    if locked_pursuit_index is not None:
        if not isinstance(locked_pursuit_index, int) or isinstance(
            locked_pursuit_index, bool
        ):
            raise ValueError("locked_pursuit_index must be an integer")
        if not 0 <= locked_pursuit_index < len(waypoints):
            raise ValueError("locked_pursuit_index is outside the route")
        index = locked_pursuit_index
    else:
        index = min(max(target_index, 0), len(waypoints) - 1)
        index = _closest_index_from(
            pose,
            waypoints,
            index,
            config.max_progress_advance_m,
            config.enforce_heading_corridor,
        )
    target = waypoints[index]
    target_distance = distance(pose, target)
    while (
        locked_pursuit_index is None
        and target_distance <= config.goal_tolerance_m
        and index < len(waypoints) - 1
    ):
        next_index = index + 1
        next_target = waypoints[next_index]
        if math.isfinite(target.yaw_rad) != math.isfinite(next_target.yaw_rad):
            # A reverse/forward or free/protected boundary is a control-mode
            # handoff.  Return one explicit zero-command cycle at the finite
            # corridor entry; the next tick may align and move forward.
            return ControllerStep(
                VelocityCommand(0.0, 0.0),
                next_index,
                False,
                distance(pose, next_target),
                next_index,
            )
        index = next_index
        target = waypoints[index]
        target_distance = distance(pose, target)

    at_final_position = index == len(waypoints) - 1 and target_distance <= config.goal_tolerance_m
    if at_final_position:
        # Planned station-approach routes carry a finite yaw on their final
        # waypoint.  Reaching only the x/y cell is insufficient for camera/QR
        # alignment, so finish with an in-place heading correction.  Ordinary
        # transit routes leave yaw as NaN and retain position-only completion.
        if math.isfinite(target.yaw_rad):
            final_heading_error = normalize_angle(target.yaw_rad - pose.yaw_rad)
            if abs(final_heading_error) > config.heading_tolerance_rad:
                angular = _clamp(
                    final_heading_error * config.rotate_gain,
                    -config.max_angular_radps,
                    config.max_angular_radps,
                )
                return ControllerStep(
                    VelocityCommand(0.0, angular),
                    index,
                    False,
                    target_distance,
                    index,
                    final_heading_error,
                )
        return ControllerStep(VelocityCommand(0.0, 0.0), index, True, target_distance, index)

    if config.enforce_heading_corridor and math.isfinite(target.yaw_rad):
        corridor_heading_error = normalize_angle(target.yaw_rad - pose.yaw_rad)
        if abs(corridor_heading_error) > config.heading_tolerance_rad:
            angular = _clamp(
                corridor_heading_error * config.rotate_gain,
                -config.max_angular_radps,
                config.max_angular_radps,
            )
            return ControllerStep(
                VelocityCommand(0.0, angular),
                index,
                False,
                target_distance,
                index,
                corridor_heading_error,
            )

    pursuit_index = (
        index
        if locked_pursuit_index is not None
        else _lookahead_index(
            pose,
            waypoints,
            index,
            config.lookahead_distance_m,
            config.enforce_heading_corridor,
        )
    )
    if _should_latch_pursuit_progress(
        waypoints,
        index,
        pursuit_index,
    ):
        # The waypoint is already the certified pursuit vertex.  Report it as
        # progress now so the follower persists it on the next control cycle.
        index = pursuit_index
        target = waypoints[index]
        target_distance = distance(pose, target)
    pursuit = waypoints[pursuit_index]
    heading = math.atan2(pursuit.y_m - pose.y_m, pursuit.x_m - pose.x_m)
    corridor_index = first_heading_corridor_index(waypoints)
    reversing_stage = (
        config.reverse_staging
        and corridor_index is not None
        and pursuit_index < corridor_index
    )
    controlled_heading = (
        normalize_angle(heading + math.pi) if reversing_stage else heading
    )
    heading_error = normalize_angle(controlled_heading - pose.yaw_rad)
    angular = _clamp(
        heading_error * config.rotate_gain,
        -config.max_angular_radps,
        config.max_angular_radps,
    )
    linear = _linear_speed_for_heading(abs(heading_error), target_distance, config)
    if reversing_stage:
        linear = -linear
    return ControllerStep(
        VelocityCommand(linear, angular),
        index,
        False,
        target_distance,
        pursuit_index,
        heading_error,
    )


def compute_start_egress_vertex_command(
    pose: Pose2D,
    waypoints: Sequence[Pose2D],
    waypoint_index: int,
    config: ControllerConfig,
    *,
    reach_tolerance_m: float = 0.02,
    egress_config: StartEgressControlConfig = StartEgressControlConfig(),
) -> ControllerStep | None:
    """Pursue one certified egress vertex without lookahead or advancement.

    ``None`` means the exact vertex is inside the tight release tolerance. The
    caller owns clearing the route-scoped lock and issuing a zero cycle before
    returning to ordinary pure-pursuit behavior.
    """

    if not math.isfinite(reach_tolerance_m) or reach_tolerance_m <= 0.0:
        raise ValueError("reach_tolerance_m must be finite and positive")
    if not isinstance(waypoint_index, int) or isinstance(waypoint_index, bool):
        raise ValueError("waypoint_index must be an integer")
    if not 0 <= waypoint_index < len(waypoints):
        raise ValueError("waypoint_index is outside the route")
    if distance(pose, waypoints[waypoint_index]) <= reach_tolerance_m:
        return None
    step = compute_waypoint_command(
        pose,
        waypoints,
        waypoint_index,
        replace(
            config,
            goal_tolerance_m=min(config.goal_tolerance_m, reach_tolerance_m),
            lookahead_distance_m=0.0,
            max_progress_advance_m=0.0,
            reverse_staging=False,
        ),
        locked_pursuit_index=waypoint_index,
    )
    # The generic controller starts tapering forward motion only at its broad
    # stop-heading threshold (1.25 rad by default).  For a keepout egress that
    # would let the robot cut diagonally away from the continuously certified
    # segment.  Hold position until the locked vertex is tightly aligned, then
    # translate slowly while retaining the normal angular correction.
    heading_error = abs(step.controlled_heading_error_rad)
    linear = 0.0
    if (
        math.isfinite(heading_error)
        and heading_error <= egress_config.alignment_tolerance_rad
    ):
        linear = min(step.command.linear_x_mps, egress_config.max_linear_mps)
    return replace(
        step,
        command=VelocityCommand(linear, step.command.angular_z_radps),
    )


def compute_join_anchor_command(
    pose: Pose2D,
    anchor: Pose2D,
    config: ControllerConfig,
    *,
    join_tolerance_m: float = 0.01,
) -> ControllerStep:
    """Pursue only a newly adopted route's collision-certified start pose."""

    if not math.isfinite(join_tolerance_m) or join_tolerance_m <= 0.0:
        raise ValueError("join_tolerance_m must be finite and positive")
    return compute_waypoint_command(
        pose,
        (anchor,),
        0,
        replace(
            config,
            goal_tolerance_m=join_tolerance_m,
            lookahead_distance_m=0.0,
            max_progress_advance_m=0.0,
        ),
    )
