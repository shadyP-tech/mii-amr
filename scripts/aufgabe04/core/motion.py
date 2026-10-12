"""Pure path controller and physical command admission for one live owner."""
from __future__ import annotations
import math
from dataclasses import dataclass, replace
from typing import Sequence
from .model import Pose2D, Transform2D, Velocity, PlannedLeg, StoppedState, finite, wrap_angle

@dataclass(frozen=True)
class ControllerConfig:
    max_linear_mps: float = 0.055
    max_angular_radps: float = 0.18
    goal_tolerance_m: float = 0.08
    terminal_goal_tolerance_m: float | None = None
    heading_tolerance_rad: float = 0.25
    rotate_gain: float = 1.2
    lookahead_distance_m: float = 0.18
    slow_heading_error_rad: float = 0.75
    stop_heading_error_rad: float = 1.25
    min_linear_speed_scale: float = 0.35
    max_progress_advance_m: float = 0.45
    enforce_heading_corridor: bool = False
    reverse_staging: bool = False
    exact_vertex_pursuit: bool = False


@dataclass(frozen=True)
class CertifiedCornerControlConfig:
    """Fail-closed controls for a material bend in a certified polyline."""

    turn_threshold_rad: float = 0.20
    release_tolerance_m: float = 0.01
    hold_tolerance_m: float = 0.025
    hard_tolerance_m: float = 0.03
    alignment_tolerance_rad: float = 0.10
    max_reacquire_attempts: int = 2

    def __post_init__(self) -> None:
        angular = {
            "turn_threshold_rad": self.turn_threshold_rad,
            "alignment_tolerance_rad": self.alignment_tolerance_rad,
        }
        for name, value in angular.items():
            if not math.isfinite(value) or value <= 0.0 or value > math.pi:
                raise ValueError(f"{name} must be finite and in (0, pi]")
        distances = {
            "release_tolerance_m": self.release_tolerance_m,
            "hold_tolerance_m": self.hold_tolerance_m,
            "hard_tolerance_m": self.hard_tolerance_m,
        }
        for name, value in distances.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.release_tolerance_m > self.hold_tolerance_m:
            raise ValueError(
                "release_tolerance_m must not exceed hold_tolerance_m"
            )
        if self.hold_tolerance_m >= self.hard_tolerance_m:
            raise ValueError(
                "hold_tolerance_m must remain strictly inside hard_tolerance_m"
            )
        if (
            not isinstance(self.max_reacquire_attempts, int)
            or isinstance(self.max_reacquire_attempts, bool)
            or self.max_reacquire_attempts < 0
        ):
            raise ValueError("max_reacquire_attempts must be a non-negative integer")
        if self.alignment_tolerance_rad >= self.turn_threshold_rad:
            raise ValueError(
                "alignment_tolerance_rad must be smaller than "
                "turn_threshold_rad"
            )


@dataclass(frozen=True)
class CertifiedCornerTransitionLatch:
    vertex_index: int
    reacquire_attempts: int = 0
    reacquiring: bool = False


@dataclass(frozen=True)
class CertifiedCornerTransitionDecision:
    step: "ControllerStep | None"
    latch: CertifiedCornerTransitionLatch | None
    failure: str = ""


@dataclass(frozen=True)
class ControllerStep:
    command: Velocity
    target_index: int
    reached_goal: bool
    distance_to_target_m: float
    pursuit_index: int = 0
    controlled_heading_error_rad: float = math.nan
    progress_mode: str = "path_tracking"


def normalize_angle(angle_rad: float) -> float:
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


def route_vertex_turn_angle_rad(
    waypoints: Sequence[Pose2D],
    vertex_index: int,
) -> float:
    """Return the unsigned heading change at one interior route vertex."""

    if not 0 < vertex_index < len(waypoints) - 1:
        return 0.0
    previous = waypoints[vertex_index - 1]
    vertex = waypoints[vertex_index]
    following = waypoints[vertex_index + 1]
    incoming_dx = vertex.x_m - previous.x_m
    incoming_dy = vertex.y_m - previous.y_m
    outgoing_dx = following.x_m - vertex.x_m
    outgoing_dy = following.y_m - vertex.y_m
    if (
        math.hypot(incoming_dx, incoming_dy) <= 1.0e-9
        or math.hypot(outgoing_dx, outgoing_dy) <= 1.0e-9
    ):
        return 0.0
    incoming = math.atan2(incoming_dy, incoming_dx)
    outgoing = math.atan2(outgoing_dy, outgoing_dx)
    return abs(normalize_angle(outgoing - incoming))


def compute_certified_corner_transition(
    pose: Pose2D,
    waypoints: Sequence[Pose2D],
    target_index: int,
    controller_config: ControllerConfig,
    latch: CertifiedCornerTransitionLatch | None = None,
    corner_config: CertifiedCornerControlConfig = CertifiedCornerControlConfig(),
) -> CertifiedCornerTransitionDecision:
    """Approach, align at, and explicitly hand off one material route bend.

    The incoming segment remains the active certificate while the robot
    rotates in place.  Only a pose tightly bound to the exact vertex may
    switch to the outgoing segment, and that switch publishes one zero-command
    control cycle before translation can resume.
    """

    if not waypoints:
        return CertifiedCornerTransitionDecision(None, None)
    if not 0 <= target_index < len(waypoints):
        raise ValueError("target_index is outside the route")
    if latch is not None and latch.vertex_index != target_index:
        latch = None
    turn_angle_rad = route_vertex_turn_angle_rad(waypoints, target_index)
    if turn_angle_rad < corner_config.turn_threshold_rad:
        return CertifiedCornerTransitionDecision(None, None)
    if not 0 < target_index < len(waypoints) - 1:
        return CertifiedCornerTransitionDecision(None, None)

    vertex = waypoints[target_index]
    distance_to_vertex_m = distance(pose, vertex)
    if latch is None and distance_to_vertex_m > corner_config.release_tolerance_m:
        approach = compute_waypoint_command(
            pose,
            waypoints,
            target_index,
            controller_config,
            locked_pursuit_index=target_index,
        )
        return CertifiedCornerTransitionDecision(
            replace(approach, progress_mode="certified_corner_approach"),
            None,
        )
    if latch is None:
        latch = CertifiedCornerTransitionLatch(vertex_index=target_index)

    following = waypoints[target_index + 1]
    outgoing_heading_rad = math.atan2(
        following.y_m - vertex.y_m,
        following.x_m - vertex.x_m,
    )
    heading_error_rad = normalize_angle(outgoing_heading_rad - pose.yaw_rad)
    if distance_to_vertex_m > corner_config.hard_tolerance_m:
        return CertifiedCornerTransitionDecision(
            ControllerStep(
                Velocity(0.0, 0.0),
                target_index,
                False,
                distance_to_vertex_m,
                target_index,
                heading_error_rad,
                "certified_corner_hard_limit_exceeded",
            ),
            latch,
            "certified corner hard tolerance exceeded",
        )
    if latch.reacquiring:
        if distance_to_vertex_m > corner_config.release_tolerance_m:
            reacquire = compute_waypoint_command(
                pose,
                waypoints,
                target_index,
                controller_config,
                locked_pursuit_index=target_index,
            )
            return CertifiedCornerTransitionDecision(
                replace(reacquire, progress_mode="certified_corner_reacquire"),
                latch,
            )
        recovered_latch = replace(latch, reacquiring=False)
        return CertifiedCornerTransitionDecision(
            ControllerStep(
                Velocity(0.0, 0.0),
                target_index,
                False,
                distance_to_vertex_m,
                target_index,
                heading_error_rad,
                "certified_corner_reacquired",
            ),
            recovered_latch,
        )
    if distance_to_vertex_m > corner_config.hold_tolerance_m:
        if latch.reacquire_attempts >= corner_config.max_reacquire_attempts:
            return CertifiedCornerTransitionDecision(
                ControllerStep(
                    Velocity(0.0, 0.0),
                    target_index,
                    False,
                    distance_to_vertex_m,
                    target_index,
                    heading_error_rad,
                    "certified_corner_reacquire_exhausted",
                ),
                latch,
                "certified corner reacquire budget exhausted",
            )
        reacquire_latch = replace(
            latch,
            reacquire_attempts=latch.reacquire_attempts + 1,
            reacquiring=True,
        )
        # Consume one complete zero-command cycle before any corrective
        # translation.  The follower's ordinary certificate check then proves
        # the live pose and exact-vertex chord remain inside the hard tube.
        return CertifiedCornerTransitionDecision(
            ControllerStep(
                Velocity(0.0, 0.0),
                target_index,
                False,
                distance_to_vertex_m,
                target_index,
                heading_error_rad,
                "certified_corner_reacquire_start",
            ),
            reacquire_latch,
        )
    if abs(heading_error_rad) > corner_config.alignment_tolerance_rad:
        angular_z_radps = _clamp(
            heading_error_rad * controller_config.rotate_gain,
            -controller_config.max_angular_radps,
            controller_config.max_angular_radps,
        )
        return CertifiedCornerTransitionDecision(
            ControllerStep(
                Velocity(0.0, angular_z_radps),
                target_index,
                False,
                distance_to_vertex_m,
                target_index,
                heading_error_rad,
                "certified_corner_alignment",
            ),
            latch,
        )

    next_index = target_index + 1
    return CertifiedCornerTransitionDecision(
        ControllerStep(
            Velocity(0.0, 0.0),
            next_index,
            False,
            distance(pose, following),
            next_index,
            heading_error_rad,
            "certified_corner_handoff",
        ),
        None,
    )


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
        # Free/constrained and changed-heading vertices are control-mode
        # handoffs even when early corridor alignment is disabled.  Crossing
        # one by Euclidean-nearest progress can skip the exact vertex that the
        # controller must acknowledge before changing modes.
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
        # A lookahead command must obey the same mode boundaries as progress
        # latching.  Otherwise it can cut past an exact free/finite-yaw vertex
        # while the target clock remains attached to that missed vertex.
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
    *,
    goal_tolerance_m: float | None = None,
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

    effective_goal_tolerance_m = (
        config.goal_tolerance_m
        if goal_tolerance_m is None
        else goal_tolerance_m
    )
    approach_distance_m = max(effective_goal_tolerance_m * 2.0, 1e-6)
    approach_scale = _clamp(target_distance_m / approach_distance_m, 0.25, 1.0)
    return config.max_linear_mps * _clamp(heading_scale, 0.0, 1.0) * approach_scale


def _goal_tolerance_for_index(
    config: ControllerConfig,
    index: int,
    waypoint_count: int,
) -> float:
    if (
        index == waypoint_count - 1
        and config.terminal_goal_tolerance_m is not None
    ):
        return config.terminal_goal_tolerance_m
    return config.goal_tolerance_m


def compute_waypoint_command(
    pose: Pose2D,
    waypoints: Sequence[Pose2D],
    target_index: int,
    config: ControllerConfig,
    *,
    locked_pursuit_index: int | None = None,
) -> ControllerStep:
    if not waypoints:
        return ControllerStep(Velocity(0.0, 0.0), 0, True, 0.0)
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
        if not config.exact_vertex_pursuit:
            index = _closest_index_from(
                pose,
                waypoints,
                index,
                config.max_progress_advance_m,
                config.enforce_heading_corridor,
            )
    target = waypoints[index]
    target_distance = distance(pose, target)
    target_tolerance_m = _goal_tolerance_for_index(
        config,
        index,
        len(waypoints),
    )
    while (
        locked_pursuit_index is None
        and target_distance <= target_tolerance_m
        and index < len(waypoints) - 1
    ):
        next_index = index + 1
        next_target = waypoints[next_index]
        if math.isfinite(target.yaw_rad) != math.isfinite(next_target.yaw_rad):
            # A reverse/forward or free/protected boundary is a control-mode
            # handoff.  Return one explicit zero-command cycle at the finite
            # corridor entry; the next tick may align and move forward.
            return ControllerStep(
                Velocity(0.0, 0.0),
                next_index,
                False,
                distance(pose, next_target),
                next_index,
                math.nan,
                "mode_handoff",
            )
        index = next_index
        target = waypoints[index]
        target_distance = distance(pose, target)
        target_tolerance_m = _goal_tolerance_for_index(
            config,
            index,
            len(waypoints),
        )

    at_final_position = (
        index == len(waypoints) - 1
        and target_distance <= target_tolerance_m
    )
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
                    Velocity(0.0, angular),
                    index,
                    False,
                    target_distance,
                    index,
                    final_heading_error,
                    "terminal_heading",
                )
        return ControllerStep(Velocity(0.0, 0.0), index, True, target_distance, index)

    if config.enforce_heading_corridor and math.isfinite(target.yaw_rad):
        corridor_heading_error = normalize_angle(target.yaw_rad - pose.yaw_rad)
        if abs(corridor_heading_error) > config.heading_tolerance_rad:
            angular = _clamp(
                corridor_heading_error * config.rotate_gain,
                -config.max_angular_radps,
                config.max_angular_radps,
            )
            return ControllerStep(
                Velocity(0.0, angular),
                index,
                False,
                target_distance,
                index,
                corridor_heading_error,
                "heading_corridor",
            )

    pursuit_index = (
        index
        if locked_pursuit_index is not None or config.exact_vertex_pursuit
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
    exact_vertex_alignment = (
        config.exact_vertex_pursuit
        and (
            not math.isfinite(pursuit.yaw_rad)
            or not config.enforce_heading_corridor
        )
        and abs(heading_error) > config.heading_tolerance_rad
    )
    linear = (
        0.0
        if exact_vertex_alignment
        else _linear_speed_for_heading(
            abs(heading_error),
            target_distance,
            config,
            goal_tolerance_m=target_tolerance_m,
        )
    )
    if reversing_stage:
        linear = -linear
    return ControllerStep(
        Velocity(linear, angular),
        index,
        False,
        target_distance,
        pursuit_index,
        heading_error,
        (
            "exact_vertex_alignment"
            if exact_vertex_alignment
            else "path_tracking"
        ),
    )
@dataclass(frozen=True)
class CommandSmoothingConfig:
    """Acceleration limits for ordinary, nonzero controller commands.

    Any requested reduction is applied immediately. Retaining more velocity
    than the latest controller or clearance decision permits could violate a
    certified in-place alignment or obstacle slowdown.
    """

    enabled: bool = True
    max_linear_accel_mps2: float = 0.10
    max_angular_accel_radps2: float = 0.60

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        for name in (
            "max_linear_accel_mps2",
            "max_angular_accel_radps2",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


class CommandSmoother:
    """Limit acceleration without delaying any reduction or zero axis."""

    def __init__(self, config: CommandSmoothingConfig) -> None:
        if not isinstance(config, CommandSmoothingConfig):
            raise ValueError("config must be a CommandSmoothingConfig")
        self.config = config
        self._last = Velocity(0.0, 0.0)
        self._has_last = False

    def reset(self) -> None:
        self._last = Velocity(0.0, 0.0)
        self._has_last = False

    def apply(self, command: Velocity, *, dt_sec: float) -> Velocity:
        shaped = shape_velocity_command(
            command,
            self._last if self._has_last else None,
            dt_sec,
            self.config,
        )
        if shaped.linear_x_mps == 0.0 and shaped.angular_z_radps == 0.0:
            self.reset()
            return shaped
        self._last = shaped
        self._has_last = True
        return shaped


def shape_velocity_command(
    command: Velocity,
    previous: Velocity | None,
    dt_sec: float,
    config: CommandSmoothingConfig = CommandSmoothingConfig(),
) -> Velocity:
    """Limit acceleration while preserving the latest safety envelope."""

    if not _finite_command(command):
        raise ValueError("velocity command must be finite")
    if command.linear_x_mps == 0.0 and command.angular_z_radps == 0.0:
        return Velocity(0.0, 0.0)
    if not config.enabled:
        return command
    if not math.isfinite(dt_sec) or dt_sec < 0.0:
        raise ValueError("command smoothing dt must be finite and non-negative")
    previous_command = previous if previous is not None else Velocity(0.0, 0.0)
    if not _finite_command(previous_command):
        previous_command = Velocity(0.0, 0.0)
    return Velocity(
        _shape_axis(
            previous_command.linear_x_mps,
            command.linear_x_mps,
            dt_sec=dt_sec,
            accel_limit=config.max_linear_accel_mps2,
        ),
        _shape_axis(
            previous_command.angular_z_radps,
            command.angular_z_radps,
            dt_sec=dt_sec,
            accel_limit=config.max_angular_accel_radps2,
        ),
    )


def _finite_command(command: Velocity) -> bool:
    return (
        isinstance(command, Velocity)
        and math.isfinite(command.linear_x_mps)
        and math.isfinite(command.angular_z_radps)
    )


def _shape_axis(
    previous: float,
    target: float,
    *,
    dt_sec: float,
    accel_limit: float,
) -> float:
    if target == previous:
        return target
    if target == 0.0:
        return 0.0
    if previous != 0.0 and math.copysign(1.0, target) != math.copysign(
        1.0, previous
    ):
        # Never cross through zero in one shaped command. The next cycle may
        # accelerate from a known zero state in the new direction.
        return 0.0
    if abs(target) < abs(previous):
        # A limiter may not retain more motion than the raw safety-scaled
        # command currently permits.
        return target
    if dt_sec == 0.0:
        return previous
    max_delta = accel_limit * dt_sec
    delta = max(-max_delta, min(max_delta, target - previous))
    return previous + delta
@dataclass(frozen=True)
class ExecutionRouteCheck:
    ok: bool
    reason: str
    pose_distance_to_segment_m: float
    maximum_chord_distance_to_segment_m: float
    active_segment_start_index: int
    active_segment_end_index: int
    target_index: int
    pursuit_index: int
    tracking_tube_radius_m: float

    def to_log_dict(self) -> dict[str, object]:
        return {
            "reason": self.reason,
            "source": "execution_route_certificate",
            "pose_distance_to_segment_m": self.pose_distance_to_segment_m,
            "maximum_chord_distance_to_segment_m": (
                self.maximum_chord_distance_to_segment_m
            ),
            "active_segment_start_index": self.active_segment_start_index,
            "active_segment_end_index": self.active_segment_end_index,
            "target_index": self.target_index,
            "pursuit_index": self.pursuit_index,
            "tracking_tube_radius_m": self.tracking_tube_radius_m,
            "fail_closed": not self.ok,
        }


def point_to_segment_distance_m(
    point: Pose2D,
    start: Pose2D,
    end: Pose2D,
) -> float:
    dx = end.x_m - start.x_m
    dy = end.y_m - start.y_m
    length_sq = dx * dx + dy * dy
    if length_sq <= 1.0e-18:
        return math.hypot(point.x_m - start.x_m, point.y_m - start.y_m)
    projection = (
        (point.x_m - start.x_m) * dx + (point.y_m - start.y_m) * dy
    ) / length_sq
    projection = max(0.0, min(1.0, projection))
    nearest_x = start.x_m + projection * dx
    nearest_y = start.y_m + projection * dy
    return math.hypot(point.x_m - nearest_x, point.y_m - nearest_y)


def check_execution_route_tube(
    pose: Pose2D,
    waypoints: Sequence[Pose2D],
    *,
    target_index: int,
    pursuit_index: int,
    tracking_tube_radius_m: float,
    chord_sample_spacing_m: float = 0.01,
) -> ExecutionRouteCheck:
    """Check the live pose and pursuit chord against the active route segment."""

    if len(waypoints) < 2:
        raise ValueError("a certified execution route requires at least two waypoints")
    if not 0 <= target_index < len(waypoints):
        raise ValueError("target_index is outside the route")
    if not 0 <= pursuit_index < len(waypoints):
        raise ValueError("pursuit_index is outside the route")
    if not math.isfinite(tracking_tube_radius_m) or tracking_tube_radius_m <= 0.0:
        raise ValueError("tracking_tube_radius_m must be finite and positive")
    if not math.isfinite(chord_sample_spacing_m) or chord_sample_spacing_m <= 0.0:
        raise ValueError("chord_sample_spacing_m must be finite and positive")

    # Exact-vertex execution never commands a chord beyond the target.  Check
    # this independently from the controller implementation so a future
    # configuration regression stops before publishing velocity.
    if pursuit_index != target_index:
        return ExecutionRouteCheck(
            ok=False,
            reason="uncertified pursuit shortcut",
            pose_distance_to_segment_m=math.inf,
            maximum_chord_distance_to_segment_m=math.inf,
            active_segment_start_index=max(0, target_index - 1),
            active_segment_end_index=target_index,
            target_index=target_index,
            pursuit_index=pursuit_index,
            tracking_tube_radius_m=tracking_tube_radius_m,
        )

    segment_end_index = target_index
    segment_start_index = max(0, segment_end_index - 1)
    segment_start = waypoints[segment_start_index]
    segment_end = waypoints[segment_end_index]
    pose_distance = point_to_segment_distance_m(pose, segment_start, segment_end)

    pursuit = waypoints[pursuit_index]
    chord_length = math.hypot(pursuit.x_m - pose.x_m, pursuit.y_m - pose.y_m)
    sample_count = max(1, int(math.ceil(chord_length / chord_sample_spacing_m)))
    maximum_chord_distance = 0.0
    for sample_index in range(sample_count + 1):
        fraction = sample_index / sample_count
        sample = Pose2D(
            pose.x_m + fraction * (pursuit.x_m - pose.x_m),
            pose.y_m + fraction * (pursuit.y_m - pose.y_m),
            pose.yaw_rad,
        )
        maximum_chord_distance = max(
            maximum_chord_distance,
            point_to_segment_distance_m(sample, segment_start, segment_end),
        )

    tolerance = tracking_tube_radius_m + 1.0e-9
    reason = ""
    if pose_distance > tolerance:
        reason = "pose left certified route tube"
    elif maximum_chord_distance > tolerance:
        reason = "pursuit chord left certified route tube"
    return ExecutionRouteCheck(
        ok=not reason,
        reason=reason,
        pose_distance_to_segment_m=pose_distance,
        maximum_chord_distance_to_segment_m=maximum_chord_distance,
        active_segment_start_index=segment_start_index,
        active_segment_end_index=segment_end_index,
        target_index=target_index,
        pursuit_index=pursuit_index,
        tracking_tube_radius_m=tracking_tube_radius_m,
    )

class MotionRejected(ValueError):
    """The sole motion owner must latch stop when a physical gate rejects."""


@dataclass(frozen=True)
class MotionConfig:
    robot_radius_m: float = .105
    collision_margin_m: float = .01
    tracking_margin_m: float = .03
    odom_drift_bound_m: float = .02
    braking_distance_m: float = .015
    uncertainty_sigma_multiplier: float = 2.
    max_linear_mps: float = .055
    max_angular_radps: float = .18
    max_linear_accel_mps2: float = .10
    max_angular_accel_radps2: float = .60
    min_obstacle_distance_m: float = .20
    hard_stop_margin_m: float = .02
    front_slow_distance_m: float = .38
    waypoint_tolerance_m: float = .02
    goal_tolerance_m: float = .03
    route_tube_radius_m: float = .03
    max_map_translation_drift_m: float = .03
    max_map_yaw_drift_rad: float = .05
    stuck_timeout_sec: float = 8.
    terminal_heading_timeout_sec: float = 24.
    motion_floor_mps: float = .005
    max_cycle_gap_sec: float = .35

    def __post_init__(self):
        from dataclasses import fields
        for field in fields(self):
            if finite(getattr(self, field.name), field.name) <= 0:
                raise ValueError("physical motion limits must be positive")
        if self.max_linear_mps > .12 or self.max_angular_radps > .8:
            raise ValueError("motion limits exceed physical exploration bounds")
        if self.route_tube_radius_m > self.tracking_margin_m:
            raise ValueError("execution tube exceeds reserved tracking margin")
        if self.waypoint_tolerance_m >= self.route_tube_radius_m:
            raise ValueError("waypoint release must remain inside route tube")
        if self.robot_radius_m+self.hard_stop_margin_m >= self.min_obstacle_distance_m:
            raise ValueError("LiDAR slowdown must exceed physical footprint stop")
        if self.front_slow_distance_m <= self.min_obstacle_distance_m:
            raise ValueError("invalid slowdown envelope")


@dataclass(frozen=True)
class MotionStep:
    command: Velocity
    complete: bool
    target_index: int


def _maximum_position_sigma(stopped: StoppedState) -> float:
    xx, xy, yy = stopped.position_covariance
    return math.sqrt(max(0., (xx+yy+math.hypot(xx-yy,2*xy))/2))


def require_route_uncertainty(leg: PlannedLeg, stopped: StoppedState, config: MotionConfig):
    """Reject before arming when uncertainty exceeds the route's free disk."""
    if leg.map_id != stopped.map_id or leg.source_epoch != stopped.epoch:
        raise MotionRejected("route has a stale stopped epoch or another map")
    if math.hypot(leg.poses[0].x_m-stopped.pose_map.x_m,leg.poses[0].y_m-stopped.pose_map.y_m)>1e-8:
        raise MotionRejected("route does not begin at its exact stopped pose")
    lever=max(math.hypot(p.x_m-stopped.pose_map.x_m,p.y_m-stopped.pose_map.y_m) for p in leg.poses)
    required=(config.robot_radius_m+config.collision_margin_m+config.tracking_margin_m
        +config.odom_drift_bound_m+config.braking_distance_m
        +config.uncertainty_sigma_multiplier*(_maximum_position_sigma(stopped)
            +math.sqrt(stopped.yaw_variance)*(lever+config.robot_radius_m)))
    if leg.minimum_clearance_m <= required:
        raise MotionRejected("route uncertainty budget exhausted")


def _scan_points(scan, base_from_scan: Transform2D):
    if not isinstance(base_from_scan,Transform2D):
        raise MotionRejected("scan requires its calibrated base transform")
    for name in ('angle_min','angle_max','angle_increment','range_min','range_max'):
        finite(getattr(scan,name), 'scan '+name)
    if not scan.ranges or scan.angle_increment == 0 or not 0 <= scan.range_min < scan.range_max:
        raise MotionRejected("invalid scan geometry")
    step=abs(scan.angle_increment)
    direction=1 if scan.angle_increment>0 else -1
    indexed_span=(len(scan.ranges)-1)*step
    reported_span=direction*(scan.angle_max-scan.angle_min)
    indexed_gap=(math.tau-indexed_span)/step
    reported_gap=(math.tau-reported_span)/step
    # The real LDS has a bounded endpoint metadata discrepancy. Preserve its
    # indexed rays and account for the uncovered seam; never synthesize a ray.
    if (step>math.radians(2) or not .9<=indexed_gap<=2.1
            or not .9<=reported_gap<=1.1 or abs(reported_span-indexed_span)>1.1*step):
        raise MotionRejected("motion requires bounded full circular scan geometry")
    c,s=math.cos(base_from_scan.yaw_rad),math.sin(base_from_scan.yaw_rad)
    points=[]
    valid_beams=0
    invalid=[]
    for i,value in enumerate(scan.ranges):
        valid = (not isinstance(value,bool) and isinstance(value,(int,float))
                 and math.isfinite(value) and scan.range_min <= value <= scan.range_max)
        invalid.append(not valid)
        if not valid:
            continue
        valid_beams+=1
        angle=scan.angle_min+i*scan.angle_increment
        x,y=value*math.cos(angle),value*math.sin(angle)
        points.append((c*x-s*y+base_from_scan.x_m,s*x+c*y+base_from_scan.y_m))
    if valid_beams < max(3,len(scan.ranges)//10):
        raise MotionRejected("scan has insufficient valid range support")
    run=maximum=0
    for bad in invalid+invalid:
        run=run+1 if bad else 0
        maximum=max(maximum,run)
    prefix=next((i for i,bad in enumerate(invalid) if not bad),len(invalid))
    suffix=next((i for i,bad in enumerate(reversed(invalid)) if not bad),len(invalid))
    blind_angle=max(min(maximum,len(invalid))*step,
                    (prefix+suffix+max(0.,indexed_gap-1))*step)
    if blind_angle>math.radians(5)+1e-12:
        raise MotionRejected("scan contains an unobserved angular sector")
    return tuple(points)


def admit_scan_command(command: Velocity, scan, base_from_scan: Transform2D,
                       config: MotionConfig, *, apply_slowdown=True) -> Velocity:
    """Current full footprint plus conservative swept arc before publication."""
    points=_scan_points(scan,base_from_scan)
    radius=config.robot_radius_m+config.hard_stop_margin_m
    if any(math.hypot(x,y)<=radius for x,y in points):
        raise MotionRejected("obstacle inside physical footprint stop")
    v,w=command.linear_x_mps,command.angular_z_radps
    if v == 0:
        return command  # circular footprint is invariant under rotation
    sign=1 if v>0 else -1
    forward=[math.hypot(x,y) for x,y in points if sign*x>0 and abs(math.atan2(y,sign*x))<=math.radians(35)]
    if not forward:
        raise MotionRejected("no valid forward scan support")
    clearance=min(forward)
    if clearance<=config.min_obstacle_distance_m:
        raise MotionRejected("front obstacle stop")
    scale=min(1.,max(0.,(clearance-config.min_obstacle_distance_m)/(config.front_slow_distance_m-config.min_obstacle_distance_m)))
    if apply_slowdown:
        v*=scale
    if 0<abs(v)<config.motion_floor_mps:
        raise MotionRejected("clearance-limited motion below controllable speed")
    # Include one worst scheduling interval plus stopping distance. Sampled
    # arc discs expand by half the travel gap, enclosing the continuous sweep.
    horizon=config.max_cycle_gap_sec+config.braking_distance_m/max(abs(v),config.motion_floor_mps)
    distance=abs(v)*horizon
    count=max(1,math.ceil(distance/.005))
    reserve=radius+distance/(2*count)
    for i in range(count+1):
        t=horizon*i/count
        x=v*t if abs(w)<1e-9 else v*math.sin(w*t)/w
        y=0. if abs(w)<1e-9 else v*(1-math.cos(w*t))/w
        if any(math.hypot(px-x,py-y)<=reserve for px,py in points):
            raise MotionRejected("obstacle intersects braking sweep")
    return Velocity(v,w)


class MotionController:
    """One immutable leg; pure math owned by the ROS publisher's control loop."""
    def __init__(self, leg: PlannedLeg, stopped: StoppedState, config: MotionConfig):
        if not isinstance(leg,PlannedLeg) or not isinstance(stopped,StoppedState):
            raise MotionRejected("typed route and stopped capture required")
        require_route_uncertainty(leg,stopped,config)
        self.leg,self.stopped,self.config=leg,stopped,config
        self.waypoints=tuple(stopped.map_from_odom.inverse_apply(p) for p in leg.poses)
        self.controller=ControllerConfig(max_linear_mps=config.max_linear_mps,
            max_angular_radps=config.max_angular_radps,goal_tolerance_m=config.waypoint_tolerance_m,
            terminal_goal_tolerance_m=config.goal_tolerance_m,
            heading_tolerance_rad=math.radians(3) if leg.kind!='coverage' else .25,
            exact_vertex_pursuit=True)
        self.smoother=CommandSmoother(CommandSmoothingConfig(True,config.max_linear_accel_mps2,config.max_angular_accel_radps2))
        self.index=0
        self.corner_latch=None
        self.started=None
        self.last_time=None
        self.progress_time=None
        self.best_distance=math.inf
        self.best_heading=math.inf
        self.progress_key=None
        self.heading_started=None
        self.completed=False
        self.fault=None
        length=sum(math.hypot(b.x_m-a.x_m,b.y_m-a.y_m) for a,b in zip(self.waypoints,self.waypoints[1:]))
        turns=sum(abs(wrap_angle(math.atan2(c.y_m-b.y_m,c.x_m-b.x_m)-math.atan2(b.y_m-a.y_m,b.x_m-a.x_m)))
                  for a,b,c in zip(self.waypoints,self.waypoints[1:],self.waypoints[2:]))
        self.timeout=max(45.,1.25*(length/config.motion_floor_mps+(turns+2*math.pi)/config.max_angular_radps)+5.)

    def step(self, pose_odom: Pose2D, scan, now: float, map_from_odom: Transform2D,
             *, base_from_scan: Transform2D) -> MotionStep:
        if self.fault:
            raise MotionRejected(self.fault)
        try:
            return self._step(pose_odom,scan,now,map_from_odom,base_from_scan)
        except (ValueError,TypeError,AttributeError,OverflowError) as exc:
            self.smoother.reset()
            self.fault=str(exc)
            raise MotionRejected(self.fault) from exc

    def _step(self, pose, scan, now, transform, scan_transform):
        finite(now,'control clock')
        finite(pose.yaw_rad,'execution yaw')
        if not isinstance(transform,Transform2D):
            raise MotionRejected("current map transform unavailable")
        if self.last_time is not None and (now<=self.last_time or now-self.last_time>self.config.max_cycle_gap_sec):
            raise MotionRejected("control cycle stale or clock reversed")
        dt=.1 if self.last_time is None else now-self.last_time
        self.last_time=now
        if self.started is None:
            self.started=self.progress_time=now
            if math.hypot(pose.x_m-self.stopped.pose_odom.x_m,pose.y_m-self.stopped.pose_odom.y_m)>.01 or abs(wrap_angle(pose.yaw_rad-self.stopped.pose_odom.yaw_rad))>math.radians(2):
                raise MotionRejected("robot moved since stopped route admission")
        if now-self.started>self.timeout:
            raise MotionRejected("route execution deadline expired")
        frozen=self.stopped.map_from_odom
        # Anchor continuity at the actual route, rather than the arbitrary odom origin.
        frozen_pose=frozen.apply(pose)
        live_pose=transform.apply(pose)
        if (math.hypot(live_pose.x_m-frozen_pose.x_m,live_pose.y_m-frozen_pose.y_m)>self.config.max_map_translation_drift_m
                or abs(wrap_angle(transform.yaw_rad-frozen.yaw_rad))>self.config.max_map_yaw_drift_rad):
            raise MotionRejected("map to odom changed during motion")
        if self.completed:
            return MotionStep(Velocity(),True,self.index)
        corner=compute_certified_corner_transition(pose,self.waypoints,self.index,self.controller,self.corner_latch,
                    CertifiedCornerControlConfig(hard_tolerance_m=self.config.route_tube_radius_m))
        if corner.failure:
            raise MotionRejected(corner.failure)
        self.corner_latch=corner.latch
        step=corner.step or compute_waypoint_command(pose,self.waypoints,self.index,self.controller)
        check=check_execution_route_tube(pose,self.waypoints,target_index=step.target_index,
            pursuit_index=step.pursuit_index,tracking_tube_radius_m=self.config.route_tube_radius_m)
        if not check.ok:
            raise MotionRejected(check.reason)
        self.index=step.target_index
        if step.reached_goal:
            self.completed=True
            self.smoother.reset()
            return MotionStep(Velocity(),True,self.index)
        key=(step.target_index,step.progress_mode)
        heading=abs(step.controlled_heading_error_rad)
        if key!=self.progress_key:
            self.progress_key=key
            self.best_distance=step.distance_to_target_m
            self.best_heading=heading
            self.progress_time=now
        elif (step.distance_to_target_m<self.best_distance-.005
              or math.isfinite(heading) and heading<self.best_heading-.03):
            self.best_distance=min(self.best_distance,step.distance_to_target_m)
            self.best_heading=min(self.best_heading,heading)
            self.progress_time=now
        elif now-self.progress_time>self.config.stuck_timeout_sec:
            raise MotionRejected("motion made no bounded progress")
        if step.progress_mode in ('terminal_heading','final_heading'):
            self.heading_started=now if self.heading_started is None else self.heading_started
            if now-self.heading_started>self.config.terminal_heading_timeout_sec:
                raise MotionRejected("terminal heading deadline expired")
        else:
            self.heading_started=None
        command=admit_scan_command(step.command,scan,scan_transform,self.config)
        command=self.smoother.apply(command,dt_sec=dt)
        command=admit_scan_command(command,scan,scan_transform,self.config,apply_slowdown=False)
        # Smoothing cannot retain an unsafe axis, and is checked independently.
        if (abs(command.linear_x_mps)>self.config.max_linear_mps+1e-12
                or abs(command.angular_z_radps)>self.config.max_angular_radps+1e-12):
            raise MotionRejected("velocity exceeds configured physical limit")
        return MotionStep(command,False,self.index)
