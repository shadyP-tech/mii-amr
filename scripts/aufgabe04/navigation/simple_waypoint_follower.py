"""Minimal ROS2 waypoint follower for one Aufgabe 04 route segment."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable, Sequence

from scripts.aufgabe04.navigation.dynamic_route_handoff import (
    RouteUpdate,
    RouteUpdateKind,
)
from scripts.aufgabe04.navigation.execution_route_certificate import (
    ExecutionRouteCheck,
    check_execution_route_tube,
)
from scripts.aufgabe04.navigation.follower_models import FollowerResult
from scripts.aufgabe04.navigation.follower_safety import (
    NO_VALID_FRONT_SECTOR_SCAN_RANGES,
    OBSTACLE_TOO_CLOSE,
    cmd_vel_ownership_failure,
    front_sector_decision,
    initial_pose_failure,
    linear_scale_for_front_clearance,
    message_freshness_failure,
    obstacle_decision,
    stuck_progress_failure,
    waypoint_timeout_failure,
)
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.ros_runtime_config import ResolvedRuntimeConfig
from scripts.aufgabe04.navigation.viewpoint_sampling_contract import (
    DEFAULT_VIEWPOINT_SAMPLING_TARGET_DISTANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M,
    INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M,
    ViewpointSamplingHoldConfig,
    viewpoint_sampling_hold_metrics,
)
from scripts.aufgabe04.navigation.waypoint_controller import (
    CertifiedCornerControlConfig,
    CertifiedCornerTransitionDecision,
    CertifiedCornerTransitionLatch,
    ControllerConfig,
    ControllerStep,
    StartEgressControlConfig,
    VelocityCommand,
    compute_certified_corner_transition,
    compute_join_anchor_command,
    compute_start_egress_vertex_command,
    compute_waypoint_command,
    normalize_angle,
    reverse_staging_is_preferred,
)


STALE_TF_RECOVERY_MAX_DURATION_SEC = 0.18
STALE_TF_RECOVERY_MAX_CALLBACKS = 48
STALE_TF_RECOVERY_SPIN_TIMEOUT_SEC = 0.005
# Gazebo odometry quaternions are unit normalized.  A 1e-3 norm tolerance
# admits only floating-point serialization drift; it is not a normalization
# or malformed-pose repair path.
SIMULATION_ODOM_FALLBACK_QUATERNION_NORM_TOLERANCE = 1.0e-3
SIMULATION_ODOM_FALLBACK_SOURCE = (
    "simulation_direct_odom_after_tf_retry"
)
# Keep capacity for /clock, scan, and odometry even when their callback
# groups are simultaneously runnable.  Production TF subscriptions live on a
# separate node/executor so they cannot be starved by this executor.
FOLLOWER_EXECUTOR_NUM_THREADS = 4
TF_LISTENER_NODE_NAME = "aufgabe04_waypoint_follower_tf_listener"
CALLBACK_SERVICE_CALLER_SPIN = "caller_spin"
CALLBACK_SERVICE_BACKGROUND_EXECUTOR = "background_executor"
INTERMEDIATE_TERMINAL_HEADING_HOLD_EXCEEDED = (
    "intermediate_terminal_heading_hold_tolerance_exceeded"
)


try:  # pragma: no cover - exercised on ROS hosts.
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.duration import Duration
    from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer, TransformException, TransformListener
except ImportError:  # pragma: no cover - keeps offline tests ROS-free.
    rclpy = None
    Twist = None
    LaserScan = None
    Odometry = None
    Duration = None
    MultiThreadedExecutor = None
    SingleThreadedExecutor = None
    Node = object
    Parameter = None
    qos_profile_sensor_data = None
    Time = None
    Buffer = None
    TransformException = Exception
    TransformListener = None


@dataclass(frozen=True)
class FollowerConfig:
    controller: ControllerConfig
    min_obstacle_distance_m: float = 0.20
    front_obstacle_slow_distance_m: float = 0.38
    front_obstacle_sector_rad: float = math.radians(35.0)
    max_scan_age_sec: float = 1.0
    max_odom_age_sec: float = 1.0
    max_tf_age_sec: float = 1.0
    max_future_timestamp_sec: float = 0.25
    allow_simulation_odom_after_stale_tf: bool = False
    initial_sensor_wait_sec: float = 2.0
    waypoint_timeout_sec: float = 45.0
    stuck_timeout_sec: float = 8.0
    stuck_progress_epsilon_m: float = 0.03
    stuck_heading_progress_epsilon_rad: float = 0.10
    initial_distance_limit_m: float = 0.35
    control_rate_hz: float = 10.0
    allowed_cmd_vel_publishers: Sequence[str] = ()
    dynamic_route_refresh_sec: float = 0.0
    dynamic_join_tolerance_m: float = 0.01
    start_egress_waypoint_tolerance_m: float = 0.02
    start_egress_alignment_tolerance_rad: float = 0.10
    start_egress_max_linear_mps: float = 0.03
    initial_start_egress_waypoint_index: int | None = None
    initial_start_join_clearance_m: float | None = None
    initial_route_kind: str = ""
    axis_acquisition_wait_timeout_sec: float = 12.0
    viewpoint_sampling_timeout_sec: float = 30.0
    viewpoint_sampling_target_timeout_sec: float = 30.0
    viewpoint_sampling_goal_tolerance_m: float = (
        INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M
    )
    viewpoint_sampling_terminal_heading_hold_tolerance_m: float = (
        INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M
    )
    viewpoint_sampling_target_distance_m: float = (
        DEFAULT_VIEWPOINT_SAMPLING_TARGET_DISTANCE_M
    )
    viewpoint_sampling_terminal_heading_target_envelope_radius_m: float = (
        INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M
    )
    viewpoint_sampling_heading_tolerance_rad: float = math.radians(5.0)
    physical_waypoint_tolerance_m: float = 0.02
    physical_goal_tolerance_m: float = 0.03
    certified_route_tube_radius_m: float = 0.03
    certified_route_chord_sample_spacing_m: float = 0.01
    certified_corner_turn_threshold_rad: float = 0.20
    certified_corner_release_tolerance_m: float = 0.01
    certified_corner_hold_tolerance_m: float = 0.025
    certified_corner_alignment_tolerance_rad: float = 0.10

    def __post_init__(self) -> None:
        if not isinstance(
            self.allow_simulation_odom_after_stale_tf,
            bool,
        ):
            raise ValueError(
                "allow_simulation_odom_after_stale_tf must be boolean"
            )
        if (
            not math.isfinite(self.dynamic_join_tolerance_m)
            or self.dynamic_join_tolerance_m <= 0.0
        ):
            raise ValueError("dynamic_join_tolerance_m must be finite and positive")
        if (
            not math.isfinite(self.start_egress_waypoint_tolerance_m)
            or self.start_egress_waypoint_tolerance_m <= 0.0
        ):
            raise ValueError(
                "start_egress_waypoint_tolerance_m must be finite and positive"
            )
        # Validate the paired egress controls through their pure value object
        # so the ROS wrapper and offline controller tests share one contract.
        StartEgressControlConfig(
            alignment_tolerance_rad=self.start_egress_alignment_tolerance_rad,
            max_linear_mps=self.start_egress_max_linear_mps,
        )
        if self.initial_start_egress_waypoint_index is not None and (
            not isinstance(self.initial_start_egress_waypoint_index, int)
            or isinstance(self.initial_start_egress_waypoint_index, bool)
            or self.initial_start_egress_waypoint_index <= 0
        ):
            raise ValueError(
                "initial_start_egress_waypoint_index must be a positive integer"
            )
        if self.initial_start_join_clearance_m is not None and (
            not math.isfinite(self.initial_start_join_clearance_m)
            or self.initial_start_join_clearance_m <= 0.0
        ):
            raise ValueError(
                "initial_start_join_clearance_m must be finite and positive"
            )
        if (self.initial_start_egress_waypoint_index is None) != (
            self.initial_start_join_clearance_m is None
        ):
            raise ValueError(
                "initial start-egress lock and start-join clearance must be paired"
            )
        if (
            not math.isfinite(self.stuck_heading_progress_epsilon_rad)
            or self.stuck_heading_progress_epsilon_rad <= 0.0
        ):
            raise ValueError(
                "stuck_heading_progress_epsilon_rad must be finite and positive"
            )
        if (
            not math.isfinite(self.axis_acquisition_wait_timeout_sec)
            or self.axis_acquisition_wait_timeout_sec <= 0.0
        ):
            raise ValueError(
                "axis_acquisition_wait_timeout_sec must be finite and positive"
            )
        if (
            not math.isfinite(self.viewpoint_sampling_timeout_sec)
            or self.viewpoint_sampling_timeout_sec <= 0.0
        ):
            raise ValueError(
                "viewpoint_sampling_timeout_sec must be finite and positive"
            )
        if (
            not math.isfinite(self.viewpoint_sampling_target_timeout_sec)
            or self.viewpoint_sampling_target_timeout_sec <= 0.0
        ):
            raise ValueError(
                "viewpoint_sampling_target_timeout_sec must be finite and positive"
            )
        if (
            not math.isfinite(self.viewpoint_sampling_goal_tolerance_m)
            or self.viewpoint_sampling_goal_tolerance_m <= 0.0
        ):
            raise ValueError(
                "viewpoint_sampling_goal_tolerance_m must be finite and positive"
            )
        if (
            not math.isfinite(
                self.viewpoint_sampling_terminal_heading_hold_tolerance_m
            )
            or self.viewpoint_sampling_terminal_heading_hold_tolerance_m <= 0.0
            or self.viewpoint_sampling_terminal_heading_hold_tolerance_m
            > INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M
            or self.viewpoint_sampling_terminal_heading_hold_tolerance_m
            < min(
                self.viewpoint_sampling_goal_tolerance_m,
                INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M,
            )
        ):
            raise ValueError(
                "viewpoint_sampling_terminal_heading_hold_tolerance_m must be "
                "finite, no smaller than the effective sampling tolerance, "
                "and no greater than 0.020"
            )
        if (
            not math.isfinite(self.viewpoint_sampling_target_distance_m)
            or self.viewpoint_sampling_target_distance_m
            <= self.viewpoint_sampling_terminal_heading_hold_tolerance_m
        ):
            raise ValueError(
                "viewpoint_sampling_target_distance_m must be finite and greater "
                "than the terminal-heading radial hold tolerance"
            )
        if (
            not math.isfinite(
                self.viewpoint_sampling_terminal_heading_target_envelope_radius_m
            )
            or self.viewpoint_sampling_terminal_heading_target_envelope_radius_m
            < self.viewpoint_sampling_terminal_heading_hold_tolerance_m
            or self.viewpoint_sampling_terminal_heading_target_envelope_radius_m
            > INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M
        ):
            raise ValueError(
                "viewpoint_sampling_terminal_heading_target_envelope_radius_m "
                "must be finite, no smaller than the radial hold tolerance, "
                "and no greater than 0.030"
            )
        if (
            not math.isfinite(self.viewpoint_sampling_heading_tolerance_rad)
            or self.viewpoint_sampling_heading_tolerance_rad <= 0.0
        ):
            raise ValueError(
                "viewpoint_sampling_heading_tolerance_rad must be finite and positive"
            )
        if (
            not math.isfinite(self.physical_waypoint_tolerance_m)
            or self.physical_waypoint_tolerance_m <= 0.0
        ):
            raise ValueError(
                "physical_waypoint_tolerance_m must be finite and positive"
            )
        if (
            not math.isfinite(self.physical_goal_tolerance_m)
            or self.physical_goal_tolerance_m <= 0.0
        ):
            raise ValueError("physical_goal_tolerance_m must be finite and positive")
        if (
            not math.isfinite(self.max_future_timestamp_sec)
            or self.max_future_timestamp_sec < 0.0
        ):
            raise ValueError("max_future_timestamp_sec must be finite and non-negative")
        if (
            not math.isfinite(self.certified_route_tube_radius_m)
            or self.certified_route_tube_radius_m <= 0.0
        ):
            raise ValueError(
                "certified_route_tube_radius_m must be finite and positive"
            )
        if (
            not math.isfinite(self.certified_route_chord_sample_spacing_m)
            or self.certified_route_chord_sample_spacing_m <= 0.0
        ):
            raise ValueError(
                "certified_route_chord_sample_spacing_m must be finite and positive"
            )
        corner_config = CertifiedCornerControlConfig(
            turn_threshold_rad=self.certified_corner_turn_threshold_rad,
            release_tolerance_m=self.certified_corner_release_tolerance_m,
            hold_tolerance_m=self.certified_corner_hold_tolerance_m,
            alignment_tolerance_rad=(
                self.certified_corner_alignment_tolerance_rad
            ),
        )
        if (
            corner_config.release_tolerance_m
            > self.physical_waypoint_tolerance_m
        ):
            raise ValueError(
                "certified corner release tolerance must not exceed the "
                "physical waypoint tolerance"
            )
        if corner_config.hold_tolerance_m >= self.certified_route_tube_radius_m:
            raise ValueError(
                "certified corner hold tolerance must remain strictly inside "
                "the certified route tube"
            )
        if self.physical_goal_tolerance_m > self.certified_route_tube_radius_m:
            raise ValueError(
                "physical_goal_tolerance_m must not exceed the certified route tube"
            )
        if (
            self.physical_waypoint_tolerance_m
            > self.certified_route_tube_radius_m
        ):
            raise ValueError(
                "physical_waypoint_tolerance_m must not exceed "
                "the certified route tube"
            )


@dataclass(frozen=True)
class CertifiedStartupRouteState:
    join_pending: bool
    join_limit_m: float | None
    egress_lock_index: int | None


@dataclass(frozen=True)
class CertifiedStaticStartupDecision:
    """Bounded initial target selection for a sealed static route."""

    ok: bool
    target_index: int | None
    route_check: ExecutionRouteCheck


def certified_startup_route_state(
    config: FollowerConfig,
    waypoint_count: int,
) -> CertifiedStartupRouteState:
    """Create the immutable startup ordering for a certified static leg."""

    index = config.initial_start_egress_waypoint_index
    if index is not None and index >= waypoint_count:
        raise ValueError("initial start-egress waypoint is outside the route")
    join_limit = config.initial_start_join_clearance_m
    return CertifiedStartupRouteState(
        join_pending=join_limit is not None,
        join_limit_m=join_limit,
        egress_lock_index=index,
    )


INTERMEDIATE_ROUTE_KINDS = frozenset(
    {"axis_acquisition", "viewpoint_sampling"}
)
DYNAMIC_PHYSICAL_ROUTE_KINDS = frozenset(
    {"synchronized_face_approach", "synchronized_viewpoint"}
)
CATALOG_PHYSICAL_ROUTE_KINDS = frozenset({"catalog_face_approach"})
STATIC_PHYSICAL_ROUTE_KINDS = CATALOG_PHYSICAL_ROUTE_KINDS | frozenset(
    {"detected_stand_preapproach", "stand_discovery_corridor"}
)
STATIC_STARTUP_SEGMENT_JOIN_ROUTE_KINDS = frozenset(
    {"detected_stand_preapproach", "stand_discovery_corridor"}
)
PHYSICAL_ROUTE_KINDS = DYNAMIC_PHYSICAL_ROUTE_KINDS | STATIC_PHYSICAL_ROUTE_KINDS
DYNAMIC_VIEWPOINT_ROUTE_KINDS = (
    INTERMEDIATE_ROUTE_KINDS | DYNAMIC_PHYSICAL_ROUTE_KINDS
)


def certified_static_startup_decision(
    pose: Pose2D,
    waypoints: Sequence[Pose2D],
    *,
    tracking_tube_radius_m: float,
    chord_sample_spacing_m: float = 0.01,
) -> CertifiedStaticStartupDecision:
    """Select waypoint 0 or 1 without leaving the certified first segment.

    A* route vertices are map-cell centers while the live localization pose is
    continuous.  At startup the robot can therefore be inside the first sealed
    segment but slightly farther than the tracking radius from vertex 0.  The
    ordinary target-0 route check treats that vertex as a zero-length segment.
    This gate prefers the exact next vertex when both the live pose and its
    pursuit chord fit inside the already-certified first route segment.  That
    prevents a small localization update from returning execution to the
    degenerate vertex-0 tube after startup.
    """

    on_first_segment = check_execution_route_tube(
        pose,
        waypoints,
        target_index=1,
        pursuit_index=1,
        tracking_tube_radius_m=tracking_tube_radius_m,
        chord_sample_spacing_m=chord_sample_spacing_m,
    )
    if on_first_segment.ok:
        return CertifiedStaticStartupDecision(
            ok=True,
            target_index=1,
            route_check=on_first_segment,
        )

    at_first_vertex = check_execution_route_tube(
        pose,
        waypoints,
        target_index=0,
        pursuit_index=0,
        tracking_tube_radius_m=tracking_tube_radius_m,
        chord_sample_spacing_m=chord_sample_spacing_m,
    )
    if at_first_vertex.ok:
        return CertifiedStaticStartupDecision(
            ok=True,
            target_index=0,
            route_check=at_first_vertex,
        )
    return CertifiedStaticStartupDecision(
        ok=False,
        target_index=None,
        route_check=on_first_segment,
    )


@dataclass(frozen=True)
class IntermediateTerminalHeadingLatch:
    """Identity of an intermediate target committed to in-place yaw control."""

    route_kind: str
    target_index: int
    target: Pose2D


@dataclass(frozen=True)
class IntermediateTerminalHeadingDecision:
    """Pure controller result plus the next immutable latch state."""

    step: ControllerStep
    latch: IntermediateTerminalHeadingLatch | None
    failure: str = ""


def reset_intermediate_terminal_heading_latch(
    latch: IntermediateTerminalHeadingLatch | None,
    *,
    material_route_revision: bool = False,
    target_changed: bool = False,
) -> IntermediateTerminalHeadingLatch | None:
    """Clear a latch at either route-identity boundary."""

    if material_route_revision or target_changed:
        return None
    return latch


def intermediate_terminal_heading_entry_tolerance_m(
    config: ControllerConfig,
) -> float:
    """Return the strict final-position entry tolerance for survey routes."""

    configured_tolerance_m = (
        config.goal_tolerance_m
        if config.terminal_goal_tolerance_m is None
        else config.terminal_goal_tolerance_m
    )
    return min(
        configured_tolerance_m,
        INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M,
    )


def intermediate_terminal_heading_hold_diagnostics(
    pose: Pose2D,
    latch: IntermediateTerminalHeadingLatch,
    *,
    hold_tolerance_m: float,
    viewpoint_sampling_target_distance_m: float,
    viewpoint_sampling_target_envelope_radius_m: float,
) -> dict[str, object]:
    """Return the pure metric predicates used by the latched-yaw safety gate."""

    target = latch.target
    comparison_epsilon_m = (
        INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M
    )
    target_envelope_distance_m = math.hypot(
        pose.x_m - target.x_m,
        pose.y_m - target.y_m,
    )
    heading_is_finite = math.isfinite(pose.yaw_rad) and math.isfinite(
        target.yaw_rad
    )
    is_viewpoint_sampling = latch.route_kind == "viewpoint_sampling"
    if is_viewpoint_sampling:
        metrics = viewpoint_sampling_hold_metrics(
            pose,
            target,
            config=ViewpointSamplingHoldConfig(
                entry_tolerance_m=min(
                    hold_tolerance_m,
                    INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M,
                ),
                hold_tolerance_m=hold_tolerance_m,
                target_envelope_radius_m=(
                    viewpoint_sampling_target_envelope_radius_m
                ),
                target_distance_m=viewpoint_sampling_target_distance_m,
                distance_comparison_epsilon_m=comparison_epsilon_m,
            ),
        )
        return metrics.to_diagnostics_dict()

    target_envelope_radius_m = hold_tolerance_m
    target_envelope_comparison_limit_m = (
        target_envelope_radius_m + comparison_epsilon_m
    )
    target_envelope_within_limit = (
        math.isfinite(target_envelope_distance_m)
        and math.isfinite(target_envelope_comparison_limit_m)
        and target_envelope_distance_m <= target_envelope_comparison_limit_m
    )

    inferred_stand_center_x_m = math.nan
    inferred_stand_center_y_m = math.nan
    inferred_stand_distance_m = math.nan
    annulus_min_m = math.nan
    annulus_max_m = math.nan
    inferred_stand_distance_within_annulus = True
    within_hold = (
        heading_is_finite
        and target_envelope_within_limit
        and inferred_stand_distance_within_annulus
    )
    return {
        "hold_model": "target_distance_disk",
        "distance_unit": "m",
        "target_yaw_unit": "rad",
        "target_yaw_rad": target.yaw_rad,
        "heading_is_finite": heading_is_finite,
        "target_envelope_distance_m": target_envelope_distance_m,
        "target_envelope_radius_m": target_envelope_radius_m,
        "target_envelope_within_limit": target_envelope_within_limit,
        "nominal_target_distance_m": viewpoint_sampling_target_distance_m,
        "inferred_stand_center_x_m": inferred_stand_center_x_m,
        "inferred_stand_center_y_m": inferred_stand_center_y_m,
        "inferred_stand_distance_m": inferred_stand_distance_m,
        "annulus_min_m": annulus_min_m,
        "annulus_max_m": annulus_max_m,
        "inferred_stand_distance_within_annulus": (
            inferred_stand_distance_within_annulus
        ),
        "distance_comparison_epsilon_m": comparison_epsilon_m,
        "within_hold": within_hold,
    }


def _latched_intermediate_terminal_heading_decision(
    pose: Pose2D,
    latch: IntermediateTerminalHeadingLatch,
    config: ControllerConfig,
    hold_tolerance_m: float,
    viewpoint_sampling_target_distance_m: float,
    viewpoint_sampling_target_envelope_radius_m: float,
) -> IntermediateTerminalHeadingDecision:
    diagnostics = intermediate_terminal_heading_hold_diagnostics(
        pose,
        latch,
        hold_tolerance_m=hold_tolerance_m,
        viewpoint_sampling_target_distance_m=(
            viewpoint_sampling_target_distance_m
        ),
        viewpoint_sampling_target_envelope_radius_m=(
            viewpoint_sampling_target_envelope_radius_m
        ),
    )
    target_envelope_distance_m = float(
        diagnostics["target_envelope_distance_m"]
    )
    if not bool(diagnostics["within_hold"]):
        return IntermediateTerminalHeadingDecision(
            ControllerStep(
                VelocityCommand(0.0, 0.0),
                latch.target_index,
                False,
                target_envelope_distance_m,
                latch.target_index,
                math.nan,
                "terminal_heading_hold_exceeded",
            ),
            latch,
            INTERMEDIATE_TERMINAL_HEADING_HOLD_EXCEEDED,
        )

    target = latch.target
    final_heading_error = normalize_angle(target.yaw_rad - pose.yaw_rad)
    reached_goal = abs(final_heading_error) <= config.heading_tolerance_rad
    angular_z_radps = 0.0
    if not reached_goal:
        angular_z_radps = max(
            -config.max_angular_radps,
            min(
                config.max_angular_radps,
                final_heading_error * config.rotate_gain,
            ),
        )
    return IntermediateTerminalHeadingDecision(
        ControllerStep(
            VelocityCommand(0.0, angular_z_radps),
            latch.target_index,
            reached_goal,
            target_envelope_distance_m,
            latch.target_index,
            final_heading_error,
            "terminal_heading",
        ),
        latch,
    )


def compute_intermediate_terminal_heading_command(
    pose: Pose2D,
    waypoints: Sequence[Pose2D],
    target_index: int,
    config: ControllerConfig,
    route_kind: str,
    latch: IntermediateTerminalHeadingLatch | None = None,
    *,
    hold_tolerance_m: float = INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M,
    viewpoint_sampling_target_distance_m: float = (
        DEFAULT_VIEWPOINT_SAMPLING_TARGET_DISTANCE_M
    ),
    viewpoint_sampling_target_envelope_radius_m: float = (
        INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M
    ),
) -> IntermediateTerminalHeadingDecision:
    """Latch final survey-yaw control without changing other route behavior."""

    if not waypoints:
        return IntermediateTerminalHeadingDecision(
            compute_waypoint_command(pose, waypoints, target_index, config),
            None,
        )

    entry_tolerance_m = intermediate_terminal_heading_entry_tolerance_m(config)
    if route_kind in INTERMEDIATE_ROUTE_KINDS and (
        not math.isfinite(hold_tolerance_m)
        or hold_tolerance_m <= 0.0
        or hold_tolerance_m > INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M
        or hold_tolerance_m < entry_tolerance_m
    ):
        raise ValueError(
            "hold_tolerance_m must be finite, no smaller than the effective "
            "entry tolerance, and no greater than 0.020"
        )
    if route_kind == "viewpoint_sampling" and (
        not math.isfinite(viewpoint_sampling_target_distance_m)
        or viewpoint_sampling_target_distance_m <= hold_tolerance_m
    ):
        raise ValueError(
            "viewpoint_sampling_target_distance_m must be finite and greater "
            "than hold_tolerance_m"
        )
    if route_kind == "viewpoint_sampling" and (
        not math.isfinite(viewpoint_sampling_target_envelope_radius_m)
        or viewpoint_sampling_target_envelope_radius_m < hold_tolerance_m
        or viewpoint_sampling_target_envelope_radius_m
        > INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M
    ):
        raise ValueError(
            "viewpoint_sampling_target_envelope_radius_m must be finite, no "
            "smaller than hold_tolerance_m, and no greater than 0.030"
        )

    current_index = min(max(target_index, 0), len(waypoints) - 1)
    current_target = waypoints[current_index]
    latch_matches_target = (
        latch is not None
        and route_kind in INTERMEDIATE_ROUTE_KINDS
        and latch.route_kind == route_kind
        and latch.target_index == current_index
        and latch.target == current_target
        and current_index == len(waypoints) - 1
        and math.isfinite(current_target.yaw_rad)
    )
    if latch is not None and not latch_matches_target:
        latch = reset_intermediate_terminal_heading_latch(
            latch,
            target_changed=True,
        )
    if latch is not None:
        return _latched_intermediate_terminal_heading_decision(
            pose,
            latch,
            config,
            hold_tolerance_m,
            viewpoint_sampling_target_distance_m,
            viewpoint_sampling_target_envelope_radius_m,
        )

    ordinary_step = compute_waypoint_command(
        pose,
        waypoints,
        target_index,
        config,
    )
    if route_kind not in INTERMEDIATE_ROUTE_KINDS:
        return IntermediateTerminalHeadingDecision(ordinary_step, None)

    final_index = len(waypoints) - 1
    if (
        current_index != final_index
        or ordinary_step.target_index != current_index
        or not math.isfinite(current_target.yaw_rad)
    ):
        # A controller-side target advance is a material target change.  Let
        # the caller persist it first; the following tick may enter the latch.
        return IntermediateTerminalHeadingDecision(ordinary_step, None)

    target_distance_m = math.hypot(
        pose.x_m - current_target.x_m,
        pose.y_m - current_target.y_m,
    )
    if (
        not math.isfinite(entry_tolerance_m)
        or entry_tolerance_m <= 0.0
        or not math.isfinite(target_distance_m)
        or target_distance_m > entry_tolerance_m
    ):
        return IntermediateTerminalHeadingDecision(ordinary_step, None)

    latch = IntermediateTerminalHeadingLatch(
        route_kind=route_kind,
        target_index=current_index,
        target=current_target,
    )
    return _latched_intermediate_terminal_heading_decision(
        pose,
        latch,
        config,
        hold_tolerance_m,
        viewpoint_sampling_target_distance_m,
        viewpoint_sampling_target_envelope_radius_m,
    )


def controller_config_for_route_kind(
    config: ControllerConfig,
    route_kind: str,
    *,
    reverse_staging: bool = False,
    viewpoint_sampling_goal_tolerance_m: float | None = None,
    viewpoint_sampling_heading_tolerance_rad: float | None = None,
    physical_waypoint_tolerance_m: float | None = None,
    physical_goal_tolerance_m: float | None = None,
) -> ControllerConfig:
    """Apply terminal-heading constraints only to physical face approaches.

    Acquisition and viewpoint-sampling waypoints may carry a finite terminal
    yaw, but enforcing that yaw throughout translation conflicts with pursuit
    of the geometric segment.  The normal final-position alignment still
    enforces the yaw once the sampling point has actually been reached.
    """

    if route_kind not in DYNAMIC_VIEWPOINT_ROUTE_KINDS | STATIC_PHYSICAL_ROUTE_KINDS:
        return config
    physical = route_kind in PHYSICAL_ROUTE_KINDS
    goal_tolerance = config.goal_tolerance_m
    intermediate = route_kind in INTERMEDIATE_ROUTE_KINDS
    if intermediate:
        # Acquisition must also end close enough that lateral position error
        # plus the terminal-yaw tolerance stays inside the observer's camera
        # centering cone.  At 0.55 m standoff, 0.03 m and 5 degrees remain
        # comfortably below the 12-degree perception gate.
        goal_tolerance = min(
            goal_tolerance,
            INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M,
        )
        if viewpoint_sampling_goal_tolerance_m is not None:
            goal_tolerance = min(
                goal_tolerance,
                viewpoint_sampling_goal_tolerance_m,
            )
    terminal_goal_tolerance = config.terminal_goal_tolerance_m
    if intermediate and terminal_goal_tolerance is not None:
        terminal_goal_tolerance = min(
            terminal_goal_tolerance,
            goal_tolerance,
        )
    if physical and physical_waypoint_tolerance_m is not None:
        goal_tolerance = min(goal_tolerance, physical_waypoint_tolerance_m)
    if physical and physical_goal_tolerance_m is not None:
        terminal_goal_tolerance = min(
            config.goal_tolerance_m,
            physical_goal_tolerance_m,
        )
    heading_tolerance = config.heading_tolerance_rad
    if (
        route_kind in INTERMEDIATE_ROUTE_KINDS
        and viewpoint_sampling_heading_tolerance_rad is not None
    ):
        # Both intermediate phases are camera observation poses.  The generic
        # 0.25 rad transit tolerance can declare an acquisition goal complete
        # while the observer's image-centering gate still rejects it (retry
        # 08 stopped at 0.224 rad).  Use the shared camera-heading tolerance
        # before starting the bounded stationary hold.
        heading_tolerance = min(
            heading_tolerance, viewpoint_sampling_heading_tolerance_rad
        )
    return replace(
        config,
        goal_tolerance_m=goal_tolerance,
        terminal_goal_tolerance_m=terminal_goal_tolerance,
        heading_tolerance_rad=heading_tolerance,
        enforce_heading_corridor=physical,
        reverse_staging=physical and reverse_staging,
        exact_vertex_pursuit=physical,
    )


def dynamic_route_kind_transition_failure(
    current_route_kind: str, next_route_kind: str
) -> str:
    """Validate monotonic acquisition -> sampling -> physical handoffs."""

    if not next_route_kind:
        return "missing dynamic route kind"
    if next_route_kind not in DYNAMIC_VIEWPOINT_ROUTE_KINDS:
        return f"unknown dynamic route kind: {next_route_kind}"
    if current_route_kind not in DYNAMIC_VIEWPOINT_ROUTE_KINDS:
        return f"unknown current dynamic route kind: {current_route_kind or '<missing>'}"
    if current_route_kind == next_route_kind:
        return ""
    if current_route_kind == "axis_acquisition" and next_route_kind in (
        {"viewpoint_sampling"} | DYNAMIC_PHYSICAL_ROUTE_KINDS
    ):
        return ""
    if (
        current_route_kind == "viewpoint_sampling"
        and next_route_kind in DYNAMIC_PHYSICAL_ROUTE_KINDS
    ):
        return ""
    if (
        current_route_kind in DYNAMIC_PHYSICAL_ROUTE_KINDS
        and next_route_kind in DYNAMIC_PHYSICAL_ROUTE_KINDS
    ):
        return ""
    return (
        "backward dynamic route phase transition: "
        f"{current_route_kind}->{next_route_kind}"
    )


def viewpoint_sampling_timeout_failure(
    *,
    route_kind: str,
    phase_started_at: float | None,
    now_monotonic: float,
    timeout_sec: float,
) -> str:
    if route_kind != "viewpoint_sampling":
        return ""
    if phase_started_at is None:
        return "viewpoint_sampling_clock_unavailable"
    if now_monotonic - phase_started_at >= timeout_sec:
        return "viewpoint_sampling_timeout"
    return ""


def viewpoint_sampling_target_timeout_failure(
    *,
    route_kind: str,
    target_started_at: float | None,
    now_monotonic: float,
    timeout_sec: float,
) -> str:
    failure = viewpoint_sampling_timeout_failure(
        route_kind=route_kind,
        phase_started_at=target_started_at,
        now_monotonic=now_monotonic,
        timeout_sec=timeout_sec,
    )
    return {
        "viewpoint_sampling_clock_unavailable": (
            "viewpoint_sampling_target_clock_unavailable"
        ),
        "viewpoint_sampling_timeout": "viewpoint_sampling_target_timeout",
    }.get(failure, failure)


def acquisition_goal_action(
    *,
    route_kind: str,
    provider_available: bool,
    hold_elapsed_sec: float,
    timeout_sec: float,
) -> str:
    """Decide whether a geometrically reached route is mission-terminal."""

    if route_kind not in INTERMEDIATE_ROUTE_KINDS:
        return "complete"
    if not provider_available:
        return "missing_dynamic_route_provider"
    if route_kind == "axis_acquisition" and hold_elapsed_sec >= timeout_sec:
        return "axis_acquisition_timeout"
    return "hold_for_physical_face"


def _require_ros() -> None:
    if rclpy is None:
        raise RuntimeError("ROS2 Python packages are not available in this environment")


def _create_dedicated_tf_listener(runtime_config: ResolvedRuntimeConfig):
    """Create an isolated owner for only the TF listener subscriptions."""

    listener_node = Node(
        TF_LISTENER_NODE_NAME,
        namespace=runtime_config.namespace,
    )
    tf_buffer = Buffer(node=listener_node)
    tf_listener = TransformListener(
        tf_buffer,
        listener_node,
        spin_thread=False,
    )
    return listener_node, tf_buffer, tf_listener


def _node_identity(endpoint) -> str:
    namespace = getattr(endpoint, "node_namespace", "") or ""
    name = getattr(endpoint, "node_name", "") or ""
    return _format_node_identity(namespace, name)


def _format_node_identity(namespace: str, name: str) -> str:
    if namespace in ("", "/"):
        return f"/{name}"
    return f"{namespace.rstrip('/')}/{name}"


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _ros_stamp_sec(stamp) -> float | None:
    try:
        value = float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class PoseLookupResult:
    pose: Pose2D | None
    details: dict[str, object] | None = None
    stamp_sec: float | None = None


def tf_lookup_failure_details(
    *,
    reason: str,
    target_frame: str,
    source_frame: str,
    max_age_sec: float,
    age_sec: float | None = None,
    exception: BaseException | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "stop_reason": "map-to-base transform unavailable",
        "source": "tf_lookup",
        "reason": reason,
        "target_frame": target_frame,
        "source_frame": source_frame,
        "max_age_sec": max_age_sec,
    }
    if age_sec is not None:
        payload["age_sec"] = age_sec
    if exception is not None:
        payload["exception_type"] = exception.__class__.__name__
        payload["exception"] = str(exception)
    return payload


def _pose_lookup_diagnostics(result: PoseLookupResult) -> dict[str, object]:
    details = dict(result.details or {})
    if not details:
        details = {
            "source": "tf_lookup",
            "reason": "fresh_transform",
        }
    if result.stamp_sec is not None:
        details["stamp_sec"] = result.stamp_sec
    return details


def _stale_tf_recovery_failure_details(
    final_details: dict[str, object],
    *,
    first_lookup: PoseLookupResult,
    retry_lookup: PoseLookupResult,
    callback_drain: dict[str, object],
) -> dict[str, object]:
    first_details = _pose_lookup_diagnostics(first_lookup)
    retry_details = _pose_lookup_diagnostics(retry_lookup)
    combined = dict(final_details)
    combined.update(
        {
            "fail_closed": True,
            "recovery_attempted": True,
            "zero_published_before_retry": True,
            "first_lookup_age_sec": first_details.get("age_sec"),
            "retry_lookup_age_sec": retry_details.get("age_sec"),
            "first_lookup_stamp_sec": first_lookup.stamp_sec,
            "retry_lookup_stamp_sec": retry_lookup.stamp_sec,
            "first_lookup": first_details,
            "retry_lookup": retry_details,
            "callback_drain": callback_drain,
        }
    )
    return combined


def dynamic_join_envelope_failure(
    pose: Pose2D,
    anchor: Pose2D,
    effective_join_limit_m: float | None,
) -> dict[str, object] | None:
    """Fail closed if a live pose leaves or cannot define the certified join disk."""

    if not all(math.isfinite(value) for value in (pose.x_m, pose.y_m, pose.yaw_rad)):
        return {
            "reason": "current robot pose is non-finite during dynamic-route join",
            "fault_code": "invalid_current_pose",
            "fail_closed": True,
        }
    if (
        effective_join_limit_m is None
        or not math.isfinite(effective_join_limit_m)
        or effective_join_limit_m <= 0.0
    ):
        return {
            "reason": "dynamic-route join envelope is invalid",
            "fault_code": "invalid_route_update",
            "fail_closed": True,
            "effective_join_limit_m": effective_join_limit_m,
        }
    join_distance = math.hypot(pose.x_m - anchor.x_m, pose.y_m - anchor.y_m)
    if not math.isfinite(join_distance):
        return {
            "reason": "dynamic-route join distance is non-finite",
            "fault_code": "invalid_current_pose",
            "fail_closed": True,
        }
    if join_distance > effective_join_limit_m:
        return {
            "reason": "robot left the certified dynamic-route join envelope",
            "fault_code": "join_envelope_exceeded",
            "fail_closed": True,
            "join_distance_m": join_distance,
            "effective_join_limit_m": effective_join_limit_m,
        }
    return None


def certified_startup_join_action(
    pose: Pose2D,
    anchor: Pose2D,
    effective_join_limit_m: float | None,
    join_tolerance_m: float,
) -> tuple[str, dict[str, object] | None]:
    """Select only stop, anchor pursuit, or the anchor-complete zero cycle."""

    failure = dynamic_join_envelope_failure(
        pose,
        anchor,
        effective_join_limit_m,
    )
    if failure is not None:
        return "stop", failure
    if not math.isfinite(join_tolerance_m) or join_tolerance_m <= 0.0:
        return "stop", {
            "reason": "dynamic-route join tolerance is invalid",
            "fault_code": "invalid_route_update",
            "fail_closed": True,
        }
    distance_m = math.hypot(pose.x_m - anchor.x_m, pose.y_m - anchor.y_m)
    return ("zero", None) if distance_m <= join_tolerance_m else ("anchor", None)


def stuck_progress_details(
    *,
    target_index: int,
    distance_to_target_m: float,
    last_progress_distance_m: float,
    elapsed_without_progress_sec: float,
    max_without_progress_sec: float,
    progress_epsilon_m: float,
    commanded_linear_x_mps: float,
    commanded_angular_z_radps: float,
    front_clearance_scale: float,
    effective_linear_x_mps: float,
    front_clearance_details: dict[str, object] | None = None,
    pursuit_index: int | None = None,
    controlled_heading_error_rad: float | None = None,
    last_progress_heading_error_rad: float | None = None,
    heading_progress_epsilon_rad: float | None = None,
    last_progress_target_index: int | None = None,
    last_progress_pursuit_index: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "stop_reason": "stuck no progress",
        "source": "progress_monitor",
        "target_index": target_index,
        "distance_to_target_m": distance_to_target_m,
        "last_progress_distance_m": last_progress_distance_m,
        "elapsed_without_progress_sec": elapsed_without_progress_sec,
        "max_without_progress_sec": max_without_progress_sec,
        "progress_epsilon_m": progress_epsilon_m,
        "commanded_linear_x_mps": commanded_linear_x_mps,
        "commanded_angular_z_radps": commanded_angular_z_radps,
        "front_clearance_scale": front_clearance_scale,
        "effective_linear_x_mps": effective_linear_x_mps,
        "pursuit_index": pursuit_index,
        "controlled_heading_error_rad": controlled_heading_error_rad,
        "last_progress_heading_error_rad": last_progress_heading_error_rad,
        "heading_progress_epsilon_rad": heading_progress_epsilon_rad,
        "last_progress_target_index": last_progress_target_index,
        "last_progress_pursuit_index": last_progress_pursuit_index,
    }
    if front_clearance_details is not None:
        payload["front_clearance"] = front_clearance_details
    return payload


class SimpleWaypointFollowerNode(Node):  # pragma: no cover - requires ROS runtime.
    def __init__(
        self,
        runtime_config: ResolvedRuntimeConfig,
        waypoints: Sequence[Pose2D],
        follower_config: FollowerConfig,
        waypoint_provider: Callable[[Pose2D], RouteUpdate | None] | None = None,
        route_update_callback: Callable[[RouteUpdate], None] | None = None,
        *,
        tf_buffer=None,
    ) -> None:
        # Topic resolution alone does not namespace the ROS node identity.
        # Create the publisher in the resolved namespace so the identity bound
        # into the execution certificate is the identity DDS actually reports.
        super().__init__(
            "aufgabe04_simple_waypoint_follower",
            namespace=runtime_config.namespace,
        )
        self.runtime_config = runtime_config
        self.waypoints = tuple(waypoints)
        self.follower_config = follower_config
        self.waypoint_provider = waypoint_provider
        self.route_update_callback = route_update_callback
        self.last_route_refresh_at = 0.0
        self.initial_route_refresh_pending = waypoint_provider is not None
        startup_state = certified_startup_route_state(
            follower_config,
            len(self.waypoints),
        )
        self.dynamic_join_pending = startup_state.join_pending
        self.dynamic_join_limit_m = startup_state.join_limit_m
        self.start_egress_lock_index = startup_state.egress_lock_index
        self.current_route_kind = follower_config.initial_route_kind
        self.certified_static_start_pending = (
            self.current_route_kind in STATIC_STARTUP_SEGMENT_JOIN_ROUTE_KINDS
            and not startup_state.join_pending
        )
        self.intermediate_terminal_heading_latch: (
            IntermediateTerminalHeadingLatch | None
        ) = None
        self.certified_corner_latch: CertifiedCornerTransitionLatch | None = None
        self._last_certified_corner_phase: tuple[int, str] | None = None
        self.reverse_staging = False
        self.axis_acquisition_hold_started_at: float | None = None
        self.axis_acquisition_target_revision: int | None = None
        self.viewpoint_sampling_started_at: float | None = None
        self.viewpoint_sampling_target_started_at: float | None = None
        self.viewpoint_sampling_target_revision: int | None = None
        self.target_index = 0
        self.latest_scan = None
        self.latest_scan_receipt = None
        self.latest_odom = None
        self.latest_odom_receipt = None
        self.latest_odom_callback_count = 0
        self.motion_published = False
        self.distance_estimate_m = 0.0
        self.last_pose = None
        self.last_progress_distance_m = math.inf
        self.last_progress_heading_error_rad = math.inf
        self.last_progress_target_index: int | None = None
        self.last_progress_pursuit_index: int | None = None
        self.last_progress_mode: str | None = None
        self.progress_heading_modes_seen: set[str] = set()
        self.progress_heading_error_by_mode: dict[str, float] = {}
        self.last_progress_at = time.monotonic()
        self.target_started_at = time.monotonic()
        self.latest_stop_details = None
        self.latest_front_clearance_details = None
        self._simulation_odom_fallback_active = False
        self._simulation_odom_fallback_episode = 0
        self._background_callback_service_enabled = False
        self._configure_sim_time(runtime_config.use_sim_time)

        self.cmd_vel_pub = self.create_publisher(Twist, runtime_config.cmd_vel_topic, 10)
        self.create_subscription(
            LaserScan,
            runtime_config.scan_topic,
            self._scan_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(Odometry, runtime_config.odom_topic, self._odom_callback, 10)
        if tf_buffer is None:
            # Preserve direct-node construction for focused ROS use.  The
            # production runner always injects the buffer serviced by its
            # isolated listener node/executor below.
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
        else:
            self.tf_buffer = tf_buffer
            self.tf_listener = None

    def _configure_sim_time(self, use_sim_time: bool) -> None:
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", use_sim_time)
            return
        self.set_parameters(
            [Parameter("use_sim_time", Parameter.Type.BOOL, use_sim_time)]
        )

    def _scan_callback(self, msg) -> None:
        self.latest_scan = msg
        # Transport freshness belongs to the process monotonic clock.  A new
        # simulation node can receive a scan before its first /clock callback;
        # recording that receipt with ROS time would store zero and then look
        # thousands of seconds stale as soon as simulated time activates.
        self.latest_scan_receipt = time.monotonic()

    def _odom_callback(self, msg) -> None:
        self.latest_odom = msg
        self.latest_odom_receipt = time.monotonic()
        self.latest_odom_callback_count = (
            getattr(self, "latest_odom_callback_count", 0) + 1
        )

    def publish_zero(self) -> None:
        self.cmd_vel_pub.publish(Twist())

    @property
    def callback_service_mode(self) -> str:
        if getattr(self, "_background_callback_service_enabled", False):
            return CALLBACK_SERVICE_BACKGROUND_EXECUTOR
        return CALLBACK_SERVICE_CALLER_SPIN

    def enable_background_callback_service(self) -> None:
        """Mark this node as owned by the continuously spinning executor."""

        self._background_callback_service_enabled = True

    def disable_background_callback_service(self) -> None:
        """Return callback ownership to the caller after executor teardown."""

        self._background_callback_service_enabled = False

    def _service_or_wait_for_callbacks(self, timeout_sec: float) -> None:
        """Wait for background callbacks, or service them in legacy caller mode."""

        if self.callback_service_mode == CALLBACK_SERVICE_BACKGROUND_EXECUTOR:
            time.sleep(timeout_sec)
            return
        rclpy.spin_once(self, timeout_sec=timeout_sec)

    def publish_repeated_zero(self, count: int = 5) -> None:
        for _ in range(count):
            self.publish_zero()
            self._service_or_wait_for_callbacks(0.02)

    def _drain_runtime_callbacks(
        self,
        max_callbacks: int = 12,
        *,
        max_duration_sec: float | None = None,
        spin_timeout_sec: float = 0.0,
    ) -> dict[str, object]:
        """Service callbacks in caller mode or wait for the background executor.

        Production runs use continuously spinning follower and TF executors.
        Their ordinary control-loop drain is therefore an immediate no-op.  A
        bounded stale-TF recovery instead waits its full safety window while
        the follower executor services scan/odometry/clock and the isolated TF
        executor services TF subscriptions.  ``spin_count`` remains scoped to
        caller spins performed here; it never claims work done by either
        background executor.  The caller-spin branch remains only for ROS-free
        focused tests and direct node use outside
        :func:`run_simple_waypoint_follower`.
        """
        if (
            not isinstance(max_callbacks, int)
            or isinstance(max_callbacks, bool)
            or max_callbacks <= 0
        ):
            raise ValueError("max_callbacks must be a positive integer")
        if max_duration_sec is not None and (
            not math.isfinite(max_duration_sec) or max_duration_sec <= 0.0
        ):
            raise ValueError("max_duration_sec must be finite and positive")
        if not math.isfinite(spin_timeout_sec) or spin_timeout_sec < 0.0:
            raise ValueError("spin_timeout_sec must be finite and non-negative")

        started_at = time.monotonic()
        if self.callback_service_mode == CALLBACK_SERVICE_BACKGROUND_EXECUTOR:
            waited_for_background_callbacks = max_duration_sec is not None
            if waited_for_background_callbacks:
                time.sleep(max_duration_sec)
            elapsed_sec = time.monotonic() - started_at
            return {
                "callback_service_mode": self.callback_service_mode,
                "spin_count": 0,
                "elapsed_sec": elapsed_sec,
                "max_callbacks": max_callbacks,
                "max_duration_sec": max_duration_sec,
                "spin_timeout_sec": spin_timeout_sec,
                "deadline_reached": waited_for_background_callbacks,
                "background_wait_requested_sec": (
                    max_duration_sec if waited_for_background_callbacks else 0.0
                ),
            }

        spin_count = 0
        deadline_reached = False
        for _ in range(max_callbacks):
            elapsed_sec = time.monotonic() - started_at
            if (
                max_duration_sec is not None
                and elapsed_sec >= max_duration_sec
            ):
                deadline_reached = True
                break
            timeout_sec = spin_timeout_sec
            if max_duration_sec is not None:
                timeout_sec = min(
                    timeout_sec,
                    max(0.0, max_duration_sec - elapsed_sec),
                )
            rclpy.spin_once(self, timeout_sec=timeout_sec)
            spin_count += 1
        elapsed_sec = time.monotonic() - started_at
        return {
            "callback_service_mode": self.callback_service_mode,
            "spin_count": spin_count,
            "elapsed_sec": elapsed_sec,
            "max_callbacks": max_callbacks,
            "max_duration_sec": max_duration_sec,
            "spin_timeout_sec": spin_timeout_sec,
            "deadline_reached": deadline_reached,
        }

    def _start_egress_command(
        self,
        pose: Pose2D,
        controller_config: ControllerConfig,
    ):
        """Return a locked command, or release the lock at its exact vertex."""

        waypoint_index = self.start_egress_lock_index
        if waypoint_index is None:
            raise ValueError("start egress command requested without an active lock")
        step = compute_start_egress_vertex_command(
            pose,
            self.waypoints,
            waypoint_index,
            controller_config,
            reach_tolerance_m=(
                self.follower_config.start_egress_waypoint_tolerance_m
            ),
            egress_config=StartEgressControlConfig(
                alignment_tolerance_rad=(
                    self.follower_config.start_egress_alignment_tolerance_rad
                ),
                max_linear_mps=self.follower_config.start_egress_max_linear_mps,
            ),
        )
        if step is not None:
            return step
        self.start_egress_lock_index = None
        if waypoint_index != self.target_index:
            self._clear_intermediate_terminal_heading_latch(
                target_changed=True,
            )
        self.target_index = waypoint_index
        self.target_started_at = time.monotonic()
        self._reset_progress_watchdog(time.monotonic())
        return None

    def _certified_corner_decision(
        self,
        pose: Pose2D,
        controller_config: ControllerConfig,
    ) -> CertifiedCornerTransitionDecision:
        """Apply the physical discovery-route sharp-vertex contract."""

        if self.current_route_kind != "stand_discovery_corridor":
            self.certified_corner_latch = None
            return CertifiedCornerTransitionDecision(None, None)
        decision = compute_certified_corner_transition(
            pose,
            self.waypoints,
            self.target_index,
            controller_config,
            self.certified_corner_latch,
            CertifiedCornerControlConfig(
                turn_threshold_rad=(
                    self.follower_config.certified_corner_turn_threshold_rad
                ),
                release_tolerance_m=(
                    self.follower_config.certified_corner_release_tolerance_m
                ),
                hold_tolerance_m=(
                    self.follower_config.certified_corner_hold_tolerance_m
                ),
                alignment_tolerance_rad=(
                    self.follower_config
                    .certified_corner_alignment_tolerance_rad
                ),
            ),
        )
        self.certified_corner_latch = decision.latch
        return decision

    def _log_certified_corner_phase(
        self,
        step: ControllerStep | None,
    ) -> None:
        if step is None or not step.progress_mode.startswith("certified_corner_"):
            self._last_certified_corner_phase = None
            return
        phase = (step.target_index, step.progress_mode)
        if phase == self._last_certified_corner_phase:
            return
        self._last_certified_corner_phase = phase
        try:
            logger = self.get_logger()
        except Exception:
            return
        logger.info(
            "certified sharp-corner transition: "
            f"phase={step.progress_mode} target_index={step.target_index} "
            f"distance_m={step.distance_to_target_m:.6f} "
            f"heading_error_rad={step.controlled_heading_error_rad:.6f} "
            f"linear_mps={step.command.linear_x_mps:.6f} "
            f"angular_radps={step.command.angular_z_radps:.6f}"
        )

    def _reset_progress_watchdog(self, now_monotonic: float) -> None:
        self.last_progress_distance_m = math.inf
        self.last_progress_heading_error_rad = math.inf
        self.last_progress_target_index = None
        self.last_progress_pursuit_index = None
        self.last_progress_mode = None
        self.progress_heading_modes_seen.clear()
        self.progress_heading_error_by_mode.clear()
        self.last_progress_at = now_monotonic

    def _clear_intermediate_terminal_heading_latch(
        self,
        *,
        material_route_revision: bool = False,
        target_changed: bool = False,
    ) -> None:
        self.intermediate_terminal_heading_latch = (
            reset_intermediate_terminal_heading_latch(
                getattr(self, "intermediate_terminal_heading_latch", None),
                material_route_revision=material_route_revision,
                target_changed=target_changed,
            )
        )

    def run(self) -> FollowerResult:
        if len(self.waypoints) < 2:
            return FollowerResult("noop", "fewer than two waypoints", 0.0, 0.0, False)
        started_at = time.monotonic()
        if self.current_route_kind == "viewpoint_sampling":
            self.viewpoint_sampling_started_at = started_at
            self.viewpoint_sampling_target_started_at = started_at
        self.publish_repeated_zero()
        startup_failure = self._wait_for_initial_runtime_inputs(started_at)
        if startup_failure:
            self.publish_repeated_zero()
            return FollowerResult(
                "stopped",
                startup_failure,
                time.monotonic() - started_at,
                self.distance_estimate_m,
                self.motion_published,
            )
        loop_sleep_sec = 1.0 / max(self.follower_config.control_rate_hz, 1.0)
        try:
            while rclpy.ok():
                self._drain_runtime_callbacks()
                safety_failure = self._safety_failure()
                if safety_failure:
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "stopped",
                        safety_failure,
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        self.latest_stop_details,
                    )
                # Endpoint graph discovery in _safety_failure can briefly
                # delay TF listener callbacks.  Recover only that resulting
                # stale-transform case, while holding an explicit zero command.
                pose_lookup = self._current_pose_lookup_with_stale_recovery()
                pose = pose_lookup.pose
                if pose is None:
                    self.publish_repeated_zero()
                    stop_reason = str(
                        (pose_lookup.details or {}).get(
                            "stop_reason",
                            "map-to-base transform unavailable",
                        )
                    )
                    return FollowerResult(
                        "stopped",
                        stop_reason,
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        pose_lookup.details,
                    )
                route_refresh = self._refresh_dynamic_route(pose)
                if route_refresh == "stopped":
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "stopped",
                        str((self.latest_stop_details or {}).get("reason", "dynamic route withdrawn")),
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        self.latest_stop_details,
                    )
                if route_refresh == "completed":
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "completed",
                        "",
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        self.latest_stop_details,
                    )
                sampling_now = time.monotonic()
                sampling_timeout = viewpoint_sampling_timeout_failure(
                    route_kind=self.current_route_kind,
                    phase_started_at=self.viewpoint_sampling_started_at,
                    now_monotonic=sampling_now,
                    timeout_sec=self.follower_config.viewpoint_sampling_timeout_sec,
                )
                if not sampling_timeout:
                    sampling_timeout = viewpoint_sampling_target_timeout_failure(
                        route_kind=self.current_route_kind,
                        target_started_at=(
                            self.viewpoint_sampling_target_started_at
                        ),
                        now_monotonic=sampling_now,
                        timeout_sec=(
                            self.follower_config.viewpoint_sampling_target_timeout_sec
                        ),
                    )
                if sampling_timeout:
                    self.latest_stop_details = {
                        "reason": sampling_timeout,
                        "route_kind": self.current_route_kind,
                        "phase_elapsed_sec": (
                            None
                            if self.viewpoint_sampling_started_at is None
                            else time.monotonic()
                            - self.viewpoint_sampling_started_at
                        ),
                        "target_elapsed_sec": (
                            None
                            if self.viewpoint_sampling_target_started_at is None
                            else time.monotonic()
                            - self.viewpoint_sampling_target_started_at
                        ),
                        "phase_timeout_sec": (
                            self.follower_config.viewpoint_sampling_timeout_sec
                        ),
                        "target_timeout_sec": (
                            self.follower_config.viewpoint_sampling_target_timeout_sec
                        ),
                        "fail_closed": True,
                    }
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "stopped",
                        sampling_timeout.replace("_", " "),
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        self.latest_stop_details,
                    )
                if route_refresh == "adopted":
                    # A verified handoff still gets one complete zero-command
                    # control period before the new route may command motion.
                    self.publish_zero()
                    time.sleep(loop_sleep_sec)
                    continue
                if self.last_pose is not None:
                    self.distance_estimate_m += math.hypot(
                        pose.x_m - self.last_pose.x_m,
                        pose.y_m - self.last_pose.y_m,
                    )
                self.last_pose = pose
                if self.target_index == 0:
                    initial_failure = initial_pose_failure(
                        pose,
                        self.waypoints[0],
                        self.follower_config.initial_distance_limit_m,
                    )
                    if initial_failure:
                        self.publish_repeated_zero()
                        return FollowerResult(
                            "stopped",
                            initial_failure,
                            time.monotonic() - started_at,
                            self.distance_estimate_m,
                            self.motion_published,
                        )
                    if self.certified_static_start_pending:
                        startup_decision = certified_static_startup_decision(
                            pose,
                            self.waypoints,
                            tracking_tube_radius_m=(
                                self.follower_config.certified_route_tube_radius_m
                            ),
                            chord_sample_spacing_m=(
                                self.follower_config
                                .certified_route_chord_sample_spacing_m
                            ),
                        )
                        self.certified_static_start_pending = False
                        if not startup_decision.ok:
                            self.latest_stop_details = {
                                **startup_decision.route_check.to_log_dict(),
                                "reason": "pose outside certified startup segment",
                                "certificate_reason": (
                                    startup_decision.route_check.reason
                                ),
                                "startup_target_candidates": [0, 1],
                                "source": "execution_route_certificate",
                                "fail_closed": True,
                            }
                            self.publish_repeated_zero()
                            return FollowerResult(
                                "stopped",
                                "pose outside certified startup segment",
                                time.monotonic() - started_at,
                                self.distance_estimate_m,
                                self.motion_published,
                                self.latest_stop_details,
                            )
                        if startup_decision.target_index == 1:
                            self.target_index = 1
                            self.target_started_at = time.monotonic()
                            self._reset_progress_watchdog(time.monotonic())
                            # Make the bounded startup handoff observable as a
                            # complete zero-command control period.  No motion
                            # is published until the next loop rechecks all
                            # runtime safety inputs and the route tube.
                            self.publish_zero()
                            time.sleep(loop_sleep_sec)
                            continue
                if self.dynamic_join_pending:
                    join_action, join_failure = certified_startup_join_action(
                        pose,
                        self.waypoints[0],
                        self.dynamic_join_limit_m,
                        self.follower_config.dynamic_join_tolerance_m,
                    )
                    if join_action == "stop":
                        assert join_failure is not None
                        self.latest_stop_details = join_failure
                        self.publish_repeated_zero()
                        return FollowerResult(
                            "stopped",
                            str(join_failure["reason"]),
                            time.monotonic() - started_at,
                            self.distance_estimate_m,
                            self.motion_published,
                            self.latest_stop_details,
                        )
                    if join_action == "zero":
                        self.dynamic_join_pending = False
                        self.dynamic_join_limit_m = None
                        if self.target_index != 0:
                            self._clear_intermediate_terminal_heading_latch(
                                target_changed=True,
                            )
                        self.target_index = 0
                        self.target_started_at = time.monotonic()
                        self._reset_progress_watchdog(time.monotonic())
                        self.publish_zero()
                        time.sleep(loop_sleep_sec)
                        continue
                    # During handoff, pursue only the collision-certified route
                    # start.  Normal progress advancement/lookahead would form
                    # an unchecked chord from the live pose to waypoint 1.
                    step = compute_join_anchor_command(
                        pose,
                        self.waypoints[0],
                        controller_config_for_route_kind(
                            self.follower_config.controller,
                            self.current_route_kind,
                            reverse_staging=self.reverse_staging,
                            viewpoint_sampling_goal_tolerance_m=(
                                self.follower_config.viewpoint_sampling_goal_tolerance_m
                            ),
                            viewpoint_sampling_heading_tolerance_rad=(
                                self.follower_config.viewpoint_sampling_heading_tolerance_rad
                            ),
                            physical_goal_tolerance_m=(
                                self.follower_config.physical_goal_tolerance_m
                            ),
                            physical_waypoint_tolerance_m=(
                                self.follower_config.physical_waypoint_tolerance_m
                            ),
                        ),
                        join_tolerance_m=self.follower_config.dynamic_join_tolerance_m,
                    )
                else:
                    route_controller_config = controller_config_for_route_kind(
                        self.follower_config.controller,
                        self.current_route_kind,
                        reverse_staging=self.reverse_staging,
                        viewpoint_sampling_goal_tolerance_m=(
                            self.follower_config.viewpoint_sampling_goal_tolerance_m
                        ),
                        viewpoint_sampling_heading_tolerance_rad=(
                            self.follower_config.viewpoint_sampling_heading_tolerance_rad
                        ),
                        physical_goal_tolerance_m=(
                            self.follower_config.physical_goal_tolerance_m
                        ),
                        physical_waypoint_tolerance_m=(
                            self.follower_config.physical_waypoint_tolerance_m
                        ),
                    )
                    if self.start_egress_lock_index is not None:
                        step = self._start_egress_command(
                            pose,
                            route_controller_config,
                        )
                        if step is None:
                            # Make the lock-to-normal transition explicit; the
                            # next control tick may resume ordinary lookahead.
                            self.publish_zero()
                            time.sleep(loop_sleep_sec)
                            continue
                    else:
                        corner_decision = self._certified_corner_decision(
                            pose,
                            route_controller_config,
                        )
                        self._log_certified_corner_phase(corner_decision.step)
                        if corner_decision.failure:
                            failed_step = corner_decision.step
                            assert failed_step is not None
                            self.latest_stop_details = {
                                "reason": corner_decision.failure,
                                "source": "execution_route_certificate",
                                "route_kind": self.current_route_kind,
                                "target_index": failed_step.target_index,
                                "pursuit_index": failed_step.pursuit_index,
                                "distance_to_vertex_m": (
                                    failed_step.distance_to_target_m
                                ),
                                "release_tolerance_m": (
                                    self.follower_config
                                    .certified_corner_release_tolerance_m
                                ),
                                "hold_tolerance_m": (
                                    self.follower_config
                                    .certified_corner_hold_tolerance_m
                                ),
                                "tracking_tube_radius_m": (
                                    self.follower_config
                                    .certified_route_tube_radius_m
                                ),
                                "fail_closed": True,
                            }
                            self.publish_repeated_zero()
                            return FollowerResult(
                                "stopped",
                                corner_decision.failure,
                                time.monotonic() - started_at,
                                self.distance_estimate_m,
                                self.motion_published,
                                self.latest_stop_details,
                            )
                        if corner_decision.step is not None:
                            step = corner_decision.step
                        else:
                            terminal_heading_decision = (
                                compute_intermediate_terminal_heading_command(
                                    pose,
                                    self.waypoints,
                                    self.target_index,
                                    route_controller_config,
                                    self.current_route_kind,
                                    self.intermediate_terminal_heading_latch,
                                    hold_tolerance_m=(
                                        self.follower_config
                                        .viewpoint_sampling_terminal_heading_hold_tolerance_m
                                    ),
                                    viewpoint_sampling_target_distance_m=(
                                        self.follower_config
                                        .viewpoint_sampling_target_distance_m
                                    ),
                                    viewpoint_sampling_target_envelope_radius_m=(
                                        self.follower_config
                                        .viewpoint_sampling_terminal_heading_target_envelope_radius_m
                                    ),
                                )
                            )
                            self.intermediate_terminal_heading_latch = (
                                terminal_heading_decision.latch
                            )
                            step = terminal_heading_decision.step
                        if (
                            corner_decision.step is None
                            and terminal_heading_decision.failure
                        ):
                            hold_diagnostics = (
                                intermediate_terminal_heading_hold_diagnostics(
                                    pose,
                                    terminal_heading_decision.latch,
                                    hold_tolerance_m=(
                                        self.follower_config
                                        .viewpoint_sampling_terminal_heading_hold_tolerance_m
                                    ),
                                    viewpoint_sampling_target_distance_m=(
                                        self.follower_config
                                        .viewpoint_sampling_target_distance_m
                                    ),
                                    viewpoint_sampling_target_envelope_radius_m=(
                                        self.follower_config
                                        .viewpoint_sampling_terminal_heading_target_envelope_radius_m
                                    ),
                                )
                                if terminal_heading_decision.latch is not None
                                else {}
                            )
                            self.latest_stop_details = {
                                "reason": terminal_heading_decision.failure,
                                "fault_code": terminal_heading_decision.failure,
                                "route_kind": self.current_route_kind,
                                "target_index": step.target_index,
                                "distance_to_target_m": step.distance_to_target_m,
                                "entry_tolerance_m": (
                                    intermediate_terminal_heading_entry_tolerance_m(
                                        route_controller_config
                                    )
                                ),
                                "hold_tolerance_m": (
                                    self.follower_config
                                    .viewpoint_sampling_terminal_heading_hold_tolerance_m
                                ),
                                "distance_comparison_epsilon_m": (
                                    INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M
                                ),
                                "effective_hold_limit_m": (
                                    self.follower_config
                                    .viewpoint_sampling_terminal_heading_hold_tolerance_m
                                    + INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M
                                ),
                                **hold_diagnostics,
                                "fail_closed": True,
                            }
                            self.publish_repeated_zero()
                            return FollowerResult(
                                "stopped",
                                terminal_heading_decision.failure.replace("_", " "),
                                time.monotonic() - started_at,
                                self.distance_estimate_m,
                                self.motion_published,
                                self.latest_stop_details,
                            )
                if (
                    self.current_route_kind in PHYSICAL_ROUTE_KINDS
                    and not self.dynamic_join_pending
                ):
                    route_check = check_execution_route_tube(
                        pose,
                        self.waypoints,
                        target_index=step.target_index,
                        pursuit_index=step.pursuit_index,
                        tracking_tube_radius_m=(
                            self.follower_config.certified_route_tube_radius_m
                        ),
                        chord_sample_spacing_m=(
                            self.follower_config.certified_route_chord_sample_spacing_m
                        ),
                    )
                    if not route_check.ok:
                        self.latest_stop_details = route_check.to_log_dict()
                        self.publish_repeated_zero()
                        return FollowerResult(
                            "stopped",
                            route_check.reason,
                            time.monotonic() - started_at,
                            self.distance_estimate_m,
                            self.motion_published,
                            self.latest_stop_details,
                        )
                if step.target_index != self.target_index:
                    self._clear_intermediate_terminal_heading_latch(
                        target_changed=True,
                    )
                    self.target_index = step.target_index
                    self.certified_corner_latch = None
                    self.target_started_at = time.monotonic()
                    self._reset_progress_watchdog(time.monotonic())
                if step.reached_goal:
                    now_monotonic = time.monotonic()
                    if self.axis_acquisition_hold_started_at is None:
                        self.axis_acquisition_hold_started_at = now_monotonic
                    hold_elapsed = (
                        now_monotonic - self.axis_acquisition_hold_started_at
                    )
                    goal_action = acquisition_goal_action(
                        route_kind=self.current_route_kind,
                        provider_available=self.waypoint_provider is not None,
                        hold_elapsed_sec=hold_elapsed,
                        timeout_sec=(
                            self.follower_config.axis_acquisition_wait_timeout_sec
                        ),
                    )
                    if goal_action == "hold_for_physical_face":
                        # Remain stationary but keep spinning sensor callbacks
                        # and polling the manifest. A physical-face revision
                        # will be adopted at the top of a subsequent cycle.
                        self.publish_zero()
                        time.sleep(loop_sleep_sec)
                        continue
                    if goal_action != "complete":
                        self.latest_stop_details = {
                            "reason": goal_action,
                            "route_kind": self.current_route_kind,
                            "hold_elapsed_sec": hold_elapsed,
                            "timeout_sec": (
                                self.follower_config.axis_acquisition_wait_timeout_sec
                            ),
                            "fail_closed": True,
                        }
                        self.publish_repeated_zero()
                        return FollowerResult(
                            "stopped",
                            goal_action.replace("_", " "),
                            time.monotonic() - started_at,
                            self.distance_estimate_m,
                            self.motion_published,
                            self.latest_stop_details,
                        )
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "completed",
                        "",
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                    )
                timeout_now = time.monotonic()
                timeout_elapsed = timeout_now - self.target_started_at
                timeout_failure = waypoint_timeout_failure(
                    timeout_elapsed,
                    self.follower_config.waypoint_timeout_sec,
                )
                if timeout_failure:
                    self.latest_stop_details = {
                        "reason": timeout_failure,
                        "route_kind": self.current_route_kind,
                        "elapsed_sec": timeout_elapsed,
                        "timeout_sec": self.follower_config.waypoint_timeout_sec,
                        "target_index": step.target_index,
                        "pursuit_index": step.pursuit_index,
                        "distance_to_target_m": step.distance_to_target_m,
                        "progress_mode": step.progress_mode,
                        "axis_acquisition_target_revision": (
                            self.axis_acquisition_target_revision
                        ),
                        "viewpoint_sampling_target_revision": (
                            self.viewpoint_sampling_target_revision
                        ),
                        "robot_pose": {
                            "x_m": pose.x_m,
                            "y_m": pose.y_m,
                            "yaw_rad": pose.yaw_rad,
                        },
                        "fail_closed": True,
                    }
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "stopped",
                        timeout_failure,
                        timeout_now - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        self.latest_stop_details,
                    )
                now_monotonic = time.monotonic()
                front_clearance_scale = self._motion_clearance_linear_scale(
                    step.command.linear_x_mps
                )
                effective_linear_x_mps = step.command.linear_x_mps * front_clearance_scale
                progress_failure = self._progress_failure(
                    step.distance_to_target_m,
                    step.controlled_heading_error_rad,
                    step.target_index,
                    step.pursuit_index,
                    now_monotonic,
                    (
                        abs(step.command.linear_x_mps) > 0.0
                        or abs(step.command.angular_z_radps) > 0.0
                    ),
                    step.progress_mode,
                )
                if progress_failure:
                    self.latest_stop_details = stuck_progress_details(
                        target_index=self.target_index,
                        distance_to_target_m=step.distance_to_target_m,
                        last_progress_distance_m=self.last_progress_distance_m,
                        elapsed_without_progress_sec=now_monotonic - self.last_progress_at,
                        max_without_progress_sec=self.follower_config.stuck_timeout_sec,
                        progress_epsilon_m=self.follower_config.stuck_progress_epsilon_m,
                        commanded_linear_x_mps=step.command.linear_x_mps,
                        commanded_angular_z_radps=step.command.angular_z_radps,
                        front_clearance_scale=front_clearance_scale,
                        effective_linear_x_mps=effective_linear_x_mps,
                        front_clearance_details=self.latest_front_clearance_details,
                        pursuit_index=step.pursuit_index,
                        controlled_heading_error_rad=(
                            step.controlled_heading_error_rad
                        ),
                        last_progress_heading_error_rad=(
                            self.last_progress_heading_error_rad
                        ),
                        heading_progress_epsilon_rad=(
                            self.follower_config.stuck_heading_progress_epsilon_rad
                        ),
                        last_progress_target_index=(
                            self.last_progress_target_index
                        ),
                        last_progress_pursuit_index=(
                            self.last_progress_pursuit_index
                        ),
                    )
                    self.publish_repeated_zero()
                    return FollowerResult(
                        "stopped",
                        progress_failure,
                        time.monotonic() - started_at,
                        self.distance_estimate_m,
                        self.motion_published,
                        self.latest_stop_details,
                    )
                twist = Twist()
                twist.linear.x = effective_linear_x_mps
                twist.angular.z = step.command.angular_z_radps
                self.cmd_vel_pub.publish(twist)
                self.motion_published = self.motion_published or abs(twist.linear.x) > 0.0 or abs(twist.angular.z) > 0.0
                time.sleep(loop_sleep_sec)
        finally:
            self.publish_repeated_zero()

    def _refresh_dynamic_route(self, pose: Pose2D) -> str:
        if self.waypoint_provider is None:
            return ""
        now = time.monotonic()
        initial_refresh = self.initial_route_refresh_pending
        if (
            not initial_refresh
            and self.follower_config.dynamic_route_refresh_sec <= 0.0
        ):
            return ""
        if (
            not initial_refresh
            and now - self.last_route_refresh_at
            < self.follower_config.dynamic_route_refresh_sec
        ):
            return ""
        self.initial_route_refresh_pending = False
        self.last_route_refresh_at = now
        try:
            update = self.waypoint_provider(pose)
        except Exception as exc:
            self.latest_stop_details = {
                "reason": f"dynamic route provider failed: {exc}",
                "fault_code": "route_provider_exception",
                "fail_closed": True,
            }
            return "stopped"
        if update is None:
            return ""
        if update.kind is RouteUpdateKind.UNCHANGED:
            return ""
        if update.kind is RouteUpdateKind.REJECT:
            self.publish_zero()
            if not self._emit_route_update(update):
                return "stopped"
            self.latest_stop_details = {
                **dict(update.event_fields),
                "reason": update.reason or "dynamic route update rejected",
                "fail_closed": True,
            }
            return "stopped"
        if update.kind is RouteUpdateKind.STOP:
            # Zero first: semantic logging is synchronous and must never leave
            # the previous nonzero Twist active if it blocks or raises.
            self.publish_zero()
            if not self._emit_route_update(update):
                return "stopped"
            self.latest_stop_details = {
                **dict(update.event_fields),
                "reason": update.reason or "dynamic route withdrawn",
            }
            return "stopped"
        if update.kind is RouteUpdateKind.COMPLETE:
            # A committed arrival estimate is the successful terminal event
            # for a survey leg.  Stop before logging so a slow callback can
            # never leave a previous non-zero Twist active.
            self.publish_zero()
            if not self._emit_route_update(update):
                return "stopped"
            self.latest_stop_details = {
                **dict(update.event_fields),
                "reason": update.reason or "survey completed",
                "fail_closed": False,
            }
            return "completed"
        replacement = tuple(update.waypoints)
        if update.kind is not RouteUpdateKind.ADOPT or len(replacement) < 2:
            self.publish_zero()
            self.latest_stop_details = {
                "reason": "dynamic route adoption contained fewer than two waypoints",
                "fault_code": "invalid_route_update",
                "fail_closed": True,
            }
            return "stopped"
        if update.target_index is None or not 0 <= update.target_index < len(replacement):
            self.publish_zero()
            self.latest_stop_details = {
                "reason": "dynamic route adoption contained an invalid target index",
                "fault_code": "invalid_route_update",
                "fail_closed": True,
            }
            return "stopped"
        try:
            join_limit = float(update.event_fields["effective_join_limit_m"])
        except (KeyError, TypeError, ValueError) as exc:
            self.publish_zero()
            self.latest_stop_details = {
                "reason": f"dynamic route adoption lacks a valid join envelope: {exc}",
                "fault_code": "invalid_route_update",
                "fail_closed": True,
            }
            return "stopped"
        if not math.isfinite(join_limit) or join_limit <= 0.0:
            self.publish_zero()
            self.latest_stop_details = {
                "reason": "dynamic route adoption join envelope is not positive and finite",
                "fault_code": "invalid_route_update",
                "fail_closed": True,
            }
            return "stopped"
        next_route_kind = str(update.event_fields.get("route_kind", ""))
        phase_failure = dynamic_route_kind_transition_failure(
            self.current_route_kind, next_route_kind
        )
        if phase_failure:
            self.publish_zero()
            self.latest_stop_details = {
                "reason": phase_failure,
                "fault_code": "invalid_route_phase",
                "current_route_kind": self.current_route_kind,
                "next_route_kind": next_route_kind,
                "fail_closed": True,
            }
            return "stopped"
        raw_egress_lock = update.event_fields.get(
            "start_egress_vertex_lock",
            False,
        )
        if not isinstance(raw_egress_lock, bool):
            self.publish_zero()
            self.latest_stop_details = {
                "reason": "dynamic route start-egress lock flag is not boolean",
                "fault_code": "invalid_route_update",
                "fail_closed": True,
            }
            return "stopped"
        next_egress_lock_index = None
        if raw_egress_lock:
            raw_lock_index = update.event_fields.get(
                "start_egress_waypoint_index"
            )
            clearance_validated = update.event_fields.get(
                "start_egress_continuous_clearance_validated"
            )
            if (
                not isinstance(raw_lock_index, int)
                or isinstance(raw_lock_index, bool)
                or raw_lock_index != 1
                or raw_lock_index >= len(replacement)
                or clearance_validated is not True
            ):
                self.publish_zero()
                self.latest_stop_details = {
                    "reason": "dynamic route start-egress certificate is malformed",
                    "fault_code": "invalid_route_update",
                    "fail_closed": True,
                }
                return "stopped"
            next_egress_lock_index = raw_lock_index
        previous_route_kind = self.current_route_kind
        self.publish_zero()
        self._clear_intermediate_terminal_heading_latch(
            material_route_revision=True,
        )
        self.certified_corner_latch = None
        self._last_certified_corner_phase = None
        self.waypoints = replacement
        # Every route replacement owns a fresh lock decision. Ordinary routes
        # explicitly clear any lock retained from the previous revision.
        self.start_egress_lock_index = next_egress_lock_index
        self.current_route_kind = next_route_kind
        self.reverse_staging = (
            next_route_kind in PHYSICAL_ROUTE_KINDS
            and reverse_staging_is_preferred(pose, replacement)
        )
        if next_route_kind in PHYSICAL_ROUTE_KINDS:
            update = replace(
                update,
                event_fields={
                    **dict(update.event_fields),
                    "staging_motion": (
                        "reverse" if self.reverse_staging else "forward"
                    ),
                    "physical_goal_tolerance_m": (
                        self.follower_config.physical_goal_tolerance_m
                    ),
                    "physical_waypoint_tolerance_m": (
                        self.follower_config.physical_waypoint_tolerance_m
                    ),
                },
            )
        if next_route_kind != previous_route_kind:
            self.axis_acquisition_hold_started_at = None
            self.axis_acquisition_target_revision = (
                update.target_revision
                if next_route_kind == "axis_acquisition"
                else None
            )
            self.viewpoint_sampling_started_at = (
                now if next_route_kind == "viewpoint_sampling" else None
            )
            self.viewpoint_sampling_target_started_at = (
                now if next_route_kind == "viewpoint_sampling" else None
            )
            self.viewpoint_sampling_target_revision = (
                update.target_revision
                if next_route_kind == "viewpoint_sampling"
                else None
            )
        elif next_route_kind == "viewpoint_sampling":
            if self.viewpoint_sampling_target_revision is None:
                self.viewpoint_sampling_target_revision = update.target_revision
            elif (
                update.target_revision is not None
                and update.target_revision
                > self.viewpoint_sampling_target_revision
            ):
                # A material target revision may move the sampling point. Give
                # the new point its own bounded convergence window; identical
                # geometry heartbeats are filtered before ADOPT and do not
                # reset this clock.
                self.viewpoint_sampling_target_started_at = now
                self.viewpoint_sampling_target_revision = (
                    update.target_revision
                )
        elif next_route_kind == "axis_acquisition":
            if self.axis_acquisition_target_revision is None:
                self.axis_acquisition_target_revision = update.target_revision
            elif (
                update.target_revision is not None
                and update.target_revision
                > self.axis_acquisition_target_revision
            ):
                # A bounded acquisition sweep installed a genuinely new ray.
                # Fresh route heartbeats are UNCHANGED and cannot extend this
                # hold window.
                self.axis_acquisition_hold_started_at = None
                self.axis_acquisition_target_revision = update.target_revision
        self.target_index = update.target_index
        self.target_started_at = now
        self._reset_progress_watchdog(now)
        self.last_pose = pose
        self.dynamic_join_pending = True
        self.dynamic_join_limit_m = join_limit
        if not self._emit_route_update(update):
            return "stopped"
        return "adopted"

    def _emit_route_update(self, update: RouteUpdate) -> bool:
        if update.event_name is None or self.route_update_callback is None:
            return True
        try:
            self.route_update_callback(update)
        except Exception as exc:
            self.latest_stop_details = {
                **dict(update.event_fields),
                "reason": f"semantic event callback failed: {exc}",
                "fault_code": "route_event_callback_exception",
                "fail_closed": True,
            }
            return False
        return True

    def _wait_for_initial_runtime_inputs(self, started_at: float) -> str:
        deadline = started_at + self.follower_config.initial_sensor_wait_sec
        last_failure = "missing scan"
        while rclpy.ok():
            self._service_or_wait_for_callbacks(0.05)
            scan_failure = self._freshness_failure(
                "scan",
                self.latest_scan,
                self.latest_scan_receipt,
                self.follower_config.max_scan_age_sec,
            )
            if scan_failure:
                last_failure = scan_failure
            else:
                odom_failure = self._freshness_failure(
                    "odom",
                    self.latest_odom,
                    self.latest_odom_receipt,
                    self.follower_config.max_odom_age_sec,
                )
                if odom_failure:
                    last_failure = odom_failure
                else:
                    pose_lookup = self._current_pose_lookup()
                    if pose_lookup.pose is None:
                        self.latest_stop_details = pose_lookup.details
                        last_failure = "map-to-base transform unavailable"
                    else:
                        return ""
            if time.monotonic() >= deadline:
                return last_failure
            self.publish_zero()
        return "ROS shutdown"

    def _safety_failure(self) -> str:
        self.latest_stop_details = None
        scan_failure = self._freshness_failure("scan", self.latest_scan, self.latest_scan_receipt, self.follower_config.max_scan_age_sec)
        if scan_failure:
            return scan_failure
        odom_failure = self._freshness_failure("odom", self.latest_odom, self.latest_odom_receipt, self.follower_config.max_odom_age_sec)
        if odom_failure:
            return odom_failure
        obstacle_failure = self._obstacle_failure()
        if obstacle_failure:
            return obstacle_failure
        ownership_failure = self._cmd_vel_ownership_failure()
        if ownership_failure:
            return ownership_failure
        return ""

    def _freshness_failure(self, name: str, msg, receipt, max_age_sec: float) -> str:
        if msg is None or receipt is None:
            failure = message_freshness_failure(
                name,
                has_message=False,
                receipt_age_sec=None,
                header_age_sec=None,
                max_age_sec=max_age_sec,
            )
            self.latest_stop_details = {
                "reason": failure,
                "source": "message_freshness",
                "sensor": name,
                "has_message": False,
                "receipt_age_sec": None,
                "header_age_sec": None,
                "max_age_sec": max_age_sec,
                "fail_closed": True,
            }
            return failure
        now = self.get_clock().now()
        receipt_age = time.monotonic() - float(receipt)
        header_age = (now - Time.from_msg(msg.header.stamp)).nanoseconds / 1_000_000_000.0
        failure = message_freshness_failure(
            name,
            has_message=True,
            receipt_age_sec=receipt_age,
            header_age_sec=header_age,
            max_age_sec=max_age_sec,
            max_future_sec=self.follower_config.max_future_timestamp_sec,
        )
        if failure:
            self.latest_stop_details = {
                "reason": failure,
                "source": "message_freshness",
                "sensor": name,
                "has_message": True,
                "receipt_age_sec": receipt_age,
                "header_age_sec": header_age,
                "max_age_sec": max_age_sec,
                "max_future_sec": self.follower_config.max_future_timestamp_sec,
                "receipt_stale": receipt_age > max_age_sec,
                "header_stale": header_age > max_age_sec,
                "receipt_future": (
                    receipt_age < -self.follower_config.max_future_timestamp_sec
                ),
                "header_future": (
                    header_age < -self.follower_config.max_future_timestamp_sec
                ),
                "fail_closed": True,
            }
        return failure

    def _scan_range_min(self) -> float | None:
        return float(getattr(self.latest_scan, "range_min")) if hasattr(self.latest_scan, "range_min") else None

    def _scan_range_max(self) -> float | None:
        return float(getattr(self.latest_scan, "range_max")) if hasattr(self.latest_scan, "range_max") else None

    def _obstacle_failure(self) -> str:
        decision = obstacle_decision(
            getattr(self.latest_scan, "ranges", None),
            self.follower_config.min_obstacle_distance_m,
            range_min_m=self._scan_range_min(),
            range_max_m=self._scan_range_max(),
        )
        if decision.stop_reason:
            self.latest_stop_details = decision.to_log_dict()
        return decision.stop_reason

    def _motion_clearance_linear_scale(self, linear_x_mps: float) -> float:
        if self.latest_scan is None:
            return 1.0
        if abs(linear_x_mps) <= 1.0e-12:
            self.latest_front_clearance_details = None
            return 1.0
        reversing = linear_x_mps < 0.0
        decision = front_sector_decision(
            getattr(self.latest_scan, "ranges", None),
            float(getattr(self.latest_scan, "angle_min", 0.0)),
            float(getattr(self.latest_scan, "angle_increment", 0.0)),
            math.pi if reversing else 0.0,
            self.follower_config.front_obstacle_sector_rad,
            self.follower_config.min_obstacle_distance_m,
            range_min_m=self._scan_range_min(),
            range_max_m=self._scan_range_max(),
            source="rear_sector" if reversing else "front_sector",
        )
        self.latest_front_clearance_details = decision.to_log_dict()
        if decision.stop_reason in (NO_VALID_FRONT_SECTOR_SCAN_RANGES, OBSTACLE_TOO_CLOSE):
            return 0.0
        return linear_scale_for_front_clearance(
            decision.nearest_valid_range_m,
            self.follower_config.min_obstacle_distance_m,
            self.follower_config.front_obstacle_slow_distance_m,
        )

    def _progress_failure(
        self,
        distance_to_target_m: float,
        controlled_heading_error_rad: float,
        target_index: int,
        pursuit_index: int,
        now_monotonic: float,
        motion_commanded: bool,
        progress_mode: str = "path_tracking",
    ) -> str:
        heading_error_abs = abs(controlled_heading_error_rad)
        target_changed = target_index != self.last_progress_target_index
        pursuit_advanced = (
            not target_changed
            and pursuit_index > self.last_progress_pursuit_index
        )
        # Pure-pursuit progress is monotonic. A pursuit-index regression or
        # same-target chatter must not renew the stuck watchdog indefinitely.
        indices_changed = target_changed or pursuit_advanced
        heading_mode_first_entry = (
            progress_mode
            in {
                "exact_vertex_alignment",
                "heading_corridor",
                "terminal_heading",
            }
            and progress_mode not in self.progress_heading_modes_seen
        )
        if indices_changed:
            self.progress_heading_modes_seen.clear()
            self.progress_heading_error_by_mode.clear()
            heading_mode_first_entry = progress_mode in {
                "exact_vertex_alignment",
                "heading_corridor",
                "terminal_heading",
            }
        if heading_mode_first_entry:
            self.progress_heading_modes_seen.add(progress_mode)
        if indices_changed or heading_mode_first_entry:
            self.last_progress_distance_m = distance_to_target_m
            self.last_progress_heading_error_rad = (
                heading_error_abs if math.isfinite(heading_error_abs) else math.inf
            )
            self.progress_heading_error_by_mode[progress_mode] = (
                self.last_progress_heading_error_rad
            )
            self.last_progress_target_index = target_index
            self.last_progress_pursuit_index = pursuit_index
            self.last_progress_mode = progress_mode
            self.last_progress_at = now_monotonic
            return ""
        self.last_progress_mode = progress_mode
        mode_heading_baseline = self.progress_heading_error_by_mode.get(
            progress_mode
        )
        if mode_heading_baseline is None:
            # Path-bearing and terminal-yaw errors are different metrics.  A
            # tolerance-boundary chatter may enter a mode for the first time
            # without constituting progress, so establish its own baseline
            # without renewing the watchdog deadline.
            mode_heading_baseline = (
                heading_error_abs if math.isfinite(heading_error_abs) else math.inf
            )
            self.progress_heading_error_by_mode[progress_mode] = (
                mode_heading_baseline
            )
        self.last_progress_heading_error_rad = mode_heading_baseline
        distance_improved = (
            distance_to_target_m + self.follower_config.stuck_progress_epsilon_m
            < self.last_progress_distance_m
        )
        heading_improved = (
            math.isfinite(heading_error_abs)
            and heading_error_abs
            + self.follower_config.stuck_heading_progress_epsilon_rad
            < mode_heading_baseline
        )
        if distance_improved:
            self.last_progress_distance_m = distance_to_target_m
        if heading_improved:
            self.progress_heading_error_by_mode[progress_mode] = heading_error_abs
            self.last_progress_heading_error_rad = heading_error_abs
        if distance_improved or heading_improved:
            self.last_progress_at = now_monotonic
            return ""
        return stuck_progress_failure(
            now_monotonic - self.last_progress_at,
            self.follower_config.stuck_timeout_sec,
            motion_commanded,
        )

    def _cmd_vel_ownership_failure(self) -> str:
        publishers = self.get_publishers_info_by_topic(self.runtime_config.cmd_vel_topic)
        publisher_identities = sorted({_node_identity(publisher) for publisher in publishers})
        self_identity = _format_node_identity(self.get_namespace(), self.get_name())
        return cmd_vel_ownership_failure(
            publisher_identities,
            self_identity,
            # Allow-lists are useful for preflight discovery, but publishing
            # begins only when this process is the sole cmd_vel owner.
            (),
        )

    def _current_pose_lookup(self) -> PoseLookupResult:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.runtime_config.map_frame,
                self.runtime_config.base_frame,
                Time(),
                timeout=Duration(seconds=0.1),
            )
        except TransformException as exc:
            return PoseLookupResult(
                None,
                tf_lookup_failure_details(
                    reason="lookup_exception",
                    target_frame=self.runtime_config.map_frame,
                    source_frame=self.runtime_config.base_frame,
                    max_age_sec=self.follower_config.max_tf_age_sec,
                    exception=exc,
                ),
            )
        transform_stamp = Time.from_msg(transform.header.stamp)
        age = (
            self.get_clock().now() - transform_stamp
        ).nanoseconds / 1_000_000_000.0
        stamp_sec = transform_stamp.nanoseconds / 1_000_000_000.0
        if age < -self.follower_config.max_future_timestamp_sec:
            return PoseLookupResult(
                None,
                tf_lookup_failure_details(
                    reason="future_transform",
                    target_frame=self.runtime_config.map_frame,
                    source_frame=self.runtime_config.base_frame,
                    max_age_sec=self.follower_config.max_tf_age_sec,
                    age_sec=age,
                ),
                stamp_sec,
            )
        if age > self.follower_config.max_tf_age_sec:
            return PoseLookupResult(
                None,
                tf_lookup_failure_details(
                    reason="stale_transform",
                    target_frame=self.runtime_config.map_frame,
                    source_frame=self.runtime_config.base_frame,
                    max_age_sec=self.follower_config.max_tf_age_sec,
                    age_sec=age,
                ),
                stamp_sec,
            )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return PoseLookupResult(
            Pose2D(translation.x, translation.y, _yaw_from_quaternion(rotation)),
            stamp_sec=stamp_sec,
        )

    def _post_stale_tf_recovery_freshness_failure(self) -> str:
        scan_failure = self._freshness_failure(
            "scan",
            self.latest_scan,
            self.latest_scan_receipt,
            self.follower_config.max_scan_age_sec,
        )
        if scan_failure:
            return scan_failure
        return self._freshness_failure(
            "odom",
            self.latest_odom,
            self.latest_odom_receipt,
            self.follower_config.max_odom_age_sec,
        )

    def _fallback_message_freshness_evidence(
        self,
        name: str,
        msg,
        receipt,
        max_age_sec: float,
    ) -> dict[str, object]:
        """Apply the ordinary freshness gate and retain its predicate evidence."""

        try:
            failure = self._freshness_failure(
                name,
                msg,
                receipt,
                max_age_sec,
            )
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            return {
                "sensor": name,
                "fresh": False,
                "failure": f"malformed {name} freshness data",
                "exception_type": exc.__class__.__name__,
                "exception": str(exc),
                "receipt_age_sec": None,
                "header_age_sec": None,
                "max_age_sec": max_age_sec,
                "max_future_sec": getattr(
                    self.follower_config,
                    "max_future_timestamp_sec",
                    None,
                ),
            }

        if failure:
            stop_details = dict(self.latest_stop_details or {})
            return {
                "sensor": name,
                "fresh": False,
                "failure": failure,
                "receipt_age_sec": stop_details.get("receipt_age_sec"),
                "header_age_sec": stop_details.get("header_age_sec"),
                "max_age_sec": max_age_sec,
                "max_future_sec": stop_details.get(
                    "max_future_sec",
                    getattr(
                        self.follower_config,
                        "max_future_timestamp_sec",
                        None,
                    ),
                ),
                "receipt_stale": stop_details.get("receipt_stale"),
                "header_stale": stop_details.get("header_stale"),
                "receipt_future": stop_details.get("receipt_future"),
                "header_future": stop_details.get("header_future"),
            }

        try:
            receipt_age_sec = time.monotonic() - float(receipt)
            header_age_sec = (
                self.get_clock().now() - Time.from_msg(msg.header.stamp)
            ).nanoseconds / 1_000_000_000.0
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            # This branch is conservative: the ordinary gate just passed, but
            # evidence extraction itself was not trustworthy.
            return {
                "sensor": name,
                "fresh": False,
                "failure": f"malformed {name} timing evidence",
                "exception_type": exc.__class__.__name__,
                "exception": str(exc),
                "receipt_age_sec": None,
                "header_age_sec": None,
                "max_age_sec": max_age_sec,
                "max_future_sec": getattr(
                    self.follower_config,
                    "max_future_timestamp_sec",
                    None,
                ),
            }
        return {
            "sensor": name,
            "fresh": True,
            "failure": "",
            "receipt_age_sec": receipt_age_sec,
            "header_age_sec": header_age_sec,
            "max_age_sec": max_age_sec,
            "max_future_sec": self.follower_config.max_future_timestamp_sec,
            "receipt_stale": False,
            "header_stale": False,
            "receipt_future": False,
            "header_future": False,
        }

    def _semantic_event_failure_lookup(
        self,
        *,
        event_name: str,
        stamp_sec: float | None,
    ) -> PoseLookupResult:
        callback_failure = dict(self.latest_stop_details or {})
        return PoseLookupResult(
            None,
            {
                "stop_reason": callback_failure.get(
                    "reason",
                    "semantic event callback failed",
                ),
                "source": "semantic_event_callback",
                "event_name": event_name,
                "semantic_event_failure": callback_failure,
                "fail_closed": True,
            },
            stamp_sec,
        )

    def _primary_tf_result_with_restore_event(
        self,
        result: PoseLookupResult,
        *,
        recovered_after_retry: bool,
    ) -> PoseLookupResult:
        if (
            result.pose is None
            or not getattr(
                self,
                "_simulation_odom_fallback_active",
                False,
            )
        ):
            return result

        # The prior command may have been nonzero.  Hold zero while the
        # semantic source transition is synchronously committed.
        self.publish_zero()
        event_name = "simulation_odom_pose_fallback_restored"
        event_fields = {
            "source": "tf_lookup",
            "pose_source": "tf_lookup",
            "primary_tf_stamp_sec": result.stamp_sec,
            "recovered_after_retry": recovered_after_retry,
            "fallback_episode": getattr(
                self,
                "_simulation_odom_fallback_episode",
                0,
            ),
            "not_real_robot_migration_evidence": True,
        }
        if not self._emit_route_update(
            RouteUpdate(
                kind=RouteUpdateKind.UNCHANGED,
                event_name=event_name,
                event_fields=event_fields,
            )
        ):
            return self._semantic_event_failure_lookup(
                event_name=event_name,
                stamp_sec=result.stamp_sec,
            )
        self._simulation_odom_fallback_active = False
        return result

    def _simulation_odom_fallback_after_stale_retry(
        self,
        *,
        first_lookup: PoseLookupResult,
        retry_lookup: PoseLookupResult,
        callback_drain: dict[str, object],
        odom_callback_count_before: int,
        odom_callback_count_after: int,
        odom_msg,
        odom_receipt,
        scan_msg,
        scan_receipt,
    ) -> PoseLookupResult:
        """Validate one explicit Gazebo-only direct-odometry recovery."""

        runtime = getattr(self, "runtime_config", None)
        follower_config = getattr(self, "follower_config", None)
        enabled = (
            getattr(
                follower_config,
                "allow_simulation_odom_after_stale_tf",
                False,
            )
            is True
        )
        if not enabled:
            disabled_details = {
                "source": SIMULATION_ODOM_FALLBACK_SOURCE,
                "pose_source": SIMULATION_ODOM_FALLBACK_SOURCE,
                "attempted": False,
                "accepted": False,
                "fail_closed": True,
                "zero_published_before_fallback": True,
                "not_real_robot_migration_evidence": True,
                "first_lookup": _pose_lookup_diagnostics(first_lookup),
                "retry_lookup": _pose_lookup_diagnostics(retry_lookup),
                "callback_drain": callback_drain,
                "predicates": {
                    "explicitly_enabled": False,
                },
                "rejection_reasons": ["explicitly_enabled"],
            }
            original_failure = _stale_tf_recovery_failure_details(
                dict(retry_lookup.details or {}),
                first_lookup=first_lookup,
                retry_lookup=retry_lookup,
                callback_drain=callback_drain,
            )
            original_failure["simulation_odom_fallback"] = disabled_details
            return PoseLookupResult(
                None,
                original_failure,
                retry_lookup.stamp_sec,
            )
        use_sim_time = getattr(runtime, "use_sim_time", False) is True
        localization_is_tf = getattr(runtime, "localization_source", "") == "tf"
        map_frame = str(getattr(runtime, "map_frame", ""))
        odom_frame = str(getattr(runtime, "odom_frame", ""))
        base_frame = str(getattr(runtime, "base_frame", ""))
        map_frame_is_odom_frame = bool(map_frame) and map_frame == odom_frame

        odom_freshness = self._fallback_message_freshness_evidence(
            "odom",
            odom_msg,
            odom_receipt,
            getattr(follower_config, "max_odom_age_sec", 1.0),
        )
        scan_freshness = self._fallback_message_freshness_evidence(
            "scan",
            scan_msg,
            scan_receipt,
            getattr(follower_config, "max_scan_age_sec", 1.0),
        )

        header = getattr(odom_msg, "header", None)
        odom_message_frame = str(getattr(header, "frame_id", ""))
        odom_child_frame = str(getattr(odom_msg, "child_frame_id", ""))
        odom_stamp_sec = _ros_stamp_sec(getattr(header, "stamp", None))
        pose_message = getattr(getattr(odom_msg, "pose", None), "pose", None)
        position = getattr(pose_message, "position", None)
        orientation = getattr(pose_message, "orientation", None)

        def numeric(attribute_owner, attribute_name: str) -> float | None:
            try:
                value = float(getattr(attribute_owner, attribute_name))
            except (AttributeError, TypeError, ValueError, OverflowError):
                return None
            return value if math.isfinite(value) else None

        x_m = numeric(position, "x")
        y_m = numeric(position, "y")
        quaternion = {
            field_name: numeric(orientation, field_name)
            for field_name in ("x", "y", "z", "w")
        }
        quaternion_finite = all(
            value is not None for value in quaternion.values()
        )
        quaternion_norm = (
            math.sqrt(
                sum(float(value) ** 2 for value in quaternion.values())
            )
            if quaternion_finite
            else None
        )
        quaternion_norm_valid = (
            quaternion_norm is not None
            and abs(quaternion_norm - 1.0)
            <= SIMULATION_ODOM_FALLBACK_QUATERNION_NORM_TOLERANCE
        )
        yaw_rad = None
        if quaternion_finite:
            yaw_candidate = _yaw_from_quaternion(
                type(
                    "_QuaternionValue",
                    (),
                    quaternion,
                )()
            )
            if math.isfinite(yaw_candidate):
                yaw_rad = yaw_candidate

        retry_reason = str((retry_lookup.details or {}).get("reason", ""))
        predicates = {
            "explicitly_enabled": enabled,
            "use_sim_time": use_sim_time,
            "localization_source_is_tf": localization_is_tf,
            "map_frame_is_odom_frame": map_frame_is_odom_frame,
            "retry_is_stale_transform": retry_reason == "stale_transform",
            "odom_callback_advanced_during_recovery": (
                odom_callback_count_after > odom_callback_count_before
            ),
            "odom_message_available": odom_msg is not None,
            "odom_parent_frame_exact": (
                bool(odom_message_frame)
                and odom_message_frame == map_frame == odom_frame
            ),
            "odom_child_frame_exact": (
                bool(odom_child_frame)
                and odom_child_frame == base_frame
            ),
            "odom_fresh": bool(odom_freshness["fresh"]),
            "scan_fresh_after_recovery": bool(scan_freshness["fresh"]),
            "odom_stamp_available": odom_stamp_sec is not None,
            "retry_tf_stamp_available": retry_lookup.stamp_sec is not None,
            "odom_stamp_newer_than_tf_retry": (
                odom_stamp_sec is not None
                and retry_lookup.stamp_sec is not None
                and odom_stamp_sec > retry_lookup.stamp_sec
            ),
            "position_xy_finite": x_m is not None and y_m is not None,
            "quaternion_finite": quaternion_finite,
            "quaternion_norm_valid": quaternion_norm_valid,
            "yaw_finite": yaw_rad is not None,
        }
        rejection_reasons = [
            predicate
            for predicate, passed in predicates.items()
            if not passed
        ]
        fallback_episode = getattr(
            self,
            "_simulation_odom_fallback_episode",
            0,
        ) + (
            0
            if getattr(
                self,
                "_simulation_odom_fallback_active",
                False,
            )
            else 1
        )
        details: dict[str, object] = {
            "source": SIMULATION_ODOM_FALLBACK_SOURCE,
            "pose_source": SIMULATION_ODOM_FALLBACK_SOURCE,
            "attempted": True,
            "accepted": not rejection_reasons,
            "fail_closed": bool(rejection_reasons),
            "zero_published_before_fallback": True,
            "not_real_robot_migration_evidence": True,
            "fallback_episode": fallback_episode,
            "first_lookup_age_sec": (first_lookup.details or {}).get(
                "age_sec"
            ),
            "retry_lookup_age_sec": (retry_lookup.details or {}).get(
                "age_sec"
            ),
            "first_lookup_stamp_sec": first_lookup.stamp_sec,
            "retry_lookup_stamp_sec": retry_lookup.stamp_sec,
            "first_lookup": _pose_lookup_diagnostics(first_lookup),
            "retry_lookup": _pose_lookup_diagnostics(retry_lookup),
            "callback_drain": callback_drain,
            "runtime": {
                "use_sim_time": getattr(runtime, "use_sim_time", None),
                "localization_source": getattr(
                    runtime,
                    "localization_source",
                    None,
                ),
                "map_frame": map_frame,
                "odom_frame": odom_frame,
                "base_frame": base_frame,
            },
            "odom": {
                "frame_id": odom_message_frame,
                "child_frame_id": odom_child_frame,
                "header_stamp_sec": odom_stamp_sec,
                "receipt_monotonic_sec": odom_receipt,
                "callback_count_before_recovery": (
                    odom_callback_count_before
                ),
                "callback_count_after_recovery": (
                    odom_callback_count_after
                ),
                "freshness": odom_freshness,
                "pose": {
                    "x_m": x_m,
                    "y_m": y_m,
                    "yaw_rad": yaw_rad,
                    "quaternion": quaternion,
                    "quaternion_norm": quaternion_norm,
                    "quaternion_norm_tolerance": (
                        SIMULATION_ODOM_FALLBACK_QUATERNION_NORM_TOLERANCE
                    ),
                },
            },
            "scan": {
                "receipt_monotonic_sec": scan_receipt,
                "freshness": scan_freshness,
            },
            "predicates": predicates,
            "rejection_reasons": rejection_reasons,
        }

        if rejection_reasons:
            original_failure = _stale_tf_recovery_failure_details(
                dict(retry_lookup.details or {}),
                first_lookup=first_lookup,
                retry_lookup=retry_lookup,
                callback_drain=callback_drain,
            )
            original_failure["simulation_odom_fallback"] = details
            return PoseLookupResult(
                None,
                original_failure,
                retry_lookup.stamp_sec,
            )

        event_name = "simulation_odom_pose_fallback_started"
        if not getattr(
            self,
            "_simulation_odom_fallback_active",
            False,
        ):
            if not self._emit_route_update(
                RouteUpdate(
                    kind=RouteUpdateKind.UNCHANGED,
                    event_name=event_name,
                    event_fields=details,
                )
            ):
                return self._semantic_event_failure_lookup(
                    event_name=event_name,
                    stamp_sec=odom_stamp_sec,
                )
            self._simulation_odom_fallback_episode = fallback_episode
            self._simulation_odom_fallback_active = True
        return PoseLookupResult(
            Pose2D(float(x_m), float(y_m), float(yaw_rad)),
            details,
            odom_stamp_sec,
        )

    def _current_pose_lookup_with_stale_recovery(self) -> PoseLookupResult:
        first_lookup = self._current_pose_lookup()
        first_details = dict(first_lookup.details or {})
        if first_lookup.pose is not None:
            return self._primary_tf_result_with_restore_event(
                first_lookup,
                recovered_after_retry=False,
            )
        if first_details.get("reason") != "stale_transform":
            return first_lookup

        # Override any preceding nonzero Twist before servicing queued
        # scan/odom/TF work.  No motion command is published during recovery.
        odom_callback_count_before = getattr(
            self,
            "latest_odom_callback_count",
            0,
        )
        self.publish_zero()
        drain_details = self._drain_runtime_callbacks(
            max_callbacks=STALE_TF_RECOVERY_MAX_CALLBACKS,
            max_duration_sec=STALE_TF_RECOVERY_MAX_DURATION_SEC,
            spin_timeout_sec=STALE_TF_RECOVERY_SPIN_TIMEOUT_SEC,
        )
        odom_callback_count_after = getattr(
            self,
            "latest_odom_callback_count",
            0,
        )
        odom_msg = getattr(self, "latest_odom", None)
        odom_receipt = getattr(self, "latest_odom_receipt", None)
        scan_msg = getattr(self, "latest_scan", None)
        scan_receipt = getattr(self, "latest_scan_receipt", None)
        retry_lookup = self._current_pose_lookup()
        retry_details = dict(retry_lookup.details or {})
        if retry_lookup.pose is None:
            if retry_details.get("reason") == "stale_transform":
                return self._simulation_odom_fallback_after_stale_retry(
                    first_lookup=first_lookup,
                    retry_lookup=retry_lookup,
                    callback_drain=drain_details,
                    odom_callback_count_before=(
                        odom_callback_count_before
                    ),
                    odom_callback_count_after=(
                        odom_callback_count_after
                    ),
                    odom_msg=odom_msg,
                    odom_receipt=odom_receipt,
                    scan_msg=scan_msg,
                    scan_receipt=scan_receipt,
                )
            # Preserve the retry failure as the top-level stop diagnostic.
            # Persistent stale_transform therefore stops exactly as before,
            # with its retry age in the legacy age_sec field.
            return PoseLookupResult(
                None,
                _stale_tf_recovery_failure_details(
                    retry_details,
                    first_lookup=first_lookup,
                    retry_lookup=retry_lookup,
                    callback_drain=drain_details,
                ),
                retry_lookup.stamp_sec,
            )
        if (
            first_lookup.stamp_sec is None
            or retry_lookup.stamp_sec is None
            or retry_lookup.stamp_sec <= first_lookup.stamp_sec
        ):
            nonadvancing_details = tf_lookup_failure_details(
                reason="nonadvancing_transform",
                target_frame=self.runtime_config.map_frame,
                source_frame=self.runtime_config.base_frame,
                max_age_sec=self.follower_config.max_tf_age_sec,
            )
            return PoseLookupResult(
                None,
                _stale_tf_recovery_failure_details(
                    nonadvancing_details,
                    first_lookup=first_lookup,
                    retry_lookup=retry_lookup,
                    callback_drain=drain_details,
                ),
                retry_lookup.stamp_sec,
            )
        freshness_failure = self._post_stale_tf_recovery_freshness_failure()
        if freshness_failure:
            freshness_details = {
                "stop_reason": freshness_failure,
                "source": "stale_tf_recovery",
                "reason": "post_recovery_sensor_freshness_failure",
                "sensor_failure": dict(self.latest_stop_details or {}),
            }
            return PoseLookupResult(
                None,
                _stale_tf_recovery_failure_details(
                    freshness_details,
                    first_lookup=first_lookup,
                    retry_lookup=retry_lookup,
                    callback_drain=drain_details,
                ),
                retry_lookup.stamp_sec,
            )
        return self._primary_tf_result_with_restore_event(
            retry_lookup,
            recovered_after_retry=True,
        )

    def _current_pose(self) -> Pose2D | None:
        return self._current_pose_lookup().pose


def run_simple_waypoint_follower(
    runtime_config: ResolvedRuntimeConfig,
    waypoints: Sequence[Pose2D],
    follower_config: FollowerConfig,
    waypoint_provider: Callable[[Pose2D], RouteUpdate | None] | None = None,
    route_update_callback: Callable[[RouteUpdate], None] | None = None,
) -> FollowerResult:
    _require_ros()
    rclpy.init(args=None)
    node = None
    listener_node = None
    tf_listener = None
    follower_executor = None
    tf_executor = None
    follower_executor_thread = None
    tf_executor_thread = None
    node_added_to_follower_executor = False
    listener_node_added_to_tf_executor = False
    try:
        listener_node, tf_buffer, tf_listener = _create_dedicated_tf_listener(
            runtime_config
        )
        node = SimpleWaypointFollowerNode(
            runtime_config,
            waypoints,
            follower_config,
            waypoint_provider,
            route_update_callback,
            tf_buffer=tf_buffer,
        )
        follower_executor = MultiThreadedExecutor(
            num_threads=FOLLOWER_EXECUTOR_NUM_THREADS,
        )
        tf_executor = SingleThreadedExecutor()
        node_added_to_follower_executor = follower_executor.add_node(node)
        if not node_added_to_follower_executor:
            raise RuntimeError(
                "failed to add waypoint follower to background executor"
            )
        listener_node_added_to_tf_executor = tf_executor.add_node(
            listener_node
        )
        if not listener_node_added_to_tf_executor:
            raise RuntimeError(
                "failed to add TF listener to isolated background executor"
            )
        node.enable_background_callback_service()
        tf_executor_thread = threading.Thread(
            target=tf_executor.spin,
            name="aufgabe04-follower-tf-listener",
            daemon=False,
        )
        follower_executor_thread = threading.Thread(
            target=follower_executor.spin,
            name="aufgabe04-follower-callbacks",
            daemon=False,
        )
        # Give TF its independent callback service before the follower starts
        # its sensor wait/control loop.
        tf_executor_thread.start()
        follower_executor_thread.start()
        return node.run()
    finally:
        if follower_executor is not None:
            follower_executor.shutdown()
        if tf_executor is not None:
            tf_executor.shutdown()
        if (
            follower_executor_thread is not None
            and follower_executor_thread.ident is not None
        ):
            follower_executor_thread.join()
        if (
            tf_executor_thread is not None
            and tf_executor_thread.ident is not None
        ):
            tf_executor_thread.join()
        if (
            follower_executor is not None
            and node is not None
            and node_added_to_follower_executor
        ):
            follower_executor.remove_node(node)
        if (
            tf_executor is not None
            and listener_node is not None
            and listener_node_added_to_tf_executor
        ):
            tf_executor.remove_node(listener_node)
        if node is not None:
            node.disable_background_callback_service()
            node.destroy_node()
        if tf_listener is not None:
            tf_listener.unregister()
        if listener_node is not None:
            listener_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
