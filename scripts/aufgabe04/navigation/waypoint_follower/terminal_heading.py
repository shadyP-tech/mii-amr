"""Pure terminal-heading latch logic for intermediate survey routes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from scripts.aufgabe04.navigation.control.driving_behavior import INTERMEDIATE_ROUTE_KINDS
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.approach.viewpoint_sampling_contract import (
    DEFAULT_VIEWPOINT_SAMPLING_TARGET_DISTANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M,
    INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M,
    ViewpointSamplingHoldConfig,
    viewpoint_sampling_hold_metrics,
)
from scripts.aufgabe04.navigation.control.waypoint_controller import (
    ControllerConfig,
    ControllerStep,
    VelocityCommand,
    compute_waypoint_command,
    normalize_angle,
)


INTERMEDIATE_TERMINAL_HEADING_HOLD_EXCEEDED = (
    "intermediate_terminal_heading_hold_tolerance_exceeded"
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
