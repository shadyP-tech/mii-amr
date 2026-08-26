"""Route-phase controller-step resolution for the follower runtime.

This component may update route-phase state, but it deliberately performs no
ROS publication, trace I/O, callback spinning, sleeping, or loop control.
"""

from __future__ import annotations

from collections.abc import Callable
import time

from scripts.aufgabe04.navigation.approach.viewpoint_sampling_contract import (
    INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M,
)
from scripts.aufgabe04.navigation.control.driving_behavior import (
    controller_config_for_route_kind,
)
from scripts.aufgabe04.navigation.control.waypoint_controller import (
    compute_join_anchor_command,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.waypoint_follower.directives import (
    StartupJoinAction,
)
from scripts.aufgabe04.navigation.waypoint_follower.route_admission import (
    certified_startup_join_action,
)
from scripts.aufgabe04.navigation.waypoint_follower.route_phases import (
    ControlStepAction,
    ControlStepResolution,
    ControlStepStopKind,
    RouteCommandPhase,
    route_command_phase,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.control_results import (
    intermediate_terminal_heading_stop_details,
)
from scripts.aufgabe04.navigation.waypoint_follower.terminal_heading import (
    compute_intermediate_terminal_heading_command,
    intermediate_terminal_heading_entry_tolerance_m,
    intermediate_terminal_heading_hold_diagnostics,
)


StartupJoinDecision = Callable[
    [Pose2D, Pose2D, float | None, float],
    tuple[StartupJoinAction, dict[str, object] | None],
]
MonotonicClock = Callable[[], float]


class RouteStepResolutionRuntimeMixin:
    """Resolve route-phase state into a typed controller-step decision."""

    def _resolve_control_step(
        self,
        pose: Pose2D,
        *,
        startup_join_action_fn: StartupJoinDecision = (
            certified_startup_join_action
        ),
        monotonic_fn: MonotonicClock = time.monotonic,
    ) -> ControlStepResolution:
        """Resolve one controller step without publishing or loop control."""

        command_phase = route_command_phase(
            dynamic_join_pending=self.dynamic_join_pending,
            start_egress_lock_index=self.start_egress_lock_index,
            start_egress_forward_alignment_index=(
                self.start_egress_forward_alignment_index
            ),
        )
        if command_phase == RouteCommandPhase.DYNAMIC_JOIN:
            join_action, join_failure = startup_join_action_fn(
                pose,
                self.waypoints[0],
                self.dynamic_join_limit_m,
                self.follower_config.dynamic_join_tolerance_m,
            )
            if join_action == StartupJoinAction.STOP:
                assert join_failure is not None
                return ControlStepResolution(
                    ControlStepAction.STOP,
                    command_phase,
                    stop_kind=ControlStepStopKind.DYNAMIC_JOIN,
                    stop_reason=str(join_failure["reason"]),
                    stop_details=join_failure,
                )
            if join_action == StartupJoinAction.ZERO:
                self.dynamic_join_pending = False
                self.dynamic_join_limit_m = None
                if self.target_index != 0:
                    self._clear_intermediate_terminal_heading_latch(
                        target_changed=True,
                    )
                self.target_index = 0
                self.target_started_at = monotonic_fn()
                self._reset_progress_watchdog(monotonic_fn())
                return ControlStepResolution(
                    ControlStepAction.ZERO_HOLD,
                    command_phase,
                )

        route_controller_config = controller_config_for_route_kind(
            self.follower_config.controller,
            self.current_route_kind,
            reverse_staging=self.reverse_staging,
            viewpoint_sampling_goal_tolerance_m=(
                self.follower_config.viewpoint_sampling_goal_tolerance_m
            ),
            viewpoint_sampling_heading_tolerance_rad=(
                self.follower_config
                .viewpoint_sampling_heading_tolerance_rad
            ),
            physical_goal_tolerance_m=(
                self.follower_config.physical_goal_tolerance_m
            ),
            physical_waypoint_tolerance_m=(
                self.follower_config.physical_waypoint_tolerance_m
            ),
        )
        if command_phase == RouteCommandPhase.DYNAMIC_JOIN:
            # During handoff, pursue only the collision-certified route start.
            # Normal lookahead would form an unchecked chord to waypoint 1.
            step = compute_join_anchor_command(
                pose,
                self.waypoints[0],
                route_controller_config,
                join_tolerance_m=(
                    self.follower_config.dynamic_join_tolerance_m
                ),
            )
            return ControlStepResolution(
                ControlStepAction.PROCEED,
                command_phase,
                step=step,
            )
        if command_phase == RouteCommandPhase.START_EGRESS:
            step = self._start_egress_command(
                pose,
                route_controller_config,
            )
            return ControlStepResolution(
                (
                    ControlStepAction.ZERO_HOLD
                    if step is None
                    else ControlStepAction.PROCEED
                ),
                command_phase,
                step=step,
            )
        if command_phase == RouteCommandPhase.REVERSE_EGRESS_ALIGNMENT:
            step = self._reverse_egress_forward_alignment_command(
                pose,
                route_controller_config,
            )
            return ControlStepResolution(
                ControlStepAction.PROCEED,
                command_phase,
                step=step,
            )

        corner_decision = self._certified_corner_decision(
            pose,
            route_controller_config,
        )
        if corner_decision.failure:
            failed_step = corner_decision.step
            assert failed_step is not None
            return ControlStepResolution(
                ControlStepAction.STOP,
                command_phase,
                step=failed_step,
                corner_step=failed_step,
                stop_kind=ControlStepStopKind.CERTIFIED_CORNER,
                stop_reason=corner_decision.failure,
            )
        if corner_decision.step is not None:
            return ControlStepResolution(
                ControlStepAction.PROCEED,
                command_phase,
                step=corner_decision.step,
                corner_step=corner_decision.step,
            )

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
        if not terminal_heading_decision.failure:
            return ControlStepResolution(
                ControlStepAction.PROCEED,
                command_phase,
                step=step,
            )

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
        stop_details = intermediate_terminal_heading_stop_details(
            reason=terminal_heading_decision.failure,
            route_kind=self.current_route_kind,
            target_index=step.target_index,
            distance_to_target_m=step.distance_to_target_m,
            entry_tolerance_m=(
                intermediate_terminal_heading_entry_tolerance_m(
                    route_controller_config
                )
            ),
            hold_tolerance_m=(
                self.follower_config
                .viewpoint_sampling_terminal_heading_hold_tolerance_m
            ),
            distance_comparison_epsilon_m=(
                INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M
            ),
            hold_diagnostics=hold_diagnostics,
        )
        return ControlStepResolution(
            ControlStepAction.STOP,
            command_phase,
            step=step,
            stop_kind=ControlStepStopKind.INTERMEDIATE_TERMINAL_HEADING,
            stop_reason=terminal_heading_decision.failure.replace("_", " "),
            stop_details=stop_details,
        )
