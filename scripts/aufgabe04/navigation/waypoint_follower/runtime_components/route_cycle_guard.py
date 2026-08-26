"""Dynamic-route and sampling-deadline admission for one control cycle."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Mapping

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.waypoint_follower.directives import (
    RouteRefreshAction,
    StringDirective,
)
from scripts.aufgabe04.navigation.waypoint_follower.route_phases import (
    viewpoint_sampling_deadline_decision,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.control_results import (
    viewpoint_sampling_timeout_stop_details,
)


class RouteCycleGuardAction(StringDirective):
    """Next loop action after route refresh and deadline admission."""

    PROCEED = "proceed"
    RETRY = "retry"
    STOP = "stop"
    COMPLETE = "complete"


@dataclass(frozen=True)
class RouteCycleGuardDecision:
    """Typed route-cycle outcome for the top-level orchestration loop."""

    action: RouteCycleGuardAction
    stop_reason: str = ""
    stop_details: Mapping[str, object] | None = None
    evaluated_at: float | None = None


class RouteCycleGuardRuntimeMixin:
    """Apply route refresh, sampling deadlines, and zero-cycle handoffs."""

    def _route_cycle_guard_decision(
        self,
        pose: Pose2D,
        loop_period_sec: float,
    ) -> RouteCycleGuardDecision:
        route_refresh = self._refresh_dynamic_route(pose)
        if route_refresh == RouteRefreshAction.STOPPED:
            self.publish_repeated_zero()
            return RouteCycleGuardDecision(
                RouteCycleGuardAction.STOP,
                stop_reason=str(
                    (self.latest_stop_details or {}).get(
                        "reason",
                        "dynamic route withdrawn",
                    )
                ),
                stop_details=self.latest_stop_details,
            )
        if route_refresh == RouteRefreshAction.COMPLETED:
            self.publish_repeated_zero()
            return RouteCycleGuardDecision(
                RouteCycleGuardAction.COMPLETE,
                stop_details=self.latest_stop_details,
            )

        sampling_now = time.monotonic()
        sampling_deadline = viewpoint_sampling_deadline_decision(
            route_kind=self.current_route_kind,
            phase_started_at=self.viewpoint_sampling_started_at,
            target_started_at=self.viewpoint_sampling_target_started_at,
            now_monotonic=sampling_now,
            phase_timeout_sec=(
                self.follower_config.viewpoint_sampling_timeout_sec
            ),
            target_timeout_sec=(
                self.follower_config.viewpoint_sampling_target_timeout_sec
            ),
        )
        if sampling_deadline.failure:
            stop_details = viewpoint_sampling_timeout_stop_details(
                reason=sampling_deadline.failure,
                route_kind=self.current_route_kind,
                phase_elapsed_sec=sampling_deadline.phase_elapsed_sec,
                target_elapsed_sec=sampling_deadline.target_elapsed_sec,
                phase_timeout_sec=(
                    self.follower_config.viewpoint_sampling_timeout_sec
                ),
                target_timeout_sec=(
                    self.follower_config
                    .viewpoint_sampling_target_timeout_sec
                ),
            )
            self.latest_stop_details = stop_details
            self.publish_repeated_zero()
            return RouteCycleGuardDecision(
                RouteCycleGuardAction.STOP,
                stop_reason=sampling_deadline.failure.replace("_", " "),
                stop_details=stop_details,
                evaluated_at=sampling_now,
            )

        if route_refresh == RouteRefreshAction.ADOPTED:
            # A verified handoff consumes one complete zero-command period;
            # motion can resume only after the next full guard chain.
            self.publish_zero()
            self._hold_zero_control_period(loop_period_sec)
            return RouteCycleGuardDecision(RouteCycleGuardAction.RETRY)

        return RouteCycleGuardDecision(RouteCycleGuardAction.PROCEED)
