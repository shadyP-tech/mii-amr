"""Atomic validation and adoption of dynamic replacement routes."""

from __future__ import annotations

import math
import time
from dataclasses import replace

from scripts.aufgabe04.navigation.execution.dynamic_route_handoff import (
    RouteUpdate,
    RouteUpdateKind,
)
from scripts.aufgabe04.navigation.control.driving_behavior import PHYSICAL_ROUTE_KINDS
from scripts.aufgabe04.navigation.execution.execution_route_certificate import (
    check_execution_route_tube,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.control.waypoint_controller import (
    reverse_staging_is_preferred,
)
from scripts.aufgabe04.navigation.waypoint_follower.route_admission import (
    dynamic_join_envelope_failure,
)
from scripts.aufgabe04.navigation.waypoint_follower.directives import (
    RouteRefreshAction,
)
from scripts.aufgabe04.navigation.waypoint_follower.route_phases import (
    dynamic_route_kind_transition_failure,
)


class DynamicRouteRuntimeMixin:
    """Dynamic-route behavior mixed into the sole follower node."""

    def _refresh_dynamic_route(self, pose: Pose2D) -> RouteRefreshAction:
        queued_update = getattr(self, "queued_route_update", None)
        if queued_update is None and self.waypoint_provider is None:
            return RouteRefreshAction.CONTINUE
        now = time.monotonic()
        initial_refresh = self.initial_route_refresh_pending
        if (
            queued_update is None
            and not initial_refresh
            and self.follower_config.dynamic_route_refresh_sec <= 0.0
        ):
            return RouteRefreshAction.CONTINUE
        if (
            queued_update is None
            and not initial_refresh
            and now - self.last_route_refresh_at
            < self.follower_config.dynamic_route_refresh_sec
        ):
            return RouteRefreshAction.CONTINUE
        self.initial_route_refresh_pending = False
        self.last_route_refresh_at = now
        if queued_update is not None:
            update = queued_update
            self.queued_route_update = None
        else:
            try:
                assert self.waypoint_provider is not None
                update = self.waypoint_provider(pose)
            except Exception as exc:
                self.latest_stop_details = {
                    "reason": f"dynamic route provider failed: {exc}",
                    "fault_code": "route_provider_exception",
                    "fail_closed": True,
                }
                return RouteRefreshAction.STOPPED
        if update is None:
            return RouteRefreshAction.CONTINUE
        if update.kind is RouteUpdateKind.UNCHANGED:
            return RouteRefreshAction.CONTINUE
        if update.kind is RouteUpdateKind.REJECT:
            self.publish_zero()
            if not self._emit_route_update(update):
                return RouteRefreshAction.STOPPED
            self.latest_stop_details = {
                **dict(update.event_fields),
                "reason": update.reason or "dynamic route update rejected",
                "fail_closed": True,
            }
            return RouteRefreshAction.STOPPED
        if update.kind is RouteUpdateKind.STOP:
            # Zero first: semantic logging is synchronous and must never leave
            # the previous nonzero Twist active if it blocks or raises.
            self.publish_zero()
            if not self._emit_route_update(update):
                return RouteRefreshAction.STOPPED
            self.latest_stop_details = {
                **dict(update.event_fields),
                "reason": update.reason or "dynamic route withdrawn",
            }
            return RouteRefreshAction.STOPPED
        if update.kind is RouteUpdateKind.COMPLETE:
            # A committed arrival estimate is the successful terminal event
            # for a survey leg.  Stop before logging so a slow callback can
            # never leave a previous non-zero Twist active.
            self.publish_zero()
            if not self._emit_route_update(update):
                return RouteRefreshAction.STOPPED
            self.latest_stop_details = {
                **dict(update.event_fields),
                "reason": update.reason or "survey completed",
                "fail_closed": False,
            }
            return RouteRefreshAction.COMPLETED
        replacement = tuple(update.waypoints)
        if update.kind is not RouteUpdateKind.ADOPT or len(replacement) < 2:
            self.publish_zero()
            self.latest_stop_details = {
                "reason": "dynamic route adoption contained fewer than two waypoints",
                "fault_code": "invalid_route_update",
                "fail_closed": True,
            }
            return RouteRefreshAction.STOPPED
        if update.target_index is None or not 0 <= update.target_index < len(replacement):
            self.publish_zero()
            self.latest_stop_details = {
                "reason": "dynamic route adoption contained an invalid target index",
                "fault_code": "invalid_route_update",
                "fail_closed": True,
            }
            return RouteRefreshAction.STOPPED
        try:
            join_limit = float(update.event_fields["effective_join_limit_m"])
        except (KeyError, TypeError, ValueError) as exc:
            self.publish_zero()
            self.latest_stop_details = {
                "reason": f"dynamic route adoption lacks a valid join envelope: {exc}",
                "fault_code": "invalid_route_update",
                "fail_closed": True,
            }
            return RouteRefreshAction.STOPPED
        if not math.isfinite(join_limit) or join_limit <= 0.0:
            self.publish_zero()
            self.latest_stop_details = {
                "reason": "dynamic route adoption join envelope is not positive and finite",
                "fault_code": "invalid_route_update",
                "fail_closed": True,
            }
            return RouteRefreshAction.STOPPED
        join_failure = dynamic_join_envelope_failure(
            pose,
            replacement[0],
            join_limit,
        )
        if join_failure is not None:
            self.publish_zero()
            self.latest_stop_details = {
                **join_failure,
                "reason": "fresh pose is outside the replacement-route join envelope",
                "certificate_reason": join_failure["reason"],
                "source": "dynamic_route_admission",
                "fail_closed": True,
            }
            return RouteRefreshAction.STOPPED
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
            return RouteRefreshAction.STOPPED
        if (
            next_route_kind in PHYSICAL_ROUTE_KINDS
            and join_limit
            <= self.follower_config.certified_route_tube_radius_m + 1.0e-9
        ):
            start_check = check_execution_route_tube(
                pose,
                replacement,
                target_index=0,
                pursuit_index=0,
                tracking_tube_radius_m=(
                    self.follower_config.certified_route_tube_radius_m
                ),
                chord_sample_spacing_m=(
                    self.follower_config.certified_route_chord_sample_spacing_m
                ),
            )
            if not start_check.ok:
                self.publish_zero()
                self.latest_stop_details = {
                    **start_check.to_log_dict(),
                    "reason": (
                        "fresh pose failed the replacement-route start-tube "
                        "certificate"
                    ),
                    "certificate_reason": start_check.reason,
                    "source": "dynamic_route_admission",
                    "fail_closed": True,
                }
                return RouteRefreshAction.STOPPED
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
            return RouteRefreshAction.STOPPED
        next_egress_lock_index = None
        next_egress_reverse = False
        next_reverse_until_index = None
        next_forward_alignment_index = None
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
                return RouteRefreshAction.STOPPED
            next_egress_lock_index = raw_lock_index
            raw_egress_motion = update.event_fields.get(
                "start_egress_motion",
                "forward",
            )
            if raw_egress_motion not in {"forward", "reverse"}:
                self.publish_zero()
                self.latest_stop_details = {
                    "reason": "dynamic route start-egress motion is invalid",
                    "fault_code": "invalid_route_update",
                    "fail_closed": True,
                }
                return RouteRefreshAction.STOPPED
            next_egress_reverse = raw_egress_motion == "reverse"
            if next_egress_reverse:
                raw_reverse_until_index = update.event_fields.get(
                    "start_egress_reverse_until_waypoint_index"
                )
                raw_forward_alignment_index = update.event_fields.get(
                    "start_egress_forward_alignment_waypoint_index"
                )
                if (
                    not isinstance(raw_reverse_until_index, int)
                    or isinstance(raw_reverse_until_index, bool)
                    or raw_reverse_until_index < raw_lock_index + 1
                    or raw_reverse_until_index >= len(replacement) - 1
                    or not isinstance(raw_forward_alignment_index, int)
                    or isinstance(raw_forward_alignment_index, bool)
                    or raw_forward_alignment_index
                    != raw_reverse_until_index + 1
                    or raw_forward_alignment_index >= len(replacement)
                ):
                    self.publish_zero()
                    self.latest_stop_details = {
                        "reason": (
                            "dynamic route reverse-egress handoff "
                            "certificate is malformed"
                        ),
                        "fault_code": "invalid_route_update",
                        "fail_closed": True,
                    }
                    return RouteRefreshAction.STOPPED
                next_reverse_until_index = raw_reverse_until_index
                next_forward_alignment_index = raw_forward_alignment_index
        previous_route_kind = self.current_route_kind
        self.publish_zero()
        self._clear_intermediate_terminal_heading_latch(
            material_route_revision=True,
        )
        self.certified_corner_latch = None
        self._last_certified_corner_phase = None
        raw_route_revision = update.route_revision
        if (
            isinstance(raw_route_revision, int)
            and not isinstance(raw_route_revision, bool)
            and raw_route_revision >= 0
        ):
            self.controller_route_revision = raw_route_revision
        else:
            self.controller_route_revision = (
                getattr(self, "controller_route_revision", 0) + 1
            )
        self.waypoints = replacement
        # Every route replacement owns a fresh lock decision. Ordinary routes
        # explicitly clear any lock retained from the previous revision.
        self.start_egress_lock_index = next_egress_lock_index
        self.start_egress_reverse = next_egress_reverse
        self.start_egress_reverse_until_index = next_reverse_until_index
        self.start_egress_forward_alignment_index = (
            next_forward_alignment_index
        )
        self.current_route_kind = next_route_kind
        self.reverse_staging = (
            next_route_kind in PHYSICAL_ROUTE_KINDS
            and next_route_kind != "stand_discovery_corridor"
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
            return RouteRefreshAction.STOPPED
        return RouteRefreshAction.ADOPTED

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
