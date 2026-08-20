"""Certified startup target selection for sealed route execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from scripts.aufgabe04.navigation.execution_route_certificate import (
    ExecutionRouteCheck,
    check_execution_route_tube,
)
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.waypoint_follower.config import FollowerConfig


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
