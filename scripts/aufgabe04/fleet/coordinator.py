"""Central, ROS-free conflict-zone arbitration and permit state."""

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from .collision_rules import status_safety_issue
from .coordination_policy import peer_is_on_right
from .models import PriorityDecision
from .reservations import (
    ReservationPermit,
    ReservationTable,
    acquire_permit,
    active_permit,
)
from .robot_status import RobotStatus
from .station_locks import (
    StationLockTable,
    acquire_station,
    active_station_lease,
    release_station,
    renew_station,
)


@dataclass(frozen=True)
class ConflictZoneRequest:
    zone_id: str
    status: RobotStatus
    requested_at_sec: float


@dataclass(frozen=True)
class RobotPermitDecision:
    robot_id: str
    decision: PriorityDecision
    reason: str
    fencing_token: int = 0


@dataclass(frozen=True)
class CoordinationOutcome:
    state: "FleetCoordinatorState"
    zone_id: str
    winner_robot_id: str
    decisions: Tuple[RobotPermitDecision, ...]
    permit: Optional[ReservationPermit] = None
    tie_break_applied: bool = False


@dataclass(frozen=True)
class StationPermitOutcome:
    state: "FleetCoordinatorState"
    station_id: str
    robot_id: str
    decision: PriorityDecision
    reason: str
    fencing_token: int = 0


@dataclass(frozen=True)
class FleetCoordinatorState:
    reservations: ReservationTable = field(default_factory=ReservationTable.empty)
    station_locks: StationLockTable = field(default_factory=StationLockTable.empty)

    @classmethod
    def empty(cls) -> "FleetCoordinatorState":
        return cls()


def request_station_permit(
    state: FleetCoordinatorState,
    station_id: str,
    robot_id: str,
    *,
    now_sec: float,
    lease_ttl_sec: float,
) -> StationPermitOutcome:
    """Acquire an exclusive station permit without silently renewing it."""

    if not math.isfinite(lease_ttl_sec) or lease_ttl_sec <= 0.0:
        raise ValueError("lease_ttl_sec must be finite and positive")
    current = active_station_lease(
        state.station_locks, station_id, now_sec=now_sec
    )
    if current is not None:
        if current.robot_id == robot_id:
            return StationPermitOutcome(
                state,
                station_id,
                robot_id,
                PriorityDecision.WAIT,
                "station permit is already active; renew with its fencing token",
            )
        return StationPermitOutcome(
            state,
            station_id,
            robot_id,
            PriorityDecision.WAIT,
            f"station is occupied by {current.robot_id}",
        )
    locks = acquire_station(
        state.station_locks,
        station_id,
        robot_id,
        now_sec=now_sec,
        expires_at_sec=now_sec + lease_ttl_sec,
    )
    lease = locks.leases[station_id]
    next_state = FleetCoordinatorState(
        reservations=state.reservations,
        station_locks=locks,
    )
    return StationPermitOutcome(
        next_state,
        station_id,
        robot_id,
        PriorityDecision.PROCEED,
        "fenced station permit granted",
        lease.fencing_token,
    )


def renew_station_permit(
    state: FleetCoordinatorState,
    station_id: str,
    robot_id: str,
    fencing_token: int,
    *,
    now_sec: float,
    lease_ttl_sec: float,
) -> FleetCoordinatorState:
    if not math.isfinite(lease_ttl_sec) or lease_ttl_sec <= 0.0:
        raise ValueError("lease_ttl_sec must be finite and positive")
    locks = renew_station(
        state.station_locks,
        station_id,
        robot_id,
        fencing_token,
        now_sec=now_sec,
        expires_at_sec=now_sec + lease_ttl_sec,
    )
    return FleetCoordinatorState(
        reservations=state.reservations,
        station_locks=locks,
    )


def release_station_permit(
    state: FleetCoordinatorState,
    station_id: str,
    robot_id: str,
    fencing_token: int,
    *,
    now_sec: float,
) -> FleetCoordinatorState:
    locks = release_station(
        state.station_locks,
        station_id,
        robot_id,
        fencing_token=fencing_token,
        now_sec=now_sec,
    )
    return FleetCoordinatorState(
        reservations=state.reservations,
        station_locks=locks,
    )


def _strict_request_issue(
    request: ConflictZoneRequest,
    *,
    zone_id: str,
    now_sec: float,
    max_status_age_sec: float,
) -> str:
    if request.zone_id != zone_id:
        return "request names a different conflict zone"
    if not math.isfinite(request.requested_at_sec) or request.requested_at_sec < 0.0:
        return "invalid request timestamp"
    if request.requested_at_sec > now_sec:
        return "request timestamp is in the future"
    issue = status_safety_issue(
        request.status,
        now_sec=now_sec,
        max_status_age_sec=max_status_age_sec,
    )
    if issue is not None:
        return issue
    if request.status.yaw_rad is None or not math.isfinite(request.status.yaw_rad):
        return "missing or invalid peer yaw"
    return ""


def _winner(
    requests: Sequence[ConflictZoneRequest], *, lateral_epsilon_m: float
) -> Tuple[ConflictZoneRequest, bool]:
    if len(requests) == 1:
        return requests[0], False
    no_right_neighbor = []
    for request in requests:
        has_right_neighbor = False
        assert request.status.x_m is not None
        assert request.status.y_m is not None
        assert request.status.yaw_rad is not None
        for peer in requests:
            if peer.status.robot_id == request.status.robot_id:
                continue
            assert peer.status.x_m is not None and peer.status.y_m is not None
            if peer_is_on_right(
                robot_x_m=request.status.x_m,
                robot_y_m=request.status.y_m,
                robot_yaw_rad=request.status.yaw_rad,
                peer_x_m=peer.status.x_m,
                peer_y_m=peer.status.y_m,
                lateral_epsilon_m=lateral_epsilon_m,
            ):
                has_right_neighbor = True
                break
        if not has_right_neighbor:
            no_right_neighbor.append(request)
    candidates = no_right_neighbor or list(requests)
    chosen = min(
        candidates,
        key=lambda request: (request.requested_at_sec, request.status.robot_id),
    )
    return chosen, len(candidates) != 1


def coordinate_conflict_zone(
    state: FleetCoordinatorState,
    zone_id: str,
    requests: Sequence[ConflictZoneRequest],
    *,
    now_sec: float,
    permit_ttl_sec: float,
    max_status_age_sec: float,
    lateral_epsilon_m: float = 1e-3,
) -> CoordinationOutcome:
    """Arbitrate contenders and issue at most one fenced zone permit."""

    zone_id = zone_id.strip()
    if not zone_id:
        raise ValueError("zone_id must not be empty")
    if not requests:
        raise ValueError("at least one conflict-zone request is required")
    robot_ids = [request.status.robot_id for request in requests]
    if len(set(robot_ids)) != len(robot_ids):
        raise ValueError("conflict-zone requests contain duplicate robot IDs")

    issues = {
        request.status.robot_id: _strict_request_issue(
            request,
            zone_id=zone_id,
            now_sec=now_sec,
            max_status_age_sec=max_status_age_sec,
        )
        for request in requests
    }
    if any(issues.values()):
        decisions = tuple(
            RobotPermitDecision(
                request.status.robot_id,
                PriorityDecision.WAIT,
                issues[request.status.robot_id] or "peer state is unsafe",
            )
            for request in requests
        )
        return CoordinationOutcome(state, zone_id, "", decisions)

    existing = active_permit(state.reservations, zone_id, now_sec=now_sec)
    if existing is not None:
        owner_present = existing.robot_id in robot_ids
        decisions = tuple(
            RobotPermitDecision(
                request.status.robot_id,
                (
                    PriorityDecision.PROCEED
                    if owner_present and request.status.robot_id == existing.robot_id
                    else PriorityDecision.WAIT
                ),
                (
                    "active fenced permit"
                    if owner_present and request.status.robot_id == existing.robot_id
                    else "conflict zone already reserved"
                ),
                existing.fencing_token
                if request.status.robot_id == existing.robot_id
                else 0,
            )
            for request in requests
        )
        return CoordinationOutcome(
            state,
            zone_id,
            existing.robot_id if owner_present else "",
            decisions,
            permit=existing,
        )

    winner, tie_break_applied = _winner(
        requests, lateral_epsilon_m=lateral_epsilon_m
    )
    reservations = acquire_permit(
        state.reservations,
        zone_id,
        winner.status.robot_id,
        now_sec=now_sec,
        ttl_sec=permit_ttl_sec,
    )
    permit = active_permit(reservations, zone_id, now_sec=now_sec)
    assert permit is not None
    next_state = FleetCoordinatorState(
        reservations=reservations,
        station_locks=state.station_locks,
    )
    decisions = tuple(
        RobotPermitDecision(
            request.status.robot_id,
            (
                PriorityDecision.PROCEED
                if request.status.robot_id == winner.status.robot_id
                else PriorityDecision.YIELD
            ),
            (
                "right-before-left permit granted"
                if request.status.robot_id == winner.status.robot_id
                else "yield to right-before-left winner"
            ),
            permit.fencing_token
            if request.status.robot_id == winner.status.robot_id
            else 0,
        )
        for request in requests
    )
    return CoordinationOutcome(
        next_state,
        zone_id,
        winner.status.robot_id,
        decisions,
        permit=permit,
        tie_break_applied=tie_break_applied,
    )
