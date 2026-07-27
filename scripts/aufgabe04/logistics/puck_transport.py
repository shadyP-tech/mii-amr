"""Pure puck pickup, transport, dropoff, and loss acknowledgements."""

import math
from dataclasses import dataclass, replace
from typing import Optional

from .carrier_profile import CarrierProfile, MotionEnvelope, build_motion_envelope
from .models import PuckState
from .puck_custody import CustodyState, PuckCustody


@dataclass(frozen=True)
class PuckTransportAssumptions:
    passive_carrier: bool = True
    requires_operator_load: bool = True
    requires_operator_unload: bool = True


@dataclass(frozen=True)
class PuckTransportState:
    """Robot-local transport state backed by a fenced custody claim."""

    robot_id: str
    puck_id: str = ""
    puck_state: PuckState = PuckState.NOT_HELD
    custody_fencing_token: int = 0
    payload_mass_kg: float = 0.0
    retention_confirmed: bool = False
    updated_at_sec: Optional[float] = None
    failure_reason: str = ""


def _checked_time(state: PuckTransportState, now_sec: float) -> float:
    if not math.isfinite(now_sec) or now_sec < 0.0:
        raise ValueError("now_sec must be finite and non-negative")
    if state.updated_at_sec is not None and now_sec < state.updated_at_sec:
        raise ValueError("transport clock moved backwards")
    return now_sec


def acknowledge_puck_loaded(
    state: PuckTransportState,
    *,
    puck_id: str,
    custody_fencing_token: int,
    payload_mass_kg: float,
    retention_confirmed: bool,
    now_sec: float,
) -> PuckTransportState:
    """Record a positive load/retention acknowledgement.

    This function intentionally does not infer loading from arrival at a
    station.  A caller must first hold the matching custody token and receive a
    physical/operator load acknowledgement.
    """

    now_sec = _checked_time(state, now_sec)
    if not state.robot_id.strip():
        raise ValueError("robot_id must not be empty")
    puck_id = puck_id.strip()
    if not puck_id:
        raise ValueError("puck_id must not be empty")
    if state.puck_state == PuckState.HELD:
        raise ValueError("a puck is already loaded")
    if state.puck_state in (PuckState.DELIVERED, PuckState.LOST):
        raise ValueError("terminal transport state cannot be loaded")
    if custody_fencing_token <= 0:
        raise ValueError("a positive custody fencing token is required")
    if not math.isfinite(payload_mass_kg) or payload_mass_kg <= 0.0:
        raise ValueError("payload_mass_kg must be finite and positive")
    if not retention_confirmed:
        raise ValueError("puck retention must be confirmed before transport")
    return replace(
        state,
        puck_id=puck_id,
        puck_state=PuckState.HELD,
        custody_fencing_token=custody_fencing_token,
        payload_mass_kg=payload_mass_kg,
        retention_confirmed=True,
        updated_at_sec=now_sec,
        failure_reason="",
    )


def acknowledge_loaded_custody(
    state: PuckTransportState,
    custody: PuckCustody,
    *,
    payload_mass_kg: float,
    retention_confirmed: bool,
    now_sec: float,
) -> PuckTransportState:
    """Synchronize local transport state from a confirmed custody record."""

    if custody.state != CustodyState.LOADED:
        raise ValueError("puck custody must be confirmed loaded")
    if custody.owner_robot_id != state.robot_id:
        raise ValueError("puck custody belongs to another robot")
    return acknowledge_puck_loaded(
        state,
        puck_id=custody.puck_id,
        custody_fencing_token=custody.fencing_token,
        payload_mass_kg=payload_mass_kg,
        retention_confirmed=retention_confirmed,
        now_sec=now_sec,
    )


def acknowledge_puck_delivered(
    state: PuckTransportState,
    *,
    custody_fencing_token: int,
    now_sec: float,
) -> PuckTransportState:
    now_sec = _checked_time(state, now_sec)
    require_puck_loaded(state.puck_state)
    if custody_fencing_token != state.custody_fencing_token:
        raise ValueError("stale puck custody fencing token")
    return replace(
        state,
        puck_state=PuckState.DELIVERED,
        payload_mass_kg=0.0,
        retention_confirmed=False,
        updated_at_sec=now_sec,
    )


def acknowledge_delivered_custody(
    state: PuckTransportState,
    custody: PuckCustody,
    *,
    now_sec: float,
) -> PuckTransportState:
    if custody.state != CustodyState.DELIVERED:
        raise ValueError("puck custody must be confirmed delivered")
    if custody.owner_robot_id != state.robot_id or custody.puck_id != state.puck_id:
        raise ValueError("puck custody does not match the loaded transport")
    return acknowledge_puck_delivered(
        state,
        custody_fencing_token=custody.fencing_token,
        now_sec=now_sec,
    )


def report_loaded_puck_lost(
    state: PuckTransportState,
    *,
    custody_fencing_token: int,
    reason: str,
    now_sec: float,
) -> PuckTransportState:
    now_sec = _checked_time(state, now_sec)
    require_puck_loaded(state.puck_state)
    if custody_fencing_token != state.custody_fencing_token:
        raise ValueError("stale puck custody fencing token")
    reason = reason.strip()
    if not reason:
        raise ValueError("lost-puck reason must not be empty")
    return replace(
        state,
        puck_state=PuckState.LOST,
        payload_mass_kg=0.0,
        retention_confirmed=False,
        updated_at_sec=now_sec,
        failure_reason=reason,
    )


def acknowledge_lost_custody(
    state: PuckTransportState,
    custody: PuckCustody,
    *,
    now_sec: float,
) -> PuckTransportState:
    if custody.state != CustodyState.LOST:
        raise ValueError("puck custody must be marked lost")
    if custody.owner_robot_id != state.robot_id or custody.puck_id != state.puck_id:
        raise ValueError("puck custody does not match the loaded transport")
    return report_loaded_puck_lost(
        state,
        custody_fencing_token=custody.fencing_token,
        reason=custody.detail,
        now_sec=now_sec,
    )


def transport_motion_envelope(
    state: PuckTransportState,
    profile: CarrierProfile,
) -> MotionEnvelope:
    if state.puck_state == PuckState.UNKNOWN:
        raise ValueError("puck state must be known before motion")
    if state.puck_state == PuckState.LOST:
        raise ValueError("motion is blocked after puck loss until operator recovery")
    return build_motion_envelope(
        profile,
        puck_state=state.puck_state,
        payload_mass_kg=state.payload_mass_kg,
        retention_confirmed=state.retention_confirmed,
    )


def require_puck_loaded(puck_state: PuckState) -> None:
    if puck_state != PuckState.HELD:
        raise ValueError("puck must be loaded before transport")
