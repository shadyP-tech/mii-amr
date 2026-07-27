"""Pure carrier geometry and payload constraints.

The navigation stack must plan with the *effective* footprint.  Keeping this
calculation in a ROS-free model makes it possible to use exactly the same
loaded envelope in planners, fleet collision checks, and offline tests.
"""

import math
from dataclasses import dataclass

from .models import PuckState


def _require_finite_non_negative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _require_finite_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class CarrierProfile:
    """Measured robot/carrier parameters used to authorize loaded motion."""

    profile_id: str
    unloaded_footprint_radius_m: float
    loaded_footprint_radius_m: float
    empty_robot_mass_kg: float
    max_payload_mass_kg: float
    retention_required: bool = True

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("profile_id must not be empty")
        _require_finite_positive(
            "unloaded_footprint_radius_m", self.unloaded_footprint_radius_m
        )
        _require_finite_positive(
            "loaded_footprint_radius_m", self.loaded_footprint_radius_m
        )
        if self.loaded_footprint_radius_m < self.unloaded_footprint_radius_m:
            raise ValueError(
                "loaded_footprint_radius_m must be at least the unloaded radius"
            )
        _require_finite_positive("empty_robot_mass_kg", self.empty_robot_mass_kg)
        _require_finite_non_negative("max_payload_mass_kg", self.max_payload_mass_kg)

    def footprint_radius_m(self, puck_state: PuckState) -> float:
        if puck_state == PuckState.HELD:
            return self.loaded_footprint_radius_m
        return self.unloaded_footprint_radius_m


@dataclass(frozen=True)
class MotionEnvelope:
    footprint_radius_m: float
    total_mass_kg: float
    payload_mass_kg: float
    loaded: bool


def build_motion_envelope(
    profile: CarrierProfile,
    *,
    puck_state: PuckState,
    payload_mass_kg: float = 0.0,
    retention_confirmed: bool = False,
) -> MotionEnvelope:
    """Return a certified footprint/mass envelope or reject unsafe inputs."""

    _require_finite_non_negative("payload_mass_kg", payload_mass_kg)
    loaded = puck_state == PuckState.HELD
    if loaded:
        if payload_mass_kg <= 0.0:
            raise ValueError("loaded motion requires a positive payload mass")
        if payload_mass_kg > profile.max_payload_mass_kg:
            raise ValueError("payload exceeds the carrier profile mass limit")
        if profile.retention_required and not retention_confirmed:
            raise ValueError("loaded motion requires confirmed puck retention")
    elif payload_mass_kg != 0.0:
        raise ValueError("payload mass must be zero when no puck is held")

    return MotionEnvelope(
        footprint_radius_m=profile.footprint_radius_m(puck_state),
        total_mass_kg=profile.empty_robot_mass_kg + payload_mass_kg,
        payload_mass_kg=payload_mass_kg,
        loaded=loaded,
    )
