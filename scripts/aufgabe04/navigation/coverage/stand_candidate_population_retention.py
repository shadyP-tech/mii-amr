"""Pure static-map disposition policy for LiDAR stand populations.

Strict static-map admission and camera-population retention are deliberately
different questions.  A candidate whose nominal stand radius fits in free
space, but whose uncertainty envelope touches mapped geometry, may remain in
the camera-validation population.  This module never authorizes motion; route
planning and live execution keep their independent safety gates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


STATIC_MAP_DISPOSITION_ADMITTED = "static_map_admitted"
STATIC_MAP_DISPOSITION_BOUNDARY_PROVISIONAL = "boundary_provisional"
STATIC_MAP_DISPOSITION_REJECTED = "rejected"

RETAINED_STATIC_MAP_DISPOSITIONS = frozenset(
    {
        STATIC_MAP_DISPOSITION_ADMITTED,
        STATIC_MAP_DISPOSITION_BOUNDARY_PROVISIONAL,
    }
)
VALID_STATIC_MAP_DISPOSITIONS = frozenset(
    {*RETAINED_STATIC_MAP_DISPOSITIONS, STATIC_MAP_DISPOSITION_REJECTED}
)

_CLEARANCE_COMPARISON_EPSILON_M = 1.0e-12


@dataclass(frozen=True)
class StaticMapPopulationRetention:
    """One motion-neutral disposition derived from immutable map geometry."""

    disposition: str
    clearance_m: float
    nominal_radius_m: float
    uncertainty_m: float
    required_clearance_m: float
    clearance_shortfall_m: float
    strictly_admitted: bool
    population_retained: bool

    @property
    def boundary_provisional(self) -> bool:
        return (
            self.disposition
            == STATIC_MAP_DISPOSITION_BOUNDARY_PROVISIONAL
        )


def classify_static_map_population_retention(
    *,
    clearance_m: float,
    candidate_radius_m: float,
    candidate_uncertainty_m: float,
) -> StaticMapPopulationRetention:
    """Classify strict admission, uncertainty-only overlap, or rejection.

    The middle interval is the only relaxation allowed for population
    retention::

        clearance >= radius + uncertainty  -> strict admission
        radius <= clearance < full envelope -> boundary provisional
        clearance < radius                  -> rejected

    A boundary-provisional result is perception evidence only.  It cannot be
    interpreted as a route or motion permit.
    """

    clearance = _finite_nonnegative(clearance_m, "static-map clearance")
    radius = _finite_nonnegative(candidate_radius_m, "candidate radius")
    if radius <= 0.0:
        raise ValueError("candidate radius must be positive")
    uncertainty = _finite_nonnegative(
        candidate_uncertainty_m,
        "candidate uncertainty",
    )
    required = radius + uncertainty
    if not math.isfinite(required):
        raise ValueError("required candidate clearance must be finite")

    strictly_admitted = (
        clearance + _CLEARANCE_COMPARISON_EPSILON_M >= required
    )
    nominal_radius_fits = (
        clearance + _CLEARANCE_COMPARISON_EPSILON_M >= radius
    )
    if strictly_admitted:
        disposition = STATIC_MAP_DISPOSITION_ADMITTED
    elif nominal_radius_fits:
        disposition = STATIC_MAP_DISPOSITION_BOUNDARY_PROVISIONAL
    else:
        disposition = STATIC_MAP_DISPOSITION_REJECTED
    return StaticMapPopulationRetention(
        disposition=disposition,
        clearance_m=clearance,
        nominal_radius_m=radius,
        uncertainty_m=uncertainty,
        required_clearance_m=required,
        clearance_shortfall_m=max(0.0, required - clearance),
        strictly_admitted=strictly_admitted,
        population_retained=(
            disposition in RETAINED_STATIC_MAP_DISPOSITIONS
        ),
    )


def validate_retained_static_map_disposition(value: str) -> str:
    """Validate a disposition that is allowed to enter the survey registry."""

    if value not in RETAINED_STATIC_MAP_DISPOSITIONS:
        raise ValueError(
            "registry static-map disposition must be static_map_admitted "
            "or boundary_provisional"
        )
    return value


def merge_retained_static_map_dispositions(
    existing: str,
    incoming: str,
) -> str:
    """Preserve strict evidence and allow a later strict observation upgrade."""

    existing = validate_retained_static_map_disposition(existing)
    incoming = validate_retained_static_map_disposition(incoming)
    if STATIC_MAP_DISPOSITION_ADMITTED in {existing, incoming}:
        return STATIC_MAP_DISPOSITION_ADMITTED
    return STATIC_MAP_DISPOSITION_BOUNDARY_PROVISIONAL


def _finite_nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


__all__ = [
    "RETAINED_STATIC_MAP_DISPOSITIONS",
    "STATIC_MAP_DISPOSITION_ADMITTED",
    "STATIC_MAP_DISPOSITION_BOUNDARY_PROVISIONAL",
    "STATIC_MAP_DISPOSITION_REJECTED",
    "StaticMapPopulationRetention",
    "VALID_STATIC_MAP_DISPOSITIONS",
    "classify_static_map_population_retention",
    "merge_retained_static_map_dispositions",
    "validate_retained_static_map_disposition",
]
