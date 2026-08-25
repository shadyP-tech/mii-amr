"""Deterministic static-map plausibility gate for confirmed stand candidates.

The gate is deliberately narrower than survey fusion: it only checks whether
the configured stand envelope can fit around each estimated centre without
touching blocked static-map geometry.  It does not merge candidates, inspect
the expected stand count, plan a route, or authorize motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.approach.dynamic_approach_planner import (
    point_clearance_to_blocked_m,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.coverage.stand_candidate_population_retention import (
    STATIC_MAP_DISPOSITION_ADMITTED,
    STATIC_MAP_DISPOSITION_BOUNDARY_PROVISIONAL,
    classify_static_map_population_retention,
)
from scripts.aufgabe04.perception.stand_confirmation import ConfirmedStand


STAND_CANDIDATE_STATIC_MAP_ADMISSION_SCHEMA_VERSION = 2
STATIC_MAP_CLEARANCE_BELOW_REQUIRED = "static_map_clearance_below_required"
STATIC_MAP_BOUNDARY_PROVISIONAL = (
    STATIC_MAP_DISPOSITION_BOUNDARY_PROVISIONAL
)
_CLEARANCE_COMPARISON_EPSILON_M = 1.0e-12


@dataclass(frozen=True)
class StandCandidateStaticMapEvidence:
    """One ordered candidate decision against immutable static geometry."""

    stand_id: str
    x_m: float
    y_m: float
    confidence: float
    hit_count: int
    source_observation_ids: tuple[str, ...]
    static_map_clearance_m: float
    required_clearance_m: float
    clearance_shortfall_m: float
    disposition: str
    population_retained: bool
    boundary_provisional: bool
    admitted: bool
    reasons: tuple[str, ...]

    def to_evidence_dict(self) -> dict[str, object]:
        return {
            "stand_id": self.stand_id,
            "pose": {"x_m": self.x_m, "y_m": self.y_m},
            "confidence": self.confidence,
            "hit_count": self.hit_count,
            "source_observation_ids": list(self.source_observation_ids),
            "static_map_clearance_m": self.static_map_clearance_m,
            "required_clearance_m": self.required_clearance_m,
            "clearance_shortfall_m": self.clearance_shortfall_m,
            "disposition": self.disposition,
            "population_retained": self.population_retained,
            "boundary_provisional": self.boundary_provisional,
            "admitted": self.admitted,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class StandCandidateStaticMapAdmission:
    """Complete ordered filtering decision for one observation epoch."""

    schema_version: int
    candidate_radius_m: float
    candidate_uncertainty_m: float
    required_clearance_m: float
    costmap_resolution_m: float
    costmap_width_cells: int
    costmap_height_cells: int
    blocked_cell_count: int
    evidence: tuple[StandCandidateStaticMapEvidence, ...]
    admitted_stands: tuple[ConfirmedStand, ...]
    boundary_provisional_stands: tuple[ConfirmedStand, ...]
    rejected_stands: tuple[ConfirmedStand, ...]

    @property
    def population_retained_stands(self) -> tuple[ConfirmedStand, ...]:
        """Return non-rejected stands retained for later camera validation.

        Boundary-provisional candidates are deliberately not part of
        ``admitted_stands`` because the static map did not prove their complete
        footprint clear.  They are retained only so camera validation can try a
        separately certified, interior-side approach later.
        """

        return tuple(
            sorted(
                self.admitted_stands + self.boundary_provisional_stands,
                key=lambda stand: (stand.x_m, stand.y_m, stand.stand_id),
            )
        )

    @property
    def camera_population_stands(self) -> tuple[ConfirmedStand, ...]:
        """Backward-compatible name for population-retained stands."""

        return self.population_retained_stands

    @property
    def disposition_by_stand_id(self) -> dict[str, str]:
        """Return explicit lineage for every stand retained in the registry."""

        return {
            item.stand_id: item.disposition
            for item in self.evidence
            if item.population_retained
        }

    def to_evidence_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "gate": "stand_candidate_static_map_admission",
            "motion_authorized": False,
            "contract": {
                "candidate_radius_m": self.candidate_radius_m,
                "candidate_uncertainty_m": self.candidate_uncertainty_m,
                "required_clearance_m": self.required_clearance_m,
                "boundary_provisional_rule": (
                    "nominal_radius_fits_but_uncertainty_envelope_does_not"
                ),
                "comparison_epsilon_m": _CLEARANCE_COMPARISON_EPSILON_M,
            },
            "costmap": {
                "resolution_m": self.costmap_resolution_m,
                "width_cells": self.costmap_width_cells,
                "height_cells": self.costmap_height_cells,
                "blocked_cell_count": self.blocked_cell_count,
            },
            "counts": {
                "evaluated": len(self.evidence),
                "admitted": len(self.admitted_stands),
                "boundary_provisional": len(self.boundary_provisional_stands),
                "population_retained": len(self.population_retained_stands),
                "rejected": len(self.rejected_stands),
            },
            "admitted_stand_ids": [
                stand.stand_id for stand in self.admitted_stands
            ],
            "rejected_stand_ids": [
                stand.stand_id for stand in self.rejected_stands
            ],
            "boundary_provisional_stand_ids": [
                stand.stand_id for stand in self.boundary_provisional_stands
            ],
            "camera_population_stand_ids": [
                stand.stand_id for stand in self.population_retained_stands
            ],
            "candidate_evidence": [
                item.to_evidence_dict() for item in self.evidence
            ],
        }


def evaluate_stand_candidate_static_map_admission(
    costmap: Costmap,
    stands: Iterable[ConfirmedStand],
    *,
    candidate_radius_m: float,
    candidate_uncertainty_m: float,
) -> StandCandidateStaticMapAdmission:
    """Filter confirmed stands that overlap the static-map safety envelope.

    Both envelope terms are explicit inputs because they belong to the frozen
    survey contract.  The input order is intentionally ignored; evidence and
    returned stand tuples use the same stable ``(x, y, stand_id)`` ordering as
    survey fusion.
    """

    _validate_contract_value(
        candidate_radius_m,
        name="candidate radius",
        strictly_positive=True,
    )
    _validate_contract_value(
        candidate_uncertainty_m,
        name="candidate uncertainty",
        strictly_positive=False,
    )
    required_clearance_m = candidate_radius_m + candidate_uncertainty_m
    if not math.isfinite(required_clearance_m):
        raise ValueError("required candidate clearance must be finite")
    stand_snapshot = tuple(stands)
    for stand in stand_snapshot:
        _validate_stand_geometry(stand)
    ordered_stands = tuple(
        sorted(
            stand_snapshot,
            key=lambda stand: (stand.x_m, stand.y_m, stand.stand_id),
        )
    )
    seen_ids: set[str] = set()
    evidence: list[StandCandidateStaticMapEvidence] = []
    admitted: list[ConfirmedStand] = []
    boundary_provisional: list[ConfirmedStand] = []
    rejected: list[ConfirmedStand] = []
    for stand in ordered_stands:
        if stand.stand_id in seen_ids:
            raise ValueError(f"duplicate confirmed stand ID: {stand.stand_id!r}")
        seen_ids.add(stand.stand_id)
        clearance_m = point_clearance_to_blocked_m(
            costmap,
            Pose2D(stand.x_m, stand.y_m, 0.0),
        )
        retention = classify_static_map_population_retention(
            clearance_m=clearance_m,
            candidate_radius_m=candidate_radius_m,
            candidate_uncertainty_m=candidate_uncertainty_m,
        )
        reasons = (
            () if retention.strictly_admitted
            else (STATIC_MAP_CLEARANCE_BELOW_REQUIRED,)
        )
        evidence.append(
            StandCandidateStaticMapEvidence(
                stand_id=stand.stand_id,
                x_m=stand.x_m,
                y_m=stand.y_m,
                confidence=stand.confidence,
                hit_count=stand.hit_count,
                source_observation_ids=stand.source_observation_ids,
                static_map_clearance_m=clearance_m,
                required_clearance_m=required_clearance_m,
                clearance_shortfall_m=retention.clearance_shortfall_m,
                disposition=retention.disposition,
                population_retained=retention.population_retained,
                boundary_provisional=retention.boundary_provisional,
                admitted=retention.strictly_admitted,
                reasons=reasons,
            )
        )
        if retention.disposition == STATIC_MAP_DISPOSITION_ADMITTED:
            admitted.append(stand)
        elif retention.disposition == STATIC_MAP_DISPOSITION_BOUNDARY_PROVISIONAL:
            boundary_provisional.append(stand)
        else:
            rejected.append(stand)

    return StandCandidateStaticMapAdmission(
        schema_version=STAND_CANDIDATE_STATIC_MAP_ADMISSION_SCHEMA_VERSION,
        candidate_radius_m=candidate_radius_m,
        candidate_uncertainty_m=candidate_uncertainty_m,
        required_clearance_m=required_clearance_m,
        costmap_resolution_m=costmap.resolution,
        costmap_width_cells=costmap.width,
        costmap_height_cells=costmap.height,
        blocked_cell_count=len(costmap.blocked_cells),
        evidence=tuple(evidence),
        admitted_stands=tuple(admitted),
        boundary_provisional_stands=tuple(boundary_provisional),
        rejected_stands=tuple(rejected),
    )


def _validate_contract_value(
    value: float,
    *,
    name: str,
    strictly_positive: bool,
) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if not strictly_positive and value < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _validate_stand_geometry(stand: ConfirmedStand) -> None:
    if not isinstance(stand, ConfirmedStand):
        raise TypeError("stands must contain ConfirmedStand values")
    if not stand.stand_id.strip():
        raise ValueError("confirmed stand ID must not be empty")
    if not math.isfinite(stand.x_m) or not math.isfinite(stand.y_m):
        raise ValueError(f"confirmed stand {stand.stand_id!r} pose must be finite")
    if not math.isfinite(stand.confidence):
        raise ValueError(
            f"confirmed stand {stand.stand_id!r} confidence must be finite"
        )
    if type(stand.hit_count) is not int or stand.hit_count <= 0:
        raise ValueError(
            f"confirmed stand {stand.stand_id!r} hit count must be positive"
        )
    if not stand.source_observation_ids or any(
        not str(observation_id).strip()
        for observation_id in stand.source_observation_ids
    ):
        raise ValueError(
            f"confirmed stand {stand.stand_id!r} source IDs must be non-empty"
        )


__all__ = [
    "STATIC_MAP_CLEARANCE_BELOW_REQUIRED",
    "STATIC_MAP_BOUNDARY_PROVISIONAL",
    "STAND_CANDIDATE_STATIC_MAP_ADMISSION_SCHEMA_VERSION",
    "StandCandidateStaticMapAdmission",
    "StandCandidateStaticMapEvidence",
    "evaluate_stand_candidate_static_map_admission",
]
