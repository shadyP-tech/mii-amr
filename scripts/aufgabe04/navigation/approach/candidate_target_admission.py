"""Read-only admission of camera targets against current geometry and evidence.

Deferral removes a candidate from camera selection only. Its frozen geometry,
source evidence, and route keepout remain part of the full snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from scripts.aufgabe04.artifacts.candidate_perception_advisory import MORPHOLOGY_CONFLICT
from scripts.aufgabe04.navigation.approach.camera_candidate_selection import CameraCandidateSelectionError
from scripts.aufgabe04.navigation.approach.candidate_preapproach_models import CandidatePreapproachUnreachableError
from scripts.aufgabe04.navigation.coverage.stand_candidate_static_map_admission import (
    StandCandidateStaticMapEvidence,
    evaluate_stand_candidate_static_map_admission,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    CoverageSurveyPlan, validate_coverage_survey_plan,
)
from scripts.aufgabe04.navigation.planning.costmap import (
    CELL_SOURCE_INFLATED, CELL_SOURCE_STATION_KEEPOUT, Costmap,
)
from scripts.aufgabe04.navigation.planning.map_io import load_occupancy_grid_with_bundle
from scripts.aufgabe04.perception.stand_confirmation import ConfirmedStand
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateGeometry, CandidateSnapshot, FrozenCandidate,
    candidate_geometry_sha256, validate_candidate_geometry,
    validate_candidate_snapshot, validate_frozen_candidate,
)


TARGET_STATIC_MAP_INCOMPATIBLE = "target_static_map_incompatible"
UNRESOLVED_MORPHOLOGY_CONFLICT = "unresolved_morphology_conflict"


@dataclass(frozen=True)
class CandidateTargetAdmission:
    candidate_uid: str
    accepted: bool
    reasons: tuple[str, ...]
    candidate_geometry_sha256: str
    target_geometry_sha256: str
    static_map_evidence: StandCandidateStaticMapEvidence
    unresolved_morphology_conflict_sha256: tuple[str, ...]

    def to_evidence(self) -> dict[str, object]:
        return {
            "candidate_uid": self.candidate_uid,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "candidate_geometry_sha256": self.candidate_geometry_sha256,
            "target_geometry_sha256": self.target_geometry_sha256,
            "static_map_evidence": self.static_map_evidence.to_evidence_dict(),
            "unresolved_morphology_conflict_sha256": list(self.unresolved_morphology_conflict_sha256),
            "motion_authorized": False,
            "candidate_rejection_authorized": False,
        }


def evaluate_candidate_target_admission(
    candidate: FrozenCandidate,
    static_costmap: Costmap,
    *,
    target_geometry: CandidateGeometry | None = None,
) -> CandidateTargetAdmission:
    """Reapply the stand envelope policy and defer unresolved contradictions.

    The map must be uninflated static geometry plus the arena overlay: robot
    inflation and stand transit keepouts are separate route-safety constraints.
    Nominal-fit boundary candidates remain eligible for a certified approach.
    """
    validate_frozen_candidate(candidate)
    geometry = candidate.geometry if target_geometry is None else target_geometry
    validate_candidate_geometry(geometry)
    if not isinstance(static_costmap, Costmap):
        raise TypeError("candidate target admission requires a static Costmap")
    if any(source in {CELL_SOURCE_INFLATED, CELL_SOURCE_STATION_KEEPOUT}
           for source in static_costmap.cell_sources.values()):
        raise ValueError("candidate target admission requires an uninflated static costmap")
    stand = ConfirmedStand(
        stand_id=candidate.candidate_uid, x_m=geometry.x_m, y_m=geometry.y_m,
        confidence=candidate.confidence, hit_count=candidate.hit_count,
        source_observation_ids=candidate.source.observation_ids,
        first_seen_sec=candidate.first_seen_sec, last_seen_sec=candidate.last_seen_sec,
        first_confirmed_at_sec=candidate.last_seen_sec,
        provenance={"source": "candidate_target_admission"},
    )
    static = evaluate_stand_candidate_static_map_admission(
        static_costmap, (stand,), candidate_radius_m=geometry.radius_m,
        candidate_uncertainty_m=geometry.uncertainty_m,
    ).evidence[0]
    conflicts = tuple(sorted(
        advisory.sha256 for advisory in candidate.source.perception_advisories
        if advisory.kind == MORPHOLOGY_CONFLICT
    ))
    reasons = (() if static.population_retained else (TARGET_STATIC_MAP_INCOMPATIBLE,))
    if conflicts:
        reasons += (UNRESOLVED_MORPHOLOGY_CONFLICT,)
    return CandidateTargetAdmission(
        candidate_uid=candidate.candidate_uid, accepted=not reasons, reasons=reasons,
        candidate_geometry_sha256=candidate_geometry_sha256(candidate.geometry),
        target_geometry_sha256=candidate_geometry_sha256(geometry),
        static_map_evidence=static, unresolved_morphology_conflict_sha256=conflicts,
    )


def validate_candidate_target_bindings(
    *, plan: CoverageSurveyPlan, snapshot: CandidateSnapshot,
) -> None:
    validate_coverage_survey_plan(plan)
    validate_candidate_snapshot(snapshot)
    if snapshot.map_bundle_sha256 != plan.map_bundle_sha256:
        raise ValueError("candidate snapshot map differs from coverage plan")
    if snapshot.planning_frame != plan.planning_frame:
        raise ValueError("candidate snapshot frame differs from coverage plan")


def load_candidate_target_costmap(
    map_yaml: Path, *, semantic_map_id: str,
    plan: CoverageSurveyPlan, snapshot: CandidateSnapshot,
) -> Costmap:
    """Load bound, uninflated static geometry including the arena boundary."""
    validate_candidate_target_bindings(plan=plan, snapshot=snapshot)
    grid, bundle = load_occupancy_grid_with_bundle(
        map_yaml, semantic_map_id=semantic_map_id, planning_frame=plan.planning_frame,
    )
    if bundle.bundle_sha256 != snapshot.map_bundle_sha256:
        raise ValueError("candidate snapshot map differs from runtime map")
    return Costmap.from_occupancy_grid(grid).with_arena_bounds(plan.arena_bounds)


def require_candidate_target_admission(decision: CandidateTargetAdmission) -> None:
    if not decision.accepted:
        error = CandidatePreapproachUnreachableError(
            decision.candidate_uid, "candidate_target_ineligible: " + ", ".join(decision.reasons),
        )
        error.target_admission_evidence = candidate_target_admission_evidence((decision,))
        raise error


def candidate_target_admission_evidence(
    decisions: Iterable[CandidateTargetAdmission],
) -> dict[str, object]:
    ordered = tuple(sorted(decisions, key=lambda item: item.candidate_uid))
    return {
        "schema_version": 1, "gate": "candidate_target_admission",
        "eligible_candidate_uids": [item.candidate_uid for item in ordered if item.accepted],
        "excluded_candidate_uids": [item.candidate_uid for item in ordered if not item.accepted],
        "candidate_decisions": {item.candidate_uid: item.to_evidence() for item in ordered},
        "motion_authorized": False, "candidate_rejection_authorized": False,
    }


class NoEligibleCameraTargetError(CameraCandidateSelectionError):
    """All unresolved targets need reconciliation before camera inspection."""

    def __init__(self, decisions: Iterable[CandidateTargetAdmission]) -> None:
        super().__init__("no_eligible_camera_target", "no unresolved camera target passed target admission")
        self.target_admission_evidence = candidate_target_admission_evidence(decisions)

    def to_evidence(self) -> dict[str, object]:
        return {
            "error_code": self.code, "reason": str(self),
            "candidate_target_admission": self.target_admission_evidence,
            "motion_authorized": False,
        }
