"""Preserve cross-view morphology conflicts without changing obstacle authority."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from scripts.aufgabe04.artifacts.candidate_perception_advisory import (
    CandidatePerceptionAdvisory, MORPHOLOGY_CONFLICT, VISIBILITY_GAP,
)
from scripts.aufgabe04.navigation.coverage.candidate_frame_registry import (
    frame_provenance_from_confirmed_stand,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    CoverageSurveyPlan, StandSurveyRegistry, STATUS_REJECTED,
    coverage_survey_plan_sha256, stand_survey_registry_sha256,
    validate_stand_survey_registry,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.perception.lidar_stand_morphology import StandMorphologyAdmission


@dataclass(frozen=True)
class CoverageMorphologyConflictEvidence:
    updated_registry: StandSurveyRegistry
    advisories: tuple[CandidatePerceptionAdvisory, ...]
    payload: dict[str, object]


def retain_candidate_perception_advisories(
    registry: StandSurveyRegistry,
    *,
    plan: CoverageSurveyPlan,
    static_costmap: Costmap,
    morphology: StandMorphologyAdmission,
    viewpoint_id: str,
    morphology_sha256: str,
    proposal_max_range_m: float,
    prior_registry: StandSurveyRegistry | None = None,
) -> CoverageMorphologyConflictEvidence:
    """Retain all spatially plausible conflicts, including ambiguous matches.

    Rejected track geometry is compared only in canonical odom. A broad
    neighboring return is insufficient to delete a candidate or remove its
    keepout. Legacy frame-free fixtures produce diagnostics, never invented
    odom associations.
    """

    validate_stand_survey_registry(registry, plan)
    historical_registry = registry if prior_registry is None else prior_registry
    validate_stand_survey_registry(historical_registry, plan)
    if not {c.candidate_uid for c in historical_registry.candidates}.issubset(
        {c.candidate_uid for c in registry.candidates}
    ):
        raise ValueError("post-fusion registry lost historical candidate identities")
    if viewpoint_id not in plan.viewpoint_ids:
        raise ValueError("morphology advisory viewpoint is not in the survey plan")
    if type(proposal_max_range_m) not in (float, int) or not math.isfinite(proposal_max_range_m) or proposal_max_range_m <= 0:
        raise ValueError("proposal range must be finite and positive")
    plan_sha = coverage_survey_plan_sha256(plan)
    def eligible_other_viewpoints(candidate):
        cell = static_costmap.world_to_grid(Pose2D(candidate.x_m, candidate.y_m, 0.0))
        return tuple(
            v.viewpoint_id for v in plan.viewpoints
            if v.viewpoint_id not in candidate.viewpoint_ids and cell in v.visible_cells
        )
    evidence_by_id = {item.stand_id: item for item in morphology.evidence}
    if len(evidence_by_id) != len(morphology.evidence):
        raise ValueError("duplicate morphology evidence stand ID")
    rejected_ids = [track.stand_id for track in morphology.rejected_stands]
    if len(rejected_ids) != len(set(rejected_ids)) or set(rejected_ids) != {
        item.stand_id for item in morphology.evidence if not item.assessment.accepted
    }:
        raise ValueError("rejected track population differs from morphology evidence")
    advisories: list[CandidatePerceptionAdvisory] = []
    rejected_tracks = []

    def common(candidate):
        return dict(
            candidate_uid=candidate.candidate_uid, survey_id=registry.survey_id,
            map_bundle_sha256=registry.map_bundle_sha256, plan_sha256=plan_sha,
            viewpoint_id=viewpoint_id, source_morphology_sha256=morphology_sha256,
            candidate_frame=candidate.frame_provenance,
            candidate_source_viewpoint_ids=tuple(sorted(candidate.viewpoint_ids)),
            source_observation_ids=tuple(sorted(candidate.source_observation_ids)),
            proposal_max_range_m=proposal_max_range_m,
            visibility_radius_m=plan.config.visibility_radius_m,
            eligible_other_viewpoint_ids=eligible_other_viewpoints(candidate),
        )

    for track in morphology.rejected_stands:
        evidence = evidence_by_id.get(track.stand_id)
        if evidence is None or evidence.assessment.accepted or not evidence.assessment.rejection_reasons:
            raise ValueError("rejected morphology track lacks rejection evidence")
        if set(evidence.source_observation_ids) != set(track.source_observation_ids):
            raise ValueError("rejected morphology track source IDs differ")
        frame = frame_provenance_from_confirmed_stand(
            track, expected_map_frame=registry.planning_frame,
            expected_map_bundle_sha256=registry.map_bundle_sha256,
        )
        matches = []
        for candidate in historical_registry.candidates:
            if candidate.status == STATUS_REJECTED or viewpoint_id in candidate.viewpoint_ids:
                continue
            candidate_frame = candidate.frame_provenance
            if (frame is None) != (candidate_frame is None):
                raise ValueError("cannot associate mixed legacy and frame-bound morphology")
            if frame is None:
                continue
            if (frame.map_frame, frame.odom_frame) != (
                candidate_frame.map_frame, candidate_frame.odom_frame
            ):
                raise ValueError("morphology conflict uses incompatible frames")
            a, b = frame.canonical_odom_point, candidate_frame.canonical_odom_point
            distance = math.hypot(a.x_m - b.x_m, a.y_m - b.y_m)
            if distance <= plan.config.candidate_merge_distance_m + 1e-12:
                matches.append((candidate, distance))
        possible_uids = tuple(sorted(candidate.candidate_uid for candidate, _ in matches))
        rejected_tracks.append({
            "track_id": track.stand_id,
            "source_observation_ids": list(track.source_observation_ids),
            "frame_provenance": None if frame is None else frame.to_mapping(),
            "rejection_reasons": list(evidence.assessment.rejection_reasons),
            "possible_candidate_uids": list(possible_uids),
            "association_status": (
                "frame_unavailable" if frame is None else
                "unassociated" if not matches else
                "ambiguous_spatial_neighbors" if len(matches) > 1 else "unique_spatial_neighbor"
            ),
        })
        for candidate, distance in matches:
            advisories.append(CandidatePerceptionAdvisory(
                kind=MORPHOLOGY_CONFLICT, **common(candidate),
                track_id=track.stand_id, track_frame=frame,
                track_source_observation_ids=tuple(sorted(track.source_observation_ids)),
                rejection_reasons=evidence.assessment.rejection_reasons,
                association_distance_m=distance,
                association_limit_m=plan.config.candidate_merge_distance_m,
                possible_candidate_uids=possible_uids,
            ))
    for candidate in registry.candidates:
        if (candidate.status != STATUS_REJECTED and candidate.frame_provenance is not None
            and len(candidate.viewpoint_ids) == 1 and not eligible_other_viewpoints(candidate)
            and proposal_max_range_m > plan.config.visibility_radius_m):
            advisories.append(CandidatePerceptionAdvisory(kind=VISIBILITY_GAP, **common(candidate)))
    by_candidate = {}
    for advisory in advisories:
        by_candidate.setdefault(advisory.candidate_uid, []).append(advisory)
    updated_candidates = []
    for candidate in registry.candidates:
        records = {item.sha256: item for item in candidate.perception_advisories}
        records.update({item.sha256: item for item in by_candidate.get(candidate.candidate_uid, ())})
        updated_candidates.append(replace(
            candidate, perception_advisories=tuple(records[key] for key in sorted(records)),
        ))
    updated = replace(registry, candidates=tuple(updated_candidates))
    validate_stand_survey_registry(updated, plan)
    return CoverageMorphologyConflictEvidence(
        updated_registry=updated, advisories=tuple(advisories),
        payload={
            "schema_version": 1, "policy": "unresolved_advisory_only",
            "motion_authorized": False, "candidate_rejection_authorized": False,
            "keepouts_changed": False, "expected_stand_count_used": False,
            "survey_id": registry.survey_id, "map_bundle_sha256": registry.map_bundle_sha256,
            "plan_sha256": plan_sha, "viewpoint_id": viewpoint_id,
            "source_morphology_sha256": morphology_sha256,
            "source_registry_sha256": stand_survey_registry_sha256(registry),
            "prior_epoch_registry_sha256": stand_survey_registry_sha256(historical_registry),
            "updated_registry_sha256": stand_survey_registry_sha256(updated),
            "proposal_max_range_m": proposal_max_range_m,
            "visibility_radius_m": plan.config.visibility_radius_m,
            "advisories": [item.to_dict() for item in advisories],
            "rejected_track_dispositions": rejected_tracks,
        },
    )
