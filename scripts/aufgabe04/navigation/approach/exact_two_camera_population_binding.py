"""Authenticate exact-two camera snapshots against the full LiDAR registry.

The immutable handoff seals the complete LiDAR candidate population, while
the camera snapshot intentionally contains only the selected candidates.  This
module owns that subset relationship and the narrowly state-aware registry
digest check needed after earlier selected candidates have been decided.  It
is ROS-free, writes nothing, and never authorizes motion.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.navigation.approach.exact_two_camera_artifacts import (
    validate_live_candidate_snapshot_binding,
)
from scripts.aufgabe04.navigation.approach.exact_two_camera_contract import (
    ExactTwoCameraAdmissionError,
    ExactTwoCameraHandoffArtifact,
    stand_survey_registry_sha256,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    REJECTION_BASIS_CAMERA,
    STATUS_CONFIRMED,
    STATUS_REJECTED,
    StandSurveyRegistry,
    SurveyCandidate,
    validate_stand_survey_registry,
)
from scripts.aufgabe04.stations.candidate_snapshot import CandidateSnapshot


def validate_live_exact_two_camera_population_binding(
    handoff: ExactTwoCameraHandoffArtifact,
    snapshot: CandidateSnapshot,
    registry: StandSurveyRegistry,
    *,
    candidate_snapshot_path: str | Path | None = None,
    authenticated_decision_statuses: Mapping[str, str] | None = None,
) -> dict[str, SurveyCandidate]:
    """Authenticate the registry and return its selected live candidates.

    Selected lifecycle changes made by prior camera decisions are normalized
    only for the source-digest comparison.  Every other registry field,
    including retained candidates that were already inactive when the camera
    population was admitted, remains byte-for-byte accountable to the
    handoff's full source-registry seal.
    """

    validate_live_candidate_snapshot_binding(
        handoff,
        snapshot,
        candidate_snapshot_path=candidate_snapshot_path,
    )
    validate_stand_survey_registry(registry)
    admission = handoff.admission_decision
    selected_uids = tuple(admission.selected_candidate_uids)
    if snapshot.candidate_uids != selected_uids:
        raise ExactTwoCameraAdmissionError(
            "live_snapshot_mismatch",
            "candidate snapshot UIDs do not match selected camera UIDs",
        )

    registry_by_uid = {
        candidate.candidate_uid: candidate
        for candidate in registry.candidates
    }
    _validate_camera_population_membership(
        handoff,
        registry_uids=set(registry_by_uid),
    )

    _validate_state_aware_full_registry_binding(
        handoff,
        registry,
        authenticated_decision_statuses=authenticated_decision_statuses,
    )
    return {uid: registry_by_uid[uid] for uid in selected_uids}


def _validate_camera_population_membership(
    handoff: ExactTwoCameraHandoffArtifact,
    *,
    registry_uids: set[str],
) -> None:
    """Require the sealed active-camera population in the full registry.

    Camera admission evidence intentionally covers only candidates that were
    active LiDAR candidates at handoff time.  The survey registry is a fuller
    lifecycle ledger and can also retain already-rejected LiDAR candidates.
    Those inactive entries are authenticated below by the exact full-registry
    digest; requiring them to appear in active-camera evidence would compare
    two deliberately different populations.
    """

    admission = handoff.admission_decision
    selected_uids = set(admission.selected_candidate_uids)
    missing_selected = selected_uids.difference(registry_uids)
    if missing_selected:
        raise ExactTwoCameraAdmissionError(
            "live_registry_population_mismatch",
            "selected camera UIDs are missing from the live registry: "
            f"{sorted(missing_selected)}",
        )

    camera_evidence_uids = {
        evidence.candidate_uid for evidence in admission.candidate_evidence
    }
    missing_camera_evidence = camera_evidence_uids.difference(registry_uids)
    if missing_camera_evidence:
        raise ExactTwoCameraAdmissionError(
            "live_registry_population_mismatch",
            "sealed camera-evidence UIDs are missing from the live registry: "
            f"{sorted(missing_camera_evidence)}",
        )


def _validate_state_aware_full_registry_binding(
    handoff: ExactTwoCameraHandoffArtifact,
    registry: StandSurveyRegistry,
    *,
    authenticated_decision_statuses: Mapping[str, str] | None,
) -> None:
    """Validate the seal and receipt-bound prior selected decisions."""

    if (
        registry.survey_id != handoff.survey_id
        or registry.planning_frame != handoff.planning_frame
        or registry.map_bundle_sha256 != handoff.map_bundle_sha256
    ):
        raise ExactTwoCameraAdmissionError(
            "live_registry_mismatch",
            "live stand registry metadata no longer matches the sealed "
            "camera handoff",
        )
    actual_registry_sha256 = stand_survey_registry_sha256(registry)
    evidence_by_uid = {
        evidence.candidate_uid: evidence
        for evidence in handoff.admission_decision.candidate_evidence
    }
    selected_uids = set(
        handoff.admission_decision.selected_candidate_uids
    )
    authenticated_statuses = _authenticated_statuses(
        authenticated_decision_statuses,
        selected_uids=selected_uids,
    )
    changed_statuses: dict[str, str] = {}
    reconstructed_candidates: list[SurveyCandidate] = []
    for candidate in registry.candidates:
        if candidate.candidate_uid not in selected_uids:
            reconstructed_candidates.append(candidate)
            continue
        evidence = evidence_by_uid[candidate.candidate_uid]
        lifecycle_unchanged = (
            candidate.status == evidence.registry_status
            and candidate.rejection_basis is None
        )
        lifecycle_decided = (
            candidate.status == STATUS_CONFIRMED
            and candidate.rejection_basis is None
        ) or (
            candidate.status == STATUS_REJECTED
            and candidate.rejection_basis == REJECTION_BASIS_CAMERA
        )
        lifecycle_is_valid = lifecycle_unchanged or lifecycle_decided
        if not lifecycle_is_valid:
            raise ExactTwoCameraAdmissionError(
                "live_registry_mismatch",
                "selected candidate has an unsealed lifecycle transition: "
                f"{candidate.candidate_uid!r}",
            )
        if lifecycle_decided and not lifecycle_unchanged:
            changed_statuses[candidate.candidate_uid] = candidate.status
        reconstructed_candidates.append(
            replace(
                candidate,
                status=evidence.registry_status,
                rejection_basis=None,
            )
        )

    if authenticated_statuses != changed_statuses:
        raise ExactTwoCameraAdmissionError(
            "live_registry_mismatch",
            "selected candidate lifecycle transitions lack matching "
            "authenticated canonical decision receipts",
        )
    if actual_registry_sha256 == handoff.source_registry_sha256:
        return

    reconstructed = replace(
        registry,
        candidates=tuple(reconstructed_candidates),
    )
    if (
        stand_survey_registry_sha256(reconstructed)
        != handoff.source_registry_sha256
    ):
        raise ExactTwoCameraAdmissionError(
            "live_registry_mismatch",
            "live stand registry no longer matches the sealed camera handoff",
        )


def _authenticated_statuses(
    value: Mapping[str, str] | None,
    *,
    selected_uids: set[str],
) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ExactTwoCameraAdmissionError(
            "live_registry_mismatch",
            "authenticated decision statuses must be a mapping",
        )
    statuses = dict(value)
    if (
        any(
            not isinstance(uid, str)
            or status not in {STATUS_CONFIRMED, STATUS_REJECTED}
            for uid, status in statuses.items()
        )
        or not set(statuses).issubset(selected_uids)
    ):
        raise ExactTwoCameraAdmissionError(
            "live_registry_mismatch",
            "authenticated decision statuses contain an invalid candidate",
        )
    return statuses
