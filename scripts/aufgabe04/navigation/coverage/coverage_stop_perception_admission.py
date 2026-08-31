"""ROS-free perception admission for one stopped coverage epoch.

The record command owns file-system transaction ordering, survey progress, and
route planning.  This module owns the independent perception gates that feed
that transaction: producer morphology-contract validation, track morphology,
static-map plausibility, validated scan-visibility receipts, and an
evidence-only cross-viewpoint reconciliation report plus its bounded
registry application.  It never publishes motion, changes survey progress,
or uses the expected stand count to rank candidates.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from scripts.aufgabe04.navigation.foundation.content_hashed_evidence import payload_sha256
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.coverage.coverage_candidate_reconciliation import (
    CoverageCandidateReconciliationConfig,
)
from scripts.aufgabe04.navigation.coverage.coverage_candidate_reconciliation_application import (
    CoverageCandidateReconciliationApplication,
    apply_negative_visibility_reconciliation_report,
)
from scripts.aufgabe04.navigation.coverage.coverage_candidate_reconciliation_report import (
    CoverageCandidateReconciliationReport,
    build_coverage_candidate_reconciliation_report,
)
from scripts.aufgabe04.navigation.coverage.coverage_visibility_reporting import (
    CoverageVisibilityEvidence,
    coverage_visibility_epoch_fields,
    validate_coverage_visibility_evidence,
)
from scripts.aufgabe04.navigation.planning.map_io import FrozenMapBundle, OccupancyGrid
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.missions.plan_first_detected_station import (
    validate_observation_provenance,
)
from scripts.aufgabe04.navigation.coverage.stand_candidate_static_map_admission import (
    StandCandidateStaticMapAdmission,
    evaluate_stand_candidate_static_map_admission,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    CoverageSurveyPlan,
    CoverageSurveyProgress,
    StandSurveyRegistry,
    SurveyViewpoint,
    coverage_survey_plan_sha256,
)
from scripts.aufgabe04.perception.lidar_stand_morphology import (
    MORPHOLOGY_PROFILE_EVIDENCE_KEY,
    MORPHOLOGY_PROFILE_SHA256_KEY,
    PROPOSAL_DETECTOR_CONFIG_EVIDENCE_KEY,
    StandMorphologyAdmission,
    evaluate_stand_morphology_admission,
    stand_width_profile_from_radius,
    validated_broad_proposal_width_bounds,
)
from scripts.aufgabe04.perception.lidar_visibility_evidence import (
    VISIBILITY_EVIDENCE_ENABLED_KEY,
)
from scripts.aufgabe04.perception.stand_confirmation import (
    ConfirmedStand,
    StandConfirmationAccumulator,
    StandConfirmationConfig,
)
from scripts.aufgabe04.perception.stand_observation import (
    StandObservation,
    load_observation_jsonl,
    validated_observation_stream_clock,
)


@dataclass(frozen=True)
class ContentHashedAdmissionArtifact:
    """One immutable content-addressed JSON artifact awaiting publication."""

    kind: str
    path: Path
    payload: dict[str, object]
    sha256: str
    hash_field: str

    def validated(self) -> "ContentHashedAdmissionArtifact":
        if not self.kind:
            raise ValueError("admission artifact kind must be non-empty")
        if not self.hash_field:
            raise ValueError("admission artifact hash field must be non-empty")
        if self.hash_field in self.payload:
            raise ValueError("unhashed admission payload contains its hash field")
        if payload_sha256(self.payload) != self.sha256:
            raise ValueError("admission artifact payload hash mismatch")
        if self.sha256 not in self.path.name:
            raise ValueError("admission artifact path is not content-addressed")
        return self


@dataclass(frozen=True)
class CoverageEpochPerceptionAdmission:
    """Independent perception decisions for the current stopped epoch."""

    raw_stands: tuple[ConfirmedStand, ...]
    morphology_admission: StandMorphologyAdmission
    static_map_admission: StandCandidateStaticMapAdmission
    visibility_evidence: CoverageVisibilityEvidence | None
    morphology_artifact: ContentHashedAdmissionArtifact
    static_map_artifact: ContentHashedAdmissionArtifact

    @property
    def admitted_stands(self) -> tuple[ConfirmedStand, ...]:
        return self.static_map_admission.admitted_stands

    @property
    def camera_population_stands(self) -> tuple[ConfirmedStand, ...]:
        return self.static_map_admission.population_retained_stands

    @property
    def registry_population_stands(self) -> tuple[ConfirmedStand, ...]:
        """Return candidates eligible for fusion, never motion authority."""

        return self.static_map_admission.population_retained_stands

    @property
    def registry_static_map_dispositions(self) -> dict[str, str]:
        """Bind each fusion input to its strict or boundary disposition."""

        return self.static_map_admission.disposition_by_stand_id

    @property
    def evidence_artifacts(self) -> tuple[ContentHashedAdmissionArtifact, ...]:
        return (self.morphology_artifact, self.static_map_artifact)


@dataclass(frozen=True)
class CoverageVisibilityReconciliationAdmission:
    """Reconciliation report, registry application, and immutable artifacts."""

    report: CoverageCandidateReconciliationReport
    application: CoverageCandidateReconciliationApplication
    updated_registry: StandSurveyRegistry
    included_viewpoint_ids: tuple[str, ...]
    artifact: ContentHashedAdmissionArtifact
    application_artifact: ContentHashedAdmissionArtifact

    @property
    def evidence_artifacts(self) -> tuple[ContentHashedAdmissionArtifact, ...]:
        return (self.artifact, self.application_artifact)


def load_stopped_observer_summary(path: Path) -> dict[str, object]:
    """Load the minimal fail-closed contract for a stopped LiDAR observer."""

    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid observer summary: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("observer summary must contain a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported observer summary schema")
    if payload.get("motion_published") is not False:
        raise ValueError("observer summary must declare motion_published=false")
    processed = payload.get("processed_scan_count")
    if type(processed) is not int or processed <= 0:
        raise ValueError("observer summary contains no processed scans")
    return payload


def observer_scan_pose(summary: Mapping[str, object]) -> Pose2D:
    """Decode the final exact-time scan-frame pose from an observer summary."""

    raw = summary.get("scan_frame_pose_in_planning_frame")
    if not isinstance(raw, dict):
        raise ValueError("observer summary has no final scan-frame pose")
    try:
        pose = Pose2D(
            float(raw["x_m"]),
            float(raw["y_m"]),
            float(raw["yaw_rad"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("observer summary scan-frame pose is invalid") from exc
    if not all(
        math.isfinite(value) for value in (pose.x_m, pose.y_m, pose.yaw_rad)
    ):
        raise ValueError("observer summary scan-frame pose must be finite")
    return pose


def load_validated_epoch_observations(
    *,
    summary: Mapping[str, object],
    observations_path: Path,
    map_yaml: Path,
    map_bundle: FrozenMapBundle,
    plan: CoverageSurveyPlan,
) -> tuple[StandObservation, ...]:
    """Load and provenance-check one stopped observation stream."""

    accepted_count = summary.get("accepted_observation_count")
    if type(accepted_count) is not int or accepted_count < 0:
        raise ValueError("observer summary accepted_observation_count is invalid")
    if accepted_count == 0:
        if observations_path.exists() and observations_path.stat().st_size > 0:
            raise ValueError(
                "observer summary reports no observations but JSONL is non-empty"
            )
        return ()
    if not observations_path.exists():
        raise ValueError("observer summary reports observations but JSONL is missing")
    observations = load_observation_jsonl(observations_path)
    if len(observations) != accepted_count:
        raise ValueError(
            "observer summary/JSONL observation count mismatch: "
            f"{accepted_count} != {len(observations)}"
        )
    runtime = summary.get("runtime_config")
    timing = summary.get("timing_limits")
    if not isinstance(runtime, dict) or not isinstance(timing, dict):
        raise ValueError("observer summary runtime/timing metadata is invalid")
    required_base_frame = str(runtime.get("base_frame", ""))
    required_localization_source = str(runtime.get("localization_source", ""))
    for observation in observations:
        validate_observation_provenance(
            observation,
            map_yaml=map_yaml,
            required_map_frame=plan.planning_frame,
            required_base_frame=required_base_frame,
            required_localization_source=required_localization_source,
            max_tf_age_sec=float(timing["max_tf_age_sec"]),
            max_scan_age_sec=float(timing["max_scan_age_sec"]),
            max_future_timestamp_sec=float(timing["max_future_timestamp_sec"]),
            max_tf_scan_skew_sec=float(timing["max_tf_scan_skew_sec"]),
            expected_map_yaml_sha256=map_bundle.yaml_sha256,
            expected_map_bundle_sha256=map_bundle.bundle_sha256,
        )
    validated_observation_stream_clock(observations)
    return observations


def build_confirmed_epoch_stands(
    observations: Sequence[StandObservation],
    plan: CoverageSurveyPlan,
) -> tuple[ConfirmedStand, ...]:
    """Confirm independent stand tracks inside one bounded LiDAR epoch."""

    if not observations:
        return ()
    accumulator = StandConfirmationAccumulator(
        config=StandConfirmationConfig(
            merge_distance_m=plan.config.candidate_merge_distance_m,
            min_hits=plan.config.minimum_candidate_hits,
            max_age_sec=plan.config.observation_epoch_max_age_sec,
            min_confidence=plan.config.minimum_candidate_confidence,
            min_boundary_clearance_m=plan.config.minimum_boundary_clearance_m,
        ),
        arena_bounds=plan.arena_bounds,
    )
    return accumulator.add_observations(observations)


def validate_observer_morphology_contract(
    summary: Mapping[str, object],
    plan: CoverageSurveyPlan,
) -> dict[str, object] | None:
    """Bind exact-two fusion to the producer's uncensored width evidence."""

    profile = stand_width_profile_from_radius(plan.config.candidate_radius_m)
    expected_profile = profile.to_evidence_dict()
    expected_sha256 = payload_sha256(expected_profile)
    reported_profile = summary.get(MORPHOLOGY_PROFILE_EVIDENCE_KEY)
    reported_sha256 = summary.get(MORPHOLOGY_PROFILE_SHA256_KEY)
    detector_config = summary.get(PROPOSAL_DETECTOR_CONFIG_EVIDENCE_KEY)
    fields_present = any(
        value is not None
        for value in (reported_profile, reported_sha256, detector_config)
    )
    required = plan.config.exact_inspection_point_count == 2
    if not fields_present and not required:
        return None
    if not isinstance(reported_profile, dict):
        raise ValueError("observer summary has no LiDAR morphology profile")
    if reported_profile != expected_profile:
        raise ValueError("observer LiDAR morphology profile differs from plan")
    if reported_sha256 != expected_sha256:
        raise ValueError("observer LiDAR morphology profile hash mismatch")
    if not isinstance(detector_config, dict):
        raise ValueError("observer summary has no LiDAR proposal detector config")
    try:
        proposal_bounds = validated_broad_proposal_width_bounds(
            profile=profile,
            proposal_min_width_m=float(detector_config["min_width_m"]),
            proposal_max_width_m=float(detector_config["max_width_m"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"observer LiDAR proposal detector would censor morphology evidence: {exc}"
        ) from exc
    return {
        "observer_morphology_profile_sha256": expected_sha256,
        "proposal_detector_config": detector_config,
        "proposal_width_evidence_preservation": proposal_bounds,
    }


def prepare_coverage_epoch_perception_admission(
    *,
    survey_root: Path,
    observer_summary_json: Path,
    observer_summary: Mapping[str, object],
    plan: CoverageSurveyPlan,
    viewpoint: SurveyViewpoint,
    occupancy_grid: OccupancyGrid,
    observations: Sequence[StandObservation],
    raw_stands: Sequence[ConfirmedStand],
) -> CoverageEpochPerceptionAdmission:
    """Validate and prepare all current-epoch perception evidence."""

    survey_root = Path(survey_root)
    observer_summary_json = Path(observer_summary_json)
    observation_snapshot = tuple(observations)
    stand_snapshot = tuple(raw_stands)
    observer_contract = validate_observer_morphology_contract(
        observer_summary,
        plan,
    )
    visibility_evidence = _validate_current_visibility_evidence(
        observer_summary=observer_summary,
        observer_summary_json=observer_summary_json,
        plan=plan,
        viewpoint=viewpoint,
    )

    morphology_admission = evaluate_stand_morphology_admission(
        stand_snapshot,
        observation_snapshot,
        profile=stand_width_profile_from_radius(plan.config.candidate_radius_m),
    )
    morphology_payload = {
        **morphology_admission.to_evidence_dict(),
        "survey_id": plan.survey_id,
        "viewpoint_id": viewpoint.viewpoint_id,
        "planning_frame": plan.planning_frame,
        "map_bundle_sha256": plan.map_bundle_sha256,
        "coverage_plan_sha256": coverage_survey_plan_sha256(plan),
        "observer_contract": observer_contract,
    }
    morphology_artifact = _content_hashed_artifact(
        kind="lidar_morphology_admission",
        survey_root=survey_root,
        viewpoint_id=viewpoint.viewpoint_id,
        filename_label="lidar_morphology_admission",
        payload=morphology_payload,
        hash_field="lidar_morphology_admission_sha256",
    )

    static_map_admission = evaluate_stand_candidate_static_map_admission(
        Costmap.from_occupancy_grid(occupancy_grid).with_arena_bounds(
            plan.arena_bounds
        ),
        morphology_admission.admitted_stands,
        candidate_radius_m=plan.config.candidate_radius_m,
        candidate_uncertainty_m=plan.config.candidate_uncertainty_m,
    )
    static_map_payload = {
        **static_map_admission.to_evidence_dict(),
        "survey_id": plan.survey_id,
        "viewpoint_id": viewpoint.viewpoint_id,
        "planning_frame": plan.planning_frame,
        "map_bundle_sha256": plan.map_bundle_sha256,
        "coverage_plan_sha256": coverage_survey_plan_sha256(plan),
    }
    static_map_artifact = _content_hashed_artifact(
        kind="static_map_candidate_admission",
        survey_root=survey_root,
        viewpoint_id=viewpoint.viewpoint_id,
        filename_label="static_map_candidate_admission",
        payload=static_map_payload,
        hash_field="static_map_candidate_admission_sha256",
    )
    return CoverageEpochPerceptionAdmission(
        raw_stands=stand_snapshot,
        morphology_admission=morphology_admission,
        static_map_admission=static_map_admission,
        visibility_evidence=visibility_evidence,
        morphology_artifact=morphology_artifact,
        static_map_artifact=static_map_artifact,
    )


def prepare_coverage_visibility_reconciliation(
    *,
    survey_root: Path,
    plan: CoverageSurveyPlan,
    prior_progress: CoverageSurveyProgress,
    completed_progress: CoverageSurveyProgress,
    current_viewpoint_id: str,
    current_evidence: CoverageVisibilityEvidence | None,
    registry: StandSurveyRegistry,
    occupancy_grid: OccupancyGrid,
) -> CoverageVisibilityReconciliationAdmission | None:
    """Build a report and apply it only at terminal exact-two coverage."""

    if current_evidence is None:
        return None
    visibility_epochs = _load_validated_visibility_epochs(
        survey_root=Path(survey_root),
        plan=plan,
        progress=prior_progress,
        current_viewpoint_id=current_viewpoint_id,
        current_evidence=current_evidence,
    )
    observer_config_sha256s = {
        evidence.observer_config_sha256 for evidence in visibility_epochs
    }
    if len(observer_config_sha256s) != 1:
        raise ValueError(
            "coverage visibility epochs use different observer configurations"
        )
    report = build_coverage_candidate_reconciliation_report(
        plan=plan,
        registry=registry,
        occupancy_grid=occupancy_grid,
        receipts=tuple(
            receipt
            for evidence in visibility_epochs
            for receipt in evidence.receipts
        ),
        config=CoverageCandidateReconciliationConfig(
            observer_config_sha256=next(iter(observer_config_sha256s))
        ),
    )
    included_viewpoint_ids = tuple(
        evidence.viewpoint_id for evidence in visibility_epochs
    )
    updated_registry, application = (
        apply_negative_visibility_reconciliation_report(
            plan=plan,
            progress=completed_progress,
            registry=registry,
            report=report,
            included_viewpoint_ids=included_viewpoint_ids,
        )
    )
    payload = {
        **report.to_evidence_dict(),
        "recorded_viewpoint_id": current_viewpoint_id,
        "included_viewpoint_ids": list(included_viewpoint_ids),
    }
    artifact = _content_hashed_artifact(
        kind="lidar_visibility_reconciliation",
        survey_root=Path(survey_root),
        viewpoint_id=current_viewpoint_id,
        filename_label="lidar_visibility_reconciliation",
        payload=payload,
        hash_field="lidar_visibility_reconciliation_sha256",
    )
    application_payload = {
        **application.to_evidence_dict(),
        "recorded_viewpoint_id": current_viewpoint_id,
        "included_viewpoint_ids": list(included_viewpoint_ids),
        "lidar_visibility_reconciliation_sha256": artifact.sha256,
    }
    application_artifact = _content_hashed_artifact(
        kind="lidar_visibility_reconciliation_application",
        survey_root=Path(survey_root),
        viewpoint_id=current_viewpoint_id,
        filename_label="lidar_visibility_reconciliation_application",
        payload=application_payload,
        hash_field="lidar_visibility_reconciliation_application_sha256",
    )
    return CoverageVisibilityReconciliationAdmission(
        report=report,
        application=application,
        updated_registry=updated_registry,
        included_viewpoint_ids=included_viewpoint_ids,
        artifact=artifact,
        application_artifact=application_artifact,
    )


def coverage_stop_perception_summary_fields(
    epoch_admission: CoverageEpochPerceptionAdmission,
    reconciliation: CoverageVisibilityReconciliationAdmission | None,
) -> dict[str, object]:
    """Return compact, JSON-safe bindings shared by epoch and summary."""

    fields = {
        **coverage_visibility_epoch_fields(epoch_admission.visibility_evidence),
        "lidar_morphology_admission_json": str(
            epoch_admission.morphology_artifact.path
        ),
        "lidar_morphology_admission_sha256": (
            epoch_admission.morphology_artifact.sha256
        ),
        "static_map_candidate_admission_json": str(
            epoch_admission.static_map_artifact.path
        ),
        "static_map_candidate_admission_sha256": (
            epoch_admission.static_map_artifact.sha256
        ),
    }
    if reconciliation is None:
        return {
            **fields,
            "lidar_visibility_reconciliation_available": False,
            "lidar_visibility_reconciliation_json": None,
            "lidar_visibility_reconciliation_sha256": None,
            "lidar_visibility_reconciliation_policy_mode": None,
            "lidar_visibility_registry_mutation_applied": False,
            "lidar_visibility_reconciliation_application_json": None,
            "lidar_visibility_reconciliation_application_sha256": None,
            "lidar_visibility_terminal_application_eligible": False,
            "lidar_visibility_application_reasons": [],
            "lidar_visibility_recommended_rejection_candidate_uids": [],
            "lidar_visibility_rejected_candidate_uids": [],
        }
    report = reconciliation.report
    application = reconciliation.application
    return {
        **fields,
        "lidar_visibility_reconciliation_available": True,
        "lidar_visibility_reconciliation_json": str(
            reconciliation.artifact.path
        ),
        "lidar_visibility_reconciliation_sha256": reconciliation.artifact.sha256,
        "lidar_visibility_reconciliation_policy_mode": application.policy_mode,
        "lidar_visibility_registry_mutation_applied": (
            application.registry_mutation_applied
        ),
        "lidar_visibility_reconciliation_application_json": str(
            reconciliation.application_artifact.path
        ),
        "lidar_visibility_reconciliation_application_sha256": (
            reconciliation.application_artifact.sha256
        ),
        "lidar_visibility_terminal_application_eligible": (
            application.terminal_application_eligible
        ),
        "lidar_visibility_application_reasons": list(
            application.application_reasons
        ),
        "lidar_visibility_recommended_rejection_candidate_uids": list(
            report.recommended_negative_visibility_candidate_uids
        ),
        "lidar_visibility_rejected_candidate_uids": list(
            application.rejected_candidate_uids
        ),
        "lidar_visibility_retained_provisional_candidate_uids": list(
            application.retained_provisional_candidate_uids
        ),
    }


def _validate_current_visibility_evidence(
    *,
    observer_summary: Mapping[str, object],
    observer_summary_json: Path,
    plan: CoverageSurveyPlan,
    viewpoint: SurveyViewpoint,
) -> CoverageVisibilityEvidence | None:
    required = plan.config.exact_inspection_point_count == 2
    evidence = None
    if required or VISIBILITY_EVIDENCE_ENABLED_KEY in observer_summary:
        evidence = validate_coverage_visibility_evidence(
            observer_summary,
            plan,
            viewpoint.viewpoint_id,
            required,
        )
    if evidence is None:
        return None
    if (
        evidence.receipts_jsonl.resolve().parent
        != observer_summary_json.resolve().parent
    ):
        raise ValueError(
            "visibility receipt JSONL must share the observer-summary epoch directory"
        )
    observer_config = evidence.observer_config
    if (
        observer_config.get(MORPHOLOGY_PROFILE_EVIDENCE_KEY)
        != observer_summary.get(MORPHOLOGY_PROFILE_EVIDENCE_KEY)
        or observer_config.get(PROPOSAL_DETECTOR_CONFIG_EVIDENCE_KEY)
        != observer_summary.get(PROPOSAL_DETECTOR_CONFIG_EVIDENCE_KEY)
    ):
        raise ValueError(
            "visibility observer config differs from morphology/proposal summary"
        )
    return evidence


def _load_validated_visibility_epochs(
    *,
    survey_root: Path,
    plan: CoverageSurveyPlan,
    progress: CoverageSurveyProgress,
    current_viewpoint_id: str,
    current_evidence: CoverageVisibilityEvidence,
) -> tuple[CoverageVisibilityEvidence, ...]:
    evidence_by_viewpoint = {current_viewpoint_id: current_evidence}
    for viewpoint_id in progress.visited_viewpoint_ids:
        epoch_path = survey_root / "epochs" / f"{viewpoint_id}.json"
        if epoch_path.is_symlink() or not epoch_path.is_file():
            raise ValueError(f"prior coverage epoch is unavailable: {epoch_path}")
        try:
            epoch = json.loads(epoch_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid prior coverage epoch: {exc}") from exc
        if (
            not isinstance(epoch, dict)
            or epoch.get("survey_id") != plan.survey_id
            or epoch.get("viewpoint_id") != viewpoint_id
        ):
            raise ValueError("prior coverage epoch identity differs from plan")
        prior_summary_path = Path(str(epoch.get("observer_summary_json", "")))
        prior_summary = load_stopped_observer_summary(prior_summary_path)
        evidence = validate_coverage_visibility_evidence(
            prior_summary,
            plan,
            viewpoint_id,
            True,
        )
        if evidence is None:  # Defensive: required=True already fails closed.
            raise ValueError("prior required visibility evidence is unavailable")
        expected_fields = coverage_visibility_epoch_fields(evidence)
        if any(epoch.get(key) != value for key, value in expected_fields.items()):
            raise ValueError(
                "prior epoch visibility fields differ from observer evidence"
            )
        evidence_by_viewpoint[viewpoint_id] = evidence
    return tuple(
        evidence_by_viewpoint[viewpoint.viewpoint_id]
        for viewpoint in plan.viewpoints
        if viewpoint.viewpoint_id in evidence_by_viewpoint
    )


def _content_hashed_artifact(
    *,
    kind: str,
    survey_root: Path,
    viewpoint_id: str,
    filename_label: str,
    payload: dict[str, object],
    hash_field: str,
) -> ContentHashedAdmissionArtifact:
    sha256 = payload_sha256(payload)
    artifact = ContentHashedAdmissionArtifact(
        kind=kind,
        path=(
            survey_root
            / "epochs"
            / f"{viewpoint_id}_{filename_label}_{sha256}.json"
        ),
        payload=payload,
        sha256=sha256,
        hash_field=hash_field,
    )
    return artifact.validated()


__all__ = [
    "ContentHashedAdmissionArtifact",
    "CoverageEpochPerceptionAdmission",
    "CoverageVisibilityReconciliationAdmission",
    "build_confirmed_epoch_stands",
    "coverage_stop_perception_summary_fields",
    "load_validated_epoch_observations",
    "load_stopped_observer_summary",
    "observer_scan_pose",
    "prepare_coverage_epoch_perception_admission",
    "prepare_coverage_visibility_reconciliation",
    "validate_observer_morphology_contract",
]
