"""Pure exact-two LiDAR to camera-population admission facade.

Core classification and the standard ``CandidateSnapshot`` projection live
here. Typed invariants are isolated in ``exact_two_camera_contract`` and
strict persistence/live binding in ``exact_two_camera_artifacts``. All three
modules are ROS-free and never mutate the survey registry or authorize motion.
"""

from __future__ import annotations

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.coverage.coverage_candidate_lifecycle import (
    ExactTwoLidarCheckpointDecision,
    classify_coverage_candidates,
    evaluate_exact_two_lidar_checkpoint,
    exact_two_lidar_checkpoint_evidence_sha256,
)
from scripts.aufgabe04.navigation.coverage.coverage_candidate_reconciliation_report import (
    evidence_only_reconciliation_policy_contract,
)
from scripts.aufgabe04.navigation.approach.exact_two_camera_artifacts import (
    ADMISSION_HASH_FIELD,
    HANDOFF_HASH_FIELD,
    exact_two_camera_admission_evidence,
    exact_two_camera_admission_payload,
    exact_two_camera_admission_sha256,
    exact_two_camera_handoff_payload,
    exact_two_camera_handoff_sha256,
    load_bound_exact_two_candidate_snapshot,
    load_exact_two_camera_admission,
    load_exact_two_camera_handoff,
    new_exact_two_camera_handoff,
    require_handoff_candidate_support,
    validate_exact_two_camera_handoff,
    validate_live_candidate_snapshot_binding,
    validate_live_registry_binding,
    write_exact_two_camera_admission,
    write_exact_two_camera_handoff,
)
from scripts.aufgabe04.navigation.approach.exact_two_camera_contract import (
    EXACT_TWO_CAMERA_ADMISSION_SCHEMA_VERSION,
    EXACT_TWO_CAMERA_HANDOFF_SCHEMA_VERSION,
    SOURCE_KIND_MULTI_VIEW,
    SOURCE_KIND_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION,
    SUPPORT_CLASS_MULTI_VIEW,
    SUPPORT_CLASS_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION,
    VALID_EXACT_TWO_CAMERA_SUPPORT_CLASSES,
    ExactTwoCameraAdmissionDecision,
    ExactTwoCameraAdmissionError,
    ExactTwoCameraCandidateEvidence,
    ExactTwoCameraHandoffArtifact,
    finite_nonnegative,
    require_admitted_candidate_support,
    required_source_kind,
    stand_survey_registry_sha256,
    validate_exact_two_camera_admission,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    STATUS_PENDING_CAMERA,
    STATUS_PROVISIONAL,
    CoverageSurveyPlan,
    CoverageSurveyProgress,
    StandSurveyRegistry,
    coverage_survey_plan_sha256,
    validate_stand_survey_registry,
)
from scripts.aufgabe04.perception.lidar_stand_morphology import (
    stand_width_profile_from_radius,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateGeometry,
    CandidateSnapshot,
    CandidateSnapshotError,
    CandidateSource,
    FrozenCandidate,
    new_candidate_snapshot,
)


def evaluate_exact_two_camera_admission(
    plan: CoverageSurveyPlan,
    progress: CoverageSurveyProgress,
    registry: StandSurveyRegistry,
    lidar_checkpoint: ExactTwoLidarCheckpointDecision,
) -> ExactTwoCameraAdmissionDecision:
    """Classify the camera population after re-evaluating the checkpoint."""

    if not isinstance(lidar_checkpoint, ExactTwoLidarCheckpointDecision):
        raise ExactTwoCameraAdmissionError(
            "invalid_checkpoint",
            "lidar_checkpoint must be an ExactTwoLidarCheckpointDecision",
        )
    try:
        expected_checkpoint = evaluate_exact_two_lidar_checkpoint(
            plan, progress, registry
        )
    except ValueError as exc:
        raise ExactTwoCameraAdmissionError(
            "invalid_input", f"invalid exact-two survey snapshots: {exc}"
        ) from exc
    if lidar_checkpoint != expected_checkpoint:
        raise ExactTwoCameraAdmissionError(
            "provenance_mismatch",
            "LiDAR checkpoint does not match the live plan/progress/registry",
        )

    population = classify_coverage_candidates(plan, registry)
    evidence = tuple(
        _camera_evidence(candidate)
        for candidate in population.candidates
        if candidate.active_lidar
    )
    multi_view = tuple(
        item.candidate_uid
        for item in evidence
        if item.support_class == SUPPORT_CLASS_MULTI_VIEW and item.admissible
    )
    single_view = tuple(
        item.candidate_uid
        for item in evidence
        if item.support_class
        == SUPPORT_CLASS_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION
        and item.admissible
    )
    blocked = tuple(item.candidate_uid for item in evidence if not item.admissible)
    expected_count = plan.config.expected_stand_count
    active_count = len(evidence)

    reasons: list[str] = []
    if not lidar_checkpoint.ready:
        reasons.append("lidar_checkpoint_not_ready")
    if expected_count is None:
        reasons.append("expected_stand_count_unset")
    elif active_count != expected_count:
        reasons.append("active_candidate_count_mismatch")
    if blocked:
        reasons.append("active_candidates_not_camera_admissible")
    if len(multi_view) + len(single_view) != active_count:
        reasons.append("camera_support_partition_incomplete")

    decision = ExactTwoCameraAdmissionDecision(
        schema_version=EXACT_TWO_CAMERA_ADMISSION_SCHEMA_VERSION,
        survey_id=plan.survey_id,
        planning_frame=plan.planning_frame,
        map_bundle_sha256=plan.map_bundle_sha256,
        plan_sha256=coverage_survey_plan_sha256(plan),
        progress_snapshot_sha256=lidar_checkpoint.progress_snapshot_sha256,
        source_registry_sha256=stand_survey_registry_sha256(registry),
        lidar_checkpoint_sha256=exact_two_lidar_checkpoint_evidence_sha256(
            lidar_checkpoint
        ),
        ready=not reasons,
        reasons=tuple(reasons),
        camera_population_ready=not reasons,
        motion_authorized=False,
        expected_stand_count=expected_count,
        active_candidate_count=active_count,
        multi_view_candidate_uids=multi_view,
        single_view_candidate_uids=single_view,
        blocked_candidate_uids=blocked,
        candidate_evidence=evidence,
    )
    validate_exact_two_camera_admission(decision)
    return decision


def build_exact_two_camera_candidate_snapshot(
    plan: CoverageSurveyPlan,
    registry: StandSurveyRegistry,
    admission: ExactTwoCameraAdmissionDecision,
    *,
    snapshot_id: str,
    created_unix_sec: float | None = None,
) -> CandidateSnapshot:
    """Project admitted UIDs into the repository's standard frozen snapshot."""

    _validate_admission_against_plan_registry(admission, plan, registry)
    if not admission.ready:
        raise ExactTwoCameraAdmissionError(
            "camera_population_not_ready",
            "cannot build a candidate snapshot from a not-ready admission",
        )
    candidates_by_uid = {
        candidate.candidate_uid: candidate for candidate in registry.candidates
    }
    selected = tuple(
        candidates_by_uid[uid] for uid in admission.admitted_candidate_uids
    )
    created = (
        max(candidate.last_seen_sec for candidate in selected)
        if created_unix_sec is None
        else finite_nonnegative(created_unix_sec, "created_unix_sec")
    )
    detector_hash = exact_two_detector_config_sha256(plan)
    try:
        return new_candidate_snapshot(
            snapshot_id=snapshot_id,
            created_unix_sec=created,
            planning_frame=plan.planning_frame,
            map_bundle_sha256=plan.map_bundle_sha256,
            candidates=(
                FrozenCandidate(
                    candidate_uid=candidate.candidate_uid,
                    geometry=CandidateGeometry(
                        x_m=candidate.x_m,
                        y_m=candidate.y_m,
                        radius_m=candidate.radius_m,
                        uncertainty_m=candidate.uncertainty_m,
                        keepout_radius_m=candidate.keepout_radius_m,
                    ),
                    source=CandidateSource(
                        source_kind=required_source_kind(
                            require_admitted_candidate_support(
                                admission, candidate.candidate_uid
                            ).support_class
                        ),
                        source_artifact_sha256=admission.source_registry_sha256,
                        detector_config_sha256=detector_hash,
                        observation_ids=tuple(
                            sorted(set(candidate.source_observation_ids))
                        ),
                    ),
                    confidence=candidate.confidence,
                    hit_count=candidate.hit_count,
                    first_seen_sec=candidate.first_seen_sec,
                    last_seen_sec=candidate.last_seen_sec,
                )
                for candidate in selected
            ),
        )
    except CandidateSnapshotError as exc:
        raise ExactTwoCameraAdmissionError(exc.code, str(exc)) from exc


def exact_two_detector_config_sha256(plan: CoverageSurveyPlan) -> str:
    if not isinstance(plan, CoverageSurveyPlan):
        raise ExactTwoCameraAdmissionError(
            "invalid_input", "plan must be a CoverageSurveyPlan"
        )
    try:
        plan_sha256 = coverage_survey_plan_sha256(plan)
    except ValueError as exc:
        raise ExactTwoCameraAdmissionError("invalid_input", str(exc)) from exc
    return payload_sha256(
        {
            "source": "stand_coverage_survey/exact_two_camera_admission",
            "plan_sha256": plan_sha256,
            "config": {
                "candidate_merge_distance_m": (
                    plan.config.candidate_merge_distance_m
                ),
                "minimum_candidate_confidence": (
                    plan.config.minimum_candidate_confidence
                ),
                "minimum_distinct_viewpoints": (
                    plan.config.minimum_distinct_viewpoints
                ),
                "minimum_candidate_hits": plan.config.minimum_candidate_hits,
                "exact_inspection_point_count": (
                    plan.config.exact_inspection_point_count
                ),
                "lidar_track_morphology_profile": (
                    stand_width_profile_from_radius(
                        plan.config.candidate_radius_m
                    ).to_evidence_dict()
                ),
                "lidar_visibility_reconciliation": (
                    evidence_only_reconciliation_policy_contract()
                ),
            },
        }
    )


def _camera_evidence(candidate) -> ExactTwoCameraCandidateEvidence:
    reasons: list[str] = []
    support_class: str | None = None
    source_kind: str | None = None
    if not candidate.lidar_static_map_admitted:
        reasons.append("static_map_admission_missing")
    if not candidate.basic_lidar_support:
        reasons.extend(candidate.support_reasons)

    multi_view = (
        candidate.registry_status == STATUS_PENDING_CAMERA
        and candidate.camera_validation_queued
        and candidate.multi_view_supported
        and candidate.distinct_known_viewpoint_count >= 2
    )
    single_view = (
        candidate.registry_status == STATUS_PROVISIONAL
        and candidate.distinct_known_viewpoint_count == 1
        and candidate.viewpoint_ids_distinct
        and not candidate.unknown_viewpoint_ids
    )
    if multi_view:
        support_class = SUPPORT_CLASS_MULTI_VIEW
        source_kind = SOURCE_KIND_MULTI_VIEW
    elif single_view:
        support_class = SUPPORT_CLASS_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION
        source_kind = SOURCE_KIND_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION
    elif candidate.registry_status not in {STATUS_PENDING_CAMERA, STATUS_PROVISIONAL}:
        reasons.append("registry_status_not_camera_admissible")
    elif candidate.registry_status == STATUS_PENDING_CAMERA:
        reasons.append("pending_camera_multi_view_support_missing")
    else:
        reasons.append("provisional_candidate_not_single_view")

    admissible = (
        candidate.active_lidar
        and candidate.lidar_static_map_admitted
        and candidate.basic_lidar_support
        and support_class is not None
        and not reasons
    )
    return ExactTwoCameraCandidateEvidence(
        candidate_uid=candidate.candidate_uid,
        registry_status=candidate.registry_status,
        active_lidar=candidate.active_lidar,
        static_map_admitted=candidate.lidar_static_map_admitted,
        basic_lidar_supported=candidate.basic_lidar_support,
        confidence=candidate.confidence,
        minimum_confidence=candidate.minimum_confidence,
        confidence_supported=candidate.confidence_supported,
        hit_count=candidate.hit_count,
        minimum_hit_count=candidate.minimum_hit_count,
        hit_count_supported=candidate.hit_count_supported,
        viewpoint_ids=candidate.viewpoint_ids,
        known_viewpoint_ids=candidate.known_viewpoint_ids,
        unknown_viewpoint_ids=candidate.unknown_viewpoint_ids,
        viewpoint_ids_distinct=candidate.viewpoint_ids_distinct,
        distinct_known_viewpoint_count=candidate.distinct_known_viewpoint_count,
        support_class=support_class,
        source_kind=source_kind,
        admissible=admissible,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _validate_admission_against_plan_registry(
    admission: ExactTwoCameraAdmissionDecision,
    plan: CoverageSurveyPlan,
    registry: StandSurveyRegistry,
) -> None:
    validate_exact_two_camera_admission(admission)
    try:
        validate_stand_survey_registry(registry, plan)
        plan_sha256 = coverage_survey_plan_sha256(plan)
    except ValueError as exc:
        raise ExactTwoCameraAdmissionError("invalid_input", str(exc)) from exc
    expected = {
        "survey_id": plan.survey_id,
        "planning_frame": plan.planning_frame,
        "map_bundle_sha256": plan.map_bundle_sha256,
        "plan_sha256": plan_sha256,
        "source_registry_sha256": stand_survey_registry_sha256(registry),
    }
    for field_name, value in expected.items():
        if getattr(admission, field_name) != value:
            raise ExactTwoCameraAdmissionError(
                "provenance_mismatch",
                f"admission {field_name} does not match plan/registry",
            )
    registry_uids = {candidate.candidate_uid for candidate in registry.candidates}
    if not set(admission.admitted_candidate_uids).issubset(registry_uids):
        raise ExactTwoCameraAdmissionError(
            "provenance_mismatch", "admission contains a missing registry candidate"
        )


__all__ = [
    "ADMISSION_HASH_FIELD",
    "EXACT_TWO_CAMERA_ADMISSION_SCHEMA_VERSION",
    "EXACT_TWO_CAMERA_HANDOFF_SCHEMA_VERSION",
    "HANDOFF_HASH_FIELD",
    "SOURCE_KIND_MULTI_VIEW",
    "SOURCE_KIND_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION",
    "SUPPORT_CLASS_MULTI_VIEW",
    "SUPPORT_CLASS_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION",
    "VALID_EXACT_TWO_CAMERA_SUPPORT_CLASSES",
    "ExactTwoCameraAdmissionDecision",
    "ExactTwoCameraAdmissionError",
    "ExactTwoCameraCandidateEvidence",
    "ExactTwoCameraHandoffArtifact",
    "build_exact_two_camera_candidate_snapshot",
    "evaluate_exact_two_camera_admission",
    "exact_two_camera_admission_evidence",
    "exact_two_camera_admission_payload",
    "exact_two_camera_admission_sha256",
    "exact_two_camera_handoff_payload",
    "exact_two_camera_handoff_sha256",
    "exact_two_detector_config_sha256",
    "load_bound_exact_two_candidate_snapshot",
    "load_exact_two_camera_admission",
    "load_exact_two_camera_handoff",
    "new_exact_two_camera_handoff",
    "require_admitted_candidate_support",
    "require_handoff_candidate_support",
    "stand_survey_registry_sha256",
    "validate_exact_two_camera_admission",
    "validate_exact_two_camera_handoff",
    "validate_live_candidate_snapshot_binding",
    "validate_live_registry_binding",
    "write_exact_two_camera_admission",
    "write_exact_two_camera_handoff",
]
