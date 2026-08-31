"""Terminal-only application of LiDAR negative-visibility recommendations.

The report remains a pure recommendation over a source registry snapshot.
This module validates its complete behavior contract and projects a new
registry only after an exact-two survey has committed evidence from every
planned viewpoint. It never authorizes motion or uses the expected count.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.coverage.coverage_candidate_reconciliation import (
    ACTION_REJECT_PROVISIONAL,
    ACTION_RETAIN,
    COVERAGE_CANDIDATE_RECONCILIATION_SCHEMA_VERSION,
    CoverageCandidateReconciliationConfig,
    CoverageCandidateReconciliationDecision,
)
from scripts.aufgabe04.navigation.coverage.coverage_candidate_reconciliation_report import (
    COVERAGE_CANDIDATE_RECONCILIATION_REPORT_SCHEMA_VERSION,
    POLICY_MODE_EVIDENCE_ONLY,
    CoverageCandidateReconciliationReport,
)
from scripts.aufgabe04.navigation.coverage.coverage_candidate_reconciliation_policy import (
    evaluate_negative_visibility_ray_policy,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    REJECTION_BASIS_NEGATIVE_VISIBILITY,
    STATUS_PROVISIONAL,
    STATUS_REJECTED,
    CoverageSurveyPlan,
    CoverageSurveyProgress,
    StandSurveyRegistry,
    coverage_survey_plan_sha256,
    stand_survey_registry_sha256,
    validate_stand_survey_registry,
    validate_survey_progress,
    visited_coverage_ratio,
)


COVERAGE_CANDIDATE_RECONCILIATION_APPLICATION_SCHEMA_VERSION = 1
POLICY_MODE_BOUNDED_NEGATIVE_VISIBILITY_REGISTRY_REJECTION = (
    "bounded_negative_visibility_registry_rejection"
)
_EPSILON = 1.0e-12


def bounded_negative_visibility_reconciliation_policy_contract() -> dict[str, object]:
    """Return the canonical behavior contract sealed by mission provenance."""

    values = CoverageCandidateReconciliationConfig(
        observer_config_sha256="0" * 64
    ).to_evidence_dict()
    del values["observer_config_sha256"]
    return {
        "policy_mode": (
            POLICY_MODE_BOUNDED_NEGATIVE_VISIBILITY_REGISTRY_REJECTION
        ),
        "recommendation_report_policy_mode": POLICY_MODE_EVIDENCE_ONLY,
        "reconciliation_config": values,
        "application_scope": "terminal_exact_two_full_viewpoint_receipt_set",
        "required_candidate_status": STATUS_PROVISIONAL,
        "rejection_basis": REJECTION_BASIS_NEGATIVE_VISIBILITY,
        "expected_stand_count_used": False,
        "registry_mutation_enabled": True,
        "motion_authorized": False,
    }


# Compatibility name for callers that do not need the policy-mode qualifier.
negative_visibility_reconciliation_policy_contract = (
    bounded_negative_visibility_reconciliation_policy_contract
)


@dataclass(frozen=True)
class CoverageCandidateReconciliationApplication:
    """Immutable receipt for one registry projection attempt."""

    schema_version: int
    survey_id: str
    planning_frame: str
    map_bundle_sha256: str
    plan_sha256: str
    progress_snapshot_sha256: str
    report_sha256: str
    source_registry_snapshot_sha256: str
    updated_registry_snapshot_sha256: str
    planned_viewpoint_ids: tuple[str, ...]
    visited_viewpoint_ids: tuple[str, ...]
    included_viewpoint_ids: tuple[str, ...]
    terminal_application_eligible: bool
    application_reasons: tuple[str, ...]
    recommended_candidate_uids: tuple[str, ...]
    rejected_candidate_uids: tuple[str, ...]
    unapplied_recommended_candidate_uids: tuple[str, ...]
    retained_provisional_candidate_uids: tuple[str, ...]
    registry_mutation_applied: bool
    policy_mode: str = (
        POLICY_MODE_BOUNDED_NEGATIVE_VISIBILITY_REGISTRY_REJECTION
    )
    motion_authorized: bool = False

    @property
    def application_sha256(self) -> str:
        return payload_sha256(self.to_evidence_dict())

    def to_evidence_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "survey": {
                "survey_id": self.survey_id,
                "planning_frame": self.planning_frame,
                "map_bundle_sha256": self.map_bundle_sha256,
                "plan_sha256": self.plan_sha256,
                "progress_snapshot_sha256": self.progress_snapshot_sha256,
            },
            "report_sha256": self.report_sha256,
            "source_registry_snapshot_sha256": (
                self.source_registry_snapshot_sha256
            ),
            "updated_registry_snapshot_sha256": (
                self.updated_registry_snapshot_sha256
            ),
            "viewpoints": {
                "planned": list(self.planned_viewpoint_ids),
                "visited": list(self.visited_viewpoint_ids),
                "included_receipt_epochs": list(self.included_viewpoint_ids),
            },
            "terminal_application_eligible": (
                self.terminal_application_eligible
            ),
            "application_reasons": list(self.application_reasons),
            "recommended_candidate_uids": list(
                self.recommended_candidate_uids
            ),
            "rejected_candidate_uids": list(self.rejected_candidate_uids),
            "unapplied_recommended_candidate_uids": list(
                self.unapplied_recommended_candidate_uids
            ),
            "retained_provisional_candidate_uids": list(
                self.retained_provisional_candidate_uids
            ),
            "rejection_basis": REJECTION_BASIS_NEGATIVE_VISIBILITY,
            "expected_stand_count_used": False,
            "registry_mutation_applied": self.registry_mutation_applied,
            "policy_mode": self.policy_mode,
            "motion_authorized": self.motion_authorized,
        }


def apply_negative_visibility_reconciliation_report(
    *,
    plan: CoverageSurveyPlan,
    progress: CoverageSurveyProgress,
    registry: StandSurveyRegistry,
    report: CoverageCandidateReconciliationReport,
    included_viewpoint_ids: tuple[str, ...],
) -> tuple[StandSurveyRegistry, CoverageCandidateReconciliationApplication]:
    """Return a terminal projection or an explicitly non-applied receipt."""

    validate_survey_progress(progress, plan)
    validate_stand_survey_registry(registry, plan)
    _validate_report_binding(plan=plan, registry=registry, report=report)
    if not isinstance(included_viewpoint_ids, tuple):
        raise ValueError("included_viewpoint_ids must be a tuple")
    if len(included_viewpoint_ids) != len(set(included_viewpoint_ids)):
        raise ValueError("included viewpoint IDs must be unique")
    if not set(included_viewpoint_ids).issubset(plan.viewpoint_ids):
        raise ValueError(
            "included visibility evidence references unknown viewpoint"
        )

    planned = plan.viewpoint_ids
    visited = tuple(
        viewpoint_id
        for viewpoint_id in planned
        if viewpoint_id in progress.visited_viewpoint_ids
    )
    included = tuple(
        viewpoint_id
        for viewpoint_id in planned
        if viewpoint_id in included_viewpoint_ids
    )
    if included != report.receipt_viewpoint_ids:
        raise ValueError(
            "included viewpoint IDs differ from report receipt viewpoints"
        )
    application_reasons: list[str] = []
    if plan.config.exact_inspection_point_count != 2:
        application_reasons.append("exact_two_reconciliation_required")
    if visited != planned:
        application_reasons.append("planned_viewpoints_incomplete")
    if included != planned:
        application_reasons.append("full_plan_visibility_receipts_missing")
    if (
        visited_coverage_ratio(plan, progress) + _EPSILON
        < plan.config.coverage_threshold
    ):
        application_reasons.append("visited_coverage_below_threshold")

    recommended = tuple(report.recommended_negative_visibility_candidate_uids)
    recommended_set = set(recommended)
    if len(recommended_set) != len(recommended):
        raise ValueError("reconciliation report recommended UIDs are not unique")
    decision_by_uid = {
        decision.candidate_uid: decision for decision in report.decisions
    }
    for uid in recommended:
        decision = decision_by_uid.get(uid)
        if decision is None or not decision.reject_provisional:
            raise ValueError(
                "reconciliation recommendation lacks reject decision"
            )

    terminal_eligible = not application_reasons
    updated_candidates = []
    rejected: list[str] = []
    for candidate in registry.candidates:
        if not terminal_eligible or candidate.candidate_uid not in recommended_set:
            updated_candidates.append(candidate)
            continue
        if candidate.status != STATUS_PROVISIONAL:
            raise ValueError(
                "negative-visibility reconciliation can reject only provisional "
                f"candidates: {candidate.candidate_uid}"
            )
        updated_candidates.append(
            replace(
                candidate,
                status=STATUS_REJECTED,
                rejection_basis=REJECTION_BASIS_NEGATIVE_VISIBILITY,
            )
        )
        rejected.append(candidate.candidate_uid)
    if terminal_eligible and set(rejected) != recommended_set:
        raise ValueError("reconciliation report references unknown candidate UID")

    updated = replace(registry, candidates=tuple(updated_candidates))
    validate_stand_survey_registry(updated, plan)
    source_sha256 = stand_survey_registry_sha256(registry)
    updated_sha256 = stand_survey_registry_sha256(updated)
    application = CoverageCandidateReconciliationApplication(
        schema_version=(
            COVERAGE_CANDIDATE_RECONCILIATION_APPLICATION_SCHEMA_VERSION
        ),
        survey_id=plan.survey_id,
        planning_frame=plan.planning_frame,
        map_bundle_sha256=plan.map_bundle_sha256,
        plan_sha256=coverage_survey_plan_sha256(plan),
        progress_snapshot_sha256=_progress_snapshot_sha256(progress),
        report_sha256=report.report_sha256,
        source_registry_snapshot_sha256=source_sha256,
        updated_registry_snapshot_sha256=updated_sha256,
        planned_viewpoint_ids=planned,
        visited_viewpoint_ids=visited,
        included_viewpoint_ids=included,
        terminal_application_eligible=terminal_eligible,
        application_reasons=tuple(application_reasons),
        recommended_candidate_uids=recommended,
        rejected_candidate_uids=tuple(rejected),
        unapplied_recommended_candidate_uids=(
            () if terminal_eligible else recommended
        ),
        retained_provisional_candidate_uids=tuple(
            candidate.candidate_uid
            for candidate in updated.candidates
            if candidate.status == STATUS_PROVISIONAL
        ),
        registry_mutation_applied=updated_sha256 != source_sha256,
    )
    return updated, application


def _validate_report_binding(
    *,
    plan: CoverageSurveyPlan,
    registry: StandSurveyRegistry,
    report: CoverageCandidateReconciliationReport,
) -> None:
    if not isinstance(report, CoverageCandidateReconciliationReport):
        raise ValueError("report must be a CoverageCandidateReconciliationReport")
    if (
        report.schema_version
        != COVERAGE_CANDIDATE_RECONCILIATION_REPORT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported reconciliation report schema")
    if report.policy_mode != POLICY_MODE_EVIDENCE_ONLY:
        raise ValueError("unsupported reconciliation report policy mode")
    if any(
        decision.schema_version
        != COVERAGE_CANDIDATE_RECONCILIATION_SCHEMA_VERSION
        for decision in report.decisions
    ):
        raise ValueError("unsupported reconciliation decision schema")
    if report.survey_id != plan.survey_id:
        raise ValueError("reconciliation report survey_id mismatch")
    if report.planning_frame != plan.planning_frame:
        raise ValueError("reconciliation report planning_frame mismatch")
    if report.map_bundle_sha256 != plan.map_bundle_sha256:
        raise ValueError("reconciliation report map_bundle_sha256 mismatch")
    if report.plan_sha256 != coverage_survey_plan_sha256(plan):
        raise ValueError("reconciliation report plan snapshot mismatch")
    if report.registry_snapshot_sha256 != stand_survey_registry_sha256(registry):
        raise ValueError("reconciliation report registry snapshot mismatch")
    if report.registry_schema_version != registry.schema_version:
        raise ValueError("reconciliation report registry schema mismatch")
    if report.registry_candidate_count != len(registry.candidates):
        raise ValueError("reconciliation report registry candidate count mismatch")
    actual_config = report.reconciliation_config.to_evidence_dict().copy()
    expected_config = CoverageCandidateReconciliationConfig(
        observer_config_sha256=(
            report.reconciliation_config.observer_config_sha256
        )
    ).to_evidence_dict()
    if actual_config != expected_config:
        raise ValueError(
            "reconciliation report uses a non-canonical policy config"
        )
    if report.reconciliation_config_sha256 != payload_sha256(actual_config):
        raise ValueError("reconciliation report config hash mismatch")
    if report.registry_mutation_applied or report.motion_authorized:
        raise ValueError("reconciliation recommendation report changed safety scope")
    if len(report.receipt_viewpoint_ids) != len(
        set(report.receipt_viewpoint_ids)
    ):
        raise ValueError("reconciliation report receipt viewpoints are not unique")
    if not set(report.receipt_viewpoint_ids).issubset(plan.viewpoint_ids):
        raise ValueError("reconciliation report references unknown receipt viewpoint")

    provisional = tuple(
        candidate
        for candidate in registry.candidates
        if candidate.status == STATUS_PROVISIONAL
    )
    expected_decision_uids = tuple(
        candidate.candidate_uid
        for candidate in provisional
        if len(set(candidate.viewpoint_ids)) == 1
    )
    decision_uids = tuple(decision.candidate_uid for decision in report.decisions)
    if decision_uids != expected_decision_uids:
        raise ValueError("reconciliation report decision population mismatch")
    candidate_by_uid = {
        candidate.candidate_uid: candidate for candidate in provisional
    }
    canonical_receipt_ids: tuple[str, ...] | None = None
    for decision in report.decisions:
        candidate = candidate_by_uid[decision.candidate_uid]
        if (
            decision.survey_id != report.survey_id
            or decision.planning_frame != report.planning_frame
            or decision.map_bundle_sha256 != report.map_bundle_sha256
            or decision.plan_sha256 != report.plan_sha256
            or decision.config != report.reconciliation_config
            or decision.candidate_status != STATUS_PROVISIONAL
        ):
            raise ValueError("reconciliation decision provenance mismatch")
        if decision.source_viewpoint_ids != tuple(
            sorted(set(candidate.viewpoint_ids))
        ):
            raise ValueError("reconciliation decision source viewpoint mismatch")
        if decision.input_receipt_set_sha256 != report.receipt_set_sha256:
            raise ValueError("reconciliation decision receipt-set hash mismatch")
        if (
            len(decision.input_receipt_ids) != report.receipt_count
            or len(decision.input_receipt_ids)
            != len(set(decision.input_receipt_ids))
        ):
            raise ValueError("reconciliation decision receipt IDs mismatch")
        if canonical_receipt_ids is None:
            canonical_receipt_ids = decision.input_receipt_ids
        elif decision.input_receipt_ids != canonical_receipt_ids:
            raise ValueError("reconciliation decisions use different receipt IDs")
        if not set(decision.receipt_viewpoint_ids).issubset(
            report.receipt_viewpoint_ids
        ):
            raise ValueError("reconciliation decision receipt viewpoint mismatch")
        ray_receipt_ids = tuple(item.receipt_id for item in decision.ray_evidence)
        if (
            len(ray_receipt_ids) != len(set(ray_receipt_ids))
            or not set(ray_receipt_ids).issubset(decision.input_receipt_ids)
        ):
            raise ValueError("reconciliation ray evidence receipt IDs mismatch")
        if decision.receipt_viewpoint_ids != tuple(
            sorted({item.viewpoint_id for item in decision.ray_evidence})
        ):
            raise ValueError("reconciliation ray evidence viewpoints mismatch")
        expected_clear_stamps = _separated_clear_scan_stamps(
            decision,
        )
        if decision.distinct_clear_scan_stamps_sec != expected_clear_stamps:
            raise ValueError("reconciliation clear-scan timestamps mismatch")
        expected_ray_policy = evaluate_negative_visibility_ray_policy(
            (item.classification for item in decision.ray_evidence),
            distinct_clear_scan_count=len(expected_clear_stamps),
            policy=decision.config.ray_policy,
        )
        if decision.ray_policy_decision != expected_ray_policy:
            raise ValueError("reconciliation ray-policy evidence mismatch")
        if decision.action not in {ACTION_RETAIN, ACTION_REJECT_PROVISIONAL}:
            raise ValueError("reconciliation decision action is invalid")
        if decision.reject_provisional != (
            decision.action == ACTION_REJECT_PROVISIONAL
        ):
            raise ValueError("reconciliation decision action is inconsistent")
        if decision.reject_provisional != (not decision.reasons):
            raise ValueError("reconciliation decision reasons are inconsistent")
        policy_reasons = decision.ray_policy_decision.reasons
        if not set(policy_reasons).issubset(decision.reasons):
            raise ValueError("reconciliation ray-policy reasons are unbound")
        if decision.ray_policy_decision.rejection_supported != (
            not policy_reasons
        ):
            raise ValueError("reconciliation ray-policy decision is inconsistent")
        if (
            decision.reject_provisional
            and not decision.ray_policy_decision.rejection_supported
        ):
            raise ValueError("reconciliation rejection lacks ray-policy support")
    expected_recommended = tuple(
        decision.candidate_uid
        for decision in report.decisions
        if decision.reject_provisional
    )
    if report.recommended_negative_visibility_candidate_uids != expected_recommended:
        raise ValueError("reconciliation report recommendation set mismatch")
    expected_retained = tuple(
        candidate.candidate_uid
        for candidate in provisional
        if candidate.candidate_uid not in set(expected_recommended)
    )
    if report.retained_provisional_candidate_uids != expected_retained:
        raise ValueError("reconciliation report retained population mismatch")
    expected_unevaluated = tuple(
        candidate.candidate_uid
        for candidate in provisional
        if len(set(candidate.viewpoint_ids)) != 1
    )
    if report.unevaluated_provisional_candidate_uids != expected_unevaluated:
        raise ValueError("reconciliation report unevaluated population mismatch")


def _progress_snapshot_sha256(progress: CoverageSurveyProgress) -> str:
    return payload_sha256(
        {
            "schema_version": progress.schema_version,
            "survey_id": progress.survey_id,
            "plan_sha256": progress.plan_sha256,
            "visited_viewpoint_ids": list(progress.visited_viewpoint_ids),
        }
    )


def _separated_clear_scan_stamps(
    decision: CoverageCandidateReconciliationDecision,
) -> tuple[float, ...]:
    clear = sorted(
        {
            item.scan_stamp_sec
            for item in decision.ray_evidence
            if item.classification == "clear"
        }
    )
    selected: list[float] = []
    minimum_separation_sec = (
        decision.config.minimum_clear_scan_separation_sec
    )
    for stamp in clear:
        if (
            not selected
            or stamp - selected[-1] + _EPSILON >= minimum_separation_sec
        ):
            selected.append(stamp)
    return tuple(selected)


__all__ = [
    "COVERAGE_CANDIDATE_RECONCILIATION_APPLICATION_SCHEMA_VERSION",
    "POLICY_MODE_BOUNDED_NEGATIVE_VISIBILITY_REGISTRY_REJECTION",
    "CoverageCandidateReconciliationApplication",
    "apply_negative_visibility_reconciliation_report",
    "bounded_negative_visibility_reconciliation_policy_contract",
    "negative_visibility_reconciliation_policy_contract",
]
