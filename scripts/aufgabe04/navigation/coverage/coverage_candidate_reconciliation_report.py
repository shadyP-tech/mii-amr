"""Evidence-only batch reporting for LiDAR candidate reconciliation.

This module deliberately stops one step before policy application.  It
evaluates every provisional candidate that has evidence from exactly one
viewpoint, records the pure reconciliation decisions, and recommends candidate
UIDs that have sufficient negative-visibility evidence.  It never mutates the
registry, changes candidate status, selects a route, or authorizes motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.coverage.coverage_candidate_reconciliation import (
    CoverageCandidateReconciliationConfig,
    CoverageCandidateReconciliationDecision,
    reconcile_provisional_candidate_visibility,
)
from scripts.aufgabe04.navigation.planning.map_io import OccupancyGrid
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    STATUS_PROVISIONAL,
    CoverageSurveyPlan,
    StandSurveyRegistry,
    SurveyCandidate,
    coverage_survey_plan_sha256,
    validate_stand_survey_registry,
)
from scripts.aufgabe04.perception.lidar_visibility_evidence import (
    LidarVisibilityReceipt,
    validate_lidar_visibility_receipt,
    visibility_receipts_sha256,
)


COVERAGE_CANDIDATE_RECONCILIATION_REPORT_SCHEMA_VERSION = 1
POLICY_MODE_EVIDENCE_ONLY = (
    "evidence_only_pending_calibrated_negative_detection"
)


def evidence_only_reconciliation_policy_contract() -> dict[str, object]:
    """Return behavior-relevant thresholds without an epoch-specific hash."""

    values = CoverageCandidateReconciliationConfig(
        observer_config_sha256="0" * 64
    ).to_evidence_dict()
    del values["observer_config_sha256"]
    return {
        "policy_mode": POLICY_MODE_EVIDENCE_ONLY,
        **values,
        "registry_mutation_enabled": False,
        "motion_authorized": False,
    }


@dataclass(frozen=True)
class CoverageCandidateReconciliationReport:
    """Immutable recommendation report over one registry snapshot."""

    schema_version: int
    survey_id: str
    planning_frame: str
    map_bundle_sha256: str
    plan_sha256: str
    registry_snapshot_sha256: str
    registry_schema_version: int
    registry_candidate_count: int
    reconciliation_config: CoverageCandidateReconciliationConfig
    reconciliation_config_sha256: str
    receipt_set_sha256: str
    receipt_count: int
    decisions: tuple[CoverageCandidateReconciliationDecision, ...]
    recommended_negative_visibility_candidate_uids: tuple[str, ...]
    retained_provisional_candidate_uids: tuple[str, ...]
    unevaluated_provisional_candidate_uids: tuple[str, ...]
    registry_mutation_applied: bool = field(default=False, init=False)
    policy_mode: str = field(default=POLICY_MODE_EVIDENCE_ONLY, init=False)
    motion_authorized: bool = field(default=False, init=False)

    @property
    def report_sha256(self) -> str:
        """Return the canonical hash of the JSON-safe evidence payload."""

        return payload_sha256(self.to_evidence_dict())

    def to_evidence_dict(self) -> dict[str, object]:
        """Return finite JSON data without adding a self-referential hash."""

        return {
            "schema_version": self.schema_version,
            "survey": {
                "survey_id": self.survey_id,
                "planning_frame": self.planning_frame,
                "map_bundle_sha256": self.map_bundle_sha256,
                "plan_sha256": self.plan_sha256,
            },
            "registry_snapshot": {
                "sha256": self.registry_snapshot_sha256,
                "schema_version": self.registry_schema_version,
                "candidate_count": self.registry_candidate_count,
            },
            "reconciliation_config": {
                "sha256": self.reconciliation_config_sha256,
                "values": self.reconciliation_config.to_evidence_dict(),
            },
            "input_receipts": {
                "receipt_set_sha256": self.receipt_set_sha256,
                "receipt_count": self.receipt_count,
            },
            "decisions": [
                decision.to_evidence_dict() for decision in self.decisions
            ],
            "decision_sha256s": [
                decision.decision_sha256 for decision in self.decisions
            ],
            "recommended_negative_visibility_candidate_uids": list(
                self.recommended_negative_visibility_candidate_uids
            ),
            "retained_provisional_candidate_uids": list(
                self.retained_provisional_candidate_uids
            ),
            "unevaluated_provisional_candidate_uids": list(
                self.unevaluated_provisional_candidate_uids
            ),
            "expected_stand_count_used": False,
            "registry_mutation_applied": self.registry_mutation_applied,
            "policy_mode": self.policy_mode,
            "motion_authorized": self.motion_authorized,
        }


def build_coverage_candidate_reconciliation_report(
    *,
    plan: CoverageSurveyPlan,
    registry: StandSurveyRegistry,
    occupancy_grid: OccupancyGrid,
    receipts: tuple[LidarVisibilityReceipt, ...],
    config: CoverageCandidateReconciliationConfig,
) -> CoverageCandidateReconciliationReport:
    """Evaluate provisional single-view candidates without applying policy.

    Candidate evaluation is independent of ``expected_stand_count``.  The
    registry is validated and hashed but never replaced or mutated.  Receipt
    order is canonicalized so equivalent validated snapshots produce the same
    report evidence.
    """

    validate_stand_survey_registry(registry, plan)
    config.validated()
    if not isinstance(receipts, tuple):
        raise ValueError("receipts must be a validated tuple")
    for receipt in receipts:
        validate_lidar_visibility_receipt(receipt)

    ordered_receipts = tuple(
        sorted(
            receipts,
            key=lambda item: (
                item.viewpoint_id,
                item.scan_stamp_sec,
                item.receipt_id,
                item.receipt_sha256,
            ),
        )
    )
    provisional_candidates = tuple(
        sorted(
            (
                candidate
                for candidate in registry.candidates
                if candidate.status == STATUS_PROVISIONAL
            ),
            key=lambda candidate: candidate.candidate_uid,
        )
    )
    candidates_to_evaluate = tuple(
        candidate
        for candidate in provisional_candidates
        if len(set(candidate.viewpoint_ids)) == 1
    )
    unevaluated_uids = tuple(
        candidate.candidate_uid
        for candidate in provisional_candidates
        if len(set(candidate.viewpoint_ids)) != 1
    )
    decisions = tuple(
        reconcile_provisional_candidate_visibility(
            plan=plan,
            candidate=candidate,
            occupancy_grid=occupancy_grid,
            receipts=ordered_receipts,
            config=config,
        )
        for candidate in candidates_to_evaluate
    )
    recommended_uids = tuple(
        decision.candidate_uid
        for decision in decisions
        if decision.reject_provisional
    )
    recommended_uid_set = set(recommended_uids)
    retained_uids = tuple(
        candidate.candidate_uid
        for candidate in provisional_candidates
        if candidate.candidate_uid not in recommended_uid_set
    )
    config_payload = config.to_evidence_dict()

    return CoverageCandidateReconciliationReport(
        schema_version=COVERAGE_CANDIDATE_RECONCILIATION_REPORT_SCHEMA_VERSION,
        survey_id=plan.survey_id,
        planning_frame=plan.planning_frame,
        map_bundle_sha256=plan.map_bundle_sha256,
        plan_sha256=coverage_survey_plan_sha256(plan),
        registry_snapshot_sha256=stand_survey_registry_snapshot_sha256(
            registry,
            plan=plan,
        ),
        registry_schema_version=registry.schema_version,
        registry_candidate_count=len(registry.candidates),
        reconciliation_config=config,
        reconciliation_config_sha256=payload_sha256(config_payload),
        receipt_set_sha256=visibility_receipts_sha256(ordered_receipts),
        receipt_count=len(ordered_receipts),
        decisions=decisions,
        recommended_negative_visibility_candidate_uids=recommended_uids,
        retained_provisional_candidate_uids=retained_uids,
        unevaluated_provisional_candidate_uids=unevaluated_uids,
    )


def stand_survey_registry_snapshot_sha256(
    registry: StandSurveyRegistry,
    *,
    plan: CoverageSurveyPlan | None = None,
) -> str:
    """Hash all registry content without importing private serializers."""

    validate_stand_survey_registry(registry, plan)
    return payload_sha256(_registry_snapshot_payload(registry))


def _registry_snapshot_payload(
    registry: StandSurveyRegistry,
) -> dict[str, object]:
    return {
        "schema_version": registry.schema_version,
        "survey_id": registry.survey_id,
        "planning_frame": registry.planning_frame,
        "map_bundle_sha256": registry.map_bundle_sha256,
        "candidates": [
            _candidate_snapshot_payload(candidate)
            for candidate in sorted(
                registry.candidates,
                key=lambda item: item.candidate_uid,
            )
        ],
    }


def _candidate_snapshot_payload(
    candidate: SurveyCandidate,
) -> dict[str, object]:
    return {
        "candidate_uid": candidate.candidate_uid,
        "x_m": candidate.x_m,
        "y_m": candidate.y_m,
        "radius_m": candidate.radius_m,
        "uncertainty_m": candidate.uncertainty_m,
        "keepout_radius_m": candidate.keepout_radius_m,
        "confidence": candidate.confidence,
        "hit_count": candidate.hit_count,
        "first_seen_sec": candidate.first_seen_sec,
        "last_seen_sec": candidate.last_seen_sec,
        "source_observation_ids": list(candidate.source_observation_ids),
        "viewpoint_ids": list(candidate.viewpoint_ids),
        "status": candidate.status,
    }


__all__ = [
    "COVERAGE_CANDIDATE_RECONCILIATION_REPORT_SCHEMA_VERSION",
    "POLICY_MODE_EVIDENCE_ONLY",
    "CoverageCandidateReconciliationReport",
    "build_coverage_candidate_reconciliation_report",
    "evidence_only_reconciliation_policy_contract",
    "stand_survey_registry_snapshot_sha256",
]
