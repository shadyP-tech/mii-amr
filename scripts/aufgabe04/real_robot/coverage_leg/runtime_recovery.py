"""Post-motion localization loss recovery and reseal transition."""

from __future__ import annotations

from pathlib import Path

from scripts.aufgabe04.navigation.runtime_localization_reseal import (
    evaluate_runtime_localization_reseal,
    evaluate_runtime_localization_reseal_budget,
)
from scripts.aufgabe04.real_robot.autonomous_child_runner import MotionLegOutcome

from .models import RuntimeLocalizationPermitContext

class RuntimeRecoveryMixin:
    def _handle_runtime_localization_reseal(
        self,
        outcome: MotionLegOutcome,
    ) -> bool:
        runtime_localization_decision = evaluate_runtime_localization_reseal(
            status=outcome.status,
            motion_published=outcome.motion_published,
            stop_details=outcome.stop_details,
        )
        if runtime_localization_decision.eligible:
            budget = evaluate_runtime_localization_reseal_budget(
                completed_reseal_count=self.runtime_localization_reseal_index,
                maximum_reseal_count=(
                    self.config.max_runtime_localization_reseals_per_leg
                ),
            )
            if not budget.allowed:
                self.emit(
                    {
                        "schema_version": 1,
                        "event": "runtime_localization_reseal_rejected",
                        "timestamp": self.effects.clock(),
                        "leg_index": self.leg_index,
                        "rejected_run_id": outcome.run_id,
                        "reason": budget.reason,
                        "runtime_localization_reseal_decision": (
                            runtime_localization_decision.to_evidence()
                        ),
                        "runtime_localization_reseal_budget": budget.to_evidence(),
                        "motion_continues_authorized": False,
                        "fail_closed": True,
                    }
                )
                raise RuntimeError(
                    "runtime localization reseal budget exhausted for "
                    f"coverage leg {self.leg_index}: {outcome.stop_reason}"
                )
            try:
                self.transient_overlay_resume_state = (
                    self.effects.advance_transient_overlay_resume_state(
                        outcome=outcome,
                        previous_state=self.transient_overlay_resume_state,
                        plan_path=self.plan_path,
                        leg_index=self.leg_index,
                        target_viewpoint_id=self.target_viewpoint_id,
                        max_replans=self.config.max_blockage_replans_per_leg,
                        require_uncertainty_admission=bool(
                            self.localization_branch_proof_id
                        ),
                        artifact_root=self.session_root,
                        survey_root=self.survey_root,
                    )
                )
                if self.transient_overlay_resume_state is not None and self.coverage_plan is None:
                    self.coverage_plan = self.effects.load_coverage_plan(self.plan_path)
            except Exception as exc:
                self.emit(
                    {
                        "schema_version": 1,
                        "event": "runtime_localization_reseal_rejected",
                        "timestamp": self.effects.clock(),
                        "leg_index": self.leg_index,
                        "rejected_run_id": outcome.run_id,
                        "reason": "adopted_transient_blockage_resume_state_invalid",
                        "failure": str(exc),
                        "runtime_localization_reseal_decision": (
                            runtime_localization_decision.to_evidence()
                        ),
                        "motion_continues_authorized": False,
                        "fail_closed": True,
                    }
                )
                raise RuntimeError(
                    "cannot resume runtime localization with adopted transient "
                    f"blockage state: {exc}"
                ) from exc

            assert budget.next_reseal_index is not None
            self.runtime_localization_reseal_index = budget.next_reseal_index
            self.localization_readiness_retry_index = 0
            self.fresh_localization_evidence_path = (
                self.session_root
                / "preflight"
                / "runtime_localization_reseals"
                / (
                    f"coverage_leg_{self.leg_index:03d}"
                    "_runtime_localization_reseal_"
                    f"{self.runtime_localization_reseal_index:03d}.json"
                )
            )
            recovery_event_base = {
                "leg_index": self.leg_index,
                "runtime_localization_reseal_index": (
                    self.runtime_localization_reseal_index
                ),
                "rejected_run_id": outcome.run_id,
                "rejected_stop_details": outcome.stop_details,
                "fresh_localization_evidence_json": str(
                    self.fresh_localization_evidence_path
                ),
                "runtime_localization_reseal_decision": (
                    runtime_localization_decision.to_evidence()
                ),
                "runtime_localization_reseal_budget": budget.to_evidence(),
                "fresh_confirmation_required": (
                    self.mission_motion_authorization_json is None
                ),
                "covered_by_initial_mission_run": (
                    self.mission_motion_authorization_json is not None
                ),
                "additional_typed_run_required": (
                    self.mission_motion_authorization_json is None
                ),
                "dynamic_overlay_preserved": (
                    self.transient_overlay_resume_state is not None
                ),
                "adopted_blockage_replan_count": (
                    0
                    if self.transient_overlay_resume_state is None
                    else self.transient_overlay_resume_state.completed_replan_count
                ),
                "remaining_blockage_replan_count": (
                    self.config.max_blockage_replans_per_leg
                    if self.transient_overlay_resume_state is None
                    else self.transient_overlay_resume_state.remaining_replans
                ),
                "motion_continues_authorized": False,
            }
            self.emit(
                {
                    **recovery_event_base,
                    "schema_version": 1,
                    "event": "runtime_localization_reseal_started",
                    "timestamp": self.effects.clock(),
                    "source_stop_requires_zero_cycle": True,
                }
            )
            try:
                admitted_pose = self.effects.admit_preplanning_localization(
                    self.config.runtime,
                    self.session_root,
                    evidence_path=self.fresh_localization_evidence_path,
                )
            except Exception as exc:
                self.emit(
                    {
                        **recovery_event_base,
                        "schema_version": 1,
                        "event": "runtime_localization_reseal_failed",
                        "timestamp": self.effects.clock(),
                        "phase": "stationary_localization_admission",
                        "failure": str(exc),
                    }
                )
                raise
            fresh_start_pose = {
                "x_m": admitted_pose.x_m,
                "y_m": admitted_pose.y_m,
                "yaw_rad": admitted_pose.yaw_rad,
            }
            self.emit(
                {
                    **recovery_event_base,
                    "schema_version": 1,
                    "event": "runtime_localization_admitted",
                    "timestamp": self.effects.clock(),
                    "fresh_start_pose": fresh_start_pose,
                }
            )
            reseal_root = (
                self.survey_root
                / "runtime_localization_reseals"
                / (
                    f"leg_{self.leg_index:03d}"
                    "_runtime_localization_reseal_"
                    f"{self.runtime_localization_reseal_index:03d}"
                )
            )
            try:
                if self.transient_overlay_resume_state is None:
                    replanned = self.effects.replan_runtime_localization_source(
                        map_yaml=self.config.map_yaml,
                        semantic_map_id=self.config.semantic_map_id,
                        survey_root=self.survey_root,
                        plan_path=self.plan_path,
                        expected_target_viewpoint_id=self.target_viewpoint_id,
                        current_pose=admitted_pose,
                        rejected_outcome=outcome,
                        reseal_index=self.runtime_localization_reseal_index,
                        output_dir=reseal_root,
                    )
                    self.transient_overlay_resume_state_path = None
                    self.transient_overlay_resume_state_digest = ""
                else:
                    if self.coverage_plan is None:
                        self.coverage_plan = self.effects.load_coverage_plan(self.plan_path)
                    (
                        replanned,
                        self.transient_overlay_resume_state,
                        self.transient_overlay_resume_state_path,
                        self.transient_overlay_resume_state_digest,
                    ) = self.effects.replan_source_preserving_transient_overlay(
                        state=self.transient_overlay_resume_state,
                        plan=self.coverage_plan,
                        map_yaml=self.config.map_yaml,
                        semantic_map_id=self.config.semantic_map_id,
                        survey_root=self.survey_root,
                        target_viewpoint_id=self.target_viewpoint_id,
                        current_pose=admitted_pose,
                        rejected_outcome=outcome,
                        output_dir=reseal_root,
                        robot_radius_m=self.config.robot_radius_m,
                        recovery_kind="runtime_localization",
                        artifact_root=self.session_root,
                    )
            except Exception as exc:
                self.emit(
                    {
                        **recovery_event_base,
                        "schema_version": 1,
                        "event": "runtime_localization_reseal_failed",
                        "timestamp": self.effects.clock(),
                        "phase": "same_target_route_replan",
                        "failure": str(exc),
                    }
                )
                raise
            self.emit(
                {
                    **recovery_event_base,
                    "schema_version": 1,
                    "event": "runtime_localization_route_replanned",
                    "timestamp": self.effects.clock(),
                    "fresh_start_pose": fresh_start_pose,
                    "replacement_route_csv": replanned["route_csv"],
                    "replacement_diagnostics_json": replanned[
                        "diagnostics_json"
                    ],
                    "replacement_summary_json": replanned["summary_json"],
                    "dynamic_overlay_preserved": (
                        self.transient_overlay_resume_state is not None
                    ),
                    "adopted_blockage_replan_count": (
                        0
                        if self.transient_overlay_resume_state is None
                        else self.transient_overlay_resume_state.completed_replan_count
                    ),
                    "remaining_blockage_replan_count": (
                        self.config.max_blockage_replans_per_leg
                        if self.transient_overlay_resume_state is None
                        else self.transient_overlay_resume_state.remaining_replans
                    ),
                    "transient_overlay_resume_state_json": (
                        ""
                        if self.transient_overlay_resume_state_path is None
                        else str(self.transient_overlay_resume_state_path)
                    ),
                    "transient_overlay_resume_state_sha256": (
                        self.transient_overlay_resume_state_digest
                    ),
                    "committed_target_viewpoint_id": self.target_viewpoint_id,
                }
            )
            self.pending_runtime_route_seal = {
                **recovery_event_base,
                "fresh_start_pose": fresh_start_pose,
                "committed_target_viewpoint_id": self.target_viewpoint_id,
                "replacement_source_route_csv": replanned["route_csv"],
                "replacement_source_diagnostics_json": replanned[
                    "diagnostics_json"
                ],
                "replacement_summary_json": replanned["summary_json"],
            }
            if self.mission_motion_authorization_json is not None:
                self.pending_runtime_permit_context = RuntimeLocalizationPermitContext(
                    mission_authorization_json=Path(
                        self.mission_motion_authorization_json
                    ).absolute(),
                    session_id=self.config.session_id,
                    leg_index=self.leg_index,
                    target_viewpoint_id=self.target_viewpoint_id,
                    reseal_index=self.runtime_localization_reseal_index,
                    max_runtime_reseals_per_leg=(
                        self.config.max_runtime_localization_reseals_per_leg
                    ),
                    rejected_run_id=outcome.run_id,
                    runtime_reseal_decision_evidence=(
                        runtime_localization_decision.to_evidence()
                    ),
                    fresh_localization_evidence_path=(
                        self.fresh_localization_evidence_path
                    ),
                    permit_json_path=(
                        self.session_root
                        / "motion_authorization"
                        / (
                            f"{self.config.session_id}_coverage_"
                            f"{self.leg_index:03d}_runtime_localization_"
                            f"reseal_{self.runtime_localization_reseal_index:03d}_"
                            "permit.json"
                        )
                    ).absolute(),
                )
            self.source_route = Path(replanned["route_csv"])
            self.source_diagnostics = Path(replanned["diagnostics_json"])
            self.fresh_confirmation_reason = "runtime_localization"
            return True
        return False
