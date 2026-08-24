"""Certified-start and prestart-localization recovery transitions."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from scripts.aufgabe04.navigation.localization.prestart_localization_reseal import (
    evaluate_prestart_localization_reseal,
)
from scripts.aufgabe04.navigation.execution.startup_reseal_motion_authorization import (
    STARTUP_RESEAL_RECOVERY_SOURCE_CERTIFIED_START_POSE_MISMATCH,
    STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY,
)
from scripts.aufgabe04.real_robot.autonomous_child_runner import MotionLegOutcome
from scripts.aufgabe04.real_robot import autonomous_coverage_replanning as replanning

from .outcomes import (
    _claims_prestart_localization_phase,
    _require_completed_motion,
)

class StartupRecoveryMixin:
    def _handle_startup_reseal(self, outcome: MotionLegOutcome) -> bool:
        self.pending_runtime_permit_context = None
        self.pending_startup_permit_context = None
        self.pending_startup_recovery = None
        self.prestart_localization_decision = evaluate_prestart_localization_reseal(
            status=outcome.status,
            motion_published=outcome.motion_published,
            stop_details=outcome.stop_details,
        )
        prestart_localization_admitted = (
            self.prestart_localization_decision.eligible
            and isinstance(outcome.stop_details, Mapping)
            and outcome.stop_reason == outcome.stop_details.get("reason")
        )
        startup_pose_mismatch = replanning.is_resealable_startup_mismatch(
            outcome
        )
        if startup_pose_mismatch or prestart_localization_admitted:
            if prestart_localization_admitted:
                if (
                    self.prestart_localization_decision.motion_published is not False
                    or not self.prestart_localization_decision.requires_fresh_localization
                    or not self.prestart_localization_decision.requires_new_route_certificate
                    or self.prestart_localization_decision.automatic_motion_authorized
                ):
                    raise RuntimeError(
                        "prestart localization recovery decision violated its "
                        "fail-closed contract"
                    )
                recovery_source_kind = (
                    STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY
                )
                rejected_route_pose: dict[str, float] | None = None
            else:
                recovery_source_kind = (
                    STARTUP_RESEAL_RECOVERY_SOURCE_CERTIFIED_START_POSE_MISMATCH
                )
                rejected_pose = replanning.startup_reseal_pose(outcome)
                rejected_route_pose = {
                    "x_m": rejected_pose.x_m,
                    "y_m": rejected_pose.y_m,
                    "yaw_rad": rejected_pose.yaw_rad,
                }
            if self.startup_reseal_index >= self.config.max_startup_reseals_per_leg:
                raise RuntimeError(
                    "startup reseal budget exhausted for coverage leg "
                    f"{self.leg_index}: {outcome.stop_reason}"
                )
            self.startup_reseal_index += 1
            self.localization_readiness_retry_index = 0
            self.fresh_localization_evidence_path = (
                self.session_root
                / "preflight"
                / "startup_reseals"
                / (
                    f"coverage_leg_{self.leg_index:03d}_startup_reseal_"
                    f"{self.startup_reseal_index:03d}.json"
                )
            )
            startup_event_base = {
                "leg_index": self.leg_index,
                "startup_reseal_index": self.startup_reseal_index,
                "target_viewpoint_id": self.target_viewpoint_id,
                "rejected_run_id": outcome.run_id,
                "rejected_stop_details": outcome.stop_details,
                "recovery_source_kind": recovery_source_kind,
                "fresh_localization_evidence_json": str(
                    self.fresh_localization_evidence_path
                ),
                "covered_by_initial_mission_run": (
                    self.startup_reseal_motion_authorization_json is not None
                ),
                "additional_typed_run_required": (
                    self.startup_reseal_motion_authorization_json is None
                ),
                "motion_continues_authorized": False,
            }
            if rejected_route_pose is not None:
                startup_event_base["rejected_route_pose"] = rejected_route_pose
            else:
                startup_event_base[
                    "prestart_localization_reseal_decision"
                ] = self.prestart_localization_decision.to_evidence()
            self.emit(
                {
                    **startup_event_base,
                    "schema_version": 1,
                    "event": "startup_reseal_started",
                    "timestamp": self.effects.clock(),
                    "source_rejection_published_motion": False,
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
                        **startup_event_base,
                        "schema_version": 1,
                        "event": "startup_reseal_failed",
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
                    **startup_event_base,
                    "schema_version": 1,
                    "event": "startup_localization_admitted",
                    "timestamp": self.effects.clock(),
                    "fresh_start_pose": fresh_start_pose,
                }
            )
            reseal_root = (
                self.survey_root
                / "startup_reseals"
                / (
                    f"leg_{self.leg_index:03d}"
                    f"_startup_reseal_{self.startup_reseal_index:03d}"
                )
            )
            if self.transient_overlay_resume_state is None:
                replanned = self.effects.replan_startup_source(
                    map_yaml=self.config.map_yaml,
                    semantic_map_id=self.config.semantic_map_id,
                    survey_root=self.survey_root,
                    plan_path=self.plan_path,
                    expected_target_viewpoint_id=self.target_viewpoint_id,
                    current_pose=admitted_pose,
                    rejected_outcome=outcome,
                    reseal_index=self.startup_reseal_index,
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
                    recovery_kind="startup",
                    artifact_root=self.session_root,
                )
            self.emit(
                {
                    "schema_version": 1,
                    "event": (
                        "startup_pose_route_resealed"
                        if startup_pose_mismatch
                        else "prestart_localization_route_resealed"
                    ),
                    "timestamp": self.effects.clock(),
                    "leg_index": self.leg_index,
                    "startup_reseal_index": self.startup_reseal_index,
                    "recovery_source_kind": recovery_source_kind,
                    "rejected_run_id": outcome.run_id,
                    "rejected_stop_details": outcome.stop_details,
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
                    "fresh_confirmation_required": (
                        self.startup_reseal_motion_authorization_json is None
                    ),
                    "covered_by_initial_mission_run": (
                        self.startup_reseal_motion_authorization_json is not None
                    ),
                    "additional_typed_run_required": (
                        self.startup_reseal_motion_authorization_json is None
                    ),
                }
            )
            self.pending_startup_recovery = {
                "rejected_run_id": outcome.run_id,
                "rejected_semantic_log_path": outcome.semantic_log_path,
                "fresh_start_pose": fresh_start_pose,
                "fresh_localization_evidence_path": (
                    self.fresh_localization_evidence_path
                ),
                "recovery_source_kind": recovery_source_kind,
            }
            self.source_route = Path(replanned["route_csv"])
            self.source_diagnostics = Path(replanned["diagnostics_json"])
            self.fresh_confirmation_reason = "startup"
            self.fresh_localization_evidence_path = None
            return True
        return False
    def _reject_claimed_prestart_phase(self, outcome: MotionLegOutcome) -> None:
        if _claims_prestart_localization_phase(outcome.stop_details):
            self.emit(
                {
                    "schema_version": 1,
                    "event": "prestart_localization_reseal_rejected",
                    "timestamp": self.effects.clock(),
                    "leg_index": self.leg_index,
                    "target_viewpoint_id": self.target_viewpoint_id,
                    "rejected_run_id": outcome.run_id,
                    "stop_reason": outcome.stop_reason,
                    "stop_details": outcome.stop_details,
                    "motion_published": outcome.motion_published,
                    "prestart_localization_reseal_decision": (
                        self.prestart_localization_decision.to_evidence()
                    ),
                    "outcome_stop_reason_matches_details": (
                        isinstance(outcome.stop_details, Mapping)
                        and outcome.stop_reason
                        == outcome.stop_details.get("reason")
                    ),
                    "motion_continues_authorized": False,
                    "fail_closed": True,
                }
            )
            _require_completed_motion(outcome)
