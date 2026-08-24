"""Injected child execution and exact permit-context handoff."""

from __future__ import annotations

from pathlib import Path

from scripts.aufgabe04.navigation.mission_leg_motion_permit import MissionLegKind
from scripts.aufgabe04.real_robot.autonomous_child_runner import MotionLegOutcome

from .models import MissionLegPermitContext

class ChildExecutionMixin:
    def _run_child(
        self,
        *,
        sealed: dict[str, object],
        run_id: str,
    ) -> MotionLegOutcome:
        outcome = self.effects.run_motion_leg(
            profile=self.profile,
            sealed=sealed,
            run_id=run_id,
            session_root=self.session_root,
            execute=True,
            coverage_plan=self.plan_path,
            coverage_transient_replan={
                "survey_root": self.survey_root,
                "session_root": self.session_root,
                "map_yaml": self.config.map_yaml,
                "semantic_map_id": self.config.semantic_map_id,
                "target_viewpoint_id": self.target_viewpoint_id,
                "robot_radius_m": self.config.robot_radius_m,
                "max_replans": self.config.max_blockage_replans_per_leg,
                "leg_index": self.leg_index,
                "resume_state_json": self.transient_overlay_resume_state_path,
            },
            require_fresh_confirmation=self.fresh_confirmation_reason is not None,
            fresh_confirmation_reason=self.fresh_confirmation_reason or "startup",
            fresh_localization_evidence_path=self.fresh_localization_evidence_path,
            uncertainty_map_yaml=(
                self.config.map_yaml if self.localization_branch_proof_id else None
            ),
            uncertainty_sigma_multiplier=self.config.uncertainty_sigma_multiplier,
            localization_branch_proof_id=self.localization_branch_proof_id,
            runtime_localization_permit_context=self.pending_runtime_permit_context,
            startup_reseal_permit_context=self.pending_startup_permit_context,
            mission_leg_permit_context=(
                None
                if (
                    self.mission_leg_motion_authorization_json is None
                    or self.fresh_confirmation_reason is not None
                    or self.pending_runtime_permit_context is not None
                )
                else MissionLegPermitContext(
                    mission_authorization_json=Path(
                        self.mission_leg_motion_authorization_json
                    ).absolute(),
                    session_id=self.config.session_id,
                    semantic_map_id=self.config.semantic_map_id,
                    mission_leg_kind=MissionLegKind.COVERAGE,
                    mission_leg_index=self.leg_index,
                    target_id=self.target_viewpoint_id,
                    permit_json_path=(
                        self.session_root
                        / "motion_authorization"
                        / "mission_legs"
                        / f"{run_id}_permit.json"
                    ).absolute(),
                )
            ),
        )
        if outcome.motion_authorization_permit_path is not None:
            self.emit(
                {
                    "schema_version": 1,
                    "event": "runtime_localization_motion_permit_issued",
                    "timestamp": self.effects.clock(),
                    "leg_index": self.leg_index,
                    "run_id": outcome.run_id,
                    "runtime_localization_motion_permit_json": str(
                        outcome.motion_authorization_permit_path
                    ),
                    "runtime_localization_motion_permit_sha256": (
                        outcome.motion_authorization_permit_sha256
                    ),
                    "covered_by_initial_mission_run": True,
                    "additional_typed_run_required": False,
                }
            )
        if outcome.startup_reseal_motion_permit_path is not None:
            if self.pending_startup_permit_context is None:
                raise RuntimeError(
                    "startup reseal child reported a permit outside an exact "
                    "recovery context"
                )
            self.emit(
                {
                    "schema_version": 1,
                    "event": "startup_reseal_motion_permit_issued",
                    "timestamp": self.effects.clock(),
                    "leg_index": self.leg_index,
                    "run_id": outcome.run_id,
                    "startup_reseal_index": self.startup_reseal_index,
                    "recovery_source_kind": (
                        self.pending_startup_permit_context.recovery_source_kind
                    ),
                    "startup_reseal_motion_permit_json": str(
                        outcome.startup_reseal_motion_permit_path
                    ),
                    "startup_reseal_motion_permit_sha256": (
                        outcome.startup_reseal_motion_permit_sha256
                    ),
                    "covered_by_initial_mission_run": True,
                    "additional_typed_run_required": False,
                }
            )
        return outcome
