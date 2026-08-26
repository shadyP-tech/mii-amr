"""Route sealing and pending-recovery authorization summaries."""

from __future__ import annotations

from pathlib import Path

from scripts.aufgabe04.real_robot.readiness.startup_reseal import (
    StartupResealPermitContext,
    write_startup_reseal_permit_summary,
)

class RouteSealingMixin:
    def _seal_current_route(
        self,
        *,
        run_id: str,
        execution_root: Path,
    ) -> dict[str, object]:
        try:
            sealed = self.effects.seal_route(
                source_route_csv=self.source_route,
                source_diagnostics_json=self.source_diagnostics,
                coverage_plan_path=self.plan_path,
                output_dir=execution_root,
            )
        except Exception as exc:
            if self.pending_runtime_route_seal is not None:
                self.emit(
                    {
                        **self.pending_runtime_route_seal,
                        "schema_version": 1,
                        "event": "runtime_localization_reseal_failed",
                        "timestamp": self.effects.clock(),
                        "phase": "route_seal",
                        "failure": str(exc),
                        "motion_continues_authorized": False,
                    }
                )
            if self.pending_startup_recovery is not None:
                self.emit(
                    {
                        "schema_version": 1,
                        "event": "startup_reseal_failed",
                        "timestamp": self.effects.clock(),
                        "phase": "route_seal",
                        "leg_index": self.leg_index,
                        "startup_reseal_index": self.startup_reseal_index,
                        "rejected_run_id": self.pending_startup_recovery[
                            "rejected_run_id"
                        ],
                        "recovery_source_kind": self.pending_startup_recovery[
                            "recovery_source_kind"
                        ],
                        "failure": str(exc),
                        "motion_continues_authorized": False,
                    }
                )
            raise
        if self.pending_runtime_route_seal is not None:
            covered_by_initial_run = self.mission_motion_authorization_json is not None
            self.emit(
                {
                    **self.pending_runtime_route_seal,
                    "schema_version": 1,
                    "event": "runtime_localization_route_sealed",
                    "timestamp": self.effects.clock(),
                    "replacement_run_id": run_id,
                    "replacement_route_csv": sealed["route_csv"],
                    "replacement_diagnostics_json": sealed[
                        "diagnostics_json"
                    ],
                    "replacement_route_certificate_json": sealed[
                        "route_certificate_json"
                    ],
                    "expected_dry_odom_execution_certificate_json": str(
                        self.session_root
                        / "odom_execution"
                        / f"{run_id}_dry_certificate.json"
                    ),
                    "expected_dry_uncertainty_budget_json": str(
                        self.session_root
                        / "odom_execution"
                        / f"{run_id}_dry_uncertainty_budget.json"
                    ),
                    "fresh_typed_run_required": not covered_by_initial_run,
                    "covered_by_initial_mission_run": covered_by_initial_run,
                    "expected_runtime_localization_motion_permit_json": (
                        ""
                        if self.pending_runtime_permit_context is None
                        else str(self.pending_runtime_permit_context.permit_json_path)
                    ),
                    "transient_overlay_resume_state_json": (
                        ""
                        if self.transient_overlay_resume_state_path is None
                        else str(self.transient_overlay_resume_state_path)
                    ),
                    "transient_overlay_resume_state_sha256": (
                        self.transient_overlay_resume_state_digest
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
            )
            self.pending_runtime_route_seal = None
        if self.pending_startup_recovery is not None:
            fresh_pose = self.pending_startup_recovery["fresh_start_pose"]
            assert isinstance(fresh_pose, dict)
            try:
                summary_path = write_startup_reseal_permit_summary(
                    self.session_root
                    / "motion_authorization"
                    / "startup_reseals"
                    / f"{run_id}_sealed_summary.json",
                    leg_index=self.leg_index,
                    target_viewpoint_id=self.target_viewpoint_id,
                    reseal_index=self.startup_reseal_index,
                    rejected_run_id=str(
                        self.pending_startup_recovery["rejected_run_id"]
                    ),
                    fresh_start_x_m=float(fresh_pose["x_m"]),
                    fresh_start_y_m=float(fresh_pose["y_m"]),
                    fresh_start_yaw_rad=float(fresh_pose["yaw_rad"]),
                    route_csv=Path(sealed["route_csv"]),
                    diagnostics_json=Path(sealed["diagnostics_json"]),
                    additional_typed_run_required=(
                        self.startup_reseal_motion_authorization_json is None
                    ),
                    recovery_source_kind=str(
                        self.pending_startup_recovery["recovery_source_kind"]
                    ),
                )
            except Exception as exc:
                self.emit(
                    {
                        "schema_version": 1,
                        "event": "startup_reseal_failed",
                        "timestamp": self.effects.clock(),
                        "phase": "authorization_summary",
                        "leg_index": self.leg_index,
                        "startup_reseal_index": self.startup_reseal_index,
                        "rejected_run_id": self.pending_startup_recovery[
                            "rejected_run_id"
                        ],
                        "recovery_source_kind": self.pending_startup_recovery[
                            "recovery_source_kind"
                        ],
                        "failure": str(exc),
                        "motion_continues_authorized": False,
                    }
                )
                raise
            if self.startup_reseal_motion_authorization_json is not None:
                self.pending_startup_permit_context = StartupResealPermitContext(
                    mission_authorization_json=Path(
                        self.startup_reseal_motion_authorization_json
                    ).absolute(),
                    session_id=self.config.session_id,
                    semantic_map_id=self.config.semantic_map_id,
                    leg_index=self.leg_index,
                    target_viewpoint_id=self.target_viewpoint_id,
                    reseal_index=self.startup_reseal_index,
                    max_startup_reseals_per_leg=(
                        self.config.max_startup_reseals_per_leg
                    ),
                    rejected_run_id=str(
                        self.pending_startup_recovery["rejected_run_id"]
                    ),
                    rejected_semantic_log_path=Path(
                        self.pending_startup_recovery["rejected_semantic_log_path"]
                    ).absolute(),
                    startup_reseal_summary_path=summary_path,
                    fresh_localization_evidence_path=Path(
                        self.pending_startup_recovery[
                            "fresh_localization_evidence_path"
                        ]
                    ).absolute(),
                    permit_json_path=(
                        self.session_root
                        / "motion_authorization"
                        / "startup_reseals"
                        / f"{run_id}_permit.json"
                    ).absolute(),
                    recovery_source_kind=str(
                        self.pending_startup_recovery["recovery_source_kind"]
                    ),
                )
            self.emit(
                {
                    "schema_version": 1,
                    "event": "startup_reseal_route_sealed",
                    "timestamp": self.effects.clock(),
                    "leg_index": self.leg_index,
                    "startup_reseal_index": self.startup_reseal_index,
                    "replacement_run_id": run_id,
                    "rejected_run_id": self.pending_startup_recovery[
                        "rejected_run_id"
                    ],
                    "recovery_source_kind": self.pending_startup_recovery[
                        "recovery_source_kind"
                    ],
                    "replacement_route_csv": sealed["route_csv"],
                    "replacement_diagnostics_json": sealed[
                        "diagnostics_json"
                    ],
                    "replacement_route_certificate_json": sealed[
                        "route_certificate_json"
                    ],
                    "startup_reseal_permit_summary_json": str(summary_path),
                    "covered_by_initial_mission_run": (
                        self.pending_startup_permit_context is not None
                    ),
                    "additional_typed_run_required": (
                        self.pending_startup_permit_context is None
                    ),
                    "motion_continues_authorized": False,
                }
            )
        return sealed
