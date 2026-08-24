"""Coverage-leg orchestration without ROS or parent-runner coupling.

This module owns the bounded coverage-leg retry/reseal state machine.  Effects
which can prompt, launch a child, sample localization, or build recovery
artifacts are supplied or resolved through public module boundaries at call
time.  Importing this module therefore cannot authorize motion or connect to
ROS.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

from scripts.aufgabe04.navigation.mission_leg_motion_permit import (
    MissionLegKind,
)
from scripts.aufgabe04.navigation.prestart_localization_reseal import (
    evaluate_prestart_localization_reseal,
)
from scripts.aufgabe04.navigation.runtime_localization_reseal import (
    evaluate_runtime_localization_reseal,
    evaluate_runtime_localization_reseal_budget,
)
from scripts.aufgabe04.navigation.startup_reseal_motion_authorization import (
    STARTUP_RESEAL_RECOVERY_SOURCE_CERTIFIED_START_POSE_MISMATCH,
    STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY,
)
from scripts.aufgabe04.navigation.stand_coverage_survey import CoverageSurveyPlan
from scripts.aufgabe04.navigation.stand_discovery_route import (
    seal_stand_discovery_route,
)
from scripts.aufgabe04.navigation.transient_overlay_resume_state import (
    TransientOverlayResumeState,
)
from scripts.aufgabe04.real_robot.autonomous_child_runner import (
    DEFAULT_UNCERTAINTY_SIGMA_MULTIPLIER,
    MotionLegOutcome,
)
from scripts.aufgabe04.real_robot import autonomous_coverage_replanning as replanning
from scripts.aufgabe04.real_robot.autonomous_localization_readiness import (
    evaluate_localization_readiness_retry,
    localization_readiness_suffix,
)
from scripts.aufgabe04.real_robot.autonomous_startup_reseal import (
    StartupResealPermitContext,
    write_startup_reseal_permit_summary,
)


from .models import (
    DEFAULT_MAX_LOCALIZATION_READINESS_RETRIES_PER_LEG,
    RuntimeLocalizationPermitContext,
    MissionLegPermitContext,
    CoverageLegConfig,
)
from .effects import (
    EventSink,
    KeywordEffect,
    CoverageLegEffects,
    _append_jsonl,
)
from .outcomes import (
    _require_completed_motion,
    _claims_prestart_localization_phase,
)















from .route_sealing import RouteSealingMixin
from .child_execution import ChildExecutionMixin
from .readiness_recovery import ReadinessRecoveryMixin
from .startup_recovery import StartupRecoveryMixin
from .runtime_recovery import RuntimeRecoveryMixin

class _CoverageLegRunner(RouteSealingMixin, ChildExecutionMixin, ReadinessRecoveryMixin, StartupRecoveryMixin, RuntimeRecoveryMixin):
    """Mutable state and bounded transitions for one coverage leg."""

    def __init__(
        self,
        *,
        profile: object,
        config: CoverageLegConfig,
        effects: CoverageLegEffects,
        session_root: Path,
        survey_root: Path,
        plan_path: Path,
        leg_index: int,
        target_viewpoint_id: str,
        source_route: Path,
        source_diagnostics: Path,
        mission_motion_authorization_json: Path | None,
        mission_leg_motion_authorization_json: Path | None,
        startup_reseal_motion_authorization_json: Path | None,
    ) -> None:
        self.profile = profile
        self.config = config
        self.effects = effects
        self.session_root = session_root
        self.survey_root = survey_root
        self.plan_path = plan_path
        self.leg_index = leg_index
        self.target_viewpoint_id = target_viewpoint_id
        self.source_route = source_route
        self.source_diagnostics = source_diagnostics
        self.mission_motion_authorization_json = mission_motion_authorization_json
        self.mission_leg_motion_authorization_json = (
            mission_leg_motion_authorization_json
        )
        self.startup_reseal_motion_authorization_json = (
            startup_reseal_motion_authorization_json
        )
        self.localization_branch_proof_id = str(
            config.localization_branch_proof_id
        ).strip()
        self.coverage_plan: CoverageSurveyPlan | None = None
        self.startup_reseal_index = 0
        self.runtime_localization_reseal_index = 0
        self.localization_readiness_retry_index = 0
        self.transient_overlay_resume_state: TransientOverlayResumeState | None = None
        self.transient_overlay_resume_state_path: Path | None = None
        self.transient_overlay_resume_state_digest = ""
        self.fresh_confirmation_reason: str | None = None
        self.fresh_localization_evidence_path: Path | None = None
        self.pending_runtime_route_seal: dict[str, object] | None = None
        self.pending_runtime_permit_context: RuntimeLocalizationPermitContext | None = None
        self.pending_startup_recovery: dict[str, object] | None = None
        self.pending_startup_permit_context: StartupResealPermitContext | None = None
        self.prestart_localization_decision = None
        self.adaptive_log = session_root / "adaptive_replans.jsonl"

    def emit(self, payload: dict[str, object]) -> None:
        self.effects.event_sink(self.adaptive_log, payload)













    def run(self) -> MotionLegOutcome:
        while True:
            suffix = replanning.coverage_reseal_suffix(
                startup_reseal_index=self.startup_reseal_index,
                runtime_localization_reseal_index=(
                    self.runtime_localization_reseal_index
                ),
            ) + localization_readiness_suffix(
                self.localization_readiness_retry_index
            )
            run_id = (
                f"{self.config.session_id}_coverage_{self.leg_index:03d}{suffix}"
            )
            execution_root = (
                self.session_root
                / "execution"
                / f"coverage_leg_{self.leg_index:03d}{suffix}"
            )
            sealed = self._seal_current_route(
                run_id=run_id,
                execution_root=execution_root,
            )
            outcome = self._run_child(sealed=sealed, run_id=run_id)
            if outcome.status == "completed":
                return outcome
            if self._handle_localization_readiness_retry(outcome):
                continue
            if self._handle_startup_reseal(outcome):
                continue
            self._reject_claimed_prestart_phase(outcome)
            if self._handle_runtime_localization_reseal(outcome):
                continue
            _require_completed_motion(outcome)


__all__ = ["_CoverageLegRunner"]
