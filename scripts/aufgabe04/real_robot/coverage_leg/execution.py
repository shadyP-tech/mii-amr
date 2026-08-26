"""Public API for bounded autonomous coverage-leg execution."""

from __future__ import annotations

from pathlib import Path

from scripts.aufgabe04.real_robot.execution.child_runner import MotionLegOutcome

from .effects import CoverageLegEffects
from .models import (
    CoverageLegConfig,
    MissionLegPermitContext,
    RuntimeLocalizationPermitContext,
)
from .runner import _CoverageLegRunner


def execute_coverage_leg_with_replans(
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
    mission_motion_authorization_json: Path | None = None,
    mission_leg_motion_authorization_json: Path | None = None,
    startup_reseal_motion_authorization_json: Path | None = None,
) -> MotionLegOutcome:
    """Run one coverage leg with bounded, fail-closed recovery branches."""

    return _CoverageLegRunner(
        profile=profile,
        config=config,
        effects=effects,
        session_root=session_root,
        survey_root=survey_root,
        plan_path=plan_path,
        leg_index=leg_index,
        target_viewpoint_id=target_viewpoint_id,
        source_route=source_route,
        source_diagnostics=source_diagnostics,
        mission_motion_authorization_json=mission_motion_authorization_json,
        mission_leg_motion_authorization_json=(
            mission_leg_motion_authorization_json
        ),
        startup_reseal_motion_authorization_json=(
            startup_reseal_motion_authorization_json
        ),
    ).run()


__all__ = [
    "CoverageLegConfig",
    "CoverageLegEffects",
    "MissionLegPermitContext",
    "RuntimeLocalizationPermitContext",
    "execute_coverage_leg_with_replans",
]
