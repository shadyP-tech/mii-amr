"""Bounded autonomous coverage-leg orchestration."""

from .execution import (
    CoverageLegConfig,
    CoverageLegEffects,
    MissionLegPermitContext,
    RuntimeLocalizationPermitContext,
    execute_coverage_leg_with_replans,
)

__all__ = [
    "CoverageLegConfig",
    "CoverageLegEffects",
    "MissionLegPermitContext",
    "RuntimeLocalizationPermitContext",
    "execute_coverage_leg_with_replans",
]
