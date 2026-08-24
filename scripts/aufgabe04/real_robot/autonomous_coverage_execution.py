"""Compatibility façade for autonomous coverage-leg execution.

The canonical implementation lives in the ``coverage_leg`` package.  The
module alias preserves the historical import path and its monkeypatch seams.
"""

from __future__ import annotations

import sys

from scripts.aufgabe04.real_robot.coverage_leg import execution as _execution


__all__ = [
    "CoverageLegConfig",
    "CoverageLegEffects",
    "MissionLegPermitContext",
    "RuntimeLocalizationPermitContext",
    "execute_coverage_leg_with_replans",
]


sys.modules[__name__] = _execution
