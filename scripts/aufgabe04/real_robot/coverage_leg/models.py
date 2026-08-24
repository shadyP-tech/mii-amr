"""Immutable configuration and permit identities for a coverage leg."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import MissionLegKind
from scripts.aufgabe04.real_robot.autonomous_child_runner import (
    DEFAULT_UNCERTAINTY_SIGMA_MULTIPLIER,
)

DEFAULT_MAX_LOCALIZATION_READINESS_RETRIES_PER_LEG = 2

@dataclass(frozen=True)
class RuntimeLocalizationPermitContext:
    """Exact mission scope needed to authorize one recovery child run."""

    mission_authorization_json: Path
    session_id: str
    leg_index: int
    target_viewpoint_id: str
    reseal_index: int
    max_runtime_reseals_per_leg: int
    rejected_run_id: str
    runtime_reseal_decision_evidence: dict[str, object]
    fresh_localization_evidence_path: Path
    permit_json_path: Path

@dataclass(frozen=True)
class MissionLegPermitContext:
    """Exact routine-leg identity authorized by the mission-level RUN."""

    mission_authorization_json: Path
    session_id: str
    semantic_map_id: str
    mission_leg_kind: MissionLegKind
    mission_leg_index: int
    target_id: str
    permit_json_path: Path

@dataclass(frozen=True)
class CoverageLegConfig:
    """Behavior-relevant, immutable settings for one coverage leg.

    ``runtime`` is already resolved by the parent.  Retaining it here avoids
    repeating profile resolution during a bounded localization-reseal loop.
    """

    session_id: str
    map_yaml: Path
    semantic_map_id: str
    runtime: object
    robot_radius_m: float
    max_blockage_replans_per_leg: int
    max_startup_reseals_per_leg: int
    max_runtime_localization_reseals_per_leg: int
    max_localization_readiness_retries_per_leg: int = (
        DEFAULT_MAX_LOCALIZATION_READINESS_RETRIES_PER_LEG
    )
    localization_branch_proof_id: str = ""
    uncertainty_sigma_multiplier: float = (
        DEFAULT_UNCERTAINTY_SIGMA_MULTIPLIER
    )

    def __post_init__(self) -> None:
        if not self.session_id.strip() or not self.semantic_map_id.strip():
            raise ValueError("coverage session and semantic map IDs must be non-empty")
        retry_limits = (
            self.max_blockage_replans_per_leg,
            self.max_startup_reseals_per_leg,
            self.max_runtime_localization_reseals_per_leg,
            self.max_localization_readiness_retries_per_leg,
        )
        if any(type(value) is not int or value < 0 for value in retry_limits):
            raise ValueError(
                "coverage retry and reseal limits must be non-negative integers"
            )
        if not math.isfinite(self.robot_radius_m) or self.robot_radius_m <= 0.0:
            raise ValueError("coverage robot radius must be finite and positive")
        if (
            not math.isfinite(self.uncertainty_sigma_multiplier)
            or self.uncertainty_sigma_multiplier <= 0.0
        ):
            raise ValueError(
                "coverage uncertainty sigma multiplier must be finite and positive"
            )
