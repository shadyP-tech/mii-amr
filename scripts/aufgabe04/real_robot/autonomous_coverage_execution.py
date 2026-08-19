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


EventSink = Callable[[Path, dict[str, object]], None]
KeywordEffect = Callable[..., Any]


@dataclass(frozen=True)
class CoverageLegEffects:
    """Injected effects and replaceable deterministic helpers.

    The first two callbacks are intentionally required: the coverage module
    must never discover a way to launch motion or sample live localization on
    its own.  The parent callback remains responsible for any typed ``RUN``
    prompt requested through ``require_fresh_confirmation``.
    """

    run_motion_leg: KeywordEffect
    admit_preplanning_localization: KeywordEffect
    seal_route: KeywordEffect = seal_stand_discovery_route
    event_sink: EventSink = lambda path, payload: _append_jsonl(path, payload)
    clock: Callable[[], float] = time.time
    replan_startup_source: KeywordEffect = lambda **kwargs: (
        replanning.replan_startup_source(**kwargs)
    )
    replan_runtime_localization_source: KeywordEffect = lambda **kwargs: (
        replanning.replan_runtime_localization_source(**kwargs)
    )
    advance_transient_overlay_resume_state: KeywordEffect = lambda **kwargs: (
        replanning.advance_transient_overlay_resume_state(**kwargs)
    )
    load_coverage_plan: KeywordEffect = lambda plan_path: (
        replanning.load_coverage_plan(plan_path)
    )
    replan_source_preserving_transient_overlay: KeywordEffect = (
        lambda **kwargs: replanning.replan_source_preserving_transient_overlay(
            **kwargs
        )
    )


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        )


def _require_completed_motion(outcome: MotionLegOutcome) -> None:
    if outcome.status != "completed":
        raise RuntimeError(
            f"physical route failed for {outcome.run_id}: {outcome.stop_reason}"
        )


def _claims_prestart_localization_phase(stop_details: object) -> bool:
    """Keep malformed before-motion evidence out of runtime recovery."""

    if not isinstance(stop_details, Mapping):
        return False
    return (
        stop_details.get("execution_phase") == "before_motion"
        or stop_details.get("phase") == "initial_runtime_input_wait"
    )


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

    localization_branch_proof_id = str(
        config.localization_branch_proof_id
    ).strip()
    coverage_plan: CoverageSurveyPlan | None = None
    startup_reseal_index = 0
    runtime_localization_reseal_index = 0
    localization_readiness_retry_index = 0
    transient_overlay_resume_state: TransientOverlayResumeState | None = None
    transient_overlay_resume_state_path: Path | None = None
    transient_overlay_resume_state_digest = ""
    fresh_confirmation_reason: str | None = None
    fresh_localization_evidence_path: Path | None = None
    pending_runtime_route_seal: dict[str, object] | None = None
    pending_runtime_permit_context: RuntimeLocalizationPermitContext | None = None
    pending_startup_recovery: dict[str, object] | None = None
    pending_startup_permit_context: StartupResealPermitContext | None = None
    adaptive_log = session_root / "adaptive_replans.jsonl"

    def emit(payload: dict[str, object]) -> None:
        effects.event_sink(adaptive_log, payload)

    while True:
        suffix = replanning.coverage_reseal_suffix(
            startup_reseal_index=startup_reseal_index,
            runtime_localization_reseal_index=runtime_localization_reseal_index,
        ) + localization_readiness_suffix(localization_readiness_retry_index)
        run_id = f"{config.session_id}_coverage_{leg_index:03d}{suffix}"
        execution_root = (
            session_root / "execution" / f"coverage_leg_{leg_index:03d}{suffix}"
        )
        try:
            sealed = effects.seal_route(
                source_route_csv=source_route,
                source_diagnostics_json=source_diagnostics,
                coverage_plan_path=plan_path,
                output_dir=execution_root,
            )
        except Exception as exc:
            if pending_runtime_route_seal is not None:
                emit(
                    {
                        **pending_runtime_route_seal,
                        "schema_version": 1,
                        "event": "runtime_localization_reseal_failed",
                        "timestamp": effects.clock(),
                        "phase": "route_seal",
                        "failure": str(exc),
                        "motion_continues_authorized": False,
                    }
                )
            if pending_startup_recovery is not None:
                emit(
                    {
                        "schema_version": 1,
                        "event": "startup_reseal_failed",
                        "timestamp": effects.clock(),
                        "phase": "route_seal",
                        "leg_index": leg_index,
                        "startup_reseal_index": startup_reseal_index,
                        "rejected_run_id": pending_startup_recovery[
                            "rejected_run_id"
                        ],
                        "recovery_source_kind": pending_startup_recovery[
                            "recovery_source_kind"
                        ],
                        "failure": str(exc),
                        "motion_continues_authorized": False,
                    }
                )
            raise
        if pending_runtime_route_seal is not None:
            covered_by_initial_run = mission_motion_authorization_json is not None
            emit(
                {
                    **pending_runtime_route_seal,
                    "schema_version": 1,
                    "event": "runtime_localization_route_sealed",
                    "timestamp": effects.clock(),
                    "replacement_run_id": run_id,
                    "replacement_route_csv": sealed["route_csv"],
                    "replacement_diagnostics_json": sealed[
                        "diagnostics_json"
                    ],
                    "replacement_route_certificate_json": sealed[
                        "route_certificate_json"
                    ],
                    "expected_dry_odom_execution_certificate_json": str(
                        session_root
                        / "odom_execution"
                        / f"{run_id}_dry_certificate.json"
                    ),
                    "expected_dry_uncertainty_budget_json": str(
                        session_root
                        / "odom_execution"
                        / f"{run_id}_dry_uncertainty_budget.json"
                    ),
                    "fresh_typed_run_required": not covered_by_initial_run,
                    "covered_by_initial_mission_run": covered_by_initial_run,
                    "expected_runtime_localization_motion_permit_json": (
                        ""
                        if pending_runtime_permit_context is None
                        else str(pending_runtime_permit_context.permit_json_path)
                    ),
                    "transient_overlay_resume_state_json": (
                        ""
                        if transient_overlay_resume_state_path is None
                        else str(transient_overlay_resume_state_path)
                    ),
                    "transient_overlay_resume_state_sha256": (
                        transient_overlay_resume_state_digest
                    ),
                    "dynamic_overlay_preserved": (
                        transient_overlay_resume_state is not None
                    ),
                    "adopted_blockage_replan_count": (
                        0
                        if transient_overlay_resume_state is None
                        else transient_overlay_resume_state.completed_replan_count
                    ),
                    "remaining_blockage_replan_count": (
                        config.max_blockage_replans_per_leg
                        if transient_overlay_resume_state is None
                        else transient_overlay_resume_state.remaining_replans
                    ),
                    "motion_continues_authorized": False,
                }
            )
            pending_runtime_route_seal = None

        if pending_startup_recovery is not None:
            fresh_pose = pending_startup_recovery["fresh_start_pose"]
            assert isinstance(fresh_pose, dict)
            try:
                summary_path = write_startup_reseal_permit_summary(
                    session_root
                    / "motion_authorization"
                    / "startup_reseals"
                    / f"{run_id}_sealed_summary.json",
                    leg_index=leg_index,
                    target_viewpoint_id=target_viewpoint_id,
                    reseal_index=startup_reseal_index,
                    rejected_run_id=str(
                        pending_startup_recovery["rejected_run_id"]
                    ),
                    fresh_start_x_m=float(fresh_pose["x_m"]),
                    fresh_start_y_m=float(fresh_pose["y_m"]),
                    fresh_start_yaw_rad=float(fresh_pose["yaw_rad"]),
                    route_csv=Path(sealed["route_csv"]),
                    diagnostics_json=Path(sealed["diagnostics_json"]),
                    additional_typed_run_required=(
                        startup_reseal_motion_authorization_json is None
                    ),
                    recovery_source_kind=str(
                        pending_startup_recovery["recovery_source_kind"]
                    ),
                )
            except Exception as exc:
                emit(
                    {
                        "schema_version": 1,
                        "event": "startup_reseal_failed",
                        "timestamp": effects.clock(),
                        "phase": "authorization_summary",
                        "leg_index": leg_index,
                        "startup_reseal_index": startup_reseal_index,
                        "rejected_run_id": pending_startup_recovery[
                            "rejected_run_id"
                        ],
                        "recovery_source_kind": pending_startup_recovery[
                            "recovery_source_kind"
                        ],
                        "failure": str(exc),
                        "motion_continues_authorized": False,
                    }
                )
                raise
            if startup_reseal_motion_authorization_json is not None:
                pending_startup_permit_context = StartupResealPermitContext(
                    mission_authorization_json=Path(
                        startup_reseal_motion_authorization_json
                    ).absolute(),
                    session_id=config.session_id,
                    semantic_map_id=config.semantic_map_id,
                    leg_index=leg_index,
                    target_viewpoint_id=target_viewpoint_id,
                    reseal_index=startup_reseal_index,
                    max_startup_reseals_per_leg=(
                        config.max_startup_reseals_per_leg
                    ),
                    rejected_run_id=str(
                        pending_startup_recovery["rejected_run_id"]
                    ),
                    rejected_semantic_log_path=Path(
                        pending_startup_recovery["rejected_semantic_log_path"]
                    ).absolute(),
                    startup_reseal_summary_path=summary_path,
                    fresh_localization_evidence_path=Path(
                        pending_startup_recovery[
                            "fresh_localization_evidence_path"
                        ]
                    ).absolute(),
                    permit_json_path=(
                        session_root
                        / "motion_authorization"
                        / "startup_reseals"
                        / f"{run_id}_permit.json"
                    ).absolute(),
                    recovery_source_kind=str(
                        pending_startup_recovery["recovery_source_kind"]
                    ),
                )
            emit(
                {
                    "schema_version": 1,
                    "event": "startup_reseal_route_sealed",
                    "timestamp": effects.clock(),
                    "leg_index": leg_index,
                    "startup_reseal_index": startup_reseal_index,
                    "replacement_run_id": run_id,
                    "rejected_run_id": pending_startup_recovery[
                        "rejected_run_id"
                    ],
                    "recovery_source_kind": pending_startup_recovery[
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
                        pending_startup_permit_context is not None
                    ),
                    "additional_typed_run_required": (
                        pending_startup_permit_context is None
                    ),
                    "motion_continues_authorized": False,
                }
            )

        outcome = effects.run_motion_leg(
            profile=profile,
            sealed=sealed,
            run_id=run_id,
            session_root=session_root,
            execute=True,
            coverage_plan=plan_path,
            coverage_transient_replan={
                "survey_root": survey_root,
                "session_root": session_root,
                "map_yaml": config.map_yaml,
                "semantic_map_id": config.semantic_map_id,
                "target_viewpoint_id": target_viewpoint_id,
                "robot_radius_m": config.robot_radius_m,
                "max_replans": config.max_blockage_replans_per_leg,
                "leg_index": leg_index,
                "resume_state_json": transient_overlay_resume_state_path,
            },
            require_fresh_confirmation=fresh_confirmation_reason is not None,
            fresh_confirmation_reason=fresh_confirmation_reason or "startup",
            fresh_localization_evidence_path=fresh_localization_evidence_path,
            uncertainty_map_yaml=(
                config.map_yaml if localization_branch_proof_id else None
            ),
            uncertainty_sigma_multiplier=config.uncertainty_sigma_multiplier,
            localization_branch_proof_id=localization_branch_proof_id,
            runtime_localization_permit_context=pending_runtime_permit_context,
            startup_reseal_permit_context=pending_startup_permit_context,
            mission_leg_permit_context=(
                None
                if (
                    mission_leg_motion_authorization_json is None
                    or fresh_confirmation_reason is not None
                    or pending_runtime_permit_context is not None
                )
                else MissionLegPermitContext(
                    mission_authorization_json=Path(
                        mission_leg_motion_authorization_json
                    ).absolute(),
                    session_id=config.session_id,
                    semantic_map_id=config.semantic_map_id,
                    mission_leg_kind=MissionLegKind.COVERAGE,
                    mission_leg_index=leg_index,
                    target_id=target_viewpoint_id,
                    permit_json_path=(
                        session_root
                        / "motion_authorization"
                        / "mission_legs"
                        / f"{run_id}_permit.json"
                    ).absolute(),
                )
            ),
        )
        if outcome.motion_authorization_permit_path is not None:
            emit(
                {
                    "schema_version": 1,
                    "event": "runtime_localization_motion_permit_issued",
                    "timestamp": effects.clock(),
                    "leg_index": leg_index,
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
            if pending_startup_permit_context is None:
                raise RuntimeError(
                    "startup reseal child reported a permit outside an exact "
                    "recovery context"
                )
            emit(
                {
                    "schema_version": 1,
                    "event": "startup_reseal_motion_permit_issued",
                    "timestamp": effects.clock(),
                    "leg_index": leg_index,
                    "run_id": outcome.run_id,
                    "startup_reseal_index": startup_reseal_index,
                    "recovery_source_kind": (
                        pending_startup_permit_context.recovery_source_kind
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
        if outcome.status == "completed":
            return outcome

        readiness_decision = evaluate_localization_readiness_retry(
            status=outcome.status,
            stop_reason=outcome.stop_reason,
            stop_details=outcome.stop_details,
            motion_published=outcome.motion_published,
        )
        if readiness_decision.retryable:
            maximum_readiness_retries = (
                config.max_localization_readiness_retries_per_leg
            )
            if localization_readiness_retry_index >= maximum_readiness_retries:
                emit(
                    {
                        "schema_version": 1,
                        "event": "localization_readiness_retry_exhausted",
                        "timestamp": effects.clock(),
                        "leg_index": leg_index,
                        "target_viewpoint_id": target_viewpoint_id,
                        "rejected_run_id": outcome.run_id,
                        "completed_retry_count": localization_readiness_retry_index,
                        "maximum_retry_count": maximum_readiness_retries,
                        "stop_reason": outcome.stop_reason,
                        "motion_published": False,
                        "motion_continues_authorized": False,
                        "fail_closed": True,
                    }
                )
                raise RuntimeError(
                    "pre-motion localization readiness retry budget exhausted "
                    f"for coverage leg {leg_index}: {outcome.stop_reason}"
                )
            localization_readiness_retry_index += 1
            emit(
                {
                    "schema_version": 1,
                    "event": "localization_readiness_retry_scheduled",
                    "timestamp": effects.clock(),
                    "leg_index": leg_index,
                    "target_viewpoint_id": target_viewpoint_id,
                    "rejected_run_id": outcome.run_id,
                    "next_retry_index": localization_readiness_retry_index,
                    "maximum_retry_count": maximum_readiness_retries,
                    "reason": readiness_decision.reason,
                    "stop_reason": outcome.stop_reason,
                    "motion_published": False,
                    "motion_continues_authorized": False,
                    "fresh_nomotion_amcl_preflight_required": True,
                    "route_limits_unchanged": True,
                }
            )
            continue

        pending_runtime_permit_context = None
        pending_startup_permit_context = None
        pending_startup_recovery = None
        prestart_localization_decision = evaluate_prestart_localization_reseal(
            status=outcome.status,
            motion_published=outcome.motion_published,
            stop_details=outcome.stop_details,
        )
        prestart_localization_admitted = (
            prestart_localization_decision.eligible
            and isinstance(outcome.stop_details, Mapping)
            and outcome.stop_reason == outcome.stop_details.get("reason")
        )
        startup_pose_mismatch = replanning.is_resealable_startup_mismatch(
            outcome
        )
        if startup_pose_mismatch or prestart_localization_admitted:
            if prestart_localization_admitted:
                if (
                    prestart_localization_decision.motion_published is not False
                    or not prestart_localization_decision.requires_fresh_localization
                    or not prestart_localization_decision.requires_new_route_certificate
                    or prestart_localization_decision.automatic_motion_authorized
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
            if startup_reseal_index >= config.max_startup_reseals_per_leg:
                raise RuntimeError(
                    "startup reseal budget exhausted for coverage leg "
                    f"{leg_index}: {outcome.stop_reason}"
                )
            startup_reseal_index += 1
            localization_readiness_retry_index = 0
            fresh_localization_evidence_path = (
                session_root
                / "preflight"
                / "startup_reseals"
                / (
                    f"coverage_leg_{leg_index:03d}_startup_reseal_"
                    f"{startup_reseal_index:03d}.json"
                )
            )
            startup_event_base = {
                "leg_index": leg_index,
                "startup_reseal_index": startup_reseal_index,
                "target_viewpoint_id": target_viewpoint_id,
                "rejected_run_id": outcome.run_id,
                "rejected_stop_details": outcome.stop_details,
                "recovery_source_kind": recovery_source_kind,
                "fresh_localization_evidence_json": str(
                    fresh_localization_evidence_path
                ),
                "covered_by_initial_mission_run": (
                    startup_reseal_motion_authorization_json is not None
                ),
                "additional_typed_run_required": (
                    startup_reseal_motion_authorization_json is None
                ),
                "motion_continues_authorized": False,
            }
            if rejected_route_pose is not None:
                startup_event_base["rejected_route_pose"] = rejected_route_pose
            else:
                startup_event_base[
                    "prestart_localization_reseal_decision"
                ] = prestart_localization_decision.to_evidence()
            emit(
                {
                    **startup_event_base,
                    "schema_version": 1,
                    "event": "startup_reseal_started",
                    "timestamp": effects.clock(),
                    "source_rejection_published_motion": False,
                }
            )
            try:
                admitted_pose = effects.admit_preplanning_localization(
                    config.runtime,
                    session_root,
                    evidence_path=fresh_localization_evidence_path,
                )
            except Exception as exc:
                emit(
                    {
                        **startup_event_base,
                        "schema_version": 1,
                        "event": "startup_reseal_failed",
                        "timestamp": effects.clock(),
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
            emit(
                {
                    **startup_event_base,
                    "schema_version": 1,
                    "event": "startup_localization_admitted",
                    "timestamp": effects.clock(),
                    "fresh_start_pose": fresh_start_pose,
                }
            )
            reseal_root = (
                survey_root
                / "startup_reseals"
                / (
                    f"leg_{leg_index:03d}"
                    f"_startup_reseal_{startup_reseal_index:03d}"
                )
            )
            if transient_overlay_resume_state is None:
                replanned = effects.replan_startup_source(
                    map_yaml=config.map_yaml,
                    semantic_map_id=config.semantic_map_id,
                    survey_root=survey_root,
                    plan_path=plan_path,
                    expected_target_viewpoint_id=target_viewpoint_id,
                    current_pose=admitted_pose,
                    rejected_outcome=outcome,
                    reseal_index=startup_reseal_index,
                    output_dir=reseal_root,
                )
                transient_overlay_resume_state_path = None
                transient_overlay_resume_state_digest = ""
            else:
                if coverage_plan is None:
                    coverage_plan = effects.load_coverage_plan(plan_path)
                (
                    replanned,
                    transient_overlay_resume_state,
                    transient_overlay_resume_state_path,
                    transient_overlay_resume_state_digest,
                ) = effects.replan_source_preserving_transient_overlay(
                    state=transient_overlay_resume_state,
                    plan=coverage_plan,
                    map_yaml=config.map_yaml,
                    semantic_map_id=config.semantic_map_id,
                    survey_root=survey_root,
                    target_viewpoint_id=target_viewpoint_id,
                    current_pose=admitted_pose,
                    rejected_outcome=outcome,
                    output_dir=reseal_root,
                    robot_radius_m=config.robot_radius_m,
                    recovery_kind="startup",
                    artifact_root=session_root,
                )
            emit(
                {
                    "schema_version": 1,
                    "event": (
                        "startup_pose_route_resealed"
                        if startup_pose_mismatch
                        else "prestart_localization_route_resealed"
                    ),
                    "timestamp": effects.clock(),
                    "leg_index": leg_index,
                    "startup_reseal_index": startup_reseal_index,
                    "recovery_source_kind": recovery_source_kind,
                    "rejected_run_id": outcome.run_id,
                    "rejected_stop_details": outcome.stop_details,
                    "replacement_route_csv": replanned["route_csv"],
                    "replacement_diagnostics_json": replanned[
                        "diagnostics_json"
                    ],
                    "replacement_summary_json": replanned["summary_json"],
                    "dynamic_overlay_preserved": (
                        transient_overlay_resume_state is not None
                    ),
                    "adopted_blockage_replan_count": (
                        0
                        if transient_overlay_resume_state is None
                        else transient_overlay_resume_state.completed_replan_count
                    ),
                    "remaining_blockage_replan_count": (
                        config.max_blockage_replans_per_leg
                        if transient_overlay_resume_state is None
                        else transient_overlay_resume_state.remaining_replans
                    ),
                    "transient_overlay_resume_state_json": (
                        ""
                        if transient_overlay_resume_state_path is None
                        else str(transient_overlay_resume_state_path)
                    ),
                    "transient_overlay_resume_state_sha256": (
                        transient_overlay_resume_state_digest
                    ),
                    "fresh_confirmation_required": (
                        startup_reseal_motion_authorization_json is None
                    ),
                    "covered_by_initial_mission_run": (
                        startup_reseal_motion_authorization_json is not None
                    ),
                    "additional_typed_run_required": (
                        startup_reseal_motion_authorization_json is None
                    ),
                }
            )
            pending_startup_recovery = {
                "rejected_run_id": outcome.run_id,
                "rejected_semantic_log_path": outcome.semantic_log_path,
                "fresh_start_pose": fresh_start_pose,
                "fresh_localization_evidence_path": (
                    fresh_localization_evidence_path
                ),
                "recovery_source_kind": recovery_source_kind,
            }
            source_route = Path(replanned["route_csv"])
            source_diagnostics = Path(replanned["diagnostics_json"])
            fresh_confirmation_reason = "startup"
            fresh_localization_evidence_path = None
            continue

        if _claims_prestart_localization_phase(outcome.stop_details):
            emit(
                {
                    "schema_version": 1,
                    "event": "prestart_localization_reseal_rejected",
                    "timestamp": effects.clock(),
                    "leg_index": leg_index,
                    "target_viewpoint_id": target_viewpoint_id,
                    "rejected_run_id": outcome.run_id,
                    "stop_reason": outcome.stop_reason,
                    "stop_details": outcome.stop_details,
                    "motion_published": outcome.motion_published,
                    "prestart_localization_reseal_decision": (
                        prestart_localization_decision.to_evidence()
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

        runtime_localization_decision = evaluate_runtime_localization_reseal(
            status=outcome.status,
            motion_published=outcome.motion_published,
            stop_details=outcome.stop_details,
        )
        if runtime_localization_decision.eligible:
            budget = evaluate_runtime_localization_reseal_budget(
                completed_reseal_count=runtime_localization_reseal_index,
                maximum_reseal_count=(
                    config.max_runtime_localization_reseals_per_leg
                ),
            )
            if not budget.allowed:
                emit(
                    {
                        "schema_version": 1,
                        "event": "runtime_localization_reseal_rejected",
                        "timestamp": effects.clock(),
                        "leg_index": leg_index,
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
                    f"coverage leg {leg_index}: {outcome.stop_reason}"
                )
            try:
                transient_overlay_resume_state = (
                    effects.advance_transient_overlay_resume_state(
                        outcome=outcome,
                        previous_state=transient_overlay_resume_state,
                        plan_path=plan_path,
                        leg_index=leg_index,
                        target_viewpoint_id=target_viewpoint_id,
                        max_replans=config.max_blockage_replans_per_leg,
                        require_uncertainty_admission=bool(
                            localization_branch_proof_id
                        ),
                        artifact_root=session_root,
                        survey_root=survey_root,
                    )
                )
                if transient_overlay_resume_state is not None and coverage_plan is None:
                    coverage_plan = effects.load_coverage_plan(plan_path)
            except Exception as exc:
                emit(
                    {
                        "schema_version": 1,
                        "event": "runtime_localization_reseal_rejected",
                        "timestamp": effects.clock(),
                        "leg_index": leg_index,
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
            runtime_localization_reseal_index = budget.next_reseal_index
            localization_readiness_retry_index = 0
            fresh_localization_evidence_path = (
                session_root
                / "preflight"
                / "runtime_localization_reseals"
                / (
                    f"coverage_leg_{leg_index:03d}"
                    "_runtime_localization_reseal_"
                    f"{runtime_localization_reseal_index:03d}.json"
                )
            )
            recovery_event_base = {
                "leg_index": leg_index,
                "runtime_localization_reseal_index": (
                    runtime_localization_reseal_index
                ),
                "rejected_run_id": outcome.run_id,
                "rejected_stop_details": outcome.stop_details,
                "fresh_localization_evidence_json": str(
                    fresh_localization_evidence_path
                ),
                "runtime_localization_reseal_decision": (
                    runtime_localization_decision.to_evidence()
                ),
                "runtime_localization_reseal_budget": budget.to_evidence(),
                "fresh_confirmation_required": (
                    mission_motion_authorization_json is None
                ),
                "covered_by_initial_mission_run": (
                    mission_motion_authorization_json is not None
                ),
                "additional_typed_run_required": (
                    mission_motion_authorization_json is None
                ),
                "dynamic_overlay_preserved": (
                    transient_overlay_resume_state is not None
                ),
                "adopted_blockage_replan_count": (
                    0
                    if transient_overlay_resume_state is None
                    else transient_overlay_resume_state.completed_replan_count
                ),
                "remaining_blockage_replan_count": (
                    config.max_blockage_replans_per_leg
                    if transient_overlay_resume_state is None
                    else transient_overlay_resume_state.remaining_replans
                ),
                "motion_continues_authorized": False,
            }
            emit(
                {
                    **recovery_event_base,
                    "schema_version": 1,
                    "event": "runtime_localization_reseal_started",
                    "timestamp": effects.clock(),
                    "source_stop_requires_zero_cycle": True,
                }
            )
            try:
                admitted_pose = effects.admit_preplanning_localization(
                    config.runtime,
                    session_root,
                    evidence_path=fresh_localization_evidence_path,
                )
            except Exception as exc:
                emit(
                    {
                        **recovery_event_base,
                        "schema_version": 1,
                        "event": "runtime_localization_reseal_failed",
                        "timestamp": effects.clock(),
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
            emit(
                {
                    **recovery_event_base,
                    "schema_version": 1,
                    "event": "runtime_localization_admitted",
                    "timestamp": effects.clock(),
                    "fresh_start_pose": fresh_start_pose,
                }
            )
            reseal_root = (
                survey_root
                / "runtime_localization_reseals"
                / (
                    f"leg_{leg_index:03d}"
                    "_runtime_localization_reseal_"
                    f"{runtime_localization_reseal_index:03d}"
                )
            )
            try:
                if transient_overlay_resume_state is None:
                    replanned = effects.replan_runtime_localization_source(
                        map_yaml=config.map_yaml,
                        semantic_map_id=config.semantic_map_id,
                        survey_root=survey_root,
                        plan_path=plan_path,
                        expected_target_viewpoint_id=target_viewpoint_id,
                        current_pose=admitted_pose,
                        rejected_outcome=outcome,
                        reseal_index=runtime_localization_reseal_index,
                        output_dir=reseal_root,
                    )
                    transient_overlay_resume_state_path = None
                    transient_overlay_resume_state_digest = ""
                else:
                    if coverage_plan is None:
                        coverage_plan = effects.load_coverage_plan(plan_path)
                    (
                        replanned,
                        transient_overlay_resume_state,
                        transient_overlay_resume_state_path,
                        transient_overlay_resume_state_digest,
                    ) = effects.replan_source_preserving_transient_overlay(
                        state=transient_overlay_resume_state,
                        plan=coverage_plan,
                        map_yaml=config.map_yaml,
                        semantic_map_id=config.semantic_map_id,
                        survey_root=survey_root,
                        target_viewpoint_id=target_viewpoint_id,
                        current_pose=admitted_pose,
                        rejected_outcome=outcome,
                        output_dir=reseal_root,
                        robot_radius_m=config.robot_radius_m,
                        recovery_kind="runtime_localization",
                        artifact_root=session_root,
                    )
            except Exception as exc:
                emit(
                    {
                        **recovery_event_base,
                        "schema_version": 1,
                        "event": "runtime_localization_reseal_failed",
                        "timestamp": effects.clock(),
                        "phase": "same_target_route_replan",
                        "failure": str(exc),
                    }
                )
                raise
            emit(
                {
                    **recovery_event_base,
                    "schema_version": 1,
                    "event": "runtime_localization_route_replanned",
                    "timestamp": effects.clock(),
                    "fresh_start_pose": fresh_start_pose,
                    "replacement_route_csv": replanned["route_csv"],
                    "replacement_diagnostics_json": replanned[
                        "diagnostics_json"
                    ],
                    "replacement_summary_json": replanned["summary_json"],
                    "dynamic_overlay_preserved": (
                        transient_overlay_resume_state is not None
                    ),
                    "adopted_blockage_replan_count": (
                        0
                        if transient_overlay_resume_state is None
                        else transient_overlay_resume_state.completed_replan_count
                    ),
                    "remaining_blockage_replan_count": (
                        config.max_blockage_replans_per_leg
                        if transient_overlay_resume_state is None
                        else transient_overlay_resume_state.remaining_replans
                    ),
                    "transient_overlay_resume_state_json": (
                        ""
                        if transient_overlay_resume_state_path is None
                        else str(transient_overlay_resume_state_path)
                    ),
                    "transient_overlay_resume_state_sha256": (
                        transient_overlay_resume_state_digest
                    ),
                    "committed_target_viewpoint_id": target_viewpoint_id,
                }
            )
            pending_runtime_route_seal = {
                **recovery_event_base,
                "fresh_start_pose": fresh_start_pose,
                "committed_target_viewpoint_id": target_viewpoint_id,
                "replacement_source_route_csv": replanned["route_csv"],
                "replacement_source_diagnostics_json": replanned[
                    "diagnostics_json"
                ],
                "replacement_summary_json": replanned["summary_json"],
            }
            if mission_motion_authorization_json is not None:
                pending_runtime_permit_context = RuntimeLocalizationPermitContext(
                    mission_authorization_json=Path(
                        mission_motion_authorization_json
                    ).absolute(),
                    session_id=config.session_id,
                    leg_index=leg_index,
                    target_viewpoint_id=target_viewpoint_id,
                    reseal_index=runtime_localization_reseal_index,
                    max_runtime_reseals_per_leg=(
                        config.max_runtime_localization_reseals_per_leg
                    ),
                    rejected_run_id=outcome.run_id,
                    runtime_reseal_decision_evidence=(
                        runtime_localization_decision.to_evidence()
                    ),
                    fresh_localization_evidence_path=(
                        fresh_localization_evidence_path
                    ),
                    permit_json_path=(
                        session_root
                        / "motion_authorization"
                        / (
                            f"{config.session_id}_coverage_"
                            f"{leg_index:03d}_runtime_localization_"
                            f"reseal_{runtime_localization_reseal_index:03d}_"
                            "permit.json"
                        )
                    ).absolute(),
                )
            source_route = Path(replanned["route_csv"])
            source_diagnostics = Path(replanned["diagnostics_json"])
            fresh_confirmation_reason = "runtime_localization"
            continue

        _require_completed_motion(outcome)


__all__ = [
    "CoverageLegConfig",
    "CoverageLegEffects",
    "MissionLegPermitContext",
    "RuntimeLocalizationPermitContext",
    "execute_coverage_leg_with_replans",
]
