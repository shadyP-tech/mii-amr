"""Generic permit context and issuer for runtime-localization recovery.

This module is ROS-free and routine-neutral.  Coverage and candidate
orchestrators provide a committed mission-leg identity plus fresh route
artifacts; the sole motion adapter uses this context only after its dry run
has passed.  Importing the module cannot launch a process, prompt an operator,
or publish velocity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
    MissionLegKind,
)
from scripts.aufgabe04.navigation.execution.runtime_motion_authorization import (
    RuntimeLocalizationMotionPermit,
    file_sha256,
    load_mission_motion_authorization,
    mission_motion_authorization_sha256,
    resolve_runtime_localization_mission_leg_identity,
    write_runtime_localization_motion_permit,
)
from scripts.aufgabe04.real_robot.autonomous_artifact_paths import (
    resolve_normal_artifact_path,
)


@dataclass(frozen=True)
class RuntimeLocalizationPermitContext:
    """Exact mission scope needed to authorize one recovery child run.

    ``leg_index`` and ``target_viewpoint_id`` remain persisted compatibility
    aliases.  The generic routine identity is authoritative and must match the
    aliases exactly.
    """

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
    mission_leg_kind: MissionLegKind = MissionLegKind.COVERAGE
    mission_leg_index: int | None = None
    target_id: str = ""
    semantic_map_id: str = ""

    def __post_init__(self) -> None:
        kind, index, target = resolve_runtime_localization_mission_leg_identity(
            mission_leg_kind=self.mission_leg_kind,
            mission_leg_index=self.mission_leg_index,
            target_id=self.target_id,
            leg_index=self.leg_index,
            target_viewpoint_id=self.target_viewpoint_id,
        )
        object.__setattr__(self, "mission_leg_kind", kind)
        object.__setattr__(self, "mission_leg_index", index)
        object.__setattr__(self, "target_id", target)
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("runtime localization session_id must be non-empty")
        if (
            type(self.reseal_index) is not int
            or self.reseal_index <= 0
        ):
            raise ValueError(
                "runtime localization reseal_index must be positive"
            )
        if (
            type(self.max_runtime_reseals_per_leg) is not int
            or self.max_runtime_reseals_per_leg <= 0
            or self.reseal_index > self.max_runtime_reseals_per_leg
        ):
            raise ValueError("runtime localization reseal budget exhausted")
        if not isinstance(self.rejected_run_id, str) or not self.rejected_run_id.strip():
            raise ValueError(
                "runtime localization rejected_run_id must be non-empty"
            )
        if not isinstance(self.runtime_reseal_decision_evidence, dict):
            raise TypeError(
                "runtime localization decision evidence must be a dict"
            )
        if not isinstance(self.semantic_map_id, str):
            raise TypeError("runtime localization semantic_map_id must be a string")


def resolved_runtime_localization_semantic_map_id(
    context: RuntimeLocalizationPermitContext,
) -> str:
    """Return the master-bound semantic map, rejecting a supplied mismatch."""

    master_path = resolve_normal_artifact_path(
        context.mission_authorization_json,
        label="runtime localization mission authorization",
    )
    authorization = load_mission_motion_authorization(master_path)
    expected = authorization.semantic_map_id
    supplied = context.semantic_map_id.strip()
    if supplied and supplied != expected:
        raise ValueError(
            "runtime localization permit context semantic_map_id mismatch"
        )
    return expected


def issue_runtime_localization_motion_permit(
    *,
    context: RuntimeLocalizationPermitContext,
    run_id: str,
    route_csv: Path,
    diagnostics_json: Path,
    map_route_certificate_json: Path,
    dry_preflight_json: Path,
    dry_odom_certificate_json: Path,
    dry_uncertainty_budget_json: Path,
) -> tuple[Path, str]:
    """Seal one exact routine recovery permit after its dry run passes."""

    if run_id == context.rejected_run_id:
        raise ValueError("runtime localization permit run_id must be new")
    master_path = resolve_normal_artifact_path(
        context.mission_authorization_json,
        label="runtime localization mission authorization",
    )
    master = load_mission_motion_authorization(master_path)
    if master.session_id != context.session_id:
        raise ValueError(
            "runtime localization permit context session_id mismatch"
        )
    if context.mission_leg_kind not in master.allowed_mission_leg_kinds:
        raise ValueError(
            "runtime localization permit mission leg kind is not authorized"
        )
    resolved_runtime_localization_semantic_map_id(context)
    decision_evidence = dict(context.runtime_reseal_decision_evidence)

    def sealed(path: Path, label: str) -> tuple[str, str]:
        canonical = resolve_normal_artifact_path(path, label=label)
        return str(canonical), file_sha256(canonical)

    fresh_path, fresh_sha256 = sealed(
        context.fresh_localization_evidence_path,
        "runtime localization fresh localization evidence",
    )
    route_path, route_sha256 = sealed(
        route_csv, "runtime localization route CSV"
    )
    diagnostics_path, diagnostics_sha256 = sealed(
        diagnostics_json, "runtime localization diagnostics JSON"
    )
    map_certificate_path, map_certificate_sha256 = sealed(
        map_route_certificate_json,
        "runtime localization map-route certificate",
    )
    dry_preflight_path, dry_preflight_sha256 = sealed(
        dry_preflight_json, "runtime localization dry preflight"
    )
    dry_certificate_path, dry_certificate_sha256 = sealed(
        dry_odom_certificate_json,
        "runtime localization dry odom certificate",
    )
    dry_budget_path, dry_budget_sha256 = sealed(
        dry_uncertainty_budget_json,
        "runtime localization dry uncertainty budget",
    )
    permit = RuntimeLocalizationMotionPermit(
        master_authorization_sha256=mission_motion_authorization_sha256(master),
        master_authorization_path=str(master_path),
        run_id=run_id,
        leg_index=context.leg_index,
        target_viewpoint_id=context.target_viewpoint_id,
        mission_leg_kind=context.mission_leg_kind,
        mission_leg_index=context.mission_leg_index,
        target_id=context.target_id,
        reseal_index=context.reseal_index,
        max_runtime_reseals_per_leg=context.max_runtime_reseals_per_leg,
        rejected_run_id=context.rejected_run_id,
        runtime_reseal_decision_evidence=decision_evidence,
        runtime_reseal_decision_sha256=payload_sha256(decision_evidence),
        fresh_localization_evidence_path=fresh_path,
        fresh_localization_evidence_sha256=fresh_sha256,
        route_csv_path=route_path,
        route_csv_sha256=route_sha256,
        diagnostics_path=diagnostics_path,
        diagnostics_sha256=diagnostics_sha256,
        map_route_certificate_path=map_certificate_path,
        map_route_certificate_sha256=map_certificate_sha256,
        dry_preflight_path=dry_preflight_path,
        dry_preflight_sha256=dry_preflight_sha256,
        dry_odom_certificate_path=dry_certificate_path,
        dry_odom_certificate_sha256=dry_certificate_sha256,
        dry_uncertainty_budget_path=dry_budget_path,
        dry_uncertainty_budget_sha256=dry_budget_sha256,
        same_target_verified=True,
        dry_run_passed=True,
        additional_typed_run_required=False,
    )
    permit_path = Path(context.permit_json_path).resolve(strict=False)
    permit_sha256 = write_runtime_localization_motion_permit(
        permit_path,
        permit,
    )
    return permit_path, permit_sha256


__all__ = [
    "RuntimeLocalizationPermitContext",
    "issue_runtime_localization_motion_permit",
    "resolved_runtime_localization_semantic_map_id",
]
