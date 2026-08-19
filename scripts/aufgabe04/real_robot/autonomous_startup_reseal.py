"""Autonomous adapter for one bounded startup-reseal motion permit.

The navigation-level authorization modules own the immutable schema and the
atomic one-use claim.  This ROS-free adapter owns only the parent/coverage
context and seals the exact artifacts produced by a replacement dry run.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from scripts.aufgabe04.navigation.startup_reseal_motion_authorization import (
    StartupResealMotionPermit,
    file_sha256,
    load_startup_reseal_motion_authorization,
    startup_reseal_motion_authorization_sha256,
    write_startup_reseal_motion_permit,
)
from scripts.aufgabe04.real_robot.autonomous_artifact_paths import (
    resolve_normal_artifact_path,
)


@dataclass(frozen=True)
class StartupResealPermitContext:
    """Exact no-motion rejection and replacement identity for one permit."""

    mission_authorization_json: Path
    session_id: str
    semantic_map_id: str
    leg_index: int
    target_viewpoint_id: str
    reseal_index: int
    max_startup_reseals_per_leg: int
    rejected_run_id: str
    rejected_semantic_log_path: Path
    startup_reseal_summary_path: Path
    fresh_localization_evidence_path: Path
    permit_json_path: Path

    def __post_init__(self) -> None:
        if not self.session_id.strip() or not self.semantic_map_id.strip():
            raise ValueError("startup reseal session and map IDs must be non-empty")
        if not self.target_viewpoint_id.strip() or not self.rejected_run_id.strip():
            raise ValueError("startup reseal target and rejected run must be non-empty")
        if type(self.leg_index) is not int or self.leg_index < 0:
            raise ValueError("startup reseal leg_index must be non-negative")
        if type(self.reseal_index) is not int or self.reseal_index <= 0:
            raise ValueError("startup reseal reseal_index must be positive")
        if (
            type(self.max_startup_reseals_per_leg) is not int
            or self.max_startup_reseals_per_leg <= 0
            or self.reseal_index > self.max_startup_reseals_per_leg
        ):
            raise ValueError("startup reseal permit exceeds its bounded budget")


def write_startup_reseal_permit_summary(
    path: Path,
    *,
    leg_index: int,
    target_viewpoint_id: str,
    reseal_index: int,
    rejected_run_id: str,
    fresh_start_x_m: float,
    fresh_start_y_m: float,
    fresh_start_yaw_rad: float,
    route_csv: Path,
    diagnostics_json: Path,
    additional_typed_run_required: bool = False,
) -> Path:
    """Write the exact sealed-route summary later bound by the permit."""

    if type(leg_index) is not int or leg_index < 0:
        raise ValueError("startup reseal leg_index must be non-negative")
    if type(reseal_index) is not int or reseal_index <= 0:
        raise ValueError("startup reseal reseal_index must be positive")
    if not target_viewpoint_id.strip() or not rejected_run_id.strip():
        raise ValueError("startup reseal target and rejected run must be non-empty")
    if type(additional_typed_run_required) is not bool:
        raise ValueError("additional_typed_run_required must be boolean")
    pose = (fresh_start_x_m, fresh_start_y_m, fresh_start_yaw_rad)
    if any(isinstance(value, bool) or not math.isfinite(value) for value in pose):
        raise ValueError("startup reseal fresh pose must be finite")
    route_path = resolve_normal_artifact_path(
        route_csv,
        label="startup reseal summary route CSV",
    )
    diagnostics_path = resolve_normal_artifact_path(
        diagnostics_json,
        label="startup reseal summary diagnostics JSON",
    )
    destination = Path(path).resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "startup_route_replanned",
        "motion_published": False,
        "reseal_kind": "startup",
        "leg_index": leg_index,
        "startup_reseal_index": reseal_index,
        "rejected_run_id": rejected_run_id,
        "target_viewpoint_id": target_viewpoint_id,
        "fresh_start_pose": {
            "x_m": fresh_start_x_m,
            "y_m": fresh_start_y_m,
            "yaw_rad": fresh_start_yaw_rad,
        },
        "route_csv": str(route_path),
        "diagnostics_json": str(diagnostics_path),
        "same_target_verified": True,
        "additional_typed_run_required": additional_typed_run_required,
    }
    try:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise ValueError(
            f"refusing to reuse startup reseal summary: {destination}"
        ) from exc
    return destination


def issue_startup_reseal_motion_permit(
    *,
    context: StartupResealPermitContext,
    run_id: str,
    route_csv: Path,
    diagnostics_json: Path,
    map_route_certificate_json: Path,
    dry_preflight_json: Path,
    dry_odom_certificate_json: Path,
    dry_uncertainty_budget_json: Path,
) -> tuple[Path, str]:
    """Publish one exact replacement permit after the dry run has passed."""

    if not isinstance(context, StartupResealPermitContext):
        raise TypeError("context must be a StartupResealPermitContext")
    if not str(run_id).strip() or run_id == context.rejected_run_id:
        raise ValueError("startup reseal replacement run_id must be new")

    master_path = resolve_normal_artifact_path(
        context.mission_authorization_json,
        label="startup reseal motion authorization",
    )
    master = load_startup_reseal_motion_authorization(master_path)
    if master.session_id != context.session_id:
        raise ValueError("startup reseal authorization session mismatch")
    if master.semantic_map_id != context.semantic_map_id:
        raise ValueError("startup reseal authorization semantic map mismatch")
    if (
        master.max_startup_reseals_per_leg
        != context.max_startup_reseals_per_leg
    ):
        raise ValueError("startup reseal authorization budget mismatch")

    def sealed(path: Path, label: str) -> tuple[str, str]:
        canonical = resolve_normal_artifact_path(path, label=label)
        return str(canonical), file_sha256(canonical)

    rejected_log_path, rejected_log_sha256 = sealed(
        context.rejected_semantic_log_path,
        "startup reseal rejected semantic log",
    )
    startup_summary_path, startup_summary_sha256 = sealed(
        context.startup_reseal_summary_path,
        "startup reseal summary",
    )
    fresh_path, fresh_sha256 = sealed(
        context.fresh_localization_evidence_path,
        "startup reseal fresh localization evidence",
    )
    route_path, route_sha256 = sealed(route_csv, "startup reseal route CSV")
    diagnostics_path, diagnostics_sha256 = sealed(
        diagnostics_json,
        "startup reseal diagnostics JSON",
    )
    certificate_path, certificate_sha256 = sealed(
        map_route_certificate_json,
        "startup reseal map-route certificate",
    )
    dry_preflight_path, dry_preflight_sha256 = sealed(
        dry_preflight_json,
        "startup reseal dry preflight",
    )
    dry_certificate_path, dry_certificate_sha256 = sealed(
        dry_odom_certificate_json,
        "startup reseal dry odom certificate",
    )
    dry_budget_path, dry_budget_sha256 = sealed(
        dry_uncertainty_budget_json,
        "startup reseal dry uncertainty budget",
    )
    permit = StartupResealMotionPermit(
        master_authorization_sha256=(
            startup_reseal_motion_authorization_sha256(master)
        ),
        master_authorization_path=str(master_path),
        run_id=run_id,
        leg_index=context.leg_index,
        target_viewpoint_id=context.target_viewpoint_id,
        reseal_index=context.reseal_index,
        max_startup_reseals_per_leg=context.max_startup_reseals_per_leg,
        rejected_run_id=context.rejected_run_id,
        rejected_semantic_log_path=rejected_log_path,
        rejected_semantic_log_sha256=rejected_log_sha256,
        startup_reseal_summary_path=startup_summary_path,
        startup_reseal_summary_sha256=startup_summary_sha256,
        fresh_stationary_localization_evidence_path=fresh_path,
        fresh_stationary_localization_evidence_sha256=fresh_sha256,
        route_csv_path=route_path,
        route_csv_sha256=route_sha256,
        diagnostics_path=diagnostics_path,
        diagnostics_sha256=diagnostics_sha256,
        map_route_certificate_path=certificate_path,
        map_route_certificate_sha256=certificate_sha256,
        dry_preflight_path=dry_preflight_path,
        dry_preflight_sha256=dry_preflight_sha256,
        dry_odom_certificate_path=dry_certificate_path,
        dry_odom_certificate_sha256=dry_certificate_sha256,
        dry_uncertainty_budget_path=dry_budget_path,
        dry_uncertainty_budget_sha256=dry_budget_sha256,
        same_target_verified=True,
        rejected_motion_published=False,
        dry_run_passed=True,
        additional_typed_run_required=False,
    )
    permit_path = Path(context.permit_json_path).resolve(strict=False)
    permit_sha256 = write_startup_reseal_motion_permit(permit_path, permit)
    return permit_path, permit_sha256


__all__ = [
    "StartupResealPermitContext",
    "issue_startup_reseal_motion_permit",
    "write_startup_reseal_permit_summary",
]
