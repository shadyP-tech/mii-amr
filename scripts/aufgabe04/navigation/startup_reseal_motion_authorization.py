"""ROS-free authorization for bounded pre-motion startup reseals.

A startup reseal is eligible only after an exact child run rejected either
its certified start pose or its prestart localization continuity before
motion.  The master authorization records the operator's mission ``RUN``
scope.  Each replacement run additionally needs an immutable permit binding
the exact recovery source, rejection log, motion-free reseal summary, fresh
stationary localization, replacement route certificates, and a passed dry
run.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Mapping

from scripts.aufgabe04.artifacts.content_store import (
    ContentStoreError,
    load_content_hashed_json,
    payload_sha256,
    write_content_hashed_json,
)
from scripts.aufgabe04.navigation.mission_leg_motion_permit import (
    ROUTINE_MISSION_LEG_KINDS,
    MissionLegKind,
)
from scripts.aufgabe04.navigation.startup_reseal_route_binding import (
    validate_startup_reseal_route_binding,
)


STARTUP_RESEAL_MOTION_AUTHORIZATION_SCHEMA_VERSION = 3
STARTUP_RESEAL_MOTION_PERMIT_SCHEMA_VERSION = 3
STARTUP_RESEAL_PERMIT_SUMMARY_SCHEMA_VERSION = 3
STARTUP_RESEAL_MOTION_AUTHORIZATION_HASH_FIELD = (
    "startup_reseal_motion_authorization_sha256"
)
STARTUP_RESEAL_MOTION_PERMIT_HASH_FIELD = (
    "startup_reseal_motion_permit_sha256"
)
STARTUP_RESEAL_RECOVERY_KIND = "startup_reseal"
STARTUP_RESEAL_RUN_CONFIRMATION = "RUN"
STARTUP_RESEAL_RECOVERY_SOURCE_CERTIFIED_START_POSE_MISMATCH = (
    "certified_start_pose_mismatch"
)
STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY = (
    "prestart_localization_continuity"
)
STARTUP_RESEAL_RECOVERY_SOURCE_KINDS = frozenset(
    {
        STARTUP_RESEAL_RECOVERY_SOURCE_CERTIFIED_START_POSE_MISMATCH,
        STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY,
    }
)
STARTUP_RESEAL_MOTION_AUTHORIZATION_SCOPE = (
    "Reuse this autonomous mission RUN only for bounded same-kind, same-leg, "
    "same-target pre-motion startup reseals after either a certified start "
    "pose mismatch or an admitted prestart localization-continuity stop. "
    "The no-motion rejection, exact recovery source, fresh stationary "
    "localization, replacement route certificates, and passed dry run must "
    "be bound by an exact one-use permit; post-motion recovery, generic "
    "stops, target or leg changes, and motion without a consumed permit are "
    "not authorized."
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema_version",
        "session_id",
        "robot_id",
        "namespace",
        "cmd_vel_topic",
        "semantic_map_id",
        "localization_branch_proof_id",
        "max_startup_reseals_per_leg",
        "scope_text",
        "operator_confirmation",
        "allowed_recovery_kind",
        "allowed_mission_leg_kinds",
    }
)
_PERMIT_FIELDS = frozenset(
    {
        "schema_version",
        "master_authorization_sha256",
        "master_authorization_path",
        "run_id",
        "leg_index",
        "target_viewpoint_id",
        "mission_leg_kind",
        "mission_leg_index",
        "target_id",
        "reseal_index",
        "max_startup_reseals_per_leg",
        "rejected_run_id",
        "rejected_semantic_log_path",
        "rejected_semantic_log_sha256",
        "startup_reseal_summary_path",
        "startup_reseal_summary_sha256",
        "fresh_stationary_localization_evidence_path",
        "fresh_stationary_localization_evidence_sha256",
        "route_csv_path",
        "route_csv_sha256",
        "diagnostics_path",
        "diagnostics_sha256",
        "map_route_certificate_path",
        "map_route_certificate_sha256",
        "dry_preflight_path",
        "dry_preflight_sha256",
        "dry_odom_certificate_path",
        "dry_odom_certificate_sha256",
        "dry_uncertainty_budget_path",
        "dry_uncertainty_budget_sha256",
        "same_target_verified",
        "rejected_motion_published",
        "dry_run_passed",
        "additional_typed_run_required",
        "recovery_source_kind",
    }
)


def _mission_leg_kind(value: object, name: str) -> MissionLegKind:
    try:
        kind = MissionLegKind(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not a known mission leg kind") from exc
    if kind not in ROUTINE_MISSION_LEG_KINDS:
        raise ValueError(f"{name} must be a routine mission leg kind")
    return kind


def _canonical_allowed_mission_leg_kinds(
    values: object,
) -> tuple[MissionLegKind, ...]:
    if not isinstance(values, (tuple, list)) or not values:
        raise ValueError(
            "allowed_mission_leg_kinds must be a non-empty sequence"
        )
    result: list[MissionLegKind] = []
    for value in values:
        kind = _mission_leg_kind(value, "allowed_mission_leg_kinds")
        if kind in result:
            raise ValueError("allowed_mission_leg_kinds contains duplicates")
        result.append(kind)
    order = {kind: index for index, kind in enumerate(ROUTINE_MISSION_LEG_KINDS)}
    if result != sorted(result, key=order.__getitem__):
        raise ValueError("allowed_mission_leg_kinds must use canonical order")
    return tuple(result)


def resolve_startup_reseal_mission_leg_identity(
    *,
    mission_leg_kind: MissionLegKind | str = MissionLegKind.COVERAGE,
    mission_leg_index: int | None = None,
    target_id: str = "",
    leg_index: int | None = None,
    target_viewpoint_id: str = "",
) -> tuple[MissionLegKind, int, str]:
    """Resolve generic identity with strict coverage-alias compatibility.

    ``leg_index`` and ``target_viewpoint_id`` are persisted compatibility
    aliases.  They never form a second identity: when generic values are also
    supplied, both representations must match exactly.
    """

    kind = _mission_leg_kind(mission_leg_kind, "mission_leg_kind")
    if mission_leg_index is None:
        mission_leg_index = leg_index
    elif leg_index is not None and mission_leg_index != leg_index:
        raise ValueError("mission_leg_index and leg_index mismatch")
    _nonnegative_integer(mission_leg_index, "mission_leg_index")

    if not str(target_id).strip():
        target_id = target_viewpoint_id
    elif str(target_viewpoint_id).strip() and target_id != target_viewpoint_id:
        raise ValueError("target_id and target_viewpoint_id mismatch")
    _require_nonempty(target_id, "target_id")
    return kind, mission_leg_index, target_id


@dataclass(frozen=True)
class StartupResealMotionAuthorization:
    """Master scope derived from the mission-level operator ``RUN``."""

    session_id: str
    robot_id: str
    namespace: str
    cmd_vel_topic: str
    semantic_map_id: str
    localization_branch_proof_id: str
    max_startup_reseals_per_leg: int
    scope_text: str
    operator_confirmation: str
    allowed_recovery_kind: str
    allowed_mission_leg_kinds: tuple[MissionLegKind, ...] = (
        MissionLegKind.COVERAGE,
    )
    schema_version: int = STARTUP_RESEAL_MOTION_AUTHORIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_mission_leg_kinds",
            _canonical_allowed_mission_leg_kinds(
                self.allowed_mission_leg_kinds
            ),
        )
        _validate_authorization(self)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "robot_id": self.robot_id,
            "namespace": self.namespace,
            "cmd_vel_topic": self.cmd_vel_topic,
            "semantic_map_id": self.semantic_map_id,
            "localization_branch_proof_id": self.localization_branch_proof_id,
            "max_startup_reseals_per_leg": self.max_startup_reseals_per_leg,
            "scope_text": self.scope_text,
            "operator_confirmation": self.operator_confirmation,
            "allowed_recovery_kind": self.allowed_recovery_kind,
            "allowed_mission_leg_kinds": [
                kind.value for kind in self.allowed_mission_leg_kinds
            ],
        }

    def to_evidence(self) -> dict[str, object]:
        return self.to_payload()


@dataclass(frozen=True)
class StartupResealMotionPermit:
    """Exact authorization for one replacement run after a startup rejection."""

    master_authorization_sha256: str
    master_authorization_path: str
    run_id: str
    leg_index: int
    target_viewpoint_id: str
    reseal_index: int
    max_startup_reseals_per_leg: int
    rejected_run_id: str
    rejected_semantic_log_path: str
    rejected_semantic_log_sha256: str
    startup_reseal_summary_path: str
    startup_reseal_summary_sha256: str
    fresh_stationary_localization_evidence_path: str
    fresh_stationary_localization_evidence_sha256: str
    route_csv_path: str
    route_csv_sha256: str
    diagnostics_path: str
    diagnostics_sha256: str
    map_route_certificate_path: str
    map_route_certificate_sha256: str
    dry_preflight_path: str
    dry_preflight_sha256: str
    dry_odom_certificate_path: str
    dry_odom_certificate_sha256: str
    dry_uncertainty_budget_path: str
    dry_uncertainty_budget_sha256: str
    same_target_verified: bool
    rejected_motion_published: bool
    dry_run_passed: bool
    additional_typed_run_required: bool
    recovery_source_kind: str = (
        STARTUP_RESEAL_RECOVERY_SOURCE_CERTIFIED_START_POSE_MISMATCH
    )
    mission_leg_kind: MissionLegKind = MissionLegKind.COVERAGE
    mission_leg_index: int | None = None
    target_id: str = ""
    schema_version: int = STARTUP_RESEAL_MOTION_PERMIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        kind, index, target = resolve_startup_reseal_mission_leg_identity(
            mission_leg_kind=self.mission_leg_kind,
            mission_leg_index=self.mission_leg_index,
            target_id=self.target_id,
            leg_index=self.leg_index,
            target_viewpoint_id=self.target_viewpoint_id,
        )
        object.__setattr__(self, "mission_leg_kind", kind)
        object.__setattr__(self, "mission_leg_index", index)
        object.__setattr__(self, "target_id", target)
        _validate_permit(self)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "master_authorization_sha256": self.master_authorization_sha256,
            "master_authorization_path": self.master_authorization_path,
            "run_id": self.run_id,
            "leg_index": self.leg_index,
            "target_viewpoint_id": self.target_viewpoint_id,
            "mission_leg_kind": self.mission_leg_kind.value,
            "mission_leg_index": self.mission_leg_index,
            "target_id": self.target_id,
            "reseal_index": self.reseal_index,
            "max_startup_reseals_per_leg": self.max_startup_reseals_per_leg,
            "rejected_run_id": self.rejected_run_id,
            "rejected_semantic_log_path": self.rejected_semantic_log_path,
            "rejected_semantic_log_sha256": self.rejected_semantic_log_sha256,
            "startup_reseal_summary_path": self.startup_reseal_summary_path,
            "startup_reseal_summary_sha256": self.startup_reseal_summary_sha256,
            "fresh_stationary_localization_evidence_path": (
                self.fresh_stationary_localization_evidence_path
            ),
            "fresh_stationary_localization_evidence_sha256": (
                self.fresh_stationary_localization_evidence_sha256
            ),
            "route_csv_path": self.route_csv_path,
            "route_csv_sha256": self.route_csv_sha256,
            "diagnostics_path": self.diagnostics_path,
            "diagnostics_sha256": self.diagnostics_sha256,
            "map_route_certificate_path": self.map_route_certificate_path,
            "map_route_certificate_sha256": self.map_route_certificate_sha256,
            "dry_preflight_path": self.dry_preflight_path,
            "dry_preflight_sha256": self.dry_preflight_sha256,
            "dry_odom_certificate_path": self.dry_odom_certificate_path,
            "dry_odom_certificate_sha256": self.dry_odom_certificate_sha256,
            "dry_uncertainty_budget_path": self.dry_uncertainty_budget_path,
            "dry_uncertainty_budget_sha256": (
                self.dry_uncertainty_budget_sha256
            ),
            "same_target_verified": self.same_target_verified,
            "rejected_motion_published": self.rejected_motion_published,
            "dry_run_passed": self.dry_run_passed,
            "additional_typed_run_required": (
                self.additional_typed_run_required
            ),
            "recovery_source_kind": self.recovery_source_kind,
        }

    def to_evidence(self) -> dict[str, object]:
        return self.to_payload()


def file_sha256(path: Path) -> str:
    """Hash a normal nonsymlink file."""

    source = Path(path)
    if source.is_symlink():
        raise ValueError(f"artifact path must not be a symlink: {source}")
    if not source.is_file():
        raise ValueError(f"artifact path must be a normal file: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"artifact is unavailable: {source}") from exc
    return digest.hexdigest()


def startup_reseal_motion_authorization_sha256(
    authorization: StartupResealMotionAuthorization,
) -> str:
    _validate_authorization(authorization)
    return payload_sha256(authorization.to_payload())


def write_startup_reseal_motion_authorization(
    path: Path,
    authorization: StartupResealMotionAuthorization,
) -> str:
    _validate_authorization(authorization)
    try:
        return write_content_hashed_json(
            Path(path),
            authorization.to_payload(),
            hash_field=STARTUP_RESEAL_MOTION_AUTHORIZATION_HASH_FIELD,
        )
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc


def load_startup_reseal_motion_authorization(
    path: Path,
) -> StartupResealMotionAuthorization:
    try:
        payload = load_content_hashed_json(
            Path(path),
            hash_field=STARTUP_RESEAL_MOTION_AUTHORIZATION_HASH_FIELD,
        )
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc
    if frozenset(payload) != _AUTHORIZATION_FIELDS:
        raise ValueError("startup reseal motion authorization fields mismatch")
    try:
        return StartupResealMotionAuthorization(
            schema_version=_integer(payload["schema_version"], "schema_version"),
            session_id=_string(payload["session_id"], "session_id"),
            robot_id=_string(payload["robot_id"], "robot_id"),
            namespace=_string(payload["namespace"], "namespace"),
            cmd_vel_topic=_string(payload["cmd_vel_topic"], "cmd_vel_topic"),
            semantic_map_id=_string(
                payload["semantic_map_id"], "semantic_map_id"
            ),
            localization_branch_proof_id=_string(
                payload["localization_branch_proof_id"],
                "localization_branch_proof_id",
            ),
            max_startup_reseals_per_leg=_integer(
                payload["max_startup_reseals_per_leg"],
                "max_startup_reseals_per_leg",
            ),
            scope_text=_string(payload["scope_text"], "scope_text"),
            operator_confirmation=_string(
                payload["operator_confirmation"], "operator_confirmation"
            ),
            allowed_recovery_kind=_string(
                payload["allowed_recovery_kind"], "allowed_recovery_kind"
            ),
            allowed_mission_leg_kinds=tuple(
                _mission_leg_kind(value, "allowed_mission_leg_kinds")
                for value in _list(
                    payload["allowed_mission_leg_kinds"],
                    "allowed_mission_leg_kinds",
                )
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid startup reseal motion authorization: {exc}") from exc


def validate_startup_reseal_motion_authorization(
    authorization_path: Path,
    *,
    session_id: str,
    robot_id: str,
    namespace: str,
    cmd_vel_topic: str,
    semantic_map_id: str,
    localization_branch_proof_id: str,
) -> StartupResealMotionAuthorization:
    authorization = load_startup_reseal_motion_authorization(authorization_path)
    _require_exact_matches(
        "startup reseal motion authorization",
        {
            "session_id": (authorization.session_id, session_id),
            "robot_id": (authorization.robot_id, robot_id),
            "namespace": (authorization.namespace, namespace),
            "cmd_vel_topic": (authorization.cmd_vel_topic, cmd_vel_topic),
            "semantic_map_id": (
                authorization.semantic_map_id,
                semantic_map_id,
            ),
            "localization_branch_proof_id": (
                authorization.localization_branch_proof_id,
                localization_branch_proof_id,
            ),
        },
    )
    return authorization


def startup_reseal_motion_permit_sha256(
    permit: StartupResealMotionPermit,
) -> str:
    _validate_permit(permit)
    return payload_sha256(permit.to_payload())


def write_startup_reseal_motion_permit(
    path: Path,
    permit: StartupResealMotionPermit,
) -> str:
    """Live-verify all references and immutably publish the permit."""

    _validate_permit(permit)
    _validate_permit_references(permit)
    try:
        return write_content_hashed_json(
            Path(path),
            permit.to_payload(),
            hash_field=STARTUP_RESEAL_MOTION_PERMIT_HASH_FIELD,
        )
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc


def load_startup_reseal_motion_permit(path: Path) -> StartupResealMotionPermit:
    try:
        payload = load_content_hashed_json(
            Path(path), hash_field=STARTUP_RESEAL_MOTION_PERMIT_HASH_FIELD
        )
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc
    if frozenset(payload) != _PERMIT_FIELDS:
        raise ValueError("startup reseal motion permit fields mismatch")
    try:
        return StartupResealMotionPermit(
            schema_version=_integer(payload["schema_version"], "schema_version"),
            master_authorization_sha256=_string(
                payload["master_authorization_sha256"],
                "master_authorization_sha256",
            ),
            master_authorization_path=_string(
                payload["master_authorization_path"],
                "master_authorization_path",
            ),
            run_id=_string(payload["run_id"], "run_id"),
            leg_index=_integer(payload["leg_index"], "leg_index"),
            target_viewpoint_id=_string(
                payload["target_viewpoint_id"], "target_viewpoint_id"
            ),
            mission_leg_kind=_mission_leg_kind(
                payload["mission_leg_kind"], "mission_leg_kind"
            ),
            mission_leg_index=_integer(
                payload["mission_leg_index"], "mission_leg_index"
            ),
            target_id=_string(payload["target_id"], "target_id"),
            reseal_index=_integer(payload["reseal_index"], "reseal_index"),
            max_startup_reseals_per_leg=_integer(
                payload["max_startup_reseals_per_leg"],
                "max_startup_reseals_per_leg",
            ),
            rejected_run_id=_string(
                payload["rejected_run_id"], "rejected_run_id"
            ),
            rejected_semantic_log_path=_string(
                payload["rejected_semantic_log_path"],
                "rejected_semantic_log_path",
            ),
            rejected_semantic_log_sha256=_string(
                payload["rejected_semantic_log_sha256"],
                "rejected_semantic_log_sha256",
            ),
            startup_reseal_summary_path=_string(
                payload["startup_reseal_summary_path"],
                "startup_reseal_summary_path",
            ),
            startup_reseal_summary_sha256=_string(
                payload["startup_reseal_summary_sha256"],
                "startup_reseal_summary_sha256",
            ),
            fresh_stationary_localization_evidence_path=_string(
                payload["fresh_stationary_localization_evidence_path"],
                "fresh_stationary_localization_evidence_path",
            ),
            fresh_stationary_localization_evidence_sha256=_string(
                payload["fresh_stationary_localization_evidence_sha256"],
                "fresh_stationary_localization_evidence_sha256",
            ),
            route_csv_path=_string(payload["route_csv_path"], "route_csv_path"),
            route_csv_sha256=_string(
                payload["route_csv_sha256"], "route_csv_sha256"
            ),
            diagnostics_path=_string(
                payload["diagnostics_path"], "diagnostics_path"
            ),
            diagnostics_sha256=_string(
                payload["diagnostics_sha256"], "diagnostics_sha256"
            ),
            map_route_certificate_path=_string(
                payload["map_route_certificate_path"],
                "map_route_certificate_path",
            ),
            map_route_certificate_sha256=_string(
                payload["map_route_certificate_sha256"],
                "map_route_certificate_sha256",
            ),
            dry_preflight_path=_string(
                payload["dry_preflight_path"], "dry_preflight_path"
            ),
            dry_preflight_sha256=_string(
                payload["dry_preflight_sha256"], "dry_preflight_sha256"
            ),
            dry_odom_certificate_path=_string(
                payload["dry_odom_certificate_path"],
                "dry_odom_certificate_path",
            ),
            dry_odom_certificate_sha256=_string(
                payload["dry_odom_certificate_sha256"],
                "dry_odom_certificate_sha256",
            ),
            dry_uncertainty_budget_path=_string(
                payload["dry_uncertainty_budget_path"],
                "dry_uncertainty_budget_path",
            ),
            dry_uncertainty_budget_sha256=_string(
                payload["dry_uncertainty_budget_sha256"],
                "dry_uncertainty_budget_sha256",
            ),
            same_target_verified=_boolean(
                payload["same_target_verified"], "same_target_verified"
            ),
            rejected_motion_published=_boolean(
                payload["rejected_motion_published"],
                "rejected_motion_published",
            ),
            dry_run_passed=_boolean(
                payload["dry_run_passed"], "dry_run_passed"
            ),
            additional_typed_run_required=_boolean(
                payload["additional_typed_run_required"],
                "additional_typed_run_required",
            ),
            recovery_source_kind=_string(
                payload["recovery_source_kind"], "recovery_source_kind"
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid startup reseal motion permit: {exc}") from exc


def validate_startup_reseal_motion_permit(
    permit_path: Path,
    *,
    master_authorization_path: Path,
    run_id: str,
    session_id: str,
    robot_id: str,
    namespace: str,
    cmd_vel_topic: str,
    semantic_map_id: str,
    target_viewpoint_id: str,
    leg_index: int,
    localization_branch_proof_id: str,
    rejected_semantic_log_path: Path,
    rejected_semantic_log_sha256: str,
    startup_reseal_summary_path: Path,
    startup_reseal_summary_sha256: str,
    fresh_stationary_localization_evidence_path: Path,
    fresh_stationary_localization_evidence_sha256: str,
    route_csv_path: Path,
    route_csv_sha256: str,
    diagnostics_path: Path,
    diagnostics_sha256: str,
    map_route_certificate_path: Path,
    map_route_certificate_sha256: str,
    dry_preflight_path: Path,
    dry_preflight_sha256: str,
    dry_odom_certificate_path: Path,
    dry_odom_certificate_sha256: str,
    dry_uncertainty_budget_path: Path,
    dry_uncertainty_budget_sha256: str,
    mission_leg_kind: MissionLegKind | str = MissionLegKind.COVERAGE,
    mission_leg_index: int | None = None,
    target_id: str = "",
) -> StartupResealMotionPermit:
    """Validate every identity, path, supplied digest, and current byte hash."""

    permit = load_startup_reseal_motion_permit(permit_path)
    expected_kind, expected_index, expected_target = (
        resolve_startup_reseal_mission_leg_identity(
            mission_leg_kind=mission_leg_kind,
            mission_leg_index=mission_leg_index,
            target_id=target_id,
            leg_index=leg_index,
            target_viewpoint_id=target_viewpoint_id,
        )
    )
    authorization = _validate_master_reference(permit, master_authorization_path)
    _require_exact_matches(
        "startup reseal motion permit",
        {
            "run_id": (permit.run_id, run_id),
            "session_id": (authorization.session_id, session_id),
            "robot_id": (authorization.robot_id, robot_id),
            "namespace": (authorization.namespace, namespace),
            "cmd_vel_topic": (authorization.cmd_vel_topic, cmd_vel_topic),
            "semantic_map_id": (
                authorization.semantic_map_id,
                semantic_map_id,
            ),
            "target_viewpoint_id": (
                permit.target_viewpoint_id,
                target_viewpoint_id,
            ),
            "leg_index": (permit.leg_index, leg_index),
            "mission_leg_kind": (
                permit.mission_leg_kind,
                expected_kind,
            ),
            "mission_leg_index": (
                permit.mission_leg_index,
                expected_index,
            ),
            "target_id": (permit.target_id, expected_target),
            "localization_branch_proof_id": (
                authorization.localization_branch_proof_id,
                localization_branch_proof_id,
            ),
        },
    )
    supplied = (
        (
            "rejected_semantic_log",
            permit.rejected_semantic_log_path,
            permit.rejected_semantic_log_sha256,
            rejected_semantic_log_path,
            rejected_semantic_log_sha256,
        ),
        (
            "startup_reseal_summary",
            permit.startup_reseal_summary_path,
            permit.startup_reseal_summary_sha256,
            startup_reseal_summary_path,
            startup_reseal_summary_sha256,
        ),
        (
            "fresh_stationary_localization_evidence",
            permit.fresh_stationary_localization_evidence_path,
            permit.fresh_stationary_localization_evidence_sha256,
            fresh_stationary_localization_evidence_path,
            fresh_stationary_localization_evidence_sha256,
        ),
        (
            "route_csv",
            permit.route_csv_path,
            permit.route_csv_sha256,
            route_csv_path,
            route_csv_sha256,
        ),
        (
            "diagnostics",
            permit.diagnostics_path,
            permit.diagnostics_sha256,
            diagnostics_path,
            diagnostics_sha256,
        ),
        (
            "map_route_certificate",
            permit.map_route_certificate_path,
            permit.map_route_certificate_sha256,
            map_route_certificate_path,
            map_route_certificate_sha256,
        ),
        (
            "dry_preflight",
            permit.dry_preflight_path,
            permit.dry_preflight_sha256,
            dry_preflight_path,
            dry_preflight_sha256,
        ),
        (
            "dry_odom_certificate",
            permit.dry_odom_certificate_path,
            permit.dry_odom_certificate_sha256,
            dry_odom_certificate_path,
            dry_odom_certificate_sha256,
        ),
        (
            "dry_uncertainty_budget",
            permit.dry_uncertainty_budget_path,
            permit.dry_uncertainty_budget_sha256,
            dry_uncertainty_budget_path,
            dry_uncertainty_budget_sha256,
        ),
    )
    for name, sealed_path, sealed_hash, live_path, supplied_hash in supplied:
        _validate_artifact_binding(
            name,
            sealed_path=sealed_path,
            sealed_sha256=sealed_hash,
            live_path=live_path,
            supplied_sha256=supplied_hash,
        )
    _validate_budget(permit, authorization)
    _validate_startup_evidence(permit)
    return permit


def validate_startup_reseal_motion_permit_for_execution(
    permit_path: Path,
    *,
    master_authorization_path: Path,
    run_id: str,
    session_id: str,
    robot_id: str,
    namespace: str,
    cmd_vel_topic: str,
    semantic_map_id: str,
    target_viewpoint_id: str,
    leg_index: int,
    localization_branch_proof_id: str,
    route_csv_path: Path,
    diagnostics_path: Path,
    map_route_certificate_path: Path,
    mission_leg_kind: MissionLegKind | str = MissionLegKind.COVERAGE,
    mission_leg_index: int | None = None,
    target_id: str = "",
) -> StartupResealMotionPermit:
    """Execution-facing validator with sealed supporting inputs derived inside."""

    permit = load_startup_reseal_motion_permit(permit_path)
    return validate_startup_reseal_motion_permit(
        permit_path,
        master_authorization_path=master_authorization_path,
        run_id=run_id,
        session_id=session_id,
        robot_id=robot_id,
        namespace=namespace,
        cmd_vel_topic=cmd_vel_topic,
        semantic_map_id=semantic_map_id,
        target_viewpoint_id=target_viewpoint_id,
        leg_index=leg_index,
        localization_branch_proof_id=localization_branch_proof_id,
        rejected_semantic_log_path=Path(permit.rejected_semantic_log_path),
        rejected_semantic_log_sha256=permit.rejected_semantic_log_sha256,
        startup_reseal_summary_path=Path(permit.startup_reseal_summary_path),
        startup_reseal_summary_sha256=permit.startup_reseal_summary_sha256,
        fresh_stationary_localization_evidence_path=Path(
            permit.fresh_stationary_localization_evidence_path
        ),
        fresh_stationary_localization_evidence_sha256=(
            permit.fresh_stationary_localization_evidence_sha256
        ),
        route_csv_path=route_csv_path,
        route_csv_sha256=permit.route_csv_sha256,
        diagnostics_path=diagnostics_path,
        diagnostics_sha256=permit.diagnostics_sha256,
        map_route_certificate_path=map_route_certificate_path,
        map_route_certificate_sha256=permit.map_route_certificate_sha256,
        dry_preflight_path=Path(permit.dry_preflight_path),
        dry_preflight_sha256=permit.dry_preflight_sha256,
        dry_odom_certificate_path=Path(permit.dry_odom_certificate_path),
        dry_odom_certificate_sha256=permit.dry_odom_certificate_sha256,
        dry_uncertainty_budget_path=Path(permit.dry_uncertainty_budget_path),
        dry_uncertainty_budget_sha256=permit.dry_uncertainty_budget_sha256,
        mission_leg_kind=mission_leg_kind,
        mission_leg_index=mission_leg_index,
        target_id=target_id,
    )


def _validate_authorization(
    authorization: StartupResealMotionAuthorization,
) -> None:
    if (
        authorization.schema_version
        != STARTUP_RESEAL_MOTION_AUTHORIZATION_SCHEMA_VERSION
    ):
        raise ValueError("unsupported startup reseal motion authorization schema")
    for name in (
        "session_id",
        "robot_id",
        "cmd_vel_topic",
        "semantic_map_id",
        "localization_branch_proof_id",
    ):
        _require_nonempty(getattr(authorization, name), name)
    if not isinstance(authorization.namespace, str):
        raise ValueError("namespace must be a string")
    if authorization.namespace != authorization.namespace.strip():
        raise ValueError("namespace must be canonical")
    _nonnegative_integer(
        authorization.max_startup_reseals_per_leg,
        "max_startup_reseals_per_leg",
    )
    if authorization.scope_text != STARTUP_RESEAL_MOTION_AUTHORIZATION_SCOPE:
        raise ValueError("startup reseal motion authorization scope_text mismatch")
    if authorization.operator_confirmation != STARTUP_RESEAL_RUN_CONFIRMATION:
        raise ValueError(
            "startup reseal motion authorization requires operator confirmation RUN"
        )
    if authorization.allowed_recovery_kind != STARTUP_RESEAL_RECOVERY_KIND:
        raise ValueError("startup reseal motion authorization recovery kind mismatch")
    _canonical_allowed_mission_leg_kinds(
        authorization.allowed_mission_leg_kinds
    )


def _validate_permit(permit: StartupResealMotionPermit) -> None:
    if permit.schema_version != STARTUP_RESEAL_MOTION_PERMIT_SCHEMA_VERSION:
        raise ValueError("unsupported startup reseal motion permit schema")
    _require_sha256(
        permit.master_authorization_sha256,
        "master_authorization_sha256",
    )
    for name in (
        "master_authorization_path",
        "rejected_semantic_log_path",
        "startup_reseal_summary_path",
        "fresh_stationary_localization_evidence_path",
        "route_csv_path",
        "diagnostics_path",
        "map_route_certificate_path",
        "dry_preflight_path",
        "dry_odom_certificate_path",
        "dry_uncertainty_budget_path",
    ):
        _require_canonical_path_string(getattr(permit, name), name)
    for name in ("run_id", "target_viewpoint_id", "rejected_run_id"):
        _require_nonempty(getattr(permit, name), name)
    if permit.mission_leg_kind not in ROUTINE_MISSION_LEG_KINDS:
        raise ValueError("startup reseal permit mission_leg_kind is not routine")
    _nonnegative_integer(permit.mission_leg_index, "mission_leg_index")
    _require_nonempty(permit.target_id, "target_id")
    if permit.mission_leg_index != permit.leg_index:
        raise ValueError("startup reseal permit mission leg index alias mismatch")
    if permit.target_id != permit.target_viewpoint_id:
        raise ValueError("startup reseal permit target alias mismatch")
    _nonnegative_integer(permit.leg_index, "leg_index")
    _positive_integer(permit.reseal_index, "reseal_index")
    _positive_integer(
        permit.max_startup_reseals_per_leg,
        "max_startup_reseals_per_leg",
    )
    if permit.reseal_index > permit.max_startup_reseals_per_leg:
        raise ValueError("startup reseal motion permit reseal budget exceeded")
    if permit.rejected_run_id == permit.run_id:
        raise ValueError("rejected_run_id must differ from replacement run_id")
    for name in (
        "rejected_semantic_log_sha256",
        "startup_reseal_summary_sha256",
        "fresh_stationary_localization_evidence_sha256",
        "route_csv_sha256",
        "diagnostics_sha256",
        "map_route_certificate_sha256",
        "dry_preflight_sha256",
        "dry_odom_certificate_sha256",
        "dry_uncertainty_budget_sha256",
    ):
        _require_sha256(getattr(permit, name), name)
    if permit.same_target_verified is not True:
        raise ValueError("startup reseal motion permit requires same_target_verified=true")
    if permit.rejected_motion_published is not False:
        raise ValueError(
            "startup reseal motion permit requires rejected_motion_published=false"
        )
    if permit.dry_run_passed is not True:
        raise ValueError("startup reseal motion permit requires dry_run_passed=true")
    if permit.additional_typed_run_required is not False:
        raise ValueError(
            "startup reseal motion permit requires "
            "additional_typed_run_required=false"
        )
    if (
        not isinstance(permit.recovery_source_kind, str)
        or permit.recovery_source_kind
        not in STARTUP_RESEAL_RECOVERY_SOURCE_KINDS
    ):
        raise ValueError(
            "startup reseal motion permit recovery_source_kind is not authorized"
        )


def _validate_permit_references(permit: StartupResealMotionPermit) -> None:
    authorization = _validate_master_reference(
        permit, Path(permit.master_authorization_path)
    )
    for name, path, digest in _permit_artifacts(permit):
        _validate_bound_artifact(name, path, digest)
    _validate_budget(permit, authorization)
    _validate_startup_evidence(permit)


def _validate_master_reference(
    permit: StartupResealMotionPermit,
    master_authorization_path: Path,
) -> StartupResealMotionAuthorization:
    observed_path = _canonical_normal_file_path(
        master_authorization_path,
        "startup reseal motion authorization",
    )
    if observed_path != permit.master_authorization_path:
        raise ValueError(
            "startup reseal motion permit master authorization path mismatch"
        )
    authorization = load_startup_reseal_motion_authorization(
        master_authorization_path
    )
    if (
        startup_reseal_motion_authorization_sha256(authorization)
        != permit.master_authorization_sha256
    ):
        raise ValueError(
            "startup reseal motion permit master authorization hash mismatch"
        )
    return authorization


def _validate_budget(
    permit: StartupResealMotionPermit,
    authorization: StartupResealMotionAuthorization,
) -> None:
    if (
        permit.max_startup_reseals_per_leg
        != authorization.max_startup_reseals_per_leg
    ):
        raise ValueError("startup reseal motion permit reseal maximum mismatch")
    if not 1 <= permit.reseal_index <= authorization.max_startup_reseals_per_leg:
        raise ValueError("startup reseal motion permit reseal budget exceeded")
    if permit.mission_leg_kind not in authorization.allowed_mission_leg_kinds:
        raise ValueError(
            "startup reseal motion permit mission leg kind is not authorized"
        )


def _validate_startup_evidence(permit: StartupResealMotionPermit) -> None:
    if (
        permit.recovery_source_kind
        == STARTUP_RESEAL_RECOVERY_SOURCE_CERTIFIED_START_POSE_MISMATCH
    ):
        _validate_rejected_semantic_log(permit)
    elif (
        permit.recovery_source_kind
        == STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY
    ):
        _validate_prestart_localization_rejected_semantic_log(permit)
    else:  # pragma: no cover - guarded by _validate_permit
        raise ValueError(
            "startup reseal motion permit recovery_source_kind is not authorized"
        )
    fresh_start_pose = _validate_startup_reseal_summary(permit)
    _validate_fresh_stationary_localization_evidence(
        permit,
        expected_route_pose=fresh_start_pose,
    )
    validate_startup_reseal_route_binding(
        route_csv_path=Path(permit.route_csv_path),
        diagnostics_path=Path(permit.diagnostics_path),
        fresh_pose=fresh_start_pose,
        require_start_pose_provenance=permit.mission_leg_kind
        in {
            MissionLegKind.CANDIDATE_PREAPPROACH,
            MissionLegKind.OPPOSITE_FACE,
        },
    )


def _event_matches_mission_leg_identity(
    event: Mapping[str, object],
    permit: StartupResealMotionPermit,
) -> bool:
    """Match generic identity, with a strict legacy fallback for coverage.

    Once any generic field is present, all three generic fields are required;
    a conflicting generic identity can never be hidden by matching coverage
    aliases.  Legacy logs without generic fields remain eligible only for the
    original coverage contract.
    """

    generic_names = ("mission_leg_kind", "mission_leg_index", "target_id")
    if any(name in event for name in generic_names):
        generic_match = (
            event.get("mission_leg_kind") == permit.mission_leg_kind.value
            and event.get("mission_leg_index") == permit.mission_leg_index
            and type(event.get("mission_leg_index")) is int
            and event.get("target_id") == permit.target_id
        )
        if not generic_match:
            return False
        if permit.mission_leg_kind is MissionLegKind.COVERAGE and any(
            name in event
            for name in ("coverage_leg_index", "target_viewpoint_id")
        ):
            return (
                event.get("coverage_leg_index") == permit.mission_leg_index
                and type(event.get("coverage_leg_index")) is int
                and event.get("target_viewpoint_id") == permit.target_id
            )
        return True
    return (
        permit.mission_leg_kind is MissionLegKind.COVERAGE
        and event.get("coverage_leg_index") == permit.mission_leg_index
        and type(event.get("coverage_leg_index")) is int
        and event.get("target_viewpoint_id") == permit.target_id
    )


def _validate_rejected_semantic_log(permit: StartupResealMotionPermit) -> None:
    same_run = _same_run_strict_nomotion_events(permit)
    matches = []
    for event in same_run:
        if event.get("event") != "startup_route_rejected":
            continue
        details = event.get("stop_details")
        if not isinstance(details, Mapping):
            continue
        if (
            _event_matches_mission_leg_identity(event, permit)
            and event.get("status") == "stopped"
            and event.get("stop_reason")
            == "pose outside certified startup segment"
            and event.get("motion_published") is False
            and details.get("source") == "execution_route_certificate"
            and details.get("phase") == "before_motion_confirmation"
            and details.get("reason")
            == "pose outside certified startup segment"
            and details.get("fail_closed") is True
        ):
            matches.append(event)
    if len(matches) != 1:
        raise ValueError(
            "rejected semantic log must contain exactly one same-run "
            "pre-motion startup rejection"
        )


def _validate_prestart_localization_rejected_semantic_log(
    permit: StartupResealMotionPermit,
) -> None:
    from scripts.aufgabe04.navigation.prestart_localization_reseal import (
        evaluate_prestart_localization_reseal,
    )

    same_run = _same_run_events(permit)
    for event in same_run:
        motion_value = event.get("motion_published")
        if (
            event.get("event") == "motion_completed"
            or motion_value is True
            or ("motion_published" in event and type(motion_value) is not bool)
        ):
            raise ValueError(
                "prestart rejected semantic log contains completed or "
                "published motion"
            )

    matches: list[tuple[int, dict[str, object]]] = []
    for index, event in enumerate(same_run):
        if (
            event.get("event") != "safety_stop"
            or not _event_matches_mission_leg_identity(event, permit)
        ):
            continue
        details = event.get("stop_details")
        decision = evaluate_prestart_localization_reseal(
            status=event.get("status"),
            motion_published=event.get("motion_published"),
            stop_details=details,
        )
        if (
            decision.eligible
            and decision.motion_published is False
            and decision.requires_fresh_localization
            and decision.requires_new_route_certificate
            and decision.automatic_motion_authorized is False
            and isinstance(details, Mapping)
            and event.get("stop_reason") == details.get("reason")
        ):
            matches.append((index, event))
    if len(matches) != 1:
        raise ValueError(
            "rejected semantic log must contain exactly one same-run eligible "
            "prestart localization-continuity safety stop"
        )
    safety_stop_index, _ = matches[0]
    _validate_prestart_attempt_sequence(
        permit,
        same_run=same_run,
        safety_stop_index=safety_stop_index,
    )


def _same_run_events(
    permit: StartupResealMotionPermit,
) -> tuple[dict[str, object], ...]:
    events = _load_jsonl_objects(Path(permit.rejected_semantic_log_path))
    return tuple(
        event
        for event in events
        if event.get("run_id") == permit.rejected_run_id
    )


def _same_run_strict_nomotion_events(
    permit: StartupResealMotionPermit,
) -> tuple[dict[str, object], ...]:
    same_run = _same_run_events(permit)
    for event in same_run:
        motion_value = event.get("motion_published")
        if (
            event.get("event") in {"motion_started", "motion_completed"}
            or motion_value is True
            or ("motion_published" in event and type(motion_value) is not bool)
        ):
            raise ValueError(
                "rejected semantic log contains published or started motion"
            )
    return same_run


def _validate_prestart_attempt_sequence(
    permit: StartupResealMotionPermit,
    *,
    same_run: tuple[dict[str, object], ...],
    safety_stop_index: int,
) -> None:
    consumed_event_names = {
        "mission_leg_motion_permit_consumed",
        "startup_reseal_motion_permit_consumed",
        "runtime_localization_motion_permit_consumed",
    }
    consumed = [
        (index, event)
        for index, event in enumerate(same_run)
        if event.get("event") in consumed_event_names
    ]
    started = [
        (index, event)
        for index, event in enumerate(same_run)
        if event.get("event") == "motion_started"
    ]
    if len(consumed) != 1:
        raise ValueError(
            "prestart rejected semantic log must contain exactly one "
            "same-run motion permit consumption"
        )
    if len(started) != 1:
        raise ValueError(
            "prestart rejected semantic log must contain exactly one "
            "same-run child execution attempt"
        )
    consumed_index, consumed_event = consumed[0]
    started_index, started_event = started[0]
    if not consumed_index < started_index < safety_stop_index:
        raise ValueError(
            "prestart rejected semantic log event ordering mismatch"
        )
    if (
        started_event.get("motion_published") is not False
        or started_event.get("event_semantics")
        != "child_execution_attempt_started_before_follower"
    ):
        raise ValueError(
            "prestart rejected semantic log child execution-attempt "
            "semantics mismatch"
        )
    for label, event in (
        ("motion permit consumption", consumed_event),
        ("child execution attempt", started_event),
    ):
        if not _event_matches_mission_leg_identity(event, permit):
            raise ValueError(
                f"prestart rejected semantic log {label} identity mismatch"
            )
    if (
        consumed_event.get("covered_by_initial_mission_run") is not True
        or consumed_event.get("additional_typed_run_required") is not False
    ):
        raise ValueError(
            "prestart rejected semantic log motion permit scope mismatch"
        )
    consumed_name = consumed_event.get("event")
    if consumed_name == "mission_leg_motion_permit_consumed" and (
        consumed_event.get("mission_leg_kind") != permit.mission_leg_kind.value
        or consumed_event.get("mission_leg_index") != permit.mission_leg_index
        or type(consumed_event.get("mission_leg_index")) is not int
        or consumed_event.get("target_id") != permit.target_id
    ):
        raise ValueError(
            "prestart rejected semantic log routine permit identity mismatch"
        )
    if consumed_name == "startup_reseal_motion_permit_consumed":
        prior_source = consumed_event.get("recovery_source_kind")
        if (
            not isinstance(prior_source, str)
            or prior_source not in STARTUP_RESEAL_RECOVERY_SOURCE_KINDS
        ):
            raise ValueError(
                "prestart rejected semantic log startup permit source mismatch"
            )


def _validate_startup_reseal_summary(
    permit: StartupResealMotionPermit,
) -> tuple[float, float, float]:
    summary = _load_json_object(Path(permit.startup_reseal_summary_path))
    expected_fields = {
        "schema_version",
        "status",
        "motion_published",
        "reseal_kind",
        "leg_index",
        "mission_leg_kind",
        "mission_leg_index",
        "startup_reseal_index",
        "rejected_run_id",
        "target_viewpoint_id",
        "target_id",
        "fresh_start_pose",
        "route_csv",
        "diagnostics_json",
        "same_target_verified",
        "additional_typed_run_required",
        "recovery_source_kind",
    }
    if frozenset(summary) != expected_fields:
        raise ValueError("startup reseal summary fields mismatch")
    expected = {
        "schema_version": STARTUP_RESEAL_PERMIT_SUMMARY_SCHEMA_VERSION,
        "status": "startup_route_replanned",
        "motion_published": False,
        "reseal_kind": "startup",
        "leg_index": permit.leg_index,
        "mission_leg_kind": permit.mission_leg_kind.value,
        "mission_leg_index": permit.mission_leg_index,
        "startup_reseal_index": permit.reseal_index,
        "rejected_run_id": permit.rejected_run_id,
        "target_viewpoint_id": permit.target_viewpoint_id,
        "target_id": permit.target_id,
        "route_csv": permit.route_csv_path,
        "diagnostics_json": permit.diagnostics_path,
        "same_target_verified": True,
        "additional_typed_run_required": False,
        "recovery_source_kind": permit.recovery_source_kind,
    }
    for name, value in expected.items():
        observed = summary.get(name)
        if observed != value or (
            isinstance(value, (bool, int)) and type(observed) is not type(value)
        ):
            raise ValueError(f"startup reseal summary {name} mismatch")
    pose = summary.get("fresh_start_pose")
    if not isinstance(pose, Mapping) or frozenset(pose) != {
        "x_m",
        "y_m",
        "yaw_rad",
    }:
        raise ValueError("startup reseal summary fresh_start_pose mismatch")
    pose_values = []
    for name in ("x_m", "y_m", "yaw_rad"):
        value = pose.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(
                "startup reseal summary fresh_start_pose must be finite"
            )
        pose_values.append(float(value))
    return tuple(pose_values)


def _validate_fresh_stationary_localization_evidence(
    permit: StartupResealMotionPermit,
    *,
    expected_route_pose: tuple[float, float, float],
) -> None:
    evidence = _load_json_object(
        Path(permit.fresh_stationary_localization_evidence_path)
    )
    expected_fields = {
        "ok",
        "failures",
        "observations",
        "runtime_config",
        "route_pose",
        "odom_pose",
        "map_from_odom",
        "stationary_amcl_samples",
        "stationary_map_from_odom_samples",
    }
    if frozenset(evidence) != expected_fields:
        raise ValueError(
            "fresh stationary localization evidence fields mismatch"
        )
    if evidence.get("ok") is not True or evidence.get("failures") != []:
        raise ValueError(
            "fresh stationary localization evidence was not admitted"
        )

    runtime_config = evidence.get("runtime_config")
    if (
        not isinstance(runtime_config, Mapping)
        or runtime_config.get("localization_source") != "amcl"
        or runtime_config.get("use_sim_time") is not False
    ):
        raise ValueError(
            "fresh stationary localization evidence runtime is not physical AMCL"
        )

    route_pose = evidence.get("route_pose")
    if not isinstance(route_pose, Mapping):
        raise ValueError(
            "fresh stationary localization evidence route_pose is missing"
        )
    if not isinstance(route_pose.get("frame_id"), str) or not str(
        route_pose.get("frame_id")
    ).strip():
        raise ValueError(
            "fresh stationary localization evidence route_pose frame is invalid"
        )
    if not isinstance(route_pose.get("child_frame_id"), str) or not str(
        route_pose.get("child_frame_id")
    ).strip():
        raise ValueError(
            "fresh stationary localization evidence route_pose child frame is invalid"
        )
    observed_route_pose = []
    for name in ("x_m", "y_m", "yaw_rad"):
        value = route_pose.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(
                "fresh stationary localization evidence route_pose must be finite"
            )
        observed_route_pose.append(float(value))
    if tuple(observed_route_pose) != expected_route_pose:
        raise ValueError(
            "fresh stationary localization evidence route_pose does not match "
            "the startup reseal summary"
        )

    samples = evidence.get("stationary_amcl_samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError(
            "fresh stationary localization evidence lacks an AMCL sample window"
        )
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(
                "fresh stationary localization evidence AMCL sample is malformed"
            )
        for name in ("x_m", "y_m", "yaw_rad"):
            value = sample.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(
                    "fresh stationary localization evidence AMCL sample "
                    f"{index} is non-finite"
                )
        covariance = sample.get("covariance")
        if (
            not isinstance(covariance, list)
            or len(covariance) != 36
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in covariance
            )
        ):
            raise ValueError(
                "fresh stationary localization evidence AMCL covariance is malformed"
            )

    observations = evidence.get("observations")
    if not isinstance(observations, list):
        raise ValueError(
            "fresh stationary localization evidence observations are malformed"
        )
    stationary = [
        observation
        for observation in observations
        if isinstance(observation, Mapping)
        and observation.get("name") == "stationary AMCL stability"
    ]
    if len(stationary) != 1 or stationary[0].get("ok") is not True:
        raise ValueError(
            "fresh stationary localization evidence lacks an admitted "
            "stationary AMCL observation"
        )
    data = stationary[0].get("data")
    if not isinstance(data, Mapping):
        raise ValueError(
            "fresh stationary localization evidence stationary observation "
            "is malformed"
        )
    sample_count = data.get("sample_count")
    required_sample_count = data.get("required_sample_count")
    service_request_count = data.get("service_request_count")
    if (
        type(sample_count) is not int
        or type(required_sample_count) is not int
        or type(service_request_count) is not int
        or required_sample_count < 2
        or sample_count < required_sample_count
        or len(samples) < required_sample_count
        or service_request_count < required_sample_count
        or data.get("position_covariance_complete") is not True
        or data.get("yaw_covariance_complete") is not True
    ):
        raise ValueError(
            "fresh stationary localization evidence did not prove a complete "
            "no-motion AMCL window"
        )
    if not isinstance(evidence.get("stationary_map_from_odom_samples"), list):
        raise ValueError(
            "fresh stationary localization evidence map-to-odom samples are malformed"
        )


def _permit_artifacts(
    permit: StartupResealMotionPermit,
) -> tuple[tuple[str, str, str], ...]:
    return (
        (
            "rejected_semantic_log",
            permit.rejected_semantic_log_path,
            permit.rejected_semantic_log_sha256,
        ),
        (
            "startup_reseal_summary",
            permit.startup_reseal_summary_path,
            permit.startup_reseal_summary_sha256,
        ),
        (
            "fresh_stationary_localization_evidence",
            permit.fresh_stationary_localization_evidence_path,
            permit.fresh_stationary_localization_evidence_sha256,
        ),
        ("route_csv", permit.route_csv_path, permit.route_csv_sha256),
        ("diagnostics", permit.diagnostics_path, permit.diagnostics_sha256),
        (
            "map_route_certificate",
            permit.map_route_certificate_path,
            permit.map_route_certificate_sha256,
        ),
        (
            "dry_preflight",
            permit.dry_preflight_path,
            permit.dry_preflight_sha256,
        ),
        (
            "dry_odom_certificate",
            permit.dry_odom_certificate_path,
            permit.dry_odom_certificate_sha256,
        ),
        (
            "dry_uncertainty_budget",
            permit.dry_uncertainty_budget_path,
            permit.dry_uncertainty_budget_sha256,
        ),
    )


def _validate_bound_artifact(name: str, path: str, expected_sha256: str) -> None:
    observed_path = _canonical_normal_file_path(Path(path), name)
    if observed_path != path:
        raise ValueError(f"startup reseal motion permit {name} path is not canonical")
    if file_sha256(Path(path)) != expected_sha256:
        raise ValueError(f"startup reseal motion permit {name} hash mismatch")


def _validate_artifact_binding(
    name: str,
    *,
    sealed_path: str,
    sealed_sha256: str,
    live_path: Path,
    supplied_sha256: str,
) -> None:
    _require_sha256(supplied_sha256, f"{name}_sha256")
    observed_path = _canonical_normal_file_path(live_path, name)
    if observed_path != sealed_path:
        raise ValueError(f"startup reseal motion permit {name} path mismatch")
    if supplied_sha256 != sealed_sha256:
        raise ValueError(
            f"startup reseal motion permit {name} supplied hash mismatch"
        )
    if file_sha256(live_path) != sealed_sha256:
        raise ValueError(f"startup reseal motion permit {name} hash mismatch")


def _canonical_normal_file_path(path: Path, label: str) -> str:
    source = Path(path)
    if not source.is_absolute() or source != Path(os.path.normpath(str(source))):
        raise ValueError(f"{label} path must be canonical absolute")
    file_sha256(source)
    return str(source)


def _load_json_object(path: Path) -> dict[str, object]:
    source = Path(path)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object_pairs,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid startup reseal JSON: {source}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"startup reseal JSON root must be an object: {source}")
    return value


def _load_jsonl_objects(path: Path) -> tuple[dict[str, object], ...]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid rejected semantic log: {path}") from exc
    events = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, object_pairs_hook=_strict_object_pairs)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"invalid rejected semantic log JSONL at line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"rejected semantic log line {line_number} is not an object"
            )
        events.append(value)
    if not events:
        raise ValueError("rejected semantic log is empty")
    return tuple(events)


def _strict_object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _require_exact_matches(
    label: str,
    checks: Mapping[str, tuple[object, object]],
) -> None:
    for name, (sealed, observed) in checks.items():
        if sealed != observed or (
            isinstance(sealed, (bool, int)) and type(sealed) is not type(observed)
        ):
            raise ValueError(f"{label} {name} mismatch")


def _require_canonical_path_string(value: object, name: str) -> None:
    _require_nonempty(value, name)
    assert isinstance(value, str)
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.normpath(value)):
        raise ValueError(f"{name} must be a canonical absolute path")


def _require_nonempty(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string")


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


__all__ = [
    "STARTUP_RESEAL_MOTION_AUTHORIZATION_HASH_FIELD",
    "STARTUP_RESEAL_MOTION_AUTHORIZATION_SCHEMA_VERSION",
    "STARTUP_RESEAL_MOTION_AUTHORIZATION_SCOPE",
    "STARTUP_RESEAL_MOTION_PERMIT_HASH_FIELD",
    "STARTUP_RESEAL_MOTION_PERMIT_SCHEMA_VERSION",
    "STARTUP_RESEAL_PERMIT_SUMMARY_SCHEMA_VERSION",
    "STARTUP_RESEAL_RECOVERY_KIND",
    "STARTUP_RESEAL_RECOVERY_SOURCE_CERTIFIED_START_POSE_MISMATCH",
    "STARTUP_RESEAL_RECOVERY_SOURCE_KINDS",
    "STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY",
    "STARTUP_RESEAL_RUN_CONFIRMATION",
    "StartupResealMotionAuthorization",
    "StartupResealMotionPermit",
    "file_sha256",
    "load_startup_reseal_motion_authorization",
    "load_startup_reseal_motion_permit",
    "resolve_startup_reseal_mission_leg_identity",
    "startup_reseal_motion_authorization_sha256",
    "startup_reseal_motion_permit_sha256",
    "validate_startup_reseal_motion_authorization",
    "validate_startup_reseal_motion_permit",
    "validate_startup_reseal_motion_permit_for_execution",
    "write_startup_reseal_motion_authorization",
    "write_startup_reseal_motion_permit",
]
