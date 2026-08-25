"""Immutable authorization for one bounded runtime-localization recovery.

This module is deliberately ROS-free.  A top-level autonomous mission may
seal one :class:`MissionMotionAuthorization` after its operator typed ``RUN``.
That authorization can then be referenced by a content-hashed permit for a
strictly post-motion, same-leg, same-target localization reseal.  The permit
does not authorize startup recovery, a generic stop, another target, or an
unbounded retry.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from scripts.aufgabe04.artifacts.content_store import (
    ContentStoreError,
    canonical_json_bytes,
    load_content_hashed_json,
    payload_sha256,
    write_content_hashed_json,
)
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
    ROUTINE_MISSION_LEG_KINDS,
    MissionLegKind,
)


MISSION_MOTION_AUTHORIZATION_SCHEMA_VERSION = 2
RUNTIME_LOCALIZATION_MOTION_PERMIT_SCHEMA_VERSION = 2
LEGACY_MISSION_MOTION_AUTHORIZATION_SCHEMA_VERSION = 1
LEGACY_RUNTIME_LOCALIZATION_MOTION_PERMIT_SCHEMA_VERSION = 1

MISSION_MOTION_AUTHORIZATION_HASH_FIELD = "mission_motion_authorization_sha256"
RUNTIME_LOCALIZATION_MOTION_PERMIT_HASH_FIELD = (
    "runtime_localization_motion_permit_sha256"
)

RUNTIME_LOCALIZATION_RESEAL_RECOVERY_KIND = "runtime_localization_reseal"
MISSION_RUN_CONFIRMATION = "RUN"
MISSION_MOTION_AUTHORIZATION_SCOPE = (
    "Reuse this autonomous mission RUN only for a bounded same-leg, "
    "same-target post-motion runtime localization reseal; startup reseals, "
    "generic stops, target changes, leg changes, and automatic motion are "
    "not authorized."
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MISSION_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "session_id",
        "robot_id",
        "namespace",
        "cmd_vel_topic",
        "semantic_map_id",
        "localization_branch_proof_id",
        "max_runtime_reseals_per_leg",
        "scope_text",
        "operator_confirmation",
        "allowed_recovery_kind",
    }
)
_MISSION_FIELDS_V2 = _MISSION_FIELDS_V1 | {"allowed_mission_leg_kinds"}
_PERMIT_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "master_authorization_sha256",
        "master_authorization_path",
        "run_id",
        "leg_index",
        "target_viewpoint_id",
        "reseal_index",
        "max_runtime_reseals_per_leg",
        "rejected_run_id",
        "runtime_reseal_decision_evidence",
        "runtime_reseal_decision_sha256",
        "fresh_localization_evidence_path",
        "fresh_localization_evidence_sha256",
        "route_csv_path",
        "route_csv_sha256",
        "diagnostics_path",
        "diagnostics_sha256",
        "map_route_certificate_path",
        "map_route_certificate_sha256",
        "dry_odom_certificate_path",
        "dry_odom_certificate_sha256",
        "dry_uncertainty_budget_path",
        "dry_uncertainty_budget_sha256",
        "dry_preflight_path",
        "dry_preflight_sha256",
        "same_target_verified",
        "dry_run_passed",
        "additional_typed_run_required",
    }
)
_PERMIT_FIELDS_V2 = _PERMIT_FIELDS_V1 | {
    "mission_leg_kind",
    "mission_leg_index",
    "target_id",
}
_DECISION_FIELDS = frozenset(
    {
        "schema_version",
        "eligible",
        "reason",
        "execution_phase",
        "motion_published",
        "continuity_reason",
        "requires_fresh_localization",
        "requires_new_route_certificate",
        "requires_fresh_typed_run",
        "automatic_motion_authorized",
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


def resolve_runtime_localization_mission_leg_identity(
    *,
    mission_leg_kind: MissionLegKind | str = MissionLegKind.COVERAGE,
    mission_leg_index: int | None = None,
    target_id: str = "",
    leg_index: int | None = None,
    target_viewpoint_id: str = "",
) -> tuple[MissionLegKind, int, str]:
    """Resolve generic identity with strict coverage-alias compatibility."""

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
class MissionMotionAuthorization:
    """The narrow scope attached to the mission-level typed ``RUN``."""

    session_id: str
    robot_id: str
    namespace: str
    cmd_vel_topic: str
    semantic_map_id: str
    localization_branch_proof_id: str
    max_runtime_reseals_per_leg: int
    scope_text: str
    operator_confirmation: str
    allowed_recovery_kind: str
    allowed_mission_leg_kinds: tuple[MissionLegKind, ...] = (
        MissionLegKind.COVERAGE,
    )
    schema_version: int = MISSION_MOTION_AUTHORIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_mission_leg_kinds",
            _canonical_allowed_mission_leg_kinds(
                self.allowed_mission_leg_kinds
            ),
        )
        _validate_mission(self)

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "robot_id": self.robot_id,
            "namespace": self.namespace,
            "cmd_vel_topic": self.cmd_vel_topic,
            "semantic_map_id": self.semantic_map_id,
            "localization_branch_proof_id": self.localization_branch_proof_id,
            "max_runtime_reseals_per_leg": self.max_runtime_reseals_per_leg,
            "scope_text": self.scope_text,
            "operator_confirmation": self.operator_confirmation,
            "allowed_recovery_kind": self.allowed_recovery_kind,
        }
        if self.schema_version >= MISSION_MOTION_AUTHORIZATION_SCHEMA_VERSION:
            payload["allowed_mission_leg_kinds"] = [
                kind.value for kind in self.allowed_mission_leg_kinds
            ]
        return payload

    def to_evidence(self) -> dict[str, object]:
        return self.to_payload()


@dataclass(frozen=True)
class RuntimeLocalizationMotionPermit:
    """One auditable child-run permit derived from the mission authorization."""

    master_authorization_sha256: str
    master_authorization_path: str
    run_id: str
    leg_index: int
    target_viewpoint_id: str
    reseal_index: int
    max_runtime_reseals_per_leg: int
    rejected_run_id: str
    runtime_reseal_decision_evidence: Mapping[str, object]
    runtime_reseal_decision_sha256: str
    fresh_localization_evidence_path: str
    fresh_localization_evidence_sha256: str
    route_csv_path: str
    route_csv_sha256: str
    diagnostics_path: str
    diagnostics_sha256: str
    map_route_certificate_path: str
    map_route_certificate_sha256: str
    dry_odom_certificate_path: str
    dry_odom_certificate_sha256: str
    dry_uncertainty_budget_path: str
    dry_uncertainty_budget_sha256: str
    dry_preflight_path: str
    dry_preflight_sha256: str
    same_target_verified: bool
    dry_run_passed: bool
    additional_typed_run_required: bool
    mission_leg_kind: MissionLegKind = MissionLegKind.COVERAGE
    mission_leg_index: int | None = None
    target_id: str = ""
    schema_version: int = RUNTIME_LOCALIZATION_MOTION_PERMIT_SCHEMA_VERSION

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
        # Prevent callers from mutating the decision after construction while
        # retaining a normal JSON mapping at the public boundary.
        evidence = _canonical_decision_copy(self.runtime_reseal_decision_evidence)
        object.__setattr__(
            self,
            "runtime_reseal_decision_evidence",
            MappingProxyType(evidence),
        )
        _validate_permit(self)

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "master_authorization_sha256": self.master_authorization_sha256,
            "master_authorization_path": self.master_authorization_path,
            "run_id": self.run_id,
            "leg_index": self.leg_index,
            "target_viewpoint_id": self.target_viewpoint_id,
            "reseal_index": self.reseal_index,
            "max_runtime_reseals_per_leg": self.max_runtime_reseals_per_leg,
            "rejected_run_id": self.rejected_run_id,
            "runtime_reseal_decision_evidence": dict(
                self.runtime_reseal_decision_evidence
            ),
            "runtime_reseal_decision_sha256": (
                self.runtime_reseal_decision_sha256
            ),
            "fresh_localization_evidence_path": (
                self.fresh_localization_evidence_path
            ),
            "fresh_localization_evidence_sha256": (
                self.fresh_localization_evidence_sha256
            ),
            "route_csv_path": self.route_csv_path,
            "route_csv_sha256": self.route_csv_sha256,
            "diagnostics_path": self.diagnostics_path,
            "diagnostics_sha256": self.diagnostics_sha256,
            "map_route_certificate_path": self.map_route_certificate_path,
            "map_route_certificate_sha256": self.map_route_certificate_sha256,
            "dry_odom_certificate_path": self.dry_odom_certificate_path,
            "dry_odom_certificate_sha256": self.dry_odom_certificate_sha256,
            "dry_uncertainty_budget_path": self.dry_uncertainty_budget_path,
            "dry_uncertainty_budget_sha256": (
                self.dry_uncertainty_budget_sha256
            ),
            "dry_preflight_path": self.dry_preflight_path,
            "dry_preflight_sha256": self.dry_preflight_sha256,
            "same_target_verified": self.same_target_verified,
            "dry_run_passed": self.dry_run_passed,
            "additional_typed_run_required": self.additional_typed_run_required,
        }
        if self.schema_version >= RUNTIME_LOCALIZATION_MOTION_PERMIT_SCHEMA_VERSION:
            payload.update(
                {
                    "mission_leg_kind": self.mission_leg_kind.value,
                    "mission_leg_index": self.mission_leg_index,
                    "target_id": self.target_id,
                }
            )
        return payload

    def to_evidence(self) -> dict[str, object]:
        return self.to_payload()


def file_sha256(path: Path) -> str:
    """Hash a normal file, rejecting absent, non-file, and symlink inputs."""

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


def mission_motion_authorization_sha256(
    authorization: MissionMotionAuthorization,
) -> str:
    _validate_mission(authorization)
    return payload_sha256(authorization.to_payload())


def write_mission_motion_authorization(
    path: Path,
    authorization: MissionMotionAuthorization,
) -> str:
    _validate_mission(authorization)
    try:
        return write_content_hashed_json(
            Path(path),
            authorization.to_payload(),
            hash_field=MISSION_MOTION_AUTHORIZATION_HASH_FIELD,
        )
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc


def load_mission_motion_authorization(path: Path) -> MissionMotionAuthorization:
    try:
        payload = load_content_hashed_json(
            Path(path), hash_field=MISSION_MOTION_AUTHORIZATION_HASH_FIELD
        )
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc
    schema_version = _integer(payload.get("schema_version"), "schema_version")
    expected_fields = (
        _MISSION_FIELDS_V1
        if schema_version == LEGACY_MISSION_MOTION_AUTHORIZATION_SCHEMA_VERSION
        else _MISSION_FIELDS_V2
    )
    if frozenset(payload) != expected_fields:
        raise ValueError("mission motion authorization fields mismatch")
    try:
        return MissionMotionAuthorization(
            schema_version=schema_version,
            session_id=_string(payload["session_id"], "session_id"),
            robot_id=_string(payload["robot_id"], "robot_id"),
            namespace=_string(payload["namespace"], "namespace"),
            cmd_vel_topic=_string(payload["cmd_vel_topic"], "cmd_vel_topic"),
            semantic_map_id=_string(payload["semantic_map_id"], "semantic_map_id"),
            localization_branch_proof_id=_string(
                payload["localization_branch_proof_id"],
                "localization_branch_proof_id",
            ),
            max_runtime_reseals_per_leg=_integer(
                payload["max_runtime_reseals_per_leg"],
                "max_runtime_reseals_per_leg",
            ),
            scope_text=_string(payload["scope_text"], "scope_text"),
            operator_confirmation=_string(
                payload["operator_confirmation"], "operator_confirmation"
            ),
            allowed_recovery_kind=_string(
                payload["allowed_recovery_kind"], "allowed_recovery_kind"
            ),
            allowed_mission_leg_kinds=(
                (MissionLegKind.COVERAGE,)
                if schema_version
                == LEGACY_MISSION_MOTION_AUTHORIZATION_SCHEMA_VERSION
                else tuple(
                    _mission_leg_kind(value, "allowed_mission_leg_kinds")
                    for value in _sequence(
                        payload["allowed_mission_leg_kinds"],
                        "allowed_mission_leg_kinds",
                    )
                )
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid mission motion authorization: {exc}") from exc


def validate_mission_motion_authorization(
    authorization_path: Path,
    *,
    session_id: str,
    robot_id: str,
    namespace: str,
    cmd_vel_topic: str,
    semantic_map_id: str,
    localization_branch_proof_id: str,
) -> MissionMotionAuthorization:
    """Integrity-load and bind a mission authorization to its live scope."""

    authorization = load_mission_motion_authorization(authorization_path)
    checks = {
        "session_id": (authorization.session_id, session_id),
        "robot_id": (authorization.robot_id, robot_id),
        "namespace": (authorization.namespace, namespace),
        "cmd_vel_topic": (authorization.cmd_vel_topic, cmd_vel_topic),
        "semantic_map_id": (authorization.semantic_map_id, semantic_map_id),
        "localization_branch_proof_id": (
            authorization.localization_branch_proof_id,
            localization_branch_proof_id,
        ),
    }
    _require_exact_matches("mission motion authorization", checks)
    return authorization


def runtime_localization_motion_permit_sha256(
    permit: RuntimeLocalizationMotionPermit,
) -> str:
    _validate_permit(permit)
    return payload_sha256(permit.to_payload())


def write_runtime_localization_motion_permit(
    path: Path,
    permit: RuntimeLocalizationMotionPermit,
) -> str:
    """Verify every reference, then immutably publish the permit."""

    _validate_permit(permit)
    _validate_permit_references(permit)
    try:
        return write_content_hashed_json(
            Path(path),
            permit.to_payload(),
            hash_field=RUNTIME_LOCALIZATION_MOTION_PERMIT_HASH_FIELD,
        )
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc


def load_runtime_localization_motion_permit(
    path: Path,
) -> RuntimeLocalizationMotionPermit:
    try:
        payload = load_content_hashed_json(
            Path(path), hash_field=RUNTIME_LOCALIZATION_MOTION_PERMIT_HASH_FIELD
        )
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc
    schema_version = _integer(payload.get("schema_version"), "schema_version")
    expected_fields = (
        _PERMIT_FIELDS_V1
        if schema_version
        == LEGACY_RUNTIME_LOCALIZATION_MOTION_PERMIT_SCHEMA_VERSION
        else _PERMIT_FIELDS_V2
    )
    if frozenset(payload) != expected_fields:
        raise ValueError("runtime localization motion permit fields mismatch")
    try:
        decision = payload["runtime_reseal_decision_evidence"]
        if not isinstance(decision, Mapping):
            raise ValueError("runtime_reseal_decision_evidence must be an object")
        return RuntimeLocalizationMotionPermit(
            schema_version=schema_version,
            master_authorization_sha256=_string(
                payload["master_authorization_sha256"],
                "master_authorization_sha256",
            ),
            master_authorization_path=_string(
                payload["master_authorization_path"], "master_authorization_path"
            ),
            run_id=_string(payload["run_id"], "run_id"),
            leg_index=_integer(payload["leg_index"], "leg_index"),
            target_viewpoint_id=_string(
                payload["target_viewpoint_id"], "target_viewpoint_id"
            ),
            reseal_index=_integer(payload["reseal_index"], "reseal_index"),
            max_runtime_reseals_per_leg=_integer(
                payload["max_runtime_reseals_per_leg"],
                "max_runtime_reseals_per_leg",
            ),
            rejected_run_id=_string(
                payload["rejected_run_id"], "rejected_run_id"
            ),
            runtime_reseal_decision_evidence=decision,
            runtime_reseal_decision_sha256=_string(
                payload["runtime_reseal_decision_sha256"],
                "runtime_reseal_decision_sha256",
            ),
            fresh_localization_evidence_path=_string(
                payload["fresh_localization_evidence_path"],
                "fresh_localization_evidence_path",
            ),
            fresh_localization_evidence_sha256=_string(
                payload["fresh_localization_evidence_sha256"],
                "fresh_localization_evidence_sha256",
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
            dry_preflight_path=_string(
                payload["dry_preflight_path"], "dry_preflight_path"
            ),
            dry_preflight_sha256=_string(
                payload["dry_preflight_sha256"], "dry_preflight_sha256"
            ),
            same_target_verified=_boolean(
                payload["same_target_verified"], "same_target_verified"
            ),
            dry_run_passed=_boolean(payload["dry_run_passed"], "dry_run_passed"),
            additional_typed_run_required=_boolean(
                payload["additional_typed_run_required"],
                "additional_typed_run_required",
            ),
            mission_leg_kind=(
                MissionLegKind.COVERAGE
                if schema_version
                == LEGACY_RUNTIME_LOCALIZATION_MOTION_PERMIT_SCHEMA_VERSION
                else _mission_leg_kind(
                    payload["mission_leg_kind"], "mission_leg_kind"
                )
            ),
            mission_leg_index=(
                _integer(payload["leg_index"], "leg_index")
                if schema_version
                == LEGACY_RUNTIME_LOCALIZATION_MOTION_PERMIT_SCHEMA_VERSION
                else _integer(
                    payload["mission_leg_index"], "mission_leg_index"
                )
            ),
            target_id=(
                _string(payload["target_viewpoint_id"], "target_viewpoint_id")
                if schema_version
                == LEGACY_RUNTIME_LOCALIZATION_MOTION_PERMIT_SCHEMA_VERSION
                else _string(payload["target_id"], "target_id")
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid runtime localization motion permit: {exc}") from exc


def validate_runtime_localization_motion_permit(
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
    route_csv_sha256: str,
    diagnostics_path: Path,
    diagnostics_sha256: str,
    map_route_certificate_path: Path,
    map_route_certificate_sha256: str,
    dry_odom_certificate_path: Path,
    dry_odom_certificate_sha256: str,
    dry_uncertainty_budget_path: Path,
    dry_uncertainty_budget_sha256: str,
    dry_preflight_path: Path,
    dry_preflight_sha256: str,
    mission_leg_kind: MissionLegKind | str = MissionLegKind.COVERAGE,
    mission_leg_index: int | None = None,
    target_id: str = "",
) -> RuntimeLocalizationMotionPermit:
    """Full exact-value validation surface for an execution child."""

    permit = load_runtime_localization_motion_permit(permit_path)
    expected_kind, expected_index, expected_target = (
        resolve_runtime_localization_mission_leg_identity(
            mission_leg_kind=mission_leg_kind,
            mission_leg_index=mission_leg_index,
            target_id=target_id,
            leg_index=leg_index,
            target_viewpoint_id=target_viewpoint_id,
        )
    )
    authorization = _validate_master_reference(permit, master_authorization_path)
    checks = {
        "run_id": (permit.run_id, run_id),
        "session_id": (authorization.session_id, session_id),
        "robot_id": (authorization.robot_id, robot_id),
        "namespace": (authorization.namespace, namespace),
        "cmd_vel_topic": (authorization.cmd_vel_topic, cmd_vel_topic),
        "semantic_map_id": (authorization.semantic_map_id, semantic_map_id),
        "target_viewpoint_id": (
            permit.target_viewpoint_id,
            target_viewpoint_id,
        ),
        "leg_index": (permit.leg_index, leg_index),
        "mission_leg_kind": (permit.mission_leg_kind, expected_kind),
        "mission_leg_index": (permit.mission_leg_index, expected_index),
        "target_id": (permit.target_id, expected_target),
        "localization_branch_proof_id": (
            authorization.localization_branch_proof_id,
            localization_branch_proof_id,
        ),
    }
    _require_exact_matches("runtime localization motion permit", checks)

    artifact_inputs = (
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
        (
            "dry_preflight",
            permit.dry_preflight_path,
            permit.dry_preflight_sha256,
            dry_preflight_path,
            dry_preflight_sha256,
        ),
    )
    for name, sealed_path, sealed_hash, live_path, supplied_hash in artifact_inputs:
        _validate_artifact_binding(
            name,
            sealed_path=sealed_path,
            sealed_sha256=sealed_hash,
            live_path=live_path,
            supplied_sha256=supplied_hash,
        )
    _validate_bound_artifact(
        "fresh_localization_evidence",
        permit.fresh_localization_evidence_path,
        permit.fresh_localization_evidence_sha256,
    )
    _validate_budget(permit, authorization)
    return permit


def validate_runtime_localization_motion_permit_for_execution(
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
) -> RuntimeLocalizationMotionPermit:
    """Validate a permit while deriving sealed dry-artifact inputs internally."""

    permit = load_runtime_localization_motion_permit(permit_path)
    # The full validator performs another immutable load.  This is intentional:
    # it keeps the single strict implementation authoritative even if the path
    # is swapped between reads; either read must be a valid sealed permit.
    return validate_runtime_localization_motion_permit(
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
        route_csv_path=route_csv_path,
        route_csv_sha256=permit.route_csv_sha256,
        diagnostics_path=diagnostics_path,
        diagnostics_sha256=permit.diagnostics_sha256,
        map_route_certificate_path=map_route_certificate_path,
        map_route_certificate_sha256=permit.map_route_certificate_sha256,
        dry_odom_certificate_path=Path(permit.dry_odom_certificate_path),
        dry_odom_certificate_sha256=permit.dry_odom_certificate_sha256,
        dry_uncertainty_budget_path=Path(permit.dry_uncertainty_budget_path),
        dry_uncertainty_budget_sha256=permit.dry_uncertainty_budget_sha256,
        dry_preflight_path=Path(permit.dry_preflight_path),
        dry_preflight_sha256=permit.dry_preflight_sha256,
        mission_leg_kind=mission_leg_kind,
        mission_leg_index=mission_leg_index,
        target_id=target_id,
    )


def _validate_mission(authorization: MissionMotionAuthorization) -> None:
    if authorization.schema_version not in {
        LEGACY_MISSION_MOTION_AUTHORIZATION_SCHEMA_VERSION,
        MISSION_MOTION_AUTHORIZATION_SCHEMA_VERSION,
    }:
        raise ValueError("unsupported mission motion authorization schema")
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
        authorization.max_runtime_reseals_per_leg,
        "max_runtime_reseals_per_leg",
    )
    if authorization.scope_text != MISSION_MOTION_AUTHORIZATION_SCOPE:
        raise ValueError("mission motion authorization scope_text mismatch")
    if authorization.operator_confirmation != MISSION_RUN_CONFIRMATION:
        raise ValueError("mission motion authorization requires operator confirmation RUN")
    if (
        authorization.allowed_recovery_kind
        != RUNTIME_LOCALIZATION_RESEAL_RECOVERY_KIND
    ):
        raise ValueError("mission motion authorization recovery kind mismatch")
    allowed = _canonical_allowed_mission_leg_kinds(
        authorization.allowed_mission_leg_kinds
    )
    if (
        authorization.schema_version
        == LEGACY_MISSION_MOTION_AUTHORIZATION_SCHEMA_VERSION
        and allowed != (MissionLegKind.COVERAGE,)
    ):
        raise ValueError(
            "legacy mission motion authorization supports coverage only"
        )


def _validate_permit(permit: RuntimeLocalizationMotionPermit) -> None:
    if permit.schema_version not in {
        LEGACY_RUNTIME_LOCALIZATION_MOTION_PERMIT_SCHEMA_VERSION,
        RUNTIME_LOCALIZATION_MOTION_PERMIT_SCHEMA_VERSION,
    }:
        raise ValueError("unsupported runtime localization motion permit schema")
    _require_sha256(
        permit.master_authorization_sha256, "master_authorization_sha256"
    )
    for name in (
        "master_authorization_path",
        "run_id",
        "target_viewpoint_id",
        "rejected_run_id",
        "fresh_localization_evidence_path",
        "route_csv_path",
        "diagnostics_path",
        "map_route_certificate_path",
        "dry_odom_certificate_path",
        "dry_uncertainty_budget_path",
        "dry_preflight_path",
    ):
        _require_nonempty(getattr(permit, name), name)
    _nonnegative_integer(permit.leg_index, "leg_index")
    kind, index, target = resolve_runtime_localization_mission_leg_identity(
        mission_leg_kind=permit.mission_leg_kind,
        mission_leg_index=permit.mission_leg_index,
        target_id=permit.target_id,
        leg_index=permit.leg_index,
        target_viewpoint_id=permit.target_viewpoint_id,
    )
    if (
        permit.schema_version
        == LEGACY_RUNTIME_LOCALIZATION_MOTION_PERMIT_SCHEMA_VERSION
        and kind is not MissionLegKind.COVERAGE
    ):
        raise ValueError(
            "legacy runtime localization permit supports coverage only"
        )
    if index != permit.leg_index or target != permit.target_viewpoint_id:
        raise ValueError("runtime localization permit identity aliases mismatch")
    _positive_integer(permit.reseal_index, "reseal_index")
    _positive_integer(
        permit.max_runtime_reseals_per_leg,
        "max_runtime_reseals_per_leg",
    )
    if permit.reseal_index > permit.max_runtime_reseals_per_leg:
        raise ValueError("runtime localization motion permit reseal budget exceeded")
    if permit.rejected_run_id == permit.run_id:
        raise ValueError("rejected_run_id must differ from run_id")
    for name in (
        "runtime_reseal_decision_sha256",
        "fresh_localization_evidence_sha256",
        "route_csv_sha256",
        "diagnostics_sha256",
        "map_route_certificate_sha256",
        "dry_odom_certificate_sha256",
        "dry_uncertainty_budget_sha256",
        "dry_preflight_sha256",
    ):
        _require_sha256(getattr(permit, name), name)
    _validate_runtime_reseal_decision(
        permit.runtime_reseal_decision_evidence,
        permit.runtime_reseal_decision_sha256,
    )
    if permit.same_target_verified is not True:
        raise ValueError("runtime localization motion permit requires same_target_verified=true")
    if permit.dry_run_passed is not True:
        raise ValueError("runtime localization motion permit requires dry_run_passed=true")
    if permit.additional_typed_run_required is not False:
        raise ValueError(
            "runtime localization motion permit requires "
            "additional_typed_run_required=false"
        )


def _validate_runtime_reseal_decision(
    evidence: Mapping[str, object], expected_sha256: str
) -> None:
    if frozenset(evidence) != _DECISION_FIELDS:
        raise ValueError("runtime reseal decision evidence fields mismatch")
    expected_values = {
        "schema_version": 1,
        "eligible": True,
        "reason": "runtime_localization_reseal_required",
        "execution_phase": "after_motion",
        "motion_published": True,
        "requires_fresh_localization": True,
        "requires_new_route_certificate": True,
        "requires_fresh_typed_run": True,
        "automatic_motion_authorized": False,
    }
    for name, expected in expected_values.items():
        if evidence.get(name) != expected or type(evidence.get(name)) is not type(
            expected
        ):
            raise ValueError(f"runtime reseal decision evidence {name} mismatch")
    _require_nonempty(evidence.get("continuity_reason"), "continuity_reason")
    actual_sha256 = payload_sha256(dict(evidence))
    if actual_sha256 != expected_sha256:
        raise ValueError("runtime reseal decision evidence hash mismatch")


def _validate_permit_references(permit: RuntimeLocalizationMotionPermit) -> None:
    authorization = _validate_master_reference(
        permit, Path(permit.master_authorization_path)
    )
    for name, path, digest in _permit_artifacts(permit):
        _validate_bound_artifact(name, path, digest)
    _validate_budget(permit, authorization)


def _validate_master_reference(
    permit: RuntimeLocalizationMotionPermit,
    master_authorization_path: Path,
) -> MissionMotionAuthorization:
    observed_path = _normal_file_path(master_authorization_path)
    if observed_path != permit.master_authorization_path:
        raise ValueError("runtime localization motion permit master authorization path mismatch")
    authorization = load_mission_motion_authorization(master_authorization_path)
    actual_sha256 = mission_motion_authorization_sha256(authorization)
    if actual_sha256 != permit.master_authorization_sha256:
        raise ValueError("runtime localization motion permit master authorization hash mismatch")
    return authorization


def _validate_budget(
    permit: RuntimeLocalizationMotionPermit,
    authorization: MissionMotionAuthorization,
) -> None:
    if (
        permit.max_runtime_reseals_per_leg
        != authorization.max_runtime_reseals_per_leg
    ):
        raise ValueError("runtime localization motion permit reseal maximum mismatch")
    if not 1 <= permit.reseal_index <= authorization.max_runtime_reseals_per_leg:
        raise ValueError("runtime localization motion permit reseal budget exceeded")
    if permit.mission_leg_kind not in authorization.allowed_mission_leg_kinds:
        raise ValueError(
            "runtime localization motion permit mission leg kind is not authorized"
        )


def _permit_artifacts(
    permit: RuntimeLocalizationMotionPermit,
) -> tuple[tuple[str, str, str], ...]:
    return (
        (
            "fresh_localization_evidence",
            permit.fresh_localization_evidence_path,
            permit.fresh_localization_evidence_sha256,
        ),
        ("route_csv", permit.route_csv_path, permit.route_csv_sha256),
        ("diagnostics", permit.diagnostics_path, permit.diagnostics_sha256),
        (
            "map_route_certificate",
            permit.map_route_certificate_path,
            permit.map_route_certificate_sha256,
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
        (
            "dry_preflight",
            permit.dry_preflight_path,
            permit.dry_preflight_sha256,
        ),
    )


def _validate_bound_artifact(name: str, path: str, expected_sha256: str) -> None:
    source = Path(path)
    observed_path = _normal_file_path(source)
    if observed_path != path:
        raise ValueError(f"runtime localization motion permit {name} path is not canonical")
    actual_sha256 = file_sha256(source)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"runtime localization motion permit {name} hash mismatch")


def _validate_artifact_binding(
    name: str,
    *,
    sealed_path: str,
    sealed_sha256: str,
    live_path: Path,
    supplied_sha256: str,
) -> None:
    _require_sha256(supplied_sha256, f"{name}_sha256")
    observed_path = _normal_file_path(live_path)
    if observed_path != sealed_path:
        raise ValueError(f"runtime localization motion permit {name} path mismatch")
    if supplied_sha256 != sealed_sha256:
        raise ValueError(f"runtime localization motion permit {name} supplied hash mismatch")
    actual_sha256 = file_sha256(live_path)
    if actual_sha256 != sealed_sha256:
        raise ValueError(f"runtime localization motion permit {name} hash mismatch")


def _normal_file_path(path: Path) -> str:
    source = Path(path)
    # Hashing first provides the stable symlink/non-file errors.
    file_sha256(source)
    return str(source.absolute())


def _require_exact_matches(
    label: str,
    checks: Mapping[str, tuple[object, object]],
) -> None:
    for name, (sealed, observed) in checks.items():
        if sealed != observed or (
            isinstance(sealed, (bool, int)) and type(sealed) is not type(observed)
        ):
            raise ValueError(f"{label} {name} mismatch")


def _canonical_decision_copy(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("runtime_reseal_decision_evidence must be an object")
    try:
        decoded = json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))
    except ContentStoreError as exc:
        raise ValueError(f"invalid runtime reseal decision evidence: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError("runtime_reseal_decision_evidence must be an object")
    return decoded


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


def _sequence(value: object, name: str) -> tuple[object, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{name} must be a sequence")
    return tuple(value)


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


__all__ = [
    "MISSION_MOTION_AUTHORIZATION_HASH_FIELD",
    "MISSION_MOTION_AUTHORIZATION_SCHEMA_VERSION",
    "MISSION_MOTION_AUTHORIZATION_SCOPE",
    "MISSION_RUN_CONFIRMATION",
    "MissionMotionAuthorization",
    "RUNTIME_LOCALIZATION_MOTION_PERMIT_HASH_FIELD",
    "RUNTIME_LOCALIZATION_MOTION_PERMIT_SCHEMA_VERSION",
    "RUNTIME_LOCALIZATION_RESEAL_RECOVERY_KIND",
    "RuntimeLocalizationMotionPermit",
    "file_sha256",
    "load_mission_motion_authorization",
    "load_runtime_localization_motion_permit",
    "mission_motion_authorization_sha256",
    "resolve_runtime_localization_mission_leg_identity",
    "runtime_localization_motion_permit_sha256",
    "validate_mission_motion_authorization",
    "validate_runtime_localization_motion_permit",
    "validate_runtime_localization_motion_permit_for_execution",
    "write_mission_motion_authorization",
    "write_runtime_localization_motion_permit",
]
