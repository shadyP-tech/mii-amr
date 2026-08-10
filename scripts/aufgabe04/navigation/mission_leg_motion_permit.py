"""Immutable motion authorization for one routine autonomous mission leg.

This module is deliberately ROS-free.  One operator-confirmed mission-level
``RUN`` may be sealed as a :class:`MissionLegMotionAuthorization`.  Each child
motion process must then receive a separate content-hashed permit that binds
one exact routine leg, target, route, certificate set, and dry-run result.

The contract does not inherit the narrower runtime-localization-reseal
authorization.  It also does not authorize startup resealing: that value is a
recognized leg kind only so callers fail with a precise fresh-``RUN`` error.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

from scripts.aufgabe04.artifacts.content_store import (
    ContentStoreError,
    load_content_hashed_json,
    payload_sha256,
    write_content_hashed_json,
)


MISSION_LEG_MOTION_AUTHORIZATION_SCHEMA_VERSION = 1
MISSION_LEG_MOTION_PERMIT_SCHEMA_VERSION = 1

MISSION_LEG_MOTION_AUTHORIZATION_HASH_FIELD = (
    "mission_leg_motion_authorization_sha256"
)
MISSION_LEG_MOTION_PERMIT_HASH_FIELD = "mission_leg_motion_permit_sha256"

MISSION_LEG_RUN_CONFIRMATION = "RUN"
MISSION_LEG_MOTION_AUTHORIZATION_SCOPE = (
    "Reuse this autonomous mission RUN only for separately sealed routine "
    "child legs whose exact run, leg, target, route, certificates, and passed "
    "dry run are bound by a mission-leg motion permit; startup reseals, "
    "recovery motion, target changes, artifact changes, and motion without an "
    "exact permit are not authorized."
)


class MissionLegKind(str, Enum):
    """Known mission leg kinds, including the explicitly excluded reseal."""

    COVERAGE = "coverage"
    CANDIDATE_PREAPPROACH = "candidate_preapproach"
    OPPOSITE_FACE = "opposite_face"
    STARTUP_RESEAL = "startup_reseal"


ROUTINE_MISSION_LEG_KINDS = (
    MissionLegKind.COVERAGE,
    MissionLegKind.CANDIDATE_PREAPPROACH,
    MissionLegKind.OPPOSITE_FACE,
)

_LEG_KIND_ORDER = {
    kind: index for index, kind in enumerate(ROUTINE_MISSION_LEG_KINDS)
}
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
        "allowed_leg_kinds",
        "scope_text",
        "operator_confirmation",
    }
)
_PERMIT_FIELDS = frozenset(
    {
        "schema_version",
        "master_authorization_sha256",
        "master_authorization_path",
        "session_id",
        "robot_id",
        "namespace",
        "cmd_vel_topic",
        "semantic_map_id",
        "localization_branch_proof_id",
        "run_id",
        "mission_leg_kind",
        "mission_leg_index",
        "target_id",
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
        "dry_run_passed",
        "additional_typed_run_required",
    }
)


@dataclass(frozen=True)
class MissionLegMotionAuthorization:
    """The routine child-leg scope attached to one mission-level ``RUN``."""

    session_id: str
    robot_id: str
    namespace: str
    cmd_vel_topic: str
    semantic_map_id: str
    localization_branch_proof_id: str
    allowed_leg_kinds: tuple[MissionLegKind, ...]
    scope_text: str
    operator_confirmation: str
    schema_version: int = MISSION_LEG_MOTION_AUTHORIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_leg_kinds",
            _canonical_allowed_leg_kinds(self.allowed_leg_kinds),
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
            "allowed_leg_kinds": [
                kind.value for kind in self.allowed_leg_kinds
            ],
            "scope_text": self.scope_text,
            "operator_confirmation": self.operator_confirmation,
        }

    def to_evidence(self) -> dict[str, object]:
        return self.to_payload()


@dataclass(frozen=True)
class MissionLegMotionPermit:
    """One exact routine child leg derived from a mission authorization."""

    master_authorization_sha256: str
    master_authorization_path: str
    session_id: str
    robot_id: str
    namespace: str
    cmd_vel_topic: str
    semantic_map_id: str
    localization_branch_proof_id: str
    run_id: str
    mission_leg_kind: MissionLegKind
    mission_leg_index: int
    target_id: str
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
    dry_run_passed: bool
    additional_typed_run_required: bool
    schema_version: int = MISSION_LEG_MOTION_PERMIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mission_leg_kind",
            _mission_leg_kind(self.mission_leg_kind, "mission_leg_kind"),
        )
        _validate_permit(self)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "master_authorization_sha256": self.master_authorization_sha256,
            "master_authorization_path": self.master_authorization_path,
            "session_id": self.session_id,
            "robot_id": self.robot_id,
            "namespace": self.namespace,
            "cmd_vel_topic": self.cmd_vel_topic,
            "semantic_map_id": self.semantic_map_id,
            "localization_branch_proof_id": (
                self.localization_branch_proof_id
            ),
            "run_id": self.run_id,
            "mission_leg_kind": self.mission_leg_kind.value,
            "mission_leg_index": self.mission_leg_index,
            "target_id": self.target_id,
            "route_csv_path": self.route_csv_path,
            "route_csv_sha256": self.route_csv_sha256,
            "diagnostics_path": self.diagnostics_path,
            "diagnostics_sha256": self.diagnostics_sha256,
            "map_route_certificate_path": self.map_route_certificate_path,
            "map_route_certificate_sha256": (
                self.map_route_certificate_sha256
            ),
            "dry_preflight_path": self.dry_preflight_path,
            "dry_preflight_sha256": self.dry_preflight_sha256,
            "dry_odom_certificate_path": self.dry_odom_certificate_path,
            "dry_odom_certificate_sha256": (
                self.dry_odom_certificate_sha256
            ),
            "dry_uncertainty_budget_path": (
                self.dry_uncertainty_budget_path
            ),
            "dry_uncertainty_budget_sha256": (
                self.dry_uncertainty_budget_sha256
            ),
            "dry_run_passed": self.dry_run_passed,
            "additional_typed_run_required": (
                self.additional_typed_run_required
            ),
        }

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


def mission_leg_motion_authorization_sha256(
    authorization: MissionLegMotionAuthorization,
) -> str:
    _validate_authorization(authorization)
    return payload_sha256(authorization.to_payload())


def write_mission_leg_motion_authorization(
    path: Path,
    authorization: MissionLegMotionAuthorization,
) -> str:
    """Immutably publish one explicit mission-level routine-leg scope."""

    _validate_authorization(authorization)
    try:
        return write_content_hashed_json(
            Path(path),
            authorization.to_payload(),
            hash_field=MISSION_LEG_MOTION_AUTHORIZATION_HASH_FIELD,
        )
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc


def load_mission_leg_motion_authorization(
    path: Path,
) -> MissionLegMotionAuthorization:
    """Integrity-load an exact mission-level authorization artifact."""

    try:
        payload = load_content_hashed_json(
            Path(path),
            hash_field=MISSION_LEG_MOTION_AUTHORIZATION_HASH_FIELD,
        )
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc
    if frozenset(payload) != _AUTHORIZATION_FIELDS:
        raise ValueError("mission leg motion authorization fields mismatch")
    try:
        allowed = payload["allowed_leg_kinds"]
        if not isinstance(allowed, list):
            raise ValueError("allowed_leg_kinds must be an array")
        return MissionLegMotionAuthorization(
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
            allowed_leg_kinds=tuple(allowed),
            scope_text=_string(payload["scope_text"], "scope_text"),
            operator_confirmation=_string(
                payload["operator_confirmation"], "operator_confirmation"
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid mission leg motion authorization: {exc}") from exc


def validate_mission_leg_motion_authorization(
    authorization_path: Path,
    *,
    session_id: str,
    robot_id: str,
    namespace: str,
    cmd_vel_topic: str,
    semantic_map_id: str,
    localization_branch_proof_id: str,
    required_leg_kind: MissionLegKind | str | None = None,
) -> MissionLegMotionAuthorization:
    """Bind a sealed master authorization to its live mission identity."""

    authorization = load_mission_leg_motion_authorization(authorization_path)
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
    _require_exact_matches("mission leg motion authorization", checks)
    if required_leg_kind is not None:
        leg_kind = _mission_leg_kind(required_leg_kind, "required_leg_kind")
        _require_routine_leg_kind(leg_kind, "required_leg_kind")
        if leg_kind not in authorization.allowed_leg_kinds:
            raise ValueError(
                "mission leg motion authorization required_leg_kind is not allowed"
            )
    return authorization


def mission_leg_motion_permit_sha256(permit: MissionLegMotionPermit) -> str:
    _validate_permit(permit)
    return payload_sha256(permit.to_payload())


def write_mission_leg_motion_permit(
    path: Path,
    permit: MissionLegMotionPermit,
) -> str:
    """Verify every reference, then immutably publish one child permit."""

    _validate_permit(permit)
    _validate_permit_references(permit)
    try:
        return write_content_hashed_json(
            Path(path),
            permit.to_payload(),
            hash_field=MISSION_LEG_MOTION_PERMIT_HASH_FIELD,
        )
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc


def load_mission_leg_motion_permit(path: Path) -> MissionLegMotionPermit:
    """Integrity-load an exact one-leg permit artifact."""

    try:
        payload = load_content_hashed_json(
            Path(path), hash_field=MISSION_LEG_MOTION_PERMIT_HASH_FIELD
        )
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc
    if frozenset(payload) != _PERMIT_FIELDS:
        raise ValueError("mission leg motion permit fields mismatch")
    try:
        return MissionLegMotionPermit(
            schema_version=_integer(payload["schema_version"], "schema_version"),
            master_authorization_sha256=_string(
                payload["master_authorization_sha256"],
                "master_authorization_sha256",
            ),
            master_authorization_path=_string(
                payload["master_authorization_path"],
                "master_authorization_path",
            ),
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
            run_id=_string(payload["run_id"], "run_id"),
            mission_leg_kind=_string(
                payload["mission_leg_kind"], "mission_leg_kind"
            ),
            mission_leg_index=_integer(
                payload["mission_leg_index"], "mission_leg_index"
            ),
            target_id=_string(payload["target_id"], "target_id"),
            route_csv_path=_string(
                payload["route_csv_path"], "route_csv_path"
            ),
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
            dry_run_passed=_boolean(
                payload["dry_run_passed"], "dry_run_passed"
            ),
            additional_typed_run_required=_boolean(
                payload["additional_typed_run_required"],
                "additional_typed_run_required",
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid mission leg motion permit: {exc}") from exc


def validate_mission_leg_motion_permit(
    permit_path: Path,
    *,
    master_authorization_path: Path,
    session_id: str,
    robot_id: str,
    namespace: str,
    cmd_vel_topic: str,
    semantic_map_id: str,
    localization_branch_proof_id: str,
    run_id: str,
    mission_leg_kind: MissionLegKind | str,
    mission_leg_index: int,
    target_id: str,
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
) -> MissionLegMotionPermit:
    """Validate exact identities, paths, supplied hashes, and live bytes."""

    permit = load_mission_leg_motion_permit(permit_path)
    authorization = _validate_master_reference(permit, master_authorization_path)
    observed_leg_kind = _mission_leg_kind(
        mission_leg_kind, "mission_leg_kind"
    )
    _require_routine_leg_kind(observed_leg_kind, "mission_leg_kind")
    checks = {
        "session_id": (permit.session_id, session_id),
        "robot_id": (permit.robot_id, robot_id),
        "namespace": (permit.namespace, namespace),
        "cmd_vel_topic": (permit.cmd_vel_topic, cmd_vel_topic),
        "semantic_map_id": (permit.semantic_map_id, semantic_map_id),
        "localization_branch_proof_id": (
            permit.localization_branch_proof_id,
            localization_branch_proof_id,
        ),
        "run_id": (permit.run_id, run_id),
        "mission_leg_kind": (permit.mission_leg_kind, observed_leg_kind),
        "mission_leg_index": (permit.mission_leg_index, mission_leg_index),
        "target_id": (permit.target_id, target_id),
    }
    _require_exact_matches("mission leg motion permit", checks)
    if permit.mission_leg_kind not in authorization.allowed_leg_kinds:
        raise ValueError(
            "mission leg motion permit leg kind is not authorized by master"
        )

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
    for name, sealed_path, sealed_hash, live_path, supplied_hash in artifact_inputs:
        _validate_artifact_binding(
            name,
            sealed_path=sealed_path,
            sealed_sha256=sealed_hash,
            live_path=live_path,
            supplied_sha256=supplied_hash,
        )
    return permit


def validate_mission_leg_motion_permit_for_execution(
    permit_path: Path,
    *,
    master_authorization_path: Path,
    session_id: str,
    robot_id: str,
    namespace: str,
    cmd_vel_topic: str,
    semantic_map_id: str,
    localization_branch_proof_id: str,
    run_id: str,
    mission_leg_kind: MissionLegKind | str,
    mission_leg_index: int,
    target_id: str,
    route_csv_path: Path,
    diagnostics_path: Path,
    map_route_certificate_path: Path,
    dry_preflight_path: Path,
    dry_odom_certificate_path: Path,
    dry_uncertainty_budget_path: Path,
) -> MissionLegMotionPermit:
    """Path-first execution check deriving all sealed hashes internally."""

    permit = load_mission_leg_motion_permit(permit_path)
    # The full validator loads the immutable permit again.  If a path is
    # swapped between reads, both reads must independently be valid and every
    # binding from the first load must still match the authoritative second.
    return validate_mission_leg_motion_permit(
        permit_path,
        master_authorization_path=master_authorization_path,
        session_id=session_id,
        robot_id=robot_id,
        namespace=namespace,
        cmd_vel_topic=cmd_vel_topic,
        semantic_map_id=semantic_map_id,
        localization_branch_proof_id=localization_branch_proof_id,
        run_id=run_id,
        mission_leg_kind=mission_leg_kind,
        mission_leg_index=mission_leg_index,
        target_id=target_id,
        route_csv_path=route_csv_path,
        route_csv_sha256=permit.route_csv_sha256,
        diagnostics_path=diagnostics_path,
        diagnostics_sha256=permit.diagnostics_sha256,
        map_route_certificate_path=map_route_certificate_path,
        map_route_certificate_sha256=permit.map_route_certificate_sha256,
        dry_preflight_path=dry_preflight_path,
        dry_preflight_sha256=permit.dry_preflight_sha256,
        dry_odom_certificate_path=dry_odom_certificate_path,
        dry_odom_certificate_sha256=permit.dry_odom_certificate_sha256,
        dry_uncertainty_budget_path=dry_uncertainty_budget_path,
        dry_uncertainty_budget_sha256=(
            permit.dry_uncertainty_budget_sha256
        ),
    )


def _validate_authorization(
    authorization: MissionLegMotionAuthorization,
) -> None:
    if (
        type(authorization.schema_version) is not int
        or authorization.schema_version
        != MISSION_LEG_MOTION_AUTHORIZATION_SCHEMA_VERSION
    ):
        raise ValueError("unsupported mission leg motion authorization schema")
    for name in (
        "session_id",
        "robot_id",
        "cmd_vel_topic",
        "semantic_map_id",
        "localization_branch_proof_id",
    ):
        _require_nonempty(getattr(authorization, name), name)
    _require_namespace(authorization.namespace)
    if not authorization.allowed_leg_kinds:
        raise ValueError("allowed_leg_kinds must contain at least one routine leg")
    for kind in authorization.allowed_leg_kinds:
        _require_routine_leg_kind(kind, "allowed_leg_kinds")
    if authorization.scope_text != MISSION_LEG_MOTION_AUTHORIZATION_SCOPE:
        raise ValueError("mission leg motion authorization scope_text mismatch")
    if authorization.operator_confirmation != MISSION_LEG_RUN_CONFIRMATION:
        raise ValueError(
            "mission leg motion authorization requires operator confirmation RUN"
        )


def _validate_permit(permit: MissionLegMotionPermit) -> None:
    if (
        type(permit.schema_version) is not int
        or permit.schema_version != MISSION_LEG_MOTION_PERMIT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported mission leg motion permit schema")
    _require_sha256(
        permit.master_authorization_sha256, "master_authorization_sha256"
    )
    for name in (
        "session_id",
        "robot_id",
        "cmd_vel_topic",
        "semantic_map_id",
        "localization_branch_proof_id",
        "run_id",
        "target_id",
    ):
        _require_nonempty(getattr(permit, name), name)
    _require_namespace(permit.namespace)
    _require_routine_leg_kind(permit.mission_leg_kind, "mission_leg_kind")
    _nonnegative_integer(permit.mission_leg_index, "mission_leg_index")
    for name in (
        "master_authorization_path",
        "route_csv_path",
        "diagnostics_path",
        "map_route_certificate_path",
        "dry_preflight_path",
        "dry_odom_certificate_path",
        "dry_uncertainty_budget_path",
    ):
        _require_canonical_absolute_path(getattr(permit, name), name)
    for name in (
        "route_csv_sha256",
        "diagnostics_sha256",
        "map_route_certificate_sha256",
        "dry_preflight_sha256",
        "dry_odom_certificate_sha256",
        "dry_uncertainty_budget_sha256",
    ):
        _require_sha256(getattr(permit, name), name)
    if permit.dry_run_passed is not True:
        raise ValueError("mission leg motion permit requires dry_run_passed=true")
    if permit.additional_typed_run_required is not False:
        raise ValueError(
            "mission leg motion permit requires "
            "additional_typed_run_required=false"
        )


def _validate_permit_references(permit: MissionLegMotionPermit) -> None:
    _validate_master_reference(permit, Path(permit.master_authorization_path))
    for name, path, digest in _permit_artifacts(permit):
        _validate_bound_artifact(name, path, digest)


def _validate_master_reference(
    permit: MissionLegMotionPermit,
    master_authorization_path: Path,
) -> MissionLegMotionAuthorization:
    observed_path = _normal_file_path(master_authorization_path)
    if observed_path != permit.master_authorization_path:
        raise ValueError(
            "mission leg motion permit master authorization path mismatch"
        )
    authorization = load_mission_leg_motion_authorization(
        master_authorization_path
    )
    actual_sha256 = mission_leg_motion_authorization_sha256(authorization)
    if actual_sha256 != permit.master_authorization_sha256:
        raise ValueError(
            "mission leg motion permit master authorization hash mismatch"
        )
    checks = {
        "session_id": (authorization.session_id, permit.session_id),
        "robot_id": (authorization.robot_id, permit.robot_id),
        "namespace": (authorization.namespace, permit.namespace),
        "cmd_vel_topic": (authorization.cmd_vel_topic, permit.cmd_vel_topic),
        "semantic_map_id": (
            authorization.semantic_map_id,
            permit.semantic_map_id,
        ),
        "localization_branch_proof_id": (
            authorization.localization_branch_proof_id,
            permit.localization_branch_proof_id,
        ),
    }
    _require_exact_matches("mission leg motion permit master", checks)
    if permit.mission_leg_kind not in authorization.allowed_leg_kinds:
        raise ValueError(
            "mission leg motion permit leg kind is not authorized by master"
        )
    return authorization


def _permit_artifacts(
    permit: MissionLegMotionPermit,
) -> tuple[tuple[str, str, str], ...]:
    return (
        ("route_csv", permit.route_csv_path, permit.route_csv_sha256),
        (
            "diagnostics",
            permit.diagnostics_path,
            permit.diagnostics_sha256,
        ),
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
    source = Path(path)
    observed_path = _normal_file_path(source)
    if observed_path != path:
        raise ValueError(f"mission leg motion permit {name} path is not canonical")
    actual_sha256 = file_sha256(source)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"mission leg motion permit {name} hash mismatch")


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
        raise ValueError(f"mission leg motion permit {name} path mismatch")
    if supplied_sha256 != sealed_sha256:
        raise ValueError(
            f"mission leg motion permit {name} supplied hash mismatch"
        )
    actual_sha256 = file_sha256(live_path)
    if actual_sha256 != sealed_sha256:
        raise ValueError(f"mission leg motion permit {name} hash mismatch")


def _normal_file_path(path: Path) -> str:
    source = Path(path)
    # Hashing first provides stable symlink and non-file errors.
    file_sha256(source)
    return str(source.absolute())


def _canonical_allowed_leg_kinds(
    values: Sequence[MissionLegKind | str],
) -> tuple[MissionLegKind, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("allowed_leg_kinds must be an ordered sequence")
    kinds = tuple(
        _mission_leg_kind(value, "allowed_leg_kinds") for value in values
    )
    for kind in kinds:
        _require_routine_leg_kind(kind, "allowed_leg_kinds")
    if len(set(kinds)) != len(kinds):
        raise ValueError("allowed_leg_kinds must be unique")
    canonical = tuple(sorted(kinds, key=_LEG_KIND_ORDER.__getitem__))
    if kinds != canonical:
        raise ValueError("allowed_leg_kinds must use canonical routine order")
    return kinds


def _mission_leg_kind(value: object, name: str) -> MissionLegKind:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a mission leg kind string")
    try:
        return MissionLegKind(value)
    except ValueError as exc:
        raise ValueError(f"{name} is not a known mission leg kind") from exc


def _require_routine_leg_kind(kind: MissionLegKind, name: str) -> None:
    if kind is MissionLegKind.STARTUP_RESEAL:
        raise ValueError(f"{name} startup_reseal requires a separate typed RUN")
    if kind not in ROUTINE_MISSION_LEG_KINDS:
        raise ValueError(f"{name} is not a routine mission leg kind")


def _require_exact_matches(
    label: str,
    checks: Mapping[str, tuple[object, object]],
) -> None:
    for name, (sealed, observed) in checks.items():
        if sealed != observed or (
            isinstance(sealed, (bool, int)) and type(sealed) is not type(observed)
        ):
            raise ValueError(f"{label} {name} mismatch")


def _require_namespace(value: object) -> None:
    if not isinstance(value, str):
        raise ValueError("namespace must be a string")
    if value != value.strip():
        raise ValueError("namespace must be canonical")


def _require_nonempty(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string")


def _require_canonical_absolute_path(value: object, name: str) -> None:
    _require_nonempty(value, name)
    assert isinstance(value, str)
    source = Path(value)
    if not source.is_absolute() or str(source.absolute()) != value:
        raise ValueError(f"{name} must be a canonical absolute path")


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _nonnegative_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


__all__ = [
    "MISSION_LEG_MOTION_AUTHORIZATION_HASH_FIELD",
    "MISSION_LEG_MOTION_AUTHORIZATION_SCHEMA_VERSION",
    "MISSION_LEG_MOTION_AUTHORIZATION_SCOPE",
    "MISSION_LEG_MOTION_PERMIT_HASH_FIELD",
    "MISSION_LEG_MOTION_PERMIT_SCHEMA_VERSION",
    "MISSION_LEG_RUN_CONFIRMATION",
    "MissionLegKind",
    "MissionLegMotionAuthorization",
    "MissionLegMotionPermit",
    "ROUTINE_MISSION_LEG_KINDS",
    "file_sha256",
    "load_mission_leg_motion_authorization",
    "load_mission_leg_motion_permit",
    "mission_leg_motion_authorization_sha256",
    "mission_leg_motion_permit_sha256",
    "validate_mission_leg_motion_authorization",
    "validate_mission_leg_motion_permit",
    "validate_mission_leg_motion_permit_for_execution",
    "write_mission_leg_motion_authorization",
    "write_mission_leg_motion_permit",
]
