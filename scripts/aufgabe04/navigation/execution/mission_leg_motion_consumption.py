"""Atomic one-use claims for routine autonomous mission-leg permits.

A mission-leg permit is an immutable authorization artifact, not a reusable
capability.  This ROS-free module consumes an already validated permit with an
exclusive receipt immediately before the child publishes its first motion
event.  The receipt path is derived from the payload-bound master and permit
content, so copying identical permit bytes cannot create another claim slot.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.artifacts.content_store import (
    ContentStoreError,
    load_content_hashed_json,
    payload_sha256,
)
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
    MissionLegKind,
    MissionLegMotionPermit,
    load_mission_leg_motion_authorization,
    load_mission_leg_motion_permit,
    mission_leg_motion_authorization_sha256,
    mission_leg_motion_permit_sha256,
    validate_mission_leg_motion_permit_for_execution,
)


MISSION_LEG_MOTION_CONSUMPTION_RECEIPT_SCHEMA_VERSION = 1
MISSION_LEG_MOTION_CONSUMPTION_RECEIPT_HASH_FIELD = (
    "mission_leg_motion_consumption_receipt_sha256"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "mission_leg_motion_permit_path",
        "mission_leg_motion_permit_sha256",
        "session_id",
        "run_id",
        "mission_leg_kind",
        "mission_leg_index",
        "target_id",
    }
)


@dataclass(frozen=True)
class MissionLegMotionConsumptionReceipt:
    """Immutable identity of one successfully claimed routine-leg permit."""

    mission_leg_motion_permit_path: str
    mission_leg_motion_permit_sha256: str
    session_id: str
    run_id: str
    mission_leg_kind: MissionLegKind
    mission_leg_index: int
    target_id: str
    schema_version: int = MISSION_LEG_MOTION_CONSUMPTION_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.mission_leg_kind, MissionLegKind):
            try:
                kind = MissionLegKind(self.mission_leg_kind)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid mission_leg_kind") from exc
            object.__setattr__(self, "mission_leg_kind", kind)
        _validate_receipt(self)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mission_leg_motion_permit_path": (
                self.mission_leg_motion_permit_path
            ),
            "mission_leg_motion_permit_sha256": (
                self.mission_leg_motion_permit_sha256
            ),
            "session_id": self.session_id,
            "run_id": self.run_id,
            "mission_leg_kind": self.mission_leg_kind.value,
            "mission_leg_index": self.mission_leg_index,
            "target_id": self.target_id,
        }


def default_mission_leg_motion_consumption_receipt_path(
    permit_path: Path,
) -> Path:
    """Return the single claim path shared by identical permit copies."""

    _, permit, permit_sha256, master_path = _load_bound_permit(permit_path)
    return _receipt_path_from_binding(
        master_path=master_path,
        run_id=permit.run_id,
        permit_sha256=permit_sha256,
    )


def consume_mission_leg_motion_permit(
    *,
    permit_path: Path,
    permit: MissionLegMotionPermit,
    session_id: str,
    run_id: str,
    mission_leg_kind: MissionLegKind | str,
    mission_leg_index: int,
    target_id: str,
) -> MissionLegMotionConsumptionReceipt:
    """Atomically consume one fully validated permit exactly once."""

    if not isinstance(permit, MissionLegMotionPermit):
        raise ValueError("permit must be a MissionLegMotionPermit")
    try:
        observed_kind = MissionLegKind(mission_leg_kind)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid mission_leg_kind") from exc
    canonical_permit_path, observed, permit_sha256, master_path = (
        _load_bound_permit(permit_path)
    )
    if (
        permit_sha256 != mission_leg_motion_permit_sha256(permit)
        or observed.to_payload() != permit.to_payload()
    ):
        raise ValueError("mission leg motion permit changed after validation")
    checks = {
        "session_id": (observed.session_id, session_id),
        "run_id": (observed.run_id, run_id),
        "mission_leg_kind": (observed.mission_leg_kind, observed_kind),
        "mission_leg_index": (observed.mission_leg_index, mission_leg_index),
        "target_id": (observed.target_id, target_id),
    }
    _require_exact_matches(checks)
    # Close the validation-to-claim window for every referenced artifact.  The
    # permit itself supplies the canonical paths, while this call live-rehashes
    # their bytes immediately before the exclusive receipt is created.
    validate_mission_leg_motion_permit_for_execution(
        Path(canonical_permit_path),
        master_authorization_path=master_path,
        session_id=session_id,
        robot_id=observed.robot_id,
        namespace=observed.namespace,
        cmd_vel_topic=observed.cmd_vel_topic,
        semantic_map_id=observed.semantic_map_id,
        localization_branch_proof_id=(
            observed.localization_branch_proof_id
        ),
        run_id=run_id,
        mission_leg_kind=observed_kind,
        mission_leg_index=mission_leg_index,
        target_id=target_id,
        route_csv_path=Path(observed.route_csv_path),
        diagnostics_path=Path(observed.diagnostics_path),
        map_route_certificate_path=Path(
            observed.map_route_certificate_path
        ),
        dry_preflight_path=Path(observed.dry_preflight_path),
        dry_odom_certificate_path=Path(
            observed.dry_odom_certificate_path
        ),
        dry_uncertainty_budget_path=Path(
            observed.dry_uncertainty_budget_path
        ),
    )

    receipt = MissionLegMotionConsumptionReceipt(
        mission_leg_motion_permit_path=canonical_permit_path,
        mission_leg_motion_permit_sha256=permit_sha256,
        session_id=session_id,
        run_id=run_id,
        mission_leg_kind=observed_kind,
        mission_leg_index=mission_leg_index,
        target_id=target_id,
    )
    receipt_path = _receipt_path_from_binding(
        master_path=master_path,
        run_id=run_id,
        permit_sha256=permit_sha256,
    )
    _claim_receipt_exclusively(receipt_path, receipt)
    return load_mission_leg_motion_consumption_receipt(receipt_path)


def mission_leg_motion_consumption_receipt_sha256(
    receipt: MissionLegMotionConsumptionReceipt,
) -> str:
    _validate_receipt(receipt)
    return payload_sha256(receipt.to_payload())


def load_mission_leg_motion_consumption_receipt(
    path: Path,
) -> MissionLegMotionConsumptionReceipt:
    """Integrity-load a receipt and revalidate its permit/master bindings."""

    canonical_receipt_path = _canonical_normal_file_path(
        path, "mission leg motion consumption receipt"
    )
    try:
        payload = load_content_hashed_json(
            Path(canonical_receipt_path),
            hash_field=MISSION_LEG_MOTION_CONSUMPTION_RECEIPT_HASH_FIELD,
        )
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc
    if frozenset(payload) != _RECEIPT_FIELDS:
        raise ValueError("mission leg motion consumption receipt fields mismatch")
    try:
        receipt = MissionLegMotionConsumptionReceipt(
            schema_version=_integer(payload["schema_version"], "schema_version"),
            mission_leg_motion_permit_path=_string(
                payload["mission_leg_motion_permit_path"],
                "mission_leg_motion_permit_path",
            ),
            mission_leg_motion_permit_sha256=_string(
                payload["mission_leg_motion_permit_sha256"],
                "mission_leg_motion_permit_sha256",
            ),
            session_id=_string(payload["session_id"], "session_id"),
            run_id=_string(payload["run_id"], "run_id"),
            mission_leg_kind=_string(
                payload["mission_leg_kind"], "mission_leg_kind"
            ),
            mission_leg_index=_integer(
                payload["mission_leg_index"], "mission_leg_index"
            ),
            target_id=_string(payload["target_id"], "target_id"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid mission leg motion consumption receipt: {exc}"
        ) from exc

    _, permit, permit_sha256, master_path = _load_bound_permit(
        Path(receipt.mission_leg_motion_permit_path)
    )
    checks = {
        "mission_leg_motion_permit_sha256": (
            permit_sha256,
            receipt.mission_leg_motion_permit_sha256,
        ),
        "session_id": (permit.session_id, receipt.session_id),
        "run_id": (permit.run_id, receipt.run_id),
        "mission_leg_kind": (
            permit.mission_leg_kind,
            receipt.mission_leg_kind,
        ),
        "mission_leg_index": (
            permit.mission_leg_index,
            receipt.mission_leg_index,
        ),
        "target_id": (permit.target_id, receipt.target_id),
    }
    _require_exact_matches(checks)
    expected_path = _receipt_path_from_binding(
        master_path=master_path,
        run_id=permit.run_id,
        permit_sha256=permit_sha256,
    )
    if Path(canonical_receipt_path) != expected_path:
        raise ValueError("mission leg motion consumption receipt path mismatch")
    return receipt


def _load_bound_permit(
    permit_path: Path,
) -> tuple[str, MissionLegMotionPermit, str, Path]:
    canonical_permit_path = _canonical_normal_file_path(
        permit_path, "mission leg motion permit"
    )
    permit = load_mission_leg_motion_permit(Path(canonical_permit_path))
    permit_sha256 = mission_leg_motion_permit_sha256(permit)
    master_path = Path(
        _canonical_normal_file_path(
            Path(permit.master_authorization_path),
            "mission leg motion authorization",
        )
    )
    authorization = load_mission_leg_motion_authorization(master_path)
    if str(master_path) != permit.master_authorization_path:
        raise ValueError(
            "mission leg motion permit master authorization path mismatch"
        )
    if (
        mission_leg_motion_authorization_sha256(authorization)
        != permit.master_authorization_sha256
    ):
        raise ValueError(
            "mission leg motion permit master authorization hash mismatch"
        )
    return canonical_permit_path, permit, permit_sha256, master_path


def _receipt_path_from_binding(
    *, master_path: Path, run_id: str, permit_sha256: str
) -> Path:
    _require_nonempty(run_id, "run_id")
    _require_sha256(permit_sha256, "mission_leg_motion_permit_sha256")
    parent = Path(master_path).parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(
            "mission leg motion receipt parent must be a normal directory"
        )
    run_digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    return parent / (
        "mission_leg_motion_consumption_"
        f"{run_digest}_{permit_sha256}.json"
    )


def _claim_receipt_exclusively(
    path: Path, receipt: MissionLegMotionConsumptionReceipt
) -> None:
    hashed = receipt.to_payload()
    hashed[MISSION_LEG_MOTION_CONSUMPTION_RECEIPT_HASH_FIELD] = (
        mission_leg_motion_consumption_receipt_sha256(receipt)
    )
    try:
        data = (
            json.dumps(
                hashed,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid mission leg motion receipt: {exc}") from exc
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ValueError(f"mission leg motion permit already consumed: {path}") from exc
    except OSError as exc:
        raise ValueError(f"cannot claim mission leg motion receipt: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        # A partial exclusive inode remains a permanent fail-closed claim.
        raise ValueError(f"cannot write mission leg motion receipt: {path}") from exc
    _fsync_directory(path.parent)


def _canonical_normal_file_path(path: Path, label: str) -> str:
    source = Path(path)
    if not source.is_absolute() or source != Path(os.path.normpath(str(source))):
        raise ValueError(f"{label} path must be canonical absolute")
    if source.is_symlink():
        raise ValueError(f"{label} path must not be a symlink: {source}")
    if not source.is_file():
        raise ValueError(f"{label} path must be a normal file: {source}")
    return str(source)


def _validate_receipt(receipt: MissionLegMotionConsumptionReceipt) -> None:
    if (
        receipt.schema_version
        != MISSION_LEG_MOTION_CONSUMPTION_RECEIPT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported mission leg motion receipt schema")
    _require_canonical_path_string(
        receipt.mission_leg_motion_permit_path,
        "mission_leg_motion_permit_path",
    )
    _require_sha256(
        receipt.mission_leg_motion_permit_sha256,
        "mission_leg_motion_permit_sha256",
    )
    for name in ("session_id", "run_id", "target_id"):
        _require_nonempty(getattr(receipt, name), name)
    if receipt.mission_leg_kind not in {
        MissionLegKind.COVERAGE,
        MissionLegKind.CANDIDATE_PREAPPROACH,
        MissionLegKind.OPPOSITE_FACE,
    }:
        raise ValueError("mission_leg_kind must be a routine leg kind")
    if (
        isinstance(receipt.mission_leg_index, bool)
        or not isinstance(receipt.mission_leg_index, int)
        or receipt.mission_leg_index < 0
    ):
        raise ValueError("mission_leg_index must be a non-negative integer")


def _require_exact_matches(
    checks: Mapping[str, tuple[object, object]],
) -> None:
    for name, (sealed, observed) in checks.items():
        if sealed != observed or (
            isinstance(sealed, (bool, int)) and type(sealed) is not type(observed)
        ):
            raise ValueError(f"mission leg motion receipt {name} mismatch")


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


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


__all__ = [
    "MISSION_LEG_MOTION_CONSUMPTION_RECEIPT_HASH_FIELD",
    "MISSION_LEG_MOTION_CONSUMPTION_RECEIPT_SCHEMA_VERSION",
    "MissionLegMotionConsumptionReceipt",
    "consume_mission_leg_motion_permit",
    "default_mission_leg_motion_consumption_receipt_path",
    "load_mission_leg_motion_consumption_receipt",
    "mission_leg_motion_consumption_receipt_sha256",
]
