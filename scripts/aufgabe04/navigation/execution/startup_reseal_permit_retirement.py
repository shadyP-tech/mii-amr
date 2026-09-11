"""Terminal disposition of an unclaimed permit after typed startup rejection.

Retirement occupies the *same* exclusive inode as consumption. Whichever
operation wins prevents the other, including byte-identical permit copies.
The tombstone has its own schema and never asserts that motion was consumed.
Dry attempts instead record a bound, explicit no-permit-issued disposition.
"""

from __future__ import annotations

import os
from pathlib import Path
from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json,
    payload_sha256,
    write_content_hashed_json,
)
from scripts.aufgabe04.navigation.execution.startup_route_rejection_evidence import (
    validate_odom_startup_rejection_log,
)


DISPOSITION_HASH_FIELD = "startup_reseal_permit_disposition_sha256"


def _load_old_permit(path: Path, kind: str):
    """Use existing public binding loaders and the existing claim namespace."""
    if kind == "mission_leg":
        from scripts.aufgabe04.navigation.execution.mission_leg_motion_consumption import (
            default_mission_leg_motion_consumption_receipt_path,
        )
        from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (
            load_mission_leg_motion_authorization,
            load_mission_leg_motion_permit,
        )
        permit = load_mission_leg_motion_permit(path)
        master = load_mission_leg_motion_authorization(Path(permit.master_authorization_path))
        claim = default_mission_leg_motion_consumption_receipt_path(path)
    elif kind == "startup_reseal":
        from scripts.aufgabe04.navigation.execution.startup_reseal_motion_consumption import (
            default_startup_reseal_motion_consumption_receipt_path,
        )
        from scripts.aufgabe04.navigation.execution.startup_reseal_motion_authorization import (
            load_startup_reseal_motion_authorization,
            load_startup_reseal_motion_permit,
        )
        permit = load_startup_reseal_motion_permit(path)
        master = load_startup_reseal_motion_authorization(Path(permit.master_authorization_path))
        claim = default_startup_reseal_motion_consumption_receipt_path(path)
    elif kind == "runtime_localization":
        from scripts.aufgabe04.navigation.execution.runtime_motion_consumption import (
            default_runtime_motion_consumption_receipt_path,
        )
        from scripts.aufgabe04.navigation.execution.runtime_motion_authorization import (
            load_mission_motion_authorization,
            load_runtime_localization_motion_permit,
        )
        permit = load_runtime_localization_motion_permit(path)
        master = load_mission_motion_authorization(Path(permit.master_authorization_path))
        claim = default_runtime_motion_consumption_receipt_path(path)
    else:
        raise ValueError("unsupported rejected permit kind")
    return permit, master, claim


def _disposition_payload(
    *,
    permit_path: Path | None,
    permit_kind: str | None,
    expected_permit_sha256: str,
    rejected_semantic_log_path: Path,
    session_id: str,
    rejected_run_id: str,
    mission_leg_kind: str,
    mission_leg_index: int,
    target_id: str,
    reseal_index: int,
) -> tuple[dict[str, object], Path | None]:
    from scripts.aufgabe04.navigation.execution.startup_reseal_motion_authorization import file_sha256

    if not session_id or type(reseal_index) is not int or reseal_index <= 0:
        raise ValueError("startup disposition session or reseal index is invalid")
    log = _normal_path(rejected_semantic_log_path)
    log_sha256 = file_sha256(log)
    details = validate_odom_startup_rejection_log(
        log, rejected_run_id=rejected_run_id,
        mission_leg_kind=mission_leg_kind, mission_leg_index=mission_leg_index,
        target_id=target_id,
    )
    if type(details.get("dry_run")) is not bool:
        raise ValueError("startup rejection must explicitly identify dry or execute attempt")
    payload: dict[str, object] = {
        "schema_version": 1,
        "session_id": session_id,
        "rejected_run_id": rejected_run_id,
        "mission_leg_kind": mission_leg_kind,
        "mission_leg_index": mission_leg_index,
        "target_id": target_id,
        "replacement_startup_reseal_index": reseal_index,
        "recovery_source_kind": "odom_startup_route_mismatch",
        "rejected_semantic_log_path": str(log),
        "rejected_semantic_log_sha256": log_sha256,
        "motion_published": False,
        "motion_authorization_consumed": False,
        "follower_started": False,
    }
    claim = None
    if details["dry_run"] is True:
        if permit_path is not None or permit_kind is not None or expected_permit_sha256:
            raise ValueError("dry rejection cannot retire an execution permit")
        if not any(rejected_run_id.startswith(session_id + separator) for separator in ("_", "-")):
            raise ValueError("dry startup rejection run does not belong to the mission session")
        payload.update(disposition="no_permit_issued", permit_path=None, permit_kind=None,
                       permit_file_sha256=None, permit_sha256=None)
    else:
        if permit_path is None or permit_kind is None:
            raise ValueError("execute rejection requires its issued unclaimed permit")
        old_path = _normal_path(permit_path)
        old_sha256 = file_sha256(old_path)
        permit, master, claim = _load_old_permit(old_path, permit_kind)
        if payload_sha256(permit.to_payload()) != expected_permit_sha256:
            raise ValueError("rejected permit issued content hash mismatch")
        if (
            master.session_id != session_id or permit.run_id != rejected_run_id
            or permit.mission_leg_kind.value != mission_leg_kind
            or permit.mission_leg_index != mission_leg_index
            or permit.target_id != target_id
        ):
            raise ValueError("rejected permit mission identity mismatch")
        if permit_kind == "startup_reseal" and permit.reseal_index != reseal_index - 1:
            raise ValueError("rejected startup permit cumulative reseal index mismatch")
        from scripts.aufgabe04.navigation.execution.execution_route_certificate import (
            execution_route_certificate_sha256, load_execution_route_certificate,
        )
        bound_certificate = load_execution_route_certificate(Path(permit.map_route_certificate_path))
        if (
            file_sha256(Path(permit.map_route_certificate_path)) != permit.map_route_certificate_sha256
            or file_sha256(Path(permit.route_csv_path)) != permit.route_csv_sha256
            or bound_certificate.route_sha256 != permit.route_csv_sha256
            or execution_route_certificate_sha256(bound_certificate)
            != details["startup_route_admission"]["source_map_execution_certificate_sha256"]
        ):
            raise ValueError("rejected permit source route certificate mismatch")
        payload.update(disposition="retired_before_motion", permit_path=str(old_path),
                       permit_kind=permit_kind, permit_file_sha256=old_sha256,
                       permit_sha256=expected_permit_sha256)
        if file_sha256(old_path) != old_sha256:
            raise ValueError("rejected permit changed during retirement validation")
    if file_sha256(log) != log_sha256:
        raise ValueError("rejection log changed during retirement validation")
    return payload, claim


def retire_odom_startup_rejected_permit(
    *,
    permit_path: Path | None,
    permit_kind: str | None,
    expected_permit_sha256: str = "",
    rejected_semantic_log_path: Path,
    session_id: str,
    rejected_run_id: str,
    mission_leg_kind: str,
    mission_leg_index: int,
    target_id: str,
    reseal_index: int,
    disposition_path: Path | None = None,
) -> Path:
    """Retire before any replacement can be authorized; existing claims fail closed."""
    payload, claim = _disposition_payload(
        permit_path=permit_path, permit_kind=permit_kind,
        expected_permit_sha256=expected_permit_sha256,
        rejected_semantic_log_path=rejected_semantic_log_path,
        session_id=session_id, rejected_run_id=rejected_run_id,
        mission_leg_kind=mission_leg_kind, mission_leg_index=mission_leg_index,
        target_id=target_id, reseal_index=reseal_index,
    )
    if claim is not None:
        if disposition_path is not None and Path(disposition_path) != claim:
            raise ValueError("retirement must use the existing exclusive permit claim path")
        destination = claim
    else:
        if disposition_path is None:
            raise ValueError("dry rejection requires an explicit disposition artifact path")
        destination = Path(disposition_path).absolute()
    # Atomic hard-link publication cannot replace a concurrent exclusive claim.
    # Identical retirement bytes are an idempotent retry, never a second permit.
    write_content_hashed_json(destination, payload, hash_field=DISPOSITION_HASH_FIELD)
    return destination


def validate_startup_reseal_permit_disposition(
    path: Path,
    *,
    expected_sha256: str,
    replacement_permit,
) -> None:
    """Revalidate terminal disposition whenever the replacement permit is loaded."""
    from scripts.aufgabe04.navigation.execution.startup_reseal_motion_authorization import (
        file_sha256, load_startup_reseal_motion_authorization,
    )

    source = _normal_path(path)
    if file_sha256(source) != expected_sha256:
        raise ValueError("startup rejected permit disposition hash mismatch")
    payload = load_content_hashed_json(source, hash_field=DISPOSITION_HASH_FIELD)
    master = load_startup_reseal_motion_authorization(Path(replacement_permit.master_authorization_path))
    old_path = payload.get("permit_path")
    expected, claim = _disposition_payload(
        permit_path=Path(old_path) if isinstance(old_path, str) else None,
        permit_kind=payload.get("permit_kind"),
        expected_permit_sha256=payload.get("permit_sha256") or "",
        rejected_semantic_log_path=Path(replacement_permit.rejected_semantic_log_path),
        session_id=master.session_id, rejected_run_id=replacement_permit.rejected_run_id,
        mission_leg_kind=replacement_permit.mission_leg_kind.value,
        mission_leg_index=replacement_permit.mission_leg_index,
        target_id=replacement_permit.target_id, reseal_index=replacement_permit.reseal_index,
    )
    if payload != expected or (claim is not None and source != claim):
        raise ValueError("startup rejected permit disposition binding mismatch")
    if old_path is not None:
        _, old_master, _ = _load_old_permit(Path(old_path), payload["permit_kind"])
        for name in (
            "session_id", "robot_id", "namespace", "cmd_vel_topic",
            "semantic_map_id", "localization_branch_proof_id",
        ):
            if getattr(old_master, name) != getattr(master, name):
                raise ValueError(f"startup rejected permit master {name} mismatch")


def _normal_path(path: Path) -> Path:
    source = Path(path)
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        raise ValueError("startup disposition evidence must be an absolute normal file")
    if Path(os.path.normpath(str(source))) != source:
        raise ValueError("startup disposition evidence path must be canonical")
    return source
