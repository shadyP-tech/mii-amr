"""Atomic one-use consumption receipts for runtime motion permits.

The runtime-localization motion permit is an authorization artifact, not a
reusable capability.  This ROS-free module turns one already validated permit
into one durable claim.  The claim path is derived from permit content and the
payload-bound master-authorization directory, so copying the same permit bytes
to another path cannot create a fresh consumption slot.
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
from scripts.aufgabe04.navigation.execution.runtime_motion_authorization import (
    RuntimeLocalizationMotionPermit,
    load_mission_motion_authorization,
    load_runtime_localization_motion_permit,
    mission_motion_authorization_sha256,
    runtime_localization_motion_permit_sha256,
)


RUNTIME_MOTION_CONSUMPTION_RECEIPT_SCHEMA_VERSION = 1
RUNTIME_MOTION_CONSUMPTION_RECEIPT_HASH_FIELD = (
    "runtime_motion_consumption_receipt_sha256"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "runtime_localization_motion_permit_path",
        "runtime_localization_motion_permit_sha256",
        "run_id",
        "session_id",
        "leg_index",
        "target_viewpoint_id",
        "reseal_index",
    }
)


@dataclass(frozen=True)
class RuntimeMotionConsumptionReceipt:
    """The immutable identity of one successfully claimed runtime permit."""

    runtime_localization_motion_permit_path: str
    runtime_localization_motion_permit_sha256: str
    run_id: str
    session_id: str
    leg_index: int
    target_viewpoint_id: str
    reseal_index: int
    schema_version: int = RUNTIME_MOTION_CONSUMPTION_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_receipt(self)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "runtime_localization_motion_permit_path": (
                self.runtime_localization_motion_permit_path
            ),
            "runtime_localization_motion_permit_sha256": (
                self.runtime_localization_motion_permit_sha256
            ),
            "run_id": self.run_id,
            "session_id": self.session_id,
            "leg_index": self.leg_index,
            "target_viewpoint_id": self.target_viewpoint_id,
            "reseal_index": self.reseal_index,
        }

    def to_evidence(self) -> dict[str, object]:
        return self.to_payload()


def default_runtime_motion_consumption_receipt_path(
    permit_path: Path,
) -> Path:
    """Derive the single claim path shared by byte-identical permit copies."""

    _, permit, permit_sha256, master_path = _load_bound_permit(permit_path)
    return _receipt_path_from_binding(
        master_path=master_path,
        run_id=permit.run_id,
        permit_sha256=permit_sha256,
    )


def derive_runtime_motion_consumption_receipt_path(permit_path: Path) -> Path:
    """Alias spelling for callers that describe the path operation explicitly."""

    return default_runtime_motion_consumption_receipt_path(permit_path)


def consume_runtime_motion_permit(
    *,
    permit_path: Path,
    permit: RuntimeLocalizationMotionPermit,
    session_id: str,
    run_id: str,
    leg_index: int,
    target_viewpoint_id: str,
    reseal_index: int,
) -> RuntimeMotionConsumptionReceipt:
    """Atomically claim one previously validated permit exactly once.

    ``permit`` is the object returned by the caller's full execution validator.
    The file is integrity-loaded again and must still be byte-semantically
    identical before the claim is created.  An existing claim path is always a
    replay rejection, including when its bytes are identical or malformed.
    """

    if not isinstance(permit, RuntimeLocalizationMotionPermit):
        raise ValueError("permit must be a RuntimeLocalizationMotionPermit")
    canonical_permit_path, observed, permit_sha256, master_path = (
        _load_bound_permit(permit_path)
    )
    validated_sha256 = runtime_localization_motion_permit_sha256(permit)
    if (
        permit_sha256 != validated_sha256
        or observed.to_payload() != permit.to_payload()
    ):
        raise ValueError("runtime localization motion permit changed after validation")

    authorization = _load_bound_master(observed, master_path)
    checks = {
        "run_id": (observed.run_id, run_id),
        "session_id": (authorization.session_id, session_id),
        "leg_index": (observed.leg_index, leg_index),
        "target_viewpoint_id": (
            observed.target_viewpoint_id,
            target_viewpoint_id,
        ),
        "reseal_index": (observed.reseal_index, reseal_index),
    }
    _require_exact_matches(checks)

    receipt = RuntimeMotionConsumptionReceipt(
        runtime_localization_motion_permit_path=canonical_permit_path,
        runtime_localization_motion_permit_sha256=permit_sha256,
        run_id=run_id,
        session_id=session_id,
        leg_index=leg_index,
        target_viewpoint_id=target_viewpoint_id,
        reseal_index=reseal_index,
    )
    receipt_path = _receipt_path_from_binding(
        master_path=master_path,
        run_id=run_id,
        permit_sha256=permit_sha256,
    )
    _claim_receipt_exclusively(receipt_path, receipt)

    # Re-load both the receipt and its permit reference after publication.  If
    # anything changed during the claim, the permanent receipt remains in
    # place and this call fails closed instead of authorizing motion.
    return load_runtime_motion_consumption_receipt(receipt_path)


def runtime_motion_consumption_receipt_sha256(
    receipt: RuntimeMotionConsumptionReceipt,
) -> str:
    _validate_receipt(receipt)
    return payload_sha256(receipt.to_payload())


def load_runtime_motion_consumption_receipt(
    path: Path,
) -> RuntimeMotionConsumptionReceipt:
    """Integrity-load a receipt and verify every permit/identity binding."""

    canonical_receipt_path = _canonical_normal_file_path(
        path, "runtime motion consumption receipt"
    )
    try:
        payload = load_content_hashed_json(
            Path(canonical_receipt_path),
            hash_field=RUNTIME_MOTION_CONSUMPTION_RECEIPT_HASH_FIELD,
        )
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc
    if frozenset(payload) != _RECEIPT_FIELDS:
        raise ValueError("runtime motion consumption receipt fields mismatch")
    try:
        receipt = RuntimeMotionConsumptionReceipt(
            schema_version=_integer(payload["schema_version"], "schema_version"),
            runtime_localization_motion_permit_path=_string(
                payload["runtime_localization_motion_permit_path"],
                "runtime_localization_motion_permit_path",
            ),
            runtime_localization_motion_permit_sha256=_string(
                payload["runtime_localization_motion_permit_sha256"],
                "runtime_localization_motion_permit_sha256",
            ),
            run_id=_string(payload["run_id"], "run_id"),
            session_id=_string(payload["session_id"], "session_id"),
            leg_index=_integer(payload["leg_index"], "leg_index"),
            target_viewpoint_id=_string(
                payload["target_viewpoint_id"], "target_viewpoint_id"
            ),
            reseal_index=_integer(payload["reseal_index"], "reseal_index"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid runtime motion consumption receipt: {exc}") from exc

    _, permit, permit_sha256, master_path = _load_bound_permit(
        Path(receipt.runtime_localization_motion_permit_path)
    )
    authorization = _load_bound_master(permit, master_path)
    checks = {
        "runtime_localization_motion_permit_sha256": (
            permit_sha256,
            receipt.runtime_localization_motion_permit_sha256,
        ),
        "run_id": (permit.run_id, receipt.run_id),
        "session_id": (authorization.session_id, receipt.session_id),
        "leg_index": (permit.leg_index, receipt.leg_index),
        "target_viewpoint_id": (
            permit.target_viewpoint_id,
            receipt.target_viewpoint_id,
        ),
        "reseal_index": (permit.reseal_index, receipt.reseal_index),
    }
    _require_exact_matches(checks)
    expected_path = _receipt_path_from_binding(
        master_path=master_path,
        run_id=permit.run_id,
        permit_sha256=permit_sha256,
    )
    if Path(canonical_receipt_path) != expected_path:
        raise ValueError("runtime motion consumption receipt path mismatch")
    return receipt


def _load_bound_permit(
    permit_path: Path,
) -> tuple[str, RuntimeLocalizationMotionPermit, str, Path]:
    canonical_permit_path = _canonical_normal_file_path(
        permit_path, "runtime localization motion permit"
    )
    permit = load_runtime_localization_motion_permit(Path(canonical_permit_path))
    permit_sha256 = runtime_localization_motion_permit_sha256(permit)
    master_path = Path(
        _canonical_normal_file_path(
            Path(permit.master_authorization_path), "mission motion authorization"
        )
    )
    _load_bound_master(permit, master_path)
    return canonical_permit_path, permit, permit_sha256, master_path


def _load_bound_master(
    permit: RuntimeLocalizationMotionPermit,
    master_path: Path,
):
    canonical_master_path = _canonical_normal_file_path(
        master_path, "mission motion authorization"
    )
    if canonical_master_path != permit.master_authorization_path:
        raise ValueError(
            "runtime localization motion permit master authorization path mismatch"
        )
    authorization = load_mission_motion_authorization(Path(canonical_master_path))
    if (
        mission_motion_authorization_sha256(authorization)
        != permit.master_authorization_sha256
    ):
        raise ValueError(
            "runtime localization motion permit master authorization hash mismatch"
        )
    return authorization


def _receipt_path_from_binding(
    *, master_path: Path, run_id: str, permit_sha256: str
) -> Path:
    _require_nonempty(run_id, "run_id")
    _require_sha256(permit_sha256, "runtime_localization_motion_permit_sha256")
    parent = Path(master_path).parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(
            "runtime motion consumption receipt parent must be a normal directory"
        )
    run_id_sha256 = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    return parent / (
        "runtime_motion_consumption_"
        f"{run_id_sha256}_{permit_sha256}.json"
    )


def _claim_receipt_exclusively(
    path: Path, receipt: RuntimeMotionConsumptionReceipt
) -> None:
    payload = receipt.to_payload()
    hashed_payload = dict(payload)
    hashed_payload[RUNTIME_MOTION_CONSUMPTION_RECEIPT_HASH_FIELD] = (
        runtime_motion_consumption_receipt_sha256(receipt)
    )
    try:
        data = (
            json.dumps(
                hashed_payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid runtime motion consumption receipt: {exc}") from exc

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ValueError(f"runtime motion permit already consumed: {path}") from exc
    except OSError as exc:
        raise ValueError(
            f"cannot claim runtime motion consumption receipt: {path}"
        ) from exc

    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        # The exclusive inode deliberately remains as a permanent fail-closed
        # claim, even if publication was interrupted and its bytes are invalid.
        raise ValueError(
            f"cannot write runtime motion consumption receipt: {path}"
        ) from exc
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


def _validate_receipt(receipt: RuntimeMotionConsumptionReceipt) -> None:
    if (
        receipt.schema_version
        != RUNTIME_MOTION_CONSUMPTION_RECEIPT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported runtime motion consumption receipt schema")
    _require_canonical_path_string(
        receipt.runtime_localization_motion_permit_path,
        "runtime_localization_motion_permit_path",
    )
    _require_sha256(
        receipt.runtime_localization_motion_permit_sha256,
        "runtime_localization_motion_permit_sha256",
    )
    for name in ("run_id", "session_id", "target_viewpoint_id"):
        _require_nonempty(getattr(receipt, name), name)
    _nonnegative_integer(receipt.leg_index, "leg_index")
    _positive_integer(receipt.reseal_index, "reseal_index")


def _require_exact_matches(
    checks: Mapping[str, tuple[object, object]],
) -> None:
    for name, (sealed, observed) in checks.items():
        if sealed != observed or (
            isinstance(sealed, (bool, int)) and type(sealed) is not type(observed)
        ):
            raise ValueError(f"runtime motion consumption receipt {name} mismatch")


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


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # Some test and in-memory filesystems do not support directory fsync.
        pass


__all__ = [
    "RUNTIME_MOTION_CONSUMPTION_RECEIPT_HASH_FIELD",
    "RUNTIME_MOTION_CONSUMPTION_RECEIPT_SCHEMA_VERSION",
    "RuntimeMotionConsumptionReceipt",
    "consume_runtime_motion_permit",
    "default_runtime_motion_consumption_receipt_path",
    "derive_runtime_motion_consumption_receipt_path",
    "load_runtime_motion_consumption_receipt",
    "runtime_motion_consumption_receipt_sha256",
]
