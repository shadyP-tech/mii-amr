"""Atomic one-use receipts for startup-reseal motion permits.

The exclusive claim is created only after the permit and every referenced
artifact have been reloaded and rehashed.  It is intended to be called at the
last fail-closed boundary immediately before the child emits ``motion_started``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Mapping

from scripts.aufgabe04.artifacts.content_store import (
    ContentStoreError,
    load_content_hashed_json,
    payload_sha256,
)
from scripts.aufgabe04.navigation.startup_reseal_motion_authorization import (
    StartupResealMotionAuthorization,
    StartupResealMotionPermit,
    load_startup_reseal_motion_authorization,
    load_startup_reseal_motion_permit,
    startup_reseal_motion_authorization_sha256,
    startup_reseal_motion_permit_sha256,
    validate_startup_reseal_motion_permit_for_execution,
)


STARTUP_RESEAL_MOTION_CONSUMPTION_RECEIPT_SCHEMA_VERSION = 1
STARTUP_RESEAL_MOTION_CONSUMPTION_RECEIPT_HASH_FIELD = (
    "startup_reseal_motion_consumption_receipt_sha256"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "startup_reseal_motion_permit_path",
        "startup_reseal_motion_permit_sha256",
        "session_id",
        "run_id",
        "leg_index",
        "target_viewpoint_id",
        "reseal_index",
    }
)


@dataclass(frozen=True)
class StartupResealMotionConsumptionReceipt:
    """Immutable identity of one successfully claimed replacement run."""

    startup_reseal_motion_permit_path: str
    startup_reseal_motion_permit_sha256: str
    session_id: str
    run_id: str
    leg_index: int
    target_viewpoint_id: str
    reseal_index: int
    schema_version: int = STARTUP_RESEAL_MOTION_CONSUMPTION_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_receipt(self)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "startup_reseal_motion_permit_path": (
                self.startup_reseal_motion_permit_path
            ),
            "startup_reseal_motion_permit_sha256": (
                self.startup_reseal_motion_permit_sha256
            ),
            "session_id": self.session_id,
            "run_id": self.run_id,
            "leg_index": self.leg_index,
            "target_viewpoint_id": self.target_viewpoint_id,
            "reseal_index": self.reseal_index,
        }

    def to_evidence(self) -> dict[str, object]:
        return self.to_payload()


def default_startup_reseal_motion_consumption_receipt_path(
    permit_path: Path,
) -> Path:
    """Derive the one claim path shared by byte-identical permit copies."""

    _, permit, permit_sha256, master_path, authorization = _load_bound_permit(
        permit_path
    )
    _revalidate_execution_artifacts(
        permit_path=Path(_canonical_normal_file_path(permit_path, "permit")),
        permit=permit,
        master_path=master_path,
        authorization=authorization,
    )
    return _receipt_path_from_binding(
        master_path=master_path,
        run_id=permit.run_id,
        permit_sha256=permit_sha256,
    )


def derive_startup_reseal_motion_consumption_receipt_path(
    permit_path: Path,
) -> Path:
    return default_startup_reseal_motion_consumption_receipt_path(permit_path)


def consume_startup_reseal_motion_permit(
    *,
    permit_path: Path,
    permit: StartupResealMotionPermit,
    session_id: str,
    run_id: str,
    leg_index: int,
    target_viewpoint_id: str,
    reseal_index: int,
) -> StartupResealMotionConsumptionReceipt:
    """Revalidate and atomically claim one startup-reseal permit exactly once."""

    if not isinstance(permit, StartupResealMotionPermit):
        raise ValueError("permit must be a StartupResealMotionPermit")
    (
        canonical_permit_path,
        observed,
        permit_sha256,
        master_path,
        authorization,
    ) = _load_bound_permit(permit_path)
    if (
        permit_sha256 != startup_reseal_motion_permit_sha256(permit)
        or observed.to_payload() != permit.to_payload()
    ):
        raise ValueError("startup reseal motion permit changed after validation")
    _require_exact_matches(
        {
            "session_id": (authorization.session_id, session_id),
            "run_id": (observed.run_id, run_id),
            "leg_index": (observed.leg_index, leg_index),
            "target_viewpoint_id": (
                observed.target_viewpoint_id,
                target_viewpoint_id,
            ),
            "reseal_index": (observed.reseal_index, reseal_index),
        }
    )
    # Close the validation-to-consumption window for every reference.  The
    # exclusive receipt is the very next durable state change.
    _revalidate_execution_artifacts(
        permit_path=Path(canonical_permit_path),
        permit=observed,
        master_path=master_path,
        authorization=authorization,
    )

    receipt = StartupResealMotionConsumptionReceipt(
        startup_reseal_motion_permit_path=canonical_permit_path,
        startup_reseal_motion_permit_sha256=permit_sha256,
        session_id=session_id,
        run_id=run_id,
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
    return load_startup_reseal_motion_consumption_receipt(receipt_path)


def startup_reseal_motion_consumption_receipt_sha256(
    receipt: StartupResealMotionConsumptionReceipt,
) -> str:
    _validate_receipt(receipt)
    return payload_sha256(receipt.to_payload())


def load_startup_reseal_motion_consumption_receipt(
    path: Path,
) -> StartupResealMotionConsumptionReceipt:
    """Integrity-load a receipt and revalidate permit, master, and artifacts."""

    canonical_receipt_path = _canonical_normal_file_path(
        path, "startup reseal motion consumption receipt"
    )
    try:
        payload = load_content_hashed_json(
            Path(canonical_receipt_path),
            hash_field=STARTUP_RESEAL_MOTION_CONSUMPTION_RECEIPT_HASH_FIELD,
        )
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc
    if frozenset(payload) != _RECEIPT_FIELDS:
        raise ValueError("startup reseal motion consumption receipt fields mismatch")
    try:
        receipt = StartupResealMotionConsumptionReceipt(
            schema_version=_integer(payload["schema_version"], "schema_version"),
            startup_reseal_motion_permit_path=_string(
                payload["startup_reseal_motion_permit_path"],
                "startup_reseal_motion_permit_path",
            ),
            startup_reseal_motion_permit_sha256=_string(
                payload["startup_reseal_motion_permit_sha256"],
                "startup_reseal_motion_permit_sha256",
            ),
            session_id=_string(payload["session_id"], "session_id"),
            run_id=_string(payload["run_id"], "run_id"),
            leg_index=_integer(payload["leg_index"], "leg_index"),
            target_viewpoint_id=_string(
                payload["target_viewpoint_id"], "target_viewpoint_id"
            ),
            reseal_index=_integer(payload["reseal_index"], "reseal_index"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid startup reseal motion consumption receipt: {exc}"
        ) from exc

    (
        _,
        permit,
        permit_sha256,
        master_path,
        authorization,
    ) = _load_bound_permit(Path(receipt.startup_reseal_motion_permit_path))
    _require_exact_matches(
        {
            "startup_reseal_motion_permit_sha256": (
                permit_sha256,
                receipt.startup_reseal_motion_permit_sha256,
            ),
            "session_id": (authorization.session_id, receipt.session_id),
            "run_id": (permit.run_id, receipt.run_id),
            "leg_index": (permit.leg_index, receipt.leg_index),
            "target_viewpoint_id": (
                permit.target_viewpoint_id,
                receipt.target_viewpoint_id,
            ),
            "reseal_index": (permit.reseal_index, receipt.reseal_index),
        }
    )
    _revalidate_execution_artifacts(
        permit_path=Path(receipt.startup_reseal_motion_permit_path),
        permit=permit,
        master_path=master_path,
        authorization=authorization,
    )
    expected_path = _receipt_path_from_binding(
        master_path=master_path,
        run_id=permit.run_id,
        permit_sha256=permit_sha256,
    )
    if Path(canonical_receipt_path) != expected_path:
        raise ValueError("startup reseal motion consumption receipt path mismatch")
    return receipt


def _load_bound_permit(
    permit_path: Path,
) -> tuple[
    str,
    StartupResealMotionPermit,
    str,
    Path,
    StartupResealMotionAuthorization,
]:
    canonical_permit_path = _canonical_normal_file_path(
        permit_path, "startup reseal motion permit"
    )
    permit = load_startup_reseal_motion_permit(Path(canonical_permit_path))
    permit_sha256 = startup_reseal_motion_permit_sha256(permit)
    master_path = Path(
        _canonical_normal_file_path(
            Path(permit.master_authorization_path),
            "startup reseal motion authorization",
        )
    )
    if str(master_path) != permit.master_authorization_path:
        raise ValueError(
            "startup reseal motion permit master authorization path mismatch"
        )
    authorization = load_startup_reseal_motion_authorization(master_path)
    if (
        startup_reseal_motion_authorization_sha256(authorization)
        != permit.master_authorization_sha256
    ):
        raise ValueError(
            "startup reseal motion permit master authorization hash mismatch"
        )
    return (
        canonical_permit_path,
        permit,
        permit_sha256,
        master_path,
        authorization,
    )


def _revalidate_execution_artifacts(
    *,
    permit_path: Path,
    permit: StartupResealMotionPermit,
    master_path: Path,
    authorization: StartupResealMotionAuthorization,
) -> None:
    validate_startup_reseal_motion_permit_for_execution(
        permit_path,
        master_authorization_path=master_path,
        run_id=permit.run_id,
        session_id=authorization.session_id,
        robot_id=authorization.robot_id,
        namespace=authorization.namespace,
        cmd_vel_topic=authorization.cmd_vel_topic,
        semantic_map_id=authorization.semantic_map_id,
        target_viewpoint_id=permit.target_viewpoint_id,
        leg_index=permit.leg_index,
        localization_branch_proof_id=(
            authorization.localization_branch_proof_id
        ),
        route_csv_path=Path(permit.route_csv_path),
        diagnostics_path=Path(permit.diagnostics_path),
        map_route_certificate_path=Path(permit.map_route_certificate_path),
    )


def _receipt_path_from_binding(
    *,
    master_path: Path,
    run_id: str,
    permit_sha256: str,
) -> Path:
    _require_nonempty(run_id, "run_id")
    _require_sha256(permit_sha256, "startup_reseal_motion_permit_sha256")
    parent = Path(master_path).parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(
            "startup reseal motion receipt parent must be a normal directory"
        )
    run_digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    return parent / (
        "startup_reseal_motion_consumption_"
        f"{run_digest}_{permit_sha256}.json"
    )


def _claim_receipt_exclusively(
    path: Path,
    receipt: StartupResealMotionConsumptionReceipt,
) -> None:
    hashed = receipt.to_payload()
    hashed[STARTUP_RESEAL_MOTION_CONSUMPTION_RECEIPT_HASH_FIELD] = (
        startup_reseal_motion_consumption_receipt_sha256(receipt)
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
        raise ValueError(
            f"invalid startup reseal motion consumption receipt: {exc}"
        ) from exc
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ValueError(
            f"startup reseal motion permit already consumed: {path}"
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"cannot claim startup reseal motion consumption receipt: {path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        # A partial inode remains as a permanent fail-closed claim.
        raise ValueError(
            f"cannot write startup reseal motion consumption receipt: {path}"
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


def _validate_receipt(receipt: StartupResealMotionConsumptionReceipt) -> None:
    if (
        receipt.schema_version
        != STARTUP_RESEAL_MOTION_CONSUMPTION_RECEIPT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported startup reseal motion receipt schema")
    _require_canonical_path_string(
        receipt.startup_reseal_motion_permit_path,
        "startup_reseal_motion_permit_path",
    )
    _require_sha256(
        receipt.startup_reseal_motion_permit_sha256,
        "startup_reseal_motion_permit_sha256",
    )
    for name in ("session_id", "run_id", "target_viewpoint_id"):
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
            raise ValueError(
                f"startup reseal motion consumption receipt {name} mismatch"
            )


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
        pass


__all__ = [
    "STARTUP_RESEAL_MOTION_CONSUMPTION_RECEIPT_HASH_FIELD",
    "STARTUP_RESEAL_MOTION_CONSUMPTION_RECEIPT_SCHEMA_VERSION",
    "StartupResealMotionConsumptionReceipt",
    "consume_startup_reseal_motion_permit",
    "default_startup_reseal_motion_consumption_receipt_path",
    "derive_startup_reseal_motion_consumption_receipt_path",
    "load_startup_reseal_motion_consumption_receipt",
    "startup_reseal_motion_consumption_receipt_sha256",
]
