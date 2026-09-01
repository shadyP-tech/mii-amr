"""Pure policy and evidence models for startup active localization.

The active-localization phase is deliberately narrower than navigation.  It
permits only an odometry-measured in-place rotation after a startup route was
rejected on uncertainty, then requires a complete stop and a fresh stationary
localization admission before planning is attempted again.  This module has no
ROS or command-publication dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.artifacts.content_store import (
    ContentStoreError,
    load_content_hashed_json,
    write_content_hashed_json,
)


DEFAULT_STARTUP_ACTIVE_LOCALIZATION_MAX_ATTEMPTS = 1
DEFAULT_STARTUP_ACTIVE_LOCALIZATION_ROTATION_RAD = 2.0 * math.pi
DEFAULT_STARTUP_ACTIVE_LOCALIZATION_ANGULAR_SPEED_RADPS = 0.12
DEFAULT_STARTUP_ACTIVE_LOCALIZATION_TIMEOUT_SEC = 70.0
DEFAULT_STARTUP_ACTIVE_LOCALIZATION_YAW_TOLERANCE_RAD = math.radians(4.0)
DEFAULT_STARTUP_ACTIVE_LOCALIZATION_CONTROL_RATE_HZ = 10.0
DEFAULT_STARTUP_ACTIVE_LOCALIZATION_PROGRESS_WINDOW_SEC = 2.0
DEFAULT_STARTUP_ACTIVE_LOCALIZATION_MIN_PROGRESS_RAD = 0.05
DEFAULT_STARTUP_ACTIVE_LOCALIZATION_MAX_TRANSLATION_M = 0.03
DEFAULT_STARTUP_ACTIVE_LOCALIZATION_MIN_CLEARANCE_M = 0.20
DEFAULT_STARTUP_ACTIVE_LOCALIZATION_STOP_COMMAND_COUNT = 10
STARTUP_ACTIVE_LOCALIZATION_SCHEMA_VERSION = 1
STARTUP_ACTIVE_LOCALIZATION_CONFIRMATION = "LOCALIZE"
STARTUP_ACTIVE_LOCALIZATION_AUTHORIZATION_HASH_FIELD = (
    "startup_active_localization_authorization_sha256"
)
STARTUP_ACTIVE_LOCALIZATION_PREFLIGHT_HASH_FIELD = (
    "startup_active_localization_preflight_sha256"
)
STARTUP_ACTIVE_LOCALIZATION_RESULT_HASH_FIELD = (
    "startup_active_localization_result_sha256"
)


@dataclass(frozen=True)
class StartupActiveLocalizationConfig:
    """Bounded phase settings shared by orchestration and the motion child."""

    enabled: bool = False
    max_attempts: int = DEFAULT_STARTUP_ACTIVE_LOCALIZATION_MAX_ATTEMPTS
    rotation_rad: float = DEFAULT_STARTUP_ACTIVE_LOCALIZATION_ROTATION_RAD
    angular_speed_radps: float = (
        DEFAULT_STARTUP_ACTIVE_LOCALIZATION_ANGULAR_SPEED_RADPS
    )
    timeout_sec: float = DEFAULT_STARTUP_ACTIVE_LOCALIZATION_TIMEOUT_SEC
    yaw_tolerance_rad: float = (
        DEFAULT_STARTUP_ACTIVE_LOCALIZATION_YAW_TOLERANCE_RAD
    )
    control_rate_hz: float = DEFAULT_STARTUP_ACTIVE_LOCALIZATION_CONTROL_RATE_HZ
    progress_window_sec: float = (
        DEFAULT_STARTUP_ACTIVE_LOCALIZATION_PROGRESS_WINDOW_SEC
    )
    minimum_progress_rad: float = (
        DEFAULT_STARTUP_ACTIVE_LOCALIZATION_MIN_PROGRESS_RAD
    )
    maximum_translation_m: float = (
        DEFAULT_STARTUP_ACTIVE_LOCALIZATION_MAX_TRANSLATION_M
    )
    minimum_clearance_m: float = (
        DEFAULT_STARTUP_ACTIVE_LOCALIZATION_MIN_CLEARANCE_M
    )
    stop_command_count: int = (
        DEFAULT_STARTUP_ACTIVE_LOCALIZATION_STOP_COMMAND_COUNT
    )

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if (
            type(self.max_attempts) is not int
            or isinstance(self.max_attempts, bool)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        if (
            not math.isfinite(self.rotation_rad)
            or self.rotation_rad <= 0.0
            or self.rotation_rad > 2.0 * math.pi
        ):
            raise ValueError("rotation_rad must be finite and in (0, 2*pi]")
        if (
            not math.isfinite(self.angular_speed_radps)
            or self.angular_speed_radps <= 0.0
        ):
            raise ValueError("angular_speed_radps must be finite and positive")
        if not math.isfinite(self.timeout_sec) or self.timeout_sec <= 0.0:
            raise ValueError("timeout_sec must be finite and positive")
        minimum_duration = self.rotation_rad / self.angular_speed_radps
        if self.timeout_sec + 1.0e-9 < minimum_duration:
            raise ValueError(
                "timeout_sec must cover rotation_rad at angular_speed_radps"
            )
        if (
            not math.isfinite(self.yaw_tolerance_rad)
            or self.yaw_tolerance_rad <= 0.0
            or self.yaw_tolerance_rad >= self.rotation_rad
        ):
            raise ValueError(
                "yaw_tolerance_rad must be finite, positive, and below rotation_rad"
            )
        for name in (
            "control_rate_hz",
            "progress_window_sec",
            "minimum_progress_rad",
            "maximum_translation_m",
            "minimum_clearance_m",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            type(self.stop_command_count) is not int
            or isinstance(self.stop_command_count, bool)
            or self.stop_command_count < 1
        ):
            raise ValueError("stop_command_count must be a positive integer")

    @property
    def target_progress_rad(self) -> float:
        return self.rotation_rad - self.yaw_tolerance_rad

    def direction_for_attempt(self, attempt_index: int) -> int:
        if (
            type(attempt_index) is not int
            or isinstance(attempt_index, bool)
            or attempt_index < 0
            or attempt_index >= self.max_attempts
        ):
            raise ValueError(
                "attempt_index is outside the active-localization budget"
            )
        return 1 if attempt_index % 2 == 0 else -1

    def to_evidence_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "max_attempts": self.max_attempts,
            "rotation_rad": self.rotation_rad,
            "angular_speed_radps": self.angular_speed_radps,
            "timeout_sec": self.timeout_sec,
            "yaw_tolerance_rad": self.yaw_tolerance_rad,
            "target_progress_rad": self.target_progress_rad,
            "control_rate_hz": self.control_rate_hz,
            "progress_window_sec": self.progress_window_sec,
            "minimum_progress_rad": self.minimum_progress_rad,
            "maximum_translation_m": self.maximum_translation_m,
            "minimum_clearance_m": self.minimum_clearance_m,
            "stop_command_count": self.stop_command_count,
            "translation_authorized": False,
        }


@dataclass(frozen=True)
class RotationProgress:
    """Unwrapped odometry progress in and against the authorized direction."""

    previous_yaw_rad: float
    accumulated_progress_rad: float = 0.0
    accumulated_reverse_rad: float = 0.0


def wrapped_yaw_delta_rad(previous_yaw_rad: float, current_yaw_rad: float) -> float:
    """Return the signed shortest odometry-yaw delta across the +/-pi seam."""

    if not all(
        math.isfinite(value)
        for value in (previous_yaw_rad, current_yaw_rad)
    ):
        raise ValueError("yaw samples must be finite")
    return math.atan2(
        math.sin(current_yaw_rad - previous_yaw_rad),
        math.cos(current_yaw_rad - previous_yaw_rad),
    )


def advance_rotation_progress(
    progress: RotationProgress,
    *,
    current_yaw_rad: float,
    direction: int,
) -> RotationProgress:
    """Accumulate only motion in the authorized direction as useful progress."""

    if not isinstance(progress, RotationProgress):
        raise ValueError("progress must be a RotationProgress")
    if direction not in {-1, 1}:
        raise ValueError("direction must be -1 or 1")
    delta = wrapped_yaw_delta_rad(progress.previous_yaw_rad, current_yaw_rad)
    directed_delta = direction * delta
    return RotationProgress(
        previous_yaw_rad=current_yaw_rad,
        accumulated_progress_rad=(
            progress.accumulated_progress_rad + max(0.0, directed_delta)
        ),
        accumulated_reverse_rad=(
            progress.accumulated_reverse_rad + max(0.0, -directed_delta)
        ),
    )


def translation_from_start_m(
    start_xy_m: tuple[float, float],
    current_xy_m: tuple[float, float],
) -> float:
    values = (*start_xy_m, *current_xy_m)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("odometry positions must be finite")
    return math.hypot(
        current_xy_m[0] - start_xy_m[0],
        current_xy_m[1] - start_xy_m[1],
    )


@dataclass(frozen=True)
class StartupActiveLocalizationMotionResult:
    """Terminal result returned by the sole Aufgabe 04 motion edge."""

    status: str
    stop_reason: str
    duration_sec: float
    requested_rotation_rad: float
    accumulated_progress_rad: float
    accumulated_reverse_rad: float
    maximum_translation_m: float
    motion_published: bool
    zero_command_count: int
    stop_details: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.status not in {"completed", "stopped"}:
            raise ValueError("status must be completed or stopped")
        numeric = (
            self.duration_sec,
            self.requested_rotation_rad,
            self.accumulated_progress_rad,
            self.accumulated_reverse_rad,
            self.maximum_translation_m,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in numeric):
            raise ValueError(
                "motion result numeric values must be finite and non-negative"
            )
        if type(self.motion_published) is not bool:
            raise ValueError("motion_published must be boolean")
        if (
            type(self.zero_command_count) is not int
            or isinstance(self.zero_command_count, bool)
            or self.zero_command_count < 1
        ):
            raise ValueError("zero_command_count must be a positive integer")

    @property
    def completed(self) -> bool:
        return self.status == "completed" and not self.stop_reason

    def to_evidence_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "stop_reason": self.stop_reason,
            "duration_sec": self.duration_sec,
            "requested_rotation_rad": self.requested_rotation_rad,
            "accumulated_progress_rad": self.accumulated_progress_rad,
            "accumulated_reverse_rad": self.accumulated_reverse_rad,
            "maximum_translation_m": self.maximum_translation_m,
            "motion_published": self.motion_published,
            "zero_command_count": self.zero_command_count,
            "stop_details": (
                None if self.stop_details is None else dict(self.stop_details)
            ),
            "translation_commanded": False,
        }


def startup_active_localization_signed_turn(
    config: StartupActiveLocalizationConfig,
    *,
    attempt_index: int,
) -> float:
    return float(config.direction_for_attempt(attempt_index)) * config.rotation_rad


def startup_active_localization_attempt_dir(
    session_root: Path,
    *,
    attempt_index: int,
) -> Path:
    if (
        type(attempt_index) is not int
        or isinstance(attempt_index, bool)
        or attempt_index < 0
    ):
        raise ValueError("attempt_index must be a non-negative integer")
    return Path(session_root) / "startup_active_localization" / (
        f"attempt_{attempt_index:03d}"
    )


def startup_active_localization_result_payload(
    *,
    run_id: str,
    attempt_index: int,
    result: StartupActiveLocalizationMotionResult,
    config: StartupActiveLocalizationConfig,
    runtime_config: Mapping[str, object],
    source_route_selection_json: Path,
    source_route_selection_sha256: str,
    preflight_json: Path,
    preflight_sha256: str,
    controller_trace_jsonl: Path,
) -> dict[str, object]:
    if not isinstance(result, StartupActiveLocalizationMotionResult):
        raise ValueError("result must be a StartupActiveLocalizationMotionResult")
    if not isinstance(config, StartupActiveLocalizationConfig):
        raise ValueError("config must be a StartupActiveLocalizationConfig")
    return {
        "schema_version": STARTUP_ACTIVE_LOCALIZATION_SCHEMA_VERSION,
        "phase": "startup_active_localization",
        "run_id": str(run_id),
        "attempt_index": attempt_index,
        "config": config.to_evidence_dict(),
        "runtime_config": dict(runtime_config),
        "source_route_selection_json": str(source_route_selection_json),
        "source_route_selection_sha256": source_route_selection_sha256,
        "preflight_json": str(preflight_json),
        "preflight_sha256": preflight_sha256,
        "controller_trace_jsonl": str(controller_trace_jsonl),
        "operator_confirmation": STARTUP_ACTIVE_LOCALIZATION_CONFIRMATION,
        "route_authorized": False,
        "mission_run_authorized": False,
        "requires_fresh_stationary_localization": True,
        "requires_separate_mission_run": True,
        **result.to_evidence_dict(),
    }


def write_startup_active_localization_result(
    path: Path,
    payload: Mapping[str, object],
) -> str:
    try:
        return write_content_hashed_json(
            Path(path),
            payload,
            hash_field=STARTUP_ACTIVE_LOCALIZATION_RESULT_HASH_FIELD,
        )
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc


def load_startup_active_localization_result(path: Path) -> dict[str, object]:
    try:
        payload = load_content_hashed_json(
            Path(path),
            hash_field=STARTUP_ACTIVE_LOCALIZATION_RESULT_HASH_FIELD,
        )
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc
    if payload.get("schema_version") != (
        STARTUP_ACTIVE_LOCALIZATION_SCHEMA_VERSION
    ):
        raise ValueError("unsupported startup active-localization result schema")
    if payload.get("phase") != "startup_active_localization":
        raise ValueError("startup active-localization result phase mismatch")
    if payload.get("status") not in {"completed", "stopped"}:
        raise ValueError("startup active-localization result status is invalid")
    if type(payload.get("motion_published")) is not bool:
        raise ValueError("startup active-localization motion flag is invalid")
    return payload


def load_startup_active_localization_authorization(
    path: Path,
) -> dict[str, object]:
    try:
        payload = load_content_hashed_json(
            Path(path),
            hash_field=STARTUP_ACTIVE_LOCALIZATION_AUTHORIZATION_HASH_FIELD,
        )
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc
    if payload.get("schema_version") != STARTUP_ACTIVE_LOCALIZATION_SCHEMA_VERSION:
        raise ValueError("unsupported startup active-localization authorization")
    if payload.get("phase") != "startup_active_localization":
        raise ValueError(
            "startup active-localization authorization phase mismatch"
        )
    if payload.get("operator_confirmation") != (
        STARTUP_ACTIVE_LOCALIZATION_CONFIRMATION
    ):
        raise ValueError("startup active-localization confirmation is invalid")
    if payload.get("route_authorized") is not False or payload.get(
        "mission_run_authorized"
    ) is not False:
        raise ValueError("startup active-localization authority is overbroad")
    return payload


def stored_content_hash(path: Path, *, hash_field: str) -> str:
    """Return a verified content-store hash without accepting duplicate keys."""

    try:
        load_content_hashed_json(Path(path), hash_field=hash_field)
        decoded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (ContentStoreError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"content-hashed evidence is invalid: {exc}") from exc
    value = decoded.get(hash_field)
    if not isinstance(value, str):
        raise ValueError(f"content-hashed evidence is missing {hash_field}")
    return value


__all__ = [
    "DEFAULT_STARTUP_ACTIVE_LOCALIZATION_ANGULAR_SPEED_RADPS",
    "DEFAULT_STARTUP_ACTIVE_LOCALIZATION_CONTROL_RATE_HZ",
    "DEFAULT_STARTUP_ACTIVE_LOCALIZATION_MAX_ATTEMPTS",
    "DEFAULT_STARTUP_ACTIVE_LOCALIZATION_MAX_TRANSLATION_M",
    "DEFAULT_STARTUP_ACTIVE_LOCALIZATION_MIN_CLEARANCE_M",
    "DEFAULT_STARTUP_ACTIVE_LOCALIZATION_MIN_PROGRESS_RAD",
    "DEFAULT_STARTUP_ACTIVE_LOCALIZATION_PROGRESS_WINDOW_SEC",
    "DEFAULT_STARTUP_ACTIVE_LOCALIZATION_ROTATION_RAD",
    "DEFAULT_STARTUP_ACTIVE_LOCALIZATION_STOP_COMMAND_COUNT",
    "DEFAULT_STARTUP_ACTIVE_LOCALIZATION_TIMEOUT_SEC",
    "DEFAULT_STARTUP_ACTIVE_LOCALIZATION_YAW_TOLERANCE_RAD",
    "RotationProgress",
    "STARTUP_ACTIVE_LOCALIZATION_AUTHORIZATION_HASH_FIELD",
    "STARTUP_ACTIVE_LOCALIZATION_CONFIRMATION",
    "STARTUP_ACTIVE_LOCALIZATION_PREFLIGHT_HASH_FIELD",
    "STARTUP_ACTIVE_LOCALIZATION_SCHEMA_VERSION",
    "STARTUP_ACTIVE_LOCALIZATION_RESULT_HASH_FIELD",
    "StartupActiveLocalizationConfig",
    "StartupActiveLocalizationMotionResult",
    "advance_rotation_progress",
    "load_startup_active_localization_authorization",
    "load_startup_active_localization_result",
    "startup_active_localization_attempt_dir",
    "startup_active_localization_result_payload",
    "startup_active_localization_signed_turn",
    "stored_content_hash",
    "translation_from_start_m",
    "wrapped_yaw_delta_rad",
    "write_startup_active_localization_result",
]
