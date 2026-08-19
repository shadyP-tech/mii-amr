"""Pure admission policy for a localization stop before any robot motion.

The waypoint follower owns the motion-side fail-closed action.  This module
only classifies the complete terminal evidence that the follower persisted
after its own bounded initial runtime-input wait.  It deliberately keeps this
case separate from :mod:`runtime_localization_reseal`, whose contract requires
that motion already occurred.

An eligible result is evidence for a bounded orchestration recovery attempt;
it is never motion authority.  Both admitted trigger classes require fresh
stationary localization and a newly sealed route certificate because the
one-use child permit was consumed by the failed follower invocation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping


PRESTART_LOCALIZATION_RESEAL_SCHEMA_VERSION = 1

LOCALIZATION_RESEAL_FAULT_CODE = "localization_reseal_required"
GLOBAL_CONSISTENCY_MONITOR_SOURCE = "global_consistency_monitor"
GLOBAL_CONSISTENCY_STOP_REASON = (
    "global localization consistency requires zero and reseal"
)
FORCE_ZERO_RESEAL_ACTION = "FORCE_ZERO_RESEAL"
RESEAL_REQUIRED_MONITOR_REASON = "reseal_required"
FORCE_ZERO_RESEAL_DECISION = "force_zero_reseal"

BEFORE_MOTION_EXECUTION_PHASE = "before_motion"
INITIAL_RUNTIME_INPUT_WAIT_PHASE = "initial_runtime_input_wait"

TF_WARMUP_RETRY = "tf_warmup_retry"
FRESH_LOCALIZATION_RESEAL = "fresh_localization_reseal"

_MISSING_CONTINUITY_REASON = "map_from_odom_missing"
_DRIFT_CONTINUITY_REASONS = frozenset(
    {
        "map_from_odom_translation_drift",
        "map_from_odom_yaw_drift",
        "map_from_odom_translation_and_yaw_drift",
    }
)
_EXACT_TF_WARMUP_WARNINGS = frozenset(
    {
        "stale_map_from_odom",
        "future_map_from_odom",
    }
)
_LOOKUP_FAILURE_WARNING_PREFIX = "map_from_odom_lookup_failed:"
_CONTINUITY_THRESHOLD_SEMANTICS = (
    "accept_if_observed_less_than_or_equal_to_limit"
)
_TRANSFORM_FIELDS = frozenset({"x_m", "y_m", "yaw_rad"})


@dataclass(frozen=True)
class PrestartLocalizationResealDecision:
    """Fail-closed classification of one before-motion follower outcome."""

    eligible: bool
    reason: str
    recovery_action: str
    execution_phase: str
    motion_published: bool | None
    continuity_reason: str
    monitor_warning: str
    requires_fresh_localization: bool
    requires_new_route_certificate: bool
    automatic_motion_authorized: bool = field(default=False, init=False)

    def to_evidence(self) -> dict[str, Any]:
        return {
            "schema_version": PRESTART_LOCALIZATION_RESEAL_SCHEMA_VERSION,
            "eligible": self.eligible,
            "reason": self.reason,
            "recovery_action": self.recovery_action,
            "execution_phase": self.execution_phase,
            "motion_published": self.motion_published,
            "continuity_reason": self.continuity_reason,
            "monitor_warning": self.monitor_warning,
            "requires_fresh_localization": self.requires_fresh_localization,
            "requires_new_route_certificate": (
                self.requires_new_route_certificate
            ),
            "automatic_motion_authorized": self.automatic_motion_authorized,
        }


def evaluate_prestart_localization_reseal(
    *,
    status: object,
    motion_published: object,
    stop_details: object,
) -> PrestartLocalizationResealDecision:
    """Classify only exact, complete before-motion localization evidence.

    Malformed, incomplete, ambiguous, or differently phased evidence returns
    an ineligible decision instead of raising.  The caller must retain its
    existing terminal-stop behavior for every such decision.
    """

    if status != "stopped":
        return _rejected("outcome_not_stopped", motion_published)
    if not isinstance(motion_published, bool):
        return _rejected("motion_published_not_boolean", None)
    if motion_published:
        return _rejected("motion_already_published", motion_published)
    if not isinstance(stop_details, Mapping):
        return _rejected("stop_details_not_mapping", motion_published)

    required_top_level = (
        ("reason", GLOBAL_CONSISTENCY_STOP_REASON),
        ("fault_code", LOCALIZATION_RESEAL_FAULT_CODE),
        ("source", GLOBAL_CONSISTENCY_MONITOR_SOURCE),
        ("execution_phase", BEFORE_MOTION_EXECUTION_PHASE),
        ("phase", INITIAL_RUNTIME_INPUT_WAIT_PHASE),
        ("execution_pose_owner", "odom"),
        ("global_consistency_monitor", "amcl"),
        ("monitor_action", FORCE_ZERO_RESEAL_ACTION),
        ("monitor_reason", RESEAL_REQUIRED_MONITOR_REASON),
        ("fail_closed", True),
    )
    for name, expected in required_top_level:
        if not _exact_value(stop_details.get(name), expected):
            return _rejected(f"invalid_{name}", motion_published)

    nested_motion = stop_details.get("motion_published")
    if nested_motion is not False:
        return _rejected(
            "conflicting_stop_details_motion_published",
            motion_published,
        )

    monitor_warning = stop_details.get("monitor_warning")
    if not isinstance(monitor_warning, str):
        return _rejected("monitor_warning_not_string", motion_published)
    if monitor_warning != monitor_warning.strip():
        return _rejected("invalid_monitor_warning", motion_published)

    continuity = stop_details.get("continuity")
    if not isinstance(continuity, Mapping):
        return _rejected("continuity_not_mapping", motion_published)

    common_continuity = (
        ("schema_version", 1),
        ("accepted", False),
        ("requires_zero_cycle", True),
        ("requires_reseal", True),
        ("decision", FORCE_ZERO_RESEAL_DECISION),
        ("fail_closed", True),
        ("threshold_semantics", _CONTINUITY_THRESHOLD_SEMANTICS),
    )
    for name, expected in common_continuity:
        if not _exact_value(continuity.get(name), expected):
            return _rejected(
                f"invalid_continuity_{name}",
                motion_published,
            )

    contract_error = _continuity_identity_error(continuity)
    if contract_error:
        return _rejected(contract_error, motion_published)

    continuity_reason = continuity.get("reason")
    if not isinstance(continuity_reason, str) or not continuity_reason:
        return _rejected("invalid_continuity_reason", motion_published)

    if continuity_reason == _MISSING_CONTINUITY_REASON:
        warning_error = _tf_warmup_warning_error(monitor_warning)
        if warning_error:
            return _rejected(warning_error, motion_published)
        missing_error = _missing_transform_contract_error(continuity)
        if missing_error:
            return _rejected(missing_error, motion_published)
        return _eligible(
            reason="prestart_tf_warmup_retry_required",
            recovery_action=TF_WARMUP_RETRY,
            motion_published=motion_published,
            continuity_reason=continuity_reason,
            monitor_warning=monitor_warning,
        )

    if continuity_reason in _DRIFT_CONTINUITY_REASONS:
        if monitor_warning:
            return _rejected(
                "unexpected_monitor_warning_for_drift",
                motion_published,
            )
        drift_error = _drift_contract_error(
            continuity,
            continuity_reason=continuity_reason,
        )
        if drift_error:
            return _rejected(drift_error, motion_published)
        return _eligible(
            reason="prestart_fresh_localization_reseal_required",
            recovery_action=FRESH_LOCALIZATION_RESEAL,
            motion_published=motion_published,
            continuity_reason=continuity_reason,
            monitor_warning=monitor_warning,
        )

    return _rejected("unsupported_continuity_reason", motion_published)


def _continuity_identity_error(continuity: Mapping[str, object]) -> str:
    digest = continuity.get("certificate_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        return "invalid_continuity_certificate_sha256"

    frames = []
    for name in ("map_frame", "odom_frame", "base_frame"):
        frame = continuity.get(name)
        if (
            not isinstance(frame, str)
            or not frame
            or frame != frame.strip()
            or frame.startswith("/")
            or any(character.isspace() for character in frame)
        ):
            return f"invalid_continuity_{name}"
        frames.append(frame)
    if len(set(frames)) != len(frames):
        return "conflicting_continuity_frames"

    if not _valid_transform(continuity.get("frozen_map_from_odom")):
        return "invalid_continuity_frozen_map_from_odom"

    for name in ("max_translation_drift_m", "max_yaw_drift_rad"):
        if _finite_nonnegative(continuity.get(name)) is None:
            return f"invalid_continuity_{name}"
    max_yaw = _finite_nonnegative(continuity.get("max_yaw_drift_rad"))
    if max_yaw is None or max_yaw > math.pi:
        return "invalid_continuity_max_yaw_drift_rad"
    return ""


def _tf_warmup_warning_error(warning: str) -> str:
    if warning in _EXACT_TF_WARMUP_WARNINGS:
        return ""
    if warning.startswith(_LOOKUP_FAILURE_WARNING_PREFIX):
        detail = warning[len(_LOOKUP_FAILURE_WARNING_PREFIX) :]
        if detail and detail.strip() and detail == detail.rstrip():
            return ""
        return "invalid_map_from_odom_lookup_warning"
    if not warning:
        return "missing_tf_warmup_warning"
    return "unsupported_monitor_warning"


def _missing_transform_contract_error(
    continuity: Mapping[str, object],
) -> str:
    if continuity.get("live_map_from_odom", object()) is not None:
        return "invalid_missing_continuity_live_map_from_odom"
    for name in (
        "relative_translation_x_m",
        "relative_translation_y_m",
        "translation_drift_m",
        "relative_yaw_rad",
        "absolute_yaw_drift_rad",
    ):
        if continuity.get(name, object()) is not None:
            return f"invalid_missing_continuity_{name}"
    if continuity.get("validation_error") != "live map_from_odom is missing":
        return "invalid_missing_continuity_validation_error"
    return ""


def _drift_contract_error(
    continuity: Mapping[str, object],
    *,
    continuity_reason: str,
) -> str:
    if not _valid_transform(continuity.get("live_map_from_odom")):
        return "invalid_drift_continuity_live_map_from_odom"
    if continuity.get("validation_error", object()) is not None:
        return "invalid_drift_continuity_validation_error"

    signed_x = _finite_number(continuity.get("relative_translation_x_m"))
    signed_y = _finite_number(continuity.get("relative_translation_y_m"))
    translation = _finite_nonnegative(continuity.get("translation_drift_m"))
    relative_yaw = _finite_number(continuity.get("relative_yaw_rad"))
    absolute_yaw = _finite_nonnegative(
        continuity.get("absolute_yaw_drift_rad")
    )
    if signed_x is None or signed_y is None:
        return "invalid_drift_continuity_relative_translation"
    if translation is None:
        return "invalid_drift_continuity_translation_drift_m"
    if relative_yaw is None or absolute_yaw is None:
        return "invalid_drift_continuity_yaw_drift"
    if not math.isclose(
        math.hypot(signed_x, signed_y),
        translation,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        return "inconsistent_drift_continuity_translation"
    if not math.isclose(
        abs(relative_yaw),
        absolute_yaw,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        return "inconsistent_drift_continuity_yaw"

    max_translation = _finite_nonnegative(
        continuity.get("max_translation_drift_m")
    )
    max_yaw = _finite_nonnegative(continuity.get("max_yaw_drift_rad"))
    if max_translation is None or max_yaw is None:
        return "invalid_drift_continuity_limits"
    translation_exceeded = translation > max_translation
    yaw_exceeded = absolute_yaw > max_yaw
    expected_reason = (
        "map_from_odom_translation_and_yaw_drift"
        if translation_exceeded and yaw_exceeded
        else "map_from_odom_translation_drift"
        if translation_exceeded
        else "map_from_odom_yaw_drift"
        if yaw_exceeded
        else ""
    )
    if continuity_reason != expected_reason:
        return "inconsistent_drift_continuity_reason"
    return ""


def _valid_transform(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != _TRANSFORM_FIELDS:
        return False
    x_m = _finite_number(value.get("x_m"))
    y_m = _finite_number(value.get("y_m"))
    yaw_rad = _finite_number(value.get("yaw_rad"))
    return (
        x_m is not None
        and y_m is not None
        and yaw_rad is not None
        and -math.pi <= yaw_rad < math.pi
    )


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _finite_nonnegative(value: object) -> float | None:
    result = _finite_number(value)
    if result is None or result < 0.0:
        return None
    return result


def _exact_value(value: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return value is expected
    if isinstance(expected, int):
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value == expected
        )
    return value == expected


def _eligible(
    *,
    reason: str,
    recovery_action: str,
    motion_published: bool,
    continuity_reason: str,
    monitor_warning: str,
) -> PrestartLocalizationResealDecision:
    return PrestartLocalizationResealDecision(
        eligible=True,
        reason=reason,
        recovery_action=recovery_action,
        execution_phase=BEFORE_MOTION_EXECUTION_PHASE,
        motion_published=motion_published,
        continuity_reason=continuity_reason,
        monitor_warning=monitor_warning,
        requires_fresh_localization=True,
        requires_new_route_certificate=True,
    )


def _rejected(
    reason: str,
    motion_published: object,
) -> PrestartLocalizationResealDecision:
    return PrestartLocalizationResealDecision(
        eligible=False,
        reason=reason,
        recovery_action="",
        execution_phase="not_admitted",
        motion_published=(
            motion_published if isinstance(motion_published, bool) else None
        ),
        continuity_reason="",
        monitor_warning="",
        requires_fresh_localization=False,
        requires_new_route_certificate=False,
    )
