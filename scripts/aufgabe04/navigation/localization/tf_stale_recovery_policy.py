"""ROS-free admission policy for bounded stale-TF recovery.

This module only classifies immutable evidence.  It does not call ROS
services, spin executors, publish velocity commands, or select routes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


TF_STATUS_FRESH = "fresh"
TF_STATUS_STALE = "stale"
TF_STATUS_FUTURE = "future"
TF_STATUS_UNAVAILABLE = "unavailable"


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative(value: float, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _angle_delta(first_rad: float, second_rad: float) -> float:
    return math.atan2(
        math.sin(second_rad - first_rad),
        math.cos(second_rad - first_rad),
    )


@dataclass(frozen=True)
class TfEdgeSample:
    """One timestamp observation for a directed TF edge.

    ``stamp_sec=None`` represents a failed lookup.  A direct AMCL
    ``map->odom`` edge may legitimately be future dated, so callers classify
    it with a separate future tolerance from a composed ``map->base`` edge.
    """

    parent_frame: str
    child_frame: str
    stamp_sec: float | None

    def __post_init__(self) -> None:
        _nonempty(self.parent_frame, "parent_frame")
        _nonempty(self.child_frame, "child_frame")
        if self.parent_frame == self.child_frame:
            raise ValueError("parent_frame and child_frame must differ")
        if self.stamp_sec is not None:
            _nonnegative(self.stamp_sec, "stamp_sec")

    @property
    def available(self) -> bool:
        return self.stamp_sec is not None

    def age_sec(self, *, now_sec: float) -> float | None:
        now = _nonnegative(now_sec, "now_sec")
        if self.stamp_sec is None:
            return None
        return now - float(self.stamp_sec)

    def to_log_dict(self, *, now_sec: float | None = None) -> dict[str, object]:
        data: dict[str, object] = {
            "parent_frame": self.parent_frame,
            "child_frame": self.child_frame,
            "available": self.available,
            "stamp_sec": self.stamp_sec,
        }
        if now_sec is not None:
            data["age_sec"] = self.age_sec(now_sec=now_sec)
        return data


@dataclass(frozen=True)
class OdomStationaritySample:
    """One odometry callback used to prove a zero-held robot stayed still."""

    callback_count: int
    stamp_sec: float
    x_m: float
    y_m: float
    yaw_rad: float
    linear_x_mps: float
    angular_z_radps: float

    def __post_init__(self) -> None:
        if type(self.callback_count) is not int or self.callback_count < 0:
            raise ValueError("callback_count must be a non-negative integer")
        _nonnegative(self.stamp_sec, "stamp_sec")
        for name in (
            "x_m",
            "y_m",
            "yaw_rad",
            "linear_x_mps",
            "angular_z_radps",
        ):
            _finite(getattr(self, name), name)

    def to_log_dict(self) -> dict[str, object]:
        return {
            "callback_count": self.callback_count,
            "stamp_sec": float(self.stamp_sec),
            "pose": {
                "x_m": float(self.x_m),
                "y_m": float(self.y_m),
                "yaw_rad": float(self.yaw_rad),
            },
            "twist": {
                "linear_x_mps": float(self.linear_x_mps),
                "angular_z_radps": float(self.angular_z_radps),
            },
        }


@dataclass(frozen=True)
class StationarityLimits:
    """Thresholds for two distinct, fresh odometry observations."""

    max_sample_age_sec: float = 0.5
    future_tolerance_sec: float = 0.02
    min_sample_separation_sec: float = 0.08
    max_translation_m: float = 0.01
    max_yaw_rad: float = math.radians(2.0)
    max_linear_speed_mps: float = 0.01
    max_angular_speed_radps: float = 0.05

    def __post_init__(self) -> None:
        _positive(self.max_sample_age_sec, "max_sample_age_sec")
        _nonnegative(self.future_tolerance_sec, "future_tolerance_sec")
        _positive(self.min_sample_separation_sec, "min_sample_separation_sec")
        if self.min_sample_separation_sec > self.max_sample_age_sec:
            raise ValueError(
                "min_sample_separation_sec must not exceed max_sample_age_sec"
            )
        for name in (
            "max_translation_m",
            "max_yaw_rad",
            "max_linear_speed_mps",
            "max_angular_speed_radps",
        ):
            _nonnegative(getattr(self, name), name)

    def to_log_dict(self) -> dict[str, object]:
        return {
            "max_sample_age_sec": float(self.max_sample_age_sec),
            "future_tolerance_sec": float(self.future_tolerance_sec),
            "min_sample_separation_sec": float(self.min_sample_separation_sec),
            "max_translation_m": float(self.max_translation_m),
            "max_yaw_rad": float(self.max_yaw_rad),
            "max_linear_speed_mps": float(self.max_linear_speed_mps),
            "max_angular_speed_radps": float(self.max_angular_speed_radps),
        }


@dataclass(frozen=True)
class StationarityDecision:
    accepted: bool
    reason: str
    reasons: tuple[str, ...]
    callback_advanced: bool
    stamp_advanced: bool
    sample_separation_sec: float
    translation_delta_m: float
    yaw_delta_rad: float
    max_abs_linear_speed_mps: float
    max_abs_angular_speed_radps: float
    first_age_sec: float
    second_age_sec: float
    limits: StationarityLimits

    def to_log_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "reasons": list(self.reasons),
            "callback_advanced": self.callback_advanced,
            "stamp_advanced": self.stamp_advanced,
            "sample_separation_sec": self.sample_separation_sec,
            "translation_delta_m": self.translation_delta_m,
            "yaw_delta_rad": self.yaw_delta_rad,
            "max_abs_linear_speed_mps": self.max_abs_linear_speed_mps,
            "max_abs_angular_speed_radps": self.max_abs_angular_speed_radps,
            "first_age_sec": self.first_age_sec,
            "second_age_sec": self.second_age_sec,
            "limits": self.limits.to_log_dict(),
        }


def evaluate_stationarity(
    first: OdomStationaritySample,
    second: OdomStationaritySample,
    *,
    now_sec: float,
    limits: StationarityLimits = StationarityLimits(),
) -> StationarityDecision:
    """Require distinct callbacks, advancing stamps, stable pose, and zero twist."""

    if not isinstance(first, OdomStationaritySample):
        raise ValueError("first must be an OdomStationaritySample")
    if not isinstance(second, OdomStationaritySample):
        raise ValueError("second must be an OdomStationaritySample")
    if not isinstance(limits, StationarityLimits):
        raise ValueError("limits must be StationarityLimits")
    now = _nonnegative(now_sec, "now_sec")

    callback_advanced = second.callback_count > first.callback_count
    stamp_advanced = second.stamp_sec > first.stamp_sec
    separation = second.stamp_sec - first.stamp_sec
    first_age = now - first.stamp_sec
    second_age = now - second.stamp_sec
    translation = math.hypot(second.x_m - first.x_m, second.y_m - first.y_m)
    yaw_delta = abs(_angle_delta(first.yaw_rad, second.yaw_rad))
    max_linear = max(abs(first.linear_x_mps), abs(second.linear_x_mps))
    max_angular = max(abs(first.angular_z_radps), abs(second.angular_z_radps))

    reasons: list[str] = []
    if not callback_advanced:
        reasons.append("odom_callback_not_advanced")
    if not stamp_advanced:
        reasons.append("odom_stamp_not_advanced")
    if stamp_advanced and separation < limits.min_sample_separation_sec:
        reasons.append("odom_sample_separation_too_short")
    if first_age > limits.max_sample_age_sec:
        reasons.append("first_odom_sample_stale")
    if second_age > limits.max_sample_age_sec:
        reasons.append("second_odom_sample_stale")
    if first_age < -limits.future_tolerance_sec:
        reasons.append("first_odom_sample_future")
    if second_age < -limits.future_tolerance_sec:
        reasons.append("second_odom_sample_future")
    if translation > limits.max_translation_m:
        reasons.append("odom_translation_not_stationary")
    if yaw_delta > limits.max_yaw_rad:
        reasons.append("odom_yaw_not_stationary")
    if max_linear > limits.max_linear_speed_mps:
        reasons.append("odom_linear_twist_not_stationary")
    if max_angular > limits.max_angular_speed_radps:
        reasons.append("odom_angular_twist_not_stationary")

    accepted = not reasons
    reason_tuple = tuple(reasons) if reasons else ("odom_stationary",)
    return StationarityDecision(
        accepted=accepted,
        reason=reason_tuple[0],
        reasons=reason_tuple,
        callback_advanced=callback_advanced,
        stamp_advanced=stamp_advanced,
        sample_separation_sec=separation,
        translation_delta_m=translation,
        yaw_delta_rad=yaw_delta,
        max_abs_linear_speed_mps=max_linear,
        max_abs_angular_speed_radps=max_angular,
        first_age_sec=first_age,
        second_age_sec=second_age,
        limits=limits,
    )


def _edge_status(
    sample: TfEdgeSample,
    *,
    now_sec: float,
    max_age_sec: float,
    future_tolerance_sec: float,
) -> str:
    if not isinstance(sample, TfEdgeSample):
        raise ValueError("TF evidence must be TfEdgeSample")
    if sample.stamp_sec is None:
        return TF_STATUS_UNAVAILABLE
    age = now_sec - sample.stamp_sec
    if age < -future_tolerance_sec:
        return TF_STATUS_FUTURE
    if age > max_age_sec:
        return TF_STATUS_STALE
    return TF_STATUS_FRESH


def _same_edge(first: TfEdgeSample, second: TfEdgeSample) -> bool:
    return (
        first.parent_frame == second.parent_frame
        and first.child_frame == second.child_frame
    )


def _advanced(first: TfEdgeSample, second: TfEdgeSample) -> bool:
    return (
        first.stamp_sec is not None
        and second.stamp_sec is not None
        and second.stamp_sec > first.stamp_sec
    )


def _regressed(first: TfEdgeSample, second: TfEdgeSample) -> bool:
    return (
        first.stamp_sec is not None
        and second.stamp_sec is not None
        and second.stamp_sec < first.stamp_sec
    )


def _edges_form_chain(
    composed: TfEdgeSample,
    map_to_odom: TfEdgeSample,
    odom_to_base: TfEdgeSample,
) -> bool:
    return (
        composed.parent_frame == map_to_odom.parent_frame
        and map_to_odom.child_frame == odom_to_base.parent_frame
        and odom_to_base.child_frame == composed.child_frame
    )


@dataclass(frozen=True)
class RecoveryEligibilityDecision:
    accepted: bool
    reason: str
    reasons: tuple[str, ...]
    composed_retry_status: str
    map_to_odom_retry_status: str
    odom_to_base_retry_status: str
    composed_advanced: bool
    map_to_odom_advanced: bool
    composed_retry_stamp_sec: float | None
    map_to_odom_retry_stamp_sec: float | None

    def to_log_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "reasons": list(self.reasons),
            "composed_retry_status": self.composed_retry_status,
            "map_to_odom_retry_status": self.map_to_odom_retry_status,
            "odom_to_base_retry_status": self.odom_to_base_retry_status,
            "composed_advanced": self.composed_advanced,
            "map_to_odom_advanced": self.map_to_odom_advanced,
            "composed_retry_stamp_sec": self.composed_retry_stamp_sec,
            "map_to_odom_retry_stamp_sec": self.map_to_odom_retry_stamp_sec,
        }


def evaluate_recovery_eligibility(
    *,
    localization_source: str,
    use_sim_time: bool,
    composed_before: TfEdgeSample,
    composed_retry: TfEdgeSample,
    map_to_odom_before: TfEdgeSample,
    map_to_odom_retry: TfEdgeSample,
    odom_to_base_retry: TfEdgeSample,
    now_sec: float,
    max_tf_age_sec: float,
    composed_future_tolerance_sec: float = 0.02,
    map_to_odom_future_tolerance_sec: float = 1.1,
) -> RecoveryEligibilityDecision:
    """Admit a no-motion AMCL request only for the diagnosed AMCL edge case."""

    source = _nonempty(localization_source, "localization_source").strip().lower()
    if type(use_sim_time) is not bool:
        raise ValueError("use_sim_time must be bool")
    now = _nonnegative(now_sec, "now_sec")
    max_age = _positive(max_tf_age_sec, "max_tf_age_sec")
    composed_future = _nonnegative(
        composed_future_tolerance_sec, "composed_future_tolerance_sec"
    )
    map_future = _nonnegative(
        map_to_odom_future_tolerance_sec,
        "map_to_odom_future_tolerance_sec",
    )
    for sample in (
        composed_before,
        composed_retry,
        map_to_odom_before,
        map_to_odom_retry,
        odom_to_base_retry,
    ):
        if not isinstance(sample, TfEdgeSample):
            raise ValueError("all TF evidence must be TfEdgeSample")

    composed_status = _edge_status(
        composed_retry,
        now_sec=now,
        max_age_sec=max_age,
        future_tolerance_sec=composed_future,
    )
    map_status = _edge_status(
        map_to_odom_retry,
        now_sec=now,
        max_age_sec=max_age,
        future_tolerance_sec=map_future,
    )
    odom_status = _edge_status(
        odom_to_base_retry,
        now_sec=now,
        max_age_sec=max_age,
        future_tolerance_sec=composed_future,
    )
    composed_advanced = _advanced(composed_before, composed_retry)
    map_advanced = _advanced(map_to_odom_before, map_to_odom_retry)
    composed_regressed = _regressed(composed_before, composed_retry)
    map_regressed = _regressed(map_to_odom_before, map_to_odom_retry)

    reasons: list[str] = []
    if source != "amcl":
        reasons.append("localization_source_not_amcl")
    if use_sim_time:
        reasons.append("sim_time_recovery_forbidden")
    if not _same_edge(composed_before, composed_retry):
        reasons.append("composed_edge_changed")
    if composed_before.stamp_sec is None:
        reasons.append("composed_before_unavailable")
    if composed_regressed:
        reasons.append("composed_retry_stamp_regressed")
    if composed_status != TF_STATUS_STALE:
        reasons.append(f"composed_retry_not_stale:{composed_status}")
    if not _same_edge(map_to_odom_before, map_to_odom_retry):
        reasons.append("map_to_odom_edge_changed")
    if map_to_odom_before.stamp_sec is None:
        reasons.append("map_to_odom_before_unavailable")
    if map_regressed:
        reasons.append("map_to_odom_retry_stamp_regressed")
    if map_status == TF_STATUS_UNAVAILABLE:
        reasons.append("map_to_odom_retry_unavailable")
    elif map_status == TF_STATUS_FUTURE:
        reasons.append("map_to_odom_retry_future")
    elif map_status != TF_STATUS_STALE and map_advanced:
        reasons.append("map_to_odom_not_stale_or_nonadvancing")
    if odom_status != TF_STATUS_FRESH:
        reasons.append(f"odom_to_base_retry_not_fresh:{odom_status}")
    if not _edges_form_chain(
        composed_retry, map_to_odom_retry, odom_to_base_retry
    ):
        reasons.append("tf_edge_topology_inconsistent")

    accepted = not reasons
    reason_tuple = (
        tuple(reasons)
        if reasons
        else ("real_amcl_stale_edge_recovery_eligible",)
    )
    return RecoveryEligibilityDecision(
        accepted=accepted,
        reason=reason_tuple[0],
        reasons=reason_tuple,
        composed_retry_status=composed_status,
        map_to_odom_retry_status=map_status,
        odom_to_base_retry_status=odom_status,
        composed_advanced=composed_advanced,
        map_to_odom_advanced=map_advanced,
        composed_retry_stamp_sec=composed_retry.stamp_sec,
        map_to_odom_retry_stamp_sec=map_to_odom_retry.stamp_sec,
    )


@dataclass(frozen=True)
class RecoveryAcceptanceDecision:
    accepted: bool
    reason: str
    reasons: tuple[str, ...]
    composed_recovered_status: str
    map_to_odom_recovered_status: str
    odom_to_base_recovered_status: str
    composed_strictly_newer: bool
    map_to_odom_strictly_newer: bool
    scan_fresh: bool
    odom_fresh: bool
    exclusive_cmd_vel_owner: bool
    stationary: bool

    def to_log_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "reasons": list(self.reasons),
            "composed_recovered_status": self.composed_recovered_status,
            "map_to_odom_recovered_status": self.map_to_odom_recovered_status,
            "odom_to_base_recovered_status": self.odom_to_base_recovered_status,
            "composed_strictly_newer": self.composed_strictly_newer,
            "map_to_odom_strictly_newer": self.map_to_odom_strictly_newer,
            "scan_fresh": self.scan_fresh,
            "odom_fresh": self.odom_fresh,
            "exclusive_cmd_vel_owner": self.exclusive_cmd_vel_owner,
            "stationary": self.stationary,
        }


def evaluate_recovery_acceptance(
    *,
    eligibility: RecoveryEligibilityDecision,
    composed_before: TfEdgeSample,
    composed_recovered: TfEdgeSample,
    map_to_odom_before: TfEdgeSample,
    map_to_odom_recovered: TfEdgeSample,
    odom_to_base_recovered: TfEdgeSample,
    stationarity: StationarityDecision,
    scan_fresh: bool,
    odom_fresh: bool,
    exclusive_cmd_vel_owner: bool,
    now_sec: float,
    max_tf_age_sec: float,
    composed_future_tolerance_sec: float = 0.02,
    map_to_odom_future_tolerance_sec: float = 1.1,
) -> RecoveryAcceptanceDecision:
    """Accept recovery only after every stopped-runtime safety gate passes."""

    if not isinstance(eligibility, RecoveryEligibilityDecision):
        raise ValueError("eligibility must be RecoveryEligibilityDecision")
    if not isinstance(stationarity, StationarityDecision):
        raise ValueError("stationarity must be StationarityDecision")
    for name, value in (
        ("scan_fresh", scan_fresh),
        ("odom_fresh", odom_fresh),
        ("exclusive_cmd_vel_owner", exclusive_cmd_vel_owner),
    ):
        if type(value) is not bool:
            raise ValueError(f"{name} must be bool")
    now = _nonnegative(now_sec, "now_sec")
    max_age = _positive(max_tf_age_sec, "max_tf_age_sec")
    composed_future = _nonnegative(
        composed_future_tolerance_sec, "composed_future_tolerance_sec"
    )
    map_future = _nonnegative(
        map_to_odom_future_tolerance_sec,
        "map_to_odom_future_tolerance_sec",
    )
    for sample in (
        composed_before,
        composed_recovered,
        map_to_odom_before,
        map_to_odom_recovered,
        odom_to_base_recovered,
    ):
        if not isinstance(sample, TfEdgeSample):
            raise ValueError("all TF evidence must be TfEdgeSample")

    composed_status = _edge_status(
        composed_recovered,
        now_sec=now,
        max_age_sec=max_age,
        future_tolerance_sec=composed_future,
    )
    map_status = _edge_status(
        map_to_odom_recovered,
        now_sec=now,
        max_age_sec=max_age,
        future_tolerance_sec=map_future,
    )
    odom_status = _edge_status(
        odom_to_base_recovered,
        now_sec=now,
        max_age_sec=max_age,
        future_tolerance_sec=composed_future,
    )

    composed_baselines = tuple(
        stamp
        for stamp in (
            composed_before.stamp_sec,
            eligibility.composed_retry_stamp_sec,
        )
        if stamp is not None
    )
    map_baselines = tuple(
        stamp
        for stamp in (
            map_to_odom_before.stamp_sec,
            eligibility.map_to_odom_retry_stamp_sec,
        )
        if stamp is not None
    )
    composed_newer = (
        composed_recovered.stamp_sec is not None
        and bool(composed_baselines)
        and composed_recovered.stamp_sec > max(composed_baselines)
    )
    map_newer = (
        map_to_odom_recovered.stamp_sec is not None
        and bool(map_baselines)
        and map_to_odom_recovered.stamp_sec > max(map_baselines)
    )

    reasons: list[str] = []
    if not eligibility.accepted:
        reasons.append("recovery_not_eligible")
    if not _same_edge(composed_before, composed_recovered):
        reasons.append("composed_edge_changed")
    if composed_status != TF_STATUS_FRESH:
        reasons.append(f"composed_recovered_not_fresh:{composed_status}")
    if not composed_newer:
        reasons.append("composed_transform_not_strictly_newer")
    if not _same_edge(map_to_odom_before, map_to_odom_recovered):
        reasons.append("map_to_odom_edge_changed")
    if map_status != TF_STATUS_FRESH:
        reasons.append(f"map_to_odom_recovered_not_fresh:{map_status}")
    if not map_newer:
        reasons.append("map_to_odom_transform_not_strictly_newer")
    if odom_status != TF_STATUS_FRESH:
        reasons.append(f"odom_to_base_recovered_not_fresh:{odom_status}")
    if not _edges_form_chain(
        composed_recovered,
        map_to_odom_recovered,
        odom_to_base_recovered,
    ):
        reasons.append("tf_edge_topology_inconsistent")
    if not scan_fresh:
        reasons.append("scan_not_fresh")
    if not odom_fresh:
        reasons.append("odom_not_fresh")
    if not exclusive_cmd_vel_owner:
        reasons.append("cmd_vel_owner_not_exclusive")
    if not stationarity.accepted:
        reasons.append("stationarity_not_confirmed")

    accepted = not reasons
    reason_tuple = tuple(reasons) if reasons else ("stale_tf_recovery_accepted",)
    return RecoveryAcceptanceDecision(
        accepted=accepted,
        reason=reason_tuple[0],
        reasons=reason_tuple,
        composed_recovered_status=composed_status,
        map_to_odom_recovered_status=map_status,
        odom_to_base_recovered_status=odom_status,
        composed_strictly_newer=composed_newer,
        map_to_odom_strictly_newer=map_newer,
        scan_fresh=scan_fresh,
        odom_fresh=odom_fresh,
        exclusive_cmd_vel_owner=exclusive_cmd_vel_owner,
        stationary=stationarity.accepted,
    )
