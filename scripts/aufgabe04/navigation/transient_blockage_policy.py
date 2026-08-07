"""ROS-free evidence policy for transient navigation blockages.

The policy deliberately separates three questions:

* whether a scaled linear command is large enough to be treated as motion,
* whether repeated front-sector returns were captured while the robot stayed
  stationary in both odom and map, and
* how much distance progress was physically reachable during an observation
  interval.

No function in this module publishes motion or mutates planner state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Iterable, Sequence

from scripts.aufgabe04.navigation.models import Pose2D


DEFAULT_LINEAR_MOTION_FLOOR_MPS = 0.01
CLEARANCE_LIMITED_MOTION_FLOOR = "clearance-limited motion floor"

LINEAR_COMMAND_ZERO = "zero"
LINEAR_COMMAND_BELOW_FLOOR = "below_motion_floor"
LINEAR_COMMAND_MOTION_CAPABLE = "at_or_above_motion_floor"


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_nonnegative(value: float, name: str) -> float:
    result = _finite_number(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _finite_positive(value: float, name: str) -> float:
    result = _finite_number(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _linear_command_class(value_mps: float, floor_mps: float) -> str:
    magnitude = abs(value_mps)
    if magnitude == 0.0:
        return LINEAR_COMMAND_ZERO
    if magnitude < floor_mps:
        return LINEAR_COMMAND_BELOW_FLOOR
    return LINEAR_COMMAND_MOTION_CAPABLE


@dataclass(frozen=True)
class LinearCommandFloorDecision:
    """Classification of nominal and clearance-scaled linear commands."""

    nominal_linear_x_mps: float
    effective_linear_x_mps: float
    linear_motion_floor_mps: float
    nominal_class: str
    effective_class: str
    output_linear_x_mps: float
    zero_hold_required: bool
    stationary_confirmation_required: bool
    fail_closed: bool
    reasons: tuple[str, ...]

    def to_log_dict(self) -> dict[str, object]:
        return {
            "nominal_linear_x_mps": self.nominal_linear_x_mps,
            "effective_linear_x_mps": self.effective_linear_x_mps,
            "linear_motion_floor_mps": self.linear_motion_floor_mps,
            "nominal_class": self.nominal_class,
            "effective_class": self.effective_class,
            "output_linear_x_mps": self.output_linear_x_mps,
            "zero_hold_required": self.zero_hold_required,
            "stationary_confirmation_required": (
                self.stationary_confirmation_required
            ),
            "fail_closed": self.fail_closed,
            "reason": self.reasons[0],
            "reasons": list(self.reasons),
        }


def classify_linear_command(
    nominal_linear_x_mps: float,
    effective_linear_x_mps: float,
    *,
    linear_motion_floor_mps: float = DEFAULT_LINEAR_MOTION_FLOOR_MPS,
) -> LinearCommandFloorDecision:
    """Fail safely when scaling leaves a nonzero but unreachable command.

    A nominal request that has already been scaled to exactly zero is also a
    stationary-confirmation case, although it does not need the sub-floor
    command itself to be replaced.
    """

    nominal = _finite_number(nominal_linear_x_mps, "nominal_linear_x_mps")
    effective = _finite_number(
        effective_linear_x_mps, "effective_linear_x_mps"
    )
    floor = _finite_positive(
        linear_motion_floor_mps, "linear_motion_floor_mps"
    )
    nominal_class = _linear_command_class(nominal, floor)
    effective_class = _linear_command_class(effective, floor)

    if effective_class == LINEAR_COMMAND_BELOW_FLOOR:
        return LinearCommandFloorDecision(
            nominal_linear_x_mps=nominal,
            effective_linear_x_mps=effective,
            linear_motion_floor_mps=floor,
            nominal_class=nominal_class,
            effective_class=effective_class,
            output_linear_x_mps=0.0,
            zero_hold_required=True,
            stationary_confirmation_required=True,
            fail_closed=True,
            reasons=(CLEARANCE_LIMITED_MOTION_FLOOR,),
        )

    if effective_class == LINEAR_COMMAND_ZERO and nominal_class != LINEAR_COMMAND_ZERO:
        return LinearCommandFloorDecision(
            nominal_linear_x_mps=nominal,
            effective_linear_x_mps=effective,
            linear_motion_floor_mps=floor,
            nominal_class=nominal_class,
            effective_class=effective_class,
            output_linear_x_mps=0.0,
            zero_hold_required=True,
            stationary_confirmation_required=True,
            fail_closed=True,
            reasons=("effective_linear_zero_during_nominal_motion",),
        )

    reason = (
        "linear_command_is_zero"
        if effective_class == LINEAR_COMMAND_ZERO
        else "effective_linear_motion_capable"
    )
    return LinearCommandFloorDecision(
        nominal_linear_x_mps=nominal,
        effective_linear_x_mps=effective,
        linear_motion_floor_mps=floor,
        nominal_class=nominal_class,
        effective_class=effective_class,
        output_linear_x_mps=effective,
        zero_hold_required=False,
        stationary_confirmation_required=False,
        fail_closed=False,
        reasons=(reason,),
    )


def _validate_pose(pose: Pose2D, name: str) -> None:
    if not isinstance(pose, Pose2D):
        raise ValueError(f"{name} must be a Pose2D")
    for field_name, value in (
        ("x_m", pose.x_m),
        ("y_m", pose.y_m),
        ("yaw_rad", pose.yaw_rad),
    ):
        _finite_number(value, f"{name}.{field_name}")


@dataclass(frozen=True)
class StationaryFrontSectorSample:
    """One front ray and both pose estimates captured during a zero hold."""

    timestamp_sec: float
    front_range_m: float
    front_bearing_rad: float
    map_pose: Pose2D
    odom_pose: Pose2D

    def __post_init__(self) -> None:
        _finite_nonnegative(self.timestamp_sec, "timestamp_sec")
        _finite_positive(self.front_range_m, "front_range_m")
        _finite_number(self.front_bearing_rad, "front_bearing_rad")
        _validate_pose(self.map_pose, "map_pose")
        _validate_pose(self.odom_pose, "odom_pose")

    def to_log_dict(self) -> dict[str, object]:
        return {
            "timestamp_sec": float(self.timestamp_sec),
            "front_range_m": float(self.front_range_m),
            "front_bearing_rad": float(self.front_bearing_rad),
            "map_pose": {
                "x_m": float(self.map_pose.x_m),
                "y_m": float(self.map_pose.y_m),
                "yaw_rad": float(self.map_pose.yaw_rad),
            },
            "odom_pose": {
                "x_m": float(self.odom_pose.x_m),
                "y_m": float(self.odom_pose.y_m),
                "yaw_rad": float(self.odom_pose.yaw_rad),
            },
        }


@dataclass(frozen=True)
class PersistentObstacleConfig:
    """All thresholds used to confirm a stationary persistent obstacle."""

    min_distinct_samples: int = 3
    max_sample_age_sec: float = 0.8
    max_sample_window_sec: float = 0.5
    min_sample_separation_sec: float = 0.08
    future_timestamp_tolerance_sec: float = 0.02
    min_front_range_m: float = 0.12
    max_front_range_m: float = 0.50
    front_sector_half_width_rad: float = math.radians(35.0)
    max_front_range_spread_m: float = 0.04
    max_map_hit_spread_m: float = 0.05
    max_map_pose_translation_spread_m: float = 0.025
    max_map_pose_yaw_spread_rad: float = math.radians(3.0)
    max_odom_pose_translation_spread_m: float = 0.015
    max_odom_pose_yaw_spread_rad: float = math.radians(2.0)
    max_map_odom_offset_spread_m: float = 0.025
    max_map_odom_yaw_offset_spread_rad: float = math.radians(2.0)

    def __post_init__(self) -> None:
        if (
            type(self.min_distinct_samples) is not int
            or self.min_distinct_samples < 2
        ):
            raise ValueError("min_distinct_samples must be an integer >= 2")
        for name in (
            "max_sample_age_sec",
            "max_sample_window_sec",
            "min_front_range_m",
            "max_front_range_m",
            "front_sector_half_width_rad",
        ):
            _finite_positive(getattr(self, name), name)
        for name in (
            "future_timestamp_tolerance_sec",
            "max_front_range_spread_m",
            "max_map_hit_spread_m",
            "max_map_pose_translation_spread_m",
            "max_map_pose_yaw_spread_rad",
            "max_odom_pose_translation_spread_m",
            "max_odom_pose_yaw_spread_rad",
            "max_map_odom_offset_spread_m",
            "max_map_odom_yaw_offset_spread_rad",
        ):
            _finite_nonnegative(getattr(self, name), name)
        _finite_positive(
            self.min_sample_separation_sec, "min_sample_separation_sec"
        )
        if self.max_sample_window_sec > self.max_sample_age_sec:
            raise ValueError(
                "max_sample_window_sec must not exceed max_sample_age_sec"
            )
        if self.min_sample_separation_sec > self.max_sample_window_sec:
            raise ValueError(
                "min_sample_separation_sec must not exceed max_sample_window_sec"
            )
        if (
            (self.min_distinct_samples - 1) * self.min_sample_separation_sec
            > self.max_sample_window_sec
        ):
            raise ValueError(
                "sample window cannot contain min_distinct_samples at the "
                "configured separation"
            )
        if self.min_front_range_m >= self.max_front_range_m:
            raise ValueError("front range bounds must be strictly increasing")
        if self.front_sector_half_width_rad > math.pi:
            raise ValueError("front_sector_half_width_rad must not exceed pi")

    def to_log_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class PersistentObstacleDecision:
    """Fail-closed confirmation result with log-ready evidence metrics."""

    confirmed: bool
    fail_closed: bool
    reasons: tuple[str, ...]
    config: PersistentObstacleConfig
    supplied_sample_count: int
    recent_sample_count: int
    window_sample_count: int
    distinct_sample_count: int
    stale_sample_count: int
    future_sample_count: int
    duplicate_sample_count: int
    first_timestamp_sec: float | None
    last_timestamp_sec: float | None
    sample_window_sec: float | None
    median_front_range_m: float | None
    median_front_bearing_rad: float | None
    front_range_spread_m: float | None
    median_map_hit_x_m: float | None
    median_map_hit_y_m: float | None
    map_hit_spread_m: float | None
    map_pose_translation_spread_m: float | None
    map_pose_yaw_spread_rad: float | None
    odom_pose_translation_spread_m: float | None
    odom_pose_yaw_spread_rad: float | None
    map_odom_offset_spread_m: float | None
    map_odom_yaw_offset_spread_rad: float | None

    def to_log_dict(self) -> dict[str, object]:
        return {
            "confirmed": self.confirmed,
            "fail_closed": self.fail_closed,
            "reason": self.reasons[0],
            "reasons": list(self.reasons),
            "thresholds": self.config.to_log_dict(),
            "supplied_sample_count": self.supplied_sample_count,
            "recent_sample_count": self.recent_sample_count,
            "window_sample_count": self.window_sample_count,
            "distinct_sample_count": self.distinct_sample_count,
            "stale_sample_count": self.stale_sample_count,
            "future_sample_count": self.future_sample_count,
            "duplicate_sample_count": self.duplicate_sample_count,
            "first_timestamp_sec": self.first_timestamp_sec,
            "last_timestamp_sec": self.last_timestamp_sec,
            "sample_window_sec": self.sample_window_sec,
            "median_front_range_m": self.median_front_range_m,
            "median_front_bearing_rad": self.median_front_bearing_rad,
            "front_range_spread_m": self.front_range_spread_m,
            "median_map_hit": (
                None
                if self.median_map_hit_x_m is None
                or self.median_map_hit_y_m is None
                else {
                    "x_m": self.median_map_hit_x_m,
                    "y_m": self.median_map_hit_y_m,
                }
            ),
            "map_hit_spread_m": self.map_hit_spread_m,
            "map_pose_translation_spread_m": (
                self.map_pose_translation_spread_m
            ),
            "map_pose_yaw_spread_rad": self.map_pose_yaw_spread_rad,
            "odom_pose_translation_spread_m": (
                self.odom_pose_translation_spread_m
            ),
            "odom_pose_yaw_spread_rad": self.odom_pose_yaw_spread_rad,
            "map_odom_offset_spread_m": self.map_odom_offset_spread_m,
            "map_odom_yaw_offset_spread_rad": (
                self.map_odom_yaw_offset_spread_rad
            ),
        }


@dataclass(frozen=True)
class StationaryClearanceDecision:
    """Fail-closed proof that a stationary front sector is clear."""

    confirmed: bool
    fail_closed: bool
    reasons: tuple[str, ...]
    config: PersistentObstacleConfig
    clearance_threshold_m: float
    supplied_sample_count: int
    recent_sample_count: int
    window_sample_count: int
    distinct_sample_count: int
    stale_sample_count: int
    future_sample_count: int
    duplicate_sample_count: int
    first_timestamp_sec: float | None
    last_timestamp_sec: float | None
    sample_window_sec: float | None
    minimum_front_range_m: float | None
    median_front_range_m: float | None
    median_front_bearing_rad: float | None
    map_pose_translation_spread_m: float | None
    map_pose_yaw_spread_rad: float | None
    odom_pose_translation_spread_m: float | None
    odom_pose_yaw_spread_rad: float | None
    map_odom_offset_spread_m: float | None
    map_odom_yaw_offset_spread_rad: float | None

    def to_log_dict(self) -> dict[str, object]:
        return {
            "confirmed": self.confirmed,
            "fail_closed": self.fail_closed,
            "reason": self.reasons[0],
            "reasons": list(self.reasons),
            "clearance_threshold_m": self.clearance_threshold_m,
            "thresholds": self.config.to_log_dict(),
            "supplied_sample_count": self.supplied_sample_count,
            "recent_sample_count": self.recent_sample_count,
            "window_sample_count": self.window_sample_count,
            "distinct_sample_count": self.distinct_sample_count,
            "stale_sample_count": self.stale_sample_count,
            "future_sample_count": self.future_sample_count,
            "duplicate_sample_count": self.duplicate_sample_count,
            "first_timestamp_sec": self.first_timestamp_sec,
            "last_timestamp_sec": self.last_timestamp_sec,
            "sample_window_sec": self.sample_window_sec,
            "minimum_front_range_m": self.minimum_front_range_m,
            "median_front_range_m": self.median_front_range_m,
            "median_front_bearing_rad": self.median_front_bearing_rad,
            "map_pose_translation_spread_m": (
                self.map_pose_translation_spread_m
            ),
            "map_pose_yaw_spread_rad": self.map_pose_yaw_spread_rad,
            "odom_pose_translation_spread_m": (
                self.odom_pose_translation_spread_m
            ),
            "odom_pose_yaw_spread_rad": self.odom_pose_yaw_spread_rad,
            "map_odom_offset_spread_m": self.map_odom_offset_spread_m,
            "map_odom_yaw_offset_spread_rad": (
                self.map_odom_yaw_offset_spread_rad
            ),
        }


def _angular_distance(first_rad: float, second_rad: float) -> float:
    return abs(
        math.atan2(
            math.sin(first_rad - second_rad),
            math.cos(first_rad - second_rad),
        )
    )


def _translation_spread(points: Sequence[tuple[float, float]]) -> float:
    return max(
        (
            math.hypot(first[0] - second[0], first[1] - second[1])
            for index, first in enumerate(points)
            for second in points[index + 1 :]
        ),
        default=0.0,
    )


def _yaw_spread(values_rad: Sequence[float]) -> float:
    return max(
        (
            _angular_distance(first, second)
            for index, first in enumerate(values_rad)
            for second in values_rad[index + 1 :]
        ),
        default=0.0,
    )


def _map_hit(sample: StationaryFrontSectorSample) -> tuple[float, float]:
    ray_yaw = sample.map_pose.yaw_rad + sample.front_bearing_rad
    return (
        sample.map_pose.x_m + sample.front_range_m * math.cos(ray_yaw),
        sample.map_pose.y_m + sample.front_range_m * math.sin(ray_yaw),
    )


@dataclass(frozen=True)
class _StationarySelection:
    samples: tuple[StationaryFrontSectorSample, ...]
    reasons: tuple[str, ...]
    supplied_sample_count: int
    recent_sample_count: int
    window_sample_count: int
    stale_sample_count: int
    future_sample_count: int
    duplicate_sample_count: int


@dataclass(frozen=True)
class _StationarityMetrics:
    map_pose_translation_spread_m: float
    map_pose_yaw_spread_rad: float
    odom_pose_translation_spread_m: float
    odom_pose_yaw_spread_rad: float
    map_odom_offset_spread_m: float
    map_odom_yaw_offset_spread_rad: float


def _stationarity_metrics(
    samples: Sequence[StationaryFrontSectorSample],
) -> _StationarityMetrics:
    map_points = tuple(
        (float(sample.map_pose.x_m), float(sample.map_pose.y_m))
        for sample in samples
    )
    odom_points = tuple(
        (float(sample.odom_pose.x_m), float(sample.odom_pose.y_m))
        for sample in samples
    )
    map_yaws = tuple(float(sample.map_pose.yaw_rad) for sample in samples)
    odom_yaws = tuple(float(sample.odom_pose.yaw_rad) for sample in samples)
    offset_points = tuple(
        (
            float(sample.map_pose.x_m - sample.odom_pose.x_m),
            float(sample.map_pose.y_m - sample.odom_pose.y_m),
        )
        for sample in samples
    )
    yaw_offsets = tuple(
        math.atan2(
            math.sin(sample.map_pose.yaw_rad - sample.odom_pose.yaw_rad),
            math.cos(sample.map_pose.yaw_rad - sample.odom_pose.yaw_rad),
        )
        for sample in samples
    )
    return _StationarityMetrics(
        map_pose_translation_spread_m=_translation_spread(map_points),
        map_pose_yaw_spread_rad=_yaw_spread(map_yaws),
        odom_pose_translation_spread_m=_translation_spread(odom_points),
        odom_pose_yaw_spread_rad=_yaw_spread(odom_yaws),
        map_odom_offset_spread_m=_translation_spread(offset_points),
        map_odom_yaw_offset_spread_rad=_yaw_spread(yaw_offsets),
    )


def _stationarity_failure_reasons(
    metrics: _StationarityMetrics,
    config: PersistentObstacleConfig,
) -> list[str]:
    reasons: list[str] = []
    if (
        metrics.map_pose_translation_spread_m
        > config.max_map_pose_translation_spread_m
    ):
        reasons.append("map_pose_not_stationary")
    if metrics.map_pose_yaw_spread_rad > config.max_map_pose_yaw_spread_rad:
        reasons.append("map_yaw_not_stationary")
    if (
        metrics.odom_pose_translation_spread_m
        > config.max_odom_pose_translation_spread_m
    ):
        reasons.append("odom_pose_not_stationary")
    if metrics.odom_pose_yaw_spread_rad > config.max_odom_pose_yaw_spread_rad:
        reasons.append("odom_yaw_not_stationary")
    if metrics.map_odom_offset_spread_m > config.max_map_odom_offset_spread_m:
        reasons.append("map_odom_localization_divergence")
    if (
        metrics.map_odom_yaw_offset_spread_rad
        > config.max_map_odom_yaw_offset_spread_rad
    ):
        reasons.append("map_odom_yaw_divergence")
    return reasons


def _select_stationary_samples(
    samples: Iterable[StationaryFrontSectorSample],
    *,
    now_sec: float,
    config: PersistentObstacleConfig,
) -> _StationarySelection:
    supplied = tuple(samples)
    if not all(
        isinstance(sample, StationaryFrontSectorSample) for sample in supplied
    ):
        raise ValueError("samples must contain StationaryFrontSectorSample values")
    if not supplied:
        return _StationarySelection(
            samples=(),
            reasons=("no_stationary_front_samples",),
            supplied_sample_count=0,
            recent_sample_count=0,
            window_sample_count=0,
            stale_sample_count=0,
            future_sample_count=0,
            duplicate_sample_count=0,
        )

    future = tuple(
        sample
        for sample in supplied
        if sample.timestamp_sec
        > now_sec + config.future_timestamp_tolerance_sec
    )
    nonfuture = tuple(
        sample
        for sample in supplied
        if sample.timestamp_sec
        <= now_sec + config.future_timestamp_tolerance_sec
    )
    recent = tuple(
        sample
        for sample in nonfuture
        if now_sec - sample.timestamp_sec <= config.max_sample_age_sec
    )
    stale_count = len(nonfuture) - len(recent)
    if future:
        return _StationarySelection(
            samples=(),
            reasons=("future_dated_stationary_sample",),
            supplied_sample_count=len(supplied),
            recent_sample_count=len(recent),
            window_sample_count=0,
            stale_sample_count=stale_count,
            future_sample_count=len(future),
            duplicate_sample_count=0,
        )
    if not recent:
        return _StationarySelection(
            samples=(),
            reasons=("no_recent_stationary_samples",),
            supplied_sample_count=len(supplied),
            recent_sample_count=0,
            window_sample_count=0,
            stale_sample_count=stale_count,
            future_sample_count=0,
            duplicate_sample_count=0,
        )

    ordered_recent = tuple(sorted(recent, key=lambda sample: sample.timestamp_sec))
    latest_timestamp = ordered_recent[-1].timestamp_sec
    windowed = tuple(
        sample
        for sample in ordered_recent
        if latest_timestamp - sample.timestamp_sec <= config.max_sample_window_sec
    )
    distinct: list[StationaryFrontSectorSample] = []
    for sample in windowed:
        if (
            not distinct
            or sample.timestamp_sec - distinct[-1].timestamp_sec
            >= config.min_sample_separation_sec
        ):
            distinct.append(sample)
    duplicate_count = len(windowed) - len(distinct)
    reasons: list[str] = []
    if (
        len(windowed) < config.min_distinct_samples
        and len(recent) >= config.min_distinct_samples
    ):
        reasons.append("samples_exceed_max_sample_window")
    if len(distinct) < config.min_distinct_samples:
        reasons.append("insufficient_distinct_recent_samples")
    return _StationarySelection(
        samples=tuple(distinct),
        reasons=tuple(reasons),
        supplied_sample_count=len(supplied),
        recent_sample_count=len(recent),
        window_sample_count=len(windowed),
        stale_sample_count=stale_count,
        future_sample_count=0,
        duplicate_sample_count=duplicate_count,
    )


def _empty_confirmation(
    *,
    config: PersistentObstacleConfig,
    selection: _StationarySelection,
) -> PersistentObstacleDecision:
    return PersistentObstacleDecision(
        confirmed=False,
        fail_closed=True,
        reasons=selection.reasons,
        config=config,
        supplied_sample_count=selection.supplied_sample_count,
        recent_sample_count=selection.recent_sample_count,
        window_sample_count=selection.window_sample_count,
        distinct_sample_count=len(selection.samples),
        stale_sample_count=selection.stale_sample_count,
        future_sample_count=selection.future_sample_count,
        duplicate_sample_count=selection.duplicate_sample_count,
        first_timestamp_sec=None,
        last_timestamp_sec=None,
        sample_window_sec=None,
        median_front_range_m=None,
        median_front_bearing_rad=None,
        front_range_spread_m=None,
        median_map_hit_x_m=None,
        median_map_hit_y_m=None,
        map_hit_spread_m=None,
        map_pose_translation_spread_m=None,
        map_pose_yaw_spread_rad=None,
        odom_pose_translation_spread_m=None,
        odom_pose_yaw_spread_rad=None,
        map_odom_offset_spread_m=None,
        map_odom_yaw_offset_spread_rad=None,
    )


def confirm_persistent_obstacle(
    samples: Iterable[StationaryFrontSectorSample],
    *,
    now_sec: float,
    config: PersistentObstacleConfig = PersistentObstacleConfig(),
) -> PersistentObstacleDecision:
    """Confirm only coherent returns from a bounded stationary sample window."""

    now = _finite_nonnegative(now_sec, "now_sec")
    if not isinstance(config, PersistentObstacleConfig):
        raise ValueError("config must be a PersistentObstacleConfig")
    selection = _select_stationary_samples(samples, now_sec=now, config=config)
    if selection.reasons:
        return _empty_confirmation(config=config, selection=selection)

    selected = selection.samples
    ranges = tuple(float(sample.front_range_m) for sample in selected)
    stationarity = _stationarity_metrics(selected)
    hits = tuple(_map_hit(sample) for sample in selected)
    median_hit_x = float(median(point[0] for point in hits))
    median_hit_y = float(median(point[1] for point in hits))

    front_range_spread = max(ranges) - min(ranges)
    map_hit_spread = max(
        math.hypot(x_m - median_hit_x, y_m - median_hit_y)
        for x_m, y_m in hits
    )
    coherence_reasons = _stationarity_failure_reasons(stationarity, config)
    if any(
        sample.front_range_m < config.min_front_range_m
        or sample.front_range_m > config.max_front_range_m
        for sample in selected
    ):
        coherence_reasons.append("front_range_outside_confirmation_bounds")
    if any(
        _angular_distance(sample.front_bearing_rad, 0.0)
        > config.front_sector_half_width_rad
        for sample in selected
    ):
        coherence_reasons.append("bearing_outside_front_sector")
    if front_range_spread > config.max_front_range_spread_m:
        coherence_reasons.append("front_range_spread_exceeded")
    if map_hit_spread > config.max_map_hit_spread_m:
        coherence_reasons.append("map_hit_cluster_spread_exceeded")
    confirmed = not coherence_reasons
    reasons = (
        ("coherent_persistent_obstacle_confirmed",)
        if confirmed
        else tuple(coherence_reasons)
    )
    first_timestamp = float(selected[0].timestamp_sec)
    last_timestamp = float(selected[-1].timestamp_sec)
    return PersistentObstacleDecision(
        confirmed=confirmed,
        fail_closed=not confirmed,
        reasons=reasons,
        config=config,
        supplied_sample_count=selection.supplied_sample_count,
        recent_sample_count=selection.recent_sample_count,
        window_sample_count=selection.window_sample_count,
        distinct_sample_count=len(selected),
        stale_sample_count=selection.stale_sample_count,
        future_sample_count=selection.future_sample_count,
        duplicate_sample_count=selection.duplicate_sample_count,
        first_timestamp_sec=first_timestamp,
        last_timestamp_sec=last_timestamp,
        sample_window_sec=last_timestamp - first_timestamp,
        median_front_range_m=float(median(ranges)),
        median_front_bearing_rad=float(
            median(sample.front_bearing_rad for sample in selected)
        ),
        front_range_spread_m=front_range_spread,
        median_map_hit_x_m=median_hit_x,
        median_map_hit_y_m=median_hit_y,
        map_hit_spread_m=map_hit_spread,
        map_pose_translation_spread_m=(
            stationarity.map_pose_translation_spread_m
        ),
        map_pose_yaw_spread_rad=stationarity.map_pose_yaw_spread_rad,
        odom_pose_translation_spread_m=(
            stationarity.odom_pose_translation_spread_m
        ),
        odom_pose_yaw_spread_rad=stationarity.odom_pose_yaw_spread_rad,
        map_odom_offset_spread_m=stationarity.map_odom_offset_spread_m,
        map_odom_yaw_offset_spread_rad=(
            stationarity.map_odom_yaw_offset_spread_rad
        ),
    )


def _empty_clearance_decision(
    *,
    config: PersistentObstacleConfig,
    clearance_threshold_m: float,
    selection: _StationarySelection,
) -> StationaryClearanceDecision:
    return StationaryClearanceDecision(
        confirmed=False,
        fail_closed=True,
        reasons=selection.reasons,
        config=config,
        clearance_threshold_m=clearance_threshold_m,
        supplied_sample_count=selection.supplied_sample_count,
        recent_sample_count=selection.recent_sample_count,
        window_sample_count=selection.window_sample_count,
        distinct_sample_count=len(selection.samples),
        stale_sample_count=selection.stale_sample_count,
        future_sample_count=selection.future_sample_count,
        duplicate_sample_count=selection.duplicate_sample_count,
        first_timestamp_sec=None,
        last_timestamp_sec=None,
        sample_window_sec=None,
        minimum_front_range_m=None,
        median_front_range_m=None,
        median_front_bearing_rad=None,
        map_pose_translation_spread_m=None,
        map_pose_yaw_spread_rad=None,
        odom_pose_translation_spread_m=None,
        odom_pose_yaw_spread_rad=None,
        map_odom_offset_spread_m=None,
        map_odom_yaw_offset_spread_rad=None,
    )


def confirm_stationary_clearance(
    samples: Iterable[StationaryFrontSectorSample],
    *,
    now_sec: float,
    clearance_threshold_m: float,
    config: PersistentObstacleConfig = PersistentObstacleConfig(),
) -> StationaryClearanceDecision:
    """Confirm clearance only from fresh, distinct, stationary front samples.

    Every selected range and their median must be strictly greater than the
    supplied threshold.  Obstacle-hit clustering is intentionally not reused:
    coherent close hits prove an obstacle, while clearance rays may validly
    terminate on different distant surfaces.
    """

    now = _finite_nonnegative(now_sec, "now_sec")
    threshold = _finite_positive(
        clearance_threshold_m, "clearance_threshold_m"
    )
    if not isinstance(config, PersistentObstacleConfig):
        raise ValueError("config must be a PersistentObstacleConfig")
    selection = _select_stationary_samples(samples, now_sec=now, config=config)
    if selection.reasons:
        return _empty_clearance_decision(
            config=config,
            clearance_threshold_m=threshold,
            selection=selection,
        )

    selected = selection.samples
    stationarity = _stationarity_metrics(selected)
    reasons = _stationarity_failure_reasons(stationarity, config)
    if any(
        sample.front_range_m < config.min_front_range_m for sample in selected
    ):
        reasons.append("front_range_below_validation_bound")
    if any(
        _angular_distance(sample.front_bearing_rad, 0.0)
        > config.front_sector_half_width_rad
        for sample in selected
    ):
        reasons.append("bearing_outside_front_sector")

    ranges = tuple(float(sample.front_range_m) for sample in selected)
    minimum_range = min(ranges)
    median_range = float(median(ranges))
    if minimum_range <= threshold:
        reasons.append("front_range_not_above_clearance_threshold")
    if median_range <= threshold:
        reasons.append("median_front_range_not_above_clearance_threshold")

    confirmed = not reasons
    decision_reasons = (
        ("stationary_front_clearance_confirmed",)
        if confirmed
        else tuple(reasons)
    )
    first_timestamp = float(selected[0].timestamp_sec)
    last_timestamp = float(selected[-1].timestamp_sec)
    return StationaryClearanceDecision(
        confirmed=confirmed,
        fail_closed=not confirmed,
        reasons=decision_reasons,
        config=config,
        clearance_threshold_m=threshold,
        supplied_sample_count=selection.supplied_sample_count,
        recent_sample_count=selection.recent_sample_count,
        window_sample_count=selection.window_sample_count,
        distinct_sample_count=len(selected),
        stale_sample_count=selection.stale_sample_count,
        future_sample_count=selection.future_sample_count,
        duplicate_sample_count=selection.duplicate_sample_count,
        first_timestamp_sec=first_timestamp,
        last_timestamp_sec=last_timestamp,
        sample_window_sec=last_timestamp - first_timestamp,
        minimum_front_range_m=minimum_range,
        median_front_range_m=median_range,
        median_front_bearing_rad=float(
            median(sample.front_bearing_rad for sample in selected)
        ),
        map_pose_translation_spread_m=(
            stationarity.map_pose_translation_spread_m
        ),
        map_pose_yaw_spread_rad=stationarity.map_pose_yaw_spread_rad,
        odom_pose_translation_spread_m=(
            stationarity.odom_pose_translation_spread_m
        ),
        odom_pose_yaw_spread_rad=stationarity.odom_pose_yaw_spread_rad,
        map_odom_offset_spread_m=stationarity.map_odom_offset_spread_m,
        map_odom_yaw_offset_spread_rad=(
            stationarity.map_odom_yaw_offset_spread_rad
        ),
    )


def reachable_distance_progress_epsilon(
    requested_epsilon_m: float,
    *,
    remaining_distance_m: float,
    waypoint_tolerance_m: float,
    expected_effective_travel_m: float,
) -> float:
    """Clamp a distance-progress test to progress that was physically reachable."""

    requested = _finite_nonnegative(requested_epsilon_m, "requested_epsilon_m")
    remaining = _finite_nonnegative(remaining_distance_m, "remaining_distance_m")
    tolerance = _finite_nonnegative(waypoint_tolerance_m, "waypoint_tolerance_m")
    expected_travel = _finite_nonnegative(
        expected_effective_travel_m, "expected_effective_travel_m"
    )
    reachable_before_waypoint = max(0.0, remaining - tolerance)
    return min(requested, reachable_before_waypoint, expected_travel)
