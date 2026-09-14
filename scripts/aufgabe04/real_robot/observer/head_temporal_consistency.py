"""Current-pixel head-border and axial-angle consistency while stopped.

This does not smooth corners, choose a historical pose, or create an angle
measurement. The caller supplies already fresh, uniquely associated current
raw-head pixels and owns the stopped motion epoch. Ambiguous current poses may
contribute veto evidence; admission of an actual axis still requires the
caller's independent pose-quality checks. All supplied choices,
including incompatible ones, remain in the bounded window until age/capacity
eviction. A rejected choice therefore cannot be dropped around six old samples
to fabricate a seven-sample result. Reported intervals are diagnostic observed
hulls, not confidence bounds or motion/planning authority.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from itertools import combinations
import math


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _axis(value):
    return (value + math.pi / 2) % math.pi - math.pi / 2


@dataclass(frozen=True)
class HeadTemporalContext:
    target_key: str
    model_sha256: str
    camera_signature: tuple
    motion_epoch: int

    def __post_init__(self):
        if (not isinstance(self.target_key, str) or not self.target_key
                or not isinstance(self.model_sha256, str) or not self.model_sha256
                or not isinstance(self.camera_signature, tuple) or not self.camera_signature
                or len(self.camera_signature) > 32
                or type(self.motion_epoch) is not int or self.motion_epoch < 0):
            raise ValueError("head temporal context requires target, model, camera and stopped epoch")
        for value in self.camera_signature:
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError("camera signature must contain immutable scalar values")
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError("camera signature must be finite")


@dataclass(frozen=True)
class HeadAxisInterval:
    """Smallest enclosing arc on an undirected, modulo-pi axis circle."""
    center_rad: float
    half_width_rad: float
    lower_rad: float
    upper_rad: float
    crosses_axis_wrap: bool

    @property
    def span_rad(self):
        return 2 * self.half_width_rad

    def metadata(self):
        return {**asdict(self), "span_rad": self.span_rad,
                "semantics": "observed_and_supplied_plausible_axes_not_confidence_bound",
                "planning_authorized": False, "motion_authorized": False}


def _interval(yaws):
    values = sorted(set(value % math.pi for value in yaws))
    if not values:
        return None
    gap, index = max(
        ((values[(i + 1) % len(values)] + (math.pi if i == len(values) - 1 else 0)
          - value, i) for i, value in enumerate(values)),
    )
    span = max(0., math.pi - gap)
    start = values[(index + 1) % len(values)]
    center = _axis(start + span / 2)
    lower, upper = _axis(center - span / 2), _axis(center + span / 2)
    return HeadAxisInterval(center, span / 2, lower, upper, lower > upper)


@dataclass(frozen=True)
class HeadTemporalDecision:
    current_sample_accepted: bool
    ready: bool
    reset_axis_evidence: bool
    reason: str
    sample_count: int
    required_samples: int
    window_stamps_sec: tuple[float, ...]
    angle_interval: HeadAxisInterval | None
    chosen_angle_interval: HeadAxisInterval | None
    corner_displacement_ratio: float
    max_axis_span_rad: float
    max_corner_displacement_ratio: float
    context_reset: bool
    expired_sample_count: int
    instability_count: int

    def metadata(self):
        payload = asdict(self)
        payload["angle_interval"] = None if self.angle_interval is None else self.angle_interval.metadata()
        payload["chosen_angle_interval"] = (
            None if self.chosen_angle_interval is None else self.chosen_angle_interval.metadata()
        )
        return {**payload, "schema_version": 1,
                "normalization": "full_image_corners_relative_to_nominal_projection_in_head_heights",
                "outlier_policy": "retain_all_choices_until_window_age_or_capacity_eviction",
                "acceptance_semantics": "temporal_compatibility_only_independent_pose_quality_required",
                "measurement_reused": False, "motion_authorized": False,
                "completion_authorized": False, "planning_authorized": False}


@dataclass(frozen=True)
class _Measurement:
    stamp_sec: float
    yaw_rad: float
    plausible_yaws_rad: tuple[float, ...]
    normalized_corners: tuple[tuple[float, float], ...]


class StationaryHeadConsistency:
    """Additional veto over current measurements; never substitutes a pose.

    Before any instability, compatible current samples may enter ordinary
    evidence while this window fills. An incompatible window withholds every
    new axis and requests clearing earlier axis evidence. Recovery enables
    only the new current sample; it does not backfill retained measurements
    into the independent seven-fresh-sample route admission contract.
    """

    def __init__(self, *, required_samples=7, max_axis_span_rad=math.radians(8.),
                 max_corner_displacement_ratio=.04, window_ttl_sec=5.):
        if type(required_samples) is not int or not 2 <= required_samples <= 32:
            raise ValueError("required_samples must be an integer in [2, 32]")
        for name, value, maximum in (
            ("max_axis_span_rad", max_axis_span_rad, math.pi / 2),
            ("max_corner_displacement_ratio", max_corner_displacement_ratio, 1.),
            ("window_ttl_sec", window_ttl_sec, 60.),
        ):
            if not 0 < _finite(value, name) <= maximum:
                raise ValueError(f"{name} is outside its bounded positive range")
        self.required_samples = required_samples
        self.max_axis_span_rad = float(max_axis_span_rad)
        self.max_corner_displacement_ratio = float(max_corner_displacement_ratio)
        self.window_ttl_sec = float(window_ttl_sec)
        self._window = deque(maxlen=required_samples)
        self._context = None
        self._last_stamp = -math.inf
        self._instability_count = 0

    def reset(self):
        """Explicit context reset; the caller must also clear prior axis evidence."""
        self._window.clear()
        self._context = None
        self._last_stamp = -math.inf
        self._instability_count = 0

    def _decision(self, *, accepted, reason, context_reset=False, expired=0, reset=False):
        window = tuple(self._window)
        chosen = _interval(item.yaw_rad for item in window)
        interval = _interval(yaw for item in window for yaw in item.plausible_yaws_rad)
        corner_jump = max(
            (math.dist(a, b) for first, second in combinations(window, 2)
             for a, b in zip(first.normalized_corners, second.normalized_corners)),
            default=0.,
        )
        return HeadTemporalDecision(
            current_sample_accepted=accepted,
            ready=accepted and len(window) >= self.required_samples,
            reset_axis_evidence=reset or context_reset, reason=reason,
            sample_count=len(window), required_samples=self.required_samples,
            window_stamps_sec=tuple(item.stamp_sec for item in window),
            angle_interval=interval, chosen_angle_interval=chosen,
            corner_displacement_ratio=corner_jump,
            max_axis_span_rad=self.max_axis_span_rad,
            max_corner_displacement_ratio=self.max_corner_displacement_ratio,
            context_reset=context_reset, expired_sample_count=expired,
            instability_count=self._instability_count,
        )

    def observe(self, *, context: HeadTemporalContext, stamp_sec: float, yaw_rad: float,
                corners_full_image, projected_center_px, expected_head_height_px: float,
                plausible_yaws_rad=()):
        """Review one fresh, associated current raw-head choice at its original stamp.

        Corners must be TL/TR/BR/BL full-image pixels, independent of crop
        offsets. Use the original nominal candidate projection, never the
        recentered proposal as its own normalization reference. Optional
        plausible axes must be supported by this same frame; no hypotheses
        are selected or excluded here. Temporal acceptance does not establish
        pose quality or resolve ambiguity. Malformed input clears local
        history and raises; the caller must also clear prior axis evidence.
        """
        try:
            if not isinstance(context, HeadTemporalContext):
                raise ValueError("context must be HeadTemporalContext")
            stamp = _finite(stamp_sec, "stamp_sec")
            yaw = _finite(yaw_rad, "yaw_rad")
            height = _finite(expected_head_height_px, "expected_head_height_px")
            if stamp < 0 or height <= 0:
                raise ValueError("stamp must be nonnegative and projected head height positive")
            if len(projected_center_px) != 2 or len(corners_full_image) != 4:
                raise ValueError("one projected center and four current corners are required")
            cx, cy = (_finite(value, "projected_center_px") for value in projected_center_px)
            normalized = []
            for point in corners_full_image:
                if len(point) != 2:
                    raise ValueError("corners must contain full-image (u, v) pairs")
                u, v = (_finite(value, "corner") for value in point)
                point_normalized = (_finite((u - cx) / height, "normalized corner"),
                                    _finite((v - cy) / height, "normalized corner"))
                # Bound malformed numeric input far outside any image geometry
                # so pairwise distances and their serialized diagnostics stay finite.
                if any(abs(value) > 1e6 for value in point_normalized):
                    raise ValueError("normalized corner exceeds numeric safety bound")
                normalized.append(point_normalized)
            if len(plausible_yaws_rad) > 8:
                raise ValueError("at most eight current plausible axes may be supplied")
            plausible = tuple(_axis(_finite(value, "plausible yaw")) for value in plausible_yaws_rad)
        except (TypeError, ValueError, OverflowError):
            self.reset()
            raise ValueError("invalid current head temporal measurement") from None
        context_reset = self._context != context
        if context_reset:
            self.reset()
            self._context = context
        if stamp <= self._last_stamp:
            return self._decision(accepted=False, reason="duplicate_or_out_of_order_head_measurement")
        self._last_stamp = stamp
        expired = 0
        while self._window and self._window[0].stamp_sec < stamp - self.window_ttl_sec:
            self._window.popleft()
            expired += 1
        self._window.append(_Measurement(stamp, _axis(yaw), (_axis(yaw), *plausible), tuple(normalized)))
        decision = self._decision(accepted=True, reason="current_head_window_consistent",
                                  context_reset=context_reset, expired=expired)
        inconsistent_axis = decision.angle_interval.span_rad > self.max_axis_span_rad + 1e-12
        inconsistent_border = decision.corner_displacement_ratio > self.max_corner_displacement_ratio + 1e-12
        if inconsistent_axis or inconsistent_border:
            self._instability_count += 1
            reason = ("head_axis_and_border_choice_unstable" if inconsistent_axis and inconsistent_border
                      else "head_axis_choice_unstable" if inconsistent_axis
                      else "head_border_choice_unstable")
            return self._decision(accepted=False, reason=reason, context_reset=context_reset,
                                  expired=expired, reset=True)
        return decision
