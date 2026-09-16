"""Retain every current-pixel orientation allowance in a stopped window.

This is not a more permissive single-angle consensus. No hypothesis is voted
away and uncertainty is never divided by the number of observations. The
result describes one interval for a downstream, whole-interval viewing check.
"""

from collections import deque
from dataclasses import dataclass
from itertools import combinations
import math

from scripts.aufgabe04.artifacts.bounded_orientation import MAXIMUM_ORIENTATION_HALF_WIDTH_RAD


def axial(value):
    return (value + math.pi / 2) % math.pi - math.pi / 2


def enclose_intervals(intervals):
    """Enclose complete axial arcs, not just their endpoints across a wrap.

    A common lift around the newest center is conservative. If the union
    cannot fit in a semicircle, return no useful orientation rather than
    accidentally selecting the complement of a large interval.
    """
    intervals = tuple(intervals)
    if not intervals:
        return None
    reference = intervals[-1][0]
    lower, upper = math.inf, -math.inf
    for center, half_width in intervals:
        if (any(type(v) not in (int, float) or not math.isfinite(v)
                for v in (center, half_width)) or not 0 <= half_width < math.pi / 2):
            return None
        lifted = reference + axial(center - reference)
        lower, upper = min(lower, lifted - half_width), max(upper, lifted + half_width)
    if upper - lower >= math.pi:
        return None
    return axial((lower + upper) / 2), (upper - lower) / 2


@dataclass(frozen=True)
class BoundedHeadSample:
    stamp_sec: float
    axis_center_rad: float
    half_width_rad: float
    model_sha256: str
    camera_signature: tuple
    face: str
    qr_id: str | None
    corners: tuple
    projected_center_px: tuple
    expected_head_height_px: float


class BoundedHeadWindow:
    def __init__(self, *, required_samples=7, ttl_sec=5.):
        if type(required_samples) is not int or not 7 <= required_samples <= 32:
            raise ValueError("bounded orientation requires 7..32 fresh samples")
        if type(ttl_sec) not in (int, float) or not math.isfinite(ttl_sec) or not 0 < ttl_sec <= 60:
            raise ValueError("bounded orientation TTL must be in (0,60]")
        self.required_samples, self.ttl_sec = required_samples, ttl_sec
        self.reset()

    def reset(self):
        self._context = None
        self._samples = deque(maxlen=self.required_samples)
        self._latest_stamp = -math.inf
        self.metadata = {"ready": False, "reason": "awaiting_current_head_bounds"}

    def observe(self, sample, *, update, observed_at_sec):
        """Caller must first run ordinary sensor, motion and identity gates."""
        snapshot = update.snapshot
        epoch = (snapshot.target_key, snapshot.motion_epoch)
        if self._context is not None and self._context[:2] != epoch:
            self.reset()
        if snapshot.poisoned:
            self.reset()
            self.metadata = {"ready": False, "reason": "poisoned_observation_epoch"}
            return None
        if type(observed_at_sec) not in (int, float) or not math.isfinite(observed_at_sec):
            self.reset()
            return None
        while self._samples and self._samples[0].stamp_sec < observed_at_sec - self.ttl_sec:
            self._samples.popleft()
        if sample is None or not update.frame_accepted:
            self.metadata = {"ready": False, "reason": "current_bounded_head_unavailable",
                             "sample_count": len(self._samples)}
            return None
        if (not isinstance(sample, BoundedHeadSample)
                or sample.face not in {"front", "backside"}
                or sample.face == "front" and (not sample.qr_id or not update.qr_sample_accepted)
                or sample.face == "backside" and (sample.qr_id is not None
                    or snapshot.tentative_qr_id is not None or snapshot.latched_qr_id is not None)
                or type(sample.stamp_sec) not in (int, float)
                or not math.isfinite(sample.stamp_sec) or sample.stamp_sec < 0
                or sample.stamp_sec > observed_at_sec + .05
                or observed_at_sec - sample.stamp_sec > self.ttl_sec
                or type(sample.expected_head_height_px) not in (int, float)
                or not math.isfinite(sample.expected_head_height_px) or sample.expected_head_height_px <= 0
                or len(sample.corners) != 4 or len(sample.projected_center_px) != 2
                or any(len(p) != 2 or any(type(v) not in (int, float) or not math.isfinite(v)
                       for v in p) for p in sample.corners)
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       for v in sample.projected_center_px)
                or enclose_intervals(((sample.axis_center_rad, sample.half_width_rad),)) is None):
            self.reset()
            return None
        context = (*epoch, sample.model_sha256, sample.camera_signature, sample.face, sample.qr_id)
        if self._context != context:
            self.reset()
            self._context = context
        if sample.stamp_sec <= self._latest_stamp:
            self.metadata = {"ready": False, "reason": "duplicate_or_out_of_order_bounds"}
            return None
        self._latest_stamp = sample.stamp_sec
        self._samples.append(sample)
        interval = enclose_intervals((s.axis_center_rad, s.half_width_rad) for s in self._samples)
        normalized = [tuple(((u - s.projected_center_px[0]) / s.expected_head_height_px,
                             (v - s.projected_center_px[1]) / s.expected_head_height_px)
                            for u, v in s.corners) for s in self._samples]
        corner_jump = max((math.dist(a, b) for first, second in combinations(normalized, 2)
                           for a, b in zip(first, second)), default=0.)
        # This only limits the artifact's representable range. The planner
        # must separately check its actual terminal pose and route.
        ready = (len(self._samples) >= self.required_samples and interval is not None
                 and interval[1] <= MAXIMUM_ORIENTATION_HALF_WIDTH_RAD and corner_jump <= .04)
        self.metadata = {
            "ready": ready, "reason": ("bounded_orientation_ready" if ready else
                "head_border_choice_unstable" if corner_jump > .04 else
                "orientation_range_requires_view_adjustment" if interval is None or interval[1] > MAXIMUM_ORIENTATION_HALF_WIDTH_RAD
                else "collecting_bounded_orientation"),
            "sample_count": len(self._samples), "required_samples": self.required_samples,
            "source_stamps_sec": [s.stamp_sec for s in self._samples],
            "center_rad": None if interval is None else interval[0],
            "half_width_rad": None if interval is None else interval[1],
            "corner_displacement_ratio": corner_jump,
            "max_corner_displacement_ratio": .04,
            "hypothesis_policy": "retain_every_interval_without_averaging_uncertainty",
            "motion_authorized": False,
        }
        if not ready:
            return None
        return {"policy": "current_head_noise_expanded_interval", "center_rad": interval[0],
                "half_width_rad": interval[1], "sample_count": len(self._samples)}
