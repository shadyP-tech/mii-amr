"""Separate repeated backside appearance from orientation readiness.

Only ordinary fresh, stationary, associated observer updates count. Appearance
cannot supply an angle, and a successful angle cannot manufacture appearance.
The accumulator is bounded by the existing observation count and TTL.
"""

from dataclasses import dataclass
import math

from scripts.aufgabe04.perception.stand_axis.head_backside_appearance import HeadBacksideAppearance


@dataclass(frozen=True)
class HeadConfidenceInput:
    frame_stamp_sec: float
    appearance: HeadBacksideAppearance | None
    complete_head_verified: bool
    marker_observed_now: bool
    marker_seen_in_epoch: bool
    angle_observable_now: bool
    angle_reason: str


class HeadObservationConfidence:
    def __init__(self, *, required_samples: int, ttl_sec: float):
        if type(required_samples) is not int or not 2 <= required_samples <= 256:
            raise ValueError("head confidence requires a bounded repeated sample count")
        if not math.isfinite(ttl_sec) or ttl_sec <= 0:
            raise ValueError("head confidence TTL must be finite and positive")
        self.required_samples, self.ttl_sec = required_samples, ttl_sec
        self._context = None
        self._samples = {}
        self._latest_stamp = -math.inf
        self._front_veto = False

    def observe(self, current: HeadConfidenceInput, *, update, observed_at_sec: float,
                angle_temporally_consistent: bool = False):
        """Consume the result of the ordinary evidence gate, never replace it."""
        snapshot = update.snapshot
        context = (snapshot.target_key, snapshot.motion_epoch)
        if context != self._context:
            self._context = context
            self._samples.clear()
            self._latest_stamp = -math.inf
            self._front_veto = False
        stamp = current.frame_stamp_sec
        finite_time = all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                          for v in (stamp, observed_at_sec))
        if finite_time:
            cutoff = max(stamp, observed_at_sec, self._latest_stamp) - self.ttl_sec
            self._samples = {s: confidence for s, confidence in self._samples.items() if s >= cutoff}
        if current.marker_observed_now or current.marker_seen_in_epoch or snapshot.poisoned:
            self._front_veto = True
            self._samples.clear()
        appearance = current.appearance
        qualifies = bool(
            finite_time and stamp > self._latest_stamp and not self._front_veto
            and update.frame_accepted and not snapshot.poisoned
            and current.complete_head_verified is True
            and isinstance(appearance, HeadBacksideAppearance)
            and appearance.accepted is True
            and appearance.supplies_angle is False and appearance.motion_authorized is False
            and type(appearance.confidence) in (int, float)
            and math.isfinite(appearance.confidence) and 0 <= appearance.confidence <= 1)
        if finite_time:
            self._latest_stamp = max(self._latest_stamp, stamp)
        if qualifies:
            self._samples[stamp] = appearance.confidence
            self._samples = dict(sorted(self._samples.items())[-self.required_samples:])
        supported = len(self._samples) >= self.required_samples and not self._front_veto
        return {
            "backside": {
                "state": ("front_marker_veto" if self._front_veto else
                          "backside_supported" if supported else
                          "collecting_backside_appearance" if self._samples else "unresolved"),
                "current_sample_accepted": qualifies,
                "sample_count": len(self._samples), "required_samples": self.required_samples,
                "confidence": min(self._samples.values()) if self._samples else None,
                "complete_head_verified_now": current.complete_head_verified,
                "appearance_reason": None if appearance is None else appearance.reason,
                "neck_required": False,
            },
            "angle": {
                "observable_now": current.angle_observable_now,
                "reason": current.angle_reason,
                "temporally_consistent": angle_temporally_consistent,
                "consensus_ready": update.axis_consensus is not None,
            },
            "target_key": snapshot.target_key, "motion_epoch": snapshot.motion_epoch,
            "motion_authorized": False,
            "opposite_side_requires": "fresh_validated_axis_receipt_and_admitted_route",
        }
