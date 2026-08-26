"""Bounded temporal evidence for one passive stand-observer target.

The ROS observer intentionally keeps sensor transport, image processing, and
motion outside this module.  This class only answers a narrower question: may
axis and QR evidence from several *stationary, LiDAR-associated* sensor tuples
be combined?  Short-lived perception misses do not erase good evidence, while
target changes, motion, stale/mismatched LiDAR, duplicate frames, and ambiguous
QR identities remain fail closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Iterable

from scripts.aufgabe04.perception.stand_axis_consensus import AxisConsensus


@dataclass(frozen=True)
class EvidencePose:
    """Planar pose used only to bind evidence to one stationary epoch."""

    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(frozen=True)
class PassiveObserverEvidenceSnapshot:
    """Current and historical counters suitable for observer audit artifacts."""

    schema_version: int
    target_key: str
    motion_epoch: int
    anchor_pose: EvidencePose
    poisoned: bool
    poison_reason: str | None
    required_axis_sample_count: int
    current_axis_sample_count: int
    peak_axis_sample_count: int
    current_axis_sample_count_by_source: dict[str, int]
    peak_axis_sample_count_by_source: dict[str, int]
    current_qr_sample_count: int
    peak_qr_sample_count: int
    tentative_qr_id: str | None
    latched_qr_id: str | None
    axis_ttl_sec: float
    qr_ttl_sec: float
    accepted_frame_count: int
    duplicate_frame_count: int
    lidar_rejection_count: int
    soft_miss_count: int
    motion_reset_count: int
    last_soft_miss_reason: str | None

    def as_dict(self) -> dict[str, object]:
        """Return a detached, JSON-compatible representation."""

        return asdict(self)


@dataclass(frozen=True)
class PassiveObserverEvidenceUpdate:
    """Result of adding one associated frame or one soft miss."""

    frame_accepted: bool
    axis_sample_accepted: bool
    qr_sample_accepted: bool
    reason: str
    axis_consensus: AxisConsensus | None
    resolved_qr_id: str | None
    axis_only: bool
    motion_epoch_reset: bool
    snapshot: PassiveObserverEvidenceSnapshot


class PassiveObserverEvidence:
    """Combine bounded evidence for one fixed candidate and motion epoch.

    Axis samples are never mixed across estimator sources.  QR identity is an
    independent channel and needs two distinct associated image frames.  A
    single QR read is therefore tentative and cannot turn an axis consensus
    into a QR-bound recommendation.
    """

    def __init__(
        self,
        *,
        target_key: str,
        anchor_pose: EvidencePose,
        required_axis_samples: int,
        max_axis_deviation_rad: float,
        axis_ttl_sec: float = 5.0,
        qr_ttl_sec: float = 5.0,
        required_qr_samples: int = 2,
        max_lidar_age_sec: float = 0.5,
        max_sensor_skew_sec: float = 0.10,
        max_future_stamp_sec: float = 0.05,
        max_anchor_translation_m: float = 0.02,
        max_anchor_rotation_rad: float = math.radians(3.0),
    ) -> None:
        if not target_key:
            raise ValueError("target_key must be non-empty")
        if required_axis_samples < 2:
            raise ValueError("required_axis_samples must be at least two")
        if required_qr_samples < 2:
            raise ValueError("required_qr_samples must be at least two")
        self._require_positive("max_axis_deviation_rad", max_axis_deviation_rad)
        self._require_positive("axis_ttl_sec", axis_ttl_sec)
        self._require_positive("qr_ttl_sec", qr_ttl_sec)
        self._require_positive("max_lidar_age_sec", max_lidar_age_sec)
        self._require_nonnegative("max_sensor_skew_sec", max_sensor_skew_sec)
        self._require_nonnegative("max_future_stamp_sec", max_future_stamp_sec)
        self._require_positive(
            "max_anchor_translation_m", max_anchor_translation_m
        )
        self._require_positive("max_anchor_rotation_rad", max_anchor_rotation_rad)
        self._validate_pose(anchor_pose)

        self.target_key = target_key
        self.required_axis_samples = required_axis_samples
        self.required_qr_samples = required_qr_samples
        self.max_axis_deviation_rad = float(max_axis_deviation_rad)
        self.axis_ttl_sec = float(axis_ttl_sec)
        self.qr_ttl_sec = float(qr_ttl_sec)
        self.max_lidar_age_sec = float(max_lidar_age_sec)
        self.max_sensor_skew_sec = float(max_sensor_skew_sec)
        self.max_future_stamp_sec = float(max_future_stamp_sec)
        self.max_anchor_translation_m = float(max_anchor_translation_m)
        self.max_anchor_rotation_rad = float(max_anchor_rotation_rad)

        self._anchor_pose = anchor_pose
        self._motion_epoch = 0
        self._axis_samples: dict[str, dict[float, float]] = {}
        self._qr_samples: dict[float, str] = {}
        self._seen_frame_stamps: set[float] = set()
        self._latest_reference_stamp = -math.inf
        self._poison_reason: str | None = None

        self._peak_axis_sample_count = 0
        self._peak_axis_sample_count_by_source: dict[str, int] = {}
        self._peak_qr_sample_count = 0
        self._accepted_frame_count = 0
        self._duplicate_frame_count = 0
        self._lidar_rejection_count = 0
        self._soft_miss_count = 0
        self._motion_reset_count = 0
        self._last_soft_miss_reason: str | None = None

    @staticmethod
    def _require_positive(name: str, value: float) -> None:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    @staticmethod
    def _require_nonnegative(name: str, value: float) -> None:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    @staticmethod
    def _validate_pose(pose: EvidencePose) -> None:
        if not all(math.isfinite(value) for value in asdict(pose).values()):
            raise ValueError("evidence pose must be finite")

    @staticmethod
    def _validate_stamp(name: str, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return value

    def _validate_target(self, target_key: str) -> None:
        if target_key != self.target_key:
            raise ValueError(
                f"evidence target is fixed to {self.target_key!r}, got {target_key!r}"
            )

    def _reset_current_epoch(self, pose: EvidencePose) -> None:
        self._anchor_pose = pose
        self._motion_epoch += 1
        self._motion_reset_count += 1
        self._axis_samples.clear()
        self._qr_samples.clear()
        self._seen_frame_stamps.clear()
        self._latest_reference_stamp = -math.inf
        self._poison_reason = None
        self._last_soft_miss_reason = None

    def _ensure_motion_epoch(self, pose: EvidencePose) -> bool:
        self._validate_pose(pose)
        translation = math.hypot(
            pose.x_m - self._anchor_pose.x_m,
            pose.y_m - self._anchor_pose.y_m,
        )
        rotation = abs(
            math.atan2(
                math.sin(pose.yaw_rad - self._anchor_pose.yaw_rad),
                math.cos(pose.yaw_rad - self._anchor_pose.yaw_rad),
            )
        )
        if (
            translation <= self.max_anchor_translation_m
            and rotation <= self.max_anchor_rotation_rad
        ):
            return False
        self._reset_current_epoch(pose)
        return True

    def _prune(self, reference_stamp: float) -> None:
        self._latest_reference_stamp = max(
            self._latest_reference_stamp, reference_stamp
        )
        axis_cutoff = self._latest_reference_stamp - self.axis_ttl_sec
        qr_cutoff = self._latest_reference_stamp - self.qr_ttl_sec
        for source in tuple(self._axis_samples):
            samples = self._axis_samples[source]
            self._axis_samples[source] = {
                stamp: yaw
                for stamp, yaw in samples.items()
                if stamp >= axis_cutoff
            }
            if not self._axis_samples[source]:
                del self._axis_samples[source]
        self._qr_samples = {
            stamp: qr_id
            for stamp, qr_id in self._qr_samples.items()
            if stamp >= qr_cutoff
        }

        # Duplicate suppression only needs to cover frames that could still
        # contribute to either bounded evidence window.
        seen_cutoff = self._latest_reference_stamp - max(
            self.axis_ttl_sec, self.qr_ttl_sec
        )
        self._seen_frame_stamps = {
            stamp for stamp in self._seen_frame_stamps if stamp >= seen_cutoff
        }

    def _remember_frame(self, frame_stamp: float) -> bool:
        if frame_stamp in self._seen_frame_stamps:
            return False
        self._seen_frame_stamps.add(frame_stamp)
        return True

    def _record_axis(self, *, source: str, stamp: float, yaw_rad: float) -> bool:
        if not source or not math.isfinite(yaw_rad):
            return False
        samples = self._axis_samples.setdefault(source, {})
        samples[stamp] = float(yaw_rad)
        newest = sorted(samples, reverse=True)[: self.required_axis_samples]
        self._axis_samples[source] = {
            sample_stamp: samples[sample_stamp] for sample_stamp in newest
        }
        count = len(self._axis_samples[source])
        self._peak_axis_sample_count_by_source[source] = max(
            self._peak_axis_sample_count_by_source.get(source, 0), count
        )
        self._peak_axis_sample_count = max(self._peak_axis_sample_count, count)
        return True

    def _record_qr(self, *, stamp: float, qr_texts: Iterable[str]) -> tuple[bool, str | None]:
        unique = tuple(sorted({str(text).strip() for text in qr_texts if str(text).strip()}))
        if len(unique) > 1:
            self._poison_reason = "multiple_qr_ids_in_associated_frame"
            self._axis_samples.clear()
            self._qr_samples.clear()
            return False, self._poison_reason
        if not unique:
            return False, None
        qr_id = unique[0]
        active_ids = set(self._qr_samples.values())
        if active_ids and active_ids != {qr_id}:
            self._poison_reason = "conflicting_qr_ids_in_motion_epoch"
            self._axis_samples.clear()
            self._qr_samples.clear()
            return False, self._poison_reason
        self._qr_samples[stamp] = qr_id
        newest = sorted(self._qr_samples, reverse=True)[: self.required_qr_samples]
        self._qr_samples = {
            sample_stamp: self._qr_samples[sample_stamp]
            for sample_stamp in newest
        }
        self._peak_qr_sample_count = max(
            self._peak_qr_sample_count, len(self._qr_samples)
        )
        return True, None

    def _axis_consensus(self) -> AxisConsensus | None:
        candidates: list[tuple[float, AxisConsensus]] = []
        for source, sample_map in self._axis_samples.items():
            if len(sample_map) < self.required_axis_samples:
                continue
            ordered = sorted(sample_map.items())[-self.required_axis_samples :]
            values = [yaw for _, yaw in ordered]
            sin_mean = sum(math.sin(value) for value in values) / len(values)
            cos_mean = sum(math.cos(value) for value in values) / len(values)
            mean = math.atan2(sin_mean, cos_mean)
            deviations = [
                abs(
                    math.atan2(
                        math.sin(value - mean),
                        math.cos(value - mean),
                    )
                )
                for value in values
            ]
            maximum = max(deviations)
            if maximum <= self.max_axis_deviation_rad:
                candidates.append(
                    (
                        ordered[-1][0],
                        AxisConsensus(mean, len(values), maximum, source),
                    )
                )
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item[0], item[1].source))[1]

    def _qr_state(self) -> tuple[str | None, str | None]:
        identities = set(self._qr_samples.values())
        if len(identities) != 1:
            return None, None
        qr_id = next(iter(identities))
        if len(self._qr_samples) >= self.required_qr_samples:
            return qr_id, None
        return None, qr_id

    def snapshot(self) -> PassiveObserverEvidenceSnapshot:
        current_by_source = {
            source: len(samples)
            for source, samples in sorted(self._axis_samples.items())
        }
        resolved_qr_id, tentative_qr_id = self._qr_state()
        return PassiveObserverEvidenceSnapshot(
            schema_version=1,
            target_key=self.target_key,
            motion_epoch=self._motion_epoch,
            anchor_pose=self._anchor_pose,
            poisoned=self._poison_reason is not None,
            poison_reason=self._poison_reason,
            required_axis_sample_count=self.required_axis_samples,
            current_axis_sample_count=max(current_by_source.values(), default=0),
            peak_axis_sample_count=self._peak_axis_sample_count,
            current_axis_sample_count_by_source=current_by_source,
            peak_axis_sample_count_by_source=dict(
                sorted(self._peak_axis_sample_count_by_source.items())
            ),
            current_qr_sample_count=len(self._qr_samples),
            peak_qr_sample_count=self._peak_qr_sample_count,
            tentative_qr_id=tentative_qr_id,
            latched_qr_id=resolved_qr_id,
            axis_ttl_sec=self.axis_ttl_sec,
            qr_ttl_sec=self.qr_ttl_sec,
            accepted_frame_count=self._accepted_frame_count,
            duplicate_frame_count=self._duplicate_frame_count,
            lidar_rejection_count=self._lidar_rejection_count,
            soft_miss_count=self._soft_miss_count,
            motion_reset_count=self._motion_reset_count,
            last_soft_miss_reason=self._last_soft_miss_reason,
        )

    def _update(
        self,
        *,
        frame_accepted: bool,
        axis_sample_accepted: bool,
        qr_sample_accepted: bool,
        reason: str,
        motion_epoch_reset: bool,
    ) -> PassiveObserverEvidenceUpdate:
        consensus = None if self._poison_reason else self._axis_consensus()
        resolved_qr_id, _ = self._qr_state()
        if self._poison_reason:
            resolved_qr_id = None
        return PassiveObserverEvidenceUpdate(
            frame_accepted=frame_accepted,
            axis_sample_accepted=axis_sample_accepted,
            qr_sample_accepted=qr_sample_accepted,
            reason=reason,
            axis_consensus=consensus,
            resolved_qr_id=resolved_qr_id,
            axis_only=consensus is not None and resolved_qr_id is None,
            motion_epoch_reset=motion_epoch_reset,
            snapshot=self.snapshot(),
        )

    def note_soft_miss(
        self,
        *,
        target_key: str,
        pose: EvidencePose,
        stamp_sec: float,
        reason: str,
    ) -> PassiveObserverEvidenceUpdate:
        """Advance TTLs without deleting still-fresh evidence."""

        self._validate_target(target_key)
        stamp = self._validate_stamp("stamp_sec", stamp_sec)
        reset = self._ensure_motion_epoch(pose)
        self._prune(stamp)
        self._soft_miss_count += 1
        self._last_soft_miss_reason = reason or "unspecified_soft_miss"
        return self._update(
            frame_accepted=False,
            axis_sample_accepted=False,
            qr_sample_accepted=False,
            reason=self._last_soft_miss_reason,
            motion_epoch_reset=reset,
        )

    def record_frame(
        self,
        *,
        target_key: str,
        pose: EvidencePose,
        frame_stamp_sec: float,
        lidar_stamp_sec: float,
        observed_at_sec: float,
        lidar_associated: bool,
        axis_yaw_rad: float | None,
        axis_source: str | None,
        qr_texts: Iterable[str] = (),
    ) -> PassiveObserverEvidenceUpdate:
        """Record one synchronized frame if its LiDAR association is fresh.

        QR and axis channels are evaluated independently after the common
        target, motion, tuple-synchrony, freshness, and duplicate-frame gates.
        """

        self._validate_target(target_key)
        frame_stamp = self._validate_stamp("frame_stamp_sec", frame_stamp_sec)
        lidar_stamp = self._validate_stamp("lidar_stamp_sec", lidar_stamp_sec)
        observed_at = self._validate_stamp("observed_at_sec", observed_at_sec)
        reset = self._ensure_motion_epoch(pose)
        self._prune(frame_stamp)

        if self._poison_reason is not None:
            return self._update(
                frame_accepted=False,
                axis_sample_accepted=False,
                qr_sample_accepted=False,
                reason=self._poison_reason,
                motion_epoch_reset=reset,
            )
        if frame_stamp < self._latest_reference_stamp - max(
            self.axis_ttl_sec, self.qr_ttl_sec
        ):
            self._soft_miss_count += 1
            self._last_soft_miss_reason = "frame_older_than_evidence_window"
            return self._update(
                frame_accepted=False,
                axis_sample_accepted=False,
                qr_sample_accepted=False,
                reason=self._last_soft_miss_reason,
                motion_epoch_reset=reset,
            )
        if frame_stamp in self._seen_frame_stamps:
            self._duplicate_frame_count += 1
            return self._update(
                frame_accepted=False,
                axis_sample_accepted=False,
                qr_sample_accepted=False,
                reason="duplicate_frame_stamp",
                motion_epoch_reset=reset,
            )

        lidar_age = observed_at - lidar_stamp
        lidar_reason = None
        if not lidar_associated:
            lidar_reason = "lidar_target_not_associated"
        elif abs(frame_stamp - lidar_stamp) > self.max_sensor_skew_sec:
            lidar_reason = "lidar_not_from_same_sensor_tuple"
        elif lidar_age > self.max_lidar_age_sec:
            lidar_reason = "lidar_tuple_stale"
        elif lidar_age < -self.max_future_stamp_sec:
            lidar_reason = "lidar_tuple_from_future"
        if lidar_reason is not None:
            self._lidar_rejection_count += 1
            self._soft_miss_count += 1
            self._last_soft_miss_reason = lidar_reason
            return self._update(
                frame_accepted=False,
                axis_sample_accepted=False,
                qr_sample_accepted=False,
                reason=lidar_reason,
                motion_epoch_reset=reset,
            )

        self._remember_frame(frame_stamp)
        self._accepted_frame_count += 1
        qr_accepted, poison_reason = self._record_qr(
            stamp=frame_stamp,
            qr_texts=qr_texts,
        )
        if poison_reason is not None:
            return self._update(
                frame_accepted=False,
                axis_sample_accepted=False,
                qr_sample_accepted=False,
                reason=poison_reason,
                motion_epoch_reset=reset,
            )

        axis_accepted = False
        axis_requested = axis_yaw_rad is not None or axis_source is not None
        if axis_yaw_rad is not None and axis_source is not None:
            axis_accepted = self._record_axis(
                source=axis_source,
                stamp=frame_stamp,
                yaw_rad=float(axis_yaw_rad),
            )
        reason = "associated_frame_recorded"
        if axis_requested and not axis_accepted:
            reason = "invalid_axis_sample"
        update = self._update(
            frame_accepted=True,
            axis_sample_accepted=axis_accepted,
            qr_sample_accepted=qr_accepted,
            reason=reason,
            motion_epoch_reset=reset,
        )
        if update.axis_consensus is not None:
            reason = (
                "axis_and_qr_consensus"
                if update.resolved_qr_id is not None
                else "axis_consensus_without_qr"
            )
            return replace(update, reason=reason)
        return update
