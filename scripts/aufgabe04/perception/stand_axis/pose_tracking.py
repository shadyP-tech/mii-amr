"""Short-lived pose prediction state with explicit invalidation keys."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.aufgabe04.perception.stand_axis.models import (
        StandAxisEdgeDebugArtifacts,
        StandAxisImageEstimate,
    )

from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import (
    PlanarPoseHypothesis,
)
from scripts.aufgabe04.perception.stand_axis.observation_freshness import (
    observation_freshness,
)


@dataclass(frozen=True)
class PosePrediction:
    state: str
    pose: PlanarPoseHypothesis | None
    age_sec: float | None
    reason: str


@dataclass(frozen=True)
class PoseTrackerUpdate:
    accepted: bool
    reason: str
    observation_age_sec: float | None


class MetricPoseTracker:
    """Retain a pose only as a bounded search prior, never as measurement."""

    def __init__(self, *, prediction_ttl_sec: float = 0.25) -> None:
        if not math.isfinite(prediction_ttl_sec) or prediction_ttl_sec <= 0.0:
            raise ValueError("prediction_ttl_sec must be finite and positive")
        self.prediction_ttl_sec = float(prediction_ttl_sec)
        self._pose: PlanarPoseHypothesis | None = None
        self._accepted_at_sec: float | None = None
        self._profile_sha256: str | None = None
        self._camera_signature: tuple[float, float, float, float] | None = None

    def reset(self) -> None:
        self._pose = None
        self._accepted_at_sec = None
        self._profile_sha256 = None
        self._camera_signature = None

    def accept(
        self,
        pose: PlanarPoseHypothesis,
        *,
        now_sec: float,
        profile_sha256: str,
        camera_signature: tuple[float, float, float, float],
    ) -> None:
        if not math.isfinite(now_sec):
            raise ValueError("now_sec must be finite")
        self._pose = pose
        self._accepted_at_sec = float(now_sec)
        self._profile_sha256 = profile_sha256
        self._camera_signature = tuple(float(value) for value in camera_signature)

    def update_from_observation(
        self,
        estimate: StandAxisImageEstimate | None,
        artifacts: StandAxisEdgeDebugArtifacts | None,
        *,
        observed_at_sec: float | None,
        completed_at_sec: float,
        profile_sha256: str,
        camera_signature: tuple[float, float, float, float],
        result_fresh: bool = True,
    ) -> PoseTrackerUpdate:
        """Retain only verified, fresh poses, dated at image observation time.

        A QR seed from a rejected fit is useful for that frame's search only.
        Rejection leaves the previous verified pose and its expiry unchanged.
        """

        freshness = observation_freshness(
            observed_at_sec=observed_at_sec,
            now_sec=completed_at_sec,
            max_age_sec=self.prediction_ttl_sec,
        )
        if not result_fresh or not freshness.accepted:
            return PoseTrackerUpdate(False, "pose_observation_stale", freshness.age_sec)
        if (
            estimate is None
            or artifacts is None
            or not estimate.usable
            or estimate.evidence_state != "fresh_refined"
            or artifacts.evidence_state != "fresh_refined"
            or artifacts.model_pose is None
        ):
            return PoseTrackerUpdate(False, "pose_not_verified", freshness.age_sec)
        if (
            estimate.model_profile_sha256 != profile_sha256
            or artifacts.model_profile_sha256 != profile_sha256
        ):
            return PoseTrackerUpdate(False, "pose_profile_mismatch", freshness.age_sec)
        if self._accepted_at_sec is not None and observed_at_sec <= self._accepted_at_sec:
            return PoseTrackerUpdate(False, "pose_observation_not_newer", freshness.age_sec)
        self.accept(
            artifacts.model_pose,
            now_sec=observed_at_sec,
            profile_sha256=profile_sha256,
            camera_signature=camera_signature,
        )
        return PoseTrackerUpdate(True, "verified_pose_observed", freshness.age_sec)

    def prediction(
        self,
        *,
        now_sec: float,
        profile_sha256: str,
        camera_signature: tuple[float, float, float, float],
        invalidated: bool = False,
    ) -> PosePrediction:
        if invalidated:
            self.reset()
            return PosePrediction("unavailable", None, None, "tracker_invalidated")
        if self._pose is None or self._accepted_at_sec is None:
            return PosePrediction("unavailable", None, None, "no_tracked_pose")
        if (
            profile_sha256 != self._profile_sha256
            or tuple(float(value) for value in camera_signature)
            != self._camera_signature
        ):
            self.reset()
            return PosePrediction("unavailable", None, None, "tracking_context_changed")
        age = float(now_sec) - self._accepted_at_sec
        if not math.isfinite(age) or age < 0.0 or age > self.prediction_ttl_sec:
            self.reset()
            return PosePrediction("stale", None, age, "tracked_pose_expired")
        return PosePrediction("predicted_only", self._pose, age, "bounded_pose_prediction")
