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

    def __init__(self, *, prediction_ttl_sec: float = 0.25,
                 search_hint_ttl_sec: float | None = None, max_soft_misses: int = 2) -> None:
        if not math.isfinite(prediction_ttl_sec) or prediction_ttl_sec <= 0.0:
            raise ValueError("prediction_ttl_sec must be finite and positive")
        self.prediction_ttl_sec = float(prediction_ttl_sec)
        search_ttl = prediction_ttl_sec if search_hint_ttl_sec is None else search_hint_ttl_sec
        if not math.isfinite(search_ttl) or not prediction_ttl_sec <= search_ttl <= 2.0:
            raise ValueError("search hint TTL must cover freshness and be at most two seconds")
        if type(max_soft_misses) is not int or not 0 <= max_soft_misses <= 3:
            raise ValueError("search hints permit at most three soft misses")
        self.search_hint_ttl_sec = float(search_ttl)
        self.max_soft_misses = max_soft_misses
        self._soft_misses = 0
        self._last_miss_at_sec: float | None = None
        self._pose: PlanarPoseHypothesis | None = None
        self._accepted_at_sec: float | None = None
        self._profile_sha256: str | None = None
        self._camera_signature: tuple[float, float, float, float] | None = None

    def reset(self) -> None:
        self._soft_misses = 0
        self._last_miss_at_sec = None
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
        self._soft_misses = 0
        self._last_miss_at_sec = None
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
        Brief rejection retains the previous verified search hint without
        renewing its source time. Exhausted misses or context changes clear it.
        """

        signature = tuple(float(value) for value in camera_signature)
        if (self._pose is not None and (profile_sha256 != self._profile_sha256
                or signature != self._camera_signature)):
            self.reset()
        freshness = observation_freshness(
            observed_at_sec=observed_at_sec,
            now_sec=completed_at_sec,
            max_age_sec=self.prediction_ttl_sec,
        )
        if not result_fresh or not freshness.accepted:
            self._record_miss(observed_at_sec, completed_at_sec)
            return PoseTrackerUpdate(False, "pose_observation_stale", freshness.age_sec)
        if (
            estimate is None
            or artifacts is None
            or not estimate.usable
            or estimate.evidence_state != "fresh_refined"
            or artifacts.evidence_state != "fresh_refined"
            or artifacts.model_pose is None
        ):
            self._record_miss(observed_at_sec, completed_at_sec)
            return PoseTrackerUpdate(False, "pose_not_verified", freshness.age_sec)
        if (
            estimate.model_profile_sha256 != profile_sha256
            or artifacts.model_profile_sha256 != profile_sha256
        ):
            self.reset()
            return PoseTrackerUpdate(False, "pose_profile_mismatch", freshness.age_sec)
        previous = max(self._accepted_at_sec if self._accepted_at_sec is not None else -math.inf,
                       self._last_miss_at_sec if self._last_miss_at_sec is not None else -math.inf)
        if observed_at_sec <= previous:
            return PoseTrackerUpdate(False, "pose_observation_not_newer", freshness.age_sec)
        self.accept(
            artifacts.model_pose,
            now_sec=observed_at_sec,
            profile_sha256=profile_sha256,
            camera_signature=camera_signature,
        )
        return PoseTrackerUpdate(True, "verified_pose_observed", freshness.age_sec)

    def _record_miss(self, observed_at_sec: float | None, now_sec: float) -> None:
        """Bound retained search information without renewing its source time."""
        if self._pose is None or self._accepted_at_sec is None:
            return
        if (observed_at_sec is None or not math.isfinite(observed_at_sec)
                or not math.isfinite(now_sec) or now_sec < observed_at_sec
                or now_sec - self._accepted_at_sec > self.search_hint_ttl_sec):
            self.reset()
            return
        previous = self._accepted_at_sec if self._last_miss_at_sec is None else self._last_miss_at_sec
        if observed_at_sec <= previous:
            return
        self._soft_misses += 1
        self._last_miss_at_sec = observed_at_sec
        if self._soft_misses > self.max_soft_misses:
            self.reset()

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
        if not math.isfinite(age) or age < 0.0 or age > self.search_hint_ttl_sec:
            self.reset()
            return PosePrediction("stale", None, age, "tracked_pose_expired")
        return PosePrediction("predicted_only", self._pose, age, "bounded_pose_prediction")
