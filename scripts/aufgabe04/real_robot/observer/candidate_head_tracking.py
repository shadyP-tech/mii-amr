"""Candidate-scoped complete-head search information, never cached evidence.

A fresh associated head may locate the next image's pixels regardless of QR or
face classification. Source-time expiry and the stopped anchor bound this hint;
the next image must independently fit geometry and pass current association.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.stand_axis.head_border_seed import validate_current_head_proposal
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.observation_freshness import observation_freshness
from scripts.aufgabe04.real_robot.configuration.geometry import ImageRoi
from scripts.aufgabe04.real_robot.observer.camera_target_registration import HeadRoiEvaluation
from scripts.aufgabe04.real_robot.observer.current_head_detection import current_head_search_pose
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import (
    HeadRoiAttempt, validate_backside_registration_center_offset_ratio,
)


@dataclass(frozen=True)
class CandidateHeadContext:
    target_key: str
    model_sha256: str
    camera_signature: tuple[object, ...]
    image_shape: tuple[int, ...]
    stationary_epoch: int


@dataclass(frozen=True)
class CandidateHeadSearch:
    attempt: HeadRoiAttempt
    pose_hint: object  # Camera coordinates, independent of crop principal point.
    full_image_corners: tuple[ImagePoint, ...]


@dataclass(frozen=True)
class _Hint:
    context: CandidateHeadContext
    observed_at_sec: float
    anchor_pose: Pose2D
    pose: object
    corners: tuple[ImagePoint, ...]


class CandidateHeadTracking:
    """Retain one verified head for bounded search across front/back images.

    Seed freshness is the caller's existing image admission limit. The separate
    two-second search lifetime permits expensive cold acquisition to bootstrap
    tracking without increasing the age allowed for an accepted current angle.
    """

    def __init__(self, *, ttl_sec: float = 2.0, max_translation_m: float = .01,
                 max_rotation_rad: float = math.radians(2.0), max_soft_misses: int = 2) -> None:
        if not math.isfinite(ttl_sec) or not 0 < ttl_sec <= 2.0:
            raise ValueError("candidate head search TTL must be within (0, 2] seconds")
        if any(not math.isfinite(v) or v < 0 for v in (max_translation_m, max_rotation_rad)):
            raise ValueError("candidate head motion limits must be finite and nonnegative")
        if type(max_soft_misses) is not int or not 0 <= max_soft_misses <= 3:
            raise ValueError("candidate head search permits at most three soft misses")
        self.ttl_sec = ttl_sec
        self.max_translation_m = max_translation_m
        self.max_rotation_rad = max_rotation_rad
        self.max_soft_misses = max_soft_misses
        self._soft_misses = 0
        self._last_miss_stamp = -math.inf
        self._hint: _Hint | None = None
        self.last_metadata: dict[str, object] = {}

    def reset(self, reason: str = "candidate_head_search_reset") -> None:
        self._hint = None
        self._soft_misses = 0
        self._last_miss_stamp = -math.inf
        self.last_metadata = {"reason": reason, "hint_retained": False,
                              "measurement_reused": False, "motion_authorized": False}

    def _moved(self, anchor: Pose2D, pose: Pose2D) -> bool:
        return (math.hypot(pose.x_m - anchor.x_m, pose.y_m - anchor.y_m) > self.max_translation_m
                or abs((pose.yaw_rad - anchor.yaw_rad + math.pi) % (2 * math.pi) - math.pi)
                > self.max_rotation_rad)

    @staticmethod
    def _finite_pose(pose: Pose2D) -> bool:
        return all(math.isfinite(v) for v in (pose.x_m, pose.y_m, pose.yaw_rad))

    def _retain_search_after_miss(self, reason, *, context, observed_at_sec, now_sec, robot_pose):
        """A missed current fit may keep a locator, never renew its evidence.

        The caller still receives False and must reject the current frame.
        Repeated misses return to cold acquisition, and the original source
        stamp and stopped anchor bound the retained locator throughout.
        """
        old = self._hint
        if (reason not in {"candidate_head_seed_unassociated", "candidate_head_seed_geometry_unverified"}
                or old is None or old.context != context
                or self._soft_misses >= self.max_soft_misses
                or not all(math.isfinite(v) for v in (observed_at_sec, now_sec))
                or observed_at_sec <= max(old.observed_at_sec, self._last_miss_stamp)
                or not 0 < now_sec - old.observed_at_sec <= self.ttl_sec
                or not self._finite_pose(robot_pose) or self._moved(old.anchor_pose, robot_pose)):
            return False
        self._soft_misses += 1
        self._last_miss_stamp = observed_at_sec
        self.last_metadata = {
            "reason": "candidate_head_search_retained_after_miss", "current_failure": reason,
            "hint_retained": True, "current_measurement_accepted": False,
            "consecutive_soft_misses": self._soft_misses, "max_soft_misses": self.max_soft_misses,
            "source_stamp_sec": old.observed_at_sec, "age_sec": now_sec - old.observed_at_sec,
            "ttl_sec": self.ttl_sec, "source_stamp_refreshed": False,
            "measurement_reused": False, "motion_authorized": False,
        }
        return True

    def remember(self, evaluation: HeadRoiEvaluation, *, context: CandidateHeadContext,
                 observed_at_sec: float, now_sec: float, max_age_sec: float,
                 robot_pose: Pose2D, candidate_associated: bool) -> bool:
        """Remember current verified pixels only after unique target association.

        QR presence, decoded identity and visible-face labels do not participate.
        A small number of fresh misses may retain the previous search locator;
        they cannot refresh its timestamp, supply a measurement or admit an
        unassociated candidate. Context changes and motion clear it immediately.
        """
        freshness = observation_freshness(observed_at_sec=observed_at_sec,
                                         now_sec=now_sec, max_age_sec=max_age_sec)
        estimate, debug = evaluation.estimate, evaluation.debug
        search_pose = current_head_search_pose(estimate, debug, profile_sha256=context.model_sha256)
        old = self._hint
        reason = None
        if not freshness.accepted:
            reason = "candidate_head_seed_stale"
        elif not self._finite_pose(robot_pose):
            reason = "candidate_head_seed_pose_invalid"
        elif (estimate.model_profile_sha256 != context.model_sha256
              or debug.model_profile_sha256 != context.model_sha256
              or (debug.head_model_quality is not None
                  and debug.head_model_quality.profile_sha256 != context.model_sha256)):
            reason = "candidate_head_seed_model_mismatch"
        elif old is not None and old.context == context:
            if observed_at_sec <= max(old.observed_at_sec, self._last_miss_stamp):
                reason = "candidate_head_seed_nonadvancing_image"
            elif self._moved(old.anchor_pose, robot_pose):
                reason = "candidate_head_anchor_moved"
        if reason is None and not candidate_associated:
            reason = "candidate_head_seed_unassociated"
        if reason is None and search_pose is None:
            reason = "candidate_head_seed_geometry_unverified"
        if reason is not None:
            if not self._retain_search_after_miss(reason, context=context,
                    observed_at_sec=observed_at_sec, now_sec=now_sec, robot_pose=robot_pose):
                self.reset(reason)
            return False
        roi = evaluation.attempt.roi
        try:
            height, width = context.image_shape[:2]
            if not (0 <= roi.x0 < roi.x1 <= width and 0 <= roi.y0 < roi.y1 <= height):
                raise ValueError("invalid full-image crop")
            corners = validate_current_head_proposal(
                estimate.corners, frame_shape=(roi.y1 - roi.y0, roi.x1 - roi.x0))
            # Complete borders need actual surrounding pixels, including when
            # the original nominal crop happened to touch the physical head.
            if not all(2 <= p.u_px <= roi.x1 - roi.x0 - 2
                       and 2 <= p.v_px <= roi.y1 - roi.y0 - 2 for p in corners):
                raise ValueError("head is clipped")
            full = tuple(ImagePoint(p.u_px + roi.x0, p.v_px + roi.y0) for p in corners)
            pose = search_pose
            if (not pose.positive_depth or not all(math.isfinite(v) for v in
                    (*pose.rotation_vector, *pose.translation_xyz_m, *pose.face_normal_xyz))):
                raise ValueError("invalid camera pose")
        except (ValueError, TypeError, AttributeError):
            self.reset("candidate_head_seed_bounds_or_pose_invalid")
            return False
        anchor = old.anchor_pose if old is not None and old.context == context else robot_pose
        self._hint = _Hint(context, observed_at_sec, anchor, pose, full)
        self._soft_misses = 0
        self._last_miss_stamp = -math.inf
        self.last_metadata = {"reason": "current_associated_head_retained", "hint_retained": True,
                              "seed_age_sec": freshness.age_sec, "ttl_sec": self.ttl_sec,
                              "source_stamp_sec": observed_at_sec, "measurement_reused": False,
                              "motion_authorized": False}
        return True

    def hint(self, roi_attempts: tuple[HeadRoiAttempt, ...], *, context: CandidateHeadContext,
             observed_at_sec: float, robot_pose: Pose2D,
             max_center_offset_ratio: float = 1.5) -> CandidateHeadSearch | None:
        """Return a bounded crop and camera pose while preserving original bounds.

        The attempt's expected center and scale remain the map projection, not
        the saved fitted center. Full-camera intrinsics must be adjusted once by
        the returned ROI origin when the caller processes the current crop.
        """
        limit = validate_backside_registration_center_offset_ratio(max_center_offset_ratio)
        old = self._hint
        reason = None
        if old is None:
            reason = "no_candidate_head_hint"
        elif old.context != context:
            reason = "candidate_head_context_changed"
        elif not math.isfinite(observed_at_sec) or not 0 < observed_at_sec - old.observed_at_sec <= self.ttl_sec:
            reason = "candidate_head_hint_expired_or_nonadvancing_image"
        elif not self._finite_pose(robot_pose) or self._moved(old.anchor_pose, robot_pose):
            reason = "candidate_head_anchor_moved"
        elif not roi_attempts:
            reason = "candidate_head_search_roi_unavailable"
        if reason is not None:
            self.reset(reason)
            return None
        nominal, allowed = roi_attempts[0], roi_attempts[-1].roi
        assert old is not None
        us, vs = [p.u_px for p in old.corners], [p.v_px for p in old.corners]
        height = max(vs) - min(vs)
        center_offset = math.hypot(sum(us) / 4 - nominal.expected_center_u_px,
                                   sum(vs) / 4 - nominal.expected_center_v_px)
        if (not math.isfinite(nominal.expected_head_height_px) or nominal.expected_head_height_px <= 0
                or center_offset > limit * nominal.expected_head_height_px):
            self.reset("candidate_head_hint_outside_current_projection_bound")
            return None
        margin = max(8., .25 * height)
        crop = ImageRoi(max(allowed.x0, math.floor(min(us) - margin)),
                        max(allowed.y0, math.floor(min(vs) - margin)),
                        min(allowed.x1, math.ceil(max(us) + margin)),
                        min(allowed.y1, math.ceil(max(vs) + margin)),
                        nominal.roi.expected_size_px)
        if not all(crop.x0 + 2 <= u <= crop.x1 - 2 and crop.y0 + 2 <= v <= crop.y1 - 2
                   for u, v in zip(us, vs)):
            self.reset("candidate_head_hint_crop_clipped")
            return None
        self.last_metadata = {"reason": "candidate_head_search_hint", "hint_retained": True,
                              "source_stamp_sec": old.observed_at_sec,
                              "age_sec": observed_at_sec - old.observed_at_sec,
                              "ttl_sec": self.ttl_sec,
                              "crop_xyxy": (crop.x0, crop.y0, crop.x1, crop.y1),
                              "consecutive_soft_misses": self._soft_misses,
                              "current_measurement_accepted": False,
                              "measurement_reused": False, "motion_authorized": False}
        return CandidateHeadSearch(replace(nominal, roi=crop, source="candidate_tracked_head_search"),
                                   old.pose, old.corners)
