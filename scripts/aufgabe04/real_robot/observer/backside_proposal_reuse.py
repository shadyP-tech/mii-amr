"""Bounded backside search hints, never cached camera measurements.

A slow but geometrically usable result may locate the next image's search.
Every returned angle is fitted from that next image, with native current QR
checks and bounded full acquisition on the ROI, retaining the original
projection's registration bound.
Freshness, LiDAR association and consensus remain the observer's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.real_robot.observer.camera_target_registration import (
    BACKSIDE_REACQUISITION_MODE,
    QR_MODEL_REACQUISITION_MODE,
    CameraTargetRegistrationSelection,
    HeadRoiEvaluation,
    select_camera_target_measurement,
)
from scripts.aufgabe04.real_robot.observer.contract import BACKSIDE_AXIS_SAMPLE_SOURCE
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import (
    HeadRoiAttempt,
    REGISTERED_BACKSIDE_REACQUISITION_SOURCE,
    REGISTERED_QR_MODEL_REACQUISITION_SOURCE,
    TARGET_CENTERED_REACQUISITION_SOURCE,
    registered_head_roi_attempt,
    validate_backside_registration_center_offset_ratio,
)


@dataclass(frozen=True)
class BacksideProposalContext:
    target_key: str
    model_sha256: str
    camera_signature: tuple[object, ...]
    image_shape: tuple[int, ...]


@dataclass(frozen=True)
class _SearchHint:
    context: BacksideProposalContext
    observed_at_sec: float
    anchor_pose: Pose2D
    corners: tuple[ImagePoint, ...]  # Full-image pixels; no angle or pose seed.


def _backside_geometry(evaluation: HeadRoiEvaluation, context: BacksideProposalContext) -> bool:
    estimate = evaluation.estimate
    return bool(
        estimate.usable
        and estimate.evidence_state == "fresh_backside"
        and estimate.source == BACKSIDE_AXIS_SAMPLE_SOURCE
        and estimate.model_profile_sha256 == context.model_sha256
        and estimate.corners is not None
        and len(estimate.corners) == 4
        and not evaluation.debug.qr_detected
        and evaluation.qr_observations == ()
    )


class BacksideProposalReuse:
    """One candidate-scoped hint with sensor-time expiry and anchored motion."""

    def __init__(self, *, ttl_sec: float = 2.0,
                 max_translation_m: float = 0.01,
                 max_rotation_rad: float = math.radians(2.0)) -> None:
        if not math.isfinite(ttl_sec) or not 0.0 < ttl_sec <= 2.0:
            raise ValueError("backside proposal TTL must be within (0, 2] seconds")
        if any(not math.isfinite(v) or v < 0.0
               for v in (max_translation_m, max_rotation_rad)):
            raise ValueError("backside proposal motion limits must be finite and nonnegative")
        self.ttl_sec = ttl_sec
        self.max_translation_m = max_translation_m
        self.max_rotation_rad = max_rotation_rad
        self._hint: _SearchHint | None = None
        self.last_metadata: dict[str, object] = {}

    def reset(self) -> None:
        self._hint = None

    def select(
        self, roi_attempts: tuple[HeadRoiAttempt, ...], *,
        context: BacksideProposalContext, observed_at_sec: float,
        robot_pose: Pose2D, marker_seen_in_stationary_epoch: bool,
        tracked_pose: object | None,
        evaluate: Callable[[HeadRoiAttempt, object | None], HeadRoiEvaluation],
        enable_reacquisition: bool, max_center_offset_ratio: float,
        acquire_registered: Callable[[HeadRoiAttempt, HeadRoiEvaluation], CameraTargetRegistrationSelection | None] | None = None,
    ) -> CameraTargetRegistrationSelection:
        """Try one strict hinted fit, otherwise use ordinary acquisition.

        A failed hinted fit is returned as the current miss and clears the
        hint. The next image reacquires normally; expensive retries on the
        already processed image cannot starve newer sensor tuples.
        """
        max_center_offset_ratio = validate_backside_registration_center_offset_ratio(
            max_center_offset_ratio
        )
        if not all(math.isfinite(v) for v in (
            observed_at_sec, robot_pose.x_m, robot_pose.y_m, robot_pose.yaw_rad,
        )):
            self.reset()
            raise ValueError("backside search context requires finite time and pose")
        hint = self._hint
        reason = self._hint_rejection(
            hint, context, observed_at_sec, robot_pose, roi_attempts,
            marker_seen_in_stationary_epoch, tracked_pose, enable_reacquisition,
        )
        self.last_metadata = {
            "mode": "ordinary_acquisition", "hint_reason": reason,
            "hint_source_stamp_sec": None if hint is None else hint.observed_at_sec,
            "hint_age_sec": None if hint is None else observed_at_sec - hint.observed_at_sec,
            "ttl_sec": self.ttl_sec, "measurement_reused": False,
        }
        self.reset()
        if reason is None:
            assert hint is not None
            wide = roi_attempts[1]
            local_corners = tuple(ImagePoint(p.u_px - wide.roi.x0, p.v_px - wide.roi.y0)
                                  for p in hint.corners)
            search = registered_head_roi_attempt(
                wide, local_corners, max_center_offset_ratio=max_center_offset_ratio,
            )
            if search.accepted and search.attempt is not None:
                if acquire_registered is not None:
                    # The old image may choose a small first search, but a
                    # current backside still needs a newly located complete
                    # head and unique current scan association. A hint is
                    # never evidence that today's QR/head is inside its crop.
                    selection = select_camera_target_measurement(
                        (search.attempt, wide), tracked_pose=None,
                        evaluate=evaluate, enable_reacquisition=enable_reacquisition,
                        max_center_offset_ratio=max_center_offset_ratio,
                        acquire_registered=acquire_registered,
                    )
                    selection = replace(selection, search_hint_used=True)
                    self.last_metadata.update(mode="strict_current_image_hint",
                                              complete_head_reverified=selection.head_acquisition is not None)
                    self._remember(selection, context, observed_at_sec, hint.anchor_pose,
                                   allowed=not marker_seen_in_stationary_epoch)
                    return selection
                current = evaluate(search.attempt, None)
                # Recompute registration from CURRENT fitted corners. The old
                # image only picked a search centre and supplies no receipt data.
                source = (REGISTERED_QR_MODEL_REACQUISITION_SOURCE
                          if current.debug.qr_detected
                          else REGISTERED_BACKSIDE_REACQUISITION_SOURCE)
                decision = registered_head_roi_attempt(
                    wide, current.estimate.corners,
                    max_center_offset_ratio=max_center_offset_ratio,
                    registered_source=source,
                )
                if not decision.accepted:
                    current = replace(
                        current,
                        attempt=replace(current.attempt, source=TARGET_CENTERED_REACQUISITION_SOURCE),
                        estimate=replace(current.estimate, usable=False,
                                         evidence_state="unobservable",
                                         reason=("model_backside_target_center_mismatch"
                                                 if current.estimate.usable
                                                 else current.estimate.reason)),
                    )
                elif current.debug.qr_detected:
                    current = replace(current, attempt=replace(current.attempt, source=source))
                selection = CameraTargetRegistrationSelection(
                    selected=current, evaluations=(current,), proposal=None,
                    decision=decision if decision.accepted else None,
                    strict_retry=current if decision.accepted else None,
                    reacquisition_mode=(QR_MODEL_REACQUISITION_MODE
                                        if current.debug.qr_detected
                                        else BACKSIDE_REACQUISITION_MODE),
                    search_hint_used=True,
                )
                self.last_metadata.update(mode="strict_current_image_hint",
                                          current_registration=decision.metadata())
                self._remember(selection, context, observed_at_sec, hint.anchor_pose,
                               allowed=not marker_seen_in_stationary_epoch)
                return selection
            self.last_metadata["hint_reason"] = search.reason
        selection = select_camera_target_measurement(
            roi_attempts, tracked_pose=tracked_pose, evaluate=evaluate,
            enable_reacquisition=enable_reacquisition,
            max_center_offset_ratio=max_center_offset_ratio,
            acquire_registered=acquire_registered,
        )
        self._remember(selection, context, observed_at_sec, robot_pose,
                       allowed=enable_reacquisition and not marker_seen_in_stationary_epoch)
        return selection

    def _hint_rejection(self, hint, context, stamp, pose, attempts,
                        marker_seen, tracked_pose, enabled) -> str | None:
        if not enabled or marker_seen or tracked_pose is not None:
            return "backside_hint_ineligible"
        if hint is None:
            return "no_backside_hint"
        if hint.context != context:
            return "backside_hint_context_changed"
        if not 0.0 < stamp - hint.observed_at_sec <= self.ttl_sec:
            return "backside_hint_expired_or_nonadvancing_image"
        anchor = hint.anchor_pose
        if (math.hypot(pose.x_m - anchor.x_m, pose.y_m - anchor.y_m) > self.max_translation_m
                or abs((pose.yaw_rad - anchor.yaw_rad + math.pi) % (2 * math.pi) - math.pi)
                > self.max_rotation_rad):
            return "backside_hint_anchor_moved"
        if len(attempts) < 2:
            return "backside_hint_no_wide_roi"
        narrow, wide = attempts[0].roi, attempts[1].roi
        if not (wide.x0 <= narrow.x0 and wide.y0 <= narrow.y0
                and wide.x1 >= narrow.x1 and wide.y1 >= narrow.y1):
            return "backside_hint_would_skip_nominal_pixels"
        return None

    def _remember(self, selection, context, stamp, anchor_pose, *, allowed):
        current = selection.selected
        if (allowed and selection.registered and _backside_geometry(current, context)
                and all(not item.debug.qr_detected and not item.qr_observations
                        for item in selection.evaluations)):
            corners = tuple(ImagePoint(p.u_px + current.attempt.roi.x0,
                                       p.v_px + current.attempt.roi.y0)
                            for p in current.estimate.corners)
            if all(math.isfinite(v) for p in corners for v in (p.u_px, p.v_px)):
                self._hint = _SearchHint(context, stamp, anchor_pose, corners)
        self.last_metadata["hint_retained"] = self._hint is not None
