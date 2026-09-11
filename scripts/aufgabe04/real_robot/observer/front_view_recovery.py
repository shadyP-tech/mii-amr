"""Bounded stopped reacquisition before abandoning a readable front view.

This policy delays only an advisory exit. It cannot latch QR identity, admit
an axis, publish motion, or extend the parent observer process deadline.
"""

from collections.abc import Mapping
import math
from scripts.aufgabe04.perception.stand_axis.head_model_quality import MEASURED_HEAD_AXIS_SOURCE


FRONT_VIEW_RECOVERY_SEC = 30.0
CORNER_REACQUISITION = "qr_corner_reacquisition"
GEOMETRY_REACQUISITION = "joint_geometry_reacquisition"
HEAD_GEOMETRY_REACQUISITION = "head_geometry_reacquisition"


def front_view_failure_kind(
    state: str, details: Mapping[str, object], *, require_candidate_association: bool = True,
) -> str | None:
    """Classify a current readable/verified marker, never a historical latch."""
    if state not in {"metric_model_measurement_unavailable", "evidence_not_committable",
                     "collecting_consensus", "axis_observation_not_committable"}:
        return None
    axis = details.get("stand_axis_debug") or {}
    if (state in {"collecting_consensus", "axis_observation_not_committable"}
            and axis.get("estimator_source") != MEASURED_HEAD_AXIS_SOURCE):
        return None
    model = axis.get("metric_model") or {}
    binding = axis.get("decoded_qr_target_binding") or {}
    registration = model.get("camera_target_registration") or {}
    acquisition = model.get("head_acquisition") or registration.get("head_acquisition") or {}
    bound_qr = binding.get("accepted") is True and binding.get("reason") == "decoded_qr_target_associated"
    associated_head = (registration.get("strict_retry_applied") is True
                       and acquisition.get("candidate_associated") is True)
    measured_head = axis.get("estimator_source") == MEASURED_HEAD_AXIS_SOURCE
    if measured_head and (axis.get("measured_head_lidar_admission") or {}).get("accepted") is True:
        associated_head = True
    # A return in the nominal projected LiDAR cone does not associate an
    # off-center front marker. Require that current marker's own ray or the
    # independently registered head proposal before retaining the view.
    if require_candidate_association and not bound_qr and not associated_head:
        return None
    texts = details.get("qr_texts") or ()
    decoded_front = (isinstance(texts, (list, tuple)) and len(texts) == 1
                     and isinstance(texts[0], str) and bool(texts[0].strip()))
    if not decoded_front and model.get("qr_marker_verified") is not True:
        return None
    if measured_head and axis.get("estimator_usable") is True and not bound_qr:
        # A good independent angle does not supply the QR's missing current
        # geometry/identity. Keep its first stopped acquisition deadline even
        # while the stronger angle channel is collecting valid samples.
        return CORNER_REACQUISITION
    if axis.get("estimator_usable") is True:
        return None
    if measured_head:
        return HEAD_GEOMETRY_REACQUISITION
    reason = details.get("estimator_reason") or axis.get("estimator_reason") or details.get("reason")
    if reason in {"model_qr_text_without_geometry", "model_pose_seed_unavailable"}:
        return CORNER_REACQUISITION
    if (reason == "model_head_qr_geometry_mismatch"
            or model.get("model_pose_fit_source") == "joint_qr_head"):
        return GEOMETRY_REACQUISITION
    return None


class FrontViewRecovery:
    """One non-renewable deadline for each target and stationary epoch."""

    def __init__(self, *, duration_sec: float = FRONT_VIEW_RECOVERY_SEC,
                 max_translation_m: float = .02,
                 max_rotation_rad: float = math.radians(2)):
        if (type(duration_sec) not in (int, float) or not math.isfinite(duration_sec)
                or not 0 < duration_sec <= FRONT_VIEW_RECOVERY_SEC):
            raise ValueError("front recovery duration must be in (0, 30] seconds")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
               for v in (max_translation_m, max_rotation_rad)):
            raise ValueError("front recovery stationary limits must be finite and positive")
        self.duration_sec = duration_sec
        self.max_translation_m = max_translation_m
        self.max_rotation_rad = max_rotation_rad
        self.reset()

    def reset(self):
        self._target_key = None
        self._anchor = None
        self._started = None
        self._deadline = None
        self._last_frame_stamp = None
        self._last_qualified_frame_stamp = None
        self._poisoned = False
        self._kind = None
        self._qualified_frames = 0
        self._deferred = False
        self._reason = "no_qualified_front_observation"

    def poison(self):
        """Conflict clears the active hold without creating a fresh budget."""
        self._poisoned = True
        self._deferred = False
        self._reason = "poisoned_observation_epoch"

    def observe(self, *, target_key: str, now_sec: float, frame_stamp_sec: float,
                robot_pose: Mapping[str, float], frame_accepted: bool,
                source_fresh: bool, poisoned: bool, motion_epoch_reset: bool,
                failure_kind: str | None) -> bool:
        """Return whether this fresh tuple should keep observing while stopped."""
        if (not isinstance(target_key, str) or not target_key
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                       for v in (now_sec, frame_stamp_sec))):
            raise ValueError("front recovery needs a target and finite timestamps")
        pose = tuple(robot_pose[key] for key in ("x_m", "y_m", "yaw_rad"))
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in pose):
            raise ValueError("front recovery needs a finite stationary pose")
        moved = self._anchor is not None and (
            math.hypot(pose[0] - self._anchor[0], pose[1] - self._anchor[1]) > self.max_translation_m
            or abs(math.remainder(pose[2] - self._anchor[2], 2 * math.pi)) > self.max_rotation_rad
        )
        if self._target_key != target_key or moved or motion_epoch_reset:
            self.reset()
            self._target_key, self._anchor = target_key, pose
        self._deferred = False
        if poisoned:
            self.poison()
        if self._poisoned:
            return False
        if frame_accepted is not True or source_fresh is not True:
            self._reason = "current_frame_not_fresh_and_associated"
            return False
        if self._last_frame_stamp is not None and frame_stamp_sec <= self._last_frame_stamp:
            self._reason = "repeated_or_out_of_order_frame"
            return False
        self._last_frame_stamp = frame_stamp_sec
        if failure_kind not in {CORNER_REACQUISITION, GEOMETRY_REACQUISITION, HEAD_GEOMETRY_REACQUISITION}:
            self._reason = "current_frame_does_not_need_front_recovery"
            return False
        self._kind = failure_kind
        self._last_qualified_frame_stamp = frame_stamp_sec
        self._qualified_frames += 1
        if self._started is None:
            self._started = now_sec
            self._deadline = now_sec + self.duration_sec
        # New QR samples, changing failure reasons and ordinary soft misses
        # cannot change either timestamp. The outer process keeps its own cap.
        self._deferred = now_sec < self._deadline
        self._reason = "stopped_front_reacquisition" if self._deferred else "front_recovery_budget_exhausted"
        return self._deferred

    def metadata(self, *, now_sec: float) -> dict[str, object]:
        remaining = None if self._deadline is None else max(0., self._deadline - now_sec)
        return {
            "schema_version": 1, "target_key": self._target_key,
            "phase": self._kind, "reason": self._reason,
            "duration_sec": self.duration_sec,
            "started_monotonic_sec": self._started, "deadline_monotonic_sec": self._deadline,
            "remaining_sec": remaining,
            "qualified_frame_count": self._qualified_frames,
            "last_qualified_source_stamp_sec": self._last_qualified_frame_stamp,
            "current_advisory_deferred": self._deferred and remaining is not None and remaining > 0,
            "budget_exhausted": remaining == 0., "poisoned": self._poisoned,
            "extends_parent_deadline": False,
            "motion_authorized": False, "completion_authorized": False,
        }
