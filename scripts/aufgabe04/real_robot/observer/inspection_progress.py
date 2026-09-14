"""Bounded evidence of an unresolved view, separate from pose authority."""

from dataclasses import dataclass
import math

from scripts.aufgabe04.artifacts.candidate_inspection_observation import (
    MAX_ADVISORY_ACCUMULATION_WINDOW_SEC,
    MIN_PROGRESS_FRAMES,
    MIN_PROGRESS_SPAN_SEC,
)


# The measured production QR/model path takes about 1.4 seconds per frame.
# Seven advisory frames therefore need more than the independent five-second
# QR/axis latch lifetime. Every sample must still pass fresh sensor gates at
# ingestion; this bounded history gives no older QR or metric pose authority.
INSPECTION_PROGRESS_WINDOW_SEC = MAX_ADVISORY_ACCUMULATION_WINDOW_SEC


@dataclass(frozen=True)
class InspectionClassification:
    classification: str
    reason: str
    camera_relative_yaw_rad: float | None = None


def classify_inspection_progress(state: str, details: dict) -> InspectionClassification:
    """Describe a current view without promoting rejected geometry to a pose."""

    axis = details.get("stand_axis_debug") or {}
    model = axis.get("metric_model") or {}
    reason = str(details.get("estimator_reason") or details.get("reason") or axis.get("estimator_reason") or state)
    yaw = axis.get("advisory_camera_relative_yaw_rad")
    if isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
        yaw = None
    front = details.get("front_observation") or {}
    if front.get("axis_state") == "unresolved" and front.get("classification") in {
        "front_readable", "front_unreadable",
    }:
        # A contradicted QR-free estimate cannot supply even the advisory
        # angle of an observed front marker.
        return InspectionClassification(front["classification"], reason, None)
    model_edge_view = (
        axis.get("estimator_usable") is True
        and yaw is not None
        and abs(abs(yaw) - math.pi / 2) <= math.radians(15)
    )
    if model_edge_view or axis.get("estimator_view_mode") == "edge_on" or axis.get("estimator_reason") == "edge_on_approx_90_deg":
        classification = "edge_on"
    elif details.get("conditioning", {}).get("reason") == "oblique_silhouette":
        classification = "oblique"
    elif details.get("qr_texts"):
        classification = "front_readable"
    elif model.get("qr_detected") is True:
        classification = "front_unreadable"
    elif (model.get("observation_confidence") or {}).get("backside", {}).get(
            "state") == "backside_supported":
        # Repeated complete-head appearance survives an ambiguous angle, but
        # cannot attach that rejected angle to a planning observation.
        return InspectionClassification("backside_unresolved", reason, None)
    elif model.get("visible_face") == "backside_candidate":
        classification = "backside_unresolved"
    else:
        classification = "unobservable"
    return InspectionClassification(classification, reason, yaw)


class InspectionProgress:
    """Keep distinct sensor tuples from one stationary, unpoisoned view."""

    def __init__(self, *, required_frames=MIN_PROGRESS_FRAMES, minimum_span_sec=MIN_PROGRESS_SPAN_SEC,
                 max_age_sec=INSPECTION_PROGRESS_WINDOW_SEC,
                 max_translation_m=0.02, max_rotation_rad=math.radians(2)):
        if isinstance(required_frames, bool) or not isinstance(required_frames, int) or required_frames < MIN_PROGRESS_FRAMES:
            raise ValueError("inspection progress needs at least seven frames")
        if not math.isfinite(minimum_span_sec) or minimum_span_sec < MIN_PROGRESS_SPAN_SEC:
            raise ValueError("inspection progress needs at least two seconds")
        if (type(max_age_sec) not in (int, float) or not math.isfinite(max_age_sec)
                or not minimum_span_sec <= max_age_sec <= INSPECTION_PROGRESS_WINDOW_SEC):
            raise ValueError("inspection window must cover its minimum span and cannot exceed 15 seconds")
        self.required_frames = required_frames
        self.minimum_span_sec = minimum_span_sec
        self.max_age_sec = max_age_sec
        self.max_translation_m = max_translation_m
        self.max_rotation_rad = max_rotation_rad
        self._samples = {}
        self._anchor = None
        self._poisoned = False
        self._seen_qr_id = None

    @property
    def poisoned(self):
        """Expose epoch conflict state without allowing advisory reset to erase it."""
        return self._poisoned

    def restart_acquisition_window(self):
        """Give new good measurements time without forgetting epoch identity."""

        self._samples.clear()

    def record(self, *, frame_stamp_sec, robot_pose, frame_accepted, poisoned, classification,
               current_qr_id=None, current_qr_sample_count=0, motion_epoch_reset=False):
        if isinstance(frame_stamp_sec, bool) or not math.isfinite(frame_stamp_sec) or frame_stamp_sec < 0:
            raise ValueError("inspection frame timestamp is invalid")
        pose = tuple(float(robot_pose[k]) for k in ("x_m", "y_m", "yaw_rad"))
        if not all(math.isfinite(v) for v in pose):
            raise ValueError("inspection pose is invalid")
        new_motion_epoch = motion_epoch_reset
        if self._anchor is not None:
            dx, dy = pose[0] - self._anchor[0], pose[1] - self._anchor[1]
            dyaw = abs(math.atan2(math.sin(pose[2] - self._anchor[2]), math.cos(pose[2] - self._anchor[2])))
            new_motion_epoch = new_motion_epoch or math.hypot(dx, dy) > self.max_translation_m or dyaw > self.max_rotation_rad
        if new_motion_epoch:
            self._samples.clear()
            self._anchor = None
            self._seen_qr_id = None
            self._poisoned = False
        if poisoned:
            self._poisoned = True
            self._samples.clear()
        if self._poisoned or frame_accepted is not True:
            return None
        if self._anchor is None:
            self._anchor = pose
        if current_qr_id is not None:
            if self._seen_qr_id is not None and current_qr_id != self._seen_qr_id:
                self._samples.clear()
                self._poisoned = True
                return None
            self._seen_qr_id = current_qr_id
        if self._samples and frame_stamp_sec <= max(self._samples):
            return None
        newest = max(frame_stamp_sec, max(self._samples, default=frame_stamp_sec))
        self._samples = {s: c for s, c in self._samples.items() if s >= newest - self.max_age_sec}
        if frame_stamp_sec < newest - self.max_age_sec:
            return None
        self._samples[frame_stamp_sec] = classification
        stamps = sorted(self._samples)
        if len(stamps) < self.required_frames or stamps[-1] - stamps[0] < self.minimum_span_sec:
            return None
        # Identity comes only from this frame's live, validated latch. Never
        # cache an expired identity or carry it across observer processes.
        if current_qr_id is not None and (not isinstance(current_qr_id, str) or not current_qr_id.strip()):
            raise ValueError("inspection current QR latch is invalid")
        latest = self._samples[stamps[-1]]
        return {
            "classification": latest.classification,
            "reasons": sorted({c.reason for c in self._samples.values()}),
            "camera_relative_yaw_rad": latest.camera_relative_yaw_rad,
            "yaw_uncertainty_rad": None if latest.camera_relative_yaw_rad is None else math.pi / 2,
            "qr_id": current_qr_id,
            "qr_sample_count": current_qr_sample_count,
            "robot_pose": dict(robot_pose),
            "sample_count": len(stamps),
            "sensor_stamps_sec": stamps,
            "first_sensor_stamp_sec": stamps[0],
            "sensor_stamp_sec": stamps[-1],
            "advisory_accumulation_window_sec": self.max_age_sec,
            "sample_gate_evidence": {k: True for k in (
                "all_samples_stationary", "all_samples_synchronized", "all_samples_lidar_associated", "all_samples_fresh",
            )},
        }
