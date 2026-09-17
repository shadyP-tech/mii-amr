"""Read-only current-scan association while comparing measured head borders.

Rough locator centers can move during raw refinement. Their preview uses a
conservative angular envelope, never an exact scan-cone decision. The callable
checks completed measured centers against the ordinary current-scan gate; the
observer still repeats association and freshness checks before publication.
"""

from dataclasses import dataclass, field
import math

from scripts.aufgabe04.perception.candidate_lidar_association import (
    associate_camera_registered_candidate_lidar_target,
    normalize_certified_camera_map_bearing_limit,
)
from scripts.aufgabe04.perception.stand_axis_handoff import rectified_pixel_bearing_in_scan
from scripts.aufgabe04.real_robot.observer.head_model_admission import measured_head_lidar_rejection


# Metric raw-refinement corridors are capped at eight pixels, with a two-pixel
# center allowance. Outer recovery grows about the seed center, not a new pose.
MAX_HINT_CENTER_SHIFT_PX = 10.


def _angle_delta(left, right):
    return math.atan2(math.sin(left-right), math.cos(left-right))


@dataclass
class CurrentScanHeadProposalFilter:
    intrinsics: object
    scan_from_camera: object
    scan: object
    map_bearing_rad: float
    cone_half_angle_rad: float
    accepted_range_m: tuple[float, float]
    now_sec: float
    max_scan_age_sec: float
    min_cluster_sample_count: int
    max_camera_map_bearing_delta_rad: float
    preview_lidar_association: object = None
    current_ros_sec: object = None
    _diagnostics: dict = field(default_factory=lambda: {
        "rough_previews": 0, "rough_rejections": 0, "rough_envelope_fallbacks": 0,
        "measured_scan_previews": 0, "measured_scan_rejections": 0,
        "measured_associations": [],
    }, init=False, repr=False)

    def __post_init__(self):
        self.max_camera_map_bearing_delta_rad = normalize_certified_camera_map_bearing_limit(
            self.max_camera_map_bearing_delta_rad)

    def _bearing(self, u, v):
        return rectified_pixel_bearing_in_scan(
            u_px=u, v_px=v, fx_px=self.intrinsics.fx_px, fy_px=self.intrinsics.fy_px,
            cx_px=self.intrinsics.cx_px, cy_px=self.intrinsics.cy_px,
            scan_from_camera=self.scan_from_camera)

    def preview(self, proposal):
        """Reject only centers whose whole refinement envelope misses the map gate.

        Projected scan rays are affine in image coordinates. When all four
        corners of the center's ten-pixel square lie in the same open angular
        half-plane, their bearing extrema bound every point inside the square.
        A singular ray or an envelope crossing that half-plane is retained.
        No scan clustering or persistence operation runs on rough hypotheses.
        """
        self._diagnostics["rough_previews"] += 1
        u, v = proposal.center_u_px, proposal.center_v_px
        try:
            center = self._bearing(u, v)
            angles = tuple(_angle_delta(self._bearing(u+du, v+dv), center)
                           for du in (-MAX_HINT_CENTER_SHIFT_PX, MAX_HINT_CENTER_SHIFT_PX)
                           for dv in (-MAX_HINT_CENTER_SHIFT_PX, MAX_HINT_CENTER_SHIFT_PX))
            if any(abs(value) >= math.pi/2 for value in angles):
                raise ValueError("bearing envelope is not in one open half-plane")
            lower, upper = min(angles), max(angles)
            target = _angle_delta(self.map_bearing_rad, center)
            distance = min(max(lower-value, 0., value-upper)
                           for value in (target-2*math.pi, target, target+2*math.pi))
            accepted = distance <= self.max_camera_map_bearing_delta_rad + 1.e-12
        except (TypeError, ValueError, ArithmeticError):
            self._diagnostics["rough_envelope_fallbacks"] += 1
            return True
        self._diagnostics["rough_rejections"] += int(not accepted)
        return accepted

    def __call__(self, proposal):
        """Check this measured center without recording persistence evidence."""
        self._diagnostics["measured_scan_previews"] += 1
        bearing, association = None, None
        try:
            bearing = self._bearing(proposal.center_u_px, proposal.center_v_px)
            association = associate_camera_registered_candidate_lidar_target(
                self.scan, map_bearing_rad=self.map_bearing_rad,
                observed_camera_bearing_rad=bearing,
                cone_half_angle_rad=self.cone_half_angle_rad,
                accepted_range_m=self.accepted_range_m,
                now_sec=self.now_sec if self.current_ros_sec is None else self.current_ros_sec(),
                max_scan_age_sec=self.max_scan_age_sec,
                min_cluster_sample_count=self.min_cluster_sample_count,
                max_camera_map_bearing_delta_rad=self.max_camera_map_bearing_delta_rad)
            if self.preview_lidar_association is not None:
                association = self.preview_lidar_association(association, self.scan)
            reason = association.rejection_reason
            if association.associated:
                reason = measured_head_lidar_rejection(
                    association.search_association, registered=True,
                    cone_half_angle_rad=self.cone_half_angle_rad,
                    registered_association=association)
            accepted = bool(association.associated and not reason)
        except (TypeError, ValueError, ArithmeticError):
            accepted, reason = False, "current_head_projection_invalid"
        self._diagnostics["measured_scan_rejections"] += int(not accepted)
        self._diagnostics["measured_associations"].append({
            "center_full_image_px": (proposal.center_u_px, proposal.center_v_px),
            "bearing_rad": bearing, "associated": accepted, "reason": reason,
        })
        return accepted

    def metadata(self):
        return {**self._diagnostics,
                "measured_associations": list(self._diagnostics["measured_associations"]),
                "policy": "current_scan_measured_proposal_preview",
                "rough_policy": "conservative_map_bearing_envelope",
                "hint_center_refinement_allowance_px": MAX_HINT_CENTER_SHIFT_PX,
                "rough_scan_clustering_performed": False,
                "persistence_read_only": True, "final_association_required": True,
                "motion_authorized": False, "completion_authorized": False}
