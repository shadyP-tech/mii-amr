"""Motion-neutral framing vetoes and a fixed opportunity for useful geometry.

These policies only decline optional centering. They never select new scan
returns, change a centering target, relax admission, or authorize a turn.
"""
from dataclasses import asdict, dataclass
import math

from scripts.aufgabe04.perception.stand_axis_handoff import rectified_pixel_bearing_in_scan
from scripts.aufgabe04.real_robot.observer.candidate_centering import (
    center_point_in_base, project_center_after_turn,
)
from scripts.aufgabe04.real_robot.observer.scan_target_persistence import scan_pose_from_camera_extrinsics

PRODUCTIVE_VIEW_OPPORTUNITY_SEC = 5.0


class ProductiveViewHold:
    """One non-renewable geometry opportunity per calibrated stationary epoch."""
    def __init__(self):
        self.context = None
        self.started = None
        self.poisoned = False

    def observe(self, *, context, now_sec, axis_sample_accepted, poisoned=False):
        if not math.isfinite(now_sec) or now_sec < 0:
            raise ValueError("productive view clock must be finite and nonnegative")
        if context != self.context:
            self.context, self.started, self.poisoned = context, None, False
        self.poisoned = self.poisoned or poisoned
        if self.poisoned:
            return False
        if self.started is None and axis_sample_accepted:
            self.started = now_sec
        return self.started is not None and 0 <= now_sec-self.started < PRODUCTIVE_VIEW_OPPORTUNITY_SEC

    def metadata(self, now_sec):
        deadline = None if self.started is None else self.started+PRODUCTIVE_VIEW_OPPORTUNITY_SEC
        return dict(policy="productive_geometry_before_optional_centering", duration_sec=PRODUCTIVE_VIEW_OPPORTUNITY_SEC,
            deadline_monotonic_sec=deadline, remaining_sec=None if deadline is None else max(0., deadline-now_sec),
            renews_on_soft_miss=False, extends_parent_deadline=False, motion_authorized=False)


@dataclass(frozen=True)
class FramingDecision:
    allowed: bool
    reason: str
    predicted_point_bearing_rad: float | None = None
    predicted_ray_bearing_rad: float | None = None
    boundary_center_rad: float | None = None
    required_clearance_rad: float | None = None
    minimum_clearance_rad: float | None = None

    def metadata(self):
        return {**asdict(self), "policy": "scan_boundary_centering_veto",
                "changes_centering_target": False, "motion_authorized": False}


def review_centering_destination(advisory, *, search_association):
    """Veto a step whose target cone touches the raw full-rotation boundary.

    Use original indexed endpoints plus one angular sample of margin. Consider
    both the depth-aware point and rotation-only ray used by head association.
    A single scan's valid seam does not establish that the next scan will have
    valid endpoint metadata. Linear/partial scanners retain their own policy.
    """
    topology = search_association.scan_topology or {}
    if topology.get("profile") != "full_rotation":
        return FramingDecision(True, "full_rotation_boundary_not_applicable")
    try:
        count = topology["sample_count"]
        start, step = topology["angle_min_rad"], topology["angle_increment_rad"]
        cone = search_association.cone_half_angle_rad
        if (type(count) is not int or count < 3
                or not all(type(v) in (int, float) and math.isfinite(v) for v in (start, step, cone))
                or step == 0 or not 0 < cone < math.pi/2):
            raise ValueError("invalid boundary geometry")
        span = (count-1)*abs(step)
        if not 0 < span < math.tau:
            raise ValueError("partial or overlapping boundary geometry")
        gap = math.tau-span
        boundary = start-math.copysign(gap/2, step)
        required = cone+gap/2+abs(step)
        point = center_point_in_base(center_px=advisory.measured_center_px,
            intrinsics=advisory.intrinsics, distance_m=advisory.associated_range_m,
            scan_from_camera=advisory.scan_from_camera, base_from_camera=advisory.base_from_camera)
        c, s = math.cos(advisory.requested_yaw_rad), math.sin(advisory.requested_yaw_rad)
        ext = scan_pose_from_camera_extrinsics(advisory.base_from_camera.translation_xyz_m,
            advisory.base_from_camera.rotation_xyzw, advisory.scan_from_camera.translation_xyz_m,
            advisory.scan_from_camera.rotation_xyzw)
        point_bearing = math.atan2(-s*point[0]+c*point[1]-ext.y_m,
                                   c*point[0]+s*point[1]-ext.x_m)-ext.yaw_rad
        u, v = project_center_after_turn(point_base=point, yaw_rad=advisory.requested_yaw_rad,
            intrinsics=advisory.intrinsics, base_from_camera=advisory.base_from_camera)
        camera = advisory.intrinsics
        ray = rectified_pixel_bearing_in_scan(u_px=u, v_px=v, fx_px=camera.fx_px, fy_px=camera.fy_px,
            cx_px=camera.cx_px, cy_px=camera.cy_px, scan_from_camera=advisory.scan_from_camera)
        clearance = min(abs(math.remainder(b-boundary, math.tau)) for b in (point_bearing, ray))
        allowed = clearance > required
        return FramingDecision(allowed, "destination_clear_of_scan_boundary" if allowed else
            "preserve_view_centering_would_cross_scan_boundary", point_bearing, ray, boundary, required, clearance)
    except (AttributeError, KeyError, TypeError, ValueError, ArithmeticError):
        return FramingDecision(False, "centering_boundary_geometry_unavailable")
