"""Current physical head borders for ID-only camera exploration.

The current border only bounds identity pixels and supplies a camera ray for
independent scan registration. It never fits or replaces the retained angle.
No QR detector, payload, marker geometry, or predicted border is used here.
"""
from dataclasses import asdict, dataclass
import math
import time

from scripts.aufgabe04.perception.stand_axis.geometry import order_corners
from scripts.aufgabe04.perception.stand_axis.head_border_seed import validate_current_head_proposal
from scripts.aufgabe04.perception.stand_axis.head_proposal import (
    _line_groups, _rough_proposals, _proposal, _same_head,
)
from scripts.aufgabe04.perception.stand_axis.head_acquisition_budget import (
    HeadAcquisitionDeadlineExceeded, check_head_acquisition_deadline,
)
from scripts.aufgabe04.perception.stand_axis.model_refinement import refine_projected_head_border
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.preprocessing import _canny_edges_from_frame

HEAD_REGION_POLICY = 'current_opposite_head_region_search_only'
HEAD_POLICY = 'current_opposite_head_region_unique_scan'
MAX_BORDER_VERIFICATIONS = 12


@dataclass(frozen=True)
class OppositeHeadRegion:
    corners_px: tuple
    full_image_center_px: tuple
    image_stamp_sec: float
    image_shape: tuple
    expected_head_height_px: float

    def metadata(self):
        return dict(policy=HEAD_REGION_POLICY, corners_px=self.corners_px,
            center_px=self.full_image_center_px, image_stamp_sec=self.image_stamp_sec,
            image_shape=self.image_shape, expected_head_height_px=self.expected_head_height_px,
            supplies_angle=False, supplies_identity=False, supplies_target_uniqueness=False,
            motion_authorized=False)


@dataclass(frozen=True)
class OppositeHeadSupport:
    corners_px: tuple
    full_image_center_px: tuple
    lidar_association: object
    image_stamp_sec: float
    image_shape: tuple
    expected_head_height_px: float
    depth_m: float
    target_reconciliation: dict | None = None
    finite_bearing: dict | None = None

    @property
    def accepted(self):
        return True

    def metadata(self):
        return dict(policy=HEAD_POLICY, accepted=True, corners_px=self.corners_px,
            center_px=self.full_image_center_px, image_shape=self.image_shape,
            image_stamp_sec=self.image_stamp_sec, expected_head_height_px=self.expected_head_height_px,
            depth_m=self.depth_m, lidar_association=asdict(self.lidar_association),
            supplies_angle=False, supplies_identity=False,
            target_reconciliation=self.target_reconciliation, finite_bearing=self.finite_bearing)


def validate_head_support_geometry(value):
    """Validate complete physical-head geometry, independently of QR corners."""
    if not isinstance(value, dict):
        raise ValueError('current opposite head region missing')
    shape = value.get('image_shape')
    if not isinstance(shape, (tuple, list)) or len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError('opposite head image shape invalid')
    try:
        points = tuple(ImagePoint(*p) for p in value.get('corners_px', ()))
        points = validate_current_head_proposal(points, frame_shape=shape)
    except (TypeError, ValueError):
        raise ValueError('opposite head region is incomplete') from None
    expected, stamp = value.get('expected_head_height_px'), value.get('image_stamp_sec')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (expected, stamp)) or expected <= 0:
        raise ValueError('opposite head scale or timestamp invalid')
    corners = tuple((p.u_px, p.v_px) for p in points)
    width = (math.dist(corners[0], corners[1])+math.dist(corners[2], corners[3]))/2
    height = (math.dist(corners[0], corners[3])+math.dist(corners[1], corners[2]))/2
    # The shared receipt also carries an already admitted ordinary head. Its
    # established physical-head scale limits must not tighten during identity
    # binding; the fresh opposite locator keeps its narrower search below.
    if not .6*expected <= height <= 1.35*expected or not .35*height <= width <= 1.35*height:
        raise ValueError('opposite head region does not match target scale')
    center = tuple(sum(p[k] for p in corners)/4 for k in (0, 1))
    if tuple(value.get('center_px', ())) != center:
        raise ValueError('opposite head center differs from region')
    return corners


def validate_opposite_head_region(value):
    if not isinstance(value, dict) or value.get('policy') != HEAD_REGION_POLICY:
        raise ValueError('current opposite head region missing')
    validate_head_support_geometry(value)
    if any(value.get(key) is not False for key in (
            'supplies_angle', 'supplies_identity', 'supplies_target_uniqueness', 'motion_authorized')):
        raise ValueError('opposite head search region cannot grant authority')
    return value


def head_region_from_corners(corners, *, image_shape, image_stamp_sec, attempt,
        model_profile=None, max_center_offset_ratio=1.5):
    """Bind already measured current full-image head corners to this search.

    Callers with an existing current head measurement reuse it here; no second
    border detector or orientation fit is required.
    """
    if (attempt is None or type(max_center_offset_ratio) not in (int, float)
            or not math.isfinite(max_center_offset_ratio) or not 0 < max_center_offset_ratio <= 1.5):
        return None
    try:
        points = order_corners(tuple(ImagePoint(*p) for p in corners))
        corners = tuple((p.u_px, p.v_px) for p in points)
        center = tuple(sum(p[k] for p in corners)/4 for k in (0, 1))
        if math.dist(center, (attempt.expected_center_u_px, attempt.expected_center_v_px)) > max_center_offset_ratio*attempt.expected_head_height_px:
            return None
        region = OppositeHeadRegion(corners, center, image_stamp_sec, tuple(image_shape),
                                    attempt.expected_head_height_px)
        validate_opposite_head_region(region.metadata())
    except (TypeError, ValueError):
        return None
    return region


def _miss(diagnostics, reason, **fields):
    if diagnostics is not None:
        diagnostics.update(reason=reason, **fields)
    return None


def detect_opposite_head_region(frame, cv2, *, attempt, model_profile,
        image_stamp_sec, now_sec, max_scan_age_sec, resources=None,
        max_elapsed_sec=.06, diagnostics=None):
    """Find a unique current head border inside the bounded candidate hint.

    The existing line endpoint locator supplies hypotheses; existing strict
    current Canny rail/corner refinement verifies all of them. An incomplete
    ambiguity comparison never yields support, including at the deadline.
    """
    if attempt is None or max_elapsed_sec <= 0:
        return _miss(diagnostics, 'search_unavailable' if attempt is None else 'support_time_budget_exhausted')
    if not 0 <= now_sec-image_stamp_sec <= max_scan_age_sec:
        return _miss(diagnostics, 'stale_support_image')
    started = time.monotonic()
    deadline = started+min(max_elapsed_sec, max_scan_age_sec-(now_sec-image_stamp_sec))
    roi = attempt.roi
    if (frame is None or len(frame.shape) != 3 or frame.shape[2] != 3
            or not 0 <= roi.x0 < roi.x1 <= frame.shape[1]
            or not 0 <= roi.y0 < roi.y1 <= frame.shape[0]):
        return _miss(diagnostics, 'head_region_input_invalid')
    pixels = frame[roi.y0:roi.y1, roi.x0:roi.x1]
    center = (attempt.expected_center_u_px-roi.x0, attempt.expected_center_v_px-roi.y0)
    height = attempt.expected_head_height_px
    try:
        check_head_acquisition_deadline(deadline, 'head_region_preprocessing')
        edges = _canny_edges_from_frame(cv2, pixels, edge_preprocess='channel_union',
            blur_kernel=5, canny_low=20, canny_high=60)
        check_head_acquisition_deadline(deadline, 'head_region_lines')
        lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(
            cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY))[0]
        rough = _rough_proposals(_line_groups(lines, expected_height=height, expected_center=center),
            expected_height=height, expected_center=center, max_center_offset=.75,
            deadline_monotonic_sec=deadline)
        if len(rough) > MAX_BORDER_VERIFICATIONS:
            return _miss(diagnostics, 'head_region_verification_budget_exceeded', hypotheses=len(rough))
        accepted = []
        for _, corners in rough:
            check_head_acquisition_deadline(deadline, 'head_region_border_verification')
            measured = refine_projected_head_border(cv2, edges, corners, corridor_half_width_px=4.)
            if measured.accepted:
                proposal = _proposal(measured, edges.shape, expected_height=height, expected_center=center)
                if (.7 <= proposal.expected_height_ratio <= 1.3
                        and head_region_from_corners(tuple((p.u_px+roi.x0, p.v_px+roi.y0) for p in proposal.corners),
                            image_shape=frame.shape[:2], image_stamp_sec=image_stamp_sec, attempt=attempt,
                            max_center_offset_ratio=.75) is not None):
                    accepted.append(proposal)
        check_head_acquisition_deadline(deadline, 'head_region_selection')
    except HeadAcquisitionDeadlineExceeded:
        return _miss(diagnostics, 'support_time_budget_exhausted', comparison_complete=False)
    if not accepted:
        return _miss(diagnostics, 'no_complete_current_head_region', hypotheses=len(rough))
    selected = max(accepted, key=lambda p: (p.head_bounds_xyxy[2]-p.head_bounds_xyxy[0])*p.observed_height_px)
    if any(not _same_head(selected, other) for other in accepted):
        return _miss(diagnostics, 'multiple_current_head_regions')
    region = head_region_from_corners(tuple((p.u_px+roi.x0, p.v_px+roi.y0) for p in selected.corners),
        image_shape=frame.shape[:2], image_stamp_sec=image_stamp_sec, attempt=attempt,
        max_center_offset_ratio=.75)
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(reason='current_head_region_search_only', hypotheses=len(rough),
            verified_borders=len(accepted), elapsed_sec=time.monotonic()-started,
            raw_edge_support=selected.raw_edge_support, comparison_complete=True)
    return region


def support_opposite_head_region(region, *, attempt, image_shape, model_profile,
        intrinsics, scan_from_camera, scan, image_stamp_sec, now_sec, map_bearing_rad,
        cone_half_angle_rad, accepted_range_m, max_scan_age_sec,
        max_camera_map_bearing_delta_rad, target_reconciliation=None,
        fragmentation=None, diagnostics=None):
    from scripts.aufgabe04.real_robot.observer.opposite_target_support import (
        _registration_context, _support_for_outline,
    )
    if (not isinstance(region, OppositeHeadRegion) or attempt is None
            or region.image_stamp_sec != image_stamp_sec
            or tuple(region.image_shape) != tuple(image_shape)):
        return _miss(diagnostics, 'head_region_source_mismatch')
    if not 0 <= now_sec-image_stamp_sec <= max_scan_age_sec:
        return _miss(diagnostics, 'stale_support_image')
    expected = head_region_from_corners(region.corners_px, image_shape=image_shape,
        image_stamp_sec=image_stamp_sec, attempt=attempt)
    if expected is None or expected != region:
        return _miss(diagnostics, 'head_region_search_context_mismatch')
    registration = _registration_context(scan=scan, image_stamp_sec=image_stamp_sec,
        now_sec=now_sec, map_bearing_rad=map_bearing_rad, cone_half_angle_rad=cone_half_angle_rad,
        accepted_range_m=accepted_range_m, max_scan_age_sec=max_scan_age_sec,
        max_camera_map_bearing_delta_rad=max_camera_map_bearing_delta_rad,
        target_reconciliation=target_reconciliation, fragmentation=fragmentation, diagnostics=diagnostics)
    if registration is None:
        return None
    result = _support_for_outline(region, intrinsics=intrinsics, scan_from_camera=scan_from_camera,
        scan=scan, now_sec=now_sec, cone_half_angle_rad=cone_half_angle_rad,
        accepted_range_m=accepted_range_m, max_scan_age_sec=max_scan_age_sec,
        target_reconciliation=target_reconciliation, registration=registration,
        diagnostics=diagnostics, support_factory=OppositeHeadSupport)
    if result is not None and diagnostics is not None:
        diagnostics.clear()
        diagnostics['reason'] = 'current_head_region_associated'
    return result


def detect_opposite_head_support(frame, cv2, *, attempt, intrinsics, model_profile,
        scan_from_camera, scan, image_stamp_sec, now_sec, map_bearing_rad,
        cone_half_angle_rad, accepted_range_m, max_scan_age_sec,
        max_camera_map_bearing_delta_rad, resources=None, max_elapsed_sec=.06,
        target_reconciliation=None, fragmentation=None, diagnostics=None):
    started = time.monotonic()
    region = detect_opposite_head_region(frame, cv2, attempt=attempt, model_profile=model_profile,
        image_stamp_sec=image_stamp_sec, now_sec=now_sec, max_scan_age_sec=max_scan_age_sec,
        max_elapsed_sec=max_elapsed_sec, diagnostics=diagnostics)
    if region is None:
        return None
    return support_opposite_head_region(region, attempt=attempt, image_shape=frame.shape[:2],
        model_profile=model_profile, intrinsics=intrinsics, scan_from_camera=scan_from_camera,
        scan=scan, image_stamp_sec=image_stamp_sec, now_sec=now_sec+time.monotonic()-started,
        map_bearing_rad=map_bearing_rad, cone_half_angle_rad=cone_half_angle_rad,
        accepted_range_m=accepted_range_m, max_scan_age_sec=max_scan_age_sec,
        max_camera_map_bearing_delta_rad=max_camera_map_bearing_delta_rad,
        target_reconciliation=target_reconciliation, fragmentation=fragmentation,
        diagnostics=diagnostics)
