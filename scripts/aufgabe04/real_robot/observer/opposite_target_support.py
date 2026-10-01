"""Shared current range support and legacy QR-outline receipt compatibility.

Camera exploration uses physical head regions from opposite_head_support.
Historical outline producers remain readable without changing their policy.
"""
from dataclasses import asdict, dataclass
import math
import time

from scripts.aufgabe04.perception.candidate_lidar_association import associate_camera_registered_candidate_lidar_target
from scripts.aufgabe04.real_robot.observer.finite_target_bearing import finite_target_bearing
from scripts.aufgabe04.real_robot.observer.qr_candidate_search import qr_registration_envelope
from scripts.aufgabe04.qr_scanning.qr_observation import validated_qr_corners

from scripts.aufgabe04.real_robot.observer.shared_scan_cluster import envelope_is_unique, bind_ray_to_envelope
from scripts.aufgabe04.real_robot.observer.scan_target_persistence import registered_target_metadata_is_unique

POLICY = 'current_opposite_qr_outline_unique_scan'
OUTLINE_POLICY = 'current_opposite_qr_outline_search_only'


@dataclass(frozen=True)
class OppositeQrOutline:
    """Current pixels only; target association is a separate proof."""

    corners_px: tuple
    full_image_center_px: tuple
    image_stamp_sec: float
    image_shape: tuple
    expected_symbol_height_px: float

    def metadata(self):
        return dict(policy=OUTLINE_POLICY, corners_px=self.corners_px,
            center_px=self.full_image_center_px, image_stamp_sec=self.image_stamp_sec,
            image_shape=self.image_shape, expected_symbol_height_px=self.expected_symbol_height_px,
            supplies_angle=False, supplies_identity=False, supplies_target_uniqueness=False,
            motion_authorized=False)


def validate_opposite_qr_outline(value):
    """Validate visual geometry, without granting scan or identity authority."""
    if not isinstance(value, dict) or value.get('policy') != OUTLINE_POLICY:
        raise ValueError('current opposite QR outline missing')
    shape = value.get('image_shape')
    if not isinstance(shape, (tuple, list)) or len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError('opposite QR image shape invalid')
    corners = validated_qr_corners(value.get('corners_px'), image_shape=shape)
    if corners is None:
        raise ValueError('opposite QR outline is incomplete')
    expected, stamp = value.get('expected_symbol_height_px'), value.get('image_stamp_sec')
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (expected, stamp))
            or expected <= 0):
        raise ValueError('opposite QR scale or timestamp invalid')
    edges = [math.dist(corners[i], corners[(i+1)%4]) for i in range(4)]
    if min(edges) < .6*expected or max(edges) > 1.4*expected:
        raise ValueError('opposite QR outline does not match target scale')
    center = tuple(sum(p[k] for p in corners)/4 for k in (0, 1))
    if tuple(value.get('center_px', ())) != center:
        raise ValueError('opposite QR center differs from outline')
    if any(value.get(key) is not False for key in (
            'supplies_angle', 'supplies_identity', 'supplies_target_uniqueness', 'motion_authorized')):
        raise ValueError('opposite QR search outline cannot grant authority')
    return value


@dataclass(frozen=True)
class OppositeTargetSupport:
    corners_px: tuple
    full_image_center_px: tuple
    lidar_association: object
    image_stamp_sec: float
    image_shape: tuple
    expected_symbol_height_px: float
    depth_m: float
    target_reconciliation: dict | None = None
    finite_bearing: dict | None = None

    @property
    def accepted(self):
        return True

    def metadata(self):
        return dict(policy=POLICY, accepted=True, corners_px=self.corners_px,
            center_px=self.full_image_center_px, image_shape=self.image_shape,
            image_stamp_sec=self.image_stamp_sec, expected_symbol_height_px=self.expected_symbol_height_px,
            depth_m=self.depth_m, lidar_association=asdict(self.lidar_association),
            supplies_angle=False, supplies_identity=False,
            target_reconciliation=self.target_reconciliation, finite_bearing=self.finite_bearing)


def validate_target_support(value):
    """Validate the persisted current-source proof, including its pixel extent."""
    from scripts.aufgabe04.real_robot.observer.opposite_head_support import (
        HEAD_POLICY, validate_head_support_geometry,
    )
    if not isinstance(value, dict) or value.get('policy') not in (POLICY, HEAD_POLICY) or value.get('accepted') is not True:
        raise ValueError('current opposite QR support missing')
    head_region = value['policy'] == HEAD_POLICY
    shape = value.get('image_shape')
    if not isinstance(shape, (tuple, list)) or len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError('opposite QR image shape invalid')
    corners = (validate_head_support_geometry(value) if head_region else
               validated_qr_corners(value.get('corners_px'), image_shape=shape))
    if corners is None:
        raise ValueError('opposite QR outline is incomplete')
    height = value.get('expected_head_height_px' if head_region else 'expected_symbol_height_px')
    depth = value.get('depth_m')
    stamp = value.get('image_stamp_sec')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (height, depth, stamp)) or min(height, depth) <= 0:
        raise ValueError('opposite QR scale/depth invalid')
    edges = [math.dist(corners[i], corners[(i+1)%4]) for i in range(4)]
    if not head_region and (not .6*height <= min(edges) or max(edges) > 1.4*height):
        raise ValueError('opposite QR outline does not match target scale')
    center = tuple(sum(p[k] for p in corners)/4 for k in (0, 1))
    if tuple(value.get('center_px', ())) != center:
        raise ValueError('opposite QR center differs from outline')
    lidar = value.get('lidar_association') or {}
    cluster = lidar.get('search_association') if isinstance(lidar, dict) else None
    if not isinstance(lidar, dict) or not isinstance(cluster, dict):
        raise ValueError('opposite QR scan support invalid')
    for key in ('distance_m', 'camera_map_bearing_delta_rad', 'max_camera_map_bearing_delta_rad'):
        if type(lidar.get(key)) not in (int, float) or not math.isfinite(lidar[key]):
            raise ValueError('opposite QR registration invalid')
    if not 0 <= lidar['camera_map_bearing_delta_rad'] <= lidar['max_camera_map_bearing_delta_rad'] <= math.radians(12)+1e-9:
        raise ValueError('opposite QR bearing exceeds registration limit')
    indices = cluster.get('selected_cluster_source_indices')
    if (not isinstance(indices, (tuple, list)) or not indices or any(type(i) is not int or i < 0 for i in indices)
            or len(set(indices)) != len(indices) or cluster.get('selected_cluster_sample_count') != len(indices)
            or type(cluster.get('scan_stamp_sec')) not in (int, float) or not math.isfinite(cluster['scan_stamp_sec'])
            or not isinstance(cluster.get('scan_frame_id'), str) or not cluster['scan_frame_id']
            or lidar['distance_m'] <= 0):
        raise ValueError('opposite QR current scan samples invalid')
    if (lidar.get('associated') is not True or cluster.get('associated') is not True
            or not registered_target_metadata_is_unique(lidar)
            or not cluster.get('selected_cluster_source_indices')
            or abs(stamp-cluster.get('scan_stamp_sec', -1.)) > .1
            or value.get('supplies_angle') is not False or value.get('supplies_identity') is not False):
        raise ValueError('opposite QR outline lacks its unique synchronized scan')
    proof = value.get('target_reconciliation')
    if proof is not None:
        from scripts.aufgabe04.real_robot.observer.target_reconciliation import validate_reconciliation
        from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
        from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics
        scan, envelope, _, reference = validate_reconciliation(proof,
            image_stamp_sec=stamp, scan_stamp_sec=cluster['scan_stamp_sec'])
        geometry = value['finite_bearing']
        bearing, uncertainty, optical_depth = finite_target_bearing(center_px=center,
            intrinsics=CameraIntrinsics(**geometry['intrinsics']),
            scan_from_camera=RigidTransform(**geometry['scan_from_camera']),
            distance_m=envelope.distance_m, range_interval_m=envelope.accepted_range_m)
        if abs(math.remainder(bearing-reference,math.tau))+uncertainty > math.radians(3)+1e-9:
            raise ValueError('opposite QR ray misses reconciled target')
        current = associate_camera_registered_candidate_lidar_target(scan,
            map_bearing_rad=reference, observed_camera_bearing_rad=bearing,
            cone_half_angle_rad=math.radians(3), accepted_range_m=envelope.accepted_range_m,
            now_sec=proof['entries'][-1]['checked_at_sec'], max_scan_age_sec=.5,
            min_cluster_sample_count=1,max_camera_map_bearing_delta_rad=math.radians(3))
        current = bind_ray_to_envelope(current, scan, envelope,
            now_sec=proof['entries'][-1]['checked_at_sec'], max_scan_age_sec=.5)
        if (not current.associated or abs(optical_depth-depth)>1e-9
                or current.distance_m != lidar['distance_m']
                or tuple(current.search_association.selected_cluster_source_indices) != tuple(indices)
                or abs(current.camera_map_bearing_delta_rad-lidar['camera_map_bearing_delta_rad'])>1e-9):
            raise ValueError('opposite QR reconciliation differs from current scan')
    return value


def _miss(diagnostics, reason, **fields):
    if diagnostics is not None:
        diagnostics.update(reason=reason, **fields)
    return None


def _outline_from_corners(corners, *, image_shape, image_stamp_sec, attempt, model_profile):
    corners = validated_qr_corners(corners, image_shape=image_shape)
    if corners is None:
        return None
    center = tuple(sum(p[k] for p in corners)/4 for k in (0, 1))
    if math.dist(center, (attempt.expected_center_u_px, attempt.expected_center_v_px)) > .75*attempt.expected_head_height_px:
        return None
    expected = attempt.expected_head_height_px*model_profile.qr_symbol_height_m/model_profile.head_height_m
    outline = OppositeQrOutline(corners, center, image_stamp_sec, tuple(image_shape), expected)
    try:
        validate_opposite_qr_outline(outline.metadata())
    except ValueError:
        return None
    return outline


def _outline_batches(frame, cv2, *, attempt, model_profile, image_stamp_sec,
                     resources, started, max_elapsed_sec):
    """Keep per-scale batches so normal association can reject other outlines."""
    roi = attempt.roi
    pixels = frame[roi.y0:roi.y1, roi.x0:roi.x1]
    detector = cv2.QRCodeDetector() if resources is None else resources.decoder('native')
    for scale in (4, 1):
        if time.monotonic()-started >= max_elapsed_sec:
            break
        view = pixels if scale == 1 else cv2.resize(pixels, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        try:
            found, points = detector.detectMulti(view)
        except (AttributeError, TypeError, ValueError, cv2.error):
            continue
        if not found or points is None:
            continue
        outlines = []
        for quad in points:
            corners = tuple((float(p[0])/scale+roi.x0, float(p[1])/scale+roi.y0) for p in quad)
            outline = _outline_from_corners(corners, image_shape=frame.shape[:2],
                image_stamp_sec=image_stamp_sec, attempt=attempt, model_profile=model_profile)
            if outline is not None:
                outlines.append(outline)
        yield tuple(outlines)


def detect_opposite_qr_outline(frame, cv2, *, attempt, model_profile,
        image_stamp_sec, now_sec, max_scan_age_sec, resources=None,
        max_elapsed_sec=.06, diagnostics=None):
    """Find one complete current symbol in a bounded search hint, without decoding.

    A search hint need not establish LiDAR uniqueness. The returned visual
    geometry cannot be used as target support until independently associated.
    """
    if attempt is None or max_elapsed_sec <= 0:
        return _miss(diagnostics, 'search_unavailable' if attempt is None else 'support_time_budget_exhausted')
    if not 0 <= now_sec-image_stamp_sec <= max_scan_age_sec:
        return _miss(diagnostics, 'stale_support_image')
    _miss(diagnostics, 'no_complete_target_outline')
    for outlines in _outline_batches(frame, cv2, attempt=attempt, model_profile=model_profile,
            image_stamp_sec=image_stamp_sec, resources=resources, started=time.monotonic(),
            max_elapsed_sec=max_elapsed_sec):
        if len(outlines) == 1:
            if diagnostics is not None:
                diagnostics.clear()
                diagnostics['reason'] = 'current_outline_search_only'
            return outlines[0]
        if len(outlines) > 1:
            return _miss(diagnostics, 'multiple_search_outlines')
    return None


def _registration_context(*, scan, image_stamp_sec, now_sec, map_bearing_rad,
        cone_half_angle_rad, accepted_range_m, max_scan_age_sec,
        max_camera_map_bearing_delta_rad, target_reconciliation, fragmentation, diagnostics):
    envelope = qr_registration_envelope(scan, map_bearing_rad=map_bearing_rad,
        cone_half_angle_rad=cone_half_angle_rad, max_camera_map_bearing_delta_rad=max_camera_map_bearing_delta_rad,
        accepted_range_m=accepted_range_m, now_sec=now_sec, max_scan_age_sec=max_scan_age_sec,
        fragmentation=fragmentation)
    reference, limit = map_bearing_rad, max_camera_map_bearing_delta_rad
    association_range = accepted_range_m
    if target_reconciliation is not None:
        from scripts.aufgabe04.real_robot.observer.candidate_position_epoch import validated_reconciliation_envelope
        try:
            _, envelope, _, reference = validated_reconciliation_envelope(target_reconciliation,
                scan=scan, image_stamp_sec=image_stamp_sec, map_bearing_rad=map_bearing_rad,
                accepted_range_m=accepted_range_m)
            association_range = envelope.accepted_range_m
            limit = min(cone_half_angle_rad, math.radians(3))
        except (ValueError, TypeError, KeyError, OSError) as exc:
            return _miss(diagnostics, 'invalid_target_reconciliation', detail=str(exc))
    if not envelope_is_unique(envelope):
        return _miss(diagnostics, 'scan_cluster_not_unique', accepted_range_m=accepted_range_m)
    return envelope, reference, limit, association_range


def _support_for_outline(outline, *, intrinsics, scan_from_camera, scan,
        now_sec, cone_half_angle_rad, accepted_range_m, max_scan_age_sec,
        target_reconciliation, registration, diagnostics,
        support_factory=OppositeTargetSupport):
    envelope, reference, limit, association_range = registration
    try:
        bearing, uncertainty, depth = finite_target_bearing(center_px=outline.full_image_center_px,
            intrinsics=intrinsics, scan_from_camera=scan_from_camera,
            distance_m=envelope.distance_m, range_interval_m=association_range)
    except ValueError:
        return None
    if abs(math.remainder(bearing-reference, math.tau))+uncertainty > limit:
        return _miss(diagnostics, 'outline_bearing_interval_exceeds_limit', bearing_rad=bearing,
            reference_rad=reference, uncertainty_rad=uncertainty, limit_rad=limit)
    lidar = associate_camera_registered_candidate_lidar_target(scan,
        map_bearing_rad=reference, observed_camera_bearing_rad=bearing,
        cone_half_angle_rad=cone_half_angle_rad, accepted_range_m=association_range,
        now_sec=now_sec, max_scan_age_sec=max_scan_age_sec,
        min_cluster_sample_count=1, max_camera_map_bearing_delta_rad=limit)
    lidar = bind_ray_to_envelope(lidar, scan, envelope,
        now_sec=now_sec, max_scan_age_sec=max_scan_age_sec)
    height = getattr(outline, 'expected_head_height_px', None)
    if height is None:
        height = outline.expected_symbol_height_px
    support = support_factory(outline.corners_px, outline.full_image_center_px, lidar,
        outline.image_stamp_sec, outline.image_shape, height,
        depth, target_reconciliation,
        dict(intrinsics=asdict(intrinsics), scan_from_camera=asdict(scan_from_camera)))
    try:
        validate_target_support(support.metadata())
    except ValueError as exc:
        return _miss(diagnostics, 'outline_scan_support_rejected', detail=str(exc),
            lidar_reason=lidar.rejection_reason, accepted_range_m=accepted_range_m)
    return support


def support_opposite_qr_outline(outline, *, attempt, image_shape, model_profile,
        intrinsics, scan_from_camera, scan, image_stamp_sec, now_sec, map_bearing_rad,
        cone_half_angle_rad, accepted_range_m, max_scan_age_sec,
        max_camera_map_bearing_delta_rad, target_reconciliation=None,
        fragmentation=None, diagnostics=None):
    """Associate already detected current pixels without another native detection.

    The caller must supply the same current image tuple and its search context.
    Retained corners, changed scale, and unproved fragmented scans cannot pass.
    """
    if (not isinstance(outline, OppositeQrOutline) or attempt is None
            or outline.image_stamp_sec != image_stamp_sec
            or tuple(outline.image_shape) != tuple(image_shape)):
        return _miss(diagnostics, 'outline_source_mismatch')
    if not 0 <= now_sec-image_stamp_sec <= max_scan_age_sec:
        return _miss(diagnostics, 'stale_support_image')
    expected = _outline_from_corners(outline.corners_px, image_shape=image_shape,
        image_stamp_sec=image_stamp_sec, attempt=attempt, model_profile=model_profile)
    if expected is None or expected != outline:
        return _miss(diagnostics, 'outline_search_context_mismatch')
    registration = _registration_context(scan=scan, image_stamp_sec=image_stamp_sec,
        now_sec=now_sec, map_bearing_rad=map_bearing_rad, cone_half_angle_rad=cone_half_angle_rad,
        accepted_range_m=accepted_range_m, max_scan_age_sec=max_scan_age_sec,
        max_camera_map_bearing_delta_rad=max_camera_map_bearing_delta_rad,
        target_reconciliation=target_reconciliation, fragmentation=fragmentation, diagnostics=diagnostics)
    if registration is None:
        return None
    result = _support_for_outline(outline, intrinsics=intrinsics, scan_from_camera=scan_from_camera,
        scan=scan, now_sec=now_sec, cone_half_angle_rad=cone_half_angle_rad,
        accepted_range_m=accepted_range_m, max_scan_age_sec=max_scan_age_sec,
        target_reconciliation=target_reconciliation, registration=registration, diagnostics=diagnostics)
    if result is not None and diagnostics is not None:
        diagnostics.clear()
        diagnostics['reason'] = 'current_outline_associated'
    return result


def detect_opposite_target_support(frame, cv2, *, attempt, intrinsics, model_profile,
        scan_from_camera, scan, image_stamp_sec, now_sec, map_bearing_rad,
        cone_half_angle_rad, accepted_range_m, max_scan_age_sec,
        max_camera_map_bearing_delta_rad, resources=None, max_elapsed_sec=.06,
        target_reconciliation=None, fragmentation=None, diagnostics=None):
    """Locate a complete foreground symbol before the payload decoder sees it.

    Ordinary support still requires a unique envelope before native detection;
    each scale is resolved using all existing registration and range gates.
    """
    if attempt is None or max_elapsed_sec <= 0:
        return _miss(diagnostics, 'search_unavailable' if attempt is None else 'support_time_budget_exhausted')
    if not 0 <= now_sec-image_stamp_sec <= max_scan_age_sec:
        return _miss(diagnostics, 'stale_support_image')
    started = time.monotonic()
    registration = _registration_context(scan=scan, image_stamp_sec=image_stamp_sec,
        now_sec=now_sec, map_bearing_rad=map_bearing_rad, cone_half_angle_rad=cone_half_angle_rad,
        accepted_range_m=accepted_range_m, max_scan_age_sec=max_scan_age_sec,
        max_camera_map_bearing_delta_rad=max_camera_map_bearing_delta_rad,
        target_reconciliation=target_reconciliation, fragmentation=fragmentation, diagnostics=diagnostics)
    if registration is None:
        return None
    _miss(diagnostics, 'no_complete_target_outline')
    for outlines in _outline_batches(frame, cv2, attempt=attempt, model_profile=model_profile,
            image_stamp_sec=image_stamp_sec, resources=resources, started=started,
            max_elapsed_sec=max_elapsed_sec):
        choices = []
        for outline in outlines:
            support = _support_for_outline(outline, intrinsics=intrinsics,
                scan_from_camera=scan_from_camera, scan=scan, now_sec=now_sec+time.monotonic()-started,
                cone_half_angle_rad=cone_half_angle_rad, accepted_range_m=accepted_range_m,
                max_scan_age_sec=max_scan_age_sec, target_reconciliation=target_reconciliation,
                registration=registration, diagnostics=diagnostics)
            if support is not None:
                choices.append(support)
        if len(choices) == 1:
            if diagnostics is not None:
                diagnostics.clear()
                diagnostics['reason'] = 'current_outline_associated'
            return choices[0]
        if len(choices) > 1:
            return _miss(diagnostics, 'multiple_supported_outlines')
    return None
