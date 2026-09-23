"""Current QR outline and range support; no payload, head pose, or angle fit."""
from dataclasses import asdict, dataclass
import math
import time

from scripts.aufgabe04.perception.candidate_lidar_association import associate_camera_registered_candidate_lidar_target
from scripts.aufgabe04.real_robot.observer.finite_target_bearing import finite_target_bearing
from scripts.aufgabe04.real_robot.observer.qr_candidate_search import qr_registration_envelope
from scripts.aufgabe04.qr_scanning.qr_observation import validated_qr_corners

POLICY = 'current_opposite_qr_outline_unique_scan'


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
    if not isinstance(value, dict) or value.get('policy') != POLICY or value.get('accepted') is not True:
        raise ValueError('current opposite QR support missing')
    shape = value.get('image_shape')
    if not isinstance(shape, (tuple, list)) or len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError('opposite QR image shape invalid')
    corners = validated_qr_corners(value.get('corners_px'), image_shape=shape)
    if corners is None:
        raise ValueError('opposite QR outline is incomplete')
    height = value.get('expected_symbol_height_px')
    depth = value.get('depth_m')
    stamp = value.get('image_stamp_sec')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (height, depth, stamp)) or min(height, depth) <= 0:
        raise ValueError('opposite QR scale/depth invalid')
    edges = [math.dist(corners[i], corners[(i+1)%4]) for i in range(4)]
    if not .6*height <= min(edges) or max(edges) > 1.4*height:
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
            or cluster.get('eligible_cluster_count') != 1
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
        if (not current.associated or abs(optical_depth-depth)>1e-9
                or current.distance_m != lidar['distance_m']
                or tuple(current.search_association.selected_cluster_source_indices) != tuple(indices)
                or abs(current.camera_map_bearing_delta_rad-lidar['camera_map_bearing_delta_rad'])>1e-9):
            raise ValueError('opposite QR reconciliation differs from current scan')
    return value


def detect_opposite_target_support(frame, cv2, *, attempt, intrinsics, model_profile,
        scan_from_camera, scan, image_stamp_sec, now_sec, map_bearing_rad,
        cone_half_angle_rad, accepted_range_m, max_scan_age_sec,
        max_camera_map_bearing_delta_rad, resources=None, max_elapsed_sec=.06,
        target_reconciliation=None):
    """Locate a complete foreground symbol before the payload decoder sees it.

    Two bounded scale attempts; background-sized symbols cannot become target
    support. Native detection works even on builds without a native QR decoder.
    """
    if attempt is None or max_elapsed_sec <= 0:
        return None
    if not 0 <= now_sec-image_stamp_sec <= max_scan_age_sec:
        return None
    started = time.monotonic()
    roi = attempt.roi
    pixels = frame[roi.y0:roi.y1, roi.x0:roi.x1]
    detector = cv2.QRCodeDetector() if resources is None else resources.decoder('native')
    expected = attempt.expected_head_height_px*model_profile.qr_symbol_height_m/model_profile.head_height_m
    envelope = qr_registration_envelope(scan,map_bearing_rad=map_bearing_rad,
        cone_half_angle_rad=cone_half_angle_rad,max_camera_map_bearing_delta_rad=max_camera_map_bearing_delta_rad,
        accepted_range_m=accepted_range_m,now_sec=now_sec,max_scan_age_sec=max_scan_age_sec)
    if not envelope.associated or envelope.eligible_cluster_count != 1:
        return None
    reference, limit = map_bearing_rad, max_camera_map_bearing_delta_rad
    if target_reconciliation is not None:
        from scripts.aufgabe04.real_robot.observer.target_reconciliation import validate_reconciliation
        try:
            proof_scan, original, _, reference = validate_reconciliation(target_reconciliation,
                image_stamp_sec=image_stamp_sec, scan_stamp_sec=scan.scan_stamp_sec)
            fields = ('scan_stamp_sec','scan_frame_id','selected_cluster_source_indices',
                      'distance_m','accepted_range_m','map_bearing_rad','cone_half_angle_rad')
            if any(getattr(original,k) != getattr(envelope,k) for k in fields):
                return None
            limit = min(cone_half_angle_rad, math.radians(3))
        except (ValueError, TypeError, KeyError, OSError):
            return None
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
        choices = []
        for quad in points:
            corners = validated_qr_corners(tuple((float(p[0])/scale+roi.x0, float(p[1])/scale+roi.y0) for p in quad), image_shape=frame.shape[:2])
            if corners is None:
                continue
            center = tuple(sum(p[k] for p in corners)/4 for k in (0, 1))
            if math.dist(center, (attempt.expected_center_u_px, attempt.expected_center_v_px)) > .75*attempt.expected_head_height_px:
                continue
            try:
                bearing, uncertainty, depth = finite_target_bearing(center_px=center,intrinsics=intrinsics,
                    scan_from_camera=scan_from_camera,distance_m=envelope.distance_m,range_interval_m=accepted_range_m)
            except ValueError:
                continue
            if abs(math.remainder(bearing-reference,math.tau))+uncertainty > limit:
                continue
            lidar = associate_camera_registered_candidate_lidar_target(scan,
                map_bearing_rad=reference, observed_camera_bearing_rad=bearing,
                cone_half_angle_rad=cone_half_angle_rad, accepted_range_m=accepted_range_m,
                now_sec=now_sec+time.monotonic()-started, max_scan_age_sec=max_scan_age_sec,
                min_cluster_sample_count=1, max_camera_map_bearing_delta_rad=limit)
            support = OppositeTargetSupport(corners, center, lidar, image_stamp_sec, tuple(frame.shape[:2]),
                expected, depth, target_reconciliation,
                dict(intrinsics=asdict(intrinsics), scan_from_camera=asdict(scan_from_camera)))
            try:
                validate_target_support(support.metadata())
            except ValueError:
                continue
            choices.append(support)
        if len(choices) == 1:
            return choices[0]
        if len(choices) > 1:
            return None
    return None
