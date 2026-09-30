"""Current decoded QR ray for framing recovery, with no stand-angle authority.

This deliberately covers only independently decoded symbols whose current
scan has a complete stopped-target reconciliation. Identity and angle evidence
continue through their existing admission paths.
"""
from dataclasses import asdict, dataclass
import math

from scripts.aufgabe04.perception.candidate_lidar_association import (
    CandidateLidarAssociation, CameraRegisteredCandidateLidarAssociation,
    associate_camera_registered_candidate_lidar_target,
)
from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
from scripts.aufgabe04.qr_scanning.qr_observation import validated_qr_corners
from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics
from scripts.aufgabe04.real_robot.observer.finite_target_bearing import finite_target_bearing
from scripts.aufgabe04.real_robot.observer.qr_candidate_search import qr_registration_envelope
from scripts.aufgabe04.real_robot.observer.shared_scan_cluster import (
    _equal, bind_ray_to_envelope, envelope_is_unique,
)
from scripts.aufgabe04.real_robot.observer.target_reconciliation import (
    POLICY as RECONCILIATION_POLICY, validate_reconciliation,
)

POLICY = 'current_decoded_qr_unique_reconciled_scan'


def _same_sensor_result(expected, actual):
    """Preserve structure and discrete fields across platform float rounding."""
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and expected.keys() == actual.keys()
                and all(_same_sensor_result(v, actual[k]) for k, v in expected.items()))
    if isinstance(expected, (tuple, list)):
        return (isinstance(actual, (tuple, list)) and len(expected) == len(actual)
                and all(_same_sensor_result(a, b) for a, b in zip(expected, actual)))
    if type(expected) is float:
        return (type(actual) in (int, float) and math.isfinite(expected)
                and math.isfinite(actual) and abs(expected-actual) <= 1e-9)
    return type(expected) is type(actual) and expected == actual


@dataclass(frozen=True)
class QrTargetSupport:
    corners_px: tuple
    full_image_center_px: tuple
    lidar_association: object
    image_stamp_sec: float
    image_shape: tuple
    target_reconciliation: dict
    finite_bearing: dict
    qr_binding: dict

    @property
    def accepted(self):
        return True

    def metadata(self):
        return dict(policy=POLICY, accepted=True, corners_px=self.corners_px,
            center_px=self.full_image_center_px, image_shape=self.image_shape,
            image_stamp_sec=self.image_stamp_sec,
            lidar_association=asdict(self.lidar_association),
            target_reconciliation=self.target_reconciliation,
            finite_bearing=self.finite_bearing, qr_binding=self.qr_binding,
            supplies_angle=False, supplies_identity=False)


def _validated_lidar(value):
    if (not isinstance(value, dict) or value.get('policy') != POLICY
            or value.get('accepted') is not True
            or value.get('supplies_angle') is not False
            or value.get('supplies_identity') is not False):
        raise ValueError('current decoded QR framing support missing')
    shape = value['image_shape']
    if (not isinstance(shape, (tuple, list)) or len(shape) != 2
            or any(type(v) is not int or v <= 0 for v in shape)):
        raise ValueError('QR framing image shape invalid')
    corners = validated_qr_corners(value['corners_px'], image_shape=shape)
    if corners is None:
        raise ValueError('QR framing requires a complete decoded quadrilateral')
    center = tuple(sum(p[k] for p in corners)/4 for k in (0, 1))
    if tuple(value['center_px']) != center:
        raise ValueError('QR framing center differs from decoded quadrilateral')
    binding = value['qr_binding']
    texts = binding['qr_texts_for_evidence']
    if (binding.get('accepted') is not True
            or binding.get('reason') != 'decoded_qr_target_associated'
            or type(binding.get('symbol_count')) is not int or binding['symbol_count'] != 1
            or not isinstance(texts, (tuple, list)) or len(texts) != 1
            or not isinstance(texts[0], str) or not texts[0].strip()
            or binding.get('current_head_binding') is not None
            or binding.get('range_resolution') is not None
            or binding.get('motion_authorized') is not False
            or binding.get('completion_authorized') is not False):
        raise ValueError('QR framing requires independently bound current decoded identity')
    proof, finite, lidar = (value[k] for k in (
        'target_reconciliation', 'finite_bearing', 'lidar_association'))
    if (not _equal(proof, binding['target_reconciliation'])
            or not _equal(finite, binding['finite_bearing'])
            or not _equal(lidar, binding['association'])
            or proof.get('policy') != RECONCILIATION_POLICY
            or proof.get('retained_orientation') is not None):
        raise ValueError('QR framing differs from its ordinary independent binding')
    cluster = lidar['search_association']
    stamp = value['image_stamp_sec']
    if type(stamp) not in (int, float) or not math.isfinite(stamp):
        raise ValueError('QR framing image timestamp invalid')
    scan, envelope, _, reference = validate_reconciliation(proof,
        image_stamp_sec=stamp, scan_stamp_sec=cluster['scan_stamp_sec'])
    if not envelope_is_unique(envelope):
        raise ValueError('QR framing requires one reconciled current cluster')
    cone = cluster['cone_half_angle_rad']
    minimum = cluster['min_cluster_sample_count']
    if (type(cone) not in (int, float) or not math.isfinite(cone)
            or not 0 < cone <= math.radians(3)+1e-9
            or cone != proof['entries'][-1]['options']['cone_half_angle_rad']
            or lidar['max_camera_map_bearing_delta_rad'] != cone
            or type(minimum) is not int or minimum < 1):
        raise ValueError('QR framing association bounds changed')
    intrinsics = CameraIntrinsics(**finite['intrinsics'])
    camera = RigidTransform(**finite['scan_from_camera'])
    if (finite.get('policy') != 'calibrated_scan_range_ray'
            or (intrinsics.height_px, intrinsics.width_px) != tuple(shape)
            or camera.parent_frame != scan.scan_frame_id
            or finite['range_m'] != envelope.distance_m
            or tuple(finite['range_interval_m']) != tuple(envelope.accepted_range_m)):
        raise ValueError('QR framing calibrated range geometry changed')
    bearing, uncertainty, depth = finite_target_bearing(center_px=center,
        intrinsics=intrinsics, scan_from_camera=camera,
        distance_m=envelope.distance_m, range_interval_m=envelope.accepted_range_m)
    if (any(type(actual) not in (int, float) or not math.isfinite(actual)
            or abs(expected-actual) > 1e-9 for expected, actual in (
                (bearing, binding['camera_bearing_rad']), (bearing, finite['bearing_rad']),
                (uncertainty, finite['uncertainty_rad']), (depth, finite['optical_depth_m'])))
            or abs(math.remainder(bearing-reference, math.tau))+uncertainty > cone+1e-9):
        raise ValueError('QR framing ray misses reconciled target')
    # The binding runs after reconciliation. Reproduce its actual checked time,
    # rather than silently replacing later processing age with an earlier age.
    age = cluster['scan_age_sec']
    if type(age) not in (int, float) or not math.isfinite(age) or not 0 <= age <= .5:
        raise ValueError('QR framing scan was not fresh')
    checked = scan.receipt_sec + age
    if (checked+1e-9 < proof['entries'][-1]['checked_at_sec']
            or any(not 0 <= checked-source <= .5 for source in (stamp, scan.scan_stamp_sec))):
        raise ValueError('QR framing sources were not fresh together')
    registration = binding.get('independent_registration')
    if registration is not None:
        entry = proof['entries'][-1]
        original = qr_registration_envelope(scan, **entry['options'], now_sec=checked,
            max_scan_age_sec=.5, fragmentation=entry.get('fragmentation'))
        if (registration.get('policy') != 'decoded_quad_unique_registration_envelope'
                or registration.get('head_geometry_required') is not False
                or registration.get('motion_authorized') is not False
                or not envelope_is_unique(original)
                or not _same_sensor_result(asdict(original), registration.get('envelope'))):
            raise ValueError('QR framing independent registration envelope changed')
    current = associate_camera_registered_candidate_lidar_target(scan,
        map_bearing_rad=reference, observed_camera_bearing_rad=bearing,
        cone_half_angle_rad=cone, accepted_range_m=envelope.accepted_range_m,
        now_sec=checked, max_scan_age_sec=.5, min_cluster_sample_count=minimum,
        max_camera_map_bearing_delta_rad=cone)
    current = bind_ray_to_envelope(current, scan, envelope, now_sec=checked, max_scan_age_sec=.5)
    if not current.associated or not _same_sensor_result(asdict(current), lidar):
        raise ValueError('QR framing association differs from current raw scan')
    # Keep the originally recorded floats in the persisted independent binding.
    # A replay on another platform may round atan2's last bit differently.
    return CameraRegisteredCandidateLidarAssociation(**{
        **lidar, 'search_association': CandidateLidarAssociation(**cluster)})


def validate_qr_target_support(value):
    """Recompute the persisted decoded ray and complete current scan proof."""
    try:
        _validated_lidar(value)
    except (TypeError, KeyError, AttributeError, ArithmeticError, OSError) as exc:
        raise ValueError('invalid current QR framing proof') from exc
    return value


def prepare_qr_target_support(current):
    """Use one prepared independent QR frame; do not infer a head or stand angle."""
    if current is None:
        return None
    try:
        binding = current.qr_binding.metadata()
        proof = binding['target_reconciliation']
        if (current.retained_backside_orientation is not None
                or current.arrival_target_reconciliation is not None
                or current.qr_corners is None
                or set(current.observed_qr_texts) != set(binding['qr_texts_for_evidence'])
                or current.target_key != proof['target_key']
                or tuple(proof['entries'][-1]['robot_pose']) != tuple(
                    getattr(current.robot_pose, k) for k in ('x_m', 'y_m', 'yaw_rad'))
                or current.scan_stamp_sec != binding['association']['search_association']['scan_stamp_sec']):
            return None
        corners = tuple(tuple(p) for p in current.qr_corners)
        fields = dict(policy=POLICY, accepted=True, corners_px=corners,
            center_px=tuple(sum(p[k] for p in corners)/4 for k in (0, 1)),
            image_shape=tuple(current.image_shape), image_stamp_sec=current.stamp_sec,
            lidar_association=binding['association'], target_reconciliation=proof,
            finite_bearing=binding['finite_bearing'], qr_binding=binding,
            supplies_angle=False, supplies_identity=False)
        lidar = _validated_lidar(fields)
        return QrTargetSupport(corners, fields['center_px'], lidar, current.stamp_sec,
            fields['image_shape'], proof, fields['finite_bearing'], binding)
    except (ValueError, TypeError, KeyError, AttributeError, ArithmeticError, OSError):
        return None
