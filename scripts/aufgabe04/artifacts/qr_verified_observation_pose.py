"""Discovery-only QR receipt with optional certified backside orientation.

A content hash protects transport integrity. It is not a motion permission. The
pose was checked at the original image time; consumers must admit any later
navigation independently rather than treating this historical receipt as live TF.
"""

from collections.abc import Mapping
from copy import deepcopy
import math
from pathlib import Path
import re

from scripts.aufgabe04.artifacts.content_store import (
    content_hashed_payload, load_content_hashed_json, payload_sha256,
)
from scripts.aufgabe04.qr_scanning.qr_observation import validated_qr_corners
from scripts.aufgabe04.navigation.foundation.models import Pose2D

HASH_FIELD = "qr_verified_observation_pose_sha256"
OBSERVATION_KIND = "qr_verified_observation_pose"
SOURCE_GATES = frozenset({"stationary", "synchronized", "lidar_associated", "source_fresh",
                          "identity_unambiguous", "exact_time_localization"})


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"QR observation {name} must be finite")
    return float(value)


def _camera_context(value, depth=0):
    if isinstance(value, str):
        return len(value) <= 256
    if type(value) in (int, float):
        return math.isfinite(value)
    return (isinstance(value, (list, tuple)) and depth < 4 and len(value) <= 256
            and all(_camera_context(item, depth + 1) for item in value))


def validate_qr_verified_observation_pose(payload: Mapping) -> dict:
    if not isinstance(payload, Mapping):
        raise ValueError("QR observation pose must be an object")
    data = deepcopy(dict(payload))
    stored = data.pop(HASH_FIELD, None)
    if not isinstance(stored, str) or stored != payload_sha256(data):
        raise ValueError("QR observation pose hash mismatch")
    if (type(data.get("schema_version")) is not int or data["schema_version"] not in (1, 2)
            or data.get("observation_kind") != OBSERVATION_KIND):
        raise ValueError("unsupported QR observation pose schema")
    retained = data.get("retained_backside_orientation")
    crop_identity = data['schema_version'] == 2
    if ((not crop_identity and (data.get("stand_axis_rad", 0) is not None or retained is not None))
            or data.get("facing_ready") is not False
            or data.get("motion_authorized") is not False
            or data.get("completion_authorized") is not True
            or data.get("completion_scope") != "discovery_only"):
        raise ValueError("QR observation pose grants discovery completion only")
    for field in ("candidate_uid", "stream_id", "planning_frame", "qr_id", "target_key"):
        if not isinstance(data.get(field), str) or not data[field].strip() or len(data[field]) > 4096:
            raise ValueError(f"QR observation {field} is missing or invalid")
    for field, keys in (("stand_center", ("x_m", "y_m")),
                        ("robot_pose", ("x_m", "y_m", "yaw_rad"))):
        value = data.get(field)
        if not isinstance(value, Mapping) or set(value) != set(keys):
            raise ValueError(f"QR observation {field} is invalid")
        for key in keys:
            _number(value[key], f"{field}.{key}")
    for field in ("robot_profile_sha256", "calibration_profile_sha256", "stand_model_profile_sha256"):
        if not isinstance(data.get(field), str) or not re.fullmatch("[0-9a-f]{64}", data[field]):
            raise ValueError(f"QR observation {field} is invalid")
    image, scan, checked = (_number(data.get(key), key)
                           for key in ("sensor_stamp_sec", "scan_stamp_sec", "checked_at_sec"))
    if min(image, scan, checked) < 0 or abs(image - scan) > .1 + 1e-9:
        raise ValueError("QR observation source timestamps are not synchronized")
    if any(not -.05 <= checked - stamp <= .5 for stamp in (image, scan)):
        raise ValueError("QR observation sources were not fresh when checked")
    shape = data.get("image_shape")
    if (not isinstance(shape, (list, tuple)) or len(shape) != 2
            or any(type(v) is not int or v <= 0 for v in shape)
            or (not crop_identity and validated_qr_corners(data.get("qr_corners_px"), image_shape=shape) is None)):
        raise ValueError("QR observation needs valid current decoded corners")
    binding = data.get("qr_binding")
    if (not isinstance(binding, Mapping) or binding.get("accepted") is not True
            or binding.get("reason") != ("decoded_qr_exclusive_opposite_crop" if crop_identity else "decoded_qr_target_associated")
            or type(binding.get("symbol_count")) is not int or binding["symbol_count"] != 1
            or tuple(binding.get("qr_texts_for_evidence") or ()) != (data["qr_id"],)):
        raise ValueError("QR observation needs one independently bound decoded identity")
    if not crop_identity:
        _number(binding.get("camera_bearing_rad"), "QR camera bearing")
    else:
        from scripts.aufgabe04.artifacts.retained_backside_orientation import (
            validate_retained_orientation, opposite_view_matches,
        )
        validate_retained_orientation(retained, candidate_uid=data['candidate_uid'],
            planning_frame=data['planning_frame'], stand_center=data['stand_center'],
            model_sha256=data['stand_model_profile_sha256'])
        if data.get('stand_axis_rad') != retained['stand_axis_rad']:
            raise ValueError('QR receipt must retain the certified backside axis')
        if any(data[key] != retained[key] for key in ('robot_profile_sha256', 'calibration_profile_sha256')):
            raise ValueError('retained orientation calibration/robot profile changed')
        if not opposite_view_matches(retained, Pose2D(**data['robot_pose'])):
            raise ValueError('QR observation is outside the certified opposite side')
        crop = binding.get('current_head_binding')
        _validate_opposite_crop(crop, data, image, scan, shape)
    association = binding.get("association")
    if not isinstance(association, Mapping) or association.get("associated") is not True:
        raise ValueError("QR observation needs an accepted LiDAR association")
    cluster = association.get("search_association", association)
    if (not isinstance(cluster, Mapping) or cluster.get("associated") is not True
            or type(cluster.get("eligible_cluster_count")) is not int
            or cluster["eligible_cluster_count"] != 1
            or cluster.get("scan_stamp_sec") != scan
            or not isinstance(cluster.get("scan_frame_id"), str)
            or not cluster["scan_frame_id"]):
        raise ValueError("QR observation needs one unique current LiDAR cluster")
    indices = cluster.get("selected_cluster_source_indices")
    if (not isinstance(indices, (list, tuple)) or not indices
            or any(type(index) is not int or index < 0 for index in indices)
            or len(indices) != len(set(indices))):
        raise ValueError("QR observation needs current LiDAR sample indices")
    registration = binding.get("independent_registration")
    if registration is not None:
        envelope = registration.get("envelope") if isinstance(registration, Mapping) else None
        if (not isinstance(envelope, Mapping)
                or registration.get("policy") != "decoded_quad_unique_registration_envelope"
                or envelope.get("associated") is not True
                or type(envelope.get("eligible_cluster_count")) is not int
                or envelope["eligible_cluster_count"] != 1
                or envelope.get("min_cluster_sample_count") != 1
                or envelope.get("scan_stamp_sec") != scan
                or envelope.get("scan_frame_id") != cluster["scan_frame_id"]
                or envelope.get("accepted_range_m") != cluster.get("accepted_range_m")):
            raise ValueError("independent QR registration needs its unique current search envelope")
        envelope_indices = envelope.get("selected_cluster_source_indices")
        if (not isinstance(envelope_indices, (list, tuple)) or not envelope_indices
                or any(type(i) is not int or i < 0 for i in envelope_indices)
                or len(envelope_indices) != len(set(envelope_indices))
                or not set(indices).issubset(envelope_indices)):
            raise ValueError("independent QR registration must retain the same cluster")
    reconciliation = binding.get('target_reconciliation')
    finite = binding.get('finite_bearing')
    if reconciliation is not None:
        from scripts.aufgabe04.real_robot.observer.target_reconciliation import validate_reconciliation
        proof_scan, envelope, _, reference = validate_reconciliation(reconciliation,
            candidate_uid=data['candidate_uid'],stand_center=tuple(data['stand_center'][k] for k in ('x_m','y_m')),
            image_stamp_sec=image,scan_stamp_sec=scan)
        if (reconciliation['target_key'] != data['target_key'] or reconciliation['epoch'] != data['motion_epoch']
                or reconciliation['planning_frame'] != data['planning_frame']
                or tuple(reconciliation['entries'][-1]['robot_pose']) != tuple(data['robot_pose'][k] for k in ('x_m','y_m','yaw_rad'))
                or not set(indices).issubset(envelope.selected_cluster_source_indices)
                or association.get('map_bearing_rad') != reference or finite is None):
            raise ValueError('QR reconciliation differs from admitted tuple')
        if (finite['range_m'] != envelope.distance_m
                or tuple(finite['range_interval_m']) != tuple(envelope.accepted_range_m)
                or abs(association.get('max_camera_map_bearing_delta_rad',math.inf)-math.radians(3)) > 1e-9):
            raise ValueError('QR reconciliation range or cone changed')
        from scripts.aufgabe04.perception.candidate_lidar_association import associate_camera_registered_candidate_lidar_target
        recomputed = associate_camera_registered_candidate_lidar_target(proof_scan,
            map_bearing_rad=reference,observed_camera_bearing_rad=binding['camera_bearing_rad'],
            cone_half_angle_rad=math.radians(3),accepted_range_m=envelope.accepted_range_m,
            now_sec=checked,max_scan_age_sec=.5,min_cluster_sample_count=1,
            max_camera_map_bearing_delta_rad=math.radians(3))
        if (not recomputed.associated or tuple(indices) != recomputed.search_association.selected_cluster_source_indices
                or association['distance_m'] != recomputed.distance_m):
            raise ValueError('QR reconciliation current scan no longer admits its ray')
    if finite is not None:
        from scripts.aufgabe04.real_robot.observer.finite_target_bearing import finite_target_bearing
        from scripts.aufgabe04.real_robot.configuration.geometry import CameraIntrinsics
        from scripts.aufgabe04.perception.stand_axis_handoff import RigidTransform
        corners = data['qr_corners_px']
        bearing, uncertainty, depth = finite_target_bearing(
            center_px=tuple(sum(p[k] for p in corners)/4 for k in (0,1)),
            intrinsics=CameraIntrinsics(**finite['intrinsics']),
            scan_from_camera=RigidTransform(**finite['scan_from_camera']),
            distance_m=finite['range_m'],range_interval_m=finite['range_interval_m'])
        if (any(abs(a-b)>1e-9 for a,b in ((bearing,binding['camera_bearing_rad']),
                (bearing,finite['bearing_rad']),(uncertainty,finite['uncertainty_rad']),(depth,finite['optical_depth_m'])))
                or abs(math.remainder(bearing-association['map_bearing_rad'],math.tau))+uncertainty
                    > association.get('max_camera_map_bearing_delta_rad',association.get('cone_half_angle_rad',0.))+1e-9):
            raise ValueError('QR finite ray differs from current calibrated geometry')
    if type(data.get("motion_epoch")) is not int or data["motion_epoch"] < 0:
        raise ValueError("QR observation needs a stopped motion epoch")
    signature = data.get("camera_signature")
    if (not isinstance(signature, (list, tuple)) or not 4 <= len(signature) <= 256
            or not _camera_context(signature)):
        raise ValueError("QR observation needs its calibrated camera context")
    gates = data.get("source_gates")
    if not isinstance(gates, Mapping) or set(gates) != SOURCE_GATES or any(v is not True for v in gates.values()):
        raise ValueError("QR observation is missing current source gates")
    provenance = data.get("localization_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("QR observation localization provenance missing")
    for field in ("map_frame", "base_frame", "scan_frame", "camera_frame"):
        if not isinstance(provenance.get(field), str) or not provenance[field]:
            raise ValueError("QR observation localization frames are missing")
    if (provenance["map_frame"] != data["planning_frame"]
            or provenance["scan_frame"] != cluster["scan_frame_id"]
            or provenance.get("exact_image_transform_stamp_sec") != image
            or provenance.get("exact_scan_transform_stamp_sec") != scan):
        raise ValueError("QR observation localization does not match its source tuple")
    return {**data, HASH_FIELD: stored}


def build_qr_verified_observation_pose(**fields) -> dict:
    retained = fields.get('retained_backside_orientation')
    return validate_qr_verified_observation_pose(content_hashed_payload({
        **fields, "schema_version": 1 if retained is None else 2, "observation_kind": OBSERVATION_KIND,
        "stand_axis_rad": None if retained is None else retained['stand_axis_rad'],
        "facing_ready": False, "motion_authorized": False,
        "completion_authorized": True, "completion_scope": "discovery_only",
    }, hash_field=HASH_FIELD))


def load_qr_verified_observation_pose(path: Path) -> dict:
    return validate_qr_verified_observation_pose(content_hashed_payload(
        load_content_hashed_json(path, hash_field=HASH_FIELD), hash_field=HASH_FIELD))


def _validate_opposite_crop(crop, data, image, scan, shape):
    if (not isinstance(crop, Mapping) or crop.get('accepted') is not True
            or crop.get('policy') != 'opposite_current_scan_exclusive_identity_crop'
            or crop.get('candidate_uid') != data['candidate_uid']
            or crop.get('image_stamp_sec') != image or crop.get('scan_stamp_sec') != scan):
        raise ValueError('QR text requires its current exclusive candidate crop')
    box = crop.get('bounds_xyxy')
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or any(type(x) is not int for x in box)
            or not 0 <= box[0] < box[2] <= shape[1] or not 0 <= box[1] < box[3] <= shape[0]):
        raise ValueError('QR identity crop bounds invalid')
    center = crop.get('target_center_px')
    if (not isinstance(center, (list, tuple)) or len(center) != 2
            or not box[0] < _number(center[0], 'crop center') < box[2]
            or not box[1] < _number(center[1], 'crop center') < box[3]):
        raise ValueError('QR identity crop excludes target center')
    support = crop.get('target_support')
    isolated = crop.get('sampling') == 'isolated_current_qr_quad'
    if isolated:
        from scripts.aufgabe04.real_robot.observer.opposite_target_support import validate_target_support
        validate_target_support(support)
        proof = support.get('target_reconciliation')
        if proof is not None and (proof['candidate_uid'] != data['candidate_uid']
                or proof['target_key'] != data['target_key'] or proof['epoch'] != data['motion_epoch']
                or proof['planning_frame'] != data['planning_frame']
                or any(proof['stand_center'][i] != data['stand_center'][k] for i,k in enumerate(('x_m','y_m')))
                or tuple(proof['entries'][-1]['robot_pose']) != tuple(data['robot_pose'][k] for k in ('x_m','y_m','yaw_rad'))):
            raise ValueError('opposite target proof differs from candidate epoch')
        cluster = support['lidar_association']['search_association']
        if (support['image_stamp_sec'] != image or tuple(support['image_shape']) != tuple(shape)
                or tuple(support['center_px']) != tuple(center)
                or cluster['scan_stamp_sec'] != scan
                or cluster['scan_frame_id'] != data['qr_binding']['association']['scan_frame_id']
                or not set(cluster['selected_cluster_source_indices']).issubset(
                    data['qr_binding']['association']['selected_cluster_source_indices'])
                or any(not box[0] <= p[0] < box[2] or not box[1] <= p[1] < box[3]
                       for p in support['corners_px'])):
            raise ValueError('isolated QR support differs from current crop')
        target_depth = crop.get('target_depth_interval_m')
        _depth_interval(target_depth)
        if not target_depth[0] <= support['depth_m'] <= target_depth[1]:
            raise ValueError('isolated target depth is outside its uncertainty interval')
    competitors = crop.get('competitors')
    if not isinstance(competitors, list):
        raise ValueError('QR identity crop lacks neighbor review')
    for competitor in competitors:
        other = competitor.get('bounds_xyxy') if isinstance(competitor, Mapping) else None
        if (not isinstance(other, (list, tuple)) or len(other) != 4
                or any(type(x) is not int for x in other)
                or other[0] >= other[2] or other[1] >= other[3]):
            raise ValueError('QR identity crop overlaps a neighboring candidate')
        overlap = box[0] < other[2] and other[0] < box[2] and box[1] < other[3] and other[1] < box[3]
        if overlap:
            if not isolated or competitor.get('occluded_by_target_symbol') is not True:
                raise ValueError('QR identity crop overlaps a neighboring candidate')
            _depth_interval(competitor.get('depth_interval_m'))
            if competitor['depth_interval_m'][0] <= target_depth[1]:
                raise ValueError('QR neighbor depth is not separated from target')
    search = crop.get('search')
    if (not isinstance(search, Mapping) or search.get('accepted') is not True
            or search.get('envelope') != data['qr_binding']['association']):
        raise ValueError('QR identity crop differs from current scan search')


def _depth_interval(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
            or not 0 <= value[0] < value[1]):
        raise ValueError('QR crop depth interval invalid')
