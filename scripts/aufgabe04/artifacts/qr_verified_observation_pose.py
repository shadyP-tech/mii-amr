"""Discovery-only QR receipt retaining the robot observation pose, never a stand yaw.

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
    if (type(data.get("schema_version")) is not int or data["schema_version"] != 1
            or data.get("observation_kind") != OBSERVATION_KIND):
        raise ValueError("unsupported QR observation pose schema")
    if (data.get("stand_axis_rad", 0) is not None or data.get("facing_ready") is not False
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
            or validated_qr_corners(data.get("qr_corners_px"), image_shape=shape) is None):
        raise ValueError("QR observation needs valid current decoded corners")
    binding = data.get("qr_binding")
    if (not isinstance(binding, Mapping) or binding.get("accepted") is not True
            or binding.get("reason") != "decoded_qr_target_associated"
            or type(binding.get("symbol_count")) is not int or binding["symbol_count"] != 1
            or tuple(binding.get("qr_texts_for_evidence") or ()) != (data["qr_id"],)):
        raise ValueError("QR observation needs one independently bound decoded identity")
    _number(binding.get("camera_bearing_rad"), "QR camera bearing")
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
    return validate_qr_verified_observation_pose(content_hashed_payload({
        **fields, "schema_version": 1, "observation_kind": OBSERVATION_KIND,
        "stand_axis_rad": None, "facing_ready": False, "motion_authorized": False,
        "completion_authorized": True, "completion_scope": "discovery_only",
    }, hash_field=HASH_FIELD))


def load_qr_verified_observation_pose(path: Path) -> dict:
    return validate_qr_verified_observation_pose(content_hashed_payload(
        load_content_hashed_json(path, hash_field=HASH_FIELD), hash_field=HASH_FIELD))
