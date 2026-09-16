"""Explicit one-frame front admission, separate from temporal consensus.

The payload retains the measured head quality, current sensor gates and an
independently decoded, candidate-bound QR. It is a checked receipt, not an
authentication mechanism or a robot motion permission.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
import math

from scripts.aufgabe04.perception.stand_axis.head_model_quality import (
    HeadModelQuality, MEASURED_HEAD_AXIS_SOURCE, validated_head_model_quality,
)
from scripts.aufgabe04.real_robot.observer.scan_target_persistence import registered_target_metadata_is_unique

CURRENT_HEAD_FRONT_POLICY = "current_head_and_bound_qr"
MAX_QR_BINDING_AGE_SEC = 1.0
MAX_CURRENT_SOURCE_AGE_SEC = .5
MAX_SENSOR_SKEW_SEC = .1
MAX_FUTURE_STAMP_SEC = .05
CURRENT_HEAD_FRONT_GATES = frozenset({
    "stationary", "synchronized", "lidar_associated", "source_fresh",
    "current_head_boundary", "complete_head", "identity_unambiguous",
})
_FIELDS = frozenset({
    "policy", "source", "head_model_quality", "head_admission", "head_corners_px",
    "qr_corners_px", "image_shape", "model_profile_sha256", "sensor_stamp_sec",
    "scan_stamp_sec", "checked_at_sec", "qr_sensor_stamp_sec", "qr_scan_stamp_sec",
    "qr_checked_at_sec", "qr_id", "qr_binding", "head_lidar_association",
    "camera_yaw_rad", "camera_heading_rad", "stand_axis_rad", "target_key",
    "motion_epoch", "camera_signature", "sample_gate_evidence",
    "axis_sample_count", "confidence_basis", "motion_authorized",
})


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"current head front {name} must be finite")
    return float(value)


def _quad(values):
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError("current head front requires four current image corners")
    points = []
    for point in values:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("current head front image corner must have two coordinates")
        points.append(tuple(_number(v, "image coordinate") for v in point))
    cross = [_cross(a, b, c) for a, b, c in zip(points, points[1:] + points[:1], points[2:] + points[:2])]
    if not (all(v > 1e-6 for v in cross) or all(v < -1e-6 for v in cross)):
        raise ValueError("current head front requires convex nondegenerate corners")
    return points if cross[0] > 0 else list(reversed(points))


def _cross(a, b, p):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _camera_value(value, depth=0):
    if isinstance(value, str):
        return len(value) <= 256  # Empty distortion_model is valid CameraInfo.
    if type(value) in (int, float):
        return math.isfinite(value)
    return (isinstance(value, (list, tuple)) and depth < 4 and len(value) <= 256
            and all(_camera_value(item, depth + 1) for item in value))


def qr_quad_inside_current_head(qr_corners_px, head_corners_px):
    """Use only geometric enclosure; QR size cannot veto a usable head fit."""
    try:
        qr, head = _quad(qr_corners_px), _quad(head_corners_px)
        return all(_cross(a, b, p) >= -1e-6
                   for a, b in zip(head, head[1:] + head[:1]) for p in qr)
    except (TypeError, ValueError):
        return False


def _association(payload, *, stamp, allow_witnessed_head=False):
    if not isinstance(payload, Mapping) or payload.get("associated") is not True:
        raise ValueError("current head front requires an accepted LiDAR association")
    cluster = payload.get("search_association", payload)
    if not isinstance(cluster, Mapping) or cluster.get("associated") is not True:
        raise ValueError("current head front LiDAR cluster is missing")
    witnessed = (allow_witnessed_head and payload.get("witnessed_fragmentation") is not None)
    if witnessed and not registered_target_metadata_is_unique(dict(payload)):
        raise ValueError("current head front witnessed target does not recompute")
    if not witnessed and (cluster.get("eligible_cluster_count") != 1 or type(cluster.get("eligible_cluster_count")) is not int):
        raise ValueError("current head front requires one unique LiDAR cluster")
    if _number(cluster.get("scan_stamp_sec"), "association scan stamp") != stamp:
        raise ValueError("current head front association is bound to another scan")
    indices = cluster.get("selected_cluster_source_indices")
    if not isinstance(indices, (list, tuple)) or not indices or any(type(i) is not int or i < 0 for i in indices):
        raise ValueError("current head front association needs current scan sample indices")
    if not isinstance(cluster.get("scan_frame_id"), str) or not cluster["scan_frame_id"]:
        raise ValueError("current head front association needs a scan frame")
    return cluster


def validated_current_head_front_evidence(payload, *, expected_sensor_stamp_sec=None,
        expected_stand_axis_rad=None, expected_target_key=None, expected_qr_id=None):
    """Validate all one-frame exceptions before a producer or planner uses them."""
    if not isinstance(payload, Mapping) or set(payload) != _FIELDS:
        raise ValueError("current head front evidence has missing or unexpected fields")
    if (payload["policy"] != CURRENT_HEAD_FRONT_POLICY or payload["source"] != MEASURED_HEAD_AXIS_SOURCE
            or payload["axis_sample_count"] != 1 or type(payload["axis_sample_count"]) is not int
            or payload["confidence_basis"] != "current_pixel_quality_not_temporal_consensus"
            or payload["motion_authorized"] is not False):
        raise ValueError("current head front evidence policy is invalid")
    gates = payload["sample_gate_evidence"]
    if not isinstance(gates, Mapping) or set(gates) != CURRENT_HEAD_FRONT_GATES or any(v is not True for v in gates.values()):
        raise ValueError("current head front evidence requires every current sample gate")
    sha = payload["model_profile_sha256"]
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("current head front requires measured profile SHA256")
    try:
        quality = HeadModelQuality(**payload["head_model_quality"])
        valid = validated_head_model_quality(quality)
    except (TypeError, ValueError, AttributeError):
        valid = False
    if not valid or quality.profile_sha256 != sha:
        raise ValueError("current head front requires strict measured head quality")
    admission = payload["head_admission"]
    if (not isinstance(admission, Mapping) or admission.get("accepted") is not True
            or admission.get("reason") != "measured_head_geometry_quality_accepted"
            or admission.get("source") != MEASURED_HEAD_AXIS_SOURCE
            or admission.get("yaw_uncertainty_deg") != quality.yaw_std_deg
            or admission.get("motion_authorized") is not False):
        raise ValueError("current head front requires the measured head admission")
    head = _quad(payload["head_corners_px"])
    shape = payload["image_shape"]
    if (not isinstance(shape, (list, tuple)) or len(shape) != 2
            or any(type(v) is not int or v < 4 for v in shape)
            or any(not (1. <= x <= shape[1]-2. and 1. <= y <= shape[0]-2.) for x, y in head)):
        raise ValueError("current head front requires a complete head inside the image")
    if not qr_quad_inside_current_head(payload["qr_corners_px"], head):
        raise ValueError("current head front QR must lie inside the current head")
    stamps = {key: _number(payload[key], key) for key in (
        "sensor_stamp_sec", "scan_stamp_sec", "checked_at_sec", "qr_sensor_stamp_sec",
        "qr_scan_stamp_sec", "qr_checked_at_sec")}
    if any(v < 0 for v in stamps.values()):
        raise ValueError("current head front sensor times cannot be negative")
    for prefix in ("", "qr_"):
        image, scan, checked = (stamps[prefix+key] for key in ("sensor_stamp_sec", "scan_stamp_sec", "checked_at_sec"))
        if (abs(image-scan) > MAX_SENSOR_SKEW_SEC + 1e-9
                or any(not -MAX_FUTURE_STAMP_SEC <= checked-v <= MAX_CURRENT_SOURCE_AGE_SEC for v in (image, scan))):
            raise ValueError("current head front sources were not fresh and synchronized")
    if not -MAX_FUTURE_STAMP_SEC <= stamps["checked_at_sec"]-stamps["qr_sensor_stamp_sec"] <= MAX_QR_BINDING_AGE_SEC:
        raise ValueError("current head front QR binding exceeds its brief stopped lifetime")
    if stamps["qr_checked_at_sec"] > stamps["checked_at_sec"] + MAX_FUTURE_STAMP_SEC:
        raise ValueError("current head front QR binding was checked in the future")
    if expected_sensor_stamp_sec is not None and stamps["sensor_stamp_sec"] != expected_sensor_stamp_sec:
        raise ValueError("current head front sensor stamp differs from recommendation")
    qr = payload["qr_binding"]
    qr_id = payload["qr_id"]
    if (not isinstance(qr_id, str) or not qr_id.strip() or len(qr_id) > 4096
            or not isinstance(qr, Mapping) or qr.get("accepted") is not True
            or qr.get("reason") != "decoded_qr_target_associated"
            or type(qr.get("symbol_count")) is not int or qr["symbol_count"] != 1
            or tuple(qr.get("qr_texts_for_evidence") or ()) != (qr_id,)):
        raise ValueError("current head front requires one independently bound decoded QR")
    if expected_qr_id is not None and qr_id != expected_qr_id:
        raise ValueError("current head front QR identity differs from recommendation")
    head_cluster = _association(payload["head_lidar_association"], stamp=stamps["scan_stamp_sec"],
                                allow_witnessed_head=True)
    qr_cluster = _association(qr.get("association"), stamp=stamps["qr_scan_stamp_sec"])
    if head_cluster["scan_frame_id"] != qr_cluster["scan_frame_id"]:
        raise ValueError("current head front head and QR use different scan frames")
    if stamps["scan_stamp_sec"] == stamps["qr_scan_stamp_sec"] and not set(
            head_cluster["selected_cluster_source_indices"]).intersection(qr_cluster["selected_cluster_source_indices"]):
        raise ValueError("current head front head and QR use different current scan clusters")
    if not isinstance(payload["target_key"], str) or not payload["target_key"]:
        raise ValueError("current head front requires a target key")
    if expected_target_key is not None and payload["target_key"] != expected_target_key:
        raise ValueError("current head front target differs from recommendation")
    if type(payload["motion_epoch"]) is not int or payload["motion_epoch"] < 0:
        raise ValueError("current head front requires a stationary observation epoch")
    signature = payload["camera_signature"]
    if (not isinstance(signature, (tuple, list)) or not 4 <= len(signature) <= 256
            or not _camera_value(signature)):
        raise ValueError("current head front requires a camera context")
    yaw, heading, axis = (_number(payload[key], key) for key in
                          ("camera_yaw_rad", "camera_heading_rad", "stand_axis_rad"))
    if abs(math.remainder(axis - (heading + yaw - math.pi/2), math.pi)) > 1e-9:
        raise ValueError("current head front camera/map axis conversion differs")
    if expected_stand_axis_rad is not None and abs(math.remainder(axis-expected_stand_axis_rad, math.pi)) > 1e-9:
        raise ValueError("current head front axis differs from recommendation")
    return deepcopy(dict(payload))


def build_current_head_front_evidence(**fields):
    """Build explicit evidence after the observer's independent runtime gates."""
    quality = fields.get("head_model_quality")
    if isinstance(quality, HeadModelQuality):
        fields["head_model_quality"] = asdict(quality)
    payload = {**fields, "policy": CURRENT_HEAD_FRONT_POLICY, "source": MEASURED_HEAD_AXIS_SOURCE,
               "axis_sample_count": 1, "confidence_basis": "current_pixel_quality_not_temporal_consensus",
               "motion_authorized": False}
    return validated_current_head_front_evidence(payload)
