"""Hashed, motion-neutral evidence for selecting another candidate view.

This receipt is deliberately weaker than a backside axis or QR recommendation.
Its angle and identity fields describe inspection progress and cannot commit a
station identity or authorize a directed opposite-face route.
"""

from collections.abc import Mapping
import math
from pathlib import Path
import re

from scripts.aufgabe04.artifacts.content_store import (
    content_hashed_payload,
    load_content_hashed_json,
    payload_sha256,
)

HASH_FIELD = "inspection_observation_sha256"
OBSERVATION_KIND = "candidate_inspection_progress"
CLASSIFICATIONS = frozenset({
    "front_readable", "front_unreadable", "backside_unresolved", "edge_on", "oblique", "unobservable",
})
MIN_PROGRESS_FRAMES = 7
MIN_PROGRESS_SPAN_SEC = 2.0


def _finite(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def validate_candidate_inspection_observation(payload: Mapping) -> dict:
    if not isinstance(payload, Mapping):
        raise ValueError("inspection observation must be an object")
    data = dict(payload)
    stored = data.pop(HASH_FIELD, None)
    if not isinstance(stored, str) or stored != payload_sha256(data):
        raise ValueError("inspection observation hash mismatch")
    if data.get("schema_version") != 1 or isinstance(data.get("schema_version"), bool):
        raise ValueError("unsupported inspection observation schema")
    if data.get("observation_kind") != OBSERVATION_KIND:
        raise ValueError("unsupported inspection observation kind")
    for field in ("motion_authorized", "completion_authorized"):
        if data.get(field) is not False:
            raise ValueError(f"inspection {field} must be false")
    if data.get("angle_authority") != "advisory_camera_relative_only":
        raise ValueError("inspection angle authority is invalid")
    for field in ("candidate_uid", "stream_id", "planning_frame"):
        if not isinstance(data.get(field), str) or not data[field].strip():
            raise ValueError(f"inspection {field} is missing")
    if data.get("classification") not in CLASSIFICATIONS:
        raise ValueError("invalid inspection classification")
    reasons = data.get("reasons")
    if not isinstance(reasons, list) or not reasons or any(not isinstance(s, str) or not s for s in reasons):
        raise ValueError("inspection reasons must be nonempty strings")
    for field, keys in (("stand_center", ("x_m", "y_m")), ("robot_pose", ("x_m", "y_m", "yaw_rad"))):
        value = data.get(field)
        if not isinstance(value, Mapping):
            raise ValueError(f"inspection {field} must be an object")
        for key in keys:
            _finite(value.get(key), f"{field}.{key}")
    for field in ("robot_profile_sha256", "calibration_profile_sha256", "stand_model_profile_sha256"):
        if not isinstance(data.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", data[field]):
            raise ValueError(f"invalid inspection {field}")
    stamps = data.get("sensor_stamps_sec")
    if not isinstance(stamps, list) or len(stamps) < MIN_PROGRESS_FRAMES:
        raise ValueError("insufficient inspection sensor stamps")
    values = [_finite(s, "sensor stamp") for s in stamps]
    if values != sorted(set(values)) or values[0] < 0:
        raise ValueError("inspection stamps must be distinct and increasing")
    if values[-1] - values[0] + 1e-9 < MIN_PROGRESS_SPAN_SEC:
        raise ValueError("inspection observation span is too short")
    if isinstance(data.get("sample_count"), bool) or not isinstance(data.get("sample_count"), int) or data.get("sample_count") != len(values):
        raise ValueError("inspection sample count differs from stamps")
    if data.get("sensor_stamp_sec") != values[-1] or data.get("first_sensor_stamp_sec") != values[0]:
        raise ValueError("inspection sensor timestamp bounds mismatch")
    gates = data.get("sample_gate_evidence")
    if not isinstance(gates, Mapping) or any(gates.get(k) is not True for k in (
        "all_samples_stationary", "all_samples_synchronized", "all_samples_lidar_associated", "all_samples_fresh",
    )):
        raise ValueError("inspection observation lacks sensor gates")
    qr_id = data.get("qr_id")
    if qr_id is not None and (not isinstance(qr_id, str) or not qr_id.strip()):
        raise ValueError("invalid inspection QR identity")
    qr_count = data.get("qr_sample_count")
    if isinstance(qr_count, bool) or not isinstance(qr_count, int) or qr_count < 0:
        raise ValueError("invalid inspection QR sample count")
    if qr_id is not None and qr_count < 2:
        raise ValueError("inspection QR identity needs a current repeated latch")
    if data.get("qr_identity_authority") != "current_stationary_latch_only":
        raise ValueError("invalid inspection QR authority")
    yaw = data.get("camera_relative_yaw_rad")
    uncertainty = data.get("yaw_uncertainty_rad")
    if (yaw is None) != (uncertainty is None):
        raise ValueError("inspection yaw and uncertainty must appear together")
    if yaw is not None:
        if abs(_finite(yaw, "camera relative yaw")) > math.pi:
            raise ValueError("inspection yaw outside normalized range")
        if not math.radians(15) <= _finite(uncertainty, "yaw uncertainty") <= math.pi:
            raise ValueError("inspection yaw uncertainty is too small or invalid")
    return {**data, HASH_FIELD: stored}


def build_candidate_inspection_observation(**fields) -> dict:
    payload = content_hashed_payload({
        "schema_version": 1,
        "observation_kind": OBSERVATION_KIND,
        "motion_authorized": False,
        "completion_authorized": False,
        "angle_authority": "advisory_camera_relative_only",
        "qr_identity_authority": "current_stationary_latch_only",
        **fields,
    }, hash_field=HASH_FIELD)
    return validate_candidate_inspection_observation(payload)


def load_candidate_inspection_observation(path: Path) -> dict:
    return validate_candidate_inspection_observation(
        content_hashed_payload(
            load_content_hashed_json(path, hash_field=HASH_FIELD),
            hash_field=HASH_FIELD,
        )
    )
