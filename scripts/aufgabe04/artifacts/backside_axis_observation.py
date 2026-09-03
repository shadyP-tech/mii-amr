"""Shared, ROS-free contract for model-backed backside-axis evidence.

The passive camera observer produces this receipt and navigation consumes it.
Keeping the schema, thresholds, validation, and opposite-face geometry here
prevents those two layers from drifting while leaving the artifact incapable
of commanding motion on its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re


LEGACY_PASSIVE_VIEWPOINT_OBSERVER_VERSION = (
    "aufgabe04-real-passive-viewpoint-v6-backside-model-evidence"
)
PASSIVE_VIEWPOINT_OBSERVER_VERSION = (
    "aufgabe04-real-passive-viewpoint-v7-bounded-camera-lidar-registration"
)
LEGACY_BACKSIDE_AXIS_OBSERVATION_SCHEMA_VERSION = 2
BACKSIDE_AXIS_OBSERVATION_SCHEMA_VERSION = 3
BACKSIDE_AXIS_OBSERVATION_KIND = "real_stand_backside_axis_without_qr"
REAL_STAND_AXIS_OBSERVATION_KIND = BACKSIDE_AXIS_OBSERVATION_KIND
BACKSIDE_AXIS_SAMPLE_SOURCE = "model_backside_current_frame"
BACKSIDE_CURRENT_FRAME_SOURCE = BACKSIDE_AXIS_SAMPLE_SOURCE
REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE = (
    "model_backside_current_frame_bounded_camera_lidar_registered"
)
BACKSIDE_MODEL_EVIDENCE_STATE = "fresh_backside"
BACKSIDE_VISIBLE_FACE = "backside_candidate"
BACKSIDE_CLASSIFICATION_BASIS = (
    "measured_head_geometry_plus_repeated_qr_marker_absence"
)

# The estimator and both receipt boundaries deliberately share these gates.
# Face classification is stronger than axial consensus because the former is
# what authorizes selecting the antipodal inspection side.
MINIMUM_BACKSIDE_FACE_CONFIDENCE = 0.70
MINIMUM_BACKSIDE_AXIS_CONFIDENCE = 0.60
MINIMUM_BACKSIDE_AXIS_SAMPLE_COUNT = 2
MINIMUM_HEAD_SCALE_RATIO = 0.60
MAXIMUM_HEAD_SCALE_RATIO = 1.35
MAXIMUM_HEAD_CENTER_ERROR_RATIO = 0.55
MAXIMUM_REGISTRATION_HEAD_CENTER_OFFSET_RATIO = 1.50
MAXIMUM_REGISTRATION_BEARING_DELTA_RAD = math.radians(12.0)
TARGET_REGISTRATION_MODE_MAP_PROJECTION = "map_projection"
TARGET_REGISTRATION_MODE_BOUNDED_CAMERA_LIDAR = (
    "bounded_camera_lidar_registration"
)
TARGET_REGISTRATION_LIDAR_SOURCE_MAP = "map_bearing"
TARGET_REGISTRATION_LIDAR_SOURCE_CAMERA = "registered_camera_bearing"
TARGET_REGISTRATION_EVIDENCE_KEYS = (
    "mode",
    "original_head_center_error_ratio",
    "center_offset_limit_ratio",
    "final_strict_head_center_error_ratio",
    "map_bearing_rad",
    "lidar_search_bearing_rad",
    "camera_map_bearing_delta_rad",
    "bearing_delta_limit_rad",
    "lidar_search_bearing_source",
    "unique_eligible_lidar_cluster_required",
    "eligible_lidar_cluster_count",
)
BACKSIDE_SAMPLE_GATE_KEYS = (
    "all_samples_stationary",
    "all_samples_synchronized",
    "all_samples_lidar_associated",
    "all_samples_current_frame_model_geometry",
    "all_samples_qr_marker_absent",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class BacksideAxisObservation:
    """Validated, immutable subset used for opposite-face planning."""

    stand_id: str
    planning_frame: str
    stand_axis_rad: float
    stand_x_m: float
    stand_y_m: float
    robot_x_m: float
    robot_y_m: float
    visible_face_confidence: float
    axis_confidence: float
    axis_sample_count: int
    stand_model_profile_sha256: str

    @property
    def opposite_face_normal_rad(self) -> float:
        """Return the stand normal antipodal to the observing robot."""

        relative_x_m = self.robot_x_m - self.stand_x_m
        relative_y_m = self.robot_y_m - self.stand_y_m
        if math.hypot(relative_x_m, relative_y_m) <= 1.0e-9:
            raise ValueError(
                "axis observation robot pose coincides with stand center"
            )

        robot_side = math.atan2(relative_y_m, relative_x_m)
        normals = (
            _normalize_angle(self.stand_axis_rad + math.pi / 2.0),
            _normalize_angle(self.stand_axis_rad - math.pi / 2.0),
        )
        selected = min(
            normals,
            key=lambda normal: math.cos(normal - robot_side),
        )
        if math.cos(selected - robot_side) > -0.5:
            raise ValueError(
                "stand axis does not resolve a sufficiently opposite "
                "inspection face"
            )
        return selected


def validated_backside_axis_observation(
    payload: Mapping[str, object],
) -> BacksideAxisObservation:
    """Validate the complete receipt needed for backside-derived planning."""

    if not isinstance(payload, Mapping):
        raise ValueError("axis observation must be a mapping")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version not in (
        LEGACY_BACKSIDE_AXIS_OBSERVATION_SCHEMA_VERSION,
        BACKSIDE_AXIS_OBSERVATION_SCHEMA_VERSION,
    ):
        raise ValueError("axis observation schema_version must be exactly 2 or 3")
    is_legacy_receipt = (
        schema_version == LEGACY_BACKSIDE_AXIS_OBSERVATION_SCHEMA_VERSION
    )
    if is_legacy_receipt:
        if "target_registration" in payload:
            raise ValueError(
                "schema-2 axis observation cannot contain target_registration"
            )
        expected_observer_version = LEGACY_PASSIVE_VIEWPOINT_OBSERVER_VERSION
        expected_axis_sample_source = BACKSIDE_AXIS_SAMPLE_SOURCE
        target_registration = None
    else:
        expected_observer_version = PASSIVE_VIEWPOINT_OBSERVER_VERSION
        target_registration = _mapping(
            payload.get("target_registration"), "target_registration"
        )
        if set(target_registration) != set(TARGET_REGISTRATION_EVIDENCE_KEYS):
            raise ValueError(
                "axis observation target_registration has unexpected fields"
            )
        mode = target_registration.get("mode")
        if mode == TARGET_REGISTRATION_MODE_MAP_PROJECTION:
            expected_axis_sample_source = BACKSIDE_AXIS_SAMPLE_SOURCE
        elif mode == TARGET_REGISTRATION_MODE_BOUNDED_CAMERA_LIDAR:
            expected_axis_sample_source = REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE
        else:
            raise ValueError(
                "axis observation target_registration.mode is unsupported"
            )
    if payload.get("observation_kind") != BACKSIDE_AXIS_OBSERVATION_KIND:
        raise ValueError("unexpected axis observation kind")
    if payload.get("visible_face") != BACKSIDE_VISIBLE_FACE:
        raise ValueError(
            "axis observation visible_face is not a backside candidate"
        )
    if payload.get("visible_face_source") != BACKSIDE_AXIS_SAMPLE_SOURCE:
        raise ValueError(
            "axis observation visible_face_source is not current-frame"
        )
    if payload.get("axis_sample_source") != expected_axis_sample_source:
        raise ValueError(
            "axis observation axis_sample_source does not match target registration"
        )
    if payload.get("model_evidence_state") != BACKSIDE_MODEL_EVIDENCE_STATE:
        raise ValueError(
            "axis observation model evidence is not fresh backside"
        )
    if payload.get("classification_basis") != BACKSIDE_CLASSIFICATION_BASIS:
        raise ValueError(
            "axis observation classification_basis is unsupported"
        )
    if payload.get("motion_capability") != "none":
        raise ValueError("axis observation motion_capability must be none")
    if payload.get("observer_version") != expected_observer_version:
        raise ValueError("axis observation observer_version is unsupported")
    if payload.get("stand_model_measurement_status") != "measured":
        raise ValueError(
            "axis observation stand_model_measurement_status must be measured"
        )
    if payload.get("qr_marker_detected") is not False:
        raise ValueError(
            "axis observation must prove qr_marker_detected is false"
        )
    qr_texts = payload.get("qr_texts")
    if type(qr_texts) is not list or qr_texts:
        raise ValueError("axis observation qr_texts must be an empty list")

    visible_face_confidence = _finite_number(
        payload.get("visible_face_confidence"),
        "visible_face_confidence",
    )
    if not MINIMUM_BACKSIDE_FACE_CONFIDENCE <= visible_face_confidence <= 1.0:
        raise ValueError(
            "axis observation visible_face_confidence must be in [0.70, 1]"
        )
    axis_confidence = _finite_number(
        payload.get("axis_confidence"), "axis_confidence"
    )
    if not MINIMUM_BACKSIDE_AXIS_CONFIDENCE <= axis_confidence <= 1.0:
        raise ValueError(
            "axis observation axis_confidence must be in [0.60, 1]"
        )
    axis_sample_count = payload.get("axis_sample_count")
    if (
        type(axis_sample_count) is not int
        or axis_sample_count < MINIMUM_BACKSIDE_AXIS_SAMPLE_COUNT
    ):
        raise ValueError(
            "axis observation axis_sample_count must be an integer >= 2"
        )
    qr_absent_sample_count = payload.get("qr_absent_sample_count")
    if (
        type(qr_absent_sample_count) is not int
        or qr_absent_sample_count != axis_sample_count
    ):
        raise ValueError(
            "axis observation qr_absent_sample_count must equal "
            "axis_sample_count"
        )
    gates = _mapping(
        payload.get("sample_gate_evidence"), "sample_gate_evidence"
    )
    if set(gates) != set(BACKSIDE_SAMPLE_GATE_KEYS):
        raise ValueError(
            "axis observation sample_gate_evidence has unexpected fields"
        )
    for gate in BACKSIDE_SAMPLE_GATE_KEYS:
        if gates.get(gate) is not True:
            raise ValueError(
                f"axis observation sample_gate_evidence.{gate} must be true"
            )

    model_sha256 = _sha256(
        payload.get("stand_model_profile_sha256"),
        "stand_model_profile_sha256",
    )
    _sha256(payload.get("robot_profile_sha256"), "robot_profile_sha256")
    _sha256(
        payload.get("calibration_profile_sha256"),
        "calibration_profile_sha256",
    )
    sensor_stamp_sec = _finite_number(
        payload.get("sensor_stamp_sec"), "sensor_stamp_sec"
    )
    if sensor_stamp_sec < 0.0:
        raise ValueError("axis observation sensor_stamp_sec must be nonnegative")
    head_scale_ratio = _finite_number(
        payload.get("head_scale_ratio"), "head_scale_ratio"
    )
    if not MINIMUM_HEAD_SCALE_RATIO <= head_scale_ratio <= MAXIMUM_HEAD_SCALE_RATIO:
        raise ValueError(
            "axis observation head_scale_ratio must be in [0.60, 1.35]"
        )
    head_center_error_ratio = _finite_number(
        payload.get("head_center_error_ratio"),
        "head_center_error_ratio",
    )
    if not 0.0 <= head_center_error_ratio <= MAXIMUM_HEAD_CENTER_ERROR_RATIO:
        raise ValueError(
            "axis observation head_center_error_ratio must be in [0, 0.55]"
        )
    if target_registration is not None:
        _validate_target_registration(
            target_registration,
            final_head_center_error_ratio=head_center_error_ratio,
        )
    for name in ("pose_reprojection_rmse_px", "pose_ambiguity_gap_px"):
        value = payload.get(name)
        if value is not None and _finite_number(value, name) < 0.0:
            raise ValueError(
                f"axis observation {name} must be nonnegative when present"
            )

    _nonempty_string(payload.get("stream_id"), "stream_id")
    stand_id = _nonempty_string(payload.get("stand_id"), "stand_id")
    planning_frame = _nonempty_string(
        payload.get("planning_frame"), "planning_frame"
    )
    stand = _mapping(payload.get("stand_center"), "stand_center")
    robot = _mapping(payload.get("robot_pose"), "robot_pose")
    _finite_number(robot.get("yaw_rad"), "robot_pose.yaw_rad")
    return BacksideAxisObservation(
        stand_id=stand_id,
        planning_frame=planning_frame,
        stand_axis_rad=_finite_number(
            payload.get("stand_axis_rad"), "stand_axis_rad"
        ),
        stand_x_m=_finite_number(stand.get("x_m"), "stand_center.x_m"),
        stand_y_m=_finite_number(stand.get("y_m"), "stand_center.y_m"),
        robot_x_m=_finite_number(robot.get("x_m"), "robot_pose.x_m"),
        robot_y_m=_finite_number(robot.get("y_m"), "robot_pose.y_m"),
        visible_face_confidence=visible_face_confidence,
        axis_confidence=axis_confidence,
        axis_sample_count=axis_sample_count,
        stand_model_profile_sha256=model_sha256,
    )


def validate_backside_axis_observation(
    payload: Mapping[str, object],
) -> None:
    """Validate a receipt when the typed result is not needed."""

    validated_backside_axis_observation(payload)


def load_backside_axis_observation(
    axis_observation_path: Path,
) -> BacksideAxisObservation:
    """Load and validate one model-backed backside observation receipt."""

    try:
        payload = json.loads(Path(axis_observation_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load axis observation: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("axis observation JSON root must be an object")
    return validated_backside_axis_observation(payload)


def opposite_face_normal_from_axis_observation(
    payload: Mapping[str, object],
) -> float:
    """Validate a receipt and derive its motion-neutral face normal."""

    return validated_backside_axis_observation(
        payload
    ).opposite_face_normal_rad


def load_opposite_face_normal(axis_observation_path: Path) -> float:
    """Load a validated observation and derive its opposite face normal."""

    return load_backside_axis_observation(
        axis_observation_path
    ).opposite_face_normal_rad


def _normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"axis observation {name} must be a mapping")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"axis observation {name} must be a non-empty string")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"axis observation {name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"axis observation {name} must be finite")
    return number


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(
            f"axis observation {name} must be lowercase SHA-256"
        )
    return value


def _validate_target_registration(
    evidence: Mapping[str, object],
    *,
    final_head_center_error_ratio: float,
) -> None:
    """Validate the exact camera/map registration evidence carried by v3."""

    if set(evidence) != set(TARGET_REGISTRATION_EVIDENCE_KEYS):
        raise ValueError(
            "axis observation target_registration has unexpected fields"
        )
    mode = evidence.get("mode")
    if mode not in (
        TARGET_REGISTRATION_MODE_MAP_PROJECTION,
        TARGET_REGISTRATION_MODE_BOUNDED_CAMERA_LIDAR,
    ):
        raise ValueError(
            "axis observation target_registration.mode is unsupported"
        )

    original_center_error = _finite_number(
        evidence.get("original_head_center_error_ratio"),
        "target_registration.original_head_center_error_ratio",
    )
    center_offset_limit = _finite_number(
        evidence.get("center_offset_limit_ratio"),
        "target_registration.center_offset_limit_ratio",
    )
    final_strict_center_error = _finite_number(
        evidence.get("final_strict_head_center_error_ratio"),
        "target_registration.final_strict_head_center_error_ratio",
    )
    if not 0.0 <= center_offset_limit <= (
        MAXIMUM_REGISTRATION_HEAD_CENTER_OFFSET_RATIO
    ):
        raise ValueError(
            "axis observation target_registration.center_offset_limit_ratio "
            "must be in [0, 1.5]"
        )
    if not 0.0 <= original_center_error <= center_offset_limit:
        raise ValueError(
            "axis observation target_registration original center error "
            "exceeds its bounded limit"
        )
    if not 0.0 <= final_strict_center_error <= (
        MAXIMUM_HEAD_CENTER_ERROR_RATIO
    ):
        raise ValueError(
            "axis observation target_registration final strict center error "
            "must be in [0, 0.55]"
        )
    if not math.isclose(
        final_strict_center_error,
        final_head_center_error_ratio,
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    ):
        raise ValueError(
            "axis observation target_registration final strict center error "
            "differs from head_center_error_ratio"
        )

    map_bearing = _finite_number(
        evidence.get("map_bearing_rad"),
        "target_registration.map_bearing_rad",
    )
    lidar_search_bearing = _finite_number(
        evidence.get("lidar_search_bearing_rad"),
        "target_registration.lidar_search_bearing_rad",
    )
    camera_map_delta = _finite_number(
        evidence.get("camera_map_bearing_delta_rad"),
        "target_registration.camera_map_bearing_delta_rad",
    )
    bearing_delta_limit = _finite_number(
        evidence.get("bearing_delta_limit_rad"),
        "target_registration.bearing_delta_limit_rad",
    )
    if not 0.0 <= bearing_delta_limit <= (
        MAXIMUM_REGISTRATION_BEARING_DELTA_RAD
    ):
        raise ValueError(
            "axis observation target_registration.bearing_delta_limit_rad "
            "must be in [0, 12 degrees]"
        )
    if not 0.0 <= camera_map_delta <= bearing_delta_limit:
        raise ValueError(
            "axis observation target_registration camera/map bearing delta "
            "exceeds its bounded limit"
        )

    lidar_source = evidence.get("lidar_search_bearing_source")
    unique_required = evidence.get(
        "unique_eligible_lidar_cluster_required"
    )
    if type(unique_required) is not bool:
        raise ValueError(
            "axis observation target_registration unique cluster flag "
            "must be boolean"
        )
    cluster_count = evidence.get("eligible_lidar_cluster_count")
    if (
        type(cluster_count) is not int
        or cluster_count < 1
    ):
        raise ValueError(
            "axis observation target_registration eligible cluster count "
            "must be an integer >= 1"
        )

    search_map_delta = abs(
        _normalize_angle(lidar_search_bearing - map_bearing)
    )
    if mode == TARGET_REGISTRATION_MODE_BOUNDED_CAMERA_LIDAR:
        if lidar_source != TARGET_REGISTRATION_LIDAR_SOURCE_CAMERA:
            raise ValueError(
                "registered axis observation must use registered_camera_bearing"
            )
        if unique_required is not True or cluster_count != 1:
            raise ValueError(
                "registered axis observation requires exactly one eligible "
                "LiDAR cluster"
            )
        if not math.isclose(
            search_map_delta,
            camera_map_delta,
            rel_tol=1.0e-9,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "registered axis observation search/map bearing delta differs "
                "from camera/map bearing delta"
            )
    else:
        if lidar_source != TARGET_REGISTRATION_LIDAR_SOURCE_MAP:
            raise ValueError(
                "nominal axis observation must use map_bearing"
            )
        if unique_required is not False:
            raise ValueError(
                "nominal axis observation cannot require a unique LiDAR cluster"
            )
        if not math.isclose(
            search_map_delta,
            0.0,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "nominal axis observation LiDAR search bearing must equal map bearing"
            )
        if not math.isclose(
            original_center_error,
            final_strict_center_error,
            rel_tol=1.0e-9,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "nominal axis observation original and final center errors differ"
            )


__all__ = [
    "BACKSIDE_AXIS_OBSERVATION_KIND",
    "BACKSIDE_AXIS_OBSERVATION_SCHEMA_VERSION",
    "BACKSIDE_AXIS_SAMPLE_SOURCE",
    "BACKSIDE_CLASSIFICATION_BASIS",
    "BACKSIDE_CURRENT_FRAME_SOURCE",
    "BACKSIDE_MODEL_EVIDENCE_STATE",
    "BACKSIDE_SAMPLE_GATE_KEYS",
    "BACKSIDE_VISIBLE_FACE",
    "BacksideAxisObservation",
    "LEGACY_BACKSIDE_AXIS_OBSERVATION_SCHEMA_VERSION",
    "LEGACY_PASSIVE_VIEWPOINT_OBSERVER_VERSION",
    "MAXIMUM_HEAD_CENTER_ERROR_RATIO",
    "MAXIMUM_REGISTRATION_BEARING_DELTA_RAD",
    "MAXIMUM_REGISTRATION_HEAD_CENTER_OFFSET_RATIO",
    "MAXIMUM_HEAD_SCALE_RATIO",
    "MINIMUM_BACKSIDE_AXIS_CONFIDENCE",
    "MINIMUM_BACKSIDE_AXIS_SAMPLE_COUNT",
    "MINIMUM_BACKSIDE_FACE_CONFIDENCE",
    "MINIMUM_HEAD_SCALE_RATIO",
    "PASSIVE_VIEWPOINT_OBSERVER_VERSION",
    "REGISTERED_BACKSIDE_AXIS_SAMPLE_SOURCE",
    "REAL_STAND_AXIS_OBSERVATION_KIND",
    "TARGET_REGISTRATION_EVIDENCE_KEYS",
    "TARGET_REGISTRATION_LIDAR_SOURCE_CAMERA",
    "TARGET_REGISTRATION_LIDAR_SOURCE_MAP",
    "TARGET_REGISTRATION_MODE_BOUNDED_CAMERA_LIDAR",
    "TARGET_REGISTRATION_MODE_MAP_PROJECTION",
    "load_backside_axis_observation",
    "load_opposite_face_normal",
    "opposite_face_normal_from_axis_observation",
    "validate_backside_axis_observation",
    "validated_backside_axis_observation",
]
