"""Immutable real-robot runtime and camera-calibration profiles."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json,
    payload_sha256,
    write_content_hashed_json,
)
from scripts.aufgabe04.navigation.foundation.ros_runtime_config import (
    ResolvedRuntimeConfig,
    RuntimeConfig,
    resolve_runtime_config,
    resolve_topic,
)


REAL_HARDWARE_PROFILE_SCHEMA_VERSION = 1
CAMERA_CALIBRATION_PROFILE_SCHEMA_VERSION = 1
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_FRAME = re.compile(r"^[A-Za-z][A-Za-z0-9_/.-]{0,127}$")
_SAFE_ROS_NAME = re.compile(r"^/?[A-Za-z][A-Za-z0-9_/]{0,254}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RigidTransform:
    translation_xyz_m: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class CameraCalibrationProfile:
    schema_version: int
    calibration_id: str
    camera_optical_frame: str
    base_frame: str
    width_px: int
    height_px: int
    distortion_model: str
    distortion_coefficients: tuple[float, ...]
    camera_matrix: tuple[float, ...]
    rectification_matrix: tuple[float, ...]
    projection_matrix: tuple[float, ...]
    base_to_camera: RigidTransform
    measured_unix_sec: float
    source: str


@dataclass(frozen=True)
class RealRobotProfile:
    schema_version: int
    profile_id: str
    robot_id: str
    namespace: str
    scan_topic: str
    odom_topic: str
    cmd_vel_topic: str
    amcl_topic: str
    compressed_image_topic: str
    camera_info_topic: str
    map_frame: str
    odom_frame: str
    base_frame: str
    scan_frame: str
    camera_optical_frame: str
    localization_source: str
    physical_site_id: str
    physical_site_sha256: str
    calibration_profile_sha256: str
    robot_radius_m: float
    scan_origin_to_base_offset_m: float
    max_linear_speed_mps: float
    max_angular_speed_radps: float

    def runtime_config(self) -> RuntimeConfig:
        return RuntimeConfig(
            namespace=self.namespace,
            scan_topic=self.scan_topic,
            odom_topic=self.odom_topic,
            cmd_vel_topic=self.cmd_vel_topic,
            amcl_topic=self.amcl_topic,
            map_frame=self.map_frame,
            odom_frame=self.odom_frame,
            base_frame=self.base_frame,
            localization_source=self.localization_source,
            use_sim_time=False,
        )

    def resolved_runtime(self) -> ResolvedRuntimeConfig:
        return resolve_runtime_config(self.runtime_config())

    @property
    def resolved_compressed_image_topic(self) -> str:
        return resolve_topic(self.compressed_image_topic, self.namespace)

    @property
    def resolved_camera_info_topic(self) -> str:
        return resolve_topic(self.camera_info_topic, self.namespace)


def validate_camera_calibration(profile: CameraCalibrationProfile) -> None:
    if profile.schema_version != CAMERA_CALIBRATION_PROFILE_SCHEMA_VERSION:
        raise ValueError("unsupported camera calibration profile schema_version")
    _validate_id(profile.calibration_id, "calibration_id")
    _validate_frame(profile.camera_optical_frame, "camera_optical_frame")
    _validate_frame(profile.base_frame, "base_frame")
    if profile.width_px <= 0 or profile.height_px <= 0:
        raise ValueError("camera calibration dimensions must be positive")
    if not profile.distortion_model:
        raise ValueError("camera calibration distortion_model must be non-empty")
    _finite_sequence(
        profile.distortion_coefficients,
        "distortion_coefficients",
        minimum_length=0,
    )
    _finite_sequence(profile.camera_matrix, "camera_matrix", exact_length=9)
    _finite_sequence(profile.rectification_matrix, "rectification_matrix", exact_length=9)
    _finite_sequence(profile.projection_matrix, "projection_matrix", exact_length=12)
    if profile.camera_matrix[0] <= 0.0 or profile.camera_matrix[4] <= 0.0:
        raise ValueError("camera calibration focal lengths must be positive")
    if profile.projection_matrix[0] <= 0.0 or profile.projection_matrix[5] <= 0.0:
        raise ValueError("camera projection focal lengths must be positive")
    _validate_transform(profile.base_to_camera)
    _finite_nonnegative(profile.measured_unix_sec, "measured_unix_sec")
    if not profile.source:
        raise ValueError("camera calibration source must be non-empty")


def validate_real_robot_profile(profile: RealRobotProfile) -> None:
    if profile.schema_version != REAL_HARDWARE_PROFILE_SCHEMA_VERSION:
        raise ValueError("unsupported real robot profile schema_version")
    for name, value in (
        ("profile_id", profile.profile_id),
        ("robot_id", profile.robot_id),
        ("physical_site_id", profile.physical_site_id),
    ):
        _validate_id(value, name)
    for name, value in (
        ("map_frame", profile.map_frame),
        ("odom_frame", profile.odom_frame),
        ("base_frame", profile.base_frame),
        ("scan_frame", profile.scan_frame),
        ("camera_optical_frame", profile.camera_optical_frame),
    ):
        _validate_frame(value, name)
    if profile.map_frame == profile.odom_frame:
        raise ValueError(
            "real profile map_frame and odom_frame must differ; use localization "
            "for map -> odom"
        )
    if profile.localization_source not in {"amcl", "tf"}:
        raise ValueError("localization_source must be 'amcl' or 'tf'")
    for name, topic in (
        ("scan_topic", profile.scan_topic),
        ("odom_topic", profile.odom_topic),
        ("cmd_vel_topic", profile.cmd_vel_topic),
        ("amcl_topic", profile.amcl_topic),
        ("compressed_image_topic", profile.compressed_image_topic),
        ("camera_info_topic", profile.camera_info_topic),
    ):
        _validate_ros_name(topic, name)
    if profile.namespace:
        _validate_ros_name(profile.namespace, "namespace")
    resolved_topics = {
        profile.resolved_runtime().scan_topic,
        profile.resolved_runtime().odom_topic,
        profile.resolved_runtime().cmd_vel_topic,
        profile.resolved_runtime().amcl_topic,
        profile.resolved_compressed_image_topic,
        profile.resolved_camera_info_topic,
    }
    if len(resolved_topics) != 6:
        raise ValueError("real profile topics must resolve to six distinct topics")
    for name, value in (
        ("physical_site_sha256", profile.physical_site_sha256),
        ("calibration_profile_sha256", profile.calibration_profile_sha256),
    ):
        if not _SHA256.fullmatch(value):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    if profile.robot_radius_m <= 0.0:
        raise ValueError("robot_radius_m must be positive")
    _finite_nonnegative(
        profile.scan_origin_to_base_offset_m,
        "scan_origin_to_base_offset_m",
    )
    if profile.max_linear_speed_mps <= 0.0:
        raise ValueError("max_linear_speed_mps must be positive")
    if profile.max_angular_speed_radps <= 0.0:
        raise ValueError("max_angular_speed_radps must be positive")
    for name, value in (
        ("robot_radius_m", profile.robot_radius_m),
        ("max_linear_speed_mps", profile.max_linear_speed_mps),
        ("max_angular_speed_radps", profile.max_angular_speed_radps),
    ):
        _finite(value, name)


def camera_calibration_sha256(profile: CameraCalibrationProfile) -> str:
    validate_camera_calibration(profile)
    return payload_sha256(_calibration_payload(profile))


def real_robot_profile_sha256(profile: RealRobotProfile) -> str:
    validate_real_robot_profile(profile)
    return payload_sha256(_robot_payload(profile))


def write_camera_calibration(path: Path, profile: CameraCalibrationProfile) -> str:
    validate_camera_calibration(profile)
    return write_content_hashed_json(
        path,
        _calibration_payload(profile),
        hash_field="calibration_profile_sha256",
    )


def load_camera_calibration(path: Path) -> CameraCalibrationProfile:
    payload = load_content_hashed_json(
        path,
        hash_field="calibration_profile_sha256",
    )
    profile = _calibration_from_payload(payload)
    validate_camera_calibration(profile)
    return profile


def write_real_robot_profile(path: Path, profile: RealRobotProfile) -> str:
    validate_real_robot_profile(profile)
    return write_content_hashed_json(
        path,
        _robot_payload(profile),
        hash_field="real_robot_profile_sha256",
    )


def load_real_robot_profile(path: Path) -> RealRobotProfile:
    payload = load_content_hashed_json(
        path,
        hash_field="real_robot_profile_sha256",
    )
    profile = _robot_from_payload(payload)
    validate_real_robot_profile(profile)
    return profile


def camera_info_mismatches(
    profile: CameraCalibrationProfile,
    camera_info,
    *,
    numeric_tolerance: float = 1.0e-6,
) -> tuple[str, ...]:
    """Compare a live ``sensor_msgs/CameraInfo`` with the sealed profile."""

    validate_camera_calibration(profile)
    observed = {
        "width_px": int(camera_info.width),
        "height_px": int(camera_info.height),
        "distortion_model": str(camera_info.distortion_model),
        "distortion_coefficients": tuple(float(value) for value in camera_info.d),
        "camera_matrix": tuple(float(value) for value in camera_info.k),
        "rectification_matrix": tuple(float(value) for value in camera_info.r),
        "projection_matrix": tuple(float(value) for value in camera_info.p),
    }
    mismatches = []
    for name in ("width_px", "height_px", "distortion_model"):
        if observed[name] != getattr(profile, name):
            mismatches.append(name)
    for name in (
        "distortion_coefficients",
        "camera_matrix",
        "rectification_matrix",
        "projection_matrix",
    ):
        expected = getattr(profile, name)
        actual = observed[name]
        if len(expected) != len(actual) or any(
            abs(first - second) > numeric_tolerance
            for first, second in zip(expected, actual)
        ):
            mismatches.append(name)
    observed_frame = str(getattr(camera_info.header, "frame_id", "")).strip("/")
    if observed_frame != profile.camera_optical_frame:
        mismatches.append("camera_optical_frame")
    return tuple(mismatches)


def transform_mismatches(
    expected: RigidTransform,
    transform,
    *,
    translation_tolerance_m: float = 0.005,
    rotation_tolerance_rad: float = math.radians(1.0),
) -> tuple[str, ...]:
    """Compare a live base<-camera TF with the measured calibration."""

    _validate_transform(expected)
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    actual_translation = (
        float(translation.x),
        float(translation.y),
        float(translation.z),
    )
    actual_rotation = _normalized_quaternion(
        (float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w))
    )
    expected_rotation = _normalized_quaternion(expected.rotation_xyzw)
    mismatches = []
    translation_error = math.sqrt(
        sum(
            (actual - wanted) ** 2
            for actual, wanted in zip(actual_translation, expected.translation_xyz_m)
        )
    )
    if translation_error > translation_tolerance_m:
        mismatches.append("base_to_camera_translation")
    quaternion_dot = abs(
        sum(actual * wanted for actual, wanted in zip(actual_rotation, expected_rotation))
    )
    quaternion_dot = min(1.0, max(-1.0, quaternion_dot))
    rotation_error = 2.0 * math.acos(quaternion_dot)
    if rotation_error > rotation_tolerance_rad:
        mismatches.append("base_to_camera_rotation")
    return tuple(mismatches)


def _calibration_payload(profile: CameraCalibrationProfile) -> dict[str, object]:
    payload = asdict(profile)
    payload["profile_kind"] = "real_camera_calibration"
    return payload


def _robot_payload(profile: RealRobotProfile) -> dict[str, object]:
    payload = asdict(profile)
    payload["profile_kind"] = "real_robot_runtime"
    payload["use_sim_time"] = False
    return payload


def _calibration_from_payload(payload: Mapping[str, object]) -> CameraCalibrationProfile:
    if payload.get("profile_kind") != "real_camera_calibration":
        raise ValueError("camera calibration profile_kind mismatch")
    transform = _mapping(payload.get("base_to_camera"), "base_to_camera")
    return CameraCalibrationProfile(
        schema_version=int(payload["schema_version"]),
        calibration_id=str(payload["calibration_id"]),
        camera_optical_frame=str(payload["camera_optical_frame"]),
        base_frame=str(payload["base_frame"]),
        width_px=int(payload["width_px"]),
        height_px=int(payload["height_px"]),
        distortion_model=str(payload["distortion_model"]),
        distortion_coefficients=_number_tuple(payload["distortion_coefficients"]),
        camera_matrix=_number_tuple(payload["camera_matrix"]),
        rectification_matrix=_number_tuple(payload["rectification_matrix"]),
        projection_matrix=_number_tuple(payload["projection_matrix"]),
        base_to_camera=RigidTransform(
            translation_xyz_m=_fixed_number_tuple(
                transform.get("translation_xyz_m"),
                "base_to_camera.translation_xyz_m",
                3,
            ),
            rotation_xyzw=_fixed_number_tuple(
                transform.get("rotation_xyzw"),
                "base_to_camera.rotation_xyzw",
                4,
            ),
        ),
        measured_unix_sec=float(payload["measured_unix_sec"]),
        source=str(payload["source"]),
    )


def _robot_from_payload(payload: Mapping[str, object]) -> RealRobotProfile:
    if payload.get("profile_kind") != "real_robot_runtime":
        raise ValueError("real robot profile_kind mismatch")
    if payload.get("use_sim_time") is not False:
        raise ValueError("real robot profile must set use_sim_time=false")
    names = (
        "profile_id",
        "robot_id",
        "namespace",
        "scan_topic",
        "odom_topic",
        "cmd_vel_topic",
        "amcl_topic",
        "compressed_image_topic",
        "camera_info_topic",
        "map_frame",
        "odom_frame",
        "base_frame",
        "scan_frame",
        "camera_optical_frame",
        "localization_source",
        "physical_site_id",
        "physical_site_sha256",
        "calibration_profile_sha256",
    )
    values = {name: str(payload[name]) for name in names}
    return RealRobotProfile(
        schema_version=int(payload["schema_version"]),
        **values,
        robot_radius_m=float(payload["robot_radius_m"]),
        scan_origin_to_base_offset_m=float(payload["scan_origin_to_base_offset_m"]),
        max_linear_speed_mps=float(payload["max_linear_speed_mps"]),
        max_angular_speed_radps=float(payload["max_angular_speed_radps"]),
    )


def _validate_transform(transform: RigidTransform) -> None:
    _finite_sequence(transform.translation_xyz_m, "translation_xyz_m", exact_length=3)
    _finite_sequence(transform.rotation_xyzw, "rotation_xyzw", exact_length=4)
    norm = math.sqrt(sum(value * value for value in transform.rotation_xyzw))
    if abs(norm - 1.0) > 1.0e-3:
        raise ValueError("calibration quaternion must be unit normalized")


def _normalized_quaternion(values: Sequence[float]) -> tuple[float, float, float, float]:
    normalized = tuple(float(value) for value in values)
    norm = math.sqrt(sum(value * value for value in normalized))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("quaternion must be finite and non-zero")
    return tuple(value / norm for value in normalized)  # type: ignore[return-value]


def _validate_id(value: str, name: str) -> None:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{name} is not a safe identifier")


def _validate_frame(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not _SAFE_FRAME.fullmatch(value)
        or value.startswith("/")
        or "//" in value
        or ".." in value
    ):
        raise ValueError(f"{name} is not a valid relative ROS frame")


def _validate_ros_name(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not _SAFE_ROS_NAME.fullmatch(value)
        or "//" in value
        or value.endswith("/")
        or ".." in value
    ):
        raise ValueError(f"{name} is not a valid ROS name")


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_nonnegative(value: float, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _finite_sequence(
    values: Sequence[float],
    name: str,
    *,
    minimum_length: int | None = None,
    exact_length: int | None = None,
) -> None:
    if exact_length is not None and len(values) != exact_length:
        raise ValueError(f"{name} must contain exactly {exact_length} values")
    if minimum_length is not None and len(values) < minimum_length:
        raise ValueError(f"{name} contains too few values")
    for index, value in enumerate(values):
        _finite(value, f"{name}[{index}]")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _number_tuple(value: object) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("numeric profile field must be an array")
    return tuple(float(item) for item in value)


def _fixed_number_tuple(
    value: object,
    name: str,
    length: int,
) -> tuple:
    result = _number_tuple(value)
    if len(result) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return result
