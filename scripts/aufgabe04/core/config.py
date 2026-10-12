"""Measured physical inputs, loaded once before the live robot session."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re


@dataclass(frozen=True)
class RigidTransform:
    translation_xyz_m: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class CameraCalibration:
    width_px: int
    height_px: int
    distortion_model: str
    distortion_coefficients: tuple[float, ...]
    camera_matrix: tuple[float, ...]
    rectification_matrix: tuple[float, ...]
    projection_matrix: tuple[float, ...]
    base_to_camera: RigidTransform
    base_frame: str
    camera_optical_frame: str


@dataclass(frozen=True)
class StandModel:
    head_width_m: float
    head_height_m: float
    head_depth_m: float
    head_top_height_m: float
    base_width_m: float
    base_depth_m: float
    qr_symbol_width_m: float
    qr_symbol_height_m: float
    qr_panel_width_m: float
    qr_panel_height_m: float
    qr_center_x_m: float
    qr_center_y_m: float
    tolerance_m: float

    @property
    def head_center_height_m(self) -> float:
        return self.head_top_height_m - self.head_height_m / 2.0

    @property
    def navigation_footprint_radius_m(self) -> float:
        return math.hypot(self.base_width_m, self.base_depth_m) / 2.0 + self.tolerance_m


@dataclass(frozen=True)
class Config:
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
    robot_radius_m: float
    scan_origin_to_base_offset_m: float
    max_linear_speed_mps: float
    max_angular_speed_radps: float
    calibration: CameraCalibration
    model: StandModel
    map_yaml: Path
    semantic_map_id: str
    expected_yaml_sha256: str
    expected_image_sha256: str
    expected_map_bundle_sha256: str
    survey_candidate_radius_m: float = 0.06
    approach_offset_m: float = 0.50
    stop_spacing_m: float = 0.70
    lidar_duration_sec: float = 8.0
    camera_timeout_sec: float = 90.0
    startup_timeout_sec: float = 15.0
    mission_timeout_sec: float = 1800.0
    uncertainty_sigma_multiplier: float = 2.0
    tracking_margin_m: float = 0.03
    collision_margin_m: float = 0.01
    lidar_stop_distance_m: float = 0.20
    lidar_clearance_margin_m: float = 0.02
    braking_distance_m: float = 0.015
    odom_drift_bound_m: float = 0.02

    @property
    def camera_offset_xy_m(self) -> tuple[float, float]:
        return self.calibration.base_to_camera.translation_xyz_m[:2]

    @property
    def camera_heading_offset_rad(self) -> float:
        qx, qy, qz, qw = self.calibration.base_to_camera.rotation_xyzw
        fx, fy = 2*(qx*qz+qw*qy), 2*(qy*qz-qw*qx)
        if math.hypot(fx, fy) < .5:
            raise ValueError("measured camera optical direction is not planar")
        return math.atan2(fy, fx)

    @property
    def inflation_radius_m(self) -> float:
        return max(0.25, self.robot_radius_m + self.tracking_margin_m,
                   self.lidar_stop_distance_m + self.scan_origin_to_base_offset_m
                   + self.lidar_clearance_margin_m + self.tracking_margin_m)

    @property
    def minimum_standoff_m(self) -> float:
        collision = (self.model.navigation_footprint_radius_m + 0.02
                     + self.robot_radius_m + self.collision_margin_m + self.tracking_margin_m)
        lidar = (0.06 + self.lidar_stop_distance_m + self.scan_origin_to_base_offset_m
                 + self.lidar_clearance_margin_m + 0.02)
        return max(collision, lidar)

    @property
    def candidate_keepout_radius_m(self) -> float:
        return max(self.minimum_standoff_m, 0.31 + self.tracking_margin_m)

    def motion_config(self):
        from .motion import MotionConfig
        return MotionConfig(robot_radius_m=self.robot_radius_m,
            collision_margin_m=self.collision_margin_m, tracking_margin_m=self.tracking_margin_m,
            odom_drift_bound_m=self.odom_drift_bound_m, braking_distance_m=self.braking_distance_m,
            uncertainty_sigma_multiplier=self.uncertainty_sigma_multiplier,
            max_linear_mps=self.max_linear_speed_mps, max_angular_radps=self.max_angular_speed_radps,
            min_obstacle_distance_m=self.lidar_stop_distance_m)


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_FRAME = re.compile(r"^[A-Za-z][A-Za-z0-9_/.-]{0,127}$")
_ROS = re.compile(r"^/?[A-Za-z_][A-Za-z0-9_/]*$")
_HASH = re.compile(r"^[0-9a-f]{64}$")


def _number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f"{name} is outside its physical range")
    return float(value)


def _text(value, name, pattern=None):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty text")
    if pattern is not None and not pattern.fullmatch(value):
        raise ValueError(f"invalid {name}")
    return value


def _sequence(value, name, length=None):
    if not isinstance(value, list) or length is not None and len(value) != length:
        raise ValueError(f"invalid {name} dimensions")
    return tuple(_number(v, name) for v in value)


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate configuration key: {key}")
        out[key] = value
    return out


def _regular(path, root=None):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"configuration may not be a symlink: {path}")
    result = path.resolve(strict=True)
    if not result.is_file():
        raise ValueError(f"configuration must be a regular file: {path}")
    if root is not None:
        result.relative_to(root)
    return result


def _read_json(path, *, hash_field=None):
    path = _regular(path)
    raw = path.read_bytes()
    def bad_constant(value):
        raise ValueError(f"nonfinite configuration value: {value}")
    data = json.loads(raw, object_pairs_hook=_pairs, parse_constant=bad_constant)
    if not isinstance(data, dict):
        raise ValueError(f"configuration must be an object: {path}")
    if hash_field is not None:
        digest = data.pop(hash_field, None)
        expected = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        if digest != expected:
            raise ValueError(f"configuration hash mismatch: {path}")
    return data, raw


def _schema(data, version, **required):
    if type(data.get("schema_version")) is not int or data["schema_version"] != version:
        raise ValueError("unsupported physical configuration schema")
    for key, value in required.items():
        if data.get(key) != value or isinstance(value, bool) and data.get(key) is not value:
            raise ValueError(f"physical configuration {key} mismatch")


def _load_calibration(path):
    data, _ = _read_json(path, hash_field="calibration_profile_sha256")
    _schema(data, 1, profile_kind="real_camera_calibration")
    for name in ("width_px", "height_px"):
        if type(data.get(name)) is not int or data[name] <= 0:
            raise ValueError("camera dimensions must be positive integers")
    transform = data.get("base_to_camera")
    if not isinstance(transform, dict):
        raise ValueError("measured base-to-camera transform required")
    translation = _sequence(transform.get("translation_xyz_m"), "translation", 3)
    quaternion = _sequence(transform.get("rotation_xyzw"), "rotation", 4)
    norm = math.sqrt(sum(v*v for v in quaternion))
    if norm < 1e-12 or abs(norm-1.0) > 1e-3:
        raise ValueError("measured camera quaternion must be normalized")
    matrix = _sequence(data.get("camera_matrix"), "camera_matrix", 9)
    projection = _sequence(data.get("projection_matrix"), "projection_matrix", 12)
    if min(matrix[0], matrix[4], projection[0], projection[5]) <= 0:
        raise ValueError("measured camera focal lengths must be positive")
    _number(data.get("measured_unix_sec"), "measured_unix_sec", nonnegative=True)
    _text(data.get("source"), "calibration source")
    return CameraCalibration(data["width_px"], data["height_px"],
        _text(data.get("distortion_model"), "distortion_model"),
        _sequence(data.get("distortion_coefficients"), "distortion_coefficients"),
        matrix, _sequence(data.get("rectification_matrix"), "rectification_matrix", 9),
        projection, RigidTransform(translation, quaternion),
        _text(data.get("base_frame"), "base_frame", _FRAME),
        _text(data.get("camera_optical_frame"), "camera_optical_frame", _FRAME)), data


def _load_model(path):
    data, _ = _read_json(path, hash_field="stand_model_sha256")
    _schema(data, 2, environment="physical", measurement_status="measured")
    _text(data.get("source"), "stand source")
    values = {name: _number(data.get(name), name, positive=name not in ("qr_center_x_m", "qr_center_y_m"))
              for name in StandModel.__dataclass_fields__}
    model = StandModel(**values)
    for dimension in ("width", "height"):
        symbol = values[f"qr_symbol_{dimension}_m"]
        panel = values[f"qr_panel_{dimension}_m"]
        head = values[f"head_{dimension}_m"]
        center = values["qr_center_x_m" if dimension == "width" else "qr_center_y_m"]
        if not symbol < panel < head or abs(center) + panel/2 >= head/2:
            raise ValueError("measured QR panel/symbol exceeds head geometry")
    if model.head_top_height_m <= model.head_height_m:
        raise ValueError("complete measured head must be above the floor")
    return model


def load_config(*, robot_profile: Path, camera_calibration: Path,
                stand_model_profile: Path, physical_site: Path, repository_root: Path,
                map_yaml: Path | None = None, semantic_map_id: str | None = None,
                approach_offset_m=0.50,
                stop_spacing_m=0.70, camera_timeout_sec=90.0) -> Config:
    """Accept existing measured input files; no runtime receipt files are needed."""
    robot, _ = _read_json(robot_profile, hash_field="real_robot_profile_sha256")
    _schema(robot, 1, profile_kind="real_robot_runtime", use_sim_time=False,
            localization_source="amcl")
    calibration, calibration_data = _load_calibration(camera_calibration)
    calibration_hash = hashlib.sha256(json.dumps(calibration_data, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if calibration_hash != robot.get("calibration_profile_sha256"):
        raise ValueError("camera calibration differs from measured robot profile")
    model = _load_model(stand_model_profile)
    root = Path(repository_root).resolve(strict=True)
    site, site_bytes = _read_json(physical_site)
    if hashlib.sha256(site_bytes).hexdigest() != robot.get("physical_site_sha256"):
        raise ValueError("physical site bytes differ from measured robot profile")
    _schema(site, 1)
    site_id = _text(robot.get("physical_site_id"), "physical_site_id", _ID)
    if site.get("physical_site_id") != site_id or Path(physical_site).stem != site_id:
        raise ValueError("physical site identity differs from robot profile")
    measurement = site.get("map_measurement")
    if not isinstance(measurement, dict):
        raise ValueError("physical site map measurement is missing")
    bound_map_id = _text(measurement.get("semantic_map_id"), "semantic_map_id", _ID)
    if semantic_map_id is not None and semantic_map_id != bound_map_id:
        raise ValueError("physical site map identity mismatch")
    semantic_map_id = bound_map_id
    paths = {}
    for name in ("map_yaml", "map_image"):
        value = _text(measurement.get(name), name)
        declared = PurePosixPath(value)
        if declared.is_absolute() or ".." in declared.parts or "\\" in value or declared.as_posix() != value:
            raise ValueError("site map references must be safe repository-relative paths")
        paths[name] = _regular(root / declared, root)
        digest = _text(measurement.get(name + "_sha256"), name + " hash", _HASH)
        if hashlib.sha256(paths[name].read_bytes()).hexdigest() != digest:
            raise ValueError("measured site map bytes changed")
    if map_yaml is not None:
        requested_map = Path(map_yaml)
        if _regular(requested_map if requested_map.is_absolute() else root / requested_map, root) != paths["map_yaml"]:
            raise ValueError("requested map differs from measured physical site")
    _text(measurement.get("map_bundle_sha256"), "map bundle hash", _HASH)
    namespace = robot.get("namespace")
    if namespace != "":
        _text(namespace, "namespace", _ROS)
    topics = {}
    for name in ("scan_topic", "odom_topic", "cmd_vel_topic", "amcl_topic",
                 "compressed_image_topic", "camera_info_topic"):
        topic = _text(robot.get(name), name, _ROS)
        topics[name] = "/" + "/".join(p for p in ((namespace or "").strip("/"), topic.strip("/")) if p) if not topic.startswith("/") else topic
    if len(set(topics.values())) != 6:
        raise ValueError("physical sensor and command topics must be distinct")
    frames = {name: _text(robot.get(name), name, _FRAME) for name in
              ("map_frame", "odom_frame", "base_frame", "scan_frame", "camera_optical_frame")}
    if frames["map_frame"] == frames["odom_frame"] or frames["base_frame"] != calibration.base_frame or frames["camera_optical_frame"] != calibration.camera_optical_frame:
        raise ValueError("measured runtime/calibration frames disagree")
    numbers = {name: _number(robot.get(name), name, positive=True) for name in
               ("robot_radius_m", "max_linear_speed_mps", "max_angular_speed_radps")}
    numbers["scan_origin_to_base_offset_m"] = _number(robot.get("scan_origin_to_base_offset_m"), "scan offset", nonnegative=True)
    config = Config(_text(robot.get("robot_id"), "robot_id", _ID), namespace,
        **topics, **frames, **numbers, calibration=calibration, model=model,
        map_yaml=paths["map_yaml"], semantic_map_id=semantic_map_id,
        expected_yaml_sha256=measurement["map_yaml_sha256"],
        expected_image_sha256=measurement["map_image_sha256"],
        expected_map_bundle_sha256=measurement["map_bundle_sha256"],
        approach_offset_m=_number(approach_offset_m, "approach offset", positive=True),
        stop_spacing_m=_number(stop_spacing_m, "stop spacing", positive=True),
        camera_timeout_sec=_number(camera_timeout_sec, "camera timeout", positive=True))
    if config.approach_offset_m < config.minimum_standoff_m:
        raise ValueError("camera approach is below physical collision/LiDAR clearance")
    config.camera_heading_offset_rad
    return config
