"""Shared calibrated-camera model and rectification helpers.

This module is ROS-free: callers may pass a real ``sensor_msgs/CameraInfo`` or
any test double exposing the same fields. OpenCV and NumPy are injected so pure
geometry imports remain available on machines without camera dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraCalibration:
    width_px: int
    height_px: int
    frame_id: str
    camera_matrix: tuple[float, ...]
    distortion: tuple[float, ...]
    rectification_matrix: tuple[float, ...]
    projection_matrix: tuple[float, ...]

    @property
    def fx_px(self) -> float:
        return self.projection_matrix[0]

    @property
    def fy_px(self) -> float:
        return self.projection_matrix[5]

    @property
    def cx_px(self) -> float:
        return self.projection_matrix[2]

    @property
    def cy_px(self) -> float:
        return self.projection_matrix[6]


def camera_calibration_from_info(camera_info: object) -> CameraCalibration:
    """Copy and validate the pixel geometry carried by ``CameraInfo``."""

    header = getattr(camera_info, "header", None)
    calibration = CameraCalibration(
        width_px=int(getattr(camera_info, "width", 0)),
        height_px=int(getattr(camera_info, "height", 0)),
        frame_id=str(getattr(header, "frame_id", "") or ""),
        camera_matrix=tuple(float(value) for value in getattr(camera_info, "k", ())),
        distortion=tuple(float(value) for value in getattr(camera_info, "d", ())),
        rectification_matrix=tuple(
            float(value) for value in getattr(camera_info, "r", ())
        ),
        projection_matrix=tuple(
            float(value) for value in getattr(camera_info, "p", ())
        ),
    )
    validate_camera_calibration(calibration)
    return calibration


def validate_camera_calibration(calibration: CameraCalibration) -> None:
    if calibration.width_px <= 0 or calibration.height_px <= 0:
        raise ValueError("camera calibration dimensions must be positive")
    if len(calibration.camera_matrix) != 9:
        raise ValueError("camera calibration K must contain 9 values")
    if len(calibration.rectification_matrix) != 9:
        raise ValueError("camera calibration R must contain 9 values")
    if len(calibration.projection_matrix) != 12:
        raise ValueError("camera calibration P must contain 12 values")
    values = (
        *calibration.camera_matrix,
        *calibration.distortion,
        *calibration.rectification_matrix,
        *calibration.projection_matrix,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("camera calibration values must be finite")
    if (
        calibration.camera_matrix[0] <= 0.0
        or calibration.camera_matrix[4] <= 0.0
    ):
        raise ValueError("camera calibration K focal lengths must be positive")
    if calibration.fx_px <= 0.0 or calibration.fy_px <= 0.0:
        raise ValueError("camera calibration P focal lengths must be positive")


def rectify_bgr_frame(
    frame: object,
    calibration_or_info: object,
    cv2_module: object,
    numpy_module: object,
) -> object:
    """Rectify a raw BGR image into the pixel geometry described by ``P``."""

    calibration = (
        calibration_or_info
        if isinstance(calibration_or_info, CameraCalibration)
        else camera_calibration_from_info(calibration_or_info)
    )
    validate_camera_calibration(calibration)
    height, width = frame.shape[:2]
    if width != calibration.width_px or height != calibration.height_px:
        raise ValueError(
            "decoded image dimensions do not match CameraInfo: "
            f"image={width}x{height}, "
            f"info={calibration.width_px}x{calibration.height_px}"
        )
    camera_matrix = numpy_module.asarray(
        calibration.camera_matrix, dtype=float
    ).reshape(3, 3)
    distortion = numpy_module.asarray(calibration.distortion, dtype=float)
    rectification = numpy_module.asarray(
        calibration.rectification_matrix, dtype=float
    ).reshape(3, 3)
    projection = numpy_module.asarray(
        calibration.projection_matrix, dtype=float
    ).reshape(3, 4)
    map_x, map_y = cv2_module.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        rectification,
        projection[:, :3],
        (width, height),
        cv2_module.CV_32FC1,
    )
    return cv2_module.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2_module.INTER_LINEAR,
        borderMode=cv2_module.BORDER_CONSTANT,
    )
