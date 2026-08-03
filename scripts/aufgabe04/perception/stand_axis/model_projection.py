"""Projection of semantic stand landmarks into a rectified camera image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from scripts.aufgabe04.perception.stand_axis.model_profile import (
    ModelPoint3D,
    StandModelProfile,
)
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import (
    PlanarPoseHypothesis,
    RectifiedCameraMatrix,
)


@dataclass(frozen=True)
class ProjectedStandModel:
    landmarks: Mapping[str, ImagePoint]
    head_corners: tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint]
    head_back_corners: tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint]


def project_model_points(
    cv2,
    points: Mapping[str, ModelPoint3D],
    pose: PlanarPoseHypothesis,
    camera: RectifiedCameraMatrix,
) -> dict[str, ImagePoint]:
    import numpy

    camera.validate()
    names = tuple(points)
    object_points = numpy.asarray(
        [[points[name].x_m, points[name].y_m, points[name].z_m] for name in names],
        dtype=numpy.float64,
    )
    camera_matrix = numpy.asarray(
        (
            (camera.fx_px, 0.0, camera.cx_px),
            (0.0, camera.fy_px, camera.cy_px),
            (0.0, 0.0, 1.0),
        ),
        dtype=numpy.float64,
    )
    projected, _jacobian = cv2.projectPoints(
        object_points,
        numpy.asarray(pose.rotation_vector, dtype=numpy.float64).reshape(3, 1),
        numpy.asarray(pose.translation_xyz_m, dtype=numpy.float64).reshape(3, 1),
        camera_matrix,
        numpy.zeros((4, 1), dtype=numpy.float64),
    )
    pixels = projected.reshape(-1, 2)
    return {
        name: ImagePoint(float(pixel[0]), float(pixel[1]))
        for name, pixel in zip(names, pixels)
    }


def project_stand_model(
    cv2,
    profile: StandModelProfile,
    pose: PlanarPoseHypothesis,
    camera: RectifiedCameraMatrix,
) -> ProjectedStandModel:
    landmarks = project_model_points(
        cv2,
        profile.semantic_landmarks,
        pose,
        camera,
    )
    head = tuple(
        landmarks[name]
        for name in (
            "head_top_left",
            "head_top_right",
            "head_bottom_right",
            "head_bottom_left",
        )
    )
    head_back = tuple(
        landmarks[name]
        for name in (
            "head_back_top_left",
            "head_back_top_right",
            "head_back_bottom_right",
            "head_back_bottom_left",
        )
    )
    return ProjectedStandModel(
        landmarks=landmarks,
        head_corners=head,
        head_back_corners=head_back,
    )
