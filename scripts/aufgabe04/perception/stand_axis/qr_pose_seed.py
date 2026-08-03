"""QR-corner acquisition and ambiguity-preserving planar pose estimation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from scripts.aufgabe04.perception.stand_axis.geometry import (
    _largest_qr_quad,
    order_corners,
)
from scripts.aufgabe04.perception.stand_axis.model_profile import ModelPoint3D
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint


@dataclass(frozen=True)
class RectifiedCameraMatrix:
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float

    def validate(self) -> None:
        values = (self.fx_px, self.fy_px, self.cx_px, self.cy_px)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("rectified camera matrix values must be finite")
        if self.fx_px <= 0.0 or self.fy_px <= 0.0:
            raise ValueError("rectified focal lengths must be positive")


@dataclass(frozen=True)
class QrQuadDetection:
    """QR symbol corners restored to the input image pixel domain."""

    corners: tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint]
    scale: float


@dataclass(frozen=True)
class PlanarPoseHypothesis:
    rotation_vector: tuple[float, float, float]
    translation_xyz_m: tuple[float, float, float]
    face_normal_xyz: tuple[float, float, float]
    yaw_deg: float
    reprojection_rmse_px: float
    positive_depth: bool


@dataclass(frozen=True)
class PlanarPoseResult:
    accepted: bool
    reason: str
    hypotheses: tuple[PlanarPoseHypothesis, ...]
    ambiguity_gap_px: float | None

    @property
    def best(self) -> PlanarPoseHypothesis | None:
        return self.hypotheses[0] if self.accepted and self.hypotheses else None

    def axis_ambiguous(
        self,
        *,
        max_residual_gap_px: float = 0.10,
        min_axis_separation_deg: float = 5.0,
    ) -> bool:
        """Whether similarly fitting planar solutions imply different axes."""

        if len(self.hypotheses) < 2 or self.ambiguity_gap_px is None:
            return False
        first, second = self.hypotheses[:2]
        axial_difference = abs(
            (first.yaw_deg - second.yaw_deg + 90.0) % 180.0 - 90.0
        )
        return (
            self.ambiguity_gap_px <= max_residual_gap_px
            and axial_difference >= min_axis_separation_deg
        )


def select_temporally_consistent_pose(
    result: PlanarPoseResult,
    reference: PlanarPoseHypothesis,
    *,
    max_axial_yaw_jump_deg: float = 8.0,
    max_translation_jump_m: float = 0.08,
    minimum_score_margin: float = 0.25,
) -> PlanarPoseHypothesis | None:
    """Resolve an IPPE ambiguity only when camera history clearly does so.

    LiDAR is deliberately excluded here: the camera pose must be independently
    estimated before it can be compared with the LiDAR axis.  Scores normalize
    axial yaw and translation changes by explicit temporal gates.  A candidate
    must pass both gates and be clearly closer than the runner-up; otherwise the
    ambiguity remains fail-closed.
    """

    if not result.accepted or not result.hypotheses:
        return None
    if (
        not math.isfinite(max_axial_yaw_jump_deg)
        or max_axial_yaw_jump_deg <= 0.0
        or not math.isfinite(max_translation_jump_m)
        or max_translation_jump_m <= 0.0
        or not math.isfinite(minimum_score_margin)
        or minimum_score_margin < 0.0
    ):
        raise ValueError("temporal pose gates must be finite and positive")

    ranked = []
    for hypothesis in result.hypotheses:
        yaw_jump = abs(
            (hypothesis.yaw_deg - reference.yaw_deg + 90.0) % 180.0 - 90.0
        )
        translation_jump = math.sqrt(
            sum(
                (current - previous) ** 2
                for current, previous in zip(
                    hypothesis.translation_xyz_m,
                    reference.translation_xyz_m,
                )
            )
        )
        score = (
            yaw_jump / max_axial_yaw_jump_deg
            + translation_jump / max_translation_jump_m
        )
        ranked.append((score, yaw_jump, translation_jump, hypothesis))
    ranked.sort(key=lambda item: item[0])
    best_score, yaw_jump, translation_jump, best = ranked[0]
    if (
        yaw_jump > max_axial_yaw_jump_deg
        or translation_jump > max_translation_jump_m
    ):
        return None
    if len(ranked) > 1 and ranked[1][0] - best_score < minimum_score_margin:
        return None
    return best


def _qr_quad_candidates(points) -> tuple[tuple[ImagePoint, ...], ...]:
    if points is None:
        return ()
    try:
        quadrilaterals = points.reshape(-1, 4, 2)
    except Exception:
        return ()
    return tuple(
        order_corners(
            tuple(
                ImagePoint(float(point[0]), float(point[1]))
                for point in quadrilateral
            )
        )
        for quadrilateral in quadrilaterals
    )


def _detect_qr_quad_corners_native(
    cv2,
    frame,
) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    """Run OpenCV's QR detector once in the supplied pixel domain."""

    detector = cv2.QRCodeDetector()
    try:
        ok, points = detector.detectMulti(frame)
    except Exception:
        ok, points = False, None
    candidates = _qr_quad_candidates(points) if ok else ()
    if candidates:
        return _largest_qr_quad(candidates)

    try:
        ok, points = detector.detect(frame)
    except Exception:
        ok, points = False, None
    candidates = _qr_quad_candidates(points) if ok else ()
    if candidates:
        return _largest_qr_quad(candidates)

    try:
        multi_result = detector.detectAndDecodeMulti(frame)
    except Exception:
        multi_result = ()
    points = multi_result[2] if len(multi_result) > 2 else None
    return _largest_qr_quad(_qr_quad_candidates(points))


def detect_qr_quad(
    cv2,
    frame,
    *,
    scales: Sequence[float] = (1.0, 2.0, 4.0),
) -> QrQuadDetection | None:
    """Acquire QR corners through a bounded image pyramid.

    The real 800x600 camera often renders the QR symbol too small for OpenCV
    4.5's native detector.  Upscaling is used only for acquisition; returned
    corners are always mapped back to the caller's original pixel domain.
    """

    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        raise ValueError("QR detection requires a non-empty image")
    if not scales:
        raise ValueError("QR detection scales must not be empty")
    for raw_scale in scales:
        scale = float(raw_scale)
        if not math.isfinite(scale) or scale < 1.0:
            raise ValueError("QR detection scales must be finite and at least 1.0")
        scaled = frame
        if scale != 1.0:
            scaled = cv2.resize(
                frame,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
        corners = _detect_qr_quad_corners_native(cv2, scaled)
        if corners is None:
            continue
        restored = tuple(
            ImagePoint(point.u_px / scale, point.v_px / scale)
            for point in corners
        )
        return QrQuadDetection(restored, scale)
    return None


def detect_qr_quad_corners(
    cv2,
    frame,
) -> tuple[ImagePoint, ImagePoint, ImagePoint, ImagePoint] | None:
    """Compatibility wrapper returning multi-scale QR symbol corners only."""

    detection = detect_qr_quad(cv2, frame)
    return None if detection is None else detection.corners


def estimate_planar_pose_ippe(
    cv2,
    image_points: Sequence[ImagePoint],
    model_points: Sequence[ModelPoint3D],
    camera: RectifiedCameraMatrix,
    *,
    max_reprojection_rmse_px: float = 2.0,
) -> PlanarPoseResult:
    """Return all physically visible IPPE hypotheses ordered by residual."""

    import numpy

    camera.validate()
    if len(image_points) != len(model_points) or len(image_points) < 4:
        raise ValueError("matching image/model point lists need at least four points")
    if not math.isfinite(max_reprojection_rmse_px) or max_reprojection_rmse_px <= 0.0:
        raise ValueError("max_reprojection_rmse_px must be finite and positive")
    model_z = [point.z_m for point in model_points]
    if max(model_z) - min(model_z) > 1.0e-9:
        raise ValueError("IPPE seed points must be coplanar")

    object_points = numpy.asarray(
        [[point.x_m, point.y_m, point.z_m] for point in model_points],
        dtype=numpy.float64,
    )
    pixels = numpy.asarray(
        [[point.u_px, point.v_px] for point in image_points],
        dtype=numpy.float64,
    )
    if not numpy.isfinite(object_points).all() or not numpy.isfinite(pixels).all():
        raise ValueError("pose points must be finite")
    camera_matrix = numpy.asarray(
        (
            (camera.fx_px, 0.0, camera.cx_px),
            (0.0, camera.fy_px, camera.cy_px),
            (0.0, 0.0, 1.0),
        ),
        dtype=numpy.float64,
    )
    distortion = numpy.zeros((4, 1), dtype=numpy.float64)
    try:
        result = cv2.solvePnPGeneric(
            object_points,
            pixels,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_IPPE,
        )
    except Exception:
        result = ()
    if not result or not bool(result[0]):
        return PlanarPoseResult(False, "ippe_failed", (), None)

    hypotheses = []
    for rotation_vector, translation_vector in zip(result[1], result[2]):
        rotation_vector = numpy.asarray(rotation_vector, dtype=numpy.float64).reshape(3, 1)
        translation_vector = numpy.asarray(translation_vector, dtype=numpy.float64).reshape(3, 1)
        try:
            projected, _jacobian = cv2.projectPoints(
                object_points,
                rotation_vector,
                translation_vector,
                camera_matrix,
                distortion,
            )
            rotation, _jacobian = cv2.Rodrigues(rotation_vector)
        except Exception:
            continue
        residual = projected.reshape(-1, 2) - pixels
        rmse = math.sqrt(float(numpy.mean(numpy.sum(residual * residual, axis=1))))
        normal = rotation @ numpy.asarray(((0.0,), (0.0,), (1.0,)), dtype=numpy.float64)
        normal_xyz = tuple(float(value) for value in normal.reshape(3))
        translation_xyz = tuple(float(value) for value in translation_vector.reshape(3))
        finite = all(math.isfinite(value) for value in (*normal_xyz, *translation_xyz, rmse))
        if not finite:
            continue
        hypotheses.append(
            PlanarPoseHypothesis(
                rotation_vector=tuple(float(value) for value in rotation_vector.reshape(3)),
                translation_xyz_m=translation_xyz,
                face_normal_xyz=normal_xyz,
                # Match the public stand-axis convention: positive image-left.
                yaw_deg=-math.degrees(math.atan2(normal_xyz[0], abs(normal_xyz[2]))),
                reprojection_rmse_px=rmse,
                positive_depth=translation_xyz[2] > 0.0,
            )
        )
    visible = tuple(
        sorted(
            (hypothesis for hypothesis in hypotheses if hypothesis.positive_depth),
            key=lambda hypothesis: hypothesis.reprojection_rmse_px,
        )
    )
    if not visible:
        return PlanarPoseResult(False, "no_positive_depth_pose", (), None)
    gap = (
        visible[1].reprojection_rmse_px - visible[0].reprojection_rmse_px
        if len(visible) >= 2
        else None
    )
    accepted = visible[0].reprojection_rmse_px <= max_reprojection_rmse_px
    return PlanarPoseResult(
        accepted,
        "pose_estimated" if accepted else "reprojection_error_too_high",
        visible,
        gap,
    )
