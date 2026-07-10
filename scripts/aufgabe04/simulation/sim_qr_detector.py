"""Decode the known Gazebo station QR matrices without an external QR backend."""

from __future__ import annotations

import math
from dataclasses import dataclass

from scripts.aufgabe04.simulation.generate_gazebo_world import QR_SIZE, qr_matrix


@dataclass(frozen=True)
class SimulatedQrDetection:
    station_id: str
    corners_px: tuple[tuple[float, float], ...]
    mismatch_fraction: float
    face_yaw_rad: float | None


def detect_simulated_station_qr_bgr(
    frame,
    cv2,
    *,
    camera_fx_px: float | None = None,
    camera_fy_px: float | None = None,
    camera_cx_px: float | None = None,
    camera_cy_px: float | None = None,
    station_ids: tuple[str, ...] = ("A", "B", "C"),
    max_mismatch_fraction: float = 0.12,
) -> SimulatedQrDetection | None:
    """Match a simulated QR and recover its face-normal yaw with square PnP."""

    try:
        import numpy
    except ImportError:
        return None
    detector = cv2.QRCodeDetector()
    try:
        found, points = detector.detect(frame)
    except Exception:
        return None
    if not found or points is None:
        return None
    corners = numpy.asarray(points, dtype=numpy.float32).reshape(-1, 2)
    if corners.shape != (4, 2):
        return None
    station_id, mismatch = _match_sampled_qr(
        frame, cv2, numpy, corners, station_ids=station_ids
    )
    if station_id is None or mismatch > max_mismatch_fraction:
        return None
    face_yaw_rad = _square_face_yaw_rad(
        cv2,
        numpy,
        corners,
        camera_fx_px=camera_fx_px,
        camera_fy_px=camera_fy_px,
        camera_cx_px=camera_cx_px,
        camera_cy_px=camera_cy_px,
    )
    return SimulatedQrDetection(
        station_id=station_id,
        corners_px=tuple((float(point[0]), float(point[1])) for point in corners),
        mismatch_fraction=mismatch,
        face_yaw_rad=face_yaw_rad,
    )


def detect_simulated_station_qr_texts_bgr(
    frame,
    cv2,
    *,
    station_ids: tuple[str, ...] = ("A", "B", "C"),
    max_mismatch_fraction: float = 0.12,
) -> tuple[str, ...]:
    """Detect a QR quadrilateral and match its sampled modules to station IDs."""

    detection = detect_simulated_station_qr_bgr(
        frame,
        cv2,
        station_ids=station_ids,
        max_mismatch_fraction=max_mismatch_fraction,
    )
    return () if detection is None else (detection.station_id,)


def _match_sampled_qr(frame, cv2, numpy, corners, *, station_ids):
    pixels_per_module = 8
    size = QR_SIZE * pixels_per_module
    destination = numpy.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=numpy.float32,
    )
    transform = cv2.getPerspectiveTransform(corners, destination)
    warped = cv2.warpPerspective(frame, transform, (size, size))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
    _threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sampled = numpy.zeros((QR_SIZE, QR_SIZE), dtype=numpy.uint8)
    for row in range(QR_SIZE):
        for col in range(QR_SIZE):
            y = int((row + 0.5) * pixels_per_module)
            x = int((col + 0.5) * pixels_per_module)
            sampled[row, col] = 1 if binary[y, x] < 128 else 0

    best_id = None
    best_mismatch = 1.0
    for station_id in station_ids:
        expected = numpy.asarray(qr_matrix(station_id), dtype=numpy.uint8)
        for turns in range(4):
            mismatch = float(numpy.mean(sampled != numpy.rot90(expected, turns)))
            if mismatch < best_mismatch:
                best_id, best_mismatch = station_id, mismatch
    return best_id, best_mismatch


def _square_face_yaw_rad(
    cv2,
    numpy,
    corners,
    *,
    camera_fx_px,
    camera_fy_px,
    camera_cx_px,
    camera_cy_px,
):
    values = (camera_fx_px, camera_fy_px, camera_cx_px, camera_cy_px)
    if any(value is None or not math.isfinite(value) for value in values):
        return None
    object_points = numpy.array(
        [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]],
        dtype=numpy.float64,
    )
    camera_matrix = numpy.array(
        [[camera_fx_px, 0.0, camera_cx_px], [0.0, camera_fy_px, camera_cy_px], [0.0, 0.0, 1.0]],
        dtype=numpy.float64,
    )
    try:
        ok, rvec, _tvec = cv2.solvePnP(
            object_points,
            corners.astype(numpy.float64),
            camera_matrix,
            numpy.zeros((4, 1), dtype=numpy.float64),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None
        rotation, _jacobian = cv2.Rodrigues(rvec)
    except Exception:
        return None
    normal = rotation @ numpy.array([[0.0], [0.0], [1.0]])
    return math.atan2(float(normal[0, 0]), abs(float(normal[2, 0])))
