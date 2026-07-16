"""Decode the known Gazebo station QR matrices without an external QR backend."""

from __future__ import annotations

from dataclasses import dataclass

from scripts.aufgabe04.simulation.generate_gazebo_world import QR_SIZE, qr_matrix


@dataclass(frozen=True)
class SimulatedQrDetection:
    station_id: str
    corners_px: tuple[tuple[float, float], ...]
    mismatch_fraction: float


def detect_simulated_station_qr_bgr(
    frame,
    cv2,
    *,
    roi: tuple[int, int, int, int] | None = None,
    station_ids: tuple[str, ...] = ("A", "B", "C"),
    max_mismatch_fraction: float = 0.12,
) -> SimulatedQrDetection | None:
    """Match a simulated QR inside an optional target ROI.

    Returned corners always use full-frame coordinates.  Callers with a known
    LiDAR stand must pass its projected ROI instead of searching the complete
    multi-stand image.
    """

    try:
        import numpy
    except ImportError:
        return None
    x_offset = 0
    y_offset = 0
    detection_frame = frame
    if roi is not None:
        x0, y0, x1, y1 = roi
        height, width = frame.shape[:2]
        x0 = max(0, min(width, int(x0)))
        x1 = max(0, min(width, int(x1)))
        y0 = max(0, min(height, int(y0)))
        y1 = max(0, min(height, int(y1)))
        if x1 <= x0 or y1 <= y0:
            return None
        x_offset, y_offset = x0, y0
        detection_frame = frame[y0:y1, x0:x1]

    detector = cv2.QRCodeDetector()
    try:
        found, points = detector.detect(detection_frame)
    except Exception:
        return None
    if not found or points is None:
        return None
    corners = numpy.asarray(points, dtype=numpy.float32).reshape(-1, 2)
    if corners.shape != (4, 2):
        return None
    station_id, mismatch = _match_sampled_qr(
        detection_frame, cv2, numpy, corners, station_ids=station_ids
    )
    if station_id is None or mismatch > max_mismatch_fraction:
        return None
    full_frame_corners = corners + numpy.asarray(
        [x_offset, y_offset], dtype=numpy.float32
    )
    return SimulatedQrDetection(
        station_id=station_id,
        corners_px=tuple(
            (float(point[0]), float(point[1])) for point in full_frame_corners
        ),
        mismatch_fraction=mismatch,
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
