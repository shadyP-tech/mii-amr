"""Current-pixel QR marker evidence, separate from quadrilateral proposals.

OpenCV can return a QR-shaped quadrilateral on a plain stand back. That proposal
must still veto backside use in its own frame, but must not become a persistent
front-face observation. This module verifies three finder patterns without
another decoder or a metric pose. It supplies no identity or motion authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.aufgabe04.perception.stand_axis.qr_pose_seed import QrQuadDetection


@dataclass(frozen=True)
class QrMarkerEvidence:
    verified: bool
    reason: str
    finder_count: int = 0
    minimum_finder_score: float | None = None


def _ordered_quad(np, points):
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    return np.roll(ordered, -int(np.argmin(ordered.sum(axis=1))), axis=0)


def _finder_score(cv2, np, binary, corners) -> float:
    """Check black/white/black rings around an independently found white hole."""

    # A finder has 7x7 modules; the enclosed white square spans 5x5. Using that
    # square avoids requiring a detached outer ring (printing can connect it to
    # the symbol or paper border). Each ring must have current pixel support.
    transform = cv2.getPerspectiveTransform(
        corners, np.float32(((12, 12), (72, 12), (72, 72), (12, 72))),
    )
    pattern = cv2.warpPerspective(
        binary, transform, (84, 84), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    cells = pattern.reshape(7, 12, 7, 12).transpose(0, 2, 1, 3)
    # Sample inside each module instead of its antialiased boundary.
    darkness = cells[:, :, 3:9, 3:9].mean(axis=(2, 3)) / 255.0
    row, column = np.indices((7, 7))
    ring = np.minimum.reduce((row, column, 6 - row, 6 - column))
    return min(
        float(darkness[ring == 0].mean()),
        float((1.0 - darkness[ring == 1]).mean()),
        float(darkness[ring >= 2].mean()),
    )


def validate_qr_marker(
    cv2,
    frame,
    detection: QrQuadDetection | None,
) -> QrMarkerEvidence:
    """Validate decoded identity or three QR finder patterns in current pixels.

    The input quadrilateral is always in ``frame`` coordinates. Its acquisition
    and current-frame veto are deliberately unchanged when verification fails.
    Callers must separately enforce frame freshness and candidate association.
    Decoded text is authoritative marker evidence even when a pose cannot fit.
    """

    if detection is None:
        return QrMarkerEvidence(False, "no_qr_quadrilateral")
    if detection.text:
        return QrMarkerEvidence(True, "decoded_qr_identity")
    if (
        frame is None or not hasattr(frame, "shape")
        or len(frame.shape) not in (2, 3)
        or min(frame.shape[:2]) <= 0
        or (len(frame.shape) == 3 and frame.shape[2] != 3)
    ):
        return QrMarkerEvidence(False, "invalid_marker_image")

    import numpy as np

    try:
        coordinates = tuple((float(p.u_px), float(p.v_px)) for p in detection.corners)
    except (TypeError, ValueError, AttributeError):
        return QrMarkerEvidence(False, "invalid_marker_quadrilateral")
    if len(coordinates) != 4 or not all(math.isfinite(v) for p in coordinates for v in p):
        return QrMarkerEvidence(False, "invalid_marker_quadrilateral")
    corners = np.asarray(coordinates, dtype=np.float32)
    height, width = frame.shape[:2]
    if (
        np.any(corners[:, 0] < 0) or np.any(corners[:, 0] > width - 1)
        or np.any(corners[:, 1] < 0) or np.any(corners[:, 1] > height - 1)
        or not cv2.isContourConvex(corners)
        or abs(cv2.contourArea(corners)) < 144.0
        or float(np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1).min()) < 12.0
    ):
        return QrMarkerEvidence(False, "invalid_marker_quadrilateral")

    # Bounded work independent of full camera resolution. The margin preserves
    # finder borders when the native detector's corner locations are biased.
    transform = cv2.getPerspectiveTransform(
        corners, np.float32(((32, 32), (223, 32), (223, 223), (32, 223))),
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    patch = cv2.warpPerspective(
        gray, transform, (256, 256), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    _, binary = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return QrMarkerEvidence(False, "qr_finder_patterns_unverified")
    hierarchy = hierarchy[0]
    # One best finder in each symbol quadrant; repeated nested contours cannot
    # count multiple times. Three finders must have comparable physical scale.
    finders: dict[tuple[bool, bool], tuple[float, float]] = {}
    for index, contour in enumerate(contours):
        child = int(hierarchy[index, 2])
        if child < 0:
            continue
        depth, parent = 0, int(hierarchy[index, 3])
        while parent >= 0:
            depth += 1
            parent = int(hierarchy[parent, 3])
        if depth % 2 != 1:  # White hole surrounded by black pixels.
            continue
        area = float(cv2.contourArea(contour))
        core_area = float(cv2.contourArea(contours[child]))
        if area < 25.0 or not 0.20 <= core_area / area <= 0.55:
            continue
        polygon = cv2.approxPolyDP(contour, 0.035 * cv2.arcLength(contour, True), True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        ordered = _ordered_quad(np, polygon.reshape(4, 2).astype(np.float32))
        center = ordered.mean(axis=0)
        normalized = (center - 32.0) / 191.0
        if (
            np.any(normalized < -0.05) or np.any(normalized > 1.05)
            or any(0.40 < float(value) < 0.60 for value in normalized)
        ):
            continue
        core_moments = cv2.moments(contours[child])
        if not core_moments["m00"]:
            continue
        core_center = np.asarray((
            core_moments["m10"] / core_moments["m00"],
            core_moments["m01"] / core_moments["m00"],
        ))
        if float(np.linalg.norm(core_center - center)) > math.sqrt(area) * 0.15:
            continue
        score = _finder_score(cv2, np, binary, ordered)
        if score < 0.80:
            continue
        quadrant = (bool(normalized[0] >= 0.5), bool(normalized[1] >= 0.5))
        if quadrant not in finders or score > finders[quadrant][0]:
            finders[quadrant] = (score, area)

    # Evaluate every bounded triple (at most four) so an unrelated fourth
    # feature cannot defeat three internally consistent finder patterns.
    from itertools import combinations

    for triple in combinations(finders.values(), 3):
        areas = tuple(item[1] for item in triple)
        if max(areas) / min(areas) <= 2.0:
            return QrMarkerEvidence(
                True, "three_current_pixel_qr_finders", 3,
                min(item[0] for item in triple),
            )
    return QrMarkerEvidence(
        False, "qr_finder_patterns_unverified", len(finders),
        min((value[0] for value in finders.values()), default=None),
    )
