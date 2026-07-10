from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from scripts.aufgabe04.perception.stand_axis_image import ImagePoint, StandAxisImageEstimate, order_corners


@dataclass(frozen=True)
class StandSideClassification:
    side: str
    confidence: float
    qr_texts: tuple[str, ...]
    color_confidence: float
    reason: str


def classify_stand_side(
    *,
    qr_texts: Sequence[str],
    color_confidence: float,
    min_color_confidence: float = 0.20,
    allow_color_only: bool = True,
) -> StandSideClassification:
    clean_qr_texts = tuple(text.strip() for text in qr_texts if str(text).strip())
    if clean_qr_texts:
        return StandSideClassification(
            side="qr_code_side",
            confidence=1.0,
            qr_texts=clean_qr_texts,
            color_confidence=color_confidence,
            reason="qr_detected",
        )
    if allow_color_only and color_confidence >= min_color_confidence:
        return StandSideClassification(
            side="basic_color_side",
            confidence=min(1.0, color_confidence / max(min_color_confidence, 1e-9)),
            qr_texts=(),
            color_confidence=color_confidence,
            reason="stand_color_detected_without_qr",
        )
    return StandSideClassification(
        side="unknown_side",
        confidence=0.0,
        qr_texts=(),
        color_confidence=color_confidence,
        reason=(
            "color_only_evidence_not_allowed"
            if not allow_color_only and color_confidence >= min_color_confidence
            else "no_qr_and_low_color_confidence"
        ),
    )


def classify_stand_side_from_frame(
    cv2,
    numpy,
    frame,
    color_mask,
    estimate: StandAxisImageEstimate,
    *,
    detect_qr_texts_bgr,
    min_color_confidence: float = 0.20,
    qr_crop_margin_px: int = 8,
    allow_color_only: bool = True,
) -> StandSideClassification:
    qr_texts = ()
    for qr_frame in _qr_scan_frames_for_estimate(
        cv2,
        numpy,
        frame,
        estimate,
        margin_px=qr_crop_margin_px,
    ):
        qr_texts = detect_qr_texts_bgr(qr_frame, cv2)
        if qr_texts:
            break
    color_confidence = color_confidence_for_estimate(cv2, numpy, color_mask, estimate)
    return classify_stand_side(
        qr_texts=qr_texts,
        color_confidence=color_confidence,
        min_color_confidence=min_color_confidence,
        allow_color_only=allow_color_only,
    )


def color_confidence_for_estimate(cv2, numpy, color_mask, estimate: StandAxisImageEstimate) -> float:
    if color_mask is None or estimate.corners is None:
        return 0.0
    polygon_mask = numpy.zeros(color_mask.shape[:2], dtype=numpy.uint8)
    polygon = numpy.array(
        [[(int(round(point.u_px)), int(round(point.v_px))) for point in order_corners(estimate.corners)]],
        dtype=numpy.int32,
    )
    cv2.fillPoly(polygon_mask, polygon, 255)
    total = int(cv2.countNonZero(polygon_mask))
    if total <= 0:
        return 0.0
    matched = cv2.bitwise_and(color_mask, color_mask, mask=polygon_mask)
    return int(cv2.countNonZero(matched)) / total


def _crop_frame_to_estimate(cv2, frame, estimate: StandAxisImageEstimate, *, margin_px: int):
    if estimate.corners is None:
        return frame
    x_min, y_min, x_max, y_max = _corner_bounds(estimate.corners)
    x_min = max(0, int(round(x_min)) - margin_px)
    y_min = max(0, int(round(y_min)) - margin_px)
    x_max = min(frame.shape[1], int(round(x_max)) + margin_px)
    y_max = min(frame.shape[0], int(round(y_max)) + margin_px)
    if x_max <= x_min or y_max <= y_min:
        return frame
    return frame[y_min:y_max, x_min:x_max]


def _qr_scan_frames_for_estimate(cv2, numpy, frame, estimate: StandAxisImageEstimate, *, margin_px: int):
    frames = []
    rectified = _rectify_frame_to_estimate(cv2, numpy, frame, estimate)
    if rectified is not None:
        frames.append(rectified)
    crop = _crop_frame_to_estimate(cv2, frame, estimate, margin_px=max(margin_px, 18))
    if crop is not frame:
        frames.append(crop)
    frames.append(frame)
    return tuple(frames)


def _rectify_frame_to_estimate(cv2, numpy, frame, estimate: StandAxisImageEstimate):
    if estimate.corners is None:
        return None
    top_left, top_right, bottom_right, bottom_left = order_corners(estimate.corners)
    width = int(round(max(_distance(top_left, top_right), _distance(bottom_left, bottom_right))))
    height = int(round(max(_distance(top_left, bottom_left), _distance(top_right, bottom_right))))
    if width < 16 or height < 16:
        return None

    src = numpy.array(
        [
            [top_left.u_px, top_left.v_px],
            [top_right.u_px, top_right.v_px],
            [bottom_right.u_px, bottom_right.v_px],
            [bottom_left.u_px, bottom_left.v_px],
        ],
        dtype=numpy.float32,
    )
    dst = numpy.array(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=numpy.float32,
    )
    try:
        transform = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, transform, (width, height))
    except Exception:
        return None


def _corner_bounds(corners: Sequence[ImagePoint]) -> tuple[float, float, float, float]:
    return (
        min(point.u_px for point in corners),
        min(point.v_px for point in corners),
        max(point.u_px for point in corners),
        max(point.v_px for point in corners),
    )


def _distance(first: ImagePoint, second: ImagePoint) -> float:
    return ((second.u_px - first.u_px) ** 2 + (second.v_px - first.v_px) ** 2) ** 0.5
