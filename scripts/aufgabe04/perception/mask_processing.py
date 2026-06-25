from __future__ import annotations

from typing import Sequence

from scripts.aufgabe04.perception.models import ColorClassification, ColorRange
from scripts.aufgabe04.perception.roi import Rect, clamp_roi


def build_mask_for_ranges(cv2, numpy, hsv_frame, color_ranges: Sequence[ColorRange]):
    mask = numpy.zeros(hsv_frame.shape[:2], dtype=numpy.uint8)
    for color_range in color_ranges:
        lower = numpy.array(color_range.lower_hsv, dtype=numpy.uint8)
        upper = numpy.array(color_range.upper_hsv, dtype=numpy.uint8)
        range_mask = cv2.inRange(hsv_frame, lower, upper)
        mask = cv2.bitwise_or(mask, range_mask)
    return mask


def apply_morphology(cv2, mask, *, kernel_size: int, close_iterations: int, open_iterations: int):
    if kernel_size <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    return cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=open_iterations)


def classify_mask_roi(cv2, mask, roi: Rect, label: str) -> ColorClassification:
    clipped = clamp_roi(roi, mask.shape)
    total = clipped.width * clipped.height
    if total <= 0:
        return ColorClassification("unknown", 0.0, 0, 0)
    mask_roi = mask[clipped.y : clipped.y + clipped.height, clipped.x : clipped.x + clipped.width]
    matched = int(cv2.countNonZero(mask_roi))
    return ColorClassification(label, matched / total, matched, total)
