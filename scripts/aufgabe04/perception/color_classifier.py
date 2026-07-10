from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from .models import ColorClassification, ColorClassifierConfig, ColorRange


HSVPixel = Tuple[int, int, int]


DEFAULT_STAND_PALETTE = (
    ColorRange("red", (0, 70, 50), (10, 255, 255)),
    ColorRange("red", (170, 70, 50), (179, 255, 255)),
    ColorRange("yellow", (20, 70, 50), (38, 255, 255)),
    ColorRange("green", (45, 55, 45), (95, 255, 255)),
    ColorRange("blue", (100, 60, 45), (130, 255, 255)),
)


def validate_color_range(color_range: ColorRange) -> None:
    for name, values in (("lower_hsv", color_range.lower_hsv), ("upper_hsv", color_range.upper_hsv)):
        if len(values) != 3:
            raise ValueError(f"{name} must contain exactly three HSV values")
    lower_h, lower_s, lower_v = color_range.lower_hsv
    upper_h, upper_s, upper_v = color_range.upper_hsv
    for hue in (lower_h, upper_h):
        if hue < 0 or hue > 179:
            raise ValueError("HSV hue must be in OpenCV range 0..179")
    for value in (lower_s, lower_v, upper_s, upper_v):
        if value < 0 or value > 255:
            raise ValueError("HSV saturation/value must be in range 0..255")
    if lower_s > upper_s or lower_v > upper_v:
        raise ValueError("HSV lower saturation/value must not exceed upper bound")


def hsv_pixel_in_range(pixel: HSVPixel, color_range: ColorRange) -> bool:
    validate_color_range(color_range)
    hue, saturation, value = pixel
    lower_h, lower_s, lower_v = color_range.lower_hsv
    upper_h, upper_s, upper_v = color_range.upper_hsv
    if saturation < lower_s or saturation > upper_s:
        return False
    if value < lower_v or value > upper_v:
        return False
    if lower_h <= upper_h:
        return lower_h <= hue <= upper_h
    return hue >= lower_h or hue <= upper_h


def flatten_hsv_pixels(hsv_pixels: Iterable[HSVPixel] | Iterable[Iterable[HSVPixel]]) -> List[HSVPixel]:
    flattened: List[HSVPixel] = []
    for item in hsv_pixels:
        if isinstance(item, tuple) and len(item) == 3:
            flattened.append((int(item[0]), int(item[1]), int(item[2])))
            continue
        for pixel in item:  # type: ignore[union-attr]
            flattened.append((int(pixel[0]), int(pixel[1]), int(pixel[2])))
    return flattened


def score_hsv_pixels(
    hsv_pixels: Sequence[HSVPixel],
    color_range: ColorRange,
) -> ColorClassification:
    total = len(hsv_pixels)
    if total == 0:
        return ColorClassification(color_range.label, 0.0, 0, 0)
    matched = sum(1 for pixel in hsv_pixels if hsv_pixel_in_range(pixel, color_range))
    return ColorClassification(
        label=color_range.label,
        confidence=matched / total,
        matched_pixels=matched,
        total_pixels=total,
    )


def classify_hsv_pixels(
    hsv_pixels: Iterable[HSVPixel] | Iterable[Iterable[HSVPixel]],
    *,
    palette: Sequence[ColorRange] = DEFAULT_STAND_PALETTE,
    config: ColorClassifierConfig | None = None,
    timestamp_sec: float | None = None,
) -> ColorClassification:
    cfg = config or ColorClassifierConfig()
    pixels = flatten_hsv_pixels(hsv_pixels)
    if not pixels:
        return ColorClassification(cfg.unknown_label, 0.0, 0, 0, timestamp_sec=timestamp_sec)

    scores = [score_hsv_pixels(pixels, color_range) for color_range in palette]
    by_label = {}
    for score in scores:
        previous = by_label.get(score.label)
        if previous is None or score.confidence > previous.confidence:
            by_label[score.label] = score
    best = max(by_label.values(), key=lambda score: score.confidence, default=None)
    if best is None or best.confidence < cfg.min_confidence:
        return ColorClassification(
            cfg.unknown_label,
            best.confidence if best is not None else 0.0,
            best.matched_pixels if best is not None else 0,
            len(pixels),
            timestamp_sec=timestamp_sec,
        )
    return ColorClassification(
        best.label,
        best.confidence,
        best.matched_pixels,
        best.total_pixels,
        timestamp_sec=timestamp_sec,
    )

