"""Shared HSV hints for physical stand borders; never measurement pixels."""

from scripts.aufgabe04.perception.mask_processing import build_mask_for_ranges
from scripts.aufgabe04.perception.models import ColorRange


# Initial palette for the five physical stands. September 11–17 camera
# recordings show purple near H=132 and a cool cast on the light-grey head
# (H=120–143, S=23–64), so achromatic grey alone would lose its borders.
# Avoid an unrestricted achromatic range: it selects the walls and QR paper.
# These are initial camera-lighting ranges, not color calibration.
STAND_EDGE_PALETTE = (
    ColorRange("light-grey", (105, 30, 65), (140, 85, 255)),
    ColorRange("blue", (96, 150, 40), (127, 255, 255)),
    ColorRange("green", (38, 65, 40), (95, 255, 255)),
    ColorRange("red", (0, 75, 40), (12, 255, 255)),
    ColorRange("red", (166, 75, 40), (179, 255, 255)),
    ColorRange("purple", (128, 100, 40), (165, 255, 255)),
)


def color_edge_support(cv2, numpy, frame, *, color="all", ranges=None):
    """Allow real edges within two pixels of colored exterior contours.

    Canny often places an outline just outside its colored surface. A small
    dilation keeps that outline without opening away thin colored rails.
    A small closing bridges segmentation pinholes; external contours omit QR
    texture enclosed by a colored border. Never generate measured edges from
    this binary mask: it only selects existing Canny pixels.
    Explicit ranges support the viewer's live HSV trackbars.
    """
    if ranges is None:
        ranges = tuple(item for item in STAND_EDGE_PALETTE
                       if color == "all" or item.label == color)
        if not ranges:
            raise ValueError(f"unknown edge color: {color}")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = build_mask_for_ranges(cv2, numpy, hsv, ranges)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    outline = numpy.zeros_like(mask)
    cv2.drawContours(outline, contours, -1, 255, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Closing can bridge gaps, but those bridges cannot license distant edges.
    return cv2.bitwise_and(cv2.dilate(outline, kernel, iterations=1),
                           cv2.dilate(mask, kernel, iterations=1))


def color_edge_exclusion(cv2, support, *, roi=None, existing=None):
    """Crop full-image color support into the detector's exact image domain."""
    if roi is not None:
        support = support[roi.y0:roi.y1, roi.x0:roi.x1]
    exclusion = cv2.bitwise_not(support)
    if existing is not None:
        if existing.shape != exclusion.shape:
            raise ValueError("existing edge exclusion must match the processing ROI")
        exclusion = cv2.bitwise_or(exclusion, existing)
    return exclusion
