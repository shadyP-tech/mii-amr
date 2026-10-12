import cv2
import numpy as np
import config


def detect_markers(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, config.HSV_LOWER, config.HSV_UPPER)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []

    for contour in contours:
        area = cv2.contourArea(contour)

        (x, y), radius = cv2.minEnclosingCircle(contour)

        perimeter = cv2.arcLength(contour, True)
        circularity = 0.0
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)

        circle_area = np.pi * radius * radius
        fill_ratio = area / circle_area if circle_area > 0 else 0.0

        if area < config.MIN_CONTOUR_AREA:
            continue

        if radius < config.MIN_RADIUS or radius > config.MAX_RADIUS:
            continue

        if perimeter == 0:
            continue

        if circularity < config.MIN_CIRCULARITY and fill_ratio < config.MIN_FILL_RATIO:
            continue

        centers.append((int(x), int(y), float(radius)))

    # sort all detected centers by radius descending
    centers = sorted(centers, key=lambda c: c[2], reverse=True)

    # separate likely large and small markers.
    large_candidates = [c for c in centers if c[2] >= 35]
    small_candidates = [c for c in centers if 12 <= c[2] < 35]

    selected = []

    if len(large_candidates) >= 2:
        m0, m1 = large_candidates[0], large_candidates[1]
        selected = [m0, m1]

        valid_small = []
        for c in small_candidates:
            d0 = np.hypot(c[0] - m0[0], c[1] - m0[1])
            d1 = np.hypot(c[0] - m1[0], c[1] - m1[1])

            if min(d0, d1) > 90:
                valid_small.append(c)

        if valid_small:
            m2 = sorted(valid_small, key=lambda c: c[2], reverse=True)[0]
            selected.append(m2)

        centers = selected

    elif len(centers) > 3:
        centers = centers[:3]

    return centers, mask
