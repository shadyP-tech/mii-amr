"""
tracker.py — Detect green circular markers in a camera frame.

Pipeline per frame:
    BGR → HSV → threshold green → morphological cleanup → find contours
    → filter by area / radius / circularity → return marker centers.

When run standalone (`python3 tracker.py`), opens a live camera feed with
annotated detections and prints pixel coordinates to stdout.
"""

import cv2
import numpy as np
import config


def detect_markers(frame):
    """Detect green circular blobs in *frame*.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (already resized if desired).

    Returns
    -------
    list of (int, int, float)
        Up to 3 detected marker centers as ``(x, y, radius)`` in pixel
        coordinates, sorted by radius descending.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, config.HSV_LOWER, config.HSV_UPPER)

    # Morphological cleanup — remove noise, fill small holes
    kernel = np.ones((config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < config.MIN_CONTOUR_AREA:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < config.MIN_RADIUS or radius > config.MAX_RADIUS:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < config.MIN_CIRCULARITY:
            continue

        centers.append((int(x), int(y), float(radius)))

    # Sort all detected centers by radius descending
    centers = sorted(centers, key=lambda c: c[2], reverse=True)

    if len(centers) >= 3:
        # Assume the 2 largest blobs are the confident markers on the robot
        m0, m1 = centers[0], centers[1]
        mid_x = (m0[0] + m1[0]) / 2.0
        mid_y = (m0[1] + m1[1]) / 2.0
        
        # Look for the third marker that is closest to the robot's midpoint
        best_m2 = None
        min_dist = float('inf')
        
        for c in centers[2:]:
            dist = np.hypot(c[0] - mid_x, c[1] - mid_y)
            # The robot is small, so the third marker must be nearby (<200 pixels)
            if dist < 200 and dist < min_dist:
                min_dist = dist
                best_m2 = c
                
        if best_m2:
            centers = [m0, m1, best_m2]
        else:
            centers = [m0, m1]
    else:
        # If 1 or 2 are detected, just return them
        pass

    return centers, mask


def draw_markers(frame, centers):
    """Annotate *frame* in-place with circles and labels for *centers*."""
    for i, (x, y, r) in enumerate(centers):
        cv2.circle(frame, (x, y), int(r), (0, 0, 255), 2)
        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
        cv2.putText(
            frame,
            f"M{i}: ({x},{y}) r={r:.0f}",
            (x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )


def main():
    """Open camera, detect markers, display and print in a loop."""
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {config.CAMERA_INDEX}")
        return

    print("tracker.py — press ESC to quit")
    print(f"HSV range: {config.HSV_LOWER} – {config.HSV_UPPER}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Failed to read frame")
            break

        frame = cv2.resize(frame, None, fx=config.RESIZE_SCALE, fy=config.RESIZE_SCALE)

        centers, mask = detect_markers(frame)
        draw_markers(frame, centers)

        # Print pixel coordinates
        if centers:
            coords = "  ".join(
                f"M{i}=({x},{y} r={r:.0f})" for i, (x, y, r) in enumerate(centers)
            )
            print(f"[{len(centers)} detected]  {coords}")

        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
