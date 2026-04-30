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
import camera


def score_exposure(frame, hsv_lower, hsv_upper):
    """
    Higher score = better frame for green-marker tracking.
    Penalizes overexposure and rewards clean green blobs.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clipped_fraction = np.mean(gray > 250)

    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    green_pixels = np.count_nonzero(mask)

    saturation = hsv[:, :, 1]
    mean_green_saturation = np.mean(saturation[mask > 0]) if green_pixels > 0 else 0

    score = green_pixels + 10.0 * mean_green_saturation

    # Strongly reject frames with too much clipping
    if clipped_fraction > config.MAX_CLIPPED_FRACTION:
        score -= 1_000_000.0 * clipped_fraction

    return score, clipped_fraction, green_pixels


def auto_select_exposure(cap, hsv_lower, hsv_upper):
    """
    Try several exposure values and keep the best one.
    Works only if the camera/backend allows manual exposure control.
    """

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_GAIN, 0)

    exposure_candidates = config.EXPOSURE_CANDIDATES

    best_exposure = None
    best_score = -float("inf")

    for exposure in exposure_candidates:
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

        # Let camera settle
        for _ in range(10):
            cap.read()

        actual_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)

        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.resize(
            frame,
            None,
            fx=config.RESIZE_SCALE,
            fy=config.RESIZE_SCALE,
        )

        score, clipped_fraction, green_pixels = score_exposure(
            frame,
            hsv_lower,
            hsv_upper,
        )

        print(
            f"Requested exposure {exposure}, actual {actual_exposure}: "
            f"score={score:.1f}, "
            f"clipped={clipped_fraction:.4f}, "
            f"green_pixels={green_pixels}"
        )

        if score > best_score:
            best_score = score
            best_exposure = exposure

    if best_exposure is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, best_exposure)

        for _ in range(10):
            cap.read()

        actual_final_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)

        print(
            f"Selected exposure: requested {best_exposure}, "
            f"actual {actual_final_exposure}"
        )
    else:
        print("Could not select exposure automatically.")

    return best_exposure


def detect_markers(frame):
    """Detect green circular blobs in *frame*.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (already resized if desired).

    Returns
    -------
    tuple
        (centers, mask), where centers is a list of (x, y, radius)
        and mask is the binary green-threshold image.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, config.HSV_LOWER, config.HSV_UPPER)

    # Fill gaps before removing noise.  The green markers often have glare or
    # camera noise that makes their masks slightly broken; opening first can
    # split a real marker into irregular fragments.
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

        if config.DEBUG_CONTOURS:
            print(
                f"contour: x={x:.0f}, y={y:.0f}, "
                f"area={area:.1f}, r={radius:.1f}, "
                f"circ={circularity:.2f}, fill={fill_ratio:.2f}"
            )

        if area < config.MIN_CONTOUR_AREA:
            if config.DEBUG_CONTOURS:
                print("  rejected: area")
            continue

        if radius < config.MIN_RADIUS or radius > config.MAX_RADIUS:
            if config.DEBUG_CONTOURS:
                print("  rejected: radius")
            continue

        if perimeter == 0:
            if config.DEBUG_CONTOURS:
                print("  rejected: perimeter")
            continue

        if circularity < config.MIN_CIRCULARITY and fill_ratio < config.MIN_FILL_RATIO:
            if config.DEBUG_CONTOURS:
                print("  rejected: shape")
            continue

        centers.append((int(x), int(y), float(radius)))

    # Sort all detected centers by radius descending
    centers = sorted(centers, key=lambda c: c[2], reverse=True)

    # Separate likely large and small markers.
    # Tune these values for your resized frame.
    large_candidates = [c for c in centers if c[2] >= 35]
    small_candidates = [c for c in centers if 12 <= c[2] < 35]

    selected = []

    if len(large_candidates) >= 2:
        # The two largest real markers
        m0, m1 = large_candidates[0], large_candidates[1]
        selected = [m0, m1]

        # Reject small blobs that are too close to either large marker.
        # These are usually reflections or fragmented parts of the large balls.
        valid_small = []
        for c in small_candidates:
            d0 = np.hypot(c[0] - m0[0], c[1] - m0[1])
            d1 = np.hypot(c[0] - m1[0], c[1] - m1[1])

            if min(d0, d1) > 90:
                valid_small.append(c)

        if valid_small:
            # Prefer the largest remaining small marker
            m2 = sorted(valid_small, key=lambda c: c[2], reverse=True)[0]
            selected.append(m2)

        centers = selected

    elif len(centers) > 3:
        centers = centers[:3]

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
    try:
        cap = camera.open_camera()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return
    
    if config.AUTO_SELECT_EXPOSURE:
        auto_select_exposure(
            cap,
            config.HSV_LOWER,
            config.HSV_UPPER,
        )

    print("tracker.py — press ESC to quit")
    print(f"HSV range: {config.HSV_LOWER} – {config.HSV_UPPER}")

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Failed to read frame")
            break

        frame = cv2.resize(frame, None, fx=config.RESIZE_SCALE, fy=config.RESIZE_SCALE)

        centers, mask = detect_markers(frame)
        draw_markers(frame, centers)

        # Print pixel coordinates
        frame_count += 1
        
        if centers and frame_count % 15 == 0:
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
