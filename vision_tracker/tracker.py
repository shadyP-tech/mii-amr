import cv2
import numpy as np
import config
import camera


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

    print("tracker.py — press ESC to quit")
    print(f"HSV range: {config.HSV_LOWER} - {config.HSV_UPPER}")

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
