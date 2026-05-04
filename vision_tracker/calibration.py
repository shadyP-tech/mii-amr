"""
calibration.py — Compute and store a pixel-to-world homography.

Usage:
    python3 calibration.py          # interactive: click 4 corners
    python3 calibration.py --verify # click test points after loading H

The homography maps image pixels to real-world meter coordinates for a
planar workspace viewed from a roughly top-down camera.

Limitations (document in report):
    - Assumes a planar workspace.
    - Neglects lens distortion (can add chessboard calibration later).
"""

import cv2
import numpy as np
import os
import config
import camera

# globals for mouse callback
_clicked_points = []
_click_done = False


def _mouse_callback(event, x, y, flags, param):
    """Collect clicked points on the calibration image."""
    global _clicked_points, _click_done
    if event == cv2.EVENT_LBUTTONDOWN and not _click_done:
        _clicked_points.append((x, y))
        print(f"  Point {len(_clicked_points)}: ({x}, {y})")
        if len(_clicked_points) >= 4:
            _click_done = True


# pixel → world transform
def load_homography(path=None):
    """Load a previously saved homography matrix.

    Returns
    -------
    np.ndarray or None
        3×3 homography matrix, or None if file does not exist.
    """
    path = path or config.HOMOGRAPHY_FILE
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return data["H"]


def pixel_to_world(pixel, H):
    """Convert a single pixel coordinate to world coordinates.

    Parameters
    ----------
    pixel : tuple of (float, float)
        (x, y) in image pixels.
    H : np.ndarray
        3×3 homography matrix (pixel → world).

    Returns
    -------
    np.ndarray
        [x, y] in world meters.
    """
    px = np.array([[[pixel[0], pixel[1]]]], dtype=np.float32)
    world = cv2.perspectiveTransform(px, H)
    return world[0, 0]  # shape (2,)


def world_to_pixel(point, H):
    """Convert a single world coordinate to image pixel coordinates.

    Parameters
    ----------
    point : tuple of (float, float)
        (x, y) in world meters.
    H : np.ndarray
        3x3 homography matrix (pixel -> world).

    Returns
    -------
    np.ndarray
        [x, y] in image pixels.
    """
    H_inv = np.linalg.inv(H)
    world = np.array([[[point[0], point[1]]]], dtype=np.float32)
    pixel = cv2.perspectiveTransform(world, H_inv)
    return pixel[0, 0]  # shape (2,)


def pixels_to_world(pixels, H):
    """Convert multiple pixel coordinates to world coordinates.

    Parameters
    ----------
    pixels : list of (float, float)
        Pixel coordinates.
    H : np.ndarray
        3×3 homography.

    Returns
    -------
    list of np.ndarray
        Each element is [x, y] in world meters.
    """
    return [pixel_to_world(p, H) for p in pixels]


# calibration procedure
def calibrate_interactive():
    """Run the interactive calibration: capture frame, click 4 corners, compute and save the homography.

    Returns
    -------
    np.ndarray
        3×3 homography matrix.
    """
    global _clicked_points, _click_done
    _clicked_points = []
    _click_done = False

    cap = camera.open_camera()

    print("Capturing calibration frame…  Press SPACE to freeze.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue  # skip failed frames during warmup
        cv2.imshow("calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            break
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Calibration cancelled")

    cap.release()

    # Resize for display consistency
    frame = cv2.resize(frame, None, fx=config.RESIZE_SCALE, fy=config.RESIZE_SCALE)

    print("\nClick the 4 corners of your reference rectangle in order:")
    print("  1) top-left   2) top-right   3) bottom-right   4) bottom-left")
    print(f"World coordinates: {config.WORLD_RECT_METERS.tolist()}\n")

    cv2.imshow("calibration", frame)
    cv2.setMouseCallback("calibration", _mouse_callback)

    while not _click_done:
        # Redraw with already-clicked points
        display = frame.copy()
        for i, (px, py) in enumerate(_clicked_points):
            cv2.circle(display, (px, py), 6, (0, 0, 255), -1)
            cv2.putText(
                display,
                str(i + 1),
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        cv2.imshow("calibration", display)
        if cv2.waitKey(50) & 0xFF == 27:
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Calibration cancelled")

    cv2.destroyAllWindows()

    image_points = np.array(_clicked_points, dtype=np.float32)
    world_points = config.WORLD_RECT_METERS

    H, status = cv2.findHomography(image_points, world_points)
    if H is None:
        raise RuntimeError("Homography computation failed")

    # Save
    os.makedirs(config.DATA_DIR, exist_ok=True)
    np.savez(config.HOMOGRAPHY_FILE, H=H)
    print(f"\nHomography saved to {config.HOMOGRAPHY_FILE}")
    print(f"Matrix:\n{H}\n")

    return H


# verification mode
def verify_interactive(H=None):
    """Click additional points and print their world coordinates to verify
    calibration accuracy."""
    if H is None:
        H = load_homography()
    if H is None:
        print("ERROR: No homography found.  Run calibration first.")
        return

    cap = camera.open_camera()

    print("Verification mode — press SPACE to freeze frame, then click points.")
    print("Press ESC to quit.\n")

    global _clicked_points, _click_done
    _clicked_points = []
    _click_done = False

    def _verify_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            world = pixel_to_world((x, y), H)
            print(f"  Pixel ({x}, {y})  →  World ({world[0]:.4f}, {world[1]:.4f}) m")
            _clicked_points.append((x, y, world[0], world[1]))

    # Live feed until SPACE is pressed (allows camera warmup)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue  # skip failed frames during warmup
        cv2.imshow("verify", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            break
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()

    frame = cv2.resize(frame, None, fx=config.RESIZE_SCALE, fy=config.RESIZE_SCALE)

    cv2.imshow("verify", frame)
    cv2.setMouseCallback("verify", _verify_click)

    while True:
        # Redraw with clicked points
        display = frame.copy()
        for px, py, wx, wy in _clicked_points:
            cv2.circle(display, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(
                display,
                f"({wx:.3f},{wy:.3f})",
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )
        cv2.imshow("verify", display)
        if cv2.waitKey(50) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


# CLI
def main():
    import sys

    if "--verify" in sys.argv:
        verify_interactive()
    else:
        H = calibrate_interactive()
        print("Run again with --verify to test additional points.")


if __name__ == "__main__":
    main()
