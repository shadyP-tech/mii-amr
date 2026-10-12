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
    path = path or config.HOMOGRAPHY_FILE
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return data["H"]


def pixel_to_world(pixel, H):
    px = np.array([[[pixel[0], pixel[1]]]], dtype=np.float32)
    world = cv2.perspectiveTransform(px, H)
    return world[0, 0]  # shape (2,)


# calibration procedure
def calibrate_interactive():
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

    # resize for display consistency
    frame = cv2.resize(frame, None, fx=config.RESIZE_SCALE, fy=config.RESIZE_SCALE)

    print("\nClick the 4 corners of your reference rectangle in order:")
    print("  1) top-left   2) top-right   3) bottom-right   4) bottom-left")
    print(f"World coordinates: {config.WORLD_RECT_METERS.tolist()}\n")

    cv2.imshow("calibration", frame)
    cv2.setMouseCallback("calibration", _mouse_callback)

    while not _click_done:
        # redraw with already-clicked points
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


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description="Calibrate the camera with four reference corners.").parse_args()
    calibrate_interactive()
