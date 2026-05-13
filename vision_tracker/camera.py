"""Shared OpenCV camera setup for the configured camera index."""

import cv2

import config


def _open_opencv_capture(index):
    if config.CAMERA_FORCE_AVFOUNDATION:
        return cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(index)


def _apply_camera_settings(cap, width=None, height=None, fps=None):
    width = config.CAMERA_FRAME_WIDTH if width is None else width
    height = config.CAMERA_FRAME_HEIGHT if height is None else height
    fps = config.CAMERA_FPS if fps is None else fps

    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

    if config.CAMERA_FOURCC:
        fourcc = cv2.VideoWriter_fourcc(*config.CAMERA_FOURCC)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)


def _read_warm_frame(cap):
    frame = None
    ok = False

    for _ in range(max(1, config.CAMERA_WARMUP_FRAMES)):
        ok, next_frame = cap.read()
        if ok:
            frame = next_frame

    return ok and frame is not None, frame


def open_camera(index=None):
    """Open the configured OpenCV camera and return an initialized capture."""
    index = config.CAMERA_INDEX if index is None else index
    cap = _open_opencv_capture(index)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open camera {index}.")

    _apply_camera_settings(cap)
    ok, _frame = _read_warm_frame(cap)
    if not ok:
        cap.release()
        raise RuntimeError(f"Camera {index} opened but did not return frames.")

    print(
        f"camera {index}: opened "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
        f"@ {cap.get(cv2.CAP_PROP_FPS):.1f} fps"
    )
    return cap
