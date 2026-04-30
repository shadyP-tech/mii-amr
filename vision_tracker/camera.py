"""
camera.py - Shared OpenCV camera setup.

macOS/AVFoundation can occasionally open a USB camera in a monochrome-looking
mode even when the same camera is capable of RGB.  Centralizing capture setup
keeps all vision scripts using the same backend, requested format, and retry
logic.
"""

import time

import cv2
import numpy as np

import config


class RealSenseCapture:
    """Small cv2.VideoCapture-like wrapper around a RealSense color stream."""

    def __init__(self, rs, pipeline, profile, width, height, fps):
        self.rs = rs
        self.pipeline = pipeline
        self.profile = profile
        self.width = width
        self.height = height
        self.fps = fps
        self.color_sensor = self._find_color_sensor()
        self._apply_sensor_options()

    def _find_color_sensor(self):
        device = self.profile.get_device()

        try:
            return device.first_color_sensor()
        except Exception:
            pass

        for sensor in device.query_sensors():
            try:
                name = sensor.get_info(self.rs.camera_info.name)
            except Exception:
                continue
            if "rgb" in name.lower() or "color" in name.lower():
                return sensor

        return None

    def _set_sensor_option(self, option, value):
        if self.color_sensor is None or value is None:
            return False
        if not self.color_sensor.supports(option):
            return False
        self.color_sensor.set_option(option, float(value))
        return True

    def _apply_sensor_options(self):
        self._set_sensor_option(
            self.rs.option.enable_auto_exposure,
            1.0 if config.REALSENSE_ENABLE_AUTO_EXPOSURE else 0.0,
        )
        self._set_sensor_option(self.rs.option.exposure, config.REALSENSE_EXPOSURE)
        self._set_sensor_option(self.rs.option.gain, config.REALSENSE_GAIN)

    def read(self):
        try:
            frames = self.pipeline.wait_for_frames(5000)
        except Exception:
            return False, None

        color_frame = frames.get_color_frame()
        if not color_frame:
            return False, None

        return True, np.asanyarray(color_frame.get_data())

    def release(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop == cv2.CAP_PROP_FPS:
            return float(self.fps)
        if self.color_sensor is None:
            return 0.0
        if prop == cv2.CAP_PROP_EXPOSURE:
            option = self.rs.option.exposure
        elif prop == cv2.CAP_PROP_GAIN:
            option = self.rs.option.gain
        elif prop == cv2.CAP_PROP_AUTO_EXPOSURE:
            option = self.rs.option.enable_auto_exposure
        else:
            return 0.0
        if not self.color_sensor.supports(option):
            return 0.0
        return float(self.color_sensor.get_option(option))

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_EXPOSURE:
            return self._set_sensor_option(self.rs.option.exposure, value)
        if prop == cv2.CAP_PROP_GAIN:
            return self._set_sensor_option(self.rs.option.gain, value)
        if prop == cv2.CAP_PROP_AUTO_EXPOSURE:
            auto_value = 1.0 if value >= 1.0 else 0.0
            return self._set_sensor_option(
                self.rs.option.enable_auto_exposure,
                auto_value,
            )
        return False


def color_stats(frame):
    """Return simple colorfulness stats for a BGR frame."""
    if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
        return 0.0, 0.0

    b, g, r = cv2.split(frame[:, :, :3])
    channel_diff = float(
        np.mean(
            (
                cv2.absdiff(b, g).astype(np.float32)
                + cv2.absdiff(b, r).astype(np.float32)
                + cv2.absdiff(g, r).astype(np.float32)
            )
            / 3.0
        )
    )

    hsv = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2HSV)
    mean_saturation = float(np.mean(hsv[:, :, 1]))

    return mean_saturation, channel_diff


def looks_like_color(frame):
    """Return True if the frame does not look like duplicated grayscale."""
    mean_saturation, channel_diff = color_stats(frame)
    return (
        mean_saturation >= config.CAMERA_MIN_MEAN_SATURATION
        or channel_diff >= config.CAMERA_MIN_CHANNEL_DIFF
    )


def _open_opencv_capture(index):
    if config.CAMERA_FORCE_AVFOUNDATION:
        return cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(index)


def _open_realsense_capture():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 is not installed") from exc

    pipeline = rs.pipeline()
    rs_config = rs.config()

    if config.REALSENSE_SERIAL:
        rs_config.enable_device(config.REALSENSE_SERIAL)

    rs_config.enable_stream(
        rs.stream.color,
        config.CAMERA_FRAME_WIDTH,
        config.CAMERA_FRAME_HEIGHT,
        rs.format.bgr8,
        config.CAMERA_FPS,
    )

    try:
        profile = pipeline.start(rs_config)
    except Exception as exc:
        raise RuntimeError(f"RealSense color stream did not start: {exc}") from exc

    return RealSenseCapture(
        rs,
        pipeline,
        profile,
        config.CAMERA_FRAME_WIDTH,
        config.CAMERA_FRAME_HEIGHT,
        config.CAMERA_FPS,
    )


def _open_capture(index, backend):
    if backend == "realsense":
        return _open_realsense_capture()
    return _open_opencv_capture(index)


def _candidate_backends():
    backend = config.CAMERA_BACKEND.lower()
    if backend == "auto":
        return ["realsense", "opencv"]
    if backend in {"realsense", "opencv"}:
        return [backend]
    raise RuntimeError(
        f"Unsupported CAMERA_BACKEND={config.CAMERA_BACKEND!r}; "
        'use "auto", "realsense", or "opencv".'
    )


def _apply_camera_settings(cap, width=None, height=None, fps=None):
    if isinstance(cap, RealSenseCapture):
        return

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


def _read_configured_frame(cap):
    if isinstance(cap, RealSenseCapture) or not config.CAMERA_USE_RGB_WAKEUP_MODE:
        _apply_camera_settings(cap)
        return _read_warm_frame(cap)

    _apply_camera_settings(
        cap,
        width=config.CAMERA_RGB_WAKEUP_FRAME_WIDTH,
        height=config.CAMERA_RGB_WAKEUP_FRAME_HEIGHT,
        fps=config.CAMERA_RGB_WAKEUP_FPS,
    )
    warm_ok, warm_frame = _read_warm_frame(cap)

    _apply_camera_settings(cap)
    ok, frame = _read_warm_frame(cap)

    if ok:
        return ok, frame
    return warm_ok, warm_frame


def _read_warm_frame(cap):
    frame = None
    ok = False

    for _ in range(max(1, config.CAMERA_WARMUP_FRAMES)):
        ok, next_frame = cap.read()
        if ok:
            frame = next_frame

    return ok and frame is not None, frame


def open_camera(index=None, require_color=None):
    """Open the configured camera, retrying if it appears monochrome."""
    index = config.CAMERA_INDEX if index is None else index
    require_color = config.CAMERA_REQUIRE_COLOR if require_color is None else require_color

    last_error = "camera did not open"

    for backend in _candidate_backends():
        label = "RealSense color stream" if backend == "realsense" else f"camera {index}"

        for attempt in range(1, config.CAMERA_OPEN_RETRIES + 1):
            try:
                cap = _open_capture(index, backend)
            except RuntimeError as exc:
                last_error = str(exc)
                print(f"{label}: {last_error}.")
                break

            if not isinstance(cap, RealSenseCapture) and not cap.isOpened():
                last_error = "camera did not open"
                cap.release()
                print(
                    f"{label}: open attempt {attempt}/"
                    f"{config.CAMERA_OPEN_RETRIES} failed."
                )
                time.sleep(config.CAMERA_RETRY_DELAY_SEC)
                continue

            ok, frame = _read_configured_frame(cap)
            if not ok:
                last_error = "camera opened but did not return frames"
                cap.release()
                print(
                    f"{label}: attempt {attempt}/"
                    f"{config.CAMERA_OPEN_RETRIES} returned no frame."
                )
                time.sleep(config.CAMERA_RETRY_DELAY_SEC)
                continue

            mean_saturation, channel_diff = color_stats(frame)
            if backend == "realsense" or not require_color or looks_like_color(frame):
                print(
                    f"{label}: opened "
                    f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                    f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
                    f"sat={mean_saturation:.1f} diff={channel_diff:.1f}"
                )
                return cap

            last_error = (
                "camera stream looks grayscale "
                f"(mean saturation {mean_saturation:.1f}, "
                f"channel diff {channel_diff:.1f})"
            )
            print(
                f"{label}: attempt {attempt}/"
                f"{config.CAMERA_OPEN_RETRIES} looks grayscale "
                f"(sat={mean_saturation:.1f}, diff={channel_diff:.1f}); retrying."
            )
            cap.release()
            time.sleep(config.CAMERA_RETRY_DELAY_SEC)

    raise RuntimeError(
        f"Cannot open color camera {index}: {last_error}. "
        "Try closing other camera apps, unplugging/replugging the USB camera, "
        "or set CAMERA_REQUIRE_COLOR = False if you intentionally want mono."
    )
