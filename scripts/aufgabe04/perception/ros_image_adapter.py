from __future__ import annotations


def compressed_msg_to_bgr_frame(msg, cv2, numpy):
    data = numpy.frombuffer(msg.data, dtype=numpy.uint8)
    if data.size == 0:
        raise ValueError("compressed ROS image data is empty")
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        image_format = getattr(msg, "format", "")
        raise ValueError(f"failed to decode compressed ROS image: {image_format!r}")
    return frame


def compressed_msg_stamp_sec(msg) -> float | None:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return None
    try:
        return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0
    except (AttributeError, TypeError, ValueError):
        return None


def raw_msg_to_bgr_frame(msg, cv2, numpy):
    """Decode the uncompressed sensor_msgs/Image emitted by Gazebo cameras."""

    encoding = str(getattr(msg, "encoding", "")).lower()
    channels_by_encoding = {
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
        "mono8": 1,
    }
    channels = channels_by_encoding.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported raw ROS image encoding: {encoding!r}")
    width = int(msg.width)
    height = int(msg.height)
    step = int(msg.step)
    if width <= 0 or height <= 0 or step < width * channels:
        raise ValueError("invalid raw ROS image dimensions or row step")
    data = numpy.frombuffer(msg.data, dtype=numpy.uint8)
    if data.size < height * step:
        raise ValueError("raw ROS image data is truncated")
    rows = data[: height * step].reshape(height, step)
    pixels = rows[:, : width * channels].reshape(height, width, channels)
    if encoding == "bgr8":
        return pixels.copy()
    if encoding == "rgb8":
        return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(pixels.reshape(height, width), cv2.COLOR_GRAY2BGR)
