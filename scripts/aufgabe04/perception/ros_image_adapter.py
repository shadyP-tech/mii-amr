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
