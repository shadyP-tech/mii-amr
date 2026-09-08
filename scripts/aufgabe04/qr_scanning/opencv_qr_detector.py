"""Pure OpenCV QR-code detection helpers for Aufgabe 04."""

from __future__ import annotations

from scripts.aufgabe04.qr_scanning.qr_observation import (
    DecodedQrObservation, validated_qr_corners,
)
from scripts.aufgabe04.qr_scanning.isolated_qr_identity import decode_isolated_native_quad


def detect_qr_texts_bgr(frame, cv2) -> tuple[str, ...]:
    """Return non-empty QR texts detected in a BGR frame.

    ``cv2`` is injected so this module stays importable in ROS-free tests and
    environments that do not have OpenCV installed.
    """

    return tuple(item.text for item in detect_qr_observations_bgr(frame, cv2))


def detect_qr_observations_bgr(frame, cv2) -> tuple[DecodedQrObservation, ...]:
    """Share text and corners from the same decoder and bounded preprocessing.

    Every corner is restored to the input crop. Multiple returned identities
    remain multiple observations; callers must not pick one as target proof.
    """
    provisional = ()
    for candidate, scale, border in _qr_decode_candidates_with_geometry(frame, cv2):
        for observations in _candidate_observations(
            candidate, cv2, image_shape=getattr(frame, "shape", None),
            scale=scale, border_px=border,
        ):
            if not observations:
                continue
            if len(observations) > 1:
                return observations
            if provisional and provisional[0].text != observations[0].text:
                return provisional + observations
            if observations[0].corners is not None:
                return observations
            # A text-only decode cannot donate its identity to unrelated
            # native geometry. Continue only to find a decoder that returns
            # both the same unique payload and that symbol's actual corners.
            if not provisional:
                provisional = observations
    return provisional


def _candidate_observations(candidate, cv2, *, image_shape, scale, border_px):
    factory = getattr(cv2, "wechat_qrcode_WeChatQRCode", None)
    if factory is not None:
        try:
            result = factory().detectAndDecode(candidate)
            yield _decoded_observations(
                result[0], result[1] if len(result) > 1 else None,
                detector="wechat", image_shape=image_shape, scale=scale, border_px=border_px,
            )
        except Exception:
            pass
    try:
        result = cv2.QRCodeDetector().detectAndDecodeMulti(candidate)
        yield _decoded_observations(
            result[1], result[2] if len(result) > 2 else None,
            detector="opencv_multi", image_shape=image_shape, scale=scale, border_px=border_px,
        ) if result and result[0] and len(result) > 1 else ()
        if len(result) > 2 and result[2] is not None:
            isolated = decode_isolated_native_quad(
                candidate, result[2], cv2, image_shape=image_shape,
                scale=scale, border_px=border_px,
            )
            if isolated is not None:
                yield (isolated,)
    except Exception:
        pass
    try:
        result = cv2.QRCodeDetector().detectAndDecode(candidate)
        yield _decoded_observations(
            result[0], result[1] if len(result) > 1 else None,
            detector="opencv_single", image_shape=image_shape, scale=scale, border_px=border_px,
        ) if result else ()
    except Exception:
        pass


def _decoded_observations(decoded, points, *, detector, image_shape, scale, border_px):
    texts = ((decoded,) if isinstance(decoded, str)
             else tuple(decoded) if decoded is not None else ())
    if hasattr(points, "tolist"):
        points = points.tolist()
    try:
        # A single symbol may be returned as 4x2 or 1x4x2, depending on
        # decoder/OpenCV version. Never flatten several symbols together.
        single_quad = len(points) == 4 and all(len(row) == 2 for row in points)
        groups = (points,) if single_quad else points
        len(groups)
    except (TypeError, ValueError):
        groups = ()
    result = []
    for index, raw in enumerate(texts):
        text = "" if raw is None else str(raw).strip()
        if not text:
            continue
        corners = validated_qr_corners(
            groups[index] if index < len(groups) else None,
            image_shape=image_shape, scale=scale, border_px=border_px,
        )
        result.append(DecodedQrObservation(text, corners, detector, float(scale)))
    return tuple(result)


def _qr_decode_candidates_with_geometry(frame, cv2):
    yield frame, 1.0, 0
    try:
        height, width = frame.shape[:2]
    except (AttributeError, ValueError):
        return
    scale = 4 if max(height, width) < 220 else 2
    enlarged = _resize_for_qr(cv2, frame, scale=scale)
    image = enlarged if enlarged is not None else frame
    effective_scale = float(scale) if enlarged is not None else 1.0
    border = max(12, int(0.08 * max(image.shape[:2])))
    if enlarged is not None:
        bordered = _add_quiet_border(cv2, enlarged, border_px=border)
        actual_border = (bordered.shape[0] - enlarged.shape[0]) // 2
        yield bordered, effective_scale, actual_border
    gray = _to_gray(cv2, image)
    if gray is not None:
        bordered = _add_quiet_border(cv2, gray, border_px=border)
        actual_border = (bordered.shape[0] - gray.shape[0]) // 2
        yield bordered, effective_scale, actual_border
        thresholded = _threshold_for_qr(cv2, bordered)
        if thresholded is not None:
            yield thresholded, effective_scale, actual_border


def _resize_for_qr(cv2, frame, *, scale: int):
    if scale <= 1:
        return frame
    try:
        return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    except Exception:
        return None


def _add_quiet_border(cv2, frame, *, border_px: int):
    try:
        if len(frame.shape) == 2:
            value = 255
        else:
            value = (255, 255, 255)
        return cv2.copyMakeBorder(
            frame,
            border_px,
            border_px,
            border_px,
            border_px,
            cv2.BORDER_CONSTANT,
            value=value,
        )
    except Exception:
        return frame


def _to_gray(cv2, frame):
    try:
        if len(frame.shape) == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None


def _threshold_for_qr(cv2, gray):
    try:
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            3,
        )
    except Exception:
        return None
