"""Bind a native QR quadrilateral to a decoder by isolating that symbol."""

from __future__ import annotations

from scripts.aufgabe04.qr_scanning.qr_observation import (
    DecodedQrObservation, validated_qr_corners,
)


def decode_isolated_native_quad(
    frame, points, cv2, *, image_shape, scale, border_px,
) -> DecodedQrObservation | None:
    """Decode exactly one native quad without borrowing a full-crop payload.

    Production OpenCV lacks QUIRC and model-free WeChat returns crop bounds
    instead of localized corners. Native detection supplies geometry; a new
    WeChat decode of only that rectified symbol supplies its own identity.
    Multiple native quads, malformed geometry, and multiple isolated payloads
    are rejected. The fixed 256-pixel output bounds the additional work.
    """
    raw = validated_qr_corners(points, image_shape=getattr(frame, "shape", None))
    restored = validated_qr_corners(
        points, image_shape=image_shape, scale=scale, border_px=border_px,
    )
    factory = getattr(cv2, "wechat_qrcode_WeChatQRCode", None)
    if raw is None or restored is None or factory is None:
        return None
    try:
        import numpy

        transform = cv2.getPerspectiveTransform(
            numpy.asarray(raw, dtype=numpy.float32),
            numpy.asarray(((0, 0), (223, 0), (223, 223), (0, 223)), dtype=numpy.float32),
        )
        symbol = cv2.warpPerspective(frame, transform, (224, 224))
        white = 255 if len(symbol.shape) == 2 else (255, 255, 255)
        isolated = cv2.copyMakeBorder(
            symbol, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=white,
        )
        result = factory().detectAndDecode(isolated)
        decoded = result[0]
        texts = (decoded,) if isinstance(decoded, str) else tuple(decoded)
        if len(texts) != 1 or not isinstance(texts[0], str) or not texts[0].strip():
            return None
        return DecodedQrObservation(
            texts[0].strip(), restored, "opencv_quad_wechat_rectified", float(scale),
        )
    except Exception:
        return None
