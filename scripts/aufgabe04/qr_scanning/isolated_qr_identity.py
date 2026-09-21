"""Bind a native QR quadrilateral to a decoder by isolating that symbol."""

from __future__ import annotations

from scripts.aufgabe04.qr_scanning.qr_observation import (
    DecodedQrObservation, validated_qr_corners,
)
from scripts.aufgabe04.qr_scanning.isolated_qr_views import (
    ISOLATED_QR_VIEWS, rectify_isolated_qr_view,
)


def decode_isolated_native_quad(
    frame, points, cv2, *, image_shape, scale, border_px,
    wechat_decoder=None, diagnostics: dict | None = None,
    budget_exhausted=None, work_budget=None,
) -> DecodedQrObservation | None:
    """Decode exactly one native quad without borrowing a full-crop payload.

    Production OpenCV lacks QUIRC and model-free WeChat returns crop bounds
    instead of localized corners. Native detection supplies geometry; a new
    WeChat decode of only that rectified symbol supplies its own identity.
    Multiple native quads, malformed geometry, and multiple isolated payloads
    are rejected. At most two views are decoded, each at most 256x256 pixels.
    A second view preserves a narrow source-pixel quiet margin that recovered
    the recorded frame 000011 in the deployed OpenCV 4.5.4 runtime. No decoded
    input-extent rectangle is used as symbol geometry.
    An optional cooperative budget is checked before each recovery view.
    """
    raw = validated_qr_corners(points, image_shape=getattr(frame, "shape", None))
    restored = validated_qr_corners(
        points, image_shape=image_shape, scale=scale, border_px=border_px,
    )
    factory = getattr(cv2, "wechat_qrcode_WeChatQRCode", None)
    if diagnostics is not None:
        diagnostics.update(views=[], ambiguous_texts=[],
                           reason="native_quad_unavailable")
    if raw is None or restored is None or (factory is None and wechat_decoder is None):
        return None
    try:
        decoder = wechat_decoder if wechat_decoder is not None else factory()
        for view in ISOLATED_QR_VIEWS:
            if budget_exhausted is not None and budget_exhausted():
                if diagnostics is not None:
                    diagnostics["reason"] = "processing_budget_exhausted"
                return None
            size = view.symbol_size_px + 2 * round(view.symbol_size_px * view.source_margin_ratio) + 2 * view.quiet_border_px
            pixels = size * size
            if work_budget is not None and not work_budget.allow("isolated_wechat", pixels):
                if diagnostics is not None:
                    diagnostics["reason"] = "processing_cost_budget_exhausted"
                return None
            def decode_view():
                isolated = rectify_isolated_qr_view(frame, raw, cv2, view)
                return isolated, decoder.detectAndDecode(isolated)
            isolated, result = (decode_view() if work_budget is None else
                                work_budget.measure("isolated_wechat", pixels, decode_view))
            decoded = result[0]
            texts = (decoded,) if isinstance(decoded, str) else tuple(decoded)
            texts = tuple(text.strip() for text in texts if isinstance(text, str) and text.strip())
            if diagnostics is not None:
                diagnostics["views"].append({
                    "view": view.name, "image_shape": list(isolated.shape),
                    "source_margin_ratio": view.source_margin_ratio,
                    "texts": list(texts),
                })
            if len(texts) > 1:
                if diagnostics is not None:
                    diagnostics.update(reason="multiple_isolated_payloads", ambiguous_texts=list(texts))
                return None
            if not texts:
                continue
            if diagnostics is not None:
                diagnostics.update(reason="isolated_payload_confirmed", selected_view=view.name)
            return DecodedQrObservation(
                texts[0], restored, "opencv_quad_wechat_rectified", float(scale),
            )
        if diagnostics is not None:
            diagnostics["reason"] = "isolated_payload_unavailable"
    except Exception:
        if diagnostics is not None:
            diagnostics["reason"] = "isolated_decode_error"
    return None
