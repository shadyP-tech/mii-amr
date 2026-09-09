"""Bounded original-frame QR identity refresh for an already tracked stand."""

from __future__ import annotations

from scripts.aufgabe04.qr_scanning.opencv_qr_detector import _decoded_observations
from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation


def detect_native_qr_observations_bgr(frame, cv2) -> tuple[DecodedQrObservation, ...]:
    """Decode this frame with native OpenCV only, without acquisition retries.

    Both text and corners come from the same native decoder result. A miss
    stays empty: no cached identity, enlarged images, isolated-symbol decoder,
    or WeChat fallback is used during a tracked refresh. Native multi-symbol
    and text-only results retain the generic decoder's admission semantics.
    """

    try:
        detector = cv2.QRCodeDetector()
    except Exception:
        return ()

    provisional = ()
    for method, source in (
        ("detectAndDecodeMulti", "opencv_multi"),
        ("detectAndDecode", "opencv_single"),
    ):
        try:
            result = getattr(detector, method)(frame)
            if method == "detectAndDecodeMulti":
                if not result or not result[0] or len(result) < 2:
                    continue
                decoded = result[1]
                points = result[2] if len(result) > 2 else None
            else:
                if not result:
                    continue
                decoded = result[0]
                points = result[1] if len(result) > 1 else None
            observations = _decoded_observations(
                decoded, points, detector=source,
                image_shape=getattr(frame, "shape", None), scale=1.0, border_px=0,
            )
        except Exception:
            continue
        if not observations:
            continue
        if len(observations) > 1:
            return observations
        if provisional and provisional[0].text != observations[0].text:
            return provisional + observations
        if observations[0].corners is not None:
            return observations
        if not provisional:
            provisional = observations
    return provisional
