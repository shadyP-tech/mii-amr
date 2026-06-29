"""Pure OpenCV QR-code detection helpers for Aufgabe 04."""

from __future__ import annotations

from collections.abc import Iterable


def _nonblank_texts(decoded: object) -> tuple[str, ...]:
    if decoded is None:
        return ()
    if isinstance(decoded, str):
        candidates: Iterable[object] = (decoded,)
    else:
        try:
            iter(decoded)  # type: ignore[arg-type]
        except TypeError:
            candidates = ()
        else:
            candidates = decoded  # type: ignore[assignment]

    texts = []
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            texts.append(text)
    return tuple(texts)


def detect_qr_texts_bgr(frame, cv2) -> tuple[str, ...]:
    """Return non-empty QR texts detected in a BGR frame.

    ``cv2`` is injected so this module stays importable in ROS-free tests and
    environments that do not have OpenCV installed.
    """

    detector = cv2.QRCodeDetector()
    multi_result = detector.detectAndDecodeMulti(frame)
    ok = bool(multi_result[0]) if multi_result else False
    decoded = multi_result[1] if len(multi_result) > 1 else ()
    if ok:
        texts = _nonblank_texts(decoded)
        if texts:
            return texts

    single_result = detector.detectAndDecode(frame)
    single_text = single_result[0] if single_result else ""
    return _nonblank_texts(single_text)
