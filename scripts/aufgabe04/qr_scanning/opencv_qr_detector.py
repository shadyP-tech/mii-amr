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

    for candidate in _qr_decode_candidates(frame, cv2):
        texts = _detect_qr_texts_single_candidate(candidate, cv2)
        if texts:
            return texts
    return ()


def _detect_qr_texts_single_candidate(frame, cv2) -> tuple[str, ...]:
    detector = cv2.QRCodeDetector()
    try:
        multi_result = detector.detectAndDecodeMulti(frame)
    except Exception:
        multi_result = ()
    ok = bool(multi_result[0]) if multi_result else False
    decoded = multi_result[1] if len(multi_result) > 1 else ()
    if ok:
        texts = _nonblank_texts(decoded)
        if texts:
            return texts

    try:
        single_result = detector.detectAndDecode(frame)
    except Exception:
        single_result = ()
    single_text = single_result[0] if single_result else ""
    texts = _nonblank_texts(single_text)
    if texts:
        return texts

    wechat_detector_factory = getattr(cv2, "wechat_qrcode_WeChatQRCode", None)
    if wechat_detector_factory is None:
        return ()

    wechat_detector = wechat_detector_factory()
    try:
        wechat_result = wechat_detector.detectAndDecode(frame)
    except Exception:
        return ()
    wechat_texts = wechat_result[0] if wechat_result else ()
    return _nonblank_texts(wechat_texts)


def _qr_decode_candidates(frame, cv2) -> tuple[object, ...]:
    candidates = [frame]
    candidates.extend(_preprocessed_qr_candidates(frame, cv2))
    return tuple(candidates)


def _preprocessed_qr_candidates(frame, cv2) -> tuple[object, ...]:
    try:
        height, width = frame.shape[:2]
    except AttributeError:
        return ()

    scale = 4 if max(height, width) < 220 else 2
    candidates = []
    enlarged = _resize_for_qr(cv2, frame, scale=scale)
    if enlarged is not None:
        candidates.append(_add_quiet_border(cv2, enlarged, border_px=max(12, int(0.08 * max(enlarged.shape[:2])))))

    gray = _to_gray(cv2, enlarged if enlarged is not None else frame)
    if gray is not None:
        bordered_gray = _add_quiet_border(cv2, gray, border_px=max(12, int(0.08 * max(gray.shape[:2]))))
        candidates.append(bordered_gray)
        thresholded = _threshold_for_qr(cv2, bordered_gray)
        if thresholded is not None:
            candidates.append(thresholded)
    return tuple(candidate for candidate in candidates if candidate is not None)


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
