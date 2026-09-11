"""Pure OpenCV QR-code detection helpers for Aufgabe 04."""

from __future__ import annotations

from scripts.aufgabe04.qr_scanning.qr_observation import (
    DecodedQrObservation, qr_corner_groups, validated_qr_corners,
)
from scripts.aufgabe04.qr_scanning.isolated_qr_identity import decode_isolated_native_quad
from scripts.aufgabe04.qr_scanning.qr_decoder_runtime import QrDecoderRuntime


MAX_DEFERRED_SINGLE_QUADS = 4


def detect_qr_texts_bgr(frame, cv2) -> tuple[str, ...]:
    """Return non-empty QR texts detected in a BGR frame.

    ``cv2`` is injected so this module stays importable in ROS-free tests and
    environments that do not have OpenCV installed.
    """

    return tuple(item.text for item in detect_qr_observations_bgr(frame, cv2))


def detect_qr_observations_bgr(frame, cv2, *, diagnostics: dict | None = None) -> tuple[DecodedQrObservation, ...]:
    """Share text and corners from the same decoder and bounded preprocessing.

    Every corner is restored to the input crop. Multiple returned identities
    remain multiple observations; callers must not pick one as target proof.
    """
    runtime = QrDecoderRuntime(cv2, diagnostics)
    provisional = ()
    deferred_single = []
    for candidate, scale, border in _qr_decode_candidates_with_geometry(frame, cv2):
        for observations in _candidate_observations(
            candidate, cv2, image_shape=getattr(frame, "shape", None),
            scale=scale, border_px=border, runtime=runtime, deferred_single=deferred_single,
        ):
            observations = runtime.conservative_observations(observations)
            if not observations:
                continue
            if len(observations) > 1:
                return runtime.finish(observations)
            if provisional and provisional[0].text != observations[0].text:
                return runtime.finish(provisional + observations)
            if observations[0].corners is not None:
                return runtime.finish(observations)
            # A text-only decode cannot donate its identity to unrelated
            # native geometry. Continue only to find a decoder that returns
            # both the same unique payload and that symbol's actual corners.
            if not provisional:
                provisional = observations
    # Prefer every normal/multi variant before spending time on native-single
    # isolation. This preserves the measured fast scale-4 path for frame 24.
    # Only these current-image variants are retained, never previous frames.
    if runtime.native_symbol_count > 1:
        return runtime.finish(provisional)
    for candidate, points, scale, border in deferred_single:
        observations = _isolated_observations(
            candidate, points, cv2, image_shape=getattr(frame, "shape", None),
            scale=scale, border_px=border, runtime=runtime, source="opencv_single_isolated_deferred",
        )
        if len(observations) > 1:
            return runtime.finish(observations)
        if observations:
            if provisional and provisional[0].text != observations[0].text:
                return runtime.finish(provisional + observations)
            return runtime.finish(observations)
    return runtime.finish(provisional)


def _candidate_observations(candidate, cv2, *, image_shape, scale, border_px,
                             runtime=None, deferred_single=None):
    runtime = runtime if runtime is not None else QrDecoderRuntime(cv2)
    wechat = runtime.decoder("wechat")
    if wechat is not None:
        try:
            result = wechat.detectAndDecode(candidate)
            corner_validation = [] if runtime.diagnostics is not None else None
            observations = _decoded_observations(
                result[0], result[1] if len(result) > 1 else None,
                detector="wechat", image_shape=image_shape, scale=scale, border_px=border_px,
                corner_diagnostics=corner_validation,
            )
            runtime.record("wechat", scale=scale, border_px=border_px, observations=observations,
                           corner_validation=corner_validation)
            yield observations
        except Exception:
            runtime.record("wechat", scale=scale, border_px=border_px, reason="decoder_error")
    native = runtime.decoder("native")
    if native is None:
        return
    multi_quad = None
    try:
        result = native.detectAndDecodeMulti(candidate)
        if result and len(result) > 2:
            runtime.observe_native_multi(result[2], image_shape=image_shape, scale=scale, border_px=border_px)
        corner_validation = [] if runtime.diagnostics is not None else None
        observations = _decoded_observations(
            result[1], result[2] if len(result) > 2 else None,
            detector="opencv_multi", image_shape=image_shape, scale=scale, border_px=border_px,
            corner_diagnostics=corner_validation,
        ) if result and result[0] and len(result) > 1 else ()
        runtime.record("opencv_multi", scale=scale, border_px=border_px, observations=observations,
                       corner_validation=corner_validation)
        yield observations
        if len(result) > 2 and result[2] is not None:
            multi_quad = validated_qr_corners(result[2], image_shape=getattr(candidate, "shape", None))
            yield _isolated_observations(
                candidate, result[2], cv2, image_shape=image_shape, scale=scale,
                border_px=border_px, runtime=runtime,
            )
    except Exception:
        runtime.record("opencv_multi", scale=scale, border_px=border_px, reason="decoder_error")
    try:
        result = native.detectAndDecode(candidate)
        corner_validation = [] if runtime.diagnostics is not None else None
        observations = _decoded_observations(
            result[0], result[1] if len(result) > 1 else None,
            detector="opencv_single", image_shape=image_shape, scale=scale, border_px=border_px,
            corner_diagnostics=corner_validation,
        ) if result else ()
        runtime.record("opencv_single", scale=scale, border_px=border_px, observations=observations,
                       corner_validation=corner_validation)
        yield observations
        if result and len(result) > 1 and deferred_single is not None:
            # Validation copies native output into immutable tuples: a later
            # backend call must not overwrite points retained for recovery.
            single_quad = validated_qr_corners(result[1], image_shape=getattr(candidate, "shape", None))
            if (runtime.native_symbol_count <= 1 and single_quad is not None and single_quad != multi_quad
                    and len(deferred_single) < MAX_DEFERRED_SINGLE_QUADS):
                deferred_single.append((candidate, single_quad, scale, border_px))
    except Exception:
        runtime.record("opencv_single", scale=scale, border_px=border_px, reason="decoder_error")


def _isolated_observations(candidate, points, cv2, *, image_shape, scale, border_px,
                           runtime, source="opencv_multi_isolated"):
    if runtime.native_symbol_count > 1:
        runtime.record(source, scale=scale, border_px=border_px, reason="multiple_native_quads")
        return ()
    wechat = runtime.decoder("wechat")
    if wechat is None:
        runtime.record(source, scale=scale, border_px=border_px, reason="wechat_unavailable")
        return ()
    diagnostics = {}
    isolated = decode_isolated_native_quad(
        candidate, points, cv2, image_shape=image_shape, scale=scale,
        border_px=border_px, wechat_decoder=wechat, diagnostics=diagnostics,
    )
    # Ambiguous isolated decoding remains conservative conflict evidence,
    # including two physical symbols with identical text. It grants no quad.
    observations = ((isolated,) if isolated is not None else tuple(
        DecodedQrObservation(text, None, "opencv_quad_wechat_rectified", float(scale))
        for text in diagnostics.get("ambiguous_texts", ())
    ))
    runtime.record(source, scale=scale, border_px=border_px,
                   observations=observations, isolated=diagnostics)
    return observations


def _decoded_observations(decoded, points, *, detector, image_shape, scale, border_px,
                          corner_diagnostics=None):
    texts = ((decoded,) if isinstance(decoded, str)
             else tuple(decoded) if decoded is not None else ())
    groups = qr_corner_groups(points)
    result = []
    for index, raw in enumerate(texts):
        text = "" if raw is None else str(raw).strip()
        if not text:
            continue
        validation = {} if corner_diagnostics is not None else None
        corners = validated_qr_corners(
            groups[index] if index < len(groups) else None,
            image_shape=image_shape, scale=scale, border_px=border_px,
            diagnostics=validation,
        )
        if corner_diagnostics is not None and len(corner_diagnostics) < 8:
            corner_diagnostics.append({"symbol_index": index, **validation})
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
