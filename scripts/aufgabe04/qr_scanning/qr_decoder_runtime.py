"""Frame-local decoder instances and bounded, optional decode provenance."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from scripts.aufgabe04.qr_scanning.qr_observation import qr_corner_groups, validated_qr_corners


MAX_TRACE_EVENTS = 32
MAX_TRACE_TEXTS = 8
MAX_TRACE_TEXT_LENGTH = 160


def trace_texts(texts):
    """Bound diagnostics only; decoded observations retain complete payloads."""
    return [str(text)[:MAX_TRACE_TEXT_LENGTH] for text in tuple(texts)[:MAX_TRACE_TEXTS]]


class QrDecoderRuntime:
    """Share backend objects only for this call, never decoded image evidence."""

    def __init__(self, cv2, diagnostics=None):
        self.cv2 = cv2
        self.diagnostics = diagnostics
        self._decoders = {}
        self.native_symbol_count = 0
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update(
                opencv_version=str(getattr(cv2, "__version__", "unknown")),
                decoder_instances={}, events=[], events_truncated=False,
            )

    def decoder(self, name):
        if name not in self._decoders:
            factory_name = ("QRCodeDetector" if name == "native"
                            else "wechat_qrcode_WeChatQRCode")
            try:
                factory = getattr(self.cv2, factory_name, None)
                self._decoders[name] = factory() if factory is not None else None
            except Exception:
                self._decoders[name] = None
            if self.diagnostics is not None:
                self.diagnostics["decoder_instances"][name] = int(self._decoders[name] is not None)
        return self._decoders[name]

    def observe_native_multi(self, points, *, image_shape, scale, border_px):
        count = sum(validated_qr_corners(group, image_shape=image_shape,
                    scale=scale, border_px=border_px) is not None
                    for group in qr_corner_groups(points))
        self.native_symbol_count = max(self.native_symbol_count, count)

    def conservative_observations(self, observations):
        # Several undecoded native symbols are enough to prevent unique
        # geometry admission. Preserve actual payloads without fabricating
        # identities for symbols the backend could not decode.
        if self.native_symbol_count > 1 and len(observations) == 1:
            return tuple(replace(item, corners=None) for item in observations)
        return observations

    def record(self, stage, *, scale, border_px, observations=(), reason=None,
               isolated=None, corner_validation=None):
        if self.diagnostics is None:
            return
        events = self.diagnostics["events"]
        if len(events) >= MAX_TRACE_EVENTS:
            self.diagnostics["events_truncated"] = True
            return
        event = {
            "stage": stage, "scale": float(scale), "border_px": int(border_px),
            "texts": trace_texts(item.text for item in observations),
            "symbol_count": len(observations),
            "own_corner_count": sum(item.corners is not None for item in observations),
            "native_symbol_count": self.native_symbol_count,
        }
        if reason is not None:
            event["reason"] = reason
        if corner_validation is not None:
            event["corner_validation"] = deepcopy(corner_validation[:MAX_TRACE_TEXTS])
        if isolated is not None:
            event["isolated"] = deepcopy(isolated)
            event["isolated"]["ambiguous_texts"] = trace_texts(isolated.get("ambiguous_texts", ()))
            for view in event["isolated"].get("views", ()):
                view["texts"] = trace_texts(view.get("texts", ()))
        events.append(event)

    def finish(self, observations):
        observations = self.conservative_observations(observations)
        if self.diagnostics is not None:
            self.diagnostics["result"] = {
                "texts": trace_texts(item.text for item in observations),
                "symbol_count": len(observations),
                "own_corner_count": sum(item.corners is not None for item in observations),
                "detectors": [item.detector for item in observations[:MAX_TRACE_TEXTS]],
                "native_symbol_count": self.native_symbol_count,
            }
            if self.native_symbol_count > 1 and len(observations) <= 1:
                self.diagnostics["result"]["geometry_suppressed_reason"] = "multiple_native_quads"
        return observations
