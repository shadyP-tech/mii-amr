"""Reuse QR decoding only within one image's bounded ROI evaluations.

Create a new cache for every input image. Keys are exact full-image crop
bounds and decoder mode; observations remain in the decoder's crop coordinates.
This helper neither stores image frames nor caches fitted geometry or identity
admission. The caller supplies the same decoder policy for each mode.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Callable, Literal

from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation


QrDecodeMode = Literal["full", "native"]
RoiBounds = tuple[int, int, int, int]
QrObservations = tuple[DecodedQrObservation, ...] | None
MAX_CACHE_ENTRIES = 3


@dataclass(frozen=True)
class RoiQrDecodeResult:
    roi: RoiBounds
    mode: QrDecodeMode
    observations: QrObservations
    cache_hit: bool
    elapsed_ms: float
    decoder_elapsed_ms: float

    def metadata(self) -> dict[str, object]:
        """Keep this call's work separate from the original decoder cost."""

        return {
            "roi": list(self.roi),
            "mode": self.mode,
            "cache_hit": self.cache_hit,
            "elapsed_ms": self.elapsed_ms,
            "decoder_elapsed_ms": self.decoder_elapsed_ms,
        }


class RoiQrDecodeCache:
    """At most three exact crop/mode results for the current image only.

    Empty tuples and unavailable (``None``) outputs remain distinct and are
    cached. A saturated cache decodes new keys without retaining them, so the
    bound cannot suppress a necessary decoder call. Exceptions are not cached.
    """

    def __init__(self, *, max_entries: int = MAX_CACHE_ENTRIES) -> None:
        if type(max_entries) is not int or not 1 <= max_entries <= MAX_CACHE_ENTRIES:
            raise ValueError("max_entries must be an integer between 1 and 3")
        self._max_entries = max_entries
        self._results: dict[tuple[RoiBounds, QrDecodeMode], RoiQrDecodeResult] = {}

    def decode(
        self,
        *,
        roi: RoiBounds,
        mode: QrDecodeMode,
        frame: object,
        decoder: Callable[[object], QrObservations],
    ) -> RoiQrDecodeResult:
        if (
            not isinstance(roi, tuple)
            or len(roi) != 4
            or any(type(value) is not int for value in roi)
            or roi[0] < 0
            or roi[1] < 0
            or roi[2] <= roi[0]
            or roi[3] <= roi[1]
        ):
            raise ValueError("roi must contain exact nonempty integer crop bounds")
        if mode not in ("full", "native"):
            raise ValueError("QR decoder mode must be full or native")
        key = (roi, mode)
        started = perf_counter()
        cached = self._results.get(key)
        if cached is not None:
            return replace(
                cached,
                cache_hit=True,
                elapsed_ms=(perf_counter() - started) * 1000.0,
            )
        observations = decoder(frame)
        elapsed_ms = (perf_counter() - started) * 1000.0
        result = RoiQrDecodeResult(
            roi=roi,
            mode=mode,
            observations=observations,
            cache_hit=False,
            elapsed_ms=elapsed_ms,
            decoder_elapsed_ms=elapsed_ms,
        )
        if len(self._results) < self._max_entries:
            self._results[key] = result
        return result


__all__ = ["RoiQrDecodeCache", "RoiQrDecodeResult"]
