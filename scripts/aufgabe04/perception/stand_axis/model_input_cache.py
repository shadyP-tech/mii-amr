"""Reuse deterministic inputs for repeated exact crops of one current image.

The cache never contains a pose, a backside classification, fitted geometry,
or admission evidence. A registered acquisition must still evaluate all of
those against its current projection and gates. Create one cache per rectified
image and pass only exact views into that image, with their full-image bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from time import perf_counter
from typing import Callable

from scripts.aufgabe04.qr_scanning.qr_observation import DecodedQrObservation


RoiBounds = tuple[int, int, int, int]
MAX_CACHE_ENTRIES = 3


@dataclass
class _InputEntry:
    values: dict[str, object] = field(default_factory=dict)
    producer_ms: dict[str, float] = field(default_factory=dict)


class _CurrentCropInputs:
    def __init__(self, entry: _InputEntry, metadata: dict[str, object]) -> None:
        self._entry = entry
        self._metadata = metadata

    def compute(self, stage: str, producer: Callable[[], object]):
        if stage not in ("edge_preprocessing", "qr_detection"):
            raise ValueError("only raw edges and QR quadrilaterals may be cached")
        started = perf_counter()
        hit = stage in self._entry.values
        if hit:
            value = self._entry.values[stage]
            # Geometry/debug consumers may mutate their own edge image. They
            # cannot thereby alter the raw evidence used by another evaluation.
            if stage == "edge_preprocessing":
                value = value.copy()
        else:
            value = producer()  # Exceptions propagate and are never cached.
            self._entry.producer_ms[stage] = (perf_counter() - started) * 1000.0
            self._entry.values[stage] = (
                value.copy() if stage == "edge_preprocessing" else value
            )
        self._metadata[stage] = {
            "cache_hit": hit,
            "elapsed_ms": (perf_counter() - started) * 1000.0,
            "producer_elapsed_ms": self._entry.producer_ms[stage],
        }
        return value


class MetricModelInputCache:
    """At most three exact crop/settings entries from one rectified image.

    Crop memory identity prevents another image from borrowing these inputs.
    A pixel digest additionally makes in-place image changes miss the cache.
    A saturated cache performs all necessary work without retaining new keys.
    """

    def __init__(self, frame, *, max_entries: int = MAX_CACHE_ENTRIES) -> None:
        if type(max_entries) is not int or not 1 <= max_entries <= MAX_CACHE_ENTRIES:
            raise ValueError("max_entries must be an integer between 1 and 3")
        if not hasattr(frame, "__array_interface__") or len(frame.shape) < 2:
            raise ValueError("model input cache requires a rectified image array")
        self._frame = frame
        self._max_entries = max_entries
        self._entries: dict[tuple, _InputEntry] = {}
        self.last_metadata: dict[str, object] = {}

    def begin(
        self,
        frame,
        *,
        roi: RoiBounds,
        cv2,
        edge_preprocess: str,
        blur_kernel: int,
        canny_low: int,
        canny_high: int,
        pose_hint_present: bool,
        qr_observations: tuple[DecodedQrObservation, ...] | None,
    ) -> _CurrentCropInputs:
        if (
            not isinstance(roi, tuple)
            or len(roi) != 4
            or any(type(value) is not int for value in roi)
            or not 0 <= roi[0] < roi[2] <= self._frame.shape[1]
            or not 0 <= roi[1] < roi[3] <= self._frame.shape[0]
        ):
            raise ValueError("roi must contain exact nonempty image crop bounds")
        expected = self._frame[roi[1]:roi[3], roi[0]:roi[2]]
        if (
            not hasattr(frame, "__array_interface__")
            or frame.shape != expected.shape
            or frame.dtype != expected.dtype
            or frame.strides != expected.strides
            or frame.__array_interface__["data"][0]
            != expected.__array_interface__["data"][0]
        ):
            raise ValueError("cached model inputs require the exact source-image ROI view")
        if qr_observations is not None and any(
            not isinstance(item, DecodedQrObservation) for item in qr_observations
        ):
            raise ValueError("metric model QR observations have an invalid type")
        qr_key = None if qr_observations is None else tuple(qr_observations)
        started = perf_counter()
        key = (
            roi, tuple(frame.shape), frame.dtype.str,
            sha256(frame.tobytes()).digest(), id(cv2), edge_preprocess,
            blur_kernel, canny_low, canny_high, bool(pose_hint_present), qr_key,
        )
        entry = self._entries.get(key)
        if entry is None:
            entry = _InputEntry()
            if len(self._entries) < self._max_entries:
                self._entries[key] = entry
        self.last_metadata = {
            "roi": list(roi),
            "mode": "native" if pose_hint_present else "full",
            "key_elapsed_ms": (perf_counter() - started) * 1000.0,
        }
        return _CurrentCropInputs(entry, self.last_metadata)


__all__ = ["MetricModelInputCache"]
