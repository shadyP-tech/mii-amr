"""JSON-safe values for frame-aligned diagnostic recordings."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import math
from pathlib import Path


def recording_metadata(value):
    """Preserve structured evidence; unavailable numerical values become null.

    Images/arrays are intentionally unsupported: source pixels belong in the
    recorder's lossless image files, not the JSON timeline.
    """

    if is_dataclass(value):
        return recording_metadata(asdict(value))
    if isinstance(value, dict):
        return {key: recording_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [recording_metadata(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported recording metadata: {type(value).__name__}")
