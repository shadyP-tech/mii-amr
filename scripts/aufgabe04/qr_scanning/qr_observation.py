"""Decoded QR identity and its own validated original-image corners."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DecodedQrObservation:
    text: str
    corners: tuple[tuple[float, float], ...] | None
    detector: str
    scale: float = 1.0

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("decoded QR observation requires a nonblank identity")
        if not isinstance(self.detector, str) or not self.detector:
            raise ValueError("decoded QR observation requires detector provenance")
        if not math.isfinite(self.scale) or self.scale < 1:
            raise ValueError("decoded QR observation scale must be finite and >= 1")
        if self.corners is not None and validated_qr_corners(self.corners) != self.corners:
            raise ValueError("decoded QR corners must be a finite convex quadrilateral")


def qr_corner_groups(points):
    """Normalize single/multi backend arrays without combining symbols."""
    if hasattr(points, "tolist"):
        points = points.tolist()
    try:
        if len(points) == 4 and all(len(row) == 2 for row in points):
            return (points,)
        return tuple(points)
    except (TypeError, ValueError):
        return ()


def validated_qr_corners(points, *, image_shape=None, scale=1.0, border_px=0,
                         diagnostics: dict | None = None):
    """Restore preprocessing geometry and reject malformed/out-of-frame quads.

    Text remains usable as provisional identity evidence when a decoder gives
    no usable quadrilateral. Corners are never borrowed from another symbol.
    """
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics["reason"] = "missing" if points is None else "malformed"
    try:
        if not math.isfinite(scale) or scale < 1.0 or not math.isfinite(border_px):
            if diagnostics is not None:
                diagnostics["reason"] = "invalid_preprocessing"
            return None
        if hasattr(points, "tolist"):
            points = points.tolist()
        if len(points) == 1:
            points = points[0]
        if len(points) != 4:
            return None
        corners = tuple(((float(p[0]) - border_px) / scale,
                         (float(p[1]) - border_px) / scale) for p in points if len(p) == 2)
        if len(corners) != 4 or not all(math.isfinite(v) for p in corners for v in p):
            return None
        if diagnostics is not None:
            bounds = [min(p[0] for p in corners), min(p[1] for p in corners),
                      max(p[0] for p in corners), max(p[1] for p in corners)]
            diagnostics.update(normalized_bounds=bounds,
                               raw_bounds=[value * scale + border_px for value in bounds])
        if image_shape is not None:
            height, width = image_shape[:2]
            if any(not (0 <= u < width and 0 <= v < height) for u, v in corners):
                if diagnostics is not None:
                    diagnostics["reason"] = "out_of_bounds"
                return None
            # OpenCV 4.5's WeChat decoder without detector models may decode
            # a payload while returning the complete input rectangle as its
            # corners. That is decoder search extent, not symbol geometry.
            if all(
                min(abs(u), abs(u - (width - 1))) <= 1.0
                and min(abs(v), abs(v - (height - 1))) <= 1.0
                for u, v in corners
            ):
                if diagnostics is not None:
                    diagnostics["reason"] = "full_input_extent"
                return None
        turns = []
        for i in range(4):
            a, b, c = (corners[(i + k) % 4] for k in range(3))
            if math.hypot(a[0] - b[0], a[1] - b[1]) < 1.0:
                if diagnostics is not None:
                    diagnostics["reason"] = "degenerate"
                return None
            turns.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        if not (all(t > 1e-6 for t in turns) or all(t < -1e-6 for t in turns)):
            if diagnostics is not None:
                diagnostics["reason"] = "nonconvex"
            return None
        if diagnostics is not None:
            diagnostics["reason"] = "valid"
        return corners
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
