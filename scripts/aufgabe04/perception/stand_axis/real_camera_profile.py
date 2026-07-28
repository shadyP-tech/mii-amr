"""Pure parameter profile for the passive real-camera stand-axis adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass


_EDGE_PREPROCESS_MODES = frozenset(("channel_union", "gray"))


@dataclass(frozen=True)
class ResolvedRealCameraStandAxisProfile:
    """Estimator parameters after resolving image-scale-dependent gates."""

    edge_preprocess: str
    canny_low: int
    canny_high: int
    expected_head_size_px: float
    min_area_px: float
    min_edge_height_px: float
    close_kernel: int
    min_aspect_ratio: float
    max_aspect_ratio: float

    def estimator_kwargs(self) -> dict[str, object]:
        """Return only the profile-owned façade estimator arguments."""

        return {
            "edge_preprocess": self.edge_preprocess,
            "canny_low": self.canny_low,
            "canny_high": self.canny_high,
            "min_area_px": self.min_area_px,
            "min_edge_height_px": self.min_edge_height_px,
            "close_kernel": self.close_kernel,
            "min_aspect_ratio": self.min_aspect_ratio,
            "max_aspect_ratio": self.max_aspect_ratio,
        }


@dataclass(frozen=True)
class RealCameraStandAxisProfile:
    """Validated offline-candidate settings for rectified real-camera crops.

    The profile deliberately exposes only the preprocessing choice and the
    bounded Canny pair as runtime choices. Geometry gates remain one fixed
    recipe and are resolved from the projected head size so a CLI cannot
    silently create a different detector.
    """

    edge_preprocess: str = "channel_union"
    canny_low: int = 20
    canny_high: int = 60
    min_aspect_ratio: float = 0.45
    max_aspect_ratio: float = 1.80

    def __post_init__(self) -> None:
        if self.edge_preprocess not in _EDGE_PREPROCESS_MODES:
            allowed = ", ".join(sorted(_EDGE_PREPROCESS_MODES))
            raise ValueError(f"edge_preprocess must be one of: {allowed}")
        if (
            isinstance(self.canny_low, bool)
            or isinstance(self.canny_high, bool)
            or not isinstance(self.canny_low, int)
            or not isinstance(self.canny_high, int)
            or not 0 <= self.canny_low < self.canny_high <= 255
        ):
            raise ValueError(
                "Canny thresholds must be integers satisfying "
                "0 <= low < high <= 255"
            )
        aspect_values = (self.min_aspect_ratio, self.max_aspect_ratio)
        if not all(math.isfinite(value) for value in aspect_values):
            raise ValueError("aspect-ratio gates must be finite")
        if not (
            0.0 < self.min_aspect_ratio <= 1.0 <= self.max_aspect_ratio
            and self.min_aspect_ratio < self.max_aspect_ratio
        ):
            raise ValueError(
                "aspect-ratio gates must straddle one and be strictly ordered"
            )

    @classmethod
    def from_cli(
        cls,
        *,
        edge_preprocess: str,
        canny_low: int,
        canny_high: int,
    ) -> "RealCameraStandAxisProfile":
        """Normalize the hyphenated CLI spelling into the façade spelling."""

        return cls(
            edge_preprocess=str(edge_preprocess).replace("-", "_"),
            canny_low=canny_low,
            canny_high=canny_high,
        )

    def resolve(
        self,
        expected_head_size_px: float,
    ) -> ResolvedRealCameraStandAxisProfile:
        """Resolve the fixed real-camera recipe at one projected head scale."""

        expected = float(expected_head_size_px)
        if not math.isfinite(expected) or expected <= 0.0:
            raise ValueError("expected_head_size_px must be finite and positive")

        min_area_px = max(40.0, 0.10 * expected**2)
        min_edge_height_px = max(5.0, min(14.0, 0.18 * expected))
        close_kernel = min(7, max(3, int(round(0.05 * expected)) | 1))
        return ResolvedRealCameraStandAxisProfile(
            edge_preprocess=self.edge_preprocess,
            canny_low=self.canny_low,
            canny_high=self.canny_high,
            expected_head_size_px=expected,
            min_area_px=min_area_px,
            min_edge_height_px=min_edge_height_px,
            close_kernel=close_kernel,
            min_aspect_ratio=self.min_aspect_ratio,
            max_aspect_ratio=self.max_aspect_ratio,
        )
