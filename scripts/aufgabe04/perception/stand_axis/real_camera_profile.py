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
    min_edge_height_px: float


@dataclass(frozen=True)
class RealCameraStandAxisProfile:
    """Validated offline-candidate settings for rectified real-camera crops.

    The measured model owns geometry. This profile exposes only the
    preprocessing choice and bounded Canny pair, plus a scale-derived minimum
    rail height for current-frame model refinement.
    """

    edge_preprocess: str = "channel_union"
    canny_low: int = 20
    canny_high: int = 60

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

        min_edge_height_px = max(5.0, min(14.0, 0.18 * expected))
        return ResolvedRealCameraStandAxisProfile(
            edge_preprocess=self.edge_preprocess,
            canny_low=self.canny_low,
            canny_high=self.canny_high,
            expected_head_size_px=expected,
            min_edge_height_px=min_edge_height_px,
        )
