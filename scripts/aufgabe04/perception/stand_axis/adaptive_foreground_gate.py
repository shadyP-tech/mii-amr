"""Conservative colour-adaptive foreground support for real-camera topology.

The silhouette fitter deliberately remains colour agnostic.  This module only
uses a *verified repeated-rib background region* to learn a local Lab colour
model and builds a generous gate around pixels that differ from it. No stand
hue is configured: a red, blue, or green stand can supply foreground support
equally. The gate is for proposal topology only; raw Canny evidence remains
untouched for the final colour-agnostic corner fit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptiveForegroundGateResult:
    """Result of a local-background foreground-gate attempt."""

    gate: object | None
    applied: bool
    background_pixel_count: int
    foreground_fraction: float
    reason: str


@dataclass(frozen=True)
class AdaptiveBackgroundModel:
    """Robust Lab statistics learned from a geometry-owned background family."""

    centre: tuple[float, float, float]
    scale: tuple[float, float, float]
    background_pixel_count: int


class AdaptiveForegroundGateTracker:
    """Reuse only the colour model when the rib seed flickers between frames.

    The binary gate is always recomputed from the current frame. This avoids a
    static image-position mask while preventing a one-frame Hough miss from
    reopening the complete heater topology.
    """

    def __init__(self, *, model_ttl_sec: float = 0.75) -> None:
        if model_ttl_sec <= 0.0:
            raise ValueError("model_ttl_sec must be positive")
        self.model_ttl_sec = float(model_ttl_sec)
        self._model: AdaptiveBackgroundModel | None = None
        self._model_at_sec: float | None = None
        self._has_activated = False

    @property
    def enforcement_active(self) -> bool:
        return self._model is not None

    @property
    def has_activated(self) -> bool:
        return self._has_activated

    def update(
        self,
        cv2,
        numpy,
        frame,
        background_seed_mask,
        *,
        now_sec: float,
        **gate_options,
    ) -> AdaptiveForegroundGateResult:
        if self._model_at_sec is not None and now_sec - self._model_at_sec > self.model_ttl_sec:
            self._model = None
            self._model_at_sec = None

        current_model = adaptive_background_model_from_seed(
            cv2,
            numpy,
            frame,
            background_seed_mask,
            min_background_pixels=int(gate_options.get("min_background_pixels", 128)),
        )
        if current_model is not None:
            self._model = current_model
            self._model_at_sec = now_sec
            self._has_activated = True

        if self._model is None:
            return AdaptiveForegroundGateResult(
                gate=None,
                applied=False,
                background_pixel_count=0,
                foreground_fraction=0.0,
                reason="background_model_unavailable",
            )

        result = adaptive_foreground_gate_from_model(
            cv2,
            numpy,
            frame,
            self._model,
            **gate_options,
        )
        if not result.applied:
            return result
        return AdaptiveForegroundGateResult(
            gate=result.gate,
            applied=True,
            background_pixel_count=result.background_pixel_count,
            foreground_fraction=result.foreground_fraction,
            reason=(
                "applied_current_model"
                if current_model is not None
                else "applied_cached_model"
            ),
        )


def _odd_kernel_size(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def adaptive_background_model_from_seed(
    cv2,
    numpy,
    frame,
    background_seed_mask,
    *,
    min_background_pixels: int = 128,
) -> AdaptiveBackgroundModel | None:
    """Learn robust Lab statistics without retaining the seed's image location."""

    if frame is None or len(frame.shape) != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a non-empty BGR image")
    if background_seed_mask is None or background_seed_mask.shape[:2] != frame.shape[:2]:
        raise ValueError("background_seed_mask must match the frame size")
    if min_background_pixels <= 0:
        raise ValueError("min_background_pixels must be positive")

    seed = background_seed_mask > 0
    background_pixel_count = int(numpy.count_nonzero(seed))
    if background_pixel_count < min_background_pixels:
        return None
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(numpy.float32)
    background = lab[seed]
    centre = numpy.median(background, axis=0)
    scale = numpy.maximum(
        1.4826 * numpy.median(numpy.abs(background - centre), axis=0),
        8.0,
    )
    return AdaptiveBackgroundModel(
        centre=tuple(float(value) for value in centre),
        scale=tuple(float(value) for value in scale),
        background_pixel_count=background_pixel_count,
    )


def adaptive_foreground_gate_from_model(
    cv2,
    numpy,
    frame,
    model: AdaptiveBackgroundModel,
    *,
    min_background_pixels: int = 128,
    robust_distance_threshold: float = 3.5,
    min_foreground_fraction: float = 0.01,
    max_foreground_fraction: float = 0.85,
    morphology_kernel_px: int = 5,
    boundary_dilate_px: int = 7,
) -> AdaptiveForegroundGateResult:
    """Apply a background colour model to the current frame."""

    if robust_distance_threshold <= 0.0:
        raise ValueError("robust_distance_threshold must be positive")
    if not 0.0 <= min_foreground_fraction < max_foreground_fraction <= 1.0:
        raise ValueError("foreground fraction bounds must be inside [0, 1]")

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(numpy.float32)
    centre = numpy.asarray(model.centre, dtype=numpy.float32)
    scale = numpy.asarray(model.scale, dtype=numpy.float32)
    normalized = (lab - centre) / scale
    distance = numpy.sqrt(numpy.sum(normalized * normalized, axis=2))
    foreground = (distance >= robust_distance_threshold).astype(numpy.uint8) * 255

    kernel_size = _odd_kernel_size(morphology_kernel_px)
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)

    foreground_fraction = float(numpy.count_nonzero(foreground)) / float(foreground.size)
    if not min_foreground_fraction <= foreground_fraction <= max_foreground_fraction:
        return AdaptiveForegroundGateResult(
            gate=None,
            applied=False,
            background_pixel_count=model.background_pixel_count,
            foreground_fraction=foreground_fraction,
            reason="foreground_coverage_unreliable",
        )

    boundary_size = _odd_kernel_size(boundary_dilate_px)
    if boundary_size > 1:
        boundary_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (boundary_size, boundary_size),
        )
        foreground = cv2.dilate(foreground, boundary_kernel, iterations=1)
    return AdaptiveForegroundGateResult(
        gate=foreground,
        applied=True,
        background_pixel_count=model.background_pixel_count,
        foreground_fraction=foreground_fraction,
        reason="applied",
    )


def adaptive_foreground_gate_from_background(
    cv2,
    numpy,
    frame,
    background_seed_mask,
    *,
    min_background_pixels: int = 128,
    robust_distance_threshold: float = 3.5,
    min_foreground_fraction: float = 0.01,
    max_foreground_fraction: float = 0.85,
    morphology_kernel_px: int = 5,
    boundary_dilate_px: int = 7,
) -> AdaptiveForegroundGateResult:
    """Return a broad foreground gate learned from a known background sample.

    ``background_seed_mask`` must originate from geometry (the repeated-rib
    detector), not a hand-positioned image area.  The model uses robust Lab
    channel statistics, making it independent of a particular foreground hue.
    A gate is intentionally rejected when it cannot safely discriminate the
    scene; callers then preserve their normal colour-agnostic Canny input.
    """

    if frame is None or len(frame.shape) != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a non-empty BGR image")
    if background_seed_mask is None or background_seed_mask.shape[:2] != frame.shape[:2]:
        raise ValueError("background_seed_mask must match the frame size")
    if min_background_pixels <= 0:
        raise ValueError("min_background_pixels must be positive")
    if robust_distance_threshold <= 0.0:
        raise ValueError("robust_distance_threshold must be positive")
    if not 0.0 <= min_foreground_fraction < max_foreground_fraction <= 1.0:
        raise ValueError("foreground fraction bounds must be inside [0, 1]")

    model = adaptive_background_model_from_seed(
        cv2,
        numpy,
        frame,
        background_seed_mask,
        min_background_pixels=min_background_pixels,
    )
    if model is None:
        return AdaptiveForegroundGateResult(
            gate=None,
            applied=False,
            background_pixel_count=0,
            foreground_fraction=0.0,
            reason="insufficient_background_sample",
        )
    return adaptive_foreground_gate_from_model(
        cv2,
        numpy,
        frame,
        model,
        min_background_pixels=min_background_pixels,
        robust_distance_threshold=robust_distance_threshold,
        min_foreground_fraction=min_foreground_fraction,
        max_foreground_fraction=max_foreground_fraction,
        morphology_kernel_px=morphology_kernel_px,
        boundary_dilate_px=boundary_dilate_px,
    )
