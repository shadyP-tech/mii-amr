"""Pure ROI policy for target-centred real-camera stand reacquisition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

from scripts.aufgabe04.real_robot.configuration.geometry import (
    CameraIntrinsics,
    ImageRoi,
    OpticalProjection,
    roi_from_projection,
)


DEFAULT_BACKSIDE_REACQUISITION_PADDING_SCALE = 4.5
MAX_BACKSIDE_REACQUISITION_PADDING_SCALE = 4.5
DEFAULT_BACKSIDE_REGISTRATION_MAX_CENTER_OFFSET_RATIO = 1.5
MAX_BACKSIDE_REGISTRATION_CENTER_OFFSET_RATIO = 1.5
DEFAULT_BACKSIDE_TARGET_CROP_HALF_WIDTH_RATIO = 1.25
BACKSIDE_REACQUISITION_TARGET_CROP_HALF_WIDTH_RATIO = 2.25


@dataclass(frozen=True)
class HeadRoiAttempt:
    """One bounded crop for model-backed target observation."""

    roi: ImageRoi
    source: str
    padding_scale: float
    expected_center_u_px: float
    expected_center_v_px: float
    expected_head_height_px: float
    backside_target_crop_half_width_ratio: float = (
        DEFAULT_BACKSIDE_TARGET_CROP_HALF_WIDTH_RATIO
    )

    def metadata(self) -> dict[str, object]:
        return {
            "source": self.source,
            "padding_scale": self.padding_scale,
            "roi": asdict(self.roi),
            "expected_center_u_px": self.expected_center_u_px,
            "expected_center_v_px": self.expected_center_v_px,
            "expected_head_height_px": self.expected_head_height_px,
            "backside_target_crop_half_width_ratio": (
                self.backside_target_crop_half_width_ratio
            ),
        }


@dataclass(frozen=True)
class HeadRoiRegistrationDecision:
    """Fail-closed decision to recenter one expanded acquisition ROI."""

    accepted: bool
    reason: str
    attempt: HeadRoiAttempt | None
    projected_center_u_px: float
    projected_center_v_px: float
    detected_center_u_px: float | None
    detected_center_v_px: float | None
    center_offset_px: float | None
    center_offset_ratio: float | None
    max_center_offset_ratio: float

    def metadata(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "projected_center_u_px": self.projected_center_u_px,
            "projected_center_v_px": self.projected_center_v_px,
            "detected_center_u_px": self.detected_center_u_px,
            "detected_center_v_px": self.detected_center_v_px,
            "center_offset_px": self.center_offset_px,
            "center_offset_ratio": self.center_offset_ratio,
            "max_center_offset_ratio": self.max_center_offset_ratio,
        }


def target_centered_head_roi_attempts(
    projection: OpticalProjection,
    intrinsics: CameraIntrinsics,
    *,
    expected_head_height_px: float,
    nominal_padding_scale: float,
    backside_reacquisition_padding_scale: float = (
        DEFAULT_BACKSIDE_REACQUISITION_PADDING_SCALE
    ),
    enable_backside_reacquisition: bool = True,
) -> tuple[HeadRoiAttempt, ...]:
    """Return ordered crops without moving the expected target centre.

    The nominal crop preserves the legacy target projection.  The optional
    reacquisition crop uses its own certified extent and stays centred on the
    same map/TF projected head centre, so downstream head-centre, bearing,
    LiDAR, consensus, and receipt gates remain responsible for rejecting a
    neighbouring stand.
    """

    _validate_padding_scale(nominal_padding_scale, "nominal_padding_scale")
    _validate_padding_scale(
        backside_reacquisition_padding_scale,
        "backside_reacquisition_padding_scale",
    )
    if (
        float(backside_reacquisition_padding_scale)
        > MAX_BACKSIDE_REACQUISITION_PADDING_SCALE
    ):
        raise ValueError(
            "backside_reacquisition_padding_scale cannot exceed the "
            f"certified {MAX_BACKSIDE_REACQUISITION_PADDING_SCALE:g} bound"
        )
    if (
        not math.isfinite(float(expected_head_height_px))
        or float(expected_head_height_px) <= 0.0
    ):
        raise ValueError("expected_head_height_px must be finite and positive")

    nominal = roi_from_projection(
        projection,
        intrinsics,
        padding_scale=nominal_padding_scale,
    )
    if nominal is None:
        return ()
    attempts = [
        HeadRoiAttempt(
            roi=nominal,
            source="nominal_projection",
            padding_scale=float(nominal_padding_scale),
            expected_center_u_px=float(projection.u_px),
            expected_center_v_px=float(projection.v_px),
            expected_head_height_px=float(expected_head_height_px),
        )
    ]
    if enable_backside_reacquisition:
        # This scale is intentionally independent of the legacy nominal ROI
        # setting.  Taking ``max(..., nominal_padding_scale)`` here allowed an
        # old, uncapped ``--head-roi-padding-scale`` value to silently enlarge
        # the proposal beyond the certified reacquisition bound.
        proposal_scale = float(backside_reacquisition_padding_scale)
        proposal_roi = roi_from_projection(
            projection,
            intrinsics,
            padding_scale=proposal_scale,
        )
        # Keep the proposal attempt even when image-boundary clipping makes
        # its outer ROI equal to the nominal ROI: its bounded internal search
        # width is still deliberately different.
        if proposal_roi is not None:
            attempts.append(
                HeadRoiAttempt(
                    roi=proposal_roi,
                    source="target_centered_backside_reacquisition",
                    padding_scale=proposal_scale,
                    expected_center_u_px=float(projection.u_px),
                    expected_center_v_px=float(projection.v_px),
                    expected_head_height_px=float(expected_head_height_px),
                    backside_target_crop_half_width_ratio=(
                        BACKSIDE_REACQUISITION_TARGET_CROP_HALF_WIDTH_RATIO
                    ),
                )
            )
    return tuple(attempts)


def registered_head_roi_attempt(
    proposal_attempt: HeadRoiAttempt,
    detected_corners: Sequence[object] | None,
    *,
    max_center_offset_ratio: float = (
        DEFAULT_BACKSIDE_REGISTRATION_MAX_CENTER_OFFSET_RATIO
    ),
) -> HeadRoiRegistrationDecision:
    """Build a strict second-pass attempt from bounded image evidence.

    A shifted search may propose a head outside the normal projection-centre
    gate, but that proposal is never itself returned as a committable
    measurement.  This function only recentres the expected head location for
    a second ordinary metric-model pass.  Candidate identity is subsequently
    rebound to the map candidate by the camera-bearing/LiDAR range gate.
    """

    max_ratio = validate_backside_registration_center_offset_ratio(
        max_center_offset_ratio
    )
    _validate_reacquisition_proposal(proposal_attempt)
    base = {
        "projected_center_u_px": proposal_attempt.expected_center_u_px,
        "projected_center_v_px": proposal_attempt.expected_center_v_px,
        "max_center_offset_ratio": max_ratio,
    }
    if detected_corners is None or len(detected_corners) != 4:
        return HeadRoiRegistrationDecision(
            accepted=False,
            reason="detected_head_corners_unavailable",
            attempt=None,
            detected_center_u_px=None,
            detected_center_v_px=None,
            center_offset_px=None,
            center_offset_ratio=None,
            **base,
        )
    try:
        local_us = tuple(float(getattr(point, "u_px")) for point in detected_corners)
        local_vs = tuple(float(getattr(point, "v_px")) for point in detected_corners)
    except (AttributeError, TypeError, ValueError):
        local_us = ()
        local_vs = ()
    if not local_us or not all(
        math.isfinite(value) for value in (*local_us, *local_vs)
    ):
        return HeadRoiRegistrationDecision(
            accepted=False,
            reason="detected_head_corners_invalid",
            attempt=None,
            detected_center_u_px=None,
            detected_center_v_px=None,
            center_offset_px=None,
            center_offset_ratio=None,
            **base,
        )
    detected_u = sum(local_us) / 4.0 + proposal_attempt.roi.x0
    detected_v = sum(local_vs) / 4.0 + proposal_attempt.roi.y0
    offset_px = math.hypot(
        detected_u - proposal_attempt.expected_center_u_px,
        detected_v - proposal_attempt.expected_center_v_px,
    )
    offset_ratio = offset_px / proposal_attempt.expected_head_height_px
    if offset_ratio > max_ratio:
        return HeadRoiRegistrationDecision(
            accepted=False,
            reason="detected_head_outside_registration_window",
            attempt=None,
            detected_center_u_px=detected_u,
            detected_center_v_px=detected_v,
            center_offset_px=offset_px,
            center_offset_ratio=offset_ratio,
            **base,
        )
    roi = proposal_attempt.roi
    if not (
        roi.x0 <= detected_u < roi.x1
        and roi.y0 <= detected_v < roi.y1
    ):
        return HeadRoiRegistrationDecision(
            accepted=False,
            reason="detected_head_center_outside_acquisition_roi",
            attempt=None,
            detected_center_u_px=detected_u,
            detected_center_v_px=detected_v,
            center_offset_px=offset_px,
            center_offset_ratio=offset_ratio,
            **base,
        )
    attempt = HeadRoiAttempt(
        roi=roi,
        source="camera_registered_backside_reacquisition",
        padding_scale=float(proposal_attempt.padding_scale),
        expected_center_u_px=detected_u,
        expected_center_v_px=detected_v,
        expected_head_height_px=proposal_attempt.expected_head_height_px,
        backside_target_crop_half_width_ratio=(
            DEFAULT_BACKSIDE_TARGET_CROP_HALF_WIDTH_RATIO
        ),
    )
    return HeadRoiRegistrationDecision(
        accepted=True,
        reason="bounded_head_center_registration_accepted",
        attempt=attempt,
        detected_center_u_px=detected_u,
        detected_center_v_px=detected_v,
        center_offset_px=offset_px,
        center_offset_ratio=offset_ratio,
        **base,
    )


def _validate_padding_scale(value: float, name: str) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and at least 1") from exc
    if not math.isfinite(numeric) or numeric < 1.0:
        raise ValueError(f"{name} must be finite and at least 1")


def validate_backside_registration_center_offset_ratio(value: float) -> float:
    """Return a finite registration limit within the certified image bound."""

    try:
        max_ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "max_center_offset_ratio must be finite, positive, and no greater "
            f"than {MAX_BACKSIDE_REGISTRATION_CENTER_OFFSET_RATIO}"
        ) from exc
    if (
        not math.isfinite(max_ratio)
        or max_ratio <= 0.0
        or max_ratio > MAX_BACKSIDE_REGISTRATION_CENTER_OFFSET_RATIO
    ):
        raise ValueError(
            "max_center_offset_ratio must be finite, positive, and no greater "
            f"than {MAX_BACKSIDE_REGISTRATION_CENTER_OFFSET_RATIO}"
        )
    return max_ratio


def _validate_reacquisition_proposal(attempt: HeadRoiAttempt) -> None:
    """Reject hand-built proposal attempts that bypass generation-time caps."""

    if attempt.source != "target_centered_backside_reacquisition":
        raise ValueError(
            "proposal_attempt must be a target-centred backside reacquisition"
        )
    _validate_padding_scale(attempt.padding_scale, "proposal_attempt.padding_scale")
    proposal_padding_scale = float(attempt.padding_scale)
    if proposal_padding_scale > MAX_BACKSIDE_REACQUISITION_PADDING_SCALE:
        raise ValueError(
            "proposal_attempt.padding_scale cannot exceed the certified "
            f"{MAX_BACKSIDE_REACQUISITION_PADDING_SCALE:g} bound"
        )
    try:
        target_crop_ratio = float(
            attempt.backside_target_crop_half_width_ratio
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "proposal_attempt must use the bounded backside reacquisition "
            "target-crop policy"
        ) from exc
    if not math.isclose(
        target_crop_ratio,
        BACKSIDE_REACQUISITION_TARGET_CROP_HALF_WIDTH_RATIO,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "proposal_attempt must use the bounded backside reacquisition "
            "target-crop policy"
        )
    try:
        expected_u = float(attempt.expected_center_u_px)
        expected_v = float(attempt.expected_center_v_px)
        expected_height = float(attempt.expected_head_height_px)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "proposal_attempt expected centre and head height must be finite, "
            "with positive head height"
        ) from exc
    if (
        not math.isfinite(expected_u)
        or not math.isfinite(expected_v)
        or not math.isfinite(expected_height)
        or expected_height <= 0.0
    ):
        raise ValueError(
            "proposal_attempt expected centre and head height must be finite, "
            "with positive head height"
        )
    roi = attempt.roi
    if (
        roi.x0 < 0
        or roi.y0 < 0
        or roi.x1 <= roi.x0
        or roi.y1 <= roi.y0
        or not (roi.x0 <= expected_u < roi.x1)
        or not (roi.y0 <= expected_v < roi.y1)
    ):
        raise ValueError(
            "proposal_attempt ROI must contain its finite expected centre"
        )
