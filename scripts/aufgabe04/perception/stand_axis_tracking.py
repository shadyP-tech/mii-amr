from __future__ import annotations

import math
from dataclasses import dataclass, replace

from scripts.aufgabe04.perception.stand_axis_image import StandAxisImageEstimate


@dataclass(frozen=True)
class _HeadCandidateSignature:
    center_x_px: float
    center_y_px: float
    extent_px: float
    width_px: float
    height_px: float
    side_direction_rad: float
    corners_px: tuple[tuple[float, float], ...]
    yaw_deg: float | None
    source: str


@dataclass(frozen=True)
class HeadTemporalSelection:
    """A fresh, bounded-held, or unavailable head decision."""

    estimate: StandAxisImageEstimate | None
    current_accepted: bool
    held: bool
    reason: str
    held_age_sec: float | None = None

    @property
    def measurement_status(self) -> str:
        if self.current_accepted:
            return "fresh"
        if self.held:
            return "held"
        return "unavailable"


def _head_candidate_signature(
    estimate: StandAxisImageEstimate,
) -> _HeadCandidateSignature | None:
    if not estimate.usable or estimate.corners is None:
        return None
    top_left, top_right, bottom_right, bottom_left = estimate.corners
    xs = [point.u_px for point in estimate.corners]
    ys = [point.v_px for point in estimate.corners]
    if not all(math.isfinite(value) for value in (*xs, *ys)):
        return None
    extent = max(max(xs) - min(xs), max(ys) - min(ys))
    if extent <= 0.0:
        return None
    width_px = (
        math.dist((top_left.u_px, top_left.v_px), (top_right.u_px, top_right.v_px))
        + math.dist((bottom_left.u_px, bottom_left.v_px), (bottom_right.u_px, bottom_right.v_px))
    ) / 2.0
    left_side = (bottom_left.u_px - top_left.u_px, bottom_left.v_px - top_left.v_px)
    right_side = (bottom_right.u_px - top_right.u_px, bottom_right.v_px - top_right.v_px)
    left_height = math.hypot(*left_side)
    right_height = math.hypot(*right_side)
    height_px = (left_height + right_height) / 2.0
    if width_px <= 0.0 or height_px <= 0.0:
        return None
    left_direction = (left_side[0] / left_height, left_side[1] / left_height)
    right_direction = (right_side[0] / right_height, right_side[1] / right_height)
    if left_direction[0] * right_direction[0] + left_direction[1] * right_direction[1] < 0.0:
        right_direction = (-right_direction[0], -right_direction[1])
    side_direction_rad = math.atan2(
        left_direction[1] + right_direction[1],
        left_direction[0] + right_direction[0],
    )
    yaw_deg = estimate.yaw_deg
    if yaw_deg is not None and not math.isfinite(yaw_deg):
        yaw_deg = None
    return _HeadCandidateSignature(
        center_x_px=(min(xs) + max(xs)) / 2.0,
        center_y_px=(min(ys) + max(ys)) / 2.0,
        extent_px=extent,
        width_px=width_px,
        height_px=height_px,
        side_direction_rad=side_direction_rad,
        corners_px=tuple((point.u_px, point.v_px) for point in estimate.corners),
        yaw_deg=yaw_deg,
        source=estimate.source,
    )


class HeadCandidateTemporalGate:
    """Reject head jumps and expose bounded holds without creating evidence."""

    def __init__(
        self,
        *,
        max_center_jump_scale: float = 0.45,
        max_size_ratio: float = 1.60,
        max_axis_jump_deg: float = 35.0,
        max_width_ratio: float = 1.25,
        max_height_ratio: float = 1.25,
        max_corner_jump_scale: float = 0.18,
        max_side_direction_jump_deg: float = 8.0,
        reacquire_frames: int = 3,
        initial_acquire_frames: int = 1,
        hold_sec: float = 0.35,
        structure_owner_memory_sec: float | None = None,
        accepted_state_timeout_sec: float = 0.75,
    ) -> None:
        if not math.isfinite(hold_sec) or hold_sec < 0.0:
            raise ValueError("hold_sec must be finite and non-negative")
        self.max_center_jump_scale = max_center_jump_scale
        self.max_size_ratio = max_size_ratio
        self.max_axis_jump_deg = max_axis_jump_deg
        self.max_width_ratio = max_width_ratio
        self.max_height_ratio = max_height_ratio
        self.max_corner_jump_scale = max_corner_jump_scale
        self.max_side_direction_jump_deg = max_side_direction_jump_deg
        self.reacquire_frames = max(1, int(reacquire_frames))
        self.initial_acquire_frames = max(1, int(initial_acquire_frames))
        self.hold_sec = hold_sec
        if (
            not math.isfinite(accepted_state_timeout_sec)
            or accepted_state_timeout_sec <= 0.0
        ):
            raise ValueError(
                "accepted_state_timeout_sec must be finite and positive"
            )
        self.accepted_state_timeout_sec = float(
            accepted_state_timeout_sec
        )
        self.structure_owner_memory_sec = (
            max(0.75, hold_sec)
            if structure_owner_memory_sec is None
            else float(structure_owner_memory_sec)
        )
        if (
            not math.isfinite(self.structure_owner_memory_sec)
            or self.structure_owner_memory_sec < 0.0
        ):
            raise ValueError(
                "structure_owner_memory_sec must be finite and non-negative"
            )
        self._accepted: _HeadCandidateSignature | None = None
        self._pending: _HeadCandidateSignature | None = None
        self._pending_count = 0
        self._last_accepted_estimate: StandAxisImageEstimate | None = None
        self._last_accepted_at_sec: float | None = None
        self._last_structure_owner_at_sec: float | None = None

    def _expire_accepted_state(self) -> None:
        self._accepted = None
        self._last_accepted_estimate = None
        self._last_accepted_at_sec = None
        self._last_structure_owner_at_sec = None
        self._clear_pending()

    def _clear_pending(self) -> None:
        self._pending = None
        self._pending_count = 0

    def _compatible(
        self,
        previous: _HeadCandidateSignature,
        current: _HeadCandidateSignature,
    ) -> bool:
        minimum_extent = max(1.0, min(previous.extent_px, current.extent_px))
        center_jump = math.hypot(
            current.center_x_px - previous.center_x_px,
            current.center_y_px - previous.center_y_px,
        )
        if center_jump > max(18.0, self.max_center_jump_scale * minimum_extent):
            return False
        size_ratio = max(previous.extent_px, current.extent_px) / minimum_extent
        if size_ratio > self.max_size_ratio:
            return False
        width_ratio = max(previous.width_px, current.width_px) / max(
            1.0, min(previous.width_px, current.width_px)
        )
        if width_ratio > self.max_width_ratio:
            return False
        height_ratio = max(previous.height_px, current.height_px) / max(
            1.0, min(previous.height_px, current.height_px)
        )
        if height_ratio > self.max_height_ratio:
            return False
        maximum_corner_jump = max(
            math.dist(previous_corner, current_corner)
            for previous_corner, current_corner in zip(
                previous.corners_px, current.corners_px
            )
        )
        if maximum_corner_jump > max(
            4.0,
            self.max_corner_jump_scale * minimum_extent,
        ):
            return False
        side_delta = abs(
            (current.side_direction_rad - previous.side_direction_rad + math.pi / 2.0)
            % math.pi
            - math.pi / 2.0
        )
        if side_delta > math.radians(self.max_side_direction_jump_deg):
            return False
        if previous.yaw_deg is not None and current.yaw_deg is not None:
            axis_delta = abs(
                (current.yaw_deg - previous.yaw_deg + 90.0) % 180.0 - 90.0
            )
            if axis_delta > self.max_axis_jump_deg:
                return False
        return True

    def _size_compatible(
        self,
        previous: _HeadCandidateSignature,
        current: _HeadCandidateSignature,
    ) -> bool:
        minimum_extent = max(1.0, min(previous.extent_px, current.extent_px))
        return (
            max(previous.extent_px, current.extent_px) / minimum_extent
            <= self.max_size_ratio
        )

    def accept(self, estimate: StandAxisImageEstimate) -> tuple[bool, str]:
        signature = _head_candidate_signature(estimate)
        if signature is None:
            self._clear_pending()
            return False, estimate.reason
        if self._accepted is None:
            if self._pending is not None and self._compatible(self._pending, signature):
                self._pending = signature
                self._pending_count += 1
            else:
                self._pending = signature
                self._pending_count = 1
            if self._pending_count < self.initial_acquire_frames:
                return False, "temporal_head_bootstrap"
            self._accepted = signature
            self._clear_pending()
            return True, "accepted"
        if signature.source == "edge_qr_scaled_front":
            self._accepted = signature
            self._clear_pending()
            return True, "accepted_qr_anchor"
        if self._compatible(self._accepted, signature):
            if self._accepted.source != "edge_qr_scaled_front":
                self._accepted = signature
                self._clear_pending()
                return True, "accepted"

        if not self._size_compatible(self._accepted, signature):
            self._clear_pending()
            return False, "temporal_head_outlier"

        if self._pending is not None and self._compatible(self._pending, signature):
            self._pending = signature
            self._pending_count += 1
        else:
            self._pending = signature
            self._pending_count = 1
        if self._pending_count >= self.reacquire_frames:
            self._accepted = signature
            self._clear_pending()
            return True, "reacquired"
        return False, "temporal_head_outlier"

    def _accept_structure_tracking(
        self,
        estimate: StandAxisImageEstimate,
        *,
        now_sec: float,
    ) -> tuple[bool, str]:
        signature = _head_candidate_signature(estimate)
        if (
            signature is None
            or self._accepted is None
            or self._last_structure_owner_at_sec is None
        ):
            self._clear_pending()
            return False, "structure_owner_unavailable"
        owner_age_sec = now_sec - self._last_structure_owner_at_sec
        if not 0.0 <= owner_age_sec <= self.structure_owner_memory_sec:
            self._clear_pending()
            return False, "structure_owner_expired"
        if not self._compatible(self._accepted, signature):
            self._clear_pending()
            return False, "structure_tracking_outlier"
        self._accepted = signature
        self._clear_pending()
        return True, "accepted_structure_tracking"

    def stabilize(
        self,
        estimate: StandAxisImageEstimate,
        *,
        now_sec: float,
        rejection_reason: str | None = None,
    ) -> HeadTemporalSelection:
        if not math.isfinite(now_sec):
            raise ValueError("now_sec must be finite")
        if (
            self._accepted is not None
            and self._last_accepted_at_sec is not None
            and now_sec - self._last_accepted_at_sec
            > self.accepted_state_timeout_sec
        ):
            # A hidden accepted signature must not outlive the visible hold
            # and block a narrower, strongly rotated head indefinitely.
            self._expire_accepted_state()
        if rejection_reason is None:
            if estimate.source == "edge_structure_tracking_candidate":
                current_accepted, decision_reason = (
                    self._accept_structure_tracking(
                        estimate,
                        now_sec=now_sec,
                    )
                )
            else:
                current_accepted, decision_reason = self.accept(estimate)
        else:
            self._clear_pending()
            current_accepted = False
            decision_reason = rejection_reason

        if current_accepted:
            if estimate.source == "edge_structure_owned_head":
                self._last_structure_owner_at_sec = now_sec
            self._last_accepted_estimate = estimate
            self._last_accepted_at_sec = now_sec
            return HeadTemporalSelection(
                estimate=estimate,
                current_accepted=True,
                held=False,
                reason=decision_reason,
            )

        if (
            self.hold_sec > 0.0
            and self._last_accepted_estimate is not None
            and self._last_accepted_at_sec is not None
        ):
            age_sec = now_sec - self._last_accepted_at_sec
            if 0.0 <= age_sec <= self.hold_sec:
                return HeadTemporalSelection(
                    estimate=replace(
                        self._last_accepted_estimate,
                        reason=f"temporal_hold_after_{decision_reason}",
                    ),
                    current_accepted=False,
                    held=True,
                    reason=decision_reason,
                    held_age_sec=age_sec,
                )

        return HeadTemporalSelection(
            estimate=None,
            current_accepted=False,
            held=False,
            reason=decision_reason,
        )
