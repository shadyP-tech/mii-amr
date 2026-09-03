"""ROS-free state machine for bounded camera target registration.

The wide image search is proposal-only.  A proposal may move the expected
head centre, but only a second invocation of the ordinary strict metric-model
pipeline may become the selected measurement.  This keeps registration policy
out of the ROS adapter and prevents relaxed search evidence from entering
consensus or a motion-authorizing receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from scripts.aufgabe04.perception.stand_axis.models import (
    StandAxisEdgeDebugArtifacts,
    StandAxisImageEstimate,
)
from scripts.aufgabe04.real_robot.observer.contract import (
    BACKSIDE_AXIS_SAMPLE_SOURCE,
)
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import (
    HeadRoiAttempt,
    HeadRoiRegistrationDecision,
    registered_head_roi_attempt,
    validate_backside_registration_center_offset_ratio,
)


BACKSIDE_REACQUISITION_TRIGGER_REASONS = frozenset(
    {
        "model_backside_head_and_neck_unavailable",
        "model_backside_target_crop_unavailable",
        "model_backside_target_center_mismatch",
    }
)


@dataclass(frozen=True)
class HeadRoiEvaluation:
    """One metric-model evaluation and its coordinate frame."""

    attempt: HeadRoiAttempt
    frame: object
    estimate: StandAxisImageEstimate
    debug: StandAxisEdgeDebugArtifacts


@dataclass(frozen=True)
class CameraTargetRegistrationSelection:
    """Selected strict result plus complete proposal/retry provenance."""

    selected: HeadRoiEvaluation
    evaluations: tuple[HeadRoiEvaluation, ...]
    proposal: HeadRoiEvaluation | None
    decision: HeadRoiRegistrationDecision | None
    strict_retry: HeadRoiEvaluation | None

    @property
    def registered(self) -> bool:
        return self.strict_retry is not None and self.selected is self.strict_retry

    def metadata(self, *, enabled: bool) -> dict[str, object]:
        proposal = self.proposal
        strict_retry = self.strict_retry
        return {
            "enabled": bool(enabled),
            "attempted": proposal is not None,
            "strict_retry_applied": self.registered,
            "measurement_accepted": (
                self.registered and self.selected.estimate.usable
            ),
            "proposal_estimator_reason": (
                None if proposal is None else proposal.estimate.reason
            ),
            "proposal_qr_detected": (
                None if proposal is None else proposal.debug.qr_detected
            ),
            "proposal_head_center_error_ratio": (
                None
                if proposal is None
                else proposal.debug.head_center_error_ratio
            ),
            "decision": (
                None if self.decision is None else self.decision.metadata()
            ),
            "final_strict_estimator_reason": (
                None if strict_retry is None else strict_retry.estimate.reason
            ),
            "final_strict_head_center_error_ratio": (
                None
                if strict_retry is None
                else strict_retry.debug.head_center_error_ratio
            ),
        }


def select_camera_target_measurement(
    roi_attempts: tuple[HeadRoiAttempt, ...],
    *,
    tracked_pose: object | None,
    evaluate: Callable[[HeadRoiAttempt, object | None], HeadRoiEvaluation],
    enable_reacquisition: bool,
    max_center_offset_ratio: float,
) -> CameraTargetRegistrationSelection:
    """Select a nominal or strictly reverified registered measurement."""

    if not roi_attempts:
        raise ValueError("roi_attempts must contain a nominal attempt")
    # Validate the safety bound regardless of which state-machine branch is
    # taken.  A disabled or unnecessary reacquisition must not make an invalid
    # caller-supplied limit appear acceptable.
    max_center_offset_ratio = (
        validate_backside_registration_center_offset_ratio(
            max_center_offset_ratio
        )
    )
    primary = evaluate(roi_attempts[0], tracked_pose)
    evaluations = [primary]
    if (
        not enable_reacquisition
        or len(roi_attempts) < 2
        or primary.estimate.usable
        or primary.debug.qr_detected
        or tracked_pose is not None
        or primary.estimate.source != BACKSIDE_AXIS_SAMPLE_SOURCE
        or primary.estimate.reason not in BACKSIDE_REACQUISITION_TRIGGER_REASONS
    ):
        return CameraTargetRegistrationSelection(
            selected=primary,
            evaluations=tuple(evaluations),
            proposal=None,
            decision=None,
            strict_retry=None,
        )

    proposal = evaluate(roi_attempts[1], None)
    evaluations.append(proposal)
    eligible = (
        proposal.estimate.corners is not None
        and not proposal.debug.qr_detected
        and proposal.estimate.source == BACKSIDE_AXIS_SAMPLE_SOURCE
        and (
            proposal.estimate.usable
            or (
                proposal.estimate.reason
                == "model_backside_target_center_mismatch"
            )
        )
    )
    decision = None
    if eligible:
        decision = registered_head_roi_attempt(
            proposal.attempt,
            proposal.estimate.corners,
            max_center_offset_ratio=max_center_offset_ratio,
        )
    if decision is not None and decision.accepted and decision.attempt is not None:
        strict_retry = evaluate(decision.attempt, None)
        evaluations.append(strict_retry)
        return CameraTargetRegistrationSelection(
            selected=strict_retry,
            evaluations=tuple(evaluations),
            proposal=proposal,
            decision=decision,
            strict_retry=strict_retry,
        )

    # The wide-search result remains proposal-only, whether it is usable or
    # failed.  Its detailed outcome is retained in ``proposal`` and metadata;
    # it never becomes the selected downstream measurement without the strict
    # registered second pass above.
    return CameraTargetRegistrationSelection(
        selected=primary,
        evaluations=tuple(evaluations),
        proposal=proposal,
        decision=decision,
        strict_retry=None,
    )
