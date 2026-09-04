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
    REGISTERED_BACKSIDE_REACQUISITION_SOURCE,
    REGISTERED_QR_MODEL_REACQUISITION_SOURCE,
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
QR_MODEL_REACQUISITION_TRIGGER_REASONS = frozenset(
    {
        "model_pose_seed_unavailable",
        "projected_head_outside_image",
    }
)
BACKSIDE_REACQUISITION_MODE = "backside"
QR_MODEL_REACQUISITION_MODE = "qr_model"
QR_MODEL_PROPOSAL_SOURCES = frozenset(
    {
        "model_projection",
        "model_refined_head",
        "model_current_frame_refined",
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
    reacquisition_mode: str | None

    @property
    def registered(self) -> bool:
        return self.strict_retry is not None and self.selected is self.strict_retry

    def metadata(self, *, enabled: bool) -> dict[str, object]:
        proposal = self.proposal
        strict_retry = self.strict_retry
        return {
            "enabled": bool(enabled),
            "attempted": proposal is not None,
            "reacquisition_mode": self.reacquisition_mode,
            "primary_estimator_reason": self.evaluations[0].estimate.reason,
            "primary_qr_detected": self.evaluations[0].debug.qr_detected,
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


def _reacquisition_mode(
    primary: HeadRoiEvaluation,
    *,
    tracked_pose: object | None,
) -> str | None:
    """Classify one failed nominal measurement without relaxing its gates."""

    estimate = primary.estimate
    debug = primary.debug
    if (
        not debug.qr_detected
        and tracked_pose is None
        and estimate.source == BACKSIDE_AXIS_SAMPLE_SOURCE
        and estimate.reason in BACKSIDE_REACQUISITION_TRIGGER_REASONS
    ):
        return BACKSIDE_REACQUISITION_MODE
    if (
        estimate.reason in QR_MODEL_REACQUISITION_TRIGGER_REASONS
        and (
            debug.qr_detected
            or estimate.source == "model_projection"
        )
    ):
        # ``model_pose_seed_unavailable`` needs positive QR evidence.  A
        # ``model_projection`` result already proves that a QR or tracked
        # metric pose existed, even when QR detection is intermittent on this
        # particular crop.
        return QR_MODEL_REACQUISITION_MODE
    return None


def _proposal_is_eligible(
    proposal: HeadRoiEvaluation,
    *,
    reacquisition_mode: str,
) -> bool:
    """Keep wide evidence proposal-only until it can seed a strict retry."""

    if proposal.estimate.corners is None:
        return False
    if reacquisition_mode == BACKSIDE_REACQUISITION_MODE:
        return (
            not proposal.debug.qr_detected
            and proposal.estimate.source == BACKSIDE_AXIS_SAMPLE_SOURCE
            and (
                proposal.estimate.usable
                or proposal.estimate.reason
                == "model_backside_target_center_mismatch"
            )
        )
    if reacquisition_mode == QR_MODEL_REACQUISITION_MODE:
        return (
            proposal.debug.qr_detected
            and getattr(proposal.debug, "model_pose", None) is not None
            and proposal.estimate.source in QR_MODEL_PROPOSAL_SOURCES
        )
    raise ValueError(f"unsupported reacquisition mode: {reacquisition_mode}")


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
    reacquisition_mode = _reacquisition_mode(
        primary,
        tracked_pose=tracked_pose,
    )
    if (
        not enable_reacquisition
        or len(roi_attempts) < 2
        or primary.estimate.usable
        or reacquisition_mode is None
    ):
        return CameraTargetRegistrationSelection(
            selected=primary,
            evaluations=tuple(evaluations),
            proposal=None,
            decision=None,
            strict_retry=None,
            reacquisition_mode=None,
        )

    proposal = evaluate(roi_attempts[1], None)
    evaluations.append(proposal)
    eligible = _proposal_is_eligible(
        proposal,
        reacquisition_mode=reacquisition_mode,
    )
    decision = None
    if eligible:
        decision = registered_head_roi_attempt(
            proposal.attempt,
            proposal.estimate.corners,
            max_center_offset_ratio=max_center_offset_ratio,
            registered_source=(
                REGISTERED_QR_MODEL_REACQUISITION_SOURCE
                if reacquisition_mode == QR_MODEL_REACQUISITION_MODE
                else REGISTERED_BACKSIDE_REACQUISITION_SOURCE
            ),
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
            reacquisition_mode=reacquisition_mode,
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
        reacquisition_mode=reacquisition_mode,
    )
