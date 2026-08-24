"""Bounded before-motion localization-readiness retry transition."""

from __future__ import annotations

from scripts.aufgabe04.real_robot.autonomous_child_runner import MotionLegOutcome
from scripts.aufgabe04.real_robot.autonomous_localization_readiness import (
    evaluate_localization_readiness_retry,
)

class ReadinessRecoveryMixin:
    def _handle_localization_readiness_retry(
        self,
        outcome: MotionLegOutcome,
    ) -> bool:
        readiness_decision = evaluate_localization_readiness_retry(
            status=outcome.status,
            stop_reason=outcome.stop_reason,
            stop_details=outcome.stop_details,
            motion_published=outcome.motion_published,
        )
        if readiness_decision.retryable:
            maximum_readiness_retries = (
                self.config.max_localization_readiness_retries_per_leg
            )
            if self.localization_readiness_retry_index >= maximum_readiness_retries:
                self.emit(
                    {
                        "schema_version": 1,
                        "event": "localization_readiness_retry_exhausted",
                        "timestamp": self.effects.clock(),
                        "leg_index": self.leg_index,
                        "target_viewpoint_id": self.target_viewpoint_id,
                        "rejected_run_id": outcome.run_id,
                        "completed_retry_count": self.localization_readiness_retry_index,
                        "maximum_retry_count": maximum_readiness_retries,
                        "stop_reason": outcome.stop_reason,
                        "motion_published": False,
                        "motion_continues_authorized": False,
                        "fail_closed": True,
                    }
                )
                raise RuntimeError(
                    "pre-motion localization readiness retry budget exhausted "
                    f"for coverage leg {self.leg_index}: {outcome.stop_reason}"
                )
            self.localization_readiness_retry_index += 1
            self.emit(
                {
                    "schema_version": 1,
                    "event": "localization_readiness_retry_scheduled",
                    "timestamp": self.effects.clock(),
                    "leg_index": self.leg_index,
                    "target_viewpoint_id": self.target_viewpoint_id,
                    "rejected_run_id": outcome.run_id,
                    "next_retry_index": self.localization_readiness_retry_index,
                    "maximum_retry_count": maximum_readiness_retries,
                    "reason": readiness_decision.reason,
                    "stop_reason": outcome.stop_reason,
                    "motion_published": False,
                    "motion_continues_authorized": False,
                    "fresh_nomotion_amcl_preflight_required": True,
                    "route_limits_unchanged": True,
                }
            )
            return True
        return False
