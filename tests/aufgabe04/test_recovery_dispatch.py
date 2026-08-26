from __future__ import annotations

import unittest

from scripts.aufgabe04.navigation.waypoint_follower.directives import (
    BlockageRecoveryAction,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.recovery_dispatch import (
    BlockageRecoveryDisposition,
    BlockageRecoveryTrigger,
    RecoveryLoopAction,
    blockage_recovery_disposition,
    blockage_recovery_eligible,
    front_sector_recovery_evidence,
)


class RecoveryDispatchTest(unittest.TestCase):
    def test_front_sector_evidence_is_extracted_from_stop_details(self):
        evidence = {"source": "front_sector", "minimum_range_m": 0.18}

        self.assertIs(
            front_sector_recovery_evidence(
                {"front_clearance": evidence}
            ),
            evidence,
        )
        self.assertIsNone(front_sector_recovery_evidence(None))
        self.assertIsNone(
            front_sector_recovery_evidence(
                {"front_clearance": "invalid"}
            )
        )
        self.assertIsNone(
            front_sector_recovery_evidence(
                {"front_clearance": {"source": "full_scan"}}
            )
        )

    def test_recovery_requires_provider_forward_motion_and_front_evidence(self):
        admitted = blockage_recovery_eligible(
            provider_available=True,
            nominal_linear_x_mps=0.03,
            front_evidence={"source": "front_sector"},
        )

        self.assertTrue(admitted)
        self.assertFalse(
            blockage_recovery_eligible(
                provider_available=False,
                nominal_linear_x_mps=0.03,
                front_evidence={"source": "front_sector"},
            )
        )
        self.assertFalse(
            blockage_recovery_eligible(
                provider_available=True,
                nominal_linear_x_mps=0.0,
                front_evidence={"source": "front_sector"},
            )
        )
        self.assertFalse(
            blockage_recovery_eligible(
                provider_available=True,
                nominal_linear_x_mps=0.03,
                front_evidence={"source": "omnidirectional_scan"},
            )
        )

    def test_clearance_adopted_or_cleared_gets_zero_hold_retry(self):
        for recovery_action in (
            BlockageRecoveryAction.ADOPTED,
            BlockageRecoveryAction.CLEARED,
        ):
            with self.subTest(recovery_action=recovery_action):
                decision = blockage_recovery_disposition(
                    trigger=BlockageRecoveryTrigger.CLEARANCE_FLOOR,
                    recovery_action=recovery_action,
                    fallback_reason="clearance floor",
                    latest_reason="latest",
                )

                self.assertEqual(
                    decision,
                    BlockageRecoveryDisposition(
                        RecoveryLoopAction.ZERO_HOLD_AND_RETRY
                    ),
                )

    def test_stuck_adopted_gets_hold_retry_without_extra_zero_directive(self):
        decision = blockage_recovery_disposition(
            trigger=BlockageRecoveryTrigger.STUCK_WATCHDOG,
            recovery_action="adopted",
            fallback_reason="stuck no progress",
            latest_reason="latest",
        )

        self.assertEqual(
            decision,
            BlockageRecoveryDisposition(RecoveryLoopAction.HOLD_AND_RETRY),
        )

    def test_obstacle_stop_distinguishes_adopted_and_cleared_retries(self):
        adopted = blockage_recovery_disposition(
            trigger=BlockageRecoveryTrigger.OBSTACLE_SAFETY_STOP,
            recovery_action=BlockageRecoveryAction.ADOPTED,
            fallback_reason="obstacle too close",
            latest_reason="latest",
        )
        cleared = blockage_recovery_disposition(
            trigger=BlockageRecoveryTrigger.OBSTACLE_SAFETY_STOP,
            recovery_action=BlockageRecoveryAction.CLEARED,
            fallback_reason="obstacle too close",
            latest_reason="latest",
        )

        self.assertEqual(adopted.action, RecoveryLoopAction.HOLD_AND_RETRY)
        self.assertEqual(
            cleared.action,
            RecoveryLoopAction.ZERO_HOLD_AND_RETRY,
        )

    def test_clear_scan_does_not_discharge_stuck_watchdog(self):
        decision = blockage_recovery_disposition(
            trigger=BlockageRecoveryTrigger.STUCK_WATCHDOG,
            recovery_action=BlockageRecoveryAction.CLEARED,
            fallback_reason="stuck no progress",
            latest_reason="controller diagnosis",
        )

        self.assertEqual(decision.action, RecoveryLoopAction.STOP)
        self.assertEqual(decision.stop_reason, "controller diagnosis")

    def test_stopped_recovery_uses_latest_reason(self):
        for trigger in BlockageRecoveryTrigger:
            with self.subTest(trigger=trigger):
                decision = blockage_recovery_disposition(
                    trigger=trigger,
                    recovery_action=BlockageRecoveryAction.STOPPED,
                    fallback_reason="fallback",
                    latest_reason="recovery failed closed",
                )

                self.assertEqual(decision.action, RecoveryLoopAction.STOP)
                self.assertEqual(
                    decision.stop_reason,
                    "recovery failed closed",
                )

    def test_not_attempted_and_completed_preserve_fallback_reason(self):
        for recovery_action in (
            BlockageRecoveryAction.NOT_ATTEMPTED,
            BlockageRecoveryAction.COMPLETED,
        ):
            for trigger in BlockageRecoveryTrigger:
                with self.subTest(
                    recovery_action=recovery_action,
                    trigger=trigger,
                ):
                    decision = blockage_recovery_disposition(
                        trigger=trigger,
                        recovery_action=recovery_action,
                        fallback_reason="fallback",
                        latest_reason="latest",
                    )

                    self.assertEqual(
                        decision,
                        BlockageRecoveryDisposition(
                            RecoveryLoopAction.STOP,
                            "fallback",
                        ),
                    )

    def test_unknown_trigger_fails_closed(self):
        decision = blockage_recovery_disposition(
            trigger="unknown",  # type: ignore[arg-type]
            recovery_action=BlockageRecoveryAction.ADOPTED,
            fallback_reason="fallback",
            latest_reason="latest",
        )

        self.assertEqual(
            decision,
            BlockageRecoveryDisposition(
                RecoveryLoopAction.STOP,
                "fallback",
            ),
        )


if __name__ == "__main__":
    unittest.main()
