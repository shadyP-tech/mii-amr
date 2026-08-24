import unittest

from scripts.aufgabe04.navigation.localization.runtime_localization_reseal import (
    evaluate_runtime_localization_reseal,
    evaluate_runtime_localization_reseal_budget,
)


def _stop_details():
    return {
        "fault_code": "localization_reseal_required",
        "source": "global_consistency_monitor",
        "execution_pose_owner": "odom",
        "global_consistency_monitor": "amcl",
        "monitor_action": "FORCE_ZERO_RESEAL",
        "fail_closed": True,
        "continuity": {
            "accepted": False,
            "requires_zero_cycle": True,
            "requires_reseal": True,
            "decision": "force_zero_reseal",
            "reason": "map_from_odom_yaw_drift",
            "fail_closed": True,
        },
    }


class RuntimeLocalizationResealTest(unittest.TestCase):
    def test_complete_post_motion_stop_is_eligible_but_never_authorizes_motion(self):
        decision = evaluate_runtime_localization_reseal(
            status="stopped",
            motion_published=True,
            stop_details=_stop_details(),
        )

        self.assertTrue(decision.eligible)
        self.assertEqual(decision.execution_phase, "after_motion")
        self.assertTrue(decision.requires_fresh_localization)
        self.assertTrue(decision.requires_new_route_certificate)
        self.assertTrue(decision.requires_fresh_typed_run)
        self.assertFalse(decision.automatic_motion_authorized)
        self.assertFalse(decision.to_evidence()["automatic_motion_authorized"])

    def test_complete_pre_motion_runtime_stop_is_not_reclassified(self):
        decision = evaluate_runtime_localization_reseal(
            status="stopped",
            motion_published=False,
            stop_details=_stop_details(),
        )

        self.assertFalse(decision.eligible)
        self.assertEqual(decision.reason, "motion_not_published")
        self.assertEqual(decision.execution_phase, "not_admitted")

    def test_wrong_status_and_malformed_top_level_evidence_are_ineligible(self):
        cases = (
            ("completed", True, _stop_details(), "outcome_not_stopped"),
            ("stopped", 1, _stop_details(), "motion_published_not_boolean"),
            ("stopped", True, None, "stop_details_not_mapping"),
        )
        for status, motion, details, reason in cases:
            with self.subTest(reason=reason):
                decision = evaluate_runtime_localization_reseal(
                    status=status,
                    motion_published=motion,
                    stop_details=details,
                )
                self.assertFalse(decision.eligible)
                self.assertEqual(decision.reason, reason)

    def test_every_required_stop_field_fails_closed_when_changed(self):
        cases = {
            "fault_code": "other",
            "source": "other",
            "execution_pose_owner": "map",
            "global_consistency_monitor": "none",
            "monitor_action": "LOG",
            "fail_closed": False,
        }
        for field, replacement in cases.items():
            with self.subTest(field=field):
                details = _stop_details()
                details[field] = replacement
                decision = evaluate_runtime_localization_reseal(
                    status="stopped",
                    motion_published=True,
                    stop_details=details,
                )
                self.assertFalse(decision.eligible)
                self.assertEqual(decision.reason, f"invalid_{field}")

    def test_every_required_continuity_field_fails_closed_when_changed(self):
        cases = {
            "accepted": True,
            "requires_zero_cycle": False,
            "requires_reseal": False,
            "decision": "continue_odom_execution",
            "reason": "",
            "fail_closed": False,
        }
        for field, replacement in cases.items():
            with self.subTest(field=field):
                details = _stop_details()
                details["continuity"][field] = replacement
                decision = evaluate_runtime_localization_reseal(
                    status="stopped",
                    motion_published=True,
                    stop_details=details,
                )
                self.assertFalse(decision.eligible)

    def test_continuity_must_be_a_mapping(self):
        details = _stop_details()
        details["continuity"] = []
        decision = evaluate_runtime_localization_reseal(
            status="stopped",
            motion_published=True,
            stop_details=details,
        )
        self.assertFalse(decision.eligible)
        self.assertEqual(decision.reason, "continuity_not_mapping")

    def test_budget_is_strictly_bounded(self):
        available = evaluate_runtime_localization_reseal_budget(
            completed_reseal_count=0,
            maximum_reseal_count=1,
        )
        exhausted = evaluate_runtime_localization_reseal_budget(
            completed_reseal_count=1,
            maximum_reseal_count=1,
        )

        self.assertTrue(available.allowed)
        self.assertEqual(available.next_reseal_index, 1)
        self.assertFalse(available.automatic_motion_authorized)
        self.assertFalse(exhausted.allowed)
        self.assertIsNone(exhausted.next_reseal_index)

    def test_budget_arguments_must_be_nonnegative_integers(self):
        for completed, maximum in ((-1, 1), (0, -1), (True, 1), (0, 1.5)):
            with self.subTest(completed=completed, maximum=maximum):
                with self.assertRaises(ValueError):
                    evaluate_runtime_localization_reseal_budget(
                        completed_reseal_count=completed,
                        maximum_reseal_count=maximum,
                    )


if __name__ == "__main__":
    unittest.main()
