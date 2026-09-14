from copy import deepcopy
from dataclasses import replace
import unittest

from scripts.aufgabe04.navigation.localization.map_odom_drift_reference import RouteDriftAnchor
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.navigation.localization.odom_route_adapter import (
    OdomExecutionContext,
    evaluate_map_odom_continuity,
)

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


def _anchored_context():
    return OdomExecutionContext(
        map_frame="map", odom_frame="odom", base_frame="base_footprint",
        certificate_sha256="a" * 64,
        frozen_map_from_odom=PlanarTransform2D(0.0, 0.0, 0.0),
        max_map_from_odom_translation_drift_m=0.10,
        max_map_from_odom_yaw_drift_rad=0.03,
        drift_reference=RouteDriftAnchor(2.0, 1.0, 2.0, 1.0),
    )


def _anchored_continuity():
    return evaluate_map_odom_continuity(
        _anchored_context(), PlanarTransform2D(0.20, 0.0, 0.0),
    ).to_evidence()


class RuntimeLocalizationResealTest(unittest.TestCase):
    def test_anchored_stop_preserves_recomputable_evidence_in_recovery_decision(self):
        details = _stop_details()
        details["continuity"] = _anchored_continuity()
        decision = evaluate_runtime_localization_reseal(
            status="stopped", motion_published=True, stop_details=details,
            execution_context=_anchored_context(),
        )
        self.assertTrue(decision.eligible, decision.reason)
        self.assertFalse(decision.automatic_motion_authorized)
        evidence = decision.to_evidence()
        self.assertEqual(evidence["schema_version"], 2)
        self.assertEqual(evidence["continuity_evidence"], details["continuity"])

    def test_anchored_stop_rejects_changed_measurement_anchor_or_schema(self):
        original = _anchored_continuity()
        for field, value in (
            ("translation_drift_m", 0.30),
            ("relative_translation_x_m", 0.30),
            ("live_map_from_odom", {"x_m": 0.30, "y_m": 0.0, "yaw_rad": 0.0}),
            ("reason", "map_from_odom_yaw_drift"),
            ("schema_version", 1), ("schema_version", True),
            ("drift_reference", None),
        ):
            with self.subTest(field=field):
                details = _stop_details()
                details["continuity"] = {**deepcopy(original), field: value}
                self.assertFalse(evaluate_runtime_localization_reseal(
                    status="stopped", motion_published=True, stop_details=details,
                ).eligible)
        details = _stop_details()
        details["continuity"] = deepcopy(original)
        del details["continuity"]["drift_reference"]
        self.assertFalse(evaluate_runtime_localization_reseal(
            status="stopped", motion_published=True, stop_details=details,
        ).eligible)
        # A stripped anchor plus changed version retains the origin diagnostic;
        # that marker must not enter the genuine schema-1 compatibility path.
        details["continuity"]["schema_version"] = 1
        self.assertFalse(evaluate_runtime_localization_reseal(
            status="stopped", motion_published=True, stop_details=details,
        ).eligible)

    def test_anchored_stop_binds_supplied_certificate_context(self):
        details = _stop_details()
        details["continuity"] = _anchored_continuity()
        context = _anchored_context()
        for changed in (
            replace(context, certificate_sha256="b" * 64),
            replace(context, drift_reference=RouteDriftAnchor(3.0, 1.0, 3.0, 1.0)),
            replace(context, drift_reference=None),
            replace(context, max_map_from_odom_translation_drift_m=0.05),
        ):
            with self.subTest(context=changed):
                self.assertFalse(evaluate_runtime_localization_reseal(
                    status="stopped", motion_published=True, stop_details=details,
                    execution_context=changed,
                ).eligible)

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
