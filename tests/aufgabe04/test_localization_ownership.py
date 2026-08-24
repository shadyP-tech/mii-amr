import sys
import unittest
from dataclasses import FrozenInstanceError, fields
from inspect import signature
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.localization.localization_ownership import (  # noqa: E402
    FAIL_AMBIGUOUS,
    FAIL_AMCL_STALE,
    FAIL_AMCL_WITH_EXTERNAL_TF,
    FAIL_EXTERNAL_MAP_TO_ODOM_OWNER,
    FAIL_FROZEN_MAP_TRANSFORM_CERTIFICATE,
    FAIL_MAP_TO_ODOM,
    FAIL_MONITOR_STALE,
    FAIL_ODOM_TO_BASE,
    FAIL_ROUTE_TRANSFORM,
    FAIL_TF_WITH_AMCL,
    MONITOR_ACTION_FORCE_ZERO_RESEAL,
    MONITOR_ACTION_LOG,
    MONITOR_ACTION_PASS,
    MONITOR_REASON_RESEAL_REQUIRED,
    MONITOR_REASON_UNCERTAINTY_BUDGET_EXHAUSTED,
    LocalizationMonitorDecision,
    LocalizationOwnershipEvidence,
    evaluate_global_consistency_monitor,
    evaluate_localization_ownership,
)


def decide(**overrides):
    values = {
        "localization_source": "amcl",
        "amcl_fresh": True,
        "map_to_odom_dynamic_fresh": True,
        "route_transform_fresh": True,
        "odom_to_base_fresh": True,
        "route_uses_odom_frame": False,
        "external_tf_owner_candidates": (),
        "ambiguous_owner_evidence": (),
    }
    values.update(overrides)
    return evaluate_localization_ownership(LocalizationOwnershipEvidence(**values))


class LocalizationOwnershipTest(unittest.TestCase):
    def test_amcl_with_fresh_amcl_and_dynamic_map_to_odom_is_ok(self):
        decision = decide()

        self.assertTrue(decision.ok)
        self.assertEqual(decision.failure, "")
        self.assertEqual(decision.data["execution_pose_owner"], "amcl")
        self.assertEqual(decision.data["global_consistency_monitor"], "none")
        self.assertEqual(
            decision.data["execution_pose_owner_action"],
            "provide_execution_pose",
        )

    def test_amcl_requires_dynamic_map_to_odom(self):
        decision = decide(map_to_odom_dynamic_fresh=False)

        self.assertFalse(decision.ok)
        self.assertEqual(decision.failure, FAIL_MAP_TO_ODOM)

    def test_static_only_map_to_odom_fails_as_missing_dynamic_tf(self):
        decision = decide(map_to_odom_dynamic_fresh=False)

        self.assertFalse(decision.ok)
        self.assertEqual(decision.failure, FAIL_MAP_TO_ODOM)

    def test_amcl_requires_fresh_amcl(self):
        decision = decide(amcl_fresh=False)

        self.assertFalse(decision.ok)
        self.assertEqual(decision.failure, FAIL_AMCL_STALE)

    def test_tf_source_with_dynamic_map_to_odom_and_no_amcl_is_ok(self):
        decision = decide(localization_source="tf", amcl_fresh=False)

        self.assertTrue(decision.ok)

    def test_tf_source_allows_odom_frame_route_without_map_to_odom(self):
        decision = decide(
            localization_source="tf",
            amcl_fresh=False,
            map_to_odom_dynamic_fresh=False,
            route_uses_odom_frame=True,
        )

        self.assertTrue(decision.ok)
        self.assertTrue(decision.data["route_uses_odom_frame"])

    def test_amcl_source_still_requires_dynamic_map_to_odom_for_map_route(self):
        decision = decide(
            localization_source="amcl",
            amcl_fresh=True,
            map_to_odom_dynamic_fresh=False,
            route_uses_odom_frame=False,
        )

        self.assertFalse(decision.ok)
        self.assertEqual(decision.failure, FAIL_MAP_TO_ODOM)

    def test_tf_source_fails_when_fresh_amcl_is_present(self):
        decision = decide(localization_source="tf", amcl_fresh=True)

        self.assertFalse(decision.ok)
        self.assertEqual(decision.failure, FAIL_TF_WITH_AMCL)

    def test_any_source_requires_route_transform(self):
        decision = decide(route_transform_fresh=False)

        self.assertFalse(decision.ok)
        self.assertEqual(decision.failure, FAIL_ROUTE_TRANSFORM)

    def test_amcl_fails_when_namespace_scoped_external_tf_owner_is_active(self):
        decision = decide(external_tf_owner_candidates=("/robot1/slam_toolbox",))

        self.assertFalse(decision.ok)
        self.assertEqual(decision.failure, FAIL_AMCL_WITH_EXTERNAL_TF)
        self.assertEqual(decision.data["external_tf_owner_candidates"], ["/robot1/slam_toolbox"])

    def test_other_robot_owner_evidence_is_filtered_before_decision(self):
        decision = decide(external_tf_owner_candidates=())

        self.assertTrue(decision.ok)
        self.assertEqual(decision.data["external_tf_owner_candidates"], [])

    def test_ambiguous_owner_evidence_fails_closed(self):
        decision = decide(ambiguous_owner_evidence=("multiple dynamic map->odom candidates",))

        self.assertFalse(decision.ok)
        self.assertEqual(decision.failure, FAIL_AMBIGUOUS)

    def test_odom_execution_with_fresh_amcl_monitor_is_ok(self):
        decision = decide(
            localization_source="tf",
            execution_pose_owner="odom",
            global_consistency_monitor="amcl",
            route_uses_odom_frame=True,
            map_to_odom_dynamic_fresh=False,
            frozen_map_transform_certified=True,
            amcl_fresh=True,
            odom_to_base_fresh=True,
        )

        self.assertTrue(decision.ok)
        self.assertEqual(decision.failure, "")
        self.assertEqual(decision.data["execution_pose_owner"], "odom")
        self.assertEqual(decision.data["global_consistency_monitor"], "amcl")
        self.assertEqual(
            decision.data["global_consistency_monitor_action"],
            "pass_log_or_force_zero_reseal_only",
        )
        self.assertEqual(
            decision.data["global_consistency_monitor_allowed_actions"],
            ["PASS", "LOG", "FORCE_ZERO_RESEAL"],
        )

    def test_odom_execution_requires_frozen_map_transform_certificate(self):
        decision = decide(
            localization_source="tf",
            execution_pose_owner="odom",
            global_consistency_monitor="amcl",
            route_uses_odom_frame=True,
            frozen_map_transform_certified=False,
        )

        self.assertFalse(decision.ok)
        self.assertEqual(decision.failure, FAIL_FROZEN_MAP_TRANSFORM_CERTIFICATE)

    def test_odom_execution_requires_fresh_odom_to_base(self):
        decision = decide(
            localization_source="tf",
            execution_pose_owner="odom",
            global_consistency_monitor="amcl",
            route_uses_odom_frame=True,
            frozen_map_transform_certified=True,
            odom_to_base_fresh=False,
        )

        self.assertFalse(decision.ok)
        self.assertEqual(decision.failure, FAIL_ODOM_TO_BASE)

    def test_required_amcl_monitor_must_be_fresh(self):
        decision = decide(
            localization_source="tf",
            execution_pose_owner="odom",
            global_consistency_monitor="amcl",
            route_uses_odom_frame=True,
            frozen_map_transform_certified=True,
            amcl_fresh=False,
        )

        self.assertFalse(decision.ok)
        self.assertEqual(decision.failure, FAIL_MONITOR_STALE)

    def test_odom_execution_rejects_external_map_to_odom_owner(self):
        decision = decide(
            localization_source="tf",
            execution_pose_owner="odom",
            global_consistency_monitor="amcl",
            route_uses_odom_frame=True,
            frozen_map_transform_certified=True,
            external_tf_owner_candidates=("/robot1/slam_toolbox",),
        )

        self.assertFalse(decision.ok)
        self.assertEqual(decision.failure, FAIL_EXTERNAL_MAP_TO_ODOM_OWNER)

    def test_odom_execution_rejects_ambiguous_map_to_odom_owner(self):
        decision = decide(
            localization_source="tf",
            execution_pose_owner="odom",
            global_consistency_monitor="amcl",
            route_uses_odom_frame=True,
            frozen_map_transform_certified=True,
            ambiguous_owner_evidence=("multiple dynamic map->odom candidates",),
        )

        self.assertFalse(decision.ok)
        self.assertEqual(decision.failure, FAIL_AMBIGUOUS)


class LocalizationMonitorDecisionTest(unittest.TestCase):
    def test_clean_monitor_passes(self):
        decision = evaluate_global_consistency_monitor()

        self.assertEqual(decision.action, MONITOR_ACTION_PASS)
        self.assertEqual(decision.reason, "")

    def test_warning_only_monitor_logs(self):
        decision = evaluate_global_consistency_monitor(
            diagnostic_warning="global residual increased",
        )

        self.assertEqual(decision.action, MONITOR_ACTION_LOG)
        self.assertEqual(decision.reason, "")
        self.assertEqual(decision.diagnostic_warning, "global residual increased")

    def test_exhausted_uncertainty_budget_forces_zero_and_reseal(self):
        decision = evaluate_global_consistency_monitor(
            uncertainty_budget_exhausted=True,
        )

        self.assertEqual(decision.action, MONITOR_ACTION_FORCE_ZERO_RESEAL)
        self.assertEqual(
            decision.reason,
            MONITOR_REASON_UNCERTAINTY_BUDGET_EXHAUSTED,
        )

    def test_reseal_reason_takes_precedence(self):
        decision = evaluate_global_consistency_monitor(
            uncertainty_budget_exhausted=True,
            reseal_required=True,
        )

        self.assertEqual(decision.action, MONITOR_ACTION_FORCE_ZERO_RESEAL)
        self.assertEqual(decision.reason, MONITOR_REASON_RESEAL_REQUIRED)

    def test_monitor_decision_is_immutable_and_has_no_control_surface(self):
        decision = evaluate_global_consistency_monitor()

        with self.assertRaises(FrozenInstanceError):
            decision.action = MONITOR_ACTION_LOG

        forbidden_terms = ("waypoint", "pursuit", "velocity", "mutation")
        field_names = {item.name for item in fields(LocalizationMonitorDecision)}
        parameter_names = set(signature(evaluate_global_consistency_monitor).parameters)
        for forbidden in forbidden_terms:
            self.assertFalse(any(forbidden in name for name in field_names))
            self.assertFalse(any(forbidden in name for name in parameter_names))

        self.assertEqual(
            {
                MONITOR_ACTION_PASS,
                MONITOR_ACTION_LOG,
                MONITOR_ACTION_FORCE_ZERO_RESEAL,
            },
            {"PASS", "LOG", "FORCE_ZERO_RESEAL"},
        )

        with self.assertRaisesRegex(
            ValueError,
            "unsupported localization monitor action",
        ):
            LocalizationMonitorDecision(action="STEER")


if __name__ == "__main__":
    unittest.main()
