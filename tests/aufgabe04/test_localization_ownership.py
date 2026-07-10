import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.localization_ownership import (  # noqa: E402
    FAIL_AMBIGUOUS,
    FAIL_AMCL_STALE,
    FAIL_AMCL_WITH_EXTERNAL_TF,
    FAIL_MAP_TO_ODOM,
    FAIL_ROUTE_TRANSFORM,
    FAIL_TF_WITH_AMCL,
    LocalizationOwnershipEvidence,
    evaluate_localization_ownership,
)


def decide(**overrides):
    values = {
        "localization_source": "amcl",
        "amcl_fresh": True,
        "map_to_odom_dynamic_fresh": True,
        "route_transform_fresh": True,
        "odom_to_base_fresh": True,
        "map_odom_identity": False,
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

    def test_tf_source_accepts_identical_map_and_odom_without_dynamic_tf(self):
        decision = decide(
            localization_source="tf",
            amcl_fresh=True,
            map_to_odom_dynamic_fresh=False,
            map_odom_identity=True,
        )

        self.assertTrue(decision.ok)
        self.assertTrue(decision.data["map_odom_identity"])

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


if __name__ == "__main__":
    unittest.main()
