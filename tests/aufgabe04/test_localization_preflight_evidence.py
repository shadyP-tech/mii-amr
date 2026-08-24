import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.localization.localization_ownership import (  # noqa: E402
    FAIL_MAP_TO_ODOM,
    LocalizationOwnershipEvidence,
    evaluate_localization_ownership,
)
from scripts.aufgabe04.navigation.localization.localization_preflight_evidence import (  # noqa: E402
    build_dynamic_map_to_odom_freshness,
    build_localization_ownership_observation_data,
    find_external_tf_owner_candidates,
)
from scripts.aufgabe04.navigation.foundation.ros_runtime_config import (  # noqa: E402
    RuntimeConfig,
    resolve_runtime_config,
)


def resolved_namespace(namespace: str) -> str:
    return resolve_runtime_config(RuntimeConfig(namespace=namespace)).namespace


class LocalizationPreflightEvidenceTest(unittest.TestCase):
    def test_root_namespace_rejects_namespaced_slam_toolbox_evidence(self):
        candidates = find_external_tf_owner_candidates(
            resolved_namespace=resolved_namespace(""),
            node_items=(("slam_toolbox", "/robot1"),),
            topic_names=("/robot1/slam_toolbox/transition_event",),
            service_names=("/robot1/slam_toolbox/get_state",),
        )

        self.assertEqual(candidates, [])

    def test_root_namespace_accepts_root_slam_toolbox_evidence(self):
        candidates = find_external_tf_owner_candidates(
            resolved_namespace=resolved_namespace("/"),
            node_items=(("slam_toolbox", "/"),),
            topic_names=("/slam_toolbox/transition_event",),
            service_names=("/slam_toolbox/get_state",),
        )

        self.assertEqual(
            candidates,
            [
                "/slam_toolbox",
                "/slam_toolbox/get_state",
                "/slam_toolbox/transition_event",
            ],
        )

    def test_matching_namespace_accepts_node_topic_and_service_evidence(self):
        candidates = find_external_tf_owner_candidates(
            resolved_namespace=resolved_namespace("robot1"),
            node_items=(("slam_toolbox", "/robot1"),),
            topic_names=("/robot1/slam_toolbox/transition_event",),
            service_names=("/robot1/slam_toolbox/get_state",),
        )

        self.assertEqual(
            candidates,
            [
                "/robot1/slam_toolbox",
                "/robot1/slam_toolbox/get_state",
                "/robot1/slam_toolbox/transition_event",
            ],
        )

    def test_other_namespace_rejects_slam_toolbox_evidence(self):
        candidates = find_external_tf_owner_candidates(
            resolved_namespace=resolved_namespace("/robot2"),
            node_items=(("slam_toolbox", "/robot1"),),
            topic_names=("/robot1/slam_toolbox/transition_event",),
            service_names=("/robot1/slam_toolbox/get_state",),
        )

        self.assertEqual(candidates, [])

    def test_candidates_are_sorted_and_deduplicated(self):
        candidates = find_external_tf_owner_candidates(
            resolved_namespace=resolved_namespace("/robot1"),
            node_items=(("slam_toolbox", "/robot1"), ("slam_toolbox", "/robot1/")),
            topic_names=(
                "/robot1/slam_toolbox/transition_event",
                "/robot1/slam_toolbox/transition_event",
            ),
            service_names=("/robot1/slam_toolbox/get_state",),
        )

        self.assertEqual(
            candidates,
            [
                "/robot1/slam_toolbox",
                "/robot1/slam_toolbox/get_state",
                "/robot1/slam_toolbox/transition_event",
            ],
        )

    def test_localization_ownership_payload_contains_expected_observation_shape(self):
        data = build_localization_ownership_observation_data(
            decision_data={
                "localization_source": "amcl",
                "external_tf_owner_candidates": ["/robot1/slam_toolbox"],
            },
            map_frame="map",
            odom_frame="odom",
            base_frame="base_footprint",
            amcl_topic="/robot1/amcl_pose",
            dynamic_tf_topics=("/tf", "/robot1/tf"),
            amcl_data={"received": True},
            map_to_odom_dynamic_data={"available": True, "dynamic": True},
            route_transform_data={"available": True},
            odom_to_base_data={"available": True},
        )

        for key in (
            "localization_source",
            "map_frame",
            "odom_frame",
            "base_frame",
            "amcl",
            "map_to_odom_dynamic",
            "route_transform",
            "external_tf_owner_candidates",
        ):
            self.assertIn(key, data)
        self.assertEqual(data["dynamic_tf_topics"], ["/tf", "/robot1/tf"])

    def test_static_only_map_to_odom_does_not_satisfy_dynamic_tf_evidence(self):
        map_to_odom_ok, map_to_odom_data = build_dynamic_map_to_odom_freshness(
            has_dynamic_transform=False,
            receipt_age_sec=None,
            header_age_sec=None,
            max_age_sec=0.5,
        )
        route_transform_data = {"available": True, "age_sec": 0.1}
        decision = evaluate_localization_ownership(
            LocalizationOwnershipEvidence(
                localization_source="amcl",
                amcl_fresh=True,
                map_to_odom_dynamic_fresh=map_to_odom_ok,
                route_transform_fresh=bool(route_transform_data["available"]),
                odom_to_base_fresh=True,
                external_tf_owner_candidates=(),
            )
        )
        payload = build_localization_ownership_observation_data(
            decision_data=decision.data,
            map_frame="map",
            odom_frame="odom",
            base_frame="base_footprint",
            amcl_topic="/amcl_pose",
            dynamic_tf_topics=("/tf",),
            amcl_data={"received": True},
            map_to_odom_dynamic_data=map_to_odom_data,
            route_transform_data=route_transform_data,
            odom_to_base_data={"available": True},
        )

        self.assertFalse(map_to_odom_ok)
        self.assertFalse(decision.ok)
        self.assertEqual(decision.failure, FAIL_MAP_TO_ODOM)
        self.assertTrue(payload["route_transform"]["available"])
        self.assertFalse(payload["map_to_odom_dynamic"]["available"])
        self.assertFalse(payload["map_to_odom_dynamic"]["dynamic"])

    def test_dynamic_map_to_odom_requires_receipt_and_header_freshness(self):
        ok, data = build_dynamic_map_to_odom_freshness(
            has_dynamic_transform=True,
            receipt_age_sec=0.1,
            header_age_sec=0.9,
            max_age_sec=0.5,
        )

        self.assertFalse(ok)
        self.assertTrue(data["available"])
        self.assertTrue(data["dynamic"])

    def test_dynamic_map_to_odom_rejects_future_timestamp(self):
        ok, data = build_dynamic_map_to_odom_freshness(
            has_dynamic_transform=True,
            receipt_age_sec=0.1,
            header_age_sec=-0.8,
            max_age_sec=0.5,
            max_future_sec=0.25,
        )

        self.assertFalse(ok)
        self.assertTrue(data["future_dated"])


if __name__ == "__main__":
    unittest.main()
