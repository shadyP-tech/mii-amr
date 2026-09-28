"""ROS-free diagnostics and eligibility checks for first sensor delivery."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from scripts.aufgabe04.navigation.waypoint_follower.initial_sensor_acquisition import (
    InitialSensorAcquisition, publisher_diagnostics,
)


class InitialSensorAcquisitionTest(unittest.TestCase):
    def test_only_explicit_never_received_inputs_are_eligible(self):
        for malformed in (False, True):
            with self.subTest(malformed=malformed):
                state = InitialSensorAcquisition()
                for name in ("scan", "odom"):
                    state.record(name, has_message=False, failure="missing message", details={
                        "source": "message_freshness", "sensor": name,
                        **({} if malformed else {"has_message": False}),
                    })
                self.assertEqual(state.waiting_only_for_first_delivery(), not malformed)

    def test_graph_reports_matches_and_qos_with_bounded_endpoints(self):
        qos = SimpleNamespace(reliability=1, durability=2, history=1, depth=10)
        subscription = SimpleNamespace(topic_name="/robot1/odom", qos_profile=qos,
                                       get_publisher_count=Mock(return_value=1))
        node = SimpleNamespace(
            initial_sensor_subscriptions={"odom": subscription},
            get_publishers_info_by_topic=Mock(return_value=[SimpleNamespace(
                node_name=f"publisher_{i}", node_namespace="/robot1", qos_profile=qos,
            ) for i in range(20)]),
        )
        evidence = publisher_diagnostics(node)["odom"]
        self.assertEqual(evidence["matched_publisher_count"], 1)
        self.assertEqual(evidence["discovered_publisher_count"], 20)
        self.assertEqual(len(evidence["publishers"]), 8)
        self.assertEqual(evidence["requested_qos"]["reliability"], 1)
        node.get_publishers_info_by_topic.assert_called_once_with("/robot1/odom")

    def test_graph_error_remains_diagnostic(self):
        subscription = SimpleNamespace(topic_name="/scan", qos_profile=SimpleNamespace(
            reliability=2, durability=2, history=1, depth=5),
            get_publisher_count=Mock(side_effect=RuntimeError("graph unavailable")))
        node = SimpleNamespace(initial_sensor_subscriptions={"scan": subscription})
        self.assertIn("RuntimeError: graph unavailable",
                      publisher_diagnostics(node)["scan"]["diagnostic_error"])


if __name__ == "__main__":
    unittest.main()
