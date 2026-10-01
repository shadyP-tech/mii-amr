"""Cold executing-listener AMCL refresh; no ROS services or robot effects."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.aufgabe04.navigation.localization.tf_stale_recovery_policy import OdomStationaritySample
from scripts.aufgabe04.navigation.waypoint_follower import runtime as follower
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components import initial_runtime_inputs
from tests.aufgabe04 import test_initial_tf_acquisition as fixtures


class InitialAmclRefreshTest(unittest.TestCase):
    def setup_case(self, *, map_delay=.25, response_delay=.25):
        fixture = fixtures.InitialTfAcquisitionTest()
        node, clock = fixture.make_sampled_node(map_at=None)
        node.runtime_config.localization_source = "amcl"
        node.runtime_config.use_sim_time = False
        node.runtime_nomotion_update_service = "/request_nomotion_update"
        requested = []
        future = Mock()
        future.done.side_effect = lambda: bool(requested) and response_delay is not None and clock.now >= requested[0]+response_delay
        future.exception.return_value = None
        future.result.return_value = object()
        node.runtime_nomotion_update_client = Mock()
        node.runtime_nomotion_update_client.service_is_ready.return_value = True
        node.runtime_nomotion_update_client.call_async.side_effect = lambda request: (requested.append(clock.now), future)[1]
        node._cmd_vel_ownership_failure = Mock(return_value="")
        node._odom_stationarity_sample = Mock(side_effect=lambda: OdomStationaritySample(
            callback_count=round(clock.now*100), stamp_sec=100.+clock.now,
            x_m=0., y_m=0., yaw_rad=0., linear_x_mps=0., angular_z_radps=0.))
        original_lookup = node.tf_buffer.lookup_transform.side_effect

        def lookup(target, source, *args, **kwargs):
            if target != "map":
                return original_lookup(target, source, *args, **kwargs)
            if not requested or map_delay is None or clock.now < requested[0]+map_delay:
                raise fixtures.LookupException('"map" passed to lookupTransform target_frame does not exist.')
            return fixture.transform(target, source, node.odom_execution_context.frozen_map_from_odom, 100.+clock.now)

        node.tf_buffer.lookup_transform.side_effect = lookup
        return fixture, node, clock, future, requested

    def run_case(self, fixture, node, clock):
        with patch.object(follower, "Empty", SimpleNamespace(Request=object)):
            return fixture.wait(node, clock)

    def test_missing_map_is_refreshed_once_in_same_listener_before_motion(self):
        fixture, node, clock, future, requested = self.setup_case()
        original_buffer, original_context = node.tf_buffer, node.odom_execution_context
        self.assertEqual(self.run_case(fixture, node, clock), "")
        self.assertEqual(requested, [2.25])
        self.assertEqual(clock.now, 2.75)
        self.assertIs(node.tf_buffer, original_buffer)
        self.assertIs(node.odom_execution_context, original_context)
        self.assertFalse(node.motion_published)
        self.assertIsNone(node.latest_stop_details)
        evidence = node.latest_initial_tf_acquisition["initial_amcl_refresh"]
        self.assertTrue(evidence["service_completed"])
        self.assertEqual(evidence["request_count"], 1)
        self.assertEqual(evidence["status"], "fresh_tf_and_continuity_observed")
        self.assertEqual(evidence["absolute_deadline_monotonic_sec"], 5.)
        future.cancel.assert_not_called()

    def test_acknowledgement_without_map_never_unlocks_and_keeps_original_deadline(self):
        fixture, node, clock, future, requested = self.setup_case(map_delay=None)
        self.assertEqual(self.run_case(fixture, node, clock), "TF transform unavailable: map <- odom")
        self.assertEqual(clock.now, 5.)
        self.assertEqual(len(requested), 1)
        self.assertTrue(node.latest_initial_tf_acquisition["deadline_exhausted"])
        self.assertFalse(node.motion_published)
        self.assertTrue(node.latest_initial_tf_acquisition["initial_amcl_refresh"]["service_completed"])

    def test_fresh_map_during_pending_service_still_holds_until_post_ack_lookup(self):
        fixture, node, clock, future, requested = self.setup_case(map_delay=.25, response_delay=.75)
        self.assertEqual(self.run_case(fixture, node, clock), "")
        self.assertEqual(clock.now, 3.25)
        self.assertFalse(node.motion_published)
        self.assertEqual(len(requested), 1)

    def test_done_response_first_observed_at_service_deadline_is_rejected(self):
        fixture, node, clock, future, requested = self.setup_case(map_delay=.25, response_delay=2.)
        self.assertIn("service_timeout", self.run_case(fixture, node, clock))
        self.assertEqual(clock.now, 4.25)
        future.result.assert_not_called()
        self.assertEqual(len(requested), 1)

    def test_map_arriving_after_absolute_deadline_cannot_unlock(self):
        fixture, node, clock, future, requested = self.setup_case(map_delay=2.75)
        self.assertTrue(self.run_case(fixture, node, clock))
        self.assertEqual(clock.now, 5.)
        self.assertFalse(node.motion_published)

    def test_timeout_or_service_error_fails_closed_and_pending_future_is_cancelled(self):
        for response_delay in (None, .25):
            with self.subTest(response_delay=response_delay):
                fixture, node, clock, future, requested = self.setup_case(response_delay=response_delay)
                if response_delay is not None:
                    future.exception.return_value = RuntimeError("service error")
                self.assertTrue(self.run_case(fixture, node, clock))
                self.assertEqual(node.latest_stop_details["source"], "initial_amcl_refresh")
                self.assertFalse(node.motion_published)
                self.assertEqual(len(requested), 1)
                if response_delay is None:
                    future.cancel.assert_called_once()
                    self.assertTrue(node.latest_stop_details["initial_amcl_refresh"]["pending_future_cancelled"])

    def test_new_map_outside_frozen_transform_bounds_keeps_continuity_rejection(self):
        fixture, node, clock, future, requested = self.setup_case()
        lookup = node.tf_buffer.lookup_transform.side_effect

        def shifted(target, *args, **kwargs):
            transform = lookup(target, *args, **kwargs)
            if target == "map":
                transform.transform.translation.x += .3
            return transform

        node.tf_buffer.lookup_transform.side_effect = shifted
        self.assertEqual(self.run_case(fixture, node, clock), "global localization consistency requires zero and reseal")
        self.assertEqual(node.latest_stop_details["source"], "global_consistency_monitor")
        self.assertEqual(len(requested), 1)
        self.assertFalse(node.motion_published)

    def test_nonphysical_or_non_amcl_runtime_never_requests_service(self):
        for attribute, value in (("use_sim_time", True), ("localization_source", "slam")):
            with self.subTest(attribute=attribute):
                fixture, node, clock, future, requested = self.setup_case()
                setattr(node.runtime_config, attribute, value)
                self.assertTrue(self.run_case(fixture, node, clock))
                self.assertEqual(requested, [])
                self.assertNotIn("initial_amcl_refresh", node.latest_initial_tf_acquisition)

    def test_ownership_failure_or_moving_odom_blocks_request(self):
        for moving in (False, True):
            with self.subTest(moving=moving):
                fixture, node, clock, future, requested = self.setup_case()
                if moving:
                    node._odom_stationarity_sample.side_effect = lambda: OdomStationaritySample(
                        callback_count=round(clock.now*100), stamp_sec=100.+clock.now,
                        x_m=0., y_m=0., yaw_rad=0., linear_x_mps=.05, angular_z_radps=0.)
                else:
                    node._cmd_vel_ownership_failure.return_value = "another velocity owner"
                self.assertTrue(self.run_case(fixture, node, clock))
                self.assertEqual(requested, [])
                self.assertFalse(node.motion_published)

    def test_stale_sensor_after_request_keeps_original_failure_and_cancels_future(self):
        fixture, node, clock, future, requested = self.setup_case(map_delay=None, response_delay=None)

        def freshness(name, *args):
            if requested and name == "scan":
                node.latest_stop_details = {"source": "message_freshness", "reason": "stale scan", "sensor": "scan", "has_message": True}
                return "stale scan"
            return ""

        node._freshness_failure.side_effect = freshness
        self.assertEqual(self.run_case(fixture, node, clock), "stale scan")
        self.assertEqual(node.latest_stop_details["source"], "message_freshness")
        future.cancel.assert_called_once()
        self.assertEqual(len(requested), 1)

    def test_naturally_arriving_tf_before_request_does_not_call_service(self):
        fixture, node, clock, future, requested = self.setup_case()
        original = node.tf_buffer.lookup_transform.side_effect

        def appears(target, source, *args, **kwargs):
            if target == "map" and clock.now >= 2.25:
                return fixture.transform(target, source, node.odom_execution_context.frozen_map_from_odom, 100.+clock.now)
            return original(target, source, *args, **kwargs)

        node.tf_buffer.lookup_transform.side_effect = appears
        self.assertEqual(self.run_case(fixture, node, clock), "")
        self.assertEqual(requested, [])
        self.assertEqual(clock.now, 2.25)

    def test_invalid_first_map_and_established_edge_loss_never_request(self):
        for mode in ("future", "wrong_frame", "established_then_lost"):
            with self.subTest(mode=mode):
                fixture, node, clock, future, requested = self.setup_case()
                original = node.tf_buffer.lookup_transform.side_effect

                def malformed(target, source, *args, **kwargs):
                    if mode == "established_then_lost" and target == "odom":
                        raise fixtures.LookupException("execution tree is still cold")
                    if target != "map":
                        return original(target, source, *args, **kwargs)
                    if mode == "established_then_lost" and clock.now >= .5:
                        raise fixtures.LookupException("established map edge disappeared")
                    transform = fixture.transform(target, source,
                        node.odom_execution_context.frozen_map_from_odom,
                        100.+clock.now+(2. if mode == "future" else 0.))
                    if mode == "wrong_frame":
                        transform.header.frame_id = "wrong_map"
                    return transform

                node.tf_buffer.lookup_transform.side_effect = malformed
                self.assertTrue(self.run_case(fixture, node, clock))
                self.assertEqual(requested, [])
                self.assertFalse(node.motion_published)

    def test_executor_failure_blocks_request(self):
        fixture, node, clock, future, requested = self.setup_case()
        node.initial_tf_executor_health_probe.return_value = {"ready": False}
        self.assertTrue(self.run_case(fixture, node, clock))
        self.assertEqual(requested, [])

    def test_global_edge_lost_while_service_pending_stops_without_second_request(self):
        fixture, node, clock, future, requested = self.setup_case(response_delay=.75)
        original = node.tf_buffer.lookup_transform.side_effect

        def lost_after_first_sample(target, source, *args, **kwargs):
            if target == "map" and clock.now >= 2.75:
                raise fixtures.LookupException("established map edge disappeared")
            return original(target, source, *args, **kwargs)

        node.tf_buffer.lookup_transform.side_effect = lost_after_first_sample
        self.assertTrue(self.run_case(fixture, node, clock))
        self.assertEqual(clock.now, 2.75)
        self.assertEqual(node.latest_initial_tf_acquisition["denial_reason"], "required_tf_edge_already_acquired")
        self.assertEqual(len(requested), 1)
        future.cancel.assert_called_once()

    def test_ros_shutdown_cancels_pending_future(self):
        fixture, node, clock, future, requested = self.setup_case(map_delay=None, response_delay=None)
        with patch.object(initial_runtime_inputs, "rclpy", SimpleNamespace(
                ok=lambda: not requested or clock.now < 2.5)):
            self.assertEqual(self.run_case(fixture, node, clock), "ROS shutdown")
        future.cancel.assert_called_once()
        self.assertEqual(len(requested), 1)


if __name__ == "__main__":
    unittest.main()
