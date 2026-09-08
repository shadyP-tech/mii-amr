"""Stopped follower TF acquisition regressions; no ROS or wall-clock sleeps."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.aufgabe04.navigation.control.waypoint_controller import ControllerConfig
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.waypoint_follower import runtime as follower
from scripts.aufgabe04.navigation.waypoint_follower.initial_tf_acquisition import (
    InitialTfAcquisition,
    TfExecutorHeartbeat,
)
from scripts.aufgabe04.navigation.waypoint_follower.runtime_components import initial_runtime_inputs


def _cold_failure(exception_type="ConnectivityException", **overrides):
    # Recorded terminal failure shape from candidate_002_inspection_001 in
    # stand_explore_exact2_camera_20260908T123459Z. Keep replay ROS/tmp free.
    return {
        "source": "tf_lookup", "reason": "lookup_exception",
        "target_frame": "odom", "source_frame": "base_footprint",
        "exception_type": exception_type,
        "exception": (
            "Could not find a connection between 'odom' and 'base_footprint' "
            "because they are not part of the same tree. "
            "Tf has two or more unconnected trees."
        ),
        "max_age_sec": 1.0,
        **overrides,
    }


class _Clock:
    def __init__(self):
        self.now = 0.0

    def wait(self, timeout):
        # A deterministic delayed callback-service cycle, still within the
        # normal sensor/extra-phase budgets.
        self.now += 0.25


class InitialTfAcquisitionTest(unittest.TestCase):
    def make_node(self, *, reconnect_at=None, details=None, extra_wait=3.0):
        clock = _Clock()
        node = object.__new__(follower.SimpleWaypointFollowerNode)
        node.follower_config = follower.FollowerConfig(
            controller=ControllerConfig(), initial_tf_acquisition_wait_sec=extra_wait,
        )
        node.odom_execution_context = SimpleNamespace(
            map_frame="map", odom_frame="odom", base_frame="base_footprint",
        )
        node.latest_scan = object()
        node.latest_odom = object()
        node.latest_scan_receipt = 0.0
        node.latest_odom_receipt = 0.0
        node.motion_published = False
        node.latest_stop_details = None
        node.tf_buffer = object()
        node.controller_trace_writer = object()
        node._append_controller_trace = Mock(return_value="")
        node._service_or_wait_for_callbacks = Mock(side_effect=clock.wait)
        node._freshness_failure = Mock(return_value="")
        failure = _cold_failure() if details is None else details
        node._current_pose_lookup = Mock(side_effect=lambda: (
            follower.PoseLookupResult(Pose2D(0.1, 0.2, 0.3))
            if reconnect_at is not None and clock.now >= reconnect_at
            else follower.PoseLookupResult(None, details=failure)
        ))
        node._global_consistency_monitor_failure = Mock(return_value="")
        node.initial_tf_executor_health_probe = Mock(return_value={
            "ready": True, "thread_alive": True, "heartbeat_count": 1,
            "heartbeat_age_sec": 0.0, "tf_delivery_proven": False,
        })
        node.publish_zero = Mock()
        return node, clock

    def wait(self, node, clock):
        with patch.object(follower, "rclpy", SimpleNamespace(ok=lambda: True)), patch.object(
            initial_runtime_inputs.time, "monotonic", side_effect=lambda: clock.now,
        ):
            return node._wait_for_initial_runtime_inputs(0.0)

    def assert_stopped(self, node, clock, *, deadline, denial):
        failure = self.wait(node, clock)
        self.assertTrue(failure)
        self.assertEqual(clock.now, deadline)
        self.assertFalse(node.motion_published)
        node.publish_zero.assert_called()
        evidence = node.latest_stop_details["initial_tf_acquisition"]
        self.assertEqual(evidence["denial_reason"], denial)
        self.assertFalse(evidence["motion_authorized"])
        self.assertNotIn("initial_runtime_input_ready", [
            call.kwargs["event"] for call in node._append_controller_trace.call_args_list
        ])
        return failure, evidence

    def test_recorded_cold_listener_reconnects_then_runs_global_admission(self):
        node, clock = self.make_node(reconnect_at=2.25)
        original_buffer = node.tf_buffer

        self.assertEqual(self.wait(node, clock), "")

        self.assertIs(node.tf_buffer, original_buffer)
        self.assertIsNone(node.latest_stop_details)
        self.assertFalse(node.motion_published)
        self.assertEqual(clock.now, 2.25)
        node._global_consistency_monitor_failure.assert_called_once_with()
        evidence = node.latest_initial_tf_acquisition
        self.assertTrue(evidence["extension_used"])
        self.assertTrue(evidence["fresh_sensor_and_localization_admission_required"])
        self.assertFalse(evidence["motion_authorized"])
        self.assertEqual(evidence["edges"]["execution_pose"]["successful_sample_count"], 1)
        self.assertEqual(evidence["edges"]["execution_pose"]["target_frame"], "odom")
        self.assertEqual(evidence["edges"]["global_consistency"]["target_frame"], "map")
        self.assertEqual(evidence["edges"]["global_consistency"]["source_frame"], "odom")
        self.assertEqual(node.publish_zero.call_count, 8)
        traces = node._append_controller_trace.call_args_list
        self.assertEqual([call.kwargs["event"] for call in traces], [
            "initial_tf_acquisition_started", "initial_runtime_input_ready",
        ])
        for call in traces:
            command = call.kwargs["effective_command"]
            self.assertEqual(command.linear_x_mps, 0.0)
            self.assertEqual(command.angular_z_radps, 0.0)

    def test_lookup_exception_also_has_one_bounded_acquisition_phase(self):
        node, clock = self.make_node(reconnect_at=2.5, details=_cold_failure("LookupException"))

        self.assertEqual(self.wait(node, clock), "")
        self.assertTrue(node.latest_initial_tf_acquisition["extension_used"])

    def test_persistent_disconnection_stops_with_actual_frames_and_executor_evidence(self):
        node, clock = self.make_node()

        failure, evidence = self.assert_stopped(
            node, clock, deadline=5.0, denial="cold_tf_acquisition_deadline_exhausted",
        )

        self.assertEqual(failure, "TF transform unavailable: odom <- base_footprint")
        self.assertEqual(node.latest_stop_details["stop_reason"], failure)
        self.assertEqual(node.latest_stop_details["exception_type"], "ConnectivityException")
        self.assertEqual(node.latest_stop_details["exception"], _cold_failure()["exception"])
        self.assertTrue(evidence["executor_health"]["ready"])
        self.assertFalse(evidence["executor_health"]["tf_delivery_proven"])
        self.assertEqual(evidence["maximum_startup_wait_sec"], 5.0)
        self.assertEqual(len(evidence["edges"]["execution_pose"]["recent_failures"]), 8)
        node._global_consistency_monitor_failure.assert_not_called()

    def test_reconnect_at_hard_deadline_cannot_admit_a_late_sample(self):
        node, clock = self.make_node(reconnect_at=5.0)

        self.assert_stopped(
            node, clock, deadline=5.0, denial="cold_tf_acquisition_deadline_exhausted",
        )
        node._global_consistency_monitor_failure.assert_not_called()

    def test_blocking_lookup_cannot_finish_admission_after_hard_deadline(self):
        node, clock = self.make_node(reconnect_at=4.75)

        def delayed_global_lookup():
            clock.now = 5.1
            return ""

        node._global_consistency_monitor_failure.side_effect = delayed_global_lookup
        failure, _ = self.assert_stopped(
            node, clock, deadline=5.1, denial="cold_tf_acquisition_deadline_exhausted",
        )
        self.assertEqual(failure, "initial TF acquisition deadline exhausted")

    def test_stale_future_malformed_and_untyped_tf_never_get_extra_wait(self):
        failures = (
            _cold_failure(reason="stale_transform", age_sec=1.5),
            _cold_failure(reason="future_transform", age_sec=-1.5),
            _cold_failure(reason="invalid_transform"),
            _cold_failure(exception_type="ExtrapolationException"),
            _cold_failure(exception_type="TransformException"),
            _cold_failure(target_frame="map"),
        )
        for details in failures:
            with self.subTest(details=details):
                node, clock = self.make_node(details=details)
                _, evidence = self.assert_stopped(
                    node, clock, deadline=2.0,
                    denial="execution_edge_has_non_acquisition_failure",
                )
                self.assertFalse(evidence["extension_used"])

    def test_stale_tf_then_disconnection_cannot_erase_stale_history(self):
        node, clock = self.make_node()
        node._current_pose_lookup.side_effect = lambda: follower.PoseLookupResult(
            None, details=(
                _cold_failure(reason="stale_transform") if clock.now == 0.25
                else _cold_failure()
            ),
        )

        self.assert_stopped(
            node, clock, deadline=2.0, denial="execution_edge_has_non_acquisition_failure",
        )

    def test_unserviced_executor_does_not_get_extra_wait_even_with_fresh_sensors(self):
        for health in ({}, {"ready": False, "thread_alive": False},
                       {"ready": False, "thread_alive": True, "heartbeat_age_sec": 0.75}):
            with self.subTest(health=health):
                node, clock = self.make_node()
                node.initial_tf_executor_health_probe.return_value = health
                self.assert_stopped(
                    node, clock, deadline=2.0, denial="tf_executor_not_ready",
                )

    def test_executor_readiness_is_required_even_when_all_tf_edges_are_ready(self):
        node, clock = self.make_node(reconnect_at=0.25)
        node.initial_tf_executor_health_probe.return_value = {"ready": False}

        failure, _ = self.assert_stopped(
            node, clock, deadline=2.0, denial="tf_executor_not_ready",
        )
        self.assertEqual(failure, "TF listener executor not ready")
        node._global_consistency_monitor_failure.assert_called()

    def test_global_consistency_still_blocks_admission_after_execution_tf_reconnect(self):
        node, clock = self.make_node(reconnect_at=2.25)

        def global_failure():
            node.latest_stop_details = {
                "source": "global_consistency_monitor", "reason": "localization drift",
                "fault_code": "translation_drift", "target_frame": "map", "source_frame": "odom",
            }
            return "global consistency monitor stopped"

        node._global_consistency_monitor_failure.side_effect = global_failure
        failure, evidence = self.assert_stopped(
            node, clock, deadline=2.25, denial="execution_edge_already_acquired",
        )

        self.assertEqual(failure, "global consistency monitor stopped")
        self.assertEqual(node.latest_stop_details["fault_code"], "translation_drift")
        self.assertEqual(evidence["edges"]["global_consistency"]["successful_sample_count"], 0)

    def test_previously_acquired_edge_loss_does_not_get_extra_wait(self):
        node, clock = self.make_node()
        node._current_pose_lookup.side_effect = lambda: (
            follower.PoseLookupResult(Pose2D(0.1, 0.2, 0.3)) if clock.now == 0.25
            else follower.PoseLookupResult(None, details=_cold_failure())
        )
        node._global_consistency_monitor_failure.return_value = "global consistency missing"

        self.assert_stopped(
            node, clock, deadline=2.0, denial="execution_edge_already_acquired",
        )

    def test_fresh_sensor_requirement_remains_live_during_acquisition(self):
        node, clock = self.make_node()

        def freshness(name, *args):
            if name == "scan" and clock.now >= 2.25:
                node.latest_stop_details = {
                    "source": "message_freshness", "reason": "stale scan", "sensor": "scan",
                }
                return "stale scan"
            return ""

        node._freshness_failure.side_effect = freshness
        failure, _ = self.assert_stopped(
            node, clock, deadline=2.25, denial="sensor_inputs_not_fresh",
        )
        self.assertEqual(failure, "stale scan")

    def test_no_post_motion_entry_or_mid_wait_extension(self):
        for enter_after_motion in (True, False):
            with self.subTest(enter_after_motion=enter_after_motion):
                node, clock = self.make_node(reconnect_at=2.25)
                if enter_after_motion:
                    node.motion_published = True
                else:
                    def wait_then_move(timeout):
                        clock.wait(timeout)
                        node.motion_published = clock.now >= 2.25
                    node._service_or_wait_for_callbacks.side_effect = wait_then_move

                self.assertEqual(
                    self.wait(node, clock),
                    "initial runtime input acquisition requested after motion",
                )
                self.assertEqual(clock.now, 0.0 if enter_after_motion else 2.25)
                self.assertEqual(node.latest_stop_details["execution_phase"], "after_motion")
                self.assertEqual(node.latest_initial_tf_acquisition["denial_reason"], "motion_already_published")
                node._global_consistency_monitor_failure.assert_not_called()
                node.publish_zero.assert_called()
                if enter_after_motion:
                    node._service_or_wait_for_callbacks.assert_not_called()
                    node._current_pose_lookup.assert_not_called()

    def test_disabled_extra_phase_preserves_original_sensor_deadline(self):
        node, clock = self.make_node(extra_wait=0.0)
        self.assert_stopped(
            node, clock, deadline=2.0, denial="cold_tf_acquisition_disabled",
        )

    def test_evidence_write_failure_cannot_authorize_startup(self):
        node, clock = self.make_node(reconnect_at=2.25)

        def failed_trace(**kwargs):
            node.latest_stop_details = {"fault_code": "artifact_append_failed"}
            return "controller trace append failed"

        node._append_controller_trace.side_effect = failed_trace
        self.assertEqual(self.wait(node, clock), "controller trace append failed")
        self.assertEqual(clock.now, 2.0)
        self.assertEqual(node.latest_stop_details["fault_code"], "artifact_append_failed")
        self.assertTrue(node.latest_stop_details["fail_closed"])
        node._global_consistency_monitor_failure.assert_not_called()


class TfExecutorHeartbeatTest(unittest.TestCase):
    def test_readiness_requires_a_recent_callback_and_a_live_thread(self):
        heartbeat = TfExecutorHeartbeat()
        with patch("scripts.aufgabe04.navigation.waypoint_follower.initial_tf_acquisition.time.monotonic", return_value=1.0):
            self.assertFalse(heartbeat.snapshot(thread_alive=True)["ready"])
            heartbeat.tick()
            health = heartbeat.snapshot(thread_alive=True)
            self.assertTrue(health["ready"])
            self.assertEqual(health["heartbeat_count"], 1)
            self.assertFalse(health["tf_delivery_proven"])
            self.assertFalse(heartbeat.snapshot(thread_alive=False)["ready"])
        with patch("scripts.aufgabe04.navigation.waypoint_follower.initial_tf_acquisition.time.monotonic", return_value=1.501):
            self.assertFalse(heartbeat.snapshot(thread_alive=True)["ready"])

    def test_policy_cannot_rebind_failure_to_another_tf_edge(self):
        policy = InitialTfAcquisition(0.0, 2.0, 3.0)
        policy.record_edge(
            "execution_pose", target_frame="odom", source_frame="base_footprint",
            ready=False, details=_cold_failure(),
        )
        self.assertFalse(policy.can_continue(
            now=2.0, motion_published=False, sensors_fresh=True,
            failure_details=_cold_failure(target_frame="map"), executor_health={"ready": True},
        ))
        self.assertEqual(policy.denial_reason, "failure_not_execution_tf_edge")


if __name__ == "__main__":
    unittest.main()
