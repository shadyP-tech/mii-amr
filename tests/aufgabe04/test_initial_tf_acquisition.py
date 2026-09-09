"""Stopped follower TF acquisition regressions; no ROS or wall-clock sleeps."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.aufgabe04.navigation.control.waypoint_controller import ControllerConfig
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.navigation.localization.odom_route_adapter import OdomExecutionContext
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



def _ready_lookup(pose, target, source, clock):
    return follower.PoseLookupResult(pose, {
        "source": "tf_lookup", "reason": "fresh_transform",
        "target_frame": target, "source_frame": source,
        "available": True, "validation_passed": True,
        "age_sec": 0.0, "max_age_sec": 1.0,
        "max_future_sec": 1.1 if target == "map" else 0.25,
    }, 100.0 + clock.now)

class _Clock:
    def __init__(self):
        self.now = 0.0

    def wait(self, timeout):
        # A deterministic delayed callback-service cycle, still within the
        # normal sensor/extra-phase budgets.
        self.now += 0.25


class _RosTime:
    def __init__(self, nanoseconds=0):
        self.nanoseconds = nanoseconds

    @classmethod
    def from_msg(cls, stamp):
        return cls(stamp.sec * 1_000_000_000 + stamp.nanosec)

    def __sub__(self, other):
        return _RosTime(self.nanoseconds - other.nanoseconds)


class LookupException(Exception):
    pass


class InitialTfAcquisitionTest(unittest.TestCase):
    def make_node(self, *, reconnect_at=None, details=None, extra_wait=3.0):
        clock = _Clock()
        node = object.__new__(follower.SimpleWaypointFollowerNode)
        node.follower_config = follower.FollowerConfig(
            controller=ControllerConfig(), initial_tf_acquisition_wait_sec=extra_wait,
        )
        node.odom_execution_context = OdomExecutionContext(
            map_frame="map", odom_frame="odom", base_frame="base_footprint",
            frozen_map_from_odom=PlanarTransform2D(0.0, 0.0, 0.0),
            certificate_sha256="a" * 64,
            max_map_from_odom_translation_drift_m=0.1,
            max_map_from_odom_yaw_drift_rad=0.1,
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
            _ready_lookup(Pose2D(0.1, 0.2, 0.3), "odom", "base_footprint", clock)
            if reconnect_at is not None and clock.now >= reconnect_at
            else follower.PoseLookupResult(None, details=failure)
        ))
        node._map_from_odom_lookup = Mock(side_effect=lambda: _ready_lookup(Pose2D(0.0, 0.0, 0.0), "map", "odom", clock))
        node.get_clock = Mock(return_value=SimpleNamespace(now=lambda: _RosTime(int((100.0 + clock.now) * 1e9))))
        node._global_consistency_monitor_failure = Mock(return_value="")
        node.initial_tf_executor_health_probe = Mock(return_value={
            "ready": True, "thread_alive": True, "heartbeat_count": 1,
            "heartbeat_age_sec": 0.0, "heartbeat_max_age_sec": 0.5, "tf_delivery_proven": False,
        })
        node.publish_zero = Mock()
        return node, clock

    def wait(self, node, clock):
        with patch.object(follower, "rclpy", SimpleNamespace(ok=lambda: True)), patch.object(
            initial_runtime_inputs.time, "monotonic", side_effect=lambda: clock.now,
        ), patch.object(follower, "Time", _RosTime), patch.object(
            follower, "Duration", lambda **kwargs: kwargs,
        ):
            return node._wait_for_initial_runtime_inputs(0.0)

    def make_sampled_node(self, *, odom_at=0.0, map_at=0.0):
        """Exercise production TF sampling and continuity on the executing buffer."""
        node, clock = self.make_node()
        for method in ("_current_pose_lookup", "_map_from_odom_lookup", "_global_consistency_monitor_failure"):
            delattr(node, method)
        # Frozen context and missing-map exception from the terminal 14:36 run.
        node.odom_execution_context = OdomExecutionContext(
            map_frame="map", odom_frame="odom", base_frame="base_footprint",
            frozen_map_from_odom=PlanarTransform2D(-1.6105036284310639, -0.08429228009562106, -0.48698294362131),
            certificate_sha256="f611e7955e26c60d8ea39092df25430ff95eaf9b5917adadc11a13ca18cf4e91",
            max_map_from_odom_translation_drift_m=0.14034156361714037,
            max_map_from_odom_yaw_drift_rad=0.09206777224224688,
        )
        node.runtime_config = SimpleNamespace(map_frame="map", odom_frame="odom", base_frame="base_footprint")
        node.get_clock = Mock(return_value=SimpleNamespace(now=lambda: _RosTime(int((100.0 + clock.now) * 1e9))))

        def lookup(target, source, *args, **kwargs):
            available_at = map_at if target == "map" else odom_at
            if available_at is None or clock.now < available_at:
                raise LookupException(f'"{target}" passed to lookupTransform argument target_frame does not exist.')
            pose = node.odom_execution_context.frozen_map_from_odom if target == "map" else Pose2D(1.2047289298, 0.5574263304, -2.5825808684)
            return self.transform(target, source, pose, 100.0 + clock.now)

        node.tf_buffer = SimpleNamespace(lookup_transform=Mock(side_effect=lookup))
        return node, clock

    def make_delayed_stale_map_node(self, *, fresh_at=None):
        # Replay the measured first stale age after 11 cold-map misses, using
        # deterministic relative time. Any later fresh sample is hypothetical.
        node, clock = self.make_sampled_node(map_at=3.0)
        original_lookup = node.tf_buffer.lookup_transform.side_effect

        def delayed_map(target, source, *args, **kwargs):
            transform = original_lookup(target, source, *args, **kwargs)
            if target == "map" and (fresh_at is None or clock.now < fresh_at):
                ns = int((100.0 + clock.now - 1.664346) * 1e9)
                transform.header.stamp.sec = ns // 1_000_000_000
                transform.header.stamp.nanosec = ns % 1_000_000_000
            return transform

        node.tf_buffer.lookup_transform.side_effect = delayed_map
        return node, clock

    @staticmethod
    def transform(target, source, pose, stamp):
        import math
        ns = int(stamp * 1e9)
        return SimpleNamespace(
            header=SimpleNamespace(frame_id=target, stamp=SimpleNamespace(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)),
            child_frame_id=source,
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=pose.x_m, y=pose.y_m, z=0.0),
                rotation=SimpleNamespace(x=0.0, y=0.0, z=math.sin(pose.yaw_rad / 2), w=math.cos(pose.yaw_rad / 2)),
            ),
        )

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
        node._global_consistency_monitor_failure.assert_called()
        self.assertEqual(node._global_consistency_monitor_failure.call_args.kwargs["map_lookup"].details["target_frame"], "map")
        evidence = node.latest_initial_tf_acquisition
        self.assertTrue(evidence["extension_used"])
        self.assertTrue(evidence["fresh_sensor_and_localization_admission_required"])
        self.assertFalse(evidence["motion_authorized"])
        self.assertEqual(evidence["edges"]["execution_pose"]["successful_sample_count"], 1)
        self.assertEqual(evidence["edges"]["execution_pose"]["target_frame"], "odom")
        self.assertEqual(evidence["edges"]["global_consistency"]["target_frame"], "map")
        self.assertEqual(evidence["edges"]["global_consistency"]["source_frame"], "odom")
        self.assertEqual(node.publish_zero.call_count, 9)
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
        node._global_consistency_monitor_failure.assert_called()

    def test_reconnect_at_hard_deadline_cannot_admit_a_late_sample(self):
        node, clock = self.make_node(reconnect_at=5.0)

        self.assert_stopped(
            node, clock, deadline=5.0, denial="cold_tf_acquisition_deadline_exhausted",
        )
        node._global_consistency_monitor_failure.assert_called()

    def test_blocking_lookup_cannot_finish_admission_after_hard_deadline(self):
        node, clock = self.make_node(reconnect_at=4.75)

        def delayed_global_lookup(**kwargs):
            if clock.now < 4.75:
                return ""
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
                    denial="required_tf_edge_has_non_acquisition_failure",
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
            node, clock, deadline=2.0, denial="required_tf_edge_has_non_acquisition_failure",
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

        def global_failure(**kwargs):
            if clock.now < 2.25:
                return ""
            node.latest_stop_details = {
                "source": "global_consistency_monitor", "reason": "localization drift",
                "fault_code": "translation_drift", "target_frame": "map", "source_frame": "odom",
            }
            return "global consistency monitor stopped"

        node._global_consistency_monitor_failure.side_effect = global_failure
        failure, evidence = self.assert_stopped(
            node, clock, deadline=2.25, denial="continuity_admission_failed",
        )

        self.assertEqual(failure, "global consistency monitor stopped")
        self.assertEqual(node.latest_stop_details["fault_code"], "translation_drift")
        self.assertGreater(evidence["edges"]["global_consistency"]["successful_sample_count"], 0)
        self.assertTrue(evidence["admission_failure_seen"])

    def test_previously_acquired_edge_loss_does_not_get_extra_wait(self):
        node, clock = self.make_node()
        node._current_pose_lookup.side_effect = lambda: (
            _ready_lookup(Pose2D(0.1, 0.2, 0.3), "odom", "base_footprint", clock) if clock.now == 0.25
            else follower.PoseLookupResult(None, details=_cold_failure())
        )
        node._global_consistency_monitor_failure.return_value = "global consistency missing"

        self.assert_stopped(
            node, clock, deadline=2.0, denial="continuity_admission_failed",
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
                if enter_after_motion:
                    node._global_consistency_monitor_failure.assert_not_called()
                else:
                    node._global_consistency_monitor_failure.assert_called()
                node.publish_zero.assert_called()
                if enter_after_motion:
                    node._service_or_wait_for_callbacks.assert_not_called()
                    node._current_pose_lookup.assert_not_called()

    def test_disabled_extra_phase_preserves_original_sensor_deadline(self):
        node, clock = self.make_node(extra_wait=0.0)
        self.assert_stopped(
            node, clock, deadline=2.0, denial="cold_tf_acquisition_deadline_exhausted",
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
        node._global_consistency_monitor_failure.assert_called()

    def test_recorded_valid_odom_missing_map_transcript_reconnects_in_same_listener(self):
        node, clock = self.make_sampled_node(map_at=2.5)
        original_buffer = node.tf_buffer

        self.assertEqual(self.wait(node, clock), "")

        self.assertEqual(clock.now, 2.5)
        self.assertIs(node.tf_buffer, original_buffer)
        evidence = node.latest_initial_tf_acquisition
        self.assertEqual(evidence["schema_version"], 2)
        self.assertTrue(evidence["extension_used"])
        self.assertEqual(evidence["required_edges"], ["execution_pose", "global_consistency"])
        self.assertEqual(evidence["edges"]["execution_pose"]["successful_sample_count"], 10)
        edge = evidence["edges"]["global_consistency"]
        self.assertEqual(edge["successful_sample_count"], 1)
        self.assertFalse(edge["non_acquisition_failure_seen"])
        self.assertEqual(edge["recent_failures"][-1]["exception_type"], "LookupException")
        self.assertTrue(edge["last_sample"]["validation_passed"])
        self.assertEqual(edge["last_sample"]["target_frame"], "map")
        self.assertEqual(edge["last_sample"]["source_frame"], "odom")
        self.assertFalse(node.motion_published)
        self.assertEqual(node.publish_zero.call_count, 10)
        for call in node._append_controller_trace.call_args_list:
            command = call.kwargs["effective_command"]
            self.assertEqual((command.linear_x_mps, command.angular_z_radps), (0.0, 0.0))

    def test_permanent_map_absence_retains_typed_certificate_bound_terminal_evidence(self):
        node, clock = self.make_sampled_node(map_at=None)

        failure, evidence = self.assert_stopped(
            node, clock, deadline=5.0, denial="cold_tf_acquisition_deadline_exhausted",
        )

        self.assertEqual(failure, "TF transform unavailable: map <- odom")
        self.assertEqual(node.latest_stop_details["source"], "tf_lookup")
        self.assertEqual(node.latest_stop_details["exception_type"], "LookupException")
        self.assertNotIn("continuity", node.latest_stop_details)
        self.assertEqual(evidence["failed_edge_role"], "global_consistency")
        self.assertTrue(evidence["deadline_exhausted"])
        self.assertEqual(evidence["elapsed_sec"], 5.0)
        self.assertTrue(evidence["sensor_inputs_fresh"])
        self.assertFalse(evidence["admission_failure_seen"])
        self.assertEqual(evidence["execution_context"]["certificate_sha256"], node.odom_execution_context.certificate_sha256)
        self.assertTrue(evidence["edges"]["execution_pose"]["current_ready"])
        self.assertEqual(evidence["edges"]["global_consistency"]["successful_sample_count"], 0)

        from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.control_results import initial_runtime_input_stop_details
        from scripts.aufgabe04.navigation.localization.prestart_localization_reseal import evaluate_prestart_localization_reseal
        terminal = initial_runtime_input_stop_details(node.latest_stop_details, reason=failure, motion_published=False)
        decision = evaluate_prestart_localization_reseal(status="stopped", motion_published=False, stop_details=terminal)
        self.assertTrue(decision.eligible, decision.reason)
        self.assertEqual(decision.recovery_action, "tf_warmup_retry")
        self.assertTrue(decision.requires_fresh_localization)
        self.assertTrue(decision.requires_new_route_certificate)
        self.assertFalse(decision.automatic_motion_authorized)

    def test_recorded_cold_then_stale_map_waits_for_hypothetical_fresh_sample(self):
        # candidate_000 in stand_explore_exact2_camera_20260909T123820Z:
        # 11 cold map lookups followed by a first map sample aged 1.664346 s.
        # Keep the clock deterministic while replaying that failure transition.
        node, clock = self.make_delayed_stale_map_node(fresh_at=3.25)
        node._global_consistency_monitor_failure = Mock(wraps=node._global_consistency_monitor_failure)
        original_buffer = node.tf_buffer

        self.assertEqual(self.wait(node, clock), "")

        self.assertEqual(clock.now, 3.25)
        self.assertIs(node.tf_buffer, original_buffer)
        self.assertIsNone(node.latest_stop_details)
        self.assertFalse(node.motion_published)
        evidence = node.latest_initial_tf_acquisition
        self.assertEqual(evidence["phase"], "cold_tf_acquisition")
        self.assertTrue(evidence["extension_used"])
        self.assertFalse(evidence["deadline_exhausted"])
        self.assertEqual(evidence["elapsed_sec"], 3.25)
        self.assertEqual(evidence["maximum_startup_wait_sec"], 5.0)
        self.assertTrue(evidence["sensor_inputs_fresh"])
        self.assertTrue(evidence["executor_health"]["ready"])
        odom_edge = evidence["edges"]["execution_pose"]
        self.assertTrue(odom_edge["current_ready"])
        self.assertEqual(odom_edge["successful_sample_count"], 13)
        map_edge = evidence["edges"]["global_consistency"]
        self.assertEqual(map_edge["attempt_count"], 13)
        self.assertEqual(map_edge["successful_sample_count"], 1)
        self.assertTrue(map_edge["non_acquisition_failure_seen"])
        self.assertEqual(map_edge["waitable_stale_sample_count"], 1)
        self.assertEqual(map_edge["recent_failures"][-2]["reason"], "lookup_exception")
        stale = map_edge["recent_failures"][-1]
        self.assertEqual(stale["attempt"], 12)
        self.assertEqual(stale["reason"], "stale_transform")
        self.assertAlmostEqual(stale["age_sec"], 1.664346, places=7)
        self.assertTrue(stale["structural_validation_passed"])
        self.assertTrue(stale["waitable_first_stale_global_sample"])
        node._global_consistency_monitor_failure.assert_called_once()
        admitted = node._global_consistency_monitor_failure.call_args.kwargs["map_lookup"]
        self.assertIsNotNone(admitted.pose)
        self.assertEqual(admitted.details["reason"], "fresh_transform")
        self.assertTrue(admitted.details["validation_passed"])

    def test_persistent_first_stale_global_sample_exhausts_same_budget_without_reseal(self):
        node, clock = self.make_delayed_stale_map_node()
        node._global_consistency_monitor_failure = Mock(wraps=node._global_consistency_monitor_failure)
        failure, evidence = self.assert_stopped(
            node, clock, deadline=5.0, denial="cold_tf_acquisition_deadline_exhausted",
        )
        self.assertEqual(failure, "TF transform unavailable: map <- odom")
        self.assertEqual(node.latest_stop_details["reason"], "stale_transform")
        self.assertAlmostEqual(node.latest_stop_details["age_sec"], 1.664346, places=7)
        self.assertFalse(node.latest_stop_details["available"])
        self.assertFalse(node.latest_stop_details["validation_passed"])
        self.assertTrue(node.latest_stop_details["structural_validation_passed"])
        edge = evidence["edges"]["global_consistency"]
        self.assertFalse(edge["current_ready"])
        self.assertEqual(edge["successful_sample_count"], 0)
        self.assertGreater(edge["waitable_stale_sample_count"], 1)
        self.assertTrue(edge["non_acquisition_failure_seen"])
        node._global_consistency_monitor_failure.assert_not_called()

        from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.control_results import initial_runtime_input_stop_details
        from scripts.aufgabe04.navigation.localization.prestart_localization_reseal import evaluate_prestart_localization_reseal
        terminal = initial_runtime_input_stop_details(node.latest_stop_details, reason=failure, motion_published=False)
        decision = evaluate_prestart_localization_reseal(status="stopped", motion_published=False, stop_details=terminal)
        self.assertFalse(decision.eligible)
        self.assertEqual(decision.reason, "invalid_initial_map_tf_stop")
        self.assertFalse(decision.automatic_motion_authorized)

    def test_waitable_stale_then_missing_retains_non_cold_history_and_cannot_reseal(self):
        node, clock = self.make_delayed_stale_map_node()
        original_lookup = node.tf_buffer.lookup_transform.side_effect

        def missing_again(target, source, *args, **kwargs):
            if target == "map" and clock.now > 3.0:
                raise LookupException('"map" does not exist')
            return original_lookup(target, source, *args, **kwargs)

        node.tf_buffer.lookup_transform.side_effect = missing_again
        failure, evidence = self.assert_stopped(
            node, clock, deadline=3.25, denial="required_tf_edge_has_non_acquisition_failure",
        )
        self.assertEqual(node.latest_stop_details["reason"], "lookup_exception")
        edge = evidence["edges"]["global_consistency"]
        self.assertTrue(edge["non_acquisition_failure_seen"])
        self.assertEqual(edge["waitable_stale_sample_count"], 1)
        self.assertEqual(edge["successful_sample_count"], 0)
        from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.control_results import initial_runtime_input_stop_details
        from scripts.aufgabe04.navigation.localization.prestart_localization_reseal import evaluate_prestart_localization_reseal
        terminal = initial_runtime_input_stop_details(node.latest_stop_details, reason=failure, motion_published=False)
        self.assertFalse(evaluate_prestart_localization_reseal(status="stopped", motion_published=False, stop_details=terminal).eligible)

    def test_old_map_sample_must_pass_real_payload_validation_before_waiting(self):
        for defect in ("frame", "quaternion", "translation", "stamp_sec", "stamp_nanosec"):
            with self.subTest(defect=defect):
                node, clock = self.make_delayed_stale_map_node(fresh_at=3.25)
                original_lookup = node.tf_buffer.lookup_transform.side_effect

                def malformed(target, source, *args, **kwargs):
                    transform = original_lookup(target, source, *args, **kwargs)
                    if target == "map":
                        if defect == "frame":
                            transform.header.frame_id = "another_map"
                        elif defect == "quaternion":
                            transform.transform.rotation.w = 2.0
                        elif defect == "translation":
                            transform.transform.translation.x = float("nan")
                        elif defect == "stamp_sec":
                            transform.header.stamp.sec = -1
                        else:
                            transform.header.stamp.nanosec = 1_000_000_000
                    return transform

                node.tf_buffer.lookup_transform.side_effect = malformed
                _, evidence = self.assert_stopped(
                    node, clock, deadline=3.0, denial="required_tf_edge_has_non_acquisition_failure",
                )
                self.assertEqual(node.latest_stop_details["reason"], "malformed_transform_stamp" if defect.startswith("stamp_") else "malformed_transform_pose")
                self.assertFalse(node.latest_stop_details["structural_validation_passed"])
                self.assertEqual(evidence["edges"]["global_consistency"]["waitable_stale_sample_count"], 0)

    def test_late_future_map_sample_cannot_use_stale_first_sample_wait(self):
        node, clock = self.make_delayed_stale_map_node(fresh_at=3.25)
        original_lookup = node.tf_buffer.lookup_transform.side_effect

        def future_map(target, source, *args, **kwargs):
            transform = original_lookup(target, source, *args, **kwargs)
            if target == "map":
                transform.header.stamp.sec += 4
            return transform

        node.tf_buffer.lookup_transform.side_effect = future_map
        _, evidence = self.assert_stopped(node, clock, deadline=3.0, denial="required_tf_edge_has_non_acquisition_failure")
        self.assertEqual(node.latest_stop_details["reason"], "future_transform")
        self.assertTrue(node.latest_stop_details["structural_validation_passed"])
        self.assertEqual(evidence["edges"]["global_consistency"]["waitable_stale_sample_count"], 0)

    def test_first_stale_global_sample_cannot_hide_execution_tf_failure(self):
        for defect in ("stale", "never_acquired"):
            with self.subTest(defect=defect):
                node, clock = self.make_delayed_stale_map_node(fresh_at=3.25)
                original_lookup = node.tf_buffer.lookup_transform.side_effect

                def failed_execution(target, source, *args, **kwargs):
                    if target == "odom" and defect == "never_acquired":
                        raise LookupException('"odom" does not exist')
                    transform = original_lookup(target, source, *args, **kwargs)
                    if target == "odom" and clock.now >= 3.0:
                        transform.header.stamp.sec -= 2
                    return transform

                node.tf_buffer.lookup_transform.side_effect = failed_execution
                _, evidence = self.assert_stopped(
                    node, clock, deadline=3.0,
                    denial="required_tf_edge_already_acquired" if defect == "stale" else "required_tf_edge_has_non_acquisition_failure",
                )
                self.assertEqual(evidence["edges"]["global_consistency"]["waitable_stale_sample_count"], 0)

    def test_previously_acquired_map_becoming_stale_cannot_use_first_sample_wait(self):
        node, clock = self.make_sampled_node(odom_at=None)
        original_lookup = node.tf_buffer.lookup_transform.side_effect

        def established_map_stale(target, source, *args, **kwargs):
            transform = original_lookup(target, source, *args, **kwargs)
            if target == "map" and clock.now >= 3.0:
                transform.header.stamp.sec -= 2
            return transform

        node.tf_buffer.lookup_transform.side_effect = established_map_stale
        _, evidence = self.assert_stopped(node, clock, deadline=3.0, denial="required_tf_edge_already_acquired")
        self.assertEqual(node.latest_stop_details["reason"], "stale_transform")
        edge = evidence["edges"]["global_consistency"]
        self.assertGreater(edge["successful_sample_count"], 0)
        self.assertEqual(edge["waitable_stale_sample_count"], 0)

    def test_first_stale_global_wait_stops_on_sensor_or_executor_failure(self):
        for defect in ("sensor", "executor"):
            with self.subTest(defect=defect):
                node, clock = self.make_delayed_stale_map_node(fresh_at=4.0)
                if defect == "sensor":
                    def freshness(*args):
                        if clock.now >= 3.25:
                            node.latest_stop_details = {"source": "message_freshness", "reason": "stale scan"}
                            return "stale scan"
                        return ""
                    node._freshness_failure.side_effect = freshness
                else:
                    health = node.initial_tf_executor_health_probe.return_value
                    node.initial_tf_executor_health_probe.side_effect = lambda: {**health, "ready": clock.now < 3.25}
                _, evidence = self.assert_stopped(
                    node, clock, deadline=3.25,
                    denial="sensor_inputs_not_fresh" if defect == "sensor" else "tf_executor_not_ready",
                )
                self.assertGreater(evidence["edges"]["global_consistency"]["waitable_stale_sample_count"], 0)

    def test_first_stale_global_wait_stops_if_motion_is_reported(self):
        node, clock = self.make_delayed_stale_map_node(fresh_at=4.0)

        def report_motion(timeout):
            clock.wait(timeout)
            if clock.now >= 3.25:
                node.motion_published = True

        node._service_or_wait_for_callbacks.side_effect = report_motion
        self.assertEqual(self.wait(node, clock), "initial runtime input acquisition requested after motion")
        self.assertEqual(clock.now, 3.25)
        self.assertEqual(node.latest_initial_tf_acquisition["denial_reason"], "motion_already_published")
        node.publish_zero.assert_called()
        self.assertNotIn("initial_runtime_input_ready", [c.kwargs["event"] for c in node._append_controller_trace.call_args_list])

    def test_fresh_replacement_after_stale_still_requires_frozen_map_continuity(self):
        node, clock = self.make_delayed_stale_map_node(fresh_at=3.25)
        original_lookup = node.tf_buffer.lookup_transform.side_effect

        def drifting_replacement(target, source, *args, **kwargs):
            transform = original_lookup(target, source, *args, **kwargs)
            if target == "map" and clock.now >= 3.25:
                transform.transform.translation.x += 0.2
            return transform

        node.tf_buffer.lookup_transform.side_effect = drifting_replacement
        failure, evidence = self.assert_stopped(node, clock, deadline=3.25, denial="continuity_admission_failed")
        self.assertTrue(evidence["admission_failure_seen"])
        self.assertEqual(node.latest_stop_details["continuity"]["reason"], "map_from_odom_translation_drift")
        from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.control_results import initial_runtime_input_stop_details
        from scripts.aufgabe04.navigation.localization.prestart_localization_reseal import evaluate_prestart_localization_reseal
        terminal = initial_runtime_input_stop_details(node.latest_stop_details, reason=failure, motion_published=False)
        self.assertFalse(evaluate_prestart_localization_reseal(status="stopped", motion_published=False, stop_details=terminal).eligible)

    def test_stale_global_wait_cannot_admit_replacement_at_or_after_hard_deadline(self):
        for blocking in (False, True):
            with self.subTest(blocking=blocking):
                node, clock = self.make_delayed_stale_map_node(fresh_at=4.75 if blocking else 5.0)
                original_lookup = node.tf_buffer.lookup_transform.side_effect

                def late_lookup(target, source, *args, **kwargs):
                    transform = original_lookup(target, source, *args, **kwargs)
                    if blocking and target == "map" and clock.now >= 4.75:
                        clock.now = 5.1
                    return transform

                node.tf_buffer.lookup_transform.side_effect = late_lookup
                self.assert_stopped(node, clock, deadline=5.1 if blocking else 5.0, denial="cold_tf_acquisition_deadline_exhausted")

    def test_either_edge_can_acquire_first_without_resetting_shared_deadline(self):
        for odom_at, map_at in ((2.25, 4.75), (4.75, 2.25)):
            with self.subTest(odom_at=odom_at, map_at=map_at):
                node, clock = self.make_sampled_node(odom_at=odom_at, map_at=map_at)
                self.assertEqual(self.wait(node, clock), "")
                self.assertEqual(clock.now, 4.75)
                self.assertEqual(node.latest_initial_tf_acquisition["maximum_startup_wait_sec"], 5.0)
                self.assertEqual([c.kwargs["event"] for c in node._append_controller_trace.call_args_list].count("initial_tf_acquisition_started"), 1)

    def test_previously_acquired_map_edge_loss_cannot_extend_other_cold_edge(self):
        node, clock = self.make_sampled_node(odom_at=None)
        original_lookup = node.tf_buffer.lookup_transform.side_effect

        def lose_map(target, source, *args, **kwargs):
            if target == "map" and clock.now > 0.25:
                raise LookupException('"map" does not exist')
            return original_lookup(target, source, *args, **kwargs)

        node.tf_buffer.lookup_transform.side_effect = lose_map
        failure, _ = self.assert_stopped(
            node, clock, deadline=2.0, denial="required_tf_edge_already_acquired",
        )
        self.assertEqual(failure, "TF transform unavailable: map <- odom")

    def test_map_stale_future_and_invalid_payload_are_not_cold_acquisition(self):
        for defect in ("stale", "future", "frame", "quaternion"):
            with self.subTest(defect=defect):
                node, clock = self.make_sampled_node()
                original_lookup = node.tf_buffer.lookup_transform.side_effect

                def invalid_map(target, source, *args, **kwargs):
                    transform = original_lookup(target, source, *args, **kwargs)
                    if target == "map":
                        if defect in ("stale", "future"):
                            transform.header.stamp.sec += -2 if defect == "stale" else 2
                        elif defect == "frame":
                            transform.header.frame_id = "another_map"
                        else:
                            transform.transform.rotation.w = 2.0
                    return transform

                node.tf_buffer.lookup_transform.side_effect = invalid_map
                self.assert_stopped(node, clock, deadline=2.0, denial="required_tf_edge_has_non_acquisition_failure")

    def test_actual_map_drift_still_blocks_after_missing_edge_acquisition(self):
        node, clock = self.make_sampled_node(map_at=2.25)
        original_lookup = node.tf_buffer.lookup_transform.side_effect

        def drifting_map(target, source, *args, **kwargs):
            transform = original_lookup(target, source, *args, **kwargs)
            if target == "map":
                transform.transform.translation.x += 0.2
            return transform

        node.tf_buffer.lookup_transform.side_effect = drifting_map
        failure, evidence = self.assert_stopped(node, clock, deadline=2.25, denial="continuity_admission_failed")
        self.assertEqual(failure, "global localization consistency requires zero and reseal")
        self.assertEqual(node.latest_stop_details["continuity"]["reason"], "map_from_odom_translation_drift")
        self.assertTrue(node.latest_stop_details["tf_sample"]["validation_passed"])
        self.assertTrue(evidence["admission_failure_seen"])

        from scripts.aufgabe04.navigation.waypoint_follower.runtime_components.control_results import initial_runtime_input_stop_details
        from scripts.aufgabe04.navigation.localization.prestart_localization_reseal import evaluate_prestart_localization_reseal
        terminal = initial_runtime_input_stop_details(node.latest_stop_details, reason=failure, motion_published=False)
        decision = evaluate_prestart_localization_reseal(status="stopped", motion_published=False, stop_details=terminal)
        self.assertTrue(decision.eligible, decision.reason)
        self.assertEqual(decision.recovery_action, "fresh_localization_reseal")
        self.assertTrue(decision.requires_fresh_localization)
        self.assertTrue(decision.requires_new_route_certificate)
        self.assertFalse(decision.automatic_motion_authorized)

    def test_ordinary_phase_callback_cannot_cross_absolute_budget_and_admit(self):
        node, clock = self.make_sampled_node()
        node._service_or_wait_for_callbacks.side_effect = lambda _: setattr(clock, "now", 5.1)
        failure, _ = self.assert_stopped(node, clock, deadline=5.1, denial="cold_tf_acquisition_deadline_exhausted")
        self.assertEqual(failure, "initial TF acquisition deadline exhausted")
        node.tf_buffer.lookup_transform.assert_not_called()
        self.assertFalse(node.latest_initial_tf_acquisition["extension_used"])

    def test_post_lookup_executor_and_sensor_rechecks_can_block_readiness(self):
        for defect in ("executor", "sensor"):
            with self.subTest(defect=defect):
                node, clock = self.make_sampled_node()
                original_lookup = node.tf_buffer.lookup_transform.side_effect

                def delayed_lookup(*args, **kwargs):
                    transform = original_lookup(*args, **kwargs)
                    clock.now = 2.25
                    return transform

                node.tf_buffer.lookup_transform.side_effect = delayed_lookup
                if defect == "executor":
                    node.initial_tf_executor_health_probe.side_effect = lambda: {"ready": clock.now < 2.0}
                else:
                    def freshness(*args):
                        if clock.now >= 2.0:
                            node.latest_stop_details = {"source": "message_freshness", "reason": "stale scan"}
                            return "stale scan"
                        return ""
                    node._freshness_failure.side_effect = freshness
                self.assert_stopped(node, clock, deadline=2.25, denial="tf_executor_not_ready" if defect == "executor" else "sensor_inputs_not_fresh")

    def test_runtime_missing_map_still_uses_zero_reseal_contract_with_typed_sample(self):
        node, clock = self.make_sampled_node(map_at=None)
        node.motion_published = True
        with patch.object(follower, "Time", _RosTime), patch.object(follower, "Duration", lambda **kwargs: kwargs):
            failure = node._global_consistency_monitor_failure()
        self.assertEqual(failure, "global localization consistency requires zero and reseal")
        self.assertEqual(node.latest_stop_details["continuity"]["reason"], "map_from_odom_missing")
        self.assertEqual(node.latest_stop_details["tf_sample"]["exception_type"], "LookupException")
        self.assertEqual(node.latest_stop_details["tf_sample"]["target_frame"], "map")
        self.assertEqual(node.latest_stop_details["tf_sample"]["source_frame"], "odom")

    def test_first_tf_sample_cannot_expire_during_second_lookup_or_continuity(self):
        for delay_phase in ("second_lookup", "continuity"):
            with self.subTest(delay_phase=delay_phase):
                node, clock = self.make_sampled_node()
                original_lookup = node.tf_buffer.lookup_transform.side_effect

                def delayed_lookup(target, source, *args, **kwargs):
                    transform = original_lookup(target, source, *args, **kwargs)
                    if target == "odom":
                        ns = int((100.0 + clock.now - 0.95) * 1e9)
                        transform.header.stamp.sec = ns // 1_000_000_000
                        transform.header.stamp.nanosec = ns % 1_000_000_000
                    elif delay_phase == "second_lookup":
                        clock.now += 0.1
                    return transform

                node.tf_buffer.lookup_transform.side_effect = delayed_lookup
                if delay_phase == "continuity":
                    monitor = node._global_consistency_monitor_failure
                    def delayed_monitor(**kwargs):
                        outcome = monitor(**kwargs)
                        clock.now += 0.1
                        return outcome
                    node._global_consistency_monitor_failure = delayed_monitor
                self.assertTrue(self.wait(node, clock))
                self.assertAlmostEqual(clock.now, 2.1)
                self.assertEqual(node.latest_stop_details["reason"], "stale_transform")
                self.assertAlmostEqual(node.latest_stop_details["age_sec"], 1.05, places=6)
                self.assertFalse(node.latest_initial_tf_acquisition["extension_used"])
                self.assertTrue(node.latest_initial_tf_acquisition["edges"]["execution_pose"]["non_acquisition_failure_seen"])
                self.assertNotIn("initial_runtime_input_ready", [c.kwargs["event"] for c in node._append_controller_trace.call_args_list])
                self.assertFalse(node.motion_published)

    def test_late_deadline_callback_does_not_report_an_old_execution_sample_as_fresh(self):
        node, clock = self.make_sampled_node(map_at=None)

        def wait(timeout):
            if clock.now >= 4.75:
                clock.now = 6.5
            else:
                clock.wait(timeout)

        node._service_or_wait_for_callbacks.side_effect = wait
        self.assertTrue(self.wait(node, clock))
        self.assertEqual(node.latest_stop_details["reason"], "stale_transform")
        self.assertAlmostEqual(node.latest_stop_details["age_sec"], 1.75)
        edge = node.latest_initial_tf_acquisition["edges"]["execution_pose"]
        self.assertFalse(edge["current_ready"])
        self.assertTrue(edge["non_acquisition_failure_seen"])


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
        self.assertEqual(policy.denial_reason, "failure_not_required_tf_edge")


if __name__ == "__main__":
    unittest.main()
