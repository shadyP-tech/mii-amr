from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.waypoint_follower.runtime import (
    STALE_TF_RECOVERY_MAX_CALLBACKS,
    STALE_TF_RECOVERY_MAX_DURATION_SEC,
    STALE_TF_RECOVERY_SPIN_TIMEOUT_SEC,
    PoseLookupResult,
    SimpleWaypointFollowerNode,
    tf_lookup_failure_details,
)
from scripts.aufgabe04.navigation.localization.tf_stale_recovery_policy import (
    OdomStationaritySample,
    TfEdgeSample,
    evaluate_stationarity,
)


def failed_lookup(
    reason: str,
    age_sec: float | None = None,
    stamp_sec: float | None = None,
) -> PoseLookupResult:
    return PoseLookupResult(
        None,
        tf_lookup_failure_details(
            reason=reason,
            target_frame="odom",
            source_frame="base_footprint",
            max_age_sec=1.0,
            age_sec=age_sec,
            exception=(
                RuntimeError("missing transform")
                if reason == "lookup_exception"
                else None
            ),
        ),
        stamp_sec,
    )


def odom_message(
    *,
    stamp_sec: float = 7595.643,
    frame_id: str = "odom",
    child_frame_id: str = "base_footprint",
    x_m: float = 1.25,
    y_m: float = -0.4,
    qx: float = 0.0,
    qy: float = 0.0,
    qz: float = 0.14943813247359922,
    qw: float = 0.9887710779360422,
    linear_x_mps: float = 0.0,
    angular_z_radps: float = 0.0,
):
    whole_sec = math.floor(stamp_sec)
    nanosec = round((stamp_sec - whole_sec) * 1_000_000_000)
    return SimpleNamespace(
        header=SimpleNamespace(
            frame_id=frame_id,
            stamp=SimpleNamespace(sec=whole_sec, nanosec=nanosec),
        ),
        child_frame_id=child_frame_id,
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x_m, y=y_m, z=0.0),
                orientation=SimpleNamespace(
                    x=qx,
                    y=qy,
                    z=qz,
                    w=qw,
                ),
            )
        ),
        twist=SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=linear_x_mps),
                angular=SimpleNamespace(z=angular_z_radps),
            )
        ),
    )


class FollowerTfStaleRecoveryTest(unittest.TestCase):
    def bare_node(self):
        return object.__new__(SimpleWaypointFollowerNode)

    def fallback_node(self):
        node = self.bare_node()
        node.runtime_config = SimpleNamespace(
            use_sim_time=True,
            localization_source="tf",
            map_frame="odom",
            odom_frame="odom",
            base_frame="base_footprint",
        )
        node.follower_config = SimpleNamespace(
            allow_simulation_odom_after_stale_tf=True,
            max_odom_age_sec=1.0,
            max_scan_age_sec=1.0,
            max_future_timestamp_sec=0.25,
        )
        node.latest_odom = odom_message()
        node.latest_odom_receipt = 100.0
        node.latest_scan = object()
        node.latest_scan_receipt = 100.0
        node.latest_odom_callback_count = 4
        node.latest_stop_details = None
        node._simulation_odom_fallback_active = False
        node._simulation_odom_fallback_episode = 0
        node._fallback_message_freshness_evidence = Mock(
            side_effect=lambda name, _msg, _receipt, max_age_sec: {
                "sensor": name,
                "fresh": True,
                "failure": "",
                "receipt_age_sec": 0.01,
                "header_age_sec": 0.02,
                "max_age_sec": max_age_sec,
                "max_future_sec": 0.25,
            }
        )
        node.route_update_callback = Mock()
        return node

    def real_amcl_node(self):
        node = self.bare_node()
        node.runtime_config = SimpleNamespace(
            use_sim_time=False,
            localization_source="amcl",
            map_frame="map",
            odom_frame="odom",
            base_frame="base_footprint",
            cmd_vel_topic="/cmd_vel",
        )
        node.follower_config = SimpleNamespace(
            runtime_nomotion_update_service="request_nomotion_update",
            runtime_nomotion_update_timeout_sec=2.0,
            max_tf_age_sec=1.0,
            max_future_timestamp_sec=0.25,
            amcl_edge_future_tolerance_sec=1.1,
            max_scan_age_sec=1.0,
            max_odom_age_sec=1.0,
            control_rate_hz=10.0,
        )
        node.runtime_nomotion_update_service = "/request_nomotion_update"
        node.latest_scan = object()
        node.latest_scan_receipt = 1.0
        node.latest_odom = odom_message(stamp_sec=99.9)
        node.latest_odom_receipt = 1.0
        node.latest_odom_callback_count = 2
        node.latest_stop_details = None
        node._emit_route_update = Mock(return_value=True)
        node._append_controller_trace = Mock(return_value="")
        node._post_stale_tf_recovery_freshness_failure = Mock(
            return_value=""
        )
        node._cmd_vel_ownership_failure = Mock(return_value="")
        node._fallback_message_freshness_evidence = Mock(
            side_effect=lambda name, _msg, _receipt, max_age: {
                "sensor": name,
                "fresh": True,
                "failure": "",
                "max_age_sec": max_age,
            }
        )
        first = OdomStationaritySample(
            1,
            99.8,
            1.0,
            2.0,
            0.1,
            0.0,
            0.0,
        )
        second = OdomStationaritySample(
            2,
            99.9,
            1.0,
            2.0,
            0.1,
            0.0,
            0.0,
        )
        stationarity = evaluate_stationarity(
            first,
            second,
            now_sec=100.0,
        )
        node._wait_for_stationary_odom_pair = Mock(
            return_value=(
                stationarity,
                {
                    "accepted": True,
                    "decision": stationarity.to_log_dict(),
                },
            )
        )
        node._ros_now_sec = Mock(return_value=100.0)
        node._service_or_wait_for_callbacks = Mock()
        return node

    def direct_fallback_result(
        self,
        node,
        *,
        count_before: int = 4,
        count_after: int = 5,
    ):
        return node._simulation_odom_fallback_after_stale_retry(
            first_lookup=failed_lookup(
                "stale_transform",
                1.462,
                7594.181,
            ),
            retry_lookup=failed_lookup(
                "stale_transform",
                1.641,
                7594.181,
            ),
            callback_drain={"elapsed_sec": 0.18, "spin_count": 0},
            odom_callback_count_before=count_before,
            odom_callback_count_after=count_after,
            odom_msg=node.latest_odom,
            odom_receipt=node.latest_odom_receipt,
            scan_msg=node.latest_scan,
            scan_receipt=node.latest_scan_receipt,
        )

    def test_repeat20_replay_accepts_advancing_exact_frame_sim_odometry(self):
        node = self.fallback_node()
        events = []
        lookups = [
            failed_lookup("stale_transform", 1.462, 7594.181),
            failed_lookup("stale_transform", 1.641, 7594.181),
        ]

        def lookup():
            events.append("lookup")
            return lookups.pop(0)

        def drain(**kwargs):
            events.append("drain")
            node.latest_odom_callback_count += 1
            return {"elapsed_sec": 0.18, "spin_count": 0, **kwargs}

        def semantic_event(update):
            events.append(update.event_name)

        node._current_pose_lookup = Mock(side_effect=lookup)
        node.publish_zero = lambda: events.append("zero")
        node._drain_runtime_callbacks = drain
        node.route_update_callback = semantic_event

        result = node._current_pose_lookup_with_stale_recovery()

        self.assertEqual(result.pose, Pose2D(1.25, -0.4, 0.3))
        self.assertAlmostEqual(result.stamp_sec, 7595.643)
        self.assertEqual(
            result.details["source"],
            "simulation_direct_odom_after_tf_retry",
        )
        self.assertTrue(result.details["accepted"])
        self.assertTrue(result.details["zero_published_before_fallback"])
        self.assertTrue(
            result.details["not_real_robot_migration_evidence"]
        )
        self.assertEqual(result.details["retry_lookup_stamp_sec"], 7594.181)
        self.assertEqual(
            result.details["odom"]["header_stamp_sec"],
            7595.643,
        )
        self.assertTrue(
            result.details["predicates"][
                "odom_callback_advanced_during_recovery"
            ]
        )
        self.assertTrue(
            result.details["predicates"][
                "odom_stamp_newer_than_tf_retry"
            ]
        )
        self.assertEqual(
            events,
            [
                "lookup",
                "zero",
                "drain",
                "lookup",
                "simulation_odom_pose_fallback_started",
            ],
        )

    def test_simulation_odom_fallback_rejection_matrix_preserves_tf_stop(self):
        cases = {
            "real_time": (
                lambda node: setattr(
                    node.runtime_config,
                    "use_sim_time",
                    False,
                ),
                "use_sim_time",
            ),
            "map_not_odom": (
                lambda node: setattr(
                    node.runtime_config,
                    "map_frame",
                    "map",
                ),
                "map_frame_is_odom_frame",
            ),
            "amcl": (
                lambda node: setattr(
                    node.runtime_config,
                    "localization_source",
                    "amcl",
                ),
                "localization_source_is_tf",
            ),
            "parent_frame": (
                lambda node: setattr(
                    node.latest_odom.header,
                    "frame_id",
                    "map",
                ),
                "odom_parent_frame_exact",
            ),
            "child_frame": (
                lambda node: setattr(
                    node.latest_odom,
                    "child_frame_id",
                    "base_link",
                ),
                "odom_child_frame_exact",
            ),
            "nonfinite_position": (
                lambda node: setattr(
                    node.latest_odom.pose.pose.position,
                    "x",
                    math.nan,
                ),
                "position_xy_finite",
            ),
            "nonfinite_quaternion": (
                lambda node: setattr(
                    node.latest_odom.pose.pose.orientation,
                    "z",
                    math.inf,
                ),
                "quaternion_finite",
            ),
            "quaternion_norm": (
                lambda node: setattr(
                    node.latest_odom.pose.pose.orientation,
                    "w",
                    0.5,
                ),
                "quaternion_norm_valid",
            ),
            "nonnewer_stamp": (
                lambda node: setattr(
                    node,
                    "latest_odom",
                    odom_message(stamp_sec=7594.181),
                ),
                "odom_stamp_newer_than_tf_retry",
            ),
        }
        for name, (mutate, failed_predicate) in cases.items():
            with self.subTest(name=name):
                node = self.fallback_node()
                mutate(node)

                result = self.direct_fallback_result(node)

                self.assertIsNone(result.pose)
                self.assertEqual(
                    result.details["reason"],
                    "stale_transform",
                )
                self.assertEqual(
                    result.details["stop_reason"],
                    "map-to-base transform unavailable",
                )
                fallback = result.details["simulation_odom_fallback"]
                self.assertFalse(fallback["accepted"])
                self.assertIn(
                    failed_predicate,
                    fallback["rejection_reasons"],
                )
                node.route_update_callback.assert_not_called()

    def test_fallback_rejects_stale_future_sensor_and_no_callback_advance(self):
        cases = (
            ("stale_odom", "odom", "odom_fresh"),
            ("future_odom", "odom", "odom_fresh"),
            ("stale_scan", "scan", "scan_fresh_after_recovery"),
        )
        for name, sensor, failed_predicate in cases:
            with self.subTest(name=name):
                node = self.fallback_node()
                node._fallback_message_freshness_evidence = Mock(
                    side_effect=lambda current, _msg, _receipt, max_age: {
                        "sensor": current,
                        "fresh": current != sensor,
                        "failure": (
                            f"{name} rejected"
                            if current == sensor
                            else ""
                        ),
                        "receipt_age_sec": (
                            1.2 if name == "stale_odom" else -0.3
                        ),
                        "header_age_sec": (
                            1.2 if name != "future_odom" else -0.3
                        ),
                        "max_age_sec": max_age,
                        "max_future_sec": 0.25,
                    }
                )

                result = self.direct_fallback_result(node)

                self.assertIsNone(result.pose)
                self.assertIn(
                    failed_predicate,
                    result.details["simulation_odom_fallback"][
                        "rejection_reasons"
                    ],
                )

        node = self.fallback_node()
        result = self.direct_fallback_result(
            node,
            count_before=4,
            count_after=4,
        )
        self.assertIsNone(result.pose)
        self.assertIn(
            "odom_callback_advanced_during_recovery",
            result.details["simulation_odom_fallback"][
                "rejection_reasons"
            ],
        )

    def test_fallback_is_default_off_and_preserves_original_tf_failure(self):
        node = self.fallback_node()
        node.follower_config.allow_simulation_odom_after_stale_tf = False

        result = self.direct_fallback_result(node)

        self.assertIsNone(result.pose)
        self.assertEqual(result.details["reason"], "stale_transform")
        fallback = result.details["simulation_odom_fallback"]
        self.assertFalse(fallback["attempted"])
        self.assertEqual(
            fallback["rejection_reasons"],
            ["explicitly_enabled"],
        )
        node._fallback_message_freshness_evidence.assert_not_called()

    def test_fallback_episode_event_emits_once_and_primary_tf_restore_resets(self):
        node = self.fallback_node()

        first = self.direct_fallback_result(node)
        second = self.direct_fallback_result(node)

        self.assertIsNotNone(first.pose)
        self.assertIsNotNone(second.pose)
        self.assertEqual(node.route_update_callback.call_count, 1)
        self.assertEqual(
            node.route_update_callback.call_args.args[0].event_name,
            "simulation_odom_pose_fallback_started",
        )

        node.publish_zero = Mock()
        primary = PoseLookupResult(
            Pose2D(2.0, 3.0, 0.4),
            stamp_sec=7596.0,
        )
        node._current_pose_lookup = Mock(return_value=primary)

        restored = node._current_pose_lookup_with_stale_recovery()

        self.assertIs(restored, primary)
        self.assertFalse(node._simulation_odom_fallback_active)
        self.assertEqual(node.route_update_callback.call_count, 2)
        self.assertEqual(
            node.route_update_callback.call_args.args[0].event_name,
            "simulation_odom_pose_fallback_restored",
        )
        node.publish_zero.assert_called_once_with()

    def test_fallback_event_callback_failure_is_fail_closed(self):
        node = self.fallback_node()
        node.route_update_callback.side_effect = RuntimeError(
            "semantic sink unavailable"
        )

        result = self.direct_fallback_result(node)

        self.assertIsNone(result.pose)
        self.assertEqual(result.details["source"], "semantic_event_callback")
        self.assertIn("semantic sink unavailable", result.details["stop_reason"])
        self.assertTrue(result.details["fail_closed"])

    def test_stale_lookup_zeros_drains_and_retries_once_before_continuing(self):
        node = self.bare_node()
        fresh = PoseLookupResult(Pose2D(1.0, 2.0, 0.3), stamp_sec=10.1)
        events = []
        node._current_pose_lookup = Mock(
            side_effect=lambda: (
                events.append("lookup")
                or (
                    failed_lookup("stale_transform", 1.4, 10.0)
                    if events.count("lookup") == 1
                    else fresh
                )
            )
        )
        node.publish_zero = lambda: events.append("zero")

        def drain(**kwargs):
            events.append("drain")
            return {"spin_count": 7, **kwargs}

        node._drain_runtime_callbacks = drain
        node._post_stale_tf_recovery_freshness_failure = Mock(return_value="")

        result = node._current_pose_lookup_with_stale_recovery()

        self.assertEqual(result.pose, fresh.pose)
        self.assertEqual(events, ["lookup", "zero", "drain", "lookup"])
        self.assertEqual(node._current_pose_lookup.call_count, 2)
        self.assertIsNone(result.details)
        node._post_stale_tf_recovery_freshness_failure.assert_called_once_with()

    def test_persistent_stale_preserves_stop_and_both_age_diagnostics(self):
        node = self.bare_node()
        node._current_pose_lookup = Mock(
            side_effect=(
                failed_lookup("stale_transform", 1.4, 10.0),
                failed_lookup("stale_transform", 1.2, 10.1),
            )
        )
        zero_calls = []
        node.publish_zero = lambda: zero_calls.append(True)
        node._drain_runtime_callbacks = Mock(
            return_value={"spin_count": 48, "elapsed_sec": 0.18}
        )

        result = node._current_pose_lookup_with_stale_recovery()

        self.assertIsNone(result.pose)
        self.assertEqual(result.details["stop_reason"], "map-to-base transform unavailable")
        self.assertEqual(result.details["reason"], "stale_transform")
        self.assertEqual(result.details["age_sec"], 1.2)
        self.assertEqual(result.details["first_lookup_age_sec"], 1.4)
        self.assertEqual(result.details["retry_lookup_age_sec"], 1.2)
        self.assertEqual(result.details["first_lookup_stamp_sec"], 10.0)
        self.assertEqual(result.details["retry_lookup_stamp_sec"], 10.1)
        self.assertTrue(result.details["recovery_attempted"])
        self.assertTrue(result.details["zero_published_before_retry"])
        self.assertTrue(result.details["fail_closed"])
        self.assertEqual(
            result.details["first_lookup"]["age_sec"],
            1.4,
        )
        self.assertEqual(
            result.details["retry_lookup"]["age_sec"],
            1.2,
        )
        self.assertEqual(zero_calls, [True])
        node._drain_runtime_callbacks.assert_called_once_with(
            max_callbacks=STALE_TF_RECOVERY_MAX_CALLBACKS,
            max_duration_sec=STALE_TF_RECOVERY_MAX_DURATION_SEC,
            spin_timeout_sec=STALE_TF_RECOVERY_SPIN_TIMEOUT_SEC,
        )

    def test_fresh_retry_must_have_a_strictly_newer_transform_timestamp(self):
        node = self.bare_node()
        node.runtime_config = SimpleNamespace(
            map_frame="odom",
            base_frame="base_footprint",
        )
        node.follower_config = SimpleNamespace(max_tf_age_sec=1.0)
        node._current_pose_lookup = Mock(
            side_effect=(
                failed_lookup("stale_transform", 1.4, 10.0),
                PoseLookupResult(Pose2D(1.0, 2.0, 0.3), stamp_sec=10.0),
            )
        )
        node.publish_zero = Mock()
        node._drain_runtime_callbacks = Mock(return_value={"spin_count": 4})
        node._post_stale_tf_recovery_freshness_failure = Mock(return_value="")

        result = node._current_pose_lookup_with_stale_recovery()

        self.assertIsNone(result.pose)
        self.assertEqual(result.details["reason"], "nonadvancing_transform")
        self.assertEqual(result.details["first_lookup_stamp_sec"], 10.0)
        self.assertEqual(result.details["retry_lookup_stamp_sec"], 10.0)
        self.assertTrue(result.details["fail_closed"])
        node._post_stale_tf_recovery_freshness_failure.assert_not_called()

    def test_fresh_retry_rechecks_scan_and_odom_before_continuing(self):
        node = self.bare_node()
        node.latest_scan = object()
        node.latest_scan_receipt = 1.0
        node.latest_odom = object()
        node.latest_odom_receipt = 2.0
        node.follower_config = SimpleNamespace(
            max_scan_age_sec=0.5,
            max_odom_age_sec=0.6,
        )
        node._freshness_failure = Mock(side_effect=("", ""))

        failure = node._post_stale_tf_recovery_freshness_failure()

        self.assertEqual(failure, "")
        self.assertEqual(
            node._freshness_failure.call_args_list[0].args,
            ("scan", node.latest_scan, 1.0, 0.5),
        )
        self.assertEqual(
            node._freshness_failure.call_args_list[1].args,
            ("odom", node.latest_odom, 2.0, 0.6),
        )

    def test_stale_sensor_after_fresh_retry_remains_fail_closed(self):
        node = self.bare_node()
        node._current_pose_lookup = Mock(
            side_effect=(
                failed_lookup("stale_transform", 1.4, 10.0),
                PoseLookupResult(Pose2D(1.0, 2.0, 0.3), stamp_sec=10.1),
            )
        )
        node.publish_zero = Mock()
        node._drain_runtime_callbacks = Mock(return_value={"spin_count": 4})
        node._post_stale_tf_recovery_freshness_failure = Mock(
            return_value="stale scan"
        )
        node.latest_stop_details = {
            "reason": "stale scan",
            "source": "message_freshness",
        }

        result = node._current_pose_lookup_with_stale_recovery()

        self.assertIsNone(result.pose)
        self.assertEqual(result.details["stop_reason"], "stale scan")
        self.assertEqual(
            result.details["reason"],
            "post_recovery_sensor_freshness_failure",
        )
        self.assertEqual(
            result.details["sensor_failure"]["source"],
            "message_freshness",
        )
        self.assertTrue(result.details["fail_closed"])

    def test_real_amcl_persistent_stale_routes_to_amcl_refresh_not_odom_fallback(self):
        node = self.real_amcl_node()
        first = failed_lookup("stale_transform", 1.2, 98.8)
        retry = failed_lookup("stale_transform", 1.3, 98.8)
        recovered = PoseLookupResult(Pose2D(1.0, 2.0, 0.1), stamp_sec=99.9)
        node._current_pose_lookup = Mock(side_effect=(first, retry))
        node._is_real_amcl_runtime = Mock(return_value=True)
        node.publish_zero = Mock()
        node._drain_runtime_callbacks = Mock(return_value={"elapsed_sec": 0.18})
        node._tf_edge_sample = Mock(
            side_effect=(
                TfEdgeSample("map", "odom", 100.5),
                TfEdgeSample("map", "odom", 100.5),
                TfEdgeSample("odom", "base_footprint", 99.9),
            )
        )
        node._real_amcl_stale_tf_recovery = Mock(return_value=recovered)
        node._simulation_odom_fallback_after_stale_retry = Mock()

        result = node._current_pose_lookup_with_stale_recovery()

        self.assertIs(result, recovered)
        node.publish_zero.assert_called_once_with()
        node._real_amcl_stale_tf_recovery.assert_called_once()
        node._simulation_odom_fallback_after_stale_retry.assert_not_called()

    def test_real_amcl_refresh_calls_service_once_under_zero_and_requires_new_tf(self):
        node = self.real_amcl_node()
        events = []
        future = SimpleNamespace(done=lambda: True, exception=lambda: None)
        client = SimpleNamespace(
            service_is_ready=lambda: True,
            call_async=Mock(
                side_effect=lambda _request: (
                    events.append("service_request") or future
                )
            ),
        )
        node.runtime_nomotion_update_client = client
        node.publish_zero = Mock(side_effect=lambda: events.append("zero"))
        fresh_lookup = PoseLookupResult(
            Pose2D(1.0, 2.0, 0.1),
            stamp_sec=99.9,
        )
        node._current_pose_lookup = Mock(return_value=fresh_lookup)
        node._tf_edge_sample = Mock(
            side_effect=(
                TfEdgeSample("map", "odom", 100.8),
                TfEdgeSample("odom", "base_footprint", 99.95),
                TfEdgeSample("map", "odom", 100.8),
                TfEdgeSample("odom", "base_footprint", 99.95),
            )
        )

        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.Empty",
            SimpleNamespace(Request=lambda: object()),
        ), patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.rclpy",
            SimpleNamespace(ok=lambda: True),
        ), patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.time.monotonic",
            return_value=0.0,
        ):
            result = node._real_amcl_stale_tf_recovery(
                first_lookup=failed_lookup("stale_transform", 1.2, 98.8),
                retry_lookup=failed_lookup("stale_transform", 1.3, 98.8),
                callback_drain={"elapsed_sec": 0.18},
                map_to_odom_before=TfEdgeSample("map", "odom", 100.5),
                map_to_odom_retry=TfEdgeSample("map", "odom", 100.5),
                odom_to_base_retry=TfEdgeSample(
                    "odom",
                    "base_footprint",
                    99.9,
                ),
            )

        self.assertEqual(result.pose, fresh_lookup.pose)
        self.assertTrue(result.details["accepted"])
        self.assertFalse(result.details["motion_authorized"])
        self.assertTrue(result.details["requires_route_tube_readmission"])
        self.assertTrue(result.details["zero_cycle_handoff_completed"])
        self.assertEqual(client.call_async.call_count, 1)
        self.assertLess(events.index("zero"), events.index("service_request"))
        self.assertGreaterEqual(events.count("zero"), 3)
        emitted_names = [
            call.args[0].event_name
            for call in node._emit_route_update.call_args_list
        ]
        self.assertEqual(
            emitted_names,
            [
                "real_amcl_stale_tf_recovery_started",
                "real_amcl_stale_tf_recovery_recovered",
            ],
        )

    def test_real_amcl_refresh_service_unavailable_is_terminal(self):
        node = self.real_amcl_node()
        client = SimpleNamespace(
            service_is_ready=lambda: False,
            call_async=Mock(),
        )
        node.runtime_nomotion_update_client = client
        node.publish_zero = Mock()

        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.time.monotonic",
            return_value=0.0,
        ):
            result = node._real_amcl_stale_tf_recovery(
                first_lookup=failed_lookup("stale_transform", 1.2, 98.8),
                retry_lookup=failed_lookup("stale_transform", 1.3, 98.8),
                callback_drain={"elapsed_sec": 0.18},
                map_to_odom_before=TfEdgeSample("map", "odom", 100.5),
                map_to_odom_retry=TfEdgeSample("map", "odom", 100.5),
                odom_to_base_retry=TfEdgeSample(
                    "odom",
                    "base_footprint",
                    99.9,
                ),
            )

        self.assertIsNone(result.pose)
        self.assertEqual(
            result.details["reason"],
            "nomotion_update_service_unavailable",
        )
        self.assertTrue(result.details["fail_closed"])
        client.call_async.assert_not_called()

    def test_stationarity_collector_spans_adjacent_twenty_hz_odom_samples(self):
        node = self.real_amcl_node()
        samples = (
            OdomStationaritySample(1, 100.0, 1.0, 2.0, 0.1, 0.0, 0.0),
            OdomStationaritySample(2, 100.05, 1.0, 2.0, 0.1, 0.0, 0.0),
            OdomStationaritySample(3, 100.10, 1.0, 2.0, 0.1, 0.0, 0.0),
        )
        node._odom_stationarity_sample = Mock(side_effect=samples)
        node._ros_now_sec = Mock(return_value=100.10)
        node.publish_zero = Mock()

        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.rclpy",
            SimpleNamespace(ok=lambda: True),
        ), patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.time.monotonic",
            return_value=0.0,
        ):
            decision, evidence = (
                SimpleWaypointFollowerNode._wait_for_stationary_odom_pair(
                    node,
                    deadline_monotonic=1.0,
                )
            )

        self.assertIsNotNone(decision)
        self.assertTrue(decision.accepted)
        self.assertAlmostEqual(decision.sample_separation_sec, 0.10)
        self.assertEqual(len(evidence["attempts"]), 2)
        self.assertEqual(node.publish_zero.call_count, 2)

    def test_real_amcl_refresh_service_timeout_is_terminal_after_one_request(self):
        node = self.real_amcl_node()
        future = SimpleNamespace(done=lambda: False)
        client = SimpleNamespace(
            service_is_ready=lambda: True,
            call_async=Mock(return_value=future),
        )
        node.runtime_nomotion_update_client = client
        node.publish_zero = Mock()

        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.Empty",
            SimpleNamespace(Request=lambda: object()),
        ), patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.rclpy",
            SimpleNamespace(ok=lambda: True),
        ), patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.time.monotonic",
            side_effect=(0.0, 0.0, 0.0, 2.1),
        ):
            result = node._real_amcl_stale_tf_recovery(
                first_lookup=failed_lookup("stale_transform", 1.2, 98.8),
                retry_lookup=failed_lookup("stale_transform", 1.3, 98.8),
                callback_drain={"elapsed_sec": 0.18},
                map_to_odom_before=TfEdgeSample("map", "odom", 100.5),
                map_to_odom_retry=TfEdgeSample("map", "odom", 100.5),
                odom_to_base_retry=TfEdgeSample(
                    "odom",
                    "base_footprint",
                    99.9,
                ),
            )

        self.assertIsNone(result.pose)
        self.assertEqual(
            result.details["reason"],
            "nomotion_update_service_timeout",
        )
        self.assertEqual(client.call_async.call_count, 1)
        self.assertGreaterEqual(node.publish_zero.call_count, 2)

    def test_real_amcl_refresh_ownership_change_after_service_is_terminal(self):
        node = self.real_amcl_node()
        future = SimpleNamespace(done=lambda: True, exception=lambda: None)
        client = SimpleNamespace(
            service_is_ready=lambda: True,
            call_async=Mock(return_value=future),
        )
        node.runtime_nomotion_update_client = client
        node.publish_zero = Mock()
        node._cmd_vel_ownership_failure = Mock(
            side_effect=("", "competing cmd_vel publisher")
        )
        node._current_pose_lookup = Mock(
            return_value=PoseLookupResult(
                Pose2D(1.0, 2.0, 0.1),
                stamp_sec=99.9,
            )
        )
        node._tf_edge_sample = Mock(
            side_effect=(
                TfEdgeSample("map", "odom", 100.8),
                TfEdgeSample("odom", "base_footprint", 99.95),
            )
        )

        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.Empty",
            SimpleNamespace(Request=lambda: object()),
        ), patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.rclpy",
            SimpleNamespace(ok=lambda: True),
        ), patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.time.monotonic",
            return_value=0.0,
        ):
            result = node._real_amcl_stale_tf_recovery(
                first_lookup=failed_lookup("stale_transform", 1.2, 98.8),
                retry_lookup=failed_lookup("stale_transform", 1.3, 98.8),
                callback_drain={"elapsed_sec": 0.18},
                map_to_odom_before=TfEdgeSample("map", "odom", 100.5),
                map_to_odom_retry=TfEdgeSample("map", "odom", 100.5),
                odom_to_base_retry=TfEdgeSample(
                    "odom",
                    "base_footprint",
                    99.9,
                ),
            )

        self.assertIsNone(result.pose)
        self.assertEqual(
            result.details["reason"],
            "cmd_vel_owner_not_exclusive",
        )
        self.assertEqual(client.call_async.call_count, 1)

    def test_future_and_lookup_exception_do_not_enter_recovery(self):
        for failure in (
            failed_lookup("future_transform", -0.3),
            failed_lookup("lookup_exception"),
        ):
            with self.subTest(reason=failure.details["reason"]):
                node = self.bare_node()
                node._current_pose_lookup = Mock(return_value=failure)
                node.publish_zero = Mock()
                node._drain_runtime_callbacks = Mock()

                result = node._current_pose_lookup_with_stale_recovery()

                self.assertIs(result, failure)
                node._current_pose_lookup.assert_called_once_with()
                node.publish_zero.assert_not_called()
                node._drain_runtime_callbacks.assert_not_called()

    def test_callback_drain_is_bounded_by_deadline_and_callback_cap(self):
        node = self.bare_node()
        spin_once = Mock()
        fake_rclpy = SimpleNamespace(spin_once=spin_once)

        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.rclpy",
            fake_rclpy,
        ), patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.time.monotonic",
            side_effect=(0.0, 0.0, 0.05, 0.10, 0.15, 0.18, 0.18),
        ):
            deadline_drain = node._drain_runtime_callbacks(
                max_callbacks=48,
                max_duration_sec=0.18,
                spin_timeout_sec=0.005,
            )

        self.assertEqual(spin_once.call_count, 4)
        self.assertEqual(deadline_drain["spin_count"], 4)
        self.assertTrue(deadline_drain["deadline_reached"])
        self.assertEqual(deadline_drain["elapsed_sec"], 0.18)

        spin_once.reset_mock()
        with patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.rclpy",
            fake_rclpy,
        ), patch(
            "scripts.aufgabe04.navigation.waypoint_follower.runtime.time.monotonic",
            return_value=0.0,
        ):
            capped_drain = node._drain_runtime_callbacks(
                max_callbacks=48,
                max_duration_sec=0.18,
                spin_timeout_sec=0.0,
            )

        self.assertEqual(spin_once.call_count, 48)
        self.assertEqual(capped_drain["spin_count"], 48)
        self.assertFalse(capped_drain["deadline_reached"])


if __name__ == "__main__":
    unittest.main()
