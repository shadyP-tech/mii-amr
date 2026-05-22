import argparse
import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "aufgabe03"))

import arena_active_spin as active_spin  # noqa: E402
import arena_geometry_localizer as arena  # noqa: E402


def fake_scan(clearance=1.0):
    ranges = [clearance] * 360
    return argparse.Namespace(
        ranges=ranges,
        angle_min=-math.pi,
        angle_increment=(2.0 * math.pi) / len(ranges),
        range_min=0.05,
        range_max=5.0,
    )


def fake_odom(yaw_rad=0.0):
    return argparse.Namespace(
        pose=argparse.Namespace(
            pose=argparse.Namespace(
                position=argparse.Namespace(x=0.0, y=0.0, z=0.0),
                orientation=argparse.Namespace(
                    x=0.0,
                    y=0.0,
                    z=math.sin(yaw_rad / 2.0),
                    w=math.cos(yaw_rad / 2.0),
                ),
            )
        )
    )


def fake_candidate(axis_side, range_m, heater_score=0.0):
    return arena.ShortWallClassification(
        wall_type=arena.WALL_UNKNOWN,
        reason="pairwise_profile_relative_heater_score_too_low",
        observed_axis_side=axis_side,
        short_wall_candidate_range_m=range_m,
        heater_profile_score=heater_score,
        profile_features={"validity_failed_reason": None},
    )


def fake_pose_not_unique_result(
    negative_range=0.7,
    positive_range=3.2,
    axis_angle_rad=0.0,
    normal_angle_rad=None,
    lateral_offset_m=None,
):
    return arena.ArenaGeometryResult(
        success=False,
        failure_reason="pose_not_unique",
        pose_unique=False,
        yaw_ambiguity_resolved=False,
        estimated_pose_prior=None,
        estimated_covariance=None,
        long_wall_fit=arena.LongWallFit(
            ok=True,
            reason="ok",
            axis_angle_rad=axis_angle_rad,
            normal_angle_rad=normal_angle_rad,
            lateral_offset_m=lateral_offset_m,
        ),
        short_wall_classification=arena.ShortWallClassification(
            arena.WALL_UNKNOWN,
            "pairwise_profile_relative_heater_score_too_low",
        ),
        short_wall_candidates={
            "axis_negative": fake_candidate("axis_negative", negative_range, 0.40),
            "axis_positive": fake_candidate("axis_positive", positive_range, 0.50),
        },
        diagnostics={},
    )


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class FakeTwist:
    def __init__(self):
        self.linear = argparse.Namespace(x=0.0, y=0.0, z=0.0)
        self.angular = argparse.Namespace(x=0.0, y=0.0, z=0.0)


class FakeNode:
    def __init__(self):
        self.subscriptions = {}
        self.publisher_count = 1

    def create_subscription(self, _msg_type, topic, callback, _qos):
        self.subscriptions[topic] = callback
        return callback

    def count_publishers(self, _topic):
        return self.publisher_count


class FakeRclpy:
    def __init__(self, node, time_box):
        self.node = node
        self.time_box = time_box
        self.yaw = 0.0

    def ok(self):
        return True

    def spin_once(self, _node, timeout_sec=0.0):
        self.time_box["now"] += timeout_sec
        self.yaw = active_spin.normalize_angle_rad(self.yaw + 0.20)
        if "/odom" in self.node.subscriptions:
            self.node.subscriptions["/odom"](fake_odom(self.yaw))
        if "/scan" in self.node.subscriptions:
            self.node.subscriptions["/scan"](fake_scan())


class PromptDelayRclpy:
    def __init__(self, node, time_box):
        self.node = node
        self.time_box = time_box
        self.yaw = 0.0

    def ok(self):
        return True

    def spin_once(self, _node, timeout_sec=0.0):
        self.time_box["now"] += timeout_sec
        if timeout_sec > 0.0:
            if "/odom" in self.node.subscriptions:
                self.node.subscriptions["/odom"](fake_odom(self.yaw))
            if "/scan" in self.node.subscriptions:
                self.node.subscriptions["/scan"](fake_scan())


class ArenaActiveSpinTest(unittest.TestCase):
    def test_diagnostics_json_accepts_array_like_values(self):
        class FakeArray:
            def tolist(self):
                return [[1.0, 2.0], [3.0, 4.0]]

        class FakeScalar:
            def item(self):
                return 0.25

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "diag.json"

            active_spin.write_diagnostics_json(
                path,
                {
                    "matrix": FakeArray(),
                    "scalar": FakeScalar(),
                    "nested": {"tuple": (FakeScalar(),)},
                },
            )
            diagnostics = json.loads(path.read_text())

        self.assertEqual(diagnostics["matrix"], [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(diagnostics["scalar"], 0.25)
        self.assertEqual(diagnostics["nested"]["tuple"], [0.25])

    def test_shortest_angle_delta_handles_wraparound(self):
        delta = active_spin.shortest_angle_delta_rad(
            math.radians(170.0),
            math.radians(-170.0),
        )

        self.assertAlmostEqual(math.degrees(delta), 20.0)

    def test_spin_direction_changes_command_sign(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            node = FakeNode()
            publisher = FakePublisher()
            config = active_spin.ArenaActiveSpinConfig(
                run_id="direction_test",
                diagnostics_path=Path(tmpdir) / "diag.json",
                spin_direction="cw",
                require_operator_confirmation=False,
            )
            session = active_spin.ArenaActiveSpinSession(
                node,
                config,
                FakeRclpy(node, {"now": 0.0}),
                FakeTwist,
                object,
                object,
                None,
                sleep_fn=lambda _delay: None,
            )

            session.publish_spin_command(publisher)

        self.assertLess(publisher.messages[-1].angular.z, 0.0)

    def test_clearance_checks_named_sectors(self):
        scan = fake_scan(clearance=1.0)
        scan.ranges[180] = 0.10
        config = active_spin.ArenaActiveSpinConfig(
            run_id="clearance_test",
            diagnostics_path=Path("unused.json"),
            require_operator_confirmation=False,
        )

        clearance = active_spin.evaluate_clearance(scan, config)

        self.assertFalse(clearance.ok)
        self.assertEqual(clearance.reason, "front_clearance_below_limit")

    def test_stop_repeatedly_publishes_multiple_zero_twists(self):
        publisher = FakePublisher()

        active_spin.stop_repeatedly(
            publisher,
            FakeTwist,
            sleep_fn=lambda _delay: None,
            count=5,
        )

        self.assertEqual(len(publisher.messages), 5)
        self.assertTrue(all(msg.angular.z == 0.0 for msg in publisher.messages))

    def test_center_reposition_action_moves_away_from_nearest_short_wall(self):
        config = active_spin.ArenaActiveSpinConfig(
            run_id="reposition_action_test",
            diagnostics_path=Path("unused.json"),
            enable_center_reposition=True,
            require_operator_confirmation=False,
        )
        result = fake_pose_not_unique_result(
            negative_range=0.72,
            positive_range=3.13,
            axis_angle_rad=0.25,
        )

        action = active_spin.choose_center_reposition_action(
            result,
            config,
            origin_yaw_rad=0.10,
        )

        self.assertTrue(action.ok)
        self.assertEqual(action.nearest_axis_side, "axis_negative")
        self.assertEqual(action.away_axis_side, "axis_positive")
        self.assertAlmostEqual(action.planned_distance_m, 0.93)
        self.assertAlmostEqual(action.local_heading_rad, 0.25)
        self.assertAlmostEqual(action.odom_heading_rad, 0.35)
        self.assertEqual([step.kind for step in action.steps], ["longitudinal"])

    def test_center_reposition_action_clamps_step_and_rejects_near_target(self):
        config = active_spin.ArenaActiveSpinConfig(
            run_id="reposition_clamp_test",
            diagnostics_path=Path("unused.json"),
            enable_center_reposition=True,
            require_operator_confirmation=False,
            center_reposition_max_step_m=0.50,
        )
        far_action = active_spin.choose_center_reposition_action(
            fake_pose_not_unique_result(negative_range=0.20, positive_range=3.70),
            config,
        )
        near_action = active_spin.choose_center_reposition_action(
            fake_pose_not_unique_result(negative_range=1.50, positive_range=2.40),
            config,
        )

        self.assertTrue(far_action.ok)
        self.assertAlmostEqual(far_action.planned_distance_m, 0.50)
        self.assertFalse(near_action.ok)
        self.assertEqual(
            near_action.reason,
            "center_reposition_not_useful_already_near_target",
        )

    def test_center_reposition_action_skips_lateral_when_offset_small(self):
        config = active_spin.ArenaActiveSpinConfig(
            run_id="reposition_lateral_skip_test",
            diagnostics_path=Path("unused.json"),
            enable_center_reposition=True,
            require_operator_confirmation=False,
        )

        action = active_spin.choose_center_reposition_action(
            fake_pose_not_unique_result(
                negative_range=0.72,
                positive_range=3.13,
                axis_angle_rad=0.25,
                normal_angle_rad=1.25,
                lateral_offset_m=0.10,
            ),
            config,
        )

        self.assertTrue(action.ok)
        self.assertEqual([step.kind for step in action.steps], ["longitudinal"])
        self.assertTrue(action.lateral_step_skipped)
        self.assertEqual(
            action.lateral_skip_reason,
            "center_reposition_lateral_offset_within_threshold",
        )

    def test_center_reposition_action_adds_lateral_step_for_positive_offset(self):
        config = active_spin.ArenaActiveSpinConfig(
            run_id="reposition_lateral_positive_test",
            diagnostics_path=Path("unused.json"),
            enable_center_reposition=True,
            require_operator_confirmation=False,
        )

        action = active_spin.choose_center_reposition_action(
            fake_pose_not_unique_result(
                negative_range=0.72,
                positive_range=3.13,
                axis_angle_rad=0.25,
                normal_angle_rad=math.pi / 2.0,
                lateral_offset_m=0.50,
            ),
            config,
            origin_yaw_rad=0.10,
        )

        self.assertTrue(action.ok)
        self.assertEqual([step.kind for step in action.steps], ["longitudinal", "lateral"])
        lateral = action.steps[1]
        self.assertAlmostEqual(lateral.planned_distance_m, 0.40)
        self.assertAlmostEqual(
            lateral.local_heading_rad,
            active_spin.normalize_angle_rad(math.pi / 2.0 + math.pi),
        )
        self.assertAlmostEqual(
            lateral.odom_heading_rad,
            active_spin.normalize_angle_rad(0.10 + math.pi / 2.0 + math.pi),
        )
        self.assertFalse(action.lateral_step_skipped)

    def test_center_reposition_action_adds_lateral_step_for_negative_offset(self):
        config = active_spin.ArenaActiveSpinConfig(
            run_id="reposition_lateral_negative_test",
            diagnostics_path=Path("unused.json"),
            enable_center_reposition=True,
            require_operator_confirmation=False,
        )

        action = active_spin.choose_center_reposition_action(
            fake_pose_not_unique_result(
                negative_range=0.72,
                positive_range=3.13,
                axis_angle_rad=0.25,
                normal_angle_rad=math.pi / 2.0,
                lateral_offset_m=-0.50,
            ),
            config,
            origin_yaw_rad=0.10,
        )

        self.assertTrue(action.ok)
        lateral = action.steps[1]
        self.assertAlmostEqual(lateral.local_heading_rad, math.pi / 2.0)
        self.assertAlmostEqual(lateral.odom_heading_rad, 0.10 + math.pi / 2.0)

    def test_center_reposition_action_clamps_lateral_step(self):
        config = active_spin.ArenaActiveSpinConfig(
            run_id="reposition_lateral_clamp_test",
            diagnostics_path=Path("unused.json"),
            enable_center_reposition=True,
            require_operator_confirmation=False,
            center_reposition_lateral_max_step_m=0.35,
        )

        action = active_spin.choose_center_reposition_action(
            fake_pose_not_unique_result(
                negative_range=1.50,
                positive_range=2.40,
                normal_angle_rad=math.pi / 2.0,
                lateral_offset_m=1.00,
            ),
            config,
        )

        self.assertTrue(action.ok)
        self.assertEqual([step.kind for step in action.steps], ["lateral"])
        self.assertAlmostEqual(action.steps[0].planned_distance_m, 0.35)
        self.assertAlmostEqual(action.lateral_planned_distance_m, 0.35)

    def test_center_reposition_action_rejects_missing_lateral_normal(self):
        config = active_spin.ArenaActiveSpinConfig(
            run_id="reposition_lateral_missing_normal_test",
            diagnostics_path=Path("unused.json"),
            enable_center_reposition=True,
            require_operator_confirmation=False,
        )

        action = active_spin.choose_center_reposition_action(
            fake_pose_not_unique_result(
                negative_range=0.72,
                positive_range=3.13,
                normal_angle_rad=None,
                lateral_offset_m=0.50,
            ),
            config,
        )

        self.assertFalse(action.ok)
        self.assertEqual(action.reason, "center_reposition_missing_lateral_normal")

    def test_center_reposition_action_rejects_invalid_range_sum(self):
        config = active_spin.ArenaActiveSpinConfig(
            run_id="reposition_range_test",
            diagnostics_path=Path("unused.json"),
            enable_center_reposition=True,
            require_operator_confirmation=False,
        )

        action = active_spin.choose_center_reposition_action(
            fake_pose_not_unique_result(negative_range=0.7, positive_range=2.0),
            config,
        )

        self.assertFalse(action.ok)
        self.assertEqual(action.reason, "center_reposition_range_sum_invalid")

    def test_center_reposition_refreshes_inputs_between_turn_and_drive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            node = FakeNode()
            publisher = FakePublisher()
            time_box = {"now": 0.0}
            config = active_spin.ArenaActiveSpinConfig(
                run_id="reposition_handoff_test",
                diagnostics_path=Path(tmpdir) / "diag.json",
                require_operator_confirmation=False,
            )
            session = active_spin.ArenaActiveSpinSession(
                node,
                config,
                FakeRclpy(node, time_box),
                FakeTwist,
                object,
                object,
                None,
                sleep_fn=lambda delay: time_box.__setitem__("now", time_box["now"] + delay),
            )
            events = []

            def freshen():
                events.append("wait")
                session.latest_scan = fake_scan()
                session.latest_scan_received_sec = session.now()
                session.latest_odom_pose = arena.Pose2D()
                session.latest_odom_yaw_rad = 0.0
                session.latest_odom_received_sec = session.now()

            session.wait_for_fresh_inputs = freshen
            session.refresh_fresh_inputs_after_prompt = lambda: None
            session.turn_to_heading = lambda _publisher, _heading: events.append("turn")
            session.drive_forward = lambda _publisher, _distance: events.append("drive") or 0.5
            action = active_spin.CenterRepositionAction(
                ok=True,
                reason="center_reposition_toward_arena_center",
                planned_distance_m=0.5,
                odom_heading_rad=0.0,
            )

            with contextlib.redirect_stdout(io.StringIO()):
                session.execute_center_reposition(publisher, action)

        self.assertEqual(events, ["wait", "turn", "wait", "drive"])

    def test_center_reposition_executes_two_steps_with_fresh_handoffs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            node = FakeNode()
            publisher = FakePublisher()
            time_box = {"now": 0.0}
            config = active_spin.ArenaActiveSpinConfig(
                run_id="reposition_two_step_test",
                diagnostics_path=Path(tmpdir) / "diag.json",
                require_operator_confirmation=False,
            )
            session = active_spin.ArenaActiveSpinSession(
                node,
                config,
                FakeRclpy(node, time_box),
                FakeTwist,
                object,
                object,
                None,
                sleep_fn=lambda delay: time_box.__setitem__("now", time_box["now"] + delay),
            )
            events = []

            def freshen():
                events.append("wait")
                session.latest_scan = fake_scan()
                session.latest_scan_received_sec = session.now()
                session.latest_odom_pose = arena.Pose2D()
                session.latest_odom_yaw_rad = 0.0
                session.latest_odom_received_sec = session.now()

            session.wait_for_fresh_inputs = freshen
            session.refresh_fresh_inputs_after_prompt = lambda: None
            session.turn_to_heading = lambda _publisher, _heading: events.append("turn")
            session.drive_forward = lambda _publisher, distance: events.append("drive") or distance
            action = active_spin.CenterRepositionAction(
                ok=True,
                reason="center_reposition_toward_arena_center",
                steps=(
                    active_spin.CenterRepositionStep(
                        "longitudinal",
                        "center_reposition_away_from_nearest_short_wall",
                        0.50,
                        0.0,
                        0.0,
                    ),
                    active_spin.CenterRepositionStep(
                        "lateral",
                        "center_reposition_reduce_lateral_offset",
                        0.30,
                        math.pi / 2.0,
                        math.pi / 2.0,
                    ),
                ),
            )

            with contextlib.redirect_stdout(io.StringIO()):
                record = session.execute_center_reposition(publisher, action)

        self.assertEqual(events, ["wait", "turn", "wait", "drive", "wait", "turn", "wait", "drive"])
        self.assertAlmostEqual(record["driven_distance_m"], 0.80)
        self.assertEqual(len(record["steps"]), 2)
        self.assertGreaterEqual(len(publisher.messages), 40)

    def test_center_reposition_second_step_failure_propagates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            node = FakeNode()
            publisher = FakePublisher()
            time_box = {"now": 0.0}
            config = active_spin.ArenaActiveSpinConfig(
                run_id="reposition_two_step_stale_test",
                diagnostics_path=Path(tmpdir) / "diag.json",
                require_operator_confirmation=False,
            )
            session = active_spin.ArenaActiveSpinSession(
                node,
                config,
                FakeRclpy(node, time_box),
                FakeTwist,
                object,
                object,
                None,
                sleep_fn=lambda delay: time_box.__setitem__("now", time_box["now"] + delay),
            )

            def freshen():
                session.latest_scan = fake_scan()
                session.latest_scan_received_sec = session.now()
                session.latest_odom_pose = arena.Pose2D()
                session.latest_odom_yaw_rad = 0.0
                session.latest_odom_received_sec = session.now()

            drive_calls = {"count": 0}

            def drive(_publisher, distance):
                drive_calls["count"] += 1
                if drive_calls["count"] == 2:
                    raise RuntimeError("stale_scan_during_reposition_drive")
                return distance

            session.wait_for_fresh_inputs = freshen
            session.refresh_fresh_inputs_after_prompt = lambda: None
            session.turn_to_heading = lambda _publisher, _heading: None
            session.drive_forward = drive
            action = active_spin.CenterRepositionAction(
                ok=True,
                reason="center_reposition_toward_arena_center",
                steps=(
                    active_spin.CenterRepositionStep("longitudinal", "long", 0.50, 0.0, 0.0),
                    active_spin.CenterRepositionStep("lateral", "lat", 0.30, 1.0, 1.0),
                ),
            )

            with self.assertRaisesRegex(RuntimeError, "stale_scan_during_reposition_drive"):
                with contextlib.redirect_stdout(io.StringIO()):
                    session.execute_center_reposition(publisher, action)

    def test_localizer_exception_writes_diagnostics_and_stops(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            node = FakeNode()
            publisher = FakePublisher()
            time_box = {"now": 0.0}
            diagnostics_path = Path(tmpdir) / "arena_active_result.json"
            config = active_spin.ArenaActiveSpinConfig(
                run_id="exception_test",
                diagnostics_path=diagnostics_path,
                spin_complete_tolerance_deg=359.0,
                min_scan_samples=1,
                max_spin_sec=2.0,
                require_operator_confirmation=False,
                stop_settle_sec=0.0,
                arena_config=arena.ArenaGeometryConfig(),
            )

            def failing_analyze(*_args, **_kwargs):
                raise RuntimeError("localizer exploded")

            with contextlib.redirect_stdout(io.StringIO()):
                result = active_spin.run_arena_active_spin(
                    node,
                    publisher,
                    config,
                    FakeRclpy(node, time_box),
                    FakeTwist,
                    object,
                    object,
                    None,
                    sleep_fn=lambda delay: time_box.__setitem__("now", time_box["now"] + delay),
                    analyze_fn=failing_analyze,
                )
            diagnostics = json.loads(diagnostics_path.read_text())

        self.assertFalse(result.success)
        self.assertIn("localizer exploded", result.failure_reason)
        self.assertEqual(diagnostics["exception"]["type"], "RuntimeError")
        self.assertGreaterEqual(len(publisher.messages), active_spin.DEFAULT_STOP_COUNT * 2)
        self.assertFalse(diagnostics["initialpose"]["published"])

    def test_prompt_delay_refreshes_scan_before_first_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            node = FakeNode()
            publisher = FakePublisher()
            time_box = {"now": 0.0}
            config = active_spin.ArenaActiveSpinConfig(
                run_id="prompt_delay_test",
                diagnostics_path=Path(tmpdir) / "diag.json",
                require_operator_confirmation=True,
                spin_complete_tolerance_deg=359.0,
                min_scan_samples=1,
                max_spin_sec=2.0,
                stop_settle_sec=0.0,
                arena_config=arena.ArenaGeometryConfig(),
            )
            session = active_spin.ArenaActiveSpinSession(
                node,
                config,
                PromptDelayRclpy(node, time_box),
                FakeTwist,
                object,
                object,
                None,
                input_fn=lambda _prompt: time_box.__setitem__("now", time_box["now"] + 1.0),
                sleep_fn=lambda delay: time_box.__setitem__("now", time_box["now"] + delay),
                analyze_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("stop after spin")
                ),
            )

            with contextlib.redirect_stdout(io.StringIO()):
                result = session.run(publisher)

        self.assertNotEqual(result.failure_reason, "stale_scan_during_spin")
        self.assertTrue(any(message.angular.z != 0.0 for message in publisher.messages))


if __name__ == "__main__":
    unittest.main()
