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


def fake_scan_with_side_clearance(left=1.0, right=1.0, front=1.0, rear=1.0):
    scan = fake_scan(clearance=2.0)
    for index in range(len(scan.ranges)):
        angle_rad = scan.angle_min + index * scan.angle_increment
        angle_deg = math.degrees(active_spin.normalize_angle_rad(angle_rad))
        if -30.0 <= angle_deg <= 30.0:
            scan.ranges[index] = front
        elif 60.0 <= angle_deg <= 120.0:
            scan.ranges[index] = left
        elif -120.0 <= angle_deg <= -60.0:
            scan.ranges[index] = right
        elif 150.0 <= angle_deg <= 180.0 or -180.0 <= angle_deg <= -150.0:
            scan.ranges[index] = rear
    return scan


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


def fake_candidate(axis_side, range_m, heater_score=0.0, validity_failed_reason=None):
    return arena.ShortWallClassification(
        wall_type=arena.WALL_UNKNOWN,
        reason="pairwise_profile_relative_heater_score_too_low",
        observed_axis_side=axis_side,
        short_wall_candidate_range_m=range_m,
        heater_profile_score=heater_score,
        profile_features={"validity_failed_reason": validity_failed_reason},
        validity_failed_reason=validity_failed_reason,
    )


def fake_pose_not_unique_result(
    negative_range=0.7,
    positive_range=3.2,
    axis_angle_rad=0.0,
    normal_angle_rad=None,
    lateral_offset_m=None,
    negative_heater_score=0.40,
    positive_heater_score=0.50,
    negative_validity_failed_reason=None,
    positive_validity_failed_reason=None,
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
            "axis_negative": fake_candidate(
                "axis_negative",
                negative_range,
                negative_heater_score,
                negative_validity_failed_reason,
            ),
            "axis_positive": fake_candidate(
                "axis_positive",
                positive_range,
                positive_heater_score,
                positive_validity_failed_reason,
            ),
        },
        diagnostics={},
    )


def fake_success_result():
    return arena.ArenaGeometryResult(
        success=True,
        failure_reason="",
        pose_unique=True,
        yaw_ambiguity_resolved=True,
        estimated_pose_prior=arena.Pose2D(0.1, 0.2, 30.0),
        estimated_covariance={"x_m2": 0.04, "y_m2": 0.01, "yaw_rad2": 0.01},
        long_wall_fit=arena.LongWallFit(ok=True, reason="ok", axis_angle_rad=0.0),
        short_wall_classification=arena.ShortWallClassification(
            arena.WALL_HEATER,
            "pairwise_profile_relative_heater_valid",
            observed_axis_side="axis_negative",
        ),
        short_wall_candidates={},
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

    def test_center_reposition_lateral_regression_for_real_negative_offset(self):
        config = active_spin.ArenaActiveSpinConfig(
            run_id="reposition_real_lateral_regression_test",
            diagnostics_path=Path("unused.json"),
            enable_center_reposition=True,
            require_operator_confirmation=False,
            center_reposition_target_nearest_short_wall_range_m=1.75,
        )

        action = active_spin.choose_center_reposition_action(
            fake_pose_not_unique_result(
                negative_range=3.0743892214964075,
                positive_range=0.78081798119409,
                axis_angle_rad=active_spin.normalize_angle_rad(-0.8028514559173914),
                normal_angle_rad=0.7679448708775052,
                lateral_offset_m=-0.3872003294910733,
            ),
            config,
            origin_yaw_rad=1.7085612863302231,
        )

        self.assertTrue(action.ok)
        self.assertEqual([step.kind for step in action.steps], ["longitudinal", "lateral"])
        lateral = action.steps[1]
        self.assertAlmostEqual(lateral.planned_distance_m, 0.2872003294910733)
        self.assertAlmostEqual(lateral.local_heading_rad, 0.7679448708775052)
        self.assertAlmostEqual(
            lateral.odom_heading_rad,
            active_spin.normalize_angle_rad(1.7085612863302231 + 0.7679448708775052),
        )
        self.assertTrue(lateral.dynamic_heading)
        self.assertEqual(lateral.dynamic_heading_source, "live_side_clearance")

    def test_dynamic_lateral_heading_turns_toward_more_open_side(self):
        left_open = active_spin.dynamic_lateral_heading_from_scan(
            fake_scan_with_side_clearance(left=1.4, right=0.5),
            current_yaw_rad=0.2,
        )
        right_open = active_spin.dynamic_lateral_heading_from_scan(
            fake_scan_with_side_clearance(left=0.5, right=1.4),
            current_yaw_rad=0.2,
        )

        self.assertEqual(left_open["direction"], "left")
        self.assertAlmostEqual(left_open["odom_heading_rad"], 0.2 + math.pi / 2.0)
        self.assertEqual(right_open["direction"], "right")
        self.assertAlmostEqual(right_open["odom_heading_rad"], 0.2 - math.pi / 2.0)

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

    def test_heater_approach_action_moves_toward_suspected_heater(self):
        config = active_spin.ArenaActiveSpinConfig(
            run_id="heater_approach_test",
            diagnostics_path=Path("unused.json"),
            enable_center_reposition=True,
            require_operator_confirmation=False,
        )
        result = fake_pose_not_unique_result(
            negative_range=2.31,
            positive_range=1.55,
            axis_angle_rad=0.25,
            negative_heater_score=0.581,
            positive_heater_score=0.143,
        )

        action = active_spin.choose_heater_approach_reposition_action(
            result,
            config,
            origin_yaw_rad=0.10,
        )

        self.assertTrue(action.ok)
        self.assertEqual(action.reason, "heater_approach_toward_suspected_heater")
        self.assertEqual(action.suspected_heater_axis_side, "axis_negative")
        self.assertAlmostEqual(action.heater_profile_delta, 0.438)
        self.assertAlmostEqual(action.planned_distance_m, 1.10)
        self.assertEqual([step.kind for step in action.steps], ["heater_approach"])
        self.assertAlmostEqual(
            action.steps[0].local_heading_rad,
            active_spin.normalize_angle_rad(0.25 + math.pi),
        )
        self.assertAlmostEqual(
            action.steps[0].odom_heading_rad,
            active_spin.normalize_angle_rad(0.10 + 0.25 + math.pi),
        )

    def test_heater_approach_action_rejects_weak_directional_evidence(self):
        config = active_spin.ArenaActiveSpinConfig(
            run_id="heater_approach_weak_test",
            diagnostics_path=Path("unused.json"),
            enable_center_reposition=True,
            require_operator_confirmation=False,
        )

        low_score = active_spin.choose_heater_approach_reposition_action(
            fake_pose_not_unique_result(
                negative_range=2.31,
                positive_range=1.55,
                negative_heater_score=0.49,
                positive_heater_score=0.10,
            ),
            config,
        )
        high_opposite = active_spin.choose_heater_approach_reposition_action(
            fake_pose_not_unique_result(
                negative_range=2.31,
                positive_range=1.55,
                negative_heater_score=0.70,
                positive_heater_score=0.35,
            ),
            config,
        )
        low_delta = active_spin.choose_heater_approach_reposition_action(
            fake_pose_not_unique_result(
                negative_range=2.31,
                positive_range=1.55,
                negative_heater_score=0.60,
                positive_heater_score=0.30,
            ),
            config,
        )

        self.assertFalse(low_score.ok)
        self.assertEqual(low_score.reason, "heater_approach_selected_score_too_low")
        self.assertFalse(high_opposite.ok)
        self.assertEqual(high_opposite.reason, "heater_approach_opposite_score_too_high")
        self.assertFalse(low_delta.ok)
        self.assertEqual(low_delta.reason, "heater_approach_delta_too_low")

    def test_heater_approach_action_rejects_invalid_profiles_and_near_target(self):
        config = active_spin.ArenaActiveSpinConfig(
            run_id="heater_approach_invalid_test",
            diagnostics_path=Path("unused.json"),
            enable_center_reposition=True,
            require_operator_confirmation=False,
        )

        invalid_profile = active_spin.choose_heater_approach_reposition_action(
            fake_pose_not_unique_result(
                negative_range=2.31,
                positive_range=1.55,
                negative_heater_score=0.581,
                positive_heater_score=0.143,
                negative_validity_failed_reason="profile_bad_fit",
            ),
            config,
        )
        near_target = active_spin.choose_heater_approach_reposition_action(
            fake_pose_not_unique_result(
                negative_range=1.12,
                positive_range=2.78,
                negative_heater_score=0.581,
                positive_heater_score=0.143,
            ),
            config,
        )

        self.assertFalse(invalid_profile.ok)
        self.assertEqual(invalid_profile.reason, "heater_approach_profile_invalid")
        self.assertFalse(near_target.ok)
        self.assertEqual(
            near_target.reason,
            "heater_approach_not_useful_already_near_target",
        )

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

    def test_center_reposition_lateral_step_uses_fresh_side_clearance_heading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            node = FakeNode()
            publisher = FakePublisher()
            time_box = {"now": 0.0}
            config = active_spin.ArenaActiveSpinConfig(
                run_id="reposition_dynamic_lateral_test",
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
            wait_calls = {"count": 0}
            headings = []

            def freshen():
                wait_calls["count"] += 1
                events.append("wait")
                if wait_calls["count"] >= 3:
                    session.latest_scan = fake_scan_with_side_clearance(left=0.5, right=1.4)
                    session.latest_odom_yaw_rad = 0.2
                else:
                    session.latest_scan = fake_scan_with_side_clearance(left=1.0, right=1.0)
                    session.latest_odom_yaw_rad = 0.0
                session.latest_scan_received_sec = session.now()
                session.latest_odom_pose = arena.Pose2D()
                session.latest_odom_received_sec = session.now()

            session.wait_for_fresh_inputs = freshen
            session.refresh_fresh_inputs_after_prompt = lambda: None
            session.turn_to_heading = lambda _publisher, heading: headings.append(heading)
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
                        "center_reposition_reduce_lateral_offset_dynamic",
                        0.30,
                        math.pi / 2.0,
                        math.pi / 2.0,
                        dynamic_heading=True,
                        dynamic_heading_source="live_side_clearance",
                    ),
                ),
            )

            with contextlib.redirect_stdout(io.StringIO()):
                record = session.execute_center_reposition(publisher, action)

        self.assertAlmostEqual(headings[0], 0.0)
        self.assertAlmostEqual(
            headings[1],
            active_spin.normalize_angle_rad(0.2 - math.pi / 2.0),
        )
        self.assertEqual(
            record["steps"][1]["dynamic_heading_result"]["direction"],
            "right",
        )

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

    def test_run_uses_heater_approach_after_center_reposition_still_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            node = FakeNode()
            publisher = FakePublisher()
            time_box = {"now": 0.0}
            diagnostics_path = Path(tmpdir) / "arena_active_result.json"
            config = active_spin.ArenaActiveSpinConfig(
                run_id="heater_approach_flow_test",
                diagnostics_path=diagnostics_path,
                enable_center_reposition=True,
                require_operator_confirmation=False,
                spin_complete_tolerance_deg=359.0,
                min_scan_samples=1,
                stop_settle_sec=0.0,
            )
            analyze_results = iter(
                [
                    fake_pose_not_unique_result(
                        negative_range=0.70,
                        positive_range=3.15,
                        axis_angle_rad=0.0,
                        normal_angle_rad=math.pi / 2.0,
                        lateral_offset_m=0.05,
                        negative_heater_score=0.25,
                        positive_heater_score=0.45,
                    ),
                    fake_pose_not_unique_result(
                        negative_range=2.31,
                        positive_range=1.55,
                        axis_angle_rad=0.0,
                        normal_angle_rad=math.pi / 2.0,
                        lateral_offset_m=0.05,
                        negative_heater_score=0.581,
                        positive_heater_score=0.143,
                    ),
                    fake_success_result(),
                ]
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
                analyze_fn=lambda *_args, **_kwargs: next(analyze_results),
            )
            executed_actions = []

            def execute_reposition(_publisher, action):
                executed_actions.append(action.reason)
                return {
                    **action.to_dict(),
                    "steps": [step.to_dict() for step in action.steps],
                    "driven_distance_m": action.planned_distance_m or 0.0,
                    "duration_sec": 0.0,
                }

            session.execute_center_reposition = execute_reposition

            with contextlib.redirect_stdout(io.StringIO()):
                result = session.run(publisher)
            diagnostics = json.loads(diagnostics_path.read_text())

        self.assertTrue(result.success)
        self.assertEqual(
            executed_actions,
            [
                "center_reposition_toward_arena_center",
                "heater_approach_toward_suspected_heater",
            ],
        )
        self.assertEqual(len(diagnostics["reposition"]["attempts"]), 2)
        self.assertEqual(diagnostics["reposition"]["attempts"][0]["stage"], "center")
        self.assertEqual(
            diagnostics["reposition"]["attempts"][1]["stage"],
            "heater_approach",
        )
        self.assertEqual(
            diagnostics["reposition"]["attempts"][1]["motion"]["steps"][0]["kind"],
            "heater_approach",
        )
        self.assertEqual(len(diagnostics["spin_attempts"]), 3)

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
