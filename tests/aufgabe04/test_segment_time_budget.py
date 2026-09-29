"""Offline deadline arithmetic and admission regression coverage."""

from dataclasses import replace
import json
import math
import unittest

from scripts.aufgabe04.navigation.control.driving_behavior import CommandSmoothingConfig
from scripts.aufgabe04.navigation.control.segment_time_budget import (
    MAX_WAYPOINT_TIMEOUT_SEC,
    MIN_WAYPOINT_TIMEOUT_SEC,
    SegmentTimeBudgetError,
    route_time_budgets,
    waypoint_time_budget,
)
from scripts.aufgabe04.navigation.control.waypoint_controller import ControllerConfig
from scripts.aufgabe04.navigation.foundation.models import Pose2D


class SegmentTimeBudgetTest(unittest.TestCase):
    def setUp(self):
        self.controller = ControllerConfig()
        self.route = (Pose2D(0.0, 0.0), Pose2D(2.0, 0.0))

    def budget(self, route=None, index=1, **kwargs):
        kwargs.setdefault("controller", self.controller)
        return waypoint_time_budget(
            self.route if route is None else route, index, **kwargs,
        )

    def test_recorded_long_approach_allows_measured_speed_and_alignment(self):
        # Geometry and initial turn from candidate_002 in 20260929T124849Z.
        length = 1.7251295190771123
        turn = 2.169932511191337
        route = (Pose2D(0.0, 0.0, -turn), Pose2D(length, 0.0))
        budget = self.budget(route)
        nominal = length / 0.055 + turn / 0.18
        self.assertAlmostEqual(budget.nominal_motion_sec, nominal)
        self.assertAlmostEqual(budget.acceleration_allowance_sec, 0.85)
        self.assertAlmostEqual(budget.required_timeout_sec, 1.25 * (nominal + 0.85) + 5.0)
        self.assertGreater(budget.timeout_sec, 45.00224110414274)
        measured_completion = 12.60038390592672 + length / 0.04992500227898095 + 3.0
        self.assertLess(measured_completion, budget.timeout_sec)
        self.assertLess(budget.timeout_sec, MAX_WAYPOINT_TIMEOUT_SEC)

    def test_short_segment_keeps_existing_floor(self):
        budget = self.budget((Pose2D(0, 0), Pose2D(0.5, 0)))
        self.assertEqual(budget.required_timeout_sec, MIN_WAYPOINT_TIMEOUT_SEC)
        self.assertEqual(budget.timeout_sec, MIN_WAYPOINT_TIMEOUT_SEC)

    def test_actual_admitted_pose_controls_distance_and_alignment(self):
        budget = self.budget(start_pose=Pose2D(0.2, 0, -math.pi / 2))
        self.assertAlmostEqual(budget.distance_m, 1.8)
        self.assertAlmostEqual(budget.alignment_turn_rad, math.pi / 2)

    def test_corner_rotation_belongs_to_incoming_target_only(self):
        route = (Pose2D(0, 0), Pose2D(2, 0), Pose2D(2, 2))
        incoming, outgoing = route_time_budgets(route, controller=self.controller)
        self.assertAlmostEqual(incoming.corner_turn_rad, math.pi / 2)
        self.assertEqual(incoming.alignment_turn_rad, 0.0)
        self.assertEqual(outgoing.corner_turn_rad, 0.0)
        self.assertEqual(outgoing.alignment_turn_rad, 0.0)
        self.assertAlmostEqual(incoming.acceleration_allowance_sec, 0.85)
        self.assertAlmostEqual(outgoing.acceleration_allowance_sec, 0.55)
        self.assertGreater(incoming.timeout_sec, outgoing.timeout_sec)

    def test_separate_alignment_phases_each_receive_acceleration_allowance(self):
        route = (Pose2D(0, 0, math.pi), Pose2D(1, 0), Pose2D(1, 1))
        budget = self.budget(route)
        self.assertAlmostEqual(budget.alignment_turn_rad, math.pi)
        self.assertAlmostEqual(budget.corner_turn_rad, math.pi / 2)
        self.assertAlmostEqual(budget.acceleration_allowance_sec, 0.55 + 2 * 0.3)

    def test_subthreshold_corner_is_not_charged_as_certified_corner(self):
        turn = 0.1
        route = (Pose2D(0, 0), Pose2D(1, 0), Pose2D(1 + math.cos(turn), math.sin(turn)))
        incoming, outgoing = route_time_budgets(route, controller=self.controller)
        self.assertEqual(incoming.corner_turn_rad, 0.0)
        self.assertAlmostEqual(outgoing.alignment_turn_rad, turn)

    def test_terminal_pose_yaw_uses_separate_heading_timer(self):
        straight = self.budget()
        terminal_turn = self.budget((self.route[0], replace(self.route[1], yaw_rad=math.pi)))
        self.assertEqual(straight, terminal_turn)

    def test_unconstrained_unused_route_yaws_do_not_enter_timing_math(self):
        route = (Pose2D(0, 0), Pose2D(2, 0, math.nan), Pose2D(2, 2, math.nan))
        ordinary = (Pose2D(0, 0), Pose2D(2, 0), Pose2D(2, 2))
        self.assertEqual(
            route_time_budgets(route, controller=self.controller),
            route_time_budgets(ordinary, controller=self.controller),
        )
        with self.assertRaises(SegmentTimeBudgetError):
            self.budget((Pose2D(0, 0, math.nan), Pose2D(2, 0)))
        with self.assertRaises(SegmentTimeBudgetError):
            self.budget(start_pose=Pose2D(0, 0, math.nan))

    def test_smoothing_disabled_removes_only_acceleration_allowance(self):
        route = (Pose2D(0, 0, math.pi), Pose2D(2, 0))
        enabled = self.budget(route)
        disabled = self.budget(route, smoothing=CommandSmoothingConfig(enabled=False))
        self.assertEqual(disabled.acceleration_allowance_sec, 0.0)
        self.assertEqual(disabled.nominal_motion_sec, enabled.nominal_motion_sec)
        self.assertAlmostEqual(enabled.timeout_sec - disabled.timeout_sec, 1.25 * 0.85)

    def test_slower_controller_and_acceleration_produce_larger_budgets(self):
        route = (Pose2D(0, 0, math.pi / 2), Pose2D(1.5, 0))
        baseline = self.budget(route)
        slower = self.budget(route, controller=replace(self.controller, max_linear_mps=0.04, max_angular_radps=0.12))
        softer = self.budget(route, smoothing=CommandSmoothingConfig(max_linear_accel_mps2=0.05, max_angular_accel_radps2=0.3))
        self.assertGreater(slower.timeout_sec, baseline.timeout_sec)
        self.assertGreater(softer.timeout_sec, baseline.timeout_sec)

    def test_explicit_timeout_must_cover_required_budget_and_stay_finite(self):
        required = self.budget().required_timeout_sec
        self.assertEqual(self.budget(timeout_limit_sec=required).timeout_sec, required)
        explicit = self.budget(timeout_limit_sec=90.0)
        self.assertEqual(explicit.timeout_sec, 90.0)
        self.assertEqual(explicit.required_timeout_sec, required)
        for invalid in (required - 0.001, 45.0, 0, -1, True, "90", math.nan, math.inf, 120.001):
            with self.subTest(timeout=invalid):
                with self.assertRaises(SegmentTimeBudgetError):
                    self.budget(timeout_limit_sec=invalid)

    def test_required_budget_above_hard_cap_is_infeasible(self):
        route = (Pose2D(0, 0), Pose2D(6, 0))
        for explicit in (None, 120.0, 300.0):
            with self.subTest(explicit=explicit):
                with self.assertRaisesRegex(SegmentTimeBudgetError, "exceeding.*cap"):
                    self.budget(route, timeout_limit_sec=explicit)

    def test_route_budgets_target_sequence_and_first_start_pose(self):
        route = (Pose2D(0, 0), Pose2D(1, 0), Pose2D(1, 1))
        budgets = route_time_budgets(route, start_pose=Pose2D(0.1, 0, -0.3), controller=self.controller)
        self.assertEqual(tuple(b.target_index for b in budgets), (1, 2))
        self.assertAlmostEqual(budgets[0].distance_m, 0.9)
        self.assertAlmostEqual(budgets[0].alignment_turn_rad, 0.3)
        self.assertAlmostEqual(budgets[1].distance_m, 1.0)
        self.assertEqual(budgets[1].alignment_turn_rad, 0.0)

    def test_single_point_and_zero_length_segment_have_finite_floor(self):
        single, = route_time_budgets((Pose2D(0, 0, math.pi),), controller=self.controller)
        self.assertEqual(single.target_index, 0)
        self.assertEqual(single.nominal_motion_sec, 0.0)
        self.assertEqual(single.acceleration_allowance_sec, 0.0)
        self.assertEqual(single.timeout_sec, 45.0)
        duplicate = self.budget((Pose2D(0, 0), Pose2D(0, 0, math.pi)))
        self.assertEqual(duplicate.nominal_motion_sec, 0.0)
        self.assertEqual(duplicate.timeout_sec, 45.0)

    def test_angle_wrap_uses_shortest_turn_even_for_large_finite_headings(self):
        wrapped = self.budget((Pose2D(0, 0, math.tau * 3 - 0.3), Pose2D(2, 0)))
        self.assertAlmostEqual(wrapped.alignment_turn_rad, 0.3)
        huge = self.budget((Pose2D(0, 0, 1.0e308), Pose2D(2, 0)))
        self.assertTrue(math.isfinite(huge.timeout_sec))
        self.assertLessEqual(huge.alignment_turn_rad, math.pi)

    def test_invalid_route_shapes_and_indices_fail_with_policy_error(self):
        for invalid in (None, [], "route", [dict(x_m=0, y_m=0)], [(0, 0)]):
            with self.subTest(route=invalid):
                with self.assertRaises(SegmentTimeBudgetError):
                    waypoint_time_budget(invalid, 0, controller=self.controller)
                with self.assertRaises(SegmentTimeBudgetError):
                    route_time_budgets(invalid, controller=self.controller)
        for invalid in (-1, 2, True, 0.5, "1"):
            with self.subTest(index=invalid):
                with self.assertRaises(SegmentTimeBudgetError):
                    self.budget(index=invalid)

    def test_nonfinite_nonnumeric_and_boolean_pose_values_are_rejected(self):
        for field in ("x_m", "y_m", "yaw_rad"):
            for invalid in (True, "0", None, math.inf, -math.inf, math.nan):
                with self.subTest(field=field, value=invalid):
                    pose = replace(Pose2D(0, 0), **{field: invalid})
                    with self.assertRaises(SegmentTimeBudgetError):
                        self.budget((pose, self.route[1]))
                    with self.assertRaises(SegmentTimeBudgetError):
                        self.budget(start_pose=pose)
        with self.assertRaises(SegmentTimeBudgetError):
            self.budget((Pose2D(-1e308, 0), Pose2D(1e308, 0)))

    def test_invalid_controls_and_arithmetic_overflow_are_rejected(self):
        for field in ("max_linear_mps", "max_angular_radps"):
            for invalid in (True, "0.1", None, 0.0, -0.1, math.nan, math.inf, 5e-324):
                with self.subTest(field=field, value=invalid):
                    with self.assertRaises(SegmentTimeBudgetError):
                        self.budget(
                            (Pose2D(0, 0, math.pi), Pose2D(2, 0)),
                            controller=replace(self.controller, **{field: invalid}),
                        )
        with self.assertRaises(SegmentTimeBudgetError):
            self.budget(controller=None)
        with self.assertRaises(SegmentTimeBudgetError):
            self.budget(smoothing=None)
        with self.assertRaises(SegmentTimeBudgetError):
            self.budget(smoothing=CommandSmoothingConfig(max_linear_accel_mps2=True))

    def test_evidence_is_json_serializable_and_complete(self):
        budget = self.budget()
        evidence = json.loads(json.dumps(budget.to_dict(), allow_nan=False))
        self.assertEqual(evidence, {
            "target_index": 1,
            "distance_m": budget.distance_m,
            "alignment_turn_rad": budget.alignment_turn_rad,
            "corner_turn_rad": budget.corner_turn_rad,
            "nominal_motion_sec": budget.nominal_motion_sec,
            "acceleration_allowance_sec": budget.acceleration_allowance_sec,
            "required_timeout_sec": budget.required_timeout_sec,
            "timeout_sec": budget.timeout_sec,
        })


if __name__ == "__main__":
    unittest.main()
