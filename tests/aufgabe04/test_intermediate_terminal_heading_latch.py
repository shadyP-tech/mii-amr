from __future__ import annotations

import json
import math
import unittest

from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.simple_waypoint_follower import (
    FollowerConfig,
    INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M,
    INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_HOLD_EXCEEDED,
    INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M,
    compute_intermediate_terminal_heading_command,
    controller_config_for_route_kind,
    intermediate_terminal_heading_hold_diagnostics,
    reset_intermediate_terminal_heading_latch,
)
from scripts.aufgabe04.navigation.viewpoint_sampling_contract import (
    DEFAULT_VIEWPOINT_SAMPLING_STRICT_ARRIVAL_TOLERANCE_M,
    DEFAULT_VIEWPOINT_SAMPLING_TARGET_DISTANCE_M,
    VIEWPOINT_SAMPLING_CONTRACT_NAME,
    VIEWPOINT_SAMPLING_CONTRACT_VERSION,
    ViewpointSamplingArrivalLatch,
    ViewpointSamplingHoldConfig,
    ViewpointSamplingMaterialTarget,
    viewpoint_sampling_hold_metrics,
)
from scripts.aufgabe04.navigation.waypoint_controller import (
    ControllerConfig,
    compute_waypoint_command,
)


class IntermediateTerminalHeadingLatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.waypoints = (
            Pose2D(0.0, 0.0, math.nan),
            Pose2D(1.0, 0.0, math.pi / 2.0),
        )
        self.config = ControllerConfig(
            goal_tolerance_m=0.018,
            heading_tolerance_rad=0.08,
            max_angular_radps=0.18,
        )

    def _enter_latch(self):
        decision = compute_intermediate_terminal_heading_command(
            Pose2D(1.0 - 0.0175, 0.0, 0.0),
            self.waypoints,
            1,
            self.config,
            "viewpoint_sampling",
        )
        self.assertIsNotNone(decision.latch)
        return decision

    def _pose_at_target_frame_offset(
        self,
        longitudinal_m: float,
        lateral_m: float,
        *,
        yaw_rad: float = 0.2,
    ) -> Pose2D:
        target = self.waypoints[-1]
        heading_x = math.cos(target.yaw_rad)
        heading_y = math.sin(target.yaw_rad)
        return Pose2D(
            target.x_m
            + longitudinal_m * heading_x
            - lateral_m * heading_y,
            target.y_m
            + longitudinal_m * heading_y
            + lateral_m * heading_x,
            yaw_rad,
        )

    def test_final_finite_yaw_target_enters_strict_sampling_latch(self):
        decision = self._enter_latch()

        self.assertEqual(
            INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M,
            0.018,
        )
        self.assertEqual(decision.failure, "")
        self.assertEqual(decision.step.progress_mode, "terminal_heading")
        self.assertEqual(decision.step.command.linear_x_mps, 0.0)
        self.assertNotEqual(decision.step.command.angular_z_radps, 0.0)
        self.assertFalse(decision.step.reached_goal)

    def test_latched_drift_at_0_0195_stays_terminal_heading_only(self):
        latch = self._enter_latch().latch

        decision = compute_intermediate_terminal_heading_command(
            Pose2D(1.0 - 0.0195, 0.0, 0.2),
            self.waypoints,
            1,
            self.config,
            "viewpoint_sampling",
            latch,
        )

        self.assertEqual(decision.failure, "")
        self.assertIs(decision.latch, latch)
        self.assertEqual(decision.step.progress_mode, "terminal_heading")
        self.assertEqual(decision.step.command.linear_x_mps, 0.0)
        # The final yaw lies counter-clockwise from the robot. A point-bearing
        # fallback would turn clockwise in this fixture.
        self.assertGreater(decision.step.command.angular_z_radps, 0.0)

    def test_micrometer_scale_hold_boundary_drift_stays_zero_linear(self):
        latch = self._enter_latch().latch

        for distance_m in (0.02000499, 0.020002421):
            with self.subTest(distance_m=distance_m):
                decision = compute_intermediate_terminal_heading_command(
                    Pose2D(1.0 - distance_m, 0.0, 0.2),
                    self.waypoints,
                    1,
                    self.config,
                    "viewpoint_sampling",
                    latch,
                )

                self.assertEqual(decision.failure, "")
                self.assertEqual(decision.step.progress_mode, "terminal_heading")
                self.assertEqual(decision.step.command.linear_x_mps, 0.0)

    def test_radial_drift_beyond_hold_comparison_epsilon_fails_closed(self):
        latch = self._enter_latch().latch
        radial_drift_m = (
            INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M
            + INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M
            + 1.0e-7
        )

        decision = compute_intermediate_terminal_heading_command(
            self._pose_at_target_frame_offset(radial_drift_m, 0.0),
            self.waypoints,
            1,
            self.config,
            "viewpoint_sampling",
            latch,
        )

        self.assertEqual(
            decision.failure,
            INTERMEDIATE_TERMINAL_HEADING_HOLD_EXCEEDED,
        )
        self.assertEqual(
            (decision.step.command.linear_x_mps, decision.step.command.angular_z_radps),
            (0.0, 0.0),
        )

    def test_intermediate_controller_cannot_enter_terminal_heading_before_latch(self):
        resolved = controller_config_for_route_kind(
            ControllerConfig(
                goal_tolerance_m=0.03,
                terminal_goal_tolerance_m=0.04,
                heading_tolerance_rad=0.08,
            ),
            "viewpoint_sampling",
            viewpoint_sampling_goal_tolerance_m=0.03,
        )

        self.assertEqual(resolved.goal_tolerance_m, 0.018)
        self.assertEqual(resolved.terminal_goal_tolerance_m, 0.018)
        decision = compute_intermediate_terminal_heading_command(
            Pose2D(1.0 - 0.0195, 0.0, 0.2),
            self.waypoints,
            1,
            resolved,
            "viewpoint_sampling",
        )
        self.assertIsNone(decision.latch)
        self.assertEqual(decision.step.progress_mode, "path_tracking")
        self.assertGreater(decision.step.command.linear_x_mps, 0.0)
        self.assertLess(decision.step.command.angular_z_radps, 0.0)

    def test_repeat14_anisotropic_drift_stays_zero_linear(self):
        latch = self._enter_latch().latch
        longitudinal_m = 0.0082
        lateral_m = 0.018275
        pose = self._pose_at_target_frame_offset(longitudinal_m, lateral_m)

        decision = compute_intermediate_terminal_heading_command(
            pose,
            self.waypoints,
            1,
            self.config,
            "viewpoint_sampling",
            latch,
        )

        target_error_m = math.hypot(longitudinal_m, lateral_m)
        inferred_stand_distance_m = math.hypot(
            0.33 - longitudinal_m,
            lateral_m,
        )
        diagnostics = intermediate_terminal_heading_hold_diagnostics(
            pose,
            latch,
            hold_tolerance_m=0.020,
            viewpoint_sampling_target_distance_m=0.33,
            viewpoint_sampling_target_envelope_radius_m=0.030,
        )

        self.assertGreater(target_error_m, 0.020)
        self.assertLess(target_error_m, 0.030)
        self.assertGreaterEqual(inferred_stand_distance_m, 0.33 - 0.020)
        self.assertLessEqual(inferred_stand_distance_m, 0.33 + 0.020)
        self.assertEqual(decision.failure, "")
        self.assertIs(decision.latch, latch)
        self.assertEqual(
            decision.step.command.linear_x_mps,
            0.0,
        )
        self.assertEqual(
            decision.step.progress_mode,
            "terminal_heading",
        )
        self.assertEqual(
            diagnostics["hold_model"],
            "target_envelope_and_inferred_stand_distance_annulus",
        )
        self.assertEqual(diagnostics["distance_unit"], "m")
        self.assertEqual(diagnostics["target_yaw_unit"], "rad")
        self.assertAlmostEqual(diagnostics["target_yaw_rad"], math.pi / 2.0)
        self.assertAlmostEqual(diagnostics["inferred_stand_center_x_m"], 1.0)
        self.assertAlmostEqual(diagnostics["inferred_stand_center_y_m"], 0.33)
        self.assertAlmostEqual(
            diagnostics["target_envelope_distance_m"],
            target_error_m,
        )
        self.assertAlmostEqual(
            diagnostics["inferred_stand_distance_m"],
            inferred_stand_distance_m,
        )
        self.assertAlmostEqual(diagnostics["annulus_min_m"], 0.31)
        self.assertAlmostEqual(diagnostics["annulus_max_m"], 0.35)
        self.assertTrue(diagnostics["target_envelope_within_limit"])
        self.assertTrue(
            diagnostics["inferred_stand_distance_within_annulus"]
        )
        self.assertTrue(diagnostics["within_hold"])

    def test_viewpoint_tangential_drift_beyond_0_030_fails_closed(self):
        latch = self._enter_latch().latch

        decision = compute_intermediate_terminal_heading_command(
            self._pose_at_target_frame_offset(0.0, 0.03002),
            self.waypoints,
            1,
            self.config,
            "viewpoint_sampling",
            latch,
        )

        self.assertEqual(
            INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M,
            0.030,
        )
        self.assertEqual(
            decision.failure,
            INTERMEDIATE_TERMINAL_HEADING_HOLD_EXCEEDED,
        )
        self.assertEqual(
            (decision.step.command.linear_x_mps, decision.step.command.angular_z_radps),
            (0.0, 0.0),
        )

    def test_axis_acquisition_retains_circular_0_020_hold(self):
        latch = compute_intermediate_terminal_heading_command(
            Pose2D(1.0 - 0.0175, 0.0, 0.0),
            self.waypoints,
            1,
            self.config,
            "axis_acquisition",
        ).latch

        decision = compute_intermediate_terminal_heading_command(
            Pose2D(1.0 - 0.0201, 0.0, 0.2),
            self.waypoints,
            1,
            self.config,
            "axis_acquisition",
            latch,
        )

        self.assertIsNotNone(latch)
        self.assertEqual(
            decision.failure,
            INTERMEDIATE_TERMINAL_HEADING_HOLD_EXCEEDED,
        )
        self.assertEqual(
            (decision.step.command.linear_x_mps, decision.step.command.angular_z_radps),
            (0.0, 0.0),
        )

    def test_nonfinite_latched_pose_components_fail_closed(self):
        latch = self._enter_latch().latch

        for pose in (
            Pose2D(math.nan, 0.0, 0.2),
            Pose2D(1.0, math.inf, 0.2),
            Pose2D(1.0, 0.0, math.nan),
        ):
            with self.subTest(pose=pose):
                decision = compute_intermediate_terminal_heading_command(
                    pose,
                    self.waypoints,
                    1,
                    self.config,
                    "viewpoint_sampling",
                    latch,
                )

                self.assertEqual(
                    decision.failure,
                    INTERMEDIATE_TERMINAL_HEADING_HOLD_EXCEEDED,
                )
                self.assertEqual(
                    (
                        decision.step.command.linear_x_mps,
                        decision.step.command.angular_z_radps,
                    ),
                    (0.0, 0.0),
                )

    def test_latched_terminal_yaw_completion_succeeds_inside_hold_disk(self):
        latch = compute_intermediate_terminal_heading_command(
            Pose2D(1.0 - 0.0175, 0.0, 0.0),
            self.waypoints,
            1,
            self.config,
            "axis_acquisition",
        ).latch

        decision = compute_intermediate_terminal_heading_command(
            Pose2D(1.0 - 0.0195, 0.0, math.pi / 2.0),
            self.waypoints,
            1,
            self.config,
            "axis_acquisition",
            latch,
        )

        self.assertIsNotNone(latch)
        self.assertEqual(decision.failure, "")
        self.assertTrue(decision.step.reached_goal)
        self.assertEqual(
            (decision.step.command.linear_x_mps, decision.step.command.angular_z_radps),
            (0.0, 0.0),
        )

    def test_material_revision_and_target_change_clear_latch(self):
        latch = self._enter_latch().latch

        self.assertIs(
            reset_intermediate_terminal_heading_latch(latch),
            latch,
        )
        self.assertIsNone(
            reset_intermediate_terminal_heading_latch(
                latch,
                material_route_revision=True,
            )
        )
        self.assertIsNone(
            reset_intermediate_terminal_heading_latch(
                latch,
                target_changed=True,
            )
        )

        revised_waypoints = (
            self.waypoints[0],
            Pose2D(1.1, 0.0, math.pi / 2.0),
        )
        revised = compute_intermediate_terminal_heading_command(
            Pose2D(0.98, 0.0, 0.0),
            revised_waypoints,
            1,
            self.config,
            "viewpoint_sampling",
            latch,
        )
        self.assertIsNone(revised.latch)
        self.assertEqual(revised.step.progress_mode, "path_tracking")
        self.assertGreater(revised.step.command.linear_x_mps, 0.0)

    def test_ordinary_and_physical_routes_keep_existing_controller_behavior(self):
        latch = self._enter_latch().latch
        pose = Pose2D(1.0 - 0.0175, 0.0, 0.0)

        for route_kind, config in (
            ("ordinary_route", self.config),
            (
                "synchronized_face_approach",
                ControllerConfig(
                    goal_tolerance_m=0.018,
                    heading_tolerance_rad=0.08,
                    max_angular_radps=0.18,
                    enforce_heading_corridor=True,
                ),
            ),
        ):
            with self.subTest(route_kind=route_kind):
                expected = compute_waypoint_command(
                    pose,
                    self.waypoints,
                    1,
                    config,
                )
                decision = compute_intermediate_terminal_heading_command(
                    pose,
                    self.waypoints,
                    1,
                    config,
                    route_kind,
                    latch,
                )

                self.assertEqual(decision.step, expected)
                self.assertIsNone(decision.latch)
                self.assertEqual(decision.failure, "")

    def test_hold_tolerance_config_is_bounded_at_0_020(self):
        configured = FollowerConfig(controller=self.config)

        self.assertEqual(
            configured.viewpoint_sampling_terminal_heading_hold_tolerance_m,
            0.020,
        )
        self.assertEqual(configured.viewpoint_sampling_target_distance_m, 0.33)
        self.assertEqual(
            configured.viewpoint_sampling_terminal_heading_target_envelope_radius_m,
            0.030,
        )
        with self.assertRaises(ValueError):
            FollowerConfig(
                controller=self.config,
                viewpoint_sampling_terminal_heading_hold_tolerance_m=0.0201,
            )
        with self.assertRaises(ValueError):
            FollowerConfig(
                controller=self.config,
                viewpoint_sampling_terminal_heading_hold_tolerance_m=0.0179,
            )
        with self.assertRaises(ValueError):
            FollowerConfig(
                controller=self.config,
                viewpoint_sampling_target_distance_m=0.020,
            )
        with self.assertRaises(ValueError):
            FollowerConfig(
                controller=self.config,
                viewpoint_sampling_target_distance_m=math.nan,
            )
        with self.assertRaises(ValueError):
            FollowerConfig(
                controller=self.config,
                viewpoint_sampling_terminal_heading_target_envelope_radius_m=0.0301,
            )
        with self.assertRaises(ValueError):
            FollowerConfig(
                controller=self.config,
                viewpoint_sampling_terminal_heading_target_envelope_radius_m=0.0199,
            )
        with self.assertRaises(ValueError):
            FollowerConfig(
                controller=self.config,
                viewpoint_sampling_terminal_heading_target_envelope_radius_m=(
                    math.inf
                ),
            )


class ViewpointSamplingContractTest(unittest.TestCase):
    def setUp(self) -> None:
        # Recorded repeat-15 detected_stand_03 target/observer pose.  The robot
        # had strictly entered on an earlier observation, then settled with a
        # 20.36 mm anisotropic offset that is safe in the shared hold geometry.
        self.repeat15_target_pose = Pose2D(
            -0.03674540082085545,
            0.19714385282555608,
            1.132226788334103,
        )
        self.repeat15_robot_pose = Pose2D(
            -0.016401899394672962,
            0.19790291610018154,
            1.0568849205675144,
        )
        self.target = ViewpointSamplingMaterialTarget(
            pose=self.repeat15_target_pose,
            face_id="sampling_near",
            target_revision=4,
        )

    def test_contract_constants_and_follower_re_exports_are_stable(self):
        self.assertEqual(VIEWPOINT_SAMPLING_CONTRACT_NAME, "viewpoint_sampling_arrival_hold")
        self.assertEqual(VIEWPOINT_SAMPLING_CONTRACT_VERSION, 1)
        self.assertEqual(DEFAULT_VIEWPOINT_SAMPLING_TARGET_DISTANCE_M, 0.33)
        self.assertEqual(
            DEFAULT_VIEWPOINT_SAMPLING_STRICT_ARRIVAL_TOLERANCE_M,
            0.017,
        )
        self.assertEqual(
            INTERMEDIATE_TERMINAL_HEADING_ENTRY_TOLERANCE_M,
            0.018,
        )
        self.assertEqual(
            INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M,
            0.020,
        )
        self.assertEqual(
            INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M,
            0.030,
        )
        self.assertEqual(
            INTERMEDIATE_TERMINAL_HEADING_DISTANCE_COMPARISON_EPSILON_M,
            1.0e-5,
        )

    def test_repeat15_metrics_match_follower_and_remain_inside_hold(self):
        metrics = viewpoint_sampling_hold_metrics(
            self.repeat15_robot_pose,
            self.repeat15_target_pose,
        )
        follower_metrics = intermediate_terminal_heading_hold_diagnostics(
            self.repeat15_robot_pose,
            type(
                "Latch",
                (),
                {
                    "route_kind": "viewpoint_sampling",
                    "target": self.repeat15_target_pose,
                },
            )(),
            hold_tolerance_m=0.020,
            viewpoint_sampling_target_distance_m=0.33,
            viewpoint_sampling_target_envelope_radius_m=0.030,
        )

        self.assertAlmostEqual(
            metrics.target_envelope_distance_m,
            0.02035765770740765,
        )
        self.assertAlmostEqual(
            metrics.inferred_stand_distance_m,
            0.32118418726967807,
        )
        self.assertGreater(metrics.target_envelope_distance_m, 0.020)
        self.assertTrue(metrics.target_envelope_within_limit)
        self.assertTrue(metrics.inferred_stand_distance_within_annulus)
        self.assertTrue(metrics.within_hold)
        self.assertEqual(
            follower_metrics,
            metrics.to_diagnostics_dict(),
        )

    def test_nonfinite_metrics_fail_closed_and_status_is_strict_json(self):
        for pose in (
            Pose2D(math.nan, 0.0, 0.0),
            Pose2D(0.0, math.inf, 0.0),
            Pose2D(0.0, 0.0, math.nan),
        ):
            with self.subTest(pose=pose):
                metrics = viewpoint_sampling_hold_metrics(
                    pose,
                    self.repeat15_target_pose,
                )
                self.assertFalse(metrics.within_hold)
                json.dumps(metrics.to_status_dict(), allow_nan=False)

    def test_arrival_latch_strict_entry_then_repeat15_hold(self):
        latch = ViewpointSamplingArrivalLatch()

        initialized = latch.update(
            pose=self.repeat15_target_pose,
            target=self.target,
        )
        self.assertFalse(initialized.armed)
        self.assertEqual(
            initialized.transition_reason,
            "target_initialized_unarmed",
        )

        entered = latch.update(
            pose=self.repeat15_target_pose,
            target=self.target,
        )
        self.assertTrue(entered.armed)
        self.assertTrue(entered.strict_ever_armed)
        self.assertEqual(entered.transition_reason, "strict_entry_armed")

        held = latch.update(
            pose=self.repeat15_robot_pose,
            target=self.target,
        )
        self.assertTrue(held.armed)
        self.assertTrue(held.hold_valid)
        self.assertFalse(held.strict_entry_within_limit)
        self.assertEqual(held.transition_reason, "armed_hold_valid")
        json.dumps(held.to_status_dict(), allow_nan=False)

    def test_material_pose_face_and_revision_changes_reset_without_rearming(self):
        latch = ViewpointSamplingArrivalLatch()

        def arm(target: ViewpointSamplingMaterialTarget) -> None:
            latch.update(pose=target.pose, target=target)
            self.assertTrue(latch.update(pose=target.pose, target=target).armed)

        arm(self.target)
        changed_targets = (
            (
                ViewpointSamplingMaterialTarget(
                    pose=Pose2D(
                        self.target.pose.x_m + 0.001,
                        self.target.pose.y_m,
                        self.target.pose.yaw_rad,
                    ),
                    face_id=self.target.face_id,
                    target_revision=self.target.target_revision,
                ),
                "material_target_pose_changed",
            ),
            (
                ViewpointSamplingMaterialTarget(
                    pose=self.target.pose,
                    face_id="face_b",
                    target_revision=self.target.target_revision,
                ),
                "material_target_pose_face_changed",
            ),
            (
                ViewpointSamplingMaterialTarget(
                    pose=self.target.pose,
                    face_id=self.target.face_id,
                    target_revision=5,
                ),
                "material_target_face_revision_changed",
            ),
        )
        for changed, expected_reason in changed_targets:
            with self.subTest(expected_reason=expected_reason):
                # Each tuple changes from the preceding tuple, so the reason
                # exposes every changed key, not only the fixture's intent.
                evidence = latch.update(pose=changed.pose, target=changed)
                self.assertFalse(evidence.armed)
                self.assertFalse(evidence.strict_ever_armed)
                self.assertEqual(evidence.reset_reason, expected_reason)

        latch.reset("observer_stream_changed")
        status = latch.to_status_dict()
        self.assertEqual(status["reset_reason"], "observer_stream_changed")
        self.assertFalse(status["armed"])
        json.dumps(status, allow_nan=False)

    def test_latch_disarm_reason_identifies_failed_hold_predicate(self):
        cases = (
            (
                Pose2D(
                    self.target.pose.x_m + 0.031,
                    self.target.pose.y_m,
                    self.target.pose.yaw_rad,
                ),
                "target_envelope_exceeded",
            ),
            (
                Pose2D(
                    self.target.pose.x_m
                    + 0.02002 * math.cos(self.target.pose.yaw_rad),
                    self.target.pose.y_m
                    + 0.02002 * math.sin(self.target.pose.yaw_rad),
                    self.target.pose.yaw_rad,
                ),
                "inferred_stand_annulus_exceeded",
            ),
            (
                Pose2D(
                    self.target.pose.x_m,
                    self.target.pose.y_m,
                    math.nan,
                ),
                "nonfinite_heading",
            ),
        )
        for pose, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                latch = ViewpointSamplingArrivalLatch()
                latch.update(pose=self.target.pose, target=self.target)
                latch.update(pose=self.target.pose, target=self.target)

                disarmed = latch.update(pose=pose, target=self.target)

                self.assertFalse(disarmed.armed)
                self.assertTrue(disarmed.strict_ever_armed)
                self.assertEqual(disarmed.disarm_reason, expected_reason)
                self.assertEqual(
                    latch.to_status_dict()["disarm_reason"],
                    expected_reason,
                )

    def test_hold_config_rejects_cross_stage_boundary_violations(self):
        for kwargs in (
            {"entry_tolerance_m": 0.0181},
            {"hold_tolerance_m": 0.0179},
            {"target_envelope_radius_m": 0.0199},
            {"target_envelope_radius_m": 0.0301},
            {"target_distance_m": 0.020},
            {"distance_comparison_epsilon_m": 1.01e-5},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ViewpointSamplingHoldConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
