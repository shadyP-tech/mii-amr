import math
import unittest
from unittest.mock import patch

from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.perception.camera_stand_observation import (
    stand_axis_from_camera_yaw,
)
from scripts.aufgabe04.navigation.viewpoint_recommendation import (
    QrBindingObservation,
    QrFaceLatch,
    StableFaceResolver,
)
from scripts.aufgabe04.simulation.sim_viewpoint_optimization import (
    AxialAngleFilter,
    DynamicPreApproachTracker,
    DynamicTargetConfig,
    StationarySettleGate,
    TimedSample,
    ViewpointConfig,
    ViewpointMeasurement,
    ViewpointSamplingLatch,
    evaluate_viewpoint,
    face_normal_candidates,
    newest_synchronized_triplet,
    nearest_timed_sample,
    normalize_angle,
)
from scripts.aufgabe04.simulation.sim_synchronized_viewpoint_node import (
    _conditioned_axis_decision,
    _conditioned_axis_input,
    build_parser as build_synchronized_parser,
    camera_yaw_from_target_line_of_sight,
    select_published_viewpoint_pose,
    provisional_viewpoint_candidates,
    should_defer_initial_physical_recommendation,
    should_reseed_face_resolver,
    suspend_qr_binding_while_identity_unresolved,
)


def measurement(**changes):
    values = dict(
        image_stamp_sec=10.0,
        scan_stamp_sec=9.95,
        robot_pose=Pose2D(0.3, 0.0, math.pi),
        linear_speed_mps=0.0,
        angular_speed_radps=0.0,
        stand_x_m=0.0,
        stand_y_m=0.0,
        distance_m=0.30,
        camera_center_error_rad=0.0,
        camera_yaw_rad=0.05,
        silhouette_usable=True,
    )
    values.update(changes)
    return ViewpointMeasurement(**values)


class SimViewpointOptimizationTest(unittest.TestCase):
    def test_off_center_optical_yaw_is_corrected_to_target_line_of_sight(self):
        # Robot-to-stand line of sight is -120 degrees. The visible face normal
        # is -133.586 degrees, while the camera optical axis points -132 degrees.
        # PnP therefore reports -1.586 degrees and the target appears 12 degrees
        # left of image center (negative image-x angle). Correcting that bearing
        # recovers the true axis.
        corrected = camera_yaw_from_target_line_of_sight(
            math.radians(-1.586),
            math.radians(-12.0),
        )
        axis = stand_axis_from_camera_yaw(
            robot_x_m=math.cos(math.radians(60.0)),
            robot_y_m=math.sin(math.radians(60.0)),
            stand_x_m=0.0,
            stand_y_m=0.0,
            camera_yaw_rad=corrected,
        )

        error = 0.5 * abs(
            normalize_angle(2.0 * (axis - math.radians(136.414)))
        )
        self.assertAlmostEqual(math.degrees(error), 0.0, places=6)

    def test_conditioned_axis_uses_synchronized_camera_map_heading(self):
        decision = _conditioned_axis_decision(
            camera_yaw_rad=math.radians(-1.586),
            silhouette_usable=True,
            estimate_mode="face_visible",
            max_obliqueness_rad=math.radians(20.0),
            robot_pose=Pose2D(
                math.cos(math.radians(60.0)),
                math.sin(math.radians(60.0)),
                math.radians(-132.0),
            ),
            stand_pose=Pose2D(0.0, 0.0),
            camera_heading_rad=math.radians(-132.0),
        )

        error = 0.5 * abs(
            normalize_angle(
                2.0 * (decision.axis_rad - math.radians(136.414))
            )
        )
        self.assertEqual(decision.reason, "well_conditioned")
        self.assertAlmostEqual(math.degrees(error), 0.0, places=6)

    def test_conditioning_uses_los_yaw_but_map_axis_uses_raw_optical_yaw(self):
        raw_optical = math.radians(23.01)
        los_relative = math.radians(17.76)
        camera_heading = math.radians(-132.0)
        decision = _conditioned_axis_decision(
            camera_yaw_rad=raw_optical,
            conditioning_yaw_rad=los_relative,
            silhouette_usable=True,
            estimate_mode="face_visible",
            max_obliqueness_rad=math.radians(20.0),
            robot_pose=Pose2D(0.3, 0.0, camera_heading),
            stand_pose=Pose2D(0.0, 0.0),
            camera_heading_rad=camera_heading,
        )

        expected_axis = normalize_angle(
            camera_heading + raw_optical - math.pi / 2.0
        )
        self.assertEqual(decision.reason, "well_conditioned")
        self.assertAlmostEqual(decision.axis_rad, expected_axis, places=7)
        self.assertAlmostEqual(
            decision.confidence,
            1.0 - abs(los_relative) / (math.pi / 2.0),
            places=7,
        )

    def test_axial_filter_treats_opposite_directions_as_same_axis(self):
        angle_filter = AxialAngleFilter()
        angle_filter.add(math.radians(10), 1.0)
        axis, confidence = angle_filter.add(math.radians(190), 1.0)
        self.assertAlmostEqual(axis, math.radians(10), places=6)
        self.assertAlmostEqual(confidence, 1.0)

    def test_unsettled_gap_resets_uncommitted_dynamic_axis_samples(self):
        tracker = DynamicPreApproachTracker(
            Pose2D(0.0, 0.0),
            DynamicTargetConfig(min_axis_samples=2),
        )
        first = tracker.update(
            robot_pose=Pose2D(0.3, 0.0, math.pi),
            axis_rad=0.0,
            measurement_confidence=1.0,
            linear_speed_mps=0.0,
            angular_speed_radps=0.0,
        )
        tracker.reset_uncommitted_samples()
        second = tracker.update(
            robot_pose=Pose2D(0.3, 0.0, math.pi),
            axis_rad=0.0,
            measurement_confidence=1.0,
            linear_speed_mps=0.0,
            angular_speed_radps=0.0,
        )

        self.assertEqual(first.reason, "insufficient_axis_samples")
        self.assertEqual(second.reason, "insufficient_axis_samples")

    def test_axial_filter_stable_cluster_ignores_isolated_outlier(self):
        angle_filter = AxialAngleFilter(max_samples=12)
        for angle_deg in (1, -1, 2, -2, 0, 1, -1, 24):
            angle_filter.add(math.radians(angle_deg), 1.0)

        stable = angle_filter.stable_inlier_estimate(
            max_deviation_rad=math.radians(5.0),
            min_samples=7,
        )

        self.assertIsNotNone(stable)
        axis, confidence, count = stable
        self.assertLess(abs(math.degrees(axis)), 1.0)
        self.assertGreater(confidence, 0.99)
        self.assertEqual(count, 7)

    def test_dynamic_target_waits_for_consensus_then_selects_nearer_face(self):
        tracker = DynamicPreApproachTracker(
            Pose2D(0.0, 0.0),
            DynamicTargetConfig(min_axis_samples=3, freeze_distance_m=0.2),
        )
        robot = Pose2D(0.0, -1.0, math.pi / 2)
        self.assertFalse(tracker.update(robot_pose=robot, axis_rad=0.0, measurement_confidence=0.9,
                                        linear_speed_mps=0.05, angular_speed_radps=0.0).accepted)
        tracker.update(robot_pose=robot, axis_rad=0.02, measurement_confidence=0.9,
                       linear_speed_mps=0.05, angular_speed_radps=0.0)
        update = tracker.update(robot_pose=robot, axis_rad=-0.02, measurement_confidence=0.9,
                                linear_speed_mps=0.05, angular_speed_radps=0.0)
        self.assertTrue(update.accepted)
        self.assertEqual(update.side_index, 1)
        self.assertLess(update.pose.y_m, 0.0)

    def test_dynamic_target_rejects_a_high_confidence_but_spread_axis_window(self):
        tracker = DynamicPreApproachTracker(
            Pose2D(0.0, 0.0),
            DynamicTargetConfig(
                min_axis_samples=3,
                min_axis_confidence=0.0,
                max_axis_deviation_rad=math.radians(5.0),
            ),
        )
        robot = Pose2D(1.0, 0.0, math.pi)
        for angle_deg in (0.0, 0.0):
            tracker.update(
                robot_pose=robot,
                axis_rad=math.radians(angle_deg),
                measurement_confidence=1.0,
                linear_speed_mps=0.0,
                angular_speed_radps=0.0,
            )
        update = tracker.update(
            robot_pose=robot,
            axis_rad=math.radians(18.0),
            measurement_confidence=1.0,
            linear_speed_mps=0.0,
            angular_speed_radps=0.0,
        )
        self.assertFalse(update.accepted)
        self.assertEqual(update.reason, "axis_samples_not_stable")

    def test_dynamic_target_rejects_fast_rotation_and_freezes_near_stand(self):
        tracker = DynamicPreApproachTracker(
            Pose2D(0.0, 0.0),
            DynamicTargetConfig(min_axis_samples=1, freeze_distance_m=0.42),
        )
        rejected = tracker.update(robot_pose=Pose2D(1.0, 0.0), axis_rad=0.0,
                                  measurement_confidence=1.0, linear_speed_mps=0.0,
                                  angular_speed_radps=0.5)
        self.assertEqual(rejected.reason, "angular_motion_too_fast")
        accepted = tracker.update(robot_pose=Pose2D(1.0, 0.0), axis_rad=0.0,
                                  measurement_confidence=1.0, linear_speed_mps=0.0,
                                  angular_speed_radps=0.0)
        self.assertTrue(accepted.accepted)
        frozen = tracker.update(robot_pose=Pose2D(0.4, 0.0), axis_rad=0.2,
                                measurement_confidence=1.0, linear_speed_mps=0.0,
                                angular_speed_radps=0.0, freeze_allowed=True)
        self.assertTrue(frozen.frozen)
        self.assertEqual(frozen.pose, accepted.pose)

    def test_dynamic_target_never_switches_to_opposite_side_without_qr(self):
        tracker = DynamicPreApproachTracker(
            Pose2D(0.0, 0.0),
            DynamicTargetConfig(min_axis_samples=1, freeze_distance_m=0.42),
        )
        first = tracker.update(robot_pose=Pose2D(1.0, 0.0), axis_rad=0.0,
                               measurement_confidence=1.0, linear_speed_mps=0.0,
                               angular_speed_radps=0.0)
        held = tracker.update(robot_pose=first.pose, axis_rad=None,
                                  measurement_confidence=0.0, linear_speed_mps=0.0,
                                  angular_speed_radps=0.0)
        self.assertFalse(held.accepted)
        self.assertTrue(held.frozen)
        self.assertEqual(held.reason, "target_committed")
        self.assertEqual(held.side_index, first.side_index)
        self.assertEqual(held.pose, first.pose)

    def test_committed_target_keeps_one_immutable_axis_pose_tuple(self):
        tracker = DynamicPreApproachTracker(
            Pose2D(0.0, 0.0),
            DynamicTargetConfig(
                min_axis_samples=3,
                min_axis_confidence=0.0,
                max_axis_deviation_rad=math.radians(5.0),
            ),
        )
        robot = Pose2D(1.0, 0.0, math.pi)
        updates = []
        for angle_deg in (0.0, 0.0, 20.0, 0.0):
            updates.append(
                tracker.update(
                    robot_pose=robot,
                    axis_rad=math.radians(angle_deg),
                    measurement_confidence=1.0,
                    linear_speed_mps=0.0,
                    angular_speed_radps=0.0,
                )
            )
        committed = updates[-1]
        self.assertTrue(committed.accepted)
        self.assertAlmostEqual(committed.stand_axis_rad, 0.0, places=7)

        held = tracker.update(
            robot_pose=robot,
            axis_rad=None,
            measurement_confidence=0.0,
            linear_speed_mps=0.0,
            angular_speed_radps=0.0,
        )

        self.assertEqual(held.reason, "target_committed")
        self.assertEqual(held.pose, committed.pose)
        self.assertEqual(held.side_index, committed.side_index)
        self.assertEqual(held.stand_axis_rad, committed.stand_axis_rad)
        self.assertEqual(held.axis_confidence, committed.axis_confidence)
        candidates = face_normal_candidates(
            Pose2D(0.0, 0.0),
            held.stand_axis_rad,
            tracker.config.approach_offset_m,
        )
        self.assertEqual(candidates[held.side_index], held.pose)
        target_normal = math.atan2(held.pose.y_m, held.pose.x_m)
        self.assertAlmostEqual(
            abs(normalize_angle(target_normal - held.stand_axis_rad)),
            math.pi / 2.0,
            places=7,
        )

    def test_dynamic_target_commit_is_independent_of_qr_evidence(self):
        tracker = DynamicPreApproachTracker(
            Pose2D(0.0, 0.0),
            DynamicTargetConfig(min_axis_samples=1, freeze_distance_m=0.42),
        )
        first = tracker.update(
            robot_pose=Pose2D(1.0, 0.0),
            axis_rad=0.0,
            measurement_confidence=1.0,
            linear_speed_mps=0.0,
            angular_speed_radps=0.0,
        )

        held = tracker.update(
            robot_pose=first.pose,
            axis_rad=None,
            measurement_confidence=0.0,
            linear_speed_mps=0.0,
            angular_speed_radps=0.0,
            allow_opposite_side_switch=False,
        )

        self.assertFalse(held.accepted)
        self.assertEqual(held.reason, "target_committed")
        self.assertTrue(held.frozen)
        self.assertEqual(held.side_index, first.side_index)

    def test_axial_filter_wrap_preserves_the_same_physical_face(self):
        tracker = DynamicPreApproachTracker(
            Pose2D(0.0, 0.0),
            DynamicTargetConfig(
                min_axis_samples=1,
                max_axis_samples=1,
                min_axis_confidence=0.0,
                min_target_translation_m=0.0,
                min_target_yaw_change_rad=0.0,
            ),
        )
        robot = Pose2D(1.0, 0.0, math.pi)
        before = tracker.update(
            robot_pose=robot,
            axis_rad=math.radians(89.0),
            measurement_confidence=1.0,
            linear_speed_mps=0.0,
            angular_speed_radps=0.0,
        )
        after = tracker.update(
            robot_pose=robot,
            axis_rad=math.radians(-89.0),
            measurement_confidence=1.0,
            linear_speed_mps=0.0,
            angular_speed_radps=0.0,
        )
        self.assertIsNotNone(before.pose)
        self.assertIsNotNone(after.pose)
        # Once selected, the robot-facing physical side and its target are
        # immutable; a later axial representation flip cannot replan it.
        self.assertEqual(before.side_index, after.side_index)
        self.assertLess(
            math.hypot(
                before.pose.x_m - after.pose.x_m,
                before.pose.y_m - after.pose.y_m,
            ),
            0.02,
        )

    def test_approximate_pairing_uses_nearest_sample_inside_tolerance(self):
        samples = [TimedSample(1.0, "old"), TimedSample(1.09, "nearest")]
        self.assertEqual(
            nearest_timed_sample(samples, 1.10, max_delta_sec=0.12).value,
            "nearest",
        )
        self.assertIsNone(nearest_timed_sample(samples, 2.0, max_delta_sec=0.12))

    def test_buffered_sync_uses_older_complete_image_instead_of_failing_latest(self):
        images = [TimedSample(1.00, "image_complete"), TimedSample(1.19, "image_ahead")]
        scans = [TimedSample(0.98, "scan")]
        odometry = [TimedSample(1.01, "odom")]

        synchronized = newest_synchronized_triplet(
            images,
            scans,
            odometry,
            min_image_stamp_exclusive=0.0,
            max_delta_sec=0.05,
        )

        self.assertIsNotNone(synchronized)
        image, scan, odom = synchronized
        self.assertEqual(image.value, "image_complete")
        self.assertEqual(scan.value, "scan")
        self.assertEqual(odom.value, "odom")

    def test_buffered_sync_waits_without_consuming_unpaired_image(self):
        synchronized = newest_synchronized_triplet(
            [TimedSample(1.19, "image")],
            [TimedSample(0.98, "scan")],
            [TimedSample(1.18, "odom")],
            min_image_stamp_exclusive=1.0,
            max_delta_sec=0.05,
        )
        self.assertIsNone(synchronized)

    def test_moving_good_view_settles_but_does_not_commit(self):
        decision = evaluate_viewpoint(measurement(linear_speed_mps=0.03))
        self.assertEqual(decision.state, "settling")
        self.assertTrue(decision.geometrically_ready)
        self.assertFalse(decision.stationary)

    def test_oblique_view_recommends_tangential_correction(self):
        decision = evaluate_viewpoint(measurement(camera_yaw_rad=math.radians(50)))
        self.assertEqual(decision.reason, "oblique_silhouette")
        self.assertGreater(decision.recommended_pose.y_m, 0.0)
        initial_error = math.radians(50.0)
        refined_bearing = math.atan2(
            decision.recommended_pose.y_m,
            decision.recommended_pose.x_m,
        )
        self.assertAlmostEqual(refined_bearing, math.radians(20.0), places=7)
        self.assertLess(
            abs(normalize_angle(math.radians(50.0) - refined_bearing)),
            initial_error,
        )

    def test_simulation_sampling_can_use_one_bounded_35_degree_correction(self):
        decision = evaluate_viewpoint(
            measurement(camera_yaw_rad=math.radians(50)),
            ViewpointConfig(max_tangential_step_rad=math.radians(35)),
        )
        refined_bearing = math.atan2(
            decision.recommended_pose.y_m,
            decision.recommended_pose.x_m,
        )

        self.assertAlmostEqual(refined_bearing, math.radians(35), places=7)

    def test_oblique_view_never_enters_dynamic_axis_filter(self):
        axis, confidence = _conditioned_axis_input(
            camera_yaw_rad=math.radians(31.0),
            silhouette_usable=True,
            estimate_mode="face_visible",
            max_obliqueness_rad=math.radians(30.0),
            robot_pose=Pose2D(1.0, 0.0),
            stand_pose=Pose2D(0.0, 0.0),
        )
        self.assertIsNone(axis)
        self.assertEqual(confidence, 0.0)

    def test_oblique_axis_decision_never_converts_yaw_to_physical_axis(self):
        with patch(
            "scripts.aufgabe04.simulation.sim_synchronized_viewpoint_node."
            "stand_axis_from_camera_yaw"
        ) as convert:
            decision = _conditioned_axis_decision(
                camera_yaw_rad=math.radians(48.0),
                silhouette_usable=True,
                estimate_mode="face_visible",
                expected_head_px=80.0,
                min_expected_head_px=50.0,
                max_obliqueness_rad=math.radians(30.0),
                robot_pose=Pose2D(1.0, 0.0),
                stand_pose=Pose2D(0.0, 0.0),
            )

        self.assertEqual(decision.reason, "oblique_silhouette")
        self.assertIsNone(decision.axis_rad)
        self.assertEqual(decision.confidence, 0.0)
        convert.assert_not_called()

    def test_sampling_target_is_latched_en_route_and_survives_frame_loss(self):
        latch = ViewpointSamplingLatch(arrival_tolerance_m=0.10)
        first = Pose2D(0.30, 0.0, math.pi)
        ignored = latch.update(
            robot_pose=Pose2D(0.55, 0.0, math.pi),
            stationary=True,
            axis_input_reason="oblique_silhouette",
            candidate_pose=first,
            allow_start=False,
        )
        started = latch.update(
            robot_pose=Pose2D(0.55, 0.0, math.pi),
            stationary=True,
            axis_input_reason="oblique_silhouette",
            candidate_pose=first,
            allow_start=True,
        )
        moving_candidate = Pose2D(0.25, 0.15, -2.60)
        held_moving = latch.update(
            robot_pose=Pose2D(0.45, 0.0, math.pi),
            stationary=False,
            axis_input_reason="oblique_silhouette",
            candidate_pose=moving_candidate,
            allow_start=True,
        )
        held_missing = latch.update(
            robot_pose=Pose2D(0.40, 0.0, math.pi),
            stationary=False,
            axis_input_reason="silhouette_unavailable",
            candidate_pose=moving_candidate,
            allow_start=True,
        )

        self.assertFalse(ignored.active)
        self.assertTrue(started.active)
        self.assertTrue(started.advanced)
        self.assertEqual(started.target_pose, first)
        self.assertEqual(held_moving.target_pose, first)
        self.assertEqual(held_missing.target_pose, first)
        self.assertFalse(held_moving.advanced)
        self.assertFalse(held_missing.advanced)

    def test_well_conditioned_acquisition_starts_closer_sampling_target(self):
        """Regression for station B in gazebo_arrival_e2e_006.

        The initial acquisition pose is intentionally outside the final camera
        observation band.  A usable frontal silhouette there must move the
        robot to the closer diagnostic viewpoint rather than leave the
        follower holding the acquisition route until timeout.
        """

        latch = ViewpointSamplingLatch(arrival_tolerance_m=0.10)
        robot = Pose2D(
            -1.0943829462328813,
            -0.4616802455957303,
            -2.7385960874576867,
        )
        closer_viewpoint = Pose2D(
            -1.414758041920721,
            -0.5079278517452288,
            -2.7766377581748034,
        )

        update = latch.update(
            robot_pose=robot,
            stationary=True,
            axis_input_reason="well_conditioned",
            candidate_pose=closer_viewpoint,
            allow_start=True,
            view_centered=True,
            view_settled=True,
        )

        stand = Pose2D(-1.695, -0.615)
        self.assertTrue(update.active)
        self.assertTrue(update.advanced)
        self.assertEqual(update.reason, "sampling_started")
        self.assertEqual(update.target_pose, closer_viewpoint)
        self.assertLess(
            math.hypot(
                closer_viewpoint.x_m - stand.x_m,
                closer_viewpoint.y_m - stand.y_m,
            ),
            math.hypot(robot.x_m - stand.x_m, robot.y_m - stand.y_m),
        )

    def test_well_conditioned_acquisition_keeps_settle_gates_fail_closed(self):
        latch = ViewpointSamplingLatch(arrival_tolerance_m=0.10)
        robot = Pose2D(0.55, 0.0, math.pi)
        candidate = Pose2D(0.30, 0.0, math.pi)
        rejected_cases = (
            {"allow_start": False, "stationary": True, "view_centered": True, "view_settled": True},
            {"allow_start": True, "stationary": False, "view_centered": True, "view_settled": True},
            {"allow_start": True, "stationary": True, "view_centered": False, "view_settled": True},
            {"allow_start": True, "stationary": True, "view_centered": True, "view_settled": False},
        )

        for flags in rejected_cases:
            with self.subTest(flags=flags):
                update = latch.update(
                    robot_pose=robot,
                    axis_input_reason="well_conditioned",
                    candidate_pose=candidate,
                    **flags,
                )
                self.assertFalse(update.active)
                self.assertFalse(update.advanced)
                self.assertIsNone(update.target_pose)
                self.assertEqual(update.reason, "acquisition_not_settled")

    def test_unusable_acquisition_cannot_start_closer_sampling_target(self):
        robot = Pose2D(0.55, 0.0, math.pi)
        candidate = Pose2D(0.30, 0.0, math.pi)

        for reason in (
            "camera_yaw_unavailable",
            "silhouette_unavailable",
            "silhouette_not_face_visible",
            "projected_head_too_small",
        ):
            with self.subTest(reason=reason):
                latch = ViewpointSamplingLatch(arrival_tolerance_m=0.10)
                update = latch.update(
                    robot_pose=robot,
                    stationary=True,
                    axis_input_reason=reason,
                    candidate_pose=candidate,
                    allow_start=True,
                    view_centered=True,
                    view_settled=True,
                )
                self.assertFalse(update.active)
                self.assertFalse(update.advanced)
                self.assertIsNone(update.target_pose)
                self.assertEqual(update.reason, "axis_not_sampleable")

    def test_sampling_target_advances_only_when_reached_stationary_and_oblique(self):
        latch = ViewpointSamplingLatch(arrival_tolerance_m=0.10)
        first = Pose2D(0.30, 0.0, math.pi)
        second = Pose2D(
            0.30 * math.cos(math.radians(20.0)),
            0.30 * math.sin(math.radians(20.0)),
            -math.pi + math.radians(20.0),
        )
        latch.update(
            robot_pose=Pose2D(0.55, 0.0, math.pi),
            stationary=True,
            axis_input_reason="oblique_silhouette",
            candidate_pose=first,
            allow_start=True,
        )
        reached_but_moving = latch.update(
            robot_pose=first,
            stationary=False,
            axis_input_reason="oblique_silhouette",
            candidate_pose=second,
            allow_start=True,
        )
        reached_and_settled = latch.update(
            robot_pose=first,
            stationary=True,
            axis_input_reason="oblique_silhouette",
            candidate_pose=second,
            allow_start=True,
        )

        self.assertEqual(reached_but_moving.target_pose, first)
        self.assertFalse(reached_but_moving.advanced)
        self.assertEqual(reached_and_settled.target_pose, second)
        self.assertTrue(reached_and_settled.advanced)

    def test_sampling_target_does_not_advance_during_off_center_yaw_alignment(self):
        latch = ViewpointSamplingLatch(arrival_tolerance_m=0.10)
        first = Pose2D(0.30, 0.0, math.pi)
        second = Pose2D(0.25, 0.16, -2.57)
        started = latch.update(
            robot_pose=Pose2D(0.55, 0.0, math.pi),
            stationary=True,
            axis_input_reason="oblique_silhouette",
            candidate_pose=first,
            allow_start=True,
            view_centered=True,
        )
        held = latch.update(
            robot_pose=first,
            stationary=True,
            axis_input_reason="oblique_silhouette",
            candidate_pose=second,
            allow_start=True,
            view_centered=False,
        )

        self.assertTrue(started.advanced)
        self.assertFalse(held.advanced)
        self.assertEqual(held.target_pose, first)

    def test_sampling_target_does_not_advance_before_centered_view_settles(self):
        latch = ViewpointSamplingLatch(arrival_tolerance_m=0.10)
        first = Pose2D(0.30, 0.0, math.pi)
        second = Pose2D(0.25, 0.16, -2.57)
        latch.update(
            robot_pose=Pose2D(0.55, 0.0, math.pi),
            stationary=True,
            axis_input_reason="oblique_silhouette",
            candidate_pose=first,
            allow_start=True,
            view_centered=True,
            view_settled=True,
        )
        held = latch.update(
            robot_pose=first,
            stationary=True,
            axis_input_reason="oblique_silhouette",
            candidate_pose=second,
            allow_start=True,
            view_centered=True,
            view_settled=False,
        )

        self.assertFalse(held.advanced)
        self.assertEqual(held.target_pose, first)

    def test_initial_physical_publish_waits_for_stable_face_ids(self):
        self.assertTrue(
            should_defer_initial_physical_recommendation(
                acquiring_axis=False,
                face_identity_resolved=False,
                hard_qr_latched=False,
            )
        )
        self.assertFalse(
            should_defer_initial_physical_recommendation(
                acquiring_axis=False,
                face_identity_resolved=True,
                hard_qr_latched=False,
            )
        )
        self.assertFalse(
            should_defer_initial_physical_recommendation(
                acquiring_axis=False,
                face_identity_resolved=False,
                hard_qr_latched=True,
            )
        )

    def test_sampling_candidates_are_unresolved_antipodal_sampling_rays(self):
        stand = Pose2D(0.0, 0.0)
        sample = Pose2D(0.0, 0.30, -math.pi / 2.0)
        candidates = provisional_viewpoint_candidates(
            stand,
            sample,
            near_id="sampling_near",
            far_id="sampling_far",
        )

        self.assertEqual(candidates[0].face_id, "sampling_near")
        self.assertAlmostEqual(candidates[0].pose.x_m, sample.x_m)
        self.assertAlmostEqual(candidates[0].pose.y_m, sample.y_m)
        self.assertAlmostEqual(candidates[0].pose.yaw_rad, sample.yaw_rad)
        self.assertFalse(candidates[0].identity_resolved)
        self.assertEqual(candidates[1].face_id, "sampling_far")
        self.assertAlmostEqual(candidates[1].pose.x_m, 0.0, places=7)
        self.assertAlmostEqual(candidates[1].pose.y_m, -0.30, places=7)
        self.assertAlmostEqual(
            abs(normalize_angle(
                candidates[1].outward_normal_rad
                - candidates[0].outward_normal_rad
            )),
            math.pi,
        )

    def test_projected_head_too_small_never_enters_dynamic_axis_filter(self):
        axis, confidence = _conditioned_axis_input(
            camera_yaw_rad=math.radians(5.0),
            silhouette_usable=True,
            estimate_mode="face_visible",
            expected_head_px=16.0,
            min_expected_head_px=35.0,
            max_obliqueness_rad=math.radians(30.0),
            robot_pose=Pose2D(1.0, 0.0),
            stand_pose=Pose2D(0.0, 0.0),
        )
        self.assertIsNone(axis)
        self.assertEqual(confidence, 0.0)

    def test_projected_head_large_enough_can_enter_dynamic_axis_filter(self):
        axis, confidence = _conditioned_axis_input(
            camera_yaw_rad=math.radians(5.0),
            silhouette_usable=True,
            estimate_mode="face_visible",
            expected_head_px=36.0,
            min_expected_head_px=35.0,
            max_obliqueness_rad=math.radians(30.0),
            robot_pose=Pose2D(1.0, 0.0),
            stand_pose=Pose2D(0.0, 0.0),
        )
        self.assertIsNotNone(axis)
        self.assertGreater(confidence, 0.0)

    def test_oblique_viewpoint_cannot_override_committed_tracker_pose(self):
        tracker = DynamicPreApproachTracker(
            Pose2D(0.0, 0.0),
            DynamicTargetConfig(min_axis_samples=1, freeze_distance_m=0.2),
        )
        robot = Pose2D(1.0, 0.0, math.pi)
        seeded = tracker.update(
            robot_pose=robot,
            axis_rad=0.0,
            measurement_confidence=1.0,
            linear_speed_mps=0.0,
            angular_speed_radps=0.0,
        )
        decision = evaluate_viewpoint(
            measurement(
                robot_pose=robot,
                camera_yaw_rad=math.radians(50.0),
            )
        )
        stale = tracker.update(
            robot_pose=robot,
            axis_rad=None,
            measurement_confidence=0.0,
            linear_speed_mps=0.0,
            angular_speed_radps=0.0,
        )
        selected = select_published_viewpoint_pose(
            decision.recommended_pose, stale
        )
        self.assertEqual(stale.reason, "target_committed")
        self.assertEqual(selected, seeded.pose)
        self.assertNotEqual(selected, decision.recommended_pose)

    def test_real_axis_reseeds_provisional_face_identity_without_hard_qr(self):
        resolver = StableFaceResolver()
        resolver.update(stream_id="run", outward_normals_rad=(0.0, math.pi))
        fallback = resolver.update(
            stream_id="run", outward_normals_rad=(0.0, math.pi)
        )
        self.assertTrue(fallback.identity_resolved)
        self.assertTrue(
            should_reseed_face_resolver(
                "provisional", "measured", hard_qr_latched=False
            )
        )
        resolver.reset("run")
        first_real = resolver.update(
            stream_id="run",
            outward_normals_rad=(math.pi / 2.0, -math.pi / 2.0),
        )
        second_real = resolver.update(
            stream_id="run",
            outward_normals_rad=(math.pi / 2.0, -math.pi / 2.0),
        )
        self.assertFalse(first_real.identity_resolved)
        self.assertTrue(second_real.identity_resolved)

    def test_hard_qr_is_suspended_while_face_identity_is_unresolved(self):
        latch = QrFaceLatch()
        latch.update(
            stream_id="run",
            observation=QrBindingObservation(
                face_id="face_a",
                confidence=1.0,
                provenance="sim_qr_consensus",
                registry_match=True,
                inside_target_roi=True,
                distinct_fresh_frame_consensus=True,
                visibility_margin_rad=math.radians(20.0),
            ),
        )
        retained = latch.update(stream_id="run", observation=None)
        suspended = suspend_qr_binding_while_identity_unresolved(
            retained,
            identity_resolved=False,
            hard_qr_latched=True,
        )
        self.assertTrue(retained.evidence.hard)
        self.assertFalse(suspended.evidence.hard)
        self.assertFalse(suspended.evidence.valid)
        self.assertEqual(
            suspended.reason, "face_identity_unresolved_latch_suspended"
        )

    def test_synchronized_sensor_frame_contract_has_explicit_defaults(self):
        args = build_synchronized_parser().parse_args(
            [
                "--stand-x",
                "0",
                "--stand-y",
                "0",
                "--status-json",
                "status.json",
                "--recommended-pose-json",
                "recommendation.json",
                "--observation-json",
                "observation.json",
            ]
        )
        self.assertEqual(args.map_frame, "odom")
        self.assertEqual(args.base_frame, "base_footprint")
        self.assertEqual(args.scan_frame, "base_scan")
        self.assertEqual(args.min_silhouette_head_px, 50.0)
        self.assertEqual(args.camera_frame, "camera_link")
        self.assertEqual(args.dynamic_min_axis_samples, 7)
        self.assertEqual(args.max_obliqueness_deg, 20.0)
        self.assertEqual(args.max_tangential_step_deg, 35.0)
        self.assertAlmostEqual(args.axis_acquisition_distance_m, 0.55)
        self.assertAlmostEqual(args.axis_acquisition_arrival_tolerance_m, 0.10)
        self.assertAlmostEqual(args.sampling_arrival_tolerance_m, 0.10)

    def test_stationary_well_conditioned_view_is_ready(self):
        decision = evaluate_viewpoint(measurement())
        self.assertEqual(decision.state, "stationary_consensus")
        self.assertTrue(decision.geometrically_ready)

    def test_settle_gate_resets_on_motion(self):
        gate = StationarySettleGate(0.4)
        self.assertFalse(gate.update(stamp_sec=1.0, ready=True))
        self.assertFalse(gate.update(stamp_sec=1.2, ready=False))
        self.assertFalse(gate.update(stamp_sec=2.0, ready=True))
        self.assertTrue(gate.update(stamp_sec=2.5, ready=True))


if __name__ == "__main__":
    unittest.main()
