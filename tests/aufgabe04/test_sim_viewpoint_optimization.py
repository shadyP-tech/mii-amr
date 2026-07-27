import json
import math
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.aufgabe04.navigation.axis_acquisition_feedback import (
    AXIS_ACQUISITION_FEEDBACK_CONTRACT,
    AXIS_ACQUISITION_FEEDBACK_SCHEMA_VERSION,
    axis_acquisition_feedback_binding,
    canonical_json_sha256,
    load_axis_acquisition_feedback,
    write_axis_acquisition_feedback,
)
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.viewpoint_sampling_contract import (
    ViewpointSamplingMaterialTarget,
)
from scripts.aufgabe04.perception.camera_stand_observation import (
    stand_axis_from_camera_yaw,
)
from scripts.aufgabe04.navigation.viewpoint_recommendation import (
    FaceCandidate,
    MaterialTarget,
    QrBindingObservation,
    QrFaceLatch,
    SideEvidence,
    StandGeometry,
    StableFaceResolver,
    SynchronizedViewpointRecommendation,
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
    ViewpointSamplingOdomSample,
    evaluate_viewpoint,
    face_normal_candidates,
    newest_synchronized_triplet,
    nearest_timed_sample,
    normalize_angle,
    sampling_recenter_pose,
    select_viewpoint_sampling_odom_replay,
    viewpoint_sampling_history_identity,
    viewpoint_sampling_history_identity_changed,
)
from scripts.aufgabe04.simulation.sim_synchronized_viewpoint_node import (
    SimSynchronizedViewpointNode,
    _conditioned_axis_decision,
    _conditioned_axis_input,
    build_parser as build_synchronized_parser,
    camera_yaw_from_target_line_of_sight,
    center_distance_with_lidar_presence,
    axis_acquisition_search_target,
    evaluate_axis_acquisition_feedback,
    main as synchronized_viewpoint_main,
    select_published_viewpoint_pose,
    provisional_viewpoint_candidates,
    should_attempt_qr_face_binding,
    should_defer_initial_physical_recommendation,
    should_reseed_face_resolver,
    suspend_qr_binding_while_identity_unresolved,
    unresolved_point_target_poses,
    unresolved_point_target_reached,
)


def acquisition_recommendation(
    *,
    index: int = 1,
    observation_unix_sec: float = 100.0,
    sensor_stamp_sec: float = 10.0,
    robot_pose: Pose2D = Pose2D(-1.0539397321, 0.3967815349, 2.9360318704),
) -> SynchronizedViewpointRecommendation:
    stand = Pose2D(-1.5991445773, 0.5427822810)
    if index == 1:
        near = Pose2D(-1.1225253454, 0.8172528765, -2.6191049830)
        far = Pose2D(-2.0757638092, 0.2683116855, 0.5224876706)
        near_normal = 0.5224876705552723
    else:
        near_normal = -0.2629108199521244 + index * 0.1
        near = Pose2D(
            stand.x_m + 0.55 * math.cos(near_normal),
            stand.y_m + 0.55 * math.sin(near_normal),
            normalize_angle(near_normal + math.pi),
        )
        far_normal = normalize_angle(near_normal + math.pi)
        far = Pose2D(
            stand.x_m + 0.55 * math.cos(far_normal),
            stand.y_m + 0.55 * math.sin(far_normal),
            normalize_angle(far_normal + math.pi),
        )
    far_normal = normalize_angle(near_normal + math.pi)
    return SynchronizedViewpointRecommendation(
        schema_version=1,
        simulation_only=True,
        stream_id="survey-v3-feedback",
        stand_id="A",
        planning_frame="odom",
        source="synchronized_lidar_camera_viewpoint",
        observation_unix_sec=observation_unix_sec,
        sensor_stamp_sec=sensor_stamp_sec,
        stand=StandGeometry(stand, 0.06, 0.02, "synchronized_lidar_cluster"),
        robot_pose=robot_pose,
        axis_confidence=0.0,
        axis_state="axis_acquisition",
        face_candidates=(
            FaceCandidate(
                f"acquisition_near_{index:02d}",
                near_normal,
                near,
                False,
            ),
            FaceCandidate(
                f"acquisition_far_{index:02d}",
                far_normal,
                far,
                False,
            ),
        ),
        side_evidence=SideEvidence(
            "none",
            0.0,
            False,
            False,
            None,
            "axis_acquisition_axis_uncommitted",
        ),
        material_target=MaterialTarget(
            f"acquisition_near_{index:02d}",
            near,
            "axis_acquisition",
        ),
    )


def pending_feedback_payload(
    recommendation: SynchronizedViewpointRecommendation,
    *,
    created_unix_sec: float = 100.0,
) -> dict:
    binding = axis_acquisition_feedback_binding(recommendation)
    return {
        "schema_version": AXIS_ACQUISITION_FEEDBACK_SCHEMA_VERSION,
        "contract": AXIS_ACQUISITION_FEEDBACK_CONTRACT,
        "simulation_only": True,
        "state": "pending",
        "created_unix_sec": created_unix_sec,
        "source_observation_unix_sec": recommendation.observation_unix_sec,
        "source_sensor_stamp_sec": recommendation.sensor_stamp_sec,
        "arrival_tolerance_m": 0.10,
        "binding": binding,
        "binding_sha256": canonical_json_sha256(binding),
        "held_active_target": {
            "face_id": "acquisition_near_00",
            "evidence_state": "axis_acquisition",
            "pose": {
                "x_m": -1.0680438671,
                "y_m": 0.3998416094,
                "yaw_rad": 2.8786821607,
            },
        },
        "held_distance_m": 0.014434,
        "rejections": [
            {
                "face_id": recommendation.face_candidates[0].face_id,
                "failure_reason": "acquisition_target_not_traversable",
            },
            {
                "face_id": recommendation.face_candidates[1].face_id,
                "failure_reason": "acquisition_target_not_traversable",
            },
        ],
    }


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
    def test_v3_feedback_consumes_newer_same_target_once_without_motion(self):
        planner_frame = acquisition_recommendation()
        newer_observer_frame = acquisition_recommendation(
            observation_unix_sec=100.2,
            sensor_stamp_sec=10.2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "axis_acquisition_feedback.json"
            write_axis_acquisition_feedback(
                path,
                pending_feedback_payload(planner_frame),
            )

            accepted = evaluate_axis_acquisition_feedback(
                path,
                last_recommendation=newer_observer_frame,
                current_robot_pose=newer_observer_frame.robot_pose,
                current_search_index=1,
                max_search_targets=7,
                arrival_tolerance_m=0.10,
                max_age_sec=3.0,
                now_unix_sec=100.25,
                current_stationary=True,
                current_view_centered=True,
            )
            duplicate = evaluate_axis_acquisition_feedback(
                path,
                last_recommendation=newer_observer_frame,
                current_robot_pose=newer_observer_frame.robot_pose,
                current_search_index=1,
                max_search_targets=7,
                arrival_tolerance_m=0.10,
                max_age_sec=3.0,
                now_unix_sec=100.3,
                current_stationary=True,
                current_view_centered=True,
            )
            stored = load_axis_acquisition_feedback(path)

        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.next_search_index, 2)
        self.assertEqual(accepted.reason, "static_target_rejection_consumed")
        self.assertFalse(duplicate.accepted)
        self.assertEqual(duplicate.reason, "feedback_already_consumed")
        self.assertEqual(stored["state"], "consumed")
        self.assertEqual(stored["consumed_unix_sec"], 100.25)

    def test_feedback_rejects_unsafe_stale_mismatched_or_unsettled_input(self):
        recommendation = acquisition_recommendation()
        cases = (
            (
                "live_drift",
                dict(
                    current_robot_pose=Pose2D(-0.85, 0.40, 2.9),
                ),
                100.1,
                "hold is unsafe",
            ),
            (
                "rotating",
                dict(current_stationary=False),
                100.1,
                "not safely settled",
            ),
            (
                "off_center",
                dict(current_view_centered=False),
                100.1,
                "not safely settled",
            ),
            (
                "stale",
                {},
                104.0,
                "stale",
            ),
        )
        for label, changes, now, expected_reason in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "axis_acquisition_feedback.json"
                write_axis_acquisition_feedback(
                    path,
                    pending_feedback_payload(recommendation),
                )
                arguments = dict(
                    last_recommendation=recommendation,
                    current_robot_pose=recommendation.robot_pose,
                    current_search_index=1,
                    max_search_targets=7,
                    arrival_tolerance_m=0.10,
                    max_age_sec=3.0,
                    now_unix_sec=now,
                    current_stationary=True,
                    current_view_centered=True,
                )
                arguments.update(changes)
                decision = evaluate_axis_acquisition_feedback(path, **arguments)
                self.assertFalse(decision.accepted)
                self.assertIn(expected_reason, decision.reason)
                self.assertEqual(
                    load_axis_acquisition_feedback(path)["state"],
                    "pending",
                )

        exhausted_recommendation = acquisition_recommendation(index=6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "axis_acquisition_feedback.json"
            write_axis_acquisition_feedback(
                path,
                pending_feedback_payload(exhausted_recommendation),
            )
            exhausted = evaluate_axis_acquisition_feedback(
                path,
                last_recommendation=exhausted_recommendation,
                current_robot_pose=exhausted_recommendation.robot_pose,
                current_search_index=6,
                max_search_targets=7,
                arrival_tolerance_m=0.10,
                max_age_sec=3.0,
                now_unix_sec=100.1,
            )
        self.assertFalse(exhausted.accepted)
        self.assertIn("bounded acquisition search", exhausted.reason)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "axis_acquisition_feedback.json"
            write_axis_acquisition_feedback(
                path,
                pending_feedback_payload(recommendation),
            )
            mismatched = evaluate_axis_acquisition_feedback(
                path,
                last_recommendation=acquisition_recommendation(index=2),
                current_robot_pose=recommendation.robot_pose,
                current_search_index=1,
                max_search_targets=7,
                arrival_tolerance_m=0.10,
                max_age_sec=3.0,
                now_unix_sec=100.1,
            )
            older_observer = acquisition_recommendation(
                observation_unix_sec=99.9,
                sensor_stamp_sec=9.9,
            )
            source_ahead = evaluate_axis_acquisition_feedback(
                path,
                last_recommendation=older_observer,
                current_robot_pose=recommendation.robot_pose,
                current_search_index=1,
                max_search_targets=7,
                arrival_tolerance_m=0.10,
                max_age_sec=3.0,
                now_unix_sec=100.1,
            )
        self.assertFalse(mismatched.accepted)
        self.assertIn("exact last recommendation", mismatched.reason)
        self.assertFalse(source_ahead.accepted)
        self.assertIn("newer than the observer", source_ahead.reason)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "axis_acquisition_feedback.json"
            path.write_text("{malformed")
            malformed = evaluate_axis_acquisition_feedback(
                path,
                last_recommendation=recommendation,
                current_robot_pose=recommendation.robot_pose,
                current_search_index=1,
                max_search_targets=7,
                arrival_tolerance_m=0.10,
                max_age_sec=3.0,
                now_unix_sec=100.1,
            )
        self.assertFalse(malformed.accepted)
        self.assertIn("feedback_rejected", malformed.reason)

    def test_feedback_binding_rejects_internally_mismatched_geometry(self):
        recommendation = acquisition_recommendation()
        payload = pending_feedback_payload(recommendation)
        payload["binding"]["material_target"]["pose"]["x_m"] += 0.01
        payload["binding_sha256"] = canonical_json_sha256(payload["binding"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "axis_acquisition_feedback.json"
            with self.assertRaisesRegex(ValueError, "geometry mismatch"):
                write_axis_acquisition_feedback(path, payload)

    def test_acquisition_search_alternates_bounded_rays_around_stand(self):
        stand = Pose2D(1.0, -2.0)
        targets = [
            axis_acquisition_search_target(
                stand,
                reference_normal_rad=0.0,
                distance_m=0.55,
                target_index=index,
                step_rad=math.pi / 4.0,
            )
            for index in range(5)
        ]
        bearings = [
            normalize_angle(
                math.atan2(target.y_m - stand.y_m, target.x_m - stand.x_m)
            )
            for target in targets
        ]

        expected = (0.0, math.pi / 4.0, -math.pi / 4.0, math.pi / 2.0, -math.pi / 2.0)
        for actual, wanted in zip(bearings, expected):
            self.assertAlmostEqual(actual, wanted)
        for target in targets:
            self.assertAlmostEqual(
                math.hypot(target.x_m - stand.x_m, target.y_m - stand.y_m),
                0.55,
            )
            self.assertAlmostEqual(
                abs(normalize_angle(target.yaw_rad - math.atan2(
                    stand.y_m - target.y_m,
                    stand.x_m - target.x_m,
                ))),
                0.0,
            )

    def test_center_standoff_gate_retains_live_lidar_presence(self):
        distance = center_distance_with_lidar_presence(
            robot_x_m=0.2740,
            robot_y_m=0.4240,
            stand_x_m=0.408758,
            stand_y_m=0.680168,
            stand_radius_m=0.06,
            stand_uncertainty_m=0.02,
            range_tolerance_m=0.03,
            lidar_surface_distance_m=0.2502,
        )

        self.assertAlmostEqual(distance, 0.2895, places=3)
        self.assertIsNone(
            center_distance_with_lidar_presence(
                robot_x_m=0.2740,
                robot_y_m=0.4240,
                stand_x_m=0.408758,
                stand_y_m=0.680168,
                stand_radius_m=0.06,
                stand_uncertainty_m=0.02,
                range_tolerance_m=0.03,
                lidar_surface_distance_m=None,
            )
        )
        self.assertIsNone(
            center_distance_with_lidar_presence(
                robot_x_m=0.9214,
                robot_y_m=0.5817,
                stand_x_m=0.408758,
                stand_y_m=0.680168,
                stand_radius_m=0.06,
                stand_uncertainty_m=0.02,
                range_tolerance_m=0.03,
                lidar_surface_distance_m=1.239,
            )
        )

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

    def test_simulation_sampling_damps_camera_correction_before_step_clamp(self):
        decision = evaluate_viewpoint(
            measurement(camera_yaw_rad=math.radians(50)),
            ViewpointConfig(max_tangential_step_rad=math.radians(35)),
        )
        refined_bearing = math.atan2(
            decision.recommended_pose.y_m,
            decision.recommended_pose.x_m,
        )

        self.assertAlmostEqual(refined_bearing, math.radians(25), places=7)

    def test_unity_tangential_gain_still_obeys_hard_step_clamp(self):
        decision = evaluate_viewpoint(
            measurement(camera_yaw_rad=math.radians(50)),
            ViewpointConfig(
                max_tangential_step_rad=math.radians(35),
                tangential_correction_gain=1.0,
            ),
        )

        self.assertAlmostEqual(
            math.atan2(
                decision.recommended_pose.y_m,
                decision.recommended_pose.x_m,
            ),
            math.radians(35),
            places=7,
        )

    def test_stand_b_near_frontal_pnp_jump_is_halved(self):
        decision = evaluate_viewpoint(
            measurement(camera_yaw_rad=math.radians(-28.732025965730422)),
            ViewpointConfig(max_tangential_step_rad=math.radians(35)),
        )

        self.assertAlmostEqual(
            math.atan2(
                decision.recommended_pose.y_m,
                decision.recommended_pose.x_m,
            ),
            math.radians(-14.366012982865211),
            places=7,
        )

    def test_tangential_correction_gain_is_bounded(self):
        for value in (0.0, -0.1, 1.01, math.nan, math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    r"tangential correction gain must be in \(0, 1\]",
                ):
                    ViewpointConfig(tangential_correction_gain=value)

    def test_repeat16_recenter_preserves_stand_center_and_exact_geometry(self):
        target = Pose2D(
            -0.037311778590026795,
            0.19741015889821922,
            1.1303302378451376,
        )
        robot = Pose2D(
            -0.016786847655917692,
            0.19744789107551103,
            1.048374408778151,
        )
        target_distance_m = 0.33

        recentered = sampling_recenter_pose(
            current_target_pose=target,
            robot_pose=robot,
            center_error_rad=math.radians(-12.134895),
            target_distance_m=target_distance_m,
            max_correction_rad=math.radians(20.0),
        )

        self.assertIsNotNone(recentered)
        assert recentered is not None
        self.assertAlmostEqual(recentered.x_m, 0.0025206826895567552)
        self.assertAlmostEqual(recentered.y_m, 0.1817058707441434)
        self.assertAlmostEqual(recentered.yaw_rad, 1.2601682809119483)
        old_stand = (
            target.x_m + target_distance_m * math.cos(target.yaw_rad),
            target.y_m + target_distance_m * math.sin(target.yaw_rad),
        )
        new_stand = (
            recentered.x_m
            + target_distance_m * math.cos(recentered.yaw_rad),
            recentered.y_m
            + target_distance_m * math.sin(recentered.yaw_rad),
        )
        self.assertAlmostEqual(new_stand[0], old_stand[0], places=14)
        self.assertAlmostEqual(new_stand[1], old_stand[1], places=14)
        self.assertAlmostEqual(
            math.hypot(
                recentered.x_m - old_stand[0],
                recentered.y_m - old_stand[1],
            ),
            target_distance_m,
            places=14,
        )
        self.assertAlmostEqual(
            normalize_angle(recentered.yaw_rad - robot.yaw_rad),
            math.radians(12.134895),
        )

    def test_sampling_recenter_clamps_center_error_and_fails_closed(self):
        target = Pose2D(0.0, 0.0, 0.0)
        robot = Pose2D(0.0, 0.0, 0.3)
        maximum = math.radians(20.0)
        bounded = sampling_recenter_pose(
            current_target_pose=target,
            robot_pose=robot,
            center_error_rad=math.radians(-50.0),
            target_distance_m=0.33,
            max_correction_rad=maximum,
        )
        self.assertIsNotNone(bounded)
        assert bounded is not None
        self.assertAlmostEqual(
            abs(normalize_angle(bounded.yaw_rad - robot.yaw_rad)),
            maximum,
        )

        invalid_cases = (
            {"current_target_pose": None},
            {"current_target_pose": Pose2D(math.nan, 0.0, 0.0)},
            {"robot_pose": Pose2D(0.0, math.inf, 0.0)},
            {"center_error_rad": None},
            {"center_error_rad": math.nan},
            {"center_error_rad": "invalid"},
            {"target_distance_m": 0.0},
            {"max_correction_rad": math.inf},
            {"max_correction_rad": 0.0},
        )
        common = {
            "current_target_pose": target,
            "robot_pose": robot,
            "center_error_rad": 0.2,
            "target_distance_m": 0.33,
            "max_correction_rad": maximum,
        }
        for changes in invalid_cases:
            with self.subTest(changes=changes):
                self.assertIsNone(
                    sampling_recenter_pose(**{**common, **changes})
                )

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
        latch = ViewpointSamplingLatch()
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

        latch = ViewpointSamplingLatch()
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
        latch = ViewpointSamplingLatch()
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
                latch = ViewpointSamplingLatch()
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
        latch = ViewpointSamplingLatch()
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

    def test_sampling_antipode_is_equivalent_only_for_unresolved_point_phase(self):
        latch = ViewpointSamplingLatch()
        near = Pose2D(0.30, 0.0, math.pi)
        far = Pose2D(-0.30, 0.0, 0.0)
        next_target = Pose2D(-0.28, 0.10, -0.34)
        latch.update(
            robot_pose=Pose2D(0.55, 0.0, math.pi),
            stationary=True,
            axis_input_reason="oblique_silhouette",
            candidate_pose=near,
            allow_start=True,
        )

        # Selecting the planner-substituted antipode is itself a material
        # target change, so it must reset before a later strict observation
        # can arm that exact target.
        reset_for_antipode = latch.update(
            robot_pose=far,
            stationary=False,
            axis_input_reason="oblique_silhouette",
            candidate_pose=next_target,
            allow_start=True,
            equivalent_target_poses=(near, far),
        )
        self.assertEqual(latch.arrival_target_pose, far)
        advanced = latch.update(
            robot_pose=far,
            stationary=True,
            axis_input_reason="oblique_silhouette",
            candidate_pose=next_target,
            allow_start=True,
            equivalent_target_poses=(near, far),
        )

        self.assertFalse(reset_for_antipode.advanced)
        self.assertTrue(advanced.advanced)
        self.assertEqual(advanced.target_pose, next_target)

    def test_unresolved_acquisition_recognizes_planner_selected_antipode(self):
        near = Pose2D(-0.55, 0.0, 0.0)
        far = Pose2D(0.55, 0.0, math.pi)
        recommendation = SynchronizedViewpointRecommendation(
            schema_version=1,
            simulation_only=True,
            stream_id="survey-test",
            stand_id="A",
            planning_frame="odom",
            source="synchronized_lidar_camera_viewpoint",
            observation_unix_sec=100.0,
            sensor_stamp_sec=10.0,
            stand=StandGeometry(Pose2D(0.0, 0.0), 0.06, 0.02, "lidar_cluster"),
            robot_pose=near,
            axis_confidence=0.0,
            axis_state="axis_acquisition",
            face_candidates=(
                FaceCandidate("acquisition_near_01", math.pi, near, False),
                FaceCandidate("acquisition_far_01", 0.0, far, False),
            ),
            side_evidence=SideEvidence(
                "none", 0.0, False, False, None, "axis_uncommitted"
            ),
            material_target=MaterialTarget(
                "acquisition_near_01", near, "axis_acquisition"
            ),
        )

        self.assertEqual(
            unresolved_point_target_poses(
                recommendation,
                expected_axis_state="axis_acquisition",
            ),
            (near, far),
        )
        self.assertTrue(
            unresolved_point_target_reached(
                recommendation,
                far,
                expected_axis_state="axis_acquisition",
                arrival_tolerance_m=0.10,
            )
        )
        resolved = recommendation.__class__(
            **{
                **recommendation.__dict__,
                "face_candidates": (
                    FaceCandidate("acquisition_near_01", math.pi, near, True),
                    FaceCandidate("acquisition_far_01", 0.0, far, True),
                ),
            }
        )
        self.assertFalse(
            unresolved_point_target_reached(
                resolved,
                far,
                expected_axis_state="axis_acquisition",
                arrival_tolerance_m=0.10,
            )
        )

    def test_sampling_target_does_not_advance_during_off_center_yaw_alignment(self):
        latch = ViewpointSamplingLatch()
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
        self.assertEqual(held.reason, "sampling_recenter_unavailable")

    def test_repeat21_buffered_odom_recovers_68ms_strict_entry_then_holds(self):
        target = Pose2D(
            -0.05681815712818519,
            0.20740931174165717,
            1.0638940917675712,
        )
        final_pose = Pose2D(
            -0.036082240181171614,
            0.20493972776397115,
            0.9861425073650157,
        )
        latch = ViewpointSamplingLatch()
        common = {
            "axis_input_reason": "oblique_silhouette",
            "candidate_pose": target,
            "allow_start": True,
            "target_face_id": "sampling_near",
        }
        initialized = latch.update(
            robot_pose=Pose2D(-0.14, 0.27, 0.7),
            stationary=True,
            **common,
        )
        self.assertFalse(initialized.arrival_status["arrived"])

        strict_distance_m = 0.0169345
        strict_pose = Pose2D(
            target.x_m
            + strict_distance_m * math.cos(target.yaw_rad + math.pi / 2.0),
            target.y_m
            + strict_distance_m * math.sin(target.yaw_rad + math.pi / 2.0),
            target.yaw_rad,
        )
        quaternion = (
            0.0,
            0.0,
            math.sin(strict_pose.yaw_rad / 2.0),
            math.cos(strict_pose.yaw_rad / 2.0),
        )
        history = tuple(
            ViewpointSamplingOdomSample(
                stamp_sec=stamp_sec,
                pose=strict_pose,
                parent_frame="odom",
                child_frame="base_footprint",
                quaternion_xyzw=quaternion,
            )
            for stamp_sec in (9078.709, 9078.743, 9078.777)
        )
        replay = select_viewpoint_sampling_odom_replay(
            history,
            target_pose=target,
            target_activation_stamp_sec=9078.5,
            last_checked_odom_stamp_sec=9078.6,
            current_image_stamp_sec=9078.9,
            expected_parent_frame="odom",
            expected_child_frame="base_footprint",
            strict_entry_tolerance_m=0.017,
        )
        self.assertEqual(
            tuple(sample.stamp_sec for sample in replay.samples),
            (9078.709, 9078.743, 9078.777),
        )
        self.assertEqual(replay.diagnostics["strict_entry_sample_count"], 3)
        self.assertAlmostEqual(
            replay.diagnostics["minimum_target_distance_m"],
            strict_distance_m,
        )
        for sample in replay.samples:
            latch.observe_arrival_pose(sample.pose)

        held = latch.update(
            robot_pose=final_pose,
            stationary=True,
            **common,
        )

        self.assertTrue(held.arrival_status["arrived"])
        self.assertTrue(held.arrival_status["strict_ever_armed"])
        self.assertTrue(held.arrival_status["hold_valid"])
        self.assertFalse(held.arrival_status["strict_entry_within_limit"])
        self.assertAlmostEqual(
            held.arrival_status["metrics"]["target_envelope_distance_m"],
            0.020882459066314207,
        )

    def test_observer_history_diagnostics_bind_strict_entry_to_target_hash(self):
        target = Pose2D(0.0, 0.0, math.pi / 2.0)
        latch = ViewpointSamplingLatch()
        latch.update(
            robot_pose=Pose2D(0.20, 0.0, 0.0),
            stationary=True,
            axis_input_reason="oblique_silhouette",
            candidate_pose=target,
            allow_start=True,
            target_face_id="sampling_near",
        )
        material_target = latch.arrival_material_target
        self.assertIsNotNone(material_target)
        assert material_target is not None

        def odom_sample(stamp_sec: float, distance_m: float):
            yaw = target.yaw_rad
            message = SimpleNamespace(
                header=SimpleNamespace(frame_id="odom"),
                child_frame_id="base_footprint",
                pose=SimpleNamespace(
                    pose=SimpleNamespace(
                        position=SimpleNamespace(
                            x=distance_m,
                            y=0.0,
                        ),
                        orientation=SimpleNamespace(
                            x=0.0,
                            y=0.0,
                            z=math.sin(yaw / 2.0),
                            w=math.cos(yaw / 2.0),
                        ),
                    )
                ),
            )
            return TimedSample(stamp_sec, message)

        node = object.__new__(SimSynchronizedViewpointNode)
        node.args = SimpleNamespace(
            stream_id="stream-repeat21",
            map_frame="odom",
            base_frame="base_footprint",
            sampling_arrival_tolerance_m=0.017,
        )
        node.sampling_latch = latch
        node.sampling_arrival_history_target = material_target
        node.sampling_arrival_history_stream_id = node.args.stream_id
        node.sampling_arrival_target_activation_stamp_sec = 100.0
        node.sampling_arrival_last_checked_odom_stamp_sec = 100.0
        node.sampling_arrival_history_diagnostics = {}
        node.odometry = deque(
            (
                odom_sample(100.1, 0.0165),
                odom_sample(100.2, 0.0160),
            ),
            maxlen=40,
        )

        self.assertTrue(
            node._replay_sampling_arrival_history(
                current_image_stamp_sec=100.3,
                current_odom_stamp_sec=100.4,
            )
        )
        diagnostics = node.sampling_arrival_history_diagnostics
        self.assertEqual(diagnostics["history_processed_sample_count"], 2)
        self.assertEqual(diagnostics["strict_history_sample_count"], 2)
        self.assertEqual(diagnostics["checked_from_stamp_sec"], 100.0)
        self.assertEqual(diagnostics["checked_to_stamp_sec"], 100.3)
        self.assertEqual(diagnostics["strict_history_min_distance_m"], 0.016)
        self.assertEqual(
            diagnostics["strict_history_min_distance_stamp_sec"],
            100.2,
        )
        self.assertTrue(diagnostics["strict_armed_by_history"])
        self.assertEqual(
            diagnostics["target_identity"]["stream_id"],
            "stream-repeat21",
        )
        self.assertEqual(len(diagnostics["target_identity"]["sha256"]), 64)
        self.assertEqual(
            node.sampling_arrival_last_checked_odom_stamp_sec,
            100.2,
        )
        node.args.stream_id = "stream-restarted"
        self.assertTrue(
            node._replay_sampling_arrival_history(
                current_image_stamp_sec=100.5,
                current_odom_stamp_sec=100.6,
            )
        )
        self.assertEqual(
            node.sampling_arrival_history_diagnostics["state"],
            "target_history_discarded",
        )
        self.assertTrue(
            node.sampling_arrival_history_diagnostics[
                "history_discarded_on_target_change"
            ]
        )
        self.assertFalse(node.sampling_latch.arrival_status()["arrived"])
        self.assertIsNone(node.sampling_arrival_history_target)

    def test_buffered_arrival_odom_is_target_bounded_and_idempotent(self):
        target = Pose2D(0.0, 0.0, math.pi / 2.0)
        quaternion = (
            0.0,
            0.0,
            math.sin(target.yaw_rad / 2.0),
            math.cos(target.yaw_rad / 2.0),
        )

        def sample(stamp_sec: float, distance_m: float):
            return ViewpointSamplingOdomSample(
                stamp_sec=stamp_sec,
                pose=Pose2D(distance_m, 0.0, target.yaw_rad),
                parent_frame="odom",
                child_frame="base_footprint",
                quaternion_xyzw=quaternion,
            )

        replay = select_viewpoint_sampling_odom_replay(
            (
                sample(10.0, 0.010),
                sample(10.1, 0.016),
                sample(10.2, 0.015),
                sample(10.3, 0.014),
            ),
            target_pose=target,
            target_activation_stamp_sec=10.0,
            last_checked_odom_stamp_sec=10.1,
            current_image_stamp_sec=10.2,
            expected_parent_frame="odom",
            expected_child_frame="base_footprint",
            strict_entry_tolerance_m=0.017,
        )

        self.assertEqual(
            tuple(item.stamp_sec for item in replay.samples),
            (10.2,),
        )
        self.assertEqual(
            replay.diagnostics["excluded_at_or_before_activation_count"],
            1,
        )
        self.assertEqual(
            replay.diagnostics["excluded_at_or_before_last_checked_count"],
            1,
        )
        self.assertEqual(replay.diagnostics["excluded_after_image_count"], 1)
        again = select_viewpoint_sampling_odom_replay(
            replay.samples,
            target_pose=target,
            target_activation_stamp_sec=10.0,
            last_checked_odom_stamp_sec=10.2,
            current_image_stamp_sec=10.4,
            expected_parent_frame="odom",
            expected_child_frame="base_footprint",
            strict_entry_tolerance_m=0.017,
        )
        self.assertEqual(again.samples, ())

    def test_buffered_arrival_odom_never_replays_at_or_after_current_pose(self):
        target = Pose2D(0.0, 0.0, 0.0)

        def sample(stamp_sec: float):
            return ViewpointSamplingOdomSample(
                stamp_sec=stamp_sec,
                pose=Pose2D(0.016, 0.0, 0.0),
                parent_frame="odom",
                child_frame="base_footprint",
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            )

        buffered = (sample(30.1), sample(30.2), sample(30.25))
        before_current = select_viewpoint_sampling_odom_replay(
            buffered,
            target_pose=target,
            target_activation_stamp_sec=30.0,
            last_checked_odom_stamp_sec=30.0,
            current_image_stamp_sec=30.3,
            current_pose_stamp_sec=30.2,
            expected_parent_frame="odom",
            expected_child_frame="base_footprint",
            strict_entry_tolerance_m=0.017,
        )
        self.assertEqual(
            tuple(item.stamp_sec for item in before_current.samples),
            (30.1,),
        )
        self.assertEqual(
            before_current.diagnostics[
                "excluded_at_or_after_current_pose_count"
            ],
            2,
        )

        after_current = select_viewpoint_sampling_odom_replay(
            buffered,
            target_pose=target,
            target_activation_stamp_sec=30.0,
            last_checked_odom_stamp_sec=30.2,
            current_image_stamp_sec=30.3,
            current_pose_stamp_sec=30.3,
            expected_parent_frame="odom",
            expected_child_frame="base_footprint",
            strict_entry_tolerance_m=0.017,
        )
        self.assertEqual(
            tuple(item.stamp_sec for item in after_current.samples),
            (30.25,),
        )
        consumed = select_viewpoint_sampling_odom_replay(
            buffered,
            target_pose=target,
            target_activation_stamp_sec=30.0,
            last_checked_odom_stamp_sec=30.3,
            current_image_stamp_sec=30.4,
            current_pose_stamp_sec=30.4,
            expected_parent_frame="odom",
            expected_child_frame="base_footprint",
            strict_entry_tolerance_m=0.017,
        )
        self.assertEqual(consumed.samples, ())

        image_before_current = select_viewpoint_sampling_odom_replay(
            (sample(40.1), sample(40.25), sample(40.3)),
            target_pose=target,
            target_activation_stamp_sec=40.0,
            last_checked_odom_stamp_sec=40.0,
            current_image_stamp_sec=40.2,
            current_pose_stamp_sec=40.3,
            expected_parent_frame="odom",
            expected_child_frame="base_footprint",
            strict_entry_tolerance_m=0.017,
        )
        self.assertEqual(
            tuple(item.stamp_sec for item in image_before_current.samples),
            (40.1,),
        )
        self.assertEqual(
            image_before_current.diagnostics[
                "history_replay_upper_bound_stamp_sec"
            ],
            40.2,
        )
        self.assertEqual(
            image_before_current.diagnostics["excluded_after_image_count"],
            2,
        )

    def test_arrival_history_identity_discards_stream_face_revision_or_pose(self):
        target = ViewpointSamplingMaterialTarget(
            pose=Pose2D(1.0, 2.0, 0.5),
            face_id="sampling_near",
            target_revision=4,
        )
        identity = viewpoint_sampling_history_identity("stream-a", target)
        self.assertEqual(identity["stream_id"], "stream-a")
        self.assertEqual(identity["material_target"], target.to_status_dict())
        self.assertEqual(len(identity["sha256"]), 64)
        self.assertFalse(
            viewpoint_sampling_history_identity_changed(
                previous_stream_id="stream-a",
                previous_target=target,
                current_stream_id="stream-a",
                current_target=target,
            )
        )

        changed_targets = (
            ViewpointSamplingMaterialTarget(
                pose=Pose2D(1.001, 2.0, 0.5),
                face_id=target.face_id,
                target_revision=target.target_revision,
            ),
            ViewpointSamplingMaterialTarget(
                pose=target.pose,
                face_id="sampling_far",
                target_revision=target.target_revision,
            ),
            ViewpointSamplingMaterialTarget(
                pose=target.pose,
                face_id=target.face_id,
                target_revision=5,
            ),
        )
        cases = (
            ("stream-b", target),
            *(("stream-a", changed) for changed in changed_targets),
        )
        for stream_id, current_target in cases:
            with self.subTest(
                stream_id=stream_id,
                current_target=current_target,
            ):
                self.assertTrue(
                    viewpoint_sampling_history_identity_changed(
                        previous_stream_id="stream-a",
                        previous_target=target,
                        current_stream_id=stream_id,
                        current_target=current_target,
                    )
                )
                changed_identity = viewpoint_sampling_history_identity(
                    stream_id,
                    current_target,
                )
                self.assertNotEqual(
                    changed_identity["sha256"],
                    identity["sha256"],
                )

    def test_observer_drops_old_history_for_every_material_identity_change(self):
        old_target = ViewpointSamplingMaterialTarget(
            pose=Pose2D(1.0, 2.0, 0.5),
            face_id="sampling_near",
            target_revision=4,
        )
        cases = (
            (
                "stream-b",
                old_target,
            ),
            (
                "stream-a",
                ViewpointSamplingMaterialTarget(
                    pose=Pose2D(1.001, 2.0, 0.5),
                    face_id=old_target.face_id,
                    target_revision=old_target.target_revision,
                ),
            ),
            (
                "stream-a",
                ViewpointSamplingMaterialTarget(
                    pose=old_target.pose,
                    face_id="sampling_far",
                    target_revision=old_target.target_revision,
                ),
            ),
            (
                "stream-a",
                ViewpointSamplingMaterialTarget(
                    pose=old_target.pose,
                    face_id=old_target.face_id,
                    target_revision=5,
                ),
            ),
        )

        for stream_id, current_target in cases:
            with self.subTest(
                stream_id=stream_id,
                current_target=current_target,
            ):
                latch = ViewpointSamplingLatch()
                latch.update(
                    robot_pose=Pose2D(1.2, 2.0, 0.5),
                    stationary=True,
                    axis_input_reason="oblique_silhouette",
                    candidate_pose=current_target.pose,
                    allow_start=True,
                    target_face_id=current_target.face_id,
                    target_revision=current_target.target_revision,
                )
                self.assertEqual(
                    latch.arrival_material_target,
                    current_target,
                )
                yaw = current_target.pose.yaw_rad
                odom = SimpleNamespace(
                    header=SimpleNamespace(frame_id="odom"),
                    child_frame_id="base_footprint",
                    pose=SimpleNamespace(
                        pose=SimpleNamespace(
                            position=SimpleNamespace(
                                x=current_target.pose.x_m,
                                y=current_target.pose.y_m,
                            ),
                            orientation=SimpleNamespace(
                                x=0.0,
                                y=0.0,
                                z=math.sin(yaw / 2.0),
                                w=math.cos(yaw / 2.0),
                            ),
                        )
                    ),
                )
                node = object.__new__(SimSynchronizedViewpointNode)
                node.args = SimpleNamespace(
                    stream_id=stream_id,
                    map_frame="odom",
                    base_frame="base_footprint",
                    sampling_arrival_tolerance_m=0.017,
                )
                node.sampling_latch = latch
                node.sampling_arrival_history_target = old_target
                node.sampling_arrival_history_stream_id = "stream-a"
                node.sampling_arrival_target_activation_stamp_sec = 50.0
                node.sampling_arrival_last_checked_odom_stamp_sec = 50.0
                node.sampling_arrival_history_diagnostics = {}
                node.odometry = deque((TimedSample(50.1, odom),), maxlen=40)

                self.assertTrue(
                    node._replay_sampling_arrival_history(
                        current_image_stamp_sec=50.2,
                        current_odom_stamp_sec=50.3,
                    )
                )
                self.assertEqual(
                    node.sampling_arrival_history_diagnostics["state"],
                    "target_history_discarded",
                )
                self.assertFalse(latch.arrival_status()["arrived"])
                self.assertIsNone(node.sampling_arrival_history_target)

    def test_buffered_arrival_odom_rejects_malformed_window_as_one_batch(self):
        target = Pose2D(0.0, 0.0, 0.0)
        valid = ViewpointSamplingOdomSample(
            stamp_sec=20.1,
            pose=Pose2D(0.016, 0.0, 0.0),
            parent_frame="odom",
            child_frame="base_footprint",
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
        cases = (
            (
                ViewpointSamplingOdomSample(
                    stamp_sec=20.2,
                    pose=valid.pose,
                    parent_frame="map",
                    child_frame="base_footprint",
                    quaternion_xyzw=valid.quaternion_xyzw,
                ),
                "odom_frame_mismatch",
            ),
            (
                ViewpointSamplingOdomSample(
                    stamp_sec=20.2,
                    pose=Pose2D(math.nan, 0.0, 0.0),
                    parent_frame="odom",
                    child_frame="base_footprint",
                    quaternion_xyzw=valid.quaternion_xyzw,
                ),
                "nonfinite_odom_pose",
            ),
            (
                ViewpointSamplingOdomSample(
                    stamp_sec=20.2,
                    pose=valid.pose,
                    parent_frame="odom",
                    child_frame="base_footprint",
                    quaternion_xyzw=(0.0, 0.0, 0.0, 0.5),
                ),
                "invalid_odom_quaternion",
            ),
            (
                ViewpointSamplingOdomSample(
                    stamp_sec=20.05,
                    pose=valid.pose,
                    parent_frame="odom",
                    child_frame="base_footprint",
                    quaternion_xyzw=valid.quaternion_xyzw,
                ),
                "nonmonotonic_odom_stamp",
            ),
        )
        for malformed, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                replay = select_viewpoint_sampling_odom_replay(
                    (valid, malformed),
                    target_pose=target,
                    target_activation_stamp_sec=20.0,
                    last_checked_odom_stamp_sec=20.0,
                    current_image_stamp_sec=20.3,
                    expected_parent_frame="odom",
                    expected_child_frame="base_footprint",
                    strict_entry_tolerance_m=0.017,
                )
                self.assertEqual(replay.samples, ())
                self.assertTrue(replay.diagnostics["fail_closed"])
                self.assertIn(
                    expected_reason,
                    replay.diagnostics["rejection_reasons"],
                )

    def test_repeat16_arrival_revises_to_explicit_recenter_and_resets_latch(self):
        target = Pose2D(
            -0.037311778590026795,
            0.19741015889821922,
            1.1303302378451376,
        )
        robot = Pose2D(
            -0.016786847655917692,
            0.19744789107551103,
            1.048374408778151,
        )
        recentered = sampling_recenter_pose(
            current_target_pose=target,
            robot_pose=robot,
            center_error_rad=math.radians(-12.134895),
            target_distance_m=0.33,
            max_correction_rad=math.radians(20.0),
        )
        self.assertIsNotNone(recentered)
        latch = ViewpointSamplingLatch()
        common = {
            "axis_input_reason": "oblique_silhouette",
            "candidate_pose": target,
            "allow_start": True,
            "target_revision": 16,
        }
        latch.update(
            robot_pose=Pose2D(0.55, 0.0, math.pi),
            stationary=True,
            **common,
        )
        armed = latch.update(
            robot_pose=target,
            stationary=False,
            **common,
        )
        self.assertTrue(armed.arrival_status["arrived"])

        revised = latch.update(
            robot_pose=robot,
            stationary=True,
            view_centered=False,
            view_settled=False,
            recenter_pose=recentered,
            **common,
        )

        self.assertTrue(revised.active)
        self.assertTrue(revised.advanced)
        self.assertEqual(
            revised.reason,
            "sampling_recentered_after_uncentered_arrival",
        )
        self.assertEqual(revised.target_pose, recentered)
        self.assertFalse(revised.arrival_status["arrived"])
        self.assertFalse(revised.arrival_status["strict_ever_armed"])
        self.assertEqual(
            revised.arrival_status["transition_reason"],
            "material_target_reset_unarmed",
        )
        self.assertIn(
            "pose",
            str(revised.arrival_status["reset_reason"]),
        )
        self.assertEqual(
            revised.arrival_status["material_target"]["target_revision"],
            16,
        )

    def test_offcenter_arrival_without_safe_recenter_never_uses_candidate(self):
        target = Pose2D(0.0, 0.0, 0.0)
        implicit_candidate = Pose2D(-0.20, 0.20, -0.8)
        latch = ViewpointSamplingLatch()
        common = {
            "axis_input_reason": "oblique_silhouette",
            "allow_start": True,
        }
        latch.update(
            robot_pose=Pose2D(0.55, 0.0, math.pi),
            stationary=True,
            candidate_pose=target,
            **common,
        )
        latch.update(
            robot_pose=target,
            stationary=False,
            candidate_pose=target,
            **common,
        )

        for recenter_pose in (
            None,
            target,
            Pose2D(math.nan, 0.0, 0.0),
        ):
            with self.subTest(recenter_pose=recenter_pose):
                unavailable = latch.update(
                    robot_pose=target,
                    stationary=True,
                    candidate_pose=implicit_candidate,
                    view_centered=False,
                    view_settled=False,
                    recenter_pose=recenter_pose,
                    **common,
                )
                self.assertFalse(unavailable.advanced)
                self.assertEqual(unavailable.target_pose, target)
                self.assertEqual(
                    unavailable.reason,
                    "sampling_recenter_unavailable",
                )
                self.assertTrue(unavailable.arrival_status["arrived"])

    def test_sampling_target_does_not_advance_outside_follower_entry_gate(self):
        latch = ViewpointSamplingLatch()
        first = Pose2D(0.30, 0.0, math.pi)
        second = Pose2D(0.28, 0.10, -2.80)
        latch.update(
            robot_pose=Pose2D(0.55, 0.0, math.pi),
            stationary=True,
            axis_input_reason="oblique_silhouette",
            candidate_pose=first,
            allow_start=True,
        )

        held = latch.update(
            robot_pose=Pose2D(0.31824, 0.0, math.pi),
            stationary=True,
            axis_input_reason="oblique_silhouette",
            candidate_pose=second,
            allow_start=True,
            view_centered=True,
            view_settled=True,
        )

        self.assertFalse(held.advanced)
        self.assertEqual(held.target_pose, first)

    def test_repeat15_arrival_requires_strict_entry_then_holds_both_envelopes(self):
        """20.36 mm is a valid hold pose only after strict 17 mm capture."""

        latch = ViewpointSamplingLatch()
        target = Pose2D(0.0, 0.0, 0.0)
        next_target = Pose2D(0.0, 0.10, -0.30)
        repeat15_pose = Pose2D(
            0.009330207878787886,
            0.018096320646435408,
            0.08,
        )
        started = latch.update(
            robot_pose=Pose2D(0.55, 0.0, math.pi),
            stationary=True,
            axis_input_reason="oblique_silhouette",
            candidate_pose=target,
            allow_start=True,
        )
        self.assertFalse(started.arrival_status["arrived"])
        self.assertEqual(
            started.arrival_status["transition_reason"],
            "target_initialized_unarmed",
        )

        unarmed = latch.update(
            robot_pose=repeat15_pose,
            stationary=True,
            axis_input_reason="oblique_silhouette",
            candidate_pose=next_target,
            allow_start=True,
        )
        self.assertFalse(unarmed.advanced)
        self.assertFalse(unarmed.arrival_status["arrived"])
        self.assertFalse(unarmed.arrival_status["strict_ever_armed"])
        self.assertTrue(unarmed.arrival_status["hold_valid"])
        self.assertFalse(
            unarmed.arrival_status["strict_entry_within_limit"]
        )
        self.assertAlmostEqual(
            unarmed.arrival_status["metrics"][
                "target_envelope_distance_m"
            ],
            0.02036,
        )
        self.assertTrue(
            unarmed.arrival_status["metrics"][
                "target_envelope_within_limit"
            ]
        )
        self.assertAlmostEqual(
            unarmed.arrival_status["metrics"][
                "inferred_stand_distance_m"
            ],
            0.32118,
        )
        self.assertTrue(
            unarmed.arrival_status["metrics"][
                "inferred_stand_distance_within_annulus"
            ]
        )

        armed = latch.update(
            robot_pose=Pose2D(0.016, 0.0, 0.08),
            stationary=False,
            axis_input_reason="oblique_silhouette",
            candidate_pose=next_target,
            allow_start=True,
        )
        self.assertTrue(armed.arrival_status["arrived"])
        self.assertTrue(armed.arrival_status["strict_ever_armed"])
        held = latch.update(
            robot_pose=repeat15_pose,
            stationary=True,
            axis_input_reason="oblique_silhouette",
            candidate_pose=target,
            allow_start=True,
        )
        self.assertFalse(held.advanced)
        self.assertTrue(held.arrival_status["arrived"])
        self.assertTrue(held.arrival_status["hold_valid"])
        self.assertFalse(held.arrival_status["strict_entry_within_limit"])

        advanced = latch.update(
            robot_pose=repeat15_pose,
            stationary=True,
            axis_input_reason="oblique_silhouette",
            candidate_pose=next_target,
            allow_start=True,
        )
        self.assertTrue(advanced.advanced)
        self.assertFalse(advanced.arrival_status["arrived"])
        self.assertFalse(advanced.arrival_status["strict_ever_armed"])
        self.assertIn(
            "pose",
            str(advanced.arrival_status["reset_reason"]),
        )
        json.dumps(advanced.arrival_status, allow_nan=False)

    def test_sampling_arrival_disarms_and_requires_strict_reentry(self):
        target = Pose2D(0.0, 0.0, 0.0)

        for robot_pose, failed_predicate, disarm_reason in (
            (
                Pose2D(0.0, 0.030001, 0.08),
                "target_envelope_within_limit",
                "target_envelope_exceeded",
            ),
            (
                Pose2D(-0.02002, 0.0, 0.08),
                "inferred_stand_distance_within_annulus",
                "inferred_stand_annulus_exceeded",
            ),
        ):
            with self.subTest(failed_predicate=failed_predicate):
                latch = ViewpointSamplingLatch()
                latch.update(
                    robot_pose=Pose2D(0.55, 0.0, math.pi),
                    stationary=True,
                    axis_input_reason="oblique_silhouette",
                    candidate_pose=target,
                    allow_start=True,
                )
                armed = latch.update(
                    robot_pose=Pose2D(0.016, 0.0, 0.08),
                    stationary=False,
                    axis_input_reason="oblique_silhouette",
                    candidate_pose=target,
                    allow_start=True,
                )
                self.assertTrue(armed.arrival_status["arrived"])

                disarmed = latch.update(
                    robot_pose=robot_pose,
                    stationary=True,
                    axis_input_reason="oblique_silhouette",
                    candidate_pose=target,
                    allow_start=True,
                )
                self.assertFalse(disarmed.arrival_status["arrived"])
                self.assertFalse(disarmed.arrival_status["hold_valid"])
                self.assertFalse(
                    disarmed.arrival_status["metrics"][failed_predicate]
                )
                self.assertEqual(
                    disarmed.arrival_status["disarm_reason"],
                    disarm_reason,
                )

                outside_strict = latch.update(
                    robot_pose=Pose2D(0.02036, 0.0, 0.08),
                    stationary=True,
                    axis_input_reason="oblique_silhouette",
                    candidate_pose=target,
                    allow_start=True,
                )
                self.assertFalse(outside_strict.arrival_status["arrived"])
                rearmed = latch.update(
                    robot_pose=Pose2D(0.016, 0.0, 0.08),
                    stationary=False,
                    axis_input_reason="oblique_silhouette",
                    candidate_pose=target,
                    allow_start=True,
                )
                self.assertTrue(rearmed.arrival_status["arrived"])

    def test_sampling_arrival_resets_on_face_revision_and_explicit_reset(self):
        latch = ViewpointSamplingLatch()
        target = Pose2D(0.0, 0.0, 0.0)
        common = {
            "axis_input_reason": "oblique_silhouette",
            "candidate_pose": target,
            "allow_start": True,
        }
        latch.update(
            robot_pose=Pose2D(0.55, 0.0, math.pi),
            stationary=True,
            target_face_id="sampling_near",
            target_revision=4,
            **common,
        )
        armed = latch.update(
            robot_pose=Pose2D(0.016, 0.0, 0.08),
            stationary=False,
            target_face_id="sampling_near",
            target_revision=4,
            **common,
        )
        self.assertTrue(armed.arrival_status["arrived"])

        face_changed = latch.update(
            robot_pose=Pose2D(0.016, 0.0, 0.08),
            stationary=False,
            target_face_id="sampling_far",
            target_revision=4,
            **common,
        )
        self.assertFalse(face_changed.arrival_status["arrived"])
        self.assertIn(
            "face",
            str(face_changed.arrival_status["reset_reason"]),
        )
        revision_changed = latch.update(
            robot_pose=Pose2D(0.016, 0.0, 0.08),
            stationary=False,
            target_face_id="sampling_far",
            target_revision=5,
            **common,
        )
        self.assertFalse(revision_changed.arrival_status["arrived"])
        self.assertIn(
            "revision",
            str(revision_changed.arrival_status["reset_reason"]),
        )

        latch.reset(reason="test_reset")
        reset_status = latch.arrival_status()
        self.assertFalse(reset_status["arrived"])
        self.assertEqual(reset_status["reset_reason"], "test_reset")
        json.dumps(reset_status, allow_nan=False)

    def test_sampling_arrival_rejects_nonfinite_geometry(self):
        latch = ViewpointSamplingLatch()
        finite = Pose2D(0.0, 0.0, 0.0)
        latch.update(
            robot_pose=Pose2D(0.55, 0.0, math.pi),
            stationary=True,
            axis_input_reason="oblique_silhouette",
            candidate_pose=finite,
            allow_start=True,
        )

        for changes, expected in (
            ({"robot_pose": Pose2D(math.nan, 0.0, 0.0)}, "robot"),
            (
                {"candidate_pose": Pose2D(0.0, math.inf, 0.0)},
                "candidate",
            ),
            (
                {
                    "equivalent_target_poses": (
                        Pose2D(0.0, 0.0, math.nan),
                    )
                },
                r"equivalent_target_poses\[0\]",
            ),
        ):
            kwargs = {
                "robot_pose": finite,
                "stationary": True,
                "axis_input_reason": "oblique_silhouette",
                "candidate_pose": finite,
                "allow_start": True,
                **changes,
            }
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    latch.update(**kwargs)

    def test_sampling_target_does_not_advance_before_centered_view_settles(self):
        latch = ViewpointSamplingLatch()
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

    def test_qr_binding_skips_provisional_faces_then_latches_physical_face(self):
        latch = QrFaceLatch()
        resolver = StableFaceResolver()

        # A valid QR consensus can coexist with a still-provisional sampling
        # target.  Those sampling IDs are intentionally unresolved, so no
        # QR-to-face binding observation may be attempted yet.
        self.assertFalse(
            should_attempt_qr_face_binding(
                acquiring_axis=True,
                consensus_available=True,
                qr_detected=True,
            )
        )
        provisional = latch.update(stream_id="run", observation=None)
        self.assertEqual(provisional.reason, "no_observation")
        self.assertIsNone(latch.latched_evidence)

        # The first committed-axis frame bootstraps stable physical IDs and is
        # deliberately deferred.  The next matching frame resolves them and
        # is the first frame allowed to bind the QR identity.
        first = resolver.update(
            stream_id="run", outward_normals_rad=(0.0, math.pi)
        )
        second = resolver.update(
            stream_id="run", outward_normals_rad=(0.0, math.pi)
        )
        self.assertFalse(first.identity_resolved)
        self.assertTrue(second.identity_resolved)
        self.assertTrue(
            should_defer_initial_physical_recommendation(
                acquiring_axis=False,
                face_identity_resolved=first.identity_resolved,
                hard_qr_latched=False,
            )
        )
        self.assertTrue(
            should_attempt_qr_face_binding(
                acquiring_axis=False,
                consensus_available=True,
                qr_detected=True,
            )
        )
        physical = latch.update(
            stream_id="run",
            observation=QrBindingObservation(
                face_id=second.faces[0].face_id,
                confidence=1.0,
                provenance="sim_qr:C:lidar_head_roi",
                registry_match=True,
                inside_target_roi=True,
                distinct_fresh_frame_consensus=True,
                visibility_margin_rad=math.radians(20.0),
                identity_resolved=second.identity_resolved,
            ),
            known_face_ids={face.face_id for face in second.faces},
        )
        self.assertEqual(physical.reason, "hard_binding_accepted")
        self.assertTrue(physical.evidence.hard)
        self.assertTrue(physical.evidence.valid)

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
        self.assertEqual(args.tangential_correction_gain, 0.5)
        self.assertAlmostEqual(args.target_distance_m, 0.33)
        self.assertAlmostEqual(args.axis_acquisition_distance_m, 0.55)
        self.assertAlmostEqual(args.axis_acquisition_arrival_tolerance_m, 0.10)
        self.assertAlmostEqual(args.axis_acquisition_search_step_deg, 45.0)
        self.assertEqual(args.axis_acquisition_search_max_targets, 7)
        self.assertAlmostEqual(args.sampling_arrival_tolerance_m, 0.017)
        self.assertAlmostEqual(
            args.viewpoint_sampling_terminal_heading_hold_tolerance_m,
            0.020,
        )
        self.assertAlmostEqual(
            args
            .viewpoint_sampling_terminal_heading_target_envelope_radius_m,
            0.030,
        )

    def test_synchronized_observer_rejects_divergent_hold_contract_before_ros(self):
        required = [
            "--stand-x", "0",
            "--stand-y", "0",
            "--status-json", "status.json",
            "--recommended-pose-json", "recommendation.json",
            "--observation-json", "observation.json",
        ]
        for options in (
            [
                "--sampling-arrival-tolerance-m", "0.019",
                "--viewpoint-sampling-terminal-heading-hold-tolerance-m",
                "0.019",
            ],
            [
                "--viewpoint-sampling-terminal-heading-target-envelope-radius-m",
                "0.030001",
            ],
        ):
            with self.subTest(options=options):
                with self.assertRaisesRegex(
                    SystemExit,
                    "invalid viewpoint-sampling arrival contract",
                ):
                    synchronized_viewpoint_main(required + options)

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
