import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json,
    write_content_hashed_json,
)
from scripts.aufgabe04.navigation.planning.map_io import freeze_map_bundle
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import (
    load_recommendation,
    recommendation_to_dict,
)
from scripts.aufgabe04.perception.stand_axis.real_camera_profile import (
    RealCameraStandAxisProfile,
)
from scripts.aufgabe04.perception.stand_axis.model_profile import (
    stand_model_from_payload,
    write_stand_model,
)
from scripts.aufgabe04.real_robot.configuration.geometry import (
    CameraIntrinsics,
    project_optical_point,
    roi_from_projection,
    rotate_vector,
    transform_point,
)
from scripts.aufgabe04.real_robot.configuration.profile import (
    CAMERA_CALIBRATION_PROFILE_SCHEMA_VERSION,
    REAL_HARDWARE_PROFILE_SCHEMA_VERSION,
    CameraCalibrationProfile,
    RealRobotProfile,
    RigidTransform,
    camera_calibration_sha256,
    camera_info_mismatches,
    load_camera_calibration,
    load_real_robot_profile,
    write_camera_calibration,
    write_real_robot_profile,
)
from scripts.aufgabe04.real_robot.observer.contract import (
    PASSIVE_VIEWPOINT_OBSERVER_VERSION,
)
from scripts.aufgabe04.real_robot.observer.head_roi_reacquisition import (
    BACKSIDE_REACQUISITION_TARGET_CROP_HALF_WIDTH_RATIO,
    DEFAULT_BACKSIDE_REACQUISITION_PADDING_SCALE,
    DEFAULT_BACKSIDE_REGISTRATION_MAX_CENTER_OFFSET_RATIO,
    target_centered_head_roi_attempts,
)
from scripts.aufgabe04.real_robot.observer import node as observer_node
from scripts.aufgabe04.real_robot.observer.node import (
    PassiveRealViewpointNode,
    _StampedMessage,
    _consensus_for_current_axis_source,
    _head_scale_gate,
    _nearest,
    _pose_is_stationary,
    _rectify_bgr_frame,
    _stand_axis_profile_from_args,
    _validate_args,
    build_parser,
)
from scripts.aufgabe04.real_robot.passive_survey.prepare import (
    main as prepare_passive_survey,
)
from scripts.aufgabe04.real_robot.passive_survey.finalize import (
    _validate_metric_model_binding,
)
from scripts.aufgabe04.real_robot.configuration.recommendation import (
    REAL_VIEWPOINT_SOURCE,
    build_real_viewpoint_recommendation,
)
from scripts.aufgabe04.real_robot.execution.run_unloaded_segment import (
    build_execution_command,
    build_runner_command,
    validate_profile_artifact_bindings,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateGeometry,
    CandidateSource,
    FrozenCandidate,
    new_candidate_snapshot,
    write_candidate_snapshot,
)
from scripts.aufgabe04.stations.station_identity_registry import (
    StationIdentity,
    new_station_identity_registry,
    write_station_identity_registry,
)


def calibration() -> CameraCalibrationProfile:
    return CameraCalibrationProfile(
        schema_version=CAMERA_CALIBRATION_PROFILE_SCHEMA_VERSION,
        calibration_id="camera_real_001",
        camera_optical_frame="camera_rgb_optical_frame",
        base_frame="base_footprint",
        width_px=640,
        height_px=480,
        distortion_model="plumb_bob",
        distortion_coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),
        camera_matrix=(400.0, 0.0, 320.0, 0.0, 400.0, 240.0, 0.0, 0.0, 1.0),
        rectification_matrix=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        projection_matrix=(
            400.0,
            0.0,
            320.0,
            0.0,
            0.0,
            400.0,
            240.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ),
        base_to_camera=RigidTransform(
            (0.08, 0.0, 0.10),
            (0.5, -0.5, 0.5, -0.5),
        ),
        measured_unix_sec=100.0,
        source="checkerboard_and_measured_tf_001",
    )


def robot_profile(calibration_sha256: str, physical_site_sha256: str) -> RealRobotProfile:
    return RealRobotProfile(
        schema_version=REAL_HARDWARE_PROFILE_SCHEMA_VERSION,
        profile_id="tb3_real_001",
        robot_id="robot_1",
        namespace="robot1",
        scan_topic="scan",
        odom_topic="odom",
        cmd_vel_topic="cmd_vel",
        amcl_topic="amcl_pose",
        compressed_image_topic="camera/image_raw/compressed",
        camera_info_topic="camera/camera_info",
        map_frame="map",
        odom_frame="odom",
        base_frame="base_footprint",
        scan_frame="base_scan",
        camera_optical_frame="camera_rgb_optical_frame",
        localization_source="amcl",
        physical_site_id="arena_real",
        physical_site_sha256=physical_site_sha256,
        calibration_profile_sha256=calibration_sha256,
        robot_radius_m=0.105,
        scan_origin_to_base_offset_m=0.032,
        max_linear_speed_mps=0.05,
        max_angular_speed_radps=0.18,
    )


def write_test_stand_model(
    path: Path,
    *,
    environment: str = "physical",
    measurement_status: str = "measured",
) -> str:
    return write_stand_model(
        path,
        stand_model_from_payload(
            {
                "schema_version": 2,
                "profile_id": f"{environment}_test_v2",
                "environment": environment,
                "measurement_status": measurement_status,
                "head_width_m": 0.078,
                "head_height_m": 0.078,
                "head_depth_m": 0.006,
                "qr_symbol_width_m": 0.062,
                "qr_symbol_height_m": 0.062,
                "qr_panel_width_m": 0.071,
                "qr_panel_height_m": 0.071,
                "qr_center_x_m": 0.0,
                "qr_center_y_m": 0.0,
                "head_top_height_m": 0.210,
                "base_width_m": 0.153,
                "base_depth_m": 0.153,
                "tolerance_m": 0.002,
                "source": "direct test metrology",
            }
        ),
    )


class RealHardwareProfileTest(unittest.TestCase):
    def test_profiles_round_trip_and_resolve_namespaced_topics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration_path = root / "calibration.json"
            calibration_digest = write_camera_calibration(
                calibration_path,
                calibration(),
            )
            profile_path = root / "robot.json"
            write_real_robot_profile(
                profile_path,
                robot_profile(calibration_digest, "a" * 64),
            )

            loaded_calibration = load_camera_calibration(calibration_path)
            loaded_profile = load_real_robot_profile(profile_path)

        self.assertEqual(loaded_calibration, calibration())
        self.assertEqual(loaded_profile.resolved_runtime().scan_topic, "/robot1/scan")
        self.assertEqual(
            loaded_profile.resolved_compressed_image_topic,
            "/robot1/camera/image_raw/compressed",
        )
        self.assertFalse(loaded_profile.runtime_config().use_sim_time)

    def test_real_profile_rejects_collapsed_map_and_odom_frames(self):
        profile = robot_profile("b" * 64, "a" * 64)
        invalid = RealRobotProfile(
            **{**profile.__dict__, "map_frame": "odom"}
        )
        with self.assertRaisesRegex(ValueError, "must differ"):
            write_real_robot_profile(Path("/tmp/not_written.json"), invalid)

    def test_live_camera_info_comparison_is_fail_closed(self):
        info = SimpleNamespace(
            width=640,
            height=480,
            distortion_model="plumb_bob",
            d=[0.0] * 5,
            k=list(calibration().camera_matrix),
            r=list(calibration().rectification_matrix),
            p=list(calibration().projection_matrix),
            header=SimpleNamespace(frame_id="camera_rgb_optical_frame"),
        )
        self.assertEqual(camera_info_mismatches(calibration(), info), ())
        info.width = 320
        self.assertEqual(camera_info_mismatches(calibration(), info), ("width_px",))


class RealCameraGeometryTest(unittest.TestCase):
    def test_rep103_optical_projection_and_roi(self):
        intrinsics = CameraIntrinsics(640, 480, 400.0, 400.0, 320.0, 240.0)
        projection = project_optical_point(
            (0.05, -0.02, 0.5),
            intrinsics,
            physical_size_m=0.08,
        )
        self.assertTrue(projection.inside_image)
        self.assertAlmostEqual(projection.u_px, 360.0)
        self.assertAlmostEqual(projection.v_px, 224.0)
        self.assertAlmostEqual(projection.expected_size_px, 64.0)
        roi = roi_from_projection(projection, intrinsics)
        self.assertIsNotNone(roi)
        self.assertLess(roi.x0, projection.u_px)
        self.assertGreater(roi.x1, projection.u_px)

    def test_transform_helpers_use_full_quaternion(self):
        half = math.sqrt(0.5)
        rotated = rotate_vector((1.0, 0.0, 0.0), (0.0, 0.0, half, half))
        self.assertAlmostEqual(rotated[0], 0.0, places=7)
        self.assertAlmostEqual(rotated[1], 1.0, places=7)
        transformed = transform_point(
            (1.0, 0.0, 0.0),
            translation_xyz=(2.0, 3.0, 0.0),
            rotation_xyzw=(0.0, 0.0, half, half),
        )
        self.assertAlmostEqual(transformed[0], 2.0)
        self.assertAlmostEqual(transformed[1], 4.0)

    def test_backside_reacquisition_roi_stays_target_centered(self):
        intrinsics = CameraIntrinsics(640, 480, 400.0, 400.0, 320.0, 240.0)
        projection = project_optical_point(
            (0.0, 0.0, 0.5),
            intrinsics,
            physical_size_m=0.08,
        )

        attempts = target_centered_head_roi_attempts(
            projection,
            intrinsics,
            expected_head_height_px=64.0,
            nominal_padding_scale=1.8,
            backside_reacquisition_padding_scale=(
                DEFAULT_BACKSIDE_REACQUISITION_PADDING_SCALE
            ),
        )

        self.assertEqual(
            [attempt.source for attempt in attempts],
            [
                "nominal_projection",
                "target_centered_backside_reacquisition",
            ],
        )
        nominal, expanded = attempts
        self.assertEqual(nominal.expected_center_u_px, projection.u_px)
        self.assertEqual(expanded.expected_center_u_px, projection.u_px)
        self.assertEqual(nominal.expected_center_v_px, projection.v_px)
        self.assertEqual(expanded.expected_center_v_px, projection.v_px)
        self.assertLess(expanded.roi.x0, nominal.roi.x0)
        self.assertGreater(expanded.roi.x1, nominal.roi.x1)
        self.assertLess(expanded.roi.y0, nominal.roi.y0)
        self.assertGreater(expanded.roi.y1, nominal.roi.y1)
        self.assertEqual(
            expanded.backside_target_crop_half_width_ratio,
            BACKSIDE_REACQUISITION_TARGET_CROP_HALF_WIDTH_RATIO,
        )

    def test_backside_reacquisition_can_be_disabled(self):
        intrinsics = CameraIntrinsics(640, 480, 400.0, 400.0, 320.0, 240.0)
        projection = project_optical_point(
            (0.0, 0.0, 0.5),
            intrinsics,
            physical_size_m=0.08,
        )

        attempts = target_centered_head_roi_attempts(
            projection,
            intrinsics,
            expected_head_height_px=64.0,
            nominal_padding_scale=1.8,
            backside_reacquisition_padding_scale=(
                DEFAULT_BACKSIDE_REACQUISITION_PADDING_SCALE
            ),
            enable_backside_reacquisition=False,
        )

        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0].source, "nominal_projection")

    def test_reacquisition_policy_survives_roi_boundary_clipping(self):
        intrinsics = CameraIntrinsics(160, 120, 400.0, 400.0, 80.0, 60.0)
        projection = project_optical_point(
            (0.0, 0.0, 0.25),
            intrinsics,
            physical_size_m=0.08,
        )

        attempts = target_centered_head_roi_attempts(
            projection,
            intrinsics,
            expected_head_height_px=128.0,
            nominal_padding_scale=4.5,
        )

        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0].roi, attempts[1].roi)
        self.assertNotEqual(
            attempts[0].backside_target_crop_half_width_ratio,
            attempts[1].backside_target_crop_half_width_ratio,
        )

class RealCameraStandAxisProfileTest(unittest.TestCase):
    def test_default_profile_resolves_model_refinement_edge_height(self):
        profile = RealCameraStandAxisProfile()
        small = profile.resolve(16.0)
        medium = profile.resolve(100.0)
        large = profile.resolve(300.0)

        self.assertEqual(profile.edge_preprocess, "channel_union")
        self.assertEqual(small.min_edge_height_px, 5.0)
        self.assertEqual(medium.min_edge_height_px, 14.0)
        self.assertEqual(large.min_edge_height_px, 14.0)

    def test_profile_rejects_invalid_modes_thresholds_and_head_sizes(self):
        with self.assertRaisesRegex(ValueError, "edge_preprocess"):
            RealCameraStandAxisProfile(edge_preprocess="outer_border")
        with self.assertRaisesRegex(ValueError, "Canny"):
            RealCameraStandAxisProfile(canny_low=60, canny_high=20)
        with self.assertRaisesRegex(ValueError, "Canny"):
            RealCameraStandAxisProfile(canny_low=-1, canny_high=20)
        profile = RealCameraStandAxisProfile()
        for invalid in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "expected_head_size_px"):
                    profile.resolve(invalid)

    def test_head_scale_gate_rejects_recorded_oversized_head(self):
        accepted = _head_scale_gate(
            expected_size_px=103.0,
            left_height_px=101.0,
            right_height_px=105.0,
        )
        rejected = _head_scale_gate(
            expected_size_px=103.0,
            left_height_px=151.0,
            right_height_px=152.0,
        )
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["reason"], "ok")
        self.assertFalse(rejected["accepted"])
        self.assertEqual(
            rejected["reason"],
            "head_size_projection_mismatch",
        )


class PassiveObservationCoreTest(unittest.TestCase):
    @staticmethod
    def parser_args(model_path: Path) -> list[str]:
        return [
            "--robot-profile",
            "robot.json",
            "--camera-calibration",
            "camera.json",
            "--stand-model-profile",
            str(model_path),
            "--stream-id",
            "stream",
            "--stand-id",
            "A",
            "--expected-qr-id",
            "A",
            "--stand-x",
            "0.1",
            "--stand-y",
            "0.2",
            "--status-json",
            "status.json",
            "--recommended-pose-json",
            "recommendation.json",
        ]

    def test_consensus_must_match_the_current_accepted_axis_source(self):
        nominal = SimpleNamespace(source="model_backside_current_frame")
        matching = SimpleNamespace(
            axis_consensus=nominal,
            axis_sample_accepted=True,
        )
        stale_other_source = SimpleNamespace(
            axis_consensus=nominal,
            axis_sample_accepted=True,
        )
        duplicate = SimpleNamespace(
            axis_consensus=nominal,
            axis_sample_accepted=False,
        )

        self.assertIs(
            _consensus_for_current_axis_source(
                matching,
                "model_backside_current_frame",
            ),
            nominal,
        )
        self.assertIsNone(
            _consensus_for_current_axis_source(
                stale_other_source,
                "model_backside_current_frame_bounded_camera_lidar_registered",
            )
        )
        self.assertIsNone(
            _consensus_for_current_axis_source(
                duplicate,
                "model_backside_current_frame",
            )
        )

    def test_parser_defaults_to_channel_union_and_wires_valid_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "measured.json"
            write_test_stand_model(model_path)
            parser = build_parser()
            defaults = parser.parse_args(self.parser_args(model_path))
            _validate_args(parser, defaults)
            default_profile = _stand_axis_profile_from_args(defaults)
            self.assertEqual(defaults.edge_preprocess, "channel-union")
            self.assertEqual(default_profile.edge_preprocess, "channel_union")
            self.assertEqual(defaults.tf_timeout_sec, 0.15)
            self.assertEqual(defaults.tf_retry_rate_hz, 50.0)
            self.assertAlmostEqual(defaults.stand_head_center_height_m, 0.171)
            self.assertEqual(
                defaults.backside_reacquisition_padding_scale,
                DEFAULT_BACKSIDE_REACQUISITION_PADDING_SCALE,
            )
            self.assertEqual(
                defaults.backside_registration_max_center_offset_ratio,
                DEFAULT_BACKSIDE_REGISTRATION_MAX_CENTER_OFFSET_RATIO,
            )
            self.assertAlmostEqual(
                defaults.backside_registration_max_bearing_delta_deg,
                12.0,
            )

            override = parser.parse_args(
                [
                    *self.parser_args(model_path),
                    "--edge-preprocess", "gray",
                    "--canny-low", "12",
                    "--canny-high", "44",
                    "--stand-head-center-height-m", "0.172",
                ]
            )
            _validate_args(parser, override)
            override_profile = _stand_axis_profile_from_args(override)
            self.assertEqual(override_profile.edge_preprocess, "gray")
            self.assertEqual(override_profile.canny_low, 12)
            self.assertEqual(override_profile.canny_high, 44)
            self.assertAlmostEqual(override.stand_head_center_height_m, 0.171)

            stale_height = parser.parse_args(
                [
                    *self.parser_args(model_path),
                    "--stand-head-center-height-m", "0.165",
                ]
            )
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                _validate_args(parser, stale_height)

            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        *self.parser_args(model_path),
                        "--edge-preprocess",
                        "outer-border",
                    ]
                )
            invalid_canny = parser.parse_args(
                [
                    *self.parser_args(model_path),
                    "--canny-low", "60",
                    "--canny-high", "20",
                ]
            )
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                _validate_args(parser, invalid_canny)

            for flag, value in (
                ("--backside-reacquisition-padding-scale", "4.51"),
                (
                    "--backside-registration-max-center-offset-ratio",
                    "1.51",
                ),
                (
                    "--backside-registration-max-bearing-delta-deg",
                    "12.01",
                ),
            ):
                with self.subTest(flag=flag):
                    unsafe = parser.parse_args(
                        [*self.parser_args(model_path), flag, value]
                    )
                    with (
                        redirect_stderr(StringIO()),
                        self.assertRaises(SystemExit),
                    ):
                        _validate_args(parser, unsafe)

    def test_parser_rejects_missing_stand_model_before_runtime(self):
        args = self.parser_args(Path("unused-model.json"))
        without_model = args[:4] + args[6:]
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(without_model)

    def test_real_robot_package_has_no_legacy_detector_dependency(self):
        real_robot_root = Path(observer_node.__file__).parents[1]
        forbidden = (
            "estimate_stand_axis_from_edges",
            "estimate_stand_axis_from_mask",
            "stand_axis_image",
            "silhouette_only",
        )
        offenders = []
        for path in sorted(real_robot_root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            offenders.extend(
                (str(path.relative_to(real_robot_root)), token)
                for token in forbidden
                if token in source
            )

        observer_source = Path(observer_node.__file__).read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "estimate_stand_axis_from_metric_model",
            observer_source,
        )
        self.assertEqual(offenders, [])

    def test_operational_observer_wires_only_strict_backside_handoff(self):
        observer_source = Path(observer_node.__file__).read_text(
            encoding="utf-8"
        )

        for expected in (
            "build_backside_axis_observation(",
            "estimate_visible_face=getattr(",
            "visible_face_confidence=getattr(",
            "consensus_source=consensus.source",
            "debug_qr_detected=debug.qr_detected",
            "snapshot.current_qr_sample_count",
            "snapshot.tentative_qr_id",
            "snapshot.latched_qr_id",
            "self._qr_marker_seen_in_stationary_epoch",
            "self._qr_marker_stationary_epoch_anchor",
            "self._reset_qr_marker_epoch()",
            "target_centered_head_roi_attempts(",
            "backside_reacquisition_padding_scale",
            "select_camera_target_measurement(",
            "build_backside_target_registration_evidence(",
            "associate_camera_registered_candidate_lidar_target(",
            "registered_lidar_association.search_association",
            "lidar_target_associated = registered_lidar_association.associated",
            '"camera_registered_candidate_lidar_association"',
            "axis_source=axis_sample_source",
            "consensus.source != axis_sample_source",
            "target_registration=target_registration",
            "head_roi_attempts",
            "attempt.expected_center_u_px - attempt_roi.x0",
            "attempt.expected_center_v_px - attempt_roi.y0",
            "expected_head_height_px=attempt.expected_head_height_px",
            "qr_texts and estimate.source == BACKSIDE_AXIS_SAMPLE_SOURCE",
            "decoded QR text conflicts",
            '"axis_observation_not_committable"',
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, observer_source)

        self.assertNotIn(
            '"observation_kind": "real_stand_axis_without_qr"',
            observer_source,
        )
        self.assertNotIn(
            '"axis_committed_qr_unresolved"',
            observer_source,
        )

    def test_passive_observer_requires_measured_model_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            measured_path = root / "measured.json"
            write_test_stand_model(measured_path)
            measured = build_parser().parse_args(
                self.parser_args(measured_path)
            )
            _validate_args(build_parser(), measured)

            provisional_path = root / "provisional.json"
            write_test_stand_model(
                provisional_path,
                measurement_status="provisional",
            )
            provisional = build_parser().parse_args(
                self.parser_args(provisional_path)
            )
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                _validate_args(build_parser(), provisional)

            simulation_path = root / "simulation.json"
            write_test_stand_model(simulation_path, environment="simulation")
            simulation = build_parser().parse_args(
                self.parser_args(simulation_path)
            )
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                _validate_args(build_parser(), simulation)

    def test_raw_image_is_rectified_into_camera_info_projection_geometry(self):
        import cv2
        import numpy

        info = SimpleNamespace(
            width=640,
            height=480,
            d=[0.0] * 5,
            k=list(calibration().camera_matrix),
            r=list(calibration().rectification_matrix),
            p=list(calibration().projection_matrix),
        )
        frame = numpy.zeros((480, 640, 3), dtype=numpy.uint8)
        rectified = _rectify_bgr_frame(frame, info, cv2, numpy)
        self.assertEqual(rectified.shape, frame.shape)

        info.width = 320
        with self.assertRaisesRegex(ValueError, "dimensions"):
            _rectify_bgr_frame(frame, info, cv2, numpy)

    def test_synchronization_and_stationarity_helpers(self):
        samples = (
            _StampedMessage(9.8, "old"),
            _StampedMessage(10.03, "near"),
        )
        self.assertEqual(
            _nearest(samples, stamp_sec=10.0, tolerance_sec=0.05).value,
            "near",
        )
        self.assertIsNone(_nearest(samples, stamp_sec=10.2, tolerance_sec=0.05))
        self.assertTrue(
            _pose_is_stationary(
                Pose2D(0.0, 0.0, 0.0),
                Pose2D(0.005, 0.0, math.radians(1.0)),
                max_translation_m=0.01,
                max_rotation_rad=math.radians(2.0),
            )
        )
        self.assertFalse(
            _pose_is_stationary(
                Pose2D(0.0, 0.0, 0.0),
                Pose2D(0.02, 0.0, 0.0),
                max_translation_m=0.01,
                max_rotation_rad=math.radians(2.0),
            )
        )

    def test_debug_writer_persists_each_available_estimator_artifact(self):
        class _FakeCv2:
            @staticmethod
            def imwrite(path, _image):
                Path(path).write_bytes(b"debug-image")
                return True

        with tempfile.TemporaryDirectory() as tmp:
            debug_dir = Path(tmp)
            adapter = PassiveRealViewpointNode.__new__(
                PassiveRealViewpointNode
            )
            adapter.args = SimpleNamespace(debug_dir=debug_dir)
            adapter.cv2 = _FakeCv2()
            mono = object()
            overlay = object()
            debug = SimpleNamespace(
                edges=mono,
                raw_edges=mono,
                face_mask=mono,
                rectangle_mask=mono,
                rectangle_overlay=overlay,
            )
            metadata = {
                "mode": "metric_model_only",
                "profile": {
                    "edge_preprocess": "channel_union",
                    "canny_low": 20,
                    "canny_high": 60,
                },
                "estimator_usable": True,
                "estimator_reason": "ok",
                "estimator_source": "metric_stand_model",
            }

            adapter._write_debug(
                overlay,
                overlay,
                debug,
                metadata=metadata,
            )
            written_metadata = json.loads(
                (debug_dir / "latest_metadata.json").read_text()
            )

            expected_images = {
                "latest_frame.png",
                "latest_head_roi.png",
                "latest_edges.png",
                "latest_raw_edges.png",
                "latest_side_evidence.png",
                "latest_rectangle_mask.png",
                "latest_rectangle_overlay.png",
            }
            self.assertEqual(
                set(written_metadata["artifacts"]),
                expected_images,
            )
            self.assertTrue(
                all((debug_dir / filename).is_file() for filename in expected_images)
            )
            self.assertEqual(
                written_metadata["stand_axis"]["profile"]["edge_preprocess"],
                "channel_union",
            )
            self.assertEqual(
                written_metadata["stand_axis"]["mode"],
                "metric_model_only",
            )

            adapter._write_debug(
                overlay,
                overlay,
                SimpleNamespace(
                    edges=mono,
                    raw_edges=None,
                    face_mask=None,
                    rectangle_mask=None,
                    rectangle_overlay=None,
                ),
                metadata=metadata,
            )
            for stale_filename in (
                "latest_raw_edges.png",
                "latest_side_evidence.png",
                "latest_rectangle_mask.png",
                "latest_rectangle_overlay.png",
            ):
                self.assertFalse((debug_dir / stale_filename).exists())
            refreshed_metadata = json.loads(
                (debug_dir / "latest_metadata.json").read_text()
            )
            self.assertEqual(
                set(refreshed_metadata["artifacts"]),
                {
                    "latest_frame.png",
                    "latest_head_roi.png",
                    "latest_edges.png",
                },
            )

    def test_real_recommendation_requires_bound_qr_and_round_trips(self):
        recommendation = build_real_viewpoint_recommendation(
            stream_id="real_session_candidate_1",
            stand_id="A",
            planning_frame="map",
            stand_center=Pose2D(0.0, 0.0),
            stand_radius_m=0.06,
            stand_uncertainty_m=0.02,
            robot_pose=Pose2D(0.5, 0.0, math.pi),
            stand_axis_rad=math.pi / 2.0,
            axis_confidence=0.9,
            axis_sample_count=7,
            sensor_stamp_sec=12.0,
            expected_qr_id="A",
            observed_qr_ids=("A",),
            target_distance_m=0.33,
            observation_unix_sec=100.0,
        )
        self.assertFalse(recommendation.simulation_only)
        self.assertEqual(recommendation.material_target.face_id, "qr_face")
        loaded = load_recommendation(
            recommendation_to_dict(recommendation),
            expected_source=REAL_VIEWPOINT_SOURCE,
            expected_simulation_only=False,
        )
        self.assertEqual(loaded, recommendation)
        with self.assertRaisesRegex(ValueError, "absent"):
            build_real_viewpoint_recommendation(
                **{
                    **recommendation_input(),
                    "observed_qr_ids": (),
                }
            )


def recommendation_input():
    return {
        "stream_id": "real_session_candidate_1",
        "stand_id": "A",
        "planning_frame": "map",
        "stand_center": Pose2D(0.0, 0.0),
        "stand_radius_m": 0.06,
        "stand_uncertainty_m": 0.02,
        "robot_pose": Pose2D(0.5, 0.0, math.pi),
        "stand_axis_rad": math.pi / 2.0,
        "axis_confidence": 0.9,
        "axis_sample_count": 7,
        "sensor_stamp_sec": 12.0,
        "expected_qr_id": "A",
        "observed_qr_ids": ("A",),
        "target_distance_m": 0.33,
        "observation_unix_sec": 100.0,
    }


class PreparePassiveSurveyTest(unittest.TestCase):
    def test_finalization_rejects_legacy_or_unbound_survey_config(self):
        digest = "a" * 64
        config = {
            "observer_version": PASSIVE_VIEWPOINT_OBSERVER_VERSION,
            "stand_model_profile_sha256": digest,
            "stand_model_environment": "physical",
            "stand_model_measurement_status": "measured",
        }
        binding = {"stand_model_profile_sha256": digest}
        self.assertEqual(
            _validate_metric_model_binding(config, binding),
            digest,
        )

        with self.assertRaisesRegex(ValueError, "metric-model-only"):
            _validate_metric_model_binding(
                {**config, "observer_version": "legacy-observer"},
                binding,
            )
        with self.assertRaisesRegex(ValueError, "differs"):
            _validate_metric_model_binding(
                config,
                {"stand_model_profile_sha256": "b" * 64},
            )

    def test_prepared_commands_are_real_time_and_observe_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "map.pgm"
            image.write_text("P2\n4 4\n255\n0 0 0 0\n0 255 255 0\n0 255 255 0\n0 0 0 0\n")
            map_yaml = root / "map.yaml"
            map_yaml.write_text(
                "image: map.pgm\nresolution: 0.05\norigin: [0, 0, 0]\n"
                "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n"
            )
            map_bundle = freeze_map_bundle(
                map_yaml,
                semantic_map_id="arena_map",
                planning_frame="map",
            )
            snapshot = new_candidate_snapshot(
                snapshot_id="snapshot_001",
                created_unix_sec=10.0,
                planning_frame="map",
                map_bundle_sha256=map_bundle.bundle_sha256,
                candidates=(
                    FrozenCandidate(
                        "candidate_a",
                        CandidateGeometry(0.1, 0.1, 0.06, 0.02, 0.25),
                        CandidateSource(
                            "lidar/stand_confirmation",
                            "1" * 64,
                            "2" * 64,
                            ("observation_a",),
                        ),
                        0.9,
                        7,
                        8.0,
                        9.0,
                    ),
                ),
            )
            snapshot_path = root / "snapshot.json"
            snapshot_sha256 = write_candidate_snapshot(snapshot_path, snapshot)
            registry = new_station_identity_registry(
                registry_id="registry_001",
                created_unix_sec=11.0,
                candidate_snapshot_sha256=snapshot_sha256,
                source_artifact_sha256="3" * 64,
                expected_candidate_uids=("candidate_a",),
                mappings=(StationIdentity("candidate_a", "A", "station_A"),),
            )
            registry_path = root / "registry.json"
            write_station_identity_registry(registry_path, registry)
            calibration_path = root / "calibration.json"
            calibration_sha256 = write_camera_calibration(
                calibration_path,
                calibration(),
            )
            site = root / "arena_real.txt"
            site.write_text("measured physical arena descriptor\n")
            import hashlib

            site_sha256 = hashlib.sha256(site.read_bytes()).hexdigest()
            robot_path = root / "robot.json"
            write_real_robot_profile(
                robot_path,
                robot_profile(calibration_sha256, site_sha256),
            )
            stand_model_path = root / "measured_stand.json"
            stand_model_sha256 = write_test_stand_model(stand_model_path)
            output_dir = root / "survey"
            stdout = StringIO()
            with redirect_stdout(stdout):
                status = prepare_passive_survey(
                    [
                        "--robot-profile",
                        str(robot_path),
                        "--camera-calibration",
                        str(calibration_path),
                        "--stand-model-profile",
                        str(stand_model_path),
                        "--physical-site",
                        str(site),
                        "--map",
                        str(map_yaml),
                        "--semantic-map-id",
                        "arena_map",
                        "--candidate-snapshot",
                        str(snapshot_path),
                        "--station-identity-registry",
                        str(registry_path),
                        "--output-dir",
                        str(output_dir),
                        "--catalog",
                        str(root / "catalog.json"),
                        "--catalog-id",
                        "real_catalog_001",
                        "--session-id",
                        "real_session_001",
                        "--survey-manifest",
                        str(root / "survey_manifest.json"),
                    ]
                )
            summary = json.loads(stdout.getvalue())
            plan = load_content_hashed_json(
                Path(summary["plan"]),
                hash_field="real_experiment_plan_sha256",
            )
            survey_config = load_content_hashed_json(
                Path(plan["survey_config"]),
                hash_field="survey_config_sha256",
            )

        self.assertEqual(status, 0)
        self.assertEqual(plan["motion_capability"], "none")
        self.assertEqual(
            survey_config["observer_version"],
            PASSIVE_VIEWPOINT_OBSERVER_VERSION,
        )
        self.assertEqual(
            survey_config["stand_model_profile_sha256"],
            stand_model_sha256,
        )
        self.assertAlmostEqual(
            survey_config["stand_head_center_height_m"],
            0.171,
        )
        observer = plan["candidate_runs"][0]["observer_command"]
        planner = plan["candidate_runs"][0]["catalog_validation_command"]
        self.assertIn("passive_viewpoint_node.py", observer[1])
        self.assertIn("--status-events-jsonl", observer)
        self.assertIn("--stand-model-profile", observer)
        self.assertAlmostEqual(
            float(observer[observer.index("--stand-head-center-height-m") + 1]),
            0.171,
        )
        self.assertNotIn("--stand-face-size-m", observer)
        self.assertNotIn("cmd_vel", " ".join(observer))
        self.assertNotIn("--allow-sim-time", planner)
        self.assertEqual(planner[planner.index("--environment") + 1], "real")
        self.assertIn("--start-from-recommendation", planner)
        self.assertEqual(planner[planner.index("--map-frame") + 1], "map")


class RealUnloadedSegmentAdapterTest(unittest.TestCase):
    def args(self, *, execute=False):
        paths = {
            name: Path(f"{name}.json")
            for name in (
                "diagnostics_json",
                "route_certificate_json",
                "route_bundle_json",
                "planner_config_json",
                "mission_plan_manifest",
                "survey_manifest",
                "runtime_map_bundle_json",
                "candidate_snapshot",
                "station_identity_registry",
                "arrival_pose_catalog",
                "task_snapshot",
            )
        }
        return SimpleNamespace(
            route_csv=Path("route.csv"),
            physical_site=Path("arena_real.txt"),
            leg_index=1,
            run_id="real_unloaded_001",
            allowed_cmd_vel_publisher=["/robot1/velocity_smoother"],
            operator_note="first unloaded leg",
            output_root=Path("results/real_runs"),
            execute=execute,
            **paths,
        )

    def test_dry_run_is_default_and_never_enables_sim_time(self):
        profile = robot_profile("b" * 64, "a" * 64)
        command = build_runner_command(self.args(), profile)
        self.assertIn("--dry-run", command)
        self.assertNotIn("--allow-sim-time", command)
        self.assertEqual(command[command.index("--namespace") + 1], "robot1")
        self.assertEqual(command[command.index("--map-frame") + 1], "map")
        self.assertIn("UNLOADED", command[command.index("--operator-note") + 1])

    def test_profile_must_match_survey_and_planner_footprint(self):
        profile = robot_profile("b" * 64, "a" * 64)
        survey = SimpleNamespace(
            planning_frame="map",
            calibration_profile=SimpleNamespace(sha256="b" * 64),
        )
        planner_config = {
            "schema_version": 1,
            "artifact_kind": "planner_config",
            "route_purpose": "logistics",
            "start_pose": {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
            "robot_radius_m": profile.robot_radius_m,
            "tracking_margin_m": 0.03,
            "collision_margin_m": 0.01,
            "inflation_radius_m": 0.20,
            "corridor_sample_spacing_m": 0.02,
            "lidar_stop_distance_m": 0.15,
            "scan_origin_to_base_offset_m": (
                profile.scan_origin_to_base_offset_m
            ),
            "lidar_clearance_margin_m": 0.02,
            "arena_bounds": {
                "length_m": 2.0,
                "width_m": 2.0,
                "center_x_m": 1.0,
                "center_y_m": 1.0,
                "yaw_deg": 0.0,
                "margin_m": 0.05,
            },
            "arena_boundary_overlay": True,
            "command_owner": "/aufgabe04_station_segment_runner",
            "algorithm": "fixed_order",
            "max_task_snapshot_age_sec": 30.0,
            "max_task_future_skew_sec": 0.1,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "planner.json"
            write_content_hashed_json(
                path,
                planner_config,
                hash_field="artifact_sha256",
            )
            validate_profile_artifact_bindings(profile, survey, path)

            mismatched = {
                **planner_config,
                "robot_radius_m": profile.robot_radius_m + 0.01,
            }
            other = Path(tmp) / "planner_mismatch.json"
            write_content_hashed_json(
                other,
                mismatched,
                hash_field="artifact_sha256",
            )
            with self.assertRaisesRegex(ValueError, "robot_radius_m"):
                validate_profile_artifact_bindings(profile, survey, other)

    def test_execute_always_uses_evidence_bundle_and_keeps_run_prompt(self):
        profile = robot_profile("b" * 64, "a" * 64)
        command = build_execution_command(self.args(execute=True), profile)
        self.assertEqual(command[0], "scripts/common/run_with_bundle.sh")
        self.assertIn("--output-root", command)
        inner = command[command.index("--") + 1 :]
        self.assertIn("run_single_station_segment.py", inner[1])
        self.assertNotIn("--dry-run", inner)
        self.assertNotIn("--allow-sim-time", inner)
        self.assertNotIn("--yes", inner)


if __name__ == "__main__":
    unittest.main()
