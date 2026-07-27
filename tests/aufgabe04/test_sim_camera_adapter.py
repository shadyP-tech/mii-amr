import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.aufgabe04.perception.debug.stand_axis_viewer import (
    SimulationRobotPose,
    _nearest_simulation_pose,
    _project_simulation_map_target,
    _simulation_pose_frame_error,
    _validate_runtime_args,
    annotate_simulation_target_roi,
    build_parser,
)
from scripts.aufgabe04.perception.ros_image_adapter import raw_msg_to_bgr_frame
from scripts.aufgabe04.simulation.prepare_burger_camera_model import main


class SimCameraAdapterTest(unittest.TestCase):
    def test_spawn_script_publishes_simulated_sensor_tf_chain(self):
        script = (
            Path("scripts/aufgabe04/simulation/spawn_burger_camera.sh").read_text()
        )
        self.assertIn("--frame-id base_footprint --child-frame-id base_link", script)
        self.assertIn("--frame-id base_link --child-frame-id base_scan", script)
        self.assertIn("--frame-id base_link --child-frame-id camera_link", script)
        self.assertIn("--model-resource-root", script)
        self.assertIn("BURGER_CAMERA_MODEL_RESOURCE_ROOT", script)

    def test_sim_raw_topic_is_explicit_and_exclusive(self):
        args = build_parser().parse_args(["--sim-raw-image-topic", "/camera/image_raw"])
        self.assertEqual(args.sim_raw_image_topic, "/camera/image_raw")
        self.assertIsNone(args.compressed_image_topic)
        with self.assertRaises(SystemExit):
            build_parser().parse_args([
                "--sim-raw-image-topic", "/camera/image_raw",
                "--compressed-image-topic", "/camera/image_raw/compressed",
            ])

    def test_map_target_mode_requires_simulation_target_and_sensors(self):
        args = build_parser().parse_args(
            [
                "--sim-raw-image-topic",
                "/camera/image_raw",
                "--axis-source",
                "edges",
                "--stand-face-size-m",
                "0.078",
                "--camera-fx-px",
                "381.36246688",
                "--lidar-bearing-source",
                "map-target",
                "--scan-topic",
                "/scan",
                "--use-lidar-distance",
                "--odom-topic",
                "/odom",
                "--stand-x",
                "-0.395",
                "--stand-y",
                "-0.415",
            ]
        )

        _validate_runtime_args(args)
        self.assertEqual(args.lidar_bearing_source, "map-target")
        self.assertEqual(args.odom_topic, "/odom")

        real_args = build_parser().parse_args(
            [
                "--compressed-image-topic",
                "/camera/image_raw/compressed",
                "--lidar-bearing-source",
                "map-target",
            ]
        )
        with self.assertRaisesRegex(ValueError, "simulation-only"):
            _validate_runtime_args(real_args)

        missing_target = build_parser().parse_args(
            [
                "--sim-raw-image-topic",
                "/camera/image_raw",
                "--stand-face-size-m",
                "0.078",
                "--camera-fx-px",
                "381.36246688",
                "--lidar-bearing-source",
                "map-target",
                "--scan-topic",
                "/scan",
                "--use-lidar-distance",
            ]
        )
        with self.assertRaisesRegex(ValueError, "--stand-x and --stand-y"):
            _validate_runtime_args(missing_target)

    def test_map_target_projection_separates_scan_and_image_bearings(self):
        pose = SimulationRobotPose(
            stamp_sec=100.0,
            frame_id="odom",
            child_frame_id="base_footprint",
            x_m=-0.10326933698754208,
            y_m=-0.1750773904064177,
            z_m=0.00850170436169892,
            yaw_rad=-2.3643049695857425,
        )

        projection = _project_simulation_map_target(
            robot_pose=pose,
            stand_x_m=-0.395,
            stand_y_m=-0.415,
            stand_head_center_height_m=0.165035,
            camera_forward_offset_m=0.076,
            camera_lateral_offset_m=0.0,
            camera_height_m=0.093,
            camera_yaw_offset_rad=0.0,
            frame_width=640,
            frame_height=480,
            camera_fx_px=381.36246688,
            camera_fy_px=381.36246688,
            camera_cx_px=320.5,
            camera_cy_px=240.5,
            stand_face_size_m=0.078,
            head_roi_padding_scale=1.6,
        )

        self.assertAlmostEqual(math.degrees(projection.scan_bearing_rad), -5.10, delta=0.1)
        self.assertAlmostEqual(math.degrees(projection.camera.bearing_rad), 6.38, delta=0.1)
        self.assertIsNotNone(projection.roi)
        self.assertAlmostEqual(
            (projection.roi.x0 + projection.roi.x1) / 2.0,
            363.0,
            delta=1.0,
        )

    def test_simulation_pose_sync_and_frames_fail_closed(self):
        poses = (
            SimulationRobotPose(9.8, "odom", "base_footprint", 0, 0, 0, 0),
            SimulationRobotPose(10.02, "odom", "base_footprint", 1, 0, 0, 0),
        )

        selected = _nearest_simulation_pose(
            poses,
            image_stamp_sec=10.0,
            tolerance_sec=0.12,
        )
        self.assertEqual(selected, poses[1])
        self.assertIsNone(
            _nearest_simulation_pose(
                poses,
                image_stamp_sec=10.4,
                tolerance_sec=0.12,
            )
        )
        self.assertIsNone(
            _simulation_pose_frame_error(
                selected,
                map_frame="/odom",
                base_frame="/base_footprint",
            )
        )
        self.assertEqual(
            _simulation_pose_frame_error(
                selected,
                map_frame="map",
                base_frame="base_footprint",
            ),
            "odom_frame_mismatch",
        )

    def test_map_target_projection_rejects_target_behind_camera(self):
        pose = SimulationRobotPose(
            stamp_sec=1.0,
            frame_id="odom",
            child_frame_id="base_footprint",
            x_m=0.0,
            y_m=0.0,
            z_m=0.0,
            yaw_rad=0.0,
        )

        projection = _project_simulation_map_target(
            robot_pose=pose,
            stand_x_m=-0.4,
            stand_y_m=0.0,
            stand_head_center_height_m=0.165035,
            camera_forward_offset_m=0.076,
            camera_lateral_offset_m=0.0,
            camera_height_m=0.093,
            camera_yaw_offset_rad=0.0,
            frame_width=640,
            frame_height=480,
            camera_fx_px=381.36246688,
            camera_fy_px=381.36246688,
            camera_cx_px=320.5,
            camera_cy_px=240.5,
            stand_face_size_m=0.078,
            head_roi_padding_scale=1.6,
        )

        self.assertLess(projection.camera.depth_m, 0.0)
        self.assertIsNone(projection.roi)

    def test_full_frame_overlay_draws_the_exact_processed_roi(self):
        class FakeCv2:
            FONT_HERSHEY_SIMPLEX = 0

            def __init__(self):
                self.rectangles = []
                self.text = []

            def rectangle(self, frame, start, end, color, thickness):
                self.rectangles.append((start, end, color, thickness))

            def putText(self, frame, text, origin, *args):
                self.text.append((text, origin))

        roi = SimpleNamespace(x0=162, y0=78, x1=324, y1=240)
        cv2 = FakeCv2()

        annotate_simulation_target_roi(
            cv2,
            object(),
            target_roi=roi,
            camera_bearing_rad=-0.20,
            scan_bearing_rad=0.15,
            camera_depth_m=0.296,
            failure_reason=None,
        )

        self.assertEqual(
            cv2.rectangles,
            [((162, 78), (323, 239), (255, 255, 0), 2)],
        )
        self.assertIn("target ROI", cv2.text[0][0])

    def test_rgb8_raw_image_becomes_bgr(self):
        import cv2
        import numpy

        message = SimpleNamespace(
            encoding="rgb8", width=1, height=1, step=3, data=bytes((1, 2, 3))
        )
        frame = raw_msg_to_bgr_frame(message, cv2, numpy)
        self.assertEqual(frame[0, 0].tolist(), [3, 2, 1])

    def test_generated_sdf_has_valid_simulation_fov(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.sdf"
            output = Path(tmp) / "generated.sdf"
            source.write_text(
                '<sdf><model name="burger"><sensor name="camera" type="wideanglecamera"><visualize>true</visualize><camera>'
                '<horizontal_fov>3.183</horizontal_fov><image><width>320</width>'
                '<height>240</height></image><lens><type>custom</type></lens>'
                '</camera></sensor></model></sdf>'
            )
            from unittest.mock import patch
            with patch("sys.argv", ["prepare", "--source", str(source), "--output", str(output)]):
                self.assertEqual(main(), 0)
            generated = output.read_text()
        self.assertIn("<horizontal_fov>1.3962634</horizontal_fov>", generated)
        self.assertNotIn("3.183", generated)
        self.assertIn('type="camera"', generated)
        self.assertNotIn("<lens>", generated)
        self.assertIn("<width>640</width>", generated)
        self.assertIn("<height>480</height>", generated)
        self.assertIn("<visualize>false</visualize>", generated)
        self.assertNotIn("<visualize>true</visualize>", generated)
        self.assertIn("libgazebo_ros_p3d.so", generated)
        self.assertIn("odom:=/gazebo_ground_truth", generated)
        self.assertIn("<frame_name>world</frame_name>", generated)

    def test_generated_sdf_resolves_turtlebot_meshes_without_server_model_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.sdf"
            output = root / "generated.sdf"
            models = root / "models"
            mesh = (
                models
                / "turtlebot3_common"
                / "meshes"
                / "bases"
                / "burger_base.stl"
            )
            mesh.parent.mkdir(parents=True)
            mesh.write_bytes(b"solid burger\nendsolid burger\n")
            source.write_text(
                '<sdf><model name="burger"><link name="base"><visual name="base">'
                "<geometry><mesh><uri>"
                "model://turtlebot3_common/meshes/bases/burger_base.stl"
                "</uri></mesh></geometry></visual></link>"
                '<sensor name="camera" type="wideanglecamera"><visualize>true</visualize>'
                "<camera><horizontal_fov>3.183</horizontal_fov>"
                "<image><width>320</width><height>240</height></image>"
                "<lens><type>custom</type></lens></camera></sensor></model></sdf>"
            )
            from unittest.mock import patch
            with patch(
                "sys.argv",
                [
                    "prepare",
                    "--source",
                    str(source),
                    "--output",
                    str(output),
                    "--model-resource-root",
                    str(models),
                ],
            ):
                self.assertEqual(main(), 0)

            generated = output.read_text()
        self.assertIn(f"<uri>{mesh.resolve().as_uri()}</uri>", generated)
        self.assertNotIn("model://turtlebot3_common", generated)

    def test_model_resource_resolution_fails_when_mesh_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.sdf"
            models = root / "models"
            (models / "turtlebot3_common").mkdir(parents=True)
            source.write_text(
                '<sdf><model name="burger"><link name="base"><visual name="base">'
                "<geometry><mesh><uri>"
                "model://turtlebot3_common/meshes/bases/missing.stl"
                "</uri></mesh></geometry></visual></link>"
                '<sensor name="camera" type="wideanglecamera"><visualize>true</visualize>'
                "<camera><horizontal_fov>3.183</horizontal_fov>"
                "<image><width>320</width><height>240</height></image>"
                "<lens><type>custom</type></lens></camera></sensor></model></sdf>"
            )
            from unittest.mock import patch
            with patch(
                "sys.argv",
                [
                    "prepare",
                    "--source",
                    str(source),
                    "--output",
                    str(root / "generated.sdf"),
                    "--model-resource-root",
                    str(models),
                ],
            ):
                with self.assertRaisesRegex(
                    SystemExit,
                    "model resource does not exist",
                ):
                    main()


if __name__ == "__main__":
    unittest.main()
