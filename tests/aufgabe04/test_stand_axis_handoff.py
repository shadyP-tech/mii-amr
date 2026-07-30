import math
import unittest

from scripts.aufgabe04.perception.stand_axis.geometry import (
    estimate_stand_axis_from_corners,
)
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint
from scripts.aufgabe04.perception.stand_axis_handoff import (
    AxialConsensusAccumulator,
    AxisHandoffConfig,
    CameraAxisEstimate,
    LidarAxisEstimate,
    RigidTransform,
    axial_difference_rad,
    camera_axis_in_scan,
    camera_face_normal_axis_in_scan,
    estimate_pooled_lidar_axis,
    evaluate_axis_handoff,
    rectified_pixel_bearing_in_scan,
    transform_point,
)
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan


CAMERA_TO_SCAN = RigidTransform(
    parent_frame="base_scan",
    child_frame="camera",
    translation_xyz_m=(0.08, 0.0, -0.05),
    rotation_xyzw=(0.5, -0.5, 0.5, -0.5),
)


def synthetic_line_scans(
    *,
    axis_rad: float,
    center_xy_m: tuple[float, float] = (0.60, 0.0),
    scan_count: int = 8,
) -> tuple[PlainLaserScan, ...]:
    angle_min = math.radians(-20.0)
    angle_increment = math.radians(0.25)
    sample_count = 161
    scans = []
    for scan_index in range(scan_count):
        ranges = [float("inf")] * sample_count
        for point_index in range(11):
            along = -0.04 + point_index * 0.008
            x_m = (
                center_xy_m[0]
                + along * math.cos(axis_rad)
                + scan_index * 0.00005
            )
            y_m = (
                center_xy_m[1]
                + along * math.sin(axis_rad)
                - scan_index * 0.00003
            )
            bearing = math.atan2(y_m, x_m)
            index = int(round((bearing - angle_min) / angle_increment))
            if 0 <= index < sample_count:
                distance = math.hypot(x_m, y_m)
                ranges[index] = min(ranges[index], distance)
        scans.append(
            PlainLaserScan(
                ranges=tuple(ranges),
                angle_min=angle_min,
                angle_increment=angle_increment,
                range_min=0.10,
                range_max=3.50,
                scan_frame_id="base_scan",
                scan_stamp_sec=10.0 + scan_index * 0.1,
                receipt_sec=20.0 + scan_index * 0.1,
            )
        )
    return tuple(scans)


class StandAxisHandoffGeometryTest(unittest.TestCase):
    def test_metric_square_pose_preserves_normal_and_center_for_handoff(self):
        import numpy

        class FakeCv2:
            SOLVEPNP_ITERATIVE = 0

            @staticmethod
            def solvePnP(*_args, **_kwargs):
                return (
                    True,
                    numpy.zeros((3, 1), dtype=float),
                    numpy.asarray([[0.10], [0.02], [0.60]], dtype=float),
                )

            @staticmethod
            def Rodrigues(_rotation_vector):
                return numpy.eye(3), None

        estimate = estimate_stand_axis_from_corners(
            (
                ImagePoint(300.0, 200.0),
                ImagePoint(340.0, 200.0),
                ImagePoint(340.0, 240.0),
                ImagePoint(300.0, 240.0),
            ),
            stand_width_m=0.078,
            camera_fx_px=640.0,
            camera_fy_px=641.0,
            camera_cx_px=320.0,
            camera_cy_px=220.0,
            cv2=FakeCv2(),
        )

        self.assertEqual(estimate.camera_face_normal_xyz, (0.0, 0.0, 1.0))
        self.assertEqual(
            estimate.camera_face_center_xyz_m,
            (0.10, 0.02, 0.60),
        )

    def test_axial_difference_wraps_across_the_180_degree_seam(self):
        difference = axial_difference_rad(
            math.radians(89.0),
            math.radians(-89.0),
        )

        self.assertAlmostEqual(math.degrees(difference), 2.0, places=6)

    def test_axial_consensus_stays_stable_across_the_180_degree_seam(self):
        accumulator = AxialConsensusAccumulator(
            required_samples=3,
            max_deviation_rad=math.radians(3.0),
        )

        self.assertIsNone(
            accumulator.add(
                angle_rad=math.radians(89.0),
                source="pnp",
                side="qr_code_side",
                qr_texts=("QR_002",),
            )
        )
        self.assertIsNone(
            accumulator.add(
                angle_rad=math.radians(-89.0),
                source="pnp",
                side="qr_code_side",
                qr_texts=("QR_002",),
            )
        )
        result = accumulator.add(
            angle_rad=math.radians(90.0),
            source="pnp",
            side="qr_code_side",
            qr_texts=("QR_002",),
        )

        self.assertIsNotNone(result)
        self.assertLess(
            axial_difference_rad(result.angle_rad, math.radians(90.0)),
            math.radians(0.1),
        )

    def test_rectified_image_center_projects_to_scan_forward(self):
        bearing = rectified_pixel_bearing_in_scan(
            u_px=406.0,
            v_px=301.0,
            fx_px=640.0,
            fy_px=641.0,
            cx_px=406.0,
            cy_px=301.0,
            scan_from_camera=CAMERA_TO_SCAN,
        )

        self.assertAlmostEqual(bearing, 0.0, places=7)

    def test_metric_pnp_center_uses_translation_in_full_extrinsic(self):
        point = transform_point((0.10, 0.02, 0.60), CAMERA_TO_SCAN)

        self.assertAlmostEqual(point[0], 0.68, places=7)
        self.assertAlmostEqual(point[1], -0.10, places=7)

    def test_frontal_camera_face_maps_to_scan_lateral_tangent(self):
        axis = camera_axis_in_scan(
            camera_yaw_rad=0.0,
            scan_from_camera=CAMERA_TO_SCAN,
        )

        self.assertAlmostEqual(math.degrees(axis), 90.0, places=7)

    def test_full_pnp_normal_matches_measured_extrinsic_axis_convention(self):
        measured_transform = RigidTransform(
            parent_frame="base_scan",
            child_frame="camera",
            translation_xyz_m=(0.07754, -0.00497, -0.05625),
            rotation_xyzw=(
                0.4671136171146707,
                -0.4831412431691032,
                0.5243683107682287,
                -0.5228931846153062,
            ),
        )
        # A unit camera-frame normal corresponding to a 69.477-degree
        # scan-frame tangent in the sealed post-battery geometry.
        normal_camera = (
            0.33478205706648645,
            0.08605832080147202,
            0.9383575756006677,
        )

        axis = camera_face_normal_axis_in_scan(
            camera_face_normal_xyz=normal_camera,
            scan_from_camera=measured_transform,
        )

        self.assertAlmostEqual(math.degrees(axis), 69.477, places=3)


class PooledLidarAxisTest(unittest.TestCase):
    def test_temporally_pooled_returns_recover_coarse_axis(self):
        expected_axis = math.radians(72.0)

        estimate = estimate_pooled_lidar_axis(
            synthetic_line_scans(axis_rad=expected_axis),
            target_bearing_rad=0.0,
            min_points=20,
            min_linearity=0.85,
            min_length_m=0.04,
            max_length_m=0.12,
        )

        self.assertTrue(estimate.usable, estimate.reason)
        self.assertGreaterEqual(estimate.scan_count, 3)
        self.assertGreaterEqual(estimate.sample_count, 20)
        self.assertLess(
            math.degrees(
                axial_difference_rad(estimate.angle_rad, expected_axis)
            ),
            3.0,
        )

    def test_no_scans_fail_closed(self):
        estimate = estimate_pooled_lidar_axis(
            (),
            target_bearing_rad=0.0,
        )

        self.assertFalse(estimate.usable)
        self.assertEqual(estimate.reason, "no_scans")


class AxisHandoffDecisionTest(unittest.TestCase):
    def setUp(self):
        self.lidar = LidarAxisEstimate(
            True,
            "axis_estimated",
            angle_rad=math.radians(90.0),
            center_xy_m=(0.60, 0.0),
            sample_count=40,
            linearity=0.98,
        )

    def test_consistent_camera_axis_is_authoritative_but_observe_only(self):
        camera = CameraAxisEstimate(
            True,
            "camera_consensus_ready",
            angle_rad=math.radians(-88.0),
            sample_count=5,
        )

        decision = evaluate_axis_handoff(lidar=self.lidar, camera=camera)

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.status, "camera_refined")
        self.assertEqual(decision.accepted_axis_rad, camera.angle_rad)
        self.assertTrue(decision.observe_only)
        self.assertFalse(decision.motion_authorized)
        self.assertAlmostEqual(decision.approach_pose.x_m, 0.15, places=7)
        self.assertAlmostEqual(decision.approach_pose.yaw_rad, 0.0, places=7)

    def test_sealed_post_battery_pair_passes_provisional_gate(self):
        lidar = LidarAxisEstimate(
            True,
            "axis_estimated",
            angle_rad=math.radians(75.25274189852891),
            center_xy_m=(0.6619404204, -0.0483194570),
            sample_count=616,
            scan_count=158,
            linearity=0.9627225326,
        )
        camera = CameraAxisEstimate(
            True,
            "metric_camera_consensus_ready",
            angle_rad=math.radians(69.47702207487575),
            sample_count=107,
        )

        decision = evaluate_axis_handoff(lidar=lidar, camera=camera)

        self.assertTrue(decision.accepted)
        self.assertAlmostEqual(
            math.degrees(decision.axial_difference_rad),
            5.77571982365316,
            places=6,
        )

    def test_inconsistent_camera_axis_is_rejected(self):
        camera = CameraAxisEstimate(
            True,
            "camera_consensus_ready",
            angle_rad=math.radians(60.0),
            sample_count=5,
        )

        decision = evaluate_axis_handoff(
            lidar=self.lidar,
            camera=camera,
            config=AxisHandoffConfig(
                max_axis_difference_rad=math.radians(15.0)
            ),
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.status, "axis_inconsistent")
        self.assertIsNone(decision.accepted_axis_rad)
        self.assertIsNotNone(decision.approach_pose)

    def test_lidar_can_offer_only_a_coarse_pose_while_camera_collects(self):
        decision = evaluate_axis_handoff(
            lidar=self.lidar,
            camera=CameraAxisEstimate(False, "camera_consensus_incomplete"),
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.status, "camera_collecting")
        self.assertIsNotNone(decision.approach_pose)
        self.assertIsNone(decision.accepted_axis_rad)


if __name__ == "__main__":
    unittest.main()
