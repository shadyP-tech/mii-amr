"""Range admission uses LaserScan's origin without weakening its other gates."""

from dataclasses import FrozenInstanceError
from contextlib import ExitStack
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.aufgabe04.perception.candidate_lidar_association import (
    associate_camera_registered_candidate_lidar_target,
)
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.real_robot.configuration.geometry import transform_point
from scripts.aufgabe04.real_robot.observer.scan_target_geometry import scan_target_geometry


POLICY = dict(stand_radius_m=.06, stand_uncertainty_m=.02,
              lidar_range_tolerance_m=.04)
FIXTURE = Path(__file__).with_name("fixtures") / "scan_target_origin_20260914T123717Z.json"


class ScanTargetGeometryTests(unittest.TestCase):
    def test_coincident_origins_preserve_existing_policy_and_planar_range(self):
        target = scan_target_geometry((.3, .4, -.182), **POLICY)
        self.assertEqual(target.center_range_m, .5)
        self.assertAlmostEqual(target.bearing_rad, math.atan2(.4, .3))
        self.assertAlmostEqual(target.accepted_range_m[0], .32)
        self.assertAlmostEqual(target.accepted_range_m[1], .54)
        with self.assertRaises(FrozenInstanceError):
            target.center_range_m = .7

    def test_rotated_robot_with_rear_scanner_shifts_range_and_bearing_together(self):
        # Base is at map (1, 2), facing +y. Scanner sits 32 mm behind it.
        # Candidate is .1 m left and .45 m ahead in robot coordinates.
        point = transform_point(
            (.9, 2.45, 0.), translation_xyz=(-1.968, 1., -.182),
            rotation_xyzw=(0., 0., -math.sqrt(.5), math.sqrt(.5)),
        )
        target = scan_target_geometry(point, **POLICY)
        self.assertAlmostEqual(point[0], .482)
        self.assertAlmostEqual(point[1], .1)
        self.assertAlmostEqual(target.center_range_m, math.hypot(.482, .1))
        self.assertAlmostEqual(target.bearing_rad, math.atan2(.1, .482))
        self.assertGreater(target.center_range_m, math.hypot(.45, .1))
        self.assertAlmostEqual(target.accepted_range_m[1] - target.accepted_range_m[0], .22)

    def test_invalid_geometry_and_policy_fail_closed(self):
        for point in (None, "xyz", (), (.5, 0), (.5, 0, 0, 0),
                      (True, 0, 0), (.5, 0, math.nan), (math.inf, 0, 0),
                      (0, 0, -.182), (1.7e308, 1.7e308, 0)):
            with self.subTest(point=point), self.assertRaises(ValueError):
                scan_target_geometry(point, **POLICY)
        for field in POLICY:
            for value in (-.1, math.nan, math.inf, True, ".04"):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    scan_target_geometry((.5, 0, 0), **{**POLICY, field: value})

    def test_recorded_scan_origin_correction_preserves_unique_cluster_gate(self):
        fixture = json.loads(FIXTURE.read_text())
        policy = fixture["policy"]
        for case in fixture["cases"]:
            with self.subTest(source=case["source_frame_path"]):
                tf = case["scan_from_map"]
                target = scan_target_geometry(
                    transform_point(fixture["candidate_map_xyz_m"],
                                    translation_xyz=tf["translation_xyz_m"],
                                    rotation_xyzw=tf["rotation_xyzw"]),
                    **{key: policy[key] for key in POLICY},
                )
                self.assertAlmostEqual(target.bearing_rad, case["map_bearing_rad"])
                self.assertAlmostEqual(target.center_range_m, .480567, delta=1e-6)
                old_range = tuple(case["old_accepted_range_m"])
                self.assertAlmostEqual(target.accepted_range_m[1] - old_range[1], .031995, delta=1e-6)
                scan = PlainLaserScan(
                    **{key: case["scan"][key] for key in ("angle_min", "angle_max", "angle_increment", "range_min", "range_max")},
                    ranges=tuple(float(value) for value in case["scan"]["ranges"]),
                    scan_frame_id=case["scan_frame_id"], scan_stamp_sec=case["scan_stamp_sec"],
                    scan_topology_profile="full_rotation",
                )
                options = dict(
                    map_bearing_rad=target.bearing_rad,
                    observed_camera_bearing_rad=case["camera_bearing_rad"],
                    now_sec=case["scan_stamp_sec"] + case["scan_age_sec"],
                    **{key: policy[key] for key in policy if key not in POLICY},
                )
                before = associate_camera_registered_candidate_lidar_target(
                    scan, accepted_range_m=old_range, **options,
                )
                after = associate_camera_registered_candidate_lidar_target(
                    scan, accepted_range_m=target.accepted_range_m, **options,
                )
                self.assertEqual(before.rejection_reason, "no_samples_in_accepted_range")
                self.assertEqual(after.associated, case["expected_corrected_associated"])
                self.assertEqual(after.rejection_reason, case["expected_corrected_reason"])
                self.assertEqual(after.search_association.eligible_cluster_count,
                                 case["expected_corrected_eligible_clusters"])

    def test_observer_uses_scan_origin_for_current_head_association(self):
        import numpy
        from tests.aufgabe04 import test_camera_observer_processing as fixture
        from tests.aufgabe04.test_head_backside_classification import classified_head

        case = fixture.CameraObserverProcessingTest()
        adapter = case.make_adapter()
        original_lookup = adapter._lookup
        adapter._lookup = lambda target, source, stamp: (
            fixture.transform((.032, 0., -.182))
            if (target, source) == ("scan", "map")
            else original_lookup(target, source, stamp)
        )
        adapter._lookup_static_transform = lambda target, source: fixture.transform(
            (.032, 0., .268) if target == "scan" else (0., 0., .45),
            (.5, -.5, .5, -.5),
        )
        adapter._next_sensor_tuple.return_value.scan.value.ranges = (.659,) * 5

        def metric(_cv2, _frame, **options):
            estimate, debug, _ = classified_head(
                u=options["expected_head_center_u_px"],
                v=options["expected_head_center_v_px"], height=52.,
                profile_sha256=adapter.stand_model_profile.sha256,
            )
            return estimate, debug

        module = "scripts.aufgabe04.real_robot.observer.node."
        with ExitStack() as stack:
            for name, arguments in {
                "camera_info_mismatches": dict(return_value=()),
                "transform_mismatches": dict(return_value=()),
                "compressed_msg_to_bgr_frame": dict(return_value=numpy.zeros((600, 800, 3), dtype=numpy.uint8)),
                "_rectify_bgr_frame": dict(side_effect=lambda value, *_: value),
                "detect_qr_observations_bgr": dict(return_value=()),
                "detect_native_qr_observations_bgr": dict(return_value=()),
                "estimate_stand_axis_from_metric_model": dict(side_effect=metric),
            }.items():
                stack.enter_context(patch(module + name, **arguments))
            adapter._process_latest()

        self.assertEqual(adapter._write_status.call_args.args, ("collecting_consensus",))
        metadata = adapter._write_status.call_args.kwargs["stand_axis_debug"]
        association = metadata["current_head_candidate_association"]
        self.assertTrue(association["accepted"])
        self.assertEqual(association["lidar_association"]["distance_m"], .659)
        bounds = association["lidar_association"]["search_association"]["accepted_range_m"]
        self.assertAlmostEqual(bounds[0], .452)
        self.assertAlmostEqual(bounds[1], .672)


if __name__ == "__main__":
    unittest.main()
