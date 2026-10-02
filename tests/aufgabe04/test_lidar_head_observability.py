from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from scripts.aufgabe04.navigation.approach.lidar_head_observability import (
    lidar_head_model_admission, verify_lidar_head_observability,
)
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.real_robot.readiness.tour_scan_contract import scan_evidence, capture_payload
from tests.aufgabe04.test_tour_scan_capture import transform, stamp


class LidarHeadObservabilityTest(unittest.TestCase):
    def setUp(self):
        self.model = load_measured_physical_stand_model(Path(__file__).parents[2] /
            "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json")

    def mount(self, value, *, height=.182, pitch=0.):
        laser, base = transform(value), transform(value, "base_footprint", 0.)
        laser.transform.translation.z = height
        laser.transform.rotation = NS(x=0., y=math.sin(pitch/2), z=0., w=math.cos(pitch/2))
        message = NS(header=NS(frame_id="laser", stamp=stamp(value)), angle_min=-1.,
                     angle_increment=.1, range_min=.05, range_max=8., ranges=(.5, .6))
        return scan_evidence(message, received_at_unix_sec=value+.01,
            scan_transform=laser, base_transform=base, odom_frame="odom",
            base_frame="base_footprint", scan_frame="laser")

    def verify(self, *, scans=None, **overrides):
        scans = scans or [self.mount(v) for v in (100., 100.1, 100.2)]
        values = dict(stand_model=self.model, base_frame="base_footprint", target_range_m=.75,
                      mount_evidence=[{**s["head_plane_mount"], "stamp_sec":s["stamp_sec"]} for s in scans],
                      source_scan_stamps_sec=[s["stamp_sec"] for s in scans])
        values.update(overrides)
        return verify_lidar_head_observability(**values)

    def test_measured_config_must_match_fitted_cross_section(self):
        self.assertTrue(lidar_head_model_admission(self.model)["accepted"])
        for model in (None, replace(self.model, head_width_m=.09),
                      replace(self.model, measurement_status="provisional")):
            self.assertFalse(lidar_head_model_admission(model)["accepted"])

    def test_exact_height_and_direction_survive_capture_hash_payload(self):
        scans = [self.mount(v, pitch=.01) for v in (100.,100.1,100.2)]
        result = capture_payload(scans, tour_id="test", odom_frame="odom", base_frame="base_footprint",
                                 scan_frame="laser", captured_at_unix_sec=100.21)
        metadata = result["scans"][0]["head_plane_mount"]
        self.assertAlmostEqual(metadata["scan_height_above_ground_m"], .182)
        self.assertAlmostEqual(metadata["scan_vertical_direction_x"], -math.sin(.01))
        self.assertEqual(metadata["exact_transform_stamp_sec"], 100.)
        self.assertTrue(self.verify(scans=scans)["accepted"])

    def test_horizontal_head_plane_is_supported_but_not_physical_calibration(self):
        result = self.verify()
        self.assertTrue(result["head_cross_section_supported"])
        self.assertFalse(result["physical_mount_calibration_verified"])
        self.assertEqual(result["stand_model_profile_sha256"], self.model.sha256)

    def test_plane_outside_head_or_tilt_at_target_range_is_unverified(self):
        for values in ({"height":.25}, {"height":.12}, {"pitch":.049}):
            result = self.verify(scans=[self.mount(v, **values) for v in (100.,100.1,100.2)])
            self.assertFalse(result["accepted"])
            self.assertEqual(result["reason"], "laser_plane_not_inside_measured_head")

    def test_missing_partial_or_wrong_ground_frame_cannot_verify(self):
        self.assertFalse(self.verify(mount_evidence=())["accepted"])
        self.assertFalse(self.verify(base_frame="base_link")["accepted"])
        for values in (dict(target_range_m=float("nan")), dict(target_range_m=-1.)):
            self.assertFalse(self.verify(**values)["accepted"])
        scans = [self.mount(v) for v in (100.,100.1,100.2)]
        self.assertFalse(self.verify(scans=scans[:2])["accepted"])
        scans[-1]["head_plane_mount"]["ground_frame"] = "other"
        with self.assertRaisesRegex(ValueError, "ground frame"):
            capture_payload(scans, tour_id="test", odom_frame="odom", base_frame="base_footprint",
                            scan_frame="laser", captured_at_unix_sec=100.21)

    def test_mount_timestamp_must_match_exact_scan_time(self):
        scans = [self.mount(v) for v in (100.,100.1,100.2)]
        scans[-1]["head_plane_mount"]["exact_transform_stamp_sec"] = 100.
        self.assertFalse(self.verify(scans=scans)["accepted"])
        with self.assertRaisesRegex(ValueError, "timestamp"):
            capture_payload(scans, tour_id="test", odom_frame="odom", base_frame="base_footprint",
                            scan_frame="laser", captured_at_unix_sec=100.21)

    def test_complete_eight_scan_cohort_checks_every_beam_plane(self):
        scans = [self.mount(100. + index*.1) for index in range(8)]
        result = self.verify(scans=scans)
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["source_scan_stamps_sec"], [s["stamp_sec"] for s in scans])
        self.assertEqual(result["source_scan_count"], 8)
        self.assertEqual(len(result["beam_height_intervals_m"]), 8)
        scans[-1]["head_plane_mount"]["scan_height_above_ground_m"] = .25
        result = self.verify(scans=scans)
        self.assertEqual(result["reason"], "laser_plane_not_inside_measured_head")

    def test_partial_mount_records_cannot_stand_in_for_complete_source_cohort(self):
        scans = [self.mount(100. + index*.1) for index in range(8)]
        mounts = [{**s["head_plane_mount"], "stamp_sec": s["stamp_sec"]} for s in scans]
        for count in (0, 2, 3, 7):
            with self.subTest(count=count):
                result = self.verify(scans=scans, mount_evidence=mounts[:count])
                self.assertFalse(result["accepted"])
                self.assertEqual(result["reason"], "complete_exact_scan_mount_records_required")

    def test_mounts_must_match_complete_source_scan_order(self):
        scans = [self.mount(100. + index*.1) for index in range(8)]
        mounts = [{**s["head_plane_mount"], "stamp_sec": s["stamp_sec"]} for s in scans]
        changed = [dict(record) for record in mounts]
        changed[-1].update(stamp_sec=101., exact_transform_stamp_sec=101.)
        for records in (mounts[:-1] + [mounts[-2]], list(reversed(mounts)), changed):
            with self.subTest(records=records):
                result = self.verify(scans=scans, mount_evidence=records)
                self.assertFalse(result["accepted"])
                self.assertEqual(result["reason"], "head_slice_mount_scan_stamp_mismatch")

    def test_source_cohort_cardinality_and_stamps_fail_closed(self):
        for stamps in ((), [100.], [100.+index*.1 for index in range(4)],
                       [100., 100., 100.2], [100.2, 100.1, 100.],
                       [100., True, 100.2], [100., float("nan"), 100.2]):
            with self.subTest(stamps=stamps):
                result = self.verify(source_scan_stamps_sec=stamps)
                self.assertFalse(result["accepted"])
                self.assertEqual(result["reason"], "head_slice_source_scan_stamps_invalid")


if __name__ == "__main__":
    unittest.main()
