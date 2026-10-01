"""A stopped target miss can defer inspection but never supply target geometry."""

import copy
from dataclasses import asdict
import math
import unittest

from scripts.aufgabe04.perception.candidate_lidar_association import associate_candidate_lidar_target
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.real_robot.observer.target_support_failure import (
    TargetSupportFailureWindow, validate_target_support_failure,
)


def binding():
    return dict(candidate_uid="survey_candidate_0005", stream_id="inspection/3",
                target_key="inspection/3:survey_candidate_0005:1.310817737:0.313812076",
                planning_frame="map", stand_center={"x_m": 1.310817737, "y_m": .313812076},
                candidate_snapshot_sha256="1"*64, robot_profile_sha256="2"*64,
                calibration_profile_sha256="3"*64, stand_model_profile_sha256="4"*64)


def observation(stamp, *, ranges=(1.03, 1.035, 1.04)):
    scan = PlainLaserScan(tuple(ranges), -.02, .02, .05, 5., "base_scan", stamp, stamp+.02)
    raw = associate_candidate_lidar_target(scan, map_bearing_rad=0.,
        cone_half_angle_rad=math.radians(15), accepted_range_m=(.368, .588),
        now_sec=stamp+.1, max_scan_age_sec=.5, min_cluster_sample_count=1)
    return dict(association=asdict(raw), frame=dict(frame_stamp_sec=stamp, scan_stamp_sec=stamp,
                robot_pose=dict(x_m=.798, y_m=.251, yaw_rad=.146), motion_epoch=0,
                tf_validated=True, poisoned=False, motion_epoch_reset=False, frame_accepted=False),
                target_binding=binding(), now_sec=stamp+.1)


class TargetSupportFailureTest(unittest.TestCase):
    def pending(self):
        window = TargetSupportFailureWindow()
        for offset in (0., .8, 1.6, 2.4, 3.2, 4.):
            self.assertIsNone(window.observe(**observation(100.+offset)))
        return window

    def receipt(self):
        result = self.pending().observe(**observation(105.))
        self.assertIsNotNone(result)
        return result

    def test_seven_distinct_fresh_stationary_misses_require_five_seconds(self):
        window = self.pending()
        self.assertIsNone(window.observe(**observation(104.8)))
        result = window.observe(**observation(105.))
        self.assertEqual(result["state"], "target_reconciliation_required")
        self.assertEqual(result["reason"], "persistent_target_support_missing")
        self.assertEqual(result["sample_count"], 8)
        self.assertEqual(result["elapsed_sec"], 5.)
        self.assertFalse(result["motion_authorized"])
        self.assertFalse(result["completion_authorized"])
        self.assertFalse(result["candidate_geometry_updated"])
        self.assertEqual(validate_target_support_failure(result, target_binding=binding()), result)

    def test_time_alone_or_many_fast_samples_cannot_complete(self):
        window = TargetSupportFailureWindow()
        for stamp in (100., 106.):
            self.assertIsNone(window.observe(**observation(stamp)))
        window = TargetSupportFailureWindow()
        for i in range(30):
            self.assertIsNone(window.observe(**observation(100.+.01*i)))

    def test_current_raw_support_successful_reconciliation_and_head_reset(self):
        for kwargs in ({"ranges": (.5, .5, .5)}, {"reconciliation_validated": True}, {"associated_head": True}):
            with self.subTest(kwargs=kwargs):
                window = self.pending()
                options = observation(104.5, ranges=kwargs.get("ranges", (1.03, 1.035, 1.04)))
                options.update({k: v for k, v in kwargs.items() if k != "ranges"})
                self.assertIsNone(window.observe(**options))
                self.assertEqual(window.samples, [])
                self.assertIn("supported", window.metadata["reason"])
                self.assertIsNone(window.observe(**observation(105.)))
                self.assertEqual(len(window.samples), 1)

    def test_a_single_current_in_range_ray_prevents_absence_even_without_a_cluster(self):
        options = observation(105., ranges=(math.nan, .5, math.nan))
        options["association"].update(associated=False,
            rejection_reason="no_contiguous_cluster_meets_minimum", eligible_cluster_count=0,
            selected_cluster_sample_count=0)
        window = self.pending()
        self.assertIsNone(window.observe(**options))
        self.assertEqual(window.samples, [])

    def test_invalid_or_all_missing_rays_do_not_prove_absence(self):
        for ranges in ((math.nan, math.inf, 0.), ()):
            window = self.pending()
            self.assertIsNone(window.observe(**observation(105., ranges=ranges)))
            self.assertEqual(window.samples, [])

    def test_reused_image_or_scan_is_never_another_sample(self):
        for field in ("frame_stamp_sec", "scan_stamp_sec"):
            window = self.pending()
            options = observation(104.1)
            options["frame"][field] = 104.
            if field == "scan_stamp_sec":
                options["association"]["scan_stamp_sec"] = 104.
            self.assertIsNone(window.observe(**options))
            self.assertEqual(len(window.samples), 6)
            self.assertEqual(window.metadata["reason"], "duplicate_or_out_of_order_tuple")

    def test_stale_tf_poison_reset_and_accepted_frames_clear_history(self):
        overrides = [("tf_validated", False), ("poisoned", True),
                     ("motion_epoch_reset", True), ("frame_accepted", True),
                     ("frame_stamp_sec", 104.), ("scan_stamp_sec", 104.)]
        for key, value in overrides:
            with self.subTest(key=key):
                window = self.pending()
                options = observation(105.)
                options["frame"][key] = value
                self.assertIsNone(window.observe(**options))
                self.assertEqual(window.samples, [])

    def test_motion_and_each_target_binding_change_start_new_window(self):
        for field in ("candidate_uid", "stream_id", "target_key", "planning_frame",
                      "stand_center", "candidate_snapshot_sha256", "robot_profile_sha256",
                      "calibration_profile_sha256", "stand_model_profile_sha256"):
            with self.subTest(field=field):
                window = self.pending()
                options = observation(105.)
                options["target_binding"][field] = ({"x_m": 1.4, "y_m": .3} if field == "stand_center"
                    else "a"*64 if field.endswith("sha256") else "changed")
                self.assertIsNone(window.observe(**options))
                self.assertEqual(len(window.samples), 1)
        for key, value in (("x_m", .83), ("yaw_rad", .19)):
            window = self.pending()
            options = observation(105.)
            options["frame"]["robot_pose"][key] = value
            self.assertIsNone(window.observe(**options))
            self.assertEqual(len(window.samples), 1)
        window = self.pending()
        options = observation(105.)
        options["frame"]["motion_epoch"] = 1
        self.assertIsNone(window.observe(**options))
        self.assertEqual(len(window.samples), 1)

    def test_expired_history_cannot_complete(self):
        window = self.pending()
        self.assertIsNone(window.observe(**observation(120.)))
        self.assertEqual(len(window.samples), 1)

    def test_malformed_binding_and_counts_cannot_count_as_a_negative(self):
        for value in (None, "A"*64, "bad", 1):
            window = self.pending()
            options = observation(105.)
            options["target_binding"]["candidate_snapshot_sha256"] = value
            self.assertIsNone(window.observe(**options))
            self.assertEqual(window.samples, [])
        for field, value in (("in_range_sample_count", False), ("cone_valid_sample_count", -1),
                             ("nearest_cone_distance_m", math.nan), ("scan_age_sec", .6),
                             ("nearest_range_delta_m", .01)):
            window = self.pending()
            options = observation(105.)
            options["association"][field] = value
            self.assertIsNone(window.observe(**options))
            self.assertEqual(window.samples, [])

    def test_receipt_rejects_tampered_binding_flags_sample_bounds_and_timestamps(self):
        result = self.receipt()
        for flag in ("motion_authorized", "completion_authorized", "candidate_geometry_updated"):
            altered = copy.deepcopy(result)
            altered[flag] = True
            with self.assertRaises(ValueError):
                validate_target_support_failure(altered)
        altered = copy.deepcopy(result)
        altered["target_binding"]["candidate_uid"] = "different"
        with self.assertRaises(ValueError):
            validate_target_support_failure(altered)
        other = binding()
        other["stream_id"] = "later_inspection"
        with self.assertRaises(ValueError):
            validate_target_support_failure(result, target_binding=other)
        mutations = [lambda p: p.update(elapsed_sec=6.),
                     lambda p: p["samples"].pop(),
                     lambda p: p["samples"][0]["frame"].update(tf_validated=False),
                     lambda p: p["samples"][1]["frame"].update(scan_stamp_sec=100.),
                     lambda p: p["samples"][-1]["association"].update(in_range_sample_count=1)]
        for mutate in mutations:
            altered = copy.deepcopy(result)
            mutate(altered)
            with self.assertRaises(ValueError):
                validate_target_support_failure(altered)

    def test_input_and_receipt_mutations_do_not_change_pending_window(self):
        window = TargetSupportFailureWindow()
        options = observation(100.)
        saved = copy.deepcopy(options)
        window.observe(**options)
        self.assertEqual(saved, options)
        options["frame"]["robot_pose"]["x_m"] = 900.
        self.assertEqual(window.samples[0]["frame"]["robot_pose"]["x_m"], .798)
        for offset in (.8, 1.6, 2.4, 3.2, 4., 5.):
            result = window.observe(**observation(100.+offset))
        result["samples"][0]["frame"]["robot_pose"]["x_m"] = 800.
        self.assertEqual(window.samples[0]["frame"]["robot_pose"]["x_m"], .798)


if __name__ == "__main__":
    unittest.main()
