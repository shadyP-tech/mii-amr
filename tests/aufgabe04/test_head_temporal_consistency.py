"""Stationary current-head choices cannot discard an inconvenient seventh fit."""

from dataclasses import replace
import json
import math
import unittest

from scripts.aufgabe04.perception.stand_axis_consensus import AxisConsensusAccumulator
from scripts.aufgabe04.real_robot.observer.head_temporal_consistency import (
    HeadTemporalContext, StationaryHeadConsistency,
)


# Saved-pixel diagnostics, not originally fresh hardware measurements. Source:
# results/audits/stand_explore_exact2_camera_all5_20260914T123717Z/
# correction_validation/backside_first_view_registered_replay_20260914T134555Z.json
# SHA256 d9298565259c347988d4cc4310e6ef4b52c1d2d243140588c853d32c86bf7904
FRAME16 = {
    "stamp_sec": 1789389832.9413471,
    "yaw_rad": math.radians(-2.7742417919913365),
    "corners_full_image": (
        (259.83321917815874, 244.8684808280917),
        (358.5724281758581, 241.91456304645493),
        (357.43985470104775, 342.0),
        (258.73407243383076, 342.0),
    ),
    "projected_center_px": (355.9481250446396, 304.7401984124795),
    "expected_head_height_px": 101.3345656510694,
}
FRAME26 = {
    "stamp_sec": 1789389837.316008,
    "yaw_rad": math.radians(-11.618127128365924),
    "corners_full_image": (
        (259.6214725568629, 241.9105563300666),
        (356.3353045502392, 238.67751732683672),
        (355.4969111057713, 342.0),
        (258.8093130804133, 342.0),
    ),
    "projected_center_px": (356.08577430140457, 304.7388722745743),
    "expected_head_height_px": 101.33318957629686,
}


class HeadTemporalConsistencyTest(unittest.TestCase):
    def setUp(self):
        self.context = HeadTemporalContext("candidate_001", "model-sha", (500., 500., 320., 240.), 0)
        self.guard = StationaryHeadConsistency()

    def observe(self, stamp=100., *, frame=None, context=None, **overrides):
        payload = dict(FRAME16 if frame is None else frame)
        payload["stamp_sec"] = stamp
        payload.update(overrides)
        return self.guard.observe(context=self.context if context is None else context, **payload)

    def test_recorded_border_choice_changes_angle_despite_similar_corners(self):
        first = self.guard.observe(context=self.context, **FRAME16)
        second = self.guard.observe(context=self.context, **FRAME26)
        self.assertTrue(first.current_sample_accepted)
        self.assertFalse(second.current_sample_accepted)
        self.assertTrue(second.reset_axis_evidence)
        self.assertEqual(second.reason, "head_axis_choice_unstable")
        self.assertEqual(second.window_stamps_sec, (FRAME16["stamp_sec"], FRAME26["stamp_sec"]))
        self.assertAlmostEqual(math.degrees(second.angle_interval.span_rad), 8.843885336374588)
        self.assertAlmostEqual(second.corner_displacement_ratio, .03961507454, places=9)
        self.assertLess(second.corner_displacement_ratio, second.max_corner_displacement_ratio)

    def test_six_good_plus_one_outlier_passes_old_mean_but_new_window_vetoes(self):
        legacy = AxisConsensusAccumulator(required_samples=7, max_deviation_rad=math.radians(8.))
        for n in range(7):
            frame = FRAME16 if n < 6 else FRAME26
            legacy_result = legacy.add(yaw_rad=frame["yaw_rad"], source="measured_head",
                                       side="unknown_side", qr_texts=(), target_key="candidate_001")
            # Synthetic timestamps isolate policy; no replay timing is hardware evidence.
            decision = self.observe(100.+n*.1, frame=frame)
        self.assertIsNotNone(legacy_result)
        self.assertFalse(decision.current_sample_accepted)
        self.assertFalse(decision.ready)
        self.assertEqual(decision.sample_count, 7)
        self.assertTrue(decision.reset_axis_evidence)

    def test_outlier_is_retained_until_seven_new_choices_then_only_current_is_admitted(self):
        ordinary = AxisConsensusAccumulator(required_samples=7)

        def record(stamp, frame):
            decision = self.observe(stamp, frame=frame)
            if decision.reset_axis_evidence:
                ordinary.reset()
            if decision.current_sample_accepted:
                ordinary.add(yaw_rad=frame["yaw_rad"], source="measured_head", side="unknown_side",
                             qr_texts=(), target_key="candidate_001")
            return decision

        for n in range(6):
            record(100.+n*.1, FRAME16)
        rejected = record(100.6, FRAME26)
        self.assertFalse(rejected.current_sample_accepted)
        self.assertEqual(ordinary.sample_count, 0)
        for n in range(1, 7):
            rejected = record(100.6+n*.1, FRAME16)
            self.assertIn(100.6, rejected.window_stamps_sec)
            self.assertFalse(rejected.current_sample_accepted)
            self.assertEqual(ordinary.sample_count, 0)
        recovered = record(101.3, FRAME16)
        self.assertNotIn(100.6, recovered.window_stamps_sec)
        self.assertTrue(recovered.current_sample_accepted)
        self.assertTrue(recovered.ready)  # Temporal window only, not route admission.
        self.assertEqual(ordinary.sample_count, 1)
        self.assertFalse(recovered.metadata()["completion_authorized"])

    def test_same_angle_with_incompatible_border_choice_is_withheld(self):
        self.observe()
        shifted = tuple((u+8., v) for u, v in FRAME16["corners_full_image"])
        decision = self.observe(100.1, corners_full_image=shifted)
        self.assertEqual(decision.reason, "head_border_choice_unstable")
        self.assertGreater(decision.corner_displacement_ratio, .07)

    def test_projection_scale_compensates_current_pixel_coordinates(self):
        self.observe()
        scale, dx, dy = 1.25, 12., -5.
        corners = tuple((u*scale+dx, v*scale+dy) for u, v in FRAME16["corners_full_image"])
        center = tuple(value*scale+offset for value, offset in
                       zip(FRAME16["projected_center_px"], (dx, dy)))
        decision = self.observe(100.1, corners_full_image=corners, projected_center_px=center,
                                expected_head_height_px=FRAME16["expected_head_height_px"]*scale)
        self.assertTrue(decision.current_sample_accepted)
        self.assertAlmostEqual(decision.corner_displacement_ratio, 0.)
        # Recentered crops restore full-image corners before reaching this API.
        roi_origin = (200., 200.)
        local = tuple((u-roi_origin[0], v-roi_origin[1]) for u, v in FRAME16["corners_full_image"])
        restored = tuple((u+roi_origin[0], v+roi_origin[1]) for u, v in local)
        decision = self.observe(100.2, corners_full_image=restored)
        self.assertTrue(decision.current_sample_accepted)
        self.assertEqual(decision.sample_count, 3)

    def test_angle_is_an_undirected_axis_and_wrap_interval_is_explicit(self):
        self.observe(yaw_rad=math.radians(89.))
        decision = self.observe(100.1, yaw_rad=math.radians(-89.))
        self.assertTrue(decision.current_sample_accepted)
        self.assertAlmostEqual(math.degrees(decision.angle_interval.span_rad), 2.)
        self.assertTrue(decision.angle_interval.crosses_axis_wrap)
        self.guard.reset()
        self.observe(yaw_rad=math.radians(30.))
        decision = self.observe(100.1, yaw_rad=math.radians(210.))
        self.assertTrue(decision.current_sample_accepted)
        self.assertAlmostEqual(decision.angle_interval.span_rad, 0.)

    def test_same_frame_competing_hypothesis_is_veto_evidence_without_pose_selection(self):
        decision = self.observe(yaw_rad=math.radians(2.),
                                plausible_yaws_rad=(math.radians(2.), math.radians(25.)))
        self.assertFalse(decision.current_sample_accepted)
        self.assertEqual(decision.reason, "head_axis_choice_unstable")
        self.assertAlmostEqual(math.degrees(decision.angle_interval.span_rad), 23.)
        self.assertAlmostEqual(decision.chosen_angle_interval.span_rad, 0.)
        self.assertEqual(decision.sample_count, 1)
        metadata = decision.metadata()
        self.assertEqual(metadata["acceptance_semantics"],
                         "temporal_compatibility_only_independent_pose_quality_required")
        self.assertFalse(metadata["measurement_reused"])
        self.assertFalse(metadata["angle_interval"]["planning_authorized"])

    def test_gradual_drift_checks_full_window_not_only_adjacent_frames(self):
        for n, degrees in enumerate((0., 3., 6., 9.)):
            decision = self.observe(100.+n*.1, yaw_rad=math.radians(degrees))
        self.assertFalse(decision.current_sample_accepted)
        self.assertAlmostEqual(math.degrees(decision.angle_interval.span_rad), 9.)

    def test_changed_target_model_camera_or_epoch_clears_prior_axis_context(self):
        for change in (dict(target_key="candidate_002"), dict(model_sha256="other-model"),
                       dict(camera_signature=(600., 600., 320., 240.)), dict(motion_epoch=1)):
            with self.subTest(change=change):
                self.guard.reset()
                for n in range(6):
                    self.observe(100.+n*.1)
                decision = self.observe(101., context=replace(self.context, **change))
                self.assertTrue(decision.context_reset)
                self.assertTrue(decision.reset_axis_evidence)
                self.assertEqual(decision.sample_count, 1)

    def test_duplicate_and_out_of_order_stamps_cannot_fill_window_or_replace_outlier(self):
        self.observe()
        for stamp in (100., 99.):
            decision = self.observe(stamp, frame=FRAME26)
            self.assertFalse(decision.current_sample_accepted)
            self.assertEqual(decision.reason, "duplicate_or_out_of_order_head_measurement")
            self.assertEqual(decision.window_stamps_sec, (100.,))
        decision = self.observe(100.1)
        self.assertTrue(decision.current_sample_accepted)
        self.assertEqual(decision.sample_count, 2)

    def test_original_stamp_expiry_permits_new_current_choice_without_backfill(self):
        self.observe()
        self.observe(100.1, frame=FRAME26)
        decision = self.observe(105.2)
        self.assertTrue(decision.current_sample_accepted)
        self.assertFalse(decision.ready)
        self.assertEqual(decision.expired_sample_count, 2)
        self.assertEqual(decision.window_stamps_sec, (105.2,))

    def test_invalid_current_measurement_clears_history_and_raises(self):
        for invalid in (
            {"stamp_sec": float("nan")}, {"yaw_rad": float("inf")},
            {"corners_full_image": ((1., 2.),)}, {"expected_head_height_px": 0.},
            {"corners_full_image": ((1e300, 1e300),)*4},
            {"plausible_yaws_rad": (0.,)*9}, {"projected_center_px": None},
            {"yaw_rad": True}, {"stamp_sec": -1.}, {"context": None},
        ):
            with self.subTest(invalid=invalid):
                self.guard.reset()
                self.observe()
                payload = {"context": self.context, **FRAME16, **invalid}
                with self.assertRaises(ValueError):
                    self.guard.observe(**payload)
                decision = self.observe(101.)
                self.assertEqual(decision.sample_count, 1)
                self.assertTrue(decision.context_reset)
                json.dumps(decision.metadata(), allow_nan=False)

    def test_configuration_and_context_validation_is_bounded(self):
        for invalid in (dict(required_samples=1), dict(required_samples=True), dict(required_samples=33),
                       dict(max_axis_span_rad=0.), dict(max_axis_span_rad=math.pi),
                       dict(max_corner_displacement_ratio=float("nan")),
                       dict(window_ttl_sec=61.), dict(window_ttl_sec=True)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                StationaryHeadConsistency(**invalid)
        for invalid in (dict(target_key=""), dict(model_sha256=""), dict(camera_signature=[]),
                       dict(camera_signature=(float("nan"),)), dict(camera_signature=([],)),
                       dict(motion_epoch=True), dict(motion_epoch=-1)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                replace(self.context, **invalid)

    def test_memory_and_diagnostics_remain_bounded_without_authority(self):
        for n in range(100):
            decision = self.observe(100.+n*.01)
        payload = decision.metadata()
        self.assertEqual(len(payload["window_stamps_sec"]), 7)
        self.assertTrue(decision.ready)
        for key in ("motion_authorized", "completion_authorized", "planning_authorized"):
            self.assertFalse(payload[key])
        self.assertNotIn("yaw_rad", payload)  # No replacement pose is returned.
        json.dumps(payload, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
