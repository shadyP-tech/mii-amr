"""A current bounded head prioritizes a small search, without pose authority."""

import math
import unittest

from scripts.aufgabe04.real_robot.observer.inspection_progress import (
    InspectionProgress, classify_inspection_progress,
)
from scripts.aufgabe04.real_robot.candidate.inspection_policy import candidate_view_options


def details():
    return {"stand_axis_debug": {
        "metric_model": {"observation_confidence": {"backside": {"state": "backside_supported"}}},
        "bounded_head_view_hint": {"camera_relative_yaw_rad": math.radians(-9),
                                   "orientation_half_width_rad": math.radians(15),
                                   "purpose": "orientation_disambiguation",
                                   "candidate_associated": True, "source_fresh": True},
    }}


class BoundedHeadViewAdjustmentTest(unittest.TestCase):
    def test_backside_retains_only_current_bound_advisory(self):
        value = classify_inspection_progress("evidence_not_committable", details())
        self.assertEqual(value.classification, "backside_unresolved")
        self.assertEqual(value.reason, "bounded_head_orientation_disambiguation")
        self.assertAlmostEqual(value.camera_relative_yaw_rad, math.radians(-9))
        options = candidate_view_options(.3, classification=value.classification,
                                         achieved_normals=[.3], attempted_normals=[],
                                         advisory_yaw_rad=value.camera_relative_yaw_rad)
        self.assertAlmostEqual(options[0] - .3, math.radians(20))
        self.assertAlmostEqual(options[1] - .3, math.radians(-20))

    def test_stale_unbound_and_malformed_hints_do_not_supply_an_angle(self):
        for overrides in ({"source_fresh": False}, {"candidate_associated": False},
                          {"purpose": "opposite_side"}, {"source_fresh": None},
                          {"camera_relative_yaw_rad": math.nan}, {"camera_relative_yaw_rad": True},
                          {"orientation_half_width_rad": math.pi / 2},
                          {"orientation_half_width_rad": -1}):
            with self.subTest(overrides=overrides):
                payload = details()
                payload["stand_axis_debug"]["bounded_head_view_hint"].update(overrides)
                value = classify_inspection_progress("evidence_not_committable", payload)
                self.assertIsNone(value.camera_relative_yaw_rad)
                options = candidate_view_options(0., classification=value.classification,
                                                 achieved_normals=[0.], attempted_normals=[],
                                                 advisory_yaw_rad=value.camera_relative_yaw_rad)
                self.assertAlmostEqual(options[0], math.pi / 2)

    def test_no_hint_preserves_legacy_backside_search_and_front_angle(self):
        payload = details()
        payload["stand_axis_debug"].pop("bounded_head_view_hint")
        value = classify_inspection_progress("evidence_not_committable", payload)
        self.assertIsNone(value.camera_relative_yaw_rad)
        options = candidate_view_options(0., classification="front_readable",
                                         achieved_normals=[0.], attempted_normals=[],
                                         advisory_yaw_rad=math.radians(-51))
        self.assertAlmostEqual(options[0], math.radians(51))

    def test_advisory_history_keeps_wide_uncertainty_and_no_pose_fields(self):
        progress = InspectionProgress()
        value = None
        classification = classify_inspection_progress("evidence_not_committable", details())
        for i in range(7):
            value = progress.record(frame_stamp_sec=10. + i / 3,
                                    robot_pose={"x_m": 0., "y_m": 0., "yaw_rad": 0.},
                                    frame_accepted=True, poisoned=False, classification=classification)
        self.assertEqual(value["yaw_uncertainty_rad"], math.pi / 2)
        self.assertNotIn("stand_normal_map_rad", value)
        self.assertNotIn("motion_authorized", value)

    def test_infeasible_certified_backside_route_gets_two_small_search_hypotheses(self):
        options = candidate_view_options(0., classification="certified_backside",
                                         achieved_normals=[0.], attempted_normals=[])
        self.assertAlmostEqual(options[0], math.radians(20))
        self.assertAlmostEqual(options[1], math.radians(-20))


if __name__ == "__main__":
    unittest.main()
