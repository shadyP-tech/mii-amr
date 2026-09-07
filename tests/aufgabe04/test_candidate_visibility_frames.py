"""Regression tests for cross-epoch candidate rejection and missing evidence."""

import math
import unittest
from dataclasses import replace

from scripts.aufgabe04.navigation.coverage.candidate_visibility_frames import project_candidate_for_visibility
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from tests.aufgabe04.test_coverage_candidate_reconciliation import (
    TARGET_INDEX, _candidate, _decision, _receipt, _ranges,
)


def _epoch_receipt(index, *, shift_y=0.04, ranges=None):
    receipt = _receipt(f"receipt_{index}", float(index), 1.5)
    pose = receipt.scan_pose_map
    return replace(
        receipt,
        frame_provenance=replace(
            receipt.frame_provenance,
            map_from_odom=PlanarTransform2D(0.0, shift_y, 0.0),
            canonical_scan_pose_odom=Pose2D(pose.x_m, pose.y_m - shift_y, pose.yaw_rad),
        ),
        ranges_m=receipt.ranges_m if ranges is None else tuple(ranges),
    )


class CandidateVisibilityFramesTest(unittest.TestCase):
    def test_canonical_point_is_projected_into_receipt_epoch(self):
        candidate = _candidate()
        receipt = _epoch_receipt(1)
        projected = project_candidate_for_visibility(candidate, receipt)
        self.assertIsNone(projected.reason)
        self.assertAlmostEqual(projected.target.x_m, candidate.x_m)
        self.assertAlmostEqual(projected.target.y_m, candidate.y_m + 0.04)
        self.assertAlmostEqual(projected.displacement_m, 0.04)
        self.assertNotEqual(projected.candidate_source_evidence_id, projected.receipt_source_evidence_id)

    def test_cross_epoch_matching_return_retains_candidate(self):
        ranges = _ranges(1.5)
        ranges[TARGET_INDEX + 3] = math.hypot(0.8, 0.04)
        decision = _decision(receipts=tuple(_epoch_receipt(i, ranges=ranges) for i in range(1, 4)))
        self.assertFalse(decision.reject_provisional)
        self.assertIn("matching_return_supports_candidate", decision.reasons)
        for evidence in decision.ray_evidence:
            self.assertEqual(evidence.selected_ray_index, TARGET_INDEX + 3)
            self.assertIn(TARGET_INDEX + 3, evidence.supporting_ray_indices)
            self.assertAlmostEqual(evidence.frame_projection.displacement_m, 0.04)

    def test_unrelated_clear_scan_still_rejects_after_reprojection(self):
        decision = _decision(receipts=tuple(_epoch_receipt(i) for i in range(1, 4)))
        self.assertTrue(decision.reject_provisional)

    def test_one_legacy_receipt_vetoes_clear_majority(self):
        receipts = tuple(_receipt(f"receipt_{i}", float(i), 1.5) for i in range(1, 5))
        legacy = replace(receipts[-1], schema_version=1, frame_provenance=None)
        decision = _decision(receipts=(*receipts[:-1], legacy))
        self.assertFalse(decision.reject_provisional)
        self.assertIn("visibility_receipt_frame_provenance_missing", decision.reasons)
        self.assertIn("candidate_visibility_frame_unavailable", decision.reasons)

    def test_missing_candidate_provenance_never_supports_negative_rejection(self):
        decision = _decision(
            candidate=_candidate(frame_provenance=None),
            receipts=tuple(_receipt(f"receipt_{i}", float(i), 1.5) for i in range(1, 4)),
        )
        self.assertFalse(decision.reject_provisional)
        self.assertIn("candidate_frame_provenance_missing", decision.reasons)
        self.assertTrue(all(e.candidate_distance_m is None for e in decision.ray_evidence))

    def test_stale_candidate_coordinates_disagreeing_with_provenance_retain(self):
        candidate = replace(_candidate(), x_m=0.88)
        decision = _decision(candidate=candidate, receipts=tuple(_receipt(f"receipt_{i}", float(i), 1.5) for i in range(1, 4)))
        self.assertFalse(decision.reject_provisional)
        self.assertIn("candidate_frozen_map_point_mismatch", decision.reasons)

    def test_map_or_odom_identity_mismatch_retains(self):
        receipt = _receipt("receipt_1", 1.0, 1.5)
        for field, value in (("map_frame", "another_map"), ("odom_frame", "other_robot_odom")):
            with self.subTest(field=field):
                candidate = _candidate()
                candidate = replace(candidate, frame_provenance=replace(candidate.frame_provenance, **{field: value}))
                projected = project_candidate_for_visibility(candidate, receipt)
                self.assertIsNone(projected.target)
                self.assertEqual(projected.reason, "candidate_visibility_frame_identity_mismatch")

    def test_same_certificate_cannot_name_two_transforms(self):
        candidate = _candidate()
        receipt = _epoch_receipt(1)
        receipt = replace(receipt, frame_provenance=replace(receipt.frame_provenance, source_evidence_id=candidate.frame_provenance.source_evidence_id))
        projected = project_candidate_for_visibility(candidate, receipt)
        self.assertEqual(projected.reason, "visibility_frame_certificate_transform_mismatch")

    def test_projected_target_outside_planned_visible_cell_retains(self):
        decision = _decision(receipts=tuple(_epoch_receipt(i, shift_y=0.2) for i in range(1, 4)))
        self.assertFalse(decision.reject_provisional)
        self.assertIn("projected_candidate_not_planned_visible", decision.reasons)


if __name__ == "__main__":
    unittest.main()
