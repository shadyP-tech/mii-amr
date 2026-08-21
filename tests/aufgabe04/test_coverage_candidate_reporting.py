from __future__ import annotations

import unittest

from scripts.aufgabe04.navigation.coverage_candidate_reporting import (
    active_lidar_registry_count_fields,
    coverage_epoch_candidate_count_fields,
    coverage_phase_completion_fields,
    fused_registry_candidate_count_fields,
)


class CoverageCandidateReportingTest(unittest.TestCase):
    def test_epoch_five_and_cumulative_fused_six_remain_distinct(self):
        fields = coverage_epoch_candidate_count_fields(
            confirmed_epoch_candidate_count=5,
            static_map_admitted_candidate_count=5,
            static_map_rejected_candidate_count=0,
            fused_registry_candidate_counts={
                "confirmed": 0,
                "pending_camera": 2,
                "provisional": 4,
                "rejected": 0,
            },
        )

        self.assertEqual(fields["epoch_static_map_admitted_candidate_count"], 5)
        self.assertEqual(fields["epoch_static_map_rejected_candidate_count"], 0)
        self.assertEqual(fields["fused_registry_active_candidate_count"], 6)
        self.assertEqual(fields["fused_registry_total_candidate_count"], 6)
        self.assertEqual(fields["static_map_candidate_admitted_count"], 5)
        self.assertEqual(
            fields["legacy_epoch_candidate_count_aliases"][
                "static_map_candidate_admitted_count"
            ],
            "epoch_static_map_admitted_candidate_count",
        )

    def test_rejected_registry_candidates_are_not_active(self):
        fields = fused_registry_candidate_count_fields(
            {"confirmed": 1, "provisional": 5, "rejected": 2}
        )

        self.assertEqual(fields["fused_registry_total_candidate_count"], 8)
        self.assertEqual(fields["fused_registry_active_candidate_count"], 6)
        self.assertEqual(fields["candidate_count"], 8)
        self.assertEqual(
            fields["legacy_fused_registry_candidate_count_aliases"][
                "candidate_count"
            ],
            "fused_registry_total_candidate_count",
        )

    def test_exact_two_count_uses_registry_name_and_marks_legacy_alias(self):
        fields = active_lidar_registry_count_fields(6)

        self.assertEqual(fields["active_lidar_registry_candidate_count"], 6)
        self.assertEqual(fields["fused_registry_active_candidate_count"], 6)
        self.assertEqual(fields["lidar_static_map_admitted_candidate_count"], 6)
        self.assertEqual(
            fields["legacy_lidar_checkpoint_candidate_count_aliases"][
                "lidar_static_map_admitted_candidate_count"
            ],
            "active_lidar_registry_candidate_count",
        )

    def test_epoch_partition_must_match_confirmed_count(self):
        with self.assertRaisesRegex(
            ValueError,
            "must equal admitted plus rejected",
        ):
            coverage_epoch_candidate_count_fields(
                confirmed_epoch_candidate_count=5,
                static_map_admitted_candidate_count=4,
                static_map_rejected_candidate_count=0,
                fused_registry_candidate_counts={"provisional": 4},
            )

    def test_phase_completion_names_remove_lidar_camera_contradiction(self):
        fields = coverage_phase_completion_fields(
            lidar_coverage_complete=True,
            camera_candidate_resolution_complete=False,
            camera_expected_stand_count_met=False,
        )

        self.assertEqual(
            fields["completion_phase"],
            "lidar_complete_camera_validation_pending",
        )
        self.assertTrue(fields["lidar_coverage_complete"])
        self.assertFalse(fields["camera_exploration_complete"])
        self.assertFalse(fields["exploration_complete"])
        self.assertEqual(
            fields["legacy_completion_aliases"],
            {"exploration_complete": "camera_exploration_complete"},
        )


if __name__ == "__main__":
    unittest.main()
