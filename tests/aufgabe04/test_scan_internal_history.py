"""Recorded internal dropout retains support without manufacturing a witness.

This is a reduced camera-capture replay, not the full independent-scan stream.
The live third frame was admitted using additional independent witnesses; this
fixture deliberately cannot admit it because only two earlier scans are loaded.
"""

import json
import math
from pathlib import Path
import unittest

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception.candidate_lidar_association import (
    associate_camera_registered_candidate_lidar_target,
)
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.real_robot.observer.scan_target_persistence import (
    ScanPersistenceContext,
    StoppedScanTargetPersistence,
)


FIXTURE = Path(__file__).parent / "fixtures/scan_internal_history_20260923.json"


class RecordedInternalHistoryTest(unittest.TestCase):
    def test_two_recorded_witnesses_survive_dropout_without_admitting_it(self):
        fixture = json.loads(FIXTURE.read_text())
        state = StoppedScanTargetPersistence()
        witness_stamps = []
        for index, row in enumerate(fixture["observations"]):
            scan = PlainLaserScan(**{
                **row["scan"],
                "ranges": tuple(math.nan if v is None else v for v in row["scan"]["ranges"]),
            })
            context = ScanPersistenceContext(**{
                **row["context"],
                **{key: Pose2D(**row["context"][key])
                   for key in ("robot_pose", "scan_pose_map", "scan_pose_robot")},
            })
            raw = associate_camera_registered_candidate_lidar_target(
                scan, **row["parameters"], now_sec=row["now_sec"],
                max_scan_age_sec=row["max_scan_age_sec"],
            )
            resolved = state.resolve(
                raw, scan, context=context, now_sec=row["now_sec"],
                max_scan_age_sec=row["max_scan_age_sec"],
            )
            if index == 2:
                self.assertFalse(raw.associated)
                self.assertEqual(raw.search_association.eligible_cluster_count, 2)
                self.assertTrue(math.isnan(scan.ranges[1]))
                self.assertEqual(scan.ranges[0], 0.5580000281333923)
                self.assertEqual(scan.ranges[2], 0.5630000233650208)
                self.assertFalse(resolved.associated)
                self.assertIsNone(resolved.witnessed_fragmentation)
                self.assertEqual(state.last_metadata["witness_scan_count"], 2)
                self.assertNotIn("history_reset_reason", state.last_metadata)
            else:
                self.assertTrue(raw.associated)
                self.assertTrue(resolved.associated, state.last_metadata)
                witness_stamps.append(scan.scan_stamp_sec)
            self.assertEqual(
                [entry["scan"]["scan_stamp_sec"] for entry in state._history],
                witness_stamps,
            )

        # Capture four supplies the third real contiguous scan; the fragmented
        # capture is retained only as a compatible observation, never a witness.
        self.assertEqual(len(witness_stamps), 3)
        self.assertNotIn(fixture["observations"][2]["scan"]["scan_stamp_sec"], witness_stamps)


if __name__ == "__main__":
    unittest.main()
