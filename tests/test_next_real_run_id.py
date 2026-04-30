import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import next_real_run_id  # noqa: E402


class NextRealRunIdTest(unittest.TestCase):
    def test_no_existing_runs_starts_at_one(self):
        self.assertEqual(next_real_run_id.next_run_id([]), "run_real_001")

    def test_existing_result_rows_increment_to_next_padded_id(self):
        run_ids = ["run_real_06", "run_real_07", "run_real_10"]
        self.assertEqual(next_real_run_id.next_run_id(run_ids), "run_real_011")

    def test_bag_directory_ids_are_included(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results_csv = root / "real_scripted_drive_runs.csv"
            bags_dir = root / "bags"
            bags_dir.mkdir()
            (bags_dir / "run_real_012").mkdir()

            with results_csv.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "run_id"])
                writer.writerow(["2026-04-30T13:00:00", "run_real_011"])

            run_ids = next_real_run_id.collect_existing_run_ids(
                results_csv=results_csv,
                bags_dir=bags_dir,
            )

        self.assertEqual(next_real_run_id.next_run_id(run_ids), "run_real_013")

    def test_custom_names_are_ignored_for_numbering(self):
        run_ids = ["run_real_003", "run_real_calibration_test", "run_fake_099"]
        self.assertEqual(next_real_run_id.next_run_id(run_ids), "run_real_004")

    def test_custom_prefix_and_width(self):
        run_ids = ["trial_0009"]
        self.assertEqual(
            next_real_run_id.next_run_id(run_ids, prefix="trial_", min_width=4),
            "trial_0010",
        )


if __name__ == "__main__":
    unittest.main()
