import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "aufgabe03"))

import analyze_waypoint_follow_runs as analyzer  # noqa: E402


def current_schema_row(**overrides):
    row = {
        "timestamp": "2026-05-27T13:48:13",
        "run_id": "handoff_path_gate_test_006",
        "status": "completed",
        "notes": "arena_prior_two_stage_run;two_stage_handoff",
        "final_status_reason": "",
        "last_replan_reason": "run_local_replan_completed",
        "run_local_no_path_reason": "",
        "replan_count": "2",
        "run_local_replan_count": "4",
        "min_scan_range_m": "0.35",
        "p05_scan_range_m": "0.97",
        "run_local_map_yaml": "results/aufgabe03/handoff_path_gate_test_006_run_local_map.yaml",
        "run_local_waypoints_csv": "results/aufgabe03/handoff_path_gate_test_006_run_local_waypoints.csv",
        "updated_map_yaml": "",
        "updated_waypoints_csv": "",
    }
    row.update(overrides)
    return row


class AnalyzeWaypointFollowRunsTest(unittest.TestCase):
    def test_completed_current_schema_row_passes(self):
        verdict = analyzer.classify_run(current_schema_row(), max_replans=2)

        self.assertEqual(verdict.verdict, "PASS")

    def test_failed_lidar_replan_max_exceeded_fails(self):
        verdict = analyzer.classify_run(
            current_schema_row(
                status="failed",
                notes=(
                    "arena_prior_two_stage_run;two_stage_handoff;"
                    "lidar_replan_failed:max_replans_exceeded"
                ),
                last_replan_reason="max_replans_exceeded",
            ),
            max_replans=2,
        )

        self.assertEqual(verdict.verdict, "FAIL")
        self.assertIn("lidar_replan_failed:max_replans_exceeded", verdict.reason)

    def test_interrupted_keyboard_interrupt_fails(self):
        verdict = analyzer.classify_run(
            current_schema_row(
                status="interrupted",
                notes="arena_prior_two_stage_run;two_stage_handoff;keyboard_interrupt",
            ),
            max_replans=2,
        )

        self.assertEqual(verdict.verdict, "FAIL")
        self.assertIn("keyboard_interrupt", verdict.reason)

    def test_completed_row_with_too_many_replans_warns(self):
        verdict = analyzer.classify_run(
            current_schema_row(replan_count="3"),
            max_replans=2,
        )

        self.assertEqual(verdict.verdict, "WARN")
        self.assertIn("replan_count=3 exceeds max=2", verdict.reason)

    def test_legacy_completed_row_without_replan_columns_warns(self):
        verdict = analyzer.classify_run(
            {
                "timestamp": "2026-05-18T14:56:27",
                "run_id": "waypoint_follow_slow_02",
                "status": "completed",
                "notes": "follow_planned_waypoints",
                "min_scan_range_m": "0.87",
                "p05_scan_range_m": "0.88",
            },
            max_replans=2,
        )

        self.assertEqual(verdict.verdict, "WARN")
        self.assertIn("missing diagnostic columns", verdict.reason)

    def test_markdown_report_includes_run_verdict_and_failure_reason(self):
        rows = [
            current_schema_row(
                run_id="run_fail",
                status="failed",
                notes="lidar_replan_failed:max_replans_exceeded",
            )
        ]
        summary = analyzer.summarize_runs(rows, max_replans=2)

        report = analyzer.render_markdown_report(summary, rows)

        self.assertIn("run_fail", report)
        self.assertIn("FAIL", report)
        self.assertIn("lidar_replan_failed:max_replans_exceeded", report)


if __name__ == "__main__":
    unittest.main()
