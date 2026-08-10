from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.aufgabe04.navigation import record_stand_coverage_stop as coverage_stop


SURVEY_ROOT = Path("survey")
MAP_YAML = Path("map.yaml")
OBSERVER_SUMMARY = Path("observer_summary.json")
VIEWPOINT_ID = "survey_vp_001"


def cli_argv() -> list[str]:
    return [
        "--survey-root",
        str(SURVEY_ROOT),
        "--map",
        str(MAP_YAML),
        "--viewpoint-id",
        VIEWPOINT_ID,
        "--observer-summary-json",
        str(OBSERVER_SUMMARY),
    ]


class RecordStandCoverageStopApiTest(unittest.TestCase):
    def test_importable_api_propagates_fusion_failure_without_system_exit(self):
        with patch.object(
            coverage_stop,
            "load_coverage_survey_plan",
            side_effect=ValueError("coverage plan is corrupt"),
        ):
            with self.assertRaisesRegex(ValueError, "coverage plan is corrupt"):
                coverage_stop.record_stand_coverage_stop(
                    survey_root=SURVEY_ROOT,
                    map_yaml=MAP_YAML,
                    viewpoint_id=VIEWPOINT_ID,
                    observer_summary_json=OBSERVER_SUMMARY,
                )

    def test_cli_translates_fusion_failure_to_exit_code_two(self):
        stderr = StringIO()
        with patch.object(
            coverage_stop,
            "load_coverage_survey_plan",
            side_effect=ValueError("coverage plan is corrupt"),
        ):
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    coverage_stop.main(cli_argv())

        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(stderr.getvalue(), "error: coverage plan is corrupt\n")

    def test_cli_prints_returned_status_and_preserves_success_code(self):
        status = {
            "schema_version": 1,
            "status": "coverage_stop_recorded",
            "motion_published": False,
        }
        stdout = StringIO()
        with patch.object(
            coverage_stop,
            "record_stand_coverage_stop",
            return_value=status,
        ) as record:
            with redirect_stdout(stdout):
                result = coverage_stop.main(cli_argv())

        self.assertEqual(result, 0)
        self.assertEqual(
            stdout.getvalue(),
            '{\n  "motion_published": false,\n  "schema_version": 1,\n'
            '  "status": "coverage_stop_recorded"\n}\n',
        )
        record.assert_called_once_with(
            survey_root=SURVEY_ROOT,
            map_yaml=MAP_YAML,
            semantic_map_id="",
            viewpoint_id=VIEWPOINT_ID,
            observer_summary_json=OBSERVER_SUMMARY,
            observations_jsonl=None,
            arrival_tolerance_m=0.18,
            scan_to_base_position_offset_m=0.05,
        )


if __name__ == "__main__":
    unittest.main()
