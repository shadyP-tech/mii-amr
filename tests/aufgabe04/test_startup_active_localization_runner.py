from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.localization.startup_active_localization import (
    StartupActiveLocalizationMotionResult,
)
from scripts.aufgabe04.navigation.localization import (
    startup_active_localization_runner,
)
from scripts.aufgabe04.navigation.missions.startup_route_uncertainty_selection import (
    STARTUP_ROUTE_UNCERTAINTY_SELECTION_HASH_FIELD,
)


main = startup_active_localization_runner.main


def _arguments(root: Path) -> list[str]:
    selection = root / "selection.json"
    digest = write_content_hashed_json(
        selection,
        {
            "schema_version": 1,
            "phase": "precheckpoint_initial_coverage_route_selection",
            "motion_authorized": False,
            "motion_published": False,
            "target_committed_before_selection": False,
            "retargeting_allowed_after_selection": False,
            "selection": {
                "decision": {
                    "ready": False,
                    "selected_option_id": None,
                }
            },
        },
        hash_field=STARTUP_ROUTE_UNCERTAINTY_SELECTION_HASH_FIELD,
    )
    return [
        "--run-id",
        "active_localization_000",
        "--attempt-index",
        "0",
        "--max-attempts",
        "1",
        "--rotation-rad",
        "0.35",
        "--angular-speed-radps",
        "0.12",
        "--maximum-angular-speed-radps",
        "0.20",
        "--timeout-sec",
        "8.0",
        "--source-route-selection-json",
        str(selection),
        "--source-route-selection-sha256",
        digest,
        "--result-json",
        str(root / "result.json"),
        "--controller-trace-jsonl",
        str(root / "controller_trace.jsonl"),
        "--semantic-log",
        str(root / "events.jsonl"),
    ]


class StartupActiveLocalizationRunnerTest(unittest.TestCase):
    def test_zero_based_attempt_reaches_motion_edge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = StartupActiveLocalizationMotionResult(
                status="completed",
                stop_reason="",
                duration_sec=3.0,
                requested_rotation_rad=0.35,
                accumulated_progress_rad=0.31,
                accumulated_reverse_rad=0.0,
                maximum_translation_m=0.001,
                motion_published=True,
                zero_command_count=20,
                stop_details={"stationary_odom": {"accepted": True}},
            )
            with patch(
                "scripts.aufgabe04.navigation.localization."
                "startup_active_localization_runner._confirm_localize"
            ), patch(
                "scripts.aufgabe04.navigation.localization."
                "startup_active_localization_runner._run_preflight",
                return_value=(root / "preflight.json", "b" * 64),
            ), patch(
                "scripts.aufgabe04.navigation.localization."
                "startup_active_localization_runner."
                "run_startup_active_localization_motion",
                return_value=result,
            ) as motion:
                status = main(_arguments(root))

        self.assertEqual(status, 0)
        self.assertEqual(motion.call_args.kwargs["attempt_index"], 0)

    def test_missing_localize_confirmation_never_calls_motion_edge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                "scripts.aufgabe04.navigation.localization."
                "startup_active_localization_runner._run_preflight",
                return_value=(root / "preflight.json", "b" * 64),
            ), patch(
                "scripts.aufgabe04.navigation.localization."
                "startup_active_localization_runner._confirm_localize",
                side_effect=RuntimeError("operator did not authorize"),
            ), patch(
                "scripts.aufgabe04.navigation.localization."
                "startup_active_localization_runner."
                "run_startup_active_localization_motion",
            ) as motion, patch("builtins.print"):
                status = main(_arguments(root))

        self.assertEqual(status, 1)
        motion.assert_not_called()

    def test_failed_ros_preflight_precedes_prompt_and_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                "scripts.aufgabe04.navigation.localization."
                "startup_active_localization_runner._run_preflight",
                side_effect=RuntimeError("ROS preflight rejected"),
            ), patch(
                "scripts.aufgabe04.navigation.localization."
                "startup_active_localization_runner._confirm_localize",
            ) as confirm, patch(
                "scripts.aufgabe04.navigation.localization."
                "startup_active_localization_runner."
                "run_startup_active_localization_motion",
            ) as motion, patch("builtins.print"):
                status = main(_arguments(root))

        self.assertEqual(status, 1)
        confirm.assert_not_called()
        motion.assert_not_called()

    def test_existing_attempt_output_stops_before_preflight_or_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arguments = _arguments(root)
            (root / "result.json").write_text("stale\n", encoding="utf-8")
            with patch(
                "scripts.aufgabe04.navigation.localization."
                "startup_active_localization_runner._run_preflight",
            ) as preflight, patch(
                "scripts.aufgabe04.navigation.localization."
                "startup_active_localization_runner._confirm_localize",
            ) as confirm:
                with redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        main(arguments)

        self.assertEqual(raised.exception.code, 2)
        preflight.assert_not_called()
        confirm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
