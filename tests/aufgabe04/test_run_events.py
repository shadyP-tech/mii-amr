from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
import csv
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation import run_single_station_segment  # noqa: E402
from scripts.aufgabe04.navigation.follower_models import FollowerResult  # noqa: E402
from scripts.aufgabe04.navigation.ros_preflight import (  # noqa: E402
    RosObservation,
    RosPreflightResult,
)
from scripts.aufgabe04.navigation.run_events import (  # noqa: E402
    build_event,
    configure_event_logger,
    emit_event,
    event_to_json,
)


ROUTE_HEADER = (
    "leg_index,point_index,grid_x,grid_y,world_x_m,world_y_m,"
    "segment_length_m,cumulative_length_m\n"
)


def write_route(path: Path) -> None:
    path.write_text(
        ROUTE_HEADER
        + "\n".join(
            [
                "0,0,0,0,0.0,0.0,0.0,0.0",
                "0,1,1,0,0.2,0.0,0.2,0.2",
            ]
        )
        + "\n"
    )


def write_diagnostics(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "legs": [
                    {
                        "diagnostics": {"status": "ok", "route_length_m": 0.2},
                        "failure": None,
                        "route_length_m": 0.2,
                        "route_point_count": 2,
                    }
                ]
            }
        )
    )


def write_failing_diagnostics(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "legs": [
                    {
                        "diagnostics": {"status": "failed", "route_length_m": 0.2},
                        "failure": {"reason": "blocked"},
                        "route_length_m": 0.2,
                        "route_point_count": 2,
                    }
                ]
            }
        )
    )


def read_events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_result_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def passing_preflight() -> RosPreflightResult:
    return RosPreflightResult(
        ok=True,
        failures=[],
        observations=[
            RosObservation(
                "cmd_vel ownership",
                True,
                "publishers=[]",
                {
                    "cmd_vel_topic": "/cmd_vel",
                    "publishers": [],
                    "allowed_publishers": [],
                },
            ),
            RosObservation(
                "scan freshness",
                True,
                "receipt_age=0.100s header_age=0.100s",
                {"receipt_age_sec": 0.1, "header_age_sec": 0.1},
            ),
        ],
        runtime_config={"cmd_vel_topic": "/cmd_vel", "scan_topic": "/scan"},
    )


def failing_preflight() -> RosPreflightResult:
    return RosPreflightResult(
        ok=False,
        failures=["unapproved cmd_vel publishers: /teleop_keyboard"],
        observations=[
            RosObservation(
                "cmd_vel ownership",
                False,
                "publishers=['/teleop_keyboard']",
                {"cmd_vel_topic": "/cmd_vel", "publishers": ["/teleop_keyboard"]},
            )
        ],
        runtime_config={"cmd_vel_topic": "/cmd_vel", "scan_topic": "/scan"},
    )


class RunEventsTest(unittest.TestCase):
    def test_runner_rejects_nav2_direct_publisher_allowlist(self):
        with self.assertRaises(SystemExit) as raised, redirect_stdout(StringIO()):
            run_single_station_segment.main(
                [
                    "--leg-index",
                    "0",
                    "--allowed-cmd-vel-publisher",
                    "/behavior_server",
                ]
            )

        self.assertEqual(raised.exception.code, 2)

    def test_event_json_is_deterministic_and_contains_core_fields(self):
        event = build_event(
            "runtime_resolved",
            run_id="run-1",
            leg_index=0,
            resolved_cmd_vel_topic="/robot1/cmd_vel",
            map_frame="map",
            base_frame="base_footprint",
        )

        encoded = event_to_json(event)

        self.assertEqual(json.loads(encoded)["event"], "runtime_resolved")
        self.assertIn('"base_frame":"base_footprint"', encoded)
        self.assertIn('"resolved_cmd_vel_topic":"/robot1/cmd_vel"', encoded)

    def test_file_logger_writes_json_line_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "events.jsonl"
            logger = configure_event_logger(log_path)

            emit_event(logger, "run_finished", run_id="run-1", final_status="dry_run_ok")

            events = read_events(log_path)

        self.assertEqual(events[0]["event"], "run_finished")
        self.assertEqual(events[0]["final_status"], "dry_run_ok")
        event_logger = logging.getLogger("aufgabe04.navigation.run_events")
        for handler in event_logger.handlers:
            handler.close()
        event_logger.handlers.clear()


class RunSingleStationSegmentEventsTest(unittest.TestCase):
    def make_paths(self, tmpdir: Path) -> dict[str, Path]:
        paths = {
            "route": tmpdir / "route.csv",
            "diagnostics": tmpdir / "diagnostics.json",
            "results": tmpdir / "station_segment_runs.csv",
            "events": tmpdir / "events.jsonl",
            "preflight": tmpdir / "preflight.json",
        }
        write_route(paths["route"])
        write_diagnostics(paths["diagnostics"])
        return paths

    def base_args(self, paths: dict[str, Path]) -> list[str]:
        return [
            "--route-csv",
            str(paths["route"]),
            "--diagnostics-json",
            str(paths["diagnostics"]),
            "--results-csv",
            str(paths["results"]),
            "--semantic-log",
            str(paths["events"]),
            "--preflight-json",
            str(paths["preflight"]),
            "--run-id",
            "run-1",
            "--leg-index",
            "0",
        ]

    def test_preflight_failure_logs_event_and_skips_follower(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            with patch.object(
                run_single_station_segment,
                "run_ros_preflight",
                return_value=failing_preflight(),
            ), patch.object(run_single_station_segment, "run_simple_waypoint_follower") as follower, redirect_stdout(
                StringIO()
            ):
                status = run_single_station_segment.main(self.base_args(paths))

            events = read_events(paths["events"])
            results = paths["results"].read_text()
            finish_events = [event for event in events if event["event"] == "run_finished"]

        self.assertEqual(status, 1)
        self.assertFalse(follower.called)
        self.assertIn("preflight_failed", [event["event"] for event in events])
        self.assertEqual(len(finish_events), 1)
        self.assertIn("unapproved cmd_vel publishers", results)

    def test_dry_run_logs_no_motion_and_skips_follower(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            args = self.base_args(paths) + ["--dry-run"]
            with patch.object(
                run_single_station_segment,
                "run_ros_preflight",
                return_value=passing_preflight(),
            ) as preflight, patch.object(run_single_station_segment, "run_simple_waypoint_follower") as follower, redirect_stdout(
                StringIO()
            ):
                status = run_single_station_segment.main(args)

            events = read_events(paths["events"])
            event_names = [event["event"] for event in events]
            dry_run_event = next(event for event in events if event["event"] == "dry_run_completed")
            finish_events = [event for event in events if event["event"] == "run_finished"]
            rows = read_result_rows(paths["results"])

        self.assertEqual(status, 0)
        self.assertTrue(preflight.called)
        self.assertFalse(follower.called)
        self.assertIn("preflight_passed", event_names)
        self.assertEqual(len(finish_events), 1)
        self.assertEqual(rows[-1]["status"], "dry_run_ok")
        self.assertFalse(dry_run_event["motion_published"])

    def test_initialpose_prompt_runs_before_preflight_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            args = self.base_args(paths) + ["--dry-run", "--prompt-for-initialpose"]
            prompts = []
            with patch.object(
                run_single_station_segment,
                "run_ros_preflight",
                return_value=passing_preflight(),
            ) as preflight, patch(
                "builtins.input",
                side_effect=lambda prompt="": prompts.append(prompt) or "",
            ), redirect_stdout(
                StringIO()
            ):
                status = run_single_station_segment.main(args)

        self.assertEqual(status, 0)
        self.assertTrue(preflight.called)
        self.assertEqual(prompts, ["Press Enter, then click 2D Pose Estimate immediately: "])

    def test_operator_abort_logs_no_motion_and_skips_follower(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            with patch.object(
                run_single_station_segment,
                "run_ros_preflight",
                return_value=passing_preflight(),
            ), patch.object(run_single_station_segment, "run_simple_waypoint_follower") as follower, patch(
                "builtins.input",
                return_value="",
            ), redirect_stdout(
                StringIO()
            ):
                status = run_single_station_segment.main(self.base_args(paths))

            events = read_events(paths["events"])
            abort_event = next(event for event in events if event["event"] == "operator_aborted")
            finish_events = [event for event in events if event["event"] == "run_finished"]
            rows = read_result_rows(paths["results"])

        self.assertEqual(status, 1)
        self.assertFalse(follower.called)
        self.assertEqual(len(finish_events), 1)
        self.assertEqual(rows[-1]["status"], "aborted")
        self.assertFalse(abort_event["motion_published"])

    def test_real_run_passes_initial_sensor_wait_to_follower_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            args = self.base_args(paths) + [
                "--initial-sensor-wait-sec",
                "3.5",
                "--allowed-cmd-vel-publisher",
                "/behavior_server",
            ]
            with patch.object(
                run_single_station_segment,
                "run_ros_preflight",
                return_value=passing_preflight(),
            ), patch.object(
                run_single_station_segment,
                "run_simple_waypoint_follower",
                return_value=FollowerResult("completed", "", 1.0, 0.2, True),
            ) as follower, patch(
                "builtins.input",
                return_value="RUN",
            ), redirect_stdout(
                StringIO()
            ):
                status = run_single_station_segment.main(args)

            events = read_events(paths["events"])
            rows = read_result_rows(paths["results"])
            follower_config = follower.call_args.args[2]

        self.assertEqual(status, 0)
        self.assertEqual(follower_config.initial_sensor_wait_sec, 3.5)
        self.assertEqual(follower_config.allowed_cmd_vel_publishers, ("/behavior_server",))
        self.assertIn("motion_started", [event["event"] for event in events])
        self.assertEqual(rows[-1]["status"], "completed")

    def test_route_diagnostics_failure_writes_result_row_and_one_terminal_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            write_failing_diagnostics(paths["diagnostics"])
            with patch.object(run_single_station_segment, "run_ros_preflight") as preflight, patch.object(
                run_single_station_segment,
                "run_simple_waypoint_follower",
            ) as follower, redirect_stdout(StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    run_single_station_segment.main(self.base_args(paths))

            events = read_events(paths["events"])
            finish_events = [event for event in events if event["event"] == "run_finished"]
            rows = read_result_rows(paths["results"])

        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(preflight.called)
        self.assertFalse(follower.called)
        self.assertEqual(len(finish_events), 1)
        self.assertEqual(finish_events[0]["final_status"], "route_validation_failed")
        self.assertEqual(rows[-1]["status"], "route_validation_failed")
        self.assertIn("diagnostics leg 0", rows[-1]["stop_reason"])

    def test_preflight_unavailable_writes_result_row_and_one_terminal_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            with patch.object(
                run_single_station_segment,
                "run_ros_preflight",
                side_effect=RuntimeError("ROS2 Python packages are not available"),
            ), patch.object(run_single_station_segment, "run_simple_waypoint_follower") as follower, redirect_stdout(
                StringIO()
            ):
                with self.assertRaises(SystemExit) as raised:
                    run_single_station_segment.main(self.base_args(paths))

            events = read_events(paths["events"])
            finish_events = [event for event in events if event["event"] == "run_finished"]
            rows = read_result_rows(paths["results"])

        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(follower.called)
        self.assertEqual(len(finish_events), 1)
        self.assertEqual(finish_events[0]["final_status"], "preflight_unavailable")
        self.assertEqual(rows[-1]["status"], "preflight_unavailable")
        self.assertIn("ROS2 Python packages", rows[-1]["stop_reason"])


if __name__ == "__main__":
    unittest.main()
