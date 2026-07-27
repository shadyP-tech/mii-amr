from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
import unittest
import csv
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation import run_single_station_segment  # noqa: E402
from scripts.aufgabe04.navigation.dynamic_route_handoff import (  # noqa: E402
    RouteUpdate,
    RouteUpdateKind,
)
from scripts.aufgabe04.navigation.follower_models import FollowerResult  # noqa: E402
from scripts.aufgabe04.navigation.models import Pose2D  # noqa: E402
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
from scripts.aufgabe04.navigation.route_revision_store import (  # noqa: E402
    RouteRevisionStore,
    read_committed_revision,
)


ROUTE_HEADER = (
    "leg_index,point_index,grid_x,grid_y,world_x_m,world_y_m,"
    "segment_length_m,cumulative_length_m,simulation_only,route_kind\n"
)


def write_route(path: Path) -> None:
    path.write_text(
        ROUTE_HEADER
        + "\n".join(
            [
                "0,0,0,0,0.0,0.0,0.0,0.0,true,legacy_simulation_waypoint",
                "0,1,1,0,0.2,0.0,0.2,0.2,true,legacy_simulation_waypoint",
            ]
        )
        + "\n"
    )


def write_dynamic_route_manifest(
    paths: dict[str, Path],
    *,
    published_at: float | None = None,
    route_kind: str = "synchronized_viewpoint",
) -> Path:
    route_text = (
        "leg_index,point_index,grid_x,grid_y,world_x_m,world_y_m,"
        "segment_length_m,cumulative_length_m,simulation_only,route_kind,stream_id\n"
        f"0,0,0,0,0.0,0.0,0.0,0.0,true,{route_kind},sim-stream\n"
        f"0,1,1,0,0.2,0.0,0.2,0.2,true,{route_kind},sim-stream\n"
    )
    now = time.time() if published_at is None else published_at
    manifest = paths["route"].with_suffix(".manifest.json")
    store = RouteRevisionStore(
        manifest, stream_id="sim-stream", writer_id="planner", now_fn=lambda: now
    )
    store.publish_active(
        route_text,
        json.loads(paths["diagnostics"].read_text()),
        target_revision=1,
        observation_unix_sec=now,
        source_robot_pose={"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
        target={"x_m": 0.2, "y_m": 0.0, "yaw_rad": 0.0},
        evidence={"kind": "none"},
        previous_route_length_m=0.0,
        new_route_length_m=0.2,
        safety_diagnostics={
            "corridor_clear": True,
            "start_join_clearance_m": 0.5,
            "arena_bounds": {
                "length_m": 3.9,
                "width_m": 1.898,
                "center_x_m": 0.0,
                "center_y_m": 0.0,
                "yaw_deg": 0.0,
                "margin_m": 0.0,
            },
            "arena_boundary_overlay": True,
        },
    )
    return manifest


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
    def test_simulation_motion_confirmation_does_not_block_for_input(self):
        args = type("Args", (), {"allow_sim_time": True})()
        with patch("builtins.input") as prompt, redirect_stdout(StringIO()):
            confirmed = run_single_station_segment._confirm_motion(args, object())

        self.assertTrue(confirmed)
        prompt.assert_not_called()

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
            "--allow-sim-time",
            "--allow-legacy-simulation-route",
        ]

    def test_controller_event_reports_effective_sampling_tolerances(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            manifest = write_dynamic_route_manifest(
                paths,
                route_kind="viewpoint_sampling",
            )
            args = self.base_args(paths) + [
                "--route-manifest",
                str(manifest),
                "--viewpoint-sampling-goal-tolerance-m",
                "0.018",
                "--viewpoint-sampling-heading-tolerance-rad",
                "0.08",
            ]
            with patch.object(
                run_single_station_segment,
                "run_ros_preflight",
                return_value=passing_preflight(),
            ), patch.object(
                run_single_station_segment,
                "run_simple_waypoint_follower",
                return_value=FollowerResult("completed", "", 1.0, 0.2, True),
            ), redirect_stdout(StringIO()):
                status = run_single_station_segment.main(args)

            event = next(
                item
                for item in read_events(paths["events"])
                if item["event"] == "controller_config_resolved"
            )

        self.assertEqual(status, 0)
        self.assertEqual(event["effective_goal_tolerance_m"], 0.018)
        self.assertEqual(
            event["effective_intermediate_goal_tolerance_m"],
            0.018,
        )
        self.assertEqual(event["effective_terminal_goal_tolerance_m"], 0.018)
        self.assertEqual(event["heading_tolerance_rad"], 0.08)
        self.assertEqual(
            event["intermediate_terminal_heading_entry_tolerance_m"],
            0.018,
        )
        self.assertEqual(
            event["intermediate_terminal_heading_hold_tolerance_m"],
            0.02,
        )
        self.assertEqual(
            event[
                "intermediate_terminal_heading_distance_comparison_epsilon_m"
            ],
            1.0e-5,
        )
        self.assertEqual(
            event["intermediate_terminal_heading_effective_hold_limit_m"],
            0.02001,
        )
        self.assertEqual(
            event["intermediate_terminal_heading_target_distance_m"],
            0.33,
        )
        self.assertEqual(
            event[
                "intermediate_terminal_heading_target_envelope_radius_m"
            ],
            0.03,
        )
        self.assertEqual(
            event["intermediate_terminal_heading_minimum_stand_distance_m"],
            0.31,
        )
        self.assertAlmostEqual(
            event["intermediate_terminal_heading_maximum_stand_distance_m"],
            0.35,
        )

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

    def test_rejects_simulation_only_route_without_sim_time_even_one_shot(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            manifest = write_dynamic_route_manifest(paths)

            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    args = self.base_args(paths)
                    args.remove("--allow-sim-time")
                    run_single_station_segment.main(
                        args + ["--route-manifest", str(manifest)]
                    )

        self.assertEqual(raised.exception.code, 2)

    def test_dynamic_manifest_handoff_callback_logs_route_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            manifest = write_dynamic_route_manifest(paths)
            args = self.base_args(paths) + [
                "--allow-sim-time",
                "--route-manifest",
                str(manifest),
                "--dynamic-route-refresh-sec",
                "0.1",
            ]

            def fake_follower(_resolved, _waypoints, _config, _provider, callback):
                callback(
                    RouteUpdate(
                        kind=RouteUpdateKind.ADOPT,
                        event_name="dynamic_route_adopted",
                        route_revision=2,
                        target_revision=1,
                        route_hash="abc",
                        event_fields={
                            "stream_id": "sim-stream",
                            "route_revision": 2,
                            "target_revision": 1,
                            "route_sha256": "abc",
                        },
                    )
                )
                return FollowerResult("completed", "", 1.0, 0.2, True)

            with patch.object(
                run_single_station_segment,
                "run_ros_preflight",
                return_value=passing_preflight(),
            ), patch.object(
                run_single_station_segment,
                "run_simple_waypoint_follower",
                side_effect=fake_follower,
            ), redirect_stdout(StringIO()):
                status = run_single_station_segment.main(args)

            events = read_events(paths["events"])
            reloaded = next(event for event in events if event["event"] == "route_reloaded")
            resolved = next(
                event
                for event in events
                if event["event"] == "authoritative_route_resolved"
            )

        self.assertEqual(status, 0)
        self.assertEqual(reloaded["route_revision"], 2)
        self.assertEqual(reloaded["route_sha256"], "abc")
        self.assertEqual(resolved["route_revision"], 1)
        self.assertEqual(resolved["target_revision"], 1)
        self.assertEqual(resolved["source_robot_pose"]["x_m"], 0.0)
        self.assertEqual(resolved["previous_route_length_m"], 0.0)
        self.assertEqual(resolved["new_route_length_m"], 0.2)
        self.assertEqual(len(resolved["route_sha256"]), 64)

    def test_stale_one_shot_authoritative_route_is_rejected_before_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            manifest = write_dynamic_route_manifest(
                paths, published_at=time.time() - 30.0
            )
            args = self.base_args(paths) + [
                "--allow-sim-time",
                "--route-manifest",
                str(manifest),
            ]

            with patch.object(
                run_single_station_segment, "run_ros_preflight"
            ) as preflight, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    run_single_station_segment.main(args)

            events = read_events(paths["events"])

        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(preflight.called)
        rejected = next(
            event for event in events if event["event"] == "route_manifest_rejected"
        )
        self.assertIn("age", rejected["stop_reason"])

    def test_one_shot_authoritative_route_still_uses_verified_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            manifest = write_dynamic_route_manifest(paths)
            args = self.base_args(paths) + [
                "--allow-sim-time",
                "--route-manifest",
                str(manifest),
                "--localization-source",
                "tf",
                "--map-frame",
                "odom",
                "--odom-frame",
                "odom",
                "--allow-simulation-odom-after-stale-tf",
            ]
            observed = {}

            def fake_follower(_resolved, _waypoints, config, provider, callback):
                observed["refresh_sec"] = config.dynamic_route_refresh_sec
                observed["simulation_odom_fallback"] = (
                    config.allow_simulation_odom_after_stale_tf
                )
                observed["provider"] = provider
                observed["update"] = provider(Pose2D(0.0, 0.0, 0.0))
                callback(
                    RouteUpdate(
                        kind=RouteUpdateKind.UNCHANGED,
                        event_name=(
                            "simulation_odom_pose_fallback_started"
                        ),
                        event_fields={
                            "source": (
                                "simulation_direct_odom_after_tf_retry"
                            ),
                            "not_real_robot_migration_evidence": True,
                        },
                    )
                )
                return FollowerResult("completed", "", 1.0, 0.2, True)

            with patch.object(
                run_single_station_segment,
                "run_ros_preflight",
                return_value=passing_preflight(),
            ), patch.object(
                run_single_station_segment,
                "run_simple_waypoint_follower",
                side_effect=fake_follower,
            ), redirect_stdout(StringIO()):
                status = run_single_station_segment.main(args)
            events = read_events(paths["events"])

        self.assertEqual(status, 0)
        self.assertEqual(observed["refresh_sec"], 0.0)
        self.assertTrue(observed["simulation_odom_fallback"])
        self.assertIsNotNone(observed["provider"])
        self.assertEqual(observed["update"].kind, RouteUpdateKind.ADOPT)
        self.assertEqual(observed["update"].target_index, 0)
        self.assertGreater(
            observed["update"].event_fields["effective_join_limit_m"], 0.0
        )
        fallback_event = next(
            event
            for event in events
            if event["event"]
            == "simulation_odom_pose_fallback_started"
        )
        self.assertEqual(
            fallback_event["source"],
            "simulation_direct_odom_after_tf_retry",
        )
        self.assertTrue(
            fallback_event["not_real_robot_migration_evidence"]
        )

    def test_manifest_change_during_preflight_is_rejected_before_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            manifest = write_dynamic_route_manifest(paths)
            args = self.base_args(paths) + [
                "--allow-sim-time",
                "--route-manifest",
                str(manifest),
            ]

            def mutate_manifest(*_args, **_kwargs):
                RouteRevisionStore(
                    manifest,
                    stream_id="sim-stream",
                    writer_id="planner",
                ).withdraw("planner stopped before motion")
                return passing_preflight()

            with patch.object(
                run_single_station_segment,
                "run_ros_preflight",
                side_effect=mutate_manifest,
            ), patch.object(
                run_single_station_segment,
                "run_simple_waypoint_follower",
            ) as follower, redirect_stdout(StringIO()):
                status = run_single_station_segment.main(args)
            events = read_events(paths["events"])

        self.assertEqual(status, 1)
        self.assertFalse(follower.called)
        rejected = next(
            event for event in events if event["event"] == "route_manifest_rejected"
        )
        self.assertEqual(rejected["phase"], "immediately_before_motion")
        self.assertIn("changed or was withdrawn", rejected["stop_reason"])

    def test_same_geometry_heartbeat_during_preflight_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            manifest = write_dynamic_route_manifest(paths)
            args = self.base_args(paths) + [
                "--allow-sim-time",
                "--route-manifest",
                str(manifest),
            ]

            def publish_heartbeat(*_args, **_kwargs):
                current = read_committed_revision(manifest)
                assert current.route_path is not None
                assert current.diagnostics_path is not None
                payload = current.manifest
                RouteRevisionStore(
                    manifest,
                    stream_id="sim-stream",
                    writer_id="planner",
                ).publish_active(
                    current.route_path.read_text(),
                    json.loads(current.diagnostics_path.read_text()),
                    target_revision=current.target_revision,
                    observation_unix_sec=time.time(),
                    source_robot_pose=payload["source_robot_pose"],
                    target=payload["target"],
                    evidence=payload["evidence"],
                    previous_route_length_m=payload["new_route_length_m"],
                    new_route_length_m=payload["new_route_length_m"],
                    safety_diagnostics=payload["safety_diagnostics"],
                )
                return passing_preflight()

            with patch.object(
                run_single_station_segment,
                "run_ros_preflight",
                side_effect=publish_heartbeat,
            ), patch.object(
                run_single_station_segment,
                "run_simple_waypoint_follower",
                return_value=FollowerResult("completed", "", 1.0, 0.2, True),
            ) as follower, redirect_stdout(StringIO()):
                status = run_single_station_segment.main(args)

        self.assertEqual(status, 0)
        self.assertTrue(follower.called)

    def test_runner_maps_dynamic_withdrawal_rejection_and_stop_events(self):
        cases = (
            (
                RouteUpdateKind.STOP,
                "dynamic_route_withdrawn",
                "route_withdrawn",
            ),
            (
                RouteUpdateKind.REJECT,
                "dynamic_route_rejected",
                "route_reload_rejected",
            ),
            (
                RouteUpdateKind.STOP,
                "dynamic_route_stopped",
                "route_reload_rejected",
            ),
        )
        for kind, source_event, expected_event in cases:
            with self.subTest(source_event=source_event), tempfile.TemporaryDirectory() as tmp:
                paths = self.make_paths(Path(tmp))
                manifest = write_dynamic_route_manifest(paths)
                args = self.base_args(paths) + [
                    "--allow-sim-time",
                    "--route-manifest",
                    str(manifest),
                ]

                def fake_follower(
                    _resolved,
                    _waypoints,
                    _config,
                    _provider,
                    callback,
                ):
                    callback(
                        RouteUpdate(
                            kind=kind,
                            reason=source_event,
                            event_name=source_event,
                            event_fields={"fault_code": source_event},
                        )
                    )
                    return FollowerResult("stopped", source_event, 0.1, 0.0, False)

                with patch.object(
                    run_single_station_segment,
                    "run_ros_preflight",
                    return_value=passing_preflight(),
                ), patch.object(
                    run_single_station_segment,
                    "run_simple_waypoint_follower",
                    side_effect=fake_follower,
                ), redirect_stdout(StringIO()):
                    status = run_single_station_segment.main(args)
                events = read_events(paths["events"])

            self.assertEqual(status, 1)
            mapped = [event for event in events if event["event"] == expected_event]
            self.assertEqual(len(mapped), 1)
            self.assertEqual(mapped[0]["fault_code"], source_event)

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
            ), patch.object(run_single_station_segment, "run_simple_waypoint_follower") as follower, patch.object(
                run_single_station_segment,
                "_confirm_motion",
                return_value=False,
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
        self.assertFalse(
            follower_config.allow_simulation_odom_after_stale_tf
        )
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
