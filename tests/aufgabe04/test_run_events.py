from __future__ import annotations

import json
import hashlib
import logging
import os
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

from scripts.aufgabe04.navigation.entrypoints import (  # noqa: E402
    run_single_station_segment,
)
from scripts.aufgabe04.navigation.execution.dynamic_route_handoff import (  # noqa: E402
    RouteUpdate,
    RouteUpdateKind,
)
from scripts.aufgabe04.navigation.control.follower_models import FollowerResult  # noqa: E402
from scripts.aufgabe04.navigation.foundation.models import Pose2D  # noqa: E402
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import (  # noqa: E402
    MissionLegKind,
)
from scripts.aufgabe04.navigation.missions.plan_stand_coverage_survey import (  # noqa: E402
    main as plan_coverage,
)
from scripts.aufgabe04.navigation.localization.ros_preflight import (  # noqa: E402
    RosObservation,
    RosPreflightResult,
)
from scripts.aufgabe04.navigation.foundation.run_events import (  # noqa: E402
    build_event,
    configure_event_logger,
    emit_event,
    event_to_json,
)
from scripts.aufgabe04.navigation.execution.route_revision_store import (  # noqa: E402
    RouteRevisionStore,
    read_committed_revision,
)
from scripts.aufgabe04.navigation.coverage.stand_blockage_replan import (  # noqa: E402
    TRANSIENT_OBSTACLE_OVERLAY_SCHEMA_VERSION,
    TransientObstacleOverlay,
    write_transient_obstacle_overlay,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (  # noqa: E402
    STATUS_PROVISIONAL,
    SurveyCandidate,
    load_coverage_survey_plan,
)
from scripts.aufgabe04.navigation.coverage.stand_discovery_route import (  # noqa: E402
    seal_stand_discovery_route,
)
from scripts.aufgabe04.navigation.coverage.transient_overlay_resume_state import (  # noqa: E402
    bind_transient_overlay_resume_state_to_diagnostics,
    update_transient_overlay_resume_state_from_events,
    write_transient_overlay_resume_state,
)
from scripts.aufgabe04.navigation.planning.waypoint_csv import (  # noqa: E402
    load_route_leg,
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


def write_resumed_coverage_fixture(root: Path) -> dict[str, object]:
    """Create real sealed coverage artifacts bound to one inherited overlay."""

    root = Path(root).resolve()
    survey = root / "survey"
    map_yaml = ROOT / "maps/aufgabe03/arena_1p898x3p9_auto.yaml"
    with redirect_stdout(StringIO()):
        status = plan_coverage(
            [
                "--map",
                str(map_yaml),
                "--semantic-map-id",
                "arena_1p898x3p9_auto",
                "--planning-frame",
                "map",
                "--start-x",
                "-0.5025319639494574",
                "--start-y",
                "-0.605412965510235",
                "--start-yaw",
                "-3.1376510363781347",
                "--survey-id",
                "resume_runner_test",
                "--output-dir",
                str(survey),
                "--lane-count",
                "1",
                "--stop-spacing-m",
                "0.70",
                "--expected-stand-count",
                "1",
            ]
        )
    if status != 0:
        raise AssertionError(f"coverage fixture planning failed: {status}")
    plan_path = survey / "coverage_plan.json"
    plan = load_coverage_survey_plan(plan_path)
    target_viewpoint_id = plan.viewpoints[0].viewpoint_id

    overlay_path = root / "overlay_replan_001.json"
    candidate = SurveyCandidate(
        candidate_uid="transient_obstacle_0001",
        x_m=1.40,
        y_m=0.60,
        radius_m=plan.config.candidate_radius_m,
        uncertainty_m=plan.config.candidate_uncertainty_m,
        keepout_radius_m=plan.config.candidate_keepout_radius_m,
        confidence=1.0,
        hit_count=1,
        first_seen_sec=0.0,
        last_seen_sec=0.0,
        source_observation_ids=("prior_blockage",),
        viewpoint_ids=(),
        status=STATUS_PROVISIONAL,
    )
    write_transient_obstacle_overlay(
        overlay_path,
        TransientObstacleOverlay(
            schema_version=TRANSIENT_OBSTACLE_OVERLAY_SCHEMA_VERSION,
            survey_id=plan.survey_id,
            planning_frame=plan.planning_frame,
            map_bundle_sha256=plan.map_bundle_sha256,
            candidates=(candidate,),
        ),
        source={"kind": "runner_resume_test"},
    )
    prior_route = root / "prior_adopted_route_001.csv"
    prior_route.write_text("prior adopted route\n", encoding="utf-8")
    state = update_transient_overlay_resume_state_from_events(
        [
            {
                "event": "transient_navigation_blockage_replanned",
                "run_id": "prior-child",
                "leg_index": 0,
                "replan_index": 1,
                "target_viewpoint_id": target_viewpoint_id,
                "semantic_survey_evidence": False,
                "transient_obstacle_overlay_json": str(overlay_path),
                "replacement_route_csv": str(prior_route),
                "source_map_route_sha256": hashlib.sha256(
                    prior_route.read_bytes()
                ).hexdigest(),
            }
        ],
        plan=plan,
        coverage_leg_index=0,
        target_viewpoint_id=target_viewpoint_id,
        max_replans=3,
        artifact_root=root,
    )
    if state is None:
        raise AssertionError("resume fixture did not create state")
    state_path = root / "resume_state.json"
    write_transient_overlay_resume_state(state_path, state, plan=plan)

    source_diagnostics = survey / "legs/leg_000_diagnostics.json"
    diagnostics_payload = json.loads(source_diagnostics.read_text())
    diagnostics_payload["metadata"]["planning_frame"] = plan.planning_frame
    source_diagnostics.write_text(
        json.dumps(diagnostics_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bound_diagnostics = root / "bound_resume_diagnostics.json"
    bind_transient_overlay_resume_state_to_diagnostics(
        source_diagnostics,
        bound_diagnostics,
        resume_state_path=state_path,
        plan=plan,
    )
    sealed = seal_stand_discovery_route(
        source_route_csv=survey / "legs/leg_000_route.csv",
        source_diagnostics_json=bound_diagnostics,
        coverage_plan_path=plan_path,
        output_dir=root / "sealed",
    )
    leg = load_route_leg(Path(sealed["route_csv"]), 0)
    return {
        "map_yaml": map_yaml,
        "survey": survey,
        "plan_path": plan_path,
        "target_viewpoint_id": target_viewpoint_id,
        "overlay_path": overlay_path,
        "state": state,
        "state_path": state_path,
        "sealed": sealed,
        "leg": leg,
    }


def resumed_coverage_runner_args(
    fixture: dict[str, object],
    *,
    root: Path,
    session: Path,
    semantic_log: Path,
) -> list[str]:
    sealed = fixture["sealed"]
    assert isinstance(sealed, dict)
    return [
        "--route-csv",
        str(sealed["route_csv"]),
        "--diagnostics-json",
        str(sealed["diagnostics_json"]),
        "--route-certificate-json",
        str(sealed["route_certificate_json"]),
        "--coverage-plan",
        str(fixture["plan_path"]),
        "--leg-index",
        "0",
        "--semantic-log",
        str(semantic_log),
        "--results-csv",
        str(root / "results.csv"),
        "--preflight-json",
        str(root / "preflight.json"),
        "--run-id",
        "resumed-child",
        "--coverage-transient-replan-survey-root",
        str(fixture["survey"]),
        "--coverage-transient-replan-session-root",
        str(session),
        "--coverage-transient-replan-map",
        str(fixture["map_yaml"]),
        "--coverage-transient-replan-semantic-map-id",
        "arena_1p898x3p9_auto",
        "--coverage-transient-replan-target-viewpoint-id",
        str(fixture["target_viewpoint_id"]),
        "--coverage-transient-replan-robot-radius-m",
        "0.105",
        "--coverage-transient-replan-max-count",
        "3",
        "--coverage-transient-replan-leg-index",
        "0",
        "--coverage-transient-replan-resume-state-json",
        str(fixture["state_path"]),
    ]


class RunEventsTest(unittest.TestCase):
    def test_simulation_motion_confirmation_does_not_block_for_input(self):
        args = type("Args", (), {"allow_sim_time": True})()
        with patch("builtins.input") as prompt, redirect_stdout(StringIO()):
            confirmed = run_single_station_segment._confirm_motion(args, object())

        self.assertTrue(confirmed)
        prompt.assert_not_called()

    def test_runtime_recovery_permit_bypasses_only_through_exact_validator(self):
        args = type(
            "Args",
            (),
            {
                "mission_motion_authorization_json": Path("mission.json"),
                "runtime_localization_motion_permit_json": Path("permit.json"),
                "dry_run": False,
                "allow_sim_time": False,
                "execution_pose_frame": "odom",
                "route_certificate_json": Path("certificate.json"),
                "coverage_transient_replan_leg_index": 0,
                "coverage_transient_replan_target_viewpoint_id": "survey_vp_001",
                "coverage_transient_replan_semantic_map_id": "arena_map",
                "mission_session_id": "mission",
                "run_id": "mission_coverage_000_runtime_localization_reseal_001",
                "robot_id": "turtlebot1",
                "localization_branch_proof_id": "known_start",
            },
        )()
        resolved = type(
            "Resolved",
            (),
            {"namespace": "", "cmd_vel_topic": "/cmd_vel"},
        )()
        sentinel = object()
        with patch.object(
            run_single_station_segment,
            "validate_runtime_localization_motion_permit_for_execution",
            return_value=sentinel,
        ) as validate:
            result = run_single_station_segment._validated_runtime_localization_motion_permit(
                args,
                resolved,
                route_csv_path=Path("route.csv"),
                diagnostics_path=Path("diagnostics.json"),
            )

        self.assertIs(result, sentinel)
        self.assertEqual(validate.call_args.kwargs["session_id"], "mission")
        self.assertEqual(
            validate.call_args.kwargs["target_viewpoint_id"],
            "survey_vp_001",
        )
        self.assertEqual(validate.call_args.kwargs["cmd_vel_topic"], "/cmd_vel")

    def test_partial_runtime_recovery_permit_fails_closed(self):
        args = type(
            "Args",
            (),
            {
                "mission_motion_authorization_json": Path("mission.json"),
                "runtime_localization_motion_permit_json": None,
            },
        )()
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            run_single_station_segment._validated_runtime_localization_motion_permit(
                args,
                object(),
                route_csv_path=Path("route.csv"),
                diagnostics_path=Path("diagnostics.json"),
            )

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

    def test_before_motion_global_consistency_evidence_reaches_safety_stop(self):
        details = {
            "reason": (
                "global localization consistency requires zero and reseal"
            ),
            "fault_code": "localization_reseal_required",
            "source": "global_consistency_monitor",
            "execution_pose_owner": "odom",
            "global_consistency_monitor": "amcl",
            "monitor_action": "FORCE_ZERO_RESEAL",
            "monitor_reason": "reseal_required",
            "monitor_warning": "",
            "execution_phase": "before_motion",
            "phase": "initial_runtime_input_wait",
            "motion_published": False,
            "continuity": {
                "schema_version": 1,
                "accepted": False,
                "decision": "force_zero_reseal",
                "reason": "map_from_odom_translation_drift",
                "fail_closed": True,
                "requires_zero_cycle": True,
                "requires_reseal": True,
                "threshold_semantics": (
                    "accept_if_observed_less_than_or_equal_to_limit"
                ),
                "certificate_sha256": "a" * 64,
                "map_frame": "map",
                "odom_frame": "odom",
                "base_frame": "base_footprint",
                "frozen_map_from_odom": {
                    "x_m": 0.0,
                    "y_m": 0.0,
                    "yaw_rad": 0.0,
                },
                "live_map_from_odom": {
                    "x_m": 0.10,
                    "y_m": 0.0,
                    "yaw_rad": 0.0,
                },
                "relative_translation_x_m": 0.10,
                "relative_translation_y_m": 0.0,
                "translation_drift_m": 0.10,
                "relative_yaw_rad": 0.0,
                "absolute_yaw_drift_rad": 0.0,
                "max_translation_drift_m": 0.03,
                "max_yaw_drift_rad": 0.03,
                "validation_error": None,
            },
            "fail_closed": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            with patch.object(
                run_single_station_segment,
                "run_ros_preflight",
                return_value=passing_preflight(),
            ), patch.object(
                run_single_station_segment,
                "run_simple_waypoint_follower",
                return_value=FollowerResult(
                    "stopped",
                    details["reason"],
                    0.2,
                    0.0,
                    False,
                    details,
                ),
            ), redirect_stdout(StringIO()):
                status = run_single_station_segment.main(self.base_args(paths))

            safety_stop = next(
                event
                for event in read_events(paths["events"])
                if event["event"] == "safety_stop"
            )
            names = [
                event["event"] for event in read_events(paths["events"])
            ]
            execution_attempt = next(
                event
                for event in read_events(paths["events"])
                if event["event"] == "motion_started"
            )

        self.assertEqual(status, 1)
        self.assertFalse(safety_stop["motion_published"])
        self.assertEqual(safety_stop["stop_details"], details)
        self.assertLess(names.index("motion_started"), names.index("safety_stop"))
        self.assertFalse(execution_attempt["motion_published"])
        self.assertEqual(
            execution_attempt["event_semantics"],
            "child_execution_attempt_started_before_follower",
        )

    def test_unexpected_follower_exception_writes_fail_closed_terminal_evidence(
        self,
    ):
        failure = NameError("name '_node_identity' is not defined")
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            with patch.object(
                run_single_station_segment,
                "run_ros_preflight",
                return_value=passing_preflight(),
            ), patch.object(
                run_single_station_segment,
                "run_simple_waypoint_follower",
                side_effect=failure,
            ), redirect_stdout(StringIO()):
                with self.assertRaises(NameError) as raised:
                    run_single_station_segment.main(self.base_args(paths))

            events = read_events(paths["events"])
            safety_stops = [
                event for event in events if event["event"] == "safety_stop"
            ]
            finish_events = [
                event for event in events if event["event"] == "run_finished"
            ]
            rows = read_result_rows(paths["results"])

        self.assertIs(raised.exception, failure)
        self.assertEqual(len(safety_stops), 1)
        self.assertEqual(len(finish_events), 1)
        self.assertNotIn("motion_completed", [event["event"] for event in events])
        safety_stop = safety_stops[0]
        self.assertEqual(safety_stop["status"], "stopped")
        self.assertTrue(safety_stop["motion_published"])
        self.assertEqual(
            safety_stop["stop_details"]["fault_code"],
            "unexpected_follower_exception",
        )
        self.assertTrue(safety_stop["stop_details"]["fail_closed"])
        self.assertTrue(
            safety_stop["stop_details"]["motion_history_uncertain"]
        )
        self.assertEqual(
            safety_stop["stop_details"]["exception_type"],
            "NameError",
        )
        self.assertEqual(
            safety_stop["stop_details"]["exception_message"],
            "name '_node_identity' is not defined",
        )
        self.assertFalse(
            safety_stop["stop_details"]["recovery_attempted"]
        )
        self.assertFalse(
            safety_stop["stop_details"]["continuation_allowed"]
        )
        self.assertEqual(finish_events[0]["final_status"], "stopped")
        self.assertTrue(finish_events[0]["motion_published"])
        self.assertTrue(finish_events[0]["motion_history_uncertain"])
        self.assertTrue(finish_events[0]["fail_closed"])
        self.assertEqual(
            finish_events[0]["fault_code"],
            "unexpected_follower_exception",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "stopped")
        self.assertEqual(rows[0]["duration_sec"], "")
        self.assertEqual(rows[0]["distance_estimate_m"], "")
        self.assertEqual(rows[0]["motion_published"], "True")

    def test_follower_exception_is_preserved_if_terminal_reporting_fails(self):
        failure = NameError("original follower failure")
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            with patch.object(
                run_single_station_segment,
                "run_ros_preflight",
                return_value=passing_preflight(),
            ), patch.object(
                run_single_station_segment,
                "run_simple_waypoint_follower",
                side_effect=failure,
            ), patch.object(
                run_single_station_segment,
                "record_unexpected_follower_exception",
                side_effect=OSError("event log unavailable"),
            ), redirect_stdout(StringIO()):
                with self.assertRaises(NameError) as raised:
                    run_single_station_segment.main(self.base_args(paths))

        self.assertIs(raised.exception, failure)

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

    def test_adopted_blockage_replan_is_appended_to_mission_adaptive_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            survey = root / "survey"
            with redirect_stdout(StringIO()):
                plan_status = plan_coverage(
                    [
                        "--map",
                        str(
                            ROOT
                            / "maps/aufgabe03/arena_1p898x3p9_auto.yaml"
                        ),
                        "--semantic-map-id",
                        "arena_1p898x3p9_auto",
                        "--planning-frame",
                        "map",
                        "--start-x",
                        "0",
                        "--start-y",
                        "0",
                        "--start-yaw",
                        "0",
                        "--survey-id",
                        "adoption_event_test",
                        "--output-dir",
                        str(survey),
                        "--lane-count",
                        "1",
                        "--stop-spacing-m",
                        "0.70",
                        "--expected-stand-count",
                        "1",
                    ]
                )
            self.assertEqual(plan_status, 0)
            sealed = seal_stand_discovery_route(
                source_route_csv=survey / "legs/leg_000_route.csv",
                source_diagnostics_json=(
                    survey / "legs/leg_000_diagnostics.json"
                ),
                coverage_plan_path=survey / "coverage_plan.json",
                output_dir=root / "sealed",
            )
            leg = load_route_leg(Path(sealed["route_csv"]), 0)
            first_pose = leg.raw_waypoints[0].pose
            session = root / "mission_session"
            semantic_log = root / "events.jsonl"
            args = [
                "--route-csv",
                sealed["route_csv"],
                "--diagnostics-json",
                sealed["diagnostics_json"],
                "--route-certificate-json",
                sealed["route_certificate_json"],
                "--coverage-plan",
                str(survey / "coverage_plan.json"),
                "--leg-index",
                "0",
                "--semantic-log",
                str(semantic_log),
                "--results-csv",
                str(root / "results.csv"),
                "--preflight-json",
                str(root / "preflight.json"),
                "--run-id",
                "run-adoption",
                "--coverage-transient-replan-survey-root",
                str(survey),
                "--coverage-transient-replan-session-root",
                str(session),
                "--coverage-transient-replan-map",
                str(
                    ROOT / "maps/aufgabe03/arena_1p898x3p9_auto.yaml"
                ),
                "--coverage-transient-replan-semantic-map-id",
                "arena_1p898x3p9_auto",
                "--coverage-transient-replan-target-viewpoint-id",
                "survey_vp_001",
                "--coverage-transient-replan-robot-radius-m",
                "0.105",
                "--coverage-transient-replan-max-count",
                "3",
                "--coverage-transient-replan-leg-index",
                "7",
            ]

            def fake_follower(
                _resolved,
                _waypoints,
                _config,
                _provider,
                callback,
                **kwargs,
            ):
                self.assertIn("blockage_recovery_provider", kwargs)
                callback(
                    RouteUpdate(
                        kind=RouteUpdateKind.ADOPT,
                        event_name=(
                            "transient_navigation_blockage_replanned"
                        ),
                        route_revision=2,
                        route_hash="route_hash_2",
                        event_fields={
                            "replan_index": 2,
                            "post_plan_runtime_revalidated": True,
                            "semantic_survey_evidence": False,
                            "target_viewpoint_id": "survey_vp_001",
                            "replacement_route_csv": "replacement.csv",
                            "transient_obstacle_overlay_json": "overlay.json",
                        },
                    )
                )
                return FollowerResult("completed", "", 1.0, 0.2, True)

            preflight = RosPreflightResult(
                ok=True,
                failures=[],
                observations=[],
                runtime_config={},
                route_pose={
                    "frame_id": "map",
                    "child_frame_id": "base_footprint",
                    "x_m": first_pose.x_m,
                    "y_m": first_pose.y_m,
                    "yaw_rad": 0.0,
                },
            )
            output = StringIO()
            with patch.object(
                run_single_station_segment,
                "run_ros_preflight",
                return_value=preflight,
            ), patch.object(
                run_single_station_segment,
                "_confirm_motion",
                return_value=True,
            ), patch.object(
                run_single_station_segment,
                "run_simple_waypoint_follower",
                side_effect=fake_follower,
            ), redirect_stdout(output):
                status = run_single_station_segment.main(args)

            self.assertEqual(status, 0, output.getvalue())
            adaptive_events = read_events(
                session / "adaptive_replans.jsonl"
            )
            motion_event = next(
                event
                for event in read_events(semantic_log)
                if event["event"] == "motion_completed"
            )

        self.assertEqual(len(adaptive_events), 1)
        self.assertEqual(
            adaptive_events[0]["event"],
            "transient_navigation_blockage_replanned",
        )
        self.assertEqual(adaptive_events[0]["run_id"], "run-adoption")
        self.assertEqual(adaptive_events[0]["leg_index"], 7)
        self.assertEqual(motion_event["leg_index"], 0)
        self.assertEqual(motion_event["coverage_leg_index"], 7)
        self.assertEqual(
            motion_event["target_viewpoint_id"],
            "survey_vp_001",
        )
        self.assertEqual(adaptive_events[0]["replan_index"], 2)
        self.assertTrue(
            adaptive_events[0]["post_plan_runtime_revalidated"]
        )
        self.assertFalse(adaptive_events[0]["semantic_survey_evidence"])
        self.assertEqual(
            adaptive_events[0]["target_viewpoint_id"],
            "survey_vp_001",
        )

    def test_bound_resume_state_seeds_next_cumulative_blockage_replan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            fixture = write_resumed_coverage_fixture(root)
            session = root / "mission_session"
            semantic_log = root / "events.jsonl"
            source_001 = (
                Path(fixture["survey"])
                / "replans/leg_000_replan_001"
            )
            execution_001 = (
                session / "execution/coverage_leg_000_replan_001"
            )
            source_001.mkdir(parents=True)
            execution_001.mkdir(parents=True)
            source_marker = source_001 / "existing.txt"
            execution_marker = execution_001 / "existing.txt"
            source_marker.write_text("prior source\n", encoding="utf-8")
            execution_marker.write_text("prior execution\n", encoding="utf-8")
            leg = fixture["leg"]
            state = fixture["state"]
            first_pose = leg.raw_waypoints[0].pose
            observed: dict[str, object] = {}

            def fake_follower(
                _resolved,
                _waypoints,
                _config,
                _provider,
                _callback,
                **kwargs,
            ):
                provider = kwargs["blockage_recovery_provider"]
                observed["initial_count"] = provider.replan_count
                observed["initial_overlay"] = provider.overlay_path
                observed["initial_hashes"] = set(
                    provider.adopted_route_hashes
                )
                update = provider(
                    Pose2D(
                        -0.858887873410987,
                        -0.46164086690318107,
                        -3.1376510363781347,
                    ),
                    "stuck no progress",
                    {
                        "stationary_obstacle_confirmation": {
                            "confirmed": True,
                            "fail_closed": False,
                            "distinct_sample_count": 3,
                            "thresholds": {"min_distinct_samples": 3},
                        },
                        "front_clearance": {
                            "source": "front_sector",
                            "nearest_valid_range_m": 0.23000000417232513,
                            "nearest_valid_bearing_rad": 0.20737460535019636,
                        },
                    },
                )
                observed["update"] = update
                observed["final_count"] = provider.replan_count
                return FollowerResult("completed", "", 1.0, 0.2, True)

            preflight = RosPreflightResult(
                ok=True,
                failures=[],
                observations=[],
                runtime_config={},
                route_pose={
                    "frame_id": "map",
                    "child_frame_id": "base_footprint",
                    "x_m": first_pose.x_m,
                    "y_m": first_pose.y_m,
                    "yaw_rad": 0.0,
                },
            )
            args = resumed_coverage_runner_args(
                fixture,
                root=root,
                session=session,
                semantic_log=semantic_log,
            )
            output = StringIO()
            with patch.object(
                run_single_station_segment,
                "run_ros_preflight",
                return_value=preflight,
            ), patch(
                "builtins.input",
                return_value="RUN",
            ), patch.object(
                run_single_station_segment,
                "run_simple_waypoint_follower",
                side_effect=fake_follower,
            ), redirect_stdout(output):
                status = run_single_station_segment.main(args)

            update = observed["update"]
            observed["source_marker"] = source_marker.read_text()
            observed["execution_marker"] = execution_marker.read_text()
            observed["events"] = read_events(semantic_log)

        self.assertEqual(status, 0, output.getvalue())
        self.assertEqual(observed["initial_count"], 1)
        self.assertEqual(
            Path(observed["initial_overlay"]),
            Path(fixture["overlay_path"]),
        )
        self.assertTrue(
            set(state.adopted_route_sha256s).issubset(
                observed["initial_hashes"]
            )
        )
        self.assertIn(leg.source_sha256, observed["initial_hashes"])
        self.assertEqual(observed["final_count"], 2)
        self.assertEqual(update.kind, RouteUpdateKind.ADOPT)
        self.assertEqual(update.route_revision, 2)
        self.assertEqual(update.event_fields["replan_index"], 2)
        self.assertIn(
            "leg_000_replan_002",
            update.event_fields["transient_obstacle_overlay_json"],
        )
        self.assertIn(
            "coverage_leg_000_replan_002",
            update.event_fields["replacement_route_csv"],
        )
        self.assertNotIn(
            "replan_001",
            "\n".join(
                [
                    update.event_fields[
                        "transient_obstacle_overlay_json"
                    ],
                    update.event_fields["replacement_route_csv"],
                    update.event_fields[
                        "replacement_route_certificate_json"
                    ],
                ]
            ),
        )
        self.assertEqual(observed["source_marker"], "prior source\n")
        self.assertEqual(observed["execution_marker"], "prior execution\n")
        event_names = [item["event"] for item in observed["events"]]
        self.assertLess(
            event_names.index("transient_overlay_resume_state_validated"),
            event_names.index("motion_started"),
        )

    def test_tampered_bound_resume_state_is_rejected_before_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            fixture = write_resumed_coverage_fixture(root)
            session = root / "mission_session"
            semantic_log = root / "events.jsonl"
            state_path = Path(fixture["state_path"])
            tampered = json.loads(state_path.read_text(encoding="utf-8"))
            tampered["remaining_replans"] = 0
            state_path.write_text(
                json.dumps(tampered, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            args = resumed_coverage_runner_args(
                fixture,
                root=root,
                session=session,
                semantic_log=semantic_log,
            )
            with patch.object(
                run_single_station_segment,
                "run_ros_preflight",
            ) as preflight, patch.object(
                run_single_station_segment,
                "run_simple_waypoint_follower",
            ) as follower, redirect_stdout(StringIO()), redirect_stderr(
                StringIO()
            ):
                with self.assertRaises(SystemExit) as raised:
                    run_single_station_segment.main(args)
            events = read_events(semantic_log)

        self.assertEqual(raised.exception.code, 2)
        preflight.assert_not_called()
        follower.assert_not_called()
        rejected = next(
            event
            for event in events
            if event["event"]
            == "transient_overlay_resume_state_rejected"
        )
        self.assertFalse(rejected["motion_published"])
        self.assertIn("hash mismatch", rejected["stop_reason"])
        self.assertNotIn("motion_started", [item["event"] for item in events])

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

    def test_runtime_permit_claim_precedes_motion_and_skips_child_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            permit = type(
                "Permit",
                (),
                {
                    "target_viewpoint_id": "survey_vp_001",
                    "leg_index": 7,
                    "reseal_index": 1,
                    "rejected_run_id": "run-0",
                },
            )()
            receipt = object()
            receipt_path = Path(tmp) / "consumption.json"
            with patch.object(
                run_single_station_segment,
                "run_ros_preflight",
                return_value=passing_preflight(),
            ), patch.object(
                run_single_station_segment,
                "_validated_runtime_localization_motion_permit",
                return_value=permit,
            ), patch.object(
                run_single_station_segment,
                "default_runtime_motion_consumption_receipt_path",
                return_value=receipt_path,
            ), patch.object(
                run_single_station_segment,
                "consume_runtime_motion_permit",
                return_value=receipt,
            ) as consume, patch.object(
                run_single_station_segment,
                "runtime_localization_motion_permit_sha256",
                return_value="a" * 64,
            ), patch.object(
                run_single_station_segment,
                "runtime_motion_consumption_receipt_sha256",
                return_value="b" * 64,
            ), patch.object(
                run_single_station_segment,
                "_confirm_motion",
                side_effect=AssertionError("permit path must not prompt"),
            ), patch.object(
                run_single_station_segment,
                "run_simple_waypoint_follower",
                return_value=FollowerResult("completed", "", 1.0, 0.2, True),
            ) as follower, redirect_stdout(StringIO()):
                status = run_single_station_segment.main(self.base_args(paths))

            events = read_events(paths["events"])
            names = [event["event"] for event in events]

        self.assertEqual(status, 0)
        self.assertTrue(consume.called)
        self.assertEqual(consume.call_args.kwargs["leg_index"], 7)
        self.assertTrue(follower.called)
        self.assertLess(
            names.index("runtime_localization_motion_permit_consumed"),
            names.index("motion_started"),
        )
        consumed = next(
            event
            for event in events
            if event["event"]
            == "runtime_localization_motion_permit_consumed"
        )
        self.assertEqual(
            consumed["runtime_motion_consumption_receipt_json"],
            str(receipt_path),
        )
        self.assertEqual(consumed["coverage_leg_index"], 7)
        self.assertEqual(
            consumed["target_viewpoint_id"],
            "survey_vp_001",
        )
        self.assertFalse(consumed["additional_typed_run_required"])

    def test_routine_leg_permit_claim_precedes_motion_and_skips_child_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            permit = type(
                "Permit",
                (),
                {
                    "mission_leg_kind": MissionLegKind.COVERAGE,
                    "mission_leg_index": 1,
                    "target_id": "survey_vp_002",
                },
            )()
            receipt = object()
            receipt_path = Path(tmp) / "mission_leg_consumption.json"
            with patch.object(
                run_single_station_segment,
                "run_ros_preflight",
                return_value=passing_preflight(),
            ), patch.object(
                run_single_station_segment,
                "_validated_runtime_localization_motion_permit",
                return_value=None,
            ), patch.object(
                run_single_station_segment,
                "_validated_mission_leg_motion_permit",
                return_value=permit,
            ), patch.object(
                run_single_station_segment,
                "default_mission_leg_motion_consumption_receipt_path",
                return_value=receipt_path,
            ), patch.object(
                run_single_station_segment,
                "consume_mission_leg_motion_permit",
                return_value=receipt,
            ) as consume, patch.object(
                run_single_station_segment,
                "mission_leg_motion_permit_sha256",
                return_value="c" * 64,
            ), patch.object(
                run_single_station_segment,
                "mission_leg_motion_consumption_receipt_sha256",
                return_value="d" * 64,
            ), patch.object(
                run_single_station_segment,
                "_confirm_motion",
                side_effect=AssertionError("mission leg permit must not prompt"),
            ), patch.object(
                run_single_station_segment,
                "run_simple_waypoint_follower",
                return_value=FollowerResult("completed", "", 1.0, 0.2, True),
            ) as follower, redirect_stdout(StringIO()):
                status = run_single_station_segment.main(self.base_args(paths))

            events = read_events(paths["events"])
            names = [event["event"] for event in events]

        self.assertEqual(status, 0)
        self.assertTrue(consume.called)
        self.assertEqual(
            consume.call_args.kwargs["mission_leg_kind"],
            MissionLegKind.COVERAGE,
        )
        self.assertTrue(follower.called)
        self.assertLess(
            names.index("mission_leg_motion_permit_consumed"),
            names.index("motion_started"),
        )
        consumed = next(
            event
            for event in events
            if event["event"] == "mission_leg_motion_permit_consumed"
        )
        self.assertEqual(
            consumed["mission_leg_motion_consumption_receipt_json"],
            str(receipt_path),
        )
        self.assertEqual(consumed["coverage_leg_index"], 1)
        self.assertEqual(
            consumed["target_viewpoint_id"],
            "survey_vp_002",
        )
        self.assertFalse(consumed["additional_typed_run_required"])

    def test_startup_reseal_permit_claim_precedes_motion_and_skips_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            permit = type(
                "Permit",
                (),
                {
                    "target_viewpoint_id": "survey_vp_001",
                    "leg_index": 0,
                    "reseal_index": 1,
                    "rejected_run_id": "run-0",
                    "recovery_source_kind": (
                        "certified_start_pose_mismatch"
                    ),
                },
            )()
            receipt = object()
            receipt_path = Path(tmp) / "startup_consumption.json"
            with patch.object(
                run_single_station_segment,
                "run_ros_preflight",
                return_value=passing_preflight(),
            ), patch.object(
                run_single_station_segment,
                "_validated_runtime_localization_motion_permit",
                return_value=None,
            ), patch.object(
                run_single_station_segment,
                "_validated_mission_leg_motion_permit",
                return_value=None,
            ), patch.object(
                run_single_station_segment,
                "_validated_startup_reseal_motion_permit",
                return_value=permit,
            ), patch.object(
                run_single_station_segment,
                "default_startup_reseal_motion_consumption_receipt_path",
                return_value=receipt_path,
            ), patch.object(
                run_single_station_segment,
                "consume_startup_reseal_motion_permit",
                return_value=receipt,
            ) as consume, patch.object(
                run_single_station_segment,
                "startup_reseal_motion_permit_sha256",
                return_value="e" * 64,
            ), patch.object(
                run_single_station_segment,
                "startup_reseal_motion_consumption_receipt_sha256",
                return_value="f" * 64,
            ), patch.object(
                run_single_station_segment,
                "_confirm_motion",
                side_effect=AssertionError("startup permit must not prompt"),
            ), patch.object(
                run_single_station_segment,
                "run_simple_waypoint_follower",
                return_value=FollowerResult("completed", "", 1.0, 0.2, True),
            ) as follower, redirect_stdout(StringIO()):
                status = run_single_station_segment.main(self.base_args(paths))

            events = read_events(paths["events"])
            names = [event["event"] for event in events]

        self.assertEqual(status, 0)
        self.assertTrue(consume.called)
        self.assertEqual(consume.call_args.kwargs["reseal_index"], 1)
        self.assertTrue(follower.called)
        self.assertLess(
            names.index("startup_reseal_motion_permit_consumed"),
            names.index("motion_started"),
        )
        consumed = next(
            event
            for event in events
            if event["event"] == "startup_reseal_motion_permit_consumed"
        )
        self.assertEqual(
            consumed[
                "startup_reseal_motion_consumption_receipt_json"
            ],
            str(receipt_path),
        )
        self.assertEqual(consumed["coverage_leg_index"], 0)
        self.assertEqual(
            consumed["target_viewpoint_id"],
            "survey_vp_001",
        )
        self.assertFalse(consumed["additional_typed_run_required"])
        self.assertEqual(
            consumed["recovery_source_kind"],
            "certified_start_pose_mismatch",
        )

    def test_runtime_permit_replay_rejects_before_motion_started(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            permit = type(
                "Permit",
                (),
                {
                    "target_viewpoint_id": "survey_vp_001",
                    "leg_index": 0,
                    "reseal_index": 1,
                    "rejected_run_id": "run-0",
                },
            )()
            with patch.object(
                run_single_station_segment,
                "run_ros_preflight",
                return_value=passing_preflight(),
            ), patch.object(
                run_single_station_segment,
                "_validated_runtime_localization_motion_permit",
                return_value=permit,
            ), patch.object(
                run_single_station_segment,
                "default_runtime_motion_consumption_receipt_path",
                return_value=Path(tmp) / "consumption.json",
            ), patch.object(
                run_single_station_segment,
                "consume_runtime_motion_permit",
                side_effect=ValueError("runtime motion permit already consumed"),
            ), patch.object(
                run_single_station_segment,
                "_confirm_motion",
                side_effect=AssertionError("replayed permit must not prompt"),
            ), patch.object(
                run_single_station_segment,
                "run_simple_waypoint_follower",
            ) as follower, redirect_stdout(StringIO()):
                status = run_single_station_segment.main(self.base_args(paths))

            events = read_events(paths["events"])
            names = [event["event"] for event in events]

        self.assertEqual(status, 1)
        self.assertFalse(follower.called)
        self.assertIn("motion_authorization_rejected", names)
        self.assertNotIn("runtime_localization_motion_permit_consumed", names)
        self.assertNotIn("motion_started", names)

    def test_startup_permit_replay_rejects_before_motion_started(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            permit = type(
                "Permit",
                (),
                {
                    "target_viewpoint_id": "survey_vp_001",
                    "leg_index": 0,
                    "reseal_index": 1,
                    "rejected_run_id": "run-0",
                },
            )()
            with patch.object(
                run_single_station_segment,
                "run_ros_preflight",
                return_value=passing_preflight(),
            ), patch.object(
                run_single_station_segment,
                "_validated_runtime_localization_motion_permit",
                return_value=None,
            ), patch.object(
                run_single_station_segment,
                "_validated_mission_leg_motion_permit",
                return_value=None,
            ), patch.object(
                run_single_station_segment,
                "_validated_startup_reseal_motion_permit",
                return_value=permit,
            ), patch.object(
                run_single_station_segment,
                "default_startup_reseal_motion_consumption_receipt_path",
                return_value=Path(tmp) / "startup_consumption.json",
            ), patch.object(
                run_single_station_segment,
                "consume_startup_reseal_motion_permit",
                side_effect=ValueError(
                    "startup reseal motion permit already consumed"
                ),
            ), patch.object(
                run_single_station_segment,
                "_confirm_motion",
                side_effect=AssertionError("replayed permit must not prompt"),
            ), patch.object(
                run_single_station_segment,
                "run_simple_waypoint_follower",
            ) as follower, redirect_stdout(StringIO()):
                status = run_single_station_segment.main(self.base_args(paths))

            events = read_events(paths["events"])
            names = [event["event"] for event in events]

        self.assertEqual(status, 1)
        self.assertFalse(follower.called)
        self.assertIn("motion_authorization_rejected", names)
        self.assertNotIn("startup_reseal_motion_permit_consumed", names)
        self.assertNotIn("motion_started", names)

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

    def test_bundled_real_run_passes_canonical_controller_trace_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_paths(Path(tmp))
            bundle = Path(tmp) / "bundle"
            bundle.mkdir()
            with patch.dict(
                os.environ,
                {"MII_AMR_RUN_BUNDLE_DIR": str(bundle)},
            ), patch.object(
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
            ), redirect_stdout(StringIO()):
                status = run_single_station_segment.main(self.base_args(paths))

            self.assertEqual(status, 0)
            self.assertEqual(
                follower.call_args.kwargs["controller_trace_path"],
                bundle / "controller_trace.jsonl",
            )

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
