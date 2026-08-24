from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.aufgabe04.navigation.entrypoints import run_single_station_segment
from scripts.aufgabe04.navigation.control.follower_models import FollowerResult
from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import MissionLegKind
from scripts.aufgabe04.navigation.localization.ros_preflight import RosPreflightResult
from scripts.aufgabe04.navigation.coverage.stand_discovery_route import (
    seal_stand_discovery_route,
)
from scripts.aufgabe04.navigation.execution.startup_reseal_motion_authorization import (
    STARTUP_RESEAL_MOTION_AUTHORIZATION_SCOPE,
    STARTUP_RESEAL_RECOVERY_KIND,
    STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY,
    STARTUP_RESEAL_RUN_CONFIRMATION,
    StartupResealMotionAuthorization,
    load_startup_reseal_motion_permit,
    startup_reseal_motion_permit_sha256,
    validate_startup_reseal_motion_permit_for_execution,
    write_startup_reseal_motion_authorization,
)
from scripts.aufgabe04.navigation.planning.waypoint_csv import load_route_leg
from scripts.aufgabe04.real_robot.autonomous_startup_reseal import (
    StartupResealPermitContext,
    issue_startup_reseal_motion_permit,
    write_startup_reseal_permit_summary,
)
from tests.aufgabe04.test_run_events import write_resumed_coverage_fixture


MISSION_LEG_INDEX = 3
REJECTED_RUN_ID = "mission-001-coverage-003"
REPLACEMENT_RUN_ID = f"{REJECTED_RUN_ID}-startup-reseal-001"
SESSION_ID = "mission-001"
SEMANTIC_MAP_ID = "arena_1p898x3p9_auto"
LOCALIZATION_BRANCH_PROOF_ID = "known-start-proof-001"


def _prestart_global_consistency_stop() -> dict[str, object]:
    reason = "global localization consistency requires zero and reseal"
    return {
        "reason": reason,
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
                "x_m": 0.2,
                "y_m": 0.0,
                "yaw_rad": 0.0,
            },
            "relative_translation_x_m": 0.2,
            "relative_translation_y_m": 0.0,
            "translation_drift_m": 0.2,
            "relative_yaw_rad": 0.0,
            "absolute_yaw_drift_rad": 0.0,
            "max_translation_drift_m": 0.1,
            "max_yaw_drift_rad": 0.1,
            "validation_error": None,
        },
        "fail_closed": True,
    }


def _write_fresh_localization_evidence(
    path: Path,
    *,
    x_m: float,
    y_m: float,
    yaw_rad: float,
) -> None:
    pose = {"x_m": x_m, "y_m": y_m, "yaw_rad": yaw_rad}
    path.write_text(
        json.dumps(
            {
                "ok": True,
                "failures": [],
                "observations": [
                    {
                        "name": "stationary AMCL stability",
                        "ok": True,
                        "detail": "samples=2/2",
                        "data": {
                            "sample_count": 2,
                            "required_sample_count": 2,
                            "service_request_count": 2,
                            "position_covariance_complete": True,
                            "yaw_covariance_complete": True,
                        },
                    }
                ],
                "runtime_config": {
                    "localization_source": "amcl",
                    "use_sim_time": False,
                },
                "route_pose": {
                    "frame_id": "map",
                    "child_frame_id": "base_footprint",
                    **pose,
                },
                "odom_pose": None,
                "map_from_odom": None,
                "stationary_amcl_samples": [
                    {**pose, "covariance": [0.0] * 36},
                    {**pose, "covariance": [0.0] * 36},
                ],
                "stationary_map_from_odom_samples": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


class PrestartResealEventChainIntegrationTest(unittest.TestCase):
    def test_real_child_event_chain_issues_exact_prestart_reseal_permit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fixture = write_resumed_coverage_fixture(root / "coverage_fixture")
            survey_root = Path(fixture["survey"])
            plan_path = Path(fixture["plan_path"])
            target_viewpoint_id = str(fixture["target_viewpoint_id"])
            sealed = seal_stand_discovery_route(
                source_route_csv=survey_root / "legs/leg_000_route.csv",
                source_diagnostics_json=(
                    survey_root / "legs/leg_000_diagnostics.json"
                ),
                coverage_plan_path=plan_path,
                output_dir=root / "execution_route",
            )
            route_csv = Path(sealed["route_csv"])
            diagnostics_json = Path(sealed["diagnostics_json"])
            route_certificate_json = Path(sealed["route_certificate_json"])
            route_leg = load_route_leg(route_csv, 0)
            first_pose = route_leg.raw_waypoints[0].pose
            route_diagnostics = json.loads(
                diagnostics_json.read_text(encoding="utf-8")
            )
            fresh_start_yaw_rad = float(
                route_diagnostics["metadata"]["exact_start_connector"]
                ["exact_start"]["yaw_rad"]
            )
            semantic_log = root / "rejected_events.jsonl"
            session_root = root / "mission_session"
            session_root.mkdir(parents=True)

            routine_permit = SimpleNamespace(
                mission_leg_kind=MissionLegKind.COVERAGE,
                mission_leg_index=MISSION_LEG_INDEX,
                target_id=target_viewpoint_id,
            )
            receipt_path = root / "routine_permit_consumption.json"
            stop_details = _prestart_global_consistency_stop()
            preflight = RosPreflightResult(
                ok=True,
                failures=[],
                observations=[],
                runtime_config={
                    "localization_source": "amcl",
                    "use_sim_time": False,
                },
                route_pose={
                    "frame_id": "map",
                    "child_frame_id": "base_footprint",
                    "x_m": first_pose.x_m,
                    "y_m": first_pose.y_m,
                    "yaw_rad": fresh_start_yaw_rad,
                },
            )
            args = [
                "--route-csv",
                str(route_csv),
                "--diagnostics-json",
                str(diagnostics_json),
                "--route-certificate-json",
                str(route_certificate_json),
                "--coverage-plan",
                str(plan_path),
                "--leg-index",
                "0",
                "--semantic-log",
                str(semantic_log),
                "--results-csv",
                str(root / "results.csv"),
                "--preflight-json",
                str(root / "preflight.json"),
                "--run-id",
                REJECTED_RUN_ID,
                "--coverage-transient-replan-survey-root",
                str(survey_root),
                "--coverage-transient-replan-session-root",
                str(session_root),
                "--coverage-transient-replan-map",
                str(fixture["map_yaml"]),
                "--coverage-transient-replan-semantic-map-id",
                SEMANTIC_MAP_ID,
                "--coverage-transient-replan-target-viewpoint-id",
                target_viewpoint_id,
                "--coverage-transient-replan-robot-radius-m",
                "0.105",
                "--coverage-transient-replan-max-count",
                "3",
                "--coverage-transient-replan-leg-index",
                str(MISSION_LEG_INDEX),
            ]
            output = StringIO()
            with patch.object(
                run_single_station_segment,
                "run_ros_preflight",
                return_value=preflight,
            ), patch.object(
                run_single_station_segment,
                "_validated_mission_leg_motion_permit",
                return_value=routine_permit,
            ), patch.object(
                run_single_station_segment,
                "default_mission_leg_motion_consumption_receipt_path",
                return_value=receipt_path,
            ), patch.object(
                run_single_station_segment,
                "consume_mission_leg_motion_permit",
                return_value=object(),
            ) as consume, patch.object(
                run_single_station_segment,
                "mission_leg_motion_permit_sha256",
                return_value="b" * 64,
            ), patch.object(
                run_single_station_segment,
                "mission_leg_motion_consumption_receipt_sha256",
                return_value="c" * 64,
            ), patch.object(
                run_single_station_segment,
                "_confirm_motion",
                side_effect=AssertionError("routine permit must avoid RUN prompt"),
            ) as confirm, patch(
                "builtins.input",
                side_effect=AssertionError("no input prompt is authorized"),
            ) as prompt, patch.object(
                run_single_station_segment,
                "run_simple_waypoint_follower",
                return_value=FollowerResult(
                    status="stopped",
                    stop_reason=str(stop_details["reason"]),
                    duration_sec=0.2,
                    distance_estimate_m=0.0,
                    motion_published=False,
                    stop_details=stop_details,
                ),
            ) as follower, redirect_stdout(output):
                status = run_single_station_segment.main(args)

            self.assertEqual(status, 1, output.getvalue())
            prompt.assert_not_called()
            confirm.assert_not_called()
            follower.assert_called_once()
            consume.assert_called_once()
            self.assertEqual(
                consume.call_args.kwargs["mission_leg_index"],
                MISSION_LEG_INDEX,
            )
            self.assertEqual(
                consume.call_args.kwargs["target_id"],
                target_viewpoint_id,
            )

            events = [
                json.loads(line)
                for line in semantic_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            chain = [
                event
                for event in events
                if event["event"]
                in {
                    "mission_leg_motion_permit_consumed",
                    "motion_started",
                    "safety_stop",
                }
            ]
            self.assertEqual(
                [event["event"] for event in chain],
                [
                    "mission_leg_motion_permit_consumed",
                    "motion_started",
                    "safety_stop",
                ],
            )
            for event in chain:
                self.assertEqual(event["run_id"], REJECTED_RUN_ID)
                self.assertEqual(event["leg_index"], 0)
                self.assertEqual(
                    event["coverage_leg_index"],
                    MISSION_LEG_INDEX,
                )
                self.assertEqual(
                    event["target_viewpoint_id"],
                    target_viewpoint_id,
                )
            consumed, execution_attempt, safety_stop = chain
            self.assertEqual(consumed["mission_leg_kind"], "coverage")
            self.assertEqual(
                consumed["mission_leg_index"],
                MISSION_LEG_INDEX,
            )
            self.assertEqual(consumed["target_id"], target_viewpoint_id)
            self.assertTrue(consumed["covered_by_initial_mission_run"])
            self.assertFalse(consumed["additional_typed_run_required"])
            self.assertFalse(execution_attempt["motion_published"])
            self.assertEqual(
                execution_attempt["event_semantics"],
                "child_execution_attempt_started_before_follower",
            )
            self.assertFalse(safety_stop["motion_published"])
            self.assertEqual(safety_stop["stop_details"], stop_details)

            master_path = root / "startup_reseal_authorization.json"
            authorization = StartupResealMotionAuthorization(
                session_id=SESSION_ID,
                robot_id="tb3",
                namespace="",
                cmd_vel_topic="/cmd_vel",
                semantic_map_id=SEMANTIC_MAP_ID,
                localization_branch_proof_id=LOCALIZATION_BRANCH_PROOF_ID,
                max_startup_reseals_per_leg=2,
                scope_text=STARTUP_RESEAL_MOTION_AUTHORIZATION_SCOPE,
                operator_confirmation=STARTUP_RESEAL_RUN_CONFIRMATION,
                allowed_recovery_kind=STARTUP_RESEAL_RECOVERY_KIND,
            )
            write_startup_reseal_motion_authorization(
                master_path,
                authorization,
            )
            fresh_localization = root / "fresh_localization.json"
            _write_fresh_localization_evidence(
                fresh_localization,
                x_m=first_pose.x_m,
                y_m=first_pose.y_m,
                yaw_rad=fresh_start_yaw_rad,
            )
            summary_path = root / "startup_reseal_summary.json"
            write_startup_reseal_permit_summary(
                summary_path,
                leg_index=MISSION_LEG_INDEX,
                target_viewpoint_id=target_viewpoint_id,
                reseal_index=1,
                rejected_run_id=REJECTED_RUN_ID,
                fresh_start_x_m=first_pose.x_m,
                fresh_start_y_m=first_pose.y_m,
                fresh_start_yaw_rad=fresh_start_yaw_rad,
                route_csv=route_csv,
                diagnostics_json=diagnostics_json,
                recovery_source_kind=(
                    STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY
                ),
            )
            dry_preflight = root / "dry_preflight.json"
            dry_odom_certificate = root / "dry_odom_certificate.json"
            dry_uncertainty_budget = root / "dry_uncertainty_budget.json"
            for path, payload in (
                (dry_preflight, {"ok": True, "motion_published": False}),
                (
                    dry_odom_certificate,
                    {"accepted": True, "execution_pose_owner": "odom"},
                ),
                (
                    dry_uncertainty_budget,
                    {"accepted": True, "remaining_margin_m": 0.02},
                ),
            ):
                path.write_text(
                    json.dumps(payload, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

            permit_path = root / "startup_reseal_permit.json"
            context = StartupResealPermitContext(
                mission_authorization_json=master_path,
                session_id=SESSION_ID,
                semantic_map_id=SEMANTIC_MAP_ID,
                leg_index=MISSION_LEG_INDEX,
                target_viewpoint_id=target_viewpoint_id,
                reseal_index=1,
                max_startup_reseals_per_leg=2,
                rejected_run_id=REJECTED_RUN_ID,
                rejected_semantic_log_path=semantic_log,
                startup_reseal_summary_path=summary_path,
                fresh_localization_evidence_path=fresh_localization,
                permit_json_path=permit_path,
                recovery_source_kind=(
                    STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY
                ),
            )
            written_path, written_sha256 = issue_startup_reseal_motion_permit(
                context=context,
                run_id=REPLACEMENT_RUN_ID,
                route_csv=route_csv,
                diagnostics_json=diagnostics_json,
                map_route_certificate_json=route_certificate_json,
                dry_preflight_json=dry_preflight,
                dry_odom_certificate_json=dry_odom_certificate,
                dry_uncertainty_budget_json=dry_uncertainty_budget,
            )
            loaded = load_startup_reseal_motion_permit(written_path)
            validated = validate_startup_reseal_motion_permit_for_execution(
                written_path,
                master_authorization_path=master_path,
                run_id=REPLACEMENT_RUN_ID,
                session_id=SESSION_ID,
                robot_id="tb3",
                namespace="",
                cmd_vel_topic="/cmd_vel",
                semantic_map_id=SEMANTIC_MAP_ID,
                target_viewpoint_id=target_viewpoint_id,
                leg_index=MISSION_LEG_INDEX,
                localization_branch_proof_id=LOCALIZATION_BRANCH_PROOF_ID,
                route_csv_path=route_csv,
                diagnostics_path=diagnostics_json,
                map_route_certificate_path=route_certificate_json,
            )

            self.assertEqual(validated, loaded)
            self.assertEqual(
                written_sha256,
                startup_reseal_motion_permit_sha256(loaded),
            )
            self.assertEqual(loaded.leg_index, MISSION_LEG_INDEX)
            self.assertEqual(
                loaded.target_viewpoint_id,
                target_viewpoint_id,
            )
            self.assertEqual(loaded.rejected_run_id, REJECTED_RUN_ID)
            self.assertEqual(loaded.run_id, REPLACEMENT_RUN_ID)
            self.assertEqual(
                loaded.recovery_source_kind,
                STARTUP_RESEAL_RECOVERY_SOURCE_PRESTART_LOCALIZATION_CONTINUITY,
            )
            self.assertFalse(loaded.additional_typed_run_required)


if __name__ == "__main__":
    unittest.main()
