from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.foundation.ros_runtime_config import (
    RuntimeConfig, resolve_runtime_config,
)
from scripts.aufgabe04.navigation.localization.ros_preflight import RosPreflightResult
from scripts.aufgabe04.navigation.planning.waypoint_csv import RouteWaypoint, SelectedRouteLeg
from scripts.aufgabe04.navigation.station_segment import runtime
from scripts.aufgabe04.navigation.station_segment.time_budget_admission import (
    admit_route_time_budget, station_motion_configs,
)


class StationSegmentTimeBudgetTest(unittest.TestCase):
    def arguments(self, *extra):
        return runtime.build_parser().parse_args(["--leg-index", "0", *extra])

    def preflight(self, *, frame="map", yaw=2.2):
        pose = dict(frame_id=frame, child_frame_id="base_footprint",
                    x_m=0.0, y_m=0.0, yaw_rad=yaw)
        return RosPreflightResult(
            ok=True, failures=[], observations=[], runtime_config={},
            route_pose=pose if frame == "map" else None,
            odom_pose=pose if frame == "odom" else None,
        )

    def admission(self, args, *, waypoints=None, resolved=None, preflight=None, egress=None):
        controller, smoothing = station_motion_configs(args, "detected_stand_preapproach")
        return admit_route_time_budget(
            args=args, resolved=resolved or resolve_runtime_config(RuntimeConfig()),
            route_kind="detected_stand_preapproach",
            waypoints=waypoints or (Pose2D(0, 0), Pose2D(1.725, 0, 0)),
            preflight=preflight or self.preflight(),
            controller=controller, smoothing=smoothing, egress_certificate=egress,
        )

    def test_automatic_approach_budget_includes_actual_initial_heading(self):
        args = self.arguments()
        self.assertIsNone(args.waypoint_timeout_sec)
        evidence = self.admission(args)
        target = evidence["targets"][0]
        self.assertGreater(target["timeout_sec"], 55.0)
        self.assertLess(target["timeout_sec"], 120.0)
        self.assertAlmostEqual(target["alignment_turn_rad"], 2.2)
        self.assertFalse(evidence["motion_authorized"])

    def test_explicit_deadline_is_honored_or_rejected(self):
        with self.assertRaisesRegex(ValueError, "insufficient"):
            self.admission(self.arguments("--waypoint-timeout-sec", "45"))
        evidence = self.admission(self.arguments("--waypoint-timeout-sec", "90"))
        self.assertEqual(evidence["targets"][0]["timeout_sec"], 90.0)

    def test_odom_execution_requires_matching_pose_and_uses_it(self):
        args = self.arguments("--execution-pose-frame", "odom")
        with self.assertRaisesRegex(ValueError, "did not provide odom pose"):
            self.admission(args)
        evidence = self.admission(args, preflight=self.preflight(frame="odom", yaw=0.0))
        self.assertEqual(evidence["execution_frame"], "odom")
        self.assertEqual(evidence["targets"][0]["alignment_turn_rad"], 0.0)
        bad = replace(self.preflight(frame="odom"), odom_pose={
            **self.preflight(frame="odom").odom_pose, "frame_id": "map",
        })
        with self.assertRaisesRegex(ValueError, "frame identity mismatch"):
            self.admission(args, preflight=bad)

    def test_simulation_keeps_legacy_timing_and_egress_uses_lower_speed(self):
        self.assertEqual(self.admission(self.arguments(), resolved=resolve_runtime_config(
            RuntimeConfig(use_sim_time=True))), {})
        ordinary = self.admission(self.arguments())["targets"][0]["timeout_sec"]
        egress = self.admission(self.arguments(), egress=SimpleNamespace(
            required=True, waypoint_index=1))["targets"][0]["timeout_sec"]
        self.assertGreater(egress, ordinary)

    def run_station(self, root, *, length=1.725, extra=(), dry=True):
        route = root / "route.csv"
        points = (RouteWaypoint(0, 0, Pose2D(0, 0), 0.0),
                  RouteWaypoint(0, 1, Pose2D(length, 0, 0), length))
        leg = SelectedRouteLeg(route, 0, points, points, length, 0.0,
                               route_kind="detected_stand_preapproach")
        diagnostics = root / "diagnostics.json"
        admission = (route, diagnostics, None, leg, SimpleNamespace(metadata={}),
                     "", None, False, None, None)
        argv = ["--route-csv", str(route), "--diagnostics-json", str(diagnostics),
                "--leg-index", "0", "--run-id", "budget-test",
                "--results-csv", str(root / "results.csv"),
                "--semantic-log", str(root / "events.jsonl"),
                "--preflight-json", str(root / "preflight.json"), *extra]
        if dry:
            argv.append("--dry-run")
        with ExitStack() as stack:
            stack.enter_context(patch.object(runtime, "admit_execution_route", return_value=admission))
            stack.enter_context(patch.object(runtime, "run_ros_preflight", return_value=self.preflight()))
            stack.enter_context(patch.object(runtime, "_prompt_for_initialpose"))
            follower = stack.enter_context(patch.object(runtime, "run_simple_waypoint_follower"))
            permit = stack.enter_context(patch.object(runtime, "_validated_mission_leg_motion_permit"))
            confirm = stack.enter_context(patch.object(runtime, "_confirm_motion"))
            stack.enter_context(redirect_stdout(StringIO()))
            status = runtime.main(argv)
            follower.assert_not_called()
            permit.assert_not_called()
            confirm.assert_not_called()
        events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
        return status, events, json.loads((root / "preflight.json").read_text())

    def test_dry_run_persists_admitted_budgets_before_success(self):
        with tempfile.TemporaryDirectory() as directory:
            status, events, preflight = self.run_station(Path(directory))
        self.assertEqual(status, 0)
        names = [event["event"] for event in events]
        self.assertLess(names.index("route_time_budget_admitted"), names.index("dry_run_completed"))
        self.assertGreater(preflight["route_time_budget"]["targets"][0]["timeout_sec"], 55.0)
        self.assertTrue(preflight["ok"])

    def test_insufficient_override_and_over_cap_route_fail_before_motion(self):
        for length, extra, dry in ((1.725, ("--waypoint-timeout-sec", "45"), False),
                                   (6.0, (), True)):
            with self.subTest(length=length), tempfile.TemporaryDirectory() as directory:
                status, events, preflight = self.run_station(
                    Path(directory), length=length, extra=extra, dry=dry)
            self.assertEqual(status, 1)
            names = [event["event"] for event in events]
            self.assertIn("route_time_budget_rejected", names)
            self.assertIn("preflight_failed", names)
            self.assertNotIn("dry_run_completed", names)
            self.assertNotIn("motion_started", names)
            self.assertFalse(preflight["ok"])
            self.assertTrue(preflight["route_time_budget"]["fail_closed"])
            self.assertEqual(events[-1]["final_status"], "preflight_failed")


if __name__ == "__main__":
    unittest.main()
