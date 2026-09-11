from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.aufgabe04.navigation.foundation.content_hashed_evidence import payload_sha256
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D, pose_route_sha256, transform_map_route_to_odom,
)
from scripts.aufgabe04.navigation.localization.ros_preflight import RosObservation, RosPreflightResult
from scripts.aufgabe04.navigation.localization.startup_route_admission import (
    OdomStartupRouteAdmissionRejected, ODOM_STARTUP_ROUTE_REJECTION_REASON,
    build_odom_startup_route_admission_evidence, evaluate_odom_startup_route_rejection,
)
from scripts.aufgabe04.navigation.station_segment import localization_admission, runtime, execution_route_admission
from scripts.aufgabe04.navigation.planning.waypoint_csv import RouteWaypoint, SelectedRouteLeg
from scripts.aufgabe04.navigation.control.follower_models import FollowerResult
from scripts.aufgabe04.navigation.station_segment.reporting import build_odom_execution_admission_stop_details
from scripts.aufgabe04.navigation.waypoint_follower.startup import certified_static_startup_decision

FIXTURE = Path(__file__).parent / "fixtures/startup_route_admission_20260911/inputs.json"


def _route_pose(raw):
    return Pose2D(raw["x_m"], raw["y_m"], math.nan if raw["yaw_rad"] is None else raw["yaw_rad"])


def recorded_evidence(stage="execute"):
    fixture = json.loads(FIXTURE.read_text())
    source = fixture["stages"][stage]
    raw = source["preflight"]
    return build_odom_startup_route_admission_evidence(
        map_route=tuple(_route_pose(pose) for pose in fixture["map_route"]),
        odom_pose=raw["odom_pose"], chained_map_pose=raw["route_pose"],
        map_from_odom=raw["map_from_odom"],
        pose_tf_observations={
            label: next(observation["data"] for observation in raw["observations"]
                        if observation["name"] == f"tf {frame}->base_footprint")
            for label, frame in (("map", "map"), ("odom", "odom"))
        },
        map_frame="map", odom_frame="odom", base_frame="base_footprint",
        tracking_tube_radius_m=fixture["tracking_tube_radius_m"],
        max_tf_age_sec=fixture["max_tf_age_sec"],
        max_composition_yaw_error_rad=.03,
        source_preflight_sha256=source["source_preflight_sha256"],
        source_map_execution_certificate_sha256=fixture["source_map_execution_certificate_sha256"],
    )


def recorded_stop_details(*, dry_run=False):
    return OdomStartupRouteAdmissionRejected(evidence=recorded_evidence(), dry_run=dry_run).to_stop_details()


def decision_for(details, **overrides):
    args = dict(status="preflight_failed", motion_published=False,
                stop_reason=ODOM_STARTUP_ROUTE_REJECTION_REASON, stop_details=details)
    args.update(overrides)
    return evaluate_odom_startup_route_rejection(**args)


class OdomStartupRouteAdmissionTests(unittest.TestCase):
    def test_recorded_dry_accepted_execute_rejected_without_relaxing_radius(self):
        fixture = json.loads(FIXTURE.read_text())
        for stage, expected in (("dry", True), ("execute", False)):
            with self.subTest(stage=stage):
                evidence = recorded_evidence(stage)
                raw = evidence["odom_pose"]
                pose = Pose2D(**{key: raw[key] for key in ("x_m", "y_m", "yaw_rad")})
                route = tuple(_route_pose(point) for point in evidence["transformed_odom_route"])
                result = certified_static_startup_decision(pose, route, tracking_tube_radius_m=.03)
                self.assertEqual(result.ok, expected)
                self.assertAlmostEqual(result.route_check.pose_distance_to_segment_m,
                                       fixture["stages"][stage]["expected_distance_m"], places=14)
                self.assertEqual(evidence["tracking_tube_radius_m"], .03)
        with self.assertRaisesRegex(ValueError, "geometry_not_startup_corridor_mismatch"):
            OdomStartupRouteAdmissionRejected(evidence=recorded_evidence("dry"), dry_run=True)
        self.assertTrue(decision_for(recorded_stop_details()).eligible)

    def test_blank_intermediate_route_yaw_round_trips_as_json_null(self):
        details = recorded_stop_details()
        evidence = details["startup_route_admission"]
        for key in ("map_route", "transformed_odom_route", "first_transformed_segment"):
            self.assertIsNone(evidence[key][0]["yaw_rad"])
            self.assertTrue(math.isnan(_route_pose(evidence[key][0]).yaw_rad))
        restored = json.loads(json.dumps(details, allow_nan=False))
        self.assertTrue(decision_for(restored).eligible)
        route = tuple(_route_pose(raw) for raw in evidence["map_route"])
        self.assertEqual(evidence["source_map_route_sha256"], pose_route_sha256(route))
        for key in ("odom_pose", "chained_map_pose", "composed_map_pose", "map_from_odom"):
            bad = deepcopy(details)
            bad["startup_route_admission"][key]["yaw_rad"] = None
            bad["startup_route_admission_sha256"] = payload_sha256(bad["startup_route_admission"])
            with self.subTest(key=key):
                self.assertFalse(decision_for(bad).eligible)
        missing = deepcopy(details)
        del missing["startup_route_admission"]["map_route"][0]["yaw_rad"]
        missing["startup_route_admission_sha256"] = payload_sha256(missing["startup_route_admission"])
        self.assertFalse(decision_for(missing).eligible)

    def test_full_preflight_inputs_match_recorded_canonical_hash(self):
        for source in json.loads(FIXTURE.read_text())["stages"].values():
            self.assertEqual(payload_sha256(source["preflight"]), source["source_preflight_sha256"])

    def test_only_explicit_typed_failure_and_no_motion_are_eligible(self):
        details = recorded_stop_details()
        for override in (
            {"status": "stopped"}, {"status": "preflight_unavailable"},
            {"motion_published": True}, {"motion_published": None}, {"motion_published": 0},
            {"stop_reason": "odom execution admission failed: generic failure"},
            {"stop_details": None}, {"stop_details": {}},
        ):
            with self.subTest(override=override):
                self.assertFalse(decision_for(details, **override).eligible)
        for key in ("motion_published", "motion_authorization_consumed", "follower_started", "motion_history_uncertain"):
            for value in (True, None, 0):
                mutated = deepcopy(details); mutated[key] = value
                with self.subTest(key=key, value=value):
                    self.assertFalse(decision_for(mutated).eligible)
        for value in (None, "false", 0):
            mutated = deepcopy(details); mutated["dry_run"] = value
            self.assertFalse(decision_for(mutated).eligible)
        self.assertTrue(decision_for(recorded_stop_details(dry_run=True)).eligible)

    def test_tampered_geometry_hash_provenance_and_nonfinite_values_reject(self):
        original = recorded_stop_details()
        changes = (
            lambda e: e["odom_pose"].update(x_m=float("nan")),
            lambda e: e["odom_pose"].update(x_m=float("inf")),
            lambda e: e["odom_pose"].update(frame_id="map"),
            lambda e: e["map_from_odom"].update(source_frame="base_footprint"),
            lambda e: e["pose_tf_observations"]["odom"].update(stamp_sec=-1),
            lambda e: e["pose_tf_observations"]["odom"].update(available=False),
            lambda e: e["pose_tf_observations"]["odom"].update(x_m=0),
            lambda e: e["pose_tf_observations"]["odom"].update(age_sec=100),
            lambda e: e["map_route"][0].update(x_m=0),
            lambda e: e["transformed_odom_route"][0].update(x_m=0),
            lambda e: e["route_check"].update(pose_distance_to_segment_m=.031),
            lambda e: e["route_check"].update(active_segment_start_index=False),
            lambda e: e.update(source_map_execution_certificate_sha256=""),
            lambda e: e.update(first_transformed_segment=[]),
            lambda e: e.update(schema_version=True),
            lambda e: e.update(tracking_tube_radius_m=.035),
        )
        for index, mutate in enumerate(changes):
            with self.subTest(index=index):
                details = deepcopy(original); mutate(details["startup_route_admission"])
                try:
                    details["startup_route_admission_sha256"] = payload_sha256(details["startup_route_admission"])
                except (TypeError, ValueError):
                    pass
                self.assertFalse(decision_for(details).eligible)
        mutated = deepcopy(original); mutated["startup_route_admission_sha256"] = "0" * 64
        self.assertFalse(decision_for(mutated).eligible)

    def test_freshness_and_composition_failures_cannot_be_relabelled_corridor_mismatch(self):
        for defect in ("stale", "future", "position", "yaw"):
            with self.subTest(defect=defect):
                details = recorded_stop_details()
                evidence = details["startup_route_admission"]
                if defect in ("stale", "future"):
                    raw = evidence["pose_tf_observations"]["odom"]
                    age = 2.0 if defect == "stale" else -2.0
                    raw["stamp_sec"] = raw["capture_time_sec"] - age
                    raw["age_sec"] = age
                else:
                    key = "x_m" if defect == "position" else "yaw_rad"
                    evidence["chained_map_pose"][key] += 0.5
                    evidence["pose_tf_observations"]["map"][key] += 0.5
                details["startup_route_admission_sha256"] = payload_sha256(evidence)
                decision = decision_for(details)
                self.assertFalse(decision.eligible)
                self.assertEqual(decision.reason, "tf_not_fresh" if defect in ("stale", "future")
                                 else "pose_composition_outside_admission_bounds")

    def test_mutating_returned_details_cannot_change_exception_evidence(self):
        error = OdomStartupRouteAdmissionRejected(evidence=recorded_evidence(), dry_run=False)
        original = error.to_stop_details(); mutated = error.to_stop_details()
        mutated["startup_route_admission"]["route_check"]["tracking_tube_radius_m"] = 9
        self.assertEqual(error.to_stop_details(), original)

    def test_producer_preserves_recorded_inputs_and_reporting_type(self):
        fixture = json.loads(FIXTURE.read_text()); raw = fixture["stages"]["execute"]["preflight"]
        preflight = RosPreflightResult(**{
            **raw, "observations": [RosObservation(**row) for row in raw["observations"]],
        })
        args = SimpleNamespace(certified_route_tube_radius_m=.03,
                               max_stationary_amcl_yaw_spread_rad=.03,
                               uncertainty_robot_radius_m=.105, dry_run=False, max_tf_age_sec=1.0)
        resolved = SimpleNamespace(map_frame="map", odom_frame="odom", base_frame="base_footprint")
        route = tuple(_route_pose(pose) for pose in fixture["map_route"])
        with patch.object(localization_admission, "poses_from_waypoints", return_value=route), \
             patch.object(localization_admission, "_resolved_map_execution_certificate",
                          return_value=(None, fixture["source_map_execution_certificate_sha256"])), \
             patch.object(localization_admission, "publish_route_uncertainty_budget") as publish, \
             self.assertRaises(OdomStartupRouteAdmissionRejected) as raised:
            localization_admission._build_odom_execution_admission(
                args=args, resolved=resolved, leg=SimpleNamespace(executable_waypoints=[]),
                preflight=preflight, diagnostics_snapshot=object(),
            )
        publish.assert_not_called()
        details = build_odom_execution_admission_stop_details(raised.exception)
        self.assertEqual(details, recorded_stop_details())
        self.assertTrue(decision_for(details).eligible)
        generic = build_odom_execution_admission_stop_details(ValueError(str(raised.exception)))
        self.assertFalse(decision_for(generic).eligible)

    def test_runtime_uses_odom_gate_and_records_identity_before_any_motion_boundary(self):
        fixture = json.loads(FIXTURE.read_text()); raw = fixture["stages"]["execute"]["preflight"]
        preflight = RosPreflightResult(**{
            **raw, "observations": [RosObservation(**row) for row in raw["observations"]],
        })
        for mode, kind, skip_map, dry_run in (("odom", "candidate_preapproach", True, False),
                                     ("odom", "opposite_face", True, False),
                                     ("odom", "candidate_preapproach", True, True),
                                     ("odom", "coverage", False, False),
                                     ("map", "candidate_preapproach", False, False)):
            with self.subTest(mode=mode, kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                points = tuple(RouteWaypoint(0, index, _route_pose(pose), index * .086076)
                               for index, pose in enumerate(fixture["map_route"]))
                leg = SelectedRouteLeg(root / "route.csv", 0, points, points, .086076, 0,
                                       route_kind="detected_stand_preapproach")
                diagnostics = SimpleNamespace(metadata={"route_certificate_path": str(root / "certificate.json")})
                args = ["--route-csv", str(root / "route.csv"), "--diagnostics-json", str(root / "diagnostics.json"),
                        "--results-csv", str(root / "results.csv"), "--semantic-log", str(root / "events.jsonl"),
                        "--preflight-json", str(root / "preflight.json"), "--run-id", "candidate-000",
                        "--leg-index", "0", "--execution-pose-frame", mode,
                        "--odom-execution-certificate-json", str(root / "odom_certificate.json"),
                        "--uncertainty-budget-json", str(root / "uncertainty.json"),
                        "--uncertainty-map-yaml", str(root / "map.yaml"),
                        "--uncertainty-robot-radius-m", "0.105", "--localization-branch-proof-id", "known-start",
                        "--max-stationary-amcl-yaw-spread-rad", "0.03", "--mission-leg-evidence-kind", kind,
                        "--mission-leg-evidence-index", "0", "--mission-leg-evidence-target-id", "survey_candidate_0003"]
                if dry_run:
                    relative_root = Path(os.path.relpath(root))
                    args = [arg.replace(str(root), str(relative_root)) for arg in args] + ["--dry-run"]
                map_failure = FollowerResult("stopped", "map startup rejected", 0, 0, False, {})
                with patch.object(runtime, "admit_execution_route", return_value=(
                    root / "route.csv", root / "diagnostics.json", None, leg, diagnostics,
                    "candidate", None, False, None, None,
                )), patch.object(runtime, "run_ros_preflight", return_value=preflight), \
                     patch.object(runtime, "_static_start_preflight_rejection", return_value=map_failure) as map_gate, \
                     patch.object(localization_admission, "_resolved_map_execution_certificate", return_value=(
                         None, fixture["source_map_execution_certificate_sha256"],
                     )), patch.object(runtime, "_validated_mission_leg_motion_permit") as validate_permit, \
                     patch.object(runtime, "consume_mission_leg_motion_permit") as consume, \
                     patch.object(runtime, "run_simple_waypoint_follower") as follower, \
                     patch.object(runtime, "_confirm_motion") as confirm, redirect_stdout(StringIO()):
                    result = runtime.main(args)
                self.assertEqual(result, 1)
                validate_permit.assert_not_called(); consume.assert_not_called()
                follower.assert_not_called(); confirm.assert_not_called()
                if not skip_map:
                    map_gate.assert_called_once()
                    continue
                map_gate.assert_not_called()
                events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
                rejection_chain = events[-3:]
                self.assertEqual([event["event"] for event in rejection_chain],
                                 ["odom_execution_admission_failed", "safety_stop", "run_finished"])
                for event in rejection_chain:
                    self.assertEqual(event["status"], "preflight_failed")
                    self.assertEqual(event["mission_leg_kind"], kind)
                    self.assertEqual(event["mission_leg_index"], 0)
                    self.assertEqual(event["target_id"], "survey_candidate_0003")
                    self.assertIs(event["motion_published"], False)
                    self.assertIs(event["dry_run"], dry_run)
                    self.assertEqual(event["stop_details"], recorded_stop_details(dry_run=dry_run))
                self.assertEqual(rejection_chain[-1]["map_route_certificate_json_path"], str((root / "certificate.json").resolve()))
                self.assertEqual(rejection_chain[-1]["preflight_json_path"], str((root / "preflight.json").resolve()))
                self.assertEqual(json.loads((root / "preflight.json").read_text()), raw)

    def test_relative_dry_run_source_metadata_is_canonical_without_rewriting_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            relative = Path(os.path.relpath(root))
            argv = ["--route-csv", str(relative / "route.csv"),
                    "--diagnostics-json", str(relative / "diagnostics.json"),
                    "--preflight-json", str(relative / "preflight.json"),
                    "--semantic-log", str(root / "events.jsonl"), "--leg-index", "0", "--dry-run"]
            with patch.object(execution_route_admission, "_load_execution_route_leg",
                              side_effect=RuntimeError("stop after metadata")) as load, \
                 self.assertRaisesRegex(RuntimeError, "stop after metadata"), redirect_stdout(StringIO()):
                runtime.main(argv)
            event = json.loads((root / "events.jsonl").read_text().splitlines()[0])
            self.assertEqual(event["event"], "run_started")
            self.assertIs(event["dry_run"], True)
            for field, name in (("route_csv", "route.csv"), ("authoritative_route_csv", "route.csv"),
                                ("diagnostics_json", "diagnostics.json"),
                                ("authoritative_diagnostics_json", "diagnostics.json"),
                                ("preflight_json_path", "preflight.json")):
                self.assertEqual(event[field], str((root / name).resolve()))
            self.assertEqual(load.call_args.args[0], relative / "route.csv")


if __name__ == "__main__":
    unittest.main()
