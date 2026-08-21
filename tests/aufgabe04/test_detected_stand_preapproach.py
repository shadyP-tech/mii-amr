from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from scripts.aufgabe04.navigation.detected_stand_preapproach import (
    DETECTED_STAND_PREAPPROACH_ROUTE_KIND,
    seal_detected_stand_preapproach,
    validate_detected_stand_preapproach_binding,
)
from scripts.aufgabe04.navigation.plan_detected_stand_exploration import main as plan
from scripts.aufgabe04.navigation.ros_preflight import RosPreflightResult
from scripts.aufgabe04.navigation.run_single_station_segment import (
    main as run_segment,
)
from scripts.aufgabe04.navigation.waypoint_csv import load_route_leg
from scripts.aufgabe04.perception.stand_observation import (
    write_observation_jsonl,
)
from tests.aufgabe04.test_detected_station_exploration import (
    observation,
    write_free_map,
)


class DetectedStandPreapproachTest(unittest.TestCase):
    def _pipeline(self, root: Path) -> Path:
        map_yaml = write_free_map(root)
        observations = root / "stand_observations.jsonl"
        write_observation_jsonl(
            observations,
            (
                observation(1, x=0.5, y=0.5, observed_at=1.0, map_yaml=map_yaml),
                observation(2, x=0.51, y=0.5, observed_at=2.0, map_yaml=map_yaml),
                observation(3, x=0.49, y=0.5, observed_at=3.0, map_yaml=map_yaml),
            ),
        )
        status = plan(
            [
                "--observations-jsonl",
                str(observations),
                "--map",
                str(map_yaml),
                "--start-x",
                "0.0",
                "--start-y",
                "0.0",
                "--approach-bearing-mode",
                "robot-to-stand",
                "--approach-offset-m",
                "0.32",
                "--candidate-transit-radius-m",
                "0.31",
                "--inflation-radius-m",
                "0.25",
                "--tracking-margin-m",
                "0.03",
                "--lidar-stop-distance-m",
                "0.20",
                "--enforce-physical-clearance",
                "--exploration-state-json",
                str(root / "exploration_state.json"),
                "--layout-json",
                str(root / "layout.json"),
                "--layout-csv",
                str(root / "layout.csv"),
                "--route-csv",
                str(root / "route.csv"),
                "--diagnostics-json",
                str(root / "route_diagnostics.json"),
                "--candidate-snapshot-json",
                str(root / "candidate_snapshot.json"),
            ]
        )
        self.assertEqual(status, 0)
        (root / "pipeline_summary.json").write_text(
            json.dumps(
                {
                    "status": "observe_and_plan_complete",
                    "motion_published": False,
                }
            )
        )
        return root

    def test_seals_robot_facing_route_and_validates_binding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._pipeline(Path(tmpdir))
            outputs = seal_detected_stand_preapproach(pipeline_root=root)
            route_path = Path(outputs["route_csv"])
            diagnostics_path = Path(outputs["diagnostics_json"])
            certificate_path = Path(outputs["route_certificate_json"])

            leg = load_route_leg(route_path, 0)
            status = validate_detected_stand_preapproach_binding(
                diagnostics_path,
                leg,
                candidate_snapshot_path=root / "candidate_snapshot.json",
            )

            self.assertTrue(status.ok, status.failures)
            self.assertEqual(leg.route_kind, DETECTED_STAND_PREAPPROACH_ROUTE_KIND)
            self.assertFalse(leg.simulation_only)
            self.assertTrue(leg.raw_waypoints[-1].protected)
            self.assertTrue(leg.raw_waypoints[-1].corridor)
            self.assertTrue(math.isfinite(leg.raw_waypoints[-1].pose.yaw_rad))
            self.assertTrue(certificate_path.exists())

    def test_route_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._pipeline(Path(tmpdir))
            outputs = seal_detected_stand_preapproach(pipeline_root=root)
            route_path = Path(outputs["route_csv"])
            with route_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
                fieldnames = list(rows[0])
            rows[-1]["world_x_m"] = "0.0"
            with route_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            leg = load_route_leg(route_path, 0)

            status = validate_detected_stand_preapproach_binding(
                Path(outputs["diagnostics_json"]),
                leg,
                candidate_snapshot_path=root / "candidate_snapshot.json",
            )

            self.assertFalse(status.ok)
            self.assertTrue(
                any("SHA-256" in failure for failure in status.failures),
                status.failures,
            )

    def test_source_connector_tamper_is_rejected_before_sealing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._pipeline(Path(tmpdir))
            diagnostics_path = root / "route_diagnostics.json"
            payload = json.loads(diagnostics_path.read_text())
            payload["metadata"]["exact_start_connector"]["exact_start"][
                "x_m"
            ] += 0.01
            diagnostics_path.write_text(json.dumps(payload))

            with self.assertRaisesRegex(
                ValueError,
                "exact-start evidence differs from route waypoint 0",
            ):
                seal_detected_stand_preapproach(pipeline_root=root)

    def test_sealed_connector_tamper_fails_binding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._pipeline(Path(tmpdir))
            outputs = seal_detected_stand_preapproach(pipeline_root=root)
            diagnostics_path = Path(outputs["diagnostics_json"])
            payload = json.loads(diagnostics_path.read_text())
            payload["metadata"]["exact_start_connector"]["anchor"][
                "y_m"
            ] += 0.01
            diagnostics_path.write_text(json.dumps(payload))
            leg = load_route_leg(Path(outputs["route_csv"]), 0)

            status = validate_detected_stand_preapproach_binding(
                diagnostics_path,
                leg,
                candidate_snapshot_path=root / "candidate_snapshot.json",
            )

            self.assertFalse(status.ok)
            self.assertTrue(
                any("exact-start anchor" in failure for failure in status.failures),
                status.failures,
            )

    def test_real_runner_accepts_sealed_route_in_dry_run_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self._pipeline(Path(tmpdir))
            outputs = seal_detected_stand_preapproach(pipeline_root=root)
            leg = load_route_leg(Path(outputs["route_csv"]), 0)
            with patch(
                "scripts.aufgabe04.navigation.run_single_station_segment."
                "run_ros_preflight",
                return_value=RosPreflightResult(
                    ok=True,
                    failures=[],
                    observations=[],
                    runtime_config={},
                    route_pose={
                        "frame_id": "map",
                        "child_frame_id": "base_footprint",
                        "x_m": leg.raw_waypoints[0].pose.x_m,
                        "y_m": leg.raw_waypoints[0].pose.y_m,
                        "yaw_rad": 0.0,
                    },
                ),
            ), patch(
                "scripts.aufgabe04.navigation.run_single_station_segment."
                "run_simple_waypoint_follower"
            ) as follower, redirect_stdout(StringIO()):
                status = run_segment(
                    [
                        "--route-csv",
                        outputs["route_csv"],
                        "--diagnostics-json",
                        outputs["diagnostics_json"],
                        "--route-certificate-json",
                        outputs["route_certificate_json"],
                        "--candidate-snapshot",
                        str(root / "candidate_snapshot.json"),
                        "--leg-index",
                        "0",
                        "--semantic-log",
                        str(root / "dry_run_events.jsonl"),
                        "--results-csv",
                        str(root / "dry_run_results.csv"),
                        "--preflight-json",
                        str(root / "preflight.json"),
                        "--dry-run",
                    ]
                )

            self.assertEqual(status, 0)
            follower.assert_not_called()
            results = (root / "dry_run_results.csv").read_text()
            self.assertIn("dry_run_ok", results)
            self.assertIn("False", results)


if __name__ == "__main__":
    unittest.main()
