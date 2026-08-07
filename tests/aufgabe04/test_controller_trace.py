from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.navigation.controller_trace import (
    CONTROLLER_TRACE_SCHEMA_VERSION,
    ControllerTraceRecord,
    ControllerTraceWriter,
    append_controller_trace,
    load_controller_traces,
)
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.waypoint_controller import VelocityCommand


class ControllerTraceTests(unittest.TestCase):
    def _record(self, *, timestamp_sec: float = 12.5, event: str = "command"):
        return ControllerTraceRecord(
            timestamp_sec=timestamp_sec,
            event=event,
            fail_closed=False,
            route_revision=4,
            route_kind="stand_discovery_corridor",
            target_index=3,
            pursuit_index=3,
            progress_mode="path_tracking",
            egress_phase="forward",
            map_pose=Pose2D(1.25, -0.75, 0.2),
            odom_pose=Pose2D(0.4, 0.8, -0.1),
            active_segment_start_index=2,
            active_segment_end_index=3,
            distance_to_target_m=0.18,
            pose_distance_to_segment_m=0.012,
            maximum_chord_distance_to_segment_m=0.014,
            tracking_tube_radius_m=0.03,
            nominal_command=VelocityCommand(0.05, -0.12),
            effective_command=VelocityCommand(0.025, -0.12),
            front_clearance={
                "nearest_valid_range_m": 0.27,
                "source": "front_sector",
                "valid_sample_count": 9,
            },
            front_cluster_summary={
                "cluster_count": 2,
                "nearest_cluster_m": 0.27,
                "bearings_rad": [-0.1, 0.2],
            },
            diagnostics={
                "tf": {
                    "map_to_odom_age_sec": 0.04,
                    "odom_to_base_age_sec": 0.02,
                },
                "recovery_attempted": False,
            },
        )

    def test_schema_round_trip_contains_controller_evidence(self):
        record = self._record()
        payload = record.to_payload()

        self.assertEqual(payload["schema_version"], CONTROLLER_TRACE_SCHEMA_VERSION)
        self.assertEqual(payload["route_revision"], 4)
        self.assertEqual(payload["route_kind"], "stand_discovery_corridor")
        self.assertEqual(payload["target_index"], 3)
        self.assertEqual(payload["pursuit_index"], 3)
        self.assertEqual(payload["progress_mode"], "path_tracking")
        self.assertEqual(payload["egress_phase"], "forward")
        self.assertEqual(payload["map_pose"]["x_m"], 1.25)
        self.assertEqual(payload["odom_pose"]["yaw_rad"], -0.1)
        self.assertEqual(payload["active_segment_start_index"], 2)
        self.assertEqual(payload["active_segment_end_index"], 3)
        self.assertEqual(payload["nominal_command"]["linear_x_mps"], 0.05)
        self.assertEqual(payload["effective_command"]["linear_x_mps"], 0.025)
        self.assertEqual(payload["front_clearance"]["valid_sample_count"], 9)
        self.assertEqual(payload["front_cluster_summary"]["cluster_count"], 2)
        self.assertEqual(payload["diagnostics"]["tf"]["map_to_odom_age_sec"], 0.04)
        self.assertEqual(ControllerTraceRecord.from_payload(payload), record)
        with self.assertRaises(FrozenInstanceError):
            record.event = "changed"

    def test_schema_v1_payload_round_trips_without_v2_diagnostics_field(self):
        v2_payload = self._record().to_payload()
        legacy_payload = {
            key: value
            for key, value in v2_payload.items()
            if key != "diagnostics"
        }
        legacy_payload["schema_version"] = 1

        record = ControllerTraceRecord.from_payload(legacy_payload)

        self.assertEqual(record.schema_version, 1)
        self.assertIsNone(record.diagnostics)
        self.assertEqual(record.to_payload(), legacy_payload)
        self.assertNotIn("diagnostics", record.to_payload())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy_controller.jsonl"
            path.write_text(
                json.dumps(legacy_payload, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_controller_traces(path), (record,))

    def test_schema_v1_constructor_rejects_diagnostics(self):
        with self.assertRaisesRegex(ValueError, "schema-v1.*diagnostics"):
            ControllerTraceRecord(
                timestamp_sec=1.0,
                event="stop",
                fail_closed=True,
                diagnostics={"tf_age_sec": 1.2},
                schema_version=1,
            )

    def test_rejects_malformed_v2_diagnostics(self):
        malformed_values = (
            [],
            {"tf_age_sec": math.nan},
            {1: "non-string key"},
            {"unexpected": object()},
        )
        for diagnostics in malformed_values:
            with self.subTest(diagnostics=diagnostics):
                payload = self._record().to_payload()
                payload["diagnostics"] = diagnostics
                with self.assertRaises(ValueError):
                    ControllerTraceRecord.from_payload(payload)

    def test_writer_creates_parents_and_appends_without_replacing_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "controller.jsonl"
            first = self._record(timestamp_sec=1.0)
            second = self._record(timestamp_sec=2.0, event="stop")

            append_controller_trace(path, first)
            first_bytes = path.read_bytes()
            ControllerTraceWriter(path).append(second)

            self.assertTrue(path.read_bytes().startswith(first_bytes))
            self.assertEqual(load_controller_traces(path), (first, second))
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            for line in lines:
                payload = json.loads(line)
                self.assertEqual(
                    line,
                    json.dumps(payload, separators=(",", ":"), sort_keys=True),
                )

    def test_fail_closed_stop_allows_unavailable_numeric_measurements(self):
        record = ControllerTraceRecord(
            timestamp_sec=4.0,
            event="stop",
            reason="map-to-base transform unavailable",
            fail_closed=True,
            route_kind="stand_discovery_corridor",
            progress_mode="sensor_gate",
            map_pose=None,
            odom_pose=None,
            nominal_command=None,
            effective_command=VelocityCommand(0.0, 0.0),
            front_clearance={"nearest_valid_range_m": None, "sample_count": 0},
            front_cluster_summary=None,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stop.jsonl"
            append_controller_trace(path, record)
            text = path.read_text(encoding="utf-8")

            self.assertNotIn("NaN", text)
            self.assertNotIn("Infinity", text)
            payload = json.loads(text)
            self.assertIsNone(payload["map_pose"])
            self.assertIsNone(payload["pose_distance_to_segment_m"])
            self.assertIsNone(payload["front_clearance"]["nearest_valid_range_m"])
            self.assertEqual(load_controller_traces(path), (record,))

    def test_route_tube_stop_preserves_exact_observed_value(self):
        observed_m = 0.03154134331426062
        record = ControllerTraceRecord(
            timestamp_sec=17.25,
            event="stop",
            reason="pose left certified route tube",
            fail_closed=True,
            route_revision=7,
            route_kind="stand_discovery_corridor",
            target_index=5,
            pursuit_index=5,
            progress_mode="path_tracking",
            map_pose=Pose2D(0.2, 0.3, -0.4),
            active_segment_start_index=4,
            active_segment_end_index=5,
            pose_distance_to_segment_m=observed_m,
            maximum_chord_distance_to_segment_m=observed_m,
            tracking_tube_radius_m=0.03,
            nominal_command=VelocityCommand(0.04, 0.01),
            effective_command=VelocityCommand(0.0, 0.0),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "route_tube_stop.jsonl"
            append_controller_trace(path, record)
            payload = json.loads(path.read_text(encoding="utf-8"))
            loaded = load_controller_traces(path)[0]

        self.assertEqual(payload["pose_distance_to_segment_m"], observed_m)
        self.assertEqual(loaded.pose_distance_to_segment_m, observed_m)
        self.assertEqual(loaded.tracking_tube_radius_m, 0.03)
        self.assertGreater(
            loaded.pose_distance_to_segment_m,
            loaded.tracking_tube_radius_m,
        )

    def test_rejects_non_finite_or_structurally_invalid_values(self):
        invalid_factories = (
            lambda: ControllerTraceRecord(math.nan, "command", False),
            lambda: ControllerTraceRecord(
                1.0,
                "command",
                False,
                map_pose=Pose2D(math.inf, 0.0, 0.0),
            ),
            lambda: ControllerTraceRecord(
                1.0,
                "command",
                False,
                nominal_command=VelocityCommand(0.1, -math.inf),
            ),
            lambda: ControllerTraceRecord(
                1.0,
                "command",
                False,
                distance_to_target_m=math.nan,
            ),
            lambda: ControllerTraceRecord(
                1.0,
                "command",
                False,
                pose_distance_to_segment_m=-0.01,
            ),
            lambda: ControllerTraceRecord(
                1.0,
                "command",
                False,
                active_segment_start_index=2,
                active_segment_end_index=None,
            ),
            lambda: ControllerTraceRecord(
                1.0,
                "command",
                False,
                front_cluster_summary={"nearest": math.inf},
            ),
        )
        for factory in invalid_factories:
            with self.subTest(factory=factory):
                with self.assertRaises(ValueError):
                    factory()

    def test_loader_rejects_non_standard_numbers_and_schema_drift(self):
        payload = self._record().to_payload()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.jsonl"
            path.write_text('{"timestamp_sec":NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                load_controller_traces(path)

            payload["unexpected"] = True
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fields mismatch"):
                load_controller_traces(path)


if __name__ == "__main__":
    unittest.main()
