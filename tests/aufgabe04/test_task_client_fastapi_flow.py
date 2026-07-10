import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.logistics.server_validation.validators import (  # noqa: E402
    build_server_task_snapshot,
    validate_server_task,
)
from scripts.aufgabe04.run_logistics_mission import main as run_logistics_main  # noqa: E402
from scripts.aufgabe04.task_client.server_response_decoder import (  # noqa: E402
    decode_robot_plans,
    decode_robot_statuses,
)


def _sample_payloads(now):
    status = [
        {
            "robot_id": "Robot_Test_01",
            "mission_id": "M-00001",
            "state": "GO_TO_DEPOT_DROPOFF",
            "target": "DEPOT_01",
            "last_qr": "START",
            "cargo": "PROCESSED_MATERIAL",
            "completed_jobs": 0,
            "score": -2,
            "penalties": -2,
            "last_seen_at": (now - timedelta(seconds=5)).isoformat().replace("+00:00", "Z"),
            "charging_visits": 0,
        }
    ]
    plans = [
        {
            "robot_id": "Robot_Test_01",
            "mode": "random",
            "processing_sequence": ["PROC_04", "PROC_08"],
            "plan_steps": ["PROC_04", "DEPOT_PICKUP", "PROC_08"],
            "expanded_path": ["START", "PROC_04", "DEPOT_01", "PROC_08", "START"],
            "qr_mappings": [
                {
                    "robot_id": "Robot_Test_01",
                    "qr_code_id": "QR_001",
                    "station_id": "DEPOT_01",
                    "station_type": "depot",
                    "display_name": "Material Depot",
                },
                {
                    "robot_id": "Robot_Test_01",
                    "qr_code_id": "QR_007",
                    "station_id": "PROC_04",
                    "station_type": "processing",
                    "display_name": "Processing Station 4",
                },
            ],
            "next_job_index": 0,
            "next_step_index": 0,
            "generated_at": (now - timedelta(seconds=10)).isoformat().replace("+00:00", "Z"),
        }
    ]
    return status, plans


class TaskClientFastApiFlowTest(unittest.TestCase):
    def test_decodes_and_validates_qr_mapping(self):
        now = datetime(2026, 6, 29, 12, 50, tzinfo=timezone.utc)
        status_payload, plans_payload = _sample_payloads(now)
        statuses = decode_robot_statuses(status_payload)
        plans = decode_robot_plans(plans_payload)

        snapshot = build_server_task_snapshot(
            robot_id="Robot_Test_01",
            scanned_qr_id="QR_001",
            statuses=statuses,
            plans=plans,
        )
        validated = validate_server_task(
            snapshot,
            local_station_ids=["DEPOT_01", "PROC_04", "PROC_08", "START"],
            now=now,
        )

        self.assertEqual(snapshot.resolved_station_id, "DEPOT_01")
        self.assertEqual(validated.target_station, "DEPOT_01")
        self.assertEqual(validated.to_navigation_request().target_station_id, "DEPOT_01")

    def test_rejects_unknown_qr(self):
        now = datetime(2026, 6, 29, 12, 50, tzinfo=timezone.utc)
        status_payload, plans_payload = _sample_payloads(now)

        with self.assertRaisesRegex(ValueError, "unknown QR"):
            build_server_task_snapshot(
                robot_id="Robot_Test_01",
                scanned_qr_id="QR_999",
                statuses=decode_robot_statuses(status_payload),
                plans=decode_robot_plans(plans_payload),
            )

    def test_rejects_stale_status(self):
        now = datetime(2026, 6, 29, 12, 50, tzinfo=timezone.utc)
        status_payload, plans_payload = _sample_payloads(now)
        status_payload[0]["last_seen_at"] = (now - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
        snapshot = build_server_task_snapshot(
            robot_id="Robot_Test_01",
            scanned_qr_id="QR_001",
            statuses=decode_robot_statuses(status_payload),
            plans=decode_robot_plans(plans_payload),
        )

        with self.assertRaisesRegex(ValueError, "status is stale"):
            validate_server_task(
                snapshot,
                local_station_ids=["DEPOT_01", "PROC_04", "PROC_08", "START"],
                now=now,
            )

    def test_cli_dry_run_with_fixture_files(self):
        now = datetime.now(timezone.utc)
        status_payload, plans_payload = _sample_payloads(now)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            status_path = tmp_path / "status.json"
            plans_path = tmp_path / "plans.json"
            qr_log = tmp_path / "qr_scans.csv"
            event_log = tmp_path / "events.jsonl"
            status_path.write_text(json.dumps(status_payload))
            plans_path.write_text(json.dumps(plans_payload))

            with redirect_stdout(StringIO()):
                exit_code = run_logistics_main(
                    [
                        "--robot-id",
                        "Robot_Test_01",
                        "--qr-id",
                        "QR_001",
                        "--status-json",
                        str(status_path),
                        "--plans-json",
                        str(plans_path),
                        "--dry-run",
                        "--print-task",
                        "--local-station",
                        "DEPOT_01",
                        "--local-station",
                        "PROC_04",
                        "--local-station",
                        "PROC_08",
                        "--local-station",
                        "START",
                        "--qr-scan-log",
                        str(qr_log),
                        "--task-event-log",
                        str(event_log),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("QR_001", qr_log.read_text())
            self.assertIn("task_validated", event_log.read_text())


if __name__ == "__main__":
    unittest.main()
