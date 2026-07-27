import copy
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import patch


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
        self.assertEqual(validated.ordered_station_ids, ("DEPOT_01", "PROC_08", "START"))
        self.assertEqual(len(validated.order_sha256), 64)
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

    def test_report_scan_refetches_live_state_before_persisting_task(self):
        now = datetime.now(timezone.utc)
        pre_status, pre_plans = _sample_payloads(now)
        post_status = copy.deepcopy(pre_status)
        post_plans = copy.deepcopy(pre_plans)
        post_status[0]["state"] = "GO_TO_PROCESSING"
        post_status[0]["target"] = "PROC_04"
        post_status[0]["last_qr"] = "QR_001"
        post_plans[0]["next_step_index"] = 1
        call_order = []

        def report_scan(*args, **kwargs):
            call_order.append("report")
            return {"accepted": True}

        def fetch_status(*args, **kwargs):
            call_order.append("status")
            return post_status

        def fetch_plans(*args, **kwargs):
            call_order.append("plans")
            return post_plans

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            status_path = tmp_path / "status.json"
            plans_path = tmp_path / "plans.json"
            task_path = tmp_path / "validated_task.json"
            qr_log = tmp_path / "qr_scans.csv"
            event_log = tmp_path / "events.jsonl"
            status_path.write_text(json.dumps(pre_status))
            plans_path.write_text(json.dumps(pre_plans))

            with patch(
                "scripts.aufgabe04.run_logistics_mission.report_scanned_qr",
                side_effect=report_scan,
            ):
                with patch(
                    "scripts.aufgabe04.run_logistics_mission.fetch_admin_status",
                    side_effect=fetch_status,
                ):
                    with patch(
                        "scripts.aufgabe04.run_logistics_mission.fetch_robot_plans",
                        side_effect=fetch_plans,
                    ):
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
                                    "--scan-endpoint-template",
                                    "/robots/{robot_id}/scan",
                                    "--report-scan",
                                    "--dry-run",
                                    "--local-station",
                                    "DEPOT_01",
                                    "--local-station",
                                    "PROC_04",
                                    "--local-station",
                                    "PROC_08",
                                    "--local-station",
                                    "START",
                                    "--validated-task-json",
                                    str(task_path),
                                    "--qr-scan-log",
                                    str(qr_log),
                                    "--task-event-log",
                                    str(event_log),
                                ]
                            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(call_order, ["report", "status", "plans"])
            persisted = json.loads(task_path.read_text())
            self.assertEqual(persisted["task"]["state"], "GO_TO_PROCESSING")
            self.assertEqual(persisted["task"]["target_station"], "PROC_04")
            self.assertNotEqual(
                persisted["task"]["target_station"],
                pre_status[0]["target"],
            )
            events = [json.loads(line)["event_type"] for line in event_log.read_text().splitlines()]
            self.assertLess(events.index("scan_reported"), events.index("post_scan_task_refetched"))
            self.assertLess(events.index("post_scan_task_refetched"), events.index("task_validated"))

    def test_report_scan_refetch_failure_does_not_persist_task(self):
        now = datetime.now(timezone.utc)
        pre_status, pre_plans = _sample_payloads(now)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            status_path = tmp_path / "status.json"
            plans_path = tmp_path / "plans.json"
            task_path = tmp_path / "validated_task.json"
            status_path.write_text(json.dumps(pre_status))
            plans_path.write_text(json.dumps(pre_plans))

            with patch(
                "scripts.aufgabe04.run_logistics_mission.report_scanned_qr",
                return_value={"accepted": True},
            ):
                with patch(
                    "scripts.aufgabe04.run_logistics_mission.fetch_admin_status",
                    side_effect=RuntimeError("status refresh failed"),
                ):
                    with redirect_stdout(StringIO()):
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "status refresh failed",
                        ):
                            run_logistics_main(
                                [
                                    "--robot-id",
                                    "Robot_Test_01",
                                    "--qr-id",
                                    "QR_001",
                                    "--status-json",
                                    str(status_path),
                                    "--plans-json",
                                    str(plans_path),
                                    "--scan-endpoint-template",
                                    "/robots/{robot_id}/scan",
                                    "--report-scan",
                                    "--dry-run",
                                    "--validated-task-json",
                                    str(task_path),
                                    "--qr-scan-log",
                                    str(tmp_path / "qr_scans.csv"),
                                    "--task-event-log",
                                    str(tmp_path / "events.jsonl"),
                                ]
                            )

            self.assertFalse(task_path.exists())


if __name__ == "__main__":
    unittest.main()
