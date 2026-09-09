import copy
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json
from scripts.aufgabe04.stations.candidate_snapshot import write_candidate_snapshot
from scripts.aufgabe04.stations.server_identity_binding import (
    bind_observed_station_identities, load_server_qr_mapping_evidence, seal_server_qr_mapping_evidence,
    write_observed_identities, write_server_qr_mapping_evidence,
)
from scripts.aufgabe04.stations.station_identity_registry import load_station_identity_registry, write_station_identity_registry
from scripts.aufgabe04.task_client.server_response_decoder import decode_robot_plans, decode_robot_statuses
from tests.aufgabe04.test_station_identity_registry import _snapshot
from tests.aufgabe04.test_task_client_fastapi_flow import _sample_payloads

ROOT = Path(__file__).resolve().parents[2]


def mapping_payload(now):
    status, plans = _sample_payloads(datetime.fromtimestamp(now, timezone.utc))
    status[0]["target"] = "station_Mixed_a"
    plans[0]["expanded_path"] = ["station_Mixed_a"]
    plans[0]["qr_mappings"] = [dict(plans[0]["qr_mappings"][0], qr_code_id="qr_Mixed_a", station_id="station_Mixed_a")]
    return status, plans


class ServerIdentityBindingTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc).timestamp()
        self.snapshot = replace(_snapshot(), candidates=_snapshot().candidates[:1])
        self.status, self.plans = mapping_payload(self.now)
        self.evidence = seal_server_qr_mapping_evidence(self.plans, robot_id="Robot_Test_01", captured_unix_sec=self.now)

    def bind(self, qr="qr_Mixed_a", evidence=None, now=None):
        return bind_observed_station_identities(candidate_snapshot=self.snapshot, observed_qr_by_candidate={"candidate_a": qr}, mapping_evidence=evidence or self.evidence, registry_id="bound", now_sec=self.now if now is None else now)

    def test_exact_qr_and_case_preserving_station_bind_without_fabrication(self):
        mapping = self.bind().mappings[0]
        self.assertEqual((mapping.qr_id, mapping.server_station_id), ("qr_Mixed_a", "station_Mixed_a"))
        with self.assertRaisesRegex(ValueError, "no authoritative"):
            self.bind("QR_MIXED_A")
        with self.assertRaisesRegex(ValueError, "no authoritative"):
            self.bind("station_Mixed_a")

    def test_robot_scope_duplicates_and_absent_mappings_rejected(self):
        for change in ("foreign", "duplicate_qr", "duplicate_station", "missing", "duplicate_robot"):
            with self.subTest(change=change):
                payload = copy.deepcopy(self.plans)
                mappings = payload[0]["qr_mappings"]
                if change == "foreign":
                    mappings[0]["robot_id"] = "another_robot"
                elif change == "duplicate_qr":
                    mappings.append(dict(mappings[0], station_id="other_station"))
                elif change == "duplicate_station":
                    mappings.append(dict(mappings[0], qr_code_id="other_QR"))
                elif change == "missing":
                    mappings.clear()
                else:
                    payload.append(copy.deepcopy(payload[0]))
                with self.assertRaises(ValueError):
                    seal_server_qr_mapping_evidence(payload, robot_id="Robot_Test_01", captured_unix_sec=self.now)

    def test_status_placeholders_keep_existing_telemetry_contract(self):
        status = copy.deepcopy(self.status)
        status[0]["target"] = "-"
        status[0]["last_qr"] = "No QR Yet"
        decoded = decode_robot_statuses(status)[0]
        self.assertEqual(decoded.target, "-")
        self.assertEqual(decoded.last_qr, "No QR Yet")
        for field in ("target", "last_qr"):
            invalid = copy.deepcopy(status)
            invalid[0][field] = ""
            with self.assertRaisesRegex(ValueError, "invalid string"):
                decode_robot_statuses(invalid)

    def test_hash_scope_freshness_and_detached_input(self):
        self.plans[0]["qr_mappings"][0]["station_id"] = "tampered_after_seal"
        self.assertEqual(self.bind().mappings[0].server_station_id, "station_Mixed_a")
        with self.assertRaisesRegex(ValueError, "stale"):
            self.bind(now=self.now + 3601)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            write_server_qr_mapping_evidence(path, self.evidence)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                load_server_qr_mapping_evidence(path, robot_id="other_robot", now_sec=self.now)
            payload = json.loads(path.read_text())
            payload["robot_plans"][0]["qr_mappings"][0]["station_id"] = "tampered_file"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_server_qr_mapping_evidence(path, robot_id="Robot_Test_01", now_sec=self.now)

    def test_actual_offline_seal_bind_and_logistics_cli_case_mismatch_regression(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plans.json").write_text(json.dumps(self.plans))
            (root / "status.json").write_text(json.dumps(self.status))
            write_candidate_snapshot(root / "candidate.json", self.snapshot)
            write_observed_identities(root / "observed.json", candidate_snapshot=self.snapshot, observed_qr_by_candidate={"candidate_a": "qr_Mixed_a"}, session_id="session", observed_unix_sec=self.now)
            seal = subprocess.run([sys.executable, "-m", "scripts.aufgabe04.stations.server_identity_binding", "seal", "--captured-unix-sec", str(self.now), "--plans-json", str(root / "plans.json"), "--server-robot-id", "Robot_Test_01", "--output-json", str(root / "evidence.json")], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(seal.returncode, 0, seal.stderr)
            bind = subprocess.run([sys.executable, "-m", "scripts.aufgabe04.stations.server_identity_binding", "bind", "--candidate-snapshot", str(root / "candidate.json"), "--observed-identities", str(root / "observed.json"), "--server-qr-mapping-evidence", str(root / "evidence.json"), "--server-robot-id", "Robot_Test_01", "--registry-id", "bound", "--output-json", str(root / "identities.json")], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(bind.returncode, 0, bind.stderr)
            registry = load_station_identity_registry(root / "identities.json", candidate_snapshot=self.snapshot)
            self.assertEqual(registry.mappings[0].server_station_id, "station_Mixed_a")
            argv = [sys.executable, "-m", "scripts.aufgabe04.run_logistics_mission", "--robot-id", "Robot_Test_01", "--qr-id", "qr_Mixed_a", "--dry-run", "--skip-health", "--status-json", str(root / "status.json"), "--plans-json", str(root / "plans.json"), "--station-identity-registry", str(root / "identities.json"), "--qr-scan-log", str(root / "scans.csv"), "--task-event-log", str(root / "events.jsonl")]
            result = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("station_Mixed_a", result.stdout)
            argv[argv.index("qr_Mixed_a")] = "QR_MIXED_A"
            wrong_case = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(wrong_case.returncode, 2)
            self.assertIn("absent", wrong_case.stderr)


if __name__ == "__main__":
    unittest.main()
