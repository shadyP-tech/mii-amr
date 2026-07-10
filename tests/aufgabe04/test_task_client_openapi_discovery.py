import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.task_client.openapi_discovery import discover_scan_endpoint_template  # noqa: E402


class OpenApiDiscoveryTest(unittest.TestCase):
    def test_discovers_clear_get_qr_endpoint(self):
        payload = {
            "paths": {
                "/api/v1/robots/{robot_id}/scan": {
                    "get": {
                        "operationId": "report_qr_scan",
                        "parameters": [
                            {"name": "robot_id", "in": "path"},
                            {"name": "qr_id", "in": "query"},
                        ],
                    }
                }
            }
        }

        self.assertEqual(
            discover_scan_endpoint_template(payload),
            "/api/v1/robots/{robot_id}/scan?qr_id={qr_id}",
        )

    def test_rejects_ambiguous_scan_endpoints(self):
        payload = {
            "paths": {
                "/scan-a": {"get": {"operationId": "scan_a", "parameters": [{"name": "qr_id"}]}},
                "/scan-b": {"get": {"operationId": "scan_b", "parameters": [{"name": "qr_id"}]}},
            }
        }

        with self.assertRaisesRegex(ValueError, "multiple"):
            discover_scan_endpoint_template(payload)


if __name__ == "__main__":
    unittest.main()
