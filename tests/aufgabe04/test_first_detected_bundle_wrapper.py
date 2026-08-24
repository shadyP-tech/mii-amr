import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "scripts"
    / "aufgabe04"
    / "navigation"
    / "entrypoints"
    / "run_first_detected_station_segment_with_bundle.sh"
)


class FirstDetectedBundleWrapperTest(unittest.TestCase):
    def test_help_documents_bundle_wrapper(self):
        result = subprocess.run(
            ["bash", str(SCRIPT), "run_test", "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("run_with_bundle.sh", result.stdout)
        self.assertIn("typed RUN", result.stdout)
        self.assertIn("--no-initialpose-prompt", result.stdout)
        self.assertIn("--preflight-observation-window-sec", result.stdout)
        self.assertIn("--initial-sensor-wait-sec", result.stdout)

    def test_missing_route_artifact_fails_before_bundle_execution(self):
        result = subprocess.run(
            [
                "bash",
                str(SCRIPT),
                "run_test",
                "--route-csv",
                "does/not/exist.csv",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("missing route CSV", result.stderr)


if __name__ == "__main__":
    unittest.main()
