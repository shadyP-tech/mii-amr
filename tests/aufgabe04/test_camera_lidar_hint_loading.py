from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.navigation.coverage.coverage_visibility_reporting import (
    validate_coverage_visibility_evidence, coverage_visibility_epoch_fields,
)
from scripts.aufgabe04.perception.lidar_visibility_evidence import (
    append_lidar_visibility_receipts, load_lidar_visibility_receipt_snapshot,
    visibility_receipts_sha256,
)
from scripts.aufgabe04.real_robot.candidate.lidar_inspection_hints import load_camera_lidar_hints
from tests.aufgabe04 import test_coverage_visibility_reporting as visibility
from tests.aufgabe04.test_lidar_inspection_hint import hint_fixture, line_receipts


class CameraLidarHintLoadingTest(unittest.TestCase):
    def fixture(self, root):
        snapshot, registry, frame = hint_fixture()
        base_plan = visibility._plan()
        plan = replace(base_plan, survey_id="survey", viewpoints=tuple(
            replace(base_plan.viewpoints[0], viewpoint_id=f"vp{i}") for i in (1, 2)))
        epochs = root / "epochs"
        epochs.mkdir()
        for view in (1, 2):
            path = root / f"v{view}.jsonl"
            summary = visibility._summary(path)
            path.write_text("")
            receipts = tuple(replace(r, observer_config_sha256=summary["lidar_visibility_observer_config_sha256"])
                             for r in line_receipts(view))
            append_lidar_visibility_receipts(path, receipts)
            receipts, digest = load_lidar_visibility_receipt_snapshot(path)
            summary.update(schema_version=1, motion_published=False, processed_scan_count=len(receipts),
                lidar_visibility_receipt_count=len(receipts), lidar_visibility_receipts_file_sha256=digest,
                lidar_visibility_receipt_set_sha256=visibility_receipts_sha256(receipts))
            summary_path = root / f"summary{view}.json"
            summary_path.write_text(json.dumps(summary))
            evidence = validate_coverage_visibility_evidence(summary, plan, f"vp{view}", True)
            epoch = {"survey_id": "survey", "viewpoint_id": f"vp{view}",
                     "observer_summary_json": str(summary_path), **coverage_visibility_epoch_fields(evidence)}
            (epochs / f"vp{view}.json").write_text(json.dumps(epoch))
        return dict(survey_root=root, plan=plan, snapshot=snapshot, registry=registry, planning_frame=frame)

    def test_completed_survey_artifacts_produce_candidate_bound_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            arguments = self.fixture(Path(tmp))
            hints, diagnostics = load_camera_lidar_hints(**arguments)
            self.assertEqual(set(hints), {"candidate_1"})
            self.assertEqual(diagnostics["unavailable_epochs"], {})

    def test_missing_tampered_and_wrong_epoch_evidence_falls_back(self):
        for mode in ("missing", "tampered_scan", "wrong_epoch", "changed_epoch_binding"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                arguments = self.fixture(root)
                if mode == "missing":
                    (root / "summary2.json").unlink()
                elif mode == "tampered_scan":
                    path = root / "v2.jsonl"
                    path.write_text(path.read_text() + "\n")  # Even byte-only changes invalidate its source hash.
                else:
                    path = root / "epochs" / "vp2.json"
                    epoch = json.loads(path.read_text())
                    epoch["survey_id" if mode == "wrong_epoch" else "lidar_visibility_receipt_set_sha256"] = "wrong"
                    path.write_text(json.dumps(epoch))
                hints, diagnostics = load_camera_lidar_hints(**arguments)
                self.assertFalse(hints)
                self.assertIn("vp2", diagnostics["unavailable_epochs"])

    def test_legacy_without_frame_keeps_existing_camera_approach(self):
        hints, diagnostics = load_camera_lidar_hints(survey_root=None, plan=None, snapshot=None,
                                                    registry=None, planning_frame=None)
        self.assertFalse(hints)
        self.assertEqual(diagnostics["reason"], "survey_planning_frame_unavailable")


if __name__ == "__main__":
    unittest.main()
