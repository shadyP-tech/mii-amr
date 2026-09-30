"""Processed-frame failures survive trailing transient sensor/TF statuses."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.aufgabe04.real_robot.observer.diagnostics import load_passive_observer_status
from scripts.aufgabe04.real_robot.observer.node import PassiveRealViewpointNode
from scripts.aufgabe04.real_robot.observer.pipeline_diagnostics import CameraProcessingDiagnostics
from tests.aufgabe04 import test_camera_observer_processing as processing


class ObserverPipelineDiagnosticsTests(unittest.TestCase):
    def test_counts_one_processed_outcome_and_retains_separate_blockers(self):
        diagnostics = CameraProcessingDiagnostics()
        for index in range(3):
            diagnostics.begin_frame()
            diagnostics.record_outcome("metric_model_measurement_unavailable", {
                "stand_axis_debug": {
                    "estimator_reason": ("head_model_planar_axis_ambiguous" if index == 2
                                         else "model_current_head_border_unavailable"),
                    "current_head_candidate_association": {"reason": "scan_persistence_current_input_invalid"},
                },
            })
            diagnostics.record_outcome("tf_pending_exact_time", {"reason": "future extrapolation"})
            diagnostics.record_outcome("tf_retry_exhausted", {"reason": "future extrapolation"})
        summary = diagnostics.snapshot()
        self.assertTrue(summary["diagnostic_only"])
        self.assertEqual(summary["state_counts"], {"metric_model_measurement_unavailable": 3})
        self.assertEqual(summary["reason_counts"], {})
        self.assertEqual(summary["estimator_reason_counts"], {
            "model_current_head_border_unavailable": 2, "head_model_planar_axis_ambiguous": 1})
        self.assertEqual(summary["association_reason_counts"], {"scan_persistence_current_input_invalid": 3})
        self.assertEqual(summary["last_frame"]["estimator_reason"], "head_model_planar_axis_ambiguous")

    def test_no_processed_frame_does_not_invent_outcomes(self):
        diagnostics = CameraProcessingDiagnostics()
        diagnostics.record_outcome("tf_pending_exact_time", {"reason": "future extrapolation"})
        self.assertIsNone(diagnostics.snapshot())

    def test_dynamic_failure_labels_are_bounded(self):
        diagnostics = CameraProcessingDiagnostics()
        for index in range(100):
            diagnostics.begin_frame()
            diagnostics.record_outcome("image_rectification_failed", {"reason": f"{index}:" + "x" * 300})
        summary = diagnostics.snapshot()
        self.assertEqual(len(summary["reason_counts"]), 65)
        self.assertEqual(summary["reason_counts"]["other_labels"], 36)
        self.assertEqual(len(summary["last_frame"]["reason"]), 256)

    def test_next_iteration_cannot_misattribute_suppressed_outcome_to_tf(self):
        adapter = processing.CameraObserverProcessingTest().make_adapter()
        adapter._camera_processing_diagnostics = CameraProcessingDiagnostics()
        adapter._camera_processing_diagnostics.begin_frame()
        adapter._next_sensor_tuple.return_value = None
        adapter._process_latest()
        adapter._camera_processing_diagnostics.record_outcome("tf_pending_exact_time", {})
        self.assertIsNone(adapter._camera_processing_diagnostics.snapshot())

    def test_status_before_processing_reports_zero_activity_without_outcomes(self):
        adapter = processing.CameraObserverProcessingTest().make_adapter()
        with TemporaryDirectory() as tmp:
            adapter.args.status_json = Path(tmp) / "status.json"
            PassiveRealViewpointNode._write_status(adapter, "tf_pending_exact_time")
            status = load_passive_observer_status(adapter.args.status_json)
        self.assertEqual(status.camera_pipeline_counts, dict(
            tf_ready_tuples=0, processed_images=0, fresh_detector_results=0))
        self.assertIsNone(status.camera_processing_outcomes)

    def test_status_writer_preserves_summary_into_parent_after_final_tf_status(self):
        adapter = processing.CameraObserverProcessingTest().make_adapter()
        adapter._camera_pipeline_counters = {"tf_ready_tuples": 217, "processed_images": 217}
        adapter._camera_processing_diagnostics = CameraProcessingDiagnostics()
        with TemporaryDirectory() as tmp:
            adapter.args.status_json = Path(tmp) / "status.json"
            adapter._camera_processing_diagnostics.begin_frame()
            PassiveRealViewpointNode._write_status(adapter,
                "metric_model_measurement_unavailable", estimator_reason="head_model_planar_axis_ambiguous",
                stand_axis_debug={"current_head_candidate_association": {
                    "reason": "scan_persistence_current_input_invalid"}})
            first = json.loads(adapter.args.status_json.read_text())
            PassiveRealViewpointNode._write_status(adapter,
                "tf_pending_exact_time", reason="future extrapolation")
            status = load_passive_observer_status(adapter.args.status_json)
        self.assertEqual(status.state, "tf_pending_exact_time")
        self.assertEqual(status.reason, "future extrapolation")
        self.assertEqual(status.camera_pipeline_counts["processed_images"], 217)
        self.assertEqual(status.camera_processing_outcomes, first["camera_processing_outcomes"])
        self.assertEqual(status.to_dict()["camera_processing_outcomes"], first["camera_processing_outcomes"])


if __name__ == "__main__":
    unittest.main()
