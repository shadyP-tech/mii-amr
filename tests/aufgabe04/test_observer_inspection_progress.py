from pathlib import Path
import math
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.aufgabe04.real_robot.observer.inspection_progress import (
    InspectionProgress, InspectionClassification, classify_inspection_progress,
)
from scripts.aufgabe04.real_robot.observer.node import PassiveRealViewpointNode
from scripts.aufgabe04.artifacts.candidate_inspection_observation import load_candidate_inspection_observation
from scripts.aufgabe04.real_robot.observer.process import monitor_passive_observer_process
from tests.aufgabe04.test_passive_observer_process import _Process


def frame(stamp, **overrides):
    return dict(frame_stamp_sec=stamp, robot_pose={"x_m":0.0,"y_m":0.0,"yaw_rad":0.0},
                frame_accepted=True,poisoned=False,
                classification=InspectionClassification("unobservable","no_head"),**overrides)


class InspectionProgressTests(unittest.TestCase):
    def test_seven_distinct_frames_and_two_seconds_required(self):
        progress=InspectionProgress()
        for i in range(6):self.assertIsNone(progress.record(**frame(10+i/3)))
        self.assertEqual(progress.record(**frame(12.0))["sample_count"],7)

    def test_duplicate_stamps_and_rejected_sensor_frames_do_not_accumulate(self):
        progress=InspectionProgress()
        for i in range(20):
            self.assertIsNone(progress.record(**frame(10.0)))
            data=frame(10.0+i);data["frame_accepted"]=False
            self.assertIsNone(progress.record(**data))

    def test_short_span_does_not_trigger_early(self):
        progress=InspectionProgress()
        for i in range(20):self.assertIsNone(progress.record(**frame(10.0+i/100)))

    def test_motion_resets_same_view_evidence(self):
        progress=InspectionProgress()
        for i in range(6):progress.record(**frame(10+i/3))
        data=frame(12);data["robot_pose"]["x_m"]=0.1
        self.assertIsNone(progress.record(**data))

    def test_poison_cannot_produce_progress(self):
        progress=InspectionProgress()
        for i in range(10):
            data=frame(10+i/3);data["poisoned"]=i==3
            self.assertIsNone(progress.record(**data))

    def test_qr_is_not_carried_after_latch_expiry(self):
        progress=InspectionProgress()
        for i in range(6):progress.record(**frame(10+i/3,current_qr_id="QR_001",current_qr_sample_count=2))
        value=progress.record(**frame(12))
        self.assertIsNone(value["qr_id"])
        self.assertEqual(value["qr_sample_count"],0)

    def test_conflicting_qr_latches_poison_even_across_soft_evidence_reset(self):
        progress=InspectionProgress()
        progress.record(**frame(10,current_qr_id="QR_001",current_qr_sample_count=2))
        for i in range(1,10):
            self.assertIsNone(progress.record(**frame(10+i/3,current_qr_id="QR_002",current_qr_sample_count=2)))

    def test_missing_qr_is_not_edge_or_backside_classification(self):
        value=classify_inspection_progress("metric_model_measurement_unavailable",{})
        self.assertEqual(value.classification,"unobservable")
        value=classify_inspection_progress("evidence_not_committable",{"stand_axis_debug":{"metric_model":{"qr_detected":True}}})
        self.assertEqual(value.classification,"front_unreadable")

    def test_expired_frames_do_not_form_a_progress_receipt(self):
        progress=InspectionProgress()
        for i in range(10):self.assertIsNone(progress.record(**frame(10+6*i)))

    def test_decoded_front_and_oblique_are_distinguished_without_authority(self):
        self.assertEqual(classify_inspection_progress("collecting_consensus",{"qr_texts":["QR_001"]}).classification,"front_readable")
        self.assertEqual(classify_inspection_progress("evidence_not_committable",{
            "qr_texts":["QR_001"],"conditioning":{"reason":"oblique_silhouette"},
        }).classification,"oblique")

    def test_real_metric_model_near_ninety_is_advisory_edge_view(self):
        details={"conditioning":{"reason":"oblique_silhouette"},"stand_axis_debug":{
            "estimator_mode":"metric_model_only", "estimator_view_mode":"face_visible",
            "estimator_usable":True,"advisory_camera_relative_yaw_rad":math.radians(82),
            "estimator_reason":"axis_estimated_model_current_frame_refined",
        }}
        self.assertEqual(classify_inspection_progress("evidence_not_committable",details).classification,"edge_on")
        details["stand_axis_debug"]["estimator_usable"]=False
        details.pop("conditioning")
        self.assertEqual(classify_inspection_progress("metric_model_measurement_unavailable",details).classification,"unobservable")


class InspectionObserverIntegrationTests(unittest.TestCase):
    def make_node(self,path):
        node=PassiveRealViewpointNode.__new__(PassiveRealViewpointNode)
        node.args=SimpleNamespace(inspection_observation_json=path,stand_id="candidate1",stream_id="run_candidate1",
            stand_x=1.0,stand_y=.2,stationary_translation_m=.02,stationary_rotation_deg=2.0)
        node.profile=SimpleNamespace(map_frame="map")
        node.calibration=object();node.stand_model_profile=SimpleNamespace(sha256="c"*64)
        node.completed=False
        return node

    def test_node_publishes_advisory_only_from_processed_frames(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch("scripts.aufgabe04.real_robot.observer.node.real_robot_profile_sha256",return_value="a"*64), \
             patch("scripts.aufgabe04.real_robot.observer.node.camera_calibration_sha256",return_value="b"*64):
            path=Path(tmp)/"progress.json";node=self.make_node(path)
            for i in range(7):
                node._inspection_frame=frame(10+i/3);node._inspection_frame.pop("classification")
                result=node._maybe_commit_inspection_progress("metric_model_measurement_unavailable",{})
            self.assertIsNotNone(result)
            self.assertFalse(load_candidate_inspection_observation(path)["completion_authorized"])
            self.assertTrue(node.completed)

    def test_tf_state_and_completed_stronger_result_cannot_emit_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"progress.json";node=self.make_node(path)
            node._inspection_frame=frame(10);node._inspection_frame.pop("classification")
            self.assertIsNone(node._maybe_commit_inspection_progress("tf_pending_exact_time",{}))
            self.assertIsNone(node._inspection_frame)
            self.assertIsNone(node._maybe_commit_inspection_progress("collecting_consensus",{}))
            node.completed=True
            self.assertIsNone(node._maybe_commit_inspection_progress("collecting_consensus",{}))
            self.assertFalse(path.exists())

    def test_consumed_update_cannot_be_reused_by_later_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            node=self.make_node(Path(tmp)/"progress.json")
            node._inspection_frame=frame(10);node._inspection_frame.pop("classification")
            self.assertIsNone(node._maybe_commit_inspection_progress("metric_model_measurement_unavailable",{}))
            self.assertIsNone(node._inspection_frame)
            for _ in range(20):
                self.assertIsNone(node._maybe_commit_inspection_progress("metric_model_measurement_unavailable",{}))

    def test_recovering_axis_acquisition_clears_prior_failed_view_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"progress.json";node=self.make_node(path)
            # Six failed samples already span more than two seconds.
            for i in range(6):
                node._inspection_frame=frame(10+i*.5)
                node._inspection_frame.pop("classification")
                self.assertIsNone(node._maybe_commit_inspection_progress("metric_model_measurement_unavailable",{}))
            for i in range(6):
                node._inspection_frame=frame(13+i*.25)
                node._inspection_frame.pop("classification")
                node._inspection_frame["axis_sample_accepted"]=True
                self.assertIsNone(node._maybe_commit_inspection_progress("collecting_consensus",{}))
                self.assertEqual(node._inspection_progress._samples,{})
                self.assertFalse(node.completed)
            # Existing observer path commits its recommendation on good
            # sample seven before status publication calls this hook.
            node.completed=True
            self.assertIsNone(node._maybe_commit_inspection_progress("recommendation_committed",{}))
            self.assertFalse(path.exists())

    def test_no_association_and_poisoned_update_cannot_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            node=self.make_node(Path(tmp)/"progress.json")
            for i in range(12):
                data=frame(10+i/3);data.pop("classification")
                data["frame_accepted"]=False
                node._inspection_frame=data
                self.assertIsNone(node._maybe_commit_inspection_progress("metric_model_measurement_unavailable",{}))
            for i in range(12):
                data=frame(20+i/3);data.pop("classification")
                data["poisoned"]=i==3
                node._inspection_frame=data
                self.assertIsNone(node._maybe_commit_inspection_progress("metric_model_measurement_unavailable",{}))

    def test_new_qr_decode_gets_time_to_establish_its_second_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"progress.json";node=self.make_node(path)
            for i in range(6):
                node._inspection_frame=frame(10+i*.5)
                node._inspection_frame.pop("classification")
                self.assertIsNone(node._maybe_commit_inspection_progress("metric_model_measurement_unavailable",{}))
            node._inspection_frame=frame(13,current_qr_sample_count=1)
            node._inspection_frame.pop("classification")
            node._inspection_frame["qr_sample_accepted"]=True
            self.assertIsNone(node._maybe_commit_inspection_progress("evidence_not_committable",{"qr_texts":["QR_001"]}))
            self.assertFalse(node.completed)
            self.assertEqual(node._inspection_progress._samples,{})
            node._inspection_frame=frame(13.25,current_qr_id="QR_001",current_qr_sample_count=2)
            node._inspection_frame.pop("classification")
            node._inspection_frame["qr_sample_accepted"]=True
            self.assertIsNone(node._maybe_commit_inspection_progress("evidence_not_committable",{"qr_texts":["QR_001"]}))
            self.assertFalse(path.exists())

    def test_acquisition_grace_preserves_prior_identity_and_conflict_poison(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"progress.json";node=self.make_node(path)
            def feed(stamp,qr_id,count,accepted=False):
                data=frame(stamp,current_qr_id=qr_id,current_qr_sample_count=count)
                data.pop("classification");data["qr_sample_accepted"]=accepted
                node._inspection_frame=data
                return node._maybe_commit_inspection_progress("evidence_not_committable",{})
            self.assertIsNone(feed(10,"QR_001",2))
            self.assertIsNone(feed(11,None,1,True))
            self.assertEqual(node._inspection_progress._seen_qr_id,"QR_001")
            for i in range(10):self.assertIsNone(feed(11.1+i/3,"QR_002",2,True))
            self.assertTrue(node._inspection_progress._poisoned)
            # Another acquisition-grace request cannot erase that poison.
            self.assertIsNone(feed(15,None,1,True))
            self.assertTrue(node._inspection_progress._poisoned)
            self.assertFalse(path.exists())

    def test_process_third_artifact_and_stronger_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);progress=root/"progress.json";progress.write_text("{}")
            kwargs=dict(recommendation_path=root/"recommendation.json",axis_observation_path=root/"axis.json",
                        inspection_observation_path=progress,timeout_sec=90)
            result=monitor_passive_observer_process(process=_Process(wait_outcomes=(0,)),**kwargs)
            self.assertEqual(result.artifact_kind,"inspection_observation")
            kwargs["axis_observation_path"].write_text("{}")
            result=monitor_passive_observer_process(process=_Process(wait_outcomes=(0,)),**kwargs)
            self.assertEqual(result.artifact_kind,"axis_observation")
            kwargs["recommendation_path"].write_text("{}")
            result=monitor_passive_observer_process(process=_Process(wait_outcomes=(0,)),**kwargs)
            self.assertEqual(result.artifact_kind,"recommendation")
