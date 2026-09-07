import json
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.artifacts.content_store import content_hashed_payload
from scripts.aufgabe04.artifacts.candidate_inspection_observation import (
    HASH_FIELD, build_candidate_inspection_observation,
    load_candidate_inspection_observation, validate_candidate_inspection_observation,
)


def observation_payload():
    return build_candidate_inspection_observation(
        candidate_uid="candidate1", stream_id="run_candidate1", planning_frame="map",
        stand_center={"x_m": 1.0, "y_m": 0.2},
        robot_pose={"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
        classification="unobservable", reasons=["no_geometry"],
        camera_relative_yaw_rad=None, yaw_uncertainty_rad=None,
        qr_id=None, qr_sample_count=0, sample_count=7,
        sensor_stamps_sec=[10.0+i/3 for i in range(7)],
        first_sensor_stamp_sec=10.0, sensor_stamp_sec=12.0,
        robot_profile_sha256="a"*64, calibration_profile_sha256="b"*64,
        stand_model_profile_sha256="c"*64,
        sample_gate_evidence={k: True for k in (
            "all_samples_stationary", "all_samples_synchronized", "all_samples_lidar_associated", "all_samples_fresh",
        )},
    )


class InspectionObservationTests(unittest.TestCase):
    def test_roundtrip_and_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"progress.json"
            data=observation_payload();path.write_text(json.dumps(data))
            self.assertEqual(load_candidate_inspection_observation(path), data)
            data["candidate_uid"]="other";path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): load_candidate_inspection_observation(path)

    def test_rehashed_invalid_authority_and_evidence_are_rejected(self):
        for key,value in (
            ("motion_authorized",True),("completion_authorized",True),
            ("angle_authority","directed_face"),("sample_count",True),
            ("classification","backside_confirmed"),("robot_profile_sha256","bad"),
            ("qr_id","QR_001"),("sample_gate_evidence",{}),
            ("sensor_stamps_sec",[10.0]*7),
            ("camera_relative_yaw_rad",0.5),
        ):
            with self.subTest(key=key):
                data=observation_payload();data.pop(HASH_FIELD);data[key]=value
                with self.assertRaises(ValueError):
                    validate_candidate_inspection_observation(content_hashed_payload(data,hash_field=HASH_FIELD))

    def test_fresh_qr_latch_is_still_advisory(self):
        data=observation_payload();data.pop(HASH_FIELD)
        data.update(qr_id="QR_001",qr_sample_count=2)
        value=build_candidate_inspection_observation(**data)
        self.assertFalse(value["completion_authorized"])
        self.assertFalse(value["motion_authorized"])

    def test_duplicate_json_key_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"progress.json"
            text=json.dumps(observation_payload())
            path.write_text(text[:-1]+', "candidate_uid": "other"}')
            with self.assertRaises(ValueError): load_candidate_inspection_observation(path)
