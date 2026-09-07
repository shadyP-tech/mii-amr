"""Parent attempt binding is independent of receipt's internal validity."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.aufgabe04.real_robot.autonomous_runner import camera_inspection_binding as binding


def _expected():
    return {
        "candidate_uid": "candidate_1",
        "stream_id": "session_1_candidate_1",
        "planning_frame": "map",
        "stand_x_m": 1.2,
        "stand_y_m": -0.3,
        "stand_model_profile_sha256": "a" * 64,
        "robot_profile_sha256": "b" * 64,
        "calibration_profile_sha256": "c" * 64,
    }


def _internally_valid_receipt():
    expected = _expected()
    return {
        **{k: v for k, v in expected.items() if k not in {"stand_x_m", "stand_y_m"}},
        "stand_center": {"x_m": 1.2, "y_m": -0.3},
        "qr_id": "QR_001",
        "motion_authorized": False,
        "completion_authorized": False,
    }


def test_accepts_current_target_and_profiles_without_promoting_progress():
    receipt = _internally_valid_receipt()
    with patch.object(binding, "load_candidate_inspection_observation", return_value=receipt):
        result = binding.load_bound_camera_inspection(Path("receipt.json"), **_expected())
    assert result["qr_id"] == "QR_001"
    assert result["motion_authorized"] is False
    assert result["completion_authorized"] is False


@pytest.mark.parametrize("field", [
    "candidate_uid", "stream_id", "planning_frame", "stand_model_profile_sha256",
    "robot_profile_sha256", "calibration_profile_sha256", "x_m", "y_m",
])
def test_rejects_internally_valid_receipt_for_another_attempt(field):
    receipt = deepcopy(_internally_valid_receipt())
    if field in {"x_m", "y_m"}:
        receipt["stand_center"][field] += 0.01
    else:
        receipt[field] = "different"
    with patch.object(binding, "load_candidate_inspection_observation", return_value=receipt):
        with pytest.raises(RuntimeError, match="not bound to this candidate attempt"):
            binding.load_bound_camera_inspection(Path("receipt.json"), **_expected())


def test_invalid_hash_or_schema_remains_terminal():
    with patch.object(binding, "load_candidate_inspection_observation", side_effect=ValueError("hash mismatch")):
        with pytest.raises(RuntimeError, match="invalid camera inspection receipt: hash mismatch"):
            binding.load_bound_camera_inspection(Path("receipt.json"), **_expected())
