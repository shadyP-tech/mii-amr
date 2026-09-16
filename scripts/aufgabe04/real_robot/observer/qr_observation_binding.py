"""Bind a QR-confirmed viewing pose to the current candidate attempt.

This receipt completes discovery only. Its robot pose is never substituted
for a stand center, stand angle, certified facing pose, or motion permit.
"""

from __future__ import annotations

import math
from pathlib import Path

from scripts.aufgabe04.artifacts.qr_verified_observation_pose import (
    load_qr_verified_observation_pose,
)


def load_bound_qr_observation_pose(
    path: Path,
    *,
    candidate_uid: str,
    stream_id: str,
    planning_frame: str,
    stand_x_m: float,
    stand_y_m: float,
    stand_model_profile_sha256: str | None = None,
    robot_profile_sha256: str | None = None,
    calibration_profile_sha256: str | None = None,
    base_frame: str | None = None,
    scan_frame: str | None = None,
    camera_frame: str | None = None,
) -> dict[str, object]:
    """Validate the hash/evidence, then the parent-owned attempt bindings."""

    try:
        observation = load_qr_verified_observation_pose(path)
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"invalid QR observation pose receipt: {exc}") from exc
    expected = {
        "candidate_uid": candidate_uid,
        "stream_id": stream_id,
        "planning_frame": planning_frame,
        "stand_model_profile_sha256": stand_model_profile_sha256,
        "robot_profile_sha256": robot_profile_sha256,
        "calibration_profile_sha256": calibration_profile_sha256,
    }
    mismatches = [
        name for name, value in expected.items()
        if value is not None and observation.get(name) != value
    ]
    center = observation["stand_center"]
    for axis, position in (("x_m", stand_x_m), ("y_m", stand_y_m)):
        if not math.isclose(center[axis], position, rel_tol=0.0, abs_tol=1.0e-6):
            mismatches.append(f"stand_center.{axis}")
    provenance = observation["localization_provenance"]
    for name, expected_frame in (
        ("map_frame", planning_frame), ("base_frame", base_frame),
        ("scan_frame", scan_frame), ("camera_frame", camera_frame),
    ):
        if expected_frame is not None and provenance.get(name) != expected_frame:
            mismatches.append(f"localization_provenance.{name}")
    if mismatches:
        raise RuntimeError(
            "QR observation pose receipt is not bound to this candidate attempt: "
            + ", ".join(mismatches)
        )
    return observation
