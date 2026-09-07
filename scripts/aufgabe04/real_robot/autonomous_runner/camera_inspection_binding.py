"""Bind non-authorizing observer progress to the current camera attempt."""

from __future__ import annotations

import math
from pathlib import Path

from scripts.aufgabe04.artifacts.candidate_inspection_observation import (
    load_candidate_inspection_observation,
)


def load_bound_camera_inspection(
    path: Path,
    *,
    candidate_uid: str,
    stream_id: str,
    planning_frame: str,
    stand_x_m: float,
    stand_y_m: float,
    stand_model_profile_sha256: str,
    robot_profile_sha256: str,
    calibration_profile_sha256: str,
) -> dict[str, object]:
    """Reject stale, cross-target, or differently calibrated progress.

    The shared artifact validator checks schema, content hash, evidence and
    lack of motion/completion authority. This adapter adds the parent-owned
    attempt identity; a valid receipt for another stand cannot steer this one.
    """

    try:
        observation = load_candidate_inspection_observation(path)
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"invalid camera inspection receipt: {exc}") from exc
    expected = {
        "candidate_uid": candidate_uid,
        "stream_id": stream_id,
        "planning_frame": planning_frame,
        "stand_model_profile_sha256": stand_model_profile_sha256,
        "robot_profile_sha256": robot_profile_sha256,
        "calibration_profile_sha256": calibration_profile_sha256,
    }
    mismatches = [
        name for name, value in expected.items() if observation.get(name) != value
    ]
    center = observation["stand_center"]
    for axis, expected_position in (("x_m", stand_x_m), ("y_m", stand_y_m)):
        if not math.isclose(
            center[axis], expected_position, rel_tol=0.0, abs_tol=1.0e-6
        ):
            mismatches.append(f"stand_center.{axis}")
    if mismatches:
        raise RuntimeError(
            "camera inspection receipt is not bound to this candidate attempt: "
            + ", ".join(mismatches)
        )
    return observation
