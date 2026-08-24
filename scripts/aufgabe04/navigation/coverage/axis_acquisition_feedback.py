"""Pure simulation-only feedback contract for rejected acquisition rays.

The planner writes one per-candidate sidecar and the observer consumes it. The
route runner never reads this file. Stable target geometry owns the binding;
source observation stamps remain monotonic diagnostics so a newer observer
frame for the exact same target can safely consume a planner decision.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Mapping


AXIS_ACQUISITION_FEEDBACK_SCHEMA_VERSION = 1
AXIS_ACQUISITION_FEEDBACK_CONTRACT = (
    "simulation_axis_acquisition_static_rejection_feedback"
)
AXIS_ACQUISITION_STATIC_TARGET_FAILURES = frozenset(
    {
        "acquisition_target_not_traversable",
        "astar_failed:no_path",
        "astar_goal_was_snapped",
    }
)


def canonical_json_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def axis_acquisition_search_index(recommendation) -> int:
    """Return the exact observer search index encoded by both provisional IDs."""

    if recommendation.axis_state != "axis_acquisition":
        raise ValueError("axis acquisition feedback requires axis_acquisition")
    if recommendation.material_target.evidence_state != "axis_acquisition":
        raise ValueError("axis acquisition feedback evidence state mismatch")
    if recommendation.side_evidence.hard or recommendation.side_evidence.valid:
        raise ValueError("axis acquisition feedback forbids side evidence")
    if any(face.identity_resolved for face in recommendation.face_candidates):
        raise ValueError("axis acquisition feedback forbids resolved face identity")
    faces = tuple(recommendation.face_candidates)
    if len(faces) != 2:
        raise ValueError("axis acquisition feedback requires two candidates")
    prefix = "acquisition_near_"
    near_id = faces[0].face_id
    if not near_id.startswith(prefix):
        raise ValueError("axis acquisition near candidate ID is malformed")
    suffix = near_id[len(prefix) :]
    if len(suffix) != 2 or not suffix.isdigit():
        raise ValueError("axis acquisition search index is malformed")
    index = int(suffix)
    if faces[1].face_id != f"acquisition_far_{index:02d}":
        raise ValueError("axis acquisition candidate IDs do not share one index")
    if recommendation.material_target.face_id not in {
        face.face_id for face in faces
    }:
        raise ValueError("axis acquisition material target is not a candidate")
    return index


def axis_acquisition_feedback_binding(recommendation) -> dict[str, object]:
    """Canonical stable identity shared by the observer and planner."""

    index = axis_acquisition_search_index(recommendation)
    return {
        "stream_id": recommendation.stream_id,
        "axis_state": recommendation.axis_state,
        "search_index": index,
        "face_candidates": [
            {
                "face_id": face.face_id,
                "pose": asdict(face.pose),
            }
            for face in recommendation.face_candidates
        ],
        "material_target": {
            "face_id": recommendation.material_target.face_id,
            "pose": asdict(recommendation.material_target.pose),
            "evidence_state": recommendation.material_target.evidence_state,
        },
    }


def finite_feedback_pose(payload: object, *, name: str) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be an object")
    try:
        pose = {
            "x_m": float(payload["x_m"]),
            "y_m": float(payload["y_m"]),
            "yaw_rad": float(payload["yaw_rad"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} is incomplete") from exc
    if not all(math.isfinite(value) for value in pose.values()):
        raise ValueError(f"{name} must be finite")
    return pose


def validate_axis_acquisition_feedback(
    payload: object,
    *,
    now_unix_sec: float | None = None,
    max_age_sec: float | None = None,
) -> dict[str, object]:
    """Validate the complete sidecar without trusting its stored digest."""

    if not isinstance(payload, Mapping):
        raise ValueError("axis acquisition feedback must be an object")
    normalized = dict(payload)
    if normalized.get("schema_version") != AXIS_ACQUISITION_FEEDBACK_SCHEMA_VERSION:
        raise ValueError("unsupported axis acquisition feedback schema")
    if normalized.get("contract") != AXIS_ACQUISITION_FEEDBACK_CONTRACT:
        raise ValueError("axis acquisition feedback contract mismatch")
    if normalized.get("simulation_only") is not True:
        raise ValueError("axis acquisition feedback must be simulation_only")
    if normalized.get("state") not in {"pending", "consumed"}:
        raise ValueError("axis acquisition feedback state is invalid")

    finite_positive_fields = (
        "arrival_tolerance_m",
        "created_unix_sec",
    )
    values: dict[str, float] = {}
    for field in finite_positive_fields:
        try:
            values[field] = float(normalized[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"axis acquisition feedback {field} is incomplete"
            ) from exc
    if values["arrival_tolerance_m"] <= 0.0 or not math.isfinite(
        values["arrival_tolerance_m"]
    ):
        raise ValueError("axis acquisition feedback arrival tolerance is invalid")
    if values["created_unix_sec"] < 0.0 or not math.isfinite(
        values["created_unix_sec"]
    ):
        raise ValueError("axis acquisition feedback creation time must be finite")
    for field in ("source_observation_unix_sec", "source_sensor_stamp_sec"):
        try:
            value = float(normalized[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"axis acquisition feedback {field} is incomplete"
            ) from exc
        if value < 0.0 or not math.isfinite(value):
            raise ValueError(f"axis acquisition feedback {field} is invalid")
    if max_age_sec is not None:
        if (
            now_unix_sec is None
            or not math.isfinite(now_unix_sec)
            or not math.isfinite(max_age_sec)
            or max_age_sec <= 0.0
        ):
            raise ValueError("axis acquisition feedback freshness bound is invalid")
        age_sec = now_unix_sec - values["created_unix_sec"]
        if age_sec < -0.25:
            raise ValueError("axis acquisition feedback is future-dated")
        if age_sec > max_age_sec:
            raise ValueError("axis acquisition feedback is stale")

    binding = normalized.get("binding")
    if not isinstance(binding, Mapping):
        raise ValueError("axis acquisition feedback binding must be an object")
    binding = dict(binding)
    stream_id = binding.get("stream_id")
    if not isinstance(stream_id, str) or not stream_id:
        raise ValueError("axis acquisition feedback stream_id is invalid")
    if binding.get("axis_state") != "axis_acquisition":
        raise ValueError("axis acquisition feedback binding state mismatch")
    search_index = binding.get("search_index")
    if (
        isinstance(search_index, bool)
        or not isinstance(search_index, int)
        or search_index < 0
    ):
        raise ValueError("axis acquisition feedback search index is invalid")
    candidates = binding.get("face_candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError("axis acquisition feedback requires two bound candidates")
    expected_ids = (
        f"acquisition_near_{search_index:02d}",
        f"acquisition_far_{search_index:02d}",
    )
    candidate_poses: dict[str, dict[str, float]] = {}
    for candidate_index, (candidate, expected_id) in enumerate(
        zip(candidates, expected_ids)
    ):
        if not isinstance(candidate, Mapping):
            raise ValueError("axis acquisition feedback candidate is malformed")
        if candidate.get("face_id") != expected_id:
            raise ValueError("axis acquisition feedback candidate ID mismatch")
        candidate_poses[expected_id] = finite_feedback_pose(
            candidate.get("pose"),
            name=f"axis acquisition feedback candidate {candidate_index} pose",
        )
    material_target = binding.get("material_target")
    if (
        not isinstance(material_target, Mapping)
        or material_target.get("face_id") not in expected_ids
        or material_target.get("evidence_state") != "axis_acquisition"
    ):
        raise ValueError("axis acquisition feedback material target mismatch")
    material_target_pose = finite_feedback_pose(
        material_target.get("pose"),
        name="axis acquisition feedback material target pose",
    )
    if material_target_pose != candidate_poses[material_target["face_id"]]:
        raise ValueError(
            "axis acquisition feedback material target geometry mismatch"
        )
    binding_sha256 = normalized.get("binding_sha256")
    if (
        not isinstance(binding_sha256, str)
        or len(binding_sha256) != 64
        or any(character not in "0123456789abcdef" for character in binding_sha256)
        or binding_sha256 != canonical_json_sha256(binding)
    ):
        raise ValueError("axis acquisition feedback binding digest mismatch")

    held_target = normalized.get("held_active_target")
    if not isinstance(held_target, Mapping):
        raise ValueError("axis acquisition feedback held target is malformed")
    held_face_id = held_target.get("face_id")
    if (
        not isinstance(held_face_id, str)
        or not held_face_id.startswith(("acquisition_near_", "acquisition_far_"))
        or held_target.get("evidence_state") != "axis_acquisition"
    ):
        raise ValueError("axis acquisition feedback held target is not unresolved")
    finite_feedback_pose(
        held_target.get("pose"),
        name="axis acquisition feedback held target pose",
    )
    try:
        held_distance_m = float(normalized["held_distance_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "axis acquisition feedback held distance is incomplete"
        ) from exc
    if not math.isfinite(held_distance_m) or held_distance_m < 0.0:
        raise ValueError("axis acquisition feedback held distance is invalid")

    rejections = normalized.get("rejections")
    if not isinstance(rejections, list) or len(rejections) != 2:
        raise ValueError("axis acquisition feedback requires two rejections")
    for rejection, expected_id in zip(rejections, expected_ids):
        if not isinstance(rejection, Mapping):
            raise ValueError("axis acquisition feedback rejection is malformed")
        if rejection.get("face_id") != expected_id:
            raise ValueError("axis acquisition feedback rejection ID mismatch")
        if (
            rejection.get("failure_reason")
            not in AXIS_ACQUISITION_STATIC_TARGET_FAILURES
        ):
            raise ValueError(
                "axis acquisition feedback rejection is not a static target failure"
            )

    if normalized["state"] == "consumed":
        try:
            consumed_unix_sec = float(normalized["consumed_unix_sec"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "consumed axis acquisition feedback lacks consumption time"
            ) from exc
        if (
            not math.isfinite(consumed_unix_sec)
            or consumed_unix_sec < values["created_unix_sec"]
        ):
            raise ValueError(
                "axis acquisition feedback consumption time is invalid"
            )
    elif "consumed_unix_sec" in normalized:
        raise ValueError("pending axis acquisition feedback cannot be consumed")
    return normalized


def load_axis_acquisition_feedback(
    path: Path,
    *,
    now_unix_sec: float | None = None,
    max_age_sec: float | None = None,
) -> dict[str, object]:
    return validate_axis_acquisition_feedback(
        json.loads(path.read_text()),
        now_unix_sec=now_unix_sec,
        max_age_sec=max_age_sec,
    )


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def write_axis_acquisition_feedback(
    path: Path,
    payload: Mapping[str, object],
) -> dict[str, object]:
    validated = validate_axis_acquisition_feedback(payload)
    _atomic_json(path, validated)
    return validated


def consume_axis_acquisition_feedback(
    path: Path,
    *,
    expected_binding_sha256: str,
    consumed_unix_sec: float,
) -> dict[str, object]:
    """Atomically mark one exact pending event before observer state advances."""

    payload = load_axis_acquisition_feedback(path)
    if payload["state"] != "pending":
        raise ValueError("axis acquisition feedback was already consumed")
    if payload["binding_sha256"] != expected_binding_sha256:
        raise ValueError("axis acquisition feedback changed before consumption")
    if not math.isfinite(consumed_unix_sec):
        raise ValueError("axis acquisition feedback consumption time must be finite")
    consumed = {
        **payload,
        "state": "consumed",
        "consumed_unix_sec": consumed_unix_sec,
    }
    validated = validate_axis_acquisition_feedback(consumed)
    _atomic_json(path, validated)
    return validated
