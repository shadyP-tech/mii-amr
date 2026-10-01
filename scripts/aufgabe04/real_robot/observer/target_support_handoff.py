"""Authenticate a passive target-support failure before deferring a candidate.

Only a successful, reaped child may report this target-local failure. A broken
transport, conflicting perception artifact, or malformed receipt remains a
terminal error; none of those conditions may trigger recovery motion.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json, payload_sha256,
)
from scripts.aufgabe04.real_robot.observer.diagnostics import load_passive_observer_status
from scripts.aufgabe04.real_robot.observer.target_reconciliation import load_reconciliation_snapshot
from scripts.aufgabe04.real_robot.observer.target_support_failure import (
    REASON, STATE, validate_target_support_failure,
)
from scripts.aufgabe04.stations.candidate_snapshot import candidate_snapshot_sha256


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate status key: {key}")
        result[key] = value
    return result


def load_target_support_failure_exit(
    *, status_path, process, candidate, snapshot_path, planning_frame, stream_id,
    robot_profile_sha256, calibration_profile_sha256, stand_model_profile_sha256,
    observation_not_before_sec=None,
):
    """Return authenticated failure evidence, or None for an ordinary outcome.

    ``robot_profile_sha256`` may be a zero-argument callable so ordinary observer
    outcomes do not acquire an additional profile-loading dependency.
    """
    status_path = Path(status_path)
    receipt_path = status_path.with_name("target_support_failure.json")
    receipt_present = receipt_path.exists() or receipt_path.is_symlink()
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, ValueError) as exc:
        if not receipt_present:
            return None
        raise RuntimeError("target support failure lacks readable final status") from exc
    declared = isinstance(status, dict) and (
        status.get("state") == STATE or "target_support_failure" in status
    )
    if not declared and not receipt_present:
        return None
    try:
        if (not isinstance(status, dict) or status.get("state") != STATE
                or status.get("reason") != REASON or status.get("motion_capability") != "none"):
            raise ValueError("target support failure final status is inconsistent")
        if (process.completion_kind != "child_exit" or process.artifact_kind is not None
                or process.artifact_path is not None or process.returncode != 0
                or process.deadline_expired or process.signals_sent
                or any(action not in ("exit_observed", "graceful_wait") for action in process.cleanup_actions)):
            raise ValueError("target support failure requires clean child exit without forced cleanup")
        for name in ("recommendation", "candidate_centering", "qr_observation_pose",
                     "inspection_observation", "axis_observation"):
            artifact = status_path.with_name(name + ".json")
            if artifact.exists() or artifact.is_symlink():
                raise ValueError("target support failure conflicts with a perception artifact")
        diagnostics = load_passive_observer_status(status_path)
        observation = status.get("observation_evidence")
        if (diagnostics.load_error is not None or not isinstance(observation, dict)
                or observation.get("poisoned") is not False or observation.get("poison_reason")):
            raise ValueError("target support failure has invalid or poisoned observation evidence")
        reference = status.get("target_support_failure")
        if (not isinstance(reference, dict) or not isinstance(reference.get("path"), str)
                or Path(reference["path"]).absolute() != receipt_path.absolute()):
            raise ValueError("target support failure reference differs from this attempt")
        payload = load_content_hashed_json(receipt_path, hash_field="target_support_failure_sha256")
        if reference.get("sha256") != payload_sha256(payload):
            raise ValueError("target support failure status digest mismatch")
        if snapshot_path is None:
            raise ValueError("target support failure lacks candidate snapshot")
        snapshot = load_reconciliation_snapshot(
            snapshot_path, candidate_uid=candidate.candidate_uid, planning_frame=planning_frame,
            center=(candidate.geometry.x_m, candidate.geometry.y_m),
        )
        geometry = snapshot.candidate_for(candidate.candidate_uid).geometry
        for name in ("radius_m", "uncertainty_m"):
            if not math.isclose(getattr(geometry, name), getattr(candidate.geometry, name), abs_tol=1e-9):
                raise ValueError("target support failure snapshot differs from selected geometry")
        binding = dict(
            candidate_uid=candidate.candidate_uid, stream_id=stream_id,
            target_key=f"{stream_id}:{candidate.candidate_uid}:{candidate.geometry.x_m:.9f}:{candidate.geometry.y_m:.9f}",
            planning_frame=planning_frame,
            stand_center=dict(x_m=candidate.geometry.x_m, y_m=candidate.geometry.y_m),
            candidate_snapshot_sha256=candidate_snapshot_sha256(snapshot),
            robot_profile_sha256=robot_profile_sha256() if callable(robot_profile_sha256) else robot_profile_sha256,
            calibration_profile_sha256=calibration_profile_sha256,
            stand_model_profile_sha256=stand_model_profile_sha256,
        )
        payload = validate_target_support_failure(payload, target_binding=binding)
        if (observation.get("target_key") != binding["target_key"]
                or type(observation.get("motion_epoch")) is not int
                or observation["motion_epoch"] != payload["samples"][-1]["frame"]["motion_epoch"]):
            raise ValueError("target support failure final observation target or epoch mismatch")
        if observation_not_before_sec is not None:
            if not math.isfinite(observation_not_before_sec) or any(
                sample["frame"][key] < observation_not_before_sec
                for sample in payload["samples"] for key in ("frame_stamp_sec", "scan_stamp_sec")
            ):
                raise ValueError("target support failure predates the current stopped observation")
    except (OSError, TypeError, ValueError, KeyError, AttributeError) as exc:
        raise RuntimeError(f"invalid target support failure: {exc}") from exc
    return {
        **diagnostics.to_dict(), "target_support_failure": {
            "path": str(receipt_path), "sha256": payload_sha256(payload),
            "state": STATE, "reason": REASON, "sample_count": payload["sample_count"],
            "elapsed_sec": payload["elapsed_sec"], "target_binding": payload["target_binding"],
            "motion_authorized": False, "completion_authorized": False,
        },
    }
