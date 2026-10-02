"""Bounded centering within one physical inspection view.

The observer remains motion-neutral. Injected turn effects own authorization,
live motion and stopped arrival admission. Every new capture owns a new epoch.
"""
from __future__ import annotations

import math
import json
import os
from pathlib import Path
import time

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.real_robot.candidate.observation_deferral import (
    CandidateObservationUnavailableError,
)
from scripts.aufgabe04.real_robot.execution.candidate_centering import CandidateCenteringDisposition


MAX_CENTERING_TURNS = 2
MAX_CENTERING_TRAVEL_RAD = math.radians(12.)


def capture_with_centering(
    *, candidate_uid, frame, output_dir: Path, view_index: int, timeout_sec: float,
    capture, turn, monotonic=time.monotonic,
):
    """Return (observation, stopped frame) without spending another view slot.

    ``capture`` receives the frame, output path, view index, advice-enabled flag,
    remaining timeout and exclusive sensor timestamp floor. ``turn`` returns a
    fresh admitted frame and a validated child outcome. Failures propagate;
    replaying an already-started centering budget is explicitly refused.
    """
    if not math.isfinite(timeout_sec) or timeout_sec <= 0:
        raise ValueError("camera timeout must be finite and positive")
    state_path = output_dir / "centering_progress.json"
    history_dir = output_dir / "centering_history"
    if state_path.exists() or state_path.is_symlink() or history_dir.exists():
        raise RuntimeError("refusing to reset an existing inspection centering budget")
    deadline = monotonic() + timeout_sec
    history = []
    travel = 0.
    not_before = None
    previous_result_path = None
    revision = 0
    turn_limit = MAX_CENTERING_TURNS
    travel_limit = MAX_CENTERING_TRAVEL_RAD
    arrival_recovery = False
    observation_status_path = None
    capture_only = False

    def persist(phase):
        nonlocal revision
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1, "candidate_uid": candidate_uid,
            "physical_view_index": view_index, "phase": phase,
            "maximum_turn_count": turn_limit, "arrival_recovery": arrival_recovery,
            "centering_disabled_after_bounded_stop": capture_only,
            "maximum_angular_travel_rad": travel_limit,
            "actual_angular_travel_rad": travel,
            "observation_not_before_sec": not_before,
            "turn_history": history, "motion_authorized": False,
            "camera_centering_status_path": observation_status_path,
            # A stopped turn/arrival is not a camera measurement. The linked
            # observer status reports centered, blocked or deferred explicitly.
            "camera_centering_verification": "requires_current_observer_measurement",
        }
        receipt = history_dir / f"revision_{revision:03d}.json"
        digest = write_content_hashed_json(receipt, payload, hash_field="candidate_centering_progress_sha256")
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({**payload, "latest_revision_path": str(receipt),
            "latest_revision_sha256": digest}, indent=2, sort_keys=True)+"\n")
        os.replace(temporary, state_path)
        revision += 1

    while True:
        remaining_sec = deadline - monotonic()
        if remaining_sec <= 0:
            persist("view_deadline_expired")
            raise CandidateObservationUnavailableError(
                candidate_uid=candidate_uid, observation_attempt_index=view_index,
                reason="candidate_centering_view_deadline_expired",
                process_evidence={"observer_started": True,
                                  "centering_progress_path": str(state_path)},
                status_evidence={"motion_authorized": False},
            )
        enabled = not capture_only and len(history) < turn_limit and travel < travel_limit
        capture_dir = (output_dir if not history else
                       output_dir / f"recenter_{len(history):02d}" / "capture")
        observation = capture(frame, capture_dir, view_index, enabled, remaining_sec, not_before)
        observation_status_path = str(capture_dir / "observer_status.json")
        # A usable stopped observation precedes optional framing. Movement still
        # requires another capture with a strictly newer sensor timestamp.
        complete = (observation.recommendation_path is not None or
                    getattr(observation, "qr_observation_pose_path", None) is not None or
                    getattr(observation, "axis_observation_path", None) is not None)
        advisory_path = getattr(observation, "centering_advisory_path", None)
        if advisory_path is None or complete or capture_only:
            persist("observation_returned")
            return observation, frame
        if not history and Path(advisory_path).is_file():
            from scripts.aufgabe04.real_robot.observer.candidate_centering import validate_camera_centering_advisory
            from scripts.aufgabe04.real_robot.observer.candidate_position_epoch import RECOVERY_TRAVEL_RAD, RECOVERY_TURNS
            advice = validate_camera_centering_advisory(json.loads(Path(advisory_path).read_text()))
            if advice.arrival_recovery:
                arrival_recovery = True
                travel_limit,turn_limit = RECOVERY_TRAVEL_RAD,RECOVERY_TURNS
        if not enabled:
            raise RuntimeError("observer emitted centering after its view budget was disabled")
        turn_index = len(history)
        history.append({"turn_index": turn_index, "state": "reserved",
                        "advisory_path": str(advisory_path)})
        persist("turn_reserved")  # A failed attempt must not renew authority.
        try:
            frame, outcome = turn(
                frame, advisory_path, output_dir / f"recenter_{turn_index + 1:02d}",
                turn_index, travel_limit - travel, previous_result_path,
            )
        except Exception as exc:
            history[-1].update(state="failed", reason=f"{type(exc).__name__}: {exc}")
            persist("turn_failed")
            raise
        result = outcome.result
        disposition = getattr(outcome, "disposition", CandidateCenteringDisposition.COMPLETED)
        if (disposition not in {CandidateCenteringDisposition.COMPLETED,
                               CandidateCenteringDisposition.STOPPED_CAPTURE_ONLY}
                or result.get("status") not in {None, "completed", "stopped"}
                or (result.get("status") == "stopped"
                    and disposition is not CandidateCenteringDisposition.STOPPED_CAPTURE_ONLY)):
            raise RuntimeError("invalid centering outcome classification")
        capture_only = disposition is CandidateCenteringDisposition.STOPPED_CAPTURE_ONLY
        actual = result.get("actual_angular_travel_rad")
        stopped = result.get("stopped_at_sec")
        if (type(actual) not in (int, float) or not math.isfinite(actual) or actual < 0
                or travel + actual > travel_limit + 1e-9
                or type(stopped) not in (int, float) or not math.isfinite(stopped)
                or stopped <= 0 or (not_before is not None and stopped <= not_before)):
            raise RuntimeError("invalid centering travel or stopped timestamp evidence")
        travel += actual
        not_before = stopped
        previous_result_path = outcome.result_path
        history[-1].update(state="stopped", result_path=str(outcome.result_path),
                           actual_angular_travel_rad=actual, stopped_at_sec=stopped,
                           disposition=disposition.value)
        persist("fresh_observation_required")
