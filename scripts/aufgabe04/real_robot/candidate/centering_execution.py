"""Bounded centering within one physical inspection view.

The observer remains motion-neutral. Injected turn effects own authorization,
live motion and stopped arrival admission. Every new capture owns a new epoch.
"""
from __future__ import annotations

import math
from pathlib import Path
import time

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.real_robot.candidate.observation_deferral import (
    CandidateObservationUnavailableError,
)


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
    if state_path.exists():
        raise RuntimeError("refusing to reset an existing inspection centering budget")
    deadline = monotonic() + timeout_sec
    history = []
    travel = 0.
    not_before = None
    previous_result_path = None

    def persist(phase):
        output_dir.mkdir(parents=True, exist_ok=True)
        write_content_hashed_json(state_path, {
            "schema_version": 1, "candidate_uid": candidate_uid,
            "physical_view_index": view_index, "phase": phase,
            "maximum_turn_count": MAX_CENTERING_TURNS,
            "maximum_angular_travel_rad": MAX_CENTERING_TRAVEL_RAD,
            "actual_angular_travel_rad": travel,
            "observation_not_before_sec": not_before,
            "turn_history": history, "motion_authorized": False,
        }, hash_field="candidate_centering_progress_sha256")

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
        enabled = len(history) < MAX_CENTERING_TURNS and travel < MAX_CENTERING_TRAVEL_RAD
        capture_dir = (output_dir if not history else
                       output_dir / f"recenter_{len(history):02d}" / "capture")
        observation = capture(frame, capture_dir, view_index, enabled, remaining_sec, not_before)
        # Completion always outranks an optional advisory, including injected
        # effects that expose both artifacts in the same result.
        complete = (observation.recommendation_path is not None or
                    getattr(observation, "qr_observation_pose_path", None) is not None)
        advisory_path = getattr(observation, "centering_advisory_path", None)
        if complete or advisory_path is None:
            if history:
                persist("observation_returned")
            return observation, frame
        if not enabled:
            raise RuntimeError("observer emitted centering after its view budget was disabled")
        turn_index = len(history)
        history.append({"turn_index": turn_index, "state": "reserved",
                        "advisory_path": str(advisory_path)})
        persist("turn_reserved")  # A failed attempt must not renew authority.
        frame, outcome = turn(
            frame, advisory_path, output_dir / f"recenter_{turn_index + 1:02d}",
            turn_index, MAX_CENTERING_TRAVEL_RAD - travel, previous_result_path,
        )
        result = outcome.result
        actual = result.get("actual_angular_travel_rad")
        stopped = result.get("stopped_at_sec")
        if (type(actual) not in (int, float) or not math.isfinite(actual) or actual < 0
                or travel + actual > MAX_CENTERING_TRAVEL_RAD + 1e-9
                or type(stopped) not in (int, float) or not math.isfinite(stopped)
                or stopped <= 0 or (not_before is not None and stopped <= not_before)):
            raise RuntimeError("invalid centering travel or stopped timestamp evidence")
        travel += actual
        not_before = stopped
        previous_result_path = outcome.result_path
        history[-1].update(state="stopped", result_path=str(outcome.result_path),
                           actual_angular_travel_rad=actual, stopped_at_sec=stopped)
        persist("fresh_observation_required")
