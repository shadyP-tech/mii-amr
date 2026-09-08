"""Candidate-local observation/search loop behind injected motion effects."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Callable, Generic, Mapping, TypeVar

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.real_robot.candidate.inspection_policy import (
    CandidateInspectionState, candidate_view_options,
)
from scripts.aufgabe04.real_robot.candidate.inspection_route_search import (
    CandidateInspectionRouteUnavailableError,
)
from scripts.aufgabe04.real_robot.candidate.observation_deferral import (
    CandidateObservationUnavailableError,
)


Frame = TypeVar("Frame")
Observation = TypeVar("Observation")


@dataclass(frozen=True)
class CandidateInspectionEffects(Generic[Frame, Observation]):
    capture: Callable[[Frame, Path, int], Observation]
    canonical_normal: Callable[[Frame], float | None]
    move_view: Callable[[Frame, float, Path, int, Path | None], Frame]
    move_opposite: Callable[[Frame, Observation, Path, int], Frame]
    progress_evidence: Callable[[Frame, Observation], dict[str, object]]
    route_search_evidence: Callable[[], Mapping[str, object]] | None = None


def execute_candidate_inspection(
    *, candidate_uid: str, candidate_root: Path, initial_frame: Frame,
    max_views: int, effects: CandidateInspectionEffects[Frame, Observation],
) -> tuple[Observation, Frame]:
    """Keep one candidate active until joint success or useful views exhaust.

    Intermediate QR and axis diagnostics are durable progress, never completion
    authority. All live effects remain injected and all failures other than
    explicitly local observation/no-motion feasibility failures propagate.
    """

    state = CandidateInspectionState(candidate_uid, max_views)
    progress_path = candidate_root / "inspection_progress.json"
    frame = initial_frame
    last_error: CandidateObservationUnavailableError | None = None
    revision = 0

    def persist() -> None:
        nonlocal revision
        payload = {**state.to_dict(), **({} if effects.route_search_evidence is None
                                        else dict(effects.route_search_evidence()))}
        receipt = candidate_root / "inspection_history" / f"revision_{revision:03d}.json"
        digest = write_content_hashed_json(receipt, payload,
                                          hash_field="candidate_inspection_progress_sha256")
        pointer = {**payload, "latest_revision_path": str(receipt),
                   "latest_revision_sha256": digest}
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(pointer, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, progress_path)
        revision += 1

    while len(state.history) < max_views:
        index = len(state.history)
        observation = None
        progress: dict[str, object] = {}
        normal = effects.canonical_normal(frame)
        try:
            observation = effects.capture(
                frame, candidate_root / f"camera_lidar_attempt_{index:02d}", index,
            )
            if observation.recommendation_path is not None:
                state.record(outcome="resolved", normal=normal,
                             observation={"qr_id": observation.qr_id,
                                          "recommendation_path": str(observation.recommendation_path)})
                state.termination_reason = "joint_observation_ready"
                persist()
                return observation, frame
            if observation.axis_observation_path is not None:
                progress = {"classification": "certified_backside",
                            "axis_observation_path": str(observation.axis_observation_path)}
            elif observation.inspection_observation_path is not None:
                progress = effects.progress_evidence(frame, observation)
            else:
                raise RuntimeError("observer returned no recommendation, certified axis, or inspection progress")
            state.record(outcome="inspection_pending", normal=normal, observation=progress)
        except CandidateObservationUnavailableError as exc:
            last_error = exc
            state.record(outcome="observation_unavailable", normal=normal, reason=str(exc),
                         observation={"classification": "unobservable", **exc.to_event_fields()})
        persist()
        if len(state.history) >= max_views:
            state.termination_reason = "view_budget_exhausted"
            break
        if observation is not None and observation.axis_observation_path is not None:
            try:
                frame = effects.move_opposite(
                    frame, observation, candidate_root / f"inspection_opposite_{index + 1:02d}", index + 1,
                )
                continue
            except CandidateInspectionRouteUnavailableError as exc:
                state.route_failures.append({"view_kind": "certified_opposite", "reason": str(exc)})
                persist()
        if normal is None:
            raise RuntimeError("candidate inspection search lacks a finite observation pose")
        classification = str(progress.get("classification", "unobservable"))
        source_path = None if observation is None else observation.inspection_observation_path
        selected = False
        for option_index, next_normal in enumerate(candidate_view_options(
            normal, classification=classification, achieved_normals=state.achieved_normals,
            attempted_normals=state.attempted_normals,
            exhausted_normals=state.exhausted_normals,
            advisory_yaw_rad=progress.get("camera_relative_yaw_rad"),
        )):
            state.attempted_normals.append(next_normal)
            persist()
            try:
                frame = effects.move_view(
                    frame, next_normal,
                    candidate_root / f"inspection_view_{index + 1:02d}_{option_index:02d}",
                    index + 1, source_path,
                )
            except CandidateInspectionRouteUnavailableError as exc:
                state.route_failures.append({
                    "requested_normal_rad": next_normal, "reason": str(exc),
                    "reason_code": exc.reason_code, "proposal_evidence": exc.evidence,
                })
                if exc.reason_code == "route_proposal_budget_exhausted":
                    state.termination_reason = "route_proposal_budget_exhausted"
                    persist()
                    break
                state.exhausted_normals.append(next_normal)
                persist()
                continue
            selected = True
            break
        if not selected:
            if state.termination_reason is None:
                state.termination_reason = "view_proposals_exhausted"
            break
    persist()
    raise CandidateObservationUnavailableError(
        candidate_uid=candidate_uid, observation_attempt_index=max(0, len(state.history) - 1),
        reason="candidate_local_inspection_exhausted",
        process_evidence={"inspection_progress_path": str(progress_path),
                          "observer_started": bool(state.history),
                          "local_view_count": len(state.history),
                          "max_candidate_inspection_views": max_views,
                          "inspection_exhaustion_reason": state.termination_reason,
                          "last_observer_failure": None if last_error is None else str(last_error)},
        status_evidence={**state.to_dict(), **({} if effects.route_search_evidence is None
                                              else dict(effects.route_search_evidence()))},
    )
