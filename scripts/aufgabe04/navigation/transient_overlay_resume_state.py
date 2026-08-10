"""Immutable restart state for adopted transient coverage obstacles.

The physical coverage follower may adopt a run-local obstacle overlay before a
later localization continuity gate requests a stopped route reseal.  This
module carries that overlay, the already consumed replan budget, and the
adopted replacement-route hashes across the process boundary.  It is pure
Python: it neither imports ROS nor authorizes motion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Mapping

from scripts.aufgabe04.artifacts.content_store import (
    ContentStoreError,
    load_content_hashed_json,
    payload_sha256,
    write_content_hashed_json,
)
from scripts.aufgabe04.navigation.stand_blockage_replan import (
    load_transient_obstacle_overlay,
)
from scripts.aufgabe04.navigation.stand_coverage_survey import (
    CoverageSurveyPlan,
    coverage_survey_plan_sha256,
)


TRANSIENT_OVERLAY_RESUME_STATE_SCHEMA_VERSION = 1
TRANSIENT_OVERLAY_RESUME_STATE_HASH_FIELD = "resume_state_sha256"
TRANSIENT_OVERLAY_RESUME_DIAGNOSTICS_BINDING_KEY = (
    "transient_overlay_resume_state"
)
TRANSIENT_OVERLAY_RESUME_DIAGNOSTICS_BINDING_SCHEMA_VERSION = 1

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TRANSIENT_CANDIDATE_ID = re.compile(r"^transient_obstacle_([0-9]{4,})$")
_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "coverage_plan_sha256",
        "survey_id",
        "planning_frame",
        "map_bundle_sha256",
        "coverage_leg_index",
        "target_viewpoint_id",
        "completed_replan_count",
        "max_replans",
        "remaining_replans",
        "transient_obstacle_overlay_path",
        "transient_obstacle_overlay_sha256",
        "overlay_candidate_ids",
        "adopted_route_paths",
        "adopted_route_sha256s",
        "source_run_ids",
        "semantic_survey_evidence",
        "motion_continues_authorized",
        "automatic_motion_authorized",
    }
)
_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "resume_state_path",
        "resume_state_sha256",
        "coverage_leg_index",
        "target_viewpoint_id",
        "completed_replan_count",
        "max_replans",
        "remaining_replans",
        "semantic_survey_evidence",
        "motion_continues_authorized",
        "automatic_motion_authorized",
    }
)


@dataclass(frozen=True)
class TransientOverlayResumeState:
    """Fail-closed state needed to resume one stopped coverage leg."""

    schema_version: int
    coverage_plan_sha256: str
    survey_id: str
    planning_frame: str
    map_bundle_sha256: str
    coverage_leg_index: int
    target_viewpoint_id: str
    completed_replan_count: int
    max_replans: int
    remaining_replans: int
    transient_obstacle_overlay_path: str
    transient_obstacle_overlay_sha256: str
    overlay_candidate_ids: tuple[str, ...]
    adopted_route_paths: tuple[str, ...]
    adopted_route_sha256s: tuple[str, ...]
    source_run_ids: tuple[str, ...]
    semantic_survey_evidence: bool = False
    motion_continues_authorized: bool = False
    automatic_motion_authorized: bool = False


def transient_overlay_resume_state_payload(
    state: TransientOverlayResumeState,
) -> dict[str, object]:
    """Return the canonical unhashed JSON payload for ``state``."""

    return {
        "schema_version": state.schema_version,
        "coverage_plan_sha256": state.coverage_plan_sha256,
        "survey_id": state.survey_id,
        "planning_frame": state.planning_frame,
        "map_bundle_sha256": state.map_bundle_sha256,
        "coverage_leg_index": state.coverage_leg_index,
        "target_viewpoint_id": state.target_viewpoint_id,
        "completed_replan_count": state.completed_replan_count,
        "max_replans": state.max_replans,
        "remaining_replans": state.remaining_replans,
        "transient_obstacle_overlay_path": (
            state.transient_obstacle_overlay_path
        ),
        "transient_obstacle_overlay_sha256": (
            state.transient_obstacle_overlay_sha256
        ),
        "overlay_candidate_ids": list(state.overlay_candidate_ids),
        "adopted_route_paths": list(state.adopted_route_paths),
        "adopted_route_sha256s": list(state.adopted_route_sha256s),
        "source_run_ids": list(state.source_run_ids),
        "semantic_survey_evidence": state.semantic_survey_evidence,
        "motion_continues_authorized": state.motion_continues_authorized,
        "automatic_motion_authorized": state.automatic_motion_authorized,
    }


def transient_overlay_resume_state_sha256(
    state: TransientOverlayResumeState,
) -> str:
    """Hash the canonical state payload without granting it any authority."""

    return payload_sha256(transient_overlay_resume_state_payload(state))


def validate_transient_overlay_resume_state(
    state: TransientOverlayResumeState,
    *,
    plan: CoverageSurveyPlan,
    expected_coverage_leg_index: int | None = None,
    expected_target_viewpoint_id: str | None = None,
    expected_max_replans: int | None = None,
) -> None:
    """Validate identity, cumulative budget, and every referenced artifact."""

    if (
        type(state.schema_version) is not int
        or state.schema_version
        != TRANSIENT_OVERLAY_RESUME_STATE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported transient overlay resume-state schema")
    _require_sha256(state.coverage_plan_sha256, "coverage_plan_sha256")
    if state.coverage_plan_sha256 != coverage_survey_plan_sha256(plan):
        raise ValueError("resume-state coverage plan differs from supplied plan")
    for name, actual, expected in (
        ("survey_id", state.survey_id, plan.survey_id),
        ("planning_frame", state.planning_frame, plan.planning_frame),
        ("map_bundle_sha256", state.map_bundle_sha256, plan.map_bundle_sha256),
    ):
        _require_nonempty(actual, name)
        if actual != expected:
            raise ValueError(f"resume-state {name} differs from supplied plan")
    _require_nonnegative_int(state.coverage_leg_index, "coverage_leg_index")
    _require_nonempty(state.target_viewpoint_id, "target_viewpoint_id")
    if plan.viewpoint_for(state.target_viewpoint_id) is None:
        raise ValueError("resume-state target viewpoint is absent from plan")
    if (
        expected_coverage_leg_index is not None
        and state.coverage_leg_index != expected_coverage_leg_index
    ):
        raise ValueError("resume-state coverage leg differs from expected leg")
    if (
        expected_target_viewpoint_id is not None
        and state.target_viewpoint_id != expected_target_viewpoint_id
    ):
        raise ValueError("resume-state target differs from expected target")

    _require_positive_int(state.completed_replan_count, "completed_replan_count")
    _require_positive_int(state.max_replans, "max_replans")
    _require_nonnegative_int(state.remaining_replans, "remaining_replans")
    if state.completed_replan_count > state.max_replans:
        raise ValueError("resume-state completed replans exceed the maximum")
    if state.remaining_replans != (
        state.max_replans - state.completed_replan_count
    ):
        raise ValueError("resume-state remaining replan budget is inconsistent")
    if expected_max_replans is not None and state.max_replans != expected_max_replans:
        raise ValueError("resume-state maximum replan budget differs from expected")

    if state.semantic_survey_evidence is not False:
        raise ValueError("transient resume state cannot be semantic survey evidence")
    if state.motion_continues_authorized is not False:
        raise ValueError("transient resume state cannot continue motion")
    if state.automatic_motion_authorized is not False:
        raise ValueError("transient resume state cannot authorize automatic motion")

    overlay_path = _require_stored_canonical_file(
        state.transient_obstacle_overlay_path,
        "transient_obstacle_overlay_path",
    )
    _require_sha256(
        state.transient_obstacle_overlay_sha256,
        "transient_obstacle_overlay_sha256",
    )
    if _file_sha256(overlay_path) != state.transient_obstacle_overlay_sha256:
        raise ValueError("transient obstacle overlay hash mismatch")
    candidate_ids = _validated_overlay_candidate_ids(overlay_path, plan=plan)
    if candidate_ids != state.overlay_candidate_ids:
        raise ValueError("resume-state overlay candidate IDs differ from overlay")

    count = state.completed_replan_count
    if not (
        len(state.adopted_route_paths)
        == len(state.adopted_route_sha256s)
        == len(state.source_run_ids)
        == count
    ):
        raise ValueError("resume-state adopted-route provenance is incomplete")
    if len(set(state.adopted_route_paths)) != count:
        raise ValueError("resume-state adopted route paths are not unique")
    if len(set(state.adopted_route_sha256s)) != count:
        raise ValueError("resume-state adopted route hashes are not unique")
    for index, (path_text, expected_sha, run_id) in enumerate(
        zip(
            state.adopted_route_paths,
            state.adopted_route_sha256s,
            state.source_run_ids,
        ),
        start=1,
    ):
        route_path = _require_stored_canonical_file(
            path_text, f"adopted_route_paths[{index - 1}]"
        )
        _require_sha256(expected_sha, f"adopted_route_sha256s[{index - 1}]")
        if _file_sha256(route_path) != expected_sha:
            raise ValueError(f"adopted route {index} hash mismatch")
        _require_nonempty(run_id, f"source_run_ids[{index - 1}]")


def write_transient_overlay_resume_state(
    path: Path,
    state: TransientOverlayResumeState,
    *,
    plan: CoverageSurveyPlan,
) -> str:
    """Atomically publish an immutable, content-hashed resume state."""

    validate_transient_overlay_resume_state(state, plan=plan)
    try:
        return write_content_hashed_json(
            Path(path),
            transient_overlay_resume_state_payload(state),
            hash_field=TRANSIENT_OVERLAY_RESUME_STATE_HASH_FIELD,
        )
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc


def load_transient_overlay_resume_state(
    path: Path,
    *,
    plan: CoverageSurveyPlan,
    expected_coverage_leg_index: int | None = None,
    expected_target_viewpoint_id: str | None = None,
    expected_max_replans: int | None = None,
) -> TransientOverlayResumeState:
    """Load a strictly shaped state and live-rehash all referenced files."""

    try:
        payload = load_content_hashed_json(
            Path(path), hash_field=TRANSIENT_OVERLAY_RESUME_STATE_HASH_FIELD
        )
    except ContentStoreError as exc:
        raise ValueError(str(exc)) from exc
    _require_exact_fields(payload, _STATE_FIELDS, "resume state")
    state = _state_from_payload(payload)
    validate_transient_overlay_resume_state(
        state,
        plan=plan,
        expected_coverage_leg_index=expected_coverage_leg_index,
        expected_target_viewpoint_id=expected_target_viewpoint_id,
        expected_max_replans=expected_max_replans,
    )
    return state


def load_jsonl_event_objects(
    path: Path,
    *,
    start_offset: int = 0,
) -> tuple[dict[str, object], ...]:
    """Strictly load complete JSONL records at or after a byte boundary."""

    source = Path(path)
    _require_nonnegative_int(start_offset, "start_offset")
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"adaptive replan log must be a normal file: {source}")
    try:
        size = source.stat().st_size
        if start_offset > size:
            raise ValueError("JSONL start_offset exceeds the current file size")
        with source.open("rb") as handle:
            if start_offset > 0:
                handle.seek(start_offset - 1)
                if handle.read(1) != b"\n":
                    raise ValueError(
                        "JSONL start_offset is not a complete-record boundary"
                    )
            handle.seek(start_offset)
            data = handle.read()
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"adaptive replan log is not UTF-8: {source}") from exc
    except OSError as exc:
        raise ValueError(f"cannot read adaptive replan log: {source}") from exc
    events: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line, object_pairs_hook=_strict_object_pairs)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"invalid adaptive replan JSONL at line {line_number}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"adaptive replan JSONL line {line_number} must be an object"
            )
        events.append(payload)
    return tuple(events)


def read_adaptive_replan_events(
    path: Path,
) -> tuple[dict[str, object], ...]:
    """Backward-readable name for loading an adaptive event stream."""

    return load_jsonl_event_objects(path)


def update_transient_overlay_resume_state_from_events(
    events: Path | Iterable[Mapping[str, object]],
    *,
    plan: CoverageSurveyPlan,
    coverage_leg_index: int,
    target_viewpoint_id: str,
    max_replans: int,
    artifact_root: Path,
    expected_survey_root: Path | None = None,
    expected_session_root: Path | None = None,
    previous_state: TransientOverlayResumeState | None = None,
    source_run_id: str | None = None,
) -> TransientOverlayResumeState | None:
    """Fold adopted blockage events into one cumulative restart contract.

    ``events`` may be an adaptive JSONL path or an already decoded event list.
    When ``source_run_id`` is supplied, other child-run events are ignored so a
    session-wide JSONL can be processed one stopped child at a time.
    """

    _require_nonnegative_int(coverage_leg_index, "coverage_leg_index")
    _require_positive_int(max_replans, "max_replans")
    _require_nonempty(target_viewpoint_id, "target_viewpoint_id")
    if plan.viewpoint_for(target_viewpoint_id) is None:
        raise ValueError("target viewpoint is absent from coverage plan")
    if source_run_id is not None:
        _require_nonempty(source_run_id, "source_run_id")
    root = _canonical_directory(artifact_root, "artifact_root")
    survey_root = (
        None
        if expected_survey_root is None
        else _canonical_directory(expected_survey_root, "expected_survey_root")
    )
    session_root = (
        None
        if expected_session_root is None
        else _canonical_directory(expected_session_root, "expected_session_root")
    )
    for label, expected_root in (
        ("expected_survey_root", survey_root),
        ("expected_session_root", session_root),
    ):
        if expected_root is not None:
            try:
                expected_root.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"{label} escapes artifact_root") from exc

    if previous_state is not None:
        validate_transient_overlay_resume_state(
            previous_state,
            plan=plan,
            expected_coverage_leg_index=coverage_leg_index,
            expected_target_viewpoint_id=target_viewpoint_id,
            expected_max_replans=max_replans,
        )
    decoded = (
        load_jsonl_event_objects(events)
        if isinstance(events, Path)
        else tuple(_require_event_mapping(item) for item in events)
    )
    adopted = [
        item
        for item in decoded
        if item.get("event") == "transient_navigation_blockage_replanned"
        and (source_run_id is None or item.get("run_id") == source_run_id)
    ]
    if not adopted:
        return previous_state

    state = previous_state
    completed = 0 if state is None else state.completed_replan_count
    candidate_ids: tuple[str, ...] = (
        () if state is None else state.overlay_candidate_ids
    )
    route_paths: tuple[str, ...] = (
        () if state is None else state.adopted_route_paths
    )
    route_hashes: tuple[str, ...] = (
        () if state is None else state.adopted_route_sha256s
    )
    run_ids: tuple[str, ...] = () if state is None else state.source_run_ids
    overlay_path_text = (
        "" if state is None else state.transient_obstacle_overlay_path
    )
    overlay_sha = (
        "" if state is None else state.transient_obstacle_overlay_sha256
    )

    for event in adopted:
        event_run_id = _required_event_string(event, "run_id")
        if source_run_id is not None and event_run_id != source_run_id:
            raise AssertionError("source-run filter admitted another run")
        event_leg_index = _required_event_int(event, "leg_index")
        if event_leg_index != coverage_leg_index:
            raise ValueError("blockage event belongs to another coverage leg")
        event_target = _required_event_string(event, "target_viewpoint_id")
        if event_target != target_viewpoint_id:
            raise ValueError("blockage event belongs to another target viewpoint")
        if event.get("semantic_survey_evidence") is not False:
            raise ValueError("blockage event must not be semantic survey evidence")
        replan_index = _required_event_int(event, "replan_index")
        expected_index = completed + 1
        if replan_index != expected_index:
            raise ValueError(
                "blockage event replan indices must be contiguous: "
                f"expected {expected_index}, got {replan_index}"
            )
        if replan_index > max_replans:
            raise ValueError("blockage event exceeds the cumulative replan budget")

        overlay_path = _resolve_event_artifact(
            _required_event_string(event, "transient_obstacle_overlay_json"),
            artifact_root=root,
            label="transient obstacle overlay",
        )
        if survey_root is not None:
            expected_overlay = _canonicalize_existing_file(
                survey_root
                / "replans"
                / (
                    f"leg_{coverage_leg_index:03d}"
                    f"_replan_{replan_index:03d}"
                )
                / "transient_obstacle_overlay.json",
                "expected transient obstacle overlay",
            )
            if overlay_path != expected_overlay:
                raise ValueError(
                    "blockage event overlay path differs from its leg/replan slot"
                )
        next_candidate_ids = _validated_overlay_candidate_ids(
            overlay_path, plan=plan
        )
        _require_monotonic_overlay_extension(candidate_ids, next_candidate_ids)
        next_overlay_sha = _file_sha256(overlay_path)

        route_path = _resolve_event_artifact(
            _required_event_string(event, "replacement_route_csv"),
            artifact_root=root,
            label="replacement route",
        )
        if session_root is not None:
            expected_route = _canonicalize_existing_file(
                session_root
                / "execution"
                / (
                    f"coverage_leg_{coverage_leg_index:03d}"
                    f"_replan_{replan_index:03d}"
                )
                / "route.csv",
                "expected replacement route",
            )
            if route_path != expected_route:
                raise ValueError(
                    "blockage event route path differs from its leg/replan slot"
                )
        route_sha = _file_sha256(route_path)
        declared_route_sha = _required_event_string(
            event, "source_map_route_sha256"
        )
        _require_sha256(declared_route_sha, "source_map_route_sha256")
        if declared_route_sha != route_sha:
            raise ValueError("blockage event replacement route hash mismatch")
        if route_sha in route_hashes:
            raise ValueError("blockage event repeats an adopted route hash")
        if str(route_path) in route_paths:
            raise ValueError("blockage event repeats an adopted route path")

        completed = replan_index
        candidate_ids = next_candidate_ids
        overlay_path_text = str(overlay_path)
        overlay_sha = next_overlay_sha
        route_paths += (str(route_path),)
        route_hashes += (route_sha,)
        run_ids += (event_run_id,)

    state = TransientOverlayResumeState(
        schema_version=TRANSIENT_OVERLAY_RESUME_STATE_SCHEMA_VERSION,
        coverage_plan_sha256=coverage_survey_plan_sha256(plan),
        survey_id=plan.survey_id,
        planning_frame=plan.planning_frame,
        map_bundle_sha256=plan.map_bundle_sha256,
        coverage_leg_index=coverage_leg_index,
        target_viewpoint_id=target_viewpoint_id,
        completed_replan_count=completed,
        max_replans=max_replans,
        remaining_replans=max_replans - completed,
        transient_obstacle_overlay_path=overlay_path_text,
        transient_obstacle_overlay_sha256=overlay_sha,
        overlay_candidate_ids=candidate_ids,
        adopted_route_paths=route_paths,
        adopted_route_sha256s=route_hashes,
        source_run_ids=run_ids,
    )
    validate_transient_overlay_resume_state(
        state,
        plan=plan,
        expected_coverage_leg_index=coverage_leg_index,
        expected_target_viewpoint_id=target_viewpoint_id,
        expected_max_replans=max_replans,
    )
    return state


def refresh_transient_overlay_resume_state(
    state: TransientOverlayResumeState,
    *,
    overlay_path: Path,
    plan: CoverageSurveyPlan,
    artifact_root: Path,
) -> TransientOverlayResumeState:
    """Replace only the live overlay reference with a monotonic extension."""

    validate_transient_overlay_resume_state(state, plan=plan)
    root = _canonical_directory(artifact_root, "artifact_root")
    resolved = _resolve_event_artifact(
        str(overlay_path), artifact_root=root, label="transient obstacle overlay"
    )
    candidate_ids = _validated_overlay_candidate_ids(resolved, plan=plan)
    _require_monotonic_overlay_extension(state.overlay_candidate_ids, candidate_ids)
    refreshed = replace(
        state,
        transient_obstacle_overlay_path=str(resolved),
        transient_obstacle_overlay_sha256=_file_sha256(resolved),
        overlay_candidate_ids=candidate_ids,
    )
    validate_transient_overlay_resume_state(refreshed, plan=plan)
    return refreshed


def add_adopted_route_hash(
    state: TransientOverlayResumeState,
    *,
    route_path: Path,
    source_run_id: str,
    replan_index: int,
    plan: CoverageSurveyPlan,
    artifact_root: Path,
) -> TransientOverlayResumeState:
    """Consume exactly one more budget unit and bind its immutable route."""

    validate_transient_overlay_resume_state(state, plan=plan)
    _require_nonempty(source_run_id, "source_run_id")
    _require_positive_int(replan_index, "replan_index")
    if replan_index != state.completed_replan_count + 1:
        raise ValueError("adopted route replan index is not contiguous")
    if replan_index > state.max_replans:
        raise ValueError("adopted route exceeds the cumulative replan budget")
    root = _canonical_directory(artifact_root, "artifact_root")
    resolved = _resolve_event_artifact(
        str(route_path), artifact_root=root, label="replacement route"
    )
    route_sha = _file_sha256(resolved)
    if route_sha in state.adopted_route_sha256s:
        raise ValueError("replacement route repeats an adopted route hash")
    if str(resolved) in state.adopted_route_paths:
        raise ValueError("replacement route repeats an adopted route path")
    updated = replace(
        state,
        completed_replan_count=replan_index,
        remaining_replans=state.max_replans - replan_index,
        adopted_route_paths=state.adopted_route_paths + (str(resolved),),
        adopted_route_sha256s=state.adopted_route_sha256s + (route_sha,),
        source_run_ids=state.source_run_ids + (source_run_id,),
    )
    validate_transient_overlay_resume_state(updated, plan=plan)
    return updated


def bind_transient_overlay_resume_state_to_diagnostics(
    source_diagnostics_path: Path,
    output_diagnostics_path: Path,
    *,
    resume_state_path: Path,
    plan: CoverageSurveyPlan,
) -> str:
    """Copy motion-free diagnostics and bind a validated resume state.

    The returned value is the SHA-256 of the bound diagnostics bytes.  The
    output is immutable and may be supplied to the existing route sealer.
    """

    source = _canonicalize_existing_file(
        source_diagnostics_path, "source diagnostics"
    )
    canonical_state_path = _canonicalize_existing_file(
        resume_state_path, "resume state"
    )
    state = load_transient_overlay_resume_state(
        canonical_state_path, plan=plan
    )
    payload = _read_json_object(source, "source diagnostics")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("source diagnostics metadata must be an object")
    metadata = dict(metadata)
    if metadata.get("motion_authorized") is not False:
        raise ValueError("source diagnostics must be explicitly motion-free")
    _validate_diagnostics_identity(metadata, state)
    binding = _diagnostics_binding(state, canonical_state_path)
    existing = metadata.get(TRANSIENT_OVERLAY_RESUME_DIAGNOSTICS_BINDING_KEY)
    if existing is not None and existing != binding:
        raise ValueError("source diagnostics carry another resume-state binding")
    metadata[TRANSIENT_OVERLAY_RESUME_DIAGNOSTICS_BINDING_KEY] = binding
    bound_payload = dict(payload)
    bound_payload["metadata"] = metadata
    data = _pretty_json_bytes(bound_payload)
    _publish_immutable_bytes(Path(output_diagnostics_path), data)
    return hashlib.sha256(data).hexdigest()


def validate_transient_overlay_resume_state_diagnostics_binding(
    diagnostics_path: Path,
    *,
    resume_state_path: Path,
    plan: CoverageSurveyPlan,
    expected_coverage_leg_index: int | None = None,
    expected_target_viewpoint_id: str | None = None,
    expected_max_replans: int | None = None,
) -> TransientOverlayResumeState:
    """Validate the state reference preserved in source or sealed diagnostics."""

    diagnostics = _canonicalize_existing_file(diagnostics_path, "diagnostics")
    canonical_state_path = _canonicalize_existing_file(
        resume_state_path, "resume state"
    )
    state = load_transient_overlay_resume_state(
        canonical_state_path,
        plan=plan,
        expected_coverage_leg_index=expected_coverage_leg_index,
        expected_target_viewpoint_id=expected_target_viewpoint_id,
        expected_max_replans=expected_max_replans,
    )
    payload = _read_json_object(diagnostics, "diagnostics")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("diagnostics metadata must be an object")
    _validate_diagnostics_identity(metadata, state)
    binding = metadata.get(TRANSIENT_OVERLAY_RESUME_DIAGNOSTICS_BINDING_KEY)
    if not isinstance(binding, Mapping):
        raise ValueError("diagnostics have no transient resume-state binding")
    _require_exact_fields(binding, _BINDING_FIELDS, "diagnostics binding")
    expected = _diagnostics_binding(state, canonical_state_path)
    if dict(binding) != expected:
        raise ValueError("diagnostics transient resume-state binding differs")
    return state


def _state_from_payload(payload: Mapping[str, object]) -> TransientOverlayResumeState:
    return TransientOverlayResumeState(
        schema_version=_payload_int(payload, "schema_version"),
        coverage_plan_sha256=_payload_string(payload, "coverage_plan_sha256"),
        survey_id=_payload_string(payload, "survey_id"),
        planning_frame=_payload_string(payload, "planning_frame"),
        map_bundle_sha256=_payload_string(payload, "map_bundle_sha256"),
        coverage_leg_index=_payload_int(payload, "coverage_leg_index"),
        target_viewpoint_id=_payload_string(payload, "target_viewpoint_id"),
        completed_replan_count=_payload_int(payload, "completed_replan_count"),
        max_replans=_payload_int(payload, "max_replans"),
        remaining_replans=_payload_int(payload, "remaining_replans"),
        transient_obstacle_overlay_path=_payload_string(
            payload, "transient_obstacle_overlay_path"
        ),
        transient_obstacle_overlay_sha256=_payload_string(
            payload, "transient_obstacle_overlay_sha256"
        ),
        overlay_candidate_ids=_payload_string_tuple(
            payload, "overlay_candidate_ids"
        ),
        adopted_route_paths=_payload_string_tuple(payload, "adopted_route_paths"),
        adopted_route_sha256s=_payload_string_tuple(
            payload, "adopted_route_sha256s"
        ),
        source_run_ids=_payload_string_tuple(payload, "source_run_ids"),
        semantic_survey_evidence=_payload_bool(
            payload, "semantic_survey_evidence"
        ),
        motion_continues_authorized=_payload_bool(
            payload, "motion_continues_authorized"
        ),
        automatic_motion_authorized=_payload_bool(
            payload, "automatic_motion_authorized"
        ),
    )


def _diagnostics_binding(
    state: TransientOverlayResumeState,
    state_path: Path,
) -> dict[str, object]:
    return {
        "schema_version": (
            TRANSIENT_OVERLAY_RESUME_DIAGNOSTICS_BINDING_SCHEMA_VERSION
        ),
        "resume_state_path": str(state_path),
        "resume_state_sha256": transient_overlay_resume_state_sha256(state),
        "coverage_leg_index": state.coverage_leg_index,
        "target_viewpoint_id": state.target_viewpoint_id,
        "completed_replan_count": state.completed_replan_count,
        "max_replans": state.max_replans,
        "remaining_replans": state.remaining_replans,
        "semantic_survey_evidence": False,
        "motion_continues_authorized": False,
        "automatic_motion_authorized": False,
    }


def _validate_diagnostics_identity(
    metadata: Mapping[str, object],
    state: TransientOverlayResumeState,
) -> None:
    expected = {
        "plan_sha256": state.coverage_plan_sha256,
        "survey_id": state.survey_id,
        "planning_frame": state.planning_frame,
        "map_bundle_sha256": state.map_bundle_sha256,
        "target_viewpoint_id": state.target_viewpoint_id,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"diagnostics {field} differs from resume state")


def _validated_overlay_candidate_ids(
    path: Path,
    *,
    plan: CoverageSurveyPlan,
) -> tuple[str, ...]:
    payload = _read_json_object(path, "transient obstacle overlay")
    if payload.get("purpose") != "transient_navigation_obstacle":
        raise ValueError("transient obstacle overlay purpose is invalid")
    if payload.get("semantic_survey_evidence") is not False:
        raise ValueError("transient obstacle overlay is semantic survey evidence")
    if payload.get("motion_published") is not False:
        raise ValueError("transient obstacle overlay claims published motion")
    overlay = load_transient_obstacle_overlay(path, plan=plan)
    ids = tuple(candidate.candidate_uid for candidate in overlay.candidates)
    indices = []
    for candidate_id in ids:
        match = _TRANSIENT_CANDIDATE_ID.fullmatch(candidate_id)
        if match is None:
            raise ValueError("transient overlay candidate ID is not canonical")
        indices.append(int(match.group(1)))
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError("transient overlay candidate IDs are not monotonic")
    return ids


def _require_monotonic_overlay_extension(
    previous_ids: tuple[str, ...],
    next_ids: tuple[str, ...],
) -> None:
    if (
        len(next_ids) < len(previous_ids)
        or next_ids[: len(previous_ids)] != previous_ids
    ):
        raise ValueError("transient obstacle overlay is not a monotonic extension")


def _resolve_event_artifact(
    path_text: str,
    *,
    artifact_root: Path,
    label: str,
) -> Path:
    raw = Path(path_text)
    normalized = Path(os.path.normpath(str(raw)))
    if raw != normalized or any(part in {".", ".."} for part in raw.parts):
        raise ValueError(f"{label} path is not canonical")
    candidate = raw if raw.is_absolute() else artifact_root / raw
    absolute = Path(os.path.abspath(candidate))
    if candidate.is_symlink():
        raise ValueError(f"{label} path must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} artifact is unavailable: {candidate}") from exc
    if absolute != resolved:
        raise ValueError(f"{label} path contains a symlink or is not canonical")
    if not resolved.is_file():
        raise ValueError(f"{label} artifact is not a normal file")
    try:
        resolved.relative_to(artifact_root)
    except ValueError as exc:
        raise ValueError(f"{label} artifact escapes artifact_root") from exc
    return resolved


def _require_stored_canonical_file(path_text: str, label: str) -> Path:
    _require_nonempty(path_text, label)
    path = Path(path_text)
    if (
        not path.is_absolute()
        or path != Path(os.path.normpath(str(path)))
        or path.is_symlink()
    ):
        raise ValueError(f"{label} must be a canonical absolute non-symlink path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} artifact is unavailable: {path}") from exc
    if resolved != path or not resolved.is_file():
        raise ValueError(f"{label} must name a canonical normal file")
    return path


def _canonicalize_existing_file(path: Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError(f"{label} path must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {candidate}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} must be a canonical normal file")
    return resolved


def _canonical_directory(path: Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {candidate}") from exc
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a canonical directory")
    return resolved


def _file_sha256(path: Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"artifact must be a normal file: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"artifact is unavailable: {source}") from exc
    return digest.hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object_pairs,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {label} JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
    return payload


def _strict_object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _require_event_mapping(item: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(item, Mapping):
        raise ValueError("adaptive replan event must be a mapping")
    return dict(item)


def _required_event_string(event: Mapping[str, object], field: str) -> str:
    value = event.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"blockage event {field} must be a non-empty string")
    return value


def _required_event_int(event: Mapping[str, object], field: str) -> int:
    value = event.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"blockage event {field} must be an integer")
    return value


def _payload_string(payload: Mapping[str, object], field: str) -> str:
    value = payload[field]
    if not isinstance(value, str):
        raise ValueError(f"resume-state {field} must be a string")
    return value


def _payload_int(payload: Mapping[str, object], field: str) -> int:
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"resume-state {field} must be an integer")
    return value


def _payload_bool(payload: Mapping[str, object], field: str) -> bool:
    value = payload[field]
    if not isinstance(value, bool):
        raise ValueError(f"resume-state {field} must be a boolean")
    return value


def _payload_string_tuple(
    payload: Mapping[str, object], field: str
) -> tuple[str, ...]:
    value = payload[field]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"resume-state {field} must be a string list")
    return tuple(value)


def _require_exact_fields(
    payload: Mapping[str, object],
    expected: frozenset[str],
    label: str,
) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{label} fields differ: missing={missing}, unknown={unknown}"
        )


def _require_sha256(value: object, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _require_nonempty(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")


def _require_positive_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")


def _require_nonnegative_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _pretty_json_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"diagnostics payload is not finite JSON: {exc}") from exc


def _publish_immutable_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"refusing immutable symlink target: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"refusing to replace immutable artifact: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
                raise ValueError(f"refusing to replace immutable artifact: {path}")
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
