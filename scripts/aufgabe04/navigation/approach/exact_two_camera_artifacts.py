"""Immutable artifacts and live bindings for exact-two camera admission."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Mapping

from scripts.aufgabe04.artifacts.content_store import (
    ContentStoreError,
    content_hashed_payload,
    load_content_hashed_json,
    payload_sha256,
    write_content_hashed_json,
)
from scripts.aufgabe04.navigation.approach.exact_two_camera_contract import (
    EXACT_TWO_CAMERA_ADMISSION_SCHEMA_VERSION,
    EXACT_TWO_CAMERA_HANDOFF_SCHEMA_VERSION,
    ExactTwoCameraAdmissionDecision,
    ExactTwoCameraAdmissionError,
    ExactTwoCameraCandidateEvidence,
    ExactTwoCameraHandoffArtifact,
    boolean,
    finite_nonnegative,
    require_admitted_candidate_support,
    stand_survey_registry_sha256,
    validate_exact_two_camera_admission,
    validate_frame,
    validate_id,
    validate_sha256,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import StandSurveyRegistry
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateSnapshot,
    CandidateSnapshotError,
    candidate_snapshot_sha256,
    load_candidate_snapshot,
    validate_candidate_snapshot,
)


ADMISSION_HASH_FIELD = "exact_two_camera_admission_sha256"
HANDOFF_HASH_FIELD = "exact_two_camera_handoff_sha256"

_ADMISSION_FIELDS = frozenset(
    {
        "schema_version",
        "survey_id",
        "planning_frame",
        "map_bundle_sha256",
        "plan_sha256",
        "progress_snapshot_sha256",
        "source_registry_sha256",
        "lidar_checkpoint_sha256",
        "ready",
        "reasons",
        "camera_population_ready",
        "motion_authorized",
        "expected_stand_count",
        "active_candidate_count",
        "multi_view_candidate_uids",
        "single_view_candidate_uids",
        "blocked_candidate_uids",
        "candidate_evidence",
    }
)
_EVIDENCE_FIELDS = frozenset(
    {
        "candidate_uid",
        "registry_status",
        "active_lidar",
        "static_map_disposition",
        "static_map_admitted",
        "static_map_population_retained",
        "basic_lidar_supported",
        "confidence",
        "minimum_confidence",
        "confidence_supported",
        "hit_count",
        "minimum_hit_count",
        "hit_count_supported",
        "viewpoint_ids",
        "known_viewpoint_ids",
        "unknown_viewpoint_ids",
        "viewpoint_ids_distinct",
        "distinct_known_viewpoint_count",
        "support_class",
        "source_kind",
        "admissible",
        "reasons",
    }
)
_HANDOFF_FIELDS = frozenset(
    {
        "schema_version",
        "handoff_id",
        "created_unix_sec",
        "survey_id",
        "planning_frame",
        "map_bundle_sha256",
        "plan_sha256",
        "progress_snapshot_sha256",
        "source_registry_sha256",
        "terminal_checkpoint_path",
        "terminal_checkpoint_sha256",
        "lidar_admission_path",
        "lidar_admission_sha256",
        "lidar_checkpoint_sha256",
        "camera_admission_path",
        "camera_admission_sha256",
        "candidate_snapshot_path",
        "candidate_snapshot_sha256",
        "candidate_snapshot_id",
        "camera_population_ready",
        "motion_authorized",
        "admission_decision",
    }
)


def exact_two_camera_admission_evidence(
    decision: ExactTwoCameraAdmissionDecision,
) -> dict[str, object]:
    validate_exact_two_camera_admission(decision)
    return decision.to_evidence_dict()


def exact_two_camera_admission_sha256(
    value: ExactTwoCameraAdmissionDecision | Mapping[str, object],
) -> str:
    if isinstance(value, ExactTwoCameraAdmissionDecision):
        payload = exact_two_camera_admission_evidence(value)
    elif isinstance(value, Mapping):
        decision = admission_from_payload(value)
        validate_exact_two_camera_admission(decision)
        payload = decision.to_evidence_dict()
    else:
        raise ExactTwoCameraAdmissionError(
            "invalid_admission", "admission must be a decision or mapping"
        )
    return payload_sha256(payload)


def exact_two_camera_admission_payload(
    decision: ExactTwoCameraAdmissionDecision,
) -> dict[str, object]:
    return content_hashed_payload(
        exact_two_camera_admission_evidence(decision),
        hash_field=ADMISSION_HASH_FIELD,
    )


def write_exact_two_camera_admission(
    path: Path, decision: ExactTwoCameraAdmissionDecision
) -> str:
    try:
        return write_content_hashed_json(
            Path(path),
            exact_two_camera_admission_evidence(decision),
            hash_field=ADMISSION_HASH_FIELD,
        )
    except ContentStoreError as exc:
        raise ExactTwoCameraAdmissionError(exc.code, str(exc)) from exc


def load_exact_two_camera_admission(
    path: Path,
) -> ExactTwoCameraAdmissionDecision:
    try:
        payload = load_content_hashed_json(Path(path), hash_field=ADMISSION_HASH_FIELD)
    except ContentStoreError as exc:
        raise ExactTwoCameraAdmissionError(exc.code, str(exc)) from exc
    decision = admission_from_payload(payload)
    validate_exact_two_camera_admission(decision)
    return decision


def new_exact_two_camera_handoff(
    *,
    handoff_id: str,
    created_unix_sec: float,
    admission: ExactTwoCameraAdmissionDecision,
    terminal_checkpoint_path: str | Path,
    terminal_checkpoint_sha256: str,
    lidar_admission_path: str | Path,
    lidar_admission_sha256: str,
    camera_admission_path: str | Path,
    camera_admission_sha256: str,
    candidate_snapshot_path: str | Path,
    candidate_snapshot: CandidateSnapshot,
) -> ExactTwoCameraHandoffArtifact:
    """Seal wrapper and bare-checkpoint hashes as distinct provenance edges."""

    validate_exact_two_camera_admission(admission)
    if not admission.ready:
        raise ExactTwoCameraAdmissionError(
            "camera_population_not_ready",
            "cannot create a handoff from a not-ready admission",
        )
    validate_sha256(lidar_admission_sha256, "lidar_admission_sha256")
    if camera_admission_sha256 != exact_two_camera_admission_sha256(admission):
        raise ExactTwoCameraAdmissionError(
            "provenance_mismatch",
            "camera admission hash does not match decision",
        )
    _validate_snapshot_against_admission(candidate_snapshot, admission)
    handoff = ExactTwoCameraHandoffArtifact(
        schema_version=EXACT_TWO_CAMERA_HANDOFF_SCHEMA_VERSION,
        handoff_id=handoff_id,
        created_unix_sec=created_unix_sec,
        survey_id=admission.survey_id,
        planning_frame=admission.planning_frame,
        map_bundle_sha256=admission.map_bundle_sha256,
        plan_sha256=admission.plan_sha256,
        progress_snapshot_sha256=admission.progress_snapshot_sha256,
        source_registry_sha256=admission.source_registry_sha256,
        terminal_checkpoint_path=path_string(
            terminal_checkpoint_path, "terminal_checkpoint_path"
        ),
        terminal_checkpoint_sha256=terminal_checkpoint_sha256,
        lidar_admission_path=path_string(
            lidar_admission_path, "lidar_admission_path"
        ),
        lidar_admission_sha256=lidar_admission_sha256,
        lidar_checkpoint_sha256=admission.lidar_checkpoint_sha256,
        camera_admission_path=path_string(
            camera_admission_path, "camera_admission_path"
        ),
        camera_admission_sha256=camera_admission_sha256,
        candidate_snapshot_path=path_string(
            candidate_snapshot_path, "candidate_snapshot_path"
        ),
        candidate_snapshot_sha256=candidate_snapshot_sha256(candidate_snapshot),
        candidate_snapshot_id=candidate_snapshot.snapshot_id,
        camera_population_ready=True,
        motion_authorized=False,
        admission_decision=admission,
    )
    validate_exact_two_camera_handoff(handoff)
    return handoff


def validate_exact_two_camera_handoff(
    handoff: ExactTwoCameraHandoffArtifact,
) -> None:
    if not isinstance(handoff, ExactTwoCameraHandoffArtifact):
        raise ExactTwoCameraAdmissionError(
            "invalid_handoff", "handoff must be an ExactTwoCameraHandoffArtifact"
        )
    if (
        type(handoff.schema_version) is not int
        or handoff.schema_version != EXACT_TWO_CAMERA_HANDOFF_SCHEMA_VERSION
    ):
        raise ExactTwoCameraAdmissionError(
            "schema_mismatch", "unsupported exact-two camera handoff schema"
        )
    validate_id(handoff.handoff_id, "handoff_id")
    finite_nonnegative(handoff.created_unix_sec, "created_unix_sec")
    validate_id(handoff.survey_id, "survey_id")
    validate_frame(handoff.planning_frame, "planning_frame")
    for field_name in (
        "map_bundle_sha256",
        "plan_sha256",
        "progress_snapshot_sha256",
        "source_registry_sha256",
        "terminal_checkpoint_sha256",
        "lidar_admission_sha256",
        "lidar_checkpoint_sha256",
        "camera_admission_sha256",
        "candidate_snapshot_sha256",
    ):
        validate_sha256(getattr(handoff, field_name), field_name)
    for field_name in (
        "terminal_checkpoint_path",
        "lidar_admission_path",
        "camera_admission_path",
        "candidate_snapshot_path",
    ):
        path_string(getattr(handoff, field_name), field_name)
    validate_id(handoff.candidate_snapshot_id, "candidate_snapshot_id")
    boolean(handoff.camera_population_ready, "camera_population_ready")
    boolean(handoff.motion_authorized, "motion_authorized")
    if not handoff.camera_population_ready or handoff.motion_authorized:
        raise ExactTwoCameraAdmissionError(
            "motion_scope_violation",
            "handoff must be camera-ready and motion-neutral",
        )
    validate_exact_two_camera_admission(handoff.admission_decision)
    decision = handoff.admission_decision
    expected = {
        "survey_id": decision.survey_id,
        "planning_frame": decision.planning_frame,
        "map_bundle_sha256": decision.map_bundle_sha256,
        "plan_sha256": decision.plan_sha256,
        "progress_snapshot_sha256": decision.progress_snapshot_sha256,
        "source_registry_sha256": decision.source_registry_sha256,
        "lidar_checkpoint_sha256": decision.lidar_checkpoint_sha256,
        "camera_admission_sha256": exact_two_camera_admission_sha256(decision),
    }
    for field_name, value in expected.items():
        if getattr(handoff, field_name) != value:
            raise ExactTwoCameraAdmissionError(
                "provenance_mismatch",
                f"handoff {field_name} does not match admission",
            )


def exact_two_camera_handoff_payload(
    handoff: ExactTwoCameraHandoffArtifact,
) -> dict[str, object]:
    return content_hashed_payload(
        _handoff_payload_without_hash(handoff), hash_field=HANDOFF_HASH_FIELD
    )


def exact_two_camera_handoff_sha256(
    handoff: ExactTwoCameraHandoffArtifact,
) -> str:
    return payload_sha256(_handoff_payload_without_hash(handoff))


def write_exact_two_camera_handoff(
    path: Path, handoff: ExactTwoCameraHandoffArtifact
) -> str:
    try:
        return write_content_hashed_json(
            Path(path),
            _handoff_payload_without_hash(handoff),
            hash_field=HANDOFF_HASH_FIELD,
        )
    except ContentStoreError as exc:
        raise ExactTwoCameraAdmissionError(exc.code, str(exc)) from exc


def load_exact_two_camera_handoff(path: Path) -> ExactTwoCameraHandoffArtifact:
    try:
        payload = load_content_hashed_json(Path(path), hash_field=HANDOFF_HASH_FIELD)
    except ContentStoreError as exc:
        raise ExactTwoCameraAdmissionError(exc.code, str(exc)) from exc
    handoff = _handoff_from_payload(payload)
    validate_exact_two_camera_handoff(handoff)
    return handoff


def require_handoff_candidate_support(
    handoff: ExactTwoCameraHandoffArtifact,
    candidate_uid: str,
    required_support_class: str | None = None,
) -> ExactTwoCameraCandidateEvidence:
    validate_exact_two_camera_handoff(handoff)
    return require_admitted_candidate_support(
        handoff.admission_decision, candidate_uid, required_support_class
    )


def validate_live_registry_binding(
    handoff: ExactTwoCameraHandoffArtifact,
    registry: StandSurveyRegistry,
) -> None:
    validate_exact_two_camera_handoff(handoff)
    actual = stand_survey_registry_sha256(registry)
    if (
        registry.survey_id != handoff.survey_id
        or registry.planning_frame != handoff.planning_frame
        or registry.map_bundle_sha256 != handoff.map_bundle_sha256
        or actual != handoff.source_registry_sha256
    ):
        raise ExactTwoCameraAdmissionError(
            "live_registry_mismatch",
            "live stand registry no longer matches the sealed camera handoff",
        )


def validate_live_candidate_snapshot_binding(
    handoff: ExactTwoCameraHandoffArtifact,
    snapshot: CandidateSnapshot,
    *,
    candidate_snapshot_path: str | Path | None = None,
) -> None:
    validate_exact_two_camera_handoff(handoff)
    _validate_snapshot_against_admission(snapshot, handoff.admission_decision)
    if candidate_snapshot_sha256(snapshot) != handoff.candidate_snapshot_sha256:
        raise ExactTwoCameraAdmissionError(
            "live_snapshot_mismatch",
            "live candidate snapshot hash does not match camera handoff",
        )
    if snapshot.snapshot_id != handoff.candidate_snapshot_id:
        raise ExactTwoCameraAdmissionError(
            "live_snapshot_mismatch",
            "live candidate snapshot ID does not match camera handoff",
        )
    if candidate_snapshot_path is not None and path_string(
        candidate_snapshot_path, "candidate_snapshot_path"
    ) != handoff.candidate_snapshot_path:
        raise ExactTwoCameraAdmissionError(
            "live_snapshot_mismatch",
            "live candidate snapshot path does not match camera handoff",
        )


def load_bound_exact_two_candidate_snapshot(
    handoff: ExactTwoCameraHandoffArtifact,
    path: Path,
) -> CandidateSnapshot:
    try:
        snapshot = load_candidate_snapshot(
            Path(path), required_map_bundle_sha256=handoff.map_bundle_sha256
        )
    except CandidateSnapshotError as exc:
        raise ExactTwoCameraAdmissionError(exc.code, str(exc)) from exc
    validate_live_candidate_snapshot_binding(
        handoff, snapshot, candidate_snapshot_path=path
    )
    return snapshot


def _validate_snapshot_against_admission(
    snapshot: CandidateSnapshot,
    admission: ExactTwoCameraAdmissionDecision,
) -> None:
    validate_exact_two_camera_admission(admission)
    try:
        validate_candidate_snapshot(
            snapshot, required_map_bundle_sha256=admission.map_bundle_sha256
        )
    except CandidateSnapshotError as exc:
        raise ExactTwoCameraAdmissionError(exc.code, str(exc)) from exc
    if snapshot.planning_frame != admission.planning_frame:
        raise ExactTwoCameraAdmissionError(
            "provenance_mismatch", "candidate snapshot planning frame mismatch"
        )
    if snapshot.candidate_uids != admission.admitted_candidate_uids:
        raise ExactTwoCameraAdmissionError(
            "live_snapshot_mismatch",
            "candidate snapshot UIDs do not match admitted camera population",
        )
    for candidate in snapshot.candidates:
        evidence = require_admitted_candidate_support(
            admission, candidate.candidate_uid
        )
        if candidate.source.source_kind != evidence.source_kind:
            raise ExactTwoCameraAdmissionError(
                "support_class_mismatch",
                f"snapshot source kind does not match {candidate.candidate_uid!r}",
            )
        if candidate.source.source_artifact_sha256 != admission.source_registry_sha256:
            raise ExactTwoCameraAdmissionError(
                "provenance_mismatch",
                f"snapshot registry digest mismatch for {candidate.candidate_uid!r}",
            )


def _handoff_payload_without_hash(
    handoff: ExactTwoCameraHandoffArtifact,
) -> dict[str, object]:
    validate_exact_two_camera_handoff(handoff)
    return {
        "schema_version": handoff.schema_version,
        "handoff_id": handoff.handoff_id,
        "created_unix_sec": handoff.created_unix_sec,
        "survey_id": handoff.survey_id,
        "planning_frame": handoff.planning_frame,
        "map_bundle_sha256": handoff.map_bundle_sha256,
        "plan_sha256": handoff.plan_sha256,
        "progress_snapshot_sha256": handoff.progress_snapshot_sha256,
        "source_registry_sha256": handoff.source_registry_sha256,
        "terminal_checkpoint_path": handoff.terminal_checkpoint_path,
        "terminal_checkpoint_sha256": handoff.terminal_checkpoint_sha256,
        "lidar_admission_path": handoff.lidar_admission_path,
        "lidar_admission_sha256": handoff.lidar_admission_sha256,
        "lidar_checkpoint_sha256": handoff.lidar_checkpoint_sha256,
        "camera_admission_path": handoff.camera_admission_path,
        "camera_admission_sha256": handoff.camera_admission_sha256,
        "candidate_snapshot_path": handoff.candidate_snapshot_path,
        "candidate_snapshot_sha256": handoff.candidate_snapshot_sha256,
        "candidate_snapshot_id": handoff.candidate_snapshot_id,
        "camera_population_ready": handoff.camera_population_ready,
        "motion_authorized": handoff.motion_authorized,
        "admission_decision": handoff.admission_decision.to_evidence_dict(),
    }


def admission_from_payload(
    payload: Mapping[str, object],
) -> ExactTwoCameraAdmissionDecision:
    item = _mapping(payload, "camera admission")
    _require_fields(item, _ADMISSION_FIELDS, "camera admission")
    expected = item["expected_stand_count"]
    return ExactTwoCameraAdmissionDecision(
        schema_version=_integer(item["schema_version"], "schema_version"),
        survey_id=_string(item["survey_id"], "survey_id"),
        planning_frame=_string(item["planning_frame"], "planning_frame"),
        map_bundle_sha256=_string(item["map_bundle_sha256"], "map_bundle_sha256"),
        plan_sha256=_string(item["plan_sha256"], "plan_sha256"),
        progress_snapshot_sha256=_string(
            item["progress_snapshot_sha256"], "progress_snapshot_sha256"
        ),
        source_registry_sha256=_string(
            item["source_registry_sha256"], "source_registry_sha256"
        ),
        lidar_checkpoint_sha256=_string(
            item["lidar_checkpoint_sha256"], "lidar_checkpoint_sha256"
        ),
        ready=_bool(item["ready"], "ready"),
        reasons=_strings(item["reasons"], "reasons"),
        camera_population_ready=_bool(
            item["camera_population_ready"], "camera_population_ready"
        ),
        motion_authorized=_bool(item["motion_authorized"], "motion_authorized"),
        expected_stand_count=(
            None if expected is None else _integer(expected, "expected_stand_count")
        ),
        active_candidate_count=_integer(
            item["active_candidate_count"], "active_candidate_count"
        ),
        multi_view_candidate_uids=_strings(
            item["multi_view_candidate_uids"], "multi_view_candidate_uids"
        ),
        single_view_candidate_uids=_strings(
            item["single_view_candidate_uids"], "single_view_candidate_uids"
        ),
        blocked_candidate_uids=_strings(
            item["blocked_candidate_uids"], "blocked_candidate_uids"
        ),
        candidate_evidence=tuple(
            _evidence_from_payload(value, index)
            for index, value in enumerate(
                _list(item["candidate_evidence"], "candidate_evidence")
            )
        ),
    )


def _evidence_from_payload(
    value: object, index: int
) -> ExactTwoCameraCandidateEvidence:
    name = f"candidate_evidence[{index}]"
    item = _mapping(value, name)
    _require_fields(item, _EVIDENCE_FIELDS, name)
    support = _optional_string(item["support_class"], f"{name}.support_class")
    source = _optional_string(item["source_kind"], f"{name}.source_kind")
    return ExactTwoCameraCandidateEvidence(
        candidate_uid=_string(item["candidate_uid"], f"{name}.candidate_uid"),
        registry_status=_string(item["registry_status"], f"{name}.registry_status"),
        active_lidar=_bool(item["active_lidar"], f"{name}.active_lidar"),
        static_map_disposition=_string(
            item["static_map_disposition"],
            f"{name}.static_map_disposition",
        ),
        static_map_admitted=_bool(
            item["static_map_admitted"], f"{name}.static_map_admitted"
        ),
        static_map_population_retained=_bool(
            item["static_map_population_retained"],
            f"{name}.static_map_population_retained",
        ),
        basic_lidar_supported=_bool(
            item["basic_lidar_supported"], f"{name}.basic_lidar_supported"
        ),
        confidence=_number(item["confidence"], f"{name}.confidence"),
        minimum_confidence=_number(
            item["minimum_confidence"], f"{name}.minimum_confidence"
        ),
        confidence_supported=_bool(
            item["confidence_supported"], f"{name}.confidence_supported"
        ),
        hit_count=_integer(item["hit_count"], f"{name}.hit_count"),
        minimum_hit_count=_integer(
            item["minimum_hit_count"], f"{name}.minimum_hit_count"
        ),
        hit_count_supported=_bool(
            item["hit_count_supported"], f"{name}.hit_count_supported"
        ),
        viewpoint_ids=_strings(item["viewpoint_ids"], f"{name}.viewpoint_ids"),
        known_viewpoint_ids=_strings(
            item["known_viewpoint_ids"], f"{name}.known_viewpoint_ids"
        ),
        unknown_viewpoint_ids=_strings(
            item["unknown_viewpoint_ids"], f"{name}.unknown_viewpoint_ids"
        ),
        viewpoint_ids_distinct=_bool(
            item["viewpoint_ids_distinct"], f"{name}.viewpoint_ids_distinct"
        ),
        distinct_known_viewpoint_count=_integer(
            item["distinct_known_viewpoint_count"],
            f"{name}.distinct_known_viewpoint_count",
        ),
        support_class=support,
        source_kind=source,
        admissible=_bool(item["admissible"], f"{name}.admissible"),
        reasons=_strings(item["reasons"], f"{name}.reasons"),
    )


def _handoff_from_payload(
    payload: Mapping[str, object],
) -> ExactTwoCameraHandoffArtifact:
    item = _mapping(payload, "camera handoff")
    _require_fields(item, _HANDOFF_FIELDS, "camera handoff")
    return ExactTwoCameraHandoffArtifact(
        schema_version=_integer(item["schema_version"], "schema_version"),
        handoff_id=_string(item["handoff_id"], "handoff_id"),
        created_unix_sec=_number(item["created_unix_sec"], "created_unix_sec"),
        survey_id=_string(item["survey_id"], "survey_id"),
        planning_frame=_string(item["planning_frame"], "planning_frame"),
        map_bundle_sha256=_string(item["map_bundle_sha256"], "map_bundle_sha256"),
        plan_sha256=_string(item["plan_sha256"], "plan_sha256"),
        progress_snapshot_sha256=_string(
            item["progress_snapshot_sha256"], "progress_snapshot_sha256"
        ),
        source_registry_sha256=_string(
            item["source_registry_sha256"], "source_registry_sha256"
        ),
        terminal_checkpoint_path=_string(
            item["terminal_checkpoint_path"], "terminal_checkpoint_path"
        ),
        terminal_checkpoint_sha256=_string(
            item["terminal_checkpoint_sha256"], "terminal_checkpoint_sha256"
        ),
        lidar_admission_path=_string(
            item["lidar_admission_path"], "lidar_admission_path"
        ),
        lidar_admission_sha256=_string(
            item["lidar_admission_sha256"], "lidar_admission_sha256"
        ),
        lidar_checkpoint_sha256=_string(
            item["lidar_checkpoint_sha256"], "lidar_checkpoint_sha256"
        ),
        camera_admission_path=_string(
            item["camera_admission_path"], "camera_admission_path"
        ),
        camera_admission_sha256=_string(
            item["camera_admission_sha256"], "camera_admission_sha256"
        ),
        candidate_snapshot_path=_string(
            item["candidate_snapshot_path"], "candidate_snapshot_path"
        ),
        candidate_snapshot_sha256=_string(
            item["candidate_snapshot_sha256"], "candidate_snapshot_sha256"
        ),
        candidate_snapshot_id=_string(
            item["candidate_snapshot_id"], "candidate_snapshot_id"
        ),
        camera_population_ready=_bool(
            item["camera_population_ready"], "camera_population_ready"
        ),
        motion_authorized=_bool(item["motion_authorized"], "motion_authorized"),
        admission_decision=admission_from_payload(
            _mapping(item["admission_decision"], "admission_decision")
        ),
    )


def path_string(value: str | Path, name: str) -> str:
    parsed = value.as_posix() if isinstance(value, Path) else value
    if not isinstance(parsed, str) or not parsed or "\x00" in parsed or "\\" in parsed:
        raise ExactTwoCameraAdmissionError(
            "invalid_path", f"{name} is not a safe canonical path"
        )
    path = PurePosixPath(parsed)
    if ".." in path.parts or path.as_posix() != parsed:
        raise ExactTwoCameraAdmissionError(
            "invalid_path", f"{name} is not a canonical path"
        )
    return parsed


def _require_fields(
    payload: Mapping[str, object], expected: frozenset[str], name: str
) -> None:
    actual = frozenset(payload)
    if actual != expected:
        raise ExactTwoCameraAdmissionError(
            "artifact_corrupt",
            f"{name} fields mismatch; missing={sorted(expected - actual)} "
            f"unknown={sorted(actual - expected)}",
        )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ExactTwoCameraAdmissionError("artifact_corrupt", f"{name} must be an object")
    return value


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ExactTwoCameraAdmissionError("artifact_corrupt", f"{name} must be an array")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ExactTwoCameraAdmissionError("artifact_corrupt", f"{name} must be a string")
    return value


def _optional_string(value: object, name: str) -> str | None:
    return None if value is None else _string(value, name)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExactTwoCameraAdmissionError("artifact_corrupt", f"{name} must be a number")
    return float(value)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExactTwoCameraAdmissionError("artifact_corrupt", f"{name} must be an integer")
    return value


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ExactTwoCameraAdmissionError("artifact_corrupt", f"{name} must be a boolean")
    return value


def _strings(value: object, name: str) -> tuple[str, ...]:
    return tuple(
        _string(item, f"{name}[{index}]")
        for index, item in enumerate(_list(value, name))
    )


__all__ = [name for name in globals() if not name.startswith("_")]
