"""ROS-free orchestration for the outer autonomous coverage transaction.

The injected leg callback owns the separate retry/reseal motion state machine.
This module starts observation only after that callback returns completed-leg
evidence, publishes checkpoints only after observation and fusion succeed, and
keeps candidate snapshots behind terminal admission and count checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Callable, Mapping

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.navigation.coverage_candidate_admission import (
    CoverageCandidateAdmissionDecision,
    coverage_candidate_admission_evidence,
    evaluate_coverage_candidate_admission,
)
from scripts.aufgabe04.navigation.stand_coverage_survey import (
    STATUS_CONFIRMED,
    STATUS_PENDING_CAMERA,
    STATUS_PROVISIONAL,
    STATUS_REJECTED,
    CoverageSurveyPlan,
    CoverageSurveyProgress,
    StandSurveyRegistry,
    load_stand_survey_registry,
    load_survey_progress,
)
from scripts.aufgabe04.real_robot.autonomous_session_manifest import (
    publish_coverage_checkpoint,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateSnapshot,
    candidate_snapshot_sha256,
    write_candidate_snapshot,
)


COVERAGE_LEG_CHECKPOINT_COMPLETE = "coverage_leg_checkpoint_complete"
COVERAGE_COMPLETE = "coverage_complete"
RESUME_FROM_CHECKPOINT = "resume-next-coverage-leg"
CANDIDATE_SNAPSHOT_READY = "candidate_snapshot_ready_for_mode_dispatch"


@dataclass(frozen=True)
class CoverageCheckpointIdentity:
    """Hardware/config hashes computed once before the coverage loop."""

    session_root: Path
    session_id: str
    run_mode: str
    robot_id: str
    robot_profile_sha256: str
    calibration_profile_sha256: str
    physical_site_sha256: str
    map_bundle_sha256: str
    config_sha256: str

    def __post_init__(self) -> None:
        for name in ("session_id", "run_mode", "robot_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        for name in (
            "robot_profile_sha256",
            "calibration_profile_sha256",
            "physical_site_sha256",
            "map_bundle_sha256",
            "config_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name)


@dataclass(frozen=True)
class CoverageMissionConfig:
    survey_root: Path
    plan: CoverageSurveyPlan
    coverage_plan_path: Path
    checkpoint_identity: CoverageCheckpointIdentity
    expected_stand_count: int
    initial_leg_index: int = 0
    coverage_leg_limit: int = 0
    parent_checkpoint_path: Path | None = None

    def __post_init__(self) -> None:
        counts = (
            self.expected_stand_count,
            self.initial_leg_index,
            self.coverage_leg_limit,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("coverage counts and limits must be non-negative integers")
        if self.plan.map_bundle_sha256 != self.checkpoint_identity.map_bundle_sha256:
            raise ValueError("coverage plan and checkpoint map bundle differ")

@dataclass(frozen=True)
class CoverageLegRequest:
    leg_index: int
    target_viewpoint_id: str
    source_route: Path
    source_diagnostics: Path


@dataclass(frozen=True)
class CompletedCoverageLeg:
    odom_execution_certificate_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.odom_execution_certificate_path, Path):
            raise ValueError(
                "completed coverage leg requires an odom execution certificate"
            )


@dataclass(frozen=True)
class CoverageCheckpointRequest:
    identity: CoverageCheckpointIdentity
    completed_coverage_legs: int
    next_viewpoint_id: str
    coverage_plan_path: Path
    coverage_progress_path: Path
    survey_summary_path: Path
    stand_registry_path: Path
    lidar_observer_summary_path: Path
    parent_checkpoint_path: Path | None


@dataclass(frozen=True)
class PublishedCoverageCheckpoint:
    manifest_path: Path
    manifest_sha256: str


@dataclass(frozen=True)
class CoverageStatus:
    coverage_complete: bool | None
    visited_coverage_ratio: float | None
    visited_viewpoint_count: int | None
    total_viewpoint_count: int | None
    candidate_counts: tuple[tuple[str, int], ...]
    next_required_action: str

    def to_summary_fields(self) -> dict[str, object]:
        result: dict[str, object] = {
            "next_required_action": self.next_required_action,
        }
        optional = {
            "coverage_complete": self.coverage_complete,
            "visited_coverage_ratio": self.visited_coverage_ratio,
            "visited_viewpoint_count": self.visited_viewpoint_count,
            "total_viewpoint_count": self.total_viewpoint_count,
        }
        result.update({key: value for key, value in optional.items() if value is not None})
        if self.candidate_counts:
            result["candidate_counts"] = dict(self.candidate_counts)
            result["candidate_count"] = sum(value for _, value in self.candidate_counts)
        return result


@dataclass(frozen=True)
class CoverageCheckpointComplete:
    run_mode: str
    completed_coverage_legs: int
    legs_completed_this_run: int
    next_viewpoint_id: str
    survey_root: Path
    checkpoint_manifest: Path
    checkpoint_manifest_sha256: str
    parent_checkpoint_path: Path | None
    coverage_status: CoverageStatus
    status: str = field(default=COVERAGE_LEG_CHECKPOINT_COMPLETE, init=False)
    motion_published: bool = field(default=True, init=False)
    motion_authorized: bool = field(default=False, init=False)

    def to_mission_summary(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "status": self.status,
            "run_mode": self.run_mode,
            "motion_published": self.motion_published,
            "prior_leg_motion_published": self.motion_published,
            "motion_authorized": self.motion_authorized,
            "checkpoint_motion_authorized": False,
            "completed_coverage_legs": self.completed_coverage_legs,
            "legs_completed_this_run": self.legs_completed_this_run,
            "next_viewpoint_id": self.next_viewpoint_id,
            "survey_root": str(self.survey_root),
            "checkpoint_manifest": str(self.checkpoint_manifest),
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "checkpoint_parent_manifest": (
                None if self.parent_checkpoint_path is None else str(self.parent_checkpoint_path)
            ),
            **self.coverage_status.to_summary_fields(),
        }


@dataclass(frozen=True)
class CoverageComplete:
    run_mode: str
    completed_coverage_legs: int
    legs_completed_this_run: int
    survey_root: Path
    candidate_snapshot_path: Path
    candidate_snapshot_sha256: str
    candidate_snapshot: CandidateSnapshot
    coverage_candidate_admission_path: Path
    coverage_candidate_admission_sha256: str
    resume_parent_checkpoint_path: Path | None
    coverage_status: CoverageStatus
    status: str = field(default=COVERAGE_COMPLETE, init=False)
    motion_published: bool = field(default=True, init=False)
    motion_authorized: bool = field(default=False, init=False)

    @property
    def stand_count(self) -> int:
        return len(self.candidate_snapshot.candidates)

    def to_mission_summary(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "status": self.status,
            "run_mode": self.run_mode,
            "motion_published": self.motion_published,
            "prior_leg_motion_published": self.motion_published,
            "motion_authorized": self.motion_authorized,
            "completed_coverage_legs": self.completed_coverage_legs,
            "legs_completed_this_run": self.legs_completed_this_run,
            "stand_count": self.stand_count,
            "candidate_snapshot": str(self.candidate_snapshot_path),
            "candidate_snapshot_sha256": self.candidate_snapshot_sha256,
            "survey_root": str(self.survey_root),
            "coverage_candidate_admission": str(self.coverage_candidate_admission_path),
            "coverage_candidate_admission_sha256": (
                self.coverage_candidate_admission_sha256
            ),
            "resume_parent_checkpoint": (
                None
                if self.resume_parent_checkpoint_path is None
                else str(self.resume_parent_checkpoint_path)
            ),
            **self.coverage_status.to_summary_fields(),
        }


def _read_summary(path: Path) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("survey summary must be a JSON object")
    return payload


def _publish_checkpoint(request: CoverageCheckpointRequest) -> PublishedCoverageCheckpoint:
    identity = request.identity
    published = publish_coverage_checkpoint(
        session_root=identity.session_root,
        session_id=identity.session_id,
        run_mode=identity.run_mode,
        robot_id=identity.robot_id,
        robot_profile_sha256=identity.robot_profile_sha256,
        calibration_profile_sha256=identity.calibration_profile_sha256,
        physical_site_sha256=identity.physical_site_sha256,
        map_bundle_sha256=identity.map_bundle_sha256,
        config_sha256=identity.config_sha256,
        completed_coverage_legs=request.completed_coverage_legs,
        next_viewpoint_id=request.next_viewpoint_id,
        coverage_plan_path=request.coverage_plan_path,
        coverage_progress_path=request.coverage_progress_path,
        survey_summary_path=request.survey_summary_path,
        stand_registry_path=request.stand_registry_path,
        lidar_observer_summary_path=request.lidar_observer_summary_path,
        parent_checkpoint_path=request.parent_checkpoint_path,
    )
    return PublishedCoverageCheckpoint(published.manifest_path, published.manifest_sha256)


def _write_admission(path: Path, decision: CoverageCandidateAdmissionDecision) -> str:
    return write_content_hashed_json(
        path,
        coverage_candidate_admission_evidence(decision),
        hash_field="coverage_candidate_admission_sha256",
    )


@dataclass(frozen=True)
class CoverageMissionEffects:
    """All environment-visible effects used by the outer transaction."""

    execute_completed_leg: Callable[[CoverageLegRequest], CompletedCoverageLeg]
    capture_lidar_epoch: Callable[[str, Path], Path]
    fuse_coverage_stop: Callable[[str, Path], Mapping[str, object]]
    build_snapshot: Callable[
        [StandSurveyRegistry, CoverageSurveyPlan, Path, str], CandidateSnapshot
    ]
    read_summary: Callable[[Path], Mapping[str, object]] = _read_summary
    publish_checkpoint: Callable[
        [CoverageCheckpointRequest], PublishedCoverageCheckpoint
    ] = _publish_checkpoint
    load_progress: Callable[
        [Path, CoverageSurveyPlan], CoverageSurveyProgress
    ] = load_survey_progress
    load_registry: Callable[
        [Path, CoverageSurveyPlan], StandSurveyRegistry
    ] = load_stand_survey_registry
    evaluate_admission: Callable[
        [CoverageSurveyPlan, CoverageSurveyProgress, StandSurveyRegistry],
        CoverageCandidateAdmissionDecision,
    ] = evaluate_coverage_candidate_admission
    write_admission: Callable[
        [Path, CoverageCandidateAdmissionDecision], str
    ] = _write_admission
    write_snapshot: Callable[[Path, CandidateSnapshot], str] = write_candidate_snapshot
    snapshot_sha256: Callable[[CandidateSnapshot], str] = candidate_snapshot_sha256


def execute_coverage_mission(
    config: CoverageMissionConfig,
    effects: CoverageMissionEffects,
) -> CoverageCheckpointComplete | CoverageComplete:
    """Run coverage observation/checkpoint/admission around an injected leg FSM."""

    leg_index = config.initial_leg_index
    legs_completed_this_run = 0
    latest_checkpoint = config.parent_checkpoint_path
    progress_path = config.survey_root / "coverage_progress.json"
    summary_path = config.survey_root / "survey_summary.json"
    registry_path = config.survey_root / "stand_registry.json"
    admission_path = config.checkpoint_identity.session_root / "coverage_candidate_admission.json"
    snapshot_path = config.checkpoint_identity.session_root / "candidate_snapshot.json"
    summary = _validated_summary(effects.read_summary(summary_path))

    while (viewpoint_id := _next_viewpoint_id(summary)) is not None:
        legs_root = config.survey_root / "legs"
        completed = effects.execute_completed_leg(
            CoverageLegRequest(
                leg_index=leg_index,
                target_viewpoint_id=viewpoint_id,
                source_route=legs_root / f"leg_{leg_index:03d}_route.csv",
                source_diagnostics=legs_root / f"leg_{leg_index:03d}_diagnostics.json",
            )
        )
        if not isinstance(completed, CompletedCoverageLeg):
            raise RuntimeError("leg callback did not return completed-leg evidence")

        observer_summary = Path(
            effects.capture_lidar_epoch(
                viewpoint_id,
                completed.odom_execution_certificate_path,
            )
        )
        summary = _validated_summary(
            effects.fuse_coverage_stop(
                viewpoint_id,
                observer_summary,
            )
        )
        if summary.get("recorded_viewpoint_id", viewpoint_id) != viewpoint_id:
            raise RuntimeError("fused summary recorded a different viewpoint")
        next_viewpoint_id = _next_viewpoint_id(summary)
        if next_viewpoint_id == viewpoint_id:
            raise RuntimeError("coverage cursor did not advance after fusion")

        leg_index += 1
        legs_completed_this_run += 1
        published = None
        checkpoint_parent = latest_checkpoint
        if next_viewpoint_id is not None:
            published = effects.publish_checkpoint(
                CoverageCheckpointRequest(
                    identity=config.checkpoint_identity,
                    completed_coverage_legs=leg_index,
                    next_viewpoint_id=next_viewpoint_id,
                    coverage_plan_path=config.coverage_plan_path,
                    coverage_progress_path=progress_path,
                    survey_summary_path=summary_path,
                    stand_registry_path=registry_path,
                    lidar_observer_summary_path=observer_summary,
                    parent_checkpoint_path=latest_checkpoint,
                )
            )
            if not isinstance(published, PublishedCoverageCheckpoint):
                raise RuntimeError("checkpoint callback returned invalid evidence")
            _require_sha256(
                published.manifest_sha256,
                "checkpoint_manifest_sha256",
            )
            latest_checkpoint = published.manifest_path

        reached_run_limit = (
            config.coverage_leg_limit > 0
            and legs_completed_this_run >= config.coverage_leg_limit
        )
        if reached_run_limit and next_viewpoint_id is not None:
            assert published is not None
            return CoverageCheckpointComplete(
                run_mode=config.checkpoint_identity.run_mode,
                completed_coverage_legs=leg_index,
                legs_completed_this_run=legs_completed_this_run,
                next_viewpoint_id=next_viewpoint_id,
                survey_root=config.survey_root,
                checkpoint_manifest=published.manifest_path,
                checkpoint_manifest_sha256=published.manifest_sha256,
                parent_checkpoint_path=checkpoint_parent,
                coverage_status=_status(summary, RESUME_FROM_CHECKPOINT),
            )
        # A final planned leg falls through to admission even when it reaches
        # the per-invocation limit; a checkpoint must never bypass that gate.

    progress = effects.load_progress(progress_path, config.plan)
    registry = effects.load_registry(registry_path, config.plan)
    admission = effects.evaluate_admission(config.plan, progress, registry)
    admission_hash = effects.write_admission(admission_path, admission)
    _require_sha256(admission_hash, "coverage_candidate_admission_sha256")
    if not admission.ready:
        raise RuntimeError(
            "coverage candidate admission rejected: " + ", ".join(admission.reasons)
        )

    pending = tuple(
        candidate
        for candidate in registry.candidates
        if candidate.status == STATUS_PENDING_CAMERA
    )
    if len(pending) != config.expected_stand_count:
        raise RuntimeError(
            "center-corridor survey did not resolve the expected stand count: "
            f"pending_camera={len(pending)} expected={config.expected_stand_count}"
        )

    snapshot = effects.build_snapshot(
        registry,
        config.plan,
        registry_path,
        f"{config.checkpoint_identity.session_id}_candidates",
    )
    written_hash = effects.write_snapshot(snapshot_path, snapshot)
    snapshot_hash = effects.snapshot_sha256(snapshot)
    _require_sha256(written_hash, "written_candidate_snapshot_sha256")
    _require_sha256(snapshot_hash, "candidate_snapshot_sha256")
    if written_hash != snapshot_hash:
        raise RuntimeError("candidate snapshot persistence hash mismatch")

    return CoverageComplete(
        run_mode=config.checkpoint_identity.run_mode,
        completed_coverage_legs=leg_index,
        legs_completed_this_run=legs_completed_this_run,
        survey_root=config.survey_root,
        candidate_snapshot_path=snapshot_path,
        candidate_snapshot_sha256=snapshot_hash,
        candidate_snapshot=snapshot,
        coverage_candidate_admission_path=admission_path,
        coverage_candidate_admission_sha256=admission_hash,
        resume_parent_checkpoint_path=config.parent_checkpoint_path,
        coverage_status=_admitted_status(admission, registry),
    )


def _validated_summary(value: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or "next_viewpoint_id" not in value:
        raise RuntimeError("survey summary has no next_viewpoint_id cursor")
    _next_viewpoint_id(value)
    _status(value, RESUME_FROM_CHECKPOINT)
    return value


def _next_viewpoint_id(summary: Mapping[str, object]) -> str | None:
    value = summary["next_viewpoint_id"]
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise RuntimeError("next_viewpoint_id must be null or a non-empty string")
    return value


def _status(summary: Mapping[str, object], default_action: str) -> CoverageStatus:
    complete = summary.get("coverage_complete")
    if complete is not None and type(complete) is not bool:
        raise RuntimeError("coverage_complete must be boolean")
    ratio = summary.get("visited_coverage_ratio")
    if ratio is not None:
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise RuntimeError("visited_coverage_ratio must be numeric")
        ratio = float(ratio)
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise RuntimeError("visited_coverage_ratio must be finite and in [0, 1]")
    action = summary.get("next_required_action", default_action)
    if not isinstance(action, str) or not action.strip():
        raise RuntimeError("next_required_action must be a non-empty string")
    return CoverageStatus(
        coverage_complete=complete,
        visited_coverage_ratio=ratio,
        visited_viewpoint_count=_optional_nonnegative_int(
            summary,
            "visited_viewpoint_count",
        ),
        total_viewpoint_count=_optional_nonnegative_int(
            summary,
            "total_viewpoint_count",
        ),
        candidate_counts=_candidate_counts(summary.get("candidate_counts")),
        next_required_action=action,
    )


def _admitted_status(
    admission: CoverageCandidateAdmissionDecision,
    registry: StandSurveyRegistry,
) -> CoverageStatus:
    statuses = (
        STATUS_CONFIRMED,
        STATUS_PENDING_CAMERA,
        STATUS_PROVISIONAL,
        STATUS_REJECTED,
    )
    counts = tuple(
        (status, sum(item.status == status for item in registry.candidates))
        for status in statuses
    )
    return CoverageStatus(
        coverage_complete=admission.coverage_threshold_met,
        visited_coverage_ratio=admission.visited_coverage_ratio,
        visited_viewpoint_count=len(admission.visited_viewpoint_ids),
        total_viewpoint_count=len(admission.planned_viewpoint_ids),
        candidate_counts=counts,
        next_required_action=CANDIDATE_SNAPSHOT_READY,
    )


def _optional_nonnegative_int(
    summary: Mapping[str, object],
    name: str,
) -> int | None:
    value = summary.get(name)
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise RuntimeError(f"{name} must be a non-negative integer")
    return value

def _candidate_counts(value: object) -> tuple[tuple[str, int], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise RuntimeError("candidate_counts must be a mapping")
    result = tuple(sorted(value.items()))
    if any(
        not isinstance(name, str) or type(count) is not int or count < 0
        for name, count in result
    ):
        raise RuntimeError("candidate_counts must map names to non-negative integers")
    return result


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
