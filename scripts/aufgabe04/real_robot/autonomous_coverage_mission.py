"""ROS-free orchestration for the outer autonomous coverage transaction.

The injected leg callback owns the separate retry/reseal motion state machine.
This module starts observation only after that callback returns completed-leg
evidence, publishes checkpoints only after observation and fusion succeed, and
keeps candidate snapshots behind terminal admission and count checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
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
from scripts.aufgabe04.navigation.coverage_candidate_lifecycle import (
    ExactTwoLidarCheckpointDecision,
    evaluate_exact_two_lidar_checkpoint,
    exact_two_lidar_checkpoint_evidence,
)
from scripts.aufgabe04.navigation.coverage_candidate_reporting import (
    active_lidar_registry_count_fields,
    fused_registry_candidate_count_fields,
)
from scripts.aufgabe04.real_robot.autonomous_exact_two_completion import (
    COVERAGE_EXACT_TWO_CAMERA_READY,
    EXACT_TWO_CAMERA_VALIDATION_READY,
    CoverageExactTwoCameraAdmissionError,
    CoverageExactTwoCameraHandoffRequest,
    CoverageExactTwoCameraReady,
    ExactTwoCameraAdmissionDecision,
    ExactTwoCameraCompletionRequest,
    ExactTwoCameraHandoffArtifact,
    build_exact_two_camera_snapshot_effect,
    complete_exact_two_camera,
    create_exact_two_camera_handoff,
    evaluate_exact_two_camera_admission,
    exact_two_camera_admission_sha256,
    exact_two_camera_handoff_sha256,
    write_bound_exact_two_camera_admission,
    write_exact_two_camera_handoff,
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
    COVERAGE_LEG_CHECKPOINT_COMPLETE,
    COVERAGE_SURVEY_TERMINAL_CHECKPOINT,
    publish_coverage_checkpoint,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateSnapshot,
    candidate_snapshot_sha256,
    write_candidate_snapshot,
)


COVERAGE_COMPLETE = "coverage_complete"
COVERAGE_LIDAR_CHECKPOINT_COMPLETE = "coverage_lidar_checkpoint_complete"
RESUME_FROM_CHECKPOINT = "resume-next-coverage-leg"
CANDIDATE_SNAPSHOT_READY = "candidate_snapshot_ready_for_mode_dispatch"
LIDAR_CHECKPOINT_READY = "inspect_lidar_checkpoint_evidence"


class CoverageCompletionPolicy(str, Enum):
    """Select the terminal evidence gate without changing motion scope."""

    CAMERA_READY = "camera-ready"
    EXACT_TWO_LIDAR_CHECKPOINT = "exact-two-lidar-checkpoint"
    EXACT_TWO_CAMERA_VALIDATION = "exact-two-camera-validation"


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
    completion_policy: CoverageCompletionPolicy = (
        CoverageCompletionPolicy.CAMERA_READY
    )

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
        if not isinstance(self.completion_policy, CoverageCompletionPolicy):
            raise ValueError("coverage completion policy must be explicit")
        if self.completion_policy in {
            CoverageCompletionPolicy.EXACT_TWO_LIDAR_CHECKPOINT,
            CoverageCompletionPolicy.EXACT_TWO_CAMERA_VALIDATION,
        }:
            if self.plan.config.exact_inspection_point_count != 2:
                raise ValueError(
                    "exact-two completion policy requires an exact-two plan"
                )
            if self.plan.config.expected_stand_count != self.expected_stand_count:
                raise ValueError(
                    "exact-two completion count differs from the frozen plan"
                )
        if (
            self.completion_policy
            is CoverageCompletionPolicy.EXACT_TWO_LIDAR_CHECKPOINT
            and self.checkpoint_identity.run_mode
            not in {
                "execute-coverage-checkpoint",
                "resume-next-coverage-leg",
            }
        ):
            raise ValueError(
                "exact-two LiDAR completion is limited to coverage "
                "checkpoint execution or its one-leg resume"
            )
        if (
            self.completion_policy
            is CoverageCompletionPolicy.EXACT_TWO_CAMERA_VALIDATION
            and self.checkpoint_identity.run_mode != "execute-exact-two-camera"
        ):
            raise ValueError(
                "exact-two camera validation requires run mode "
                "execute-exact-two-camera"
            )


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
    next_viewpoint_id: str | None
    coverage_plan_path: Path
    coverage_progress_path: Path
    survey_summary_path: Path
    stand_registry_path: Path
    lidar_observer_summary_path: Path
    parent_checkpoint_path: Path | None
    checkpoint_status: str = COVERAGE_LEG_CHECKPOINT_COMPLETE


@dataclass(frozen=True)
class PublishedCoverageCheckpoint:
    manifest_path: Path
    manifest_sha256: str


@dataclass(frozen=True)
class CoverageNextLegPreparationRequest:
    """Evidence-bound request to prepare the already-checkpointed next leg."""

    leg_index: int
    recorded_viewpoint_id: str
    target_viewpoint_id: str
    lidar_observer_summary_path: Path
    checkpoint_manifest: Path
    checkpoint_manifest_sha256: str

    def __post_init__(self) -> None:
        if type(self.leg_index) is not int or self.leg_index < 0:
            raise ValueError("next coverage leg index must be a non-negative integer")
        if (
            not isinstance(self.recorded_viewpoint_id, str)
            or not self.recorded_viewpoint_id.strip()
        ):
            raise ValueError("recorded coverage viewpoint must be non-empty")
        if (
            not isinstance(self.target_viewpoint_id, str)
            or not self.target_viewpoint_id.strip()
        ):
            raise ValueError("next coverage target must be a non-empty string")
        for name in ("lidar_observer_summary_path", "checkpoint_manifest"):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"{name} must be a Path")
        _require_sha256(
            self.checkpoint_manifest_sha256,
            "checkpoint_manifest_sha256",
        )


@dataclass(frozen=True)
class PreparedCoverageLeg:
    """Route artifacts prepared for exactly one checkpointed continuation."""

    leg_index: int
    target_viewpoint_id: str
    source_route: Path
    source_diagnostics: Path

    def __post_init__(self) -> None:
        if type(self.leg_index) is not int or self.leg_index < 0:
            raise ValueError("prepared coverage leg index must be non-negative")
        if (
            not isinstance(self.target_viewpoint_id, str)
            or not self.target_viewpoint_id.strip()
        ):
            raise ValueError("prepared coverage target must be a non-empty string")
        for name in ("source_route", "source_diagnostics"):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"prepared coverage {name} must be a Path")


class CoverageContinuationPreparationError(RuntimeError):
    """Preparation failed after a durable checkpoint made the leg resumable."""

    def __init__(
        self,
        *,
        request: CoverageNextLegPreparationRequest,
        completed_coverage_legs: int,
        legs_completed_this_run: int,
        preparation_error: Exception,
    ) -> None:
        self.checkpoint_manifest = request.checkpoint_manifest
        self.checkpoint_manifest_sha256 = request.checkpoint_manifest_sha256
        self.completed_coverage_legs = completed_coverage_legs
        self.legs_completed_this_run = legs_completed_this_run
        self.recorded_viewpoint_id = request.recorded_viewpoint_id
        self.next_viewpoint_id = request.target_viewpoint_id
        self.lidar_observer_summary_path = request.lidar_observer_summary_path
        self.preparation_error = preparation_error
        super().__init__(
            "coverage continuation preparation failed after checkpoint "
            f"{request.checkpoint_manifest}: {preparation_error}"
        )

    def to_failure_fields(self) -> dict[str, object]:
        nested_fields: dict[str, object] = {}
        nested = getattr(self.preparation_error, "to_failure_fields", None)
        if callable(nested):
            try:
                value = nested()
            except Exception:
                value = None
            if isinstance(value, Mapping):
                nested_fields.update(value)
        nested_phase = nested_fields.get("failure_phase")
        if isinstance(nested_phase, str) and nested_phase.strip():
            nested_fields["preparation_failure_phase"] = nested_phase
        nested_fields.update(
            {
                "failure_phase": "coverage_continuation_preparation",
                "checkpoint_manifest": str(self.checkpoint_manifest),
                "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
                "completed_coverage_legs": self.completed_coverage_legs,
                "legs_completed_this_run": self.legs_completed_this_run,
                "recorded_viewpoint_id": self.recorded_viewpoint_id,
                "next_viewpoint_id": self.next_viewpoint_id,
                "lidar_observer_summary": str(self.lidar_observer_summary_path),
                "motion_published": True,
                "prior_leg_motion_published": True,
                "motion_authorized": False,
            }
        )
        return nested_fields


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
            result.update(
                fused_registry_candidate_count_fields(
                    dict(self.candidate_counts)
                )
            )
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
            "lidar_checkpoint_complete": False,
            "camera_validation_population_ready": False,
            "candidate_snapshot_ready": False,
            **self.coverage_status.to_summary_fields(),
        }


@dataclass(frozen=True)
class CoverageLidarCheckpointComplete:
    """Terminal exact-two LiDAR evidence; never a camera-motion handoff."""

    run_mode: str
    completed_coverage_legs: int
    legs_completed_this_run: int
    survey_root: Path
    checkpoint_manifest: Path
    checkpoint_manifest_sha256: str
    checkpoint_parent_manifest: Path | None
    lidar_checkpoint_admission_path: Path
    lidar_checkpoint_admission_sha256: str
    decision: ExactTwoLidarCheckpointDecision
    coverage_status: CoverageStatus
    status: str = field(default=COVERAGE_LIDAR_CHECKPOINT_COMPLETE, init=False)
    motion_published: bool = field(default=True, init=False)
    motion_authorized: bool = field(default=False, init=False)

    def to_mission_summary(self) -> dict[str, object]:
        population = self.decision.population
        return {
            "schema_version": 1,
            "status": self.status,
            "run_mode": self.run_mode,
            "motion_published": self.motion_published,
            "prior_leg_motion_published": self.motion_published,
            "motion_authorized": self.motion_authorized,
            "checkpoint_motion_authorized": False,
            "camera_approach_authorized": False,
            "completed_coverage_legs": self.completed_coverage_legs,
            "legs_completed_this_run": self.legs_completed_this_run,
            "next_viewpoint_id": None,
            "survey_root": str(self.survey_root),
            "checkpoint_manifest": str(self.checkpoint_manifest),
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "checkpoint_parent_manifest": (
                None
                if self.checkpoint_parent_manifest is None
                else str(self.checkpoint_parent_manifest)
            ),
            "lidar_checkpoint_admission": str(
                self.lidar_checkpoint_admission_path
            ),
            "lidar_checkpoint_admission_sha256": (
                self.lidar_checkpoint_admission_sha256
            ),
            "expected_stand_count": self.decision.expected_stand_count,
            **active_lidar_registry_count_fields(
                self.decision.active_lidar_candidate_count
            ),
            "lidar_static_map_admitted_candidate_uids": list(
                self.decision.admitted_lidar_candidate_uids
            ),
            "multi_view_supported_candidate_uids": list(
                population.multi_view_supported_candidate_uids
            ),
            "camera_validation_queue_candidate_uids": list(
                population.camera_queue_candidate_uids
            ),
            "camera_confirmed_candidate_uids": list(
                population.camera_confirmed_candidate_uids
            ),
            "candidate_snapshot": None,
            "lidar_checkpoint_complete": True,
            "camera_validation_population_ready": False,
            "candidate_snapshot_ready": False,
            **self.coverage_status.to_summary_fields(),
        }


class CoverageLidarCheckpointAdmissionError(RuntimeError):
    """LiDAR policy failed after a terminal survey checkpoint was published."""

    def __init__(
        self,
        *,
        checkpoint: PublishedCoverageCheckpoint,
        checkpoint_parent_manifest: Path | None,
        admission_path: Path,
        admission_sha256: str,
        decision: ExactTwoLidarCheckpointDecision,
        completed_coverage_legs: int,
        legs_completed_this_run: int,
    ) -> None:
        self.checkpoint = checkpoint
        self.checkpoint_parent_manifest = checkpoint_parent_manifest
        self.admission_path = admission_path
        self.admission_sha256 = admission_sha256
        self.decision = decision
        self.completed_coverage_legs = completed_coverage_legs
        self.legs_completed_this_run = legs_completed_this_run
        super().__init__(
            "exact-two LiDAR checkpoint admission rejected after terminal "
            "checkpoint: " + ", ".join(decision.reasons)
        )

    def to_failure_fields(self) -> dict[str, object]:
        return {
            "failure_phase": "exact_two_lidar_checkpoint_admission",
            "checkpoint_manifest": str(self.checkpoint.manifest_path),
            "checkpoint_manifest_sha256": self.checkpoint.manifest_sha256,
            "checkpoint_parent_manifest": (
                None
                if self.checkpoint_parent_manifest is None
                else str(self.checkpoint_parent_manifest)
            ),
            "lidar_checkpoint_admission": str(self.admission_path),
            "lidar_checkpoint_admission_sha256": self.admission_sha256,
            "lidar_checkpoint_admission_reasons": list(self.decision.reasons),
            "expected_stand_count": self.decision.expected_stand_count,
            **active_lidar_registry_count_fields(
                self.decision.active_lidar_candidate_count
            ),
            "completed_coverage_legs": self.completed_coverage_legs,
            "legs_completed_this_run": self.legs_completed_this_run,
            "next_viewpoint_id": None,
            "terminal_checkpoint_published": True,
            "motion_published": True,
            "prior_leg_motion_published": True,
            "motion_authorized": False,
            "camera_approach_authorized": False,
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
    terminal_checkpoint_manifest: Path | None = None
    terminal_checkpoint_manifest_sha256: str | None = None
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
            "terminal_checkpoint_manifest": (
                None
                if self.terminal_checkpoint_manifest is None
                else str(self.terminal_checkpoint_manifest)
            ),
            "terminal_checkpoint_manifest_sha256": (
                self.terminal_checkpoint_manifest_sha256
            ),
            "lidar_checkpoint_complete": True,
            "camera_validation_population_ready": True,
            "candidate_snapshot_ready": True,
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
        status=request.checkpoint_status,
    )
    return PublishedCoverageCheckpoint(published.manifest_path, published.manifest_sha256)


def _write_admission(path: Path, decision: CoverageCandidateAdmissionDecision) -> str:
    return write_content_hashed_json(
        path,
        coverage_candidate_admission_evidence(decision),
        hash_field="coverage_candidate_admission_sha256",
    )


def _write_lidar_checkpoint_admission(
    path: Path,
    decision: ExactTwoLidarCheckpointDecision,
    checkpoint: PublishedCoverageCheckpoint,
) -> str:
    return write_content_hashed_json(
        path,
        {
            "schema_version": 1,
            "admission_kind": "exact_two_lidar_terminal_checkpoint",
            "terminal_checkpoint_manifest": str(checkpoint.manifest_path),
            "terminal_checkpoint_manifest_sha256": checkpoint.manifest_sha256,
            "motion_authorized": False,
            "camera_approach_authorized": False,
            "decision": exact_two_lidar_checkpoint_evidence(decision),
        },
        hash_field="lidar_checkpoint_admission_sha256",
    )


def _prepare_next_leg_unconfigured(
    request: CoverageNextLegPreparationRequest,
) -> PreparedCoverageLeg:
    del request
    raise RuntimeError("next-leg preparation effect is not configured")


@dataclass(frozen=True)
class CoverageMissionEffects:
    """All environment-visible effects used by the outer transaction."""

    execute_completed_leg: Callable[[CoverageLegRequest], CompletedCoverageLeg]
    capture_lidar_epoch: Callable[[str, Path], Path]
    fuse_coverage_stop: Callable[[str, Path], Mapping[str, object]]
    build_snapshot: Callable[
        [StandSurveyRegistry, CoverageSurveyPlan, Path, str],
        CandidateSnapshot,
    ]
    prepare_next_leg: Callable[
        [CoverageNextLegPreparationRequest], PreparedCoverageLeg
    ] = _prepare_next_leg_unconfigured
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
    evaluate_lidar_checkpoint: Callable[
        [CoverageSurveyPlan, CoverageSurveyProgress, StandSurveyRegistry],
        ExactTwoLidarCheckpointDecision,
    ] = evaluate_exact_two_lidar_checkpoint
    write_lidar_checkpoint_admission: Callable[
        [
            Path,
            ExactTwoLidarCheckpointDecision,
            PublishedCoverageCheckpoint,
        ],
        str,
    ] = _write_lidar_checkpoint_admission
    evaluate_exact_two_camera_admission: Callable[
        [
            CoverageSurveyPlan,
            CoverageSurveyProgress,
            StandSurveyRegistry,
            ExactTwoLidarCheckpointDecision,
        ],
        ExactTwoCameraAdmissionDecision,
    ] = evaluate_exact_two_camera_admission
    write_exact_two_camera_admission: Callable[
        [
            Path,
            ExactTwoCameraAdmissionDecision,
            PublishedCoverageCheckpoint,
            Path,
            str,
        ],
        str,
    ] = write_bound_exact_two_camera_admission
    exact_two_camera_admission_sha256: Callable[
        [ExactTwoCameraAdmissionDecision], str
    ] = exact_two_camera_admission_sha256
    build_exact_two_camera_snapshot: Callable[
        [
            CoverageSurveyPlan,
            StandSurveyRegistry,
            ExactTwoCameraAdmissionDecision,
            str,
        ],
        CandidateSnapshot,
    ] = build_exact_two_camera_snapshot_effect
    create_exact_two_camera_handoff: Callable[
        [CoverageExactTwoCameraHandoffRequest],
        ExactTwoCameraHandoffArtifact,
    ] = create_exact_two_camera_handoff
    write_exact_two_camera_handoff: Callable[
        [Path, ExactTwoCameraHandoffArtifact], str
    ] = write_exact_two_camera_handoff
    exact_two_camera_handoff_sha256: Callable[
        [ExactTwoCameraHandoffArtifact], str
    ] = exact_two_camera_handoff_sha256
    write_snapshot: Callable[[Path, CandidateSnapshot], str] = write_candidate_snapshot
    snapshot_sha256: Callable[[CandidateSnapshot], str] = candidate_snapshot_sha256


def execute_coverage_mission(
    config: CoverageMissionConfig,
    effects: CoverageMissionEffects,
) -> (
    CoverageCheckpointComplete
    | CoverageLidarCheckpointComplete
    | CoverageExactTwoCameraReady
    | CoverageComplete
):
    """Run coverage observation/checkpoint/admission around an injected leg FSM."""

    leg_index = config.initial_leg_index
    legs_completed_this_run = 0
    latest_checkpoint = config.parent_checkpoint_path
    progress_path = config.survey_root / "coverage_progress.json"
    summary_path = config.survey_root / "survey_summary.json"
    registry_path = config.survey_root / "stand_registry.json"
    admission_path = config.checkpoint_identity.session_root / "coverage_candidate_admission.json"
    lidar_admission_path = (
        config.checkpoint_identity.session_root
        / "coverage_lidar_checkpoint_admission.json"
    )
    camera_admission_path = (
        config.checkpoint_identity.session_root
        / "coverage_exact_two_camera_admission.json"
    )
    camera_handoff_path = (
        config.checkpoint_identity.session_root
        / "coverage_exact_two_camera_handoff.json"
    )
    snapshot_path = config.checkpoint_identity.session_root / "candidate_snapshot.json"
    summary = _validated_summary(effects.read_summary(summary_path))
    prepared_leg: PreparedCoverageLeg | None = None
    terminal_checkpoint: PublishedCoverageCheckpoint | None = None
    terminal_checkpoint_parent: Path | None = None

    while (viewpoint_id := _next_viewpoint_id(summary)) is not None:
        legs_root = config.survey_root / "legs"
        if prepared_leg is None:
            source_route = legs_root / f"leg_{leg_index:03d}_route.csv"
            source_diagnostics = (
                legs_root / f"leg_{leg_index:03d}_diagnostics.json"
            )
        else:
            _validate_prepared_leg(
                prepared_leg,
                leg_index=leg_index,
                target_viewpoint_id=viewpoint_id,
            )
            source_route = prepared_leg.source_route
            source_diagnostics = prepared_leg.source_diagnostics
            prepared_leg = None
        completed = effects.execute_completed_leg(
            CoverageLegRequest(
                leg_index=leg_index,
                target_viewpoint_id=viewpoint_id,
                source_route=source_route,
                source_diagnostics=source_diagnostics,
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
        checkpoint_parent = latest_checkpoint
        checkpoint_status = (
            COVERAGE_LEG_CHECKPOINT_COMPLETE
            if next_viewpoint_id is not None
            else COVERAGE_SURVEY_TERMINAL_CHECKPOINT
        )
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
                checkpoint_status=checkpoint_status,
            )
        )
        if not isinstance(published, PublishedCoverageCheckpoint):
            raise RuntimeError("checkpoint callback returned invalid evidence")
        _require_sha256(
            published.manifest_sha256,
            "checkpoint_manifest_sha256",
        )
        latest_checkpoint = published.manifest_path
        if next_viewpoint_id is None:
            terminal_checkpoint = published
            terminal_checkpoint_parent = checkpoint_parent

        reached_run_limit = (
            config.coverage_leg_limit > 0
            and legs_completed_this_run >= config.coverage_leg_limit
        )
        if reached_run_limit and next_viewpoint_id is not None:
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
        if next_viewpoint_id is not None:
            preparation_request = CoverageNextLegPreparationRequest(
                leg_index=leg_index,
                recorded_viewpoint_id=viewpoint_id,
                target_viewpoint_id=next_viewpoint_id,
                lidar_observer_summary_path=observer_summary,
                checkpoint_manifest=published.manifest_path,
                checkpoint_manifest_sha256=published.manifest_sha256,
            )
            try:
                candidate = effects.prepare_next_leg(preparation_request)
                prepared_leg = _validate_prepared_leg(
                    candidate,
                    leg_index=preparation_request.leg_index,
                    target_viewpoint_id=preparation_request.target_viewpoint_id,
                )
            except Exception as exc:
                raise CoverageContinuationPreparationError(
                    request=preparation_request,
                    completed_coverage_legs=leg_index,
                    legs_completed_this_run=legs_completed_this_run,
                    preparation_error=exc,
                ) from exc

    progress = effects.load_progress(progress_path, config.plan)
    registry = effects.load_registry(registry_path, config.plan)
    if config.completion_policy in {
        CoverageCompletionPolicy.EXACT_TWO_LIDAR_CHECKPOINT,
        CoverageCompletionPolicy.EXACT_TWO_CAMERA_VALIDATION,
    }:
        return _finish_exact_two_completion(
            config=config,
            effects=effects,
            progress=progress,
            registry=registry,
            snapshot_path=snapshot_path,
            lidar_admission_path=lidar_admission_path,
            camera_admission_path=camera_admission_path,
            camera_handoff_path=camera_handoff_path,
            terminal_checkpoint=terminal_checkpoint,
            terminal_checkpoint_parent=terminal_checkpoint_parent,
            completed_coverage_legs=leg_index,
            legs_completed_this_run=legs_completed_this_run,
        )

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
        terminal_checkpoint_manifest=(
            None
            if terminal_checkpoint is None
            else terminal_checkpoint.manifest_path
        ),
        terminal_checkpoint_manifest_sha256=(
            None
            if terminal_checkpoint is None
            else terminal_checkpoint.manifest_sha256
        ),
    )


def _finish_exact_two_completion(
    *,
    config: CoverageMissionConfig,
    effects: CoverageMissionEffects,
    progress: CoverageSurveyProgress,
    registry: StandSurveyRegistry,
    snapshot_path: Path,
    lidar_admission_path: Path,
    camera_admission_path: Path,
    camera_handoff_path: Path,
    terminal_checkpoint: PublishedCoverageCheckpoint | None,
    terminal_checkpoint_parent: Path | None,
    completed_coverage_legs: int,
    legs_completed_this_run: int,
) -> CoverageLidarCheckpointComplete | CoverageExactTwoCameraReady:
    """Finish one exact-two policy in strict evidence-publication order."""

    if terminal_checkpoint is None:
        raise RuntimeError(
            "exact-two completion requires a terminal checkpoint published "
            "after the final leg"
        )
    lidar_decision = effects.evaluate_lidar_checkpoint(
        config.plan,
        progress,
        registry,
    )
    lidar_admission_hash = effects.write_lidar_checkpoint_admission(
        lidar_admission_path,
        lidar_decision,
        terminal_checkpoint,
    )
    _require_sha256(
        lidar_admission_hash,
        "lidar_checkpoint_admission_sha256",
    )
    if not lidar_decision.ready:
        raise CoverageLidarCheckpointAdmissionError(
            checkpoint=terminal_checkpoint,
            checkpoint_parent_manifest=terminal_checkpoint_parent,
            admission_path=lidar_admission_path,
            admission_sha256=lidar_admission_hash,
            decision=lidar_decision,
            completed_coverage_legs=completed_coverage_legs,
            legs_completed_this_run=legs_completed_this_run,
        )

    coverage_status = _lidar_checkpoint_status(lidar_decision, registry)
    if (
        config.completion_policy
        is CoverageCompletionPolicy.EXACT_TWO_LIDAR_CHECKPOINT
    ):
        return CoverageLidarCheckpointComplete(
            run_mode=config.checkpoint_identity.run_mode,
            completed_coverage_legs=completed_coverage_legs,
            legs_completed_this_run=legs_completed_this_run,
            survey_root=config.survey_root,
            checkpoint_manifest=terminal_checkpoint.manifest_path,
            checkpoint_manifest_sha256=terminal_checkpoint.manifest_sha256,
            checkpoint_parent_manifest=terminal_checkpoint_parent,
            lidar_checkpoint_admission_path=lidar_admission_path,
            lidar_checkpoint_admission_sha256=lidar_admission_hash,
            decision=lidar_decision,
            coverage_status=coverage_status,
        )

    return complete_exact_two_camera(
        ExactTwoCameraCompletionRequest(
            run_mode=config.checkpoint_identity.run_mode,
            session_id=config.checkpoint_identity.session_id,
            expected_stand_count=config.expected_stand_count,
            plan=config.plan,
            progress=progress,
            registry=registry,
            completed_coverage_legs=completed_coverage_legs,
            legs_completed_this_run=legs_completed_this_run,
            survey_root=config.survey_root,
            terminal_checkpoint=terminal_checkpoint,
            terminal_checkpoint_parent=terminal_checkpoint_parent,
            lidar_admission_path=lidar_admission_path,
            lidar_admission_sha256=lidar_admission_hash,
            lidar_decision=lidar_decision,
            camera_admission_path=camera_admission_path,
            candidate_snapshot_path=snapshot_path,
            camera_handoff_path=camera_handoff_path,
            coverage_status=_exact_two_camera_status(
                lidar_decision,
                registry,
            ),
        ),
        effects,
    )


def _validate_prepared_leg(
    value: object,
    *,
    leg_index: int,
    target_viewpoint_id: str,
) -> PreparedCoverageLeg:
    if not isinstance(value, PreparedCoverageLeg):
        raise RuntimeError("next-leg preparation returned invalid evidence")
    if value.leg_index != leg_index:
        raise RuntimeError(
            "prepared coverage leg index does not match the continuation request"
        )
    if value.target_viewpoint_id != target_viewpoint_id:
        raise RuntimeError(
            "prepared coverage target does not match the continuation request"
        )
    if not isinstance(value.source_route, Path):
        raise RuntimeError("prepared coverage route must be a Path")
    if not isinstance(value.source_diagnostics, Path):
        raise RuntimeError("prepared coverage diagnostics must be a Path")
    return value


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
    return CoverageStatus(
        coverage_complete=admission.coverage_threshold_met,
        visited_coverage_ratio=admission.visited_coverage_ratio,
        visited_viewpoint_count=len(admission.visited_viewpoint_ids),
        total_viewpoint_count=len(admission.planned_viewpoint_ids),
        candidate_counts=_registry_candidate_counts(registry),
        next_required_action=CANDIDATE_SNAPSHOT_READY,
    )


def _lidar_checkpoint_status(
    decision: ExactTwoLidarCheckpointDecision,
    registry: StandSurveyRegistry,
) -> CoverageStatus:
    return CoverageStatus(
        coverage_complete=(
            decision.all_planned_viewpoints_visited
            and decision.coverage_threshold_met
        ),
        visited_coverage_ratio=decision.visited_coverage_ratio,
        visited_viewpoint_count=len(decision.visited_viewpoint_ids),
        total_viewpoint_count=len(decision.planned_viewpoint_ids),
        candidate_counts=_registry_candidate_counts(registry),
        next_required_action=LIDAR_CHECKPOINT_READY,
    )


def _exact_two_camera_status(
    decision: ExactTwoLidarCheckpointDecision,
    registry: StandSurveyRegistry,
) -> CoverageStatus:
    return CoverageStatus(
        coverage_complete=(
            decision.all_planned_viewpoints_visited
            and decision.coverage_threshold_met
        ),
        visited_coverage_ratio=decision.visited_coverage_ratio,
        visited_viewpoint_count=len(decision.visited_viewpoint_ids),
        total_viewpoint_count=len(decision.planned_viewpoint_ids),
        candidate_counts=_registry_candidate_counts(registry),
        next_required_action=EXACT_TWO_CAMERA_VALIDATION_READY,
    )


def _registry_candidate_counts(
    registry: StandSurveyRegistry,
) -> tuple[tuple[str, int], ...]:
    statuses = (
        STATUS_CONFIRMED,
        STATUS_PENDING_CAMERA,
        STATUS_PROVISIONAL,
        STATUS_REJECTED,
    )
    return tuple(
        (status, sum(item.status == status for item in registry.candidates))
        for status in statuses
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
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
