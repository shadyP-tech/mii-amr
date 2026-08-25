"""Exact-two LiDAR-to-camera completion without coverage-runner coupling.

The outer coverage transaction owns motion, terminal checkpoint publication,
and LiDAR admission.  This module starts only after that admission succeeded;
it publishes the camera population, candidate snapshot, and immutable handoff
in that order.  Its request and effects are structural so this module never
imports the parent coverage mission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol

from scripts.aufgabe04.navigation.coverage.coverage_candidate_lifecycle import (
    ExactTwoLidarCheckpointDecision,
)
from scripts.aufgabe04.navigation.approach.exact_two_camera_admission import (
    ExactTwoCameraAdmissionDecision,
    ExactTwoCameraHandoffArtifact,
    build_exact_two_camera_candidate_snapshot,
    evaluate_exact_two_camera_admission,
    exact_two_camera_admission_sha256,
    exact_two_camera_handoff_sha256,
    new_exact_two_camera_handoff,
    write_exact_two_camera_admission,
    write_exact_two_camera_handoff,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    CoverageSurveyPlan,
    CoverageSurveyProgress,
    StandSurveyRegistry,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateSnapshot,
    candidate_snapshot_sha256,
)


COVERAGE_EXACT_TWO_CAMERA_READY = "coverage_exact_two_camera_ready"
EXACT_TWO_CAMERA_VALIDATION_READY = "begin_exact_two_camera_validation"


class PublishedCoverageCheckpointLike(Protocol):
    """Checkpoint evidence required by camera completion."""

    manifest_path: Path
    manifest_sha256: str


class CoverageStatusLike(Protocol):
    """Summary projection supplied by the coverage transaction."""

    def to_summary_fields(self) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class CoverageExactTwoCameraHandoffRequest:
    """All immutable evidence needed to construct the camera handoff."""

    plan: CoverageSurveyPlan
    terminal_checkpoint: PublishedCoverageCheckpointLike
    lidar_admission_path: Path
    lidar_admission_sha256: str
    camera_admission_path: Path
    camera_admission_sha256: str
    camera_admission: ExactTwoCameraAdmissionDecision
    candidate_snapshot_path: Path
    candidate_snapshot_sha256: str
    candidate_snapshot: CandidateSnapshot


@dataclass(frozen=True)
class CoverageExactTwoCameraReady:
    """Evidence-bound exact-two handoff into per-candidate camera validation.

    This result exposes a camera-validation population but never authorizes
    motion itself. Candidate motion still requires the existing one-use
    mission-leg permit chain in the parent runner.
    """

    run_mode: str
    completed_coverage_legs: int
    legs_completed_this_run: int
    survey_root: Path
    checkpoint_manifest: Path
    checkpoint_manifest_sha256: str
    checkpoint_parent_manifest: Path | None
    lidar_checkpoint_admission_path: Path
    lidar_checkpoint_admission_sha256: str
    lidar_decision: ExactTwoLidarCheckpointDecision
    camera_validation_admission_path: Path
    camera_validation_admission_sha256: str
    camera_validation_decision: ExactTwoCameraAdmissionDecision
    candidate_snapshot_path: Path
    candidate_snapshot_sha256: str
    candidate_snapshot: CandidateSnapshot
    camera_handoff_path: Path
    camera_handoff_sha256: str
    camera_handoff: ExactTwoCameraHandoffArtifact
    coverage_status: CoverageStatusLike
    status: str = field(default=COVERAGE_EXACT_TWO_CAMERA_READY, init=False)
    motion_published: bool = field(default=True, init=False)
    motion_authorized: bool = field(default=False, init=False)

    @property
    def stand_count(self) -> int:
        return len(self.candidate_snapshot.candidates)

    def to_mission_summary(self) -> dict[str, object]:
        decision = self.camera_validation_decision
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
            "camera_validation_admission": str(
                self.camera_validation_admission_path
            ),
            "camera_validation_admission_sha256": (
                self.camera_validation_admission_sha256
            ),
            "camera_handoff": str(self.camera_handoff_path),
            "camera_handoff_sha256": self.camera_handoff_sha256,
            "candidate_snapshot": str(self.candidate_snapshot_path),
            "candidate_snapshot_sha256": self.candidate_snapshot_sha256,
            "stand_count": self.stand_count,
            "expected_stand_count": decision.expected_stand_count,
            "lidar_static_map_admitted_candidate_count": len(
                decision.lidar_static_map_admitted_candidate_uids
            ),
            "lidar_boundary_provisional_candidate_count": len(
                decision.lidar_boundary_provisional_candidate_uids
            ),
            "lidar_population_retained_candidate_count": len(
                decision.lidar_population_retained_candidate_uids
            ),
            "lidar_static_map_admitted_candidate_uids": list(
                decision.lidar_static_map_admitted_candidate_uids
            ),
            "lidar_boundary_provisional_candidate_uids": list(
                decision.lidar_boundary_provisional_candidate_uids
            ),
            "lidar_population_retained_candidate_uids": list(
                decision.lidar_population_retained_candidate_uids
            ),
            "camera_validation_candidate_uids": list(
                decision.admitted_candidate_uids
            ),
            "multi_view_candidate_uids": list(
                decision.multi_view_candidate_uids
            ),
            "single_view_requires_camera_validation_candidate_uids": list(
                decision.single_view_candidate_uids
            ),
            "source_registry_sha256": decision.source_registry_sha256,
            "lidar_checkpoint_complete": True,
            "camera_validation_population_ready": True,
            "candidate_snapshot_ready": True,
            **self.coverage_status.to_summary_fields(),
        }


class CoverageExactTwoCameraAdmissionError(RuntimeError):
    """Camera population admission failed after durable LiDAR evidence."""

    def __init__(
        self,
        *,
        checkpoint: PublishedCoverageCheckpointLike,
        checkpoint_parent_manifest: Path | None,
        lidar_admission_path: Path,
        lidar_admission_sha256: str,
        camera_admission_path: Path,
        camera_admission_sha256: str,
        decision: ExactTwoCameraAdmissionDecision,
        completed_coverage_legs: int,
        legs_completed_this_run: int,
    ) -> None:
        self.checkpoint = checkpoint
        self.checkpoint_parent_manifest = checkpoint_parent_manifest
        self.lidar_admission_path = lidar_admission_path
        self.lidar_admission_sha256 = lidar_admission_sha256
        self.camera_admission_path = camera_admission_path
        self.camera_admission_sha256 = camera_admission_sha256
        self.decision = decision
        self.completed_coverage_legs = completed_coverage_legs
        self.legs_completed_this_run = legs_completed_this_run
        super().__init__(
            "exact-two camera validation admission rejected after terminal "
            "LiDAR checkpoint: " + ", ".join(decision.reasons)
        )

    def to_failure_fields(self) -> dict[str, object]:
        return {
            "failure_phase": "exact_two_camera_validation_admission",
            "checkpoint_manifest": str(self.checkpoint.manifest_path),
            "checkpoint_manifest_sha256": self.checkpoint.manifest_sha256,
            "checkpoint_parent_manifest": (
                None
                if self.checkpoint_parent_manifest is None
                else str(self.checkpoint_parent_manifest)
            ),
            "lidar_checkpoint_admission": str(self.lidar_admission_path),
            "lidar_checkpoint_admission_sha256": self.lidar_admission_sha256,
            "camera_validation_admission": str(self.camera_admission_path),
            "camera_validation_admission_sha256": self.camera_admission_sha256,
            "camera_validation_admission_reasons": list(self.decision.reasons),
            "expected_stand_count": self.decision.expected_stand_count,
            "camera_validation_candidate_count": len(
                self.decision.admitted_candidate_uids
            ),
            "active_lidar_candidate_count": self.decision.active_candidate_count,
            "lidar_static_map_admitted_candidate_count": len(
                self.decision.lidar_static_map_admitted_candidate_uids
            ),
            "lidar_boundary_provisional_candidate_count": len(
                self.decision.lidar_boundary_provisional_candidate_uids
            ),
            "lidar_population_retained_candidate_count": len(
                self.decision.lidar_population_retained_candidate_uids
            ),
            "lidar_static_map_admitted_candidate_uids": list(
                self.decision.lidar_static_map_admitted_candidate_uids
            ),
            "lidar_boundary_provisional_candidate_uids": list(
                self.decision.lidar_boundary_provisional_candidate_uids
            ),
            "lidar_population_retained_candidate_uids": list(
                self.decision.lidar_population_retained_candidate_uids
            ),
            "completed_coverage_legs": self.completed_coverage_legs,
            "legs_completed_this_run": self.legs_completed_this_run,
            "next_viewpoint_id": None,
            "terminal_checkpoint_published": True,
            "lidar_checkpoint_complete": True,
            "camera_validation_population_ready": False,
            "candidate_snapshot_ready": False,
            "motion_published": True,
            "prior_leg_motion_published": True,
            "motion_authorized": False,
            "camera_approach_authorized": False,
        }


@dataclass(frozen=True)
class ExactTwoCameraCompletionRequest:
    """Already-admitted LiDAR evidence for the camera completion transaction."""

    run_mode: str
    session_id: str
    expected_stand_count: int
    survey_root: Path
    plan: CoverageSurveyPlan
    progress: CoverageSurveyProgress
    registry: StandSurveyRegistry
    terminal_checkpoint: PublishedCoverageCheckpointLike
    terminal_checkpoint_parent: Path | None
    lidar_admission_path: Path
    lidar_admission_sha256: str
    lidar_decision: ExactTwoLidarCheckpointDecision
    camera_admission_path: Path
    candidate_snapshot_path: Path
    camera_handoff_path: Path
    completed_coverage_legs: int
    legs_completed_this_run: int
    coverage_status: CoverageStatusLike


class ExactTwoCameraCompletionEffects(Protocol):
    """Structural camera effects supplied by ``CoverageMissionEffects``."""

    evaluate_exact_two_camera_admission: Callable[
        [
            CoverageSurveyPlan,
            CoverageSurveyProgress,
            StandSurveyRegistry,
            ExactTwoLidarCheckpointDecision,
        ],
        ExactTwoCameraAdmissionDecision,
    ]
    write_exact_two_camera_admission: Callable[
        [
            Path,
            ExactTwoCameraAdmissionDecision,
            PublishedCoverageCheckpointLike,
            Path,
            str,
        ],
        str,
    ]
    exact_two_camera_admission_sha256: Callable[
        [ExactTwoCameraAdmissionDecision], str
    ]
    build_exact_two_camera_snapshot: Callable[
        [
            CoverageSurveyPlan,
            StandSurveyRegistry,
            ExactTwoCameraAdmissionDecision,
            str,
        ],
        CandidateSnapshot,
    ]
    write_snapshot: Callable[[Path, CandidateSnapshot], str]
    snapshot_sha256: Callable[[CandidateSnapshot], str]
    create_exact_two_camera_handoff: Callable[
        [CoverageExactTwoCameraHandoffRequest],
        ExactTwoCameraHandoffArtifact,
    ]
    write_exact_two_camera_handoff: Callable[
        [Path, ExactTwoCameraHandoffArtifact], str
    ]
    exact_two_camera_handoff_sha256: Callable[
        [ExactTwoCameraHandoffArtifact], str
    ]


def write_bound_exact_two_camera_admission(
    path: Path,
    decision: ExactTwoCameraAdmissionDecision,
    terminal_checkpoint: PublishedCoverageCheckpointLike,
    lidar_admission_path: Path,
    lidar_admission_sha256: str,
) -> str:
    """Publish camera admission only with structurally valid LiDAR provenance."""

    checkpoint_path = getattr(terminal_checkpoint, "manifest_path", None)
    checkpoint_sha256 = getattr(terminal_checkpoint, "manifest_sha256", None)
    if not isinstance(checkpoint_path, Path):
        raise RuntimeError("exact-two camera admission requires a checkpoint")
    if not isinstance(checkpoint_sha256, str):
        raise RuntimeError("exact-two camera admission requires a checkpoint")
    _require_sha256(
        checkpoint_sha256,
        "terminal_checkpoint_manifest_sha256",
    )
    if not isinstance(lidar_admission_path, Path):
        raise RuntimeError("exact-two camera admission requires a LiDAR path")
    _require_sha256(lidar_admission_sha256, "lidar_checkpoint_admission_sha256")
    return write_exact_two_camera_admission(path, decision)


def build_exact_two_camera_snapshot_effect(
    plan: CoverageSurveyPlan,
    registry: StandSurveyRegistry,
    decision: ExactTwoCameraAdmissionDecision,
    snapshot_id: str,
) -> CandidateSnapshot:
    return build_exact_two_camera_candidate_snapshot(
        plan,
        registry,
        decision,
        snapshot_id=snapshot_id,
    )


def create_exact_two_camera_handoff(
    request: CoverageExactTwoCameraHandoffRequest,
) -> ExactTwoCameraHandoffArtifact:
    if (
        candidate_snapshot_sha256(request.candidate_snapshot)
        != request.candidate_snapshot_sha256
    ):
        raise RuntimeError(
            "exact-two handoff request changed the candidate snapshot"
        )
    return new_exact_two_camera_handoff(
        handoff_id=request.plan.survey_id,
        created_unix_sec=request.candidate_snapshot.created_unix_sec,
        admission=request.camera_admission,
        terminal_checkpoint_path=request.terminal_checkpoint.manifest_path,
        terminal_checkpoint_sha256=(
            request.terminal_checkpoint.manifest_sha256
        ),
        lidar_admission_path=request.lidar_admission_path,
        lidar_admission_sha256=request.lidar_admission_sha256,
        camera_admission_path=request.camera_admission_path,
        camera_admission_sha256=request.camera_admission_sha256,
        candidate_snapshot_path=request.candidate_snapshot_path,
        candidate_snapshot=request.candidate_snapshot,
    )


def complete_exact_two_camera(
    request: ExactTwoCameraCompletionRequest,
    effects: ExactTwoCameraCompletionEffects,
) -> CoverageExactTwoCameraReady:
    """Publish camera admission, snapshot, and handoff in strict order."""

    camera_decision = effects.evaluate_exact_two_camera_admission(
        request.plan,
        request.progress,
        request.registry,
        request.lidar_decision,
    )
    camera_admission_hash = effects.write_exact_two_camera_admission(
        request.camera_admission_path,
        camera_decision,
        request.terminal_checkpoint,
        request.lidar_admission_path,
        request.lidar_admission_sha256,
    )
    _require_sha256(
        camera_admission_hash,
        "exact_two_camera_admission_sha256",
    )
    expected_camera_admission_hash = (
        effects.exact_two_camera_admission_sha256(camera_decision)
    )
    _require_sha256(
        expected_camera_admission_hash,
        "computed_exact_two_camera_admission_sha256",
    )
    if camera_admission_hash != expected_camera_admission_hash:
        raise RuntimeError(
            "exact-two camera admission persistence hash mismatch"
        )
    if not camera_decision.ready:
        raise CoverageExactTwoCameraAdmissionError(
            checkpoint=request.terminal_checkpoint,
            checkpoint_parent_manifest=request.terminal_checkpoint_parent,
            lidar_admission_path=request.lidar_admission_path,
            lidar_admission_sha256=request.lidar_admission_sha256,
            camera_admission_path=request.camera_admission_path,
            camera_admission_sha256=camera_admission_hash,
            decision=camera_decision,
            completed_coverage_legs=request.completed_coverage_legs,
            legs_completed_this_run=request.legs_completed_this_run,
        )

    snapshot = effects.build_exact_two_camera_snapshot(
        request.plan,
        request.registry,
        camera_decision,
        f"{request.session_id}_candidates",
    )
    _require_exact_two_snapshot_population(
        snapshot,
        camera_decision,
        expected_stand_count=request.expected_stand_count,
    )
    written_snapshot_hash = effects.write_snapshot(
        request.candidate_snapshot_path,
        snapshot,
    )
    snapshot_hash = effects.snapshot_sha256(snapshot)
    _require_sha256(
        written_snapshot_hash,
        "written_candidate_snapshot_sha256",
    )
    _require_sha256(snapshot_hash, "candidate_snapshot_sha256")
    if written_snapshot_hash != snapshot_hash:
        raise RuntimeError("candidate snapshot persistence hash mismatch")

    handoff = effects.create_exact_two_camera_handoff(
        CoverageExactTwoCameraHandoffRequest(
            plan=request.plan,
            terminal_checkpoint=request.terminal_checkpoint,
            lidar_admission_path=request.lidar_admission_path,
            lidar_admission_sha256=request.lidar_admission_sha256,
            camera_admission_path=request.camera_admission_path,
            camera_admission_sha256=camera_admission_hash,
            camera_admission=camera_decision,
            candidate_snapshot_path=request.candidate_snapshot_path,
            candidate_snapshot_sha256=snapshot_hash,
            candidate_snapshot=snapshot,
        )
    )
    written_handoff_hash = effects.write_exact_two_camera_handoff(
        request.camera_handoff_path,
        handoff,
    )
    handoff_hash = effects.exact_two_camera_handoff_sha256(handoff)
    _require_sha256(
        written_handoff_hash,
        "written_exact_two_camera_handoff_sha256",
    )
    _require_sha256(handoff_hash, "exact_two_camera_handoff_sha256")
    if written_handoff_hash != handoff_hash:
        raise RuntimeError("exact-two camera handoff persistence hash mismatch")

    return CoverageExactTwoCameraReady(
        run_mode=request.run_mode,
        completed_coverage_legs=request.completed_coverage_legs,
        legs_completed_this_run=request.legs_completed_this_run,
        survey_root=request.survey_root,
        checkpoint_manifest=request.terminal_checkpoint.manifest_path,
        checkpoint_manifest_sha256=(
            request.terminal_checkpoint.manifest_sha256
        ),
        checkpoint_parent_manifest=request.terminal_checkpoint_parent,
        lidar_checkpoint_admission_path=request.lidar_admission_path,
        lidar_checkpoint_admission_sha256=request.lidar_admission_sha256,
        lidar_decision=request.lidar_decision,
        camera_validation_admission_path=request.camera_admission_path,
        camera_validation_admission_sha256=camera_admission_hash,
        camera_validation_decision=camera_decision,
        candidate_snapshot_path=request.candidate_snapshot_path,
        candidate_snapshot_sha256=snapshot_hash,
        candidate_snapshot=snapshot,
        camera_handoff_path=request.camera_handoff_path,
        camera_handoff_sha256=handoff_hash,
        camera_handoff=handoff,
        coverage_status=request.coverage_status,
    )


def _require_exact_two_snapshot_population(
    snapshot: CandidateSnapshot,
    decision: ExactTwoCameraAdmissionDecision,
    *,
    expected_stand_count: int,
) -> None:
    try:
        snapshot_uids = tuple(
            candidate.candidate_uid for candidate in snapshot.candidates
        )
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            "exact-two candidate snapshot returned invalid evidence"
        ) from exc
    admitted_uids = tuple(sorted(decision.admitted_candidate_uids))
    if snapshot_uids != admitted_uids:
        raise RuntimeError(
            "exact-two candidate snapshot population differs from admission"
        )
    if len(snapshot_uids) != expected_stand_count:
        raise RuntimeError(
            "exact-two candidate snapshot does not contain the expected stands"
        )


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


__all__ = (
    "COVERAGE_EXACT_TWO_CAMERA_READY",
    "EXACT_TWO_CAMERA_VALIDATION_READY",
    "CoverageExactTwoCameraAdmissionError",
    "CoverageExactTwoCameraHandoffRequest",
    "CoverageExactTwoCameraReady",
    "ExactTwoCameraAdmissionDecision",
    "ExactTwoCameraCompletionEffects",
    "ExactTwoCameraCompletionRequest",
    "ExactTwoCameraHandoffArtifact",
    "PublishedCoverageCheckpointLike",
    "build_exact_two_camera_snapshot_effect",
    "complete_exact_two_camera",
    "create_exact_two_camera_handoff",
    "evaluate_exact_two_camera_admission",
    "exact_two_camera_admission_sha256",
    "exact_two_camera_handoff_sha256",
    "write_bound_exact_two_camera_admission",
    "write_exact_two_camera_handoff",
)
