"""Persisted preauthorization readiness for the first coverage route.

The parent runner injects every effect.  This module owns the immutable path
contract and orchestration only: one route seal, bounded no-motion dry
readiness, ordered semantic events, and one content-hashed readiness receipt.
It deliberately has no ROS, subprocess, or operator-input dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Callable, Mapping

from scripts.aufgabe04.real_robot.readiness.initial import (
    InitialReadinessDryRequest,
    InitialReadinessRejected,
    InitialReadinessResult,
    SealedRoutePaths,
    run_initial_readiness,
)


_RUN_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _path(value: object, field: str, suffix: str | None = None) -> Path:
    if not isinstance(value, Path):
        raise ValueError(f"{field} must be a Path")
    if value == Path(".") or not str(value).strip() or "\x00" in str(value):
        raise ValueError(f"{field} must identify one non-empty path")
    if ".." in value.parts:
        raise ValueError(f"{field} must not contain traversal")
    if suffix is not None and value.suffix != suffix:
        raise ValueError(f"{field} must end in {suffix}")
    return value


@dataclass(frozen=True)
class PreauthorizationReadinessPaths:
    """All derived paths for exactly one first-route readiness admission."""

    source_route_csv: Path
    source_diagnostics_json: Path
    coverage_plan_json: Path
    readiness_root: Path
    sealed_route_csv: Path
    sealed_diagnostics_json: Path
    sealed_route_certificate_json: Path
    event_log_jsonl: Path
    readiness_evidence_json: Path

    def __post_init__(self) -> None:
        suffixes = {
            "source_route_csv": ".csv",
            "source_diagnostics_json": ".json",
            "coverage_plan_json": ".json",
            "sealed_route_csv": ".csv",
            "sealed_diagnostics_json": ".json",
            "sealed_route_certificate_json": ".json",
            "event_log_jsonl": ".jsonl",
            "readiness_evidence_json": ".json",
        }
        for field, suffix in suffixes.items():
            object.__setattr__(
                self,
                field,
                _path(getattr(self, field), field, suffix),
            )
        object.__setattr__(
            self,
            "readiness_root",
            _path(self.readiness_root, "readiness_root"),
        )
        artifact_paths = (
            self.source_route_csv,
            self.source_diagnostics_json,
            self.coverage_plan_json,
            self.sealed_route_csv,
            self.sealed_diagnostics_json,
            self.sealed_route_certificate_json,
            self.event_log_jsonl,
            self.readiness_evidence_json,
        )
        if len(set(artifact_paths)) != len(artifact_paths):
            raise ValueError(
                "preauthorization readiness artifact paths must be distinct"
            )
        for sealed in (
            self.sealed_route_csv,
            self.sealed_diagnostics_json,
            self.sealed_route_certificate_json,
            self.readiness_evidence_json,
        ):
            try:
                sealed.relative_to(self.readiness_root)
            except ValueError as exc:
                raise ValueError(
                    "sealed readiness artifacts must be inside readiness_root"
                ) from exc

    @property
    def sealed_output_dir(self) -> Path:
        return self.sealed_route_csv.parent

    def to_evidence(self) -> dict[str, str]:
        return {
            "source_route_csv": str(self.source_route_csv),
            "source_diagnostics_json": str(self.source_diagnostics_json),
            "coverage_plan_json": str(self.coverage_plan_json),
            "readiness_root": str(self.readiness_root),
            "sealed_route_csv": str(self.sealed_route_csv),
            "sealed_diagnostics_json": str(self.sealed_diagnostics_json),
            "sealed_route_certificate_json": str(
                self.sealed_route_certificate_json
            ),
            "readiness_event_log_jsonl": str(self.event_log_jsonl),
            "readiness_evidence_json": str(self.readiness_evidence_json),
        }


@dataclass(frozen=True)
class PreauthorizationReadinessConfig:
    session_root: Path
    survey_root: Path
    coverage_plan_path: Path
    session_id: str
    initial_leg_index: int
    maximum_localization_readiness_retries: int
    observation_tf_evidence_path: Path
    observation_tf_evidence_sha256: str

    def __post_init__(self) -> None:
        session_root = _path(self.session_root, "session_root")
        survey_root = _path(self.survey_root, "survey_root")
        coverage_plan = _path(
            self.coverage_plan_path,
            "coverage_plan_path",
            ".json",
        )
        observation_tf = _path(
            self.observation_tf_evidence_path,
            "observation_tf_evidence_path",
            ".json",
        )
        object.__setattr__(self, "session_root", session_root)
        object.__setattr__(self, "survey_root", survey_root)
        object.__setattr__(self, "coverage_plan_path", coverage_plan)
        object.__setattr__(self, "observation_tf_evidence_path", observation_tf)
        if not isinstance(self.session_id, str) or not _RUN_TOKEN.fullmatch(
            self.session_id
        ):
            raise ValueError("session_id must be a path-safe token")
        if session_root.name != self.session_id:
            raise ValueError("session_root must end with the exact session_id")
        if survey_root != session_root / "coverage":
            raise ValueError("survey_root must be the session coverage directory")
        if coverage_plan != survey_root / "coverage_plan.json":
            raise ValueError(
                "coverage_plan_path must be the survey coverage_plan.json"
            )
        expected_observation_tf = (
            session_root / "preflight/lidar_scan_tf_before_authorization.json"
        )
        if observation_tf != expected_observation_tf:
            raise ValueError(
                "observation_tf_evidence_path must be the preauthorization "
                "scan-TF receipt"
            )
        if type(self.initial_leg_index) is not int or self.initial_leg_index < 0:
            raise ValueError("initial_leg_index must be a non-negative integer")
        if (
            type(self.maximum_localization_readiness_retries) is not int
            or self.maximum_localization_readiness_retries < 0
        ):
            raise ValueError(
                "maximum_localization_readiness_retries must be a non-negative integer"
            )
        if not isinstance(self.observation_tf_evidence_sha256, str) or not (
            _SHA256.fullmatch(self.observation_tf_evidence_sha256)
        ):
            raise ValueError("observation_tf_evidence_sha256 must be a SHA-256")
        try:
            observation_tf.relative_to(self.readiness_root)
        except ValueError:
            pass
        else:
            raise ValueError(
                "observation TF evidence must be outside authorization_readiness"
            )

    @property
    def readiness_root(self) -> Path:
        return (
            self.session_root
            / "authorization_readiness"
            / f"coverage_leg_{self.initial_leg_index:03d}"
        )

    @property
    def paths(self) -> PreauthorizationReadinessPaths:
        readiness_root = self.readiness_root
        sealed_root = readiness_root / "sealed"
        return PreauthorizationReadinessPaths(
            source_route_csv=(
                self.survey_root
                / "legs"
                / f"leg_{self.initial_leg_index:03d}_route.csv"
            ),
            source_diagnostics_json=(
                self.survey_root
                / "legs"
                / f"leg_{self.initial_leg_index:03d}_diagnostics.json"
            ),
            coverage_plan_json=self.coverage_plan_path,
            readiness_root=readiness_root,
            sealed_route_csv=sealed_root / "route.csv",
            sealed_diagnostics_json=sealed_root / "route_diagnostics.json",
            sealed_route_certificate_json=sealed_root / "route_certificate.json",
            event_log_jsonl=self.session_root / "adaptive_replans.jsonl",
            readiness_evidence_json=readiness_root / "readiness.json",
        )

    @property
    def run_id_prefix(self) -> str:
        return (
            f"{self.session_id}_preauthorization_coverage_"
            f"{self.initial_leg_index:03d}"
        )


@dataclass(frozen=True)
class PreauthorizationReadinessEffects:
    seal_route: Callable[..., Mapping[str, object]]
    run_dry_motion_leg: Callable[[InitialReadinessDryRequest], object]
    append_event: Callable[[Path, dict[str, object]], None]
    publish_hashed_json: Callable[..., str]
    wall_clock: Callable[[], float]
    notify: Callable[[str], None]
    prepare_localization_attempt: (
        Callable[[InitialReadinessDryRequest], None] | None
    ) = None

    def __post_init__(self) -> None:
        for field in (
            "seal_route",
            "run_dry_motion_leg",
            "append_event",
            "publish_hashed_json",
            "wall_clock",
            "notify",
        ):
            if not callable(getattr(self, field)):
                raise ValueError(f"{field} must be callable")
        if self.prepare_localization_attempt is not None and not callable(
            self.prepare_localization_attempt
        ):
            raise ValueError("prepare_localization_attempt must be callable or None")


@dataclass(frozen=True)
class PreauthorizationReadinessOutcome:
    result: InitialReadinessResult
    paths: PreauthorizationReadinessPaths
    evidence_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.result, InitialReadinessResult) or not self.result.ready:
            raise ValueError("outcome requires a ready InitialReadinessResult")
        if not isinstance(self.paths, PreauthorizationReadinessPaths):
            raise ValueError("paths must be PreauthorizationReadinessPaths")
        if not isinstance(self.evidence_sha256, str) or not _SHA256.fullmatch(
            self.evidence_sha256
        ):
            raise ValueError("evidence_sha256 must be a SHA-256")

    @property
    def evidence_path(self) -> Path:
        return self.paths.readiness_evidence_json


class PreauthorizationReadinessContractError(RuntimeError):
    """Fail-closed adapter error before any operator or motion authority."""

    def __init__(
        self,
        reason_code: str,
        *,
        paths: PreauthorizationReadinessPaths,
        cause: BaseException | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.paths = paths
        self.cause = cause
        super().__init__(f"preauthorization readiness contract failed: {reason_code}")

    def to_failure_fields(self) -> dict[str, object]:
        return {
            "failure_phase": "preauthorization_first_leg_readiness",
            "preauthorization_readiness_reason": self.reason_code,
            **self.paths.to_evidence(),
            "typed_run_requested": False,
            "operator_input_requested": False,
            "motion_authorized": False,
            "motion_published": False,
            "permit_issued": False,
            "reusable_as_motion_permit": False,
            "route_limits_unchanged": True,
        }


def _contract_error(
    reason_code: str,
    paths: PreauthorizationReadinessPaths,
    cause: BaseException | None = None,
) -> PreauthorizationReadinessContractError:
    return PreauthorizationReadinessContractError(
        reason_code,
        paths=paths,
        cause=cause,
    )


def _validated_timestamp(
    effects: PreauthorizationReadinessEffects,
    paths: PreauthorizationReadinessPaths,
) -> float:
    try:
        timestamp = effects.wall_clock()
    except Exception as exc:
        raise _contract_error("event_clock_failed", paths, exc) from exc
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not math.isfinite(float(timestamp))
    ):
        raise _contract_error("event_clock_invalid", paths)
    return float(timestamp)


def admit_preauthorization_readiness(
    config: PreauthorizationReadinessConfig,
    effects: PreauthorizationReadinessEffects,
) -> PreauthorizationReadinessOutcome:
    """Seal once and persist a bounded, no-motion first-route admission."""

    if not isinstance(config, PreauthorizationReadinessConfig):
        raise ValueError("config must be PreauthorizationReadinessConfig")
    if not isinstance(effects, PreauthorizationReadinessEffects):
        raise ValueError("effects must be PreauthorizationReadinessEffects")
    paths = config.paths
    try:
        sealed_payload = effects.seal_route(
            source_route_csv=paths.source_route_csv,
            source_diagnostics_json=paths.source_diagnostics_json,
            coverage_plan_path=paths.coverage_plan_json,
            output_dir=paths.sealed_output_dir,
        )
        sealed_route = SealedRoutePaths.from_mapping(sealed_payload)
    except Exception as exc:
        raise _contract_error("first_route_seal_failed", paths, exc) from exc
    expected_sealed = SealedRoutePaths(
        paths.sealed_route_csv,
        paths.sealed_diagnostics_json,
        paths.sealed_route_certificate_json,
    )
    if sealed_route != expected_sealed:
        raise _contract_error("sealed_route_path_mismatch", paths)

    def dry_runner(request: InitialReadinessDryRequest) -> object:
        if request.attempt_index > 0:
            effects.notify(
                "First-route AMCL uncertainty is not ready. Correct the "
                "RViz 2D Pose Estimate at the known physical start; the "
                f"no-motion admission is retrying ({request.attempt_index}/"
                f"{config.maximum_localization_readiness_retries})."
            )
        if effects.prepare_localization_attempt is not None:
            try:
                effects.prepare_localization_attempt(request)
            except Exception as exc:
                raise _contract_error(
                    "localization_attempt_preparation_failed",
                    paths,
                    exc,
                ) from exc
        outcome = effects.run_dry_motion_leg(request)
        if getattr(outcome, "status", None) != "dry_run_ok":
            reason = getattr(outcome, "stop_reason", "unparseable dry outcome")
            effects.notify(
                "First-route readiness rejected without motion: " + str(reason)
            )
        return outcome

    result = run_initial_readiness(
        sealed_route=sealed_route,
        session_root=config.session_root,
        run_id_prefix=config.run_id_prefix,
        maximum_retries=config.maximum_localization_readiness_retries,
        dry_runner=dry_runner,
    )
    for event in result.to_events():
        payload = {**event, "timestamp": _validated_timestamp(effects, paths)}
        try:
            effects.append_event(paths.event_log_jsonl, payload)
        except Exception as exc:
            raise _contract_error(
                "readiness_event_persistence_failed", paths, exc
            ) from exc

    evidence = {
        **result.to_evidence(),
        **paths.to_evidence(),
        "observation_tf_readiness_json": str(
            config.observation_tf_evidence_path
        ),
        "observation_tf_readiness_sha256": (
            config.observation_tf_evidence_sha256
        ),
    }
    try:
        evidence_sha256 = effects.publish_hashed_json(
            paths.readiness_evidence_json,
            evidence,
            hash_field="initial_readiness_sha256",
        )
    except Exception as exc:
        raise _contract_error(
            "readiness_evidence_persistence_failed", paths, exc
        ) from exc
    if not isinstance(evidence_sha256, str) or not _SHA256.fullmatch(
        evidence_sha256
    ):
        raise _contract_error("readiness_evidence_sha256_invalid", paths)
    if not result.ready:
        raise InitialReadinessRejected(
            result,
            evidence_path=paths.readiness_evidence_json,
            evidence_sha256=evidence_sha256,
        )
    return PreauthorizationReadinessOutcome(result, paths, evidence_sha256)


__all__ = [
    "PreauthorizationReadinessConfig",
    "PreauthorizationReadinessContractError",
    "PreauthorizationReadinessEffects",
    "PreauthorizationReadinessOutcome",
    "PreauthorizationReadinessPaths",
    "admit_preauthorization_readiness",
]
