"""Bounded no-motion localization admission after a coverage observation.

The observation at the current stopped viewpoint is a separate transaction
from preparing the following motion leg.  This module owns only the latter
readiness policy.  It can repeat a fresh stationary preflight when, and only
when, the persisted evidence shows that the direct dynamic ``map<-odom``
message was the sole failed gate.  It never plans a route, issues a permit, or
publishes velocity.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Callable, Mapping

from scripts.aufgabe04.navigation.localization.localization_ownership import (
    FAIL_MAP_TO_ODOM,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D


DYNAMIC_MAP_TO_ODOM_UNAVAILABLE = FAIL_MAP_TO_ODOM
POST_OBSERVATION_LOCALIZATION_PHASE = "post_observation_localization"


@dataclass(frozen=True)
class PostObservationLocalizationConfig:
    session_root: Path
    session_id: str
    recorded_viewpoint_id: str
    maximum_retry_count: int

    def __post_init__(self) -> None:
        if not str(self.session_id).strip():
            raise ValueError("session_id must be non-empty")
        if not str(self.recorded_viewpoint_id).strip():
            raise ValueError("recorded_viewpoint_id must be non-empty")
        if (
            type(self.maximum_retry_count) is not int
            or self.maximum_retry_count < 0
        ):
            raise ValueError("maximum_retry_count must be a non-negative integer")


@dataclass(frozen=True)
class PostObservationLocalizationDecision:
    retryable: bool
    reason: str


@dataclass(frozen=True)
class PostObservationLocalizationAdmission:
    pose: Pose2D
    evidence_path: Path
    retry_count: int


class PostObservationLocalizationError(RuntimeError):
    """Terminal readiness failure with checkpoint-friendly evidence fields."""

    def __init__(
        self,
        *,
        config: PostObservationLocalizationConfig,
        reason: str,
        evidence_paths: tuple[Path, ...],
        retry_count: int,
        cause: BaseException,
    ) -> None:
        self.config = config
        self.reason_code = reason
        self.evidence_paths = evidence_paths
        self.retry_count = retry_count
        self.cause = cause
        super().__init__(
            "post-observation localization admission failed after "
            f"{retry_count + 1} attempt(s): {cause}"
        )

    def to_failure_fields(self) -> dict[str, object]:
        return {
            "failure_phase": POST_OBSERVATION_LOCALIZATION_PHASE,
            "recorded_viewpoint_id": self.config.recorded_viewpoint_id,
            "post_observation_localization_reason": self.reason_code,
            "post_observation_localization_retry_count": self.retry_count,
            "post_observation_localization_maximum_retry_count": (
                self.config.maximum_retry_count
            ),
            "post_observation_localization_evidence": [
                str(path) for path in self.evidence_paths
            ],
            "post_observation_localization_last_evidence": (
                None
                if not self.evidence_paths
                else str(self.evidence_paths[-1])
            ),
            "post_observation_retry_motion_published": False,
            "post_observation_retry_motion_authorized": False,
            "additional_typed_run_required": False,
        }


def _read_evidence(path: Path) -> Mapping[str, object]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("post-observation localization evidence is unavailable")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"invalid post-observation localization evidence: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("post-observation localization evidence must be an object")
    return payload


def _ignore_event(_event: Mapping[str, object]) -> None:
    return None


@dataclass(frozen=True)
class PostObservationLocalizationEffects:
    admit_localization: Callable[[Path], Pose2D]
    read_evidence: Callable[[Path], Mapping[str, object]] = _read_evidence
    event_sink: Callable[[Mapping[str, object]], None] = _ignore_event
    clock: Callable[[], float] = time.time


def evaluate_post_observation_localization_retry(
    evidence: Mapping[str, object],
) -> PostObservationLocalizationDecision:
    """Admit only the exact, sole direct-dynamic-transform observation gap."""

    if not isinstance(evidence, Mapping):
        return PostObservationLocalizationDecision(False, "evidence_not_mapping")
    if evidence.get("ok") is not False:
        return PostObservationLocalizationDecision(False, "evidence_not_failed")
    failures = evidence.get("failures")
    if failures != [DYNAMIC_MAP_TO_ODOM_UNAVAILABLE]:
        return PostObservationLocalizationDecision(
            False,
            "failure_set_not_exact_dynamic_map_to_odom_gap",
        )
    observations = evidence.get("observations")
    if (
        not isinstance(observations, list)
        or not observations
        or any(not isinstance(item, Mapping) for item in observations)
    ):
        return PostObservationLocalizationDecision(
            False,
            "observations_missing_or_invalid",
        )
    ownership = [
        item
        for item in observations
        if isinstance(item, Mapping)
        and item.get("name") == "localization transform ownership"
    ]
    if len(ownership) != 1:
        return PostObservationLocalizationDecision(
            False,
            "ownership_observation_not_unique",
        )
    observation = ownership[0]
    data = observation.get("data")
    dynamic = data.get("map_to_odom_dynamic") if isinstance(data, Mapping) else None
    exact_ownership_state = (
        isinstance(data, Mapping)
        and data.get("localization_source") == "amcl"
        and data.get("execution_pose_owner") == "amcl"
        and data.get("amcl_fresh") is True
        and data.get("route_transform_fresh") is True
        and data.get("map_to_odom_dynamic_fresh") is False
        and data.get("external_tf_owner_candidates") == []
        and data.get("ambiguous_owner_evidence") == []
    )
    if (
        observation.get("ok") is not False
        or observation.get("detail") != DYNAMIC_MAP_TO_ODOM_UNAVAILABLE
        or not exact_ownership_state
        or not isinstance(dynamic, Mapping)
        or dynamic.get("available") is not False
        or dynamic.get("dynamic") is not False
    ):
        return PostObservationLocalizationDecision(
            False,
            "ownership_observation_not_exact_dynamic_gap",
        )
    if any(
        item.get("ok") is not True
        for item in observations
        if item is not observation
    ):
        return PostObservationLocalizationDecision(
            False,
            "additional_preflight_gate_failed",
        )
    return PostObservationLocalizationDecision(
        True,
        "fresh_no_motion_dynamic_tf_admission_allowed",
    )


def post_observation_localization_evidence_path(
    config: PostObservationLocalizationConfig,
    attempt_index: int,
) -> Path:
    if type(attempt_index) is not int or attempt_index < 0:
        raise ValueError("attempt_index must be a non-negative integer")
    suffix = "" if attempt_index == 0 else f"_retry_{attempt_index:03d}"
    return (
        Path(config.session_root)
        / "preflight"
        / (
            f"{config.session_id}_{config.recorded_viewpoint_id}"
            f"_post_observation_localization{suffix}.json"
        )
    )


def admit_post_observation_localization(
    config: PostObservationLocalizationConfig,
    effects: PostObservationLocalizationEffects,
) -> PostObservationLocalizationAdmission:
    """Obtain fresh next-leg localization with a bounded no-motion retry."""

    evidence_paths: list[Path] = []
    retry_count = 0
    while True:
        evidence_path = post_observation_localization_evidence_path(
            config,
            retry_count,
        )
        evidence_paths.append(evidence_path)
        try:
            pose = effects.admit_localization(evidence_path)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            try:
                decision = evaluate_post_observation_localization_retry(
                    effects.read_evidence(evidence_path)
                )
            except (OSError, TypeError, ValueError):
                decision = PostObservationLocalizationDecision(
                    False,
                    "evidence_unavailable_or_invalid",
                )
            if (
                not decision.retryable
                or retry_count >= config.maximum_retry_count
            ):
                raise PostObservationLocalizationError(
                    config=config,
                    reason=decision.reason,
                    evidence_paths=tuple(evidence_paths),
                    retry_count=retry_count,
                    cause=exc,
                ) from exc
            retry_count += 1
            effects.event_sink(
                {
                    "schema_version": 1,
                    "event": "post_observation_localization_retry_scheduled",
                    "timestamp": float(effects.clock()),
                    "recorded_viewpoint_id": config.recorded_viewpoint_id,
                    "rejected_evidence_json": str(evidence_path),
                    "next_evidence_json": str(
                        post_observation_localization_evidence_path(
                            config,
                            retry_count,
                        )
                    ),
                    "next_retry_index": retry_count,
                    "maximum_retry_count": config.maximum_retry_count,
                    "reason": decision.reason,
                    "fresh_nomotion_amcl_preflight_required": True,
                    "motion_published": False,
                    "motion_authorized": False,
                    "additional_typed_run_required": False,
                    "safety_limits_unchanged": True,
                }
            )
            continue

        if not isinstance(pose, Pose2D) or not all(
            math.isfinite(value)
            for value in (pose.x_m, pose.y_m, pose.yaw_rad)
        ):
            raise PostObservationLocalizationError(
                config=config,
                reason="admitted_pose_invalid",
                evidence_paths=tuple(evidence_paths),
                retry_count=retry_count,
                cause=ValueError("admitted post-observation pose is invalid"),
            )
        effects.event_sink(
            {
                "schema_version": 1,
                "event": "post_observation_localization_admitted",
                "timestamp": float(effects.clock()),
                "recorded_viewpoint_id": config.recorded_viewpoint_id,
                "localization_evidence_json": str(evidence_path),
                "retry_count": retry_count,
                "fresh_start_pose": {
                    "x_m": pose.x_m,
                    "y_m": pose.y_m,
                    "yaw_rad": pose.yaw_rad,
                },
                "motion_published": False,
                "motion_authorized": False,
                "additional_typed_run_required": False,
            }
        )
        return PostObservationLocalizationAdmission(
            pose=pose,
            evidence_path=evidence_path,
            retry_count=retry_count,
        )


__all__ = [
    "DYNAMIC_MAP_TO_ODOM_UNAVAILABLE",
    "POST_OBSERVATION_LOCALIZATION_PHASE",
    "PostObservationLocalizationAdmission",
    "PostObservationLocalizationConfig",
    "PostObservationLocalizationDecision",
    "PostObservationLocalizationEffects",
    "PostObservationLocalizationError",
    "admit_post_observation_localization",
    "evaluate_post_observation_localization_retry",
    "post_observation_localization_evidence_path",
]
