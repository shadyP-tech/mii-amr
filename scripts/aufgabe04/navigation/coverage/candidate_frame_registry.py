"""Frozen-frame provenance helpers for persistent survey-candidate fusion.

This module bridges detector evidence to canonical odom geometry.  It is
independent of the survey registry dataclasses so the long-lived coverage
state machine can retain a narrow orchestration role.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.approach.candidate_frame_reprojection import (
    CandidateFrameProvenance,
    CandidatePoint2D,
    current_map_point_from_canonical_odom,
)
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D,
)
from scripts.aufgabe04.perception.stand_confirmation import ConfirmedStand


FROZEN_FRAME_EVIDENCE_KEY = "frozen_odom_observation_geometry"


def frame_provenance_from_confirmed_stand(
    stand: ConfirmedStand,
    *,
    expected_map_frame: str,
    expected_map_bundle_sha256: str,
) -> CandidateFrameProvenance | None:
    """Recover canonical odom geometry from one confirmed LiDAR track.

    Synthetic and legacy observations may omit the contract.  A producer that
    declares frozen-odom geometry must provide a complete, consistent record;
    malformed physical evidence is never silently downgraded.
    """

    outer = stand.provenance
    if not isinstance(outer, Mapping):
        raise ValueError("confirmed stand provenance must be an object")
    selected = outer.get("provenance")
    if selected is None:
        return None
    if not isinstance(selected, Mapping):
        raise ValueError("confirmed stand selected provenance must be an object")
    runtime_config = selected.get("runtime_config")
    if not isinstance(runtime_config, Mapping):
        raise ValueError("confirmed stand runtime_config must be an object")
    frozen = runtime_config.get(FROZEN_FRAME_EVIDENCE_KEY)
    observer_version = str(selected.get("observer_version", ""))
    if frozen is None:
        if "frozen-odom" in observer_version:
            raise ValueError("frozen-odom observation is missing frame evidence")
        return None
    if not isinstance(frozen, Mapping):
        raise ValueError("frozen observation frame evidence must be an object")
    if frozen.get("schema_version") != 1:
        raise ValueError("unsupported frozen observation frame schema")
    if frozen.get("mode") != "frozen_map_from_odom":
        raise ValueError("unsupported frozen observation geometry mode")
    frames = frozen.get("source_frames")
    transform = frozen.get("map_from_odom")
    if not isinstance(frames, Mapping) or not isinstance(transform, Mapping):
        raise ValueError("frozen observation frame evidence is incomplete")
    map_frame = str(frames.get("map_frame", "")).strip("/")
    odom_frame = str(frames.get("odom_frame", "")).strip("/")
    if map_frame != expected_map_frame.strip("/"):
        raise ValueError("frozen observation map frame differs from survey")
    if str(selected.get("map_frame", "")).strip("/") != map_frame:
        raise ValueError("observation and frozen evidence map frames differ")
    if str(frozen.get("scan_tf_target_frame", "")).strip("/") != odom_frame:
        raise ValueError("frozen observation scan target is not its odom frame")
    if selected.get("map_bundle_sha256") != expected_map_bundle_sha256:
        raise ValueError("frozen observation map bundle differs from survey")
    certificate_sha256 = frozen.get("odom_execution_certificate_sha256")
    _validate_sha256(certificate_sha256, "odom_execution_certificate_sha256")
    try:
        frozen_map_from_odom = PlanarTransform2D(
            x_m=float(transform["x_m"]),
            y_m=float(transform["y_m"]),
            yaw_rad=float(transform["yaw_rad"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"frozen observation map_from_odom is invalid: {exc}"
        ) from exc
    return CandidateFrameProvenance.from_frozen_map_observation(
        map_frame=map_frame,
        odom_frame=odom_frame,
        frozen_map_point=CandidatePoint2D(stand.x_m, stand.y_m),
        frozen_map_from_odom=frozen_map_from_odom,
        source_evidence_id=str(certificate_sha256),
    )


def candidate_spatial_match_points(
    candidate_map_points: Sequence[tuple[float, float]],
    stand_map_points: Sequence[tuple[float, float]],
    *,
    candidate_frames: Sequence[CandidateFrameProvenance | None],
    stand_frames: Sequence[CandidateFrameProvenance | None],
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    """Select one coherent coordinate surface for spatial assignment."""

    if len(candidate_map_points) != len(candidate_frames):
        raise ValueError("candidate frame provenance count mismatch")
    if len(stand_map_points) != len(stand_frames):
        raise ValueError("stand frame provenance count mismatch")
    available_frames = (*candidate_frames, *stand_frames)
    if not any(frame is not None for frame in available_frames):
        return tuple(candidate_map_points), tuple(stand_map_points)
    if any(frame is None for frame in available_frames):
        raise ValueError(
            "cannot spatially fuse mixed legacy and frame-bound stand geometry"
        )
    typed_frames = tuple(
        frame for frame in available_frames if frame is not None
    )
    frame_pairs = {(frame.map_frame, frame.odom_frame) for frame in typed_frames}
    if len(frame_pairs) != 1:
        raise ValueError("candidate frame provenance uses incompatible frames")
    return (
        tuple(
            (
                frame.canonical_odom_point.x_m,
                frame.canonical_odom_point.y_m,
            )
            for frame in candidate_frames
            if frame is not None
        ),
        tuple(
            (
                frame.canonical_odom_point.x_m,
                frame.canonical_odom_point.y_m,
            )
            for frame in stand_frames
            if frame is not None
        ),
    )


def merge_candidate_frame_provenance(
    existing: CandidateFrameProvenance | None,
    incoming: CandidateFrameProvenance | None,
    *,
    existing_weight: int,
    incoming_weight: int,
) -> CandidateFrameProvenance | None:
    """Fuse geometry in odom and retain the newest frozen-map reference."""

    if existing is None and incoming is None:
        return None
    if existing is None or incoming is None:
        raise ValueError("cannot merge legacy and frame-bound candidate geometry")
    if (existing.map_frame, existing.odom_frame) != (
        incoming.map_frame,
        incoming.odom_frame,
    ):
        raise ValueError("candidate frame provenance uses incompatible frames")
    if existing_weight <= 0 or incoming_weight <= 0:
        raise ValueError("candidate frame fusion weights must be positive")
    total_weight = existing_weight + incoming_weight
    canonical_odom_point = CandidatePoint2D(
        x_m=(
            existing.canonical_odom_point.x_m * existing_weight
            + incoming.canonical_odom_point.x_m * incoming_weight
        )
        / total_weight,
        y_m=(
            existing.canonical_odom_point.y_m * existing_weight
            + incoming.canonical_odom_point.y_m * incoming_weight
        )
        / total_weight,
    )
    incoming_transform = incoming.frozen_map_from_odom
    if incoming_transform is None:
        raise ValueError(
            "incoming candidate frame provenance lacks its frozen transform"
        )
    return CandidateFrameProvenance(
        map_frame=existing.map_frame,
        odom_frame=existing.odom_frame,
        canonical_odom_point=canonical_odom_point,
        frozen_map_point=current_map_point_from_canonical_odom(
            canonical_odom_point,
            incoming_transform,
        ),
        frozen_map_from_odom=incoming_transform,
        source_evidence_id=payload_sha256(
            {
                "schema_version": 1,
                "fusion": "weighted_canonical_odom_geometry",
                "existing": existing.to_mapping(),
                "incoming": incoming.to_mapping(),
                "existing_weight": existing_weight,
                "incoming_weight": incoming_weight,
            }
        ),
    )


def _validate_sha256(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


__all__ = [
    "FROZEN_FRAME_EVIDENCE_KEY",
    "candidate_spatial_match_points",
    "frame_provenance_from_confirmed_stand",
    "merge_candidate_frame_provenance",
]
