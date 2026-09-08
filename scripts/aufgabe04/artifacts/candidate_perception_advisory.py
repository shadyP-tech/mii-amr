"""Immutable unresolved LiDAR evidence; never candidate or motion authority."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Mapping

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.approach.candidate_frame_reprojection import (
    CandidateFrameProvenance,
)


MORPHOLOGY_CONFLICT = "cross_view_morphology_conflict"
VISIBILITY_GAP = "visibility_eligibility_gap"


@dataclass(frozen=True)
class CandidatePerceptionAdvisory:
    """Bind a historical ambiguity to the exact candidate and rejected track.

    Frames are snapshots: later candidate fusion must preserve this record,
    rather than pretending the historical evidence used the new centroid.
    """

    kind: str
    candidate_uid: str
    survey_id: str
    map_bundle_sha256: str
    plan_sha256: str
    viewpoint_id: str
    source_morphology_sha256: str
    candidate_frame: CandidateFrameProvenance
    candidate_source_viewpoint_ids: tuple[str, ...]
    source_observation_ids: tuple[str, ...]
    proposal_max_range_m: float
    visibility_radius_m: float
    eligible_other_viewpoint_ids: tuple[str, ...]
    track_id: str | None = None
    track_frame: CandidateFrameProvenance | None = None
    track_source_observation_ids: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    association_distance_m: float | None = None
    association_limit_m: float | None = None
    possible_candidate_uids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {MORPHOLOGY_CONFLICT, VISIBILITY_GAP}:
            raise ValueError("unknown candidate perception advisory kind")
        for name in ("candidate_uid", "survey_id", "viewpoint_id"):
            _identifier(getattr(self, name), name)
        for name in ("map_bundle_sha256", "plan_sha256", "source_morphology_sha256"):
            _sha256(getattr(self, name), name)
        for name in (
            "candidate_source_viewpoint_ids", "source_observation_ids",
            "eligible_other_viewpoint_ids", "track_source_observation_ids",
            "rejection_reasons", "possible_candidate_uids",
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise ValueError(f"{name} must be an immutable tuple of unique IDs")
            for value in values:
                _identifier(value, name)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique IDs")
        if not self.source_observation_ids or not self.candidate_source_viewpoint_ids:
            raise ValueError("advisory candidate source evidence is missing")
        _frame(self.candidate_frame)
        for name in ("proposal_max_range_m", "visibility_radius_m"):
            _positive(getattr(self, name), name)
        if set(self.eligible_other_viewpoint_ids) & set(self.candidate_source_viewpoint_ids):
            raise ValueError("eligible other viewpoints contain a source viewpoint")
        if self.kind == VISIBILITY_GAP:
            if self.eligible_other_viewpoint_ids:
                raise ValueError("visibility gap has an eligible other viewpoint")
            if self.proposal_max_range_m <= self.visibility_radius_m:
                raise ValueError("visibility gap requires wider proposal range")
            if any((self.track_id, self.track_frame, self.track_source_observation_ids,
                    self.rejection_reasons, self.possible_candidate_uids)) or (
                self.association_distance_m is not None or self.association_limit_m is not None
            ):
                raise ValueError("visibility gap must not claim a rejected track")
            return
        _identifier(self.track_id, "track_id")
        _frame(self.track_frame)
        if self.viewpoint_id in self.candidate_source_viewpoint_ids:
            raise ValueError("morphology conflict must come from another viewpoint")
        if not self.track_source_observation_ids or not self.rejection_reasons:
            raise ValueError("morphology conflict is missing rejected track evidence")
        if self.candidate_uid not in self.possible_candidate_uids:
            raise ValueError("morphology conflict does not include its candidate")
        if set(self.source_observation_ids) & set(self.track_source_observation_ids):
            raise ValueError("conflicting track reuses candidate observation IDs")
        _positive(self.association_limit_m, "association_limit_m")
        _positive(self.association_distance_m, "association_distance_m", allow_zero=True)
        if self.association_distance_m > self.association_limit_m + 1e-12:
            raise ValueError("morphology conflict exceeds association radius")
        if (self.candidate_frame.map_frame, self.candidate_frame.odom_frame) != (
            self.track_frame.map_frame, self.track_frame.odom_frame
        ):
            raise ValueError("conflicting candidate and track frames differ")
        a = self.candidate_frame.canonical_odom_point
        b = self.track_frame.canonical_odom_point
        if abs(math.hypot(a.x_m - b.x_m, a.y_m - b.y_m) - self.association_distance_m) > 1e-9:
            raise ValueError("morphology association distance differs from frame evidence")

    def _payload(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": 1,
            "status": "unresolved",
            "motion_authorized": False,
            "candidate_rejection_authorized": False,
        }
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, CandidateFrameProvenance):
                value = value.to_mapping()
            elif isinstance(value, tuple):
                value = list(value)
            result[field.name] = value
        return result

    @property
    def sha256(self) -> str:
        return payload_sha256(self._payload())

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "advisory_sha256": self.sha256}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CandidatePerceptionAdvisory":
        if not isinstance(value, Mapping):
            raise ValueError("candidate perception advisory must be an object")
        payload = dict(value)
        digest = payload.pop("advisory_sha256", None)
        _sha256(digest, "advisory_sha256")
        if digest != payload_sha256(payload):
            raise ValueError("candidate perception advisory hash mismatch")
        constants = {"schema_version": 1, "status": "unresolved",
                     "motion_authorized": False, "candidate_rejection_authorized": False}
        for key, expected in constants.items():
            actual = payload.pop(key, None)
            if actual != expected or type(actual) is not type(expected):
                raise ValueError(f"invalid advisory {key}")
        if set(payload) != {field.name for field in fields(cls)}:
            raise ValueError("candidate perception advisory fields differ from schema")
        for key in ("candidate_frame", "track_frame"):
            if payload[key] is not None:
                payload[key] = CandidateFrameProvenance.from_mapping(payload[key])
        for key in ("candidate_source_viewpoint_ids", "source_observation_ids",
                    "eligible_other_viewpoint_ids", "track_source_observation_ids",
                    "rejection_reasons", "possible_candidate_uids"):
            if not isinstance(payload[key], list):
                raise ValueError(f"advisory {key} must be a list")
            payload[key] = tuple(payload[key])
        return cls(**payload)


def _identifier(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"advisory {name} must be nonempty")


def _sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"advisory {name} must be SHA256")


def _frame(value: object) -> None:
    if not isinstance(value, CandidateFrameProvenance):
        raise ValueError("advisory requires canonical frame provenance")
    CandidateFrameProvenance.from_mapping(value.to_mapping())
    _sha256(value.source_evidence_id, "frame source_evidence_id")


def _positive(value: object, name: str, *, allow_zero: bool = False) -> None:
    if type(value) not in (float, int) or not math.isfinite(value) or (
        value < 0 if allow_zero else value <= 0
    ):
        raise ValueError(f"advisory {name} must be finite and positive")


def validate_candidate_perception_advisory(
    value: CandidatePerceptionAdvisory, *, candidate_uid: str | None = None,
) -> CandidatePerceptionAdvisory:
    if not isinstance(value, CandidatePerceptionAdvisory):
        raise ValueError("expected CandidatePerceptionAdvisory")
    value.__post_init__()
    if candidate_uid is not None and value.candidate_uid != candidate_uid:
        raise ValueError("perception advisory belongs to another candidate")
    return value


def advisory_payload(value: CandidatePerceptionAdvisory) -> dict[str, object]:
    return validate_candidate_perception_advisory(value).to_dict()


def advisory_from_payload(value: Mapping[str, object]) -> CandidatePerceptionAdvisory:
    return CandidatePerceptionAdvisory.from_mapping(value)
