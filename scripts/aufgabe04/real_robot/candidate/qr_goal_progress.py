"""Bounded hypothesis dispositions and distinct, validated QR mission progress.

The immutable survey snapshot remains the obstacle/route authority. This ledger
records QR discovery separately from geometry-backed facing readiness;
rejected, exhausted, ambiguous, and unvisited hypotheses are never removed from
that snapshot. A duplicate QR quarantines every claimant instead of choosing a
spatial identity or counting the same message twice.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import os
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json, write_content_hashed_json,
)
from scripts.aufgabe04.navigation.coverage.candidate_inspection_pool import (
    candidate_inspection_pool_count_reasons, candidate_inspection_pool_policy_evidence,
)
from scripts.aufgabe04.real_robot.candidate.observation_deferral import (
    CandidateApproachIncompleteError,
    CandidateObservationAttemptEvidence,
)
from scripts.aufgabe04.stations.station_identity_registry import (
    StationIdentityRegistry,
)
from scripts.aufgabe04.stations.station_ids import canonical_qr_id

from scripts.aufgabe04.stations.candidate_snapshot import CandidateSnapshot


GOAL_PROGRESS_HASH_FIELD = "candidate_goal_progress_sha256"
GEOMETRY_FACING_EVIDENCE = "geometry_validated_facing_pose"
QR_OBSERVATION_EVIDENCE = "qr_verified_observation_pose"


def resolve_candidate_qr_goal(*, configured_count: int | None,
                              plan_count: int | None) -> int:
    """Require one positive goal bound to the immutable coverage plan."""
    if type(plan_count) is not int or plan_count <= 0:
        raise ValueError("coverage plan must seal a positive expected_stand_count")
    selected = plan_count if configured_count is None else configured_count
    if type(selected) is not int or selected != plan_count:
        raise ValueError("candidate expected_stand_count differs from the sealed coverage plan")
    return selected


class CandidateQrGoalIncompleteError(CandidateApproachIncompleteError):
    """The bounded candidate pool did not yield the configured identity goal."""

    def __init__(self, progress: "CandidateQrGoalProgress", *,
                 attempt_evidence: Iterable[CandidateObservationAttemptEvidence]) -> None:
        confirmed = progress.confirmed_candidate_uids
        super().__init__(
            resolved_candidate_uids=confirmed,
            unresolved_candidate_uids=set(progress.candidate_uids) - set(confirmed),
            attempt_evidence=attempt_evidence,
            max_attempts_per_candidate=1,
            final_pass_index=0,
        )
        self.progress = progress.to_dict()
        self.args = (
            "candidate approach incomplete: bounded candidate pool yielded "
            f"{len(confirmed)} unambiguous validated QR identities; "
            f"expected {progress.expected_stand_count}",
        )

    def to_failure_fields(self) -> dict[str, object]:
        return {
            **super().to_failure_fields(),
            "failure_phase": "candidate_qr_goal_incomplete",
            **self.progress,
            "motion_continues_authorized": False,
            "fail_closed": True,
        }


class CandidateQrGoalProgress:
    """Mission goal independent of the number of LiDAR hypotheses."""

    def __init__(self, candidate_uids: Iterable[str], *, expected_stand_count: int,
                 candidate_snapshot_sha256: str,
                 perception_advisories_by_uid: Mapping[str, list[dict[str, object]]] | None = None) -> None:
        uids = tuple(sorted(candidate_uids))
        if not uids or len(set(uids)) != len(uids) or any(
            not isinstance(uid, str) or not uid.strip() for uid in uids
        ):
            raise ValueError("candidate goal requires unique non-empty candidate UIDs")
        count_reasons = candidate_inspection_pool_count_reasons(expected_stand_count, len(uids))
        if count_reasons:
            raise ValueError("invalid candidate inspection pool: " + ", ".join(count_reasons))
        self.candidate_uids = uids
        self.expected_stand_count = expected_stand_count
        self.candidate_snapshot_sha256 = candidate_snapshot_sha256
        self._records: dict[str, dict[str, object]] = {
            uid: {"candidate_uid": uid, "disposition": "pending", "qr_id": None,
                  "perception_advisories": list((perception_advisories_by_uid or {}).get(uid, []))}
            for uid in uids
        }
        self._qr_claims: dict[str, list[str]] = {}
        self._inspection_order: list[str] = []

    @property
    def confirmed_candidate_uids(self) -> tuple[str, ...]:
        return tuple(uid for uid in self.candidate_uids
                     if self._records[uid]["disposition"] == "confirmed_unique_qr")

    @property
    def complete(self) -> bool:
        return len(self.confirmed_candidate_uids) == self.expected_stand_count

    @property
    def facing_ready_candidate_uids(self) -> tuple[str, ...]:
        return tuple(uid for uid in self.confirmed_candidate_uids
                     if self._records[uid]["evidence_kind"] == GEOMETRY_FACING_EVIDENCE)

    @property
    def qr_only_candidate_uids(self) -> tuple[str, ...]:
        return tuple(uid for uid in self.confirmed_candidate_uids
                     if self._records[uid]["evidence_kind"] == QR_OBSERVATION_EVIDENCE)

    @property
    def facing_complete(self) -> bool:
        return len(self.facing_ready_candidate_uids) == self.expected_stand_count

    def _record(self, uid: str) -> dict[str, object]:
        if uid not in self._records:
            raise ValueError(f"unknown goal candidate {uid!r}")
        return self._records[uid]

    def mark_inspection_started(self, uid: str) -> None:
        record = self._record(uid)
        if uid in self._inspection_order:
            raise RuntimeError("candidate must not repeat its local inspection episode")
        self._inspection_order.append(uid)
        record["disposition"] = "inspection_started"

    def mark_unavailable(self, uid: str, *, disposition: str,
                         evidence: Mapping[str, object]) -> None:
        if disposition not in {
            "inspection_exhausted", "no_feasible_route", "route_admission_deferred",
            "route_admission_exhausted",
        }:
            raise ValueError("unsupported candidate unavailability disposition")
        record = self._record(uid)
        if record["qr_id"] is not None:
            raise RuntimeError("cannot discard a validated QR claim as unavailable")
        record.update(disposition=disposition, evidence=dict(evidence))

    def record_validated_identity(self, uid: str, qr_id: str, *,
                                  recommendation_path: Path) -> bool:
        """Record only after joint recommendation and facing validation passed.

        Return true only for a presently unambiguous mapping. A second claimant
        revokes the first claimant from the completion count, retaining both
        observations and all geometry as explicit duplicate ambiguity.
        """
        return self._record_identity(
            uid, qr_id, evidence_kind=GEOMETRY_FACING_EVIDENCE,
            evidence_path_field="recommendation_path", evidence_path=recommendation_path,
        )

    def record_observed_identity(self, uid: str, qr_id: str, *,
                                 observation_pose_path: Path) -> bool:
        """Count a validated QR observation without asserting a stand angle.

        The caller validates current, stopped, uniquely associated observation
        evidence before recording it. The robot observation pose is not a stand
        pose or a geometry-backed facing target. Duplicate quarantine is shared
        with geometry-backed observations.
        """
        return self._record_identity(
            uid, qr_id, evidence_kind=QR_OBSERVATION_EVIDENCE,
            evidence_path_field="observation_pose_path", evidence_path=observation_pose_path,
        )

    def _record_identity(self, uid: str, qr_id: str, *, evidence_kind: str,
                         evidence_path_field: str, evidence_path: Path) -> bool:
        record = self._record(uid)
        qr_id = canonical_qr_id(qr_id)
        if record["qr_id"] is not None:
            raise RuntimeError("candidate already has a validated QR claim")
        if not isinstance(evidence_path, Path) or str(evidence_path) == ".":
            raise ValueError("validated QR claim requires an evidence artifact path")
        record.update(qr_id=qr_id, evidence_kind=evidence_kind,
                      facing_ready=evidence_kind == GEOMETRY_FACING_EVIDENCE)
        record[evidence_path_field] = str(evidence_path)
        claimants = self._qr_claims.setdefault(qr_id, [])
        claimants.append(uid)
        if len(claimants) == 1:
            record["disposition"] = "confirmed_unique_qr"
            return True
        for claimant in claimants:
            self._records[claimant].update(
                disposition="ambiguous_duplicate_qr",
                facing_ready=False,
                conflicting_candidate_uids=sorted(claimants),
                spatial_merge_authorized=False,
            )
        return False

    def finalize_goal(self) -> None:
        if not self.complete:
            raise RuntimeError("cannot finalize an incomplete QR goal")
        for record in self._records.values():
            if record["disposition"] == "pending":
                record["disposition"] = "not_visited_goal_reached"

    def to_dict(self) -> dict[str, object]:
        confirmed = self.confirmed_candidate_uids
        # Detach nested evidence so callers cannot mutate the ledger.
        dispositions = json.loads(json.dumps([
            self._records[uid] for uid in self.candidate_uids
        ], allow_nan=False))
        return {
            "schema_version": 1,
            "artifact_kind": "candidate_qr_goal_progress",
            "candidate_snapshot_sha256": self.candidate_snapshot_sha256,
            "expected_stand_count": self.expected_stand_count,
            "inspection_pool_policy": candidate_inspection_pool_policy_evidence(self.expected_stand_count),
            "candidate_pool_count": len(self.candidate_uids),
            "goal_completed": self.complete,
            "confirmed_stand_count": len(confirmed),
            "confirmed_candidate_uids": list(confirmed),
            "facing_complete": self.facing_complete,
            "facing_ready_stand_count": len(self.facing_ready_candidate_uids),
            "facing_ready_candidate_uids": list(self.facing_ready_candidate_uids),
            "qr_only_stand_count": len(self.qr_only_candidate_uids),
            "qr_only_candidate_uids": list(self.qr_only_candidate_uids),
            "confirmed_qr_ids": sorted(str(self._records[uid]["qr_id"]) for uid in confirmed),
            "remaining_candidate_uids": sorted(set(self.candidate_uids) - set(confirmed)),
            "unvisited_candidate_uids": sorted(set(self.candidate_uids) - set(self._inspection_order)),
            "candidate_dispositions": dispositions,
            "inspection_order": list(self._inspection_order),
            "keepout_candidate_uids": list(self.candidate_uids),
            "candidate_geometry_unchanged": True,
            "motion_authorized": False,
        }


class CandidateQrGoalProgressStore:
    """Publish immutable revisions and an atomic human-readable latest pointer."""

    def __init__(self, session_root: Path) -> None:
        self.session_root = session_root
        self.revision = 0

    def write(self, progress: CandidateQrGoalProgress) -> tuple[Path, str]:
        payload = progress.to_dict()
        path = (self.session_root / "candidate_goal_history"
                / f"revision_{self.revision:03d}.json")
        digest = write_content_hashed_json(path, payload, hash_field=GOAL_PROGRESS_HASH_FIELD)
        pointer = self.session_root / "candidate_goal_progress.json"
        temporary = pointer.with_suffix(".tmp")
        temporary.write_text(json.dumps({
            **payload, "latest_revision_path": str(path),
            "latest_revision_sha256": digest,
        }, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, pointer)
        self.revision += 1
        return path, digest


def validate_candidate_qr_goal_completion(
    progress_path: Path, *, candidate_snapshot: CandidateSnapshot,
    confirmed_candidate_snapshot: CandidateSnapshot, identity_registry: StationIdentityRegistry | None = None,
    expected_stand_count: int, observed_qr_by_candidate: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    """Verify final progress against both snapshots and the identity registry.

    This validates the resulting artifact graph; it does not grant motion or
    manufacture camera/pose evidence owned by the caller's observation validator.
    """
    from dataclasses import replace
    from scripts.aufgabe04.stations.candidate_snapshot import candidate_snapshot_sha256
    from scripts.aufgabe04.stations.station_identity_registry import validate_station_identity_registry

    payload = load_content_hashed_json(progress_path, hash_field=GOAL_PROGRESS_HASH_FIELD)
    pool_uids = candidate_snapshot.candidate_uids
    confirmed_uids = confirmed_candidate_snapshot.candidate_uids
    expected_payload = {
        "schema_version": 1,
        "artifact_kind": "candidate_qr_goal_progress",
        "candidate_snapshot_sha256": candidate_snapshot_sha256(candidate_snapshot),
        "expected_stand_count": expected_stand_count,
        "inspection_pool_policy": candidate_inspection_pool_policy_evidence(expected_stand_count),
        "candidate_pool_count": len(pool_uids),
        "goal_completed": True,
        "confirmed_stand_count": expected_stand_count,
        "confirmed_candidate_uids": list(confirmed_uids),
        "remaining_candidate_uids": sorted(set(pool_uids) - set(confirmed_uids)),
        "keepout_candidate_uids": list(pool_uids),
        "candidate_geometry_unchanged": True,
        "motion_authorized": False,
    }
    for field, expected in expected_payload.items():
        if type(payload.get(field)) is not type(expected) or payload[field] != expected:
            raise ValueError(f"candidate QR goal completion mismatch: {field}")
    if type(expected_stand_count) is not int or expected_stand_count <= 0:
        raise ValueError("expected_stand_count must be a positive integer")
    if len(confirmed_uids) != expected_stand_count or not set(confirmed_uids) <= set(pool_uids):
        raise ValueError("confirmed candidate set does not satisfy the configured goal")
    count_reasons = candidate_inspection_pool_count_reasons(expected_stand_count, len(pool_uids))
    if count_reasons:
        raise ValueError("invalid final candidate pool: " + ", ".join(count_reasons))
    subset = replace(candidate_snapshot, candidates=tuple(
        candidate for candidate in candidate_snapshot.candidates
        if candidate.candidate_uid in set(confirmed_uids)
    ))
    if subset != confirmed_candidate_snapshot:
        raise ValueError("confirmed snapshot changed source candidate geometry or provenance")
    if identity_registry is not None:
        validate_station_identity_registry(identity_registry, candidate_snapshot=confirmed_candidate_snapshot)
        identities = {mapping.candidate_uid: mapping.qr_id for mapping in identity_registry.mappings}
        if observed_qr_by_candidate is not None and dict(observed_qr_by_candidate) != identities:
            raise ValueError("observed QR identities differ from the identity registry")
    else:
        if observed_qr_by_candidate is None:
            raise ValueError("goal validation requires observed identities or a bound registry")
        identities = {uid: canonical_qr_id(qr) for uid, qr in observed_qr_by_candidate.items()}
        if set(identities) != set(confirmed_uids) or len(set(identities.values())) != expected_stand_count:
            raise ValueError("observed identities must uniquely resolve the confirmed snapshot")
    if payload.get("confirmed_qr_ids") != sorted(identities.values()):
        raise ValueError("goal QR identities differ from the identity registry")
    dispositions = payload.get("candidate_dispositions")
    if not isinstance(dispositions, list) or any(not isinstance(item, dict) for item in dispositions):
        raise ValueError("candidate goal dispositions are invalid")
    if [item.get("candidate_uid") for item in dispositions] != list(pool_uids):
        raise ValueError("candidate goal dispositions do not retain the full hypothesis pool")
    allowed = {"confirmed_unique_qr", "inspection_exhausted", "no_feasible_route",
               "route_admission_deferred", "route_admission_exhausted",
               "ambiguous_duplicate_qr", "not_visited_goal_reached"}
    claims: dict[str, list[str]] = {}
    facing_uids: list[str] = []
    qr_only_uids: list[str] = []
    modern_evidence = False
    for item in dispositions:
        uid = item["candidate_uid"]
        if item.get("disposition") not in allowed:
            raise ValueError("candidate goal contains unfinished or unknown disposition")
        if (item["disposition"] == "confirmed_unique_qr") != (uid in identities):
            raise ValueError("candidate confirmed disposition differs from final identities")
        if uid in identities and item.get("qr_id") != identities[uid]:
            raise ValueError("candidate disposition QR differs from final identity")
        if item.get("qr_id") is not None:
            claims.setdefault(str(item["qr_id"]), []).append(uid)
            kind = item.get("evidence_kind", GEOMETRY_FACING_EVIDENCE)
            modern_evidence = modern_evidence or "evidence_kind" in item
            if kind not in {GEOMETRY_FACING_EVIDENCE, QR_OBSERVATION_EVIDENCE}:
                raise ValueError("candidate QR claim has unsupported evidence kind")
            has_facing_geometry = kind == GEOMETRY_FACING_EVIDENCE
            facing_ready = has_facing_geometry and item["disposition"] == "confirmed_unique_qr"
            if ("evidence_kind" in item or "facing_ready" in item) and item.get("facing_ready") is not facing_ready:
                raise ValueError("candidate QR claim has untruthful facing readiness")
            path_field = "recommendation_path" if has_facing_geometry else "observation_pose_path"
            evidence_path = item.get(path_field)
            if not isinstance(evidence_path, str) or not evidence_path.strip() or evidence_path == ".":
                raise ValueError("candidate QR claim lacks its evidence artifact reference")
            if not has_facing_geometry and item.get("recommendation_path") is not None:
                raise ValueError("QR-only discovery must not claim a facing recommendation")
            if uid in identities:
                (facing_uids if facing_ready else qr_only_uids).append(uid)
        candidate = candidate_snapshot.candidate_for(uid)
        advisories = [advisory.to_dict() for advisory in candidate.source.perception_advisories]
        if item.get("perception_advisories") != advisories:
            raise ValueError("candidate goal dropped or changed perception advisories")
    readiness_fields = {
        "facing_complete": len(facing_uids) == expected_stand_count,
        "facing_ready_stand_count": len(facing_uids),
        "facing_ready_candidate_uids": facing_uids,
        "qr_only_stand_count": len(qr_only_uids),
        "qr_only_candidate_uids": qr_only_uids,
    }
    # Old geometry-only ledgers did not contain readiness metadata. Any new
    # evidence or readiness field requires the whole internally consistent set.
    if modern_evidence or any(field in payload for field in readiness_fields):
        for field, expected in readiness_fields.items():
            if type(payload.get(field)) is not type(expected) or payload[field] != expected:
                raise ValueError(f"candidate QR goal readiness mismatch: {field}")
    for item in dispositions:
        if item["disposition"] == "ambiguous_duplicate_qr":
            claimants = claims.get(str(item.get("qr_id")), [])
            if (len(claimants) < 2 or item.get("conflicting_candidate_uids") != sorted(claimants)
                    or item.get("spatial_merge_authorized") is not False):
                raise ValueError("duplicate QR disposition lacks complete quarantine evidence")
    if any(len(uids) > 1 and qr_id in identities.values() for qr_id, uids in claims.items()):
        raise ValueError("duplicate QR must not contribute to the completion goal")
    visited = payload.get("inspection_order")
    if (not isinstance(visited, list) or any(not isinstance(uid, str) for uid in visited)
            or len(visited) != len(set(visited)) or not set(visited) <= set(pool_uids)
            or not set(confirmed_uids) <= set(visited)):
        raise ValueError("candidate goal inspection history is invalid")
    if payload.get("unvisited_candidate_uids") != sorted(set(pool_uids) - set(visited)):
        raise ValueError("candidate goal unvisited disposition differs from inspection history")
    return payload
