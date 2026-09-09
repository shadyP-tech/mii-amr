"""Offline, hash-bound server identity evidence and observed QR promotion.

The seal proves content integrity, not a server signature or a scan ACK. A
caller must provide the saved robot-plans response and the intended server
robot ID explicitly. Discovery observations alone never invent station IDs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json, payload_sha256, write_content_hashed_json,
)
from scripts.aufgabe04.stations.candidate_snapshot import CandidateSnapshot, candidate_snapshot_sha256
from scripts.aufgabe04.stations.station_identity_registry import (
    StationIdentity, StationIdentityRegistry, new_station_identity_registry,
)
from scripts.aufgabe04.stations.station_ids import canonical_qr_id
from scripts.aufgabe04.task_client.server_response_decoder import decode_robot_plans

SERVER_MAPPING_HASH_FIELD = "server_qr_mapping_evidence_sha256"
OBSERVED_IDENTITIES_HASH_FIELD = "observed_station_identities_sha256"
DEFAULT_MAPPING_MAX_AGE_SEC = 3600.0  # Same bound as server task validation.


@dataclass(frozen=True)
class ServerQrMappingEvidence:
    """Validated immutable JSON representation; mappings are derived on use."""

    canonical_payload_json: str

    def to_payload(self) -> dict[str, object]:
        return json.loads(self.canonical_payload_json)

    @property
    def sha256(self) -> str:
        return payload_sha256(self.to_payload())

    @property
    def robot_id(self) -> str:
        return str(self.to_payload()["robot_id"])


def _finite(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def _selected_plan(plans_payload: object, robot_id: str):
    if not isinstance(plans_payload, list) or not all(isinstance(item, dict) for item in plans_payload):
        raise ValueError("robot-plans response must be a list of objects")
    if not isinstance(robot_id, str) or not robot_id or robot_id != robot_id.strip():
        raise ValueError("explicit server robot_id is required")
    plans = [plan for plan in decode_robot_plans(plans_payload) if plan.robot_id == robot_id]
    if len(plans) != 1:
        raise ValueError("robot-plans response must contain exactly one plan for the server robot_id")
    plan = plans[0]
    if not plan.qr_mappings:
        raise ValueError("selected robot plan has no authoritative qr_mappings")
    return plan


def _generated_sec(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("server plan generated_at must include a timezone")
    return _finite(parsed.astimezone(timezone.utc).timestamp(), "generated_at")


def seal_server_qr_mapping_evidence(
    plans_payload: Sequence[Mapping[str, object]], *, robot_id: str, captured_unix_sec: float,
) -> ServerQrMappingEvidence:
    # JSON roundtrip also detaches mutable input. Strict hashing rejects NaN.
    detached = json.loads(json.dumps(plans_payload, allow_nan=False))
    plan = _selected_plan(detached, robot_id)
    captured = _finite(captured_unix_sec, "captured_unix_sec")
    generated = _generated_sec(plan.generated_at)
    if generated > captured:
        raise ValueError("server plan generated_at is after evidence capture")
    payload = {
        "schema_version": 1,
        "source_kind": "offline_robot_plans_qr_mappings",
        "robot_id": robot_id,
        "captured_unix_sec": captured,
        "plan_generated_unix_sec": generated,
        "source_robot_plans_sha256": payload_sha256({"robot_plans": detached}),
        "robot_plans": detached,
        "motion_authorized": False,
        "server_ack_proven": False,
    }
    return ServerQrMappingEvidence(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))


def validate_server_qr_mapping_evidence(
    evidence: ServerQrMappingEvidence, *, robot_id: str, now_sec: float,
    max_age_sec: float = DEFAULT_MAPPING_MAX_AGE_SEC,
) -> None:
    payload = evidence.to_payload()
    expected = seal_server_qr_mapping_evidence(
        payload.get("robot_plans"), robot_id=robot_id,
        captured_unix_sec=payload.get("captured_unix_sec"),
    )
    if payload != expected.to_payload():
        raise ValueError("server QR mapping evidence schema, robot scope, or source hash mismatch")
    now = _finite(now_sec, "now_sec")
    max_age = _finite(max_age_sec, "max_age_sec")
    if max_age <= 0 or max_age > DEFAULT_MAPPING_MAX_AGE_SEC:
        raise ValueError("mapping maximum age must be positive and no greater than the server task bound")
    if now < payload["captured_unix_sec"] or now < payload["plan_generated_unix_sec"]:
        raise ValueError("server QR mapping evidence timestamp is in the future")
    if now - payload["plan_generated_unix_sec"] > max_age:
        raise ValueError("server QR mapping evidence is stale")


def write_server_qr_mapping_evidence(path: Path, evidence: ServerQrMappingEvidence) -> str:
    payload = evidence.to_payload()
    validate_server_qr_mapping_evidence(evidence, robot_id=evidence.robot_id, now_sec=payload["captured_unix_sec"])
    return write_content_hashed_json(path, payload, hash_field=SERVER_MAPPING_HASH_FIELD)


def load_server_qr_mapping_evidence(
    path: Path, *, robot_id: str, now_sec: float,
    max_age_sec: float = DEFAULT_MAPPING_MAX_AGE_SEC,
) -> ServerQrMappingEvidence:
    payload = load_content_hashed_json(path, hash_field=SERVER_MAPPING_HASH_FIELD)
    evidence = ServerQrMappingEvidence(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
    validate_server_qr_mapping_evidence(evidence, robot_id=robot_id, now_sec=now_sec, max_age_sec=max_age_sec)
    return evidence


def observed_identities_payload(
    *, candidate_snapshot: CandidateSnapshot, observed_qr_by_candidate: Mapping[str, str],
    session_id: str, observed_unix_sec: float,
) -> dict[str, object]:
    if set(observed_qr_by_candidate) != set(candidate_snapshot.candidate_uids):
        raise ValueError("observations must resolve exactly the supplied confirmed candidate snapshot")
    observed = {uid: canonical_qr_id(qr) for uid, qr in observed_qr_by_candidate.items()}
    if len(set(observed.values())) != len(observed):
        raise ValueError("observed QR identity is ambiguous across candidates")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("observation session_id is required")
    return {
        "schema_version": 1, "artifact_kind": "observed_candidate_qr_identities",
        "session_id": session_id, "observed_unix_sec": _finite(observed_unix_sec, "observed_unix_sec"),
        "candidate_snapshot_sha256": candidate_snapshot_sha256(candidate_snapshot),
        "observed_qr_by_candidate": dict(sorted(observed.items())),
        "binding_status": "server_binding_pending", "motion_authorized": False,
    }


def write_observed_identities(path: Path, **kwargs) -> str:
    return write_content_hashed_json(path, observed_identities_payload(**kwargs), hash_field=OBSERVED_IDENTITIES_HASH_FIELD)


def load_observed_identities(path: Path, *, candidate_snapshot: CandidateSnapshot) -> dict[str, object]:
    payload = load_content_hashed_json(path, hash_field=OBSERVED_IDENTITIES_HASH_FIELD)
    expected = observed_identities_payload(
        candidate_snapshot=candidate_snapshot,
        observed_qr_by_candidate=payload.get("observed_qr_by_candidate", {}),
        session_id=payload.get("session_id"), observed_unix_sec=payload.get("observed_unix_sec"),
    )
    if payload != expected:
        raise ValueError("observed identities schema or candidate snapshot hash mismatch")
    return payload


def bind_observed_station_identities(
    *, candidate_snapshot: CandidateSnapshot, observed_qr_by_candidate: Mapping[str, str],
    mapping_evidence: ServerQrMappingEvidence, registry_id: str, now_sec: float,
) -> StationIdentityRegistry:
    validate_server_qr_mapping_evidence(mapping_evidence, robot_id=mapping_evidence.robot_id, now_sec=now_sec)
    observations = observed_identities_payload(
        candidate_snapshot=candidate_snapshot, observed_qr_by_candidate=observed_qr_by_candidate,
        session_id=registry_id, observed_unix_sec=now_sec,
    )["observed_qr_by_candidate"]
    plan = _selected_plan(mapping_evidence.to_payload()["robot_plans"], mapping_evidence.robot_id)
    by_qr = {mapping.qr_code_id: mapping.station_id for mapping in plan.qr_mappings}
    missing = sorted(set(observations.values()) - set(by_qr))
    if missing:
        raise ValueError(f"observed QR has no authoritative server mapping: {missing}")
    return new_station_identity_registry(
        registry_id=registry_id, created_unix_sec=now_sec,
        candidate_snapshot_sha256=candidate_snapshot_sha256(candidate_snapshot),
        source_artifact_sha256=payload_sha256({
            "source_kind": "observed_qr_and_robot_scoped_server_mapping",
            "server_qr_mapping_evidence_sha256": mapping_evidence.sha256,
            "observed_qr_by_candidate": observations,
        }),
        expected_candidate_uids=candidate_snapshot.candidate_uids,
        mappings=(StationIdentity(uid, qr, by_qr[qr]) for uid, qr in observations.items()),
    )


def main(argv=None) -> int:
    """Seal saved server evidence, or bind already observed identities offline."""
    import argparse
    import sys
    import time
    from scripts.aufgabe04.stations.candidate_snapshot import load_candidate_snapshot
    from scripts.aufgabe04.stations.station_identity_registry import write_station_identity_registry

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("seal")
    seal.add_argument("--plans-json", type=Path, required=True)
    seal.add_argument("--server-robot-id", required=True)
    seal.add_argument("--output-json", type=Path, required=True)
    seal.add_argument("--captured-unix-sec", type=float, required=True)
    bind = commands.add_parser("bind")
    bind.add_argument("--candidate-snapshot", type=Path, required=True)
    bind.add_argument("--observed-identities", type=Path, required=True)
    bind.add_argument("--server-qr-mapping-evidence", type=Path, required=True)
    bind.add_argument("--server-robot-id", required=True)
    bind.add_argument("--registry-id", required=True)
    bind.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        now = time.time()
        if args.command == "seal":
            evidence = seal_server_qr_mapping_evidence(
                json.loads(args.plans_json.read_text()), robot_id=args.server_robot_id,
                captured_unix_sec=args.captured_unix_sec,
            )
            digest = write_server_qr_mapping_evidence(args.output_json, evidence)
        else:
            snapshot = load_candidate_snapshot(args.candidate_snapshot)
            observed = load_observed_identities(args.observed_identities, candidate_snapshot=snapshot)
            evidence = load_server_qr_mapping_evidence(args.server_qr_mapping_evidence, robot_id=args.server_robot_id, now_sec=now)
            registry = bind_observed_station_identities(
                candidate_snapshot=snapshot, observed_qr_by_candidate=observed["observed_qr_by_candidate"],
                mapping_evidence=evidence, registry_id=args.registry_id, now_sec=now,
            )
            digest = write_station_identity_registry(args.output_json, registry)
        print(json.dumps({"ok": True, "output_json": str(args.output_json), "sha256": digest, "motion_authorized": False}))
        return 0
    except (ValueError, TypeError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
