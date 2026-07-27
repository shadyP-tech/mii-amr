"""Create a sealed candidate/QR/server-station identity registry.

This command is offline and ROS-free. It validates and publishes identity
metadata only; it does not scan QR codes, start a mission, or move a robot.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateSnapshot,
    candidate_snapshot_sha256,
    load_candidate_snapshot,
)
from scripts.aufgabe04.stations.station_identity_registry import (
    StationIdentity,
    StationIdentityRegistry,
    new_station_identity_registry,
    station_identity_registry_sha256,
    validate_station_identity,
    write_station_identity_registry,
)


def _parse_mapping(value: str) -> StationIdentity:
    parts = tuple(part.strip() for part in value.split("="))
    if len(parts) != 3 or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "mapping must be CANDIDATE_UID=QR_ID=SERVER_STATION_ID"
        )
    return StationIdentity(
        candidate_uid=parts[0],
        qr_id=parts[1],
        server_station_id=parts[2],
    )


def mapping_source_sha256(mappings: Iterable[StationIdentity]) -> str:
    """Hash the canonical operator-supplied mapping as its source evidence."""

    canonical = tuple(
        sorted(
            mappings,
            key=lambda item: (
                item.candidate_uid,
                item.qr_id,
                item.server_station_id,
            ),
        )
    )
    for mapping in canonical:
        validate_station_identity(mapping)
    return payload_sha256(
        {
            "source_kind": "operator_cli_mappings_v1",
            "mappings": [
                {
                    "candidate_uid": mapping.candidate_uid,
                    "qr_id": mapping.qr_id,
                    "server_station_id": mapping.server_station_id,
                }
                for mapping in canonical
            ],
        }
    )


def create_registry(
    *,
    candidate_snapshot: CandidateSnapshot,
    mappings: Iterable[StationIdentity],
    registry_id: Optional[str],
    created_unix_sec: float,
) -> tuple[StationIdentityRegistry, str]:
    """Build a registry and return it with the canonical mapping-source hash."""

    selected_mappings = tuple(mappings)
    source_sha256 = mapping_source_sha256(selected_mappings)
    snapshot_sha256 = candidate_snapshot_sha256(candidate_snapshot)
    selected_registry_id = registry_id or (
        f"station_identity_{snapshot_sha256[:12]}_{source_sha256[:12]}"
    )
    registry = new_station_identity_registry(
        registry_id=selected_registry_id,
        created_unix_sec=created_unix_sec,
        candidate_snapshot_sha256=snapshot_sha256,
        source_artifact_sha256=source_sha256,
        expected_candidate_uids=candidate_snapshot.candidate_uids,
        mappings=selected_mappings,
    )
    return registry, source_sha256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-snapshot",
        required=True,
        type=Path,
        help="Immutable candidate snapshot JSON to bind.",
    )
    parser.add_argument(
        "--mapping",
        action="append",
        required=True,
        type=_parse_mapping,
        metavar="CANDIDATE_UID=QR_ID=SERVER_STATION_ID",
        help=(
            "One explicit identity mapping. Repeat exactly once for every "
            "candidate in the snapshot."
        ),
    )
    parser.add_argument(
        "--registry-id",
        default=None,
        help="Optional stable registry ID; defaults to snapshot/mapping hashes.",
    )
    parser.add_argument(
        "--created-unix-sec",
        type=float,
        default=None,
        help=(
            "Creation timestamp. Defaults to the current time; supply it for "
            "byte-identical reproducible retries."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Output path. By default, writes beside the snapshot using the "
            "full registry content hash."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        snapshot = load_candidate_snapshot(args.candidate_snapshot)
        created_unix_sec = (
            time.time()
            if args.created_unix_sec is None
            else args.created_unix_sec
        )
        registry, source_sha256 = create_registry(
            candidate_snapshot=snapshot,
            mappings=args.mapping,
            registry_id=args.registry_id,
            created_unix_sec=created_unix_sec,
        )
        registry_sha256 = station_identity_registry_sha256(registry)
        output_json = args.output_json or (
            args.candidate_snapshot.parent
            / f"station_identity_registry_{registry_sha256}.json"
        )
        written_sha256 = write_station_identity_registry(output_json, registry)
        if written_sha256 != registry_sha256:
            raise ValueError("published registry hash differs from prepared registry")
        print(
            json.dumps(
                {
                    "candidate_snapshot_sha256": registry.candidate_snapshot_sha256,
                    "mapping_count": len(registry.mappings),
                    "mapping_source_sha256": source_sha256,
                    "ok": True,
                    "output_json": str(output_json),
                    "registry_id": registry.registry_id,
                    "station_identity_registry_sha256": registry_sha256,
                },
                indent=2,
                sort_keys=True,
            )
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
