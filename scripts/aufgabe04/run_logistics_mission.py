"""Top-level single-robot Aufgabe 04 mission CLI.

This CLI currently implements the FastAPI-backed dry-run task layer only. It
does not publish motion and does not invoke the navigation follower.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.logistics.server_validation.validators import (
    build_server_task_snapshot,
    validate_server_task,
)
from scripts.aufgabe04.logistics.server_validation.artifacts import (
    validated_task_snapshot_sha256,
    write_validated_task_snapshot,
)
from scripts.aufgabe04.qr_scanning.qr_id_decoder import decode_qr_id
from scripts.aufgabe04.qr_scanning.scan_logger import append_qr_scan
from scripts.aufgabe04.stations.station_map import DEFAULT_STATIONS
from scripts.aufgabe04.stations.station_identity_registry import (
    load_station_identity_registry,
)
from scripts.aufgabe04.task_client.event_logger import append_task_event
from scripts.aufgabe04.task_client.fastapi_client import (
    fetch_admin_status,
    fetch_openapi,
    fetch_robot_plans,
    health,
    report_scanned_qr,
)
from scripts.aufgabe04.task_client.models import FastApiConfig
from scripts.aufgabe04.task_client.openapi_discovery import discover_scan_endpoint_template
from scripts.aufgabe04.task_client.server_response_decoder import (
    decode_robot_plans,
    decode_robot_statuses,
)


DEFAULT_QR_SCAN_LOG = Path("results/aufgabe04/qr_scans.csv")
DEFAULT_TASK_EVENT_LOG = Path("results/aufgabe04/task_server_events.jsonl")
DEFAULT_TASK_SNAPSHOT_DIR = Path("results/aufgabe04/task_snapshots")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-base-url", default="http://192.168.0.105:8000")
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--qr-id", required=True)
    parser.add_argument("--timeout-sec", type=float, default=3.0)
    parser.add_argument("--scan-endpoint-template", default="")
    parser.add_argument("--status-json", type=Path, default=None)
    parser.add_argument("--plans-json", type=Path, default=None)
    parser.add_argument("--station-identity-registry", type=Path, default=None)
    parser.add_argument("--validated-task-json", type=Path, default=None)
    parser.add_argument("--skip-health", action="store_true")
    parser.add_argument("--report-scan", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-task", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--qr-scan-log", type=Path, default=DEFAULT_QR_SCAN_LOG)
    parser.add_argument("--task-event-log", type=Path, default=DEFAULT_TASK_EVENT_LOG)
    parser.add_argument(
        "--local-station",
        action="append",
        default=[],
        help="Known local station id. May be repeated.",
    )
    parser.add_argument(
        "--max-status-age-sec",
        type=float,
        default=300.0,
    )
    parser.add_argument(
        "--max-plan-age-sec",
        type=float,
        default=3600.0,
    )
    return parser


def _load_json(path: Path):
    return json.loads(path.read_text())


def _local_station_ids(extra_ids: Iterable[str]) -> tuple[str, ...]:
    station_ids = {station_id.upper() for station_id in DEFAULT_STATIONS}
    station_ids.update(station_id.strip().upper() for station_id in extra_ids if station_id.strip())
    return tuple(sorted(station_ids))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("run_logistics_mission currently supports FastAPI task flow only with --dry-run")
    args.run_id = args.run_id or f"aufgabe04-task-{uuid.uuid4().hex[:8]}"
    config = FastApiConfig(
        base_url=args.server_base_url,
        robot_id=args.robot_id,
        timeout_sec=args.timeout_sec,
        scanned_qr_endpoint_template=args.scan_endpoint_template or None,
    )
    scanned = decode_qr_id(args.qr_id, source="cli")
    identity_registry = (
        None
        if args.station_identity_registry is None
        else load_station_identity_registry(args.station_identity_registry)
    )
    scanned_identity = (
        None
        if identity_registry is None
        else identity_registry.for_qr(scanned.qr_id)
    )
    if identity_registry is not None and scanned_identity is None:
        parser.error("scanned QR is absent from --station-identity-registry")
    append_task_event(
        args.task_event_log,
        "task_flow_started",
        {
            "run_id": args.run_id,
            "robot_id": args.robot_id,
            "server_base_url": args.server_base_url,
            "qr_id": scanned.qr_id,
            "dry_run": args.dry_run,
        },
    )
    if not args.skip_health and args.status_json is None:
        health_payload = health(config)
        append_task_event(args.task_event_log, "health_checked", {"run_id": args.run_id, "payload": health_payload})
    status_payload = _load_json(args.status_json) if args.status_json else fetch_admin_status(config)
    plans_payload = _load_json(args.plans_json) if args.plans_json else fetch_robot_plans(config)
    statuses = decode_robot_statuses(status_payload)
    plans = decode_robot_plans(plans_payload)
    snapshot = build_server_task_snapshot(
        robot_id=args.robot_id,
        scanned_qr_id=scanned.qr_id,
        statuses=statuses,
        plans=plans,
    )
    if args.report_scan:
        if config.scanned_qr_endpoint_template is None:
            openapi_payload = fetch_openapi(config)
            discovered_template = discover_scan_endpoint_template(openapi_payload)
            config = FastApiConfig(
                base_url=config.base_url,
                robot_id=config.robot_id,
                timeout_sec=config.timeout_sec,
                scanned_qr_endpoint_template=discovered_template,
            )
            append_task_event(
                args.task_event_log,
                "scan_endpoint_discovered",
                {"run_id": args.run_id, "endpoint_template": discovered_template},
            )
        report_payload = report_scanned_qr(
            config,
            qr_id=scanned.qr_id,
            station_id=snapshot.resolved_station_id,
        )
        append_task_event(args.task_event_log, "scan_reported", {"run_id": args.run_id, "payload": report_payload})
        # Reporting a scan may advance the server-side mission.  Never validate
        # or persist the pre-report fixture/cache as the task to execute: fetch
        # both authoritative endpoints again after the POST and rebuild the
        # snapshot from that refreshed pair.  Exceptions deliberately
        # propagate so a partial/failed refresh cannot produce a validated task.
        status_payload = fetch_admin_status(config)
        plans_payload = fetch_robot_plans(config)
        statuses = decode_robot_statuses(status_payload)
        plans = decode_robot_plans(plans_payload)
        snapshot = build_server_task_snapshot(
            robot_id=args.robot_id,
            scanned_qr_id=scanned.qr_id,
            statuses=statuses,
            plans=plans,
        )
        append_task_event(
            args.task_event_log,
            "post_scan_task_refetched",
            {
                "run_id": args.run_id,
                "robot_id": args.robot_id,
                "resolved_station_id": snapshot.resolved_station_id,
                "status_count": len(statuses),
                "plan_count": len(plans),
            },
        )
    validated = validate_server_task(
        snapshot,
        local_station_ids=(
            _local_station_ids(args.local_station)
            if identity_registry is None
            else tuple(
                identity.server_station_id
                for identity in identity_registry.mappings
            )
        ),
        now=datetime.now(timezone.utc),
        max_status_age_sec=args.max_status_age_sec,
        max_plan_age_sec=args.max_plan_age_sec,
    )
    if (
        scanned_identity is not None
        and validated.resolved_current_station
        != scanned_identity.server_station_id
    ):
        parser.error(
            "server QR mapping disagrees with --station-identity-registry"
        )
    task_digest = validated_task_snapshot_sha256(validated)
    task_snapshot_directory = (
        DEFAULT_TASK_SNAPSHOT_DIR
        if args.task_event_log == DEFAULT_TASK_EVENT_LOG
        else args.task_event_log.parent / "task_snapshots"
    )
    task_snapshot_path = args.validated_task_json or task_snapshot_directory / (
        f"task_{validated.mission_id}_{task_digest[:16]}.json"
    )
    write_validated_task_snapshot(task_snapshot_path, validated)
    navigation_request = validated.to_navigation_request()
    append_qr_scan(
        args.qr_scan_log,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": args.run_id,
            "robot_id": args.robot_id,
            "raw_text": scanned.raw_text,
            "qr_id": scanned.qr_id,
            "resolved_station_id": snapshot.resolved_station_id,
            "source": scanned.source,
            "confidence": scanned.confidence,
            "status": "validated",
            "reason": "",
        },
    )
    append_task_event(
        args.task_event_log,
        "task_validated",
        {
            "run_id": args.run_id,
            "validated_task": validated,
            "navigation_request": navigation_request,
            "validated_task_snapshot": str(task_snapshot_path),
            "validated_task_snapshot_sha256": task_digest,
        },
    )
    if args.print_task:
        print(json.dumps(navigation_request, default=lambda value: getattr(value, "__dict__", str(value)), indent=2, sort_keys=True))
    else:
        print(
            "Validated dry-run task: "
            f"robot={navigation_request.robot_id} "
            f"current={navigation_request.current_station_id} "
            f"target={navigation_request.target_station_id} "
            f"state={navigation_request.server_state}"
        )
        print(f"Validated task snapshot: {task_snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
