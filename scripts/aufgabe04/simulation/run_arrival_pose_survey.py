#!/usr/bin/env python3
"""Survey every simulated stand and record arrivals without visiting them."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.route_revision_store import (  # noqa: E402
    RouteRevisionError,
    read_route_revision,
)
from scripts.aufgabe04.stations.arrival_pose_catalog import (  # noqa: E402
    arrival_pose_catalog_sha256,
    load_arrival_pose_catalog,
)
from scripts.aufgabe04.stations.arrival_pose_models import (  # noqa: E402
    CatalogProvenance,
)


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
DEFAULT_KNOWN_STAND_KEEPOUT_RADIUS_M = 0.26


@dataclass(frozen=True)
class SurveyCandidate:
    candidate_uid: str
    stand_id: str
    x_m: float
    y_m: float
    keepout_radius_m: float | None = None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _catalog_provenance(
    args,
    *,
    map_sha256: str,
    world_sha256: str,
) -> CatalogProvenance:
    """Bind every coordinator catalog read to this exact simulation run."""

    return CatalogProvenance(
        planning_frame=args.map_frame,
        map_yaml_sha256=map_sha256,
        world_id=args.world.stem,
        world_sha256=world_sha256,
        session_id=args.session_id,
        environment="simulation",
    )


def _survey_stream_id(session_id: str, candidate_uid: str) -> str:
    """Return a bounded stream identity unique to one survey session/candidate."""

    identity = f"{session_id}\0{candidate_uid}".encode("utf-8")
    return f"survey-{hashlib.sha256(identity).hexdigest()[:32]}"


def _load_candidates(path: Path) -> tuple[SurveyCandidate, ...]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        raise ValueError("candidates JSON must contain a candidates list")
    candidates = []
    seen = set()
    for index, item in enumerate(payload["candidates"]):
        if not isinstance(item, dict):
            raise ValueError(f"candidates[{index}] must be an object")
        candidate = SurveyCandidate(
            candidate_uid=str(item["candidate_uid"]).strip(),
            stand_id=str(item.get("stand_id", item["candidate_uid"])).strip(),
            x_m=float(item["x_m"]),
            y_m=float(item["y_m"]),
            keepout_radius_m=(
                None
                if item.get("keepout_radius_m") is None
                else float(item["keepout_radius_m"])
            ),
        )
        if not _SAFE_ID.fullmatch(candidate.candidate_uid):
            raise ValueError(
                "candidate_uid values must be safe identifiers containing only "
                "letters, digits, '.', '_', or '-'"
            )
        if not _SAFE_ID.fullmatch(candidate.stand_id):
            raise ValueError(f"candidates[{index}].stand_id is not a safe identifier")
        if candidate.candidate_uid in seen:
            raise ValueError("candidate_uid values must be unique")
        if not math.isfinite(candidate.x_m) or not math.isfinite(candidate.y_m):
            raise ValueError(f"candidates[{index}] coordinates must be finite")
        if candidate.keepout_radius_m is not None and (
            not math.isfinite(candidate.keepout_radius_m)
            or candidate.keepout_radius_m <= 0.0
        ):
            raise ValueError(
                f"candidates[{index}].keepout_radius_m must be finite and positive"
            )
        seen.add(candidate.candidate_uid)
        candidates.append(candidate)
    if not candidates:
        raise ValueError("at least one candidate is required")
    return tuple(candidates)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-json", required=True, type=Path)
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--world", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--catalog-id", default="sim_arrival_survey")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--map-frame", default="odom")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_footprint")
    parser.add_argument("--scan-frame", default="base_scan")
    parser.add_argument("--camera-frame", default="camera_link")
    parser.add_argument("--image-topic", default="/camera/image_raw")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--initial-start-x", type=float, default=0.0)
    parser.add_argument("--initial-start-y", type=float, default=0.0)
    parser.add_argument("--initial-start-yaw", type=float, default=0.0)
    parser.add_argument("--startup-timeout-sec", type=float, default=20.0)
    parser.add_argument("--candidate-timeout-sec", type=float, default=180.0)
    parser.add_argument(
        "--preflight-observation-window-sec",
        type=float,
        default=6.0,
        help="DDS/TF discovery window for each newly started simulation runner.",
    )
    parser.add_argument(
        "--initial-sensor-wait-sec",
        type=float,
        default=6.0,
        help="Follower startup window for scan, odometry, and TF discovery.",
    )
    parser.add_argument("--dynamic-route-refresh-sec", type=float, default=0.10)
    parser.add_argument(
        "--known-stand-keepout-radius-m",
        type=float,
        default=DEFAULT_KNOWN_STAND_KEEPOUT_RADIUS_M,
        help=(
            "Default total robot-center exclusion radius passed for every known "
            "stand; a candidate keepout_radius_m field overrides it."
        ),
    )
    parser.add_argument(
        "--allowed-cmd-vel-publisher",
        action="append",
        default=["/behavior_server", "/velocity_smoother"],
    )
    return parser


def _observer_command(args, candidate: SurveyCandidate, output: Path, stream_id: str):
    return [
        sys.executable,
        "scripts/aufgabe04/simulation/sim_synchronized_viewpoint_node.py",
        "--image-topic", args.image_topic,
        "--scan-topic", args.scan_topic,
        "--odom-topic", args.odom_topic,
        "--stand-x", str(candidate.x_m),
        "--stand-y", str(candidate.y_m),
        "--stand-id", candidate.stand_id,
        "--stream-id", stream_id,
        "--map-frame", args.map_frame,
        "--base-frame", args.base_frame,
        "--scan-frame", args.scan_frame,
        "--camera-frame", args.camera_frame,
        "--status-json", str(output / "observer_status.json"),
        "--recommended-pose-json", str(output / "recommendation.json"),
        "--observation-json", str(output / "camera_observation.json"),
        "--debug-dir", str(output / "perception_debug"),
    ]


def _planner_command(
    args,
    candidate: SurveyCandidate,
    candidates: tuple[SurveyCandidate, ...],
    output: Path,
    stream_id: str,
    world_sha256: str,
):
    command = [
        sys.executable,
        "scripts/aufgabe04/navigation/plan_synchronized_viewpoint.py",
        "--map", str(args.map),
        "--start-x", str(args.initial_start_x),
        "--start-y", str(args.initial_start_y),
        "--start-yaw", str(args.initial_start_yaw),
        "--recommended-pose-json", str(output / "recommendation.json"),
        "--route-csv", str(output / "survey_route.csv"),
        "--diagnostics-json", str(output / "survey_route_diagnostics.json"),
        "--route-manifest", str(output / "survey_route.manifest.json"),
        "--stream-id", stream_id,
        "--writer-id", f"planner-{stream_id}",
        "--workflow-mode", "survey-only",
        "--arrival-pose-catalog", str(args.catalog),
        "--catalog-id", args.catalog_id,
        "--candidate-uid", candidate.candidate_uid,
        "--world-id", args.world.stem,
        "--world-sha256", world_sha256,
        "--session-id", args.session_id,
        "--map-frame", args.map_frame,
        "--watch",
    ]
    for item in candidates:
        command.extend(["--expected-candidate-uid", item.candidate_uid])
        if item.candidate_uid == candidate.candidate_uid:
            # The current stand has target-specific body keepout and LiDAR
            # standoff validation inside plan_axis_acquisition / fixed arrival
            # validation.  Applying the larger non-target transit disk here
            # would rasterize valid 0.30 m viewpoint samples as obstacles.
            continue
        command.extend(
            [
                "--known-stand-keepout",
                str(item.x_m),
                str(item.y_m),
                str(
                    args.known_stand_keepout_radius_m
                    if item.keepout_radius_m is None
                    else item.keepout_radius_m
                ),
            ]
        )
    return command


def _runner_command(args, candidate: SurveyCandidate, output: Path):
    command = [
        sys.executable,
        "scripts/aufgabe04/navigation/run_single_station_segment.py",
        "--leg-index", "0",
        "--route-csv", str(output / "survey_route.csv"),
        "--diagnostics-json", str(output / "survey_route_diagnostics.json"),
        "--route-manifest", str(output / "survey_route.manifest.json"),
        "--run-id", f"survey_{args.session_id}_{candidate.candidate_uid}",
        "--scan-topic", args.scan_topic,
        "--odom-topic", args.odom_topic,
        "--cmd-vel-topic", args.cmd_vel_topic,
        "--localization-source", "tf",
        "--map-frame", args.map_frame,
        "--odom-frame", args.odom_frame,
        "--base-frame", args.base_frame,
        "--allow-sim-time",
        "--preflight-observation-window-sec",
        str(args.preflight_observation_window_sec),
        "--initial-sensor-wait-sec",
        str(args.initial_sensor_wait_sec),
        "--dynamic-route-refresh-sec", str(args.dynamic_route_refresh_sec),
        "--operator-note", f"survey-only candidate {candidate.candidate_uid}",
    ]
    for publisher in args.allowed_cmd_vel_publisher:
        command.extend(["--allowed-cmd-vel-publisher", publisher])
    return command


def _wait_for_route(
    manifest: Path,
    stream_id: str,
    timeout_sec: float,
    *,
    not_before_unix_sec: float | None = None,
) -> str:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            loaded = read_route_revision(
                manifest,
                expected_stream_id=stream_id,
                verify_artifacts=True,
            )
        except (OSError, RouteRevisionError):
            time.sleep(0.10)
            continue
        if (
            not_before_unix_sec is not None
            and float(loaded.manifest["published_unix_sec"]) < not_before_unix_sec
        ):
            # The output directory is intentionally resumable.  Never hand an
            # old active manifest to a newly launched follower, even when the
            # same session/candidate stream is retried.
            time.sleep(0.10)
            continue
        if loaded.status == "active":
            return "active"
        if loaded.status == "survey_complete":
            return "survey_complete"
        time.sleep(0.10)
    raise TimeoutError("timed out waiting for an active survey route")


def _terminate(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)


def _survey_completion_available(
    *,
    catalog_path: Path,
    provenance: CatalogProvenance,
    manifest_path: Path,
    stream_id: str,
    candidate_uid: str,
) -> bool:
    """Return true only when both durable hand-offs prove survey success."""

    try:
        catalog = load_arrival_pose_catalog(
            catalog_path,
            required_provenance=provenance,
        )
        terminal = read_route_revision(
            manifest_path,
            expected_stream_id=stream_id,
            verify_artifacts=False,
        )
    except (OSError, ValueError, RouteRevisionError):
        return False
    if catalog.record_for(candidate_uid) is None or terminal.status != "survey_complete":
        return False
    completion = terminal.manifest.get("completion")
    if not isinstance(completion, dict):
        return False
    return (
        completion.get("candidate_uid") == candidate_uid
        and completion.get("catalog_sha256")
        == arrival_pose_catalog_sha256(catalog)
    )


def _survey_one(
    args,
    candidate: SurveyCandidate,
    candidates: tuple[SurveyCandidate, ...],
    world_sha256: str,
    provenance: CatalogProvenance,
) -> None:
    output = args.output_dir / candidate.candidate_uid
    output.mkdir(parents=True, exist_ok=True)
    stream_id = _survey_stream_id(args.session_id, candidate.candidate_uid)
    observer_log = (output / "observer.log").open("w")
    planner_log = (output / "planner.log").open("w")
    observer = planner = runner = None
    try:
        launched_unix_sec = time.time()
        observer = subprocess.Popen(
            _observer_command(args, candidate, output, stream_id),
            stdout=observer_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        planner = subprocess.Popen(
            _planner_command(
                args, candidate, candidates, output, stream_id, world_sha256
            ),
            stdout=planner_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        state = _wait_for_route(
            output / "survey_route.manifest.json",
            stream_id,
            args.startup_timeout_sec,
            not_before_unix_sec=launched_unix_sec,
        )
        if state == "active":
            runner = subprocess.Popen(
                _runner_command(args, candidate, output),
            )
            returncode = runner.wait(timeout=args.candidate_timeout_sec)
            if returncode != 0 and not _survey_completion_available(
                catalog_path=args.catalog,
                provenance=provenance,
                manifest_path=output / "survey_route.manifest.json",
                stream_id=stream_id,
                candidate_uid=candidate.candidate_uid,
            ):
                raise RuntimeError(
                    f"survey follower failed for {candidate.candidate_uid}: "
                    f"exit {returncode}"
                )
        catalog = load_arrival_pose_catalog(
            args.catalog,
            required_provenance=provenance,
        )
        if catalog.record_for(candidate.candidate_uid) is None:
            raise RuntimeError(
                f"survey ended without catalog record for {candidate.candidate_uid}"
            )
    finally:
        # Stop the motion process first while its ROS context is still alive,
        # allowing the follower's finally block to publish repeated zero Twist.
        _terminate(runner)
        _terminate(planner)
        _terminate(observer)
        observer_log.close()
        planner_log.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        candidates = _load_candidates(args.candidates_json)
        for name, value in (
            ("catalog_id", args.catalog_id),
            ("session_id", args.session_id),
            ("world_id", args.world.stem),
        ):
            if not _SAFE_ID.fullmatch(value):
                raise ValueError(f"{name} is not a safe identifier: {value!r}")
        positive_values = {
            "startup_timeout_sec": args.startup_timeout_sec,
            "candidate_timeout_sec": args.candidate_timeout_sec,
            "preflight_observation_window_sec": (
                args.preflight_observation_window_sec
            ),
            "initial_sensor_wait_sec": args.initial_sensor_wait_sec,
            "known_stand_keepout_radius_m": args.known_stand_keepout_radius_m,
        }
        for name, value in positive_values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        map_sha256 = _file_sha256(args.map)
        world_sha256 = _file_sha256(args.world)
        provenance = _catalog_provenance(
            args,
            map_sha256=map_sha256,
            world_sha256=world_sha256,
        )
        for candidate in candidates:
            if args.catalog.exists():
                catalog = load_arrival_pose_catalog(
                    args.catalog,
                    required_provenance=provenance,
                )
                if catalog.record_for(candidate.candidate_uid) is not None:
                    print(f"Skipping already surveyed candidate {candidate.candidate_uid}")
                    continue
            print(f"Surveying {candidate.candidate_uid} at ({candidate.x_m}, {candidate.y_m})")
            _survey_one(
                args,
                candidate,
                candidates,
                world_sha256,
                provenance,
            )
        catalog = load_arrival_pose_catalog(
            args.catalog,
            required_provenance=provenance,
        )
        if not catalog.complete:
            unresolved = sorted(
                set(catalog.expected_candidate_uids)
                - set(catalog.resolved_candidate_uids)
            )
            raise RuntimeError(f"arrival survey incomplete: {unresolved}")
        print(
            json.dumps(
                {
                    "ok": True,
                    "catalog": str(args.catalog),
                    "catalog_revision": catalog.revision,
                    "candidate_count": len(catalog.records),
                    "complete": catalog.complete,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
