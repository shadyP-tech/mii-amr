"""Compact console rendering for station-segment execution.

Full-fidelity runtime and preflight evidence belongs in JSON/JSONL artifacts.
These helpers keep the operator terminal focused on run identity, resolved ROS
edges, artifact locations, and actionable failures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence


def compact_runtime_summary(
    *,
    run_id: str,
    leg_index: int,
    runtime_config: Mapping[str, object],
    raw_waypoint_count: int,
    executable_waypoint_count: int,
    route_length_m: float,
    semantic_log_path: Path,
    results_csv_path: Path,
    preflight_json_path: Path | None,
) -> str:
    """Render the essential immutable setup in four scan-friendly lines."""

    namespace = str(runtime_config.get("namespace", "")) or "/"
    localization_source = str(runtime_config.get("localization_source", ""))
    map_frame = str(runtime_config.get("map_frame", ""))
    odom_frame = str(runtime_config.get("odom_frame", ""))
    base_frame = str(runtime_config.get("base_frame", ""))
    preflight_artifact = (
        str(preflight_json_path)
        if preflight_json_path is not None
        else "not configured (use --verbose-console for full JSON)"
    )
    return "\n".join(
        (
            (
                f"Run setup: {run_id} | leg={leg_index} | "
                f"waypoints={raw_waypoint_count}->{executable_waypoint_count} | "
                f"length={route_length_m:.3f} m"
            ),
            (
                f"ROS: namespace={namespace} | localization={localization_source} | "
                f"frames={map_frame}->{odom_frame}->{base_frame}"
            ),
            (
                "Topics: "
                f"cmd_vel={runtime_config.get('cmd_vel_topic', '')} | "
                f"scan={runtime_config.get('scan_topic', '')} | "
                f"odom={runtime_config.get('odom_topic', '')} | "
                f"amcl={runtime_config.get('amcl_topic', '')}"
            ),
            (
                f"Artifacts: preflight={preflight_artifact} | "
                f"events={semantic_log_path} | results={results_csv_path}"
            ),
        )
    )


def compact_preflight_summary(
    *,
    ok: bool,
    failures: Sequence[str],
    observation_count: int,
    preflight_json_path: Path | None,
) -> str:
    """Render one status line plus every actionable preflight failure."""

    status = "PASS" if ok else "FAIL"
    lines = [
        (
            f"Preflight: {status} | observations={observation_count} | "
            f"failures={len(failures)}"
        )
    ]
    lines.extend(f"  - {failure}" for failure in failures)
    if preflight_json_path is not None:
        lines.append(f"Full preflight JSON: {preflight_json_path}")
    elif not ok:
        lines.append(
            "Full preflight JSON was not configured; rerun with --preflight-json."
        )
    return "\n".join(lines)
