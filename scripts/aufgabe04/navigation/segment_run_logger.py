"""Append-only station segment run logging."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping


SEGMENT_RUN_HEADER = [
    "timestamp",
    "run_id",
    "robot_id",
    "namespace",
    "configured_cmd_vel_topic",
    "resolved_cmd_vel_topic",
    "configured_scan_topic",
    "resolved_scan_topic",
    "configured_odom_topic",
    "resolved_odom_topic",
    "map_frame",
    "odom_frame",
    "base_frame",
    "leg_index",
    "raw_point_count",
    "executable_point_count",
    "route_length_m",
    "preflight_ok",
    "status",
    "stop_reason",
    "duration_sec",
    "distance_estimate_m",
    "motion_published",
    "operator_note",
]


def append_segment_run(path: Path, row: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SEGMENT_RUN_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in SEGMENT_RUN_HEADER})
