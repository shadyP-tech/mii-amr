"""CSV logging helpers for Aufgabe 04 mission evidence."""

import csv
from pathlib import Path
from typing import Iterable, Mapping


MISSION_RUNS_HEADER = [
    "timestamp",
    "run_id",
    "robot_id",
    "phase",
    "station_id",
    "status",
    "reason",
    "artifact_path",
]


def write_rows(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MISSION_RUNS_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in MISSION_RUNS_HEADER})

