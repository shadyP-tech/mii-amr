"""CSV evidence logging for Aufgabe 04 QR scans."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping


QR_SCAN_HEADER = [
    "timestamp",
    "run_id",
    "robot_id",
    "raw_text",
    "qr_id",
    "resolved_station_id",
    "source",
    "confidence",
    "status",
    "reason",
]


def append_qr_scan(path: Path, row: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=QR_SCAN_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in QR_SCAN_HEADER})

