from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .models import Pose2D, ScanSample


def write_json(path: Path | str, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def load_scan_samples_json(path: Path | str):
    path = Path(path)
    with path.open() as file:
        data = json.load(file)
    samples = []
    for row in data.get("scan_samples", data if isinstance(data, list) else []):
        odom = row.get("odom_pose")
        samples.append(
            ScanSample(
                ranges=row["ranges"],
                angle_min=float(row["angle_min"]),
                angle_increment=float(row["angle_increment"]),
                range_min=float(row.get("range_min", 0.0)),
                range_max=float(row.get("range_max", float("inf"))),
                odom_pose=(
                    None
                    if odom is None
                    else Pose2D(
                        float(odom.get("x", 0.0)),
                        float(odom.get("y", 0.0)),
                        float(odom.get("yaw_deg", 0.0)),
                    )
                ),
            )
        )
    return samples


def iter_points(points: Iterable[tuple[float, float]]):
    for x, y in points:
        yield float(x), float(y)
