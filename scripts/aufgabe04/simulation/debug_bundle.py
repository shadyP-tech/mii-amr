"""Build a compact, model-readable debug bundle from an Aufgabe 04 simulation run.

The builder is ROS-free.  Live ROS capture is handled by
``debug_capture_node.py``; this module validates and merges the resulting
telemetry with existing semantic run events, copies perception artifacts, and
creates small visual summaries suitable for multimodal debugging.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PERCEPTION_GROUPS = (
    ("edges", ("edge",)),
    ("face_masks", ("face_mask", "facemask")),
    ("rectangles", ("rectangle", "quad")),
    ("head_roi", ("head_roi", "roi")),
    ("annotated", ("frame", "annotated")),
)


def validate_run_id(run_id: str) -> str:
    value = run_id.strip()
    if not value or value in {".", ".."}:
        raise ValueError("run ID must not be empty")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in value):
        raise ValueError("run ID may contain only letters, digits, '.', '_' and '-'")
    return value


def _timestamp_sec(record: Mapping[str, object]) -> float | None:
    for key in ("wall_time_sec", "time_sec", "timestamp_sec"):
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    value = record.get("timestamp")
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def read_jsonl(path: Path | None, *, source: str) -> tuple[list[dict[str, object]], list[str]]:
    if path is None:
        return [], []
    path = Path(path)
    if not path.is_file():
        return [], [f"missing {source} JSONL: {path}"]
    records: list[dict[str, object]] = []
    warnings: list[str] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            warnings.append(f"invalid {source} JSONL line {line_number}: {exc.msg}")
            continue
        if not isinstance(payload, dict):
            warnings.append(f"ignored non-object {source} JSONL line {line_number}")
            continue
        payload = dict(payload)
        payload.setdefault("source", source)
        records.append(payload)
    return records, warnings


def merge_timeline(
    telemetry_records: Sequence[Mapping[str, object]],
    semantic_records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    sortable: list[tuple[float, int, dict[str, object]]] = []
    unknown_time_base = 10**20
    for insertion_index, original in enumerate((*telemetry_records, *semantic_records)):
        record = dict(original)
        timestamp = _timestamp_sec(record)
        sortable.append((unknown_time_base + insertion_index if timestamp is None else timestamp, insertion_index, record))
    sortable.sort(key=lambda item: (item[0], item[1]))
    known_times = [item[0] for item in sortable if item[0] < unknown_time_base]
    origin = min(known_times) if known_times else None
    timeline: list[dict[str, object]] = []
    for sequence, (sort_time, _insertion_index, record) in enumerate(sortable):
        record["sequence"] = sequence
        if origin is not None and sort_time < unknown_time_base:
            record["relative_time_sec"] = round(sort_time - origin, 6)
        timeline.append(record)
    return timeline


def detect_telemetry_events(
    telemetry_records: Sequence[Mapping[str, object]],
    *,
    obstacle_threshold_m: float = 0.18,
) -> list[dict[str, object]]:
    """Derive conservative observations without replacing controller safety logic."""

    derived: list[dict[str, object]] = []
    obstacle_active = False
    previous_sign = 0
    sign_changes: list[float] = []
    motion_window: list[tuple[float, float, float]] = []
    stuck_active = False
    for record in telemetry_records:
        timestamp = _timestamp_sec(record)
        nearest = record.get("nearest_obstacle_m")
        is_near = isinstance(nearest, (int, float)) and float(nearest) < obstacle_threshold_m
        if is_near and not obstacle_active:
            derived.append(
                {
                    "source": "derived",
                    "event": "obstacle_threshold_crossed",
                    "wall_time_sec": timestamp,
                    "nearest_obstacle_m": float(nearest),
                    "threshold_m": obstacle_threshold_m,
                }
            )
        obstacle_active = bool(is_near)

        command = record.get("command")
        pose = _pose_xy(record)
        linear = command.get("linear_x") if isinstance(command, Mapping) else None
        if (
            timestamp is not None
            and pose is not None
            and isinstance(linear, (int, float))
            and abs(float(linear)) >= 0.02
        ):
            motion_window.append((timestamp, pose[0], pose[1]))
            motion_window = [sample for sample in motion_window if timestamp - sample[0] <= 3.0]
            if motion_window and timestamp - motion_window[0][0] >= 2.5:
                displacement = math.hypot(pose[0] - motion_window[0][1], pose[1] - motion_window[0][2])
                if displacement < 0.02 and not stuck_active:
                    derived.append(
                        {
                            "source": "derived",
                            "event": "no_progress_candidate",
                            "wall_time_sec": timestamp,
                            "window_sec": timestamp - motion_window[0][0],
                            "displacement_m": displacement,
                        }
                    )
                    stuck_active = True
                elif displacement >= 0.02:
                    stuck_active = False
        else:
            motion_window.clear()
            stuck_active = False

        angular = command.get("angular_z") if isinstance(command, Mapping) else None
        if not isinstance(angular, (int, float)) or abs(float(angular)) < 0.03 or timestamp is None:
            continue
        sign = 1 if float(angular) > 0 else -1
        if previous_sign and sign != previous_sign:
            sign_changes.append(timestamp)
            sign_changes = [value for value in sign_changes if timestamp - value <= 3.0]
            if len(sign_changes) == 4:
                derived.append(
                    {
                        "source": "derived",
                        "event": "angular_oscillation_candidate",
                        "wall_time_sec": timestamp,
                        "sign_changes_in_3_sec": len(sign_changes),
                    }
                )
        previous_sign = sign
    return derived


def _perception_group(path: Path) -> str:
    name = path.stem.lower()
    for group, markers in PERCEPTION_GROUPS:
        if any(marker in name for marker in markers):
            return group
    return "other"


def copy_perception_artifacts(
    source_dirs: Iterable[Path], bundle_dir: Path
) -> tuple[list[Path], list[str]]:
    copied: list[Path] = []
    warnings: list[str] = []
    seen_names: set[str] = set()
    for source_dir in source_dirs:
        source_dir = Path(source_dir)
        if not source_dir.is_dir():
            warnings.append(f"missing perception directory: {source_dir}")
            continue
        for source in sorted(source_dir.rglob("*")):
            if not source.is_file() or source.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            group = _perception_group(source)
            target_dir = bundle_dir / "perception" / group
            target_dir.mkdir(parents=True, exist_ok=True)
            candidate = source.name
            counter = 2
            while candidate in seen_names:
                candidate = f"{source.stem}_{counter}{source.suffix.lower()}"
                counter += 1
            seen_names.add(candidate)
            target = target_dir / candidate
            shutil.copy2(source, target)
            copied.append(target)
    return copied, warnings


def _load_cv2():
    try:
        import cv2
        import numpy
    except ImportError:
        return None, None
    return cv2, numpy


def build_contact_sheet(image_paths: Sequence[Path], output_path: Path, *, limit: int = 25) -> bool:
    cv2, numpy = _load_cv2()
    if cv2 is None or not image_paths:
        return False
    paths = list(image_paths)
    if len(paths) <= limit:
        selected = paths
    elif limit == 1:
        selected = [paths[0]]
    else:
        selected = [paths[round(index * (len(paths) - 1) / (limit - 1))] for index in range(limit)]
    tiles = []
    for path in selected:
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        scale = min(320.0 / max(width, 1), 220.0 / max(height, 1))
        resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))
        tile = numpy.full((260, 340, 3), 245, dtype=numpy.uint8)
        y = 10 + (220 - resized.shape[0]) // 2
        x = 10 + (320 - resized.shape[1]) // 2
        tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        cv2.putText(tile, path.name[:42], (8, 248), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1)
        tiles.append(tile)
    if not tiles:
        return False
    columns = min(4, len(tiles))
    rows = math.ceil(len(tiles) / columns)
    blank = numpy.full_like(tiles[0], 245)
    while len(tiles) < rows * columns:
        tiles.append(blank.copy())
    sheet = cv2.vconcat([cv2.hconcat(tiles[row * columns : (row + 1) * columns]) for row in range(rows)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), sheet))


def _pose_xy(record: Mapping[str, object]) -> tuple[float, float] | None:
    pose = record.get("ground_truth_pose") or record.get("pose")
    if not isinstance(pose, Mapping):
        return None
    x = pose.get("x_m", pose.get("x"))
    y = pose.get("y_m", pose.get("y"))
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    return float(x), float(y)


def build_plots(telemetry_records: Sequence[Mapping[str, object]], output_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ImportError:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    poses = [pose for record in telemetry_records if (pose := _pose_xy(record)) is not None]
    if poses:
        figure, axis = pyplot.subplots(figsize=(6, 5))
        axis.plot([pose[0] for pose in poses], [pose[1] for pose in poses], "-", linewidth=1.4)
        axis.scatter([poses[0][0]], [poses[0][1]], label="start", marker="o")
        axis.scatter([poses[-1][0]], [poses[-1][1]], label="final", marker="x")
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_title("Simulation trajectory")
        axis.grid(True, alpha=0.3)
        axis.legend()
        path = output_dir / "trajectory.png"
        figure.tight_layout()
        figure.savefig(path, dpi=140)
        pyplot.close(figure)
        outputs.append(path)

    samples = []
    for record in telemetry_records:
        command = record.get("command")
        timestamp = _timestamp_sec(record)
        if not isinstance(command, Mapping) or timestamp is None:
            continue
        linear = command.get("linear_x")
        angular = command.get("angular_z")
        if isinstance(linear, (int, float)) and isinstance(angular, (int, float)):
            samples.append((timestamp, float(linear), float(angular)))
    if samples:
        origin = samples[0][0]
        figure, axis = pyplot.subplots(figsize=(7, 4))
        axis.plot([item[0] - origin for item in samples], [item[1] for item in samples], label="linear x [m/s]")
        axis.plot([item[0] - origin for item in samples], [item[2] for item in samples], label="angular z [rad/s]")
        axis.set_xlabel("time [s]")
        axis.set_title("Commanded velocity")
        axis.grid(True, alpha=0.3)
        axis.legend()
        path = output_dir / "velocity.png"
        figure.tight_layout()
        figure.savefig(path, dpi=140)
        pyplot.close(figure)
        outputs.append(path)
    return outputs


def _git_value(arguments: Sequence[str]) -> str:
    try:
        return subprocess.run(arguments, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_bundle(
    *,
    bundle_dir: Path,
    run_id: str,
    telemetry_jsonl: Path | None,
    semantic_jsonl: Path | None,
    perception_dirs: Sequence[Path],
    expected_behavior: str,
    observed_behavior: str,
    world: str,
    command_exit_code: int | None,
    bag_path: Path | None,
) -> dict[str, object]:
    run_id = validate_run_id(run_id)
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    telemetry, warnings = read_jsonl(telemetry_jsonl, source="telemetry")
    semantics, semantic_warnings = read_jsonl(semantic_jsonl, source="semantic")
    warnings.extend(semantic_warnings)
    derived = detect_telemetry_events(telemetry)
    timeline = merge_timeline(telemetry, [*semantics, *derived])
    timeline_path = bundle_dir / "timeline.jsonl"
    timeline_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in timeline))

    copied, copy_warnings = copy_perception_artifacts(perception_dirs, bundle_dir)
    warnings.extend(copy_warnings)
    captured_frames = sorted((bundle_dir / "frames").rglob("*.jpg")) if (bundle_dir / "frames").exists() else []
    sheet_inputs = [*captured_frames, *copied]
    contact_sheet = bundle_dir / "frames" / "contact_sheet.jpg"
    contact_sheet_created = build_contact_sheet(sheet_inputs, contact_sheet)
    plots = build_plots(telemetry, bundle_dir / "plots")

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "world": world,
        "expected_behavior": expected_behavior,
        "observed_behavior": observed_behavior,
        "command_exit_code": command_exit_code,
        "command": (bundle_dir / "command.txt").read_text().strip()
        if (bundle_dir / "command.txt").is_file()
        else "",
        "git_commit": _git_value(["git", "rev-parse", "HEAD"]),
        "git_status_short": _git_value(["git", "status", "--short"]),
        "telemetry_jsonl": str(telemetry_jsonl) if telemetry_jsonl else "",
        "semantic_jsonl": str(semantic_jsonl) if semantic_jsonl else "",
        "bag_path": str(bag_path) if bag_path else "",
        "telemetry_record_count": len(telemetry),
        "semantic_event_count": len(semantics),
        "derived_event_count": len(derived),
        "timeline_record_count": len(timeline),
        "perception_image_count": len(copied),
        "captured_frame_count": len(captured_frames),
        "contact_sheet_created": contact_sheet_created,
        "plot_files": [str(path.relative_to(bundle_dir)) for path in plots],
        "warnings": warnings,
    }
    _write_json(bundle_dir / "manifest.json", manifest)

    event_names = [str(record.get("event")) for record in timeline if record.get("event")]
    summary_lines = [
        f"# Simulation debug bundle: {run_id}",
        "",
        "## Debug question",
        "",
        f"Expected behavior: {expected_behavior or 'not supplied'}",
        "",
        f"Observed behavior: {observed_behavior or 'not supplied'}",
        "",
        "Identify the first divergence, likely root cause, supporting evidence, alternatives, and the next discriminating test. Prefer telemetry or Gazebo ground truth over estimating precise motion from images.",
        "",
        "## Evidence index",
        "",
        f"- Timeline: `timeline.jsonl` ({len(timeline)} records)",
        f"- Raw ROS bag: `{bag_path}`" if bag_path else "- Raw ROS bag: not recorded",
        f"- Contact sheet: `frames/contact_sheet.jpg`" if contact_sheet_created else "- Contact sheet: unavailable",
        f"- Plots: {', '.join(f'`{path.relative_to(bundle_dir)}`' for path in plots) or 'unavailable'}",
        f"- Perception images: {len(copied)}",
        "",
        "## Event sequence",
        "",
        ", ".join(event_names) if event_names else "No semantic or derived events were available.",
    ]
    if warnings:
        summary_lines.extend(["", "## Capture warnings", "", *[f"- {warning}" for warning in warnings]])
    (bundle_dir / "summary.md").write_text("\n".join(summary_lines) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--telemetry-jsonl", type=Path)
    parser.add_argument("--semantic-jsonl", type=Path)
    parser.add_argument("--perception-dir", type=Path, action="append", default=[])
    parser.add_argument("--expected-behavior", default="")
    parser.add_argument("--observed-behavior", default="")
    parser.add_argument("--world", default="")
    parser.add_argument("--command-exit-code", type=int)
    parser.add_argument("--bag-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = build_bundle(
            bundle_dir=args.bundle_dir,
            run_id=args.run_id,
            telemetry_jsonl=args.telemetry_jsonl,
            semantic_jsonl=args.semantic_jsonl,
            perception_dirs=args.perception_dir,
            expected_behavior=args.expected_behavior,
            observed_behavior=args.observed_behavior,
            world=args.world,
            command_exit_code=args.command_exit_code,
            bag_path=args.bag_path,
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(args.bundle_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
