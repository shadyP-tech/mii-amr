#!/usr/bin/env python3
"""
Summarize Aufgabe 03 waypoint follower runs.

This is a ROS-free reporting helper. It reads the follower CSV log and reports
which obstacle/remapping runs are usable evidence for the Aufgabe 03 write-up.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT = Path("results/aufgabe03/aufgabe03_waypoint_follow_runs.csv")
DEFAULT_MAX_REPLANS = 2

FAIL_STATUSES = {"failed", "blocked", "timeout", "interrupted"}
FAILURE_REASON_FIELDS = (
    "final_status_reason",
    "last_replan_reason",
    "run_local_no_path_reason",
    "notes",
)
NEW_DIAGNOSTIC_COLUMNS = (
    "replan_count",
    "last_replan_reason",
    "run_local_replan_count",
    "run_local_map_yaml",
    "run_local_waypoints_csv",
)
FAILURE_TOKENS = (
    "lidar_replan_failed:",
    "stale_scan",
    "keyboard_interrupt",
)


@dataclass(frozen=True)
class RunVerdict:
    verdict: str
    reason: str


@dataclass(frozen=True)
class ReportSummary:
    total_runs: int
    verdict_counts: Counter
    failure_reason_counts: Counter
    max_replans: int


def empty_to_none(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return text


def parse_int_or_none(value):
    text = empty_to_none(value)
    if text is None:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_float_or_none(value):
    text = empty_to_none(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def has_failure_token(value):
    text = empty_to_none(value)
    if text is None:
        return False
    lowered = text.lower()
    return any(token in lowered for token in FAILURE_TOKENS)


def first_failure_reason(row):
    for field in FAILURE_REASON_FIELDS:
        value = empty_to_none(row.get(field))
        if value and has_failure_token(value):
            return value
    for field in FAILURE_REASON_FIELDS:
        value = empty_to_none(row.get(field))
        if value:
            return value
    return "status=" + str(row.get("status", "")).strip()


def artifact_value(row, primary, fallback):
    return empty_to_none(row.get(primary)) or empty_to_none(row.get(fallback))


def classify_run(row, max_replans):
    status = (empty_to_none(row.get("status")) or "").lower()
    if status in FAIL_STATUSES or any(
        has_failure_token(row.get(field)) for field in FAILURE_REASON_FIELDS
    ):
        return RunVerdict("FAIL", first_failure_reason(row))

    if status != "completed":
        return RunVerdict("WARN", f"unexpected status={status or 'missing'}")

    warnings = []
    missing_columns = [field for field in NEW_DIAGNOSTIC_COLUMNS if field not in row]
    if missing_columns:
        warnings.append("missing diagnostic columns: " + ", ".join(missing_columns))

    replan_count = parse_int_or_none(row.get("replan_count"))
    if "replan_count" in row and replan_count is None:
        warnings.append("blank replan_count")
    elif replan_count is not None and replan_count > max_replans:
        warnings.append(f"replan_count={replan_count} exceeds max={max_replans}")

    run_local_replan_count = parse_int_or_none(row.get("run_local_replan_count"))
    if "run_local_replan_count" in row and run_local_replan_count is None:
        warnings.append("blank run_local_replan_count")

    if not artifact_value(row, "run_local_map_yaml", "updated_map_yaml"):
        warnings.append("missing map artifact")
    if not artifact_value(row, "run_local_waypoints_csv", "updated_waypoints_csv"):
        warnings.append("missing waypoint artifact")
    if (
        parse_float_or_none(row.get("min_scan_range_m")) is None
        or parse_float_or_none(row.get("p05_scan_range_m")) is None
    ):
        warnings.append("missing scan stats")

    if warnings:
        return RunVerdict("WARN", "; ".join(warnings))
    return RunVerdict("PASS", "completed")


def summarize_runs(rows, max_replans=DEFAULT_MAX_REPLANS):
    verdict_counts = Counter()
    failure_reason_counts = Counter()
    for row in rows:
        verdict = classify_run(row, max_replans)
        verdict_counts[verdict.verdict] += 1
        if verdict.verdict == "FAIL":
            for field in FAILURE_REASON_FIELDS:
                value = empty_to_none(row.get(field))
                if value:
                    failure_reason_counts[f"{field}: {value}"] += 1
            if not any(empty_to_none(row.get(field)) for field in FAILURE_REASON_FIELDS):
                failure_reason_counts[verdict.reason] += 1
    return ReportSummary(
        total_runs=len(rows),
        verdict_counts=verdict_counts,
        failure_reason_counts=failure_reason_counts,
        max_replans=max_replans,
    )


def compact(value, width=42):
    text = empty_to_none(value)
    if text is None:
        return "-"
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..."


def format_float(value):
    number = parse_float_or_none(value)
    if number is None:
        return "-"
    return f"{number:.3f}"


def format_table(headers, rows):
    table = [headers] + rows
    widths = [
        max(len(str(row[index])) for row in table)
        for index in range(len(headers))
    ]
    lines = []
    lines.append(
        "  ".join(
            str(cell).ljust(widths[index])
            for index, cell in enumerate(headers)
        )
    )
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append(
            "  ".join(
                str(cell).ljust(widths[index])
                for index, cell in enumerate(row)
            )
        )
    return "\n".join(lines)


def run_table_rows(rows, max_replans):
    rendered = []
    for row in rows:
        verdict = classify_run(row, max_replans)
        rendered.append([
            compact(row.get("timestamp"), 19),
            compact(row.get("run_id"), 28),
            compact(row.get("status"), 11),
            verdict.verdict,
            compact(row.get("replan_count"), 7),
            compact(row.get("run_local_replan_count"), 9),
            compact(verdict.reason, 46),
            f"{format_float(row.get('min_scan_range_m'))}/{format_float(row.get('p05_scan_range_m'))}",
            compact(artifact_value(row, "run_local_map_yaml", "updated_map_yaml"), 34),
            compact(artifact_value(row, "run_local_waypoints_csv", "updated_waypoints_csv"), 34),
        ])
    return rendered


def render_text_report(summary, rows):
    lines = [
        "Aufgabe 03 waypoint follower run report",
        f"runs: {summary.total_runs}",
        (
            "verdicts: "
            f"PASS={summary.verdict_counts.get('PASS', 0)}, "
            f"WARN={summary.verdict_counts.get('WARN', 0)}, "
            f"FAIL={summary.verdict_counts.get('FAIL', 0)}"
        ),
        f"max replans threshold: {summary.max_replans}",
        "",
    ]
    if rows:
        headers = [
            "timestamp",
            "run_id",
            "status",
            "verdict",
            "replan",
            "run_local",
            "reason",
            "scan min/p05",
            "map artifact",
            "waypoint artifact",
        ]
        lines.append(format_table(headers, run_table_rows(rows, summary.max_replans)))
    else:
        lines.append("No rows matched the selected filters.")

    lines.extend(["", "Failure reason counts:"])
    if summary.failure_reason_counts:
        failure_rows = [
            [str(count), compact(reason, 100)]
            for reason, count in summary.failure_reason_counts.most_common()
        ]
        lines.append(format_table(["count", "reason"], failure_rows))
    else:
        lines.append("-")
    return "\n".join(lines)


def markdown_escape(value):
    text = empty_to_none(value) or "-"
    return text.replace("|", "\\|").replace("\n", " ")


def markdown_row(values):
    return "| " + " | ".join(markdown_escape(value) for value in values) + " |"


def render_markdown_report(summary, rows):
    lines = [
        "# Aufgabe 03 Waypoint Follow Run Report",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| Runs | {summary.total_runs} |",
        f"| PASS | {summary.verdict_counts.get('PASS', 0)} |",
        f"| WARN | {summary.verdict_counts.get('WARN', 0)} |",
        f"| FAIL | {summary.verdict_counts.get('FAIL', 0)} |",
        f"| Max replan threshold | {summary.max_replans} |",
        "",
        "## Runs",
        "",
        markdown_row([
            "timestamp",
            "run_id",
            "status",
            "verdict",
            "replan",
            "run_local",
            "reason",
            "scan min/p05",
            "map artifact",
            "waypoint artifact",
        ]),
        markdown_row([
            "---",
            "---",
            "---",
            "---",
            "---:",
            "---:",
            "---",
            "---",
            "---",
            "---",
        ]),
    ]
    for row in run_table_rows(rows, summary.max_replans):
        lines.append(markdown_row(row))
    if not rows:
        lines.append(markdown_row([
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "No rows matched the selected filters.",
            "-",
            "-",
            "-",
        ]))

    lines.extend(["", "## Failure Reason Counts", ""])
    lines.append(markdown_row(["count", "reason"]))
    lines.append(markdown_row(["---:", "---"]))
    if summary.failure_reason_counts:
        for reason, count in summary.failure_reason_counts.most_common():
            lines.append(markdown_row([str(count), reason]))
    else:
        lines.append(markdown_row(["0", "-"]))
    return "\n".join(lines) + "\n"


def load_rows(path):
    with Path(path).open(newline="") as file:
        return list(csv.DictReader(file))


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Analyze Aufgabe 03 waypoint follower run logs.",
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, type=Path)
    parser.add_argument("--run-id", help="Only include rows with this exact run_id.")
    parser.add_argument(
        "--latest",
        type=int,
        help="Keep only the newest N rows after filtering.",
    )
    parser.add_argument("--max-replans", default=DEFAULT_MAX_REPLANS, type=int)
    parser.add_argument("--output-md", type=Path, help="Optional Markdown report path.")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.latest is not None and args.latest < 1:
        parser.error("--latest must be >= 1")
    if args.max_replans < 0:
        parser.error("--max-replans must be >= 0")

    rows = load_rows(args.input)
    if args.run_id:
        rows = [row for row in rows if row.get("run_id") == args.run_id]
    if args.latest is not None:
        rows = rows[-args.latest:]

    summary = summarize_runs(rows, max_replans=args.max_replans)
    report = render_text_report(summary, rows)
    print(report)

    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown_report(summary, rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
