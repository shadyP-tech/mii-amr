#!/usr/bin/env python3
"""
Print the next sequential real-run ID.

The helper scans both the real-run result CSV and bag directory.  This avoids
reusing an ID when a run created a bag but failed before appending results.
"""

import argparse
import csv
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "results" / "real_scripted_drive_runs.csv"
DEFAULT_BAGS_DIR = PROJECT_ROOT / "bags" / "real"
DEFAULT_PREFIX = "run_real_"
DEFAULT_WIDTH = 3


def parse_run_number(run_id, prefix=DEFAULT_PREFIX):
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    match = pattern.match(run_id)
    if match is None:
        return None

    digits = match.group(1)
    return int(digits), len(digits)


def collect_existing_run_ids(results_csv=DEFAULT_RESULTS_CSV, bags_dir=DEFAULT_BAGS_DIR):
    run_ids = []

    results_csv = Path(results_csv)
    if results_csv.exists():
        with results_csv.open(newline="") as f:
            for row in csv.DictReader(f):
                run_id = row.get("run_id")
                if run_id:
                    run_ids.append(run_id)

    bags_dir = Path(bags_dir)
    if bags_dir.exists():
        for path in bags_dir.iterdir():
            run_ids.append(path.name)

    return run_ids


def next_run_id(run_ids, prefix=DEFAULT_PREFIX, min_width=DEFAULT_WIDTH):
    max_number = 0
    width = min_width

    for run_id in run_ids:
        parsed = parse_run_number(run_id, prefix=prefix)
        if parsed is None:
            continue

        number, digits = parsed
        max_number = max(max_number, number)
        width = max(width, digits)

    return f"{prefix}{max_number + 1:0{width}d}"


def main():
    parser = argparse.ArgumentParser(description="Print the next real-run ID.")
    parser.add_argument(
        "--results-csv",
        default=DEFAULT_RESULTS_CSV,
        type=Path,
        help="Path to real_scripted_drive_runs.csv.",
    )
    parser.add_argument(
        "--bags-dir",
        default=DEFAULT_BAGS_DIR,
        type=Path,
        help="Directory containing real-run bag directories.",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Run ID prefix before the numeric suffix.",
    )
    parser.add_argument(
        "--width",
        default=DEFAULT_WIDTH,
        type=int,
        help="Minimum zero-padding width for the numeric suffix.",
    )
    args = parser.parse_args()

    run_ids = collect_existing_run_ids(
        results_csv=args.results_csv,
        bags_dir=args.bags_dir,
    )
    print(next_run_id(run_ids, prefix=args.prefix, min_width=args.width))


if __name__ == "__main__":
    main()
