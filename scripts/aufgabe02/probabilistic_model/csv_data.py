import csv
import math
import re
from pathlib import Path

from .errors import DataError


def parse_run_number(run_id):
    match = re.search(r"(\d+)$", str(run_id or ""))
    if match is None:
        return None
    return int(match.group(1))


def parse_run_range(text):
    if text is None or text == "":
        return None

    match = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", text)
    if match is None:
        raise ValueError("Run range must use START:END, for example 21:50")

    start = int(match.group(1))
    end = int(match.group(2))
    if start > end:
        raise ValueError("Run range start must be <= end")

    return start, end


def read_csv_rows(path):
    path = Path(path)
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        rows = []
        for line_number, row in enumerate(reader, start=2):
            copied = dict(row)
            copied["_row_number"] = line_number
            rows.append(copied)

    return fieldnames, rows


def require_columns(fieldnames, columns, csv_path):
    missing = [column for column in columns if column not in fieldnames]
    if missing:
        raise DataError(
            f"{csv_path} is missing required column(s): {', '.join(missing)}"
        )


def filter_rows_by_run_range(rows, run_range_text):
    run_range = parse_run_range(run_range_text)
    if run_range is None:
        return list(rows)

    start, end = run_range
    selected = []
    for row in rows:
        number = parse_run_number(row.get("run_id"))
        if number is not None and start <= number <= end:
            selected.append(row)

    return selected


def filter_latest_rows(rows, count):
    if count is None:
        return list(rows)
    if count <= 0:
        raise ValueError("--sim-last-n must be a positive integer")
    return list(rows[-count:])


def finite_float(row, column):
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{column} is missing or not numeric")

    if not math.isfinite(value):
        raise ValueError(f"{column} is not finite")

    return value


def valid_rows_with_columns(rows, columns):
    valid = []
    skipped = []

    for row in rows:
        try:
            for column in columns:
                if column == "run_id":
                    if not row.get("run_id"):
                        raise ValueError("run_id is missing")
                else:
                    finite_float(row, column)
        except ValueError as exc:
            skipped.append(skip_record(row, str(exc)))
            continue
        valid.append(row)

    return valid, skipped


def skip_record(row, reason):
    return {
        "row_number": int(row.get("_row_number", 0) or 0),
        "run_id": row.get("run_id", ""),
        "reason": reason,
    }


def extract_points(rows, x_column, y_column):
    return [
        [finite_float(row, x_column), finite_float(row, y_column)]
        for row in rows
    ]


def extract_yaws(rows, yaw_column):
    return [finite_float(row, yaw_column) for row in rows]


def run_ids(rows):
    return [row.get("run_id", "") for row in rows]
