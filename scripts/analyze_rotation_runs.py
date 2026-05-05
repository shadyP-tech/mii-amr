#!/usr/bin/env python3
"""
Analyze repeated real in-place rotation runs.

The tracker yaw currently uses the opposite sign from ROS/odom for clockwise
turns, so the default analysis applies a -1 sign correction before computing
tracker yaw statistics.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path


DEFAULT_ROTATION_CSV = "results/real_rotation_runs.csv"
DEFAULT_RUN_RANGE = "18:32"
DEFAULT_TRACKER_YAW_SIGN = -1.0

REQUIRED_COLUMNS = [
    "run_id",
    "command_angle_deg",
    "tracker_start_yaw_deg",
    "tracker_yaw_change_deg",
    "tracker_dx",
    "tracker_dy",
    "tracker_position_drift_m",
    "odom_yaw_change_deg",
    "odom_yaw_error_deg",
    "odom_position_drift_m",
]


class DataError(ValueError):
    """Raised when rotation CSV data is missing or unusable."""


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
        raise ValueError("Run range must use START:END, for example 18:32")

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


def filter_rows(rows, run_range_text=None, run_id_prefix=None):
    run_range = parse_run_range(run_range_text)
    selected = []

    for row in rows:
        run_id = row.get("run_id", "")
        if run_id_prefix and not run_id.startswith(run_id_prefix):
            continue

        number = parse_run_number(row.get("run_id"))
        if run_range is not None:
            start, end = run_range
            if number is None or number < start or number > end:
                continue

        selected.append(row)

    return selected


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
            if not row.get("run_id"):
                raise ValueError("run_id is missing")
            for column in columns:
                if column != "run_id":
                    finite_float(row, column)
        except ValueError as exc:
            skipped.append(
                {
                    "row_number": int(row.get("_row_number", 0) or 0),
                    "run_id": row.get("run_id", ""),
                    "reason": str(exc),
                }
            )
            continue
        valid.append(row)

    return valid, skipped


def shortest_angle_delta_deg(start_deg, end_deg):
    return (end_deg - start_deg + 180.0) % 360.0 - 180.0


def sample_mean(values):
    values = list(values)
    if not values:
        raise ValueError("At least one value is required")
    return sum(values) / len(values)


def sample_std(values):
    values = list(values)
    if len(values) < 2:
        return 0.0
    mean = sample_mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def value_summary(values):
    values = list(values)
    return {
        "mean": sample_mean(values),
        "std": sample_std(values),
        "min": min(values),
        "max": max(values),
    }


def mean_cov_2d(points):
    points = list(points)
    if len(points) < 2:
        raise ValueError("At least 2 points are required for covariance")

    mu = [
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    ]

    sxx = 0.0
    sxy = 0.0
    syy = 0.0
    for point in points:
        dx = point[0] - mu[0]
        dy = point[1] - mu[1]
        sxx += dx * dx
        sxy += dx * dy
        syy += dy * dy

    denom = len(points) - 1
    sigma = [[sxx / denom, sxy / denom], [sxy / denom, syy / denom]]
    return mu, sigma


def rotation_matrix(theta_rad):
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    return [[c, -s], [s, c]]


def mat_vec(matrix, vector):
    return [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
    ]


def tracker_local_drift(row):
    yaw_rad = math.radians(finite_float(row, "tracker_start_yaw_deg"))
    drift_world = [
        finite_float(row, "tracker_dx"),
        finite_float(row, "tracker_dy"),
    ]
    return mat_vec(rotation_matrix(-yaw_rad), drift_world)


def build_rotation_analysis(
    csv_path,
    run_range,
    tracker_yaw_sign,
    run_id_prefix=None,
):
    fieldnames, rows = read_csv_rows(csv_path)
    require_columns(fieldnames, REQUIRED_COLUMNS, csv_path)
    selected_rows = filter_rows(
        rows,
        run_range_text=run_range,
        run_id_prefix=run_id_prefix,
    )
    valid_rows, skipped_rows = valid_rows_with_columns(
        selected_rows,
        REQUIRED_COLUMNS,
    )

    if len(valid_rows) < 2:
        raise DataError("At least 2 valid rotation rows are required")

    command_angles = [finite_float(row, "command_angle_deg") for row in valid_rows]
    corrected_tracker_yaw_changes = [
        tracker_yaw_sign * finite_float(row, "tracker_yaw_change_deg")
        for row in valid_rows
    ]
    corrected_tracker_yaw_errors = [
        shortest_angle_delta_deg(command_angle, yaw_change)
        for command_angle, yaw_change in zip(
            command_angles,
            corrected_tracker_yaw_changes,
        )
    ]

    tracker_drifts = [
        [finite_float(row, "tracker_dx"), finite_float(row, "tracker_dy")]
        for row in valid_rows
    ]
    tracker_drift_mu, tracker_drift_sigma = mean_cov_2d(tracker_drifts)
    tracker_local_drifts = [tracker_local_drift(row) for row in valid_rows]
    tracker_local_drift_mu, tracker_local_drift_sigma = mean_cov_2d(
        tracker_local_drifts
    )

    tracker_drift_magnitudes = [
        finite_float(row, "tracker_position_drift_m") for row in valid_rows
    ]
    odom_yaw_changes = [
        finite_float(row, "odom_yaw_change_deg") for row in valid_rows
    ]
    odom_yaw_errors = [
        finite_float(row, "odom_yaw_error_deg") for row in valid_rows
    ]
    odom_drift_magnitudes = [
        finite_float(row, "odom_position_drift_m") for row in valid_rows
    ]

    model = {
        "units": {
            "position": "m",
            "angle": "deg",
            "covariance": "m^2",
        },
        "data_selection": {
            "rotation_csv": str(csv_path),
            "run_range": run_range,
            "run_id_prefix": run_id_prefix,
            "selected_run_ids": [row["run_id"] for row in valid_rows],
            "skipped_rows": skipped_rows,
        },
        "tracker_yaw_sign_correction": float(tracker_yaw_sign),
        "command_angle_deg": value_summary(command_angles),
        "tracker_rotation_model": {
            "n": len(valid_rows),
            "corrected_yaw_change_deg": value_summary(
                corrected_tracker_yaw_changes
            ),
            "corrected_yaw_error_deg": value_summary(corrected_tracker_yaw_errors),
            "drift_mu": tracker_drift_mu,
            "drift_sigma": tracker_drift_sigma,
            "drift_std": [
                math.sqrt(max(tracker_drift_sigma[0][0], 0.0)),
                math.sqrt(max(tracker_drift_sigma[1][1], 0.0)),
            ],
            "local_drift_mu": tracker_local_drift_mu,
            "local_drift_sigma": tracker_local_drift_sigma,
            "local_drift_std": [
                math.sqrt(max(tracker_local_drift_sigma[0][0], 0.0)),
                math.sqrt(max(tracker_local_drift_sigma[1][1], 0.0)),
            ],
            "drift_magnitude_m": value_summary(tracker_drift_magnitudes),
        },
        "odom_reference": {
            "yaw_change_deg": value_summary(odom_yaw_changes),
            "yaw_error_deg": value_summary(odom_yaw_errors),
            "drift_magnitude_m": value_summary(odom_drift_magnitudes),
        },
    }

    return model


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def write_summary_csv(path, model):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tracker = model["tracker_rotation_model"]
    odom = model["odom_reference"]
    drift_sigma = tracker["drift_sigma"]
    local_drift_sigma = tracker["local_drift_sigma"]

    rows = [
        ("valid_run_count", tracker["n"], "count"),
        ("tracker_yaw_sign_correction", model["tracker_yaw_sign_correction"], "unitless"),
        (
            "tracker_corrected_yaw_change_mean",
            tracker["corrected_yaw_change_deg"]["mean"],
            "deg",
        ),
        (
            "tracker_corrected_yaw_change_std",
            tracker["corrected_yaw_change_deg"]["std"],
            "deg",
        ),
        (
            "tracker_corrected_yaw_error_mean",
            tracker["corrected_yaw_error_deg"]["mean"],
            "deg",
        ),
        (
            "tracker_corrected_yaw_error_std",
            tracker["corrected_yaw_error_deg"]["std"],
            "deg",
        ),
        ("tracker_drift_mu_x", tracker["drift_mu"][0], "m"),
        ("tracker_drift_mu_y", tracker["drift_mu"][1], "m"),
        ("tracker_drift_sigma_xx", drift_sigma[0][0], "m^2"),
        ("tracker_drift_sigma_xy", drift_sigma[0][1], "m^2"),
        ("tracker_drift_sigma_yy", drift_sigma[1][1], "m^2"),
        ("tracker_drift_std_x", tracker["drift_std"][0], "m"),
        ("tracker_drift_std_y", tracker["drift_std"][1], "m"),
        ("tracker_local_drift_mu_x", tracker["local_drift_mu"][0], "m"),
        ("tracker_local_drift_mu_y", tracker["local_drift_mu"][1], "m"),
        ("tracker_local_drift_sigma_xx", local_drift_sigma[0][0], "m^2"),
        ("tracker_local_drift_sigma_xy", local_drift_sigma[0][1], "m^2"),
        ("tracker_local_drift_sigma_yy", local_drift_sigma[1][1], "m^2"),
        ("tracker_local_drift_std_x", tracker["local_drift_std"][0], "m"),
        ("tracker_local_drift_std_y", tracker["local_drift_std"][1], "m"),
        (
            "tracker_drift_magnitude_mean",
            tracker["drift_magnitude_m"]["mean"],
            "m",
        ),
        (
            "tracker_drift_magnitude_std",
            tracker["drift_magnitude_m"]["std"],
            "m",
        ),
        ("odom_yaw_change_mean", odom["yaw_change_deg"]["mean"], "deg"),
        ("odom_yaw_change_std", odom["yaw_change_deg"]["std"], "deg"),
        ("odom_yaw_error_mean", odom["yaw_error_deg"]["mean"], "deg"),
        ("odom_yaw_error_std", odom["yaw_error_deg"]["std"], "deg"),
        ("odom_drift_magnitude_mean", odom["drift_magnitude_m"]["mean"], "m"),
        ("odom_drift_magnitude_std", odom["drift_magnitude_m"]["std"], "m"),
    ]

    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value", "unit"])
        writer.writerows(rows)


def print_report(model):
    tracker = model["tracker_rotation_model"]
    odom = model["odom_reference"]
    sigma = tracker["drift_sigma"]
    local_sigma = tracker["local_drift_sigma"]

    print("Selected rotation runs:")
    print(", ".join(model["data_selection"]["selected_run_ids"]))
    print(f"Skipped rows: {len(model['data_selection']['skipped_rows'])}")
    print(f"Tracker yaw sign correction: {model['tracker_yaw_sign_correction']:+.1f}")

    print("\nTracker rotation model:")
    print(f"  n = {tracker['n']}")
    print(
        "  corrected yaw change = "
        f"{tracker['corrected_yaw_change_deg']['mean']:.3f} ± "
        f"{tracker['corrected_yaw_change_deg']['std']:.3f} deg"
    )
    print(
        "  corrected yaw error = "
        f"{tracker['corrected_yaw_error_deg']['mean']:.3f} ± "
        f"{tracker['corrected_yaw_error_deg']['std']:.3f} deg"
    )
    print(
        "  drift mu = "
        f"[{tracker['drift_mu'][0]:.6f}, {tracker['drift_mu'][1]:.6f}] m"
    )
    print(
        "  drift sigma = "
        f"[[{sigma[0][0]:.10f}, {sigma[0][1]:.10f}], "
        f"[{sigma[1][0]:.10f}, {sigma[1][1]:.10f}]] m^2"
    )
    print(
        "  local drift mu = "
        f"[{tracker['local_drift_mu'][0]:.6f}, "
        f"{tracker['local_drift_mu'][1]:.6f}] m"
    )
    print(
        "  local drift sigma = "
        f"[[{local_sigma[0][0]:.10f}, {local_sigma[0][1]:.10f}], "
        f"[{local_sigma[1][0]:.10f}, {local_sigma[1][1]:.10f}]] m^2"
    )
    print(
        "  drift magnitude = "
        f"{tracker['drift_magnitude_m']['mean']:.6f} ± "
        f"{tracker['drift_magnitude_m']['std']:.6f} m"
    )

    print("\nOdometry reference:")
    print(
        "  yaw change = "
        f"{odom['yaw_change_deg']['mean']:.3f} ± "
        f"{odom['yaw_change_deg']['std']:.3f} deg"
    )
    print(
        "  yaw error = "
        f"{odom['yaw_error_deg']['mean']:.3f} ± "
        f"{odom['yaw_error_deg']['std']:.3f} deg"
    )
    print(
        "  drift magnitude = "
        f"{odom['drift_magnitude_m']['mean']:.6f} ± "
        f"{odom['drift_magnitude_m']['std']:.6f} m"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze repeated real rotate-in-place runs.",
    )
    parser.add_argument("--rotation-csv", default=DEFAULT_ROTATION_CSV)
    parser.add_argument("--run-range", default=DEFAULT_RUN_RANGE)
    parser.add_argument(
        "--run-id-prefix",
        default=None,
        help="Only include run IDs starting with this prefix.",
    )
    parser.add_argument(
        "--tracker-yaw-sign",
        type=float,
        default=DEFAULT_TRACKER_YAW_SIGN,
        help="Sign correction applied to tracker_yaw_change_deg.",
    )
    parser.add_argument(
        "--output-json",
        default="results/real_rotation_model.json",
    )
    parser.add_argument(
        "--summary-csv",
        default="results/real_rotation_model_summary.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = build_rotation_analysis(
        csv_path=args.rotation_csv,
        run_range=args.run_range,
        tracker_yaw_sign=args.tracker_yaw_sign,
        run_id_prefix=args.run_id_prefix,
    )
    write_json(args.output_json, model)
    write_summary_csv(args.summary_csv, model)
    print_report(model)
    print("\nGenerated outputs:")
    print(f"  {args.output_json}")
    print(f"  {args.summary_csv}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DataError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
    except OSError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
