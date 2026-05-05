#!/usr/bin/env python3
"""
Build empirical motion primitives for probabilistic path endpoint prediction.

The script is independent of ROS and camera hardware.  It uses already-recorded
tracker CSV rows to estimate local-frame displacement and yaw-change models for
F30, CW90, and CCW90 commands.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import analyze_probabilistic_endpoint_model as endpoint_model
import analyze_rotation_runs as rotation_analysis


EIGEN_TOL = -1e-12
FORWARD_COLUMNS = [
    "run_id",
    "tracker_start_x",
    "tracker_start_y",
    "tracker_start_yaw_deg",
    "tracker_final_x",
    "tracker_final_y",
    "tracker_final_yaw_deg",
]
ROTATION_COLUMNS = [
    "run_id",
    "tracker_start_x",
    "tracker_start_y",
    "tracker_start_yaw_deg",
    "tracker_final_x",
    "tracker_final_y",
    "tracker_final_yaw_deg",
]


class PrimitiveModelError(ValueError):
    """Raised when primitive model inputs or outputs are invalid."""


def finite_float(row, column):
    return endpoint_model.finite_float(row, column)


def pose_local_delta_and_yaw_delta(row, prefix, tracker_yaw_sign):
    start = [
        finite_float(row, f"{prefix}_start_x"),
        finite_float(row, f"{prefix}_start_y"),
    ]
    final = [
        finite_float(row, f"{prefix}_final_x"),
        finite_float(row, f"{prefix}_final_y"),
    ]
    start_yaw_deg = finite_float(row, f"{prefix}_start_yaw_deg")
    final_yaw_deg = finite_float(row, f"{prefix}_final_yaw_deg")

    world_delta = endpoint_model.vec_sub(final, start)
    local_delta = endpoint_model.mat_vec(
        endpoint_model.rotation_matrix(-math.radians(start_yaw_deg)),
        world_delta,
    )
    raw_yaw_delta = endpoint_model.normalize_angle_deg(
        final_yaw_deg - start_yaw_deg
    )
    yaw_delta = endpoint_model.normalize_angle_deg(
        tracker_yaw_sign * raw_yaw_delta
    )

    return local_delta, yaw_delta


def angle_delta_summary_deg(values):
    values = [endpoint_model.normalize_angle_deg(value) for value in values]
    if not values:
        raise ValueError("At least one yaw delta is required")

    circular = endpoint_model.circular_yaw_summary_deg(values)
    mean = circular["mean_deg"]
    if len(values) < 2:
        std = 0.0
    else:
        residuals = [
            endpoint_model.normalize_angle_deg(value - mean) for value in values
        ]
        std = math.sqrt(
            sum(residual * residual for residual in residuals)
            / (len(residuals) - 1)
        )

    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
    }


def build_primitive(name, primitive_type, rows, prefix, tracker_yaw_sign):
    if len(rows) < 2:
        raise PrimitiveModelError(f"{name} requires at least 2 valid rows")

    local_deltas = []
    yaw_deltas = []
    for row in rows:
        local_delta, yaw_delta = pose_local_delta_and_yaw_delta(
            row,
            prefix,
            tracker_yaw_sign,
        )
        local_deltas.append(local_delta)
        yaw_deltas.append(yaw_delta)

    local_delta_mu, local_delta_sigma = endpoint_model.empirical_mean_cov(
        local_deltas
    )
    yaw_summary = angle_delta_summary_deg(yaw_deltas)

    primitive = {
        "type": primitive_type,
        "local_delta_mu": local_delta_mu,
        "local_delta_sigma": local_delta_sigma,
        "local_delta_std": endpoint_model.matrix_std(local_delta_sigma),
        "yaw_delta_mean_deg": yaw_summary["mean"],
        "yaw_delta_std_deg": yaw_summary["std"],
        "yaw_delta_min_deg": yaw_summary["min"],
        "yaw_delta_max_deg": yaw_summary["max"],
        "n": len(rows),
        "selected_run_ids": endpoint_model.run_ids(rows),
    }

    validate_primitive(name, primitive)
    return primitive


def validate_primitive(name, primitive):
    sigma = primitive["local_delta_sigma"]
    validate_covariance(name, sigma)

    yaw_mean = primitive["yaw_delta_mean_deg"]
    if name == "CW90" and yaw_mean >= 0.0:
        raise PrimitiveModelError(
            f"{name} yaw_delta_mean_deg must be negative, got {yaw_mean:.6f}"
        )
    if name == "CCW90" and yaw_mean <= 0.0:
        raise PrimitiveModelError(
            f"{name} yaw_delta_mean_deg must be positive, got {yaw_mean:.6f}"
        )


def validate_covariance(name, sigma):
    if len(sigma) != 2 or any(len(row) != 2 for row in sigma):
        raise PrimitiveModelError(f"{name} covariance must be 2x2")

    values = [sigma[0][0], sigma[0][1], sigma[1][0], sigma[1][1]]
    if not all(math.isfinite(value) for value in values):
        raise PrimitiveModelError(f"{name} covariance contains non-finite values")

    if abs(sigma[0][1] - sigma[1][0]) > 1e-12:
        raise PrimitiveModelError(f"{name} covariance must be symmetric")

    if sigma[0][0] < 0.0 or sigma[1][1] < 0.0:
        raise PrimitiveModelError(
            f"{name} covariance diagonal entries must be non-negative"
        )

    eigvals, _ = endpoint_model.symmetric_eigen_2x2(sigma)
    if min(eigvals) < EIGEN_TOL:
        raise PrimitiveModelError(
            f"{name} covariance has a negative eigenvalue: {min(eigvals):.12g}"
        )


def load_forward_rows(path, run_range):
    fieldnames, rows = endpoint_model.read_csv_rows(path)
    endpoint_model.require_columns(fieldnames, FORWARD_COLUMNS, path)
    selected_rows = endpoint_model.filter_rows_by_run_range(rows, run_range)
    valid_rows, skipped_rows = endpoint_model.valid_rows_with_columns(
        selected_rows,
        FORWARD_COLUMNS,
    )
    return valid_rows, skipped_rows


def load_rotation_rows(path, run_range, run_id_prefix):
    fieldnames, rows = endpoint_model.read_csv_rows(path)
    endpoint_model.require_columns(fieldnames, ROTATION_COLUMNS, path)
    selected_rows = rotation_analysis.filter_rows(
        rows,
        run_range_text=run_range,
        run_id_prefix=run_id_prefix,
    )
    valid_rows, skipped_rows = endpoint_model.valid_rows_with_columns(
        selected_rows,
        ROTATION_COLUMNS,
    )
    return valid_rows, skipped_rows


def build_motion_primitives_model(
    forward_csv,
    forward_run_range,
    rotation_csv,
    cw_prefix,
    cw_run_range,
    ccw_prefix,
    ccw_run_range,
    tracker_yaw_sign,
):
    forward_rows, skipped_forward = load_forward_rows(
        forward_csv,
        forward_run_range,
    )
    cw_rows, skipped_cw = load_rotation_rows(rotation_csv, cw_run_range, cw_prefix)
    ccw_rows, skipped_ccw = load_rotation_rows(
        rotation_csv,
        ccw_run_range,
        ccw_prefix,
    )

    primitives = {
        "F30": build_primitive(
            "F30",
            "forward",
            forward_rows,
            "tracker",
            tracker_yaw_sign,
        ),
        "CW90": build_primitive(
            "CW90",
            "rotation",
            cw_rows,
            "tracker",
            tracker_yaw_sign,
        ),
        "CCW90": build_primitive(
            "CCW90",
            "rotation",
            ccw_rows,
            "tracker",
            tracker_yaw_sign,
        ),
    }

    return {
        "units": {
            "position": "m",
            "angle": "deg",
            "covariance": "m^2",
        },
        "yaw_source": {
            "tracker_yaw_sign_correction": float(tracker_yaw_sign),
            "yaw_delta_convention": "positive CCW, degrees",
        },
        "coordinate_frame": "robot local frame at primitive start",
        "data_selection": {
            "forward_csv": str(forward_csv),
            "forward_run_range": forward_run_range,
            "selected_forward_run_ids": endpoint_model.run_ids(forward_rows),
            "skipped_forward_rows": skipped_forward,
            "rotation_csv": str(rotation_csv),
            "cw_prefix": cw_prefix,
            "cw_run_range": cw_run_range,
            "selected_cw_run_ids": endpoint_model.run_ids(cw_rows),
            "skipped_cw_rows": skipped_cw,
            "ccw_prefix": ccw_prefix,
            "ccw_run_range": ccw_run_range,
            "selected_ccw_run_ids": endpoint_model.run_ids(ccw_rows),
            "skipped_ccw_rows": skipped_ccw,
        },
        "primitives": primitives,
        "assumptions": [
            "Primitive errors are treated as independent between actions.",
            "Translation uncertainty and yaw uncertainty are modeled separately.",
            "The model describes current measured commands, not idealized controls.",
        ],
        "limitations": [
            "No full joint covariance over x, y, and yaw is learned.",
            "No full trajectory, obstacle, SLAM, or map uncertainty is modeled.",
        ],
    }


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def write_summary_csv(path, model):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, primitive in model["primitives"].items():
        sigma = primitive["local_delta_sigma"]
        rows.extend(
            [
                (name, "type", primitive["type"], ""),
                (name, "n", primitive["n"], "count"),
                (name, "local_delta_mu_x", primitive["local_delta_mu"][0], "m"),
                (name, "local_delta_mu_y", primitive["local_delta_mu"][1], "m"),
                (name, "local_delta_sigma_xx", sigma[0][0], "m^2"),
                (name, "local_delta_sigma_xy", sigma[0][1], "m^2"),
                (name, "local_delta_sigma_yy", sigma[1][1], "m^2"),
                (name, "local_delta_std_x", primitive["local_delta_std"][0], "m"),
                (name, "local_delta_std_y", primitive["local_delta_std"][1], "m"),
                (
                    name,
                    "yaw_delta_mean",
                    primitive["yaw_delta_mean_deg"],
                    "deg",
                ),
                (
                    name,
                    "yaw_delta_std",
                    primitive["yaw_delta_std_deg"],
                    "deg",
                ),
            ]
        )

    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["primitive", "metric", "value", "unit"])
        writer.writerows(rows)


def print_report(model):
    print("Motion primitive model:")
    print(f"  coordinate_frame = {model['coordinate_frame']}")
    print(
        "  tracker_yaw_sign_correction = "
        f"{model['yaw_source']['tracker_yaw_sign_correction']:+.1f}"
    )

    for name, primitive in model["primitives"].items():
        sigma = primitive["local_delta_sigma"]
        print(f"\n{name}:")
        print(f"  type = {primitive['type']}")
        print(f"  n = {primitive['n']}")
        print(
            "  local_delta_mu = "
            f"[{primitive['local_delta_mu'][0]:.6f}, "
            f"{primitive['local_delta_mu'][1]:.6f}] m"
        )
        print(
            "  local_delta_sigma = "
            f"[[{sigma[0][0]:.10f}, {sigma[0][1]:.10f}], "
            f"[{sigma[1][0]:.10f}, {sigma[1][1]:.10f}]] m^2"
        )
        print(
            "  yaw_delta = "
            f"{primitive['yaw_delta_mean_deg']:.3f} ± "
            f"{primitive['yaw_delta_std_deg']:.3f} deg"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build empirical F30/CW90/CCW90 motion primitives.",
    )
    parser.add_argument(
        "--forward-csv",
        default="results/real_scripted_drive_runs.csv",
    )
    parser.add_argument("--forward-run-range", default="21:50")
    parser.add_argument(
        "--rotation-csv",
        default="results/real_rotation_runs.csv",
    )
    parser.add_argument("--cw-prefix", default="run_real_rot_cw90_")
    parser.add_argument("--cw-run-range", default="18:32")
    parser.add_argument("--ccw-prefix", default="run_real_rot_ccw90_")
    parser.add_argument("--ccw-run-range", default="1:15")
    parser.add_argument("--tracker-yaw-sign", type=float, default=-1.0)
    parser.add_argument(
        "--output-json",
        default="results/probabilistic_motion_primitives_model.json",
    )
    parser.add_argument(
        "--summary-csv",
        default="results/probabilistic_motion_primitives_model_summary.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = build_motion_primitives_model(
        forward_csv=args.forward_csv,
        forward_run_range=args.forward_run_range,
        rotation_csv=args.rotation_csv,
        cw_prefix=args.cw_prefix,
        cw_run_range=args.cw_run_range,
        ccw_prefix=args.ccw_prefix,
        ccw_run_range=args.ccw_run_range,
        tracker_yaw_sign=args.tracker_yaw_sign,
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
    except (PrimitiveModelError, endpoint_model.DataError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
    except OSError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
