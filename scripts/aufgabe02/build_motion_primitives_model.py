#!/usr/bin/env python3
"""Build measured forward and turn primitives in the robot's local start frame."""
import argparse
import math

if __package__:
    from .probabilistic_model import csv_data as data, geometry, statistics
else:
    from probabilistic_model import csv_data as data, geometry, statistics

EIGEN_TOL = -1e-12


class PrimitiveModelError(ValueError):
    """Raised when primitive model inputs or outputs are invalid."""


def pose_local_delta_and_yaw_delta(row, prefix, tracker_yaw_sign):
    start = [
        data.finite_float(row, f"{prefix}_start_x"),
        data.finite_float(row, f"{prefix}_start_y"),
    ]
    final = [
        data.finite_float(row, f"{prefix}_final_x"),
        data.finite_float(row, f"{prefix}_final_y"),
    ]
    start_yaw_deg = data.finite_float(row, f"{prefix}_start_yaw_deg")
    final_yaw_deg = data.finite_float(row, f"{prefix}_final_yaw_deg")

    world_delta = geometry.vec_sub(final, start)
    local_delta = geometry.mat_vec(
        geometry.rotation_matrix(-math.radians(start_yaw_deg)),
        world_delta,
    )
    raw_yaw_delta = geometry.normalize_angle_deg(
        final_yaw_deg - start_yaw_deg
    )
    yaw_delta = geometry.normalize_angle_deg(
        tracker_yaw_sign * raw_yaw_delta
    )

    return local_delta, yaw_delta


def angle_delta_summary_deg(values):
    values = [geometry.normalize_angle_deg(value) for value in values]
    if not values:
        raise ValueError("At least one yaw delta is required")

    circular = statistics.circular_yaw_summary_deg(values)
    mean = circular["mean_deg"]
    if len(values) < 2:
        std = 0.0
    else:
        residuals = [
            geometry.normalize_angle_deg(value - mean) for value in values
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

    local_delta_mu, local_delta_sigma = statistics.empirical_mean_cov(
        local_deltas
    )
    yaw_summary = angle_delta_summary_deg(yaw_deltas)

    primitive = {
        "type": primitive_type,
        "local_delta_mu": local_delta_mu,
        "local_delta_sigma": local_delta_sigma,
        "local_delta_std": statistics.matrix_std(local_delta_sigma),
        "yaw_delta_mean_deg": yaw_summary["mean"],
        "yaw_delta_std_deg": yaw_summary["std"],
        "yaw_delta_min_deg": yaw_summary["min"],
        "yaw_delta_max_deg": yaw_summary["max"],
        "n": len(rows),
        "selected_run_ids": data.run_ids(rows),
    }

    validate_primitive(name, primitive)
    return primitive


def validate_primitive(name, primitive):
    sigma = primitive["local_delta_sigma"]
    validate_covariance(name, sigma)

    yaw_mean = primitive["yaw_delta_mean_deg"]
    if name.startswith("CW") and yaw_mean >= 0.0:
        raise PrimitiveModelError(
            f"{name} yaw_delta_mean_deg must be negative, got {yaw_mean:.6f}"
        )
    if name.startswith("CCW") and yaw_mean <= 0.0:
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

    eigvals, _ = statistics.symmetric_eigen_2x2(sigma)
    if min(eigvals) < EIGEN_TOL:
        raise PrimitiveModelError(
            f"{name} covariance has a negative eigenvalue: {min(eigvals):.12g}"
        )


def load_primitive_rows(path, run_range, run_id_prefix):
    fields, rows = data.read_csv_rows(path)
    columns = data.pose_columns("tracker")
    data.require_columns(fields, columns, path)
    selected = data.filter_rows_by_run_range(rows, run_range)
    if run_id_prefix:
        selected = [row for row in selected if row.get("run_id", "").startswith(run_id_prefix)]
    return data.valid_rows_with_columns(selected, columns)


def build_motion_primitives_model(
    forward_csv, forward_run_range, rotation_csv, cw_prefix, cw_run_range,
    ccw_prefix, ccw_run_range, tracker_yaw_sign,
    extra_forward_specs=None, extra_rotation_specs=None,
):
    specs = [
        ("F30", "forward", forward_csv, forward_run_range, None),
        ("CW90", "rotation", rotation_csv, cw_run_range, cw_prefix),
        ("CCW90", "rotation", rotation_csv, ccw_run_range, ccw_prefix),
    ]
    for kind, path, extras in [
        ("forward", forward_csv, extra_forward_specs),
        ("rotation", rotation_csv, extra_rotation_specs),
    ]:
        for spec in extras or []:
            specs.append((spec["name"], kind, path, spec["run_range"], spec["run_id_prefix"]))

    primitives = {}
    primitive_sources = {}
    for name, kind, path, run_range, prefix in specs:
        rows, skipped = load_primitive_rows(path, run_range, prefix)
        primitives[name] = build_primitive(name, kind, rows, "tracker", tracker_yaw_sign)
        primitive_sources[name] = {
            "csv": str(path),
            "run_range": run_range,
            "run_id_prefix": prefix,
            "selected_run_ids": data.run_ids(rows),
            "skipped_rows": skipped,
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
            "selected_forward_run_ids": primitive_sources["F30"]["selected_run_ids"],
            "skipped_forward_rows": primitive_sources["F30"]["skipped_rows"],
            "rotation_csv": str(rotation_csv),
            "cw_prefix": cw_prefix,
            "cw_run_range": cw_run_range,
            "selected_cw_run_ids": primitive_sources["CW90"]["selected_run_ids"],
            "skipped_cw_rows": primitive_sources["CW90"]["skipped_rows"],
            "ccw_prefix": ccw_prefix,
            "ccw_run_range": ccw_run_range,
            "selected_ccw_run_ids": primitive_sources["CCW90"]["selected_run_ids"],
            "skipped_ccw_rows": primitive_sources["CCW90"]["skipped_rows"],
            "primitive_sources": primitive_sources,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-csv", default="results/aufgabe02/real_scripted_drive_runs.csv")
    parser.add_argument("--forward-run-range", default="21:50")
    parser.add_argument("--rotation-csv", default="results/aufgabe02/real_rotation_runs.csv")
    parser.add_argument("--cw-prefix", default="run_real_rot_cw90_")
    parser.add_argument("--cw-run-range", default="18:32")
    parser.add_argument("--ccw-prefix", default="run_real_rot_ccw90_")
    parser.add_argument("--ccw-run-range", default="1:15")
    for name in ("f50", "ccw45", "cw45", "ccw180"):
        parser.add_argument(f"--{name}-prefix")
        parser.add_argument(f"--{name}-run-range", default="1:15")
    parser.add_argument("--tracker-yaw-sign", type=float, default=-1.0)
    parser.add_argument("--output-json", default="results/aufgabe02/probabilistic_motion_primitives_model.json")
    args = parser.parse_args()
    extras = {"forward": [], "rotation": []}
    for name in ("f50", "ccw45", "cw45", "ccw180"):
        prefix = getattr(args, f"{name}_prefix")
        if prefix:
            extras["forward" if name == "f50" else "rotation"].append({
                "name": name.upper(), "run_id_prefix": prefix,
                "run_range": getattr(args, f"{name}_run_range"),
            })
    model = build_motion_primitives_model(
        args.forward_csv, args.forward_run_range, args.rotation_csv,
        args.cw_prefix, args.cw_run_range, args.ccw_prefix, args.ccw_run_range,
        args.tracker_yaw_sign, extras["forward"], extras["rotation"],
    )
    data.write_json(args.output_json, model)
    print(args.output_json)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}")
