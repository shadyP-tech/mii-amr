#!/usr/bin/env python3
"""
Analyze empirical endpoint and 30 cm motion-error models for Aufgabe 2.

The script is independent of ROS, Gazebo, and camera hardware.  It also avoids
mandatory third-party analysis dependencies so ``python3`` can run it on the
editing machine.  If Matplotlib is installed it is used for plots; otherwise a
small built-in PNG fallback is used.
"""

import argparse
import csv
import json
import math
import re
import struct
import zlib
from pathlib import Path


CHI2_95_2D = 5.991
DEFAULT_STEP_DISTANCE_M = 0.30
NEAR_SINGULAR_DET = 1e-14
NEAR_SINGULAR_COND = 1e12

REAL_ENDPOINT_COLUMNS = [
    "run_id",
    "tracker_start_x",
    "tracker_start_y",
    "tracker_start_yaw_deg",
    "tracker_final_x",
    "tracker_final_y",
    "tracker_final_yaw_deg",
]

SIM_ENDPOINT_COLUMNS = [
    "run_id",
    "odom_start_x",
    "odom_start_y",
    "odom_start_yaw_deg",
    "odom_final_x",
    "odom_final_y",
    "odom_final_yaw_deg",
]


class DataError(ValueError):
    """Raised when CSV data is missing required fields or valid samples."""


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


def rotation_matrix(theta_rad):
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    return [[c, -s], [s, c]]


def mat_vec(matrix, vector):
    return [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
    ]


def mat_mul(a, b):
    return [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]


def mat_transpose(matrix):
    return [[matrix[0][0], matrix[1][0]], [matrix[0][1], matrix[1][1]]]


def mat_add(a, b):
    return [
        [a[0][0] + b[0][0], a[0][1] + b[0][1]],
        [a[1][0] + b[1][0], a[1][1] + b[1][1]],
    ]


def mat_scale(scale, matrix):
    return [
        [scale * matrix[0][0], scale * matrix[0][1]],
        [scale * matrix[1][0], scale * matrix[1][1]],
    ]


def vec_add(a, b):
    return [a[0] + b[0], a[1] + b[1]]


def vec_sub(a, b):
    return [a[0] - b[0], a[1] - b[1]]


def vec_scale(scale, vector):
    return [scale * vector[0], scale * vector[1]]


def local_displacements(rows, prefix):
    displacements = []

    for row in rows:
        start = [
            finite_float(row, f"{prefix}_start_x"),
            finite_float(row, f"{prefix}_start_y"),
        ]
        final = [
            finite_float(row, f"{prefix}_final_x"),
            finite_float(row, f"{prefix}_final_y"),
        ]
        yaw_rad = math.radians(finite_float(row, f"{prefix}_start_yaw_deg"))
        delta_world = vec_sub(final, start)
        delta_local = mat_vec(rotation_matrix(-yaw_rad), delta_world)
        displacements.append(delta_local)

    return displacements


def motion_errors(local_delta, step_distance_m):
    command = [step_distance_m, 0.0]
    return [vec_sub(delta, command) for delta in local_delta]


def empirical_mean_cov(points):
    points = as_points(points)
    if len(points) < 2:
        raise ValueError("At least 2 valid points are required for covariance")

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


def as_points(points):
    normalized = []
    for point in points:
        if len(point) != 2:
            raise ValueError("Expected 2D points")
        x = float(point[0])
        y = float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Point array contains non-finite values")
        normalized.append([x, y])
    return normalized


def circular_yaw_summary_deg(values_deg):
    values = [float(value) for value in values_deg]
    if not values:
        raise ValueError("At least one yaw value is required")

    radians = [math.radians(value) for value in values]
    sin_mean = sum(math.sin(value) for value in radians) / len(radians)
    cos_mean = sum(math.cos(value) for value in radians) / len(radians)
    mean_rad = math.atan2(sin_mean, cos_mean)
    resultant = math.hypot(sin_mean, cos_mean)
    resultant = min(max(resultant, 1e-12), 1.0)
    std_rad = math.sqrt(-2.0 * math.log(resultant))

    return {
        "mean_deg": normalize_angle_deg(math.degrees(mean_rad)),
        "std_deg": math.degrees(std_rad),
    }


def normalize_angle_deg(value):
    normalized = (value + 180.0) % 360.0 - 180.0
    if normalized == -180.0 and value > 0:
        return 180.0
    return normalized


def symmetric_eigen_2x2(matrix):
    a = float(matrix[0][0])
    b = 0.5 * (float(matrix[0][1]) + float(matrix[1][0]))
    d = float(matrix[1][1])
    trace_half = 0.5 * (a + d)
    diff_half = 0.5 * (a - d)
    radius = math.hypot(diff_half, b)
    eig1 = trace_half + radius
    eig2 = trace_half - radius

    vec1 = eigenvector_for_value(a, b, d, eig1)
    vec2 = [-vec1[1], vec1[0]]
    return [eig1, eig2], [vec1, vec2]


def eigenvector_for_value(a, b, d, eig):
    if abs(b) > 1e-15 or abs(eig - a) > 1e-15:
        vector = [b, eig - a]
    else:
        vector = [1.0, 0.0] if a >= d else [0.0, 1.0]

    norm = math.hypot(vector[0], vector[1])
    if norm == 0.0:
        return [1.0, 0.0]
    return [vector[0] / norm, vector[1] / norm]


def covariance_warning(sigma):
    det = determinant_2x2(sigma)
    eigvals, _ = symmetric_eigen_2x2(sigma)
    max_abs = max(abs(value) for value in eigvals)
    min_abs = min(abs(value) for value in eigvals)
    condition = math.inf if min_abs == 0.0 else max_abs / min_abs

    if abs(det) < NEAR_SINGULAR_DET or condition > NEAR_SINGULAR_COND:
        return (
            "Covariance is singular or nearly singular; "
            "using a pseudo-inverse for Mahalanobis distances."
        )

    return None


def determinant_2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def pseudo_inverse_symmetric_2x2(matrix):
    eigvals, eigvecs = symmetric_eigen_2x2(matrix)
    result = [[0.0, 0.0], [0.0, 0.0]]
    for eigval, eigvec in zip(eigvals, eigvecs):
        if abs(eigval) <= 1e-15:
            continue
        inv = 1.0 / eigval
        result[0][0] += inv * eigvec[0] * eigvec[0]
        result[0][1] += inv * eigvec[0] * eigvec[1]
        result[1][0] += inv * eigvec[1] * eigvec[0]
        result[1][1] += inv * eigvec[1] * eigvec[1]
    return result


def mahalanobis_squared(points, mu, sigma):
    points = as_points(points)
    sigma_inv = pseudo_inverse_symmetric_2x2(sigma)
    distances = []
    for point in points:
        residual = vec_sub(point, mu)
        weighted = mat_vec(sigma_inv, residual)
        distances.append(residual[0] * weighted[0] + residual[1] * weighted[1])
    return distances


def outlier_records(rows, points, mu, sigma, threshold=CHI2_95_2D):
    distances = mahalanobis_squared(points, mu, sigma)
    outliers = []
    for row, distance in zip(rows, distances):
        if distance > threshold:
            outliers.append(
                {
                    "run_id": row.get("run_id", ""),
                    "mahalanobis_squared": float(distance),
                }
            )
    return outliers


def ellipse_parameters(mu, sigma, chi2_value=CHI2_95_2D):
    eigvals, eigvecs = symmetric_eigen_2x2(sigma)
    eigvals = [max(value, 0.0) for value in eigvals]
    semi_axes = [math.sqrt(value * chi2_value) for value in eigvals]
    angle_rad = math.atan2(eigvecs[0][1], eigvecs[0][0])

    return {
        "center": [float(mu[0]), float(mu[1])],
        "chi2_value": float(chi2_value),
        "semi_major_m": float(semi_axes[0]),
        "semi_minor_m": float(semi_axes[1]),
        "major_axis_length_m": float(2.0 * semi_axes[0]),
        "minor_axis_length_m": float(2.0 * semi_axes[1]),
        "orientation_deg": normalize_angle_deg(math.degrees(angle_rad)),
        "area_m2": float(math.pi * semi_axes[0] * semi_axes[1]),
    }


def build_analysis_model(
    real_csv,
    real_run_range,
    sim_csv,
    sim_last_n,
    step_distance_m,
    compare_sim_real=True,
):
    real_fieldnames, real_rows = read_csv_rows(real_csv)
    require_columns(real_fieldnames, REAL_ENDPOINT_COLUMNS, real_csv)
    real_selected_rows = filter_rows_by_run_range(real_rows, real_run_range)
    real_valid_rows, skipped_real_rows = valid_rows_with_columns(
        real_selected_rows,
        REAL_ENDPOINT_COLUMNS,
    )

    if len(real_valid_rows) < 2:
        raise DataError("At least 2 valid real rows are required")

    real_endpoint_points = extract_points(
        real_valid_rows,
        "tracker_final_x",
        "tracker_final_y",
    )
    endpoint_mu, endpoint_sigma = empirical_mean_cov(real_endpoint_points)

    real_local_delta = local_displacements(real_valid_rows, "tracker")
    real_errors = motion_errors(real_local_delta, step_distance_m)
    error_mu, error_sigma = empirical_mean_cov(real_errors)

    yaw_summary = circular_yaw_summary_deg(
        extract_yaws(real_valid_rows, "tracker_final_yaw_deg")
    )

    endpoint_warning = covariance_warning(endpoint_sigma)
    error_warning = covariance_warning(error_sigma)
    endpoint_outliers = outlier_records(
        real_valid_rows,
        real_endpoint_points,
        endpoint_mu,
        endpoint_sigma,
    )
    error_outliers = outlier_records(
        real_valid_rows,
        real_errors,
        error_mu,
        error_sigma,
    )

    selected_sim_run_ids = []
    skipped_sim_rows = []
    sim2real_bias = {
        "dx_m": None,
        "dy_m": None,
        "magnitude_m": None,
    }
    sim_valid_rows = []
    sim_errors = []
    sim_local_delta = []
    sim_error_mu = None
    sim_error_sigma = None

    if compare_sim_real:
        sim_fieldnames, sim_rows = read_csv_rows(sim_csv)
        require_columns(sim_fieldnames, SIM_ENDPOINT_COLUMNS, sim_csv)
        sim_selected_rows = filter_latest_rows(sim_rows, sim_last_n)
        sim_valid_rows, skipped_sim_rows = valid_rows_with_columns(
            sim_selected_rows,
            SIM_ENDPOINT_COLUMNS,
        )

        if len(sim_valid_rows) < 2:
            raise DataError("At least 2 valid simulation rows are required")

        sim_local_delta = local_displacements(sim_valid_rows, "odom")
        sim_errors = motion_errors(sim_local_delta, step_distance_m)
        sim_error_mu, sim_error_sigma = empirical_mean_cov(sim_errors)
        bias = vec_sub(error_mu, sim_error_mu)
        sim2real_bias = {
            "dx_m": float(bias[0]),
            "dy_m": float(bias[1]),
            "magnitude_m": float(math.hypot(bias[0], bias[1])),
        }
        selected_sim_run_ids = run_ids(sim_valid_rows)

    selected_real_run_ids = run_ids(real_valid_rows)
    warnings = [warning for warning in [endpoint_warning, error_warning] if warning]

    model = {
        "units": {
            "position": "m",
            "angle": "deg",
        },
        "coordinate_frames": {
            "absolute_endpoint_model": "camera/world tracker frame",
            "motion_primitive_error_model": "robot local start frame",
        },
        "data_selection": {
            "real_csv": str(real_csv),
            "real_run_range": real_run_range,
            "selected_real_run_ids": selected_real_run_ids,
            "skipped_real_rows": skipped_real_rows,
            "sim_csv": str(sim_csv) if compare_sim_real else None,
            "sim_last_n": sim_last_n if compare_sim_real else None,
            "selected_sim_run_ids": selected_sim_run_ids,
            "skipped_sim_rows": skipped_sim_rows,
        },
        "absolute_endpoint_model": {
            "n": len(real_valid_rows),
            "mu": endpoint_mu,
            "sigma": endpoint_sigma,
            "std": matrix_std(endpoint_sigma),
            "ellipse_95": ellipse_parameters(endpoint_mu, endpoint_sigma),
            "outliers_95": endpoint_outliers,
        },
        "motion_primitive_error_model": {
            "step_distance_m": float(step_distance_m),
            "mu_error": error_mu,
            "sigma_error": error_sigma,
            "std_error": matrix_std(error_sigma),
            "ellipse_95": ellipse_parameters(error_mu, error_sigma),
            "outliers_95": error_outliers,
        },
        "yaw_summary": yaw_summary,
        "sim2real_displacement_bias": sim2real_bias,
        "warnings": warnings,
        "limitations": [
            "The model propagates only the empirical 30 cm forward-motion error.",
            "Turn uncertainty at waypoint corners is not modeled.",
            "Segment errors are assumed independent.",
        ],
    }

    arrays = {
        "real_endpoint_points": real_endpoint_points,
        "real_errors": real_errors,
        "real_local_delta": real_local_delta,
        "sim_errors": sim_errors,
        "sim_local_delta": sim_local_delta,
        "endpoint_mu": endpoint_mu,
        "endpoint_sigma": endpoint_sigma,
        "error_mu": error_mu,
        "error_sigma": error_sigma,
        "sim_error_mu": sim_error_mu,
        "sim_error_sigma": sim_error_sigma,
    }

    return model, arrays


def matrix_std(sigma):
    return [math.sqrt(max(sigma[0][0], 0.0)), math.sqrt(max(sigma[1][1], 0.0))]


def run_ids(rows):
    return [row.get("run_id", "") for row in rows]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def write_summary_csv(path, model):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    endpoint = model["absolute_endpoint_model"]
    motion = model["motion_primitive_error_model"]
    yaw = model["yaw_summary"]
    bias = model["sim2real_displacement_bias"]

    rows = [
        ("real_valid_run_count", endpoint["n"], "count"),
        ("endpoint_mu_x", endpoint["mu"][0], "m"),
        ("endpoint_mu_y", endpoint["mu"][1], "m"),
        ("endpoint_sigma_xx", endpoint["sigma"][0][0], "m^2"),
        ("endpoint_sigma_xy", endpoint["sigma"][0][1], "m^2"),
        ("endpoint_sigma_yy", endpoint["sigma"][1][1], "m^2"),
        ("endpoint_std_x", endpoint["std"][0], "m"),
        ("endpoint_std_y", endpoint["std"][1], "m"),
        ("endpoint_ellipse_95_major_axis", endpoint["ellipse_95"]["major_axis_length_m"], "m"),
        ("endpoint_ellipse_95_minor_axis", endpoint["ellipse_95"]["minor_axis_length_m"], "m"),
        ("motion_error_mu_x", motion["mu_error"][0], "m"),
        ("motion_error_mu_y", motion["mu_error"][1], "m"),
        ("motion_error_sigma_xx", motion["sigma_error"][0][0], "m^2"),
        ("motion_error_sigma_xy", motion["sigma_error"][0][1], "m^2"),
        ("motion_error_sigma_yy", motion["sigma_error"][1][1], "m^2"),
        ("motion_error_std_x", motion["std_error"][0], "m"),
        ("motion_error_std_y", motion["std_error"][1], "m"),
        ("motion_error_ellipse_95_major_axis", motion["ellipse_95"]["major_axis_length_m"], "m"),
        ("motion_error_ellipse_95_minor_axis", motion["ellipse_95"]["minor_axis_length_m"], "m"),
        ("yaw_mean", yaw["mean_deg"], "deg"),
        ("yaw_std", yaw["std_deg"], "deg"),
        ("sim2real_bias_dx", bias["dx_m"], "m"),
        ("sim2real_bias_dy", bias["dy_m"], "m"),
        ("sim2real_bias_magnitude", bias["magnitude_m"], "m"),
    ]

    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value", "unit"])
        writer.writerows(rows)


def plot_scatter_with_ellipse(points, mu, sigma, path, title, xlabel, ylabel, labels=None):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
    except ModuleNotFoundError:
        write_fallback_plot(
            path,
            groups=[(points, (46, 92, 170)), ([mu], (180, 40, 40))],
            ellipses=[(mu, sigma, (20, 130, 60))],
        )
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    params = ellipse_parameters(mu, sigma)
    fig, ax = plt.subplots()
    ax.scatter([point[0] for point in points], [point[1] for point in points], label=labels or "samples")
    ax.scatter([mu[0]], [mu[1]], marker="x", s=100, label="mean")
    ellipse = Ellipse(
        xy=mu,
        width=params["major_axis_length_m"],
        height=params["minor_axis_length_m"],
        angle=params["orientation_deg"],
        fill=False,
        linewidth=2,
        label="95% ellipse",
    )
    ax.add_patch(ellipse)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sim_vs_real_errors(real_errors, sim_errors, real_mu, sim_mu, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        write_fallback_plot(
            path,
            groups=[
                (real_errors, (46, 92, 170)),
                (sim_errors, (220, 120, 40)),
                ([real_mu], (180, 40, 40)),
                ([sim_mu], (40, 130, 60)),
            ],
            ellipses=[],
        )
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.scatter([point[0] for point in real_errors], [point[1] for point in real_errors], label="real local error")
    ax.scatter([point[0] for point in sim_errors], [point[1] for point in sim_errors], label="simulation local error")
    ax.scatter([real_mu[0]], [real_mu[1]], marker="x", s=100, label="real mean")
    ax.scatter([sim_mu[0]], [sim_mu[1]], marker="+", s=120, label="simulation mean")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title("Local displacement error after 30 cm command")
    ax.set_xlabel("forward error [m]")
    ax.set_ylabel("lateral error [m]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_fallback_plot(path, groups, ellipses=None, polylines=None):
    ellipses = ellipses or []
    polylines = polylines or []
    width = 900
    height = 700
    margin = 70
    white = (255, 255, 255)
    pixels = [[white for _ in range(width)] for _ in range(height)]

    all_points = []
    for points, _color in groups:
        all_points.extend(points)
    for points, _color in polylines:
        all_points.extend(points)
    for mu, sigma, _color in ellipses:
        all_points.extend(ellipse_sample_points(mu, sigma))

    if not all_points:
        all_points = [[0.0, 0.0], [1.0, 1.0]]

    to_pixel = plot_transform(all_points, width, height, margin)

    zero = to_pixel([0.0, 0.0])
    draw_line(pixels, (margin, zero[1]), (width - margin, zero[1]), (220, 220, 220))
    draw_line(pixels, (zero[0], margin), (zero[0], height - margin), (220, 220, 220))

    for points, color in polylines:
        pixel_points = [to_pixel(point) for point in points]
        for start, end in zip(pixel_points, pixel_points[1:]):
            draw_line(pixels, start, end, color)
        for point in pixel_points:
            draw_circle(pixels, point, 4, color)

    for mu, sigma, color in ellipses:
        pixel_points = [to_pixel(point) for point in ellipse_sample_points(mu, sigma)]
        for start, end in zip(pixel_points, pixel_points[1:] + pixel_points[:1]):
            draw_line(pixels, start, end, color)

    for points, color in groups:
        for point in points:
            draw_circle(pixels, to_pixel(point), 4, color)

    write_png(path, pixels)


def ellipse_sample_points(mu, sigma, chi2_value=CHI2_95_2D, count=96):
    params = ellipse_parameters(mu, sigma, chi2_value=chi2_value)
    angle = math.radians(params["orientation_deg"])
    rot = rotation_matrix(angle)
    a = params["semi_major_m"]
    b = params["semi_minor_m"]
    points = []
    for index in range(count):
        t = 2.0 * math.pi * index / count
        local = [a * math.cos(t), b * math.sin(t)]
        points.append(vec_add(mu, mat_vec(rot, local)))
    return points


def plot_transform(points, width, height, margin):
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    padding_x = 0.12 * span_x
    padding_y = 0.12 * span_y
    min_x -= padding_x
    max_x += padding_x
    min_y -= padding_y
    max_y += padding_y
    span_x = max_x - min_x
    span_y = max_y - min_y
    scale = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y)
    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    pixel_center_x = width / 2
    pixel_center_y = height / 2

    def to_pixel(point):
        x = int(round(pixel_center_x + (point[0] - center_x) * scale))
        y = int(round(pixel_center_y - (point[1] - center_y) * scale))
        return x, y

    return to_pixel


def draw_circle(pixels, center, radius, color):
    cx, cy = center
    height = len(pixels)
    width = len(pixels[0])
    for y in range(cy - radius, cy + radius + 1):
        if y < 0 or y >= height:
            continue
        for x in range(cx - radius, cx + radius + 1):
            if x < 0 or x >= width:
                continue
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius * radius:
                pixels[y][x] = color


def draw_line(pixels, start, end, color):
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    height = len(pixels)
    width = len(pixels[0])

    while True:
        if 0 <= x0 < width and 0 <= y0 < height:
            pixels[y0][x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def write_png(path, pixels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height = len(pixels)
    width = len(pixels[0])
    raw = bytearray()
    for row in pixels:
        raw.append(0)
        for r, g, b in row:
            raw.extend([r, g, b])

    def chunk(chunk_type, data):
        payload = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + payload
            + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), level=9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def print_report(model):
    selection = model["data_selection"]
    endpoint = model["absolute_endpoint_model"]
    motion = model["motion_primitive_error_model"]
    yaw = model["yaw_summary"]
    bias = model["sim2real_displacement_bias"]

    print("Selected real runs:")
    print(", ".join(selection["selected_real_run_ids"]))
    print(f"Skipped real rows: {len(selection['skipped_real_rows'])}")
    for row in selection["skipped_real_rows"]:
        print(f"  row {row['row_number']} {row['run_id']}: {row['reason']}")

    if selection["selected_sim_run_ids"]:
        print("\nSelected simulation runs:")
        print(", ".join(selection["selected_sim_run_ids"]))
        print(f"Skipped simulation rows: {len(selection['skipped_sim_rows'])}")
        for row in selection["skipped_sim_rows"]:
            print(f"  row {row['row_number']} {row['run_id']}: {row['reason']}")

    print("\nAbsolute endpoint model:")
    print(f"  n = {endpoint['n']}")
    print(f"  mu = [{endpoint['mu'][0]:.6f}, {endpoint['mu'][1]:.6f}] m")
    print(f"  sigma = {format_matrix(endpoint['sigma'])} m^2")
    print(f"  std = [{endpoint['std'][0]:.6f}, {endpoint['std'][1]:.6f}] m")
    print(
        "  95% ellipse axes = "
        f"{endpoint['ellipse_95']['major_axis_length_m']:.6f} m x "
        f"{endpoint['ellipse_95']['minor_axis_length_m']:.6f} m"
    )

    print("\nMotion-primitive error model:")
    print(f"  step_distance_m = {motion['step_distance_m']:.3f}")
    print(
        f"  mu_error = [{motion['mu_error'][0]:.6f}, "
        f"{motion['mu_error'][1]:.6f}] m"
    )
    print(f"  sigma_error = {format_matrix(motion['sigma_error'])} m^2")
    print(
        "  95% ellipse axes = "
        f"{motion['ellipse_95']['major_axis_length_m']:.6f} m x "
        f"{motion['ellipse_95']['minor_axis_length_m']:.6f} m"
    )

    print("\nYaw summary:")
    print(f"  mean = {yaw['mean_deg']:.3f} deg")
    print(f"  std = {yaw['std_deg']:.3f} deg")

    if bias["dx_m"] is not None:
        print("\nSim2Real local displacement-error bias:")
        print(
            f"  dx={bias['dx_m']:.6f} m, dy={bias['dy_m']:.6f} m, "
            f"magnitude={bias['magnitude_m']:.6f} m"
        )

    endpoint_outliers = endpoint["outliers_95"]
    motion_outliers = motion["outliers_95"]
    print("\nOutliers:")
    if endpoint_outliers:
        print(f"  Endpoint 95% ellipse outliers: {endpoint_outliers}")
    else:
        print("  No endpoint 95% ellipse outliers.")
    if motion_outliers:
        print(f"  Motion-error 95% ellipse outliers: {motion_outliers}")
    else:
        print("  No motion-error 95% ellipse outliers.")

    for warning in model["warnings"]:
        print(f"\nWARNING: {warning}")

    print("\nInterpretation:")
    print(
        "  The endpoint model describes the repeated 30 cm real-run final "
        "positions in the tracker frame. The path prediction should use the "
        "local motion-primitive error model, which assumes independent segment "
        "errors and does not model turn uncertainty at waypoint corners."
    )


def format_matrix(matrix):
    return (
        "["
        f"[{matrix[0][0]:.8f}, {matrix[0][1]:.8f}], "
        f"[{matrix[1][0]:.8f}, {matrix[1][1]:.8f}]"
        "]"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build empirical endpoint and motion-error models.",
    )
    parser.add_argument(
        "--real-csv",
        default="results/real_scripted_drive_runs.csv",
    )
    parser.add_argument("--real-run-range", default="21:50")
    parser.add_argument("--sim-csv", default="results/scripted_drive_runs.csv")
    parser.add_argument("--sim-last-n", type=int, default=15)
    parser.add_argument("--step-distance-m", type=float, default=DEFAULT_STEP_DISTANCE_M)
    parser.add_argument("--compare-sim-real", action="store_true")
    parser.add_argument(
        "--output-json",
        default="results/probabilistic_endpoint_model.json",
    )
    parser.add_argument(
        "--summary-csv",
        default="results/probabilistic_endpoint_model_summary.csv",
    )
    parser.add_argument(
        "--endpoint-plot",
        default="results/real_endpoint_gaussian_ellipse.png",
    )
    parser.add_argument(
        "--motion-error-plot",
        default="results/real_motion_error_gaussian_ellipse.png",
    )
    parser.add_argument(
        "--sim-real-plot",
        default="results/sim_vs_real_displacement_error_scatter.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model, arrays = build_analysis_model(
        real_csv=args.real_csv,
        real_run_range=args.real_run_range,
        sim_csv=args.sim_csv,
        sim_last_n=args.sim_last_n,
        step_distance_m=args.step_distance_m,
        compare_sim_real=args.compare_sim_real,
    )

    write_json(args.output_json, model)
    write_summary_csv(args.summary_csv, model)
    plot_scatter_with_ellipse(
        arrays["real_endpoint_points"],
        arrays["endpoint_mu"],
        arrays["endpoint_sigma"],
        args.endpoint_plot,
        "Real final positions after 30 cm scripted drive",
        "x [m]",
        "y [m]",
        labels="real tracker final positions",
    )
    plot_scatter_with_ellipse(
        arrays["real_errors"],
        arrays["error_mu"],
        arrays["error_sigma"],
        args.motion_error_plot,
        "Real local motion error after 30 cm command",
        "forward error [m]",
        "lateral error [m]",
        labels="real local errors",
    )

    if args.compare_sim_real:
        plot_sim_vs_real_errors(
            arrays["real_errors"],
            arrays["sim_errors"],
            arrays["error_mu"],
            arrays["sim_error_mu"],
            args.sim_real_plot,
        )

    print_report(model)
    print("\nGenerated outputs:")
    for path in [
        args.output_json,
        args.summary_csv,
        args.endpoint_plot,
        args.motion_error_plot,
    ]:
        print(f"  {path}")
    if args.compare_sim_real:
        print(f"  {args.sim_real_plot}")

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
