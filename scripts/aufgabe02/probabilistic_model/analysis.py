import math

from .constants import REAL_ENDPOINT_COLUMNS, SIM_ENDPOINT_COLUMNS
from .csv_data import (
    extract_points,
    extract_yaws,
    filter_latest_rows,
    filter_rows_by_run_range,
    read_csv_rows,
    require_columns,
    run_ids,
    valid_rows_with_columns,
)
from .errors import DataError
from .geometry import local_displacements, motion_errors, vec_sub
from .statistics import (
    circular_yaw_summary_deg,
    covariance_warning,
    empirical_mean_cov,
    ellipse_parameters,
    matrix_std,
    outlier_records,
)


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
