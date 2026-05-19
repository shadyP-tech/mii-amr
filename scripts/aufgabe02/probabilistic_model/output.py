import csv
import json
from pathlib import Path


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
