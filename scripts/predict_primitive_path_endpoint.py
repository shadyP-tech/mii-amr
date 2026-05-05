#!/usr/bin/env python3
"""
Predict endpoint uncertainty by composing empirical motion primitives.

The predictor samples F30/CW90/CCW90 primitives from a JSON model created by
``build_motion_primitives_model.py`` and propagates the final pose with Monte
Carlo simulation.
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import analyze_probabilistic_endpoint_model as endpoint_model


VALIDATION_COLUMNS = [
    "timestamp",
    "run_id",
    "actions",
    "tracker_start_x",
    "tracker_start_y",
    "tracker_start_yaw_deg",
    "tracker_final_x",
    "tracker_final_y",
    "tracker_final_yaw_deg",
    "notes",
]


def parse_actions(text):
    actions = [action.strip().upper() for action in str(text or "").split(",")]
    actions = [action for action in actions if action]
    if not actions:
        raise ValueError("At least one action is required")
    return actions


def normalized_actions_text(actions):
    return ",".join(parse_actions(",".join(actions)))


def parse_pose(text):
    parts = [part.strip() for part in str(text or "").split(",")]
    if len(parts) != 3:
        raise ValueError("Start pose must use 'x,y,yaw_deg'")
    try:
        return [float(parts[0]), float(parts[1]), float(parts[2])]
    except ValueError as exc:
        raise ValueError("Start pose values must be numeric") from exc


def parse_fixed_points(text):
    if text is None or text == "":
        return []

    points = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 2:
            raise ValueError("Fixed points must use 'x,y;x,y;...' format")
        try:
            points.append([float(parts[0]), float(parts[1])])
        except ValueError as exc:
            raise ValueError(f"Invalid fixed point: {item}") from exc

    return points


def load_primitive_model(path):
    with Path(path).open() as file:
        data = json.load(file)

    try:
        primitives = data["primitives"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Model JSON is missing primitives") from exc

    for name, primitive in primitives.items():
        validate_primitive_shape(name, primitive)

    return data


def validate_primitive_shape(name, primitive):
    required = [
        "local_delta_mu",
        "local_delta_sigma",
        "yaw_delta_mean_deg",
        "yaw_delta_std_deg",
    ]
    for key in required:
        if key not in primitive:
            raise ValueError(f"Primitive {name} is missing {key}")

    mu = primitive["local_delta_mu"]
    sigma = primitive["local_delta_sigma"]
    if len(mu) != 2:
        raise ValueError(f"Primitive {name} local_delta_mu must be 2D")
    if len(sigma) != 2 or any(len(row) != 2 for row in sigma):
        raise ValueError(f"Primitive {name} local_delta_sigma must be 2x2")

    values = [
        float(mu[0]),
        float(mu[1]),
        float(sigma[0][0]),
        float(sigma[0][1]),
        float(sigma[1][0]),
        float(sigma[1][1]),
        float(primitive["yaw_delta_mean_deg"]),
        float(primitive["yaw_delta_std_deg"]),
    ]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Primitive {name} contains non-finite values")
    if float(primitive["yaw_delta_std_deg"]) < 0.0:
        raise ValueError(f"Primitive {name} yaw std must be non-negative")


def sample_gaussian_2d(mu, sigma, rng):
    eigvals, eigvecs = endpoint_model.symmetric_eigen_2x2(sigma)
    z1 = rng.gauss(0.0, 1.0)
    z2 = rng.gauss(0.0, 1.0)
    result = [float(mu[0]), float(mu[1])]

    for z, eigval, eigvec in zip([z1, z2], eigvals, eigvecs):
        scale = math.sqrt(max(float(eigval), 0.0)) * z
        result[0] += scale * eigvec[0]
        result[1] += scale * eigvec[1]

    return result


def apply_primitive_to_pose(pose, primitive, rng):
    x, y, yaw_deg = pose
    local_delta = sample_gaussian_2d(
        primitive["local_delta_mu"],
        primitive["local_delta_sigma"],
        rng,
    )
    yaw_delta = rng.gauss(
        float(primitive["yaw_delta_mean_deg"]),
        float(primitive["yaw_delta_std_deg"]),
    )
    rotation = endpoint_model.rotation_matrix(math.radians(yaw_deg))
    world_delta = endpoint_model.mat_vec(rotation, local_delta)

    return [
        x + world_delta[0],
        y + world_delta[1],
        endpoint_model.normalize_angle_deg(yaw_deg + yaw_delta),
    ]


def mean_path_points(actions, primitives, start_pose):
    pose = list(start_pose)
    points = [[pose[0], pose[1]]]

    for action in actions:
        primitive = primitives[action]
        rotation = endpoint_model.rotation_matrix(math.radians(pose[2]))
        world_delta = endpoint_model.mat_vec(
            rotation,
            primitive["local_delta_mu"],
        )
        pose = [
            pose[0] + world_delta[0],
            pose[1] + world_delta[1],
            endpoint_model.normalize_angle_deg(
                pose[2] + primitive["yaw_delta_mean_deg"]
            ),
        ]
        points.append([pose[0], pose[1]])

    return points


def empirical_mean_cov_or_zero(points):
    points = endpoint_model.as_points(points)
    if len(points) == 1:
        return points[0], [[0.0, 0.0], [0.0, 0.0]]
    return endpoint_model.empirical_mean_cov(points)


def predict_action_sequence(model, actions, start_pose, samples, seed):
    if samples <= 0:
        raise ValueError("--samples must be greater than zero")

    primitives = model["primitives"]
    missing = [action for action in actions if action not in primitives]
    if missing:
        raise ValueError(f"Unknown action(s): {', '.join(missing)}")

    rng = random.Random(seed)
    final_poses = []
    for _ in range(samples):
        pose = list(start_pose)
        for action in actions:
            pose = apply_primitive_to_pose(pose, primitives[action], rng)
        final_poses.append(pose)

    final_points = [[pose[0], pose[1]] for pose in final_poses]
    final_yaws = [pose[2] for pose in final_poses]
    mu, sigma = empirical_mean_cov_or_zero(final_points)
    yaw_summary = endpoint_model.circular_yaw_summary_deg(final_yaws)

    return {
        "final_poses": final_poses,
        "final_points": final_points,
        "final_yaws": final_yaws,
        "endpoint_mu": mu,
        "endpoint_sigma": sigma,
        "endpoint_std": endpoint_model.matrix_std(sigma),
        "yaw_summary": yaw_summary,
        "mean_path_points": mean_path_points(actions, primitives, start_pose),
    }


def load_validation_row(path, run_id, expected_actions):
    if path is None and run_id is None:
        return None
    if path is None or run_id is None:
        raise ValueError("--validation-csv and --validation-run-id must be used together")

    fieldnames, rows = endpoint_model.read_csv_rows(path)
    endpoint_model.require_columns(fieldnames, VALIDATION_COLUMNS, path)

    for row in rows:
        if row.get("run_id") != run_id:
            continue

        warning = None
        row_actions = normalized_actions_text([row.get("actions", "")])
        expected = normalized_actions_text(expected_actions)
        if row_actions != expected:
            warning = (
                f"validation actions {row_actions!r} do not match "
                f"prediction actions {expected!r}"
            )

        return {
            "run_id": row["run_id"],
            "actions": row.get("actions", ""),
            "tracker_start_pose": [
                endpoint_model.finite_float(row, "tracker_start_x"),
                endpoint_model.finite_float(row, "tracker_start_y"),
                endpoint_model.finite_float(row, "tracker_start_yaw_deg"),
            ],
            "tracker_final_pose": [
                endpoint_model.finite_float(row, "tracker_final_x"),
                endpoint_model.finite_float(row, "tracker_final_y"),
                endpoint_model.finite_float(row, "tracker_final_yaw_deg"),
            ],
            "notes": row.get("notes", ""),
            "warning": warning,
        }

    raise ValueError(f"Validation run_id {run_id!r} was not found in {path}")


def validation_metrics(validation, endpoint_mu, endpoint_sigma):
    if validation is None:
        return None

    final_xy = validation["tracker_final_pose"][:2]
    residual = endpoint_model.vec_sub(final_xy, endpoint_mu)
    mahalanobis = endpoint_model.mahalanobis_squared(
        [final_xy],
        endpoint_mu,
        endpoint_sigma,
    )[0]

    result = dict(validation)
    result.update(
        {
            "residual_xy_m": residual,
            "residual_magnitude_m": math.hypot(residual[0], residual[1]),
            "mahalanobis_squared": mahalanobis,
            "inside_95_endpoint_ellipse": mahalanobis <= endpoint_model.CHI2_95_2D,
        }
    )
    return result


def build_output_model(
    model_path,
    actions,
    start_pose,
    fixed_points,
    samples,
    seed,
    prediction,
    validation=None,
):
    ellipse = endpoint_model.ellipse_parameters(
        prediction["endpoint_mu"],
        prediction["endpoint_sigma"],
    )
    validation = validation_metrics(
        validation,
        prediction["endpoint_mu"],
        prediction["endpoint_sigma"],
    )

    return {
        "units": {
            "position": "m",
            "angle": "deg",
            "covariance": "m^2",
        },
        "model": str(model_path),
        "actions": actions,
        "start_pose": {
            "x": start_pose[0],
            "y": start_pose[1],
            "yaw_deg": start_pose[2],
        },
        "fixed_points": fixed_points,
        "monte_carlo": {
            "samples": samples,
            "seed": seed,
        },
        "prediction": {
            "endpoint_mu": prediction["endpoint_mu"],
            "endpoint_sigma": prediction["endpoint_sigma"],
            "endpoint_std": prediction["endpoint_std"],
            "endpoint_ellipse_95": ellipse,
            "final_yaw_mean_deg": prediction["yaw_summary"]["mean_deg"],
            "final_yaw_std_deg": prediction["yaw_summary"]["std_deg"],
            "mean_path_points": prediction["mean_path_points"],
        },
        "validation": validation,
        "assumptions": [
            "Primitive samples are independent.",
            "Yaw uncertainty is sampled separately from x/y displacement.",
            "The action sequence approximates the fixed-point path.",
        ],
    }


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def write_summary_csv(path, output):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = output["prediction"]
    sigma = pred["endpoint_sigma"]
    ellipse = pred["endpoint_ellipse_95"]
    rows = [
        ("actions", ",".join(output["actions"]), ""),
        ("samples", output["monte_carlo"]["samples"], "count"),
        ("seed", output["monte_carlo"]["seed"], ""),
        ("endpoint_mu_x", pred["endpoint_mu"][0], "m"),
        ("endpoint_mu_y", pred["endpoint_mu"][1], "m"),
        ("endpoint_sigma_xx", sigma[0][0], "m^2"),
        ("endpoint_sigma_xy", sigma[0][1], "m^2"),
        ("endpoint_sigma_yy", sigma[1][1], "m^2"),
        ("endpoint_std_x", pred["endpoint_std"][0], "m"),
        ("endpoint_std_y", pred["endpoint_std"][1], "m"),
        ("endpoint_ellipse_95_major_axis", ellipse["major_axis_length_m"], "m"),
        ("endpoint_ellipse_95_minor_axis", ellipse["minor_axis_length_m"], "m"),
        ("final_yaw_mean", pred["final_yaw_mean_deg"], "deg"),
        ("final_yaw_std", pred["final_yaw_std_deg"], "deg"),
    ]

    validation = output["validation"]
    if validation is not None:
        rows.extend(
            [
                ("validation_run_id", validation["run_id"], ""),
                (
                    "validation_residual_magnitude",
                    validation["residual_magnitude_m"],
                    "m",
                ),
                (
                    "validation_mahalanobis_squared",
                    validation["mahalanobis_squared"],
                    "",
                ),
                (
                    "validation_inside_95_endpoint_ellipse",
                    validation["inside_95_endpoint_ellipse"],
                    "bool",
                ),
            ]
        )

    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value", "unit"])
        writer.writerows(rows)


def plot_prediction(prediction, output, plot_path):
    fixed_points = output["fixed_points"]
    validation = output["validation"]
    sampled_points = prediction["final_points"]
    endpoint_mu = prediction["endpoint_mu"]
    endpoint_sigma = prediction["endpoint_sigma"]
    mean_path = prediction["mean_path_points"]
    ellipse = output["prediction"]["endpoint_ellipse_95"]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
    except ModuleNotFoundError:
        groups = [
            (sampled_points, (120, 150, 210)),
            ([endpoint_mu], (180, 40, 40)),
        ]
        if validation is not None:
            groups.append(([validation["tracker_final_pose"][:2]], (40, 130, 60)))
        polylines = [(mean_path, (180, 40, 40))]
        if fixed_points:
            polylines.append((fixed_points, (46, 92, 170)))
        endpoint_model.write_fallback_plot(
            plot_path,
            groups=groups,
            ellipses=[(endpoint_mu, endpoint_sigma, (20, 130, 60))],
            polylines=polylines,
        )
        return

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    if fixed_points:
        ax.plot(
            [point[0] for point in fixed_points],
            [point[1] for point in fixed_points],
            marker="o",
            label="fixed-point path",
        )
    ax.plot(
        [point[0] for point in mean_path],
        [point[1] for point in mean_path],
        marker="+",
        label="primitive mean path",
    )
    ax.scatter(
        [point[0] for point in sampled_points],
        [point[1] for point in sampled_points],
        s=8,
        alpha=0.18,
        label="sampled final endpoints",
    )
    ax.scatter(
        [endpoint_mu[0]],
        [endpoint_mu[1]],
        marker="x",
        s=110,
        label="predicted mean final",
    )
    if validation is not None:
        final_pose = validation["tracker_final_pose"]
        ax.scatter(
            [final_pose[0]],
            [final_pose[1]],
            marker="*",
            s=140,
            label="measured validation final",
        )

    patch = Ellipse(
        xy=endpoint_mu,
        width=ellipse["major_axis_length_m"],
        height=ellipse["minor_axis_length_m"],
        angle=ellipse["orientation_deg"],
        fill=False,
        linewidth=2,
        label="95% endpoint ellipse",
    )
    ax.add_patch(patch)
    ax.set_title("Primitive path endpoint prediction")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_report(output):
    pred = output["prediction"]
    print("Primitive path endpoint prediction:")
    print(f"  actions = {','.join(output['actions'])}")
    print(f"  samples = {output['monte_carlo']['samples']}")
    print(
        "  endpoint_mu = "
        f"[{pred['endpoint_mu'][0]:.6f}, {pred['endpoint_mu'][1]:.6f}] m"
    )
    print(
        "  endpoint_sigma = "
        f"{endpoint_model.format_matrix(pred['endpoint_sigma'])} m^2"
    )
    print(
        "  95% ellipse axes = "
        f"{pred['endpoint_ellipse_95']['major_axis_length_m']:.6f} m x "
        f"{pred['endpoint_ellipse_95']['minor_axis_length_m']:.6f} m"
    )
    print(
        "  final_yaw = "
        f"{pred['final_yaw_mean_deg']:.3f} ± "
        f"{pred['final_yaw_std_deg']:.3f} deg"
    )

    validation = output["validation"]
    if validation is not None:
        print("\nValidation endpoint:")
        print(f"  run_id = {validation['run_id']}")
        print(
            "  residual = "
            f"{validation['residual_magnitude_m']:.6f} m, "
            f"inside_95={validation['inside_95_endpoint_ellipse']}"
        )
        if validation["warning"]:
            print(f"  WARNING: {validation['warning']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict a primitive action-sequence endpoint region.",
    )
    parser.add_argument(
        "--model",
        default="results/probabilistic_motion_primitives_model.json",
    )
    parser.add_argument("--actions", required=True)
    parser.add_argument("--start-pose", default="0,0,0")
    parser.add_argument("--fixed-points", default=None)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-json",
        default="results/primitive_path_prediction.json",
    )
    parser.add_argument(
        "--summary-csv",
        default="results/primitive_path_prediction_summary.csv",
    )
    parser.add_argument(
        "--plot",
        default="results/primitive_path_prediction.png",
    )
    parser.add_argument("--validation-csv", default=None)
    parser.add_argument("--validation-run-id", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_primitive_model(args.model)
    actions = parse_actions(args.actions)
    start_pose = parse_pose(args.start_pose)
    fixed_points = parse_fixed_points(args.fixed_points)
    validation = load_validation_row(
        args.validation_csv,
        args.validation_run_id,
        actions,
    )

    prediction = predict_action_sequence(
        model,
        actions,
        start_pose,
        args.samples,
        args.seed,
    )
    output = build_output_model(
        args.model,
        actions,
        start_pose,
        fixed_points,
        args.samples,
        args.seed,
        prediction,
        validation=validation,
    )

    write_json(args.output_json, output)
    write_summary_csv(args.summary_csv, output)
    plot_prediction(prediction, output, args.plot)
    print_report(output)
    print("\nGenerated outputs:")
    print(f"  {args.output_json}")
    print(f"  {args.summary_csv}")
    print(f"  {args.plot}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, endpoint_model.DataError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
    except OSError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
