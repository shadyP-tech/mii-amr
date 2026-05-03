#!/usr/bin/env python3
"""
Predict a rough final endpoint region for a fixed-coordinate waypoint path.

The prediction composes the local 30 cm forward-motion error model produced by
``analyze_probabilistic_endpoint_model.py``.  It does not model turn error at
waypoint corners.
"""

import argparse
import json
import math
from pathlib import Path

import analyze_probabilistic_endpoint_model as endpoint_model


CONFIDENCE_TO_CHI2_2D = {
    0.68: 2.30,
    0.95: endpoint_model.CHI2_95_2D,
    0.99: 9.21,
}

SEGMENT_TOLERANCE_M = 1e-6


def parse_waypoints(text):
    waypoints = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 2:
            raise ValueError("Waypoints must use 'x,y;x,y;...' format")
        try:
            waypoints.append([float(parts[0]), float(parts[1])])
        except ValueError as exc:
            raise ValueError(f"Invalid waypoint: {item}") from exc

    if len(waypoints) < 2:
        raise ValueError("At least two waypoints are required")

    return waypoints


def chi2_for_confidence(confidence):
    for known, value in CONFIDENCE_TO_CHI2_2D.items():
        if abs(confidence - known) < 1e-9:
            return value
    raise ValueError("Supported confidence values are 0.68, 0.95, and 0.99")


def segment_primitives(
    waypoints,
    step_distance_m,
    allow_remainder_scaling=False,
    tolerance_m=SEGMENT_TOLERANCE_M,
):
    if step_distance_m <= 0.0:
        raise ValueError("step_distance_m must be positive")

    primitives = []
    remainder_segments = []

    for index in range(len(waypoints) - 1):
        start = waypoints[index]
        end = waypoints[index + 1]
        delta = endpoint_model.vec_sub(end, start)
        length = math.hypot(delta[0], delta[1])
        if length <= tolerance_m:
            raise ValueError(f"Waypoint segment {index + 1} has zero length")

        theta = math.atan2(delta[1], delta[0])
        full_steps = int(math.floor((length + tolerance_m) / step_distance_m))
        remainder = length - full_steps * step_distance_m

        if abs(remainder) <= tolerance_m:
            remainder = 0.0
        elif abs(remainder - step_distance_m) <= tolerance_m:
            full_steps += 1
            remainder = 0.0

        for _ in range(full_steps):
            primitives.append(
                {
                    "segment_index": index,
                    "theta_rad": theta,
                    "scale": 1.0,
                    "distance_m": step_distance_m,
                }
            )

        if remainder > tolerance_m:
            if not allow_remainder_scaling:
                raise ValueError(
                    f"Waypoint segment {index + 1} length {length:.6f} m is not "
                    f"divisible by step distance {step_distance_m:.6f} m. "
                    "Use --allow-remainder-scaling to enable the extra approximation."
                )

            scale = remainder / step_distance_m
            primitives.append(
                {
                    "segment_index": index,
                    "theta_rad": theta,
                    "scale": scale,
                    "distance_m": remainder,
                }
            )
            remainder_segments.append(
                {
                    "segment_index": index,
                    "remainder_m": remainder,
                    "scale": scale,
                }
            )

    return primitives, remainder_segments


def predict_endpoint_region(
    waypoints,
    mu_error,
    sigma_error,
    step_distance_m,
    allow_remainder_scaling=False,
):
    primitives, remainder_segments = segment_primitives(
        waypoints,
        step_distance_m,
        allow_remainder_scaling=allow_remainder_scaling,
    )

    accumulated_mu = [0.0, 0.0]
    accumulated_sigma = [[0.0, 0.0], [0.0, 0.0]]

    for primitive in primitives:
        rotation = endpoint_model.rotation_matrix(primitive["theta_rad"])
        scale = primitive["scale"]
        local_mu = endpoint_model.vec_scale(scale, mu_error)
        local_sigma = endpoint_model.mat_scale(scale, sigma_error)
        accumulated_mu = endpoint_model.vec_add(
            accumulated_mu,
            endpoint_model.mat_vec(rotation, local_mu),
        )
        accumulated_sigma = endpoint_model.mat_add(
            accumulated_sigma,
            endpoint_model.mat_mul(
                endpoint_model.mat_mul(rotation, local_sigma),
                endpoint_model.mat_transpose(rotation),
            ),
        )

    nominal_final = list(waypoints[-1])
    predicted_mu = endpoint_model.vec_add(nominal_final, accumulated_mu)

    return {
        "nominal_final": nominal_final,
        "predicted_mu": predicted_mu,
        "sigma": accumulated_sigma,
        "primitive_count": len(primitives),
        "remainder_segments": remainder_segments,
    }


def load_motion_model(path):
    with Path(path).open() as file:
        data = json.load(file)

    try:
        motion = data["motion_primitive_error_model"]
        mu_error = [float(value) for value in motion["mu_error"]]
        sigma_error = [
            [float(motion["sigma_error"][0][0]), float(motion["sigma_error"][0][1])],
            [float(motion["sigma_error"][1][0]), float(motion["sigma_error"][1][1])],
        ]
        step_distance_m = float(motion["step_distance_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Model JSON is missing the motion_primitive_error_model") from exc

    if len(mu_error) != 2:
        raise ValueError("Motion-primitive model must contain a 2D mean")

    return data, mu_error, sigma_error, step_distance_m


def plot_waypoint_region(waypoints, prediction, plot_path, confidence):
    chi2_value = chi2_for_confidence(confidence)
    params = endpoint_model.ellipse_parameters(
        prediction["predicted_mu"],
        prediction["sigma"],
        chi2_value=chi2_value,
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
    except ModuleNotFoundError:
        endpoint_model.write_fallback_plot(
            plot_path,
            groups=[
                ([prediction["nominal_final"]], (180, 40, 40)),
                ([prediction["predicted_mu"]], (40, 130, 60)),
            ],
            ellipses=[(prediction["predicted_mu"], prediction["sigma"], (20, 130, 60))],
            polylines=[(waypoints, (46, 92, 170))],
        )
        return params

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(
        [point[0] for point in waypoints],
        [point[1] for point in waypoints],
        marker="o",
        label="nominal path",
    )
    ax.scatter(
        [prediction["nominal_final"][0]],
        [prediction["nominal_final"][1]],
        marker="x",
        s=100,
        label="nominal final",
    )
    ax.scatter(
        [prediction["predicted_mu"][0]],
        [prediction["predicted_mu"][1]],
        marker="+",
        s=120,
        label="predicted mean final",
    )

    ellipse = Ellipse(
        xy=prediction["predicted_mu"],
        width=params["major_axis_length_m"],
        height=params["minor_axis_length_m"],
        angle=params["orientation_deg"],
        fill=False,
        linewidth=2,
        label=f"{int(confidence * 100)}% endpoint ellipse",
    )
    ax.add_patch(ellipse)

    ax.set_title("Predicted endpoint region for waypoint path")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return params


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict a rough endpoint region for a waypoint path.",
    )
    parser.add_argument(
        "--model",
        default="results/probabilistic_endpoint_model.json",
    )
    parser.add_argument(
        "--waypoints",
        required=True,
        help="Waypoint list formatted as 'x,y;x,y;...'.",
    )
    parser.add_argument("--step-distance-m", type=float, default=None)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--allow-remainder-scaling",
        action="store_true",
        help="Scale the 30 cm error model for non-multiple segment remainders.",
    )
    parser.add_argument(
        "--plot",
        default="results/waypoint_endpoint_region.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_data, mu_error, sigma_error, model_step_distance_m = load_motion_model(
        args.model
    )
    waypoints = parse_waypoints(args.waypoints)
    step_distance_m = (
        args.step_distance_m
        if args.step_distance_m is not None
        else model_step_distance_m
    )

    prediction = predict_endpoint_region(
        waypoints,
        mu_error,
        sigma_error,
        step_distance_m,
        allow_remainder_scaling=args.allow_remainder_scaling,
    )
    ellipse = plot_waypoint_region(
        waypoints,
        prediction,
        args.plot,
        args.confidence,
    )

    warning = endpoint_model.covariance_warning(prediction["sigma"])
    if warning:
        print(f"WARNING: {warning}")

    print("Waypoint endpoint prediction:")
    print(
        "  nominal_final = "
        f"[{prediction['nominal_final'][0]:.6f}, "
        f"{prediction['nominal_final'][1]:.6f}] m"
    )
    print(
        "  predicted_mean_final = "
        f"[{prediction['predicted_mu'][0]:.6f}, "
        f"{prediction['predicted_mu'][1]:.6f}] m"
    )
    print(
        "  sigma_final = "
        f"{endpoint_model.format_matrix(prediction['sigma'])} m^2"
    )
    print(
        "  95% ellipse axes = "
        f"{ellipse['major_axis_length_m']:.6f} m x "
        f"{ellipse['minor_axis_length_m']:.6f} m"
    )
    print(f"  primitive_count = {prediction['primitive_count']}")
    if prediction["remainder_segments"]:
        print(
            "  Remainder scaling used; this is an extra approximation beyond "
            "the measured 30 cm command."
        )

    print(
        "  Limitation: this model propagates only the empirical forward-motion "
        "error and ignores turn/yaw uncertainty at waypoint corners."
    )
    print(f"Generated plot: {args.plot}")

    _ = model_data
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
