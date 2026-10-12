#!/usr/bin/env python3
"""Predict a route's endpoint distribution from independent measured primitives."""
import argparse
import json
import math
import random
from pathlib import Path

if __package__:
    from .probabilistic_model import csv_data as data, geometry, statistics
else:
    from probabilistic_model import csv_data as data, geometry, statistics


def parse_actions(text):
    actions = [action.strip().upper() for action in str(text or "").split(",")]
    actions = [action for action in actions if action]
    if not actions:
        raise ValueError("At least one action is required")
    return actions


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
    eigvals, eigvecs = statistics.symmetric_eigen_2x2(sigma)
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
    rotation = geometry.rotation_matrix(math.radians(yaw_deg))
    world_delta = geometry.mat_vec(rotation, local_delta)

    return [
        x + world_delta[0],
        y + world_delta[1],
        geometry.normalize_angle_deg(yaw_deg + yaw_delta),
    ]


def mean_path_points(actions, primitives, start_pose):
    pose = list(start_pose)
    points = [[pose[0], pose[1]]]

    for action in actions:
        primitive = primitives[action]
        rotation = geometry.rotation_matrix(math.radians(pose[2]))
        world_delta = geometry.mat_vec(
            rotation,
            primitive["local_delta_mu"],
        )
        pose = [
            pose[0] + world_delta[0],
            pose[1] + world_delta[1],
            geometry.normalize_angle_deg(
                pose[2] + primitive["yaw_delta_mean_deg"]
            ),
        ]
        points.append([pose[0], pose[1]])

    return points


def empirical_mean_cov_or_zero(points):
    points = statistics.as_points(points)
    if len(points) == 1:
        return points[0], [[0.0, 0.0], [0.0, 0.0]]
    return statistics.empirical_mean_cov(points)


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
    yaw_summary = statistics.circular_yaw_summary_deg(final_yaws)

    return {
        "final_poses": final_poses,
        "final_points": final_points,
        "final_yaws": final_yaws,
        "endpoint_mu": mu,
        "endpoint_sigma": sigma,
        "endpoint_std": statistics.matrix_std(sigma),
        "yaw_summary": yaw_summary,
        "mean_path_points": mean_path_points(actions, primitives, start_pose),
    }


def build_output_model(
    model_path,
    actions,
    execution_actions,
    start_pose,
    fixed_points,
    samples,
    seed,
    prediction,
):
    ellipse = statistics.ellipse_parameters(
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
        "execution_actions": execution_actions,
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
        "assumptions": [
            "actions are model-frame labels; execution_actions are physical route commands.",
            "Primitive samples are independent.",
            "Yaw uncertainty is sampled separately from x/y displacement.",
            "The action sequence approximates the fixed-point path.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="results/aufgabe02/probabilistic_motion_primitives_model.json")
    parser.add_argument("--actions", required=True)
    parser.add_argument("--execution-actions", help="Physical actions when different from model-frame labels.")
    parser.add_argument("--start-pose", default="0,0,0")
    parser.add_argument("--fixed-points")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-json", default="results/aufgabe02/primitive_path_prediction.json")
    args = parser.parse_args()
    model = load_primitive_model(args.model)
    actions = parse_actions(args.actions)
    execution_actions = (
        parse_actions(args.execution_actions) if args.execution_actions is not None else actions
    )
    if len(execution_actions) != len(actions):
        raise ValueError("--execution-actions must have the same length as --actions")
    start_pose = parse_pose(args.start_pose)
    prediction = predict_action_sequence(model, actions, start_pose, args.samples, args.seed)
    output = build_output_model(
        args.model, actions, execution_actions, start_pose,
        parse_fixed_points(args.fixed_points), args.samples, args.seed, prediction,
    )
    data.write_json(args.output_json, output)
    print(args.output_json)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}")
