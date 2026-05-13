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


LEGACY_VALIDATION_COLUMNS = [
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

FINAL_ONLY_VALIDATION_COLUMNS = [
    "timestamp",
    "run_id",
    "actions",
    "tracker_final_x",
    "tracker_final_y",
    "tracker_final_yaw_deg",
    "notes",
]

MODEL_FRAME_FINAL_COLUMNS = [
    "tracker_final_x_model",
    "tracker_final_y_model",
    "tracker_final_yaw_model_deg",
]

MODEL_MIRROR_Y_FRAME = "model_mirror_y"
MODEL_FRAME = "model"
PHYSICAL_FRAME = "physical"
CONFIDENCE_LEVELS = [0.50, 0.68, 0.80, 0.95]
CONTOUR_STYLES = [
    (0.50, "#a7d7b5", 1.0),
    (0.68, "#6fba86", 1.2),
    (0.80, "#2f8a58", 1.5),
    (0.95, "#145c36", 2.2),
]
FALLBACK_CONTOUR_COLORS = [
    (167, 215, 181),
    (111, 186, 134),
    (47, 138, 88),
    (20, 92, 54),
]

SMALL_FONT = {
    " ": ["000", "000", "000", "000", "000", "000", "000"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    ".": ["000", "000", "000", "000", "000", "011", "011"],
    ",": ["000", "000", "000", "000", "000", "011", "010"],
    ":": ["000", "011", "011", "000", "011", "011", "000"],
    ";": ["000", "011", "011", "000", "011", "010", "100"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "=": ["00000", "11111", "00000", "11111", "00000", "00000", "00000"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    "%": ["11001", "11010", "00010", "00100", "01000", "01011", "10011"],
    "[": ["111", "100", "100", "100", "100", "100", "111"],
    "]": ["111", "001", "001", "001", "001", "001", "111"],
    "(": ["001", "010", "100", "100", "100", "010", "001"],
    ")": ["100", "010", "001", "001", "001", "010", "100"],
    "|": ["1", "1", "1", "1", "1", "1", "1"],
}


def parse_actions(text):
    actions = [action.strip().upper() for action in str(text or "").split(",")]
    actions = [action for action in actions if action]
    if not actions:
        raise ValueError("At least one action is required")
    return actions


def normalized_actions_text(actions):
    return ",".join(parse_actions(",".join(actions)))


def has_columns(fieldnames, columns):
    return all(column in fieldnames for column in columns)


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


def mirror_y_points(points):
    return [[point[0], -point[1]] for point in points]


def output_plot_frame(output):
    if output.get("execution_actions") != output.get("actions"):
        return PHYSICAL_FRAME

    validation = output.get("validation")
    if validation is not None:
        return validation.get("comparison_frame") or MODEL_FRAME

    return MODEL_FRAME


def fixed_points_frame(output):
    if output.get("execution_actions") != output.get("actions"):
        return PHYSICAL_FRAME
    return output_plot_frame(output)


def fixed_points_for_plot(output):
    fixed_points = output["fixed_points"]
    plot_frame = output_plot_frame(output)
    source_frame = fixed_points_frame(output)

    if source_frame == PHYSICAL_FRAME and plot_frame == MODEL_MIRROR_Y_FRAME:
        return mirror_y_points(fixed_points)

    return fixed_points


def should_mirror_model_to_physical(output):
    plot_frame = output.get("plot_frame", output_plot_frame(output))
    return (
        output.get("execution_actions") != output.get("actions")
        and plot_frame == PHYSICAL_FRAME
    )


def model_points_for_plot(points, output):
    if should_mirror_model_to_physical(output):
        return mirror_y_points(points)
    return points


def model_point_for_plot(point, output):
    return model_points_for_plot([point], output)[0]


def model_sigma_for_plot(sigma, output):
    if not should_mirror_model_to_physical(output):
        return sigma
    return [
        [float(sigma[0][0]), -float(sigma[0][1])],
        [-float(sigma[1][0]), float(sigma[1][1])],
    ]


def validation_pose_for_plot(validation, output):
    if validation is None:
        return None

    plot_frame = output.get("plot_frame", output_plot_frame(output))
    if plot_frame == PHYSICAL_FRAME:
        return validation.get("tracker_final_pose_raw") or validation["tracker_final_pose"]

    return validation["tracker_final_pose"]


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
    if has_columns(fieldnames, LEGACY_VALIDATION_COLUMNS):
        has_start_pose = True
    else:
        endpoint_model.require_columns(fieldnames, FINAL_ONLY_VALIDATION_COLUMNS, path)
        has_start_pose = False
    has_model_frame = has_columns(fieldnames, MODEL_FRAME_FINAL_COLUMNS)

    for row in rows:
        if row.get("run_id") != run_id:
            continue

        warning = None
        row_actions_source = row.get("model_actions") or row.get("actions", "")
        row_actions = normalized_actions_text([row_actions_source])
        expected = normalized_actions_text(expected_actions)
        if row_actions != expected:
            warning = (
                f"validation actions {row_actions!r} do not match "
                f"prediction actions {expected!r}"
            )

        raw_final_pose = [
            endpoint_model.finite_float(row, "tracker_final_x"),
            endpoint_model.finite_float(row, "tracker_final_y"),
            endpoint_model.finite_float(row, "tracker_final_yaw_deg"),
        ]
        if has_model_frame:
            tracker_final_pose = [
                endpoint_model.finite_float(row, "tracker_final_x_model"),
                endpoint_model.finite_float(row, "tracker_final_y_model"),
                endpoint_model.finite_float(row, "tracker_final_yaw_model_deg"),
            ]
            comparison_frame = row.get("comparison_frame") or "model_mirror_y"
        else:
            tracker_final_pose = raw_final_pose
            comparison_frame = "raw_tracker"

        return {
            "run_id": row["run_id"],
            "actions": row.get("actions", ""),
            "model_actions": row.get("model_actions", ""),
            "tracker_start_pose": (
                [
                    endpoint_model.finite_float(row, "tracker_start_x"),
                    endpoint_model.finite_float(row, "tracker_start_y"),
                    endpoint_model.finite_float(row, "tracker_start_yaw_deg"),
                ]
                if has_start_pose
                else None
            ),
            "tracker_final_pose": tracker_final_pose,
            "tracker_final_pose_raw": raw_final_pose,
            "comparison_frame": comparison_frame,
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
    execution_actions,
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
    frame_context = {
        "actions": actions,
        "execution_actions": execution_actions,
        "fixed_points": fixed_points,
        "validation": validation,
    }
    plot_frame = output_plot_frame(frame_context)
    fixed_points_source_frame = fixed_points_frame(frame_context)

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
        "fixed_points_frame": fixed_points_source_frame,
        "plot_frame": plot_frame,
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
            "actions are model-frame labels; execution_actions are physical route commands.",
            "Validation tracker_final_pose uses the comparison_frame reported by the CSV.",
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
        ("execution_actions", ",".join(output["execution_actions"]), ""),
        ("plot_frame", output.get("plot_frame", output_plot_frame(output)), ""),
        (
            "fixed_points_frame",
            output.get("fixed_points_frame", fixed_points_frame(output)),
            "",
        ),
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
                    "validation_comparison_frame",
                    validation.get("comparison_frame", "raw_tracker"),
                    "",
                ),
                ("validation_residual_x", validation["residual_xy_m"][0], "m"),
                ("validation_residual_y", validation["residual_xy_m"][1], "m"),
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
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["metric", "value", "unit"])
        writer.writerows(rows)


def confidence_chi2_value(confidence):
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if abs(confidence - 0.95) < 1e-12:
        return endpoint_model.CHI2_95_2D
    return -2.0 * math.log(1.0 - confidence)


def confidence_contours(mu, sigma):
    contours = []
    for level in CONFIDENCE_LEVELS:
        contours.append(
            {
                "level": level,
                "label": f"{int(round(level * 100))}%",
                "ellipse": endpoint_model.ellipse_parameters(
                    mu,
                    sigma,
                    chi2_value=confidence_chi2_value(level),
                ),
            }
        )
    return contours


def axis_labels_for_plot_frame(plot_frame):
    x_label = "x distance from start [m]"
    if plot_frame == MODEL_MIRROR_Y_FRAME:
        y_label = "lateral drift y, mirrored model frame [m]"
    elif plot_frame == PHYSICAL_FRAME:
        y_label = "lateral drift y, physical frame [m]"
    else:
        y_label = "lateral drift y [m]"
    return x_label, y_label


def display_frame_name(plot_frame):
    return str(plot_frame).replace("_", " ")


def target_endpoint_for_plot(output):
    fixed_points = fixed_points_for_plot(output)
    if not fixed_points:
        return None
    return fixed_points[-1]


def draw_fallback_rect(pixels, x0, y0, x1, y1, color, fill=None):
    if fill is not None:
        height = len(pixels)
        width = len(pixels[0])
        for y in range(max(y0, 0), min(y1 + 1, height)):
            for x in range(max(x0, 0), min(x1 + 1, width)):
                pixels[y][x] = fill
    endpoint_model.draw_line(pixels, (x0, y0), (x1, y0), color)
    endpoint_model.draw_line(pixels, (x1, y0), (x1, y1), color)
    endpoint_model.draw_line(pixels, (x1, y1), (x0, y1), color)
    endpoint_model.draw_line(pixels, (x0, y1), (x0, y0), color)


def fallback_text_size(text, scale=2):
    width = 0
    for char in text.upper():
        glyph = SMALL_FONT.get(char, SMALL_FONT[" "])
        width += (len(glyph[0]) + 1) * scale
    if width:
        width -= scale
    return width, 7 * scale


def draw_fallback_text(pixels, x, y, text, color=(30, 30, 30), scale=2):
    cursor = int(x)
    for char in text.upper():
        glyph = SMALL_FONT.get(char, SMALL_FONT[" "])
        for row_index, row in enumerate(glyph):
            for col_index, value in enumerate(row):
                if value != "1":
                    continue
                draw_fallback_rect(
                    pixels,
                    cursor + col_index * scale,
                    y + row_index * scale,
                    cursor + (col_index + 1) * scale - 1,
                    y + (row_index + 1) * scale - 1,
                    color,
                    fill=color,
                )
        cursor += (len(glyph[0]) + 1) * scale


def draw_fallback_centered_text(pixels, x, y, text, color=(30, 30, 30), scale=2):
    width, _height = fallback_text_size(text, scale=scale)
    draw_fallback_text(pixels, int(round(x - width / 2)), y, text, color, scale=scale)


def draw_fallback_cross(pixels, center, radius, color):
    cx, cy = center
    endpoint_model.draw_line(pixels, (cx - radius, cy), (cx + radius, cy), color)
    endpoint_model.draw_line(pixels, (cx, cy - radius), (cx, cy + radius), color)


def draw_fallback_diamond(pixels, center, radius, color):
    cx, cy = center
    points = [
        (cx, cy - radius),
        (cx + radius, cy),
        (cx, cy + radius),
        (cx - radius, cy),
    ]
    for start, end in zip(points, points[1:] + points[:1]):
        endpoint_model.draw_line(pixels, start, end, color)


def fallback_plot_transform(points, rect):
    left, top, right, bottom = rect
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    min_x -= 0.08 * span_x
    max_x += 0.08 * span_x
    min_y -= 0.18 * span_y
    max_y += 0.18 * span_y
    span_x = max_x - min_x
    span_y = max_y - min_y
    scale = min((right - left) / span_x, (bottom - top) / span_y)
    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    pixel_center_x = 0.5 * (left + right)
    pixel_center_y = 0.5 * (top + bottom)

    def to_pixel(point):
        x = int(round(pixel_center_x + (point[0] - center_x) * scale))
        y = int(round(pixel_center_y - (point[1] - center_y) * scale))
        return x, y

    return to_pixel, (min_x, max_x, min_y, max_y)


def nice_tick_step(span, target_count=6):
    rough = max(span / target_count, 1e-9)
    exponent = math.floor(math.log10(rough))
    fraction = rough / (10**exponent)
    if fraction <= 1.0:
        nice = 1.0
    elif fraction <= 2.0:
        nice = 2.0
    elif fraction <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return nice * (10**exponent)


def fallback_ticks(min_value, max_value, target_count=6):
    step = nice_tick_step(max_value - min_value, target_count=target_count)
    start = math.ceil(min_value / step) * step
    ticks = []
    value = start
    while value <= max_value + 1e-9:
        ticks.append(value)
        value += step
    return ticks


def draw_fallback_plot_axes(pixels, rect, to_pixel, bounds, plot_frame):
    left, top, right, bottom = rect
    min_x, max_x, min_y, max_y = bounds
    axis_color = (80, 80, 80)
    grid_color = (228, 228, 228)
    text_color = (45, 45, 45)
    draw_fallback_rect(pixels, left, top, right, bottom, (180, 180, 180))

    for tick in fallback_ticks(min_x, max_x, target_count=7):
        x, _ = to_pixel([tick, 0.0])
        endpoint_model.draw_line(pixels, (x, top), (x, bottom), grid_color)
        endpoint_model.draw_line(pixels, (x, bottom), (x, bottom + 6), axis_color)
        draw_fallback_centered_text(
            pixels,
            x,
            bottom + 11,
            f"{tick:.1f}",
            color=text_color,
            scale=1,
        )

    for tick in fallback_ticks(min_y, max_y, target_count=5):
        _, y = to_pixel([0.0, tick])
        endpoint_model.draw_line(pixels, (left, y), (right, y), grid_color)
        endpoint_model.draw_line(pixels, (left - 6, y), (left, y), axis_color)
        draw_fallback_text(
            pixels,
            left - 48,
            y - 4,
            f"{tick:.1f}",
            color=text_color,
            scale=1,
        )

    zero_x, zero_y = to_pixel([0.0, 0.0])
    if left <= zero_x <= right:
        endpoint_model.draw_line(pixels, (zero_x, top), (zero_x, bottom), (190, 190, 190))
    if top <= zero_y <= bottom:
        endpoint_model.draw_line(pixels, (left, zero_y), (right, zero_y), (190, 190, 190))

    x_label, y_label = axis_labels_for_plot_frame(plot_frame)
    draw_fallback_centered_text(
        pixels,
        0.5 * (left + right),
        bottom + 34,
        x_label,
        color=text_color,
        scale=2,
    )
    draw_fallback_text(pixels, left, top - 26, y_label, color=text_color, scale=2)


def draw_fallback_legend(pixels, x, y, has_target, has_validation):
    text_color = (45, 45, 45)
    draw_fallback_text(pixels, x, y, "LEGEND", color=text_color, scale=2)
    y += 28
    items = [
        ((205, 45, 45), "PREDICTED MEAN"),
        ((120, 150, 210), "SAMPLED ENDPOINTS"),
    ]
    if has_validation:
        items.append(((35, 125, 75), "VALIDATION FINAL"))
    if has_target:
        items.append(((46, 92, 170), "TARGET ENDPOINT"))
    for color, label in items:
        endpoint_model.draw_circle(pixels, (x + 8, y + 7), 5, color)
        draw_fallback_text(pixels, x + 24, y, label, color=text_color, scale=1)
        y += 20

    y += 6
    draw_fallback_text(pixels, x, y, "CONFIDENCE", color=text_color, scale=2)
    y += 26
    for level, color in zip(CONFIDENCE_LEVELS, FALLBACK_CONTOUR_COLORS):
        endpoint_model.draw_line(pixels, (x, y + 6), (x + 18, y + 6), color)
        draw_fallback_text(
            pixels,
            x + 24,
            y,
            f"{int(round(level * 100))}% ENDPOINT",
            color=text_color,
            scale=1,
        )
        y += 18


def plot_prediction_fallback(
    plot_path,
    sampled_points,
    endpoint_mu,
    endpoint_sigma,
    mean_path,
    fixed_points,
    validation_pose,
    plot_frame,
):
    width = 1220
    height = 760
    white = (255, 255, 255)
    pixels = [[white for _ in range(width)] for _ in range(height)]
    plot_rect = (88, 82, 850, 610)
    target_point = fixed_points[-1] if fixed_points else None
    contours = confidence_contours(endpoint_mu, endpoint_sigma)

    all_points = []
    all_points.extend(sampled_points)
    all_points.extend(mean_path)
    all_points.extend(fixed_points)
    all_points.append(endpoint_mu)
    if validation_pose is not None:
        all_points.append(validation_pose[:2])
    for contour in contours:
        all_points.extend(
            endpoint_model.ellipse_sample_points(
                endpoint_mu,
                endpoint_sigma,
                chi2_value=contour["ellipse"]["chi2_value"],
            )
        )
    if not all_points:
        all_points = [[0.0, 0.0], [1.0, 1.0]]

    to_pixel, bounds = fallback_plot_transform(all_points, plot_rect)
    draw_fallback_plot_axes(pixels, plot_rect, to_pixel, bounds, plot_frame)

    if fixed_points:
        fixed_pixels = [to_pixel(point) for point in fixed_points]
        for start, end in zip(fixed_pixels, fixed_pixels[1:]):
            endpoint_model.draw_line(pixels, start, end, (46, 92, 170))
        for point in fixed_pixels[:-1]:
            endpoint_model.draw_circle(pixels, point, 4, (46, 92, 170))

    mean_pixels = [to_pixel(point) for point in mean_path]
    for start, end in zip(mean_pixels, mean_pixels[1:]):
        endpoint_model.draw_line(pixels, start, end, (190, 55, 55))
    for point in mean_pixels:
        draw_fallback_cross(pixels, point, 4, (190, 55, 55))

    stride = max(len(sampled_points) // 1600, 1)
    for point in sampled_points[::stride]:
        px, py = to_pixel(point)
        if 0 <= px < width and 0 <= py < height:
            pixels[py][px] = (120, 150, 210)

    for contour, color in zip(contours, FALLBACK_CONTOUR_COLORS):
        points = endpoint_model.ellipse_sample_points(
            endpoint_mu,
            endpoint_sigma,
            chi2_value=contour["ellipse"]["chi2_value"],
        )
        contour_pixels = [to_pixel(point) for point in points]
        for start, end in zip(contour_pixels, contour_pixels[1:] + contour_pixels[:1]):
            endpoint_model.draw_line(pixels, start, end, color)

    if validation_pose is not None:
        endpoint_model.draw_circle(
            pixels,
            to_pixel(validation_pose[:2]),
            7,
            (35, 125, 75),
        )

    mean_pixel = to_pixel(endpoint_mu)
    endpoint_model.draw_circle(pixels, mean_pixel, 11, white)
    endpoint_model.draw_circle(pixels, mean_pixel, 8, (255, 196, 68))
    draw_fallback_cross(pixels, mean_pixel, 11, (120, 25, 25))

    if target_point is not None:
        target_pixel = to_pixel(target_point)
        endpoint_model.draw_circle(pixels, target_pixel, 13, white)
        draw_fallback_diamond(pixels, target_pixel, 11, (46, 92, 170))
        draw_fallback_text(
            pixels,
            target_pixel[0] - 84,
            target_pixel[1] - 26,
            "TARGET",
            color=(46, 92, 170),
            scale=1,
        )

    draw_fallback_text(
        pixels,
        plot_rect[0],
        24,
        f"FINAL ROUTE ENDPOINT PREDICTION ({display_frame_name(plot_frame)})",
        color=(30, 30, 30),
        scale=2,
    )
    draw_fallback_legend(
        pixels,
        888,
        94,
        target_point is not None,
        validation_pose is not None,
    )

    endpoint_model.write_png(plot_path, pixels)


def plot_prediction(prediction, output, plot_path):
    fixed_points = fixed_points_for_plot(output)
    plot_frame = output.get("plot_frame", output_plot_frame(output))
    validation = output["validation"]
    sampled_points = model_points_for_plot(prediction["final_points"], output)
    endpoint_mu = model_point_for_plot(prediction["endpoint_mu"], output)
    endpoint_sigma = model_sigma_for_plot(prediction["endpoint_sigma"], output)
    mean_path = model_points_for_plot(prediction["mean_path_points"], output)
    contours = confidence_contours(endpoint_mu, endpoint_sigma)
    target_point = target_endpoint_for_plot(output)
    validation_pose = validation_pose_for_plot(validation, output)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
    except ModuleNotFoundError:
        plot_prediction_fallback(
            plot_path,
            sampled_points,
            endpoint_mu,
            endpoint_sigma,
            mean_path,
            fixed_points,
            validation_pose,
            plot_frame,
        )
        return

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    fig.subplots_adjust(right=0.76)
    if fixed_points:
        fixed_points_label = "supervisor route"
        if output.get("fixed_points_frame") == PHYSICAL_FRAME:
            fixed_points_label += f" ({display_frame_name(plot_frame)})"
        ax.plot(
            [point[0] for point in fixed_points],
            [point[1] for point in fixed_points],
            color="#2e5da8",
            linewidth=1.6,
            label=fixed_points_label,
            zorder=2,
        )
        if len(fixed_points) > 1:
            ax.scatter(
                [point[0] for point in fixed_points[:-1]],
                [point[1] for point in fixed_points[:-1]],
                marker="o",
                s=32,
                color="#2e5da8",
                zorder=3,
            )
    ax.scatter(
        [point[0] for point in sampled_points],
        [point[1] for point in sampled_points],
        s=8,
        alpha=0.16,
        color="#7896d2",
        edgecolors="none",
        label="sampled final endpoints",
        zorder=1,
    )
    for contour, (_level, color, linewidth) in zip(contours, CONTOUR_STYLES):
        ellipse = contour["ellipse"]
        patch = Ellipse(
            xy=endpoint_mu,
            width=ellipse["major_axis_length_m"],
            height=ellipse["minor_axis_length_m"],
            angle=ellipse["orientation_deg"],
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            label=f"{contour['label']} confidence contour",
            zorder=4,
        )
        ax.add_patch(patch)
    ax.plot(
        [point[0] for point in mean_path],
        [point[1] for point in mean_path],
        marker="+",
        color="#b61f2a",
        linewidth=1.4,
        label="model prediction mean path",
        zorder=5,
    )
    ax.scatter(
        [endpoint_mu[0]],
        [endpoint_mu[1]],
        marker="o",
        s=280,
        facecolors="white",
        edgecolors="white",
        linewidths=1.0,
        zorder=6,
    )
    ax.scatter(
        [endpoint_mu[0]],
        [endpoint_mu[1]],
        marker="o",
        s=210,
        facecolors="#ffd166",
        edgecolors="#7a1e1e",
        linewidths=2.4,
        label="predicted mean final",
        zorder=7,
    )
    ax.scatter(
        [endpoint_mu[0]],
        [endpoint_mu[1]],
        marker="x",
        s=150,
        color="#7a1e1e",
        linewidths=2.2,
        zorder=8,
    )
    if validation_pose is not None:
        final_pose = validation_pose
        frame = PHYSICAL_FRAME if plot_frame == PHYSICAL_FRAME else validation.get(
            "comparison_frame",
            "raw_tracker",
        )
        ax.scatter(
            [final_pose[0]],
            [final_pose[1]],
            marker="*",
            s=170,
            color="#247a46",
            edgecolors="white",
            linewidths=0.8,
            label=f"measured validation final ({display_frame_name(frame)})",
            zorder=6,
        )

    if target_point is not None:
        ax.scatter(
            [target_point[0]],
            [target_point[1]],
            marker="D",
            s=260,
            facecolors="white",
            edgecolors="white",
            linewidths=1.0,
            zorder=9,
        )
        ax.scatter(
            [target_point[0]],
            [target_point[1]],
            marker="D",
            s=155,
            facecolors="white",
            edgecolors="#2e5da8",
            linewidths=2.8,
            label="target endpoint",
            zorder=10,
        )
        ax.annotate(
            "target endpoint",
            xy=target_point,
            xytext=(-74, 24),
            textcoords="offset points",
            fontsize=8,
            color="#2e5da8",
            ha="right",
            arrowprops={"arrowstyle": "-", "color": "#2e5da8", "linewidth": 0.8},
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "#d9e2f4",
                "alpha": 0.95,
            },
            zorder=11,
        )

    ax.annotate(
        "predicted mean",
        xy=endpoint_mu,
        xytext=(16, 16),
        textcoords="offset points",
        fontsize=8,
        color="#7a1e1e",
        arrowprops={"arrowstyle": "-", "color": "#7a1e1e", "linewidth": 0.8},
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#f3d2d2",
            "alpha": 0.95,
        },
        zorder=11,
    )

    x_label, y_label = axis_labels_for_plot_frame(plot_frame)
    ax.set_title(f"Final route endpoint prediction ({display_frame_name(plot_frame)})")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.axis("equal")
    ax.grid(True, color="#e2e2e2", linewidth=0.8)
    ax.minorticks_on()
    ax.grid(which="minor", color="#eeeeee", linewidth=0.5, alpha=0.55)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize=9,
        borderpad=0.9,
        labelspacing=1.05,
        handlelength=2.8,
        handleheight=1.8,
        handletextpad=0.9,
        markerscale=0.9,
    )
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_report(output):
    pred = output["prediction"]
    print("Primitive path endpoint prediction:")
    print(f"  actions = {','.join(output['actions'])}")
    if output["execution_actions"] != output["actions"]:
        print(f"  execution_actions = {','.join(output['execution_actions'])}")
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
        print(f"  comparison_frame = {validation.get('comparison_frame', 'raw_tracker')}")
        print(
            "  tracker_final_pose = "
            f"[{validation['tracker_final_pose'][0]:.6f}, "
            f"{validation['tracker_final_pose'][1]:.6f}, "
            f"{validation['tracker_final_pose'][2]:.3f} deg]"
        )
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
    parser.add_argument(
        "--execution-actions",
        default=None,
        help=(
            "Optional physical route actions to store for the real runner when "
            "they differ from the model-frame action labels."
        ),
    )
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
    execution_actions = (
        parse_actions(args.execution_actions)
        if args.execution_actions is not None
        else actions
    )
    if len(execution_actions) != len(actions):
        raise ValueError("--execution-actions must have the same length as --actions")
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
        execution_actions,
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
