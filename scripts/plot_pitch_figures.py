#!/usr/bin/env python3
"""
Generate consistent presentation plots for Aufgabe 2 experiment data.

The plots are designed to match the visual style of the final route endpoint
prediction figure while staying fully reproducible from CSV/JSON files in
``results/``.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import analyze_probabilistic_endpoint_model as endpoint_model
import build_motion_primitives_model as primitive_builder


CONFIDENCE_LEVELS = [0.50, 0.68, 0.80, 0.95]
CONTOUR_STYLES = [
    (0.50, "#a7d7b5", 1.0),
    (0.68, "#6fba86", 1.2),
    (0.80, "#2f8a58", 1.5),
    (0.95, "#145c36", 2.2),
]

BLUE = "#2e5da8"
LIGHT_BLUE = "#7896d2"
RED = "#b61f2a"
DARK_RED = "#7a1e1e"
ORANGE = "#ffd166"
GREEN = "#247a46"
GRID = "#e2e2e2"
MINOR_GRID = "#eeeeee"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate pitch plots for the recorded Aufgabe 2 experiments.",
    )
    parser.add_argument(
        "--primitive-model",
        default="results/probabilistic_motion_primitives_model.json",
    )
    parser.add_argument(
        "--endpoint-model",
        default="results/probabilistic_endpoint_model.json",
    )
    parser.add_argument(
        "--start-pose-csv",
        default="results/real_start_pose_checks.csv",
    )
    parser.add_argument(
        "--real-forward-csv",
        default="results/real_scripted_drive_runs.csv",
    )
    parser.add_argument(
        "--sim-forward-csv",
        default="results/scripted_drive_runs.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
    )
    return parser.parse_args()


def require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse

    return plt, Circle, Ellipse


def read_csv_rows(path):
    path = Path(path)
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def read_json(path):
    path = Path(path)
    with path.open() as file:
        return json.load(file)


def finite(row, column):
    return endpoint_model.finite_float(row, column)


def rows_by_run_id(path, run_ids):
    run_id_set = set(run_ids)
    return [row for row in read_csv_rows(path) if row.get("run_id") in run_id_set]


def primitive_rows(model, name):
    source = model["data_selection"]["primitive_sources"][name]
    run_ids = source["selected_run_ids"]
    return rows_by_run_id(source["csv"], run_ids)


def local_deltas_and_yaws(rows):
    deltas = []
    yaws = []
    for row in rows:
        delta, yaw_delta = primitive_builder.pose_local_delta_and_yaw_delta(
            row,
            "tracker",
            tracker_yaw_sign=-1.0,
        )
        deltas.append(delta)
        yaws.append(yaw_delta)
    return deltas, yaws


def command_distance_m(name):
    if not name.startswith("F"):
        raise ValueError(f"{name} is not a forward primitive")
    return float(name[1:]) / 100.0


def commanded_angle_deg(name):
    if name.startswith("CCW"):
        return float(name[3:])
    if name.startswith("CW"):
        return -float(name[2:])
    raise ValueError(f"{name} is not a rotation primitive")


def chi2_for_confidence(confidence):
    if abs(confidence - 0.95) < 1e-12:
        return endpoint_model.CHI2_95_2D
    return -2.0 * math.log(1.0 - confidence)


def confidence_ellipse_params(mu, sigma):
    return [
        (level, endpoint_model.ellipse_parameters(mu, sigma, chi2_for_confidence(level)))
        for level in CONFIDENCE_LEVELS
    ]


def apply_common_axis_style(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.minorticks_on()
    ax.grid(which="minor", color=MINOR_GRID, linewidth=0.5, alpha=0.55)


def add_confidence_contours(ax, mu, sigma, Ellipse, label_prefix=""):
    for (level, params), (_style_level, color, linewidth) in zip(
        confidence_ellipse_params(mu, sigma),
        CONTOUR_STYLES,
    ):
        label = f"{label_prefix}{int(round(level * 100))}% confidence contour"
        patch = Ellipse(
            xy=mu,
            width=params["major_axis_length_m"],
            height=params["minor_axis_length_m"],
            angle=params["orientation_deg"],
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            label=label,
            zorder=4,
        )
        ax.add_patch(patch)


def add_outside_legend(fig, ax, fontsize=9):
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize=fontsize,
        borderpad=0.9,
        labelspacing=1.05,
        handlelength=2.8,
        handleheight=1.8,
        handletextpad=0.9,
        markerscale=0.9,
    )
    fig.subplots_adjust(right=0.75)


def dedupe_handles_labels(axes):
    handles = []
    labels = []
    seen = set()
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label in seen:
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)
    return handles, labels


def add_figure_legend(fig, axes, fontsize=9):
    handles, labels = dedupe_handles_labels(axes)
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.77, 0.93),
        frameon=True,
        fontsize=fontsize,
        borderpad=0.9,
        labelspacing=1.05,
        handlelength=2.8,
        handleheight=1.8,
        handletextpad=0.9,
        markerscale=0.9,
    )
    fig.subplots_adjust(right=0.74, wspace=0.34)


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")


def plot_start_pose_repeatability(start_pose_csv, output_path):
    plt, Circle, _Ellipse = require_matplotlib()
    rows = [
        row for row in read_csv_rows(start_pose_csv)
        if str(row.get("accepted", "")).strip() in {"1", "true", "True"}
    ]
    if not rows:
        raise ValueError(f"No accepted start-pose rows in {start_pose_csv}")

    dx = [finite(row, "dx") for row in rows]
    dy = [finite(row, "dy") for row in rows]
    yaw_errors = [finite(row, "yaw_error_deg") for row in rows]
    pos_tol = max(finite(row, "position_tolerance_m") for row in rows)
    yaw_tol = max(finite(row, "yaw_tolerance_deg") for row in rows)
    mean_xy = [sum(dx) / len(dx), sum(dy) / len(dy)]
    mean_yaw = sum(yaw_errors) / len(yaw_errors)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    fig.suptitle("Start-pose repeatability before real runs")

    ax = axes[0]
    ax.scatter(dx, dy, s=34, color=LIGHT_BLUE, alpha=0.45, label="accepted starts")
    ax.scatter(
        [0.0],
        [0.0],
        marker="D",
        s=120,
        facecolors="white",
        edgecolors=BLUE,
        linewidths=2.2,
        label="reference start",
        zorder=5,
    )
    ax.scatter(
        [mean_xy[0]],
        [mean_xy[1]],
        marker="o",
        s=160,
        facecolors=ORANGE,
        edgecolors=DARK_RED,
        linewidths=2.0,
        label="mean start error",
        zorder=6,
    )
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            radius=pos_tol,
            fill=False,
            edgecolor=GREEN,
            linewidth=2.2,
            label=f"{pos_tol:.2f} m tolerance",
        )
    )
    max_abs = max(max(abs(value) for value in dx + dy), pos_tol)
    limit = max_abs * 1.35
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    apply_common_axis_style(ax, "Position gate", "start dx [m]", "start dy [m]")

    ax = axes[1]
    indices = list(range(1, len(yaw_errors) + 1))
    ax.scatter(indices, yaw_errors, s=32, color=LIGHT_BLUE, alpha=0.55, label="yaw error")
    ax.axhline(0.0, color="#777777", linewidth=1.0)
    ax.axhline(yaw_tol, color=GREEN, linewidth=1.8, label=f"+/- {yaw_tol:.0f} deg tolerance")
    ax.axhline(-yaw_tol, color=GREEN, linewidth=1.8)
    ax.axhline(mean_yaw, color=RED, linewidth=1.8, label="mean yaw error")
    ax.set_ylim(-yaw_tol * 1.25, yaw_tol * 1.25)
    apply_common_axis_style(ax, "Yaw gate", "accepted run index", "yaw error [deg]")

    add_figure_legend(fig, axes)
    save_figure(fig, output_path)
    plt.close(fig)


def plot_forward_primitives(primitive_model, output_path):
    plt, _Circle, Ellipse = require_matplotlib()
    primitive_names = ["F30", "F50"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    fig.suptitle("Forward motion primitives in robot-local frame")

    for ax, name in zip(axes, primitive_names):
        rows = primitive_rows(primitive_model, name)
        deltas, _yaws = local_deltas_and_yaws(rows)
        mu, sigma = endpoint_model.empirical_mean_cov(deltas)
        command = command_distance_m(name)

        ax.scatter(
            [point[0] for point in deltas],
            [point[1] for point in deltas],
            s=34,
            color=LIGHT_BLUE,
            alpha=0.48,
            label="measured endpoints",
            zorder=2,
        )
        add_confidence_contours(ax, mu, sigma, Ellipse)
        ax.scatter(
            [command],
            [0.0],
            marker="D",
            s=140,
            facecolors="white",
            edgecolors=BLUE,
            linewidths=2.2,
            label="commanded endpoint",
            zorder=6,
        )
        ax.scatter(
            [mu[0]],
            [mu[1]],
            marker="o",
            s=160,
            facecolors=ORANGE,
            edgecolors=DARK_RED,
            linewidths=2.0,
            label="empirical mean",
            zorder=7,
        )
        ax.scatter([mu[0]], [mu[1]], marker="x", s=95, color=DARK_RED, zorder=8)
        ax.axhline(0.0, color="#777777", linewidth=0.9)
        ax.axvline(command, color="#c9d4ea", linewidth=1.0)
        ax.set_aspect("equal", adjustable="box")
        apply_common_axis_style(
            ax,
            f"{name} (n={len(deltas)})",
            "forward displacement [m]",
            "lateral drift [m]",
        )

    add_figure_legend(fig, axes)
    save_figure(fig, output_path)
    plt.close(fig)


def plot_rotation_bias(primitive_model, output_path):
    plt, _Circle, _Ellipse = require_matplotlib()
    names = ["CW90", "CW45", "CCW45", "CCW90", "CCW180"]
    rows_by_name = {name: primitive_rows(primitive_model, name) for name in names}
    commands = [commanded_angle_deg(name) for name in names]
    measured = [primitive_model["primitives"][name]["yaw_delta_mean_deg"] for name in names]
    yaw_std = [primitive_model["primitives"][name]["yaw_delta_std_deg"] for name in names]
    drift_mean = []
    drift_std = []

    for name in names:
        deltas, _yaws = local_deltas_and_yaws(rows_by_name[name])
        drifts = [math.hypot(delta[0], delta[1]) for delta in deltas]
        mean = sum(drifts) / len(drifts)
        if len(drifts) > 1:
            variance = sum((value - mean) ** 2 for value in drifts) / (len(drifts) - 1)
        else:
            variance = 0.0
        drift_mean.append(mean)
        drift_std.append(math.sqrt(variance))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    fig.suptitle("Rotation primitive bias and drift")

    ax = axes[0]
    min_angle = min(commands + measured) - 8.0
    max_angle = max(commands + measured) + 8.0
    ax.plot(
        [min_angle, max_angle],
        [min_angle, max_angle],
        color=BLUE,
        linewidth=1.6,
        label="ideal yaw response",
    )
    ax.errorbar(
        commands,
        measured,
        yerr=yaw_std,
        fmt="o",
        markersize=8,
        markerfacecolor=ORANGE,
        markeredgecolor=DARK_RED,
        markeredgewidth=1.8,
        ecolor=RED,
        elinewidth=1.5,
        capsize=4,
        label="measured yaw mean +/- std",
        zorder=4,
    )
    for name, x, y in zip(names, commands, measured):
        if x > 130.0:
            xytext = (-12, -14)
            ha = "right"
        else:
            xytext = (6, 6)
            ha = "left"
        ax.annotate(
            name,
            (x, y),
            xytext=xytext,
            textcoords="offset points",
            fontsize=8,
            ha=ha,
        )
    ax.set_xlim(min_angle, max_angle)
    ax.set_ylim(min_angle, max_angle)
    ax.set_aspect("equal", adjustable="box")
    apply_common_axis_style(ax, "Commanded vs measured yaw", "commanded yaw [deg]", "measured yaw [deg]")

    ax = axes[1]
    x_pos = list(range(len(names)))
    ax.bar(
        x_pos,
        drift_mean,
        yerr=drift_std,
        color="#dce8fb",
        edgecolor=BLUE,
        linewidth=1.6,
        capsize=4,
        label="mean translation drift +/- std",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=25, ha="right")
    apply_common_axis_style(ax, "Translation drift during turns", "rotation primitive", "drift magnitude [m]")

    add_figure_legend(fig, axes)
    save_figure(fig, output_path)
    plt.close(fig)


def plot_sim_vs_real(real_forward_csv, sim_forward_csv, output_path):
    plt, _Circle, Ellipse = require_matplotlib()
    model, arrays = endpoint_model.build_analysis_model(
        real_csv=real_forward_csv,
        real_run_range="21:50",
        sim_csv=sim_forward_csv,
        sim_last_n=15,
        step_distance_m=0.30,
        compare_sim_real=True,
    )
    real_errors = arrays["real_errors"]
    sim_errors = arrays["sim_errors"]
    real_mu = arrays["error_mu"]
    real_sigma = arrays["error_sigma"]
    sim_mu = arrays["sim_error_mu"]
    sim_sigma = arrays["sim_error_sigma"]

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.scatter(
        [point[0] for point in sim_errors],
        [point[1] for point in sim_errors],
        s=34,
        color="#9b9b9b",
        alpha=0.45,
        label="simulation errors",
        zorder=2,
    )
    ax.scatter(
        [point[0] for point in real_errors],
        [point[1] for point in real_errors],
        s=34,
        color=LIGHT_BLUE,
        alpha=0.5,
        label="real tracker errors",
        zorder=3,
    )
    add_confidence_contours(ax, real_mu, real_sigma, Ellipse, label_prefix="real ")
    sim_params = endpoint_model.ellipse_parameters(sim_mu, sim_sigma)
    ax.add_patch(
        Ellipse(
            xy=sim_mu,
            width=sim_params["major_axis_length_m"],
            height=sim_params["minor_axis_length_m"],
            angle=sim_params["orientation_deg"],
            fill=False,
            edgecolor="#777777",
            linestyle="--",
            linewidth=1.6,
            label="simulation 95% contour",
            zorder=4,
        )
    )
    ax.scatter(
        [sim_mu[0]],
        [sim_mu[1]],
        marker="o",
        s=130,
        facecolors="white",
        edgecolors="#555555",
        linewidths=2.0,
        label="simulation mean",
        zorder=6,
    )
    ax.scatter(
        [real_mu[0]],
        [real_mu[1]],
        marker="o",
        s=170,
        facecolors=ORANGE,
        edgecolors=DARK_RED,
        linewidths=2.0,
        label="real mean",
        zorder=7,
    )
    ax.annotate(
        "",
        xy=real_mu,
        xytext=sim_mu,
        arrowprops={"arrowstyle": "->", "color": RED, "linewidth": 1.8},
        zorder=8,
    )
    bias = model["sim2real_displacement_bias"]["magnitude_m"]
    ax.text(
        0.02,
        0.96,
        f"sim-to-real bias: {bias:.4f} m",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#d5d5d5",
            "alpha": 0.95,
        },
    )
    ax.axhline(0.0, color="#777777", linewidth=0.9)
    ax.axvline(0.0, color="#777777", linewidth=0.9)
    ax.set_aspect("equal", adjustable="box")
    apply_common_axis_style(
        ax,
        "Sim-vs-real local displacement error after 30 cm command",
        "forward error [m]",
        "lateral error [m]",
    )
    add_outside_legend(fig, ax)
    save_figure(fig, output_path)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    primitive_model = read_json(args.primitive_model)

    outputs = {
        "start_pose": output_dir / "pitch_start_pose_repeatability.png",
        "forward": output_dir / "pitch_forward_primitives.png",
        "rotation": output_dir / "pitch_rotation_primitives.png",
        "sim_vs_real": output_dir / "pitch_sim_vs_real_30cm.png",
    }

    plot_start_pose_repeatability(args.start_pose_csv, outputs["start_pose"])
    plot_forward_primitives(primitive_model, outputs["forward"])
    plot_rotation_bias(primitive_model, outputs["rotation"])
    plot_sim_vs_real(args.real_forward_csv, args.sim_forward_csv, outputs["sim_vs_real"])

    print("Generated pitch plots:")
    for path in outputs.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
