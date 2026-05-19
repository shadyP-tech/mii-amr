import argparse

from .analysis import build_analysis_model
from .constants import DEFAULT_STEP_DISTANCE_M
from .output import print_report, write_json, write_summary_csv
from .plots import plot_scatter_with_ellipse, plot_sim_vs_real_errors


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build empirical endpoint and motion-error models.",
    )
    parser.add_argument(
        "--real-csv",
        default="results/aufgabe02/real_scripted_drive_runs.csv",
    )
    parser.add_argument("--real-run-range", default="21:50")
    parser.add_argument("--sim-csv", default="results/aufgabe02/scripted_drive_runs.csv")
    parser.add_argument("--sim-last-n", type=int, default=15)
    parser.add_argument("--step-distance-m", type=float, default=DEFAULT_STEP_DISTANCE_M)
    parser.add_argument("--compare-sim-real", action="store_true")
    parser.add_argument(
        "--output-json",
        default="results/aufgabe02/probabilistic_endpoint_model.json",
    )
    parser.add_argument(
        "--summary-csv",
        default="results/aufgabe02/probabilistic_endpoint_model_summary.csv",
    )
    parser.add_argument(
        "--endpoint-plot",
        default="results/aufgabe02/real_endpoint_gaussian_ellipse.png",
    )
    parser.add_argument(
        "--motion-error-plot",
        default="results/aufgabe02/real_motion_error_gaussian_ellipse.png",
    )
    parser.add_argument(
        "--sim-real-plot",
        default="results/aufgabe02/sim_vs_real_displacement_error_scatter.png",
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
