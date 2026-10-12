#!/usr/bin/env python3
"""Fit the measured endpoint and local-error distributions; optionally compare simulation."""
import argparse

if __package__:
    from .probabilistic_model.analysis import build_analysis_model
    from .probabilistic_model.csv_data import write_json
else:
    from probabilistic_model.analysis import build_analysis_model
    from probabilistic_model.csv_data import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-csv", default="results/aufgabe02/real_scripted_drive_runs.csv")
    parser.add_argument("--real-run-range", default="21:50")
    parser.add_argument("--sim-csv", default="results/aufgabe02/scripted_drive_runs.csv")
    parser.add_argument("--sim-last-n", type=int, default=15)
    parser.add_argument("--step-distance-m", type=float, default=0.30)
    parser.add_argument("--compare-sim-real", action="store_true")
    parser.add_argument("--output-json", default="results/aufgabe02/probabilistic_endpoint_model.json")
    args = parser.parse_args()
    model = build_analysis_model(
        args.real_csv, args.real_run_range, args.sim_csv, args.sim_last_n,
        args.step_distance_m, args.compare_sim_real,
    )
    write_json(args.output_json, model)
    print(args.output_json)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}")
