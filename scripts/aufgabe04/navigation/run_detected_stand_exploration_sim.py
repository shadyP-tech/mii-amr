"""Run a multi-leg detected-stand exploration route in Gazebo simulation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_ROUTE_CSV = Path("results/aufgabe04/routes/detected_stand_exploration_route.csv")
DEFAULT_DIAGNOSTICS_JSON = Path("results/aufgabe04/routes/detected_stand_exploration_route_diagnostics.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-csv", type=Path, default=DEFAULT_ROUTE_CSV)
    parser.add_argument("--diagnostics-json", type=Path, default=DEFAULT_DIAGNOSTICS_JSON)
    parser.add_argument("--run-id-prefix", default="aufgabe04_sim_detected_explore")
    parser.add_argument("--start-leg-index", type=int, default=0)
    parser.add_argument("--max-legs", type=int, default=0, help="0 means run all remaining legs")
    parser.add_argument("--namespace", default="")
    parser.add_argument("--scan-topic", default="scan")
    parser.add_argument("--odom-topic", default="odom")
    parser.add_argument("--cmd-vel-topic", default="cmd_vel")
    parser.add_argument("--map-frame", default="odom")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_footprint")
    parser.add_argument("--max-tf-age-sec", type=float, default=10.0)
    parser.add_argument("--max-scan-age-sec", type=float, default=3.0)
    parser.add_argument("--max-odom-age-sec", type=float, default=3.0)
    parser.add_argument("--initial-sensor-wait-sec", type=float, default=6.0)
    parser.add_argument("--min-obstacle-distance-m", type=float, default=0.18)
    parser.add_argument("--front-obstacle-slow-distance-m", type=float, default=0.42)
    parser.add_argument("--front-obstacle-sector-rad", type=float, default=0.6108652381980153)
    parser.add_argument("--max-linear-mps", type=float, default=0.035)
    parser.add_argument("--max-angular-radps", type=float, default=0.10)
    parser.add_argument("--goal-tolerance-m", type=float, default=0.16)
    parser.add_argument("--thinning-min-spacing-m", type=float, default=0.15)
    parser.add_argument("--stuck-timeout-sec", type=float, default=8.0)
    parser.add_argument("--stuck-progress-epsilon-m", type=float, default=0.03)
    parser.add_argument(
        "--allowed-cmd-vel-publisher",
        action="append",
        default=["/behavior_server", "/velocity_smoother"],
    )
    parser.add_argument("--operator-note", default="sim detected-stand sequential exploration")
    return parser


def _leg_count(diagnostics_json: Path) -> int:
    try:
        payload = json.loads(Path(diagnostics_json).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid diagnostics JSON: {exc}") from exc
    legs = payload.get("legs")
    if not isinstance(legs, list):
        raise ValueError("diagnostics JSON missing legs list")
    return len(legs)


def _segment_command(args, leg_index: int) -> list[str]:
    run_id = f"{args.run_id_prefix}_leg_{leg_index:02d}"
    command = [
        sys.executable,
        "scripts/aufgabe04/navigation/run_single_station_segment.py",
        "--leg-index",
        str(leg_index),
        "--route-csv",
        str(args.route_csv),
        "--diagnostics-json",
        str(args.diagnostics_json),
        "--run-id",
        run_id,
        "--namespace",
        args.namespace,
        "--scan-topic",
        args.scan_topic,
        "--odom-topic",
        args.odom_topic,
        "--cmd-vel-topic",
        args.cmd_vel_topic,
        "--localization-source",
        "tf",
        "--map-frame",
        args.map_frame,
        "--odom-frame",
        args.odom_frame,
        "--base-frame",
        args.base_frame,
        "--allow-sim-time",
        "--allow-unbound-survey-simulation-route",
        "--max-tf-age-sec",
        str(args.max_tf_age_sec),
        "--max-scan-age-sec",
        str(args.max_scan_age_sec),
        "--max-odom-age-sec",
        str(args.max_odom_age_sec),
        "--initial-sensor-wait-sec",
        str(args.initial_sensor_wait_sec),
        "--min-obstacle-distance-m",
        str(args.min_obstacle_distance_m),
        "--front-obstacle-slow-distance-m",
        str(args.front_obstacle_slow_distance_m),
        "--front-obstacle-sector-rad",
        str(args.front_obstacle_sector_rad),
        "--max-linear-mps",
        str(args.max_linear_mps),
        "--max-angular-radps",
        str(args.max_angular_radps),
        "--goal-tolerance-m",
        str(args.goal_tolerance_m),
        "--thinning-min-spacing-m",
        str(args.thinning_min_spacing_m),
        "--operator-note",
        f"{args.operator_note}; leg={leg_index}",
        "--stuck-timeout-sec",
        str(args.stuck_timeout_sec),
        "--stuck-progress-epsilon-m",
        str(args.stuck_progress_epsilon_m),
    ]
    for publisher in args.allowed_cmd_vel_publisher:
        command.extend(["--allowed-cmd-vel-publisher", publisher])
    return command


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        total_legs = _leg_count(args.diagnostics_json)
        if args.start_leg_index < 0 or args.start_leg_index >= total_legs:
            raise ValueError(f"--start-leg-index must be in [0, {total_legs - 1}]")
        end_leg = total_legs
        if args.max_legs > 0:
            end_leg = min(total_legs, args.start_leg_index + args.max_legs)
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")

    for leg_index in range(args.start_leg_index, end_leg):
        command = _segment_command(args, leg_index)
        print(f"\n=== Running detected-stand exploration leg {leg_index}/{total_legs - 1} ===")
        print(" ".join(command))
        status = subprocess.run(command, check=False).returncode
        if status != 0:
            print(f"stopping after leg {leg_index}: command exited with {status}", file=sys.stderr)
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
