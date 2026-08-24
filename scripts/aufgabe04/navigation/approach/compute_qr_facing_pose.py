"""Convert camera side/axis evidence into a final QR-facing pose JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.foundation.artifacts import write_diagnostics_json, write_route_csv
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.global_planner import plan_route
from scripts.aufgabe04.navigation.planning.map_io import load_occupancy_grid
from scripts.aufgabe04.navigation.approach.two_stage_approach import qr_facing_pose_from_camera
from scripts.aufgabe04.stations.models import Station, StationPose
from scripts.aufgabe04.perception.camera_stand_observation import (
    load_camera_observation,
    validate_camera_observation,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-json", type=Path, default=None)
    parser.add_argument("--stand-x", type=float)
    parser.add_argument("--stand-y", type=float)
    parser.add_argument("--robot-x", type=float)
    parser.add_argument("--robot-y", type=float)
    parser.add_argument("--stand-axis-rad", type=float)
    parser.add_argument("--side", choices=["qr_code_side", "basic_color_side"])
    parser.add_argument("--required-map-frame", default="odom")
    parser.add_argument("--min-axis-confidence", type=float, default=0.60)
    parser.add_argument("--min-side-confidence", type=float, default=0.60)
    parser.add_argument("--max-observation-age-sec", type=float, default=None)
    parser.add_argument("--approach-offset-m", type=float, default=0.30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map", type=Path, default=None)
    parser.add_argument("--route-csv", type=Path, default=None)
    parser.add_argument("--diagnostics-json", type=Path, default=None)
    parser.add_argument("--keepout-radius-m", type=float, default=0.20)
    parser.add_argument("--inflation-radius-m", type=float, default=0.05)
    args = parser.parse_args(argv)
    observation = None
    if args.observation_json is not None:
        observation = load_camera_observation(args.observation_json)
        validate_camera_observation(
            observation,
            required_map_frame=args.required_map_frame,
            min_axis_confidence=args.min_axis_confidence,
            min_side_confidence=args.min_side_confidence,
            max_age_sec=args.max_observation_age_sec,
        )
        args.stand_x, args.stand_y = observation.stand_x_m, observation.stand_y_m
        args.robot_x, args.robot_y = observation.robot_x_m, observation.robot_y_m
        args.stand_axis_rad, args.side = observation.stand_axis_rad, observation.side
    elif any(value is None for value in (
        args.stand_x, args.stand_y, args.robot_x, args.robot_y,
        args.stand_axis_rad, args.side,
    )):
        parser.error("provide --observation-json or all legacy geometry arguments")
    result = qr_facing_pose_from_camera(
        Pose2D(args.stand_x, args.stand_y),
        Pose2D(args.robot_x, args.robot_y),
        stand_axis_rad=args.stand_axis_rad,
        side=args.side,
        offset_m=args.approach_offset_m,
    )
    payload = {
        "schema_version": 1,
        "source": "camera_side_and_stand_axis",
        "camera_observation_json": "" if args.observation_json is None else str(args.observation_json),
        "observation_provenance": None if observation is None else {
            "observed_at_sec": observation.observed_at_sec,
            "image_topic": observation.image_topic,
            "camera_frame": observation.camera_frame,
            "map_frame": observation.map_frame,
            "axis_confidence": observation.axis_confidence,
            "side_confidence": observation.side_confidence,
            "qr_texts": list(observation.qr_texts),
        },
        "stand_axis_rad": args.stand_axis_rad,
        "observation_robot_pose": {"x_m": args.robot_x, "y_m": args.robot_y},
        "side": result.side,
        "qr_normal_rad": result.qr_normal_rad,
        "stand": {"x_m": result.stand.x_m, "y_m": result.stand.y_m},
        "final_qr_approach": {
            "x_m": result.final_qr_approach.x_m,
            "y_m": result.final_qr_approach.y_m,
            "yaw_rad": result.final_qr_approach.yaw_rad,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    route_outputs = (args.route_csv, args.diagnostics_json)
    if args.map is None and any(value is not None for value in route_outputs):
        parser.error("--map is required when writing final route artifacts")
    if args.map is not None:
        if any(value is None for value in route_outputs):
            parser.error("--route-csv and --diagnostics-json are required with --map")
        costmap = Costmap.from_occupancy_grid(load_occupancy_grid(args.map))
        hidden_yaw_placeholder = 0.0
        station = Station(
            "camera_resolved_stand",
            StationPose(args.stand_x, args.stand_y, hidden_yaw_placeholder),
            approach_offset_m=args.approach_offset_m,
            keepout_radius_m=args.keepout_radius_m,
        )
        planning_costmap = costmap.with_station_keepouts([station]).with_inflation(
            args.inflation_radius_m
        )
        result_route = plan_route(
            planning_costmap,
            Pose2D(args.robot_x, args.robot_y),
            result.final_qr_approach,
            snap_radius_m=0.30,
        )
        write_route_csv(
            args.route_csv,
            [result_route],
            final_yaw_by_leg={0: result.final_qr_approach.yaw_rad},
        )
        write_diagnostics_json(
            args.diagnostics_json,
            [result_route],
            metadata={
                "stage": "camera_resolved_qr_approach",
                "camera_pose_json": str(args.output),
                "hidden_layout_yaw_used": False,
                "inflation_radius_m": args.inflation_radius_m,
            },
        )
        if result_route.failure is not None:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
