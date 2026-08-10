"""Create a motion-free, map-derived stand coverage survey plan.

The command writes pure planning artifacts and the first A* leg.  It never
publishes velocity and does not invoke a navigation runner.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.artifacts import (
    write_diagnostics_json,
    write_route_csv,
)
from scripts.aufgabe04.navigation.map_io import load_occupancy_grid_with_bundle
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.stand_coverage_survey import (
    CoverageSurveyConfig,
    build_coverage_survey_plan,
    coverage_survey_plan_sha256,
    new_stand_survey_registry,
    new_survey_progress,
    plan_next_survey_leg,
    survey_status,
    write_coverage_survey_plan,
    write_stand_survey_registry,
    write_survey_progress,
)


DEFAULT_MAP = Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--semantic-map-id", default="")
    parser.add_argument("--planning-frame", default="map")
    parser.add_argument("--start-x", type=float, required=True)
    parser.add_argument("--start-y", type=float, required=True)
    parser.add_argument("--start-yaw", type=float, default=0.0)
    parser.add_argument("--survey-id", default="")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--lane-count", type=int, default=2)
    parser.add_argument("--stop-spacing-m", type=float, default=0.90)
    parser.add_argument(
        "--exact-inspection-point-count",
        type=int,
        help=(
            "select exactly two complementary centerline inspection points; "
            "requires --lane-count 1"
        ),
    )
    parser.add_argument("--visibility-radius-m", type=float, default=1.35)
    parser.add_argument("--inflation-radius-m", type=float, default=0.25)
    parser.add_argument("--snap-radius-m", type=float, default=0.30)
    parser.add_argument("--minimum-boundary-clearance-m", type=float, default=0.10)
    parser.add_argument("--coverage-threshold", type=float, default=0.95)
    parser.add_argument("--candidate-merge-distance-m", type=float, default=0.18)
    parser.add_argument("--observation-epoch-max-age-sec", type=float, default=8.0)
    parser.add_argument("--minimum-candidate-confidence", type=float, default=0.55)
    parser.add_argument("--minimum-distinct-viewpoints", type=int, default=2)
    parser.add_argument("--minimum-candidate-hits", type=int, default=3)
    parser.add_argument("--candidate-radius-m", type=float, default=0.06)
    parser.add_argument("--candidate-uncertainty-m", type=float, default=0.02)
    parser.add_argument("--candidate-keepout-radius-m", type=float, default=0.31)
    parser.add_argument("--expected-stand-count", type=int)
    parser.add_argument("--arena-length-m", type=float, default=ArenaBounds.length_m)
    parser.add_argument("--arena-width-m", type=float, default=ArenaBounds.width_m)
    parser.add_argument(
        "--arena-center-x-m", type=float, default=ArenaBounds.center_x_m
    )
    parser.add_argument(
        "--arena-center-y-m", type=float, default=ArenaBounds.center_y_m
    )
    parser.add_argument("--arena-yaw-deg", type=float, default=ArenaBounds.yaw_deg)
    parser.add_argument("--arena-margin-m", type=float, default=ArenaBounds.margin_m)
    return parser


def _default_survey_id(now_sec: float | None = None) -> str:
    stamp = time.strftime(
        "%Y%m%d_%H%M%S",
        time.gmtime(time.time() if now_sec is None else now_sec),
    )
    return f"stand_coverage_{stamp}"


def _paths(output_dir: Path) -> dict[str, Path]:
    return {
        "plan": output_dir / "coverage_plan.json",
        "progress": output_dir / "coverage_progress.json",
        "registry": output_dir / "stand_registry.json",
        "next_route": output_dir / "legs" / "leg_000_route.csv",
        "next_diagnostics": output_dir / "legs" / "leg_000_diagnostics.json",
        "summary": output_dir / "survey_summary.json",
    }


def _require_new_paths(paths: dict[str, Path]) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise ValueError(
            "refusing to overwrite existing survey artifacts: "
            + ", ".join(existing)
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    survey_id = args.survey_id.strip() or _default_survey_id()
    output_dir = args.output_dir or Path("results/aufgabe04") / survey_id
    paths = _paths(output_dir)
    try:
        if not all(
            math.isfinite(value)
            for value in (args.start_x, args.start_y, args.start_yaw)
        ):
            raise ValueError("start pose must be finite")
        _require_new_paths(paths)
        arena = ArenaBounds(
            length_m=args.arena_length_m,
            width_m=args.arena_width_m,
            center_x_m=args.arena_center_x_m,
            center_y_m=args.arena_center_y_m,
            yaw_deg=args.arena_yaw_deg,
            margin_m=args.arena_margin_m,
        )
        config = CoverageSurveyConfig(
            lane_count=args.lane_count,
            stop_spacing_m=args.stop_spacing_m,
            exact_inspection_point_count=args.exact_inspection_point_count,
            visibility_radius_m=args.visibility_radius_m,
            inflation_radius_m=args.inflation_radius_m,
            snap_radius_m=args.snap_radius_m,
            minimum_boundary_clearance_m=args.minimum_boundary_clearance_m,
            coverage_threshold=args.coverage_threshold,
            candidate_merge_distance_m=args.candidate_merge_distance_m,
            observation_epoch_max_age_sec=args.observation_epoch_max_age_sec,
            minimum_candidate_confidence=args.minimum_candidate_confidence,
            minimum_distinct_viewpoints=args.minimum_distinct_viewpoints,
            minimum_candidate_hits=args.minimum_candidate_hits,
            candidate_radius_m=args.candidate_radius_m,
            candidate_uncertainty_m=args.candidate_uncertainty_m,
            candidate_keepout_radius_m=args.candidate_keepout_radius_m,
            expected_stand_count=args.expected_stand_count,
        ).validated()
        semantic_map_id = args.semantic_map_id or args.map.stem
        grid, map_bundle = load_occupancy_grid_with_bundle(
            args.map,
            semantic_map_id=semantic_map_id,
            planning_frame=args.planning_frame,
        )
        start = Pose2D(args.start_x, args.start_y, args.start_yaw)
        plan = build_coverage_survey_plan(
            grid,
            map_bundle_sha256=map_bundle.bundle_sha256,
            start=start,
            survey_id=survey_id,
            planning_frame=args.planning_frame,
            arena_bounds=arena,
            config=config,
        )
        progress = new_survey_progress(plan)
        registry = new_stand_survey_registry(plan)
        next_leg = plan_next_survey_leg(
            grid,
            plan=plan,
            progress=progress,
            registry=registry,
            current_pose=start,
        )
        if next_leg is None:
            raise ValueError("coverage plan unexpectedly has no first viewpoint")

        output_dir.mkdir(parents=True, exist_ok=True)
        write_coverage_survey_plan(paths["plan"], plan)
        write_survey_progress(paths["progress"], progress, plan)
        write_stand_survey_registry(paths["registry"], registry, plan)
        write_route_csv(
            paths["next_route"],
            (next_leg.route_result,),
            final_yaw_by_leg={0: next_leg.viewpoint.pose.yaw_rad},
        )
        write_diagnostics_json(
            paths["next_diagnostics"],
            (next_leg.route_result,),
            metadata={
                "schema_version": 1,
                "route_kind": "stand_coverage_survey",
                "motion_authorized": False,
                "survey_id": plan.survey_id,
                "plan_sha256": coverage_survey_plan_sha256(plan),
                "map_bundle_sha256": plan.map_bundle_sha256,
                "target_viewpoint_id": next_leg.viewpoint.viewpoint_id,
                "target_pose": {
                    "x_m": next_leg.viewpoint.pose.x_m,
                    "y_m": next_leg.viewpoint.pose.y_m,
                    "yaw_rad": next_leg.viewpoint.pose.yaw_rad,
                },
                "candidate_keepout_count": 0,
                "inflation_radius_m": plan.config.inflation_radius_m,
                "exact_start_connector": (
                    next_leg.exact_start_connector.to_metadata()
                ),
                "arena_boundary_overlay": True,
                "arena_bounds": plan.arena_bounds.to_metadata(),
            },
        )
        summary = {
            "schema_version": 1,
            "status": "coverage_plan_ready",
            "motion_published": False,
            **survey_status(plan, progress, registry),
            "next_viewpoint_id": next_leg.viewpoint.viewpoint_id,
            "next_route_length_m": next_leg.route_result.route.length_m,
            "artifacts": {name: str(path) for name, path in paths.items()},
        }
        paths["summary"].write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
