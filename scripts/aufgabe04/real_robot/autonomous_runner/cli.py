"""Command-line contract for autonomous stand exploration."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.aufgabe04.real_robot.autonomous_child_runner import (
    DEFAULT_UNCERTAINTY_SIGMA_MULTIPLIER,
)
from scripts.aufgabe04.real_robot.autonomous_modes import AutonomousRunMode

DEFAULT_MAP = Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml")
DEFAULT_OUTPUT_ROOT = Path("results/aufgabe04/real/autonomous_exploration")
DEFAULT_MAX_BLOCKAGE_REPLANS_PER_LEG = 3
DEFAULT_MAX_STARTUP_RESEALS_PER_LEG = 3
DEFAULT_MAX_RUNTIME_LOCALIZATION_RESEALS_PER_LEG = 1
DEFAULT_MAX_LOCALIZATION_READINESS_RETRIES_PER_LEG = 2

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-profile", required=True, type=Path)
    parser.add_argument("--camera-calibration", required=True, type=Path)
    parser.add_argument("--physical-site", required=True, type=Path)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--semantic-map-id", default="arena_1p898x3p9_auto")
    parser.add_argument("--session-id", default="")
    parser.add_argument(
        "--run-mode",
        choices=tuple(mode.value for mode in AutonomousRunMode),
        default=None,
        help=(
            "Explicit mutually exclusive workflow mode. Legacy --execute, "
            "--coverage-leg-limit, and --stop-after-coverage remain accepted "
            "only when they resolve to the same mode."
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help=(
            "Immutable autonomous coverage checkpoint to continue. Required "
            "only with --run-mode resume-next-coverage-leg. The continuation "
            "uses a new session, fresh AMCL/TF, fresh A*, fresh dry-run, and "
            "fresh typed RUN."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--expected-stand-count",
        type=int,
        default=None,
        help=(
            "Optional assertion of the arena stand count. The canonical "
            "value is loaded from --physical-site; a mismatch fails before "
            "planning or motion authorization."
        ),
    )
    parser.add_argument("--inspection-stop-spacing-m", type=float, default=0.70)
    parser.add_argument(
        "--exact-inspection-point-count",
        type=int,
        choices=(2,),
        default=None,
        help=(
            "Select exactly two complementary centerline LiDAR inspection "
            "points. This is required by execute-exact-two-camera and is "
            "also supported by the LiDAR-only checkpoint workflow. Omit it "
            "for execute-full so stop spacing determines the redundant "
            "coverage set."
        ),
    )
    parser.add_argument("--lidar-epoch-sec", type=float, default=8.0)
    parser.add_argument("--candidate-approach-offset-m", type=float, default=0.70)
    parser.add_argument("--final-facing-offset-m", type=float, default=0.35)
    parser.add_argument("--axis-sample-count", type=int, default=7)
    parser.add_argument("--camera-timeout-sec", type=float, default=90.0)
    parser.add_argument(
        "--localization-branch-proof-id",
        default="",
        help=(
            "Operator evidence ID for a known physical start or an asymmetric "
            "landmark that resolves the saved map's symmetric pose branch. "
            "Required for every physical execution mode; covariance alone "
            "is insufficient."
        ),
    )
    parser.add_argument(
        "--stand-model-profile",
        type=Path,
        default=None,
        help="Optional content-hashed measured physical stand model.",
    )
    parser.add_argument(
        "--coverage-leg-limit",
        type=int,
        default=0,
        help=(
            "Coverage checkpoint leg count. A positive value is required by "
            "--run-mode execute-coverage-checkpoint. The dedicated exact-two "
            "camera mode accepts zero/omitted or exactly two and resolves to "
            "two; use zero for other modes."
        ),
    )
    parser.add_argument(
        "--max-blockage-replans-per-leg",
        type=int,
        default=DEFAULT_MAX_BLOCKAGE_REPLANS_PER_LEG,
        help=(
            "Maximum front-LiDAR transient-overlay A* recovery attempts for "
            "one coverage leg. Zero disables adaptive blockage recovery."
        ),
    )
    parser.add_argument(
        "--max-startup-reseals-per-leg",
        type=int,
        default=DEFAULT_MAX_STARTUP_RESEALS_PER_LEG,
        help=(
            "Maximum fresh-pose A* reseals after a route is rejected before "
            "motion because AMCL left its certified startup segment or the "
            "live map<-odom consistency monitor invalidated the frozen odom "
            "certificate. In an execute-* mission, the initial typed RUN "
            "covers only bounded same-leg, same-target replacements that "
            "obtain fresh stationary localization and consume a dedicated "
            "one-use recovery permit."
        ),
    )
    parser.add_argument(
        "--max-runtime-localization-reseals-per-leg",
        type=int,
        default=DEFAULT_MAX_RUNTIME_LOCALIZATION_RESEALS_PER_LEG,
        help=(
            "Maximum fresh stationary AMCL/TF admissions and A* reseals after "
            "motion has stopped because the global localization consistency "
            "monitor invalidated the odom execution certificate. The initial "
            "mission RUN may cover these bounded same-leg, same-target retries "
            "after a fresh immutable motion permit is admitted."
        ),
    )
    parser.add_argument(
        "--max-localization-readiness-retries-per-leg",
        type=int,
        default=DEFAULT_MAX_LOCALIZATION_READINESS_RETRIES_PER_LEG,
        help=(
            "Maximum fresh no-motion AMCL admissions for a sole transient "
            "dynamic map->odom gap after an observation, and for a certified "
            "route uncertainty budget exhausted during the pre-RUN first-route "
            "rehearsal or before later motion. Other failed gates remain "
            "terminal; zero disables these bounded retries."
        ),
    )
    parser.add_argument(
        "--uncertainty-sigma-multiplier",
        type=float,
        default=DEFAULT_UNCERTAINTY_SIGMA_MULTIPLIER,
        help=(
            "AMCL covariance multiplier charged to every route-clearance "
            "admission and reused as the live map<-odom continuity envelope. "
            "The route is rejected when this larger allowance exhausts "
            "clearance; hard transform-drift caps remain unchanged."
        ),
    )
    parser.add_argument(
        "--stop-after-coverage",
        action="store_true",
        help=(
            "Legacy alias: finish the center-corridor LiDAR survey and "
            "candidate snapshot, then stop before candidate approaches. "
            "Prefer --run-mode execute-coverage-only."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Legacy execution selector. Prefer an explicit --run-mode; typed "
            "RUN and exact one-use permits remain mandatory."
        ),
    )
    return parser

