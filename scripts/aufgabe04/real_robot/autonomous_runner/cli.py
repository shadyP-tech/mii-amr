"""Command-line contract for autonomous stand exploration."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.aufgabe04.real_robot.execution.child_runner import (
    DEFAULT_UNCERTAINTY_SIGMA_MULTIPLIER,
)
from scripts.aufgabe04.real_robot.mission.modes import AutonomousRunMode
from scripts.aufgabe04.navigation.localization.startup_active_localization import (
    DEFAULT_STARTUP_ACTIVE_LOCALIZATION_ANGULAR_SPEED_RADPS,
    DEFAULT_STARTUP_ACTIVE_LOCALIZATION_MAX_ATTEMPTS,
    DEFAULT_STARTUP_ACTIVE_LOCALIZATION_ROTATION_RAD,
    DEFAULT_STARTUP_ACTIVE_LOCALIZATION_TIMEOUT_SEC,
)

DEFAULT_MAP = Path("maps/aufgabe03/arena_1p898x3p9_auto.yaml")
DEFAULT_OUTPUT_ROOT = Path("results/aufgabe04/real/autonomous_exploration")
DEFAULT_MAX_BLOCKAGE_REPLANS_PER_LEG = 3
DEFAULT_MAX_STARTUP_RESEALS_PER_LEG = 3
DEFAULT_MAX_RUNTIME_LOCALIZATION_RESEALS_PER_LEG = 1
DEFAULT_MAX_LOCALIZATION_READINESS_RETRIES_PER_LEG = 2
DEFAULT_MAX_CAMERA_OBSERVATION_ATTEMPTS_PER_CANDIDATE = 2
DEFAULT_MAX_CANDIDATE_INSPECTION_VIEWS = 8
DEFAULT_MAX_ROUTE_ADMISSION_ATTEMPTS_PER_CANDIDATE = 2


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
    parser.add_argument(
        "--scan-topology-profile", choices=("linear", "full_rotation"), default="linear",
        help="Explicit scan topology assertion; full_rotation still validates each message's seam geometry.",
    )
    parser.add_argument(
        "--candidate-approach-offset-m", type=float, default=0.50,
        help=(
            "Preferred robot-base-to-stand-center distance for camera inspection "
            "(default: 0.50 m). Also sets the preferred local inspection "
            "standoff; physical clearance and route admission still apply."
        ),
    )
    parser.add_argument("--final-facing-offset-m", type=float, default=0.35)
    parser.add_argument("--axis-sample-count", type=int, default=7)
    parser.add_argument("--camera-timeout-sec", type=float, default=90.0)
    parser.add_argument(
        "--stop-after-camera-candidates", type=int,
        help="Stop after this many validated candidates as an incomplete pilot checkpoint; preserves the arena goal and full obstacle pool.",
    )
    parser.add_argument("--camera-capture-max-frames", type=int, default=64)
    parser.add_argument("--camera-capture-max-bytes", type=int, default=33554432)
    parser.add_argument(
        "--server-qr-mapping-evidence", type=Path,
        help="Sealed robot-scoped saved server mappings; without this, discovery records binding-pending identities.",
    )
    parser.add_argument("--server-robot-id", help="Exact robot ID in the saved server response; required with mapping evidence.")
    parser.add_argument(
        "--max-camera-observation-attempts-per-candidate",
        type=int,
        default=DEFAULT_MAX_CAMERA_OBSERVATION_ATTEMPTS_PER_CANDIDATE,
        help=(
            "Legacy camera-attempt bound retained for compatibility. "
            "Candidate inspection now uses --max-candidate-inspection-views "
            "to finish bounded local views before selecting another stand."
        ),
    )
    parser.add_argument(
        "--max-candidate-inspection-views",
        type=int,
        choices=range(1, 17),
        default=DEFAULT_MAX_CANDIDATE_INSPECTION_VIEWS,
        help=(
            "Maximum observation views per candidate, including its initial "
            "view (1-16). Each new view uses fresh route and motion admission; "
            "exhausted candidates remain explicitly incomplete."
        ),
    )
    parser.add_argument(
        "--max-route-admission-attempts-per-candidate",
        type=int,
        default=DEFAULT_MAX_ROUTE_ADMISSION_ATTEMPTS_PER_CANDIDATE,
        help=(
            "Bound candidate-local no-motion route-uncertainty deferrals. "
            "An exact pre-motion route admission rejection excludes that "
            "candidate while other candidates run; every retry still uses a "
            "fresh route, dry preflight, route uncertainty budget, and "
            "one-use permit gate. Other child failures remain terminal."
        ),
    )
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
        "--prompt-for-initialpose",
        action="store_true",
        help=(
            "Pause once before preplanning localization admission so the "
            "operator can click RViz 2D Pose Estimate while the robot is "
            "stopped. The first route is planned from the post-click AMCL "
            "pose. This is not motion authorization."
        ),
    )
    parser.add_argument(
        "--initialpose-prompt-window-sec",
        type=float,
        default=2.0,
        help=(
            "Human-facing time window printed by --prompt-for-initialpose. "
            "It documents when the RViz click must happen; it does not widen "
            "the ROS preflight or route-clearance gates."
        ),
    )
    parser.add_argument(
        "--stand-model-profile",
        type=Path,
        default=None,
        help=(
            "Content-hashed measured physical stand model. Required by "
            "execute-exact-two-camera and execute-full; those workflows "
            "have no legacy image-detector fallback."
        ),
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
        "--enable-startup-active-localization",
        action="store_true",
        help=(
            "After the stopped startup route selector rejects every initial "
            "route on uncertainty budget, allow a separately typed LOCALIZE "
            "phase with bounded in-place rotation, then recollect stopped "
            "AMCL evidence and retry the same exact selector. This does not "
            "authorize the mission route or bypass the later RUN prompt."
        ),
    )
    parser.add_argument(
        "--max-startup-active-localization-attempts",
        type=int,
        default=DEFAULT_STARTUP_ACTIVE_LOCALIZATION_MAX_ATTEMPTS,
    )
    parser.add_argument(
        "--startup-active-localization-rotation-rad",
        "--startup-active-localization-turn-rad",
        dest="startup_active_localization_rotation_rad",
        type=float,
        default=DEFAULT_STARTUP_ACTIVE_LOCALIZATION_ROTATION_RAD,
    )
    parser.add_argument(
        "--startup-active-localization-angular-speed-radps",
        type=float,
        default=DEFAULT_STARTUP_ACTIVE_LOCALIZATION_ANGULAR_SPEED_RADPS,
    )
    parser.add_argument(
        "--startup-active-localization-timeout-sec",
        type=float,
        default=DEFAULT_STARTUP_ACTIVE_LOCALIZATION_TIMEOUT_SEC,
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
