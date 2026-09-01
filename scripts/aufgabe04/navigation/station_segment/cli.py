"""Command-line contract for single-station segment execution."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from scripts.aufgabe04.navigation.execution.mission_leg_motion_permit import MissionLegKind
from scripts.aufgabe04.navigation.execution.route_uncertainty_defaults import (
    DEFAULT_COLLISION_MARGIN_M,
    DEFAULT_TRACKING_TUBE_RADIUS_M,
    DEFAULT_UNCERTAINTY_BRAKING_LATENCY_DISTANCE_M,
    DEFAULT_UNCERTAINTY_CLEARANCE_SAMPLE_SPACING_M,
    DEFAULT_UNCERTAINTY_ODOM_DRIFT_BOUND_M,
)
from scripts.aufgabe04.navigation.coverage.transient_blockage_policy import (
    DEFAULT_LINEAR_MOTION_FLOOR_MPS,
)
from scripts.aufgabe04.navigation.approach.viewpoint_sampling_contract import (
    INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M,
    INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M,
)

DEFAULT_ROUTE_CSV = Path("results/aufgabe04/routes/station_route.csv")
DEFAULT_DIAGNOSTICS_JSON = Path(
    "results/aufgabe04/routes/station_route_diagnostics.json"
)
DEFAULT_RUN_LOG = Path("results/aufgabe04/station_segment_runs.csv")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-csv", type=Path, default=DEFAULT_ROUTE_CSV)
    parser.add_argument("--diagnostics-json", type=Path, default=DEFAULT_DIAGNOSTICS_JSON)
    parser.add_argument("--route-certificate-json", type=Path, default=None)
    parser.add_argument("--mission-plan-manifest", type=Path, default=None)
    parser.add_argument("--survey-manifest", type=Path, default=None)
    parser.add_argument("--route-bundle-json", type=Path, default=None)
    parser.add_argument("--planner-config-json", type=Path, default=None)
    parser.add_argument("--runtime-map-bundle-json", type=Path, default=None)
    parser.add_argument("--runtime-environment", type=Path, default=None)
    parser.add_argument("--candidate-snapshot", type=Path, default=None)
    parser.add_argument("--coverage-plan", type=Path, default=None)
    parser.add_argument("--coverage-transient-replan-survey-root", type=Path)
    parser.add_argument("--coverage-transient-replan-session-root", type=Path)
    parser.add_argument("--coverage-transient-replan-map", type=Path)
    parser.add_argument("--coverage-transient-replan-semantic-map-id", default="")
    parser.add_argument("--coverage-transient-replan-target-viewpoint-id", default="")
    parser.add_argument("--coverage-transient-replan-robot-radius-m", type=float)
    parser.add_argument("--coverage-transient-replan-max-count", type=int, default=0)
    parser.add_argument("--coverage-transient-replan-leg-index", type=int)
    parser.add_argument(
        "--coverage-transient-replan-resume-state-json",
        type=Path,
        help=(
            "Internal immutable state for resuming a cumulative transient "
            "overlay after a stopped localization reseal."
        ),
    )
    parser.add_argument("--station-identity-registry", type=Path, default=None)
    parser.add_argument("--arrival-pose-catalog", type=Path, default=None)
    parser.add_argument("--task-snapshot", type=Path, default=None)
    parser.add_argument("--leg-index", type=int, required=True)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RUN_LOG)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--robot-id", default="tb3")
    parser.add_argument("--namespace", default="")
    parser.add_argument("--scan-topic", default="scan")
    parser.add_argument("--odom-topic", default="odom")
    parser.add_argument("--cmd-vel-topic", default="cmd_vel")
    parser.add_argument("--amcl-topic", default="amcl_pose")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--base-frame", default="base_footprint")
    parser.add_argument("--localization-source", default="amcl", choices=["amcl", "tf"])
    parser.add_argument(
        "--execution-pose-frame",
        choices=["map", "odom"],
        default="map",
        help=(
            "Controller pose/route frame. The odom option freezes one "
            "map->odom transform after stopped preflight; AMCL then becomes "
            "a consistency monitor and never supplies pursuit poses."
        ),
    )
    parser.add_argument("--odom-execution-certificate-json", type=Path)
    parser.add_argument("--uncertainty-budget-json", type=Path)
    parser.add_argument("--uncertainty-map-yaml", type=Path)
    parser.add_argument("--localization-branch-proof-id", default="")
    parser.add_argument(
        "--mission-motion-authorization-json",
        type=Path,
        help=(
            "Internal autonomous-wrapper evidence for the mission-level RUN. "
            "It never bypasses gates by itself."
        ),
    )
    parser.add_argument(
        "--runtime-localization-motion-permit-json",
        type=Path,
        help=(
            "Internal one-run, same-target runtime-localization recovery permit."
        ),
    )
    parser.add_argument(
        "--runtime-localization-mission-leg-kind",
        choices=[
            MissionLegKind.COVERAGE.value,
            MissionLegKind.CANDIDATE_PREAPPROACH.value,
            MissionLegKind.OPPOSITE_FACE.value,
        ],
    )
    parser.add_argument("--runtime-localization-mission-leg-index", type=int)
    parser.add_argument("--runtime-localization-target-id", default="")
    parser.add_argument(
        "--runtime-localization-target-viewpoint-id", default=""
    )
    parser.add_argument("--runtime-localization-semantic-map-id", default="")
    parser.add_argument(
        "--startup-reseal-motion-authorization-json",
        type=Path,
        help=(
            "Internal autonomous-wrapper evidence that the mission RUN covers "
            "bounded same-target pre-motion startup recovery."
        ),
    )
    parser.add_argument(
        "--startup-reseal-motion-permit-json",
        type=Path,
        help="Internal one-use permit for this exact startup-reseal child.",
    )
    parser.add_argument(
        "--startup-reseal-mission-leg-kind",
        choices=[
            MissionLegKind.COVERAGE.value,
            MissionLegKind.CANDIDATE_PREAPPROACH.value,
            MissionLegKind.OPPOSITE_FACE.value,
        ],
    )
    parser.add_argument("--startup-reseal-mission-leg-index", type=int)
    parser.add_argument("--startup-reseal-target-id", default="")
    parser.add_argument("--startup-reseal-target-viewpoint-id", default="")
    parser.add_argument("--startup-reseal-semantic-map-id", default="")
    parser.add_argument(
        "--mission-leg-evidence-kind",
        choices=[
            MissionLegKind.COVERAGE.value,
            MissionLegKind.CANDIDATE_PREAPPROACH.value,
            MissionLegKind.OPPOSITE_FACE.value,
        ],
        help=(
            "Non-authorizing routine-leg identity emitted into dry/live "
            "semantic evidence. It never bypasses operator confirmation."
        ),
    )
    parser.add_argument("--mission-leg-evidence-index", type=int)
    parser.add_argument("--mission-leg-evidence-target-id", default="")
    parser.add_argument(
        "--mission-leg-motion-authorization-json",
        type=Path,
        help=(
            "Internal autonomous-wrapper evidence for the one-time mission RUN "
            "covering separately sealed routine child legs."
        ),
    )
    parser.add_argument(
        "--mission-leg-motion-permit-json",
        type=Path,
        help="Internal one-use permit for this exact routine child leg.",
    )
    parser.add_argument(
        "--mission-leg-kind",
        choices=[
            MissionLegKind.COVERAGE.value,
            MissionLegKind.CANDIDATE_PREAPPROACH.value,
            MissionLegKind.OPPOSITE_FACE.value,
        ],
    )
    parser.add_argument("--mission-leg-index", type=int)
    parser.add_argument("--mission-leg-target-id", default="")
    parser.add_argument("--mission-leg-semantic-map-id", default="")
    parser.add_argument("--mission-leg-dry-preflight-json", type=Path)
    parser.add_argument(
        "--mission-leg-dry-odom-certificate-json", type=Path
    )
    parser.add_argument(
        "--mission-leg-dry-uncertainty-budget-json", type=Path
    )
    parser.add_argument(
        "--mission-session-id",
        default="",
        help="Session identity bound by an internal motion permit.",
    )
    parser.add_argument("--uncertainty-robot-radius-m", type=float)
    parser.add_argument(
        "--uncertainty-collision-margin-m",
        type=float,
        default=DEFAULT_COLLISION_MARGIN_M,
    )
    parser.add_argument(
        "--uncertainty-odom-drift-bound-m",
        type=float,
        default=DEFAULT_UNCERTAINTY_ODOM_DRIFT_BOUND_M,
    )
    parser.add_argument(
        "--uncertainty-braking-latency-distance-m",
        type=float,
        default=DEFAULT_UNCERTAINTY_BRAKING_LATENCY_DISTANCE_M,
    )
    parser.add_argument(
        "--uncertainty-sigma-multiplier", type=float, default=1.0
    )
    parser.add_argument(
        "--uncertainty-heading-lever-arm-m", type=float, default=None
    )
    parser.add_argument(
        "--uncertainty-clearance-sample-spacing-m",
        type=float,
        default=DEFAULT_UNCERTAINTY_CLEARANCE_SAMPLE_SPACING_M,
    )
    parser.add_argument(
        "--max-map-odom-yaw-drift-rad", type=float, default=0.10
    )
    parser.add_argument(
        "--max-map-odom-translation-drift-m",
        type=float,
        default=0.15,
        help=(
            "Hard cap for live map<-odom translation drift. The effective "
            "threshold is the smaller of this cap and the localization "
            "position allowance already charged to the route budget."
        ),
    )
    parser.add_argument("--allow-sim-time", action="store_true")
    parser.add_argument(
        "--allow-simulation-odom-after-stale-tf",
        action="store_true",
        help=(
            "Simulation-survey-only recovery after the existing zero plus "
            "bounded stale-TF retry. Disabled by default and never admitted "
            "for logistics or real-time runs."
        ),
    )
    parser.add_argument("--max-linear-mps", type=float, default=0.055)
    parser.add_argument("--max-angular-radps", type=float, default=0.18)
    parser.add_argument("--goal-tolerance-m", type=float, default=0.08)
    parser.add_argument(
        "--physical-waypoint-tolerance-m",
        type=float,
        default=0.02,
        help=(
            "Capture tolerance for intermediate vertices on certified physical "
            "routes; it must remain inside the certified route tube."
        ),
    )
    parser.add_argument(
        "--physical-goal-tolerance-m",
        type=float,
        default=0.03,
        help=(
            "Simulation dynamic physical-face routes use at most this terminal "
            "position tolerance; acquisition and sampling retain --goal-tolerance-m."
        ),
    )
    parser.add_argument("--heading-tolerance-rad", type=float, default=0.25)
    parser.add_argument("--lookahead-distance-m", type=float, default=0.18)
    parser.add_argument("--slow-heading-error-rad", type=float, default=0.75)
    parser.add_argument("--stop-heading-error-rad", type=float, default=1.25)
    parser.add_argument("--min-linear-speed-scale", type=float, default=0.35)
    parser.add_argument("--max-progress-advance-m", type=float, default=0.45)
    parser.add_argument(
        "--disable-command-smoothing",
        action="store_true",
        help=(
            "Publish raw controller commands after safety checks. By default "
            "normal commands are rate-limited; hard stops still publish zero "
            "immediately."
        ),
    )
    parser.add_argument(
        "--max-linear-accel-mps2",
        type=float,
        default=0.10,
        help="Maximum acceleration for ordinary shaped linear commands.",
    )
    parser.add_argument(
        "--max-angular-accel-radps2",
        type=float,
        default=0.60,
        help="Maximum acceleration for ordinary shaped angular commands.",
    )
    parser.add_argument("--min-obstacle-distance-m", type=float, default=0.20)
    parser.add_argument(
        "--omnidirectional-hard-stop-distance-m",
        type=float,
        default=0.12,
        help=(
            "Unconditional all-angle LiDAR stop used during certified "
            "directional blockage recovery. Normal translation retains "
            "--min-obstacle-distance-m in its commanded-motion sector."
        ),
    )
    parser.add_argument("--front-obstacle-slow-distance-m", type=float, default=0.38)
    parser.add_argument("--front-obstacle-sector-rad", type=float, default=0.6108652381980153)
    parser.add_argument("--thinning-min-spacing-m", type=float, default=0.15)
    parser.add_argument("--max-scan-age-sec", type=float, default=1.0)
    parser.add_argument("--max-odom-age-sec", type=float, default=1.0)
    parser.add_argument("--max-tf-age-sec", type=float, default=1.0)
    parser.add_argument("--max-amcl-age-sec", type=float, default=2.0)
    parser.add_argument(
        "--max-future-timestamp-sec",
        type=float,
        default=0.25,
        help="Reject sensor or TF stamps farther than this into the future.",
    )
    parser.add_argument(
        "--certified-route-tube-radius-m",
        type=float,
        default=DEFAULT_TRACKING_TUBE_RADIUS_M,
        help=(
            "Maximum live base/pursuit deviation from the active certified "
            "polyline segment on physical stand approaches."
        ),
    )
    parser.add_argument(
        "--max-localization-tf-future-sec",
        type=float,
        default=1.1,
        help=(
            "AMCL map->odom forward-stamp allowance. Keep this at least as "
            "large as AMCL transform_tolerance; it does not relax sensor stamps."
        ),
    )
    parser.add_argument(
        "--certified-route-chord-sample-spacing-m",
        type=float,
        default=0.01,
        help="Sampling spacing for runtime pursuit-chord certificate checks.",
    )
    parser.add_argument(
        "--certified-corner-max-reacquire-attempts",
        type=int,
        default=2,
        help=(
            "Maximum bounded exact-vertex reacquisitions while a discovery "
            "corner remains inside the certified route tube."
        ),
    )
    parser.add_argument("--preflight-observation-window-sec", type=float, default=2.0)
    parser.add_argument(
        "--nomotion-update-service",
        default="/request_nomotion_update",
        help="Stationary AMCL refresh service used after preflight subscribers exist.",
    )
    parser.add_argument(
        "--nomotion-update-timeout-sec",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--runtime-nomotion-update-service",
        default="request_nomotion_update",
        help=(
            "Namespace-relative AMCL no-motion service used only for bounded "
            "runtime TF reacquisition. This is separate from the preflight "
            "refresh service."
        ),
    )
    parser.add_argument(
        "--runtime-nomotion-update-timeout-sec",
        type=float,
        default=2.0,
        help=(
            "Fail-closed runtime AMCL refresh budget in seconds; must be "
            "positive and no greater than 2.0."
        ),
    )
    parser.add_argument(
        "--stationary-amcl-sample-count",
        type=int,
        default=5,
        help="Forced no-motion AMCL samples required before physical motion.",
    )
    parser.add_argument(
        "--stationary-amcl-sample-interval-sec",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--max-stationary-amcl-position-spread-m",
        type=float,
        default=0.015,
    )
    parser.add_argument(
        "--max-stationary-amcl-yaw-spread-rad",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--max-stationary-amcl-position-std-m",
        type=float,
        default=0.015,
        help=(
            "Maximum reported AMCL planar one-sigma uncertainty before "
            "physical motion; defaults to half the 0.03 m route tube."
        ),
    )
    parser.add_argument(
        "--max-stationary-amcl-yaw-std-rad",
        type=float,
        default=0.03,
        help="Maximum reported AMCL yaw one-sigma uncertainty before motion.",
    )
    parser.add_argument(
        "--skip-nomotion-update-before-preflight",
        action="store_true",
        help="Disable the automatic stationary AMCL refresh.",
    )
    parser.add_argument("--initial-sensor-wait-sec", type=float, default=2.0)
    parser.add_argument("--waypoint-timeout-sec", type=float, default=45.0)
    parser.add_argument(
        "--axis-acquisition-wait-timeout-sec",
        type=float,
        default=12.0,
        help=(
            "Simulation-only stationary hold at an axis_acquisition goal while "
            "waiting for a committed physical-face route revision."
        ),
    )
    parser.add_argument(
        "--viewpoint-sampling-timeout-sec",
        type=float,
        default=30.0,
        help=(
            "Simulation-only total budget for the viewpoint-sampling phase, "
            "including travel and stationary observation."
        ),
    )
    parser.add_argument(
        "--viewpoint-sampling-target-timeout-sec",
        type=float,
        default=30.0,
        help=(
            "Simulation-only convergence budget for one material sampling "
            "target; newer targets reset this clock but not the total phase clock."
        ),
    )
    parser.add_argument(
        "--viewpoint-sampling-goal-tolerance-m",
        type=float,
        default=0.01,
        help=(
            "Simulation-only position tolerance for axis acquisition and "
            "tangential camera samples; kept tighter than generic transit so "
            "angular viewpoint corrections are not consumed by position error."
        ),
    )
    parser.add_argument(
        "--viewpoint-sampling-terminal-heading-hold-tolerance-m",
        type=float,
        default=INTERMEDIATE_TERMINAL_HEADING_HOLD_TOLERANCE_M,
        help=(
            "Simulation-only bounded terminal-yaw latch radius. Once the "
            "strict position target is captured, leaving this radius stops "
            "the follower instead of resuming point-bearing pursuit."
        ),
    )
    parser.add_argument(
        "--viewpoint-sampling-target-distance-m",
        type=float,
        default=0.33,
        help=(
            "Simulation-only nominal stand-center distance encoded by each "
            "intermediate target pose."
        ),
    )
    parser.add_argument(
        "--viewpoint-sampling-terminal-heading-target-envelope-radius-m",
        type=float,
        default=INTERMEDIATE_TERMINAL_HEADING_TARGET_ENVELOPE_RADIUS_M,
        help=(
            "Simulation-only maximum pose-to-target drift retained during "
            "zero-linear terminal yaw; the stand-distance annulus is checked "
            "independently."
        ),
    )
    parser.add_argument(
        "--viewpoint-sampling-heading-tolerance-rad",
        type=float,
        default=math.radians(5.0),
        help=(
            "Simulation-only terminal heading tolerance for axis acquisition "
            "and camera sampling; kept tight enough for the stand to satisfy "
            "the image-centering gate."
        ),
    )
    parser.add_argument("--stuck-timeout-sec", type=float, default=8.0)
    parser.add_argument("--stuck-progress-epsilon-m", type=float, default=0.03)
    parser.add_argument(
        "--stuck-heading-progress-epsilon-rad",
        type=float,
        default=0.10,
        help=(
            "Minimum controlled-heading improvement that resets the stuck "
            "watchdog while turning toward the active pursuit waypoint."
        ),
    )
    parser.add_argument(
        "--linear-motion-floor-mps",
        type=float,
        default=DEFAULT_LINEAR_MOTION_FLOOR_MPS,
        help=(
            "Minimum clearance-scaled nonzero linear command considered capable "
            "of physical motion; smaller commands hold zero and enter stationary "
            "blockage confirmation on certified routes."
        ),
    )
    parser.add_argument(
        "--blockage-confirmation-timeout-sec",
        type=float,
        default=1.2,
        help="Bounded zero-hold window for persistent front-obstacle evidence.",
    )
    parser.add_argument(
        "--blockage-confirmation-min-samples",
        type=int,
        default=3,
        help="Minimum fresh, distinct front scans required while stopped.",
    )
    parser.add_argument("--initial-distance-limit-m", type=float, default=0.35)
    parser.add_argument(
        "--dynamic-route-refresh-sec",
        type=float,
        default=0.0,
        help="Simulation-only: hot-reload an atomically replaced A* route at this interval.",
    )
    parser.add_argument(
        "--route-manifest",
        type=Path,
        default=None,
        help="Authoritative simulation route-revision manifest (required for dynamic viewpoint routes).",
    )
    parser.add_argument("--max-route-manifest-age-sec", type=float, default=7.0)
    parser.add_argument("--max-route-observation-age-sec", type=float, default=6.0)
    parser.add_argument("--max-route-join-distance-m", type=float, default=0.35)
    parser.add_argument(
        "--dynamic-route-join-tolerance-m",
        type=float,
        default=0.02,
        help=(
            "Simulation-only tolerance for completing the certified join anchor "
            "after a live route revision."
        ),
    )
    parser.add_argument(
        "--start-egress-waypoint-tolerance-m",
        type=float,
        default=0.02,
        help=(
            "Simulation-only release tolerance for waypoint 1 when a route "
            "uses a certified start-cell raster exemption."
        ),
    )
    parser.add_argument(
        "--start-egress-alignment-tolerance-rad",
        type=float,
        default=0.10,
        help=(
            "Simulation-only heading error below which translation may begin "
            "toward a certified start-egress vertex."
        ),
    )
    parser.add_argument(
        "--start-egress-max-linear-mps",
        type=float,
        default=0.03,
        help=(
            "Simulation-only linear-speed cap while pursuing a certified "
            "start-egress vertex."
        ),
    )
    parser.add_argument(
        "--dynamic-route-terminal-lock-distance-m",
        type=float,
        default=0.42,
        help=(
            "Simulation-only distance at which the installed terminal route remains "
            "valid while newer target revisions continue to be polled."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-legacy-simulation-route",
        action="store_true",
        help=(
            "Explicitly allow route_kind=legacy_simulation_waypoint. The CSV "
            "must also declare simulation_only=true and --allow-sim-time is required."
        ),
    )
    parser.add_argument(
        "--allow-unbound-survey-simulation-route",
        action="store_true",
        help=(
            "Allow a static route_purpose=survey demonstration only when both "
            "the CSV and runtime are explicitly simulation-only. Logistics "
            "routes can never use this escape hatch."
        ),
    )
    parser.add_argument("--allow-noop", action="store_true")
    parser.add_argument(
        "--prompt-for-initialpose",
        action="store_true",
        help=(
            "Pause immediately before ROS preflight so the operator can click "
            "RViz 2D Pose Estimate and refresh AMCL."
        ),
    )
    parser.add_argument("--operator-note", default="")
    parser.add_argument("--preflight-json", type=Path, default=None)
    parser.add_argument("--semantic-log", type=Path, default=None)
    parser.add_argument(
        "--verbose-console",
        action="store_true",
        help=(
            "Print full resolved-runtime and ROS preflight JSON to the terminal. "
            "By default the terminal shows a compact summary while artifacts "
            "retain full evidence."
        ),
    )
    parser.add_argument(
        "--controller-trace-jsonl",
        type=Path,
        default=None,
        help=(
            "Append-only controller evidence path. Bundled physical runs derive "
            "controller_trace.jsonl automatically."
        ),
    )
    parser.add_argument(
        "--allowed-cmd-vel-publisher",
        action="append",
        default=[],
        help="Namespace-qualified node identity allowed in preflight, e.g. /robot1/controller_server",
    )
    return parser
