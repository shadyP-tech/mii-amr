"""Initial coverage planning and optional startup-localization composition."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Callable, Mapping

from scripts.aufgabe04.navigation.coverage.exact_two_viewpoint_selection import (
    DEFAULT_EXACT_TWO_CANDIDATE_SPACING_M,
    DEFAULT_MINIMUM_EXACT_TWO_VIEWPOINT_BASELINE_M,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    CoverageSurveyPlan,
    load_coverage_survey_plan,
)
from scripts.aufgabe04.navigation.execution.route_uncertainty_defaults import (
    DEFAULT_UNCERTAINTY_BRAKING_LATENCY_DISTANCE_M,
    DEFAULT_UNCERTAINTY_CLEARANCE_SAMPLE_SPACING_M,
    DEFAULT_UNCERTAINTY_ODOM_DRIFT_BOUND_M,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.startup_active_localization import (
    StartupActiveLocalizationConfig,
)
from scripts.aufgabe04.navigation.missions.plan_stand_coverage_survey import (
    main as plan_stand_coverage_survey,
)
from scripts.aufgabe04.real_robot.execution.child_runner import (
    DEFAULT_COLLISION_MARGIN_M,
    DEFAULT_TRACKING_TUBE_RADIUS_M,
)
from scripts.aufgabe04.real_robot.execution.startup_active_localization import (
    StartupActiveLocalizationChildRequest,
    run_startup_active_localization_child,
)
from scripts.aufgabe04.real_robot.readiness.active_localization import (
    InitialCoveragePlanningAttempt,
    StartupActiveLocalizationPlanningConfig,
    StartupActiveLocalizationPlanningEffects,
    plan_with_optional_startup_active_localization,
)


class InitialCoveragePlanningStatusError(RuntimeError):
    """Preserve a nonzero legacy planner return code across composition."""

    def __init__(self, status: int) -> None:
        if type(status) is not int or status == 0:
            raise ValueError("planning status must be a nonzero integer")
        self.status = status
        super().__init__(
            f"initial coverage planning failed with status {status}"
        )


def startup_active_localization_config_from_args(
    args,
) -> StartupActiveLocalizationConfig:
    """Build the shared, validated policy from autonomous CLI arguments."""

    return StartupActiveLocalizationConfig(
        enabled=args.enable_startup_active_localization,
        max_attempts=args.max_startup_active_localization_attempts,
        rotation_rad=args.startup_active_localization_rotation_rad,
        angular_speed_radps=(
            args.startup_active_localization_angular_speed_radps
        ),
        timeout_sec=args.startup_active_localization_timeout_sec,
    )


def build_initial_coverage_planning_command(
    *,
    args,
    profile,
    session_root: Path,
    survey_root: Path,
    start: Pose2D,
    inflation_radius_m: float,
    candidate_keepout_radius_m: float,
    route_selection_preflight_path: Path | None,
    route_selection_evidence_path: Path | None = None,
) -> list[str]:
    """Build one motion-free initial survey-planner invocation."""

    command = [
        "--map",
        str(args.map),
        "--semantic-map-id",
        args.semantic_map_id,
        "--planning-frame",
        profile.map_frame,
        "--start-x",
        str(start.x_m),
        "--start-y",
        str(start.y_m),
        "--start-yaw",
        str(start.yaw_rad),
        "--survey-id",
        args.session_id,
        "--output-dir",
        str(survey_root),
        "--lane-count",
        "1",
        "--stop-spacing-m",
        str(args.inspection_stop_spacing_m),
        "--inflation-radius-m",
        str(inflation_radius_m),
        "--candidate-keepout-radius-m",
        str(candidate_keepout_radius_m),
        "--expected-stand-count",
        str(args.expected_stand_count),
    ]
    if args.exact_inspection_point_count is None:
        return command

    if route_selection_preflight_path is None:
        route_selection_preflight_path = (
            session_root / "preflight/preplanning_localization.json"
        )
    command.extend(
        [
            "--exact-inspection-point-count",
            str(args.exact_inspection_point_count),
            "--exact-two-candidate-spacing-m",
            str(DEFAULT_EXACT_TWO_CANDIDATE_SPACING_M),
            "--minimum-exact-two-viewpoint-baseline-m",
            str(DEFAULT_MINIMUM_EXACT_TWO_VIEWPOINT_BASELINE_M),
            "--startup-route-selection-preflight-json",
            str(route_selection_preflight_path),
            "--startup-route-selection-robot-radius-m",
            str(profile.robot_radius_m),
            "--startup-route-selection-collision-margin-m",
            str(DEFAULT_COLLISION_MARGIN_M),
            "--startup-route-selection-tracking-tube-radius-m",
            str(DEFAULT_TRACKING_TUBE_RADIUS_M),
            "--startup-route-selection-odom-drift-bound-m",
            str(DEFAULT_UNCERTAINTY_ODOM_DRIFT_BOUND_M),
            "--startup-route-selection-braking-latency-distance-m",
            str(DEFAULT_UNCERTAINTY_BRAKING_LATENCY_DISTANCE_M),
            "--startup-route-selection-sigma-multiplier",
            str(args.uncertainty_sigma_multiplier),
            "--startup-route-selection-clearance-sample-spacing-m",
            str(DEFAULT_UNCERTAINTY_CLEARANCE_SAMPLE_SPACING_M),
        ]
    )
    if route_selection_evidence_path is not None:
        command.extend(
            [
                "--startup-route-selection-evidence-json",
                str(route_selection_evidence_path),
            ]
        )
    return command


def plan_initial_coverage(
    *,
    args,
    profile,
    session_root: Path,
    survey_root: Path,
    start: Pose2D,
    inflation_radius_m: float,
    candidate_keepout_radius_m: float,
    admit_stationary_localization: Callable[[Path], Pose2D],
    append_event: Callable[[Path, dict[str, object]], None],
    planner: Callable[..., int] = plan_stand_coverage_survey,
    active_localization_child: Callable[..., object] = (
        run_startup_active_localization_child
    ),
    load_plan: Callable[[Path], CoverageSurveyPlan] = load_coverage_survey_plan,
    wall_clock: Callable[[], float] = time.time,
) -> tuple[Path, CoverageSurveyPlan, int, Pose2D]:
    """Compose planner, active-localization child, and fresh AMCL admission."""

    motion_config = startup_active_localization_config_from_args(args)

    def plan_attempt(attempt: InitialCoveragePlanningAttempt) -> int:
        command = build_initial_coverage_planning_command(
            args=args,
            profile=profile,
            session_root=session_root,
            survey_root=survey_root,
            start=attempt.start,
            inflation_radius_m=inflation_radius_m,
            candidate_keepout_radius_m=candidate_keepout_radius_m,
            route_selection_preflight_path=attempt.preflight_path,
            route_selection_evidence_path=attempt.selection_evidence_path,
        )
        if attempt.propagate_route_selection_rejection:
            return planner(
                command,
                propagate_startup_route_selection_rejection=True,
            )
        return planner(command)

    def run_active(attempt_index, rejection) -> Mapping[str, object]:
        outcome = active_localization_child(
            StartupActiveLocalizationChildRequest(
                session_id=args.session_id,
                session_root=session_root,
                profile=profile,
                config=motion_config,
                attempt_index=attempt_index,
                rejected_selection=rejection,
            )
        )
        result = getattr(outcome, "result", None)
        if not isinstance(result, Mapping):
            raise RuntimeError(
                "startup active-localization child returned no result mapping"
            )
        return result

    planning = plan_with_optional_startup_active_localization(
        StartupActiveLocalizationPlanningConfig(
            session_root=session_root,
            motion=motion_config,
        ),
        StartupActiveLocalizationPlanningEffects(
            plan_initial_route=plan_attempt,
            run_active_localization=run_active,
            admit_stationary_localization=admit_stationary_localization,
            append_event=append_event,
            wall_clock=wall_clock,
        ),
        initial_start=start,
    )
    if planning.planning_status != 0:
        raise InitialCoveragePlanningStatusError(planning.planning_status)
    plan_path = survey_root / "coverage_plan.json"
    return plan_path, load_plan(plan_path), 0, planning.start


__all__ = [
    "InitialCoveragePlanningStatusError",
    "build_initial_coverage_planning_command",
    "plan_initial_coverage",
    "startup_active_localization_config_from_args",
]
