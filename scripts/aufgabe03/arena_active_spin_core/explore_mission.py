from __future__ import annotations

from dataclasses import asdict, dataclass

from arena_active_explore import (
    ActiveExplorePlan,
    grid_cell_counts,
    obstacle_shadow_unknown_cells,
)

from .active_explore_policy import (
    ActiveExplorePolicy,
    candidate_path_needs_motion,
    candidate_is_localization_pose_candidate,
    candidate_visible_shadow_count,
)
from .models import (
    ACTIVE_EXPLORE_FRONTIER_UNREACHABLE_REASONS,
    ACTIVE_EXPLORE_PHASE_LOCALIZATION_POSE,
    ACTIVE_EXPLORE_PHASE_LOCALIZATION_SPIN,
    ACTIVE_EXPLORE_PHASE_SHADOW,
    ArenaActiveSpinConfig,
)
from .temporary_map import temporary_grid_localizer_obstacle_mask


EXPLORE_PHASE_SHADOW_MAPPING = "shadow_mapping"
EXPLORE_PHASE_SHADOW_CONFIRM = "shadow_confirm"
EXPLORE_PHASE_HEATER_APPROACH = "heater_approach"
EXPLORE_PHASE_LOCALIZATION_SPIN = "localization_spin"
EXPLORE_PHASE_COMPLETE = "complete"
EXPLORE_PHASE_FAILED = "failed"

EXPLORE_ACTION_DRIVE_CANDIDATE = "drive_candidate"
EXPLORE_ACTION_CONFIRM_SHADOW_MAP = "confirm_shadow_map"
EXPLORE_ACTION_RUN_LOCALIZATION_SPIN = "run_localization_spin"
EXPLORE_ACTION_COMPLETE = "complete"
EXPLORE_ACTION_FAIL = "fail"


@dataclass(frozen=True)
class ExploreMissionDecision:
    action: str
    phase: str
    plan: ActiveExplorePlan
    selected: object | None
    reason: str
    diagnostics: dict

    def to_dict(self):
        return {
            "action": self.action,
            "phase": self.phase,
            "reason": self.reason,
            "selected": None if self.selected is None else self.selected.to_dict(),
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class ExploreMissionMotionResult:
    execution_record: dict
    goal_reached: bool
    final_target_distance_m: float | None
    candidate_kind: str | None
    driven_distance_m: float
    stop_reason: str
    executed: bool

    @classmethod
    def from_execution_record(cls, record):
        return cls(
            execution_record=dict(record),
            goal_reached=bool(record.get("goal_reached", False)),
            final_target_distance_m=record.get("final_target_distance_m"),
            candidate_kind=record.get("candidate_kind"),
            driven_distance_m=float(record.get("driven_distance_m", 0.0) or 0.0),
            stop_reason=str(record.get("stop_reason", "")),
            executed=bool(record.get("executed", False)),
        )

    def to_dict(self):
        return asdict(self)


def shadow_map_status(grid, plan=None):
    candidates = () if plan is None else tuple(plan.candidates)
    frontier_candidates = [
        candidate
        for candidate in candidates
        if candidate is not None and candidate.kind == "obstacle_shadow_frontier"
    ]
    accepted_frontiers = [
        candidate
        for candidate in frontier_candidates
        if candidate.accepted
    ]
    reachable_frontiers = [
        candidate
        for candidate in accepted_frontiers
        if (
            candidate_visible_shadow_count(candidate) > 0
            and candidate_path_needs_motion(candidate)
        )
    ]
    unreachable_frontiers = [
        candidate
        for candidate in frontier_candidates
        if (
            not candidate.accepted
            and candidate.rejection_reason in ACTIVE_EXPLORE_FRONTIER_UNREACHABLE_REASONS
        )
    ]
    accepted_localization_candidates = [
        candidate
        for candidate in candidates
        if candidate_is_localization_pose_candidate(candidate)
    ]
    active_shadow_frontiers = [*reachable_frontiers, *unreachable_frontiers]

    if grid is None:
        shadow_unknown_cells = None
        raw_shadow_unknown_cell_count = None
        cell_counts = None
        obstacle_mask = set()
        protected_wall_cells = set()
        mask_diagnostics = {}
    else:
        shadow_unknown_cells = obstacle_shadow_unknown_cells(grid)
        raw_shadow_unknown_cell_count = len(shadow_unknown_cells)
        if not active_shadow_frontiers:
            shadow_unknown_cells = set()
        cell_counts = grid_cell_counts(grid)
        obstacle_mask, protected_wall_cells, mask_diagnostics = (
            temporary_grid_localizer_obstacle_mask(grid)
        )

    status = {
        "available": grid is not None,
        "shadow_unknown_cell_count": (
            None if shadow_unknown_cells is None else len(shadow_unknown_cells)
        ),
        "raw_shadow_unknown_cell_count": raw_shadow_unknown_cell_count,
        "cell_counts": cell_counts,
        "obstacle_mask_cell_count": len(obstacle_mask),
        "protected_wall_cell_count": len(protected_wall_cells),
        "shadow_frontier_candidate_count": len(frontier_candidates),
        "active_shadow_frontier_count": len(active_shadow_frontiers),
        "unreachable_shadow_frontier_count": len(unreachable_frontiers),
        "reachable_shadow_frontier_count": len(reachable_frontiers),
        "accepted_shadow_frontier_count": len(accepted_frontiers),
        "accepted_localization_candidate_count": len(
            accepted_localization_candidates
        ),
    }
    status.update(mask_diagnostics)
    return status


class ExploreMissionController:
    def __init__(
        self,
        config: ArenaActiveSpinConfig,
        diagnostics,
        policy: ActiveExplorePolicy | None = None,
    ):
        self.config = config
        self.diagnostics = diagnostics
        self.policy = policy or ActiveExplorePolicy(diagnostics)
        self.phase = EXPLORE_PHASE_SHADOW_MAPPING
        self.motion_attempts = 0
        self.shadow_confirmation_count = 0
        self.shadow_stall_replans = 0
        self.localization_pose_attempts = 0
        self.last_shadow_unknown_cell_count = None
        self.last_selected_candidate = None
        self.sync_from_policy()
        self._update_diagnostics()

    def sync_from_policy(self):
        if self.policy.phase == ACTIVE_EXPLORE_PHASE_LOCALIZATION_SPIN:
            self.phase = EXPLORE_PHASE_LOCALIZATION_SPIN
        elif self.policy.phase == ACTIVE_EXPLORE_PHASE_LOCALIZATION_POSE:
            self.phase = EXPLORE_PHASE_HEATER_APPROACH
        else:
            self.phase = EXPLORE_PHASE_SHADOW_MAPPING
        if self.policy.shadow_explore_complete and self.phase == EXPLORE_PHASE_SHADOW_MAPPING:
            self.phase = EXPLORE_PHASE_HEATER_APPROACH
        self._sync_policy_phase()

    def _sync_policy_phase(self):
        if self.phase in {
            EXPLORE_PHASE_SHADOW_MAPPING,
            EXPLORE_PHASE_SHADOW_CONFIRM,
        }:
            self.policy.set_phase(ACTIVE_EXPLORE_PHASE_SHADOW)
            self.policy.shadow_explore_complete = False
        elif self.phase == EXPLORE_PHASE_HEATER_APPROACH:
            self.policy.shadow_explore_complete = True
            self.policy.set_phase(ACTIVE_EXPLORE_PHASE_LOCALIZATION_POSE)
        elif self.phase == EXPLORE_PHASE_LOCALIZATION_SPIN:
            self.policy.shadow_explore_complete = True
            self.policy.set_phase(ACTIVE_EXPLORE_PHASE_LOCALIZATION_SPIN)
        elif self.phase == EXPLORE_PHASE_COMPLETE:
            self.policy.shadow_explore_complete = True
            self.policy.set_phase(ACTIVE_EXPLORE_PHASE_LOCALIZATION_SPIN)
        self.policy.shadow_frontier_empty_replans = self.shadow_confirmation_count
        self.policy.update_phase_diagnostics()

    def _update_diagnostics(self):
        active = self.diagnostics["active_explore"]
        active["mission"] = {
            "phase": self.phase,
            "motion_attempts": self.motion_attempts,
            "max_motion_attempts": self.config.active_explore_max_attempts,
            "shadow_confirmation_count": self.shadow_confirmation_count,
            "shadow_completion_confirmations_required": (
                self.config.active_explore_shadow_completion_confirmations
            ),
            "shadow_stall_replans": self.shadow_stall_replans,
            "max_shadow_stall_replans": (
                self.config.active_explore_max_shadow_stall_replans
            ),
            "localization_pose_attempts": self.localization_pose_attempts,
            "max_localization_pose_attempts": (
                self.config.active_explore_max_localization_pose_attempts
            ),
            "last_shadow_unknown_cell_count": self.last_shadow_unknown_cell_count,
            "last_selected_candidate_kind": (
                None
                if self.last_selected_candidate is None
                else self.last_selected_candidate.kind
            ),
        }
        active["motion_attempts"] = self.motion_attempts
        self._sync_policy_phase()

    def _effective_plan(self, plan, selected, ok=True, reason=None):
        return ActiveExplorePlan(
            ok,
            plan.reason if reason is None else reason,
            selected,
            plan.candidates,
            plan.grid,
        )

    def _shadow_frontier_status(self, plan):
        status = self.policy.shadow_frontier_status_from_plan(plan)
        status["empty_replans"] = self.shadow_confirmation_count
        status["complete"] = self.policy.shadow_explore_complete
        self.diagnostics["active_explore"]["shadow_frontier_status"] = status
        return status

    def _diagnostics(
        self,
        plan,
        map_status,
        selected,
        selection_policy,
        **extra,
    ):
        shadow_frontier_status = extra.pop(
            "shadow_frontier_status",
            self._shadow_frontier_status(plan),
        )
        diagnostics = {
            "mission_phase": self.phase,
            "active_explore_phase": self.policy.phase,
            "motion_attempts": self.motion_attempts,
            "shadow_confirmation_count": self.shadow_confirmation_count,
            "shadow_frontier_empty_replans": self.shadow_confirmation_count,
            "shadow_stall_replans": self.shadow_stall_replans,
            "shadow_explore_complete": self.policy.shadow_explore_complete,
            "shadow_map_status": map_status,
            "shadow_frontier_status": shadow_frontier_status,
            "default_selected": (
                None if plan.selected is None else plan.selected.to_dict()
            ),
            "effective_selected": (
                None if selected is None else selected.to_dict()
            ),
            "selection_policy": selection_policy,
            "persistent_frontier_goal": self.policy.frontier_goal_diagnostics(),
            "persistent_frontier_match": extra.pop("persistent_frontier_match", None),
            "persistent_frontier_abandon_reason": extra.pop(
                "persistent_frontier_abandon_reason",
                None,
            ),
            "localization_candidate_policy": extra.pop(
                "localization_candidate_policy",
                None,
            ),
            "shadow_approach_fallback_policy": extra.pop(
                "shadow_approach_fallback_policy",
                None,
            ),
            "continue_without_motion": extra.pop("continue_without_motion", False),
        }
        diagnostics.update(extra)
        self.diagnostics["active_explore"]["shadow_map_status"] = map_status
        self.diagnostics["active_explore"]["localization_candidate_policy"] = (
            diagnostics["localization_candidate_policy"]
        )
        self.diagnostics["active_explore"]["shadow_approach_fallback_policy"] = (
            diagnostics["shadow_approach_fallback_policy"]
        )
        self._update_diagnostics()
        return diagnostics

    def _decision(
        self,
        action,
        plan,
        selected,
        reason,
        diagnostics,
    ):
        self._update_diagnostics()
        return ExploreMissionDecision(
            action=action,
            phase=self.phase,
            plan=plan,
            selected=selected,
            reason=reason,
            diagnostics=diagnostics,
        )

    def _drive_decision(self, plan, selected, reason, diagnostics):
        if self.motion_attempts >= self.config.active_explore_max_attempts:
            fail_plan = self._effective_plan(
                plan,
                None,
                ok=False,
                reason="active_explore_motion_attempts_exhausted",
            )
            diagnostics = {
                **diagnostics,
                "effective_selected": None,
                "continue_without_motion": False,
            }
            return self._decision(
                EXPLORE_ACTION_FAIL,
                fail_plan,
                None,
                "active_explore_motion_attempts_exhausted",
                diagnostics,
            )
        self.last_selected_candidate = selected
        return self._decision(
            EXPLORE_ACTION_DRIVE_CANDIDATE,
            self._effective_plan(plan, selected),
            selected,
            reason,
            diagnostics,
        )

    def _confirm_shadow_decision(self, plan, map_status, reason):
        self.phase = EXPLORE_PHASE_SHADOW_CONFIRM
        self.shadow_confirmation_count += 1
        self.shadow_stall_replans = 0
        self.last_shadow_unknown_cell_count = map_status.get("shadow_unknown_cell_count")
        if (
            self.shadow_confirmation_count
            < self.config.active_explore_shadow_completion_confirmations
        ):
            wait_plan = self._effective_plan(plan, None, ok=False, reason=reason)
            diagnostics = self._diagnostics(
                wait_plan,
                map_status,
                None,
                "shadow_map_confirmation",
                continue_without_motion=True,
            )
            return self._decision(
                EXPLORE_ACTION_CONFIRM_SHADOW_MAP,
                wait_plan,
                None,
                reason,
                diagnostics,
            )

        self.phase = EXPLORE_PHASE_HEATER_APPROACH
        self.policy.clear_frontier_goal("shadow_map_confirmed_complete")
        self._sync_policy_phase()
        return self._select_heater_pose(plan, map_status)

    def _stall_or_fail_shadow(self, plan, map_status, reason, diagnostics):
        self.shadow_confirmation_count = 0
        self.shadow_stall_replans += 1
        self.last_shadow_unknown_cell_count = map_status.get("shadow_unknown_cell_count")
        if self.shadow_stall_replans >= self.config.active_explore_max_shadow_stall_replans:
            fail_plan = self._effective_plan(
                plan,
                None,
                ok=False,
                reason="shadow_mapping_incomplete",
            )
            fail_diagnostics = {
                **diagnostics,
                "effective_selected": None,
                "continue_without_motion": False,
                "shadow_mapping_incomplete_reason": reason,
            }
            return self._decision(
                EXPLORE_ACTION_FAIL,
                fail_plan,
                None,
                "shadow_mapping_incomplete",
                fail_diagnostics,
            )
        wait_plan = self._effective_plan(plan, None, ok=False, reason=reason)
        wait_diagnostics = {
            **diagnostics,
            "effective_selected": None,
            "continue_without_motion": True,
        }
        return self._decision(
            EXPLORE_ACTION_CONFIRM_SHADOW_MAP,
            wait_plan,
            None,
            reason,
            wait_diagnostics,
        )

    def _select_shadow_candidate(self, plan, map_status, current_pose_point):
        shadow_status = self._shadow_frontier_status(plan)
        moving_frontiers = self.policy.moving_shadow_frontier_candidates(plan)
        persistent_match = None
        abandon_reason = None
        if moving_frontiers:
            selected = None
            selection_policy = "shadow_frontier_best"
            if self.policy.frontier_goal is not None:
                selected, persistent_match = (
                    self.policy.matching_active_explore_frontier_candidate(
                        plan,
                        candidates=moving_frontiers,
                    )
                )
                if selected is None:
                    abandon_reason = "no_matching_accepted_frontier"
                    self.policy.clear_frontier_goal(abandon_reason)
                else:
                    selection_policy = "persistent_frontier"
            if selected is None:
                selected = self.policy.best_scored_candidate(moving_frontiers)
            if selected is not None:
                self.policy.store_frontier_goal(selected, self.motion_attempts)
                self.shadow_confirmation_count = 0
                self.shadow_stall_replans = 0
                diagnostics = self._diagnostics(
                    plan,
                    map_status,
                    selected,
                    selection_policy,
                    shadow_frontier_status=shadow_status,
                    persistent_frontier_match=persistent_match,
                    persistent_frontier_abandon_reason=abandon_reason,
                )
                return self._drive_decision(
                    plan,
                    selected,
                    "shadow_frontier_selected",
                    diagnostics,
                )

        shadow_approach_policy = None
        if shadow_status["shadow_frontier_state"] == "unreachable":
            fallback, shadow_approach_policy = (
                self.policy.shadow_approach_fallback_candidate(
                    plan,
                    current_pose_point,
                )
            )
            self.diagnostics["active_explore"]["shadow_approach_fallback_policy"] = (
                shadow_approach_policy
            )
            if fallback is not None:
                self.shadow_confirmation_count = 0
                self.shadow_stall_replans = 0
                diagnostics = self._diagnostics(
                    plan,
                    map_status,
                    fallback,
                    "shadow_approach_fallback",
                    shadow_frontier_status=shadow_status,
                    shadow_approach_fallback_policy=shadow_approach_policy,
                )
                return self._drive_decision(
                    plan,
                    fallback,
                    "shadow_approach_fallback_selected",
                    diagnostics,
                )
            if shadow_approach_policy["candidate_count"] <= 0:
                reason = "shadow_frontier_unreachable_no_approach_candidate"
            elif shadow_approach_policy["reason"] == "no_safe_open_corridor_candidate":
                reason = "shadow_frontier_unreachable_no_safe_approach_candidate"
            else:
                reason = "shadow_frontier_unreachable_no_goal_approach_candidate"
        else:
            reason = "shadow_mapping_incomplete_no_progress_candidate"

        diagnostics = self._diagnostics(
            plan,
            map_status,
            None,
            "shadow_approach_fallback",
            shadow_frontier_status=shadow_status,
            shadow_approach_fallback_policy=shadow_approach_policy,
            continue_without_motion=True,
        )
        return self._stall_or_fail_shadow(plan, map_status, reason, diagnostics)

    def _select_heater_pose(self, plan, map_status):
        selected, localization_policy = self.policy.localization_pose_candidate(plan)
        self.diagnostics["active_explore"]["localization_candidate_policy"] = (
            localization_policy
        )
        if selected is None:
            self.localization_pose_attempts += 1
            reason = localization_policy["reason"]
            diagnostics = self._diagnostics(
                plan,
                map_status,
                None,
                "localization_pose_required",
                localization_candidate_policy=localization_policy,
                continue_without_motion=(
                    self.localization_pose_attempts
                    < self.config.active_explore_max_localization_pose_attempts
                ),
            )
            if (
                self.localization_pose_attempts
                >= self.config.active_explore_max_localization_pose_attempts
            ):
                fail_plan = self._effective_plan(plan, None, ok=False, reason=reason)
                return self._decision(
                    EXPLORE_ACTION_FAIL,
                    fail_plan,
                    None,
                    reason,
                    diagnostics,
                )
            wait_plan = self._effective_plan(plan, None, ok=False, reason=reason)
            return self._decision(
                EXPLORE_ACTION_CONFIRM_SHADOW_MAP,
                wait_plan,
                None,
                reason,
                diagnostics,
            )

        diagnostics = self._diagnostics(
            plan,
            map_status,
            selected,
            "localization_pose",
            localization_candidate_policy=localization_policy,
        )
        return self._drive_decision(
            plan,
            selected,
            "localization_pose_selected",
            diagnostics,
        )

    def next_decision(self, result, plan, map_status, current_pose_point=None):
        self.last_shadow_unknown_cell_count = map_status.get("shadow_unknown_cell_count")
        if getattr(result, "success", False):
            self.phase = EXPLORE_PHASE_COMPLETE
            diagnostics = self._diagnostics(plan, map_status, None, "result_success")
            return self._decision(
                EXPLORE_ACTION_COMPLETE,
                plan,
                None,
                "result_success",
                diagnostics,
            )

        if self.phase == EXPLORE_PHASE_LOCALIZATION_SPIN:
            diagnostics = self._diagnostics(
                plan,
                map_status,
                None,
                "localization_spin_ready",
            )
            return self._decision(
                EXPLORE_ACTION_RUN_LOCALIZATION_SPIN,
                plan,
                None,
                "localization_spin_ready",
                diagnostics,
            )

        if self.phase in {EXPLORE_PHASE_COMPLETE, EXPLORE_PHASE_FAILED}:
            action = (
                EXPLORE_ACTION_COMPLETE
                if self.phase == EXPLORE_PHASE_COMPLETE
                else EXPLORE_ACTION_FAIL
            )
            diagnostics = self._diagnostics(plan, map_status, None, self.phase)
            return self._decision(action, plan, None, self.phase, diagnostics)

        if not plan.ok:
            diagnostics = self._diagnostics(
                plan,
                map_status,
                None,
                "plan_not_ok",
            )
            return self._decision(
                EXPLORE_ACTION_FAIL,
                self._effective_plan(plan, None, ok=False, reason=plan.reason),
                None,
                plan.reason,
                diagnostics,
            )

        if self.phase == EXPLORE_PHASE_HEATER_APPROACH:
            return self._select_heater_pose(plan, map_status)

        shadow_unknown = map_status.get("shadow_unknown_cell_count")
        if shadow_unknown == 0:
            return self._confirm_shadow_decision(
                plan,
                map_status,
                "shadow_frontier_empty_replan_wait",
            )

        if shadow_unknown is None:
            shadow_status = self._shadow_frontier_status(plan)
            if shadow_status["shadow_frontier_state"] == "absent":
                return self._confirm_shadow_decision(
                    plan,
                    map_status,
                    "shadow_map_unavailable_frontier_confirm_replan_wait",
                )

        return self._select_shadow_candidate(plan, map_status, current_pose_point)

    def record_motion(self, decision, motion_result: ExploreMissionMotionResult):
        if decision.action != EXPLORE_ACTION_DRIVE_CANDIDATE:
            return
        if motion_result.executed:
            self.motion_attempts += 1
        self.policy.update_frontier_progress(motion_result.driven_distance_m)
        selected = decision.selected
        if selected is None:
            self._update_diagnostics()
            return
        if selected.kind in {"obstacle_shadow_frontier", "open_corridor"}:
            if decision.diagnostics.get("selection_policy") == "localization_pose":
                self.localization_pose_attempts += 1
                self.phase = EXPLORE_PHASE_LOCALIZATION_SPIN
            else:
                self.phase = EXPLORE_PHASE_SHADOW_MAPPING
        elif candidate_is_localization_pose_candidate(selected):
            self.localization_pose_attempts += 1
            self.phase = EXPLORE_PHASE_LOCALIZATION_SPIN
        else:
            self.phase = EXPLORE_PHASE_SHADOW_MAPPING
        self._update_diagnostics()

    def record_spin(self, result):
        success = bool(getattr(result, "success", False))
        if success:
            self.phase = EXPLORE_PHASE_COMPLETE
            self.policy.clear_frontier_goal("localization_success")
            self._update_diagnostics()
            return
        if (
            self.localization_pose_attempts
            >= self.config.active_explore_max_localization_pose_attempts
        ):
            self.phase = EXPLORE_PHASE_FAILED
        else:
            self.phase = EXPLORE_PHASE_HEATER_APPROACH
        self._update_diagnostics()
