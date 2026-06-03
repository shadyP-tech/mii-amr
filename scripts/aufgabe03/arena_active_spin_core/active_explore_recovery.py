from __future__ import annotations

from .explore_mission import (
    EXPLORE_ACTION_CONFIRM_SHADOW_MAP,
    EXPLORE_ACTION_DRIVE_CANDIDATE,
    EXPLORE_ACTION_FAIL,
    EXPLORE_ACTION_RUN_LOCALIZATION_SPIN,
    EXPLORE_PHASE_FAILED,
    EXPLORE_PHASE_LOCALIZATION_SPIN,
    EXPLORE_PHASE_SHADOW_MAPPING,
    ExploreMissionMotionResult,
    shadow_map_status,
)
from .models import ActiveExploreMotionError


class ActiveExploreRecoveryRunner:
    def __init__(self, session):
        self.session = session

    def run(self, publisher, result):
        session = self.session
        if session.config.recovery_executor not in {"dry_run", "cmd_vel"}:
            raise RuntimeError(
                f"active_explore_executor_unknown:{session.config.recovery_executor}"
            )

        session.active_explore_mission.sync_from_policy()
        total_distance = float(
            session.diagnostics["active_explore"].get("total_distance_m", 0.0)
        )
        while not result.success:
            attempt_index = len(session.diagnostics["active_explore"]["attempts"])
            plan = session.plan_active_explore_recovery(result)
            map_status = shadow_map_status(plan.grid, plan)
            session.diagnostics["active_explore"]["shadow_map_status"] = map_status
            decision = session.active_explore_mission.next_decision(
                result,
                plan,
                map_status,
                current_pose_point=session.latest_odom_point(),
            )
            effective_plan = decision.plan
            selection_diagnostics = decision.diagnostics
            plan_dict = plan.to_dict()
            rejected_unknown = [
                candidate
                for candidate in effective_plan.candidates
                if candidate.rejection_reason == "goal_unknown"
            ]
            attempt_record = {
                "attempt_index": attempt_index,
                "stage": "active_explore",
                "executor": session.config.recovery_executor,
                "previous_failure_reason": result.failure_reason,
                "previous_classifier_reason": result.short_wall_classification.reason,
                "plan": effective_plan.to_dict(),
                "raw_plan": plan_dict,
                "mission_decision": decision.to_dict(),
                **selection_diagnostics,
                "local_grid_stats": (
                    None
                    if effective_plan.grid is None
                    else effective_plan.grid.to_dict()["cell_counts"]
                ),
                "shadow_map_status": map_status,
                "rejected_unknown_space_candidates": len(rejected_unknown),
                "execution": {
                    "executed": False,
                    "stop_reason": "not_started",
                    "driven_distance_m": 0.0,
                },
            }
            session.diagnostics["active_explore"]["attempts"].append(attempt_record)
            preview_limit = min(
                session.config.active_explore_max_single_move_m,
                max(
                    0.0,
                    session.config.active_explore_max_total_distance_m
                    - total_distance,
                ),
            )
            session.publish_active_explore_plan_if_ready(effective_plan, preview_limit)

            if decision.action == EXPLORE_ACTION_CONFIRM_SHADOW_MAP:
                attempt_record["execution"]["stop_reason"] = decision.reason
                continue

            if decision.action == EXPLORE_ACTION_FAIL:
                attempt_record["execution"]["stop_reason"] = decision.reason
                break

            if decision.action == EXPLORE_ACTION_RUN_LOCALIZATION_SPIN:
                spin_result = session.run_active_explore_localization_spin(
                    publisher,
                    attempt_record,
                )
                if spin_result is not None:
                    result = spin_result
                if session.active_explore_mission.phase == EXPLORE_PHASE_LOCALIZATION_SPIN:
                    continue
                if session.active_explore_mission.phase == EXPLORE_PHASE_FAILED:
                    break
                continue

            if decision.action != EXPLORE_ACTION_DRIVE_CANDIDATE:
                attempt_record["execution"]["stop_reason"] = decision.reason
                break

            if session.config.recovery_executor == "dry_run":
                attempt_record["execution"] = {
                    "executor": "dry_run",
                    "executed": False,
                    "stop_reason": "dry_run",
                    "driven_distance_m": 0.0,
                }
                break

            remaining_distance = (
                session.config.active_explore_max_total_distance_m - total_distance
            )
            if remaining_distance <= 0.0:
                attempt_record["execution"]["stop_reason"] = (
                    "active_explore_total_distance_exhausted"
                )
                break
            session.diagnostics["fallback_used"] = True
            try:
                motion_record = session.execute_active_explore_cmd_vel(
                    publisher,
                    decision.selected,
                    distance_limit_m=remaining_distance,
                )
            except ActiveExploreMotionError as exc:
                motion_record = exc.record
                attempt_record["execution"] = motion_record
                total_distance += float(motion_record.get("driven_distance_m", 0.0))
                session.diagnostics["active_explore"][
                    "total_distance_m"
                ] = total_distance
                session.update_active_explore_frontier_progress(
                    motion_record.get("driven_distance_m", 0.0)
                )
                session.clear_active_explore_frontier_goal(exc.reason)
                raise
            except Exception:
                session.clear_active_explore_frontier_goal(
                    "active_explore_motion_failed"
                )
                raise
            total_distance += float(motion_record.get("driven_distance_m", 0.0))
            session.update_active_explore_frontier_progress(
                motion_record.get("driven_distance_m", 0.0)
            )
            if (
                total_distance
                > session.config.active_explore_max_total_distance_m + 1e-6
            ):
                motion_record["stop_reason"] = "active_explore_total_distance_exceeded"
                attempt_record["execution"] = motion_record
                session.diagnostics["active_explore"][
                    "total_distance_m"
                ] = total_distance
                session.clear_active_explore_frontier_goal(
                    "active_explore_total_distance_exceeded"
                )
                raise RuntimeError("active_explore_total_distance_exceeded")

            attempt_record["execution"] = motion_record
            session.diagnostics["active_explore"]["total_distance_m"] = total_distance
            motion_result = ExploreMissionMotionResult.from_execution_record(
                motion_record
            )
            session.active_explore_mission.record_motion(decision, motion_result)
            if session.active_explore_mission.phase == EXPLORE_PHASE_SHADOW_MAPPING:
                decision = {
                    "action": "skip",
                    "reason": "shadow_exploration_not_complete",
                    "active_explore_phase": session.active_explore_phase,
                    "shadow_explore_complete": session.shadow_explore_complete,
                    "shadow_frontier_status": session.diagnostics[
                        "active_explore"
                    ].get("shadow_frontier_status"),
                }
                attempt_record["post_motion_spin_decision"] = decision
                attempt_record["post_recovery_spin_skipped"] = True
                attempt_record["post_recovery_spin_skip_reason"] = decision["reason"]
                session.print_active_explore_phase_spin_skip(decision["reason"])
                continue

            spin_result = session.run_active_explore_localization_spin(
                publisher,
                attempt_record,
            )
            if spin_result is not None:
                result = spin_result
        return result
