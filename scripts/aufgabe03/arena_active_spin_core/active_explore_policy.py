from __future__ import annotations

import math

from arena_active_explore import ActiveExplorePlan

from .math_utils import distance_2d, finite_point_2d
from .models import (
    ACTIVE_EXPLORE_FRONTIER_CLUSTER_MATCH_M,
    ACTIVE_EXPLORE_FRONTIER_REACHED_PATH_M,
    ACTIVE_EXPLORE_FRONTIER_TARGET_MATCH_M,
    ACTIVE_EXPLORE_FRONTIER_UNREACHABLE_REASONS,
    ACTIVE_EXPLORE_LOCALIZATION_CANDIDATE_KINDS,
    ACTIVE_EXPLORE_PHASE_LOCALIZATION_POSE,
    ACTIVE_EXPLORE_PHASE_SHADOW,
    ACTIVE_EXPLORE_SHADOW_APPROACH_GOAL_DRIFT_TOLERANCE_M,
    ACTIVE_EXPLORE_SHADOW_APPROACH_MIN_PATH_CLEARANCE_M,
    ACTIVE_EXPLORE_SHADOW_EMPTY_REPLANS_TO_COMPLETE,
)


def candidate_visible_shadow_count(candidate):
    if candidate is None:
        return 0
    metadata = candidate.metadata or {}
    value = metadata.get("visible_cluster_shadow_count")
    if value is None:
        value = candidate.score_components.get("visible_shadow_unknown_count")
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def candidate_cluster_centroid(candidate):
    return finite_point_2d((candidate.metadata or {}).get("cluster_centroid_world"))


def candidate_path_needs_motion(candidate):
    path_length = None if candidate is None else candidate.path_length_m
    return path_length is None or path_length > ACTIVE_EXPLORE_FRONTIER_REACHED_PATH_M


def candidate_is_accepted_shadow_frontier(candidate):
    return (
        candidate is not None
        and candidate.accepted
        and candidate.kind == "obstacle_shadow_frontier"
    )


def candidate_is_moving_shadow_frontier(candidate):
    return (
        candidate_is_accepted_shadow_frontier(candidate)
        and candidate_visible_shadow_count(candidate) > 0
        and candidate_path_needs_motion(candidate)
    )


def candidate_is_localization_pose_candidate(candidate):
    return (
        candidate is not None
        and candidate.accepted
        and candidate.kind in ACTIVE_EXPLORE_LOCALIZATION_CANDIDATE_KINDS
    )


def candidate_is_accepted_open_corridor(candidate):
    return (
        candidate is not None
        and candidate.accepted
        and candidate.kind == "open_corridor"
    )


def candidate_path_min_clearance_m(candidate):
    if candidate is None:
        return None
    value = (candidate.score_components or {}).get("path_min_clearance_m")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def candidate_has_safe_shadow_approach_clearance(candidate):
    clearance = candidate_path_min_clearance_m(candidate)
    return (
        clearance is not None
        and clearance >= ACTIVE_EXPLORE_SHADOW_APPROACH_MIN_PATH_CLEARANCE_M
    )


def candidate_target_point(candidate):
    if candidate is None:
        return None
    return finite_point_2d([candidate.target_x, candidate.target_y])


def candidate_path_start_point(candidate):
    if candidate is None:
        return None
    if candidate.path_world:
        return finite_point_2d(candidate.path_world[0])
    if candidate.simplified_path_world:
        return finite_point_2d(candidate.simplified_path_world[0])
    return None


def candidate_path_unknown_ratio(candidate):
    if candidate is None:
        return 0.0
    value = (candidate.score_components or {}).get("path_unknown_ratio", 0.0)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


class ActiveExplorePolicy:
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        self.frontier_goal = None
        self.phase = ACTIVE_EXPLORE_PHASE_SHADOW
        self.shadow_frontier_empty_replans = 0
        self.shadow_explore_complete = False

    def frontier_goal_diagnostics(self):
        if self.frontier_goal is None:
            return None
        goal = dict(self.frontier_goal)
        if goal.get("cluster_centroid_world") is not None:
            goal["cluster_centroid_world"] = list(goal["cluster_centroid_world"])
        return goal

    def clear_frontier_goal(self, _reason):
        self.frontier_goal = None
        self.diagnostics["active_explore"]["persistent_frontier_goal"] = None

    def store_frontier_goal(self, candidate, attempt_index):
        metadata = candidate.metadata or {}
        previous = self.frontier_goal or {}
        cluster_centroid = candidate_cluster_centroid(candidate)
        goal = {
            "target_x": float(candidate.target_x),
            "target_y": float(candidate.target_y),
            "cluster_centroid_world": cluster_centroid,
            "cluster_size": metadata.get("cluster_size"),
            "visible_cluster_shadow_count": candidate_visible_shadow_count(candidate),
            "created_attempt_index": previous.get(
                "created_attempt_index",
                attempt_index,
            ),
            "last_matched_attempt_index": attempt_index,
            "driven_toward_goal_m": float(previous.get("driven_toward_goal_m", 0.0)),
        }
        self.frontier_goal = goal
        self.diagnostics["active_explore"]["persistent_frontier_goal"] = (
            self.frontier_goal_diagnostics()
        )
        return goal

    def update_frontier_progress(self, driven_distance_m):
        if self.frontier_goal is None:
            return
        self.frontier_goal["driven_toward_goal_m"] = float(
            self.frontier_goal.get("driven_toward_goal_m", 0.0)
        ) + max(0.0, float(driven_distance_m))
        self.diagnostics["active_explore"]["persistent_frontier_goal"] = (
            self.frontier_goal_diagnostics()
        )

    def set_phase(self, phase):
        self.phase = phase
        self.diagnostics["active_explore"]["active_explore_phase"] = phase

    def update_phase_diagnostics(self):
        self.diagnostics["active_explore"]["active_explore_phase"] = (
            self.phase
        )
        self.diagnostics["active_explore"]["shadow_frontier_empty_replans"] = (
            self.shadow_frontier_empty_replans
        )
        self.diagnostics["active_explore"]["shadow_explore_complete"] = (
            self.shadow_explore_complete
        )

    def shadow_frontier_status_from_plan(self, plan):
        frontier_candidates = [
            candidate
            for candidate in plan.candidates
            if candidate is not None and candidate.kind == "obstacle_shadow_frontier"
        ]
        accepted_frontiers = [
            candidate
            for candidate in frontier_candidates
            if candidate.accepted
        ]
        rejected_frontiers = [
            candidate
            for candidate in frontier_candidates
            if not candidate.accepted
        ]
        visible_frontiers = [
            candidate
            for candidate in accepted_frontiers
            if candidate_visible_shadow_count(candidate) > 0
        ]
        moving_frontiers = [
            candidate
            for candidate in visible_frontiers
            if candidate_path_needs_motion(candidate)
        ]
        path_lengths = [
            candidate.path_length_m
            for candidate in visible_frontiers
            if candidate.path_length_m is not None
        ]
        all_visible_counts = [
            candidate_visible_shadow_count(candidate)
            for candidate in frontier_candidates
            if candidate_visible_shadow_count(candidate) > 0
        ]
        rejection_reasons = {}
        unreachable_frontiers = []
        for candidate in rejected_frontiers:
            reason = candidate.rejection_reason or "unknown"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            if reason in ACTIVE_EXPLORE_FRONTIER_UNREACHABLE_REASONS:
                unreachable_frontiers.append(candidate)
        if moving_frontiers:
            shadow_frontier_state = "reachable"
        elif frontier_candidates:
            shadow_frontier_state = "unreachable"
        else:
            shadow_frontier_state = "absent"
        status = {
            "frontier_candidate_count": len(frontier_candidates),
            "accepted_frontier_count": len(accepted_frontiers),
            "rejected_frontier_count": len(rejected_frontiers),
            "frontier_rejection_reasons": rejection_reasons,
            "unreachable_frontier_count": len(unreachable_frontiers),
            "visible_shadow_frontier_count": len(visible_frontiers),
            "moving_shadow_frontier_count": len(moving_frontiers),
            "best_visible_shadow_count": (
                max(all_visible_counts)
                if all_visible_counts
                else 0
            ),
            "min_visible_frontier_path_m": min(path_lengths) if path_lengths else None,
            "max_visible_frontier_path_m": max(path_lengths) if path_lengths else None,
            "frontier_motion_threshold_m": ACTIVE_EXPLORE_FRONTIER_REACHED_PATH_M,
            "empty_replans_required": (
                ACTIVE_EXPLORE_SHADOW_EMPTY_REPLANS_TO_COMPLETE
            ),
            "shadow_frontier_state": shadow_frontier_state,
            "empty": shadow_frontier_state == "absent",
        }
        return status

    def update_shadow_explore_phase_from_plan(self, plan):
        status = self.shadow_frontier_status_from_plan(plan)
        if self.phase != ACTIVE_EXPLORE_PHASE_SHADOW:
            status["empty_replans"] = self.shadow_frontier_empty_replans
            status["complete"] = self.shadow_explore_complete
            self.diagnostics["active_explore"]["shadow_frontier_status"] = status
            self.update_phase_diagnostics()
            return status

        if status["shadow_frontier_state"] == "reachable":
            self.shadow_frontier_empty_replans = 0
            self.shadow_explore_complete = False
        elif status["shadow_frontier_state"] == "absent":
            self.shadow_frontier_empty_replans += 1
            if (
                self.shadow_frontier_empty_replans
                >= ACTIVE_EXPLORE_SHADOW_EMPTY_REPLANS_TO_COMPLETE
            ):
                self.shadow_explore_complete = True
                self.clear_frontier_goal("shadow_frontier_exhausted")
                self.set_phase(ACTIVE_EXPLORE_PHASE_LOCALIZATION_POSE)
        else:
            self.shadow_frontier_empty_replans = 0
            self.shadow_explore_complete = False

        status["empty_replans"] = self.shadow_frontier_empty_replans
        status["complete"] = self.shadow_explore_complete
        self.diagnostics["active_explore"]["shadow_frontier_status"] = status
        self.update_phase_diagnostics()
        return status

    def moving_shadow_frontier_candidates(self, plan):
        return tuple(
            candidate
            for candidate in plan.candidates
            if candidate_is_moving_shadow_frontier(candidate)
        )

    def best_scored_candidate(self, candidates):
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda candidate: (
                -(candidate.score if candidate.score is not None else -math.inf),
                (
                    candidate.path_length_m
                    if candidate.path_length_m is not None
                    else math.inf
                ),
            ),
        )[0]

    def shadow_approach_goal_reference(self, plan):
        goal = self.frontier_goal or {}
        target = finite_point_2d([goal.get("target_x"), goal.get("target_y")])
        if target is not None:
            return target, "persistent_frontier_target"

        cluster = finite_point_2d(goal.get("cluster_centroid_world"))
        if cluster is not None:
            return cluster, "persistent_frontier_cluster"

        frontier_candidates = [
            candidate
            for candidate in plan.candidates
            if candidate is not None and candidate.kind == "obstacle_shadow_frontier"
        ]
        frontier_candidates.sort(
            key=lambda candidate: (
                -candidate_visible_shadow_count(candidate),
                (
                    0
                    if candidate.rejection_reason
                    in ACTIVE_EXPLORE_FRONTIER_UNREACHABLE_REASONS
                    else 1
                ),
                (
                    candidate.path_length_m
                    if candidate.path_length_m is not None
                    else math.inf
                ),
            )
        )
        for candidate in frontier_candidates:
            target = candidate_target_point(candidate)
            if target is not None:
                return target, "generated_frontier_target"
        for candidate in frontier_candidates:
            cluster = candidate_cluster_centroid(candidate)
            if cluster is not None:
                return cluster, "generated_frontier_cluster"
        return None, None

    def shadow_approach_start_point(self, candidate, current_pose_point=None):
        if current_pose_point is not None:
            return [float(current_pose_point[0]), float(current_pose_point[1])]
        return candidate_path_start_point(candidate)

    def shadow_approach_goal_metrics(self, candidate, goal_point, current_pose_point=None):
        start = self.shadow_approach_start_point(candidate, current_pose_point)
        target = candidate_target_point(candidate)
        if target is None and candidate.path_world:
            target = finite_point_2d(candidate.path_world[-1])
        if target is None and candidate.simplified_path_world:
            target = finite_point_2d(candidate.simplified_path_world[-1])
        before = distance_2d(start, goal_point) if start is not None else None
        after = distance_2d(target, goal_point) if target is not None else None
        progress = (
            before - after
            if before is not None and after is not None
            else None
        )
        return {
            "sector_center_deg": (candidate.metadata or {}).get("sector_center_deg"),
            "target_x": None if target is None else target[0],
            "target_y": None if target is None else target[1],
            "frontier_distance_before_m": before,
            "frontier_distance_after_m": after,
            "frontier_progress_m": progress,
            "path_min_clearance_m": candidate_path_min_clearance_m(candidate),
            "path_unknown_ratio": candidate_path_unknown_ratio(candidate),
            "score": candidate.score,
        }

    def shadow_approach_fallback_candidate(self, plan, current_pose_point=None):
        candidates = [
            candidate
            for candidate in plan.candidates
            if candidate_is_accepted_open_corridor(candidate)
        ]
        safe_candidates = [
            candidate
            for candidate in candidates
            if candidate_has_safe_shadow_approach_clearance(candidate)
        ]
        goal_point, goal_source = self.shadow_approach_goal_reference(plan)
        policy = {
            "candidate_count": len(candidates),
            "safe_candidate_count": len(safe_candidates),
            "min_path_clearance_m": (
                ACTIVE_EXPLORE_SHADOW_APPROACH_MIN_PATH_CLEARANCE_M
            ),
            "goal_drift_tolerance_m": (
                ACTIVE_EXPLORE_SHADOW_APPROACH_GOAL_DRIFT_TOLERANCE_M
            ),
            "goal_reference": (
                None
                if goal_point is None
                else {
                    "source": goal_source,
                    "x": goal_point[0],
                    "y": goal_point[1],
                }
            ),
            "goal_aware_candidate_count": 0,
            "goal_drift_rejected_count": 0,
            "candidate_goal_metrics": [],
            "selected_frontier_distance_before_m": None,
            "selected_frontier_distance_after_m": None,
            "selected_frontier_progress_m": None,
            "selected_kind": None,
            "selected_sector_center_deg": None,
            "reason": "",
        }
        if not candidates:
            policy["reason"] = "no_open_corridor_candidate"
            return None, policy
        if not safe_candidates:
            policy["reason"] = "no_safe_open_corridor_candidate"
            return None, policy
        ranked = []
        for candidate in safe_candidates:
            metrics = (
                self.shadow_approach_goal_metrics(candidate, goal_point, current_pose_point)
                if goal_point is not None
                else {
                    "sector_center_deg": (candidate.metadata or {}).get(
                        "sector_center_deg"
                    ),
                    "target_x": candidate.target_x,
                    "target_y": candidate.target_y,
                    "frontier_distance_before_m": None,
                    "frontier_distance_after_m": None,
                    "frontier_progress_m": None,
                    "path_min_clearance_m": candidate_path_min_clearance_m(candidate),
                    "path_unknown_ratio": candidate_path_unknown_ratio(candidate),
                    "score": candidate.score,
                }
            )
            progress = metrics["frontier_progress_m"]
            accepted_for_goal = (
                goal_point is None
                or progress is None
                or progress >= -ACTIVE_EXPLORE_SHADOW_APPROACH_GOAL_DRIFT_TOLERANCE_M
            )
            metrics["accepted_goal_aware"] = accepted_for_goal
            metrics["rejection_reason"] = "" if accepted_for_goal else "goal_drift"
            policy["candidate_goal_metrics"].append(metrics)
            if not accepted_for_goal:
                policy["goal_drift_rejected_count"] += 1
                continue
            ranked.append((candidate, metrics))

        policy["goal_aware_candidate_count"] = len(ranked)
        if not ranked:
            policy["reason"] = "no_goal_progress_open_corridor_candidate"
            return None, policy

        selected, selected_metrics = sorted(
            ranked,
            key=lambda item: (
                -(
                    item[1]["frontier_progress_m"]
                    if item[1]["frontier_progress_m"] is not None
                    else 0.0
                ),
                -(
                    item[1]["path_min_clearance_m"]
                    if item[1]["path_min_clearance_m"] is not None
                    else -math.inf
                ),
                item[1]["path_unknown_ratio"],
                -(item[0].score if item[0].score is not None else -math.inf),
                (
                    item[0].path_length_m
                    if item[0].path_length_m is not None
                    else math.inf
                ),
            ),
        )[0]
        policy["reason"] = "selected"
        policy["selected_kind"] = selected.kind
        policy["selected_sector_center_deg"] = (selected.metadata or {}).get(
            "sector_center_deg"
        )
        policy["selected_frontier_distance_before_m"] = selected_metrics[
            "frontier_distance_before_m"
        ]
        policy["selected_frontier_distance_after_m"] = selected_metrics[
            "frontier_distance_after_m"
        ]
        policy["selected_frontier_progress_m"] = selected_metrics[
            "frontier_progress_m"
        ]
        return selected, policy

    def localization_pose_candidate(self, plan):
        candidates = [
            candidate
            for candidate in plan.candidates
            if candidate_is_localization_pose_candidate(candidate)
        ]
        policy = {
            "eligible_kinds": list(ACTIVE_EXPLORE_LOCALIZATION_CANDIDATE_KINDS),
            "candidate_count": len(candidates),
            "selected_kind": None,
            "reason": "",
        }
        if not candidates:
            policy["reason"] = "no_localization_pose_candidate"
            return None, policy

        priority = {
            "suspected_heater_approach": 0,
            "provisional_center": 1,
            "lateral_recenter": 1,
        }
        candidates.sort(
            key=lambda candidate: (
                priority.get(candidate.kind, 99),
                -(candidate.score if candidate.score is not None else -math.inf),
                (
                    candidate.path_length_m
                    if candidate.path_length_m is not None
                    else math.inf
                ),
            )
        )
        selected = candidates[0]
        policy["selected_kind"] = selected.kind
        policy["reason"] = "selected"
        return selected, policy

    def frontier_goal_candidate_match(self, goal, candidate):
        match = {
            "matched": False,
            "reason": "",
            "target_distance_m": None,
            "cluster_centroid_distance_m": None,
            "visible_cluster_shadow_count": candidate_visible_shadow_count(candidate),
            "candidate": None if candidate is None else candidate.to_dict(),
        }
        if candidate is None or candidate.kind != "obstacle_shadow_frontier":
            match["reason"] = "not_obstacle_shadow_frontier"
            return match
        if not candidate.accepted:
            match["reason"] = candidate.rejection_reason or "candidate_rejected"
            return match
        if match["visible_cluster_shadow_count"] <= 0:
            match["reason"] = "visible_shadow_zero"
            return match

        goal_target = [goal.get("target_x"), goal.get("target_y")]
        if all(value is not None for value in goal_target):
            match["target_distance_m"] = distance_2d(
                goal_target,
                [candidate.target_x, candidate.target_y],
            )

        goal_cluster = finite_point_2d(goal.get("cluster_centroid_world"))
        candidate_cluster = candidate_cluster_centroid(candidate)
        if goal_cluster is not None and candidate_cluster is not None:
            match["cluster_centroid_distance_m"] = distance_2d(
                goal_cluster,
                candidate_cluster,
            )

        target_ok = (
            match["target_distance_m"] is not None
            and match["target_distance_m"] <= ACTIVE_EXPLORE_FRONTIER_TARGET_MATCH_M
        )
        cluster_ok = (
            match["cluster_centroid_distance_m"] is not None
            and match["cluster_centroid_distance_m"]
            <= ACTIVE_EXPLORE_FRONTIER_CLUSTER_MATCH_M
        )
        if not target_ok and not cluster_ok:
            match["reason"] = "frontier_goal_mismatch"
            return match
        match["matched"] = True
        match["reason"] = "matched"
        return match

    def matching_active_explore_frontier_candidate(self, plan, candidates=None):
        if self.frontier_goal is None:
            return None, None
        matches = []
        candidate_iterable = plan.candidates if candidates is None else candidates
        for candidate in candidate_iterable:
            match = self.frontier_goal_candidate_match(
                self.frontier_goal,
                candidate,
            )
            if match["matched"]:
                target_distance = match["target_distance_m"]
                path_length = candidate.path_length_m
                matches.append(
                    (
                        (
                            target_distance if target_distance is not None else math.inf,
                            -match["visible_cluster_shadow_count"],
                            path_length if path_length is not None else math.inf,
                        ),
                        candidate,
                        match,
                    )
                )
        if not matches:
            return None, None
        matches.sort(key=lambda item: item[0])
        return matches[0][1], matches[0][2]

    def select_with_persistent_frontier(self, plan, attempt_index):
        default_selected = plan.selected
        effective_selected = default_selected
        selection_policy = "score_best"
        persistent_match = None
        abandon_reason = None

        if not plan.ok or default_selected is None:
            if self.frontier_goal is not None:
                abandon_reason = plan.reason or "plan_not_ok"
                self.clear_frontier_goal(abandon_reason)
            return plan, {
                "default_selected": None,
                "effective_selected": None,
                "selection_policy": selection_policy,
                "persistent_frontier_goal": self.frontier_goal_diagnostics(),
                "persistent_frontier_match": None,
                "persistent_frontier_abandon_reason": abandon_reason,
            }

        if self.frontier_goal is not None:
            matched_candidate, persistent_match = (
                self.matching_active_explore_frontier_candidate(plan)
            )
            if matched_candidate is not None:
                effective_selected = matched_candidate
                selection_policy = "persistent_frontier"
            else:
                abandon_reason = "no_matching_accepted_frontier"
                self.clear_frontier_goal(abandon_reason)

        if effective_selected.kind == "obstacle_shadow_frontier":
            visible_count = candidate_visible_shadow_count(effective_selected)
            path_length = effective_selected.path_length_m
            if visible_count <= 0:
                abandon_reason = abandon_reason or "selected_frontier_visible_shadow_zero"
                self.clear_frontier_goal(abandon_reason)
            elif (
                path_length is not None
                and path_length <= ACTIVE_EXPLORE_FRONTIER_REACHED_PATH_M
            ):
                abandon_reason = abandon_reason or "persistent_frontier_goal_reached"
                self.clear_frontier_goal(abandon_reason)
            else:
                self.store_frontier_goal(
                    effective_selected,
                    attempt_index,
                )
        elif self.frontier_goal is not None:
            abandon_reason = abandon_reason or "selected_candidate_not_frontier"
            self.clear_frontier_goal(abandon_reason)

        effective_plan = plan
        if effective_selected is not default_selected:
            effective_plan = ActiveExplorePlan(
                plan.ok,
                plan.reason,
                effective_selected,
                plan.candidates,
                plan.grid,
            )

        return effective_plan, {
            "default_selected": default_selected.to_dict(),
            "effective_selected": effective_selected.to_dict(),
            "selection_policy": selection_policy,
            "persistent_frontier_goal": self.frontier_goal_diagnostics(),
            "persistent_frontier_match": persistent_match,
            "persistent_frontier_abandon_reason": abandon_reason,
        }

    def select_for_phase(self, plan, attempt_index, current_pose_point=None):
        default_selected = plan.selected
        shadow_status = self.update_shadow_explore_phase_from_plan(plan)
        localization_policy = None
        shadow_approach_policy = None
        persistent_match = None
        abandon_reason = None
        continue_without_motion = False

        def diagnostics(effective_selected, selection_policy):
            return {
                "active_explore_phase": self.phase,
                "shadow_frontier_empty_replans": self.shadow_frontier_empty_replans,
                "shadow_explore_complete": self.shadow_explore_complete,
                "shadow_frontier_status": shadow_status,
                "default_selected": (
                    None if default_selected is None else default_selected.to_dict()
                ),
                "effective_selected": (
                    None if effective_selected is None else effective_selected.to_dict()
                ),
                "selection_policy": selection_policy,
                "persistent_frontier_goal": self.frontier_goal_diagnostics(),
                "persistent_frontier_match": persistent_match,
                "persistent_frontier_abandon_reason": abandon_reason,
                "localization_candidate_policy": localization_policy,
                "shadow_approach_fallback_policy": shadow_approach_policy,
                "continue_without_motion": continue_without_motion,
            }

        if not plan.ok:
            if self.frontier_goal is not None:
                abandon_reason = plan.reason or "plan_not_ok"
                self.clear_frontier_goal(abandon_reason)
            return plan, diagnostics(None, "plan_not_ok")

        if self.phase == ACTIVE_EXPLORE_PHASE_SHADOW:
            moving_frontiers = self.moving_shadow_frontier_candidates(plan)
            if not moving_frontiers:
                if shadow_status["shadow_frontier_state"] == "unreachable":
                    fallback, shadow_approach_policy = (
                        self.shadow_approach_fallback_candidate(plan, current_pose_point)
                    )
                    self.diagnostics["active_explore"][
                        "shadow_approach_fallback_policy"
                    ] = shadow_approach_policy
                    if fallback is None:
                        if shadow_approach_policy["candidate_count"] <= 0:
                            reason = "shadow_frontier_unreachable_no_approach_candidate"
                        elif (
                            shadow_approach_policy["reason"]
                            == "no_safe_open_corridor_candidate"
                        ):
                            reason = "shadow_frontier_unreachable_no_safe_approach_candidate"
                        else:
                            reason = "shadow_frontier_unreachable_no_goal_approach_candidate"
                        gated_plan = ActiveExplorePlan(
                            False,
                            reason,
                            None,
                            plan.candidates,
                            plan.grid,
                        )
                        return gated_plan, diagnostics(
                            None,
                            "shadow_approach_fallback",
                        )
                    effective_plan = ActiveExplorePlan(
                        True,
                        plan.reason,
                        fallback,
                        plan.candidates,
                        plan.grid,
                    )
                    return effective_plan, diagnostics(
                        fallback,
                        "shadow_approach_fallback",
                    )
                if self.frontier_goal is not None:
                    abandon_reason = "no_moving_shadow_frontier"
                    self.clear_frontier_goal(abandon_reason)
                continue_without_motion = True
                gated_plan = ActiveExplorePlan(
                    False,
                    "shadow_frontier_empty_replan_wait",
                    None,
                    plan.candidates,
                    plan.grid,
                )
                return gated_plan, diagnostics(None, "shadow_frontier_required")

            effective_selected = None
            selection_policy = "shadow_frontier_best"
            if self.frontier_goal is not None:
                effective_selected, persistent_match = (
                    self.matching_active_explore_frontier_candidate(
                        plan,
                        candidates=moving_frontiers,
                    )
                )
                if effective_selected is not None:
                    selection_policy = "persistent_frontier"
                else:
                    abandon_reason = "no_matching_accepted_frontier"
                    self.clear_frontier_goal(abandon_reason)

            if effective_selected is None:
                effective_selected = self.best_scored_candidate(moving_frontiers)

            if effective_selected is not None:
                self.store_frontier_goal(effective_selected, attempt_index)
                effective_plan = ActiveExplorePlan(
                    True,
                    plan.reason,
                    effective_selected,
                    plan.candidates,
                    plan.grid,
                )
                return effective_plan, diagnostics(effective_selected, selection_policy)

        localization_candidate, localization_policy = (
            self.localization_pose_candidate(plan)
        )
        self.diagnostics["active_explore"]["localization_candidate_policy"] = (
            localization_policy
        )
        if self.frontier_goal is not None:
            abandon_reason = abandon_reason or "shadow_explore_complete"
            self.clear_frontier_goal(abandon_reason)
        if localization_candidate is None:
            no_pose_plan = ActiveExplorePlan(
                False,
                localization_policy["reason"],
                None,
                plan.candidates,
                plan.grid,
            )
            return no_pose_plan, diagnostics(None, "localization_pose_required")

        effective_plan = ActiveExplorePlan(
            True,
            plan.reason,
            localization_candidate,
            plan.candidates,
            plan.grid,
        )
        return effective_plan, diagnostics(
            localization_candidate,
            "localization_pose",
        )
