from __future__ import annotations

import math

from .diagnostics import effective_recovery_mode
from .models import ACTIVE_EXPLORE_PHASE_LOCALIZATION_SPIN
from .temporary_map import (
    filter_scan_samples_with_temporary_obstacle_map,
    temporary_grid_localizer_obstacle_mask,
    valid_scan_range_count,
)


class ActiveExploreLocalizerFilter:
    def __init__(self, session):
        self.session = session
        self.last_sample_point_limits = None

    def reason_disabled(self):
        session = self.session
        if effective_recovery_mode(session.config) != "active_explore":
            return "not_active_explore"
        if not session.config.active_explore_use_accumulated_map:
            return "accumulated_map_disabled"
        attempt_index = session.diagnostics.get("spin", {}).get("attempt_index")
        if attempt_index == 0:
            return "first_spin"
        if session.active_explore_phase != ACTIVE_EXPLORE_PHASE_LOCALIZATION_SPIN:
            return "not_final_active_explore_localization_spin"
        if not session.shadow_explore_complete:
            return "shadow_explore_not_complete"
        return None

    def memory_samples(self):
        session = self.session
        if session.active_explore_final_spin_memory_samples is not None:
            return list(session.active_explore_final_spin_memory_samples)
        return list(session.explore_samples)

    def dedupe_samples_by_identity(self, sample_groups):
        deduped = []
        seen = set()
        for samples in sample_groups:
            for sample in samples:
                key = id(sample)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(sample)
        return deduped

    def stride_valid_range_count_for_sample(self, sample):
        stride = max(1, int(self.session.config.range_stride))
        count = 0
        for index, value in enumerate(sample.ranges):
            if index % stride != 0:
                continue
            if value is None or not math.isfinite(value):
                continue
            if value < sample.range_min or value > sample.range_max:
                continue
            count += 1
        return count

    def select_samples_for_point_budget(self, samples, point_budget):
        selected, _limits, _diagnostics = self.select_samples_and_point_limits(
            samples,
            point_budget,
        )
        return selected

    def select_samples_and_point_limits(self, samples, point_budget):
        samples = list(samples)
        if not samples or point_budget <= 0:
            return [], {}, {"selected_point_limit_total": 0}
        point_counts = [
            self.stride_valid_range_count_for_sample(sample)
            for sample in samples
        ]
        valid_indices = [
            index
            for index, point_count in enumerate(point_counts)
            if point_count > 0
        ]
        if not valid_indices:
            return [], {}, {"selected_point_limit_total": 0}
        total_points = sum(point_counts[index] for index in valid_indices)
        if total_points <= point_budget:
            limits = {
                id(samples[index]): point_counts[index]
                for index in valid_indices
            }
            return (
                [samples[index] for index in valid_indices],
                limits,
                {"selected_point_limit_total": sum(limits.values())},
            )

        target_count = max(1, int(point_budget))
        target_count = min(target_count, len(valid_indices))
        if target_count == 1:
            selected_indices = [valid_indices[len(valid_indices) // 2]]
        else:
            selected_indices = []
            max_position = len(valid_indices) - 1
            for selection_index in range(target_count):
                position = round(selection_index * max_position / (target_count - 1))
                selected_indices.append(valid_indices[position])
            selected_indices = sorted(set(selected_indices))

        selected = []
        for index in selected_indices:
            selected.append(samples[index])
        if not selected:
            selected.append(samples[selected_indices[0]])

        index_by_id = {
            id(samples[index]): index
            for index in valid_indices
        }
        limits = {id(sample): 1 for sample in selected}
        remaining = max(0, int(point_budget) - len(selected))
        while remaining > 0:
            changed = False
            for sample in selected:
                key = id(sample)
                index = index_by_id[key]
                if limits[key] >= point_counts[index]:
                    continue
                limits[key] += 1
                remaining -= 1
                changed = True
                if remaining <= 0:
                    break
            if not changed:
                break
        return (
            selected,
            limits,
            {"selected_point_limit_total": sum(limits.values())},
        )

    def pose_bin_for_mapping_sample(self, sample):
        pose = sample.odom_pose
        if pose is None:
            return None
        try:
            x = float(pose.x)
            y = float(pose.y)
            yaw_deg = float(pose.yaw_deg)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw_deg)):
            return None
        yaw_wrapped_deg = ((yaw_deg + 180.0) % 360.0) - 180.0
        return (
            math.floor(x / 0.15),
            math.floor(y / 0.15),
            math.floor(yaw_wrapped_deg / 20.0),
        )

    def mapping_memory_candidates(self, memory_samples, excluded_ids):
        candidates = []
        pose_bins = set()
        for sample in memory_samples:
            if id(sample) in excluded_ids:
                continue
            pose_bin = self.pose_bin_for_mapping_sample(sample)
            if pose_bin is None or pose_bin in pose_bins:
                continue
            pose_bins.add(pose_bin)
            candidates.append(sample)
        return candidates, len(pose_bins)

    def point_budgets(self, final_samples, startup_samples):
        max_points = max(1, int(self.session.config.max_points or 1))
        final_budget = int(round(max_points * 0.40))
        startup_budget = int(round(max_points * 0.40))
        mapping_budget = max(0, max_points - final_budget - startup_budget)
        if not startup_samples:
            final_budget += startup_budget
            startup_budget = 0
        if not final_samples:
            startup_budget += final_budget
            final_budget = 0
        if not final_samples and not startup_samples:
            mapping_budget = max_points
        return final_budget, startup_budget, mapping_budget

    def balanced_samples(self, memory_samples):
        session = self.session
        memory_samples = list(memory_samples)
        final_samples = list(session.samples)
        startup_samples = list(session.active_explore_startup_spin_samples)
        raw_combined_samples = self.dedupe_samples_by_identity(
            (memory_samples, final_samples)
        )
        final_budget, startup_budget, mapping_budget = self.point_budgets(
            final_samples,
            startup_samples,
        )

        selected_final, final_limits, final_limit_diagnostics = self.select_samples_and_point_limits(
            final_samples,
            final_budget,
        )
        selected_startup, startup_limits, startup_limit_diagnostics = self.select_samples_and_point_limits(
            startup_samples,
            startup_budget,
        )
        excluded_ids = {id(sample) for sample in final_samples}
        excluded_ids.update(id(sample) for sample in startup_samples)
        mapping_candidates, pose_bin_count = self.mapping_memory_candidates(
            memory_samples,
            excluded_ids,
        )
        selected_mapping, mapping_limits, mapping_limit_diagnostics = self.select_samples_and_point_limits(
            mapping_candidates,
            mapping_budget,
        )

        balanced_samples = []
        seen = set()
        sample_point_limits = {}
        selected_counts = {
            "final_spin": 0,
            "startup_spin": 0,
            "mapping_memory": 0,
        }
        for group_name, samples, limits in (
            ("final_spin", selected_final, final_limits),
            ("startup_spin", selected_startup, startup_limits),
            ("mapping_memory", selected_mapping, mapping_limits),
        ):
            for sample in samples:
                key = id(sample)
                if key in seen:
                    continue
                seen.add(key)
                balanced_samples.append(sample)
                if key in limits:
                    sample_point_limits[key] = limits[key]
                selected_counts[group_name] += 1

        self.last_sample_point_limits = sample_point_limits
        diagnostics = {
            "raw_combined_sample_count": len(raw_combined_samples),
            "startup_spin_sample_count": len(startup_samples),
            "selected_final_spin_sample_count": selected_counts["final_spin"],
            "selected_startup_spin_sample_count": selected_counts["startup_spin"],
            "selected_mapping_memory_sample_count": selected_counts["mapping_memory"],
            "balanced_sample_count": len(balanced_samples),
            "localizer_sample_order": [
                group_name
                for group_name in (
                    "final_spin",
                    "startup_spin",
                    "mapping_memory",
                )
                if selected_counts[group_name] > 0
            ],
            "final_spin_point_budget": final_budget,
            "startup_spin_point_budget": startup_budget,
            "mapping_memory_point_budget": mapping_budget,
            "mapping_memory_candidate_count": len(mapping_candidates),
            "mapping_memory_pose_bin_count": pose_bin_count,
            "final_spin_selected_point_limit": final_limit_diagnostics["selected_point_limit_total"],
            "startup_spin_selected_point_limit": startup_limit_diagnostics["selected_point_limit_total"],
            "mapping_memory_selected_point_limit": mapping_limit_diagnostics["selected_point_limit_total"],
            "sample_point_limit_count": len(sample_point_limits),
            "sample_point_limit_total": sum(sample_point_limits.values()),
        }
        return balanced_samples, diagnostics

    def remap_point_limits_by_position(self, original_samples, filtered_samples):
        limits = getattr(self, "last_sample_point_limits", None)
        if not limits:
            return None
        remapped = {}
        for original, filtered in zip(original_samples, filtered_samples):
            limit = limits.get(id(original))
            if limit is not None:
                remapped[id(filtered)] = limit
        return remapped

    def filtered_samples(self):
        session = self.session
        diagnostics = {
            "enabled": False,
            "reason": "",
            "input_sample_count": len(session.samples),
            "output_sample_count": len(session.samples),
            "memory_sample_count": 0,
            "final_spin_sample_count": len(session.samples),
            "startup_spin_sample_count": len(
                session.active_explore_startup_spin_samples
            ),
            "raw_combined_sample_count": len(session.samples),
            "combined_sample_count": len(session.samples),
            "selected_final_spin_sample_count": len(session.samples),
            "selected_startup_spin_sample_count": 0,
            "selected_mapping_memory_sample_count": 0,
            "balanced_sample_count": len(session.samples),
            "localizer_sample_order": ["raw_current_spin"] if session.samples else [],
            "final_spin_point_budget": 0,
            "startup_spin_point_budget": 0,
            "mapping_memory_point_budget": 0,
            "mapping_memory_candidate_count": 0,
            "mapping_memory_pose_bin_count": 0,
            "final_spin_selected_point_limit": 0,
            "startup_spin_selected_point_limit": 0,
            "mapping_memory_selected_point_limit": 0,
            "sample_point_limit_count": 0,
            "sample_point_limit_total": 0,
            "used_accumulated_memory": False,
            "valid_ranges_before": valid_scan_range_count(session.samples),
            "valid_ranges_after": valid_scan_range_count(session.samples),
            "filtered_range_count": 0,
            "obstacle_mask_cell_count": 0,
            "protected_wall_cell_count": 0,
            "temporary_grid_cell_counts": None,
            "final_spin_attempt_index": session.diagnostics.get("spin", {}).get(
                "attempt_index"
            ),
        }
        disabled_reason = session.active_explore_localizer_filter_reason_disabled()
        if disabled_reason is not None:
            diagnostics["reason"] = disabled_reason
            self.last_sample_point_limits = None
            session.diagnostics["active_explore"]["localizer_filter"] = diagnostics
            return session.samples

        memory_samples = session.active_explore_localizer_memory_samples()
        localizer_samples, balance_diagnostics = (
            session.balanced_active_explore_localizer_samples(memory_samples)
        )
        valid_ranges_before = valid_scan_range_count(localizer_samples)
        diagnostics["memory_sample_count"] = len(memory_samples)
        diagnostics.update(balance_diagnostics)
        diagnostics["combined_sample_count"] = len(localizer_samples)
        diagnostics["input_sample_count"] = len(localizer_samples)
        diagnostics["output_sample_count"] = len(localizer_samples)
        diagnostics["used_accumulated_memory"] = (
            balance_diagnostics["selected_startup_spin_sample_count"] > 0
            or balance_diagnostics["selected_mapping_memory_sample_count"] > 0
        )
        diagnostics["valid_ranges_before"] = valid_ranges_before
        diagnostics["valid_ranges_after"] = valid_ranges_before

        grid, grid_reason = session.active_explore_localizer_filter_grid(
            memory_samples
        )
        if grid is None:
            diagnostics["reason"] = grid_reason
            session.diagnostics["active_explore"]["localizer_filter"] = diagnostics
            return localizer_samples

        diagnostics["temporary_grid_cell_counts"] = grid.to_dict()["cell_counts"]
        obstacle_mask, protected_wall_cells, mask_diagnostics = (
            temporary_grid_localizer_obstacle_mask(grid)
        )
        diagnostics.update(mask_diagnostics)
        diagnostics["obstacle_mask_cell_count"] = len(obstacle_mask)
        diagnostics["protected_wall_cell_count"] = len(protected_wall_cells)
        if not obstacle_mask:
            diagnostics["reason"] = "no_temporary_obstacle_mask"
            session.diagnostics["active_explore"]["localizer_filter"] = diagnostics
            return localizer_samples

        filtered_samples, filtered_range_count = (
            filter_scan_samples_with_temporary_obstacle_map(
                localizer_samples,
                grid,
                obstacle_mask,
            )
        )
        filtered_valid_ranges_after = valid_scan_range_count(filtered_samples)
        if valid_ranges_before > 0 and filtered_valid_ranges_after <= 0:
            diagnostics["reason"] = "obstacle_filter_removed_all_ranges"
            diagnostics["filtered_range_count"] = filtered_range_count
            diagnostics["filtered_valid_ranges_after"] = filtered_valid_ranges_after
            session.diagnostics["active_explore"]["localizer_filter"] = diagnostics
            return localizer_samples

        self.last_sample_point_limits = self.remap_point_limits_by_position(
            localizer_samples,
            filtered_samples,
        )
        diagnostics["enabled"] = True
        diagnostics["reason"] = "filtered_temporary_obstacles"
        diagnostics["output_sample_count"] = len(filtered_samples)
        diagnostics["filtered_range_count"] = filtered_range_count
        diagnostics["valid_ranges_after"] = filtered_valid_ranges_after
        session.diagnostics["active_explore"]["localizer_filter"] = diagnostics
        return filtered_samples
