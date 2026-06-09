from __future__ import annotations


class ReplanManager:
    """Compatibility facade around the existing runtime replan methods."""

    def __init__(self, runtime):
        self.runtime = runtime

    def initialize_route(self, current_pose, waypoints):
        return self.runtime.initialize_run_local_route(current_pose, waypoints)

    def replan_after_blockage(self, current_pose, old_remaining_waypoints, trigger):
        return self.runtime.replan_after_blockage(
            current_pose,
            old_remaining_waypoints,
            trigger=trigger,
        )

    def prune_after_progress(self, current_pose, remaining_waypoints):
        prune = getattr(
            self.runtime,
            "prune_run_local_obstacles_after_progress",
            None,
        )
        if prune is None:
            return None
        return prune(current_pose, remaining_waypoints)

    def corridor_blocked_cells(self, current_pose, remaining_waypoints):
        return self.runtime.corridor_blocked_cells(
            current_pose,
            remaining_waypoints,
        )
