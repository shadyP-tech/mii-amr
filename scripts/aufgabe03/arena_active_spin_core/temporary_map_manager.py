from __future__ import annotations

from dataclasses import astuple

from arena_active_explore import (
    build_local_grid_from_scan_samples,
    build_observed_local_grid_from_scan_samples,
)

from .diagnostics import active_explore_config_from_arena_config, effective_recovery_mode


class TemporaryMapManager:
    def __init__(self, session):
        self.session = session
        self.samples = []
        self.version = 0
        self.last_publish_sec = None
        self._grid_cache = {}

    def set_samples(self, samples):
        self.samples = list(samples)
        self.invalidate()
        self._update_stored_sample_count()

    def append_sample(self, sample):
        self.samples.append(sample)
        max_samples = max(1, int(self.session.config.active_explore_map_max_samples))
        if len(self.samples) > max_samples:
            del self.samples[: len(self.samples) - max_samples]
        self.invalidate()
        self._update_stored_sample_count()

    def invalidate(self):
        self.version += 1
        self._grid_cache.clear()

    def _update_stored_sample_count(self):
        self.session.diagnostics["active_explore"]["temporary_map"][
            "scan_samples_stored"
        ] = len(self.samples)

    def _active_config(self, active_config=None):
        return active_config or active_explore_config_from_arena_config(
            self.session.config
        )

    def _pose_key(self, robot_pose):
        if robot_pose is None:
            return None
        return (
            float(robot_pose.x),
            float(robot_pose.y),
        )

    def _sample_signature(self, samples):
        samples = tuple(samples)
        return (
            self.version,
            len(samples),
            tuple(id(sample) for sample in samples),
        )

    def _cache_key(self, kind, samples, robot_pose, active_config):
        return (
            kind,
            self._sample_signature(samples),
            self._pose_key(robot_pose),
            astuple(active_config),
        )

    def planning_grid(self, samples=None, robot_pose=None, active_config=None):
        samples = self.samples if samples is None else list(samples)
        robot_pose = self.session.latest_odom_pose if robot_pose is None else robot_pose
        active_config = self._active_config(active_config)
        key = self._cache_key("planning", samples, robot_pose, active_config)
        if key not in self._grid_cache:
            self._grid_cache[key] = build_local_grid_from_scan_samples(
                samples,
                robot_pose,
                active_config,
            )
        return self._grid_cache[key]

    def display_grid(self, samples=None, robot_pose=None, active_config=None):
        samples = self.samples if samples is None else list(samples)
        robot_pose = self.session.latest_odom_pose if robot_pose is None else robot_pose
        active_config = self._active_config(active_config)
        key = self._cache_key("display", samples, robot_pose, active_config)
        if key not in self._grid_cache:
            self._grid_cache[key] = build_observed_local_grid_from_scan_samples(
                samples,
                robot_pose,
                active_config,
            )
        return self._grid_cache[key]

    def update_diagnostics(self, planning_grid, display_grid=None):
        display_counts = None
        if display_grid is not None:
            display_counts = display_grid.to_dict()
        planning_counts = planning_grid.to_dict()
        self.session.diagnostics["active_explore"]["temporary_map"] = {
            "frame": "odom",
            "source": "accumulated_spin_and_recovery_scans",
            "scan_samples_stored": len(self.samples),
            "display_grid": display_counts,
            "planning_grid": planning_counts,
            "grid": planning_counts,
        }

    def publish_if_ready(self, force=False, grid=None, display_grid=None):
        if self.session.temporary_map_callback is None:
            return
        if (
            effective_recovery_mode(self.session.config) != "active_explore"
            or not self.session.config.active_explore_use_accumulated_map
            or not self.samples
            or self.session.latest_odom_pose is None
        ):
            return
        now = self.session.now()
        period_sec = self.session.config.active_explore_temporary_map_publish_period_sec
        if (
            not force
            and self.last_publish_sec is not None
            and now - self.last_publish_sec < period_sec
        ):
            return
        if grid is None:
            grid = self.planning_grid()
        if display_grid is None:
            display_grid = self.display_grid()
        self.update_diagnostics(grid, display_grid=display_grid)
        self.last_publish_sec = now
        try:
            self.session.temporary_map_callback(display_grid, grid)
        except Exception as exc:
            self.session.diagnostics["active_explore"]["temporary_map"][
                "publish_error"
            ] = str(exc)
