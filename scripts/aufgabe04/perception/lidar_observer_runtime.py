"""Immutable runtime evidence shared by LiDAR detection, receipts and summaries.

This module has no ROS dependency. Per-scan geometry and cluster diagnostics
belong in observations, never in the static visibility configuration hash.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from scripts.aufgabe04.navigation.foundation.ros_runtime_config import ResolvedRuntimeConfig
from scripts.aufgabe04.perception.lidar_visibility_session import LidarVisibilitySession
from scripts.aufgabe04.perception.scan_topology import SCAN_TOPOLOGY_PROFILES


@dataclass(frozen=True)
class LidarObserverRuntime:
    """Capture the resolved ROS settings and selected topology once per node.

    Both ResolvedRuntimeConfig and its configured RuntimeConfig are frozen
    dataclasses. Their log serialization returns detached nested dictionaries,
    so callers can add observation diagnostics without mutating this evidence.
    """

    runtime: ResolvedRuntimeConfig
    scan_topology_profile: str = "linear"

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, ResolvedRuntimeConfig):
            raise TypeError("LiDAR observer runtime requires resolved immutable ROS settings")
        if self.scan_topology_profile not in SCAN_TOPOLOGY_PROFILES:
            raise ValueError("LiDAR observer scan topology profile must be linear or full_rotation")

    def as_log_dict(self) -> dict[str, object]:
        return {
            **self.runtime.as_log_dict(),
            "scan_topology_profile": self.scan_topology_profile,
        }

    def create_visibility_session(
        self, *, output_path: Path | None, survey_id: str = "", viewpoint_id: str = "",
        timing_limits: Mapping[str, object] | None = None,
        map_bundle_sha256: str | None = None,
        observation_geometry_mode: str | None = None,
        proposal_detector_config: Mapping[str, object] | None = None,
        morphology_profile: Mapping[str, object] | None = None,
    ) -> LidarVisibilitySession:
        """Bind the receipt session to the same runtime used by the summary."""

        return LidarVisibilitySession.create(
            output_path=output_path,
            survey_id=survey_id,
            viewpoint_id=viewpoint_id,
            runtime_config=self.as_log_dict(),
            timing_limits=timing_limits,
            map_bundle_sha256=map_bundle_sha256,
            observation_geometry_mode=observation_geometry_mode,
            proposal_detector_config=proposal_detector_config,
            morphology_profile=morphology_profile,
        )
