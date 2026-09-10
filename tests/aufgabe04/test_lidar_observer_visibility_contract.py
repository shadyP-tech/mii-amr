"""Production observer artifacts must survive the stopped-epoch importer.

These tests use the real static-runtime factory, visibility session, receipt
writer, observer-summary writer, and importer. Only the transport-free node
state is assembled here; no producer or importer function is mocked and no
matching runtime dictionaries are hand-built in the fixtures.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.coverage.coverage_stop_perception_admission import (
    _validate_current_visibility_evidence,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    SURVEY_PLAN_SCHEMA_VERSION,
    CoverageSurveyConfig,
    CoverageSurveyPlan,
    SurveyViewpoint,
)
from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.foundation.models import GridCell, Pose2D
from scripts.aufgabe04.navigation.foundation.ros_runtime_config import (
    RuntimeConfig,
    resolve_runtime_config,
)
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    OdomExecutionCertificate,
    PlanarTransform2D,
    odom_pose_to_map,
    write_odom_execution_certificate,
)
from scripts.aufgabe04.perception import stand_explorer_node as observer
from scripts.aufgabe04.perception.lidar_observer_runtime import LidarObserverRuntime
from scripts.aufgabe04.perception.lidar_stand_morphology import (
    MORPHOLOGY_PROFILE_EVIDENCE_KEY,
    PROPOSAL_DETECTOR_CONFIG_EVIDENCE_KEY,
    stand_width_profile_from_radius,
)
from scripts.aufgabe04.perception.lidar_visibility_evidence import (
    VISIBILITY_EVIDENCE_ENABLED_KEY,
    VISIBILITY_OBSERVER_CONFIG_KEY,
    VISIBILITY_OBSERVER_CONFIG_SHA256_KEY,
    VISIBILITY_RECEIPT_COUNT_KEY,
    lidar_visibility_receipt_from_scan,
)
from scripts.aufgabe04.perception.lidar_visibility_frames import (
    LidarVisibilityFrameProvenance,
)
from scripts.aufgabe04.perception.lidar_visibility_session import (
    FROZEN_ODOM_OBSERVATION_GEOMETRY,
    LIVE_MAP_OBSERVATION_GEOMETRY,
    proposal_detector_config_evidence,
)
from scripts.aufgabe04.perception.models import LidarStandDetectorConfig


MAP_SHA256 = "a" * 64
SURVEY_ID = "survey_contract"
VIEWPOINT_ID = "viewpoint_01"


def _plan() -> CoverageSurveyPlan:
    cell = GridCell(0, 0)
    second_cell = GridCell(1, 0)
    cells = tuple(sorted((cell, second_cell)))
    return CoverageSurveyPlan(
        schema_version=SURVEY_PLAN_SCHEMA_VERSION,
        survey_id=SURVEY_ID,
        planning_frame="map",
        map_bundle_sha256=MAP_SHA256,
        arena_bounds=ArenaBounds(),
        config=CoverageSurveyConfig(
            lane_count=1, exact_inspection_point_count=2,
            exact_two_candidate_spacing_m=0.45,
            minimum_exact_two_viewpoint_baseline_m=0.75,
        ),
        viewpoints=(
            SurveyViewpoint(VIEWPOINT_ID, Pose2D(0.0, 0.0, 0.0), cell, cells),
            SurveyViewpoint("viewpoint_02", Pose2D(1.0, 0.0, 0.0), second_cell, cells),
        ),
        surveyable_cells=cells,
        planned_covered_cells=cells,
        planned_coverage_ratio=1.0,
    )


def _frozen_frame(root: Path):
    certificate = OdomExecutionCertificate(
        source_map_route_sha256="b" * 64,
        source_map_execution_certificate_sha256="c" * 64,
        transformed_odom_route_sha256="d" * 64,
        map_frame="map",
        odom_frame="odom",
        base_frame="base_footprint",
        map_from_odom=PlanarTransform2D(0.4, -0.2, math.pi / 5.0),
        transform_stamp_sec=9.0,
        transform_capture_time_sec=9.1,
        waypoint_count=2,
        tracking_tube_radius_m=0.03,
        command_owner="/aufgabe04_simple_waypoint_follower",
        uncertainty_budget_sha256="e" * 64,
        ambiguity_evidence_sha256="f" * 64,
    )
    path = root / "odom_execution_certificate.json"
    write_odom_execution_certificate(path, certificate)
    return observer.load_frozen_observer_frame(
        path, map_frame="map", odom_frame="odom", base_frame="base_footprint"
    )


def _observer_state(root: Path, *, profile: str, frozen: bool):
    runtime = resolve_runtime_config(
        RuntimeConfig(namespace="robot_1", ros_domain_id="47")
    )
    observer_runtime = LidarObserverRuntime(runtime, profile)
    timing_limits = observer.DEFAULT_OBSERVATION_TIMING_LIMITS
    detector_config = LidarStandDetectorConfig()
    morphology_profile = stand_width_profile_from_radius(0.06)
    frozen_frame = _frozen_frame(root) if frozen else None
    session = observer_runtime.create_visibility_session(
        output_path=root / "visibility_receipts.jsonl",
        survey_id=SURVEY_ID,
        viewpoint_id=VIEWPOINT_ID,
        timing_limits=timing_limits.as_dict(),
        map_bundle_sha256=MAP_SHA256,
        observation_geometry_mode=(
            FROZEN_ODOM_OBSERVATION_GEOMETRY
            if frozen else LIVE_MAP_OBSERVATION_GEOMETRY
        ),
        proposal_detector_config=proposal_detector_config_evidence(detector_config),
        morphology_profile=morphology_profile.to_evidence_dict(),
    )
    return SimpleNamespace(
        args=SimpleNamespace(scan_topology_profile=profile),
        runtime=runtime,
        observer_runtime=observer_runtime,
        timing_limits=timing_limits,
        detector_config=detector_config,
        morphology_profile=morphology_profile,
        visibility_session=session,
        frozen_observer_frame=frozen_frame,
        started_unix_sec=10.0,
        output_jsonl=root / "observations.jsonl",
        map_bundle=SimpleNamespace(bundle_sha256=MAP_SHA256),
        last_scan_pose_map=None,
        last_processed_scan_stamp_sec=None,
        last_scan_topology=None,
        processed_scan_count=0,
        detected_candidate_count=0,
        accepted_observation_count=0,
        last_confirmed_stand_count=0,
    )


def _buffer_scan(node, index: int) -> None:
    scan_pose = Pose2D(0.15 * index, -0.1, 0.05 * index)
    frame = node.frozen_observer_frame
    provenance = None
    if frame is not None:
        provenance = LidarVisibilityFrameProvenance(
            map_frame=node.runtime.map_frame,
            odom_frame=node.runtime.odom_frame,
            map_from_odom=frame.certificate.map_from_odom,
            canonical_scan_pose_odom=scan_pose,
            source_evidence_id=frame.certificate_sha256,
        )
        scan_pose = odom_pose_to_map(scan_pose, frame.certificate.map_from_odom)
    stamp = 10.0 + index * 0.1
    receipt = lidar_visibility_receipt_from_scan(
        receipt_id=f"{VIEWPOINT_ID}_{index:06d}",
        survey_id=SURVEY_ID,
        viewpoint_id=VIEWPOINT_ID,
        planning_frame=node.runtime.map_frame,
        scan_frame="base_scan",
        scan_topic=node.runtime.scan_topic,
        map_bundle_sha256=MAP_SHA256,
        observer_config_sha256=node.visibility_session.observer_config_sha256,
        scan_stamp_sec=stamp,
        pose_stamp_sec=stamp,
        observer_clock_sec=stamp + 0.01,
        scan_pose_map=scan_pose,
        angle_min_rad=-math.pi,
        angle_increment_rad=math.pi / 2.0,
        range_min_m=0.08,
        range_max_m=3.5,
        ranges_m=(1.0, math.inf, 2.0, 0.8),
        frame_provenance=provenance,
    )
    node.visibility_session.buffer_receipt(receipt)
    node.processed_scan_count += 1
    node.last_processed_scan_stamp_sec = stamp
    node.last_scan_pose_map = {
        "x_m": scan_pose.x_m,
        "y_m": scan_pose.y_m,
        "yaw_rad": scan_pose.yaw_rad,
    }


def _write_summary(root: Path, node, *, name="observer_summary.json"):
    path = root / name
    observer.write_observer_summary(path, node)
    return path, json.loads(path.read_text())


def _import_summary(path: Path, payload):
    plan = _plan()
    return _validate_current_visibility_evidence(
        observer_summary=payload,
        observer_summary_json=path,
        plan=plan,
        viewpoint=plan.viewpoint_for(VIEWPOINT_ID),
    )


class LidarObserverVisibilityContractTest(unittest.TestCase):
    def test_real_producer_artifacts_are_accepted_for_every_profile_and_frame(self):
        for profile in ("linear", "full_rotation"):
            for frozen in (False, True):
                with self.subTest(profile=profile, frozen=frozen):
                    with tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        node = _observer_state(root, profile=profile, frozen=frozen)
                        _buffer_scan(node, 1)
                        _buffer_scan(node, 2)
                        path, summary = _write_summary(root, node)
                        evidence = _import_summary(path, summary)

                        self.assertEqual(len(evidence.receipts), 2)
                        self.assertEqual(summary[VISIBILITY_RECEIPT_COUNT_KEY], 2)
                        self.assertEqual(summary["scan_topology_profile"], profile)
                        self.assertEqual(
                            summary["runtime_config"],
                            evidence.observer_config["runtime_config"],
                        )
                        self.assertEqual(
                            summary["runtime_config"]["scan_topology_profile"], profile
                        )
                        self.assertEqual(
                            summary["runtime_config"]["scan_topic"], "/robot_1/scan"
                        )
                        self.assertEqual(
                            evidence.observer_config[MORPHOLOGY_PROFILE_EVIDENCE_KEY],
                            summary[MORPHOLOGY_PROFILE_EVIDENCE_KEY],
                        )
                        self.assertEqual(
                            evidence.observer_config[PROPOSAL_DETECTOR_CONFIG_EVIDENCE_KEY],
                            summary[PROPOSAL_DETECTOR_CONFIG_EVIDENCE_KEY],
                        )
                        self.assertEqual(
                            evidence.receipts[0].frame_provenance is not None, frozen
                        )

    def test_original_missing_nested_profile_is_rejected_at_epoch_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            node = _observer_state(root, profile="full_rotation", frozen=True)
            _buffer_scan(node, 1)
            path, summary = _write_summary(root, node)
            # The failed real pilot had the right top-level field, but omitted
            # the profile from this independent runtime configuration binding.
            del summary["runtime_config"]["scan_topology_profile"]
            self.assertEqual(summary["scan_topology_profile"], "full_rotation")
            with self.assertRaisesRegex(ValueError, "runtime config differs"):
                _import_summary(path, summary)

    def test_rehashed_configuration_substitution_does_not_bypass_binding(self):
        for mutation in ("summary_profile", "summary_topic", "observer_profile", "both_profiles", "timing", "disabled"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                node = _observer_state(root, profile="full_rotation", frozen=True)
                _buffer_scan(node, 1)
                path, summary = _write_summary(root, node)
                config = summary[VISIBILITY_OBSERVER_CONFIG_KEY]
                if mutation == "summary_profile":
                    summary["runtime_config"]["scan_topology_profile"] = "linear"
                elif mutation == "summary_topic":
                    summary["runtime_config"]["scan_topic"] = "/another_robot/scan"
                elif mutation == "observer_profile":
                    config["runtime_config"]["scan_topology_profile"] = "linear"
                elif mutation == "both_profiles":
                    config["runtime_config"]["scan_topology_profile"] = "linear"
                    summary["runtime_config"]["scan_topology_profile"] = "linear"
                elif mutation == "timing":
                    config["timing_limits"]["max_scan_age_sec"] = 9.0
                else:
                    summary[VISIBILITY_EVIDENCE_ENABLED_KEY] = False
                summary[VISIBILITY_OBSERVER_CONFIG_SHA256_KEY] = payload_sha256(config)
                with self.assertRaisesRegex(ValueError, "runtime config differs|receipt identity differs|timing limits differ|required LiDAR visibility evidence is disabled"):
                    _import_summary(path, summary)

    def test_scan_diagnostics_and_later_arguments_cannot_change_captured_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            node = _observer_state(root, profile="full_rotation", frozen=True)
            config_sha = node.visibility_session.observer_config_sha256
            detached = node.observer_runtime.as_log_dict()
            detached["scan_topology_profile"] = "linear"
            detached["configured"]["scan_topic"] = "another_scan"
            _buffer_scan(node, 1)
            node.last_scan_topology = {"periodic": False, "reason": "partial_scan"}
            _buffer_scan(node, 2)
            first_path, first = _write_summary(root, node)

            node.args.scan_topology_profile = "linear"
            node.last_scan_topology = {"periodic": True, "wrapped_cluster_count": 1}
            node.detected_candidate_count = 3
            second_path, second = _write_summary(root, node, name="later_summary.json")

            self.assertNotEqual(first["last_scan_topology"], second["last_scan_topology"])
            self.assertEqual(first["runtime_config"], second["runtime_config"])
            self.assertEqual(second["scan_topology_profile"], "full_rotation")
            self.assertEqual(node.visibility_session.observer_config_sha256, config_sha)
            self.assertEqual(first[VISIBILITY_OBSERVER_CONFIG_SHA256_KEY], config_sha)
            self.assertEqual(second[VISIBILITY_OBSERVER_CONFIG_SHA256_KEY], config_sha)
            self.assertNotIn("last_scan_topology", second["runtime_config"])
            self.assertNotIn("lidar_wrapped_clusters", second["runtime_config"])
            for path, summary in ((first_path, first), (second_path, second)):
                self.assertEqual(len(_import_summary(path, summary).receipts), 2)

    def test_receipt_file_and_frozen_frame_tampering_remain_rejected(self):
        for mutation in ("file_bytes", "certificate", "odom_frame", "transform"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                node = _observer_state(root, profile="full_rotation", frozen=True)
                _buffer_scan(node, 1)
                path, summary = _write_summary(root, node)
                if mutation == "file_bytes":
                    receipts = node.visibility_session.output_path
                    receipts.write_bytes(receipts.read_bytes() + b"\n")
                else:
                    geometry = summary[observer.FROZEN_FRAME_EVIDENCE_KEY]
                    if mutation == "certificate":
                        geometry["odom_execution_certificate_sha256"] = "0" * 64
                    elif mutation == "odom_frame":
                        geometry["source_frames"]["odom_frame"] = "foreign_odom"
                    else:
                        geometry["map_from_odom"]["x_m"] += 0.1
                with self.assertRaisesRegex(ValueError, "file SHA-256 mismatch|frozen frame identity differs|frozen transform differs"):
                    _import_summary(path, summary)

    def test_highest_importer_requires_receipts_and_summary_in_same_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            node = _observer_state(root, profile="linear", frozen=False)
            _buffer_scan(node, 1)
            path, summary = _write_summary(root, node)
            foreign_path = root / "different_epoch" / path.name
            foreign_path.parent.mkdir()
            foreign_path.write_text(path.read_text())
            with self.assertRaisesRegex(ValueError, "share the observer-summary epoch"):
                _import_summary(foreign_path, summary)


if __name__ == "__main__":
    unittest.main()
