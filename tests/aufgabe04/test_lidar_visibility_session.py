import ast
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.artifacts.content_store import payload_sha256  # noqa: E402
from scripts.aufgabe04.navigation.foundation.models import Pose2D  # noqa: E402
from scripts.aufgabe04.perception.lidar_stand_morphology import (  # noqa: E402
    MORPHOLOGY_PROFILE_EVIDENCE_KEY,
    PROPOSAL_DETECTOR_CONFIG_EVIDENCE_KEY,
    stand_width_profile_from_radius,
)
from scripts.aufgabe04.perception.lidar_visibility_evidence import (  # noqa: E402
    VISIBILITY_EVIDENCE_ENABLED_KEY,
    VISIBILITY_OBSERVER_CONFIG_KEY,
    VISIBILITY_OBSERVER_CONFIG_SHA256_KEY,
    VISIBILITY_RECEIPT_COUNT_KEY,
    VISIBILITY_RECEIPTS_FILE_SHA256_KEY,
    VISIBILITY_RECEIPT_SET_SHA256_KEY,
    lidar_visibility_receipt_from_scan,
)
from scripts.aufgabe04.perception.lidar_visibility_session import (  # noqa: E402
    FROZEN_ODOM_OBSERVATION_GEOMETRY,
    LidarVisibilitySession,
    disabled_visibility_summary_fields,
    proposal_detector_config_evidence,
)
from scripts.aufgabe04.perception.models import (  # noqa: E402
    LidarStandDetectorConfig,
)
import scripts.aufgabe04.perception.lidar_visibility_session as session_module  # noqa: E402,E501


MAP_SHA256 = "a" * 64
SURVEY_ID = "survey_01"
VIEWPOINT_ID = "viewpoint_01"


def _create_session(path: Path) -> LidarVisibilitySession:
    profile = stand_width_profile_from_radius(0.06)
    return LidarVisibilitySession.create(
        output_path=path,
        survey_id=SURVEY_ID,
        viewpoint_id=VIEWPOINT_ID,
        runtime_config={"map_frame": "map", "scan_topic": "/scan"},
        timing_limits={"max_scan_age_sec": 1.0},
        map_bundle_sha256=MAP_SHA256,
        observation_geometry_mode=FROZEN_ODOM_OBSERVATION_GEOMETRY,
        proposal_detector_config=proposal_detector_config_evidence(
            LidarStandDetectorConfig()
        ),
        morphology_profile=profile.to_evidence_dict(),
    )


def _receipt(
    session: LidarVisibilitySession,
    *,
    receipt_id: str = "viewpoint_01_000001",
    survey_id: str = SURVEY_ID,
    viewpoint_id: str = VIEWPOINT_ID,
):
    return lidar_visibility_receipt_from_scan(
        receipt_id=receipt_id,
        survey_id=survey_id,
        viewpoint_id=viewpoint_id,
        planning_frame="map",
        scan_frame="base_scan",
        scan_topic="/scan",
        map_bundle_sha256=MAP_SHA256,
        observer_config_sha256=session.observer_config_sha256,
        scan_stamp_sec=1.0,
        pose_stamp_sec=1.0,
        observer_clock_sec=1.01,
        scan_pose_map=Pose2D(0.0, 0.0, 0.0),
        angle_min_rad=-1.0,
        angle_increment_rad=1.0,
        range_min_m=0.08,
        range_max_m=3.5,
        ranges_m=(1.0, math.inf, 2.0),
    )


class LidarVisibilitySessionTest(unittest.TestCase):
    def test_session_module_is_ros_free_and_node_imports_only_public_api(self):
        session_path = (
            ROOT
            / "scripts"
            / "aufgabe04"
            / "perception"
            / "lidar_visibility_session.py"
        )
        node_path = (
            ROOT
            / "scripts"
            / "aufgabe04"
            / "perception"
            / "stand_explorer_node.py"
        )
        session_tree = ast.parse(
            session_path.read_text(),
            filename=str(session_path),
        )
        ros_imports = []
        for node in ast.walk(session_tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            ros_imports.extend(
                module
                for module in modules
                if module.startswith(
                    ("rclpy", "sensor_msgs", "geometry_msgs", "tf2_ros")
                )
            )

        node_tree = ast.parse(node_path.read_text(), filename=str(node_path))
        private_imports = []
        for node in ast.walk(node_tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if (
                node.module
                != "scripts.aufgabe04.perception.lidar_visibility_session"
            ):
                continue
            private_imports.extend(
                alias.name for alias in node.names if alias.name.startswith("_")
            )

        self.assertEqual(ros_imports, [])
        self.assertEqual(private_imports, [])

    def test_disabled_contract_is_explicit_and_rejects_orphan_identity(self):
        session = LidarVisibilitySession.create(output_path=None)

        self.assertFalse(session.enabled)
        self.assertEqual(
            session.finalize(processed_scan_count=9),
            disabled_visibility_summary_fields(),
        )
        with self.assertRaisesRegex(ValueError, "IDs require"):
            LidarVisibilitySession.create(
                output_path=None,
                survey_id=SURVEY_ID,
            )

    def test_enabled_contract_fails_before_reserving_incomplete_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visibility.jsonl"

            with self.assertRaisesRegex(ValueError, "morphology profile"):
                LidarVisibilitySession.create(
                    output_path=path,
                    survey_id=SURVEY_ID,
                    viewpoint_id=VIEWPOINT_ID,
                    runtime_config={},
                    timing_limits={},
                    map_bundle_sha256=MAP_SHA256,
                    observation_geometry_mode=(
                        FROZEN_ODOM_OBSERVATION_GEOMETRY
                    ),
                    proposal_detector_config={},
                    morphology_profile=None,
                )

            self.assertFalse(path.exists())

    def test_observer_config_is_detached_and_binds_broad_and_morphology_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            session = _create_session(Path(directory) / "visibility.jsonl")

            first = session.observer_config
            first["runtime_config"]["map_frame"] = "mutated"
            second = session.observer_config

            self.assertEqual(second["runtime_config"]["map_frame"], "map")
            self.assertIn(PROPOSAL_DETECTOR_CONFIG_EVIDENCE_KEY, second)
            self.assertIn(MORPHOLOGY_PROFILE_EVIDENCE_KEY, second)
            self.assertEqual(
                payload_sha256(second),
                session.observer_config_sha256,
            )

    def test_buffers_in_memory_and_publishes_one_verified_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visibility.jsonl"
            session = _create_session(path)
            session.buffer_receipt(_receipt(session))
            real_append = session_module.append_lidar_visibility_receipts

            with patch.object(
                session_module,
                "append_lidar_visibility_receipts",
                wraps=real_append,
            ) as append_mock:
                first = session.finalize(processed_scan_count=1)
                second = session.finalize(processed_scan_count=1)

            self.assertEqual(append_mock.call_count, 1)
            self.assertEqual(first, second)
            self.assertTrue(first[VISIBILITY_EVIDENCE_ENABLED_KEY])
            self.assertEqual(first[VISIBILITY_RECEIPT_COUNT_KEY], 1)
            self.assertEqual(
                len(first[VISIBILITY_RECEIPTS_FILE_SHA256_KEY]),
                64,
            )
            self.assertEqual(
                len(first[VISIBILITY_RECEIPT_SET_SHA256_KEY]),
                64,
            )
            self.assertEqual(
                payload_sha256(first[VISIBILITY_OBSERVER_CONFIG_KEY]),
                first[VISIBILITY_OBSERVER_CONFIG_SHA256_KEY],
            )
            self.assertEqual(len(path.read_text().splitlines()), 1)

    def test_scan_count_mismatch_fails_before_batch_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visibility.jsonl"
            session = _create_session(path)
            session.buffer_receipt(_receipt(session))

            with self.assertRaisesRegex(ValueError, "not every processed scan"):
                session.finalize(processed_scan_count=2)

            self.assertEqual(path.read_bytes(), b"")
            self.assertFalse(session.finalized)

    def test_receipt_identity_and_duplicate_ids_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            session = _create_session(Path(directory) / "visibility.jsonl")
            with self.assertRaisesRegex(ValueError, "identity differs"):
                session.buffer_receipt(
                    _receipt(session, viewpoint_id="viewpoint_02")
                )

            valid = _receipt(session)
            session.buffer_receipt(valid)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                session.buffer_receipt(valid)

    def test_summary_requires_clean_finalization(self):
        with tempfile.TemporaryDirectory() as directory:
            session = _create_session(Path(directory) / "visibility.jsonl")

            with self.assertRaisesRegex(ValueError, "must be finalized"):
                session.summary_fields(processed_scan_count=0)


if __name__ == "__main__":
    unittest.main()
