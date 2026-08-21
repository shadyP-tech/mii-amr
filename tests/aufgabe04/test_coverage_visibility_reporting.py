from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.coverage_visibility_reporting import (
    CoverageVisibilityEvidence,
    coverage_visibility_epoch_fields,
    validate_coverage_visibility_evidence,
)
from scripts.aufgabe04.navigation.models import GridCell, Pose2D
from scripts.aufgabe04.navigation.stand_coverage_survey import (
    SURVEY_PLAN_SCHEMA_VERSION,
    CoverageSurveyConfig,
    CoverageSurveyPlan,
    SurveyViewpoint,
)
from scripts.aufgabe04.perception.lidar_visibility_evidence import (
    VISIBILITY_EVIDENCE_ENABLED_KEY,
    VISIBILITY_OBSERVER_CONFIG_KEY,
    VISIBILITY_OBSERVER_CONFIG_SHA256_KEY,
    VISIBILITY_RECEIPT_COUNT_KEY,
    VISIBILITY_RECEIPTS_FILE_SHA256_KEY,
    VISIBILITY_RECEIPTS_JSONL_KEY,
    VISIBILITY_RECEIPT_SET_SHA256_KEY,
    append_lidar_visibility_receipts,
    lidar_visibility_receipt_from_scan,
    load_lidar_visibility_receipt_snapshot,
    visibility_receipts_sha256,
)


MAP_SHA256 = "a" * 64
SURVEY_ID = "survey_01"
VIEWPOINT_ID = "viewpoint_01"


def _plan() -> CoverageSurveyPlan:
    cell = GridCell(0, 0)
    return CoverageSurveyPlan(
        schema_version=SURVEY_PLAN_SCHEMA_VERSION,
        survey_id=SURVEY_ID,
        planning_frame="map",
        map_bundle_sha256=MAP_SHA256,
        arena_bounds=ArenaBounds(),
        config=CoverageSurveyConfig(),
        viewpoints=(
            SurveyViewpoint(
                viewpoint_id=VIEWPOINT_ID,
                pose=Pose2D(0.0, 0.0, 0.0),
                cell=cell,
                visible_cells=(cell,),
            ),
        ),
        surveyable_cells=(cell,),
        planned_covered_cells=(cell,),
        planned_coverage_ratio=1.0,
    )


def _observer_config(
    *,
    runtime_config: dict[str, object] | None = None,
    timing_limits: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "map_bundle_sha256": MAP_SHA256,
        "runtime_config": runtime_config
        or {"map_frame": "map", "scan_topic": "/scan"},
        "timing_limits": timing_limits or {"max_scan_age_sec": 1.0},
        "observation_geometry_mode": (
            "frozen_map_from_odom_plus_exact_odom_from_scan"
        ),
    }


def _receipt(*, observer_config_sha256: str, **overrides):
    values = {
        "receipt_id": "receipt_000001",
        "survey_id": SURVEY_ID,
        "viewpoint_id": VIEWPOINT_ID,
        "planning_frame": "map",
        "scan_frame": "base_scan",
        "scan_topic": "/scan",
        "map_bundle_sha256": MAP_SHA256,
        "observer_config_sha256": observer_config_sha256,
        "scan_stamp_sec": 10.0,
        "pose_stamp_sec": 10.0,
        "observer_clock_sec": 10.01,
        "scan_pose_map": Pose2D(0.0, 0.0, 0.0),
        "angle_min_rad": -1.0,
        "angle_increment_rad": 1.0,
        "range_min_m": 0.08,
        "range_max_m": 3.5,
        "ranges_m": (1.0, None, 2.0),
    }
    values.update(overrides)
    return lidar_visibility_receipt_from_scan(**values)


def _summary(
    path: Path,
    *,
    receipt_overrides=None,
    observer_config=None,
    summary_runtime_config=None,
    summary_timing_limits=None,
):
    observer_config = observer_config or _observer_config()
    config_sha256 = payload_sha256(observer_config)
    receipt = _receipt(
        observer_config_sha256=config_sha256,
        **(receipt_overrides or {}),
    )
    append_lidar_visibility_receipts(path, (receipt,))
    receipts, file_sha256 = load_lidar_visibility_receipt_snapshot(path)
    return {
        VISIBILITY_EVIDENCE_ENABLED_KEY: True,
        VISIBILITY_RECEIPTS_JSONL_KEY: str(path),
        VISIBILITY_RECEIPT_COUNT_KEY: 1,
        VISIBILITY_RECEIPTS_FILE_SHA256_KEY: file_sha256,
        VISIBILITY_RECEIPT_SET_SHA256_KEY: visibility_receipts_sha256(receipts),
        VISIBILITY_OBSERVER_CONFIG_KEY: observer_config,
        VISIBILITY_OBSERVER_CONFIG_SHA256_KEY: config_sha256,
        "processed_scan_count": 1,
        "planning_frame": "map",
        "map_bundle_sha256": MAP_SHA256,
        "runtime_config": summary_runtime_config
        or dict(observer_config["runtime_config"]),
        "timing_limits": summary_timing_limits
        or dict(observer_config["timing_limits"]),
    }


class CoverageVisibilityReportingTest(unittest.TestCase):
    def test_valid_summary_returns_frozen_evidence_and_json_safe_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visibility_receipts.jsonl"
            evidence = validate_coverage_visibility_evidence(
                _summary(path),
                _plan(),
                VIEWPOINT_ID,
                True,
            )

            self.assertIsInstance(evidence, CoverageVisibilityEvidence)
            self.assertEqual(evidence.receipt_count, 1)
            self.assertEqual(evidence.receipts[0].viewpoint_id, VIEWPOINT_ID)
            fields = coverage_visibility_epoch_fields(evidence)
            self.assertTrue(fields[VISIBILITY_EVIDENCE_ENABLED_KEY])
            self.assertEqual(fields["lidar_visibility_survey_id"], SURVEY_ID)
            json.dumps(fields, allow_nan=False)
            with self.assertRaises(FrozenInstanceError):
                evidence.receipt_count = 2

    def test_explicitly_disabled_optional_evidence_is_the_only_none_case(self):
        evidence = validate_coverage_visibility_evidence(
            {VISIBILITY_EVIDENCE_ENABLED_KEY: False},
            _plan(),
            VIEWPOINT_ID,
            False,
        )

        self.assertIsNone(evidence)
        fields = coverage_visibility_epoch_fields(evidence)
        self.assertFalse(fields[VISIBILITY_EVIDENCE_ENABLED_KEY])
        json.dumps(fields, allow_nan=False)

    def test_required_disabled_or_missing_flag_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "required.*disabled"):
            validate_coverage_visibility_evidence(
                {VISIBILITY_EVIDENCE_ENABLED_KEY: False},
                _plan(),
                VIEWPOINT_ID,
                True,
            )
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            validate_coverage_visibility_evidence(
                {},
                _plan(),
                VIEWPOINT_ID,
                False,
            )

    def test_tampered_line_raw_hash_set_hash_and_counts_fail_closed(self):
        mutations = {
            "raw_file_hash": lambda summary: summary.__setitem__(
                VISIBILITY_RECEIPTS_FILE_SHA256_KEY, "c" * 64
            ),
            "receipt_set_hash": lambda summary: summary.__setitem__(
                VISIBILITY_RECEIPT_SET_SHA256_KEY, "d" * 64
            ),
            "receipt_count": lambda summary: summary.__setitem__(
                VISIBILITY_RECEIPT_COUNT_KEY, 2
            ),
            "processed_count": lambda summary: summary.__setitem__(
                "processed_scan_count", 2
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "visibility_receipts.jsonl"
                summary = _summary(path)
                mutate(summary)
                with self.assertRaises(ValueError):
                    validate_coverage_visibility_evidence(
                        summary,
                        _plan(),
                        VIEWPOINT_ID,
                        True,
                    )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visibility_receipts.jsonl"
            summary = _summary(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["ranges_m"][0] = 3.0
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_coverage_visibility_evidence(
                    summary,
                    _plan(),
                    VIEWPOINT_ID,
                    True,
                )

    def test_substituted_receipt_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visibility_receipts.jsonl"
            summary = _summary(path, receipt_overrides={"viewpoint_id": "viewpoint_02"})
            with self.assertRaisesRegex(ValueError, "receipt identity differs"):
                validate_coverage_visibility_evidence(
                    summary,
                    _plan(),
                    VIEWPOINT_ID,
                    True,
                )

    def test_rehashed_runtime_or_timing_config_substitution_is_rejected(self):
        cases = (
            (
                "runtime config differs",
                _observer_config(
                    runtime_config={
                        "map_frame": "map",
                        "scan_topic": "/substituted_scan",
                    }
                ),
                {"scan_topic": "/substituted_scan"},
                {"map_frame": "map", "scan_topic": "/scan"},
                None,
            ),
            (
                "timing limits differ",
                _observer_config(timing_limits={"max_scan_age_sec": 9.0}),
                None,
                None,
                {"max_scan_age_sec": 1.0},
            ),
        )
        for (
            expected_error,
            observer_config,
            receipt_overrides,
            summary_runtime,
            summary_timing,
        ) in cases:
            with self.subTest(expected_error=expected_error):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "visibility_receipts.jsonl"
                    summary = _summary(
                        path,
                        observer_config=observer_config,
                        receipt_overrides=receipt_overrides,
                        summary_runtime_config=summary_runtime,
                        summary_timing_limits=summary_timing,
                    )

                    with self.assertRaisesRegex(ValueError, expected_error):
                        validate_coverage_visibility_evidence(
                            summary,
                            _plan(),
                            VIEWPOINT_ID,
                            True,
                        )

    def test_unhashed_observer_config_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visibility_receipts.jsonl"
            summary = _summary(path)
            summary[VISIBILITY_OBSERVER_CONFIG_KEY]["schema_version"] = 2
            with self.assertRaisesRegex(ValueError, "config SHA-256 mismatch"):
                validate_coverage_visibility_evidence(
                    summary,
                    _plan(),
                    VIEWPOINT_ID,
                    True,
                )

    def test_symlink_receipt_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.jsonl"
            summary = _summary(target)
            link = root / "substituted.jsonl"
            link.symlink_to(target)
            summary[VISIBILITY_RECEIPTS_JSONL_KEY] = str(link)

            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                validate_coverage_visibility_evidence(
                    summary,
                    _plan(),
                    VIEWPOINT_ID,
                    True,
                )


if __name__ == "__main__":
    unittest.main()
