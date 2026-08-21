"""Validate one stopped coverage observer's LiDAR visibility evidence.

The observer summary is only an index.  Admission therefore reopens the
referenced JSONL, verifies both byte-level and canonical receipt-set hashes,
and binds every receipt to the frozen survey/viewpoint/config identities.  The
module is ROS-free and has no registry or motion side effects.
"""

from __future__ import annotations

import json
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import (
    canonical_json_bytes,
    payload_sha256,
)
from scripts.aufgabe04.navigation.stand_coverage_survey import (
    CoverageSurveyPlan,
    validate_coverage_survey_plan,
)
from scripts.aufgabe04.perception.lidar_visibility_evidence import (
    VISIBILITY_EVIDENCE_ENABLED_KEY,
    VISIBILITY_OBSERVER_CONFIG_KEY,
    VISIBILITY_OBSERVER_CONFIG_SHA256_KEY,
    VISIBILITY_RECEIPT_COUNT_KEY,
    VISIBILITY_RECEIPTS_FILE_SHA256_KEY,
    VISIBILITY_RECEIPTS_JSONL_KEY,
    VISIBILITY_RECEIPT_SET_SHA256_KEY,
    LidarVisibilityReceipt,
    load_lidar_visibility_receipt_snapshot,
    visibility_receipts_sha256,
)
from scripts.aufgabe04.perception.lidar_visibility_session import (
    FROZEN_ODOM_OBSERVATION_GEOMETRY,
    LIDAR_VISIBILITY_OBSERVER_CONFIG_SCHEMA_VERSION,
    LIVE_MAP_OBSERVATION_GEOMETRY,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CoverageVisibilityEvidence:
    """Validated, content-bound visibility evidence for one survey viewpoint."""

    survey_id: str
    viewpoint_id: str
    planning_frame: str
    map_bundle_sha256: str
    receipts_jsonl: Path
    receipt_count: int
    receipts_file_sha256: str
    receipt_set_sha256: str
    observer_config: dict[str, object]
    observer_config_sha256: str
    receipts: tuple[LidarVisibilityReceipt, ...]


def validate_coverage_visibility_evidence(
    summary: Mapping[str, object],
    plan: CoverageSurveyPlan,
    viewpoint_id: str,
    required: bool,
) -> CoverageVisibilityEvidence | None:
    """Validate a summary-backed receipt snapshot or reject it fail closed.

    ``None`` has exactly one meaning: visibility evidence was explicitly
    disabled and the caller declared it optional.  Missing flags and malformed
    enabled evidence never degrade to that optional state.
    """

    if not isinstance(summary, Mapping):
        raise ValueError("observer summary must be a mapping")
    if not isinstance(plan, CoverageSurveyPlan):
        raise ValueError("plan must be a CoverageSurveyPlan")
    validate_coverage_survey_plan(plan)
    if type(required) is not bool:
        raise ValueError("required must be a boolean")
    if not isinstance(viewpoint_id, str) or not viewpoint_id:
        raise ValueError("viewpoint_id must be a non-empty string")
    if plan.viewpoint_for(viewpoint_id) is None:
        raise ValueError("viewpoint_id is not present in the coverage survey plan")

    enabled = summary.get(VISIBILITY_EVIDENCE_ENABLED_KEY)
    if type(enabled) is not bool:
        raise ValueError(
            f"{VISIBILITY_EVIDENCE_ENABLED_KEY} must be a boolean"
        )
    if not enabled:
        if required:
            raise ValueError("required LiDAR visibility evidence is disabled")
        return None

    summary_planning_frame = _required_string(summary, "planning_frame")
    if summary_planning_frame != plan.planning_frame:
        raise ValueError("observer summary planning frame differs from survey plan")
    summary_map_sha256 = _required_sha256(summary, "map_bundle_sha256")
    if summary_map_sha256 != plan.map_bundle_sha256:
        raise ValueError("observer summary map bundle differs from survey plan")

    processed_scan_count = _required_positive_integer(
        summary,
        "processed_scan_count",
    )
    receipt_count = _required_positive_integer(
        summary,
        VISIBILITY_RECEIPT_COUNT_KEY,
    )
    if receipt_count != processed_scan_count:
        raise ValueError(
            "LiDAR visibility receipt count differs from processed_scan_count"
        )

    path_text = _required_string(summary, VISIBILITY_RECEIPTS_JSONL_KEY)
    receipts_path = Path(path_text)
    _validate_regular_jsonl_path(receipts_path)
    receipts, actual_file_sha256 = load_lidar_visibility_receipt_snapshot(
        receipts_path
    )
    if len(receipts) != receipt_count:
        raise ValueError(
            "persisted LiDAR visibility receipt count differs from observer summary"
        )

    expected_file_sha256 = _required_sha256(
        summary,
        VISIBILITY_RECEIPTS_FILE_SHA256_KEY,
    )
    if actual_file_sha256 != expected_file_sha256:
        raise ValueError("LiDAR visibility receipt file SHA-256 mismatch")

    expected_set_sha256 = _required_sha256(
        summary,
        VISIBILITY_RECEIPT_SET_SHA256_KEY,
    )
    actual_set_sha256 = visibility_receipts_sha256(receipts)
    if actual_set_sha256 != expected_set_sha256:
        raise ValueError("LiDAR visibility canonical receipt-set SHA-256 mismatch")

    raw_observer_config = summary.get(VISIBILITY_OBSERVER_CONFIG_KEY)
    if not isinstance(raw_observer_config, Mapping):
        raise ValueError(f"{VISIBILITY_OBSERVER_CONFIG_KEY} must be a mapping")
    observer_config = _json_safe_mapping(raw_observer_config)
    observer_config_sha256 = _required_sha256(
        summary,
        VISIBILITY_OBSERVER_CONFIG_SHA256_KEY,
    )
    if payload_sha256(observer_config) != observer_config_sha256:
        raise ValueError("LiDAR visibility observer config SHA-256 mismatch")
    _validate_observer_config_binding(
        observer_config,
        summary=summary,
        plan=plan,
    )
    expected_scan_topic = _required_string(
        _required_mapping(observer_config, "runtime_config"),
        "scan_topic",
    )

    for receipt in receipts:
        if (
            receipt.survey_id != plan.survey_id
            or receipt.viewpoint_id != viewpoint_id
            or receipt.planning_frame != plan.planning_frame
            or receipt.map_bundle_sha256 != plan.map_bundle_sha256
            or receipt.observer_config_sha256 != observer_config_sha256
            or receipt.scan_topic != expected_scan_topic
        ):
            raise ValueError(
                "LiDAR visibility receipt identity differs from "
                "survey/viewpoint/planning-frame/map/config"
            )

    return CoverageVisibilityEvidence(
        survey_id=plan.survey_id,
        viewpoint_id=viewpoint_id,
        planning_frame=plan.planning_frame,
        map_bundle_sha256=plan.map_bundle_sha256,
        receipts_jsonl=receipts_path,
        receipt_count=receipt_count,
        receipts_file_sha256=actual_file_sha256,
        receipt_set_sha256=actual_set_sha256,
        observer_config=observer_config,
        observer_config_sha256=observer_config_sha256,
        receipts=receipts,
    )


def _validate_observer_config_binding(
    observer_config: Mapping[str, object],
    *,
    summary: Mapping[str, object],
    plan: CoverageSurveyPlan,
) -> None:
    """Cross-bind receipt configuration to the stopped-observer summary."""

    if (
        observer_config.get("schema_version")
        != LIDAR_VISIBILITY_OBSERVER_CONFIG_SCHEMA_VERSION
    ):
        raise ValueError("unsupported LiDAR visibility observer config schema")
    if observer_config.get("map_bundle_sha256") != plan.map_bundle_sha256:
        raise ValueError(
            "LiDAR visibility observer config map bundle differs from survey plan"
        )
    geometry_mode = observer_config.get("observation_geometry_mode")
    if geometry_mode not in {
        LIVE_MAP_OBSERVATION_GEOMETRY,
        FROZEN_ODOM_OBSERVATION_GEOMETRY,
    }:
        raise ValueError(
            "LiDAR visibility observer config geometry mode is invalid"
        )
    summary_runtime = _json_safe_mapping(
        _required_mapping(summary, "runtime_config")
    )
    summary_timing = _json_safe_mapping(
        _required_mapping(summary, "timing_limits")
    )
    if observer_config.get("runtime_config") != summary_runtime:
        raise ValueError(
            "LiDAR visibility observer runtime config differs from summary"
        )
    if observer_config.get("timing_limits") != summary_timing:
        raise ValueError(
            "LiDAR visibility observer timing limits differ from summary"
        )


def coverage_visibility_epoch_fields(
    evidence: CoverageVisibilityEvidence | None,
) -> dict[str, object]:
    """Return compact JSON-safe fields for a stopped coverage-epoch artifact."""

    if evidence is None:
        return {
            VISIBILITY_EVIDENCE_ENABLED_KEY: False,
            VISIBILITY_RECEIPTS_JSONL_KEY: None,
            VISIBILITY_RECEIPT_COUNT_KEY: 0,
            VISIBILITY_RECEIPTS_FILE_SHA256_KEY: None,
            VISIBILITY_RECEIPT_SET_SHA256_KEY: None,
            VISIBILITY_OBSERVER_CONFIG_KEY: None,
            VISIBILITY_OBSERVER_CONFIG_SHA256_KEY: None,
        }
    if not isinstance(evidence, CoverageVisibilityEvidence):
        raise ValueError("evidence must be CoverageVisibilityEvidence or None")
    fields = {
        VISIBILITY_EVIDENCE_ENABLED_KEY: True,
        VISIBILITY_RECEIPTS_JSONL_KEY: str(evidence.receipts_jsonl),
        VISIBILITY_RECEIPT_COUNT_KEY: evidence.receipt_count,
        VISIBILITY_RECEIPTS_FILE_SHA256_KEY: evidence.receipts_file_sha256,
        VISIBILITY_RECEIPT_SET_SHA256_KEY: evidence.receipt_set_sha256,
        VISIBILITY_OBSERVER_CONFIG_KEY: _json_safe_mapping(
            evidence.observer_config
        ),
        VISIBILITY_OBSERVER_CONFIG_SHA256_KEY: (
            evidence.observer_config_sha256
        ),
        "lidar_visibility_survey_id": evidence.survey_id,
        "lidar_visibility_viewpoint_id": evidence.viewpoint_id,
        "lidar_visibility_planning_frame": evidence.planning_frame,
        "lidar_visibility_map_bundle_sha256": evidence.map_bundle_sha256,
    }
    # Assert the public helper never leaks Path, tuple, NaN, or another
    # implementation-only value into an epoch JSON payload.
    canonical_json_bytes(fields)
    return fields


def _validate_regular_jsonl_path(path: Path) -> None:
    if path.suffix != ".jsonl":
        raise ValueError("LiDAR visibility receipt path must end in .jsonl")
    if path.is_symlink():
        raise ValueError("LiDAR visibility receipt path must not be a symlink")
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise ValueError("LiDAR visibility receipt JSONL is unavailable") from exc
    if not stat.S_ISREG(mode):
        raise ValueError("LiDAR visibility receipt path must be a regular file")


def _required_string(summary: Mapping[str, object], key: str) -> str:
    value = summary.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_mapping(
    summary: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = summary.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _required_sha256(summary: Mapping[str, object], key: str) -> str:
    value = summary.get(key)
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{key} must be a lowercase SHA-256")
    return value


def _required_positive_integer(summary: Mapping[str, object], key: str) -> int:
    value = summary.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _json_safe_mapping(value: Mapping[str, object]) -> dict[str, object]:
    try:
        decoded = json.loads(canonical_json_bytes(dict(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "LiDAR visibility observer config must be finite JSON"
        ) from exc
    if not isinstance(decoded, dict):  # Defensive: the input is already a mapping.
        raise ValueError("LiDAR visibility observer config must be a JSON object")
    return decoded


__all__ = [
    "CoverageVisibilityEvidence",
    "coverage_visibility_epoch_fields",
    "validate_coverage_visibility_evidence",
]
