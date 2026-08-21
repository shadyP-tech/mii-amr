"""ROS-free lifecycle for one stopped LiDAR visibility-observer epoch.

The session owns the optional output contract, immutable observer configuration,
in-memory receipt buffering, and one-shot evidence publication.  ROS adapters
remain responsible only for converting sensor messages into validated
``LidarVisibilityReceipt`` objects.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping

from scripts.aufgabe04.artifacts.content_store import (
    canonical_json_bytes,
    payload_sha256,
)
from scripts.aufgabe04.perception.lidar_stand_morphology import (
    MORPHOLOGY_PROFILE_EVIDENCE_KEY,
    PROPOSAL_DETECTOR_CONFIG_EVIDENCE_KEY,
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
    append_lidar_visibility_receipts,
    load_lidar_visibility_receipt_snapshot,
    validate_lidar_visibility_receipt,
    visibility_receipts_sha256,
)
from scripts.aufgabe04.perception.models import LidarStandDetectorConfig


LIDAR_VISIBILITY_OBSERVER_CONFIG_SCHEMA_VERSION = 1
LIVE_MAP_OBSERVATION_GEOMETRY = "live_map_from_scan"
FROZEN_ODOM_OBSERVATION_GEOMETRY = (
    "frozen_map_from_odom_plus_exact_odom_from_scan"
)
_OBSERVATION_GEOMETRY_MODES = frozenset(
    {
        LIVE_MAP_OBSERVATION_GEOMETRY,
        FROZEN_ODOM_OBSERVATION_GEOMETRY,
    }
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def proposal_detector_config_evidence(
    config: LidarStandDetectorConfig,
) -> dict[str, object]:
    """Serialize the broad proposal extractor without implying admission."""

    if not isinstance(config, LidarStandDetectorConfig):
        raise ValueError("detector config must be a LidarStandDetectorConfig")
    return {
        "role": "broad_proposal_extraction_preserves_track_width_history",
        "min_range_m": config.min_range_m,
        "max_range_m": config.max_range_m,
        "max_cluster_gap_m": config.max_cluster_gap_m,
        "min_cluster_points": config.min_cluster_points,
        "min_width_m": config.min_width_m,
        "max_width_m": config.max_width_m,
    }


def disabled_visibility_summary_fields() -> dict[str, object]:
    """Return the explicit evidence shape for an observer without receipts."""

    return {
        VISIBILITY_EVIDENCE_ENABLED_KEY: False,
        VISIBILITY_RECEIPTS_JSONL_KEY: None,
        VISIBILITY_RECEIPT_COUNT_KEY: 0,
        VISIBILITY_RECEIPTS_FILE_SHA256_KEY: None,
        VISIBILITY_RECEIPT_SET_SHA256_KEY: None,
        VISIBILITY_OBSERVER_CONFIG_KEY: None,
        VISIBILITY_OBSERVER_CONFIG_SHA256_KEY: None,
    }


class LidarVisibilitySession:
    """Buffer and publish one immutable set of exact-time scan receipts.

    Use :meth:`create` rather than constructing this class directly.  A session
    with no output path is disabled and still produces explicit summary fields.
    Enabled sessions append and fsync their complete batch exactly once when
    :meth:`finalize` is called during clean observer shutdown.
    """

    def __init__(
        self,
        *,
        output_path: Path | None,
        survey_id: str,
        viewpoint_id: str,
        map_bundle_sha256: str | None,
        observer_config_json: bytes | None,
        observer_config_sha256: str | None,
    ) -> None:
        self._output_path = output_path
        self._survey_id = survey_id
        self._viewpoint_id = viewpoint_id
        self._map_bundle_sha256 = map_bundle_sha256
        self._observer_config_json = observer_config_json
        self._observer_config_sha256 = observer_config_sha256
        self._receipts: list[LidarVisibilityReceipt] = []
        self._receipt_ids: set[str] = set()
        self._finalized_summary: dict[str, object] | None = None

    @classmethod
    def create(
        cls,
        *,
        output_path: Path | None,
        survey_id: str = "",
        viewpoint_id: str = "",
        runtime_config: Mapping[str, object] | None = None,
        timing_limits: Mapping[str, object] | None = None,
        map_bundle_sha256: str | None = None,
        observation_geometry_mode: str | None = None,
        proposal_detector_config: Mapping[str, object] | None = None,
        morphology_profile: Mapping[str, object] | None = None,
    ) -> "LidarVisibilitySession":
        """Validate the optional output contract and reserve a fresh artifact."""

        normalized_survey_id = str(survey_id).strip()
        normalized_viewpoint_id = str(viewpoint_id).strip()
        identity_supplied = bool(normalized_survey_id or normalized_viewpoint_id)
        if output_path is None:
            if identity_supplied:
                raise ValueError(
                    "visibility survey/viewpoint IDs require a receipt output path"
                )
            return cls(
                output_path=None,
                survey_id="",
                viewpoint_id="",
                map_bundle_sha256=None,
                observer_config_json=None,
                observer_config_sha256=None,
            )

        _require_safe_id(normalized_survey_id, "visibility survey ID")
        _require_safe_id(normalized_viewpoint_id, "visibility viewpoint ID")
        if not isinstance(map_bundle_sha256, str) or _SHA256.fullmatch(
            map_bundle_sha256
        ) is None:
            raise ValueError(
                "visibility receipt output requires a frozen map bundle hash"
            )
        if observation_geometry_mode not in _OBSERVATION_GEOMETRY_MODES:
            raise ValueError("visibility observation geometry mode is invalid")
        if not isinstance(runtime_config, Mapping):
            raise ValueError("visibility receipt output requires runtime config")
        if not isinstance(timing_limits, Mapping):
            raise ValueError("visibility receipt output requires timing limits")
        if not isinstance(proposal_detector_config, Mapping):
            raise ValueError(
                "visibility receipt output requires proposal detector evidence"
            )
        if not isinstance(morphology_profile, Mapping):
            raise ValueError(
                "visibility receipt output requires a bound morphology profile"
            )

        observer_config = {
            "schema_version": LIDAR_VISIBILITY_OBSERVER_CONFIG_SCHEMA_VERSION,
            "runtime_config": dict(runtime_config),
            "timing_limits": dict(timing_limits),
            "map_bundle_sha256": map_bundle_sha256,
            "observation_geometry_mode": observation_geometry_mode,
            PROPOSAL_DETECTOR_CONFIG_EVIDENCE_KEY: dict(
                proposal_detector_config
            ),
            MORPHOLOGY_PROFILE_EVIDENCE_KEY: dict(morphology_profile),
        }
        observer_config_json = canonical_json_bytes(observer_config)
        observer_config_sha256 = payload_sha256(observer_config)

        target = Path(output_path)
        if target.is_symlink() or target.exists():
            raise ValueError(
                f"refusing existing visibility receipt output: {target}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch(exist_ok=False)
        return cls(
            output_path=target,
            survey_id=normalized_survey_id,
            viewpoint_id=normalized_viewpoint_id,
            map_bundle_sha256=map_bundle_sha256,
            observer_config_json=observer_config_json,
            observer_config_sha256=observer_config_sha256,
        )

    @property
    def enabled(self) -> bool:
        return self._output_path is not None

    @property
    def output_path(self) -> Path | None:
        return self._output_path

    @property
    def survey_id(self) -> str:
        return self._survey_id

    @property
    def viewpoint_id(self) -> str:
        return self._viewpoint_id

    @property
    def observer_config(self) -> dict[str, object] | None:
        """Return a detached copy so callers cannot mutate bound evidence."""

        if self._observer_config_json is None:
            return None
        value = json.loads(self._observer_config_json)
        if not isinstance(value, dict):  # pragma: no cover - construction invariant.
            raise ValueError("visibility observer config is not an object")
        return value

    @property
    def observer_config_sha256(self) -> str | None:
        return self._observer_config_sha256

    @property
    def receipt_count(self) -> int:
        return len(self._receipts)

    @property
    def finalized(self) -> bool:
        return self._finalized_summary is not None

    def buffer_receipt(self, receipt: LidarVisibilityReceipt) -> None:
        """Validate and retain one receipt without performing disk I/O."""

        if not self.enabled:
            raise ValueError("cannot buffer a receipt in a disabled session")
        if self.finalized:
            raise ValueError("cannot buffer a receipt after visibility finalization")
        validate_lidar_visibility_receipt(receipt)
        if (
            receipt.survey_id != self._survey_id
            or receipt.viewpoint_id != self._viewpoint_id
        ):
            raise ValueError("visibility receipt identity differs from session")
        if receipt.map_bundle_sha256 != self._map_bundle_sha256:
            raise ValueError("visibility receipt map bundle differs from session")
        if receipt.observer_config_sha256 != self._observer_config_sha256:
            raise ValueError("visibility receipt observer config differs from session")
        if receipt.receipt_id in self._receipt_ids:
            raise ValueError("duplicate visibility receipt_id in session")
        self._receipts.append(receipt)
        self._receipt_ids.add(receipt.receipt_id)

    def finalize(self, *, processed_scan_count: int) -> dict[str, object]:
        """Publish one batch and return verified, content-hashed summary fields."""

        count = _require_nonnegative_integer(
            processed_scan_count,
            "processed_scan_count",
        )
        if not self.enabled:
            if count < 0:  # pragma: no cover - validated above.
                raise ValueError("processed_scan_count must be non-negative")
            return disabled_visibility_summary_fields()
        if self._finalized_summary is not None:
            if count != len(self._receipts):
                raise ValueError(
                    "processed scan count differs from finalized visibility receipts"
                )
            return _detached_json_object(self._finalized_summary)
        if len(self._receipts) != count:
            raise ValueError("not every processed scan has a visibility receipt")

        output_path = self._output_path
        if output_path is None:  # pragma: no cover - enabled invariant.
            raise ValueError("enabled visibility session has no output path")
        append_lidar_visibility_receipts(output_path, tuple(self._receipts))
        receipts, file_sha256 = load_lidar_visibility_receipt_snapshot(
            output_path
        )
        if len(receipts) != len(self._receipts):
            raise ValueError("visibility receipt counter differs from persisted JSONL")
        observer_config = self.observer_config
        if (
            not isinstance(observer_config, dict)
            or payload_sha256(observer_config) != self._observer_config_sha256
        ):
            raise ValueError("visibility observer config hash mismatch")
        summary = {
            VISIBILITY_EVIDENCE_ENABLED_KEY: True,
            VISIBILITY_RECEIPTS_JSONL_KEY: str(output_path),
            VISIBILITY_RECEIPT_COUNT_KEY: len(receipts),
            VISIBILITY_RECEIPTS_FILE_SHA256_KEY: file_sha256,
            VISIBILITY_RECEIPT_SET_SHA256_KEY: visibility_receipts_sha256(
                receipts
            ),
            VISIBILITY_OBSERVER_CONFIG_KEY: observer_config,
            VISIBILITY_OBSERVER_CONFIG_SHA256_KEY: (
                self._observer_config_sha256
            ),
        }
        self._finalized_summary = _detached_json_object(summary)
        return _detached_json_object(summary)

    def summary_fields(self, *, processed_scan_count: int) -> dict[str, object]:
        """Return finalized fields without causing observer disk I/O."""

        count = _require_nonnegative_integer(
            processed_scan_count,
            "processed_scan_count",
        )
        if not self.enabled:
            return disabled_visibility_summary_fields()
        if self._finalized_summary is None:
            raise ValueError(
                "visibility receipts must be finalized before summary creation"
            )
        if count != len(self._receipts):
            raise ValueError(
                "processed scan count differs from finalized visibility receipts"
            )
        return _detached_json_object(self._finalized_summary)


def _require_safe_id(value: str, name: str) -> None:
    if _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a safe non-empty identifier")


def _require_nonnegative_integer(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _detached_json_object(value: Mapping[str, object]) -> dict[str, object]:
    detached = json.loads(canonical_json_bytes(value))
    if not isinstance(detached, dict):  # pragma: no cover - input is a mapping.
        raise ValueError("expected a JSON object")
    return detached


__all__ = [
    "FROZEN_ODOM_OBSERVATION_GEOMETRY",
    "LIDAR_VISIBILITY_OBSERVER_CONFIG_SCHEMA_VERSION",
    "LIVE_MAP_OBSERVATION_GEOMETRY",
    "LidarVisibilitySession",
    "disabled_visibility_summary_fields",
    "proposal_detector_config_evidence",
]
