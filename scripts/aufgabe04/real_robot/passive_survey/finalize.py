#!/usr/bin/env python3
"""Freeze a complete real catalog and publish its sealed survey manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json,
    payload_sha256,
)
from scripts.aufgabe04.artifacts.manifest_store import write_survey_manifest
from scripts.aufgabe04.artifacts.models import (
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    SurveyManifest,
    artifact_reference,
)
from scripts.aufgabe04.navigation.planning.map_io import freeze_map_bundle
from scripts.aufgabe04.real_robot.configuration.profile import (
    camera_calibration_sha256,
    load_camera_calibration,
    load_real_robot_profile,
)
from scripts.aufgabe04.stations.arrival_pose_catalog import (
    arrival_pose_catalog_sha256,
    freeze_arrival_pose_catalog,
    load_arrival_pose_catalog,
    write_arrival_pose_catalog,
)
from scripts.aufgabe04.stations.arrival_pose_models import CatalogProvenance
from scripts.aufgabe04.stations.candidate_snapshot import (
    candidate_snapshot_sha256,
    load_candidate_snapshot,
)
from scripts.aufgabe04.stations.station_identity_registry import (
    load_station_identity_registry,
    station_identity_registry_sha256,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-profile", required=True, type=Path)
    parser.add_argument("--camera-calibration", required=True, type=Path)
    parser.add_argument("--physical-site", required=True, type=Path)
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--semantic-map-id", required=True)
    parser.add_argument("--candidate-snapshot", required=True, type=Path)
    parser.add_argument("--station-identity-registry", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--survey-config", required=True, type=Path)
    parser.add_argument("--survey-input-binding", required=True, type=Path)
    parser.add_argument("--survey-manifest", required=True, type=Path)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        profile = load_real_robot_profile(args.robot_profile)
        calibration = load_camera_calibration(args.camera_calibration)
        calibration_sha256 = camera_calibration_sha256(calibration)
        if calibration_sha256 != profile.calibration_profile_sha256:
            raise ValueError("robot profile and camera calibration differ")
        site_sha256 = _file_sha256(args.physical_site)
        if (
            site_sha256 != profile.physical_site_sha256
            or args.physical_site.stem != profile.physical_site_id
        ):
            raise ValueError("physical site descriptor differs from robot profile")
        map_bundle = freeze_map_bundle(
            args.map,
            semantic_map_id=args.semantic_map_id,
            planning_frame=profile.map_frame,
        )
        snapshot = load_candidate_snapshot(
            args.candidate_snapshot,
            required_map_bundle_sha256=map_bundle.bundle_sha256,
        )
        registry = load_station_identity_registry(
            args.station_identity_registry,
            candidate_snapshot=snapshot,
        )
        survey_config = load_content_hashed_json(
            args.survey_config,
            hash_field="survey_config_sha256",
        )
        survey_config_sha256 = payload_sha256(survey_config)
        binding = load_content_hashed_json(
            args.survey_input_binding,
            hash_field="survey_input_binding_sha256",
        )
        binding_sha256 = payload_sha256(binding)
        provenance = {
            "planning_frame": profile.map_frame,
            "map_yaml_sha256": map_bundle.yaml_sha256,
            "world_id": profile.physical_site_id,
            "world_sha256": site_sha256,
            "session_id": args.session_id,
            "environment": "real",
            "map_bundle_sha256": map_bundle.bundle_sha256,
            "candidate_snapshot_sha256": candidate_snapshot_sha256(snapshot),
            "station_identity_registry_sha256": (
                station_identity_registry_sha256(registry)
            ),
            "survey_config_sha256": survey_config_sha256,
            "calibration_profile_sha256": calibration_sha256,
            "survey_input_binding_sha256": binding_sha256,
        }
        catalog = load_arrival_pose_catalog(args.catalog)
        if catalog.provenance != CatalogProvenance(**provenance):
            raise ValueError("arrival catalog provenance differs from real survey inputs")
        if not catalog.complete:
            unresolved = sorted(
                set(catalog.expected_candidate_uids)
                - set(catalog.resolved_candidate_uids)
            )
            raise ValueError(f"real arrival catalog is incomplete: {unresolved}")
        if not catalog.frozen:
            catalog = freeze_arrival_pose_catalog(
                catalog,
                frozen_unix_sec=max(time.time(), catalog.updated_unix_sec),
            )
            write_arrival_pose_catalog(args.catalog, catalog)
        catalog_sha256 = arrival_pose_catalog_sha256(catalog)
        manifest = SurveyManifest(
            schema_version=ARTIFACT_MANIFEST_SCHEMA_VERSION,
            manifest_id=f"survey_{args.session_id}_{catalog_sha256[:16]}",
            created_unix_sec=catalog.updated_unix_sec,
            session_id=args.session_id,
            environment="real",
            planning_frame=profile.map_frame,
            map_bundle=artifact_reference(
                "map_bundle",
                map_bundle.semantic_map_id,
                map_bundle.bundle_sha256,
            ),
            candidate_snapshot=artifact_reference(
                "candidate_snapshot",
                snapshot.snapshot_id,
                candidate_snapshot_sha256(snapshot),
            ),
            environment_descriptor=artifact_reference(
                "physical_site",
                profile.physical_site_id,
                site_sha256,
            ),
            survey_config=artifact_reference(
                "survey_config",
                f"survey_config_{survey_config_sha256[:16]}",
                survey_config_sha256,
            ),
            calibration_profile=artifact_reference(
                "calibration_profile",
                calibration.calibration_id,
                calibration_sha256,
            ),
            arrival_pose_catalog=artifact_reference(
                "arrival_pose_catalog",
                catalog.catalog_id,
                catalog_sha256,
            ),
        )
        manifest_sha256 = write_survey_manifest(args.survey_manifest, manifest)
        print(
            f"survey_manifest={args.survey_manifest}\n"
            f"survey_manifest_sha256={manifest_sha256}\n"
            f"arrival_pose_catalog_sha256={catalog_sha256}"
        )
        return 0
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
