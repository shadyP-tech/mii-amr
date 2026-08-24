from __future__ import annotations

import ast
import dataclasses
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json,
    payload_sha256,
)
from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.map_io import (
    CELL_FREE,
    CELL_OCCUPIED,
    MapMetadata,
    OccupancyGrid,
    load_occupancy_grid_with_bundle,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.coverage.record_stand_coverage_stop import (
    commit_stand_coverage_stop,
)
from scripts.aufgabe04.navigation.coverage import (
    record_stand_coverage_stop as coverage_stop,
)
from scripts.aufgabe04.navigation.coverage import (
    coverage_stop_perception_admission as perception_admission,
)
from scripts.aufgabe04.navigation.coverage.stand_candidate_static_map_admission import (
    STATIC_MAP_CLEARANCE_BELOW_REQUIRED,
    evaluate_stand_candidate_static_map_admission,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    build_coverage_survey_plan,
    load_stand_survey_registry,
    new_stand_survey_registry,
    new_survey_progress,
    write_coverage_survey_plan,
    write_stand_survey_registry,
    write_survey_progress,
)
from scripts.aufgabe04.perception.stand_confirmation import ConfirmedStand
from scripts.aufgabe04.perception.lidar_stand_morphology import (
    MORPHOLOGY_PROFILE_EVIDENCE_KEY,
    MORPHOLOGY_PROFILE_SHA256_KEY,
    PROPOSAL_DETECTOR_CONFIG_EVIDENCE_KEY,
    stand_width_profile_from_radius,
)


def costmap_from_rows(
    rows,
    *,
    resolution=0.1,
    origin=(0.0, 0.0, 0.0),
) -> Costmap:
    metadata = MapMetadata(
        yaml_path=Path("map.yaml"),
        image_path=Path("map.pgm"),
        resolution=resolution,
        origin=origin,
        negate=0,
        occupied_thresh=0.65,
        free_thresh=0.20,
        mode="trinary",
    )
    grid = OccupancyGrid(
        metadata=metadata,
        width=len(rows[0]),
        height=len(rows),
        cells=tuple(tuple(row) for row in rows),
    )
    return Costmap.from_occupancy_grid(grid)


def _costmap_fixture() -> Costmap:
    rows = [[CELL_FREE] * 10 for _ in range(10)]
    rows[4][4] = CELL_OCCUPIED
    return costmap_from_rows(rows)


def stand(stand_id: str, x_m: float, y_m: float) -> ConfirmedStand:
    return ConfirmedStand(
        stand_id=stand_id,
        x_m=x_m,
        y_m=y_m,
        confidence=0.9,
        hit_count=3,
        first_seen_sec=1.0,
        last_seen_sec=2.0,
        first_confirmed_at_sec=2.0,
        source_observation_ids=(f"observation_{stand_id}",),
        provenance={"source": "synthetic_test"},
    )


class StandCandidateStaticMapAdmissionTest(unittest.TestCase):
    def test_filters_static_overlap_with_explicit_radius_and_uncertainty(self):
        too_close = stand("stand_close", 0.55, 0.45)
        clear = stand("stand_clear", 0.60, 0.45)

        result = evaluate_stand_candidate_static_map_admission(
            _costmap_fixture(),
            (clear, too_close),
            candidate_radius_m=0.06,
            candidate_uncertainty_m=0.02,
        )

        self.assertAlmostEqual(result.required_clearance_m, 0.08)
        self.assertEqual(result.admitted_stands, (clear,))
        self.assertEqual(result.rejected_stands, (too_close,))
        self.assertEqual(
            tuple(item.stand_id for item in result.evidence),
            ("stand_close", "stand_clear"),
        )
        self.assertAlmostEqual(result.evidence[0].static_map_clearance_m, 0.049999)
        self.assertEqual(
            result.evidence[0].reasons,
            (STATIC_MAP_CLEARANCE_BELOW_REQUIRED,),
        )
        self.assertEqual(result.evidence[0].confidence, too_close.confidence)
        self.assertEqual(result.evidence[0].hit_count, too_close.hit_count)
        self.assertEqual(
            result.evidence[0].source_observation_ids,
            too_close.source_observation_ids,
        )
        serialized = result.to_evidence_dict()["candidate_evidence"][0]
        self.assertEqual(
            serialized["source_observation_ids"],
            list(too_close.source_observation_ids),
        )
        self.assertAlmostEqual(result.evidence[1].static_map_clearance_m, 0.099999)
        self.assertEqual(result.evidence[1].reasons, ())

    def test_boundary_and_blocked_cell_centres_fail_closed(self):
        boundary = stand("stand_boundary", 0.05, 0.75)
        occupied = stand("stand_occupied", 0.45, 0.45)

        result = evaluate_stand_candidate_static_map_admission(
            _costmap_fixture(),
            (boundary, occupied),
            candidate_radius_m=0.06,
            candidate_uncertainty_m=0.02,
        )

        self.assertEqual(result.admitted_stands, ())
        self.assertEqual(result.rejected_stands, (boundary, occupied))
        self.assertAlmostEqual(result.evidence[0].static_map_clearance_m, 0.049999)
        self.assertEqual(result.evidence[1].static_map_clearance_m, 0.0)

    def test_evidence_is_order_independent_and_reports_ordered_counts(self):
        first = stand("stand_b", 0.60, 0.45)
        second = stand("stand_a", 0.70, 0.45)

        left = evaluate_stand_candidate_static_map_admission(
            _costmap_fixture(),
            (second, first),
            candidate_radius_m=0.06,
            candidate_uncertainty_m=0.02,
        )
        right = evaluate_stand_candidate_static_map_admission(
            _costmap_fixture(),
            (first, second),
            candidate_radius_m=0.06,
            candidate_uncertainty_m=0.02,
        )

        self.assertEqual(left.to_evidence_dict(), right.to_evidence_dict())
        self.assertEqual(
            payload_sha256(left.to_evidence_dict()),
            payload_sha256(right.to_evidence_dict()),
        )
        self.assertEqual(
            left.to_evidence_dict()["counts"],
            {"evaluated": 2, "admitted": 2, "rejected": 0},
        )
        self.assertEqual(
            left.to_evidence_dict()["admitted_stand_ids"],
            ["stand_b", "stand_a"],
        )

    def test_empty_epoch_is_valid_and_does_not_invent_candidates(self):
        result = evaluate_stand_candidate_static_map_admission(
            _costmap_fixture(),
            (),
            candidate_radius_m=0.06,
            candidate_uncertainty_m=0.02,
        )

        self.assertEqual(result.admitted_stands, ())
        self.assertEqual(result.rejected_stands, ())
        self.assertEqual(
            result.to_evidence_dict()["counts"],
            {"evaluated": 0, "admitted": 0, "rejected": 0},
        )

    def test_latest_run_vp5_condensed_diagnostic_replay(self):
        """Replay consumed VP5 centroids; this is not promotion evidence."""

        # The relevant static-map geometry is the 5 cm raster wall whose free
        # face begins at map x=1.83 m.  This condensed fixture intentionally
        # retains only that decision boundary and the nine raw VP5 centroids
        # from stand_explore_full_20260819T130925Z.
        rows = [[CELL_FREE] * 113 for _ in range(71)]
        for row in rows:
            row[93] = CELL_OCCUPIED
        costmap = costmap_from_rows(
            rows,
            resolution=0.05,
            origin=(-2.82, -1.69, 0.0),
        ).with_arena_bounds(
            ArenaBounds(length_m=3.9, width_m=1.898)
        )
        replayed = (
            stand("detected_stand_02", -1.0516118130316774, -0.5241597212355844),
            stand("detected_stand_09", -0.21617733474685488, 0.6331782102273432),
            stand("detected_stand_03", 0.09043282118752823, -0.7404896966552552),
            stand("detected_stand_01", 1.1446875115011619, 0.7827202779165287),
            stand("detected_stand_04", 1.4153178127328256, -0.21133267880943943),
            stand("detected_stand_08", 1.7796128911065858, 0.6875749491279549),
            stand("detected_stand_06", 1.7993301894495424, 0.4198307329624607),
            stand("detected_stand_05", 1.803308911361011, 0.20791979303784403),
            stand("detected_stand_07", 1.825087442681177, -0.7004310207578732),
        )

        result = evaluate_stand_candidate_static_map_admission(
            costmap,
            replayed,
            candidate_radius_m=0.06,
            candidate_uncertainty_m=0.02,
        )

        self.assertEqual(
            tuple(item.stand_id for item in result.admitted_stands),
            (
                "detected_stand_02",
                "detected_stand_09",
                "detected_stand_03",
                "detected_stand_01",
                "detected_stand_04",
            ),
        )
        self.assertEqual(
            tuple(item.stand_id for item in result.rejected_stands),
            (
                "detected_stand_08",
                "detected_stand_06",
                "detected_stand_05",
                "detected_stand_07",
            ),
        )
        self.assertTrue(
            all(
                item.static_map_clearance_m < result.required_clearance_m
                for item in result.evidence[-4:]
            )
        )

    def test_rejects_invalid_contract_and_ambiguous_candidate_ids(self):
        valid = stand("stand_a", 0.60, 0.45)
        for radius, uncertainty in (
            (0.0, 0.02),
            (0.06, -0.01),
            (math.nan, 0.02),
            (0.06, math.inf),
        ):
            with self.subTest(radius=radius, uncertainty=uncertainty):
                with self.assertRaises(ValueError):
                    evaluate_stand_candidate_static_map_admission(
                        _costmap_fixture(),
                        (valid,),
                        candidate_radius_m=radius,
                        candidate_uncertainty_m=uncertainty,
                    )

        with self.assertRaisesRegex(ValueError, "duplicate confirmed stand ID"):
            evaluate_stand_candidate_static_map_admission(
                _costmap_fixture(),
                (valid, dataclasses.replace(valid, x_m=0.70)),
                candidate_radius_m=0.06,
                candidate_uncertainty_m=0.02,
            )

    def test_module_remains_ros_free(self):
        module_path = (
            Path(__file__).resolve().parents[2]
            / "scripts/aufgabe04/navigation/coverage/stand_candidate_static_map_admission.py"
        )
        tree = ast.parse(module_path.read_text())
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        )
        self.assertFalse(
            any(
                name == "rclpy" or name.startswith("rclpy.")
                for name in imported
            )
        )


class StandCandidateStaticMapAdmissionIntegrationTest(unittest.TestCase):
    def test_exact_two_observer_morphology_contract_fails_closed(self):
        plan = SimpleNamespace(
            config=SimpleNamespace(
                candidate_radius_m=0.06,
                exact_inspection_point_count=2,
            )
        )
        profile = stand_width_profile_from_radius(0.06)
        profile_payload = profile.to_evidence_dict()
        valid = {
            MORPHOLOGY_PROFILE_EVIDENCE_KEY: profile_payload,
            MORPHOLOGY_PROFILE_SHA256_KEY: payload_sha256(
                profile_payload
            ),
            PROPOSAL_DETECTOR_CONFIG_EVIDENCE_KEY: {
                "min_width_m": 0.03,
                "max_width_m": 0.45,
            },
        }

        with self.assertRaisesRegex(ValueError, "no LiDAR morphology profile"):
            perception_admission.validate_observer_morphology_contract({}, plan)
        with self.assertRaisesRegex(ValueError, "profile differs"):
            perception_admission.validate_observer_morphology_contract(
                {
                    **valid,
                    MORPHOLOGY_PROFILE_EVIDENCE_KEY: {
                        **profile_payload,
                        "expected_diameter_m": 0.20,
                    },
                },
                plan,
            )
        with self.assertRaisesRegex(ValueError, "would censor"):
            perception_admission.validate_observer_morphology_contract(
                {
                    **valid,
                    PROPOSAL_DETECTOR_CONFIG_EVIDENCE_KEY: {
                        "min_width_m": 0.03,
                        "max_width_m": 0.18,
                    },
                },
                plan,
            )

        evidence = perception_admission.validate_observer_morphology_contract(
            valid,
            plan,
        )
        self.assertTrue(
            evidence["proposal_width_evidence_preservation"]
            ["preserves_track_morphology_evidence"]
        )

    def test_commit_filters_before_fusion_and_binds_hashed_epoch_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_yaml = self._write_map_with_central_static_cell(root)
            grid, bundle = load_occupancy_grid_with_bundle(
                map_yaml,
                semantic_map_id=map_yaml.stem,
                planning_frame="map",
            )
            plan = build_coverage_survey_plan(
                grid,
                map_bundle_sha256=bundle.bundle_sha256,
                start=Pose2D(-1.5, 0.0, 0.0),
                survey_id="static_map_admission_integration",
                arena_bounds=ArenaBounds(length_m=4.0, width_m=2.0),
            )
            survey_root = root / "survey"
            write_coverage_survey_plan(
                survey_root / "coverage_plan.json",
                plan,
            )
            write_survey_progress(
                survey_root / "coverage_progress.json",
                new_survey_progress(plan),
                plan,
            )
            write_stand_survey_registry(
                survey_root / "stand_registry.json",
                new_stand_survey_registry(plan),
                plan,
            )
            viewpoint = plan.viewpoints[0]
            observations_path = root / "negative_observations.jsonl"
            observer_summary = root / "observer_summary.json"
            observer_summary.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "motion_published": False,
                        "processed_scan_count": 5,
                        "accepted_observation_count": 0,
                        "map_bundle_sha256": bundle.bundle_sha256,
                        "planning_frame": "map",
                        "output_jsonl": str(observations_path),
                        "scan_frame_pose_in_planning_frame": {
                            "x_m": viewpoint.pose.x_m,
                            "y_m": viewpoint.pose.y_m,
                            "yaw_rad": viewpoint.pose.yaw_rad,
                        },
                    }
                )
            )
            occupied = next(iter(Costmap.from_occupancy_grid(grid).blocked_cells))
            occupied_center = Costmap.from_occupancy_grid(grid).grid_to_world(
                occupied
            )
            occupied_right_edge = occupied_center.x_m + grid.metadata.resolution / 2
            rejected = stand(
                "stand_static_overlap",
                occupied_right_edge + 0.05,
                occupied_center.y_m,
            )
            admitted = stand(
                "stand_free_space",
                occupied_right_edge + 0.15,
                occupied_center.y_m,
            )

            morphology = SimpleNamespace(
                admitted_stands=(admitted, rejected),
                rejected_stands=(),
                to_evidence_dict=lambda: {
                    "schema_version": 1,
                    "gate": "lidar_stand_track_morphology_admission",
                    "counts": {"evaluated": 2, "admitted": 2, "rejected": 0},
                },
            )
            with (
                patch.object(
                    coverage_stop,
                    "build_confirmed_epoch_stands",
                    return_value=(admitted, rejected),
                ),
                patch.object(
                    perception_admission,
                    "evaluate_stand_morphology_admission",
                    return_value=morphology,
                ),
            ):
                status = commit_stand_coverage_stop(
                    survey_root=survey_root,
                    map_yaml=map_yaml,
                    viewpoint_id=viewpoint.viewpoint_id,
                    observer_summary_json=observer_summary,
                )

            registry = load_stand_survey_registry(
                survey_root / "stand_registry.json",
                plan,
            )
            epoch = json.loads(Path(str(status["epoch_json"])).read_text())
            evidence_path = Path(
                str(status["static_map_candidate_admission_json"])
            )
            evidence = load_content_hashed_json(
                evidence_path,
                hash_field="static_map_candidate_admission_sha256",
            )
            morphology_evidence_path = Path(
                str(status["lidar_morphology_admission_json"])
            )
            morphology_evidence = load_content_hashed_json(
                morphology_evidence_path,
                hash_field="lidar_morphology_admission_sha256",
            )

        self.assertEqual(len(registry.candidates), 1)
        self.assertAlmostEqual(registry.candidates[0].x_m, admitted.x_m)
        self.assertEqual(status["epoch_confirmed_lidar_candidate_count"], 2)
        self.assertEqual(status["epoch_morphology_admitted_candidate_count"], 2)
        self.assertEqual(status["epoch_morphology_rejected_candidate_count"], 0)
        self.assertEqual(status["epoch_static_map_admitted_candidate_count"], 1)
        self.assertEqual(status["epoch_static_map_rejected_candidate_count"], 1)
        self.assertEqual(status["fused_registry_active_candidate_count"], 1)
        self.assertEqual(status["fused_registry_total_candidate_count"], 1)
        self.assertEqual(status["confirmed_epoch_candidate_count"], 2)
        self.assertEqual(status["static_map_candidate_admitted_count"], 1)
        self.assertEqual(status["static_map_candidate_rejected_count"], 1)
        self.assertEqual(
            status["legacy_epoch_candidate_count_aliases"][
                "static_map_candidate_admitted_count"
            ],
            "epoch_static_map_admitted_candidate_count",
        )
        self.assertEqual(
            epoch["static_map_candidate_admission_sha256"],
            status["static_map_candidate_admission_sha256"],
        )
        self.assertIn(
            status["static_map_candidate_admission_sha256"],
            evidence_path.name,
        )
        self.assertEqual(
            evidence["counts"],
            {"evaluated": 2, "admitted": 1, "rejected": 1},
        )
        self.assertEqual(evidence["rejected_stand_ids"], [rejected.stand_id])
        self.assertEqual(
            morphology_evidence["counts"],
            {"evaluated": 2, "admitted": 2, "rejected": 0},
        )
        self.assertEqual(
            epoch["lidar_morphology_admission_sha256"],
            status["lidar_morphology_admission_sha256"],
        )
        self.assertIn(
            status["lidar_morphology_admission_sha256"],
            morphology_evidence_path.name,
        )

    @staticmethod
    def _write_map_with_central_static_cell(root: Path) -> Path:
        width = 50
        height = 30
        pixels = [255] * (width * height)
        pixels[(height // 2) * width + width // 2] = 0
        (root / "map.pgm").write_text(
            f"P2\n{width} {height}\n255\n"
            + " ".join(str(pixel) for pixel in pixels)
            + "\n"
        )
        map_yaml = root / "map.yaml"
        map_yaml.write_text(
            "\n".join(
                [
                    "image: map.pgm",
                    "resolution: 0.1",
                    "origin: [-2.5, -1.5, 0.0]",
                    "negate: 0",
                    "occupied_thresh: 0.65",
                    "free_thresh: 0.20",
                    "mode: trinary",
                ]
            )
            + "\n"
        )
        return map_yaml


if __name__ == "__main__":
    unittest.main()
