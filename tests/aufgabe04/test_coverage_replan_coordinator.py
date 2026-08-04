from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.aufgabe04.navigation.coverage_replan_coordinator import (
    CoverageReplanCoordinator,
    _front_evidence,
)
from scripts.aufgabe04.navigation.dynamic_route_handoff import RouteUpdateKind
from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.navigation.plan_stand_coverage_survey import (
    main as plan_coverage,
)
from scripts.aufgabe04.navigation.stand_discovery_route import (
    STAND_DISCOVERY_ROUTE_KIND,
)


class CoverageReplanCoordinatorTest(unittest.TestCase):
    @staticmethod
    def _front_details(range_m: float = 0.23):
        return {
            "front_clearance": {
                "source": "front_sector",
                "nearest_valid_range_m": range_m,
                "nearest_valid_bearing_rad": 0.1,
            }
        }

    def test_only_bearing_bound_front_stops_are_recoverable(self):
        self.assertIsNotNone(
            _front_evidence("stuck no progress", self._front_details())
        )
        self.assertIsNotNone(
            _front_evidence("obstacle too close", self._front_details(0.19))
        )
        self.assertIsNone(
            _front_evidence(
                "obstacle too close",
                {
                    "source": "global_hard_scan",
                    "nearest_valid_range_m": 0.10,
                },
            )
        )

    def test_replans_twice_in_process_and_preserves_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            survey = root / "survey"
            session = root / "session"
            survey.mkdir()
            (survey / "coverage_plan.json").write_text("{}\n")
            coordinator = CoverageReplanCoordinator(
                survey_root=survey,
                session_root=session,
                map_yaml=Path("map.yaml"),
                semantic_map_id="arena",
                target_viewpoint_id="survey_vp_001",
                run_id="mission_coverage_000",
                coverage_leg_index=2,
                robot_radius_m=0.105,
                max_replans=2,
                tracking_tube_radius_m=0.03,
            )
            pose = Pose2D(-0.86, -0.46, 3.141592653589793)
            route_1 = (
                Pose2D(pose.x_m, pose.y_m),
                Pose2D(-0.74, -0.46),
                Pose2D(-1.59, -0.01, 0.0),
            )
            route_2 = (
                Pose2D(-0.74, -0.46),
                Pose2D(-0.64, -0.46),
                Pose2D(-1.59, -0.01, 0.0),
            )
            legs = [
                SimpleNamespace(
                    executable_waypoints=tuple(
                        SimpleNamespace(pose=item) for item in route_1
                    ),
                    route_kind=STAND_DISCOVERY_ROUTE_KIND,
                    source_sha256="route_hash_1",
                ),
                SimpleNamespace(
                    executable_waypoints=tuple(
                        SimpleNamespace(pose=item) for item in route_2
                    ),
                    route_kind=STAND_DISCOVERY_ROUTE_KIND,
                    source_sha256="route_hash_2",
                ),
            ]

            def transient_artifacts(**kwargs):
                index = coordinator.replan_count
                overlay = root / f"overlay_{index}.json"
                overlay.write_text("{}\n")
                return {
                    "route_csv": str(root / f"source_route_{index}.csv"),
                    "diagnostics_json": str(root / f"source_diag_{index}.json"),
                    "summary_json": str(root / f"summary_{index}.json"),
                    "transient_obstacle_overlay_json": str(overlay),
                }

            def sealed_artifacts(**kwargs):
                index = coordinator.replan_count
                return {
                    "route_csv": str(root / f"sealed_route_{index}.csv"),
                    "diagnostics_json": str(root / f"sealed_diag_{index}.json"),
                    "route_certificate_json": str(root / f"cert_{index}.json"),
                }

            with patch(
                "scripts.aufgabe04.navigation.coverage_replan_coordinator."
                "record_transient_blockage_replan",
                side_effect=transient_artifacts,
            ) as record, patch(
                "scripts.aufgabe04.navigation.coverage_replan_coordinator."
                "seal_stand_discovery_route",
                side_effect=sealed_artifacts,
            ), patch(
                "scripts.aufgabe04.navigation.coverage_replan_coordinator."
                "load_route_leg",
                side_effect=legs,
            ):
                first = coordinator(
                    pose,
                    "stuck no progress",
                    self._front_details(),
                )
                second = coordinator(
                    route_2[0],
                    "obstacle too close",
                    self._front_details(0.19),
                )
                exhausted = coordinator(
                    route_2[1],
                    "stuck no progress",
                    self._front_details(),
                )

            self.assertEqual(first.kind, RouteUpdateKind.ADOPT)
            self.assertEqual(first.event_fields["start_egress_motion"], "reverse")
            self.assertEqual(
                first.event_fields["target_viewpoint_id"],
                "survey_vp_001",
            )
            self.assertFalse(first.event_fields["semantic_survey_evidence"])
            self.assertEqual(second.kind, RouteUpdateKind.ADOPT)
            self.assertEqual(exhausted.kind, RouteUpdateKind.STOP)
            self.assertEqual(record.call_count, 2)
            self.assertIn(
                "leg_002_replan_001",
                str(record.call_args_list[0].kwargs["output_dir"]),
            )
            self.assertIsNone(record.call_args_list[0].kwargs["existing_overlay_path"])
            self.assertEqual(
                record.call_args_list[1].kwargs["existing_overlay_path"],
                root / "overlay_1.json",
            )
            self.assertEqual(
                len((session / "adaptive_replans.jsonl").read_text().splitlines()),
                2,
            )

    def test_latest_physical_blockage_geometry_plans_a_sealed_reverse_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            survey = root / "survey"
            with redirect_stdout(StringIO()):
                status = plan_coverage(
                    [
                        "--map",
                        "maps/aufgabe03/arena_1p898x3p9_auto.yaml",
                        "--semantic-map-id",
                        "arena_1p898x3p9_auto",
                        "--planning-frame",
                        "map",
                        "--start-x",
                        "-0.5025319639494574",
                        "--start-y",
                        "-0.605412965510235",
                        "--start-yaw",
                        "-3.1376510363781347",
                        "--survey-id",
                        "coordinator_integration",
                        "--output-dir",
                        str(survey),
                        "--lane-count",
                        "1",
                        "--stop-spacing-m",
                        "0.70",
                        "--expected-stand-count",
                        "5",
                    ]
                )
            self.assertEqual(status, 0)
            coordinator = CoverageReplanCoordinator(
                survey_root=survey,
                session_root=root / "session",
                map_yaml=Path(
                    "maps/aufgabe03/arena_1p898x3p9_auto.yaml"
                ),
                semantic_map_id="arena_1p898x3p9_auto",
                target_viewpoint_id="survey_vp_001",
                run_id="physical_geometry_coverage_000",
                coverage_leg_index=0,
                robot_radius_m=0.105,
                max_replans=3,
                tracking_tube_radius_m=0.03,
            )

            update = coordinator(
                Pose2D(
                    -0.858887873410987,
                    -0.46164086690318107,
                    -3.1376510363781347,
                ),
                "stuck no progress",
                {
                    "front_clearance": {
                        "source": "front_sector",
                        "nearest_valid_range_m": 0.23000000417232513,
                        "nearest_valid_bearing_rad": 0.20737460535019636,
                    }
                },
            )

            self.assertEqual(update.kind, RouteUpdateKind.ADOPT)
            self.assertEqual(update.event_fields["start_egress_motion"], "reverse")
            self.assertTrue(
                Path(update.event_fields["replacement_route_csv"]).is_file()
            )
            self.assertTrue(
                Path(
                    update.event_fields[
                        "replacement_route_certificate_json"
                    ]
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
