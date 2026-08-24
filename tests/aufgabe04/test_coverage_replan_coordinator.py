from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.aufgabe04.navigation.coverage.coverage_replan_coordinator import (
    CoverageReplanCoordinator,
    _front_evidence,
    _load_escape_metadata,
)
from scripts.aufgabe04.navigation.execution.dynamic_route_handoff import RouteUpdateKind
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.missions.plan_stand_coverage_survey import (
    main as plan_coverage,
)
from scripts.aufgabe04.navigation.coverage.stand_discovery_route import (
    STAND_DISCOVERY_ROUTE_KIND,
)


class CoverageReplanCoordinatorTest(unittest.TestCase):
    @staticmethod
    def _front_details(range_m: float = 0.23, bearing_rad: float = 0.1):
        return {
            "stationary_obstacle_confirmation": {
                "confirmed": True,
                "fail_closed": False,
                "distinct_sample_count": 3,
                "thresholds": {"min_distinct_samples": 3},
            },
            "front_clearance": {
                "source": "front_sector",
                "nearest_valid_range_m": range_m,
                "nearest_valid_bearing_rad": bearing_rad,
            }
        }

    def test_only_bearing_bound_front_stops_are_recoverable(self):
        self.assertIsNotNone(
            _front_evidence("stuck no progress", self._front_details())
        )
        self.assertIsNotNone(
            _front_evidence("obstacle too close", self._front_details(0.19))
        )
        self.assertIsNotNone(
            _front_evidence(
                "clearance-limited motion floor",
                self._front_details(0.234),
            )
        )
        without_confirmation = self._front_details()
        without_confirmation.pop("stationary_obstacle_confirmation")
        self.assertIsNone(
            _front_evidence("obstacle too close", without_confirmation)
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
        self.assertIsNone(
            _front_evidence(
                "pose left certified route tube",
                self._front_details(0.234),
            )
        )
        only_two_scans = self._front_details(0.234)
        only_two_scans["stationary_obstacle_confirmation"].update(
            {
                "distinct_sample_count": 2,
                "thresholds": {"min_distinct_samples": 2},
            }
        )
        self.assertIsNone(
            _front_evidence(
                "clearance-limited motion floor",
                only_two_scans,
            )
        )

    def test_forward_escape_metadata_accepts_its_outgoing_waypoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            diagnostics = Path(tmp) / "diagnostics.json"
            waypoints = (
                Pose2D(0.0, 0.0, 0.0),
                Pose2D(0.1, 0.0, 0.0),
                Pose2D(0.2, 0.0, 0.0),
            )
            diagnostics.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "egress_mode": "forward",
                            "egress_anchor": {"x_m": 0.1, "y_m": 0.0},
                            "egress_transition_anchor": {
                                "x_m": 0.1,
                                "y_m": 0.0,
                            },
                            "egress_transition_waypoint_index": 1,
                            "egress_forward_waypoint_index": 2,
                            "forward_translation_heading_limit_rad": 1.25,
                            "reverse_connector_alignment_tolerance_rad": 0.10,
                            "tracking_tube_radius_m": 0.03,
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            metadata = _load_escape_metadata(
                diagnostics,
                waypoints,
                tracking_tube_radius_m=0.03,
                forward_translation_heading_limit_rad=1.25,
                reverse_connector_alignment_tolerance_rad=0.10,
            )
            with self.assertRaisesRegex(
                ValueError,
                "reverse-alignment tolerance differs",
            ):
                _load_escape_metadata(
                    diagnostics,
                    waypoints,
                    tracking_tube_radius_m=0.03,
                    forward_translation_heading_limit_rad=1.25,
                    reverse_connector_alignment_tolerance_rad=0.08,
                )

        self.assertEqual(metadata["egress_mode"], "forward")
        self.assertEqual(metadata["egress_forward_waypoint_index"], 2)

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
                Pose2D(-0.69, -0.46),
                Pose2D(-0.69, -0.41, 0.0),
            )
            route_2 = (
                Pose2D(-0.74, -0.46),
                Pose2D(-0.64, -0.46),
                Pose2D(-0.59, -0.46),
                Pose2D(-0.59, -0.41, 0.0),
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
                diagnostics = root / f"source_diag_{index}.json"
                route = route_1 if index == 1 else route_2
                overlay.write_text(
                    '{"candidates":[{"x_m":-1.10,"y_m":-0.46,'
                    '"keepout_radius_m":0.30}]}\n'
                )
                diagnostics.write_text(
                    json.dumps(
                        {
                            "metadata": {
                                "egress_mode": "straight_reverse",
                                "egress_anchor": {
                                    "x_m": route[1].x_m,
                                    "y_m": route[1].y_m,
                                },
                                "egress_transition_anchor": {
                                    "x_m": route[2].x_m,
                                    "y_m": route[2].y_m,
                                },
                                "egress_transition_waypoint_index": 2,
                                "egress_forward_waypoint_index": 3,
                                "forward_translation_heading_limit_rad": 1.25,
                                "reverse_connector_alignment_tolerance_rad": 0.10,
                                "reverse_connector_heading_error_rad": 0.0,
                                "minimum_transition_keepout_tube_clearance_m": 0.01,
                                "tracking_tube_radius_m": 0.03,
                            }
                        }
                    )
                    + "\n"
                )
                return {
                    "route_csv": str(root / f"source_route_{index}.csv"),
                    "diagnostics_json": str(diagnostics),
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
                "scripts.aufgabe04.navigation.coverage.coverage_replan_coordinator."
                "record_transient_blockage_replan",
                side_effect=transient_artifacts,
            ) as record, patch(
                "scripts.aufgabe04.navigation.coverage.coverage_replan_coordinator."
                "seal_stand_discovery_route",
                side_effect=sealed_artifacts,
            ), patch(
                "scripts.aufgabe04.navigation.coverage.coverage_replan_coordinator."
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
                first.event_fields[
                    "start_egress_reverse_until_waypoint_index"
                ],
                2,
            )
            self.assertEqual(
                first.event_fields[
                    "start_egress_forward_alignment_waypoint_index"
                ],
                3,
            )
            self.assertEqual(
                first.event_fields["target_viewpoint_id"],
                "survey_vp_001",
            )
            self.assertFalse(first.event_fields["semantic_survey_evidence"])
            self.assertEqual(second.kind, RouteUpdateKind.ADOPT)
            self.assertEqual(exhausted.kind, RouteUpdateKind.STOP)
            self.assertEqual(record.call_count, 2)
            self.assertEqual(
                record.call_args_list[0].kwargs["tracking_tube_radius_m"],
                0.03,
            )
            self.assertIn(
                "leg_002_replan_001",
                str(record.call_args_list[0].kwargs["output_dir"]),
            )
            self.assertIsNone(record.call_args_list[0].kwargs["existing_overlay_path"])
            self.assertEqual(
                record.call_args_list[1].kwargs["existing_overlay_path"],
                root / "overlay_1.json",
            )
            self.assertFalse((session / "adaptive_replans.jsonl").exists())

    def test_resumed_replans_preserve_overlay_revision_and_cumulative_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            survey = root / "survey"
            session = root / "session"
            survey.mkdir()
            (survey / "coverage_plan.json").write_text("{}\n")
            initial_overlay = root / "overlay_1.json"
            initial_overlay.write_text(
                '{"candidates":[{"x_m":-1.10,"y_m":-0.46,'
                '"keepout_radius_m":0.30}]}\n'
            )
            coordinator = CoverageReplanCoordinator(
                survey_root=survey,
                session_root=session,
                map_yaml=Path("map.yaml"),
                semantic_map_id="arena",
                target_viewpoint_id="survey_vp_001",
                run_id="mission_coverage_000_runtime_localization_reseal_001",
                coverage_leg_index=2,
                robot_radius_m=0.105,
                max_replans=3,
                tracking_tube_radius_m=0.03,
                replan_count=1,
                overlay_path=initial_overlay,
                adopted_route_hashes={"route_hash_1"},
            )
            pose = Pose2D(-0.86, -0.46, 3.141592653589793)
            routes = {
                2: (
                    Pose2D(pose.x_m, pose.y_m),
                    Pose2D(-0.74, -0.46),
                    Pose2D(-0.69, -0.46),
                    Pose2D(-0.69, -0.41, 0.0),
                ),
                3: (
                    Pose2D(-0.74, -0.46),
                    Pose2D(-0.64, -0.46),
                    Pose2D(-0.59, -0.46),
                    Pose2D(-0.59, -0.41, 0.0),
                ),
            }

            def transient_artifacts(**_kwargs):
                index = coordinator.replan_count
                overlay = root / f"overlay_{index}.json"
                diagnostics = root / f"source_diag_{index}.json"
                route = routes[index]
                overlay.write_text(
                    '{"candidates":[{"x_m":-1.10,"y_m":-0.46,'
                    '"keepout_radius_m":0.30}]}\n'
                )
                diagnostics.write_text(
                    json.dumps(
                        {
                            "metadata": {
                                "egress_mode": "straight_reverse",
                                "egress_anchor": {
                                    "x_m": route[1].x_m,
                                    "y_m": route[1].y_m,
                                },
                                "egress_transition_anchor": {
                                    "x_m": route[2].x_m,
                                    "y_m": route[2].y_m,
                                },
                                "egress_transition_waypoint_index": 2,
                                "egress_forward_waypoint_index": 3,
                                "forward_translation_heading_limit_rad": 1.25,
                                "reverse_connector_alignment_tolerance_rad": 0.10,
                                "reverse_connector_heading_error_rad": 0.0,
                                "minimum_transition_keepout_tube_clearance_m": 0.01,
                                "tracking_tube_radius_m": 0.03,
                            }
                        }
                    )
                    + "\n"
                )
                return {
                    "route_csv": str(root / f"source_route_{index}.csv"),
                    "diagnostics_json": str(diagnostics),
                    "summary_json": str(root / f"summary_{index}.json"),
                    "transient_obstacle_overlay_json": str(overlay),
                }

            def sealed_artifacts(**_kwargs):
                index = coordinator.replan_count
                return {
                    "route_csv": str(root / f"sealed_route_{index}.csv"),
                    "diagnostics_json": str(root / f"sealed_diag_{index}.json"),
                    "route_certificate_json": str(root / f"cert_{index}.json"),
                }

            def loaded_leg(*_args, **_kwargs):
                index = coordinator.replan_count
                return SimpleNamespace(
                    executable_waypoints=tuple(
                        SimpleNamespace(pose=item) for item in routes[index]
                    ),
                    route_kind=STAND_DISCOVERY_ROUTE_KIND,
                    source_sha256=f"route_hash_{index}",
                )

            with patch(
                "scripts.aufgabe04.navigation.coverage.coverage_replan_coordinator."
                "record_transient_blockage_replan",
                side_effect=transient_artifacts,
            ) as record, patch(
                "scripts.aufgabe04.navigation.coverage.coverage_replan_coordinator."
                "seal_stand_discovery_route",
                side_effect=sealed_artifacts,
            ) as seal, patch(
                "scripts.aufgabe04.navigation.coverage.coverage_replan_coordinator."
                "load_route_leg",
                side_effect=loaded_leg,
            ):
                second = coordinator(
                    pose,
                    "stuck no progress",
                    self._front_details(),
                )
                third = coordinator(
                    routes[2][1],
                    "obstacle too close",
                    self._front_details(0.19),
                )
                exhausted = coordinator(
                    routes[3][1],
                    "stuck no progress",
                    self._front_details(),
                )

            self.assertEqual(second.kind, RouteUpdateKind.ADOPT)
            self.assertEqual(second.route_revision, 2)
            self.assertEqual(second.event_fields["replan_index"], 2)
            self.assertEqual(third.kind, RouteUpdateKind.ADOPT)
            self.assertEqual(third.route_revision, 3)
            self.assertEqual(third.event_fields["replan_index"], 3)
            self.assertEqual(exhausted.kind, RouteUpdateKind.STOP)
            self.assertEqual(coordinator.replan_count, 3)
            self.assertEqual(record.call_count, 2)
            self.assertEqual(
                record.call_args_list[0].kwargs["existing_overlay_path"],
                initial_overlay,
            )
            self.assertEqual(
                record.call_args_list[1].kwargs["existing_overlay_path"],
                root / "overlay_2.json",
            )
            self.assertIn(
                "leg_002_replan_002",
                str(record.call_args_list[0].kwargs["output_dir"]),
            )
            self.assertIn(
                "leg_002_replan_003",
                str(record.call_args_list[1].kwargs["output_dir"]),
            )
            self.assertIn(
                "coverage_leg_002_replan_002",
                str(seal.call_args_list[0].kwargs["output_dir"]),
            )
            self.assertIn(
                "coverage_leg_002_replan_003",
                str(seal.call_args_list[1].kwargs["output_dir"]),
            )
            self.assertNotIn(
                "replan_001",
                "\n".join(
                    [
                        *(
                            str(call.kwargs["output_dir"])
                            for call in record.call_args_list
                        ),
                        *(
                            str(call.kwargs["output_dir"])
                            for call in seal.call_args_list
                        ),
                    ]
                ),
            )
            self.assertFalse(
                (survey / "replans" / "leg_002_replan_001").exists()
            )
            self.assertEqual(
                coordinator.adopted_route_hashes,
                {"route_hash_1", "route_hash_2", "route_hash_3"},
            )

    def test_resume_count_and_overlay_must_be_supplied_together(self):
        common = {
            "survey_root": Path("survey"),
            "session_root": Path("session"),
            "map_yaml": Path("map.yaml"),
            "semantic_map_id": "arena",
            "target_viewpoint_id": "survey_vp_001",
            "run_id": "mission_coverage_000",
            "coverage_leg_index": 2,
            "robot_radius_m": 0.105,
            "max_replans": 3,
            "tracking_tube_radius_m": 0.03,
        }
        with self.assertRaisesRegex(ValueError, "requires an obstacle overlay"):
            CoverageReplanCoordinator(**common, replan_count=1)
        with self.assertRaisesRegex(ValueError, "requires a positive replan_count"):
            CoverageReplanCoordinator(
                **common,
                replan_count=0,
                overlay_path=Path("overlay.json"),
            )
        with self.assertRaisesRegex(ValueError, "inside the cumulative budget"):
            CoverageReplanCoordinator(
                **common,
                replan_count=4,
                overlay_path=Path("overlay.json"),
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
                self._front_details(
                    0.23000000417232513,
                    0.20737460535019636,
                ),
            )

            self.assertEqual(update.kind, RouteUpdateKind.ADOPT)
            self.assertEqual(update.event_fields["start_egress_motion"], "reverse")
            reverse_until = update.event_fields[
                "start_egress_reverse_until_waypoint_index"
            ]
            self.assertGreaterEqual(reverse_until, 2)
            self.assertEqual(
                update.event_fields[
                    "start_egress_forward_alignment_waypoint_index"
                ],
                reverse_until + 1,
            )
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
