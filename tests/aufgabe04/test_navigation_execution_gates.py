import json
import hashlib
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.ros_runtime_config import (  # noqa: E402
    RuntimeConfig,
    resolve_runtime_config,
)
from scripts.aufgabe04.navigation.run_single_station_segment import (  # noqa: E402
    _covariance_bounded_continuity_limits,
    _execution_initial_distance_limit,
    _load_execution_route_leg,
    _simulation_odom_fallback_admission_failure,
    build_parser,
    main as run_segment_main,
)
from scripts.aufgabe04.navigation.route_uncertainty_budget import (  # noqa: E402
    PlanarCovariance,
)
from scripts.aufgabe04.navigation.mission_execution_gate import (  # noqa: E402
    load_diagnostics_snapshot,
    validate_planner_config_descriptor,
    validate_route_bundle_descriptor,
)
from scripts.aufgabe04.navigation.safety_checks import (  # noqa: E402
    catalog_start_egress_certificate,
    validate_route_diagnostics_json,
    validate_speed_limits,
)
from scripts.aufgabe04.navigation.waypoint_csv import load_route_leg  # noqa: E402


ROUTE_HEADER = (
    "leg_index,point_index,grid_x,grid_y,world_x_m,world_y_m,"
    "segment_length_m,cumulative_length_m\n"
)


def write_route(path, rows):
    path.write_text(ROUTE_HEADER + "\n".join(rows) + "\n")


def write_diagnostics(path, *, status="ok", failure=None, count=2, length=0.5):
    path.write_text(
        json.dumps(
            {
                "metadata": {},
                "legs": [
                    {
                        "diagnostics": {"status": status, "route_length_m": length},
                        "failure": failure,
                        "route_length_m": length,
                        "route_point_count": count,
                    }
                ]
            }
        )
    )


class WaypointCsvTest(unittest.TestCase):
    def test_loaded_leg_keeps_digest_of_the_exact_parsed_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            route_csv = Path(tmpdir) / "route.csv"
            write_route(
                route_csv,
                [
                    "0,0,0,0,0.0,0.0,0.0,0.0",
                    "0,1,1,0,0.10,0.0,0.10,0.10",
                ],
            )
            original = route_csv.read_bytes()
            leg = load_route_leg(route_csv, 0)
            write_route(
                route_csv,
                [
                    "0,0,0,0,5.0,5.0,0.0,0.0",
                    "0,1,1,0,5.10,5.0,0.10,0.10",
                ],
            )

        import hashlib

        self.assertEqual(leg.source_sha256, hashlib.sha256(original).hexdigest())
        self.assertEqual(leg.source_waypoint_count, 2)
        self.assertEqual(leg.raw_waypoints[0].pose.x_m, 0.0)

    def test_loads_selected_leg_and_thins_deterministically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            route_csv = Path(tmpdir) / "route.csv"
            write_route(
                route_csv,
                [
                    "0,0,0,0,0.0,0.0,0.0,0.0",
                    "0,1,1,0,0.05,0.0,0.05,0.05",
                    "0,2,2,0,0.10,0.0,0.05,0.10",
                    "0,3,3,0,0.20,0.0,0.10,0.20",
                ],
            )

            leg = load_route_leg(route_csv, 0, thinning_min_spacing_m=0.11)

        self.assertEqual(len(leg.raw_waypoints), 4)
        self.assertEqual([wp.point_index for wp in leg.executable_waypoints], [0, 3])

    def test_rejects_missing_header_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            route_csv = Path(tmpdir) / "route.csv"
            route_csv.write_text("leg_index,point_index\n0,0\n")

            with self.assertRaisesRegex(ValueError, "missing columns"):
                load_route_leg(route_csv, 0)

    def test_rejects_non_contiguous_points(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            route_csv = Path(tmpdir) / "route.csv"
            write_route(
                route_csv,
                [
                    "0,0,0,0,0.0,0.0,0.0,0.0",
                    "0,2,2,0,0.10,0.0,0.10,0.10",
                ],
            )

            with self.assertRaisesRegex(ValueError, "contiguous"):
                load_route_leg(route_csv, 0)

    def test_rejects_nan_coordinate_and_missing_leg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            route_csv = Path(tmpdir) / "route.csv"
            write_route(route_csv, ["0,0,0,0,nan,0.0,0.0,0.0"])

            with self.assertRaisesRegex(ValueError, "finite"):
                load_route_leg(route_csv, 0, require_motion=False)
            with self.assertRaisesRegex(ValueError, "not found"):
                load_route_leg(route_csv, 5, require_motion=False)

    def test_rejects_zero_length_motion_but_allows_noop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            route_csv = Path(tmpdir) / "route.csv"
            write_route(route_csv, ["0,0,0,0,0.0,0.0,0.0,0.0"])

            with self.assertRaisesRegex(ValueError, "fewer than two"):
                load_route_leg(route_csv, 0)
            leg = load_route_leg(route_csv, 0, require_motion=False)

        self.assertEqual(leg.route_length_m, 0.0)

    def test_protected_corridor_points_survive_thinning_and_provenance_loads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            route_csv = Path(tmpdir) / "route.csv"
            route_csv.write_text(
                "leg_index,point_index,grid_x,grid_y,world_x_m,world_y_m,yaw_rad,"
                "segment_length_m,cumulative_length_m,protected,simulation_only,"
                "route_kind,stream_id,route_revision,target_revision,manifest_path\n"
                "0,0,0,0,0.0,0.0,,0.0,0.0,false,true,synchronized_viewpoint,s1,4,2,route.manifest.json\n"
                "0,1,1,0,0.05,0.0,,0.05,0.05,true,true,synchronized_viewpoint,s1,4,2,route.manifest.json\n"
                "0,2,2,0,0.10,0.0,0.0,0.05,0.10,true,true,synchronized_viewpoint,s1,4,2,route.manifest.json\n"
            )

            leg = load_route_leg(route_csv, 0, thinning_min_spacing_m=0.20)

        self.assertEqual([waypoint.point_index for waypoint in leg.executable_waypoints], [0, 1, 2])
        self.assertTrue(leg.simulation_only)
        self.assertEqual(leg.route_kind, "synchronized_viewpoint")
        self.assertEqual(leg.stream_id, "s1")
        self.assertEqual(leg.route_revision, 4)
        self.assertEqual(leg.target_revision, 2)
        self.assertEqual(leg.manifest_path, Path("route.manifest.json"))


class DiagnosticsGateTest(unittest.TestCase):
    def test_snapshot_hash_and_validation_use_the_same_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostics = Path(tmpdir) / "diagnostics.json"
            write_diagnostics(diagnostics)
            original = diagnostics.read_bytes()
            snapshot = load_diagnostics_snapshot(diagnostics)
            diagnostics.write_text("{changed and invalid")

            status = validate_route_diagnostics_json(
                diagnostics,
                0,
                csv_point_count=2,
                diagnostics_payload=snapshot.payload,
            )

        self.assertTrue(status.ok)
        self.assertEqual(snapshot.sha256, hashlib.sha256(original).hexdigest())
        self.assertTrue(snapshot.source_path.is_absolute())
        mutable_view = snapshot.payload
        mutable_view["metadata"]["route_purpose"] = "tampered"
        self.assertNotIn("route_purpose", snapshot.metadata)

    def test_snapshot_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostics = Path(tmpdir) / "diagnostics.json"
            diagnostics.write_text(
                '{"metadata":{},"metadata":{},"legs":[]}'
            )

            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_diagnostics_snapshot(diagnostics)

            diagnostics.write_text('{"metadata":{},"legs":[],"cost":NaN}')
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_diagnostics_snapshot(diagnostics)

    def test_execution_descriptors_reject_unknown_fields(self):
        digest = "a" * 64
        route_bundle = {
            "schema_version": 1,
            "artifact_kind": "route_bundle",
            "route_csv_sha256": digest,
            "diagnostics_sha256": digest,
            "visit_order_sha256": digest,
            "required_edge_costs_sha256": digest,
            "catalog_snapshot_sha256": digest,
            "route_certificate_sha256": digest,
        }
        planner_config = {
            "schema_version": 1,
            "artifact_kind": "planner_config",
            "route_purpose": "logistics",
            "start_pose": {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
            "robot_radius_m": 0.105,
            "tracking_margin_m": 0.03,
            "collision_margin_m": 0.02,
            "inflation_radius_m": 0.135,
            "corridor_sample_spacing_m": 0.05,
            "lidar_stop_distance_m": 0.18,
            "scan_origin_to_base_offset_m": 0.0,
            "lidar_clearance_margin_m": 0.02,
            "arena_bounds": {
                "length_m": 3.9,
                "width_m": 1.898,
                "center_x_m": 0.0,
                "center_y_m": 0.0,
                "yaw_deg": 0.0,
                "margin_m": 0.0,
            },
            "arena_boundary_overlay": True,
            "command_owner": "/tb3/follower",
            "algorithm": "fixed_task_order_a_star",
            "max_task_snapshot_age_sec": 30.0,
            "max_task_future_skew_sec": 2.0,
        }

        validate_route_bundle_descriptor(route_bundle)
        validate_planner_config_descriptor(planner_config)
        validate_planner_config_descriptor(
            {**planner_config, "scan_origin_to_base_offset_m": -0.04}
        )
        with self.assertRaisesRegex(ValueError, "overlay"):
            validate_planner_config_descriptor(
                {**planner_config, "arena_boundary_overlay": False}
            )
        with self.assertRaisesRegex(ValueError, "arena margin"):
            validate_planner_config_descriptor(
                {
                    **planner_config,
                    "arena_bounds": {
                        **planner_config["arena_bounds"],
                        "margin_m": 1.0,
                    },
                }
            )
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_route_bundle_descriptor({**route_bundle, "extra": True})
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_planner_config_descriptor({**planner_config, "extra": True})

    def test_accepts_matching_ok_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostics = Path(tmpdir) / "diagnostics.json"
            write_diagnostics(diagnostics)

            status = validate_route_diagnostics_json(diagnostics, 0, csv_point_count=2)

        self.assertTrue(status.ok)

    def test_rejects_boolean_numeric_evidence_and_permissive_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostics = Path(tmpdir) / "diagnostics.json"
            diagnostics.write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "legs": [
                            {
                                "diagnostics": {"status": "ok"},
                                "failure": None,
                                "route_length_m": True,
                                "route_point_count": True,
                            }
                        ],
                    }
                )
            )
            boolean_status = validate_route_diagnostics_json(
                diagnostics,
                0,
                csv_point_count=1,
            )
            diagnostics.write_text(
                '{"metadata":{},"metadata":{},"legs":[]}'
            )
            duplicate_status = validate_route_diagnostics_json(
                diagnostics,
                0,
                csv_point_count=1,
            )

        self.assertFalse(boolean_status.ok)
        self.assertTrue(
            any("route_point_count" in item for item in boolean_status.failures)
        )
        self.assertTrue(
            any("route_length_m" in item for item in boolean_status.failures)
        )
        self.assertFalse(duplicate_status.ok)
        self.assertTrue(
            any("duplicate" in item for item in duplicate_status.failures)
        )

    def test_rejects_failed_or_mismatched_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostics = Path(tmpdir) / "diagnostics.json"
            write_diagnostics(diagnostics, status="failed", failure={"reason": "blocked"}, count=3)

            status = validate_route_diagnostics_json(diagnostics, 0, csv_point_count=2)

        self.assertFalse(status.ok)
        self.assertGreaterEqual(len(status.failures), 3)

    def test_rejects_zero_length_motion_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostics = Path(tmpdir) / "diagnostics.json"
            write_diagnostics(diagnostics, count=1, length=0.0)

            status = validate_route_diagnostics_json(diagnostics, 0, csv_point_count=1)
            noop_status = validate_route_diagnostics_json(
                diagnostics,
                0,
                csv_point_count=1,
                require_motion=False,
            )

        self.assertFalse(status.ok)
        self.assertTrue(noop_status.ok)

    def test_rejects_unsafe_speed_caps(self):
        self.assertTrue(validate_speed_limits(0.05, 0.10).ok)
        self.assertFalse(validate_speed_limits(0.20, 0.10).ok)
        self.assertFalse(validate_speed_limits(0.05, math.inf).ok)


class SegmentRunnerCliGateTest(unittest.TestCase):
    def simulation_odom_admission(
        self,
        *,
        requested=True,
        allow_sim_time=True,
        resolved_use_sim_time=True,
        localization_source="tf",
        map_frame="odom",
        odom_frame="odom",
        simulation_only=True,
        route_kind="catalog_face_approach",
        route_purpose="survey",
        authoritative_dynamic_route=False,
        allow_unbound_survey=True,
    ):
        return _simulation_odom_fallback_admission_failure(
            SimpleNamespace(
                allow_simulation_odom_after_stale_tf=requested,
                allow_sim_time=allow_sim_time,
                allow_unbound_survey_simulation_route=(
                    allow_unbound_survey
                ),
            ),
            SimpleNamespace(
                use_sim_time=resolved_use_sim_time,
                localization_source=localization_source,
                map_frame=map_frame,
                odom_frame=odom_frame,
            ),
            SimpleNamespace(
                simulation_only=simulation_only,
                route_kind=route_kind,
            ),
            route_purpose=route_purpose,
            authoritative_dynamic_route=authoritative_dynamic_route,
        )

    def test_simulation_odom_fallback_admits_only_survey_simulation_contracts(
        self,
    ):
        self.assertEqual(self.simulation_odom_admission(), "")
        self.assertEqual(
            self.simulation_odom_admission(
                route_kind="viewpoint_sampling",
                route_purpose="",
                authoritative_dynamic_route=True,
                allow_unbound_survey=False,
            ),
            "",
        )

        rejected = {
            "real_time": {"allow_sim_time": False},
            "unresolved_sim_time": {"resolved_use_sim_time": False},
            "amcl": {"localization_source": "amcl"},
            "map_route": {"map_frame": "map"},
            "non_simulation_leg": {"simulation_only": False},
            "logistics": {"route_purpose": "logistics"},
            "unknown_static": {"route_purpose": ""},
            "unbound_static": {"allow_unbound_survey": False},
            "legacy": {
                "route_kind": "legacy_simulation_waypoint",
                "route_purpose": "",
            },
            "non_authoritative_dynamic": {
                "route_kind": "viewpoint_sampling",
                "route_purpose": "",
                "authoritative_dynamic_route": False,
            },
            "dynamic_logistics": {
                "route_kind": "viewpoint_sampling",
                "route_purpose": "logistics",
                "authoritative_dynamic_route": True,
            },
        }
        for name, overrides in rejected.items():
            with self.subTest(name=name):
                self.assertTrue(
                    self.simulation_odom_admission(**overrides)
                )

    def test_simulation_odom_fallback_is_off_by_default(self):
        args = build_parser().parse_args(["--leg-index", "0"])

        self.assertFalse(args.allow_simulation_odom_after_stale_tf)

    def test_runtime_nomotion_refresh_has_separate_bounded_defaults(self):
        args = build_parser().parse_args(["--leg-index", "0"])

        self.assertEqual(args.nomotion_update_service, "/request_nomotion_update")
        self.assertEqual(args.nomotion_update_timeout_sec, 15.0)
        self.assertEqual(
            args.runtime_nomotion_update_service,
            "request_nomotion_update",
        )
        self.assertEqual(args.runtime_nomotion_update_timeout_sec, 2.0)
        self.assertEqual(args.max_localization_tf_future_sec, 1.1)
        self.assertEqual(args.max_stationary_amcl_position_std_m, 0.015)
        self.assertEqual(args.max_stationary_amcl_yaw_std_rad, 0.03)
        self.assertEqual(args.execution_pose_frame, "map")
        self.assertEqual(args.uncertainty_sigma_multiplier, 1.0)
        self.assertEqual(args.uncertainty_odom_drift_bound_m, 0.02)
        self.assertEqual(args.max_map_odom_translation_drift_m, 0.15)

        configured = build_parser().parse_args(
            [
                "--leg-index",
                "0",
                "--runtime-nomotion-update-service",
                "amcl/request_nomotion_update",
                "--runtime-nomotion-update-timeout-sec",
                "1.25",
                "--max-localization-tf-future-sec",
                "0.75",
            ]
        )
        self.assertEqual(
            configured.runtime_nomotion_update_service,
            "amcl/request_nomotion_update",
        )
        self.assertEqual(configured.runtime_nomotion_update_timeout_sec, 1.25)
        self.assertEqual(configured.max_localization_tf_future_sec, 0.75)

    def test_live_map_odom_limits_reuse_covariance_budget_with_hard_caps(self):
        translation_m, yaw_rad = _covariance_bounded_continuity_limits(
            PlanarCovariance(0.094**2, 0.0, 0.094**2),
            heading_sigma_rad=0.03,
            sigma_multiplier=1.0,
            translation_hard_cap_m=0.15,
            yaw_hard_cap_rad=0.10,
        )
        self.assertAlmostEqual(translation_m, 0.094)
        self.assertAlmostEqual(yaw_rad, 0.03)

        capped = _covariance_bounded_continuity_limits(
            PlanarCovariance(0.094**2, 0.0, 0.094**2),
            heading_sigma_rad=0.06,
            sigma_multiplier=2.0,
            translation_hard_cap_m=0.15,
            yaw_hard_cap_rad=0.10,
        )
        self.assertEqual(capped, (0.15, 0.10))

    def test_runtime_nomotion_refresh_timeout_must_be_in_bounded_interval(self):
        for value in ("0", "2.01", "nan", "inf"):
            with self.subTest(value=value), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    run_segment_main(
                        [
                            "--leg-index",
                            "0",
                            "--runtime-nomotion-update-timeout-sec",
                            value,
                        ]
                    )

            self.assertEqual(raised.exception.code, 2)

    def test_amcl_position_admission_bounds_cannot_exceed_half_route_tube(self):
        for flag in (
            "--max-stationary-amcl-position-spread-m",
            "--max-stationary-amcl-position-std-m",
        ):
            with self.subTest(flag=flag), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    run_segment_main(
                        [
                            "--leg-index",
                            "0",
                            flag,
                            "0.0151",
                            "--certified-route-tube-radius-m",
                            "0.03",
                        ]
                    )

            self.assertEqual(raised.exception.code, 2)

    def test_odom_execution_requires_complete_uncertainty_artifacts(self):
        stderr = StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            run_segment_main(
                [
                    "--leg-index",
                    "0",
                    "--execution-pose-frame",
                    "odom",
                    "--dry-run",
                ]
            )

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("odom execution requires", stderr.getvalue())
        self.assertIn(
            "--localization-branch-proof-id", stderr.getvalue()
        )

    def test_odom_execution_disallows_simulation_stale_tf_fallback(self):
        stderr = StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            run_segment_main(
                [
                    "--leg-index",
                    "0",
                    "--execution-pose-frame",
                    "odom",
                    "--odom-execution-certificate-json",
                    "odom_certificate.json",
                    "--uncertainty-budget-json",
                    "uncertainty.json",
                    "--uncertainty-map-yaml",
                    "map.yaml",
                    "--localization-branch-proof-id",
                    "known_start_marker_20260807",
                    "--uncertainty-robot-radius-m",
                    "0.105",
                    "--allow-simulation-odom-after-stale-tf",
                    "--dry-run",
                ]
            )

        self.assertEqual(raised.exception.code, 2)
        self.assertIn(
            "may not enable the simulation stale-TF fallback",
            stderr.getvalue(),
        )

    def test_odom_execution_delegates_large_covariance_to_route_budget(self):
        stderr = StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            run_segment_main(
                [
                    "--route-csv",
                    "missing_route.csv",
                    "--diagnostics-json",
                    "missing_diagnostics.json",
                    "--leg-index",
                    "0",
                    "--execution-pose-frame",
                    "odom",
                    "--odom-execution-certificate-json",
                    "odom_certificate.json",
                    "--uncertainty-budget-json",
                    "uncertainty.json",
                    "--uncertainty-map-yaml",
                    "map.yaml",
                    "--localization-branch-proof-id",
                    "known_start_marker_20260807",
                    "--uncertainty-robot-radius-m",
                    "0.105",
                    "--max-stationary-amcl-position-std-m",
                    "0.30",
                    "--dry-run",
                ]
            )

        self.assertEqual(raised.exception.code, 2)
        self.assertNotIn("half the certified route tube", stderr.getvalue())
        self.assertIn("route validation failed", stderr.getvalue())

    def test_missing_route_kind_is_rejected_before_ros_preflight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            route_csv = root / "route.csv"
            diagnostics = root / "diagnostics.json"
            write_route(
                route_csv,
                [
                    "0,0,0,0,0.0,0.0,0.0,0.0",
                    "0,1,1,0,0.10,0.0,0.10,0.10",
                ],
            )
            write_diagnostics(diagnostics, length=0.1)
            with patch(
                "scripts.aufgabe04.navigation.run_single_station_segment.run_ros_preflight"
            ) as preflight, redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    run_segment_main(
                        [
                            "--route-csv", str(route_csv),
                            "--diagnostics-json", str(diagnostics),
                            "--semantic-log", str(root / "events.jsonl"),
                            "--results-csv", str(root / "results.csv"),
                            "--leg-index", "0",
                            "--dry-run",
                        ]
                    )

        self.assertEqual(raised.exception.code, 2)
        preflight.assert_not_called()

    def test_static_route_without_arena_overlay_is_rejected_before_preflight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            route_csv = root / "route.csv"
            route_csv.write_text(
                "leg_index,point_index,world_x_m,world_y_m,yaw_rad,"
                "cumulative_length_m,protected,corridor,simulation_only,"
                "route_kind,source_arrival_id,target_arrival_id,catalog_sha256\n"
                f"0,0,0.0,0.0,0.0,0.0,false,false,true,"
                f"catalog_face_approach,mission_start,A::face_0,{'a' * 64}\n"
                f"0,1,0.1,0.0,0.0,0.1,false,false,true,"
                f"catalog_face_approach,mission_start,A::face_0,{'a' * 64}\n"
            )
            diagnostics = root / "diagnostics.json"
            diagnostics.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "route_purpose": "survey",
                            "planning_frame": "map",
                        },
                        "legs": [
                            {
                                "diagnostics": {
                                    "status": "ok",
                                    "route_length_m": 0.1,
                                },
                                "failure": None,
                                "route_length_m": 0.1,
                                "route_point_count": 2,
                            }
                        ],
                    }
                )
            )
            stderr = StringIO()
            with patch(
                "scripts.aufgabe04.navigation.run_single_station_segment."
                "run_ros_preflight"
            ) as preflight, redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    run_segment_main(
                        [
                            "--route-csv",
                            str(route_csv),
                            "--diagnostics-json",
                            str(diagnostics),
                            "--semantic-log",
                            str(root / "events.jsonl"),
                            "--results-csv",
                            str(root / "results.csv"),
                            "--leg-index",
                            "0",
                            "--allow-sim-time",
                            "--allow-unbound-survey-simulation-route",
                            "--dry-run",
                        ]
                    )

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("arena_boundary_overlay must be true", stderr.getvalue())
        preflight.assert_not_called()

    def test_exact_e2e_006_leg2_activates_certified_vertex_one_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            route_csv = Path(tmpdir) / "route.csv"
            route_csv.write_text(
                "leg_index,point_index,world_x_m,world_y_m,yaw_rad,"
                "cumulative_length_m,protected,corridor,simulation_only,route_kind,"
                "source_arrival_id,target_arrival_id,catalog_sha256\n"
                f"2,0,-0.18491188596302915,-0.200843551718869,,0.0,false,false,true,"
                f"catalog_face_approach,station_A::face_b,station_B::face_b,{'a' * 64}\n"
                f"2,1,-0.19499999999999984,-0.11499999999999977,,0.08643428380297428,"
                f"false,false,true,catalog_face_approach,station_A::face_b,station_B::face_b,{'a' * 64}\n"
                f"2,2,-0.6449999999999996,-0.11499999999999977,,0.536434283802974,"
                f"false,false,true,catalog_face_approach,station_A::face_b,station_B::face_b,{'a' * 64}\n"
            )
            diagnostics = Path(tmpdir) / "diagnostics.json"
            diagnostics.write_text(
                json.dumps(
                    {
                        "legs": [{}, {}, {
                            "diagnostics": {
                                "fixed_arrival": {
                                    "start_join_clearance_m": 0.03508711403697043,
                                },
                            },
                            "non_target_keepout_overlay": {
                                "rasterized_cell_count": 218,
                                "blocked_cell_count": 217,
                                "start_cell": {"x": 52, "y": 29},
                                "start_cell_was_rasterized": True,
                                "start_cell_exempted": True,
                                "exact_start_minimum_margin_m": 0.04,
                                "cell_center_minimum_margin_m": 0.02284271247461922,
                                "start_connector_minimum_margin_m": 0.02284271247461922,
                            },
                            "non_target_stand_clearances": [{
                                "station_id": "station_A",
                                "x_m": -0.395,
                                "y_m": -0.415,
                                "radius_m": 0.26,
                                "minimum_route_clearance_m": 0.3,
                            }],
                        }],
                    }
                )
            )
            leg = load_route_leg(route_csv, 2)

            certificate = catalog_start_egress_certificate(diagnostics, leg)
            malformed = json.loads(diagnostics.read_text())
            malformed["legs"][2]["non_target_keepout_overlay"][
                "blocked_cell_count"
            ] = 218
            diagnostics.write_text(json.dumps(malformed))
            with self.assertRaisesRegex(ValueError, "remove exactly one"):
                catalog_start_egress_certificate(diagnostics, leg)

        self.assertTrue(certificate.required)
        self.assertEqual(certificate.waypoint_index, 1)
        self.assertAlmostEqual(certificate.minimum_route_clearance_m, 0.3)
        self.assertAlmostEqual(
            certificate.start_join_clearance_m,
            0.03508711403697043,
        )

    def test_exterior_anchor_egress_activates_vertex_one_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            route_csv = root / "route.csv"
            route_csv.write_text(
                "leg_index,point_index,world_x_m,world_y_m,yaw_rad,"
                "cumulative_length_m,protected,corridor,simulation_only,route_kind,"
                "source_arrival_id,target_arrival_id,catalog_sha256\n"
                f"1,0,1.285,1.025,,0.0,false,false,true,catalog_face_approach,"
                f"source::face,target::face,{'a' * 64}\n"
                f"1,1,1.325,1.025,,0.04,false,false,true,catalog_face_approach,"
                f"source::face,target::face,{'a' * 64}\n"
                f"1,2,2.0,1.025,,0.715,false,false,true,catalog_face_approach,"
                f"source::face,target::face,{'a' * 64}\n"
            )
            diagnostics = root / "diagnostics.json"
            diagnostics.write_text(
                json.dumps(
                    {
                        "legs": [
                            {},
                            {
                                "diagnostics": {
                                    "fixed_arrival": {
                                        "start_join_clearance_m": 0.01,
                                    },
                                },
                                "non_target_keepout_overlay": {
                                    "rasterized_cell_count": 120,
                                    "blocked_cell_count": 120,
                                    "start_cell": {"x": 25, "y": 20},
                                    "start_cell_was_rasterized": True,
                                    "start_cell_exempted": False,
                                    "exact_start_minimum_margin_m": 0.001,
                                    "cell_center_minimum_margin_m": -0.005,
                                    "start_connector_minimum_margin_m": -0.005,
                                    "egress_anchor": {
                                        "x_m": 1.325,
                                        "y_m": 1.025,
                                        "yaw_rad": 0.0,
                                    },
                                    "egress_anchor_cell": {"x": 26, "y": 20},
                                    "egress_cells": [
                                        {"x": 25, "y": 20},
                                        {"x": 26, "y": 20},
                                    ],
                                    "egress_connector_minimum_margin_m": 0.001,
                                    "egress_continuous_clearance_validated": True,
                                    "egress_failure_reason": None,
                                },
                                "non_target_stand_clearances": [
                                    {
                                        "station_id": "source",
                                        "x_m": 1.024,
                                        "y_m": 1.025,
                                        "radius_m": 0.26,
                                        "minimum_route_clearance_m": 0.261,
                                    }
                                ],
                            },
                        ],
                    }
                )
            )
            leg = load_route_leg(route_csv, 1)

            certificate = catalog_start_egress_certificate(
                diagnostics,
                leg,
            )
            tampered = json.loads(diagnostics.read_text())
            tampered["legs"][1]["non_target_keepout_overlay"][
                "egress_anchor"
            ]["x_m"] = 1.35
            diagnostics.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError, "waypoint 1"):
                catalog_start_egress_certificate(diagnostics, leg)

        self.assertTrue(certificate.required)
        self.assertEqual(certificate.waypoint_index, 1)
        self.assertAlmostEqual(certificate.minimum_route_clearance_m, 0.261)

    def test_static_catalog_route_tightens_unchecked_initial_join(self):
        self.assertEqual(
            _execution_initial_distance_limit(0.35, "catalog_face_approach"),
            0.15,
        )
        self.assertEqual(_execution_initial_distance_limit(0.35, ""), 0.35)

    def test_static_catalog_route_is_never_rethinned_into_unchecked_chords(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            route_csv = Path(tmpdir) / "route.csv"
            route_csv.write_text(
                "leg_index,point_index,world_x_m,world_y_m,yaw_rad,"
                "cumulative_length_m,protected,simulation_only,route_kind\n"
                "0,0,0.0,0.0,,0.0,false,true,catalog_face_approach\n"
                "0,1,0.05,0.0,,0.05,false,true,catalog_face_approach\n"
                "0,2,0.10,0.0,,0.10,false,true,catalog_face_approach\n"
                "0,3,0.20,0.0,0.0,0.20,true,true,catalog_face_approach\n"
            )

            leg = _load_execution_route_leg(
                route_csv,
                0,
                require_motion=True,
                requested_thinning_min_spacing_m=0.15,
                authoritative_dynamic_route=False,
            )

        self.assertEqual(leg.thinning_min_spacing_m, 0.0)
        self.assertEqual(
            [waypoint.point_index for waypoint in leg.executable_waypoints],
            [0, 1, 2, 3],
        )

    def test_ordinary_static_route_retains_requested_legacy_thinning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            route_csv = Path(tmpdir) / "route.csv"
            write_route(
                route_csv,
                [
                    "0,0,0,0,0.0,0.0,0.0,0.0",
                    "0,1,1,0,0.05,0.0,0.05,0.05",
                    "0,2,2,0,0.10,0.0,0.05,0.10",
                    "0,3,3,0,0.20,0.0,0.10,0.20",
                ],
            )

            leg = _load_execution_route_leg(
                route_csv,
                0,
                require_motion=True,
                requested_thinning_min_spacing_m=0.15,
                authoritative_dynamic_route=False,
            )

        self.assertEqual(leg.thinning_min_spacing_m, 0.15)
        self.assertEqual(
            [waypoint.point_index for waypoint in leg.executable_waypoints],
            [0, 3],
        )

    def test_physical_face_route_has_tighter_default_goal_tolerance(self):
        args = build_parser().parse_args(["--leg-index", "0"])

        self.assertEqual(args.goal_tolerance_m, 0.08)
        self.assertEqual(args.viewpoint_sampling_goal_tolerance_m, 0.01)
        self.assertEqual(args.physical_waypoint_tolerance_m, 0.02)
        self.assertEqual(args.physical_goal_tolerance_m, 0.03)

    def test_yes_bypass_argument_is_rejected(self):
        parser = build_parser()

        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--leg-index", "0", "--yes"])


class RuntimeConfigTest(unittest.TestCase):
    def test_namespaces_relative_topics_only(self):
        resolved = resolve_runtime_config(
            RuntimeConfig(
                namespace="robot1",
                scan_topic="scan",
                odom_topic="/odom",
                cmd_vel_topic="cmd_vel",
                amcl_topic="/amcl_pose",
            )
        )

        self.assertEqual(resolved.scan_topic, "/robot1/scan")
        self.assertEqual(resolved.odom_topic, "/odom")
        self.assertEqual(resolved.cmd_vel_topic, "/robot1/cmd_vel")
        self.assertEqual(resolved.amcl_topic, "/amcl_pose")
        self.assertEqual(resolved.base_frame, "base_footprint")


if __name__ == "__main__":
    unittest.main()
