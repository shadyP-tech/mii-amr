from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json
from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.missions.startup_route_uncertainty_selection import (
    STARTUP_ROUTE_UNCERTAINTY_SELECTION_HASH_FIELD,
    StartupRouteUncertaintySelectionRejected,
    load_startup_route_uncertainty_selector,
)
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.map_io import (
    CELL_FREE,
    MapMetadata,
    OccupancyGrid,
)


def free_costmap(arena_width_m: float = 2.0) -> Costmap:
    grid = OccupancyGrid(
        metadata=MapMetadata(
            yaml_path=Path("map.yaml"),
            image_path=Path("map.pgm"),
            resolution=0.05,
            origin=(-2.0, -1.0, 0.0),
            negate=0,
            occupied_thresh=0.65,
            free_thresh=0.20,
            mode="trinary",
        ),
        width=80,
        height=40,
        cells=tuple(tuple([CELL_FREE] * 80) for _ in range(40)),
    )
    return Costmap.from_occupancy_grid(grid).with_arena_bounds(
        ArenaBounds(length_m=3.0, width_m=arena_width_m)
    )


def fake_leg(viewpoint_id: str, poses: tuple[Pose2D, ...]):
    return SimpleNamespace(
        viewpoint=SimpleNamespace(viewpoint_id=viewpoint_id),
        route_result=SimpleNamespace(
            route=SimpleNamespace(
                points=tuple(SimpleNamespace(pose=pose) for pose in poses)
            )
        ),
    )


def preflight_payload(start: Pose2D) -> dict[str, object]:
    covariance = [0.0] * 36
    return {
        "ok": True,
        "failures": [],
        "route_pose": {
            "frame_id": "map",
            "child_frame_id": "base_footprint",
            "x_m": start.x_m,
            "y_m": start.y_m,
            "yaw_rad": start.yaw_rad,
        },
        "stationary_amcl_samples": [
            {"covariance": list(covariance)} for _ in range(5)
        ],
    }


class StartupRouteUncertaintySelectionTest(unittest.TestCase):
    def selector(self, root: Path, start: Pose2D):
        preflight = root / "preflight.json"
        preflight.write_text(
            json.dumps(preflight_payload(start), sort_keys=True) + "\n"
        )
        evidence = root / "startup_selection.json"
        return (
            load_startup_route_uncertainty_selector(
                preflight_json=preflight,
                evidence_json=evidence,
                expected_start=start,
                planning_frame="map",
                robot_radius_m=0.105,
                collision_margin_m=0.02,
                tracking_tube_radius_m=0.03,
                odom_drift_bound_m=0.02,
                braking_latency_distance_m=0.015,
                sigma_multiplier=2.0,
                clearance_sample_spacing_m=0.005,
            ),
            evidence,
        )

    def test_persists_all_options_and_selects_the_larger_exact_margin(self):
        start = Pose2D(-1.0, 0.0, 0.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            selector, evidence = self.selector(Path(tmpdir), start)
            selected_id, compact = selector(
                free_costmap(),
                (
                    fake_leg(
                        "near_wall_shorter",
                        (
                            start,
                            Pose2D(-0.9, 0.75, 0.0),
                            Pose2D(-0.5, 0.75, 0.0),
                        ),
                    ),
                    fake_leg(
                        "center_longer",
                        (start, Pose2D(1.0, 0.0, 0.0)),
                    ),
                ),
            )
            payload = load_content_hashed_json(
                evidence,
                hash_field=STARTUP_ROUTE_UNCERTAINTY_SELECTION_HASH_FIELD,
            )

        self.assertEqual(selected_id, "center_longer")
        self.assertEqual(compact["selected_viewpoint_id"], "center_longer")
        self.assertFalse(compact["motion_authorized"])
        self.assertEqual(
            payload["selection"]["decision"]["selected_option_id"],
            "center_longer",
        )
        self.assertEqual(len(payload["selection"]["options"]), 2)
        self.assertFalse(payload["motion_published"])
        self.assertFalse(payload["target_committed_before_selection"])

    def test_all_rejected_options_are_persisted_before_typed_failure(self):
        start = Pose2D(-1.0, 0.0, 0.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            selector, evidence = self.selector(Path(tmpdir), start)
            with self.assertRaises(
                StartupRouteUncertaintySelectionRejected
            ) as captured:
                selector(
                    free_costmap(arena_width_m=0.30),
                    (
                        fake_leg("first", (start, Pose2D(0.0, 0.0, 0.0))),
                        fake_leg("second", (start, Pose2D(0.5, 0.0, 0.0))),
                    ),
                )
            payload = load_content_hashed_json(
                evidence,
                hash_field=STARTUP_ROUTE_UNCERTAINTY_SELECTION_HASH_FIELD,
            )
            stored_hash = compact_hash_from_file(evidence)

        self.assertEqual(captured.exception.evidence_path, evidence)
        self.assertEqual(
            captured.exception.evidence_sha256,
            stored_hash,
        )
        self.assertFalse(payload["selection"]["decision"]["ready"])
        self.assertIsNone(
            payload["selection"]["decision"]["selected_option_id"]
        )
        self.assertTrue(
            all(
                not option["accepted"]
                for option in payload["selection"]["options"]
            )
        )

    def test_preflight_pose_must_match_the_admitted_planning_start(self):
        start = Pose2D(-1.0, 0.0, 0.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            preflight = root / "preflight.json"
            preflight.write_text(
                json.dumps(
                    preflight_payload(Pose2D(-0.9, 0.0, 0.0)),
                    sort_keys=True,
                )
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_startup_route_uncertainty_selector(
                    preflight_json=preflight,
                    evidence_json=root / "selection.json",
                    expected_start=start,
                    planning_frame="map",
                    robot_radius_m=0.105,
                    collision_margin_m=0.02,
                    tracking_tube_radius_m=0.03,
                    odom_drift_bound_m=0.02,
                    braking_latency_distance_m=0.015,
                    sigma_multiplier=2.0,
                    clearance_sample_spacing_m=0.005,
                )


def compact_hash_from_file(path: Path) -> str:
    payload = json.loads(path.read_text())
    return str(payload[STARTUP_ROUTE_UNCERTAINTY_SELECTION_HASH_FIELD])


if __name__ == "__main__":
    unittest.main()
