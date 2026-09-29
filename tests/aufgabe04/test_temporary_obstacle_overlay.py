"""Offline ODOM occupancy updates, immutable projection and route consumption."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json, load_content_hashed_json
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.execution.execution_route_certificate import file_sha256
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.map_io import load_occupancy_grid_with_bundle
from scripts.aufgabe04.navigation.planning.temporary_obstacle_overlay import TemporaryObstacleMap, apply_bound_temporary_obstacles
from scripts.aufgabe04.navigation.planning.temporary_obstacle_projection import load_validated_projection
from scripts.aufgabe04.navigation.planning import temporary_obstacle_overlay as overlay_module
from tests.aufgabe04.test_detected_station_exploration import write_free_map


def capture_payload(*, start=100., ranges=(1., None), base=(0., 0., 0.), tour_id="tour", origin=None):
    source = base if origin is None else origin
    return {
        "schema_version": 1, "artifact_kind": "stored_pose_tour_scan_capture", "tour_id": tour_id,
        "odom_frame": "odom", "base_frame": "base_footprint", "scan_frame": "base_scan",
        "captured_at_unix_sec": start+.22,
        "scans": [{"stamp_sec": start+i*.1, "received_at_unix_sec": start+i*.1+.01,
            "scan_pose_stamp_sec": start+i*.1, "base_pose_stamp_sec": start+i*.1,
            "scan_pose_odom": dict(zip(("x_m", "y_m", "yaw_rad"), source)),
            "base_pose_odom": dict(zip(("x_m", "y_m", "yaw_rad"), base)),
            "angle_min": 0., "angle_increment": .02, "range_min": .10, "range_max": 10.,
            "ranges": list(ranges)} for i in range(3)],
    }


def write_capture(root, name="capture.json", **kwargs):
    path = root / name
    payload = capture_payload(**kwargs)
    write_content_hashed_json(path, payload, hash_field="tour_scan_capture_sha256")
    return path


class TemporaryObstacleOverlayTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        map_yaml = write_free_map(self.root)
        grid, bundle = load_occupancy_grid_with_bundle(map_yaml, semantic_map_id="arena", planning_frame="map")
        self.base = Costmap.from_occupancy_grid(grid)
        self.bundle = bundle
        self.map = TemporaryObstacleMap("tour", "odom", bundle.bundle_sha256)
        self.frame = CandidatePlanningFrame(Pose2D(0., 0., 0.), PlanarTransform2D(0., 0., 0.))

    def binding(self, path, frame=None):
        return {"route_purpose": "stored_pose_tour", "tour_id": "tour", "planning_frame": "map",
            "map_bundle_sha256": self.bundle.bundle_sha256, "planning_frame_admission": (frame or self.frame).to_evidence(),
            "temporary_obstacle_overlay_json": str(path), "temporary_obstacle_overlay_sha256": file_sha256(path)}

    def test_three_stationary_scans_mark_endpoints_without_robot_margin(self):
        self.map.update_from_capture(write_capture(self.root), now_sec=100.23)
        self.assertEqual([(c["x"], c["y"]) for c in self.map.active_cells(100.23)], [(20, 0)])
        output = self.map.write_projection(self.root / "overlay.json", self.frame, now_sec=100.24)
        projected = apply_bound_temporary_obstacles(self.base, self.binding(output))
        self.assertEqual(projected.cells, self.base.cells)
        new = projected.blocked_cells-self.base.blocked_cells
        self.assertTrue(new)
        self.assertTrue(all(abs(projected.grid_to_world(cell).x_m-1.025) <= .12 for cell in new))
        self.assertNotIn(projected.world_to_grid(.75, 0.), new)

    def test_only_two_distinct_finite_rays_clear_previous_cells(self):
        self.map.update_from_capture(write_capture(self.root), now_sec=100.23)
        one = capture_payload(start=101., ranges=(None, None))
        one["scans"][0]["ranges"][0] = 2.
        one_path = self.root / "one.json"
        write_content_hashed_json(one_path, one, hash_field="tour_scan_capture_sha256")
        self.map.update_from_capture(one_path, now_sec=101.23)
        self.assertIn((20, 0), [(c["x"], c["y"]) for c in self.map.active_cells(101.23)])
        two = capture_payload(start=102., ranges=(2., None))
        two["scans"][2]["ranges"][0] = None
        path = self.root / "two.json"
        write_content_hashed_json(path, two, hash_field="tour_scan_capture_sha256")
        self.map.update_from_capture(path, now_sec=102.23)
        self.assertNotIn((20, 0), [(c["x"], c["y"]) for c in self.map.active_cells(102.23)])

    def test_none_and_out_of_range_rays_never_clear_and_cap_has_no_endpoint(self):
        self.map.update_from_capture(write_capture(self.root), now_sec=100.23)
        self.map.update_from_capture(write_capture(self.root, "invalid.json", start=101., ranges=(None, 15.)), now_sec=101.23)
        self.assertEqual(len(self.map.active_cells(101.23)), 1)
        self.map.update_from_capture(write_capture(self.root, "capped.json", start=102., ranges=(5., None)), now_sec=102.23)
        self.assertEqual(self.map.active_cells(102.23), [])

    def test_expiry_affects_next_projection_but_never_mutates_sealed_route(self):
        self.map.update_from_capture(write_capture(self.root), now_sec=100.23)
        old = self.map.write_projection(self.root / "old.json", self.frame, now_sec=100.24)
        old_bytes = old.read_bytes()
        fresh = self.map.write_projection(self.root / "expired.json", self.frame, now_sec=131.)
        self.assertEqual(load_validated_projection(fresh)["active_odom_cells"], [])
        self.assertEqual(old.read_bytes(), old_bytes)
        self.assertTrue(apply_bound_temporary_obstacles(self.base, self.binding(old)).blocked_cells-self.base.blocked_cells)

    def test_reprojection_rotates_odom_square_and_preserves_static_sources(self):
        self.map.update_from_capture(write_capture(self.root), now_sec=100.23)
        frame = CandidatePlanningFrame(Pose2D(0., 0., 0.), PlanarTransform2D(.2, -.4, math.pi/4))
        output = self.map.write_projection(self.root / "rotated.json", frame, now_sec=100.24)
        projected = apply_bound_temporary_obstacles(self.base, self.binding(output, frame))
        for dx, dy in ((0.,0.), (.05,0.), (.05,.05), (0.,.05), (.025,.025)):
            x = .2+math.cos(math.pi/4)*(1.+dx)-math.sin(math.pi/4)*dy
            y = -.4+math.sin(math.pi/4)*(1.+dx)+math.cos(math.pi/4)*dy
            self.assertTrue(projected.is_blocked(projected.world_to_grid(x,y)))
        for cell in self.base.blocked_cells:
            self.assertEqual(projected.cell_sources[cell], self.base.cell_sources[cell])

    def test_changed_sources_or_fabricated_cells_are_rejected(self):
        source = write_capture(self.root)
        self.map.update_from_capture(source, now_sec=100.23)
        output = self.map.write_projection(self.root / "overlay.json", self.frame, now_sec=100.24)
        payload = load_content_hashed_json(output, hash_field="temporary_obstacle_overlay_sha256")
        payload["active_odom_cells"][0]["x"] += 1
        tampered = self.root / "tampered.json"
        write_content_hashed_json(tampered, payload, hash_field="temporary_obstacle_overlay_sha256")
        with self.assertRaisesRegex(ValueError, "replayed"):
            apply_bound_temporary_obstacles(self.base, self.binding(tampered))
        source.write_text("changed")
        with self.assertRaisesRegex(ValueError, "source capture hash"):
            apply_bound_temporary_obstacles(self.base, self.binding(output))

    def test_wrong_tour_frame_and_scope_are_rejected(self):
        output = self.map.write_projection(self.root / "empty.json", self.frame, now_sec=100.)
        for key, bad in (("tour_id", "other"), ("route_purpose", "return_to_start"), ("planning_frame", "other")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                apply_bound_temporary_obstacles(self.base, {**self.binding(output), key: bad})
        changed = replace(self.frame, map_from_odom=PlanarTransform2D(.1, 0., 0.))
        with self.assertRaisesRegex(ValueError, "planning frame"):
            apply_bound_temporary_obstacles(self.base, self.binding(output, changed))

    def test_bad_stamped_stationary_cohorts_fail_without_updating(self):
        mutations = (
            lambda p: p["scans"][1].update(stamp_sec=100.),
            lambda p: p["scans"][1].update(scan_pose_stamp_sec=99.),
            lambda p: p["scans"][1]["base_pose_odom"].update(x_m=.02),
            lambda p: p["scans"][1].update(received_at_unix_sec=101.),
            lambda p: p.update(odom_frame="other"),
            lambda p: p["scans"][1]["scan_pose_odom"].update(y_m=.1),
        )
        for index, change in enumerate(mutations):
            data = capture_payload()
            change(data)
            path = self.root / f"bad{index}.json"
            write_content_hashed_json(path, data, hash_field="tour_scan_capture_sha256")
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.map.update_from_capture(path, now_sec=100.23)
        self.assertEqual(self.map.active_cells(100.23), [])

    def test_odom_discontinuity_and_reused_capture_rejected(self):
        source = write_capture(self.root)
        self.map.update_from_capture(source, now_sec=100.23)
        with self.assertRaises(ValueError):
            self.map.update_from_capture(source, now_sec=100.25)
        jump = write_capture(self.root, "jump.json", start=101., base=(2., 0., 0.))
        with self.assertRaisesRegex(ValueError, "odom discontinuity"):
            self.map.update_from_capture(jump, now_sec=101.23)

    def test_capture_bound_covers_expanded_tours_and_is_shared_with_replay(self):
        self.assertGreaterEqual(overlay_module.MAX_TOUR_CAPTURE_SOURCES, 6*(4*100+2+10+1))
        self.map.update_from_capture(write_capture(self.root), now_sec=100.23)
        self.map.update_from_capture(write_capture(self.root, "second.json", start=101.), now_sec=101.23)
        output = self.map.write_projection(self.root / "overlay.json", self.frame, now_sec=101.24)
        with patch.object(overlay_module, "MAX_TOUR_CAPTURE_SOURCES", 2):
            third = write_capture(self.root, "third.json", start=102.)
            with self.assertRaisesRegex(ValueError, "bounded capture storage"):
                self.map.update_from_capture(third, now_sec=102.23)
            self.assertEqual(len(load_validated_projection(output)["capture_sources"]), 2)
        with patch.object(overlay_module, "MAX_TOUR_CAPTURE_SOURCES", 1):
            with self.assertRaisesRegex(ValueError, "bounded capture storage"):
                load_validated_projection(output)


if __name__ == "__main__":
    unittest.main()
