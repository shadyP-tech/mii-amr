import tempfile
import unittest
from pathlib import Path

from scripts.aufgabe04.navigation.map_io import (
    FrozenMapBundleError,
    freeze_map_bundle,
    frozen_map_bundle_payload,
    load_frozen_map_bundle,
    load_occupancy_grid_with_bundle,
    write_frozen_map_bundle,
)


class FrozenMapBundleTest(unittest.TestCase):
    def _write_map(self, root: Path, pixels: str = "0 255\n255 0\n") -> Path:
        (root / "map.pgm").write_text(f"P2\n2 2\n255\n{pixels}")
        yaml_path = root / "map.yaml"
        yaml_path.write_text(
            "image: map.pgm\n"
            "resolution: 0.05\n"
            "origin: [-1.0, -2.0, 0.0]\n"
            "negate: 0\n"
            "occupied_thresh: 0.65\n"
            "free_thresh: 0.196\n"
            "mode: trinary\n"
        )
        return yaml_path

    def test_grid_and_descriptor_come_from_same_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_map(Path(tmpdir))
            grid, bundle = load_occupancy_grid_with_bundle(
                path,
                semantic_map_id="temporary_arena_001",
                planning_frame="map",
            )

        self.assertEqual((grid.width, grid.height), (2, 2))
        self.assertEqual((bundle.width, bundle.height, bundle.maxval), (2, 2, 255))
        self.assertEqual(bundle.semantic_map_id, "temporary_arena_001")
        self.assertEqual(len(bundle.yaml_sha256), 64)
        self.assertEqual(len(bundle.image_sha256), 64)
        self.assertEqual(len(bundle.bundle_sha256), 64)
        self.assertEqual(bundle.content_sha256, bundle.bundle_sha256)
        self.assertEqual(frozen_map_bundle_payload(bundle)["origin"], [-1.0, -2.0, 0.0])

    def test_image_mutation_invalidates_bundle_even_when_yaml_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = self._write_map(root)
            before = freeze_map_bundle(
                path, semantic_map_id="arena", planning_frame="map"
            )
            (root / "map.pgm").write_text("P2\n2 2\n255\n255 255\n255 255\n")
            after = freeze_map_bundle(
                path, semantic_map_id="arena", planning_frame="map"
            )

        self.assertEqual(before.yaml_sha256, after.yaml_sha256)
        self.assertNotEqual(before.image_sha256, after.image_sha256)
        self.assertNotEqual(before.bundle_sha256, after.bundle_sha256)

    def test_semantic_identity_and_frame_are_part_of_bundle_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_map(Path(tmpdir))
            arena_a = freeze_map_bundle(
                path, semantic_map_id="arena_a", planning_frame="map"
            )
            arena_b = freeze_map_bundle(
                path, semantic_map_id="arena_b", planning_frame="map"
            )
            odom = freeze_map_bundle(
                path, semantic_map_id="arena_a", planning_frame="odom"
            )

        self.assertNotEqual(arena_a.bundle_sha256, arena_b.bundle_sha256)
        self.assertNotEqual(arena_a.bundle_sha256, odom.bundle_sha256)

    def test_descriptor_is_path_independent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            first_bundle = freeze_map_bundle(
                self._write_map(first), semantic_map_id="arena", planning_frame="map"
            )
            second_bundle = freeze_map_bundle(
                self._write_map(second), semantic_map_id="arena", planning_frame="map"
            )

        self.assertEqual(first_bundle, second_bundle)
        self.assertEqual(first_bundle.bundle_sha256, second_bundle.bundle_sha256)

    def test_descriptor_round_trip_is_hashed_and_immutable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle = freeze_map_bundle(
                self._write_map(root), semantic_map_id="arena", planning_frame="map"
            )
            descriptor_path = root / "artifacts" / "map_bundle.json"
            written = write_frozen_map_bundle(descriptor_path, bundle)
            retry = write_frozen_map_bundle(descriptor_path, bundle)
            loaded = load_frozen_map_bundle(
                descriptor_path,
                required_semantic_map_id="arena",
                required_planning_frame="map",
            )
            other = freeze_map_bundle(
                root / "map.yaml",
                semantic_map_id="another_arena",
                planning_frame="map",
            )
            with self.assertRaises(FrozenMapBundleError) as raised:
                write_frozen_map_bundle(descriptor_path, other)

        self.assertEqual(loaded, bundle)
        self.assertEqual(written, bundle.bundle_sha256)
        self.assertEqual(retry, written)
        self.assertEqual(raised.exception.code, "immutable_conflict")

    def test_invalid_semantic_identity_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_map(Path(tmpdir))
            with self.assertRaises(FrozenMapBundleError) as raised:
                freeze_map_bundle(
                    path, semantic_map_id="arena with spaces", planning_frame="map"
                )

        self.assertEqual(raised.exception.code, "invalid_map")


if __name__ == "__main__":
    unittest.main()
