import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from scripts.aufgabe04.navigation.map_io import freeze_map_bundle
from scripts.aufgabe04.real_robot.hardware_profile import (
    REAL_HARDWARE_PROFILE_SCHEMA_VERSION,
    RealRobotProfile,
    load_real_robot_profile,
    real_robot_profile_sha256,
)
from scripts.aufgabe04.real_robot.physical_site_contract import (
    PhysicalSiteContractError,
    load_physical_site,
    resolve_expected_stand_count,
    validate_physical_site_contract,
)


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_SITE = ROOT / "docs/setups/aufgabe04_lab_20260817.json"
PRODUCTION_PROFILE = (
    ROOT
    / "configs/aufgabe04/real_robot_profiles/turtlebot1_unloaded_20260817.json"
)


class PhysicalSiteFixture:
    def __init__(self, root: Path):
        self.root = root
        self.map_dir = root / "maps"
        self.map_dir.mkdir(parents=True)
        self.map_yaml = self.map_dir / "arena.yaml"
        self.map_image = self.map_dir / "arena.pgm"
        self.map_image.write_bytes(b"P2\n2 2\n255\n254 0\n254 0\n")
        self.map_yaml.write_text(
            "image: arena.pgm\n"
            "resolution: 0.05\n"
            "origin: [0.0, 0.0, 0.0]\n"
            "negate: 0\n"
            "occupied_thresh: 0.65\n"
            "free_thresh: 0.196\n"
            "mode: trinary\n",
            encoding="utf-8",
        )
        self.bundle = freeze_map_bundle(
            self.map_yaml, semantic_map_id="arena", planning_frame="map"
        )
        self.site_path = root / "docs" / "site_v1.json"
        self.payload = {
            "schema_version": 1,
            "physical_site_id": "site_v1",
            "description": "Test arena with five stands",
            "recorded_date": "2026-08-17",
            "map_measurement": {
                "semantic_map_id": "arena",
                "map_yaml": "maps/arena.yaml",
                "map_yaml_sha256": self._digest(self.map_yaml),
                "map_image": "maps/arena.pgm",
                "map_image_sha256": self._digest(self.map_image),
                "map_bundle_sha256": self.bundle.bundle_sha256,
            },
            "station_setup": {
                "expected_stand_count": 5,
                "stand_coordinates_supplied": False,
                "placement": "Unknown positions",
                "orientation": "Unknown orientations",
            },
        }
        self.write_site()

    @staticmethod
    def _digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_site(self) -> None:
        self.site_path.parent.mkdir(parents=True, exist_ok=True)
        self.site_path.write_text(
            json.dumps(self.payload, indent=2) + "\n", encoding="utf-8"
        )

    def profile(self) -> RealRobotProfile:
        return RealRobotProfile(
            schema_version=REAL_HARDWARE_PROFILE_SCHEMA_VERSION,
            profile_id="robot_v1",
            robot_id="robot_1",
            namespace="",
            scan_topic="scan",
            odom_topic="odom",
            cmd_vel_topic="cmd_vel",
            amcl_topic="amcl_pose",
            compressed_image_topic="camera/image_raw/compressed",
            camera_info_topic="camera/camera_info",
            map_frame="map",
            odom_frame="odom",
            base_frame="base_footprint",
            scan_frame="base_scan",
            camera_optical_frame="camera",
            localization_source="amcl",
            physical_site_id="site_v1",
            physical_site_sha256=self._digest(self.site_path),
            calibration_profile_sha256="a" * 64,
            robot_radius_m=0.105,
            scan_origin_to_base_offset_m=0.0,
            max_linear_speed_mps=0.055,
            max_angular_speed_radps=0.18,
        )


class PhysicalSiteContractTest(unittest.TestCase):
    def test_versioned_five_stand_artifacts_are_valid_and_bound(self):
        profile_payload = json.loads(PRODUCTION_PROFILE.read_text(encoding="utf-8"))
        profile = load_real_robot_profile(PRODUCTION_PROFILE)

        contract = validate_physical_site_contract(
            PRODUCTION_SITE,
            profile=profile,
            semantic_map_id="arena_1p898x3p9_auto",
            map_yaml=ROOT / "maps/aufgabe03/arena_1p898x3p9_auto.yaml",
            repository_root=ROOT,
        )

        self.assertEqual(contract.expected_stand_count, 5)
        self.assertEqual(
            contract.physical_site_sha256,
            "7c359a3f1d84cbae8a25573d642142676829e7658fed29a56ca365792cb2679a",
        )
        self.assertEqual(
            profile_payload["real_robot_profile_sha256"],
            real_robot_profile_sha256(profile),
        )
        self.assertEqual(
            contract.map_bundle.bundle_sha256,
            "a0dc0d4c73e96f113ec262640bcdece2bfc2e1f68e8394701ecafd5e569ae17a",
        )

    def test_optional_count_resolves_only_to_canonical_value(self):
        site = load_physical_site(PRODUCTION_SITE)
        self.assertEqual(resolve_expected_stand_count(site, None), 5)
        self.assertEqual(resolve_expected_stand_count(site, 5), 5)
        with self.assertRaisesRegex(
            PhysicalSiteContractError, "requested=3 canonical=5"
        ) as caught:
            resolve_expected_stand_count(site, 3)
        self.assertEqual(caught.exception.code, "stand_count_mismatch")

    def test_full_validation_accepts_matching_requested_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = PhysicalSiteFixture(Path(tmp))
            contract = validate_physical_site_contract(
                fixture.site_path,
                profile=fixture.profile(),
                requested_expected_stand_count=5,
                semantic_map_id="arena",
                map_yaml=fixture.map_yaml,
                map_bundle=fixture.bundle,
                repository_root=fixture.root,
            )
        self.assertEqual(contract.expected_stand_count, 5)

    def test_schema_is_strict_and_rejects_boolean_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = PhysicalSiteFixture(Path(tmp))
            fixture.payload["station_setup"]["expected_stand_count"] = True
            fixture.write_site()
            with self.assertRaisesRegex(
                PhysicalSiteContractError, "expected_stand_count must be an integer"
            ):
                load_physical_site(fixture.site_path)

            fixture.payload["station_setup"]["expected_stand_count"] = 5
            fixture.payload["unreviewed_override"] = True
            fixture.write_site()
            with self.assertRaisesRegex(
                PhysicalSiteContractError, r"unknown=\['unreviewed_override'\]"
            ):
                load_physical_site(fixture.site_path)

    def test_filename_stem_must_equal_site_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = PhysicalSiteFixture(Path(tmp))
            fixture.payload["physical_site_id"] = "different_site"
            fixture.write_site()
            with self.assertRaisesRegex(
                PhysicalSiteContractError, "filename stem"
            ) as caught:
                load_physical_site(fixture.site_path)
        self.assertEqual(caught.exception.code, "site_id_mismatch")

    def test_profile_site_id_and_exact_byte_hash_are_both_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = PhysicalSiteFixture(Path(tmp))
            profile = fixture.profile()
            for changed, code in (
                (
                    replace(profile, physical_site_id="other_site"),
                    "profile_site_id_mismatch",
                ),
                (
                    replace(profile, physical_site_sha256="b" * 64),
                    "profile_site_hash_mismatch",
                ),
            ):
                with self.subTest(code=code):
                    with self.assertRaises(PhysicalSiteContractError) as caught:
                        validate_physical_site_contract(
                            fixture.site_path,
                            profile=changed,
                            repository_root=fixture.root,
                        )
                    self.assertEqual(caught.exception.code, code)

    def test_requested_semantic_map_path_and_bundle_cannot_override_site(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = PhysicalSiteFixture(Path(tmp))
            profile = fixture.profile()
            other_yaml = fixture.map_dir / "other.yaml"
            other_yaml.write_bytes(fixture.map_yaml.read_bytes())
            cases = (
                ({"semantic_map_id": "other"}, "semantic_map_mismatch"),
                ({"map_yaml": other_yaml}, "map_yaml_mismatch"),
                (
                    {
                        "map_bundle": replace(
                            fixture.bundle, semantic_map_id="other"
                        )
                    },
                    "map_bundle_mismatch",
                ),
            )
            for kwargs, code in cases:
                with self.subTest(code=code):
                    with self.assertRaises(PhysicalSiteContractError) as caught:
                        validate_physical_site_contract(
                            fixture.site_path,
                            profile=profile,
                            repository_root=fixture.root,
                            **kwargs,
                        )
                    self.assertEqual(caught.exception.code, code)

    def test_map_byte_tampering_is_rejected_before_bundle_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = PhysicalSiteFixture(Path(tmp))
            profile = fixture.profile()
            fixture.map_image.write_bytes(b"P2\n2 2\n255\n0 0\n0 0\n")
            with self.assertRaises(PhysicalSiteContractError) as caught:
                validate_physical_site_contract(
                    fixture.site_path,
                    profile=profile,
                    repository_root=fixture.root,
                )
        self.assertEqual(caught.exception.code, "map_image_hash_mismatch")

    def test_repository_path_traversal_is_rejected_at_load_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = PhysicalSiteFixture(Path(tmp))
            fixture.payload["map_measurement"]["map_yaml"] = "../arena.yaml"
            fixture.write_site()
            with self.assertRaisesRegex(
                PhysicalSiteContractError, "safe repository-relative path"
            ):
                load_physical_site(fixture.site_path)


if __name__ == "__main__":
    unittest.main()
