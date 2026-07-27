import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.logistics.carrier_profile import (  # noqa: E402
    CarrierProfile,
    build_motion_envelope,
)
from scripts.aufgabe04.logistics.models import PuckState  # noqa: E402


class CarrierProfileTest(unittest.TestCase):
    def setUp(self):
        self.profile = CarrierProfile(
            profile_id="burger-passive-v1",
            unloaded_footprint_radius_m=0.105,
            loaded_footprint_radius_m=0.145,
            empty_robot_mass_kg=1.0,
            max_payload_mass_kg=0.25,
        )

    def test_loaded_envelope_uses_larger_radius_and_total_mass(self):
        envelope = build_motion_envelope(
            self.profile,
            puck_state=PuckState.HELD,
            payload_mass_kg=0.1,
            retention_confirmed=True,
        )

        self.assertAlmostEqual(envelope.footprint_radius_m, 0.145)
        self.assertAlmostEqual(envelope.total_mass_kg, 1.1)
        self.assertTrue(envelope.loaded)

    def test_loaded_motion_requires_retention(self):
        with self.assertRaisesRegex(ValueError, "retention"):
            build_motion_envelope(
                self.profile,
                puck_state=PuckState.HELD,
                payload_mass_kg=0.1,
            )

    def test_rejects_payload_over_profile_limit(self):
        with self.assertRaisesRegex(ValueError, "mass limit"):
            build_motion_envelope(
                self.profile,
                puck_state=PuckState.HELD,
                payload_mass_kg=0.3,
                retention_confirmed=True,
            )

    def test_profile_cannot_shrink_when_loaded(self):
        with self.assertRaisesRegex(ValueError, "at least"):
            CarrierProfile("bad", 0.2, 0.1, 1.0, 0.2)


if __name__ == "__main__":
    unittest.main()
