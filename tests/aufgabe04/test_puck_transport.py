import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.logistics.models import PuckState  # noqa: E402
from scripts.aufgabe04.logistics.carrier_profile import CarrierProfile  # noqa: E402
from scripts.aufgabe04.logistics.puck_transport import (  # noqa: E402
    PuckTransportState,
    acknowledge_loaded_custody,
    acknowledge_puck_delivered,
    acknowledge_puck_loaded,
    report_loaded_puck_lost,
    require_puck_loaded,
    transport_motion_envelope,
)
from scripts.aufgabe04.logistics.puck_custody import (  # noqa: E402
    PuckCustodyLedger,
    claim_puck,
    confirm_puck_loaded,
    register_puck,
)


class PuckTransportTest(unittest.TestCase):
    def test_requires_loaded_puck(self):
        with self.assertRaisesRegex(ValueError, "puck must be loaded"):
            require_puck_loaded(PuckState.NOT_HELD)

    def test_explicit_load_and_drop_acknowledgements(self):
        state = acknowledge_puck_loaded(
            PuckTransportState(robot_id="robot-1"),
            puck_id="puck-1",
            custody_fencing_token=4,
            payload_mass_kg=0.1,
            retention_confirmed=True,
            now_sec=1.0,
        )
        delivered = acknowledge_puck_delivered(
            state, custody_fencing_token=4, now_sec=2.0
        )

        self.assertEqual(delivered.puck_state, PuckState.DELIVERED)
        self.assertEqual(delivered.payload_mass_kg, 0.0)

    def test_stale_custody_token_cannot_drop(self):
        state = acknowledge_puck_loaded(
            PuckTransportState(robot_id="robot-1"),
            puck_id="puck-1",
            custody_fencing_token=4,
            payload_mass_kg=0.1,
            retention_confirmed=True,
            now_sec=1.0,
        )

        with self.assertRaisesRegex(ValueError, "stale"):
            acknowledge_puck_delivered(
                state, custody_fencing_token=3, now_sec=2.0
            )

    def test_loss_is_terminal_and_clears_loaded_mass(self):
        state = acknowledge_puck_loaded(
            PuckTransportState(robot_id="robot-1"),
            puck_id="puck-1",
            custody_fencing_token=4,
            payload_mass_kg=0.1,
            retention_confirmed=True,
            now_sec=1.0,
        )
        lost = report_loaded_puck_lost(
            state,
            custody_fencing_token=4,
            reason="puck dropped",
            now_sec=2.0,
        )

        self.assertEqual(lost.puck_state, PuckState.LOST)
        self.assertEqual(lost.payload_mass_kg, 0.0)
        with self.assertRaisesRegex(ValueError, "terminal"):
            acknowledge_puck_loaded(
                lost,
                puck_id="puck-1",
                custody_fencing_token=5,
                payload_mass_kg=0.1,
                retention_confirmed=True,
                now_sec=3.0,
            )
        profile = CarrierProfile("carrier", 0.1, 0.15, 1.0, 0.2)
        with self.assertRaisesRegex(ValueError, "blocked"):
            transport_motion_envelope(lost, profile)

    def test_transport_envelope_uses_carrier_loaded_radius(self):
        state = acknowledge_puck_loaded(
            PuckTransportState(robot_id="robot-1"),
            puck_id="puck-1",
            custody_fencing_token=4,
            payload_mass_kg=0.1,
            retention_confirmed=True,
            now_sec=1.0,
        )
        profile = CarrierProfile("carrier", 0.1, 0.15, 1.0, 0.2)

        self.assertAlmostEqual(
            transport_motion_envelope(state, profile).footprint_radius_m, 0.15
        )

    def test_transport_can_be_bound_to_confirmed_custody(self):
        ledger = register_puck(
            PuckCustodyLedger.empty(), "puck-1", now_sec=1.0
        )
        ledger = claim_puck(ledger, "puck-1", "robot-1", now_sec=2.0)
        token = ledger.records["puck-1"].fencing_token
        ledger = confirm_puck_loaded(
            ledger, "puck-1", "robot-1", token, now_sec=3.0
        )

        state = acknowledge_loaded_custody(
            PuckTransportState(robot_id="robot-1"),
            ledger.records["puck-1"],
            payload_mass_kg=0.1,
            retention_confirmed=True,
            now_sec=3.0,
        )

        self.assertEqual(state.puck_state, PuckState.HELD)
        self.assertEqual(state.custody_fencing_token, token)

    def test_transport_rejects_custody_owned_by_other_robot(self):
        ledger = register_puck(
            PuckCustodyLedger.empty(), "puck-1", now_sec=1.0
        )
        ledger = claim_puck(ledger, "puck-1", "robot-2", now_sec=2.0)
        token = ledger.records["puck-1"].fencing_token
        ledger = confirm_puck_loaded(
            ledger, "puck-1", "robot-2", token, now_sec=3.0
        )

        with self.assertRaisesRegex(ValueError, "another robot"):
            acknowledge_loaded_custody(
                PuckTransportState(robot_id="robot-1"),
                ledger.records["puck-1"],
                payload_mass_kg=0.1,
                retention_confirmed=True,
                now_sec=3.0,
            )


if __name__ == "__main__":
    unittest.main()
