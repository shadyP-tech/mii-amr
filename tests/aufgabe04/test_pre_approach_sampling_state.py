import tempfile
import unittest
from pathlib import Path

from scripts.aufgabe04.navigation.approach.pre_approach_sampling_state import (
    initial_sampling_state,
    load_sampling_state,
    write_sampling_state,
)


class PreApproachSamplingStateTest(unittest.TestCase):
    def test_rejection_advances_and_persists_reason(self):
        state = initial_sampling_state(
            stand_id="detected_stand_04",
            reference_x_m=1.0,
            reference_y_m=-0.5,
            candidate_count=8,
        ).reject_current("oblique_silhouette")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sampling.json"
            write_sampling_state(path, state)
            loaded = load_sampling_state(path)

        self.assertEqual(loaded.candidate_index, 1)
        self.assertEqual(loaded.rejected[0].candidate_index, 0)
        self.assertEqual(loaded.rejected[0].reason, "oblique_silhouette")

    def test_exhaustion_fails_closed(self):
        state = initial_sampling_state(
            stand_id="stand", reference_x_m=0.0, reference_y_m=0.0, candidate_count=1
        )
        with self.assertRaisesRegex(ValueError, "exhausted"):
            state.reject_current("no_usable_observation")


if __name__ == "__main__":
    unittest.main()
