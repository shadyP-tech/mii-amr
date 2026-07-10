import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.logistics.models import PuckState  # noqa: E402
from scripts.aufgabe04.logistics.puck_transport import require_puck_loaded  # noqa: E402


class PuckTransportTest(unittest.TestCase):
    def test_requires_loaded_puck(self):
        with self.assertRaisesRegex(ValueError, "puck must be loaded"):
            require_puck_loaded(PuckState.NOT_HELD)


if __name__ == "__main__":
    unittest.main()

