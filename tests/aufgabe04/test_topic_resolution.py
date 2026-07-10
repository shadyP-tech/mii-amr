import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.qr_scanning.topic_resolution import resolve_topic  # noqa: E402


class TopicResolutionTest(unittest.TestCase):
    def test_absolute_topic_is_unchanged(self):
        self.assertEqual(resolve_topic("/camera/image_raw/compressed", "robot1"), "/camera/image_raw/compressed")

    def test_relative_topic_resolves_without_namespace(self):
        self.assertEqual(resolve_topic("camera/image_raw/compressed"), "/camera/image_raw/compressed")

    def test_relative_topic_resolves_under_namespace(self):
        self.assertEqual(
            resolve_topic("camera/image_raw/compressed", "/robot1/"),
            "/robot1/camera/image_raw/compressed",
        )

    def test_empty_topic_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "topic"):
            resolve_topic(" ")


if __name__ == "__main__":
    unittest.main()
