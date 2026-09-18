"""Execute one sealed camera-centering turn through the sole motion node."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.real_robot.execution.candidate_centering import main

if __name__ == "__main__":
    raise SystemExit(main())
