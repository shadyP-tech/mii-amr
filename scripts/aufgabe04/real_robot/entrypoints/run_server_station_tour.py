#!/usr/bin/env python3
"""Preview or execute an independent FastAPI tour of saved stand poses."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.real_robot.mission.station_tour_runtime import build_parser, main


if __name__ == "__main__":
    raise SystemExit(main())
