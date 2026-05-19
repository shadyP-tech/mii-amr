#!/usr/bin/env python3
"""Public entrypoint for the two-stage waypoint runner."""

import sys

from two_stage_waypoint.cli import main


if __name__ == "__main__":
    sys.exit(main())
