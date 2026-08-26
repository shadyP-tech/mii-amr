#!/usr/bin/env python3
"""Compatibility entry point for autonomous stand exploration."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.real_robot.autonomous_runner import runtime as _runtime


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(_runtime.main())

sys.modules[__name__] = _runtime
