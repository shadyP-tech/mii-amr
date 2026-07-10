#!/usr/bin/env python3
"""Generate a simulation-only Burger camera SDF with valid pinhole optics."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizontal-fov-rad", type=float, default=1.3962634)
    args = parser.parse_args()
    if not 0.1 < args.horizontal_fov_rad < 3.0:
        parser.error("horizontal FOV must be between 0.1 and 3.0 radians")
    text = args.source.read_text()
    patched, count = re.subn(
        r"<horizontal_fov>[^<]+</horizontal_fov>",
        f"<horizontal_fov>{args.horizontal_fov_rad:.7f}</horizontal_fov>",
        text,
        count=1,
    )
    if count != 1:
        raise SystemExit("source SDF does not contain exactly one camera horizontal_fov")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(patched)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
