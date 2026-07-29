"""Seal an observe-and-plan result for the real runner's dry-run gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.detected_stand_preapproach import (
    DEFAULT_COMMAND_OWNER,
    DEFAULT_TRACKING_TUBE_RADIUS_M,
    seal_detected_stand_preapproach,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--command-owner", default=DEFAULT_COMMAND_OWNER)
    parser.add_argument(
        "--tracking-tube-radius-m",
        type=float,
        default=DEFAULT_TRACKING_TUBE_RADIUS_M,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        outputs = seal_detected_stand_preapproach(
            pipeline_root=args.pipeline_root,
            output_dir=args.output_dir,
            command_owner=args.command_owner,
            tracking_tube_radius_m=args.tracking_tube_radius_m,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
