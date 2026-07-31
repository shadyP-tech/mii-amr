#!/usr/bin/env python3
"""Create one immutable content-hashed metric stand profile."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.aufgabe04.perception.stand_axis.model_profile import (
    stand_model_from_payload,
    write_stand_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--environment", choices=("physical", "simulation"), required=True)
    parser.add_argument(
        "--measurement-status",
        choices=("measured", "provisional"),
        required=True,
    )
    parser.add_argument("--head-width-m", required=True, type=float)
    parser.add_argument("--head-height-m", required=True, type=float)
    parser.add_argument("--head-depth-m", required=True, type=float)
    parser.add_argument("--qr-symbol-width-m", required=True, type=float)
    parser.add_argument("--qr-symbol-height-m", required=True, type=float)
    parser.add_argument("--qr-center-x-m", type=float, default=0.0)
    parser.add_argument("--qr-center-y-m", type=float, default=0.0)
    parser.add_argument("--stem-width-m", type=float)
    parser.add_argument("--stem-visible-height-m", type=float)
    parser.add_argument("--tolerance-m", required=True, type=float)
    parser.add_argument("--source", required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    profile = stand_model_from_payload(
        {
            "schema_version": 1,
            "profile_id": args.profile_id,
            "environment": args.environment,
            "measurement_status": args.measurement_status,
            "head_width_m": args.head_width_m,
            "head_height_m": args.head_height_m,
            "head_depth_m": args.head_depth_m,
            "qr_symbol_width_m": args.qr_symbol_width_m,
            "qr_symbol_height_m": args.qr_symbol_height_m,
            "qr_center_x_m": args.qr_center_x_m,
            "qr_center_y_m": args.qr_center_y_m,
            "stem_width_m": args.stem_width_m,
            "stem_visible_height_m": args.stem_visible_height_m,
            "tolerance_m": args.tolerance_m,
            "source": args.source,
        }
    )
    digest = write_stand_model(args.output, profile)
    print(f"wrote immutable stand model: {args.output} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
