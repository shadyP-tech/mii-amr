"""Validate an autonomous facing catalog and publish a frozen survey bundle offline."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.stations.autonomous_arrival_catalog import AutonomousCatalogInputs, promote_autonomous_arrival_catalog


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "facing-catalog", "candidate-snapshot", "confirmed-candidate-snapshot", "observed-identities",
        "source-stand-registry", "coverage-plan", "target-frame-projection", "map-yaml",
        "robot-profile", "camera-calibration", "physical-site", "server-qr-mapping-evidence", "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-identity-registry", type=Path)
    parser.add_argument("--semantic-map-id", required=True)
    parser.add_argument("--server-robot-id", required=True)
    args = parser.parse_args(argv)
    try:
        result = promote_autonomous_arrival_catalog(AutonomousCatalogInputs(**vars(args)), now_sec=time.time())
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
