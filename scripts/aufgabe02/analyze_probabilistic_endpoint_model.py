#!/usr/bin/env python3
from probabilistic_model import *  # noqa: F401,F403
from probabilistic_model.cli import main
from probabilistic_model.errors import DataError


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DataError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
    except OSError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
