"""JSONL evidence logging for FastAPI task-client events."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value)!r} is not JSON serializable")


def append_task_event(path: Path, event_type: str, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        **dict(payload),
    }
    with path.open("a") as file:
        file.write(json.dumps(event, default=_json_default, sort_keys=True) + "\n")

