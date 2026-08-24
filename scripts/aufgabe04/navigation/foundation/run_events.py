"""Structured semantic run events for Aufgabe 04 navigation runners.

This module is intentionally ROS-free. Runner CLIs configure logging handlers;
other modules should only build or emit deterministic event payloads.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


LOGGER_NAME = "aufgabe04.navigation.run_events"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_event(event: str, **fields: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "timestamp": utc_timestamp(),
        "event": event,
    }
    payload.update(fields)
    return payload


def event_to_json(event: Mapping[str, object]) -> str:
    return json.dumps(dict(event), sort_keys=True, separators=(",", ":"))


def configure_event_logger(log_path: Path | None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)
    else:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)
    return logger


def emit_event(logger: logging.Logger, event: str, **fields: object) -> dict[str, object]:
    payload = build_event(event, **fields)
    logger.info(event_to_json(payload))
    return payload
