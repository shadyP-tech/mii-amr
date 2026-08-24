"""Explicit side-effect boundary for coverage-leg orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable

from scripts.aufgabe04.navigation.coverage.stand_discovery_route import (
    seal_stand_discovery_route,
)
from scripts.aufgabe04.real_robot import autonomous_coverage_replanning as replanning

EventSink = Callable[[Path, dict[str, object]], None]

KeywordEffect = Callable[..., Any]

@dataclass(frozen=True)
class CoverageLegEffects:
    """Injected effects and replaceable deterministic helpers.

    The first two callbacks are intentionally required: the coverage module
    must never discover a way to launch motion or sample live localization on
    its own.  The parent callback remains responsible for any typed ``RUN``
    prompt requested through ``require_fresh_confirmation``.
    """

    run_motion_leg: KeywordEffect
    admit_preplanning_localization: KeywordEffect
    seal_route: KeywordEffect = seal_stand_discovery_route
    event_sink: EventSink = lambda path, payload: _append_jsonl(path, payload)
    clock: Callable[[], float] = time.time
    replan_startup_source: KeywordEffect = lambda **kwargs: (
        replanning.replan_startup_source(**kwargs)
    )
    replan_runtime_localization_source: KeywordEffect = lambda **kwargs: (
        replanning.replan_runtime_localization_source(**kwargs)
    )
    advance_transient_overlay_resume_state: KeywordEffect = lambda **kwargs: (
        replanning.advance_transient_overlay_resume_state(**kwargs)
    )
    load_coverage_plan: KeywordEffect = lambda plan_path: (
        replanning.load_coverage_plan(plan_path)
    )
    replan_source_preserving_transient_overlay: KeywordEffect = (
        lambda **kwargs: replanning.replan_source_preserving_transient_overlay(
            **kwargs
        )
    )

def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        )
