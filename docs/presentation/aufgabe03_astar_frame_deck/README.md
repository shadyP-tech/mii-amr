# Aufgabe 03 A* GIF Frame Deck

This folder contains a visual-first frame deck for explaining the A* path
generation used in Aufgabe 03.

Generated assets:

- `frames/frame_*.png`: numbered 1920x1080 animation frames
- `astar_path_generation.gif`: presentation-ready animated GIF
- `contact_sheet.png`: overview of all frames
- `manifest.json`: frame names, durations, and algorithm notes

Regenerate with the bundled Python runtime:

```bash
/Users/stephpark/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  scripts/aufgabe03/render_astar_frame_deck.py
```

The animation mirrors the planner choices in
`scripts/aufgabe03/map_path_planner.py`: occupancy-grid planning, inflated
blocked space, start/goal snapping, 8-neighbor expansion without corner
cutting, Euclidean `h`, `f = g + h`, dense path reconstruction, waypoint
simplification, and the same A* loop after a run-local obstacle overlay.
