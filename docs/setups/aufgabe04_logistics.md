# Aufgabe 04 Logistics Runbook

Draft placeholder. Fill this after QR payload format, station coordinates, puck
handling rules, and real parkour constraints are confirmed.

Initial implementation order:

1. Pure QR parsing and station ordering.
2. Pure station map and route selection.
3. Dry-run mission state machine.
4. Navigation adapter around Aufgabe 03 planning.
5. ROS wrappers and real-robot run commands only after safety gates exist.

Dry-run artifact layout:

- Generated station layouts: `results/aufgabe04/layouts/`
- Dry-run station routes and diagnostics: `results/aufgabe04/routes/`
- Mission/evidence logs remain grouped by purpose under `results/aufgabe04/`
  or a more specific subfolder when a feature starts producing repeated files.
