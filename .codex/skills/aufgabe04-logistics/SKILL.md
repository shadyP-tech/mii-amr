---
name: aufgabe04-logistics
description: >
  Handle Aufgabe 04 logistics work in the mii-amr repository: QR payload
  parsing, onboard camera QR scanning, station maps and approach targets,
  station routing, puck transport assumptions, single-robot mission state,
  mission logging, fleet coordination, station locks, right-before-left policy,
  collision checks, navigation adapters, tests, runbooks, and real-parkour
  evidence.
---

# Aufgabe 04 Logistics

## Purpose

Use this skill for `docs/tasks/Aufgabe_04.pdf` work: QR scanning, station order
derivation, station routing, puck transport, single-robot logistics missions,
two-robot coordination, safety checks, runbooks, and evidence collection.

Start from the current skeleton; do not treat Aufgabe 04 as missing code.

## Code Map

- `scripts/aufgabe04/qr_scanning/`: QR payload parsing, station order models,
  and future onboard camera node.
- `scripts/aufgabe04/stations/`: station IDs, poses, approach targets, and
  station visit routing.
- `scripts/aufgabe04/logistics/`: mission state, puck transport assumptions,
  mission result models, and CSV logging.
- `scripts/aufgabe04/fleet/`: robot IDs, station leases, right-before-left
  decisions, robot status, and shared-course conflict checks.
- `scripts/aufgabe04/navigation/`: adapter around Aufgabe 03 A*/waypoint
  planning, waypoint generation, pure CSV/diagnostics gates, ROS runtime config,
  strict ROS preflight, single-segment runner, simple waypoint follower, and
  append-only segment run logging.
- `scripts/common/run_with_bundle.sh`: task-agnostic real-run evidence wrapper.
  It records diagnostics and wraps an explicitly supplied command, but it is not
  a safety gate and never publishes motion itself.
- `tests/aufgabe04/`: pure offline tests for the current skeleton.
- `tests/test_run_with_bundle.py`: ROS-free tests for the common run-bundle
  wrapper.
- `docs/setups/aufgabe04_*.md`: draft QR, logistics, real parkour, and
  two-robot runbooks.
- `results/aufgabe04/`: CSV/Markdown evidence targets.
- `results/real_runs/<run_id>/`: raw/debug bundles for physical runs.

## Implementation Order

Prefer this order unless the user explicitly redirects:

1. QR payload parsing and station order.
2. Station map, approach targets, and station routing.
3. Navigation route adapter and waypoint generation.
4. Single-robot mission state and logging.
5. Fleet coordination policy, station locks, and collision rules.
6. ROS/onboard camera integration and real-robot CLIs last.
7. Wrap every physical run with `scripts/common/run_with_bundle.sh` once the
   strict dry-run/preflight path is clean.

Keep most logic pure first. Add ROS wrappers only where they are needed.

## Safety Rules

Never execute physical TurtleBot motion automatically.

Before suggesting real robot motion, state the physical safety requirements:
clear arena, operator beside each robot, Ctrl+C ready, physical stop possible,
and a separate `/cmd_vel` stop terminal available.

For Aufgabe 04 real runs, require:

- per-robot namespace/topic clarity
- `/cmd_vel` ownership or mux check
- fresh `/scan`, `/odom`, TF, and AMCL when used
- no active competing Nav2/custom follower goal during handoff
- station keepout and approach-zone validation
- single-robot validation before two-robot operation
- `scripts/aufgabe04/navigation/entrypoints/run_single_station_segment.py --dry-run` must
  pass before real single-segment motion
- `scripts/common/run_with_bundle.sh` should wrap every physical run and use
  the same namespace/topic/frame options as the wrapped command
- the bundle is evidence capture only; it does not replace `ros_preflight.py`,
  the runner's typed `RUN` confirmation, or the physical stop requirements

## Tests

Run the current pure Aufgabe 04 tests with:

```bash
python3 -m unittest discover tests/aufgabe04
```

Run the common bundle wrapper tests with:

```bash
python3 -m unittest tests.test_run_with_bundle
```

When adding behavior, add or update tests in `tests/aufgabe04/` before wiring
ROS motion. Pure tests should not require ROS, Gazebo, camera hardware, or the
real robot.

## Evidence Targets

Use `results/aufgabe04/` for report-ready evidence:

- `qr_scans.csv`
- `station_visits.csv`
- `logistics_mission_runs.csv`
- `fleet_coordination_events.csv`
- `real_parkour_notes.md`
- `station_segment_runs.csv`

Do not claim real-parkour success unless the relevant CSV/log notes exist and
match the described run.

Use `results/real_runs/<run_id>/` for raw/debug evidence bundles:

- `manifest.txt`
- `command.txt`
- `terminal_run.log`
- ROS topic/node/action snapshots
- `/cmd_vel`, `/scan`, `/odom`, `/amcl_pose`, and TF captures
- `archive_hint.txt`

Do not confuse bundle diagnostics with safety approval. For motion claims,
cross-check the wrapped command's preflight JSON and Aufgabe 04 CSV logs.
