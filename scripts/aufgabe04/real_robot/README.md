# Aufgabe 04 real-robot package

Canonical implementations are grouped by responsibility:

- `configuration/`: immutable hardware/camera profiles, site contracts,
  geometry, recommendation building, and profile-capture CLIs.
- `observer/`: passive observer contracts, temporal evidence, process/TF
  handling, diagnostics, and the ROS camera/LiDAR node.
- `passive_survey/`: passive-survey preparation and finalization workflows.
- `candidate/`: candidate approach, camera-observation deferral, and bounded
  startup/runtime recovery policies.
- `readiness/`: localization, exact-time TF, preauthorization,
  post-observation, and startup-reseal gates.
- `mission/`: coverage mission state, exact-two completion, checkpoint resume,
  session manifests, run modes, and reporting.
- `execution/`: canonical artifact paths, child-runner contracts, runtime
  localization permits, and the unloaded-segment entry point.
- `entrypoints/`: executable command adapters for real-robot workflows.
- `coverage_leg/`: one coverage leg and its route-sealing/recovery phases.
- `autonomous_runner/`: the CLI contract and thin mission composition runtime.

Only `__init__.py` remains as Python code directly in `real_robot/`. Import
library code from its responsibility package and invoke commands through
`entrypoints/`.

None of this structure changes motion ownership: the real-robot orchestration
packages do not publish `/cmd_vel`; certified navigation remains behind the
existing navigation runner and its physical-run gates.
