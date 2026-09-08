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

Exact-two camera exploration separates the inspection pool from the QR goal.
`navigation/coverage/candidate_inspection_pool.py` admits every usable strict
or boundary hypothesis when the pool contains between the expected stand
count and twice that count. For the five-stand site this is five to ten
hypotheses. Too few or too many still produces a durable failure; the selector
never silently chooses a best-five subset.

`candidate/qr_goal_progress.py` records each bounded inspection episode and
finishes at five distinct QR identities with validated stand poses. Duplicate
QR claimants are all marked ambiguous and excluded from the completion count.
Exhausted and unvisited hypotheses remain in the full `candidate_snapshot.json`
used by route planning and keepouts. The separate
`confirmed_candidate_snapshot.json` binds the final station identity registry;
it is completion metadata and must not replace the full pool in route planning.
Immutable `candidate_goal_history/` revisions preserve all dispositions, while
`candidate_goal_progress.json` points to the latest revision. Completion can
therefore report the QR goal achieved with some hypotheses still unresolved.

Cross-view rejected morphology and conservative visibility gaps are preserved
as immutable candidate-source advisories by
`navigation/coverage/coverage_morphology_conflict.py`. They survive registry,
snapshot and inspection-progress persistence. They grant neither a stand
identity nor permission to reject a candidate or change its keepout. New
admission and handoff schemas bind the pool policy; old camera handoffs must
be regenerated from a fresh admitted session.
