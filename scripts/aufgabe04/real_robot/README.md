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
finishes at five distinct, candidate-bound QR identities. The preferred outcome
has validated head geometry and a facing pose. If geometry fails, a fresh QR
decode with valid corners, unique candidate/LiDAR association, and stopped,
synchronized localization can finish that candidate's discovery immediately.
The robot pose at the successful image timestamp is saved as a
`qr_verified_observation_pose`; the stand angle remains unknown and the LiDAR
candidate position is unchanged. Duplicate
QR claimants are all marked ambiguous and excluded from the completion count.
Exhausted and unvisited hypotheses remain in the full `candidate_snapshot.json`
used by route planning and keepouts. The separate
`confirmed_candidate_snapshot.json` binds the final station identity registry;
it is completion metadata and must not replace the full pool in route planning.
Immutable `candidate_goal_history/` revisions preserve all dispositions, while
`candidate_goal_progress.json` points to the latest revision. Completion can
therefore report the QR goal achieved with some hypotheses still unresolved.

`observer/qr_observation_pose.py` owns QR-only admission;
`artifacts/qr_verified_observation_pose.py` validates its evidence;
`candidate/qr_pose_discovery.py` retains the observation-frame ancestry and
writes its discovery receipt. Autonomous camera exploration enables this
fallback automatically. Geometry is attempted first on the current frame;
successful geometry takes precedence. A fallback ends local inspection without
requesting another view solely for the angle. Every subsequent motion still
requires the ordinary fresh localization, route and permit checks.

`qr_observation_pose_catalog.json` contains only the fallback viewing poses.
`stand_facing_catalog.json` contains only validated geometry records and marks
`facing_complete` false when some discoveries used the fallback. Mission
summaries report separate `facing_ready_stand_count` and `qr_only_stand_count`;
`camera_geometry_complete` can be false while QR exploration is complete.
Logistics catalog promotion rejects an incomplete facing catalog. A stored QR
viewpoint is historical discovery evidence, not a docking pose or a return-route
permission. See [the fallback validation note](../../../docs/setups/aufgabe04_qr_observation_pose_fallback_20260916.md).

Cross-view rejected morphology and conservative visibility gaps are preserved
as immutable candidate-source advisories by
`navigation/coverage/coverage_morphology_conflict.py`. They survive registry,
snapshot and inspection-progress persistence. They grant neither a stand
identity nor permission to reject a candidate or change its keepout. New
admission and handoff schemas bind the pool policy; old camera handoffs must
be regenerated from a fresh admitted session.
