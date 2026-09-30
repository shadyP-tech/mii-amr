# Initial execution TF failure: 20260929T151758Z

The run failed before motion on its second coverage leg. A new follower's
initial odometry transforms arrived late and in a burst. Its first available
`odom <- base_footprint` transform was structurally valid but stale. The startup
policy aborted immediately, although 2.65 seconds remained in the existing
stopped acquisition budget. It already handled this situation for
`map <- odom`, but not for the execution transform.

The correction extends the same bounded wait to the execution edge. It does
not admit the stale transform, increase the freshness threshold or extend the
deadline. A fresh replacement and all existing admission checks are required
before startup can succeed.

## Recorded evidence

- Run: `stand_explore_exact2_camera_all5_20260929T151758Z`.
- Clean workstation revision: `3e6dfe14c829ad89aade52b3dbb60d1a57fe764d`.
- Coverage leg 000 completed at **15:20:27.575 UTC**, after 22.277 seconds and
  approximately 0.829508 m of motion.
- Coverage leg 001 passed its dry run and live preflight. Preflight sampled
  fresh execution TF at **15:21:05.353459 UTC**, age **0.025295863 seconds**.
- Its follower created a new TF buffer at **15:21:07.081927 UTC**.
- The first twelve execution-pose lookups failed with `ConnectivityException`.
  At startup elapsed **2.036360 seconds**, the follower correctly entered its
  cold-TF acquisition phase. The total budget was **5 seconds**.
- Lookup 13 produced the first execution transform at elapsed
  **2.345157 seconds**. Its age was **1.643782593 seconds**, above the existing
  **1-second** maximum. Structural validation passed; no execution sample had
  previously passed freshness validation.
- The policy stopped with `required_tf_edge_has_non_acquisition_failure`,
  `deadline_exhausted: false`, and `motion_published: false`. The CSV records
  zero distance and a 2.450-second stopped child.

The execution buffer had just received fourteen odometry transforms. Its
last eight receipt records span about four milliseconds, with ascending
timestamps and decreasing ages from 1.989 to 1.643 seconds. There were no
ingestion exceptions. Meanwhile the global transform, scan and odometry
messages were fresh, and both executors had healthy heartbeats. Both
controller trace commands were zero.

These receipts observe entry into the TF buffer, not DDS receive time. They
establish delayed/batched delivery to this listener, but do not identify the
network or middleware layer responsible. The run stopped too early to prove
whether a fresh replacement would have arrived within the remaining budget.

The mission therefore never reached candidate inspection, camera centering
or opposite-side motion. This failure does not provide evidence of a new
LiDAR association error. The asymmetric first-stale policy predates the latest
LiDAR changes; its global-only helper was introduced in commit `1c1a282` on
September 9.

## Implemented correction

`initial_tf_acquisition.py` now applies the existing first-stale exception to
either required TF edge in a certified odometry execution context. Eligibility
requires an already-entered cold acquisition phase, no successful sample on
the delayed edge, a structurally valid stale sample, and a current fresh peer
transform with clean acquisition history.

`initial_runtime_inputs.py` records fresh peers and failures of established
peers before first samples. This ensures the symmetric check uses the current
iteration's peer state, including a peer that has just become invalid.

The existing checks still require zero motion, fresh sensors, serviced
executors, exact frames, global continuity and the original absolute deadline.
A transform that becomes stale after successful acquisition does not qualify.
Future or malformed transforms do not qualify. A later lookup failure cannot
erase the stale history. Persistent stale input fails at the existing deadline.

The non-acquisition flags and stale counters remain in the evidence, and the
existing recovery validator still rejects them for localization resealing or
a new warmup retry. The old global diagnostic field is preserved; a separate
execution-edge field identifies the new wait case. No motion or recovery
authorization has been added.

## Validation and preserved artifacts

Evidence is stored under
`results/implementation_checks/run_audit_20260929T151758Z/`. Eleven principal
source files were verified against workstation SHA-256 hashes in
`source_sha256.json`.

`replay_initial_execution_tf.py` replays the recorded policy transition against
the run's committed code and the correction. The old code reproduces the exact
denial. The correction permits only continued stopped acquisition, with
**2.654843 seconds remaining**, no available execution pose, and no motion
authorization. The output is `initial_execution_tf_replay.json`.

The deterministic regression tests exercise production transform validation
and continuity with the measured stale age after twelve cold misses. A later
fresh replacement in these tests is explicitly hypothetical. Safety cases
cover persistent stale input, missing/future/malformed transforms, established
edge loss, current-cycle peer failures, continuity drift, health loss, motion
and the hard deadline. The existing global-edge tests remain unchanged.

Validation completed: **148 tests and 237 subtests passed** across the startup,
TF receipt/recovery, resealing, certified startup and follower module suites.
The startup file contains 60 tests, including 10 new regression methods. Six
new methods replayed against the original policy produce eleven expected
assertion/subtest failures and no errors. `git diff --check` also passes.
The combined results are saved in `correction_tests.xml`.

No code was deployed and no robot run was started as part of this correction.
A newer run, `stand_explore_exact2_camera_all5_20260929T152359Z`, appeared during
the audit. At the read-only check it had completed both coverage legs with the
unchanged workstation code and had no recorded mission failure. It is not
validation of this patch.
