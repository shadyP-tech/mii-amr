# Run audit: LiDAR-to-camera handoff, 9 September 2026

The first camera preapproach stopped before publishing motion because its new
TF listener received an old `map <- odom` transform after eleven missing-frame
lookups. The LiDAR survey and candidate-population handoff had completed.
Visited coverage was **95.31%** with all five candidates retained and **0/5 QR
identities confirmed**. Camera inspection had not started. The terminal message concealed the specific
`stale_transform` cause behind the generic wording “TF transform unavailable”.

Audited session: `stand_explore_exact2_camera_20260909T123820Z`.
The workstation and local checkout both used clean revision
`4627099b04da379dce222a9544e00f633e170f43` before the audit changes.
Artifacts were copied read-only from `mii001`; no robot command was executed.

| Recorded point (UTC) | Evidence |
| --- | --- |
| 12:39:38.660 | Coverage leg 000 completed with motion. |
| 12:40:41.926 | Coverage leg 001 completed with motion. Both planned LiDAR viewpoints were visited. |
| 12:40:51.702 | Exact-two camera handoff sealed five selected candidates: two multi-view and three single-view candidates requiring camera validation. |
| 12:41:10.986 | First candidate dry run completed. Target: `survey_candidate_0003`; preapproach length about 0.069 m. |
| 12:41:23.627 | Execution preflight passed all 15 observations. Its global TF age was −0.894 s, within the configured 1.1 s future allowance. |
| 12:41:23.758 | One-use routine-child permit consumed. `motion_started` records execution-attempt entry; its `motion_published` field is **false**. |
| 12:41:26.967 | Follower recorded `safety_stop`, `motion_published=false`; mission subsequently failed closed. |

The executing listener's initial acquisition evidence establishes:

- `odom <- base_footprint`: 12 successful samples; last age 0.177 s.
- `map <- odom`: 11 `LookupException` results, then `stale_transform` on attempt 12;
  zero accepted global samples. The last stamp was `1788957685.168265` and its
  measured age was **1.664346432 s**, exceeding the **1.0 s** limit.
- Fresh scan and odometry; TF executor alive, 61 heartbeats, last heartbeat age
  0.01925 s. Executor liveness explicitly did not prove TF delivery.
- Acquisition elapsed **3.065874 s** of the existing **5.0 s** maximum.
  `deadline_exhausted=false` and
  `denial_reason=required_tf_edge_has_non_acquisition_failure`.

Preflight and execution create separate ROS contexts and TF listeners. Preflight
readiness therefore does not establish readiness in the follower's new buffer.
The existing isolated TF executor was running in this failure. The earlier
diagnostic frame graph also contained a dynamic `map -> odom` edge; neither that
graph nor node listings establish timely delivery during the failing interval.
There is no evidence that the handoff stopped AMCL. The logs cannot distinguish
publisher delay, transport/discovery delay, or queued delivery; they do not
record direct per-edge callback receipt timing across the transition.
Backside classification, opposite-face inspection and QR-loop termination were
not exercised, so this run cannot validate or invalidate their behavior.

Two software gaps were found:

1. **An old first global sample prematurely ended the stopped acquisition.**
   Once cold acquisition began, any stale sample was treated as terminal even
   while the original deadline still had time remaining. The old sample could
   not safely be used for motion. Its pose was not structurally validated before
   freshness rejection, so the saved failure also does not prove that its
   translation, quaternion, or returned frame IDs were valid.
2. **Valid missing-TF recovery was blocked by incompatible reason fields.**
   Candidate recovery, coverage recovery, and startup permit validation compared
   the outer display reason with the typed inner reason. A correctly formed
   outer `TF transform unavailable: map <- odom` cannot equal the required inner
   `lookup_exception`. This is an additional defect, not the cause of this
   recorded stale sample's correct recovery rejection.

The local hardening changes keep the original deadline and freshness limits.
A structurally validated old first global sample can remain in stopped
acquisition only after cold acquisition has already begun and while the exact
certified frame pair, fresh execution TF and sensors, healthy executor, and
absence of motion or continuity failure remain established. It never supplies
a usable pose. A later fresh sample must pass the full existing localization,
drift, and route-start gates. Stale history remains recorded so a stale-then-missing
timeout cannot enter the pure missing-TF reseal policy. Future or malformed TF,
stale execution TF, and loss of an established global edge remain failures.

A shared reason-binding check now connects typed recovery evidence to its exact
outer frame label in all three recovery/permit paths. Eligibility still requires
the full existing evidence, bounded preparation, fresh stationary localization,
a new certificate, and a new one-use child permit. Terminal candidate failures
now show the typed cause, age/limit, acquisition denial, elapsed/budget, and
specific policy rejection without changing the stored child stop reason.

New diagnostic tracing records up to eight receipts per required dynamic edge
at the executing buffer's `set_transform` call. It retains ROS and monotonic
receipt times, header stamps, age at receipt, ingestion-call completion, and an
immediate newest-buffer stamp observation. This is the buffer used by the
actual listener, not another subscriber. Call return and receipt do not prove
that TF core accepted the insertion, and none of these diagnostic fields grants
readiness. Preflight already records direct TF receipt times, allowing the next
run to compare both sides of the transition. The wrapper follows the official
[Humble listener callback](https://raw.githubusercontent.com/ros2/geometry2/humble/tf2_ros_py/tf2_ros/transform_listener.py)
and [buffer API](https://raw.githubusercontent.com/ros2/geometry2/humble/tf2_ros_py/tf2_ros/buffer.py).

The remaining time belongs to the same stopped child acquisition. A fresh
replacement transform still has to agree with that child's frozen localization
certificate and pass route-start admission. A replacement child after a failed
attempt remains a separate preparation path requiring fresh localization,
recertification and another one-use permit. The wait change does not refresh,
reuse or bypass a consumed permit.

Validation: **287 tests passed**, with zero failures, errors or skips, across
20 focused follower, recovery, permit, receipt-tracing and child-runner modules.
`git diff --check` passed. The exact module list and output are retained in the
local audit directory as `validation.json` and `validation.txt`.
Offline regressions cover the observed missing-to-stale
sequence, simulated subsequent fresh delivery, terminal failure boundaries,
coverage/candidate recovery dispatch, and actual startup-permit validation.
The listener/executor integration tests use ROS-free fakes; DDS transport and
an actual ROS listener have not been validated by these tests.
Successful simulated later delivery is not evidence that this run would have
succeeded: execution stopped before any later sample was observed. Hardware
validation of the updated handoff remains outstanding; the workstation checkout
was not modified or rerun by this audit.

Local source evidence is retained under
`results/audits/stand_explore_exact2_camera_20260909T123820Z/`, including
`source_manifest.json` with hashes of the principal artifacts. The archive SHA-256
is `5a4c4d77299411e015b801342fcabe9058735cb0d6da4b5eba5887720462e95e`.
The decisive files are `mission_failure.json`, the candidate's
`controller_trace.jsonl`, its execution preflight and semantic event log,
`coverage_exact_two_camera_handoff.json`, and `station_segment_runs.csv`.
