# Pilot audit: LiDAR evidence import regression, 9 September 2026

The latest run, `stand_explore_exact2_camera_pilot_20260909T134752Z`, failed
because the LiDAR observer wrote incompatible descriptions of its runtime
configuration. **This is a software regression introduced in our last change,
commit `1c1a282`.** The first coverage leg completed; importing its stopped
LiDAR epoch then raised:

> LiDAR visibility observer runtime config differs from summary

The second survey leg and all camera operations were never reached. This run
therefore does not validate the LiDAR-to-camera execution handoff or camera
freshness/QR processing fixes.

## Run identity and sequence

The parent bundle spans **13:47:52–13:50:34 UTC (15:47:52–15:50:34 CEST)**.
The artifacts were retrieved read-only through `mii001`; the container records
hostname `mii0002` and working directory `/workspace/mii-amr`. Both execution
bundles record clean `main` at
`1c1a282a878d92a35f6ae14bd0415c8af433ff7b`, matching the audited local code.
The command includes `--stop-after-camera-candidates 1`,
`--scan-topology-profile full_rotation`, the unchanged five-stand goal, and
the new bounded camera capture options.

| UTC | Recorded outcome |
| --- | --- |
| 13:48:23.728–13:49:18.866 | Separately authorized startup localization completed: 55.099 s, 6.216 rad rotation, maximum odometry translation 0.547 mm, and accepted stationary-odom evidence. |
| Before / after localization | Startup route admission improved from 0/2 to 2/2 accepted options; selected remaining uncertainty margin was 61.556 mm. |
| 13:49:57.512 | First coverage execution preflight passed all 15 observations. |
| 13:49:58.478 | The exact first-leg motion permit was consumed. |
| 13:50:23.317 | Coverage leg `000` completed: 24.690 s, estimated travel 0.847757 m on a 0.880614 m route. |
| 13:50:25.354–13:50:33.429 | Observer epoch `survey_vp_001` processed 83 scans and wrote 83 visibility receipts plus 431 accepted observations. |
| By 13:50:34 | The importer rejected the mismatched configuration; parent exited with code 2 and `status: failed_closed`. |

The observer proposed 686 candidates across those scans, accepted 431
observations, and reported seven accumulated local tracks. Those seven tracks
are **not seven validated physical stands or QR identities**. The epoch was
not committed: `coverage_progress.json` retains an empty visited-viewpoint list,
the mission stand registry is empty, and no coverage checkpoint or camera
candidate snapshot was produced. The summary's **95.31% is planned coverage**;
its admitted visited coverage remains zero because import failed. The physical
first-leg movement is independently recorded in the segment ledger.

## Exact defect and causal evidence

At the deployed revision `1c1a282`,
[observer construction](../../scripts/aufgabe04/perception/stand_explorer_node.py)
at line 507 constructed the visibility session with:

```python
runtime_config={**self.runtime.as_log_dict(),
                "scan_topology_profile": getattr(args, "scan_topology_profile", "linear")}
```

This dictionary is part of the sealed observer configuration referenced by
every visibility receipt. However, `observer_summary_payload` at line 1067
writes only `node.runtime.as_log_dict()` into the summary's `runtime_config`.
The summary has a separate top-level `scan_topology_profile`, but that does not
make the two nested dictionaries equal.

The real artifact has exactly one differing key:

| Location | `scan_topology_profile` |
| --- | --- |
| `lidar_visibility_observer_config.runtime_config` | `full_rotation` |
| `runtime_config` | Missing |

[Visibility admission](../../scripts/aufgabe04/navigation/coverage/coverage_visibility_reporting.py)
at lines 276–278 correctly requires exact equality. The prior `4627099`
producer used the base runtime dictionary on both sides; `1c1a282` changed
only the session side. **The defect also affects `linear`**, because the new
producer adds that key even when the CLI option is omitted.

The saved [offline replay](../../results/audits/stand_explore_exact2_camera_pilot_20260909T134752Z/replay_visibility_failure.py)
uses the real stopped-epoch visibility importer and copied artifacts. It only
relocates the receipt path to the local copy in memory. The original summary
reproduces the exact exception. Adding the missing nested profile to a separate
in-memory summary admits **all 83 receipts**, including the existing hash,
timing, frame, sibling-directory and morphology/proposal checks. No source
artifact or production file was changed. The
[replay receipt](../../results/audits/stand_explore_exact2_camera_pilot_20260909T134752Z/replay_visibility_failure.json)
records unchanged input hashes and matching deployed/local source hashes.
This isolates the visibility-admission blocker; it does not establish that a
second leg or later camera mission would succeed.

## What the hardware evidence does establish

The first coverage controller became ready after **1.781 s**, with six
`map ← odom` lookup exceptions followed by an accepted seventh transform.
Its age was −0.869 s, inside the configured 1.1 s future allowance; odom TF
was fresh on all seven samples. `extension_used=false` and both stale-wait
counters are zero. **The new stale-first-sample extension was not exercised.**
The actual-buffer receipt instrumentation did record live TF ingestion:
28 odom and 18 map calls, with no recorded ingestion exceptions. That proves
the tracing ran, not the transport cause of the previous failure.

The 181 recorded tracking cycles all passed front-clearance checks, with a
minimum 0.307 m against the 0.200 m threshold. Maximum cross-track error was
9.586 mm within the 30 mm tube. The observer received and processed all 83
scans, with zero queue drops or exact-TF timeouts. Mean processing time was
10.93 ms, mean scan age 58.55 ms and maximum scan age 314.44 ms. These records
do not point to a LiDAR/TF freshness failure as this run's terminal cause.
Coverage completion and post-run publisher shutdown are logged; the audit did
not independently observe the physical stop.

Topology diagnostics were also exercised. The 83 distinct scan stamps contain
216–219 returns: 12 qualified for circular endpoint adjacency, 51 fell back to
linear because endpoint metadata was inconsistent, and 20 fell back because
the indexed seam gap was not one step. These are conservative geometry
decisions, not the configuration-mismatch exception. Keep the original angular
metadata and investigate the driver convention before changing seam tolerance.
The counts derive from saved per-observation scan provenance, not a raw range
recording or physical stand ground truth.

## Why the offline suite missed it and required correction

The previous 747 passing tests and 572 subtests did not connect the actual
producer summary to the importer. The nearest producer test in
[test_stand_explorer_tf.py](../../tests/aufgabe04/test_stand_explorer_tf.py)
at line 269 already creates different runtime dictionaries for its session and
summary; it checks receipt counts and hashes without calling admission.
Conversely, the importer fixture in
[test_coverage_visibility_reporting.py](../../tests/aufgabe04/test_coverage_visibility_reporting.py)
at line 150 copies the configuration into a fabricated summary, guaranteeing
agreement. Runner command tests replace the observer subprocess with a minimal
summary. Those tests verify their separate components but miss this contract
between them.

The correction is to build **one immutable observer runtime configuration**
containing the selected scan topology and reuse it for both session construction
and summary serialization. Keep changing per-scan geometry and wrapped-cluster
diagnostics outside that static configuration. Preserve the importer's equality
and hash checks.

Before another robot run, add a regression that uses that real production
configuration path, real `LidarVisibilitySession` receipt persistence and
`write_observer_summary`, then passes the result through the actual stopped-epoch
importer. Cover both topology profiles and live/frozen observation geometry,
and retain rejection tests for conflicting topic/profile/config hashes.

This audit made **no production edits, remote changes or robot commands**.
All 94 copied source files are indexed in the
[source manifest](../../results/audits/stand_explore_exact2_camera_pilot_20260909T134752Z/source_manifest.json).
Derived counts are in
[run_metrics.json](../../results/audits/stand_explore_exact2_camera_pilot_20260909T134752Z/run_metrics.json).

## Implemented correction — 10 September 2026

The local implementation now captures the resolved ROS settings and selected
scan topology once in the ROS-free
[LidarObserverRuntime](../../scripts/aufgabe04/perception/lidar_observer_runtime.py)
module. Its session factory binds the receipt configuration to the same frozen
runtime used by the observer summary, detector and observation provenance.
Each serialization returns detached dictionaries, so per-scan diagnostics do
not alter the static receipt configuration hash. The receipt schema and strict
importer checks are unchanged.

The new
[producer/importer contract tests](../../tests/aufgabe04/test_lidar_observer_visibility_contract.py)
exercise real receipt persistence and summary writing through the full
stopped-epoch visibility importer. They cover `linear` and `full_rotation` with
live and frozen frames, reproduce the missing nested field, and reject changed
topics, profiles, timing, receipt bytes, frozen-frame evidence and epoch paths.
An observer callback test also checks that changing the argument namespace after
capture cannot change the detector's profile. The existing receipt-flush test
now calls the real importer instead of checking only counts and hashes.

Validation: **86 tests and 54 subtests passed** across the observer, visibility,
stopped-epoch admission/transaction and scan-topology modules. `git diff --check`
passed.

The [corrected-producer replay](../../results/audits/aufgabe04_visibility_config_fix_20260910/replay_new_visibility_producer.py)
reconstructs the recorded runtime and creates a new session through the shared
factory, buffers the original 83 receipts, writes a new summary through the real
producer and calls the same visibility importer. **All 83 receipts pass.** The
configuration hash, receipt payloads and serialized receipt bytes match the
originals. The [replay result](../../results/audits/aufgabe04_visibility_config_fix_20260910/replay_new_visibility_producer.json)
records the generated summary, source hashes and unchanged input hashes. All 94
copied source artifacts also still match the audit manifest.

This is an offline code correction; the replay does not rerun detection or
commit coverage. No deployment or robot experiment was performed, and camera
handoff success still requires hardware evidence.
