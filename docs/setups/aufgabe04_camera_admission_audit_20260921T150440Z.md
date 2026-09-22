# Camera admission and viewer flicker, 21 September 2026

The latest inspected mission is `stand_explore_exact2_camera_all5_20260921T150440Z`.
The latest viewer recording is `recording_20260921_171015_606607553`.
Workstation and local code both report revision
`9213976b91a5c856c4f75eee724635e9ffed426e`.
The read-only artifact snapshot was taken at **15:13:20 UTC**. Times below are
UTC; add two hours for local CEST.

The observed failure is reproducible from saved scans without ROS or concurrent
callbacks. The main cause is scan-boundary fragmentation affecting candidate
association. The mission also terminates its camera observation on the first
valid QR-only result, which makes obtaining a facing angle sensitive to the
first frame's geometry outcome. These records do not establish a thread race.

## Mission: immediate identity admission, unavailable facing angle

The first inspected candidate was `survey_candidate_0003`:

| Event | UTC |
| --- | --- |
| Approach completed | 15:09:11.820 |
| Camera observer started | 15:09:18.227 |
| Observer's own TF buffer became ready | 15:09:20.289 |
| First tuple's exact-time lookup exhausted retries | 15:09:20.449 |
| `QR_003` observation pose committed | 15:09:20.771 |

There were two synchronized tuples, but only **one processed detector image**.
That image committed QR discovery about **372 ms after its camera stamp** and
2.54 seconds after observer startup. The observer exited normally with code 0;
its process evidence records an artifact completion and no signals or deadline
expiry. The parent records `confirmed_unique_qr`, **1/5 confirmed identities**,
and `facing_ready=false`.

The detector found six measured head proposals, all rejected by
`CurrentScanHeadProposalFilter` with
`ambiguous_registered_camera_clusters`. Replaying those six recorded bearings
against the original scan reproduces every rejection. Each camera-centred cone
contains two eligible fragments:

- Index `[0]`, at approximately 0.552 m.
- Indices `[223,224]` or `[222,223,224]`, at approximately 0.564–0.576 m.

The scan has 225 samples. Its endpoint metadata differs from its indexed
geometry by 0.565 sampling steps, so the existing topology contract disables
joining the last and first samples. The head association requires one eligible
cluster and rejects the two fragments. This is consistent with one stand
straddling the scan boundary; closeness alone is not authority to merge returns.

The ordinary map-centred QR association sees a single eligible cluster
`[222,223,224]` and accepts the independent `QR_003` decode. The different cone
centres explain why QR identity succeeds while head association fails.

The recorded immediate-front admission reason is
`current_complete_associated_head_fit_required`. Its required axis and QR
sample counts are both **one**. Waiting for seven axis samples was not the
blocker in this capture.

The QR-only fallback has `delay_sec=0`. Status publication tries the immediate
front result, then bounded head evidence, then QR-only discovery. The first
two decline and the last succeeds; `commit_qr_observation_pose` marks the
observer completed. No subsequent frame can improve the facing estimate in
that observer. This is deterministic result-selection policy, with timing
sensitivity to which scan/image arrives first.

Relevant code:

- `real_robot/observer/current_scan_head_proposal_filter.py:94`
- `perception/candidate_lidar_association.py:374`
- `real_robot/observer/node.py:3045`
- `real_robot/observer/node.py:3323`
- `real_robot/observer/qr_observation_pose.py:121`

## Viewer: deterministic scan-dependent flicker

The recording contains **120 distinct camera timestamps**, spanning
15:10:15.548–15:10:25.520, approximately 55 seconds after the mission observer
had already completed. It is a separate observation-only viewer configured
with `head_target=nearest`, `no_qr_decode=true`, and no observation output path.
Its green overlay certifies current image geometry; it cannot publish a mission
candidate admission in this configuration.

| Recorded geometry outcome | Frames |
| --- | ---: |
| Nearest scan candidate ambiguous | 82 |
| Current measured head accepted | 20 |
| Multiple head proposals | 10 |
| Head acquisition verification budget exceeded | 8 |

All **82 nearest-target ambiguities** involve the two nearest groups on opposite
ends of the original scan array. Their range difference is a median **17.1 mm**
and at most **24.3 mm**, below the viewer's 30 mm ambiguity threshold. In 71 of
these frames, endpoint metadata is inconsistent; in 11, the seam is not one
sampling step. Circular joining is therefore disabled in all 82.

Replaying the unchanged `nearest_scan_head_search` on every saved scan reproduces
all **120 target-selection reasons and candidate source-index groups**. The
replay uses the recorded detector-completion time reconstructed on the receipt
wall clock for freshness checking; it is a scan-gate replay, not a new image
detector benchmark or an end-to-end mission replay.

The viewer resets its pose tracker whenever nearest-target selection fails.
When the scan again offers an unambiguous target, it must reacquire current
borders; ambiguity or the 12-verification budget can then reject the image.
This produces the visible accepted/rejected alternation in an almost stationary
scene. There is no need for concurrent frame mutation to reproduce the main
selection failures.

Detector time was median 6.0 ms, p95 188.5 ms, maximum 226.9 ms. Many very fast
frames had already failed target selection before image geometry was attempted.
116/120 result-freshness checks passed. Processing latency is a secondary issue
in this recording, not the explanation for the 82 target ambiguities.

Relevant code:

- `perception/stand_axis/nearest_scan_head.py:79`
- `perception/scan_topology.py:70`
- `perception/debug/stand_axis_viewer.py:3429`

## TF optimization result and remaining startup loss

The preflight service was reused across the recorded mission preflights under
PID 15724. All 11 captured stationary-window consistency checks had zero
reacquisitions and no timeout; their wait measurements were **0.084–0.094 ms**.
The new preflight synchronization is not the observed admission bottleneck.

The camera observer still owns a separate, short-lived TF buffer. It spent
2.06 seconds reaching startup readiness. Its first tuple requested a transform
about 1.8 ms earlier than the earliest stored sample, exhausted a roughly
153 ms retry, and was replaced by a fresh tuple. Fourteen scan witnesses expired
before exact TF was available. This is a genuine startup timing inefficiency,
but the following processed image was fresh and committed successfully.

## Recommended changes

1. Investigate the scanner's published sample count, angle increment, and endpoint
   metadata. Preserve raw geometry and correct its production contract if it is
   wrong. If legitimate boundary gaps must be supported, introduce an explicit
   spatial/temporal fragment-association rule shared by viewer and mission;
   preserve genuine multiple-object rejection. Increasing global ambiguity or
   freshness tolerances would conceal the evidence problem.
2. If a facing angle is desired before leaving the stop, give the existing
   same-stop geometry grace a short bounded mission-level setting. The immediate
   complete-head path should still finish on its first valid frame; QR-only
   fallback should require a newly fresh, associated decode after the grace.
   Waiting alone is not demonstrated to resolve this run's association failure.
3. Preserve a warm exact-time camera TF ingestion path across stops, and drop
   startup tuples already older than the buffer's available history. Do not
   substitute latest-time transforms for camera/scan timestamps.

At the snapshot the second candidate had motion-start evidence but no camera
observer artifacts or terminal result. No matching mission process appeared in
the 15:14:24 UTC process inspection. The saved records do not establish why that
execution ended; the viewer recording does not prove a second mission observer
was running or waiting on camera admission.

## Reproduction and scope

Local artifacts are under
`results/implementation_checks/camera_admission_audit_20260921T150440Z/`:

- `source_snapshot.json`: retrieved text artifacts with SHA-256, size, and mtime.
- `analyze.py`: deterministic scan-gate replay and timeline extraction.
- `audit_summary.json`: computed counts, timings, and per-frame replay evidence.
- `viewer_annotated.avi` and `viewer_frame_000.png`, `viewer_frame_023.png`,
  `viewer_frame_029.png`: original annotated recording and extracted frames.

The investigation itself changed no production code, workstation files, ROS
state, or robot motion.

## Implemented corrections

The autonomous mission now forwards `--qr-pose-fallback-delay-sec`, default
**1.5 seconds**, to every camera observer. The window starts at the first fresh,
independently associated QR decode at the current stopped pose. Geometry may
complete immediately, including on the first frame. Otherwise, QR-only
completion requires another fresh associated decode after both the sensor and
monotonic clocks cover the window. Centering and inspection-progress advisories
cannot end observation during this window. The overall camera timeout remains
unchanged; if fresh evidence does not return, expiry of the grace alone cannot
authorize an artifact. Target, motion, calibration, and identity checks remain
in force. The standalone observer retains its explicit zero-delay default.

The previous mission command automatically receives this behavior. An explicit
`--qr-pose-fallback-delay-sec 1.5` is optional; zero disables the grace and values
outside the finite 0–10 second range are rejected by the mission parser.

Endpoint-gap eligibility now lives in the shared perception module
`perception/scan_endpoint_fragments.py`. Mission association continues to require
three independent earlier valid scan witnesses before interpreting eligible
fragments as a single target. Raw scan angles, topology validation, eligible
cluster counts, and current sensor freshness gates are preserved.

The viewer uses the same eligibility rule in a separate
`stand_axis/endpoint_search_hint.py` module. Two nearby endpoint fragments can
supply one bounded **image search region** when their physical extent and
endpoint separation satisfy the existing spatial limits. They remain two raw
groups, explicitly marked `target_uniqueness_proven=false`. This is neither
LiDAR continuity proof nor camera geometry evidence: current borders must still
be measured and verified. Ordinary two-object ambiguity, competing third
targets, stale scans, missing endpoints, and invalid gap metadata still reject.

Replaying all 120 recorded scans preserves their original candidate index
groups. All 82 former endpoint-ambiguity cases now supply a search region; the
other 38 retain their prior successful selection. This is a scan-selection
replay, not an image-fitting benchmark or proof of successful physical mission
admission. The mission saved only one processed camera frame, so it cannot
demonstrate which future geometry would have been admitted during the new grace.

Validation: **199 tests and 117 subtests passed**, covering endpoint and internal
fragment witnesses, original topology, registered association, viewer search,
CLI forwarding, QR fallback, immediate front geometry, backside handoff,
centering, and bounded-head observation. No selected tests were excluded. The
QR producer test's outdated dummy OpenCV/model inputs were updated so its real
processing path could execute. A regression demonstrates a failed geometry
frame remaining active after its QR decode, then completing with current
geometry 0.4 seconds later, inside the grace window.

The representative recorded scan is retained as
`tests/aufgabe04/fixtures/endpoint_search_20260921.json`. Full local replay output
is `optimized_target_replay.json` beside the original audit snapshot; run
`PYTHONPATH=. python results/implementation_checks/camera_admission_audit_20260921T150440Z/replay_endpoint_correction.py`
from the repository root to reproduce it. Validation ran offline with Python
3.14/OpenCV 5; ROS Humble hardware validation remains outstanding. No workstation
deployment or robot motion was performed. The separate cold camera TF buffer
startup inefficiency described above is unchanged by this patch.
