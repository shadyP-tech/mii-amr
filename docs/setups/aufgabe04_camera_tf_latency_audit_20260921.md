# Camera exploration and TF latency audit — 21 September 2026

Inspected checkout: `18644ff`. Latest recorded robot run inspected:
`stand_explore_exact2_camera_all5_20260921T131738Z`, revision `56892e3`.
Also compared the available 18 September observer logs. This is a read-only
runtime analysis; no robot commands or production changes were made.

The camera does spend substantial time processing each observation, but the
latest long failure is not explained by camera results becoming older than
usable TF. Image computation, TF availability, and failed target acquisition
are three different sources of lost time in these records.

## Measured timing

The second candidate on 21 September has 492 status events spanning 89.39 s:
2,657 received images, 375 synchronized tuples, 373 processed images, 373 fresh
detector results, and zero admitted candidate frames. Effective processing is
4.17 Hz against a configured 5 Hz. Skipping intervening camera frames is
intentional: the observer selects the latest image rather than queuing jobs.

| Measurement | Median | p95 | Maximum |
| --- | ---: | ---: | ---: |
| Image stamp to callback receipt | 49.9 ms | 63.3 ms | 75.3 ms |
| Callback receipt to processing start | 26.0 ms | 58.9 ms | 93.0 ms |
| Image age at processing start | 72.4 ms | 103.9 ms | 114.6 ms |
| Instrumented detector interval | 122.2 ms | 293.9 ms | 323.3 ms |
| Independent head acquisition stage | 93.3 ms | 123.1 ms | 215.6 ms |
| Full QR decoding, 88 calls only | 160.2 ms | 176.6 ms | 179.1 ms |
| Image age at detector completion | 198.7 ms | 373.0 ms | 409.3 ms |
| Detector completion to status timestamp | 22.6 ms | 23.4 ms | 25.8 ms |
| Recorded odom-to-base stamp to receipt | 7.8 ms | 22.1 ms | 52.9 ms |

Percentiles are computed per column; they are not additive. Detector, head,
and QR intervals overlap hierarchically. The final row uses 1,436 unique
receipts retained in bounded TF snapshots, not every published transform.
Source-to-receipt values include clock offset as well as transport/scheduling;
they are not pure network latency. The post-detector interval includes
association/evidence/debug work but excludes serialization and fsync of that
status event itself, since its timestamp is taken before those writes.

Reproduce from the repository root with
`python3 results/implementation_checks/camera_tf_latency_audit_20260921/analyze.py`.
The accompanying `timing_summary.json` contains measurements for all eight
available observer event logs. p95 uses the nearest-rank convention.

## Findings and priorities

### 1. Repeated full-image head acquisition is the largest steady compute cost

`real_robot/observer/viewer_head_acquisition.py:114` sends the full rectified
800×600 image through shared geometry acquisition. In
`perception/stand_axis/head_cold_acquisition.py:659`, a miss recursively retries
with size-prioritized rails and then guided rails. Every one of the 373 frames
in the long failed inspection records both retries: three search passes.

The recursive calls reuse raw Canny edges but repeat other work, including
contours, grayscale/line extraction, rail handling, and candidate screening.
`_rail_endpoint_hints` runs full-image LSD and Hough before filtering their
results. Independent acquisition accounts for 36.23 s of the 58.70 s summed
detector time. This is a stronger optimization target than Canny itself, whose
median recorded edge-preprocessing stage is only 1.59 ms.

First optimization: retain frame-local contour/line/rail intermediates across
the three search orderings. Preserve the current full-image sampling: the code
explicitly documents that cropping changes LSD's subpixel endpoints. Any later
ROI or downsampling change needs geometry/association regression evidence.

The underlying failed search matters more than micro-optimization: all 373
frames returned `model_current_head_border_unavailable`. The existing second
candidate audit identifies a displaced candidate prior. Current HEAD adds
bounded stopped-LiDAR search reconciliation, but that repair has no new recorded
robot run. The old 89-second failure is not a benchmark of that repair.

### 2. QR's 120 ms budget is cooperative and is exceeded in practice

`real_robot/observer/qr_acquisition_policy.py:18` allows one full crop per image,
normally probing once per second without a positive marker signal.
`qr_scanning/opencv_qr_detector.py:27` checks its elapsed budget between backend
calls; a running OpenCV call cannot be interrupted. Candidate generation also
resizes/adds borders before the outer loop checks the deadline again.

All 88 full decodes in the long failed inspection exhausted the 120 ms budget.
Median overrun was 40.2 ms, maximum 59.1 ms. Full decoding consumed 14.18 s;
total identity work, including native checks, consumed 16.53 s. Ninety-one of
373 detector passes exceeded the 200 ms period of the requested 5 Hz schedule.

First optimization: avoid starting enlarged/backend variants without enough
remaining budget based on measured per-stage cost, and cache reusable decoder
objects separately from image evidence. Add per-backend and preprocessing
timings before deciding which variants to remove. Preserve independent periodic
QR discovery when head fitting misses. A worker process with bounded job/result
slots is an option if a hard compute cutoff is required; another timer cannot
make an atomic OpenCV call meet a hard deadline.

### 3. Debug images and durable status writes remain on the processing thread

`real_robot/observer/node.py:2904` synchronously writes up to seven PNGs per
processed frame. The autonomous runner always supplies `--debug-dir`
(`autonomous_runner/runtime.py:1521`). In the viewer path, frame and selected
ROI can both be the full image. Debug metadata, status JSON, and status-event
JSONL use synchronous serialization/fsync (`node.py:349`, `node.py:373`).

The 22.6 ms median post-detector interval is a measured combined cost, not an
isolated PNG benchmark. Its sum is 8.45 s over the long inspection; the event
log alone is 19.78 MB. Actual write costs require separate spans around encoding,
serialization, and fsync. Raw capture history already has a bounded writer
thread; moving it again would not address synchronous debug/status work.

First optimization: sample or asynchronously write diagnostic PNGs and verbose
trace snapshots through bounded latest-only storage. Keep authoritative receipt
publication durable and retain its final source-freshness check. Do not allow
an asynchronous output queue to accumulate old images.

### 4. TF waiting is primarily startup/availability, not detector starvation here

The observer's dedicated ingestion thread services sensor callbacks and its TF
buffer (`node.py:3465`, `observer/ingestion_runtime.py:92`). The main owner thread
does detection and periodic exact-time retries. TF lookup is nonblocking and
uses the image or scan timestamp, not the eventual processing time
(`node.py:1004`, `node.py:1262`). Ingestion still shares process CPU/GIL resources,
but the records do not establish significant TF starvation from that contention.

In the long inspection, 116 tuples needed a retry, 114 subsequently reached
processing, and only two exhausted the retry budget. Both failures were at
startup: missing `base_footprint`, then a query older than the local TF history.
Waiting longer cannot recover a past transform that was never retained.
The buffers recorded 145 overwritten ingress images; this reflects bounded
latest-image buffering during slower passes, not a growing detector backlog.

An earlier `20260918T143813Z/.../recenter_01/capture` observer was more severely
affected: 28 of 30 synchronized tuples exhausted TF retries during 6.69 s.
The errors say the `map` frame does not exist. Odometry receipts continued while
map-to-odom receipts were absent from this observer. Only two detector passes
ran, totaling 335 ms. That delay cannot be attributed to slow image computation.
These logs do not isolate why map TF delivery was absent (publisher, discovery,
transport, or another cause).

First improvement: measure and establish TF history readiness within the actual
observer instance before spending its useful inspection budget, and distinguish
missing frames, too-old queries, and future extrapolation. The runner currently
starts a new observer process for each capture (`autonomous_runner/runtime.py:1535`),
so another preflight node's ready buffer does not make this new buffer ready.
Consider a persistent ingestion owner across stopped inspections, with explicit
evidence/target epoch resets. Preserve exact-time lookup and bounded waiting.

### 5. The later driving preflight failure is a separate snapshot-ordering issue

The previously recorded recovery failure compared a final map-to-odom sample
with a newer stationary-window sample and found the final stamp 100.995 ms
older. Current HEAD already services callbacks and reacquires the final sample
up to five times (`navigation/localization/ros_preflight.py:1748`). It still
rejects persistent lag. The earlier audit documents the evidence; the new
implementation has not been confirmed by a subsequent physical run.

The waypoint follower also has its own dedicated TF listener/executor
(`navigation/waypoint_follower/runtime.py:173`, `runtime.py:821`). Camera
inspection is a passive stopped phase, with an explicit stationarity check
before detection (`observer/node.py:1323`). Camera latency mainly delays or
prevents a fresh motion recommendation; it does not directly set the follower's
control-loop frequency. A transform for an older image is correct when sampled
at that image's timestamp. Substituting latest TF would introduce motion error.

## Verification and limits

38 existing offline tests passed across ingestion scheduling, cooperative QR
budgeting, head scheduling, publication freshness, and exact-TF retry handling.
The environment has no pytest, so the separate pytest-based final-TF
reacquisition tests could not run here. No new tests or production fixes were
introduced. The timing analysis uses saved robot measurements, not local-Mac
OpenCV runtime as a proxy for robot speed.

For the next instrumented physical run, retain image/scan source stamps and
separately measure callback receipt, TF-ready time, each acquisition retry,
each QR backend, PNG encoding, serialization/fsync, and authoritative artifact
commit. Compare their p95 and deadline misses on the same captured scenarios.
The existing 0.5 s freshness checks should remain the acceptance criterion;
increasing that limit would conceal delays rather than reduce them.
