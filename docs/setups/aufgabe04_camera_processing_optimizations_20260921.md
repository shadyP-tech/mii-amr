# Modular camera-processing optimizations — 21 September 2026

Implemented against `18644ff`, following the camera/TF latency audit. No robot
motion, deployment, freshness-limit change or calibration change was performed.

## Modules and integration

| Module | Responsibility |
| --- | --- |
| `perception/stand_axis/head_search_inputs.py` | One frame's extraction cache for recursive cold searches |
| `qr_scanning/qr_work_budget.py` | Bounded backend/preprocessing cost history and admission before atomic work |
| `qr_scanning/qr_decoder_resources.py` | Single-owner native/WeChat instances; no observation evidence |
| `real_robot/observer/debug_output.py` | Bounded asynchronous debug images and metadata |
| `real_robot/observer/tf_startup.py` | Readiness of this observer's own TF buffer before pinning a sensor tuple |

Paths in this table are relative to `scripts/aufgabe04/`.

Cold acquisition shares masked locator edges, contours, grayscale, LSD,
Hough, and distance-transform results across the original, size-prioritized,
and guided search orderings. Frame, raw-edge and region identity must match.
Candidate screening, verification, ambiguity resolution and association still
rerun under their existing deadlines. The cache ends with that image's call;
it does not hold a selected head, pose or QR identity across images.

QR decoding checks measured operation costs before launching more backend or
preprocessing work. A bounded 32-entry history uses recent peak cost, a 20%
margin and 2 ms reserve; upward image-size scaling is conservative. Unused
forecasts expire after eight full-decode calls, allowing recovery from transient
slow calls. Every eighth call also permits probing an unmeasured larger size,
so size extrapolation cannot permanently exclude the only decodable variant.
The first unmeasured operation remains a cooperative probe and may
overrun; this is not a hard real-time cutoff. Each call reports measured and
declined operations in decoder provenance under `work_stages`.

The generator's deadline is checked before advancing to another image variant,
so an already-expired call does not begin resizing. Isolated-symbol recovery
has its own measured admission. The observer reuses native and WeChat objects
on its processing owner thread, while payloads, corners and native symbol
counts remain call-local. Failed backend construction can retry on a later
call. Existing native checks and independent periodic full QR probes remain.

The debug writer owns one active and one replaceable pending snapshot. Arrays
and metadata are detached before submission; identical image objects are copied
and PNG-encoded once per snapshot. Image data is capped at 16 MiB per snapshot.
Disk errors, overwritten pending snapshots, write duration and worker state are
reported in observer status. Shutdown waits at most two seconds. The writer
only receives diagnostic images and metadata. Status files, status-event
history, and authoritative motion-neutral recommendations/receipts retain
their synchronous durable publication and source-age rechecks.

Startup checks the actual observer buffer for map/base, map/scan and
base/camera connectivity through zero-timeout queries. Until ready, it does
not select and hold an image tuple, and reports missing chains at most once
per second. Readiness never substitutes latest TF for image-time TF: normal
exact image/scan lookups still run afterward. The existing parent observation
deadline bounds startup as before; no timeout is extended. This reduces retry
churn but does not repair an absent broadcaster or eliminate process startup.
The existing final preflight TF reacquisition fix remains unchanged.

## Validation

The regression selection passed 570 tests and 612 subtests, with one existing
skip and two explicitly deselected baseline failures. Coverage includes cold
head acquisition, current-head association, QR recovery and ambiguity,
ingestion, delayed publication, TF retry, and final preflight reacquisition.
Additional cache/resource tests verify that one image cannot donate evidence
to another, a slow backend is not restarted into an insufficient budget,
diagnostic backlogs stay bounded, and startup cannot pin an old sensor tuple.
The skip requires the deployed WeChat backend, which this local OpenCV lacks.

The two pre-existing failures were reproduced by loading the original HEAD
observer source before running the tests:

- `test_qr_observation_pose.py::QrObservationPoseTests::test_processing_head_acquisition_failure_still_commits_current_qr_pose`:
  its injected `object()` camera backend cannot provide
  `initUndistortRectifyMap` required by the existing source-support path.
- `test_real_robot_pipeline.py::PassiveObservationCoreTest::test_operational_observer_wires_only_strict_backside_handoff`:
  the source-string assertion still requires `select_camera_target_measurement(`,
  which is absent from the original observer source as well.

These tests were not modified or treated as successful checks.

All changed/new Python files parse with Python 3.10 grammar. `git diff --check`
passes. Tests use the existing local `/tmp/a04-inspection-audit-venv` environment
(OpenCV 5.0.0); they do not establish physical ROS/Humble performance.

## Local timing evidence

On recorded backside frame 4, alternating 15 measured cached and uncached
calls after warmup produced identical complete acquisition results and
diagnostics. Each of contours/grayscale/LSD/Hough/distance-transform extraction
ran three times uncached and once cached.

| Extraction-cache ablation | Median | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| Cache disabled | 111.62 ms | 109.83 ms | 117.87 ms |
| Cache enabled | 72.92 ms | 71.78 ms | 77.63 ms |

Median reduction: 34.67%. This is a local macOS/OpenCV 5 replay of one
three-pass cold search, not the entire detector or a robot throughput claim.
The original candidate prior is deliberately retained to exercise all three
search orderings. Production's stopped-LiDAR reconciliation remains in place.

Reproduce from the repository root:

```sh
/tmp/a04-inspection-audit-venv/bin/python results/implementation_checks/camera_optimizations_20260921/benchmark_head_search.py
```

That directory also contains `head_search_benchmark.json` and the regression
output `tests.txt`. The next physical run should compare the existing source-age
and TF-delivery metrics, new QR `work_stages`, and `debug_writer` counters without
relaxing the 0.5-second sensor freshness limit.
