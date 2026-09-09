# Aufgabe04 improvement decisions — 9 September 2026

Implementation follow-up: the reproduced handoff/perception fixes, checked
identity/catalog adapters and optional camera pilot checkpoint are recorded in
[the implementation note](aufgabe04_handoff_fixes_20260909.md). The decisions
below preserve the assessment before those changes; hardware evidence and full
logistics/loaded/fleet runtime remain outstanding.

The feedback identifies real remaining work. Implement the reproducible camera,
LiDAR and integration defects; use hardware measurements to decide calibration
and architecture changes. The first physical milestone remains one unloaded
candidate approach, stopped observation, target-bound QR identity and validated
facing pose. Full Aufgabe04 additionally requires ordered jobs, puck transport
and two-robot coordination.

This assessment compares the [readiness audit](aufgabe04_pipeline_readiness_audit_20260909.md)
with the current checkout, including the preceding TF hardening. It adds no
production changes and performs no server calls or robot operations.

| Area | Decision | Next concrete deliverable |
| --- | --- | --- |
| Execution TF handoff | Implemented locally; validate next | Actual ROS listener/executor integration and repeated physical transitions using the existing receipt tracing. No additional timeout/freshness relaxation. |
| Camera timing | Implement focused fixes now | Final source-age check before authoritative commit; decoder reuse for identical retry pixels; bounded raw evidence and complete timing. |
| Camera geometry | Validate before further algorithm changes | Known-angle, synchronized original-image matrix comparing current production and diagnostic geometry. |
| LiDAR seam | Implement guarded shared fix now | Full-scan-aware endpoint adjacency in discovery and camera association, tested at real angular spacing and with partial/dropout scans. |
| Identity and arrival catalogs | Required; implement in parallel | Authoritative QR-to-server-station binding and checked conversion into the existing frozen arrival catalog. |
| Single-robot logistics | Required next integration stage | Real initial QR → server task → certified navigation → stopped expected QR → acknowledged job transition. |
| Loaded and fleet execution | Required later stages | Integrate measured carrier/load limits, custody, live coordination and physically evidenced station release. |

**TF: keep the completed changes and validate their actual runtime.**

The preceding changes already include bounded stopped acquisition, structural
validation before waiting on a first old global sample, common recovery-reason
binding and receipts at the executing buffer. Its 287 focused tests passed;
the integration fixtures use ROS-free fakes. This supersedes the older audit's
82-test snapshot for those changed modules, but proves neither DDS behavior nor
hardware success. [Run audit and validation scope](aufgabe04_run_audit_20260909T123820Z.md).

Use a ROS-enabled integration test and then the proposed ten physical
transitions as an initial engineering acceptance series. Record every attempt,
including failures and interventions. A timer heartbeat or passed preflight is
insufficient: the same executing listener must acquire valid fresh edges and
the child must reach stopped camera observation. Ten successes would support a
milestone, not establish a general reliability rate. Retaining listener lifetime
across children is a later option if timing identifies startup as dominant.

**Camera: there are two additional code defects worth fixing before the trial.**

First, the operational freshness check occurs after detection, while later
association/evidence work and synchronous debug writes happen before the final
recommendation/backside commit. `_write_debug` can write seven PNGs and JSON.
The creation timestamp subsequently used by generic recommendation freshness
does not substitute for source-image/scan freshness. Add a final source-age
check immediately before each authoritative commit, retain source timestamps,
and record precommit age. Inject slow debug/association work in tests and require
expired evidence to produce no authoritative result. Move expensive diagnostic
encoding off the critical path without removing evidence needed for review.
See [observer commit path](../../scripts/aufgabe04/real_robot/observer/node.py),
[recommendation construction](../../scripts/aufgabe04/real_robot/configuration/recommendation.py)
and [recommendation freshness](../../scripts/aufgabe04/navigation/approach/viewpoint_recommendation.py).

Second, a registered strict retry deliberately uses the proposal's exact ROI,
but full QR decoding is repeated. Reuse the decoded observation within one
processing invocation, keyed by image, exact crop and decoder mode. Preserve
empty results, multiple symbols, text/corner identity and crop coordinates;
rerun strict geometry with its corrected centre and corridor. Tests must prove
one decode for identical proposal/retry pixels and a new decode for changed
pixels/crops/modes. A native miss must not suppress full acquisition.
See [registration](../../scripts/aufgabe04/real_robot/observer/camera_target_registration.py)
and [strict retry ROI](../../scripts/aufgabe04/real_robot/observer/head_roi_reacquisition.py).

The 12/19 images older than 250 ms at receipt are confirmed. Their minimum,
median and maximum ages were 114.521, 265.399 and 396.548 ms. The interpretation
needs precision: tracker renewal uses completion-minus-image age ≤250 ms;
prediction compares image timestamps; operational result freshness has a
separate default 500 ms limit. Those twelve frames could not renew the tracker
if the measured timing persisted, even with zero processing cost. They do not
prove tracker starvation or invalid final publication in this run, because the
observer never started. [Tracking contract](../../scripts/aufgabe04/perception/stand_axis/pose_tracking.py),
[timing contract](../../scripts/aufgabe04/real_robot/readiness/sensor_timing_contract.py).

Extend existing status/stage diagnostics with capture→receipt→start→completion→
commit timing, per-attempt decoder/cache data and received/paired/fresh/refined/
associated/consensus counts. Preserve a bounded set of original compressed
frames, calibration, TF, ROI and corners for successful and failed observations.
Set frame/byte limits and capture-drop counters; avoid another synchronous PNG
writer. Existing status history and latest debug images are useful starting points.

Verified-only tracking, absolute stale-result checks, bounded native tracked
decoding, per-evaluation text/corner reuse, residual diagnostics and refined
overlays already exist. Their reimplementation is unnecessary. Validate the
78 mm head, 71 mm panel and 62 mm QR border associations on the same original
images before changing geometry. Compare production yaw/heading with the
viewer's full-normal geometry across known headings and measured pitch/roll.
QR-free backside detection intentionally has an undirected yaw and no full
normal; requiring one indiscriminately would disable that path. Neither a
longer expiry nor the viewer's angular tolerance should be copied without evidence.

**LiDAR: repair both clustering paths, with a conservative circularity contract.**

The seam defect reproduces using the actual scan geometry:

| Scan geometry | Interior two-return stand | Equivalent endpoint stand |
| --- | --- | --- |
| 360 rays, 1° | One candidate | No candidate |
| 224 rays, 1.607° | One candidate, 53.29 mm width | No candidate, 54.68 mm geometric width |
| 222 rays, 1.621° | One candidate, 53.76 mm width | No candidate, 56.11 mm geometric width |

These are synthetic two-return probes using recorded scan geometry, not
ground-truth stand detections from the physical run.

Camera target association also splits the endpoint object; its registered-camera
path can report two ambiguous clusters even when minimum samples is one.
Share endpoint-adjacency logic between
[discovery](../../scripts/aufgabe04/perception/lidar_stand_detector.py) and
[association](../../scripts/aufgabe04/perception/candidate_lidar_association.py).
Carry original scan size/angular metadata and a validated full-rotation/profile
contract; bare filtered point lists remain linear. Require valid indices 0 and
N−1, a seam gap compatible with one sampling step, and each caller's existing
distance/range/cone checks. Represent wrapped membership explicitly rather than
pretending it is an ordinary contiguous start/end interval.

Recorded seam gaps range from 1.026 to 2.026 beam steps. A broad near-360° test
can bridge a missing sector; exact floating-point equality can reject every
real scan. Test a conservative documented tolerance, partial FOV, missing
endpoints/interior points, two-step gaps and malformed/overlapping metadata.
Add received/drop/TF-timeout and latency summary counters using the observer's
existing warning paths. Correct misleading one-degree/head-width commentary.

Keep morphology thresholds pending measured laser-plane cross-sections, range,
incidence and clutter data: the current 0.12 m diameter is a navigation-radius
proxy. Extra adaptive viewpoints remain a separately specified mode after
perception validation; preserve the defined exact-two checkpoint and all
unresolved candidates as keepouts. Modelled coverage is not measured stand visibility.

**Logistics: reuse the existing contracts, then add the missing adapters.**

An offline call through the actual logistics CLI reproduced rejection when
discovery's `station_QR_001` met the server decoder's `STATION_QR_001`.
Case is only part of the defect: server station names need not be derived from
QR text. Join observations to the authoritative robot-specific `qr_mappings`,
apply one explicit station-ID policy and seal the result with existing registry
construction. Reject absent, duplicate or conflicting mappings.
See [discovery identity creation](../../scripts/aufgabe04/real_robot/candidate/approach.py),
[server decoding](../../scripts/aufgabe04/task_client/server_response_decoder.py)
and [registry creation](../../scripts/aufgabe04/stations/create_station_identity_registry.py).

The custom facing catalog has no production consumer. Build its explicit
adapter into `ArrivalPoseCatalog` using the existing recommendation conversion,
fixed-target/corridor validation and freeze/provenance flow. The bare converter
sets clearance flags; calling it alone would manufacture validation. Retain
the full hypothesis pool for obstacles while binding confirmed identities to
arrival records. Test stale/wrong-map/calibration evidence, registry mismatch
and unresolved keepouts. [Existing validated recording flow](../../scripts/aufgabe04/navigation/missions/plan_synchronized_viewpoint.py).

`run_logistics_mission.py` is dry-run-only, `MissionController` has no production
instantiation, and the two-robot entry point is a stub. Reuse their pure policies
and the existing sole navigation motion owner. Supply real navigation completion,
strict post-dispatch QR events, server acknowledgement and persisted progress.
Test repeated station jobs, wrong/stale QR, failed legs, delayed/duplicate
responses and restart. The current scan-report client uses GET despite a POST
comment; establish actual server advancement/idempotency semantics before
designing retry behavior around that comment.

Loaded execution needs measured carrier geometry/retention/stopping evidence
before connecting loaded envelopes and custody to navigation. Fleet integration
must enforce station occupancy, right-before-left and puck ownership from live
status. Expired leases cannot establish physical clearance. These are required
by [Aufgabe04](../../Tasks_04.md), but do not unblock the first camera observation.

**Make the one-candidate milestone explicit.**

The current top-level modes do not offer stop-after-one-camera-candidate.
`--max-candidate-inspection-views` bounds views per candidate, while
`--expected-stand-count` asserts the physical site's five-stand contract.
For repeatable bounded pilots, a small per-candidate checkpoint/stop control is
worth adding through existing orchestration and permits. Preserve the five-stand
goal and full obstacle pool, record the pilot result separately from mission
completion, and stop only after the validated candidate receipt or a terminal
failure. Do not obtain this behavior by truncating candidates or setting the
site count to one. [Modes](../../scripts/aufgabe04/real_robot/mission/modes.py),
[CLI](../../scripts/aufgabe04/real_robot/autonomous_runner/cli.py).

Recommended sequence: final camera-age guard and exact-pixel decoder reuse;
bounded production evidence and guarded LiDAR seam repair; repeatable one-candidate
physical validation; five-station discovery; unloaded ordered logistics; measured
loaded transport; two-robot operation. Identity/catalog work can proceed in
parallel with sensing validation. A compact run index can reuse existing session
IDs, manifests and hashes; keep physical, simulation, replay and unit-test results distinct.

Verification in this assessment: 39 LiDAR tests, 77 camera tests with 59 subtests,
and 46 logistics/registry/catalog/puck/fleet component tests passed; an independent
geometry review also passed 42 tests. These scopes overlap and are not summed.
The seam and identity failures were reproduced through public/offline entry points.
These checks do not certify physical angle accuracy, transport latency or full-task
execution. Detailed source fingerprints and validation scope are retained in
`results/audits/aufgabe04_improvement_decisions_20260909/assessment.json`.
