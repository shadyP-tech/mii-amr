# Camera exploration: cross-run investigation, October 1, 2026

The strongest next optimization is to make successful travel reliably lead to
camera acquisition, and to move on promptly when a stopped view produces no
useful evidence. Today's recordings already demonstrate five-identity discovery,
bounded local target reconciliation, approximate-front admission, and reliable
path following separately. They do not yet demonstrate one complete mission
with five identities and return to Start together. Five facing-ready poses remain
a separate downstream requirement if needed.

This investigation checked workstation alias `mii002` (hostname `mii001`) and
independently verified **2,416 original files** against both the local copies and
fresh workstation SHA-256 hashes. Three finished or interrupted October 1 parent
exploration runs were present at the full inventory check. A fourth ongoing run
appeared during the investigation, described separately below. Times are
**Europe/Berlin, UTC+02:00**. Candidate numbers
are local to a run; repeated numbers do not establish physical identity.

| Run start | Recorded revision | Identity discovery | Facing-ready | Motion legs completed | Outcome |
|---|---|---:|---:|---:|---|
| 12:46:18 | `fbe6926` | 5/5 | 3/5 | 8/8 | Exact stored Start endpoint blocked during return planning; no return motion |
| 14:28:01 | `915250e` | 1/5 | 1/5 | 5/6 | Start's opposite-side identity blocked by split scan support; later recovery drive stopped on localization drift, then parent interrupted |
| 15:37:59 | `915250e` | 3/5 | 3/5 | 8/8 | Two candidates rejected before camera startup; terminal `candidate_qr_goal_incomplete` |

All three completed the configured two-view coverage survey at **95.31%**.
Coverage percentage is not a measure of QR or facing completion. The early run
found all five IDs; the final run's visited stands yielded three certified facing
records, including an approximate-front result for QR_001.

The workstation changed during this read-only investigation. The first check
found `915250e`; the fresh full inventory at **16:08:36 Berlin** found clean
**`144f954`**, including `cad847b`'s opposite-side endpoint correction and the
independent **10° passive acquisition** allowance. The recorded files remained
unchanged. Those fixes are now deployed, but these three runs did not exercise
them together. This investigation did not deploy code, modify production code,
or command the robot.

Run `stand_explore_exact2_camera_all5_20261001T141116Z` started at **16:11:16**
on recorded revision `144f954`. A read-only snapshot at **16:15:54** shows one
confirmed identity/facing pose (QR_003), completed 95.31% coverage, and candidate
0001 marked `inspection_started`. The parent has no terminal summary, failure
receipt or exit entry in that snapshot. Treat it as ongoing and exclude it from
the completed-leg and outcome aggregates below. Its live files are not part of
the 2,416-file immutable-source comparison. The captured values are retained in
`pipeline_audit_20261001/new_run_snapshot.json`; they do not establish success
or failure of the combined fixes.

**Candidate localization and eligibility**

The frozen-to-current frame projections are materially different between
selection epochs. In the latest run's last selection, candidate 0005 shifts
**16.03 cm**, 0006 **14.89 cm**, and 0007 **17.89 cm** from its frozen coordinates.
These are coherent map/odometry frame transformations, not measurements of
physical stand movement or proof of a broken localization transform.

Current target support is therefore essential. In the first run, candidate 0005
had a **10.17 cm** epoch displacement. Its ordinary surface-range interval
0.3771–0.5971 m missed the real returns just beyond 0.60 m. Existing bounded
reconciliation recovered one compact four-beam group in each of three fresh
stopped scans at 0.6040/0.6055/0.6040 m; subsequent camera/QR association obtained
QR_001. This is evidence that the current recovery mechanism works. Preserve its
authenticated frozen/current hypotheses, 8–35 cm displacement envelope, current
scan requirements, competitor exclusion and unchanged frozen geometry.

The LiDAR group centroid is a surface observation. Its 4–6 cm offset from a
roughly 6 cm-radius stand hypothesis is not automatically center-localization
error. Keep surface support, fitted physical center, frozen candidate uncertainty
and current route/localization uncertainty distinct. Do not add the full frame
displacement to a covariance or overwrite the immutable center with a surface
mean. The frozen 2 cm uncertainty alone also does not describe all current pose
uncertainty.

A concrete lifecycle defect remains: a temporary static deferral removes the
candidate from the selectable population for the rest of the mission.
Independent replay reproduced all 17 recorded static-admission decisions in the
latest run. Candidate 0007's later projections would pass the same gate:

| Latest-run selection | Candidate 0007 static clearance | Existing 80 mm requirement |
|---|---:|---|
| 000 | 20.54 mm | Reject |
| 001 | 18.02 mm | Reject |
| 002 | 37.46 mm | Reject |
| 003 | 103.22 mm | Pass |
| 004 | 124.36 mm | Pass |

Only the first selection actually evaluated it. In
`real_robot/candidate/approach.py:2518–2541`, `defer_ineligible_targets` removes
excluded IDs from `unresolved`, and subsequent selections intersect with that
reduced set. Candidate 0004, by contrast, stays statically incompatible and has
an unresolved cross-view morphology conflict. Its repeated LiDAR hits do not
make it a real stand.

Implement a bounded queue for **unvisited, static-only deferrals**, reconsidered
when a newly admitted localization epoch is already available. Passing this gate
only restores eligibility; it must still obtain fresh local support and all
normal route/uncertainty admission. Preserve every keepout and candidate hash.
Never treat unchanged epochs, repeated unsuccessful retries or unresolved
morphology conflicts as reasons to loop indefinitely. Candidate 0007 is a lost
inspection opportunity, not a demonstrated missing QR identity.

**Stand admission and camera usefulness**

There are three separate angular quantities: the route controller's final yaw
tolerance (**3°**), the allowed optical bearing for starting camera acquisition
(**6° in the runs, 10° now**), and the head plane's permitted front-view deviation
(**30° including uncertainty** under the coarse-front policy). Increasing one
does not resolve failures in another.

In the latest run, candidates 0006 and 0001 arrived with optical bearing errors
**−6.088° and +7.019°**, although their stopped route errors were within the
controller's 30 mm/3° contract. Both passed range and static target admission;
both have `observer_started=false` and no camera frames from the rejected
arrival. Their identities cannot be assigned from this run.

The deployed 10° correction admits both recorded geometries for passive
acquisition and subsequent measured centering. The remaining centering limits
are 6° per turn, two turns, 12° total. Fresh target association is still required.
This removes their recorded immediate blocker; absent camera frames prevent an
offline claim that they would decode successfully.

Initial calibrated admission runs before the inspection state machine
(`real_robot/candidate/inspection_adapters.py:590`). A rejection consumes the one
episode and is labeled `inspection_exhausted`, even though zero camera views ran
(`approach.py:2504`, `:2814`). Keep bounded acquisition recovery within that local
episode, count actual started observations separately from acquisition/route
attempts, and report the actual failure stage. Do not blindly enable the old
map-derived correction loop: that would be a separate motion change.

The middle run reached the complete front of Start, but **25 fresh tuples** all
contained two LiDAR fragments at the scan-array seam. The retained metric center
was projected correctly. No unique current target could be certified, so live
QR decoding never started, despite readable pixels. This was not an angle or
decoder failure. The now-deployed endpoint correction uses the current isolated
QR outline to validate this bounded fragmentation case while preserving both
raw groups and disabled circular adjacency. Its recorded-frame offline replay
is positive; workstation latency and live opposite-side completion remain to be
verified. Global endpoint merging would remove the evidence distinction.

The biggest remaining camera-time loss is a real stand seen from an unproductive
angle. Latest-run QR_002's first view lasted **90.27 seconds**:

- 362 processed frames, with **337 (93.1%)** lacking a usable head border.
- Only 23 associated frames; four backside samples were separated by
  9.22–23.96 seconds, unable to make seven samples inside the five-second window.
- A changed view then produced a committed recommendation after **two processed
  frames**, in a 2.43-second observer window.

The original images show an oblique rear/side view followed by a readable front:
[first view](../../results/implementation_checks/run_audit_20261001T133759Z/camera_qr002_initial.jpg),
[successful view](../../results/implementation_checks/run_audit_20261001T133759Z/camera_qr002_recovery_committed.jpg).

The existing support-failure timeout concerns absent range support. It does not
cover a present but visually unproductive stand. The five-second inspection
opportunity is a minimum grace before a seven-associated-frame advisory can be
published, not a no-progress deadline. Unassociated frames are excluded by
`observer/inspection_progress.py:169–198`; isolated axis samples clear advisory
history at `observer/node.py:3277`.

Add a separate, fixed **no-useful-progress outcome for the current stopped view**.
It should use fresh processed-tuple diagnostics and be unable to certify an
identity, angle, absent target or motion. It should hand back to the existing
bounded view planner. Sparse isolated axis samples must not renew the view
budget forever. Retain the hard timeout for hung processes. Replay all successful
and failed traces to choose a shorter threshold; this one recording does not
justify declaring any particular five- or ten-second limit safe for every view.

Processing and evidence capture deserve secondary improvements. All 362 processed
tuples in that poor view were fresh (image-age median 0.164 s, p95 0.285 s),
despite 46 exact-TF retry exhaustions among 408 synchronized tuples. This supports
view geometry as the dominant recorded problem; reduce exact-TF delivery losses
without extending freshness limits. The 256-image capture cap omitted the later
152 submitted captures. A bounded rolling archive plus retained transition and
commit frames, including raw scan/TF witnesses, would improve diagnosis of long
failures without requiring unlimited recording or synchronous callback writes.

The 30° coarse-front policy itself worked in the latest run: QR_001 obtained seven
samples with **±17.233°** retained orientation uncertainty. Its strict individual
angle fits remained rejected. Keep the uncertainty-bearing interval policy;
discarding alternate planar solutions or reducing uncertainty numerically would
not improve the measurement.

**Driving, route choice and time**

Independent aggregation of the raw traces finds **21/22 motion legs completed**,
sampled distance to the active segment below **19.90 mm**, and a recorded tracking
tube of **30 mm**. Linear commands peaked at **0.055 m/s**. These are controller
trace observations, not independent ground-truth position measurements.

| Run start | Mission wall time | Motion-leg time | Outside motion legs | Estimated traveled distance |
|---|---:|---:|---:|---:|
| 12:46:18 | 652.0 s | 248.2 s | 403.8 s | 7.67 m |
| 14:28:01 | 567.6 s to interruption | 180.2 s | 387.4 s | 5.08 m |
| 15:37:59 | 810.0 s | 246.6 s | 563.4 s | 7.82 m |

Only **30–38%** of elapsed time was inside motion legs. The rest includes
localization, survey observation, planning, readiness, process startup,
camera work and teardown; it cannot all be attributed to image processing.
Prioritize successful observations and record phase latencies before increasing
speed. The two camera-rejected arrivals alone consumed **3.69 m and 96.71 s**,
about **39% of the latest run's motion time**.

Tighter controller yaw alone would not solve those arrivals: even exact
attainment of the executed goals leaves base-bearing errors −4.842°/+3.016°
relative to subsequently projected targets. Plan terminal optical alignment
using the same current target epoch and calibration where possible, then retain
the stopped acquisition check and evidence-based centering. Candidate ranking
already incorporates route time, turn burden and LiDAR support
(`navigation/approach/camera_candidate_selection.py:409`). Extend it with observed
view usefulness instead of creating a new global planner. Existing view policy
already offers diverse angular hypotheses (`candidate/inspection_policy.py:28`).

Both early runs rejected the first 0.50 m opposite-side standoff during dry
uncertainty admission, with margins **−32.71/−38.79 mm**; the 0.45 m alternative
then completed. Preview the existing uncertainty/endpoint checks across bounded
standoffs before launching expensive dry attempts, retaining final checks. This
is an efficiency opportunity, not evidence that inflation should be reduced.

The middle run's one stopped drive reached 23.34 mm from its goal and was in
terminal heading correction when map–odom yaw drift exceeded its certified
limit: **3.7848° > 3.6820°**. Translation drift remained within its limit. The
existing runtime recovery recorded both a ready handoff and the start of stopped
localization resealing; the parent was interrupted about 3.36 seconds after the
stop. The replacement outcome is unknown. Validate that existing bounded recovery
sequence end to end; do not describe it as absent or relax the stop threshold.

First-run return failure was a continuous/raster mismatch: the exact Start pose
remained 0.341708 m from a 0.34 m candidate keepout, but a fresh projection changed
its grid offset from (−7,−1) to (−7,0). The implemented correction retains the exact
saved goal and all keepouts through a separately certified 4.10 cm final
connection. Offline planning produces a 2.829 m route, with recorded uncertainty
admitting only the first 2.2255 m stage. A complete physical return still needs
new stopped localization and admission for later stages. Later runs never
reached this branch.

**Recommended implementation and verification order**

| Priority | Action | Evidence of success required |
|---|---|---|
| 1 | Exercise deployed `144f954` as one integrated pipeline | Both off-center arrivals actually start the observer; split-endpoint opposite observation yields live identity; exact Start return completes its admitted stages |
| 2 | Add bounded no-useful-progress exit and reuse existing diverse-view recovery | Recorded QR_002 failure leaves its unproductive view before 90 s; successful consensus traces retain enough opportunity; no weakened measurement gates |
| 3 | Reconsider unvisited static-only deferrals on fresh localization epochs | Recorded 0007 becomes selectable at selection 003; 0004 stays excluded; identical epochs and exhausted budgets cannot loop; original keepouts remain |
| 4 | Separate acquisition, observation and route attempt accounting | A pre-camera failure reports zero observed views and its actual cause; bounded corrective opportunities are available without repeated global tours |
| 5 | Improve feasible-view/standoff ranking and phase timing | Fewer rejected route proposals and less travel per admitted stand, while final route and uncertainty checks remain unchanged |
| 6 | Unify terminal reporting and track unfinished geometry explicitly | Identity count, facing count, return outcome and interrupted/failed phase agree across final artifacts |

The configured goal counts five distinct validated identities; it does not require
five facing receipts (`candidate/qr_goal_progress.py:112–127`). QR-only success
ends the local episode. If subsequent work requires all five facing poses, retain
missing geometry as an explicit remaining task after identity discovery. A
five-second grace improves its opportunity but does not guarantee completion.
Track **identity discovery, facing readiness and return-to-Start separately**.

Reporting also needs correction: the first run's survey registry still says three
confirmed while the mission goal correctly says five identities. The interrupted
run lacks a terminal mission summary; the last run has a failure receipt but no
mission summary. Source one terminal report from the existing goal ledger,
preserve interruption behavior, and mark continuing motion unauthorized. Label
the legacy survey summary according to its narrower registry scope.

**Validation and evidence limits**

This investigation reran six relevant modules on local `144f954`: candidate
arrival, autonomous candidate approach, recorded Start return, recorded coarse
front, opposite endpoint integration and observer inspection progress. Result:
**113 tests and 63 subtests passed**, in 38.05 seconds. These tests validate
recorded geometry and local policy contracts, not successful physical runs.

The new reproducible aggregate and fresh inventory are in
`results/implementation_checks/pipeline_audit_20261001/`:
`aggregate.py`, `aggregate.json`, `workstation_inventory.json`,
`source_verification.json`, `recorded_regressions.txt`, and the separately timed
`new_run_snapshot.json`.
Run `python3 results/implementation_checks/pipeline_audit_20261001/aggregate.py`
to recheck copied source hashes against the captured workstation manifest and
recompute the aggregate. It issues no robot or network commands.

Detailed per-run replay and provenance are linked from
[12:46 audit](aufgabe04_run_audit_20261001T104618Z.md),
[14:28 audit](aufgabe04_run_audit_20261001T122801Z.md), and
[15:37 audit](aufgabe04_run_audit_20261001T133759Z.md). Their earlier statements
that fixes were local are historical; the workstation check above supersedes
those deployment statements. No recorded evidence yet establishes that all
remaining failures have been eliminated by the current combined revision.
