# Simplification audit: two-view camera exploration

The findings and baseline results below describe the pre-change audit. The
subsequent local implementation is recorded in the final section; it has not
been deployed or exercised on the robot.

## Contract and inspected baseline

The requested behavior is: approach each candidate; compute its head frame or
usable proposal; attempt QR decoding; accept a front identity or inspect the
opposite side using the retained head angle; on the opposite side decode the
QR without another angle fit. A decoder that was never invoked is distinct
from an actual attempt returning no QR. The latter can justify a presumed
backside / identity-unresolved inspection branch without claiming physical
proof that the face is blank.

This audit inspected clean local and workstation revision
`ef3ef7172d09f7d04858ee0558b8f5d8f0b2771e`, the October 1–2 run evidence, and
recorded regression fixtures. The previous uncommitted arrival correction is
now committed and deployed. It removes the mandatory new eight-scan target
cohort before passive camera capture; it does not remove selection-time or
observer-internal target gates.

The new workstation run `stand_explore_exact2_camera_all5_20261002T121410Z`
started at 14:14:10 Europe/Berlin with this revision. Its completed first
camera attempt provides new direct evidence below. It was active at the
initial inspection. At the final read-only snapshot, **14:54:41 Berlin**, it
had zero recorded IDs, no matching exploration process, and no parent end/
exit marker or `mission_failure.json`. Treat it as an incomplete recording;
these findings do not establish its terminal cause.

No production code, robot process, or workstation artifact was changed by
this audit. Raw recordings stayed on the workstation. Paths below are
relative to the repository root unless stated otherwise.

## Principal finding: QR acquisition discards an accepted head association

The arrival correction worked for the first approached candidate in the new
run: candidate 0003 completed its approach at 14:18:58.843 and entered camera
inspection. Its `camera_lidar_attempt_00` then recorded:

| Quantity | Recorded value |
| --- | --- |
| Processed images | 38 |
| Verified geometry results | 31 |
| Associated frames | 21 |
| Axis-sample frames | 14 |
| Final accepted axis window | 7 / 7 samples |
| Unique image records with a QR decoder attempt | **0 / 38** |
| Identity rejection before complete head association | 24 frames |
| Identity rejection after head association, while constructing identity region | 14 frames |
| Final observer outcome | `inspection_progress_committed`, `measured_head_front_identity_unresolved` |

The final frame has accepted measured-head quality, an accepted head-to-LiDAR
association, and accepted orientation bounds. The head angle is approximately
17.317° with a 4.912° interval half-width. Its model reprojection RMSE is
0.323 pixels and raw border support is 1.0. These are implementation
measurements, not independent physical calibration claims.

Two different association policies then act on the same scan:

| Policy | Bearing source | Cone half-angle | Eligible clusters | Result |
| --- | --- | ---: | ---: | --- |
| Current head association | Measured camera head | 3° | 1 | `current_head_unique_lidar_cluster` |
| Subsequent QR search | Historical map bearing | 15° | 2 | `qr_search_cluster_not_unique` |

The first selects scan indices 215–216. The broader search selects a
four-point group at 213–216 but sees a second eligible group elsewhere in
the broader cone. The QR search returns no region, so the decoder is not
called. This establishes a mismatch between two admission policies; it does
not prove that every broader-cone cluster belongs to the head or that a QR
would decode successfully.

The source image and scan stamps are respectively `1790943554.365476` and
`1790943554.4052246`. The final association and QR diagnostics refer to that
same pair. The 38-attempt count above was deduplicated by image timestamp.

### R1 — consume the current head association instead of repeating discovery

Priority: **first implementation change**. Direct current-run evidence.

`observer/current_head_identity.py:156` restarts `current_scan_qr_search` after
requiring an already accepted current head association. It then calls
`support_opposite_head_region`; that path also invokes
`opposite_target_support.py:245` and repeats the broader unique-cluster gate
before re-associating the head. Removing only the first search is insufficient.

Use the existing accepted `CurrentHeadCandidateAssociation`, its same-image
corners, and its actual selected scan association to construct the existing
head-support/crop representation. Preserve the real 3° cone, source indices,
timestamps, finite-depth relation and reconciliation context. Do not relabel
the rejected broader search as accepted.

Continue through `exclusive_identity_crop` and `masked_identity_pixels`:
their neighbor projection, depth, candidate UID and masked-pixel checks still
determine whether decoded text belongs to this candidate. This removes
repeated discovery while retaining final identity association. It requires
no new search/retry framework.

Acceptance evidence: replay the new run's accepted-head / broad-search-
ambiguous tuple; prove that the payload backend is invoked. Test one valid
candidate-bound ID, empty decode, decoder failure, a neighboring QR,
multiple texts, mismatched image/scan, different calibration/candidate and
expired publication. Cornerless payloads must remain restricted to a proven
exclusive head region; a backend's whole-input rectangle is not QR geometry.

## Further simplifications, in execution order

### R2 — give usable inspection evidence priority over optional centering

Priority: **high**, demonstrated historical failure and current policy.

`observer/node.py:3339` gives pending centering exclusive publication priority
over immediate front, bounded head and QR-only completion. The preparation
path can also clear accumulated evidence. Tests deliberately assert this
behavior; passing those tests does not establish alignment with the requested
workflow.

The October 1 17:11 run reached three confirmed IDs, then a 0.5885° centering
request stopped with 0.308° remaining against a 0.300° tolerance. The robot
confirmed a stationary stop, but the generic child exception terminated the
mission. It never made the next observation.

Prefer a currently valid identity/frame result over an optional centering
advisory. If the image is unusable, a separately admitted centering move can
still run. After any actual movement, invalidate pre-motion image evidence
and acquire a fresh image. A known-stopped, bounded tolerance miss should
permit a fresh observation or candidate-local deferral; an unconfirmed stop,
unexpected translation, authority failure or integrity error remains terminal.

Use one typed centering outcome at
`real_robot/execution/candidate_centering.py:123` and consume it in the existing
`candidate/centering_execution.py` / `inspection_execution.py` flow. Avoid
another blind-turn retry loop or simply extending the timeout.

Tests: ready first-view identity beats centering; genuine unusable view still
requests correction; the recorded tiny-turn stop reaches a fresh camera
capture; unsafe/unknown stops do not. Existing centering-precedence assertions
must be deliberately revised.

### R3 — reuse the existing geometry-plus-empty-decode path

Priority: **after R1 validation**; the new run did not actually attempt
decoding, so it does not demonstrate failure of the empty-decode branch.

The bounded/unidentified branch already supports geometry plus an empty
current decode leading to the opposite side. Use it; do not add another
backside classifier or fallback module. `observer/node.py:3030` can instead
report `measured_head_front_identity_unresolved` when those prerequisites
have not produced a receipt. The resulting generic inspection advisory has
only ±90° angle authority despite the accepted precise window in the new
run, where decoding never occurred.

Keep one geometry result in the existing candidate inspection state and a
separate QR outcome: decoded, attempted-empty, or unavailable. Verify that
both precise geometry and bounded proposals can use the existing
retained-orientation route. An actual empty decode should select the
opposite-side inspection without requiring a separate visual proof of
backside appearance. A skipped or failed decoder should preserve geometry
while acquisition remains pending.

Do not globally lower every angle or sample threshold. A proposal must carry
its actual interval into the opposite-view/route feasibility calculation.
Review the generic 15° proposal cap separately from route feasibility rather
than replacing it with another arbitrary larger constant.

Tests: usable precise head + empty decode and bounded head + empty decode
reach the same opposite dispatch; missing decode does not classify backside;
the retained interval survives the move without new axis samples; wider
proposals are accepted only when their full interval supports a feasible view.

### R4 — permit opposite-side decoding before optional border reacquisition

Priority: **high**, current recorded-fixture failure, independent of the old
QR-outline bug already fixed by `8f566e1`.

The opposite branch already returns before angle fitting (`node.py:1751`).
It retains the first angle correctly. However, scan/crop/head-support checks
can prevent the payload backend from being invoked at all.

The current regression
`test_opposite_target_geometry.py::test_recorded_producer_immediate_qr_receipt_keeps_angle_and_candidate_identity[True]`
fails locally with OpenCV 4.14.0 on the September 28 recorded fixture: valid retained angle/center,
supplied reconciliation, and a unique current scan search; zero complete
current head-region hypotheses; rectangular crop overlaps background
candidate 0003; decoder never invoked. Its payload backend is stubbed, so
this is evidence of the pre-decoder barrier, not proof of pixel decode success.

Allow a diagnostic/uncommitted decode from the projected retained target
region. If the decoder supplies genuine symbol corners, existing ray/target
association can establish identity without a complete new head-border fit.
For cornerless text, keep exclusive current head-region evidence as the
admission requirement. Decoding and accepting the decoded identity are
different decisions. Preserve neighbor exclusion and duplicate-ID handling.

Keep existing QR discovery independent of final facing-route feasibility.
`candidate/qr_pose_discovery.py:65` already preserves the angle; a failed
optional facing promotion does not discard discovery. A lower-priority
inconsistency is that `artifacts/retained_facing.py:72` promotes bounded
orientations but not all certified precise orientations.

### R5 — distinguish observation-route eligibility from current visibility

Priority: **medium-high**, code-level obstacle to inspecting every candidate.

`candidate/approach.py:2623` filters the pool to candidates visible in one
current LiDAR cohort. `:2837` ends the remaining goal if none pass. The next
view may be needed precisely because a candidate is not visible here.

A safe survey-target observation route already exists below this filter:
`navigation/approach/candidate_preapproach_selection.py:239` computes the
fallback when there is no current fit. `candidate_preapproach_compute.py`
retains the full candidate keepout population, static target and standoff
checks, continuous route clearance and uncertainty admission.

Make current target estimates optional refinements for an **observation
approach**, preserving the bounded survey target when visibility is merely
missing. This also requires changing the exact estimate-key equality at
`candidate_preapproach_selection.py:128` and retaining the executed survey
target's provenance at arrival. Precision alignment/close translation still
needs the appropriate current target evidence. A wall contradiction, corrupt
source, invalid frame or unsafe route must not become a visibility fallback.

This is more useful than adding retries alone: retries cannot remove a
persistent occlusion. Test both a safe occluded-candidate observation approach
and unchanged rejection of wall/competing/unsafe cases. Do not silently choose
the best five hypotheses or remove unvisited keepouts.

### R6 — give candidate-local failures one disposition policy

Priority: **medium**, partly demonstrated by earlier runs.

One outer inspection episode can currently be exhausted before its first
camera view. Historical morphology advisories have no current-evidence
reconciliation path. Centering failures use generic exceptions even though
opposite-route recovery already distinguishes certain verified stopped cases.

Use the existing inspection state and ledgers to distinguish no view yet,
temporarily unavailable, actual inspection exhaustion, and systemic failure.
Fresh alternative planning after a verified stop needs fresh localization
and exact new motion authority; it must not reuse the rejected route/permit.
Preserve retained geometry across this transition.

The October 2 10:41 runtime-replacement failure is **already addressed in
current code** by `candidate/opposite_runtime_retry.py:36`, introduced in
`7aef892` along with the smaller collision reserve. It allows one extra fresh
planning epoch for a validated rejection while disabling renewed reseal
budgets. `approach.py:2364` routes eligible runtime errors into that mechanism.
Retain and test it; do not add another opposite-specific retry layer. Its
typed outcome concepts are useful for the remaining centering and deferral
gaps. The October 1 wall fixture demonstrates why morphology contradiction
needs explicit reconciliation rather than blanket deletion.

The 5–10 pool prerequisite and seven-frame temporal policies are lower-priority
workflow choices. They are not the direct cause of the new run's zero decoder
attempts. Avoid changing them in the first implementation patch.

## Small implementation sequence

1. **Share the admitted head association with first-view identity extraction.**
   Reproduce the new run; retain receipt replay and negative association tests.
2. **Make completion/retained-geometry outcomes own the inspection transition.**
   Unify precise/bounded no-QR handling and make centering subordinate when
   useful evidence is already available; type verified stopped outcomes.
3. **Finish the opposite identity boundary.** Attempt decoding without another
   angle fit, then apply the appropriate same-candidate identity evidence.
4. **Simplify scheduling/recovery.** Permit safe observation approaches to
   temporarily invisible hypotheses and reuse existing candidate-local recovery.

Prefer changes to the existing association object, frame and inspection state.
Avoid parallel fallback modules, another set of retry counters, duplicate
evidence formats, or a blanket increase in timeouts and geometric thresholds.

## Evidence and validation

Workstation root:
`/home/amr_gruppe01/Apptainer_Turtle/workspace_ROS2_Humble/mii-amr`.
New-run first-view evidence is under
`results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20261002T121410Z/candidates/000_survey_candidate_0003/camera_lidar_attempt_00/`:
`observer_status.json`, deduplicated `observer_events.jsonl`,
`inspection_observation.json`, and `observer_process.json`.

The October 1 17:11 run also records a successful backside-axis → opposite
`Start` QR observation → facing promotion. That working path should remain a
positive control. Historical explanations are in the neighboring run audits
for `20261001T141116Z`, `20261001T151113Z`, `20261002T084113Z` and
`20261002T104247Z`; older offline decoder/scan replays were not rerun here.

Focused current-baseline test results and exact reproduction commands are
recorded below. Tests exercise existing behavior; this audit implements none
of the recommendations and cannot establish future five-candidate success.

Across five invocations: **376 passed, 5 failed, 143 subtests passed**. These
are focused suites, not the entire repository test suite. All used the
existing `/private/tmp/a04-id-only-venv/bin/python` from the repository root;
no dependencies were installed.

| Suite | Passed | Failed | Subtests passed |
| --- | ---: | ---: | ---: |
| First-view geometry, identity and centering policy | 115 | 0 | 51 |
| Population, inspection orchestration and stopped motion | 103 | 0 | 55 |
| Target/planning fallback and current LiDAR integration | 40 | 0 | 34 |
| Opposite producer and retained-facing replays | 92 | 5 | 0 |
| QR discovery, overlap and acquisition opportunity | 26 | 0 | 3 |

One failure is the R4 replay, at
`test_opposite_target_geometry.py:135`: expected one decoder invocation,
observed zero. Four failures are `projection`, `stamp`, `other_quad`, and
`missing_corners` variants of
`test_persisted_raw_payload_binding_replays_calibration_stamp_and_own_corners`
in `test_opposite_endpoint_integration.py`: the common fixture assertion at line 380
expects corner error below 1 pixel and observes 1.664 pixels, before testing
the requested negative mutation. The cause of that numerical discrepancy
is unproven; it must not be silently counted as passing or attributed to a
new production change. Local OpenCV is 4.14.0. No current workstation pixel
decoder replay was performed in this audit.

Exact commands (run separately):

```sh
/private/tmp/a04-id-only-venv/bin/python -m pytest -q \
  tests/aufgabe04/test_unidentified_head_orientation.py \
  tests/aufgabe04/test_bounded_head_observation.py \
  tests/aufgabe04/test_immediate_front_observation.py \
  tests/aufgabe04/test_observer_candidate_centering.py \
  tests/aufgabe04/test_current_head_identity.py \
  tests/aufgabe04/test_recorded_coarse_front_facing.py \
  tests/aufgabe04/test_backside_center_completion.py

/private/tmp/a04-id-only-venv/bin/python -m pytest -q \
  tests/aufgabe04/test_exact_two_camera_admission.py \
  tests/aufgabe04/test_exact_two_camera_seed_selection.py \
  tests/aufgabe04/test_candidate_inspection_execution.py \
  tests/aufgabe04/test_candidate_centering_child.py \
  tests/aufgabe04/test_candidate_centering_runtime.py \
  tests/aufgabe04/test_opposite_runtime_retry.py

/private/tmp/a04-id-only-venv/bin/python -m pytest -q \
  tests/aufgabe04/test_lidar_inspection_planning.py \
  tests/aufgabe04/test_candidate_target_admission.py \
  tests/aufgabe04/test_current_lidar_approach_integration.py

/private/tmp/a04-id-only-venv/bin/python -m pytest -q \
  tests/aufgabe04/test_opposite_identity.py \
  tests/aufgabe04/test_opposite_head_identity_integration.py \
  tests/aufgabe04/test_opposite_endpoint_integration.py \
  tests/aufgabe04/test_opposite_reconciliation.py \
  tests/aufgabe04/test_opposite_target_geometry.py \
  tests/aufgabe04/test_retained_facing.py \
  tests/aufgabe04/test_projected_retained_facing.py

/private/tmp/a04-id-only-venv/bin/python -m pytest -q \
  tests/aufgabe04/test_qr_pose_discovery_execution.py \
  tests/aufgabe04/test_opposite_identity_overlap.py \
  tests/aufgabe04/test_opposite_identity_opportunity.py
```

Before a robot validation, the changed policy needs the R1 same-frame replay,
unchanged neighbor/identity negative controls, the R2 verified-stop replay,
and the retained-angle → opposite QR round-trip. Record per candidate the
first unresolved transition, decoder-attempt count and retained-angle source
using existing diagnostics. Success means five unique candidate-bound IDs
and the required retained head geometry; absence of an exception alone is
not success. Evaluate the subsequent return-to-Start phase separately.

## Local implementation follow-up

The implementation removes the duplicated admission prerequisites in the
first patch sequence above, while keeping current candidate association and
motion admission. No workstation deployment or new robot run was performed.

- **R1:** `current_head_identity.py` reuses the accepted measured-head
  association through `support_from_current_head`. The source tuple and the
  same narrow scan association are revalidated; the wider map-centered search
  is no longer a prerequisite for this identity crop. The existing masked
  head crop and neighbor exclusion still apply.
- **R2/R3:** ready geometry, QR identity and retained orientation precede
  optional centering in both observer publication and artifact selection.
  Advice no longer clears accepted geometry. A typed, verified bounded-stop
  outcome permits one fresh capture after the stop, with further centering
  disabled for that view. Unknown or unsafe stops remain errors. Existing
  precise and bounded geometry paths retain their actual orientation interval
  after an actual empty decoder result; decoder errors or skipped decoding
  cannot supply backside evidence.
- **R4:** opposite-side acquisition can decode a bounded search crop before
  head-border reacquisition. Genuine decoded symbol corners can establish
  current ray support and use the existing isolated-symbol receipt. For the
  ordinary ray path, the existing finite-range proof replaces broad-cone
  uniqueness and is replayed on receipt validation. Reconciled/witnessed
  support keeps its existing proof path. Cornerless text still needs an
  exclusive current head crop. The retained angle is never refitted.
- **R5:** missing visibility can keep a bounded survey hypothesis eligible
  for an initial observation approach. Current target refinements may cover
  only part of the selection pool, and stale alignment hints are suppressed
  for the remaining survey targets. The original full snapshot and support
  assessment are replayed at dispatch and through arrival. Replanning retains
  the actual completed route's target, including an upgrade to newly visible
  current LiDAR geometry. Wall, competing-target, unstable and malformed
  evidence do not qualify; all candidate keepouts remain present.

The known R6 centering failure is covered by the typed stop outcome. The
existing bounded opposite-runtime recovery remains in use. The audit's
separate morphology-reconciliation question and pool/window tuning are not
changed by these admission simplifications.

Regression evidence includes a clearly labeled synthetic reproduction of
one accepted narrow cluster plus two broad-search clusters; it is not a raw
replay of the October 2 workstation tuple. The September 28 opposite fixture
uses its recorded image and genuine desktop-decoder symbol corners, replayed
deterministically at the backend boundary. The four old calibration tests
now verify recomputed corner error against the production tolerance rather
than an unrelated subpixel fixture expectation; production tolerance is
unchanged.

### Post-change validation

One combined invocation on the final working tree passed **713 tests and
301 subtests** in 126.07 seconds. SHA-256 fingerprints of the modified Python
files and selected tests were unchanged from before to after the run.
`git diff --check` passed, and all 18 modified production modules parsed with
Python 3.10 syntax rules. The execution environment was the existing desktop
venv with OpenCV 4.14.0; this is not a ROS or hardware execution result.

Exact combined command:

```sh
/private/tmp/a04-id-only-venv/bin/python -m pytest -q \
  tests/aufgabe04/test_current_head_identity.py \
  tests/aufgabe04/test_current_head_association.py \
  tests/aufgabe04/test_opposite_head_support.py \
  tests/aufgabe04/test_qr_observation_pose.py \
  tests/aufgabe04/test_unidentified_head_orientation.py \
  tests/aufgabe04/test_bounded_head_observation.py \
  tests/aufgabe04/test_immediate_front_observation.py \
  tests/aufgabe04/test_observer_candidate_centering.py \
  tests/aufgabe04/test_recorded_coarse_front_facing.py \
  tests/aufgabe04/test_backside_center_completion.py \
  tests/aufgabe04/test_exact_two_camera_admission.py \
  tests/aufgabe04/test_exact_two_camera_seed_selection.py \
  tests/aufgabe04/test_candidate_inspection_execution.py \
  tests/aufgabe04/test_camera_inspection_binding.py \
  tests/aufgabe04/test_candidate_lidar_acquisition.py \
  tests/aufgabe04/test_candidate_centering_child.py \
  tests/aufgabe04/test_candidate_centering_runtime.py \
  tests/aufgabe04/test_candidate_centering.py \
  tests/aufgabe04/test_inspection_four_qr_centering.py \
  tests/aufgabe04/test_clipped_cluster_head_recovery.py \
  tests/aufgabe04/test_passive_observer_process.py \
  tests/aufgabe04/test_opposite_runtime_retry.py \
  tests/aufgabe04/test_lidar_inspection_planning.py \
  tests/aufgabe04/test_candidate_target_admission.py \
  tests/aufgabe04/test_current_lidar_approach_integration.py \
  tests/aufgabe04/test_current_lidar_targets.py \
  tests/aufgabe04/test_current_lidar_reacquisition.py \
  tests/aufgabe04/test_candidate_target_retention.py \
  tests/aufgabe04/test_retained_lidar_target.py \
  tests/aufgabe04/test_current_lidar_target_preflight.py \
  tests/aufgabe04/test_autonomous_candidate_approach.py \
  tests/aufgabe04/test_survey_observation_targets.py \
  tests/aufgabe04/test_camera_candidate_selection.py \
  tests/aufgabe04/test_opposite_identity.py \
  tests/aufgabe04/test_opposite_head_identity_integration.py \
  tests/aufgabe04/test_opposite_endpoint_integration.py \
  tests/aufgabe04/test_opposite_reconciliation.py \
  tests/aufgabe04/test_opposite_target_geometry.py \
  tests/aufgabe04/test_retained_facing.py \
  tests/aufgabe04/test_projected_retained_facing.py \
  tests/aufgabe04/test_qr_pose_discovery_execution.py \
  tests/aufgabe04/test_opposite_identity_overlap.py \
  tests/aufgabe04/test_opposite_identity_opportunity.py \
  tests/aufgabe04/test_observer_inspection_progress.py \
  tests/aufgabe04/test_candidate_inspection_observation.py
```

Concurrent edits to inspection progress, passive identity retry and LiDAR
sampling were preserved. Their two new integration modules also passed in a
separate invocation: **21 tests and 6 subtests**.

```sh
/private/tmp/a04-id-only-venv/bin/python -m pytest -q \
  tests/aufgabe04/test_inspection_identity_pending.py \
  tests/aufgabe04/test_inspection_identity_reason.py
```

These tests establish the local transition and evidence contracts. Whether
the revised mission reaches all five real stands still requires a new
controlled workstation/robot run with decoder-attempt and retained-angle
evidence.
