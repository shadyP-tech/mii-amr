# Current measured-head backside handoff correction

This corrects the first-view backside failure for `survey_candidate_0001` in
`stand_explore_exact2_camera_all5_20260914T113458Z`. The base revision is
`98fc4fa1cb78acb1ca7787467528fac37feaf7ba`. See the separate
[run audit](aufgabe04_run_audit_20260914T113458Z.md) for the original timeline.

## Resulting behavior

A quality-admitted current head fit can now supply a separate backside-candidate
classification. Repeated fresh, stopped, synchronized and uniquely LiDAR-associated
samples produce the existing schema-3 backside receipt. The candidate inspection
loop consumes that receipt and selects its opposite-side branch before proposing
generic local views. That branch still plans a collision-checked route around the
stand, with ordinary live admission and one-use execution authorization.

The angle comes from the four measured physical head corners. Neck pixels verify
the outer-head structure; they do not become pose landmarks or contribute another
angle. QR corners and previous overlays do not select the head angle or resolve
an ambiguous head pose. The classifier retains `backside_candidate` semantics:
it is supported geometry plus repeated marker absence, not a directed front
normal inferred from the symmetric head plane.

## Module boundaries

| Module | Responsibility |
| --- | --- |
| `perception/stand_axis/raw_neck_support.py` | Measure two separated, connected rails on current raw edges, including bounded slanted/rasterized paths. |
| `head_model_neck.py` and `head_neck_connectivity.py` | Verify the strict outer-head junction and paper-panel separation using those actual pixel paths. |
| `head_backside_classification.py` | Attach side evidence to an independently admitted head fit; bind it to the current corners, yaw and measured profile. |
| `head_model_admission.py` | Require the same head quality/uncertainty contract for neutral and classified heads, including below the generic angle limit. |
| `real_robot/observer/qr_acquisition_policy.py` | Run current native QR and model marker checks; schedule bounded full payload recovery without repeating empty pyramids on every crop. |
| `real_robot/observer/registration_evidence.py` | Bind nominal or recentered current head measurements to their exact accepted camera/LiDAR association. |
| `real_robot/observer/node.py` | Accumulate evidence, preserve marker vetoes and publish the validated receipt while stopped. |

The legacy topology locator may propose raw-supported head corners even when its
old stem-anchor heuristic fails. Such corners must pass the complete independent
head fit and current neck-junction verification before they gain angle or side
authority. A rejected legacy pose is never reused.

## Processing and safety contracts

- Keep the 500 ms image/scan freshness limit and the configured seven-sample
  consensus requirement. Recheck source freshness at publication.
- Every evaluated crop receives current native QR decoding and model
  quadrilateral/finder checks. An empty budgeted decoder result alone is not
  backside evidence. New QR observations trigger a fit on the same image, and
  evidence from other evaluated crops retains its veto/conflict role.
- Full QR recovery is limited to one crop per image, with a cooperative 120 ms
  ceiling and 50 ms publication reserve. QR-negative probes occur periodically,
  alternating nominal/wider opportunities. OpenCV calls cannot be preempted;
  any overrun still fails the final freshness gate.
- A current tentative QR quadrilateral vetoes its frame. Verified finder pixels
  or decoded text veto the stationary epoch and clear existing backside samples.
- A classification cannot bypass independent head quality by changing its source
  string. Its structured proof is saved in metric diagnostics.
- Preserve strict camera bearing, scale and unique LiDAR-cluster checks. No scan
  ranges, increments, endpoint metadata or topology thresholds are rewritten.

## Validation evidence

The lossless latest nominal 179×179 ROI and original raw edges are retained in
`tests/aufgabe04/fixtures/backside_neck_20260914`, with source hashes, adjusted
intrinsics and expected projection metadata. They are actual saved pixels;
the reference corners are replay diagnostics, not measured angle ground truth.

Local whole-pipeline replay (Python 3.12 / OpenCV 4.13) produces a usable
measured-head-backed backside candidate from the clear first view:

- Latest nominal ROI: yaw approximately **−12.929°**.
- Original capture `frame_000010`: nominal yaw approximately **−12.023°**.
- Original capture `frame_000005`: remains rejected as a planar-axis ambiguity.

The QR scheduling comparison uses 35 saved images and 60 recorded ROI evaluations.
Full decoder calls fall from 59 to 17; every sampled geometry outcome and decoded
identity is unchanged between schedules, including `QR_003` on all 11 first-candidate
images. Median local QR work falls from 235.5 ms to 19.8 ms; median crop computation
falls from 441.6 ms to 288.6 ms. These timings exclude rectification/publication
and do not reproduce live ROI selection, registration or ROS delivery. The local
OpenCV native decoder succeeds on those front images, so that result alone does
not validate the deployed WeChat recovery schedule.

Unit and integration regressions cover raw slanted rails, disconnected pixels,
single/thick rails, missing rows, inner-rectangle pose rejection, malformed inputs,
classification provenance, unchanged head angle above 35°, nominal registered
receipt production, seven-sample consensus, opposite-branch priority, marker
contradictions, ambiguous LiDAR and publication freshness. The observer integration
uses synthetic live epochs and injected pixel fits; it proves control flow and
admission behavior, not live robot performance.

The final focused pytest matrix across 64 modules passes **644 tests and 690
subtests**, with one local WeChat-backend-dependent test skipped. All 36 changed
Python files parse under the Python 3.10 grammar; `git diff --check` passes.
All **705 original source artifacts** still match the saved SHA-256 manifest.
The complete test list, result XML, code hashes and logs are in
`derived/backside_correction_validation/pytest_summary.json`, `pytest.xml` and
`pytest.log` under the audit directory.

Replay artifacts and test logs are under
`results/audits/stand_explore_exact2_camera_all5_20260914T113458Z/derived/`.
Original recordings are not changed.

## Remaining hardware evidence

Read-only environment inspection confirmed Python 3.10.12, OpenCV 4.5.4 and
NumPy 1.21.5 on `mii001`. The approved in-memory two-image replay completed before
the request to stop transfers could take effect. It transmitted a 324,418-byte
prepared payload; the repository was mounted read-only, no ROS modules were
loaded, and the workstation checkout was not updated. No further workstation
transfers or remote commands were made.

That completed diagnostic reproduces the **−12.929°** backside classification
and preserves **QR_003** in the front image, including three bounded-policy
repetitions per image using the deployed WeChat backend. The replay's prepared
module hashes match the final local code. Its output is retained locally in
`derived/backside_review/target_runtime_result.json`. This is saved-image CPU
validation, not evidence of live sensor freshness or robot handoff completion.

The saved scans frequently have native seam/endpoint inconsistencies. In those
frames, target returns can appear as two eligible clusters and remain rejected.
The captured sequence does not establish seven valid synchronized samples in a
five-second window. Faster processing may expose more valid live tuples, but a
new real run must demonstrate that; no recorded frame is redated to manufacture
freshness or consensus.

The next hardware milestone is a first-view backside receipt, a successfully
admitted opposite-side route, then current front QR confirmation. Existing
experiment flags remain applicable after updating the robot checkout. This
correction itself launches no ROS nodes and commands no robot motion.
