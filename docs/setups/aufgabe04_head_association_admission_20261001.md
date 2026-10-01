# Head acquisition, LiDAR association and opposite-side admission

Investigation of `stand_explore_exact2_camera_all5_20261001T141116Z`, running
clean revision `144f954` on workstation alias `mii002` (hostname `mii001`).
Times are Europe/Berlin on October 1, 2026. **638 original files** independently
match their copied-source hashes and a fresh workstation inventory at 16:24:43.

The light-grey candidate did obtain an accepted backside angle and reach the
correct opposite-side pose. It then failed **current QR-outline acquisition and
exclusive crop admission**, before the live QR decoder was invoked. The records
contain no successful live QR identity that was subsequently discarded. The
original pixels are readable as `Start` in offline diagnostics; that is separate
from the recorded mission's identity admission.

The general pipeline weakness is that its robust payload decoder sits behind a
less robust outline detector. When another candidate's conservative projection
overlaps the target crop, an outline is needed before decoding may start. If
outline acquisition fails, better decoding capabilities downstream are never
used, even with valid head orientation and current LiDAR support.

**What actually succeeded and failed**

| Stage for this run's candidate 0001 | Recorded result | Meaning |
|---|---|---|
| Initial view | 19 processed images, nine associated; `head_border_choice_unstable` advisory | Individual head fits did not yet make a stable admitted axis; bounded view recovery was appropriate |
| Changed backside view | 14 processed images, nine associated; seven-sample bounded orientation committed | Head acquisition, camera/LiDAR registration and backside angle admission succeeded |
| Retained orientation | Half-width **±3.8833°**, preserved through reprojection | The opposite branch did not lose or refit the angle |
| Opposite travel | 1.136 m planned route; 1.120 m executed in 51.74 s; completed 16:18:31.932 | No controller or localization-drift rejection |
| Stopped arrival | Range 0.47438 m; optical bearing **+2.13139°**; static clearance 0.50459 m | Passed even strict 3° arrival bearing; the new 10° allowance was not needed here |
| Current LiDAR | 18 associated frames; three-scan target reconciliation ready | Current target support was obtained |
| Current QR outline / crop | 21 `no_complete_target_outline` and `target_crop_overlap_unresolved` results; one stale search | The exclusive identity crop never became available |
| Live decoding | Zero decoder diagnostic entries, zero nonempty QR texts, zero QR samples | Live QR decoding was never reached |
| Observer result | Clean exit with `inspection_observation`, `qr_id=null`, classification `unobservable` | An advisory was returned, not a QR receipt or facing recommendation |
| Parent | Four blocked standoffs, then another inspection route; `KeyboardInterrupt` at 16:19:29.527 | Candidate remained `inspection_started`; only QR_003 was confirmed |

The opposite observer processed 22 images during its 8.44-second call. Twenty-one
usable crop decisions failed the same overlap gate; one search was stale. Its
receipt uses 18 associated samples and explicitly grants no identity, completion
or motion authority. The final parent manifest has no normal exit entry or final
mission summary. The interruption is recorded in the candidate handoff JSONL,
even though the parent terminal log ends earlier.

The final stopped pose was within **22.97 mm / 1.746°** of the executed goal in
odometry, inside the controller's 30 mm/3° contract. It was only 2.228° away from
the retained directed opposite normal. Reprojection moved the retained center
1.44 cm and rotated the axis 0.243°. These observations do not support a
wrong-side drive or bad head-angle diagnosis.

**Why successful LiDAR association was insufficient**

Frame 10 illustrates the exact geometry. A unique contiguous five-beam group at
0.549 m and −9.050° supplies the local target search. Its surface centroid is
9.56 cm from the projected survey target but only 3.99 cm from the authenticated
frozen hypothesis. Ordinary bounded position-epoch reconciliation succeeds.
These surface residuals are not ground-truth errors in the stand center.

| Image/depth quantity | Value |
|---|---|
| Current scan projected center | (505.95, 299.09) px |
| Expected head height | 107.06 px |
| Broad search rectangle | [377, 170, 635, 428] |
| Fallback identity rectangle | [420, 213, 592, 385] |
| Candidate 0003 projected exclusion | [503, 315, 556, 368] |
| Candidate 0003 depth interval | 1.399–1.517 m |

The search contains the visible foreground head. It is not aimed at a wholly
unrelated image region. Its fallback identity rectangle also intersects a
farther candidate's projected head envelope.

`observer/opposite_identity_crop.py:59–116` permits this overlap only when a
complete current QR quadrilateral is independently bound to the foreground
LiDAR target. That support supplies a target-depth interval and lets the decoder
sample the isolated foreground symbol. Without the outline, the broad crop
cannot establish which pixels belong to the target. The known LiDAR distance
alone does not authorize every QR inside that rectangle.

For all 21 usable search envelopes there was one contiguous current cluster;
both frames with fully valid circular scan topology still failed this same crop
gate. This is **not a recurrence of the earlier scan-seam fragmentation failure**.
The later exact-TF retry losses reduced available evidence but followed many
already-failed crop decisions, so they do not explain the primary rejection.

**The outline acquisition gap**

`observer/opposite_target_support.py:181–204` tries native OpenCV
`QRCodeDetector.detectMulti` on the rectified search crop at scales **4 and 1**,
under a **60 ms cooperative budget**. It does not use the photometric variants
available in `qr_scanning/opencv_qr_detector.py`. Completeness, center and physical
scale filters then decide whether a detected quadrilateral qualifies.

Offline replay of original frames 5, 16 and 31 with recorded CameraInfo reproduced
native outline failure. Raw versus rectified pixels, whole frame versus bounded
crop, and a longer diagnostic outline budget did not by themselves resolve it.
The production payload decoder can recover `Start` from frame 31 using its existing
preprocessing variants. The live branch cannot try that richer processing:
`observer/opposite_identity.py:114–144` invokes decoding only after
`exclusive_identity_crop` returns an admitted attempt.

The emitted `no_complete_target_outline` reason is also too coarse for diagnosis:
it can mean no native corners, rejected corner geometry, or budget exhaustion
before a later scale. Record the actual variant, returned outline count,
center/scale rejection and elapsed/remaining budget separately. This run's pixel
replay distinguishes detector robustness from a crop-position or angle problem.
Also report retained angle, current LiDAR association, outline detection, decoder
invocation, decoded payload and admitted identity as separate stages. Here
`committed_artifacts=1` refers to an inspection advisory; zero fresh head-detector
results in the opposite branch is expected because it retains the earlier angle.

A counterfactual replay of **frame 31** goes further than decoding the pixels.
Its existing thresholded variant finds the current symbol's own corners. After
mapping them back to the original rectified crop, the existing production
three-scan reconciliation and support/crop validators accept them:

- Corners approximately (457.50, 242.00), (543.94, 239.98), (549.16, 329.48),
  (459.45, 331.71); center (502.51, 285.79), within existing scale/center limits.
- Finite camera-ray registration error **0.0724°**, below the unchanged 3° limit.
- Three-beam subset **207–209**, within the original target group **206–210**.
- Isolated crop **[457, 239, 551, 333]** and foreground depth interval
  **[0.38946, 0.54946] m**. The existing rule correctly excludes the farther
  candidate at **[1.399, 1.517] m**.

No overlap, range or angle threshold needed to change for that replay. It uses
original recorded sensor geometry and regenerates reconciliation from the saved
raw scans, reproducing all 22 recorded reconciliation readiness/sample-count
states. It establishes a feasible acquisition improvement for this frame,
not a completed live mission or blanket recovery of every frame: frames 5 and 16
did not yield an outline through the tested existing variants. Desktop replay
uses **OpenCV 5.0.0**; its timing is not a workstation timing guarantee. Full
advancing-clock publication and persisted-receipt validation remain necessary.

Outline recovery alone is not yet an end-to-end solution. An advancing-clock
diagnostic recovered the outline in about 16 ms and passed support/crop admission,
but the existing `native_quad` isolated decode still returned no payload in its
remaining approximately 78 ms. Total processing was 168 ms; final image/scan
ages were 0.402/0.424 seconds. A separate existing four-percent source-quiet-margin
view also returned no payload within 120 ms. Broad-crop
decoding with a longer diagnostic budget can succeed, but cannot substitute for
an admitted isolated-symbol decode. The image variant, source quiet margin and
decoder work ordering therefore need validation together with outline recovery.
These are native-only desktop experiments; the deployed workstation backend
must be tested before attributing the isolated-decode result to live behavior.

The bounded backside artifact contains seven-sample orientation and target
registration, but no separately certified `validated_target_center`. Therefore
it uses ordinary three-scan reconciliation. The recently added endpoint-proof
path is not entered; all endpoint-confirmation diagnostics are empty. Ordinary
reconciliation succeeds, so fabricating a metric-center certificate or forcing
the endpoint exception would address the wrong prerequisite.

The original-image decoder with calibrated corner binding is currently confined
to the endpoint branch (`opposite_identity.py:129–143`). Ordinary reconciled
opposite observations instead use rectified pixels. Shared, same-frame outline
and payload handling should cover both supported target-evidence types without
changing their different LiDAR proof requirements.

**What should change in the general pipeline**

1. **Improve current outline acquisition before changing admission thresholds.**
   Reuse bounded photometric variants that already work in QR processing. Restore
   every detected corner through the exact scale, border and calibration
   transforms to the source image. Keep the symbol's own corners; a head box or
   decoded text alone must not substitute for them. Preserve the accepted
   backside angle and its uncertainty throughout.
2. **Carry the same validated quadrilateral through association and decoding.**
   Use its finite-distance camera ray to associate current LiDAR, then apply
   target/neighbor depth exclusion and decode the isolated symbol. If an
   original-image variant is needed, validate its mapping back to that same
   current quad. Support both ordinary reconciled targets and certified-center
   endpoint targets through shared image-processing helpers, retaining the
   proof rules specific to each branch.
3. **Prioritize useful variants within one real freshness budget.**
   Trying all expensive variants sequentially can exceed the 0.5-second source
   age even if a later variant works offline. Profile and order useful variants,
   preserve the 60 ms support / 120 ms decoder ceilings, and check publication
   age with an advancing clock. A one-second offline decode is evidence of
   readable pixels, not live timing compliance.
4. **Preserve the failure cause when requesting another view.**
   Current advisory generation reduces the crop conflict to generic
   `unobservable` (`inspection_progress.py:52–95`), causing the generic ±90°
   search order. Preserve whether the problem is missing outline, unresolved
   neighbor overlap, scan support, or freshness. With retained orientation and
   known projection conflict, a bounded small view change can be considered
   before a large circuit. It remains a hypothesis requiring normal route and
   live admission, not a reason to bypass the crop check.
5. **Test the entire ordinary opposite path with actual recorded pixels.**
   Existing tests include real imagery, but some use a 0.5-second support budget,
   mock decoded text, or stop at crop validation. The genuine native
   image-to-persisted-receipt test covers the special endpoint case. Add this
   ordinary reconciled target, with the background overlap retained, through
   actual outline acquisition, scan binding, isolated decoding, publication and
   `load_bound_qr_observation_pose`. No injected corners or mocked successful
   decoder should replace the failing stage.

Negative regressions should retain rejection for absent outlines, unseparated
depths, shifted corners, multiple identities, changed calibration and expired
source evidence. The successful case must preserve the original seven-sample
angle, add no synthetic current angle samples, and grant no motion authority.

The mission's downstream QR/facing separation already behaves correctly. If a
valid QR receipt exists and optional retained-facing validation fails, the code
preserves QR-only discovery (`candidate/retained_facing.py:30`,
`candidate/approach.py:2869`). That path was never reached here. Relaxing final
catalog admission or recomputing an already accepted angle would not repair this
run's missing current outline.

After the failed opposite observation, four blocked standoffs and another
direction consumed 33.44 seconds before the next inspection-motion call. Five
fresh planning-frame captures account for 25.29 seconds. Once acquisition is
fixed, pre-screening bounded geometric alternatives against a shared stopped
snapshot, then freshly certifying the selected motion, is a useful secondary
optimization. Faster driving would not remove the outline gate.

Evidence and reproducibility live in
`results/implementation_checks/admission_investigation_20261001T141116Z/`.
`evidence_summary.py` extracts counts, artifacts, retained angle and interruption
from original recordings; `source_verification.json` records the independent
638-file comparison. `camera_replay.py` and `camera_replay.json` preserve the
original-image/preprocessing experiments and production support/crop checks.
The original copies remain under
`results/implementation_checks/run_audit_20261001T141116Z/source/`.
This investigation does not modify production code or command the robot.
