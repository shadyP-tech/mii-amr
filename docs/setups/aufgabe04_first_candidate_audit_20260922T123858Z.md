# First candidate at 0.40 m — 22 September 2026

The observer never reached the head-frame angle solver. Every captured frame
failed cold head selection with `head_cold_acquisition_verification_budget_exceeded`.
The public `model_current_head_border_unavailable` message hides that distinction:
several physical-border refinements succeeded, but unresolved alternatives remained.

## Evidence

- Run: `stand_explore_exact2_camera_all5_20260922T123858Z`.
- First candidate: `000_survey_candidate_0003`, `camera_lidar_attempt_00`.
- Workstation revision: `bab57a20687ccdde1f225cb7b7ecc7662c180d27`;
  workstation status was clean when inspected.
- Recorded command confirms `--candidate-approach-offset-m 0.40`.
- Read-only workstation snapshot and analysis are under
  `results/implementation_checks/first_candidate_audit_20260922T123858Z/`.
  `audit_summary.json` records its SHA-256 and per-frame results.
- All 11 processed images were fresh, synchronized, TF-ready and associated at
  the observer's sensor-frame admission stage. This is distinct from accepting
  a measured head and registering its camera bearing to the candidate.

## Why no angle was produced

| Recorded quantity | First-candidate result |
| --- | --- |
| Cold proposals per image | 26–57 |
| Border families per image | 17–39 |
| Strict verification limit / used | 12 / 12, in every image |
| Successful border refinements per image | 5–10 |
| Remaining unverified independent hypotheses | 4–27 |
| Acquisition failure | Verification budget exceeded, 11/11 |
| Published head pose / quality | None / None, 11/11 |
| Axis evidence samples | 0 |

`head_cold_acquisition.py:639–641` returns without a selected proposal whenever
uncovered hypotheses remain after the twelve strict checks. Selection and pose
fitting have not yet happened. `physical_head_pipeline.py:173–176` maps this
producer reason to the generic missing-border message. This was a count budget,
not a processing deadline: median detector time was 183 ms, and all results
passed freshness. Median image header-to-receipt age was 38.7 ms.

The saved rectified image shows a complete, sizeable red head, with a visible
QR and no head clipping. Proposals combine the outer red border, inset edges,
and slightly different line fits around the same panel. In the final image,
10 of 12 refinements pass. Representative final corners include:

- Outer-like frame: `(292.7,197.3), (417.4,211.3), (416.6,336.3), (291.9,334.7)`.
- Inset frame: `(292.1,205.0), (410.1,215.3), (409.9,329.0), (291.9,329.0)`.
- Mixed frame: `(300.0,211.0), (417.0,211.0), (417.0,336.3), (300.0,334.8)`.

All recorded alias-resolution counters remain zero. These alternatives consume
the bounded search and prevent a unique physical rectangle from reaching PnP.
The current same-border tests require supporting-pixel agreement or evidence
of a single thin ridge; proximity or enclosure alone does not merge them.

A diagnostic replay using the final saved image, raw edges, and its 10 accepted
refinements leaves all 10 after family deduplication and reports
`distinct_current_heads_ambiguous`. This deliberately omits the unverified
hypotheses to inspect the already-tested set; it is **not** an admissible live
result. It shows that merely ignoring the budget veto does not resolve the
physical border selection. Local replay used OpenCV 5.0.0; the live decoder
reported OpenCV 4.5.4. Saved debug acquisition metadata matches frame 11.

## What the shorter distance changed

The final recorded robot-to-mapped-center distance is 0.432 m; the projected
camera optical depth is 0.389 m. The command's 0.40 m offset is therefore not
the final camera-to-panel depth. The projected head height is about 128 px.

For comparison, the preceding run's first candidate had a successful frame at
0.492 m projected optical depth and about 102 px projected head size. That
frame had 11 proposals, nine border families and nine strict verifications;
it selected a head and estimated yaw 23.21° with accepted quality.

The closer view makes the head about 26% larger in this comparison. It also
presents many more separately resolved border alternatives to the current
selector. This is consistent with a scale-sensitive border-family/refinement
problem, rather than insufficient pixel size or clipping. The two runs have
different actual viewpoints and are not a controlled distance experiment;
distance alone cannot be assigned as the sole cause.

## Why QR fallback did not rescue this view

Unlike the previous run's second-candidate decoder starvation, QR decoding did
run here. Frames 1, 6 and 10 decode `QR_003` using native quad detection and the
isolated WeChat decoder. Each is rejected by `camera_bearing_outside_map_cone`:
its measured bearing differs from the map bearing by 4.69–4.72°, exceeding the
unregistered path's 3° cone. The remaining eight frames have no decoded QR
geometry admitted to this binder.

Because no head registration was accepted, `qr_target_binding.py:71–79` uses
the original candidate cone. No independently bound QR sample is admitted,
and the QR-only grace cannot start. The repeated terminal QUIRC warnings are
not proof that identity decoding was impossible: the alternate backend did
produce `QR_003` three times. The source of the map/image bearing displacement
cannot be separated into mapping, localization or calibration error from this
capture alone.

## Why it moved on quickly

Eight eligible unresolved samples span 2.014 seconds and produce an
`unobservable` inspection advisory at approximately 14:43:52 CEST. The observer
exits normally with return code 0, no signal and no deadline expiry. The
90-second camera timeout is a maximum; unresolved-inspection completion can
finish earlier. The parent then plans another generic inspection view. No
certified head angle exists to drive an angle-based opposite-side route.

The copied mission log ends after that next view's dry preflight passes;
this audit does not claim a final mission termination cause beyond the failed
first-view acquisition.

## Recommended correction

Repair physical outer-frame selection and the handling of duplicate border
hypotheses at this image scale. Use these eleven images as regression inputs,
including true competing-head and inset/outer-boundary cases. Preserve current
pixel evidence and uniqueness; selecting the first/largest fit or merely
increasing the twelve-check cap does not establish a correct angle.

Expose the actual producer failure in observer diagnostics so a work-budget
failure is distinguishable from absent borders. Independently repair bounded
QR-to-candidate association for the measured 4.7° displacement, rather than
requiring successful head geometry before identity-only recovery can operate.

No production code, robot configuration, ROS state or robot motion was changed
by this audit. Reproduce the counts and diagnostic selection with:

```sh
PYTHONPATH=. /private/tmp/a04-inspection-audit-venv/bin/python results/implementation_checks/first_candidate_audit_20260922T123858Z/analyze.py
```
