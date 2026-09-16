# Cluttered-background camera acquisition audit

Audited recording: `recording_20260916_141146_741705309`, 20 distinct source
images, September 16, 2026, **12:11:46.221–12:11:50.853 UTC**
(14:11:46–14:11:50 CEST).

The recording and source images remained on mii001 at:

```
/home/amr_gruppe01/Apptainer_Turtle/workspace_ROS2_Humble/mii-amr/results/aufgabe04/stand_axis_debug_recordings/recording_20260916_141146_741705309
```

The remote checkout was clean at `a9aac95e743e17c3ceb671280b4e2be8de0b81fd`.
The local immediate-front admission/acquisition changes from the preceding
implementation were not deployed there. The recording does not validate those
changes. No production files were modified during this audit; offline image
replays used the existing OpenCV 4.5.4 container and a read-only repository mount.

## Findings

**The recorded failure occurs before a 3D head pose is computed.** Every frame
has `model_current_head_border_unavailable`, with no model pose, head-quality
result or orientation bounds. Nineteen frames report
`head_cold_acquisition_verification_budget_exceeded`; one has 1,027 contours,
exceeding the 1,024-contour limit. All 20 also fail freshness and tracker update.

The viewer searched the full 800 × 600 rectified image (`target_roi=null`), with
no tracked pose or candidate center/size. It considered a median 48.5 hypotheses
and used all 12 strict border verifications on each non-contour-limit frame.
Four to fourteen unverified independent hypotheses remained in those frames.
The code deliberately returns no proposal in that condition; it does not select
the first convincing rectangle.

The numeric proposal diagnostics are consistent with the user's description
of windows, radiators and finder-like squares competing with the stand. For
example, frame 0 verifies a near-canvas-height rectangle approximately
`[475,6,791,594]` and a smaller rectangle near `[336,304,406,360]`.
Other frames verify a larger central-right rectangle near
`[413,199,542,331]` and small squares near `[418,205,457,244]`,
`[493,206,532,244]`, and `[420,284,461,324]`. A verified rectangular boundary
alone does not identify the intended physical stand. The recording has no
mission candidate binding that would establish which rectangle is candidate 2.

`head_cold_acquisition.py:244–267` ranks families, verifies at most 12 and then
rejects uncovered alternatives. `head_proposal_selection.py` handles repeated
inset texture only during final selection, after the exhaustion check. Thus
that later grouping cannot rescue these frames.

## Timing is a separate blocker

| Recorded metric | Median | Range |
| --- | ---: | ---: |
| Source age at receipt | 244.35 ms | 216.70–249.36 ms |
| Independent head acquisition | 159.65 ms | 1.33–180.12 ms |
| Native QR stage | 62.47 ms | 24.08–91.46 ms |
| Complete metric processing | 225.47 ms | 71.73–256.36 ms |
| Source age at rendering | 492.34 ms | 426.39–568.25 ms |

The short acquisition minimum is the contour-limit early exit, not successful
fast geometry. The viewer requires both receipt-to-result age ≤180 ms and
source-to-result age ≤250 ms. The median incoming image leaves **5.65 ms** of
source-age budget before any decode, rectification, geometry or rendering.
Even a 20 ms tracked fit would exceed the budget for that median frame.
`obsolete_detector_result` therefore masks an additional acquisition failure;
freshness is not the reason the 3D solver never received a head proposal.

`--no-qr-decode` disables the viewer's background identity decoder. It does not
disable marker processing in `model_pipeline.py:178–199`. That stage still runs
after physical-head acquisition fails. Moreover,
`qr_pose_seed.py:163–190` tries `detectMulti`, `detect`, and then
`detectAndDecodeMulti`; the outer `allow_decode_fallback=False` does not skip
the last call inside the native helper. This explains why work remains even
with identity decoding disabled.

The receiver already uses depth-one sensor QoS, a dedicated spin thread and an
overwrite-latest buffer. The metadata cannot distinguish camera capture or
publication delay, transport/callback delay, or source/receiver clock offset.
Do not attribute the 244 ms age to a viewer FIFO without additional evidence.
This recording establishes viewer timing, not the autonomous observer's
transport latency or its separately configured 500 ms source-age limit.

## In-place saved-image replay

Raw source PNGs were rectified using each frame's saved calibration. Crop
experiments subtracted crop origins from the principal point exactly once.
They ran the unchanged physical-head solver and acceptance checks.

On frames 0, 5, 10, 15 and 19:

| Search | Crop in full-image coordinates | Usable fits | Median processing |
| --- | --- | ---: | ---: |
| Full image | `[0,0,800,600]` | 0/5 | 237.7 ms |
| Wider central-right crop | `[330,150,590,405]` | 0/5 | 83.8 ms |
| Tight central-right crop | `[390,180,560,350]` | 0/5 | 97.8 ms |
| Smaller left/below rectangle | `[315,280,430,385]` | 0/5 | 67.5 ms |

All failed the same comparison limit. Cropping reduced time, but it did not
solve selection. In the tight central-right crop, the locator generated
194–218 hypotheses instead of 41–51 in the corresponding full images. Cropping
changes contour endpoints, line selection and scale/position strata; fewer
pixels do not necessarily mean fewer competing rectangles.

A second replay supplied diagnostic expected centers and heights to the
existing candidate-projection locator, before geometric fitting:

| Diagnostic region | Expected center / height | Usable fits | Accepted angle range | Median processing |
| --- | --- | ---: | ---: | ---: |
| Central-right rectangle | `(477,265)` / 132 px | 19/20 | 7.87°–15.77° | 424.5 ms |
| Smaller left/below rectangle | `(371,332)` / 56 px | 20/20 | 1.54°–2.87° | 243.3 ms |

These manually specified image priors are a diagnostic experiment, not
candidate-association evidence. Neither result establishes the physical angle
or correct identity of candidate 2. The larger region changes border selection
across frames, accounting for a substantial angle spread. Both regions produce
accepted geometric fits, demonstrating why simply accepting any plausible
overlay or increasing the comparison limit is insufficient. The unbounded
projection-locator replay also exceeds the live timing budget.

## Correction indicated by the evidence

1. Apply the candidate's bearing and measured-head scale bounds **before**
   ranking/strictly verifying generic rectangles. Use map/LiDAR projection or
   a previous verified detection only to locate pixels. Keep complete-head
   margins and refit the four current borders. Do not derive the angle from QR
   size or restore the neck requirement.
2. Share this candidate-constrained acquisition between viewer and exploration.
   The newly implemented observer center crop is useful but still applies size
   filtering after a generic cold search; it cannot resolve this comparison
   failure by itself. The debug viewer still takes its full-image path.
3. Preserve physical-border families across current fits and test their
   consistency, rather than switching between nearby inner and outer rails.
   Wrong-candidate/background rectangles must remain rejected by association.
4. Budget QR marker work explicitly. When no complete head exists, skip or
   defer unnecessary marker/decode work. Record skipped marker checks as
   **unknown**, never as QR absence or backside evidence. Run the required
   current marker checks once a complete associated head is available.
5. Trace image capture, publication and callback receipt times with clock
   alignment. Reducing locator work alone cannot recover a 250 ms freshness
   budget already nearly exhausted at arrival.

The next validation should require complete-head acquisition on this saved
sequence, stable border/angle selection on the intended associated target, and
fresh live completion. It must also retain rejection of the competing
background rectangle. The simpler frontside admission rule applies after
these measurements exist; it cannot cure missing proposals or expired images.
