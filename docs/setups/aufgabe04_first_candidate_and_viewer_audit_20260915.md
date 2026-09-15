# First-candidate admission and viewer acquisition audit — 2026-09-15

The first candidate was recognized as QR_003, but inconsistent current head
borders and processing latency prevented seven simultaneous admissible angles.
The latest viewer recordings additionally establish a new frontside acquisition
regression: QR finder rectangles exhaust the cold-search budget and then count
as competing heads. The recorded frontside contains usable 3D head geometry.

## Scope and provenance

- Real run: `stand_explore_exact2_camera_all5_20260915T134522Z`, recorded commit
  `1a451ac3710fe0500a1a6c3509f09e3ff0729d8a`.
- First candidate: `survey_candidate_0003`; only camera attempt 00 is recorded.
- Later viewer recordings: `recording_20260915_155843_422965485` and
  `recording_20260915_160107_616628722`. The final directory check found no newer
  recording. Their source timestamps are 13:58:43–13:58:49 and 14:01:07–14:01:13
  UTC, respectively. They do not record the earlier driving transition.
- Run JSON, saved scans and viewer source images were inspected in place on
  mii001. Only derived diagnostics returned. No images/recordings were copied.
- Image replays used the existing container with the repository mounted read-only,
  Python 3.10.12/OpenCV 4.5.4, recorded calibration and original decoded-source
  PNGs. No ROS, robot motion, source changes or deployment occurred.
- Viewer metadata does not record the code revision/runtime. Its acquisition
  policy matches current remote HEAD `1a451ac`; that is not a recorded commit proof.

## The rotation, approach and localization evidence

| Event | UTC | Evidence |
| --- | --- | --- |
| Execution transform sealed | 13:50:20.777 | `map←odom=(-1.558193,-0.359867,-0.208473 rad)` |
| Initial alignment starts | 13:50:22.024 | Zero linear command; angular command starts at −0.06 rad/s |
| First forward command | 13:50:33.924 | Linear command 0.0100 m/s after approximately 11.9 s of alignment |
| Terminal alignment | 13:50:39.824 | Translation stops; odom goal error 0.0294 m |
| Approach completed | 13:50:41.260 | Recorded distance 0.2756 m; no continuity stop |
| Stopped arrival localized | 13:50:47.225 | `map←odom=(-1.547231,-0.348541,-0.218511 rad)` |
| Camera observations | 13:50:48.984–13:51:45.983 | QR identity succeeds; joint angle completion fails |

Between execution sealing and stopped arrival, the transform changes by **0.575°**,
**1.58 cm at the odom origin**, and **1.62 cm at the certified route-start anchor**.
This leg allowed **11.10 cm anchor drift and 0.10 rad (5.73°) yaw drift**. These
are endpoint comparisons, not the maximum transient motion correction.

The stopped-arrival geometry passed: expected range **0.5394 m**, bearing error
**0.900°**, within the 0.33–0.70 m / 3° limits. Those checks use estimated poses;
they do not establish scan-to-wall accuracy.

The controller trace's `map_pose` field is the execution-frame pose on this odom
leg, not a continuous independent AMCL trajectory. Its control cycles contain
driving-behavior diagnostics but no per-cycle numerical `map←odom` history.
Consequently the saved records cannot locate or measure the reported transient
AMCL jump exactly when forward motion began.

Saved scans corroborate a modest persistent wall offset. Long straight scan
segments at the stopped camera pose have median angular deviation **+0.98°**
from map-wall axes (143 segments across 57 exactly synchronized scans), compared
with **+1.79°** at survey viewpoint 2. Fitted north/south wall intercepts move
**4.24 / 3.53 cm south**, and the west wall approximately **3.11 cm east**.
Against approximate map inner faces, horizontal walls are about **4.4–4.5 cm south**.
The map resolution is 5 cm; survey transforms are frozen and camera transforms
are current. This is corroborating displacement evidence, not external ground
truth or proof of a particular AMCL/odometry fault. The scans cover the stopped
period 13:50:49–13:51:22, not the driving transition.

## Why the first view failed admission

The observer processed **100 distinct images**: 42 fresh, 58 obsolete. It produced
28 usable 3D head fits, all fresh; 24 passed current-head LiDAR association.
Seven actual axis insertions occurred across the entire 57-second window, but
only **four coexisted in the required seven-sample window**. Of 25 temporal
reviews, seven were consistent and 18 found unstable angles/borders.

Two recorded changes identify the geometry failure:

1. At source stamp `1789480255.567292`, the nominal crop selected corners spanning
   approximately x=316–373 and admitted **+57.572°**. At `1789480255.982584`, a
   recentered crop selected the complete roughly 100-pixel-wide frame at
   x=277–379 and fitted **+24.612°**. The complete head's left border was outside
   the nominal crop beginning at x=286. The first result's QR binding said
   `qr_outside_current_head`; this is diagnostic evidence of a bad head region,
   not a reason to restore QR-size angle vetoes.
2. At `1789480300.975155`, four angles had accumulated with a current **+25.428°**
   fit. At `1789480302.166518`, changed top/left borders produced **−14.968°**.
   The largest corner change was **10.047 pixels**, while the target projection
   moved only **0.102637 pixels** and expected head height changed **0.000445 pixels**.
   The resulting 40.75° temporal span correctly cleared angle evidence.

There were **zero motion-epoch resets**. Across the observation window, the
projected horizontal center moved only **4.682 pixels**, vertical center 0.0594
pixels and projected depth 0.000110 m. The critical angle switch is therefore
explained by different selected current borders, not an AMCL-induced target
projection jump at that moment. All 100 records lack a QR-ratio boundary veto,
as intended by the latest change.

Processing also consumed the freshness budget. Image age at selection was
**71.2 ms median / 93.5 ms p90**, but detector time was **435.5 / 564.2 ms** and
result age **511.2 / 647.6 ms**. There were **92 same-image geometry refits**,
100 nominal ROI evaluations and 26 recentered evaluations. Median head
acquisition cost 125.6 ms, head fitting 9.1 ms and full QR decoding 5.1 ms.

QR_003 was decoded on 80 selected frames, including 25 fresh frames; its two-sample
identity latch succeeded. The final publication was `front_readable`, with no
angle or completion authority, after the 30-second front-geometry recovery budget
expired. Its seven advisory observations are not seven admitted angles. Another
inspection route passed dry preflight; the terminal ends at 13:52:01 during its
pre-run diagnostics. No subsequent camera view or normal mission completion is
established by this bundle.

## What the two viewer recordings add

The recordings use identical settings: full 800×600 images, no target ROI,
no stand coordinates/distance, fixed LiDAR bearing, `map_frame=odom`, no background
QR decoding, and unchanged 250 ms source / 180 ms receipt-age limits.

| Result | 15:58:43 recording, QR visible | 16:01:07 recording, mostly QR absent |
| --- | ---: | ---: |
| Frames | 57 | 54 |
| Fresh detector and rendered results | 57 | 54 |
| Accepted 3D fits / purple overlays | 0 | 46 |
| Cold verification-budget failures | 57 | 5 |
| Other acquisition failures | 0 | 3 |
| Median detector time | 44.51 ms | 42.88 ms |
| Median rendered source age | 109.51 ms | 140.18 ms |

The frontside failure is unrelated to obsolete results or AMCL. Every frame has
23–38 distinct locator hypotheses. At
`perception/stand_axis/head_cold_acquisition.py:225`, their count exceeds 12 and
aborts acquisition **before any strict border verification**.

The backside recording contains 44 current refits around tracked search locations
and two successful cold acquisitions. Its 46 admitted angles range −18.61° to
−12.95°, with reported uncertainty 1.46–1.88°. The absence of a registered
candidate projection still prevents backside *classification* on 45 fits;
one has a marker-presence veto. A purple undirected angle is not a registered
backside/opposite-side routing receipt.

## In-place original-pixel replay

For diagnostic purposes only, all bounded locator hypotheses were verified in
memory with a larger search-work allowance. No physical angle, raw-border or
uncertainty threshold changed. This does **not** establish a production selection
policy or fresh motion authority.

Front frames 0, 28 and 56 contain three accepted small raw rectangles at roughly
`(287,237)–(316,267)`, `(344,240)–(370,270)` and `(287,297)–(316,327)`.
Current QR checks independently verify three finders at these locations. All
three small rectangles fail the physical head's planar-angle ambiguity check.
Complete outer-head rectangles around `(277,227)–(379,336)` produce usable
angles near +25°. Yet the locator compares the internal finder rectangles as
separate heads, returning `head_proposal_ambiguous` once the budget abort is removed.
Simply increasing the budget therefore does not fix the pipeline.

Across **all 57 front frames**, the largest quality-valid current enclosing head
fit exists. Its yaw ranges **24.400–25.772°**, median **24.960°**, standard deviation
**0.289°**. This diagnostic selection is not proof that “largest rectangle” is
safe in other scenes; a room frame enclosing a real stand must remain a negative case.

Initializing the existing tracker from the first complete outer-head fit and
then running its unchanged current-pixel physical pipeline succeeds on **57/57
frames**, with yaw **24.76–25.12°**. Each angle is refitted from that frame; no
historical angle or QR geometry replaces current measurement. This demonstrates
that acquisition/selection is blocking usable frontside geometry.

A simple crop alone did not solve this recording: the diagnostic ROI
`(257,207,400,357)` still produced 179–190 hypotheses and hit the budget limit on
the three sampled frames. Current internal texture, not merely distant clutter,
must be handled in the acquisition design.

## Recommended correction boundaries

Use the complete current head proposal directly as the input to the measured
3D fit. Combine raw four-border verification, physical pose validity and angular
uncertainty into one geometric result. QR should remain identity/side evidence;
neck validation remains irrelevant to the angle.

Acquisition must rank and group complete outer-head hypotheses before an
all-or-nothing budget decision, retain a bounded candidate-aware workload, and
distinguish contained texture rectangles from genuinely competing physical heads.
Do not restore a fixed QR-size relation, accept arbitrary largest rectangles,
or promote a geometrically ambiguous angle solely because its outline looks aligned.
Keep the complete-head crop after reacquisition and reuse the same current head
fit when decoded identity arrives, avoiding repeated geometry work and nominal/
recentered border switches.

For robot motion, separately retain fresh candidate binding, stable stopped
evidence and collision-free route admission. A current undirected angle can be
computed/displayed before a stand has enough evidence for motion commitment.

Localization merits its own follow-up: record numerical live `map←odom`, odom
pose, AMCL pose/covariance and commanded phase around turn-to-translation; compare
fresh scan-wall alignment while stopped before translating; and bound landmark
drift since survey. Current arrival reprojection applies one transform to robot
and canonical-odom target, so a pure completed global correction cancels from
their relative bearing/range. Continued localization changes during observation
can still shift the fixed arrival-map target or clear evidence. That mechanism
exists in code, but it was not the observed reset mechanism in this first view.

## Integrity and local verification

Selected source SHA-256 values were computed remotely:

- Run observer events: `72776eb2373aff4bc2af55848a8faefb90b1806b420c9d9a99ff4b0bb72b2aac`.
- Candidate controller trace: `98e24e3ca2d4647c01319d6c6865d750a56291bc31eeea3af80172c4df717edd`.
- Front viewer source frame 0: `9faffe86858acf70af0c68db6f9ae2a7b0a260f3da776148de909fb14e33f3eb`.
- Front viewer source frame 28: `f885974e8ee34af3c5c48086326e8685f2a8158c3d5e88fec0d769df73532952`.
- Front viewer source frame 56: `9ec38002150ff99d76f02af3daedf950aa144d55157854fe0c9cacd067889bef`.

Seven existing localization/candidate-association test suites passed **84 tests
and 132 subtests** on the audited source. Audit only; production code unchanged.
