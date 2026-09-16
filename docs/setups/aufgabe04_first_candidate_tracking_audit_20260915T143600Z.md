# First-inspection admission audit — 2026-09-15 14:36 UTC run

The first inspection point contains usable frontside head geometry and a
decodable QR_003. Admission fails because the mission cannot establish tracking,
repeatedly reacquires inconsistent or ambiguous borders, and never retains seven
simultaneous accepted angles. The debug viewer uses the tracking path that the
named-candidate observer currently cannot use.

## Evidence and scope

- Run: `stand_explore_exact2_camera_all5_20260915T143600Z`, commit
  `5781ca1dcc3a14bf57470f36ab426b1cc087d1de`.
- Candidate: `000_survey_candidate_0003`, camera attempt `camera_lidar_attempt_00`.
- New debug recording: `recording_20260915_164324_041155730`, 59 images,
  14:43:23.960–14:43:30.268 UTC / 16:43 CEST. Viewer metadata does not itself
  record a git revision; the inspected/replayed checkout is 5781ca1.
- Run logs, scans, original compressed captures and viewer PNGs stayed on
  mii001. Only derived diagnostics returned. Replays used the existing
  Python 3.10 / OpenCV 4.5.4 container, read-only repository mount, recorded
  camera calibration, and unchanged production geometry thresholds.
- No production code changes, deployment, ROS commands or motion occurred
  during this audit.

## What the new recording establishes

All **59/59 recorded frames** are fresh and geometrically accepted, using
`tracked_head_search`, with QR marker evidence in every frame. Raw four-border
support is complete; the head model's angular uncertainty is approximately
1.6–1.7 degrees. The viewer has no named candidate projection or ROI.

The recording was made with `no_qr_decode=true`, so its marker evidence alone
does not establish decoded identity. An in-place replay of its original images
with the production QR decoder decoded **QR_003 in 59/59 frames**. Starting
without a manually supplied pose, then tracking by refitting each current image,
produced **59/59 usable head fits**, yaw **23.7009–23.8665 degrees**.

Cold acquisition independently on each image produced 31 usable fits, 22
ambiguous proposals and six acquisition failures. This isolates the benefit
of locating the existing complete head instead of restarting acquisition.

## First inspection: evidence funnel

Arrival passed at 0.538918 m and bearing error 1.5508 degrees. The observer ran
14:41:13.187–14:42:42.825 UTC and ended at its 90-second parent deadline.

| Stage | Recorded result |
| --- | ---: |
| Distinct processed images | 187 |
| Observer-fresh images (500 ms limit) | 129 |
| Fresh usable 3D fits | 60 |
| Fresh acquisition ambiguities | 69 |
| Fresh geometry rejected by LiDAR association | 11 |
| Geometry + association temporal reviews | 49 |
| Successful angle insertions over the whole observation | 39 |
| Temporal instability rejections/resets | 10 |
| Maximum simultaneous accepted angles | **6 / 7 required** |
| Tracker updates accepted | **0 / 187** |

The remaining 58 obsolete images contain eight usable fits, 48 acquisition
ambiguities and two corner failures. QR_003 was decoded in 125 selected images;
35 images jointly had bound QR and usable geometry. Its two-sample identity
latch succeeded at source time **14:41:23.148 UTC**. Identity availability was
therefore not the final blocking condition.

At source **14:42:19.739 UTC**, frame 140 reached six angle samples. The next
twelve processed images contained nine fresh acquisition ambiguities, one fresh
geometry/LiDAR rejection and two obsolete results. All six samples expired
within their five-second lifetime. Frame 153 at 14:42:25.507 restarted at one.

## Why the viewer succeeds while the mission fails

### Tracker seed age prevents every update

The observer constructs `MetricPoseTracker(prediction_ttl_sec=0.25)` in
`real_robot/observer/node.py`. `perception/stand_axis/pose_tracking.py` also uses
that TTL as the maximum image age at processing completion before retaining a
pose. All 187 images were older than this limit: minimum **251.66 ms**, median
**443.62 ms**. Even the 129 observer-fresh images had median age **407.70 ms**.
Every update reports `pose_observation_stale`.

Independent acquisition costs **225.05 ms median**, while the actual 3D fitting
cost is **7.22 ms median**. The expensive acquisition cannot bootstrap the fast
tracking path under the current timing contract. The previous correction's
same-image QR optimization is working: **143 geometry reuses, zero refits**.

### Named-candidate fitting ignores the tracker hint

Every observer ROI call supplies expected head center and size. In
`perception/stand_axis/physical_head_pipeline.py`, those fields select
`candidate_projection` before the `pose_hint` branch. The latter is only in the
no-projection branch used by the viewer. Fixing seed freshness alone would
therefore leave the mission repeating projected acquisition.

### Complete front-head search context is not retained

`camera_target_registration.py` starts each ordinary frame with the nominal
ROI. `backside_proposal_reuse.py` retains only QR-empty backside evidence and
rejects hints after marker evidence or a tracked pose. It cannot retain a
successfully registered front head. The run selected 173 nominal crops and 14
registered crops.

This run does **not** require crop clipping to explain the failure. Its initial
nominal crop, x=277–460/y=213–396, already contains the complete observed head
around x=289–389/y=229–336. Repeated selection of different current borders is
the observed defect.

Frame 104 at **14:42:03.339** fits the complete head at **23.2927 degrees**,
with corners approximately `(290.12,230.01), (388.49,236.95), (387.69,336.29),
(289.28,334.82)`. Frame 109 at **14:42:05.922** selects the inner rectangle
`(295,229.30), (383,236.87), (383,331), (295,331)` and returns **32.7183 degrees**.
The target projection moved only **0.144 pixels**. Temporal consistency correctly
rejects the 9.54-degree interval and 7.09%-of-head-height corner change. Returning
to the outer frame at frame 111 does not instantly remove the retained outlier.

There were zero motion resets and a constant stopped anchor pose. The complete
observation's projection spans only 3.373 horizontal pixels, 0.03894 vertical
pixels and 0.000121 m depth. No evidence ties these abrupt angle switches to an
AMCL reset during observation.

## Replay of the actual first-inspection captures

The capture limit saved 64 records; **50** contain completed detector metadata
suitable for exact selected-crop replay. Recorded calibration, selected crop,
crop-adjusted intrinsics and supplied neutral proposal (when registered) were
used. The baseline replay reproduces the recorded estimator reason in **50/50**:
20 usable fits, 29 proposal ambiguities and one corner failure. QR_003 decodes
in **50/50** selected crops.

A diagnostic comparison retained the camera-space pose from the first replayed
fit whose recorded current-head candidate association was accepted:
`frame_000010.json`, source **14:41:16.360 UTC**. For each following saved image,
its pose located the current four-border search; all accepted corners and angles
were recomputed from that image's raw pixels using the unchanged physical fitter.
The replay did not borrow a historical angle or add QR geometry to the fit.

From that seed onward, **46/46 current geometric fits succeed**, yaw
**23.6956–24.5045 degrees**. Seven usable geometric samples exist by
`frame_000017`, **3.184 seconds after the seed**, within the current five-second
sample window. Across all 50 captures, the comparison yields 48 usable fits;
the two failures precede the first associated seed.

This establishes geometric feasibility on the actual mission images, not a
replayed motion authorization: fresh runtime delivery, temporal border review,
current unique LiDAR association and QR-to-head binding must still be validated
together. Only the saved first approximately 22 seconds were replayed; the late
six-sample expiry and border switch are established by event diagnostics.

## Secondary LiDAR association issue

Of eleven fresh usable-head association failures, nine split into ambiguous
registered clusters and lack **three independent witnessed scans**; two reject
stale persistence sources. Captures 5, 23 and 34 contain eligible indices
`[1,3,4]` with index 2 missing/NaN, at ranges around 0.55 m. This is an internal
missing beam, not endpoint seam splitting. Bad circular metadata in those scans
does not explain this split.

Current persistence is fed through the geometry/registration association paths,
so repeated head-proposal failures also starve scan witness history. The fix
should retain independent fresh stopped-scan evidence with its exact transform
and candidate context, without joining arbitrary competing targets.

On 77 fresh frames, QR binding reports `camera_bearing_outside_map_cone` when
no independently accepted head enables camera registration. Observed QR bearing
is about 0.085 rad versus projected 0.025 rad, roughly 3.4 degrees apart and
outside the native 3-degree cone. Stable complete-head association is the
existing route to re-registration; enlarging every cone would conceal ambiguity.

## Final recorded run stop

After the camera deadline, the parent records `unobservable` and
`candidate_observation_unavailable` and proposes local inspection 001. Although
the terminal excerpt ends after general preflight PASS, events and CSV establish
a later admission failure at **14:42:56.949 UTC**, before motion:

`route uncertainty budget exhausted: limiting_segment=segment:0000:0146 remaining_margin=-0.060079 m`

The 0.734673 m proposed route has limiting clearance **0.322500 m** versus
required **0.382579 m**. Components are robot radius 0.105 m, localization 2σ
0.102733 m, heading contribution 0.089846 m, tracking 0.030 m, empirical drift
0.020 m, braking 0.015 m and collision margin 0.020 m. Stationary map/odom
continuity passed with zero sampled drift. This final stop is a route-clearance
uncertainty failure following the camera timeout, not a TF startup failure.

## Recommended correction and acceptance test

Introduce one small candidate-scoped complete-head search context shared across
front/back observations. Seed it from a fresh, geometrically verified,
candidate-associated head. Retain full-image corners/crop and camera-space pose
as **search information only**, independent of QR presence or side label.

- Separate seed eligibility under observer freshness from search-hint expiry;
  account for the measured acquisition latency and inter-image interval. Keep
  current-angle publication freshness unchanged.
- Make the named-candidate path actually use this context, retaining complete
  head bounds and correct crop-adjusted intrinsics. Current pixels must verify
  every border; lost/ambiguous current geometry grants no angle.
- Revalidate original candidate bounds, current unique scan association and
  current QR binding on every image. Invalidate on candidate/calibration/model
  changes, movement, expiry or head loss. A hint carries no side classification.
- Feed scan persistence from independent fresh stopped scans with appropriate
  transforms instead of only sparse successful geometry frames.

Acceptance must exercise the real named-candidate observer path: **seven fresh,
temporally consistent, uniquely associated current head angles plus bound QR_003
from the first inspection point**, then normal completion without a fallback
local inspection. Include disappearing/clipped heads, neighboring stands,
movement and stale inputs as rejection cases. The debug viewer's success alone
is insufficient validation of that integration.

Selected recording PNG SHA-256 values for provenance:

- Frame 0: `f36f5c16011d9c02378e1bb3fb8fd123115d865365a856862b2a2b5f54ccef0c`
- Frame 20: `d3143b38f206fb782bf73bde9e03345471dbc18018c4f498902bb5634b2d8b18`
- Frame 40: `92b83730270aa0ed56ad2efd306a2ed0b8b89e1e6a225af5c2c3c3b0f39249a4`
- Frame 58: `1e9307cec1ad9cb4ff26b03a8010fbbd72fb3b48b049b8c431e20ba0d1bdd168`
