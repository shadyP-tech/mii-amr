# Light-grey head filtering and camera-exploration background suppression

The current hard color mask damages the head borders. Combining it with the
new LiDAR candidate search region does not repair that damage. The useful
combination is **spatially constrained search with original measurement edges**;
color can guide contour proposals separately. Threshold changes recover usable
border evidence but do not, by themselves, produce an accepted pose for the
latest front-view recording.

This investigation changes no production filtering, acceptance gate, or robot
configuration. Experiments are offline and have no live processing deadline.

## Data and experiment scope

- Latest viewer recording: `recording_20260917_162111_735876885`, all 50 original
  images, rectified using each frame's saved calibration. Original channel-union
  Canny settings (blur 5, thresholds 20/60), measured model and source-image
  support checks were retained. No QR or previous pose was supplied.
- Exploration fixture: `mission_20260917_140949`, candidate 1, attempt 0, frame 25.
  This is the light-grey backside. The original image, scan, image-time
  transforms, candidate projection and association metadata reconstruct the
  new spatial restriction. Its bounds are `[136,168,574,402]` in an 800×600 image.
- The latest viewer recording does **not** contain the matching mapped
  candidate and synchronized scan needed to reproduce that exploration filter.
  Tests using a manually selected region on this recording are explicitly
  diagnostic location-oracle experiments, not LiDAR-backed validation.
- Runtime: local OpenCV 4.13.0. The 50-frame sequence records source hashes and
  confirms that perception source files did not change during that replay.
  These are not workstation timing measurements or orientation ground truth.

## Why the current grey mask fails

The viewer's light-grey range is H=105–140, S=30–85, V=65–255. It models the
cool-tinted backside but does not cover the front frame under mixed lighting.
The middle front-view frame has warm top/bottom border pixels (median H≈10/12)
and cool side pixels (median H≈136/145). Side saturation medians are only 25/24,
below the current minimum of 30.

Raw Canny pixels in a narrow band around a separately fitted diagnostic head
rectangle give the following retention on frame 25. This reference is a
successful fit from an approximate supplied location, not a ground-truth label.

| Border | Current five-color mask | Current grey-only mask | Hue-independent grey, S≤110, V≥35, radius 2 |
| --- | ---: | ---: | ---: |
| Top | 64% | 12% | 100% |
| Right | 54% | 50% | 100% |
| Bottom | 21% | 0% | 100% |
| Left | 81% | 81% | 100% |

Removing the external-contour step alone gives the same retention on these
border bands. The hue/saturation rejection is therefore the first problem.
Increasing dilation to twelve pixels still preserves only about 71% of the
bottom border and produces proposal ambiguity. Large dilation is not a reliable
repair for incorrect color membership.

![Retained edges in white and deleted edges in red](../../results/aufgabe04/implementation_checks/light_grey_optimization_20260918/border_evidence_comparison.png)

## Actual combination with the exploration filter

The observer currently uses the new spatial filter and **does not pass the
viewer color mask into fitting**. The following combinations were injected only
by the offline experiment, using the recorded exploration fixture's real
candidate/scan/transform context.

| Configuration on exploration frame 25 | Accepted fit | Evidence |
| --- | --- | --- |
| Candidate search, spatial region disabled, color off | Yes | Baseline |
| New spatial region, color off | Yes | 4,544 of 14,628 contour edges retained; 169 line segments excluded before quotas |
| New spatial region + current hard five-color mask | No | No usable head border |
| New spatial region + hue-independent grey hard mask | Yes | Same fitted result as spatial filtering alone; still 4,544 contour edges |
| New spatial region + current color mask on contour proposals only | Yes | 1,838 contour-search edges; original 14,628 raw edges retained for refinement |

The last row is an analysis-only wrapper around the region's `locator_edges`.
It leaves original LSD/Hough inputs, regional line screening and raw corner/
border fitting untouched. It is not an implemented production integration.
The hue-independent hard mask removes only 11 full-image edges on this fixture,
and none of the retained regional contour edges: its success does not establish
added benefit over the spatial filter.

This one successful backside fixture does not establish front-view or field
reliability. In particular, spatial filtering cannot recover pixels already
deleted by a hard color mask.

## Latest front-view sequence: optimization helps proposals, not accepted poses

All variants below use automatic full-image search with no fabricated candidate
region and unchanged pose-quality gates.

| Filter | Frames producing a selected head proposal | Accepted poses |
| --- | ---: | ---: |
| No color mask | 8/50 | 0/50 |
| Current five-color mask | 0/50 | 0/50 |
| H unrestricted, S≤110, V≥35, radius 2; no contour-outline restriction | 33/50 | 0/50 |
| H unrestricted, S≤60, V≥80, radius 4; no contour-outline restriction | 39/50 | 0/50 |

The final variant was selected from a 72-setting pixel-retention sweep on frames
0/25/49. It is an in-sample candidate setting, not a generally calibrated
threshold. It retained all measured border-band pixels on those samples while
retaining about 68% of all edge pixels. Its full-sequence result was 35 planar
pose ambiguities, four excessive-yaw-uncertainty failures, ten ambiguous head
proposals and one remaining verification-budget failure.

Frame 25 illustrates the remaining problem: the restored automatic proposal has
complete raw border support but a planar-pose ambiguity gap of only 0.045 px.
A separately supplied approximate location reaches another supported rectangle
with a 1.55 px ambiguity gap and an accepted fit. This demonstrates sensitivity
to competing border choices; it does not prove which inferred angle is correct.
The current selector initially chooses the largest verified border before
comparing related border families. The selection and physical-border ownership
need investigation; lowering pose-quality thresholds would hide the ambiguity.

Across all 50 frames, supplied-location diagnostics accept 37 with unmasked
edges and the same 37 with the broad grey mask, with identical per-frame
accept/reject reasons. This confirms that the broader mask preserves the
existing fitting capability, while automatic acquisition remains unresolved.

Region-only diagnostics on frames 0/25/49, and a five-margin region sweep on
frame 25, also failed to produce an accepted fit. These image-derived boxes
must not be represented as a valid replay of LiDAR filtering on the latest
recording.

## Recommended implementation sequence

1. Preserve original Canny edges for all physical-head refinements and corner
   checks. For light grey, remove the viewer's hard color exclusion from that
   measurement path. Keep the color mask available as a visual diagnostic.
2. Use the existing verified candidate/scan region to restrict contour and line
   search. Missing or ambiguous scan context must retain its existing explicit
   fallback behavior; a color match cannot manufacture association.
3. If color is needed, add a distinct locator-support input and use grey
   confidence to prioritize proposals inside that region. Use hue-independent
   low/moderate saturation as the starting hypothesis, with small spatial
   tolerance. Do not apply a global outer-contour or narrow-hue veto to grey
   measurement pixels. Keep saturated-color handling separate.
4. Resolve the front frame's competing border choices before pose estimation.
   Test outer-frame/inset/QR boundary ownership using current raw evidence and
   the measured model. Do not select an alternative solely because it yields
   an accepted yaw, and do not loosen ambiguity/uncertainty gates.
5. Validate the actual combination on a synchronized front-view capture with
   candidate, scan and image-time transforms, plus multiple stands, no-stand
   scenes and lighting changes. Require complete-border and accepted-fit
   rates, not just a reduction in edge counts.

## Reproducible evidence

Experiment directory:
`results/aufgabe04/implementation_checks/light_grey_optimization_20260918/`.

- [Initial ablations and border statistics](../../results/aufgabe04/implementation_checks/light_grey_optimization_20260918/probe.json), [script](../../results/aufgabe04/implementation_checks/light_grey_optimization_20260918/probe.py).
- [50-frame sequence summary](../../results/aufgabe04/implementation_checks/light_grey_optimization_20260918/sequence_summary.json), [full results and hashes](../../results/aufgabe04/implementation_checks/light_grey_optimization_20260918/sequence.json).
- [Threshold sweep](../../results/aufgabe04/implementation_checks/light_grey_optimization_20260918/threshold_sweep.json), [final threshold and unmasked location controls](../../results/aufgabe04/implementation_checks/light_grey_optimization_20260918/final_checks.json).
- [Manual-region sensitivity diagnostics](../../results/aufgabe04/implementation_checks/light_grey_optimization_20260918/region_sensitivity_DIAGNOSTIC.json).

Existing feature regression checks: 34 tests passed across
`test_lidar_head_edge_region.py` and `test_viewer_color_edges.py`. Passing these
tests does not override the recorded front-view detection failures above.
