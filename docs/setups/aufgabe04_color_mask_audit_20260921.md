# Whole-stand color-mask audit, 21 September 2026

Latest workstation run verified over SSH:
`stand_explore_exact2_camera_all5_20260921T131738Z`, deployed revision
`56892e38c9f5aece9e9d65f7f4db8b5ab6c19c85`. No newer autonomous-exploration
run was present. Local HEAD is newer and has additional work in progress;
this audit changed no production code or robot state.

## Actual mask contract

The physical metric head path computes the shared five-color HSV palette on
the full rectified image. `color_edge_support` closes small holes, takes external
contours and creates a thin dilated outline-support band. It does not return
a filled stand-instance segmentation. The metric estimator uses this support
to rank otherwise supported raw rails. It does not delete all other raw edges.
The recorded policy is `coherent_rail_ranking_only`, with
`raw_edges_unchanged=true`, in all 373 second-candidate processed-frame events.
The shared palette and mask routines are unchanged between deployed and local
code; reconstructed masks use those routines, not new thresholds.

The separate LiDAR head-volume mask limits head-location hypotheses. It is
not a mask of the entire stand. It was available for 321 second-candidate
processed frames and unavailable for 52. Current raw evidence still has to
verify proposed borders.

## Recorded images and measurements

The first inspected stand is red and front-facing. Its color selection visibly
connects head border, post and visible base. Some base geometry lies beyond the
image. Selected small interior patches of these parts have 100% combined-palette
coverage in the final saved debug image. Background pixels are also selected;
this is not an exclusive instance mask.

The second inspected stand is light grey and back-facing. It is not captured
as a complete object by the color mask. The following percentages are exact
mask coverage of manually chosen interior patches, **not segmentation recall
against a pixel-perfect whole-object annotation**:

| Grey-stand patch | Median over 256 saved compressed frames | Final debug image |
| --- | ---: | ---: |
| Head interior | 100% | 92.6% |
| Post interior | 99.4% | 10.3% |
| Base socket | 0% | 0% |
| Foot top surface | 0% | 0% |

The 256 captures cover the first 60.52 seconds of the camera session; the final
debug frame is approximately 28 seconds later. Do not confuse early capture
coverage with end-of-session coverage. Every capture was rectified with its own
saved CameraInfo before applying the unchanged palette. Replay used local
OpenCV 5.0.0; the robot used OpenCV 4.5.4.

The grey range is H=105–140, S=30–85, V=65–255 in OpenCV's HSV units. In the final
image the head patch has median HSV (107,35,211), close to the thresholds. The
post has (105,27,191), below the saturation minimum. The socket is (90,8,201),
and the selected foot surface is (8,33,124), outside the grey hue range. These
measurements establish appearance variation; they do not isolate its cause
between illumination, reflections and automatic camera settings.

The final combined mask's largest component covers mainly the head:
99×109 pixels, 7,145 selected pixels. Another large component covers floor
texture: 130×131 pixels, 5,489 pixels. A global all-palette mask therefore both
misses stand parts and includes background. Simply relaxing grey thresholds
would also admit more neutral wall, floor and metal pixels.

All 373 second-candidate processed frames reported
`model_current_head_border_unavailable`. Head/post color coverage was already
high in the early recording, so missing color is not a sufficient explanation
for the failed head acquisition. The previously audited displaced candidate
projection and finite-distance association issues remain separate.

## What a whole-stand segmentation would add

A useful next design is candidate-specific instance segmentation, with color
as a seed and current LiDAR, image boundaries and connectivity constraining
its growth. It should include attached white/black QR regions and desaturated
or shadowed stand parts rather than treating every non-palette pixel as
background. Foreground confidence should remain distinct from verified raw
borders; uncertain masks must not manufacture or erase physical measurements.

A model fit can then score visible head, post and base evidence jointly while
allowing occlusion and clipping. That can strengthen location and reject
background edges, but cannot guarantee pose/side observability or QR identity.
Registration, freshness, ambiguity and temporal-consistency checks remain
necessary. It should not require every stand pixel or the entire base to be
visible before accepting a well-supported head.

The active measured model provides a 78×78×6 mm head, 210 mm total height and
153×153 mm base extents. It supplies no measured stem width/visible-height
values and no foot/socket mesh or silhouette. A complete-stand silhouette fit
therefore needs additional measured geometry; treating the base extents as a
filled square would not represent the recorded star-shaped base.

## Reproduction artifacts

`results/implementation_checks/color_mask_audit_20260921/` contains:

- `analyze.py` and `metrics.json`: latest red and grey image masks, component
  statistics, exact patch coordinates and HSV statistics.
- `sequence.py` and `sequence_metrics.json`: all 256 captured grey frames and
  counts from the recorded observer events.
- `grey_image_vs_mask.png`: recorded image next to reconstructed combined HSV
  selection. White means palette membership, not certified stand identity.
- `grey_mask_analysis.png` and `red_mask_analysis.png`: image, HSV pixels,
  contour-support band and diagnostic intersection with saved raw Canny edges.
  That last intersection illustrates a hypothetical hard color filter; it is
  not the physical metric path's actual edge-exclusion policy.
