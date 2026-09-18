# Stand viewer color edge mask

**Detection limitation confirmed by full replay:** the latest light-grey front
recording (`recording_20260917_162111_735876885`) produces no accepted head fit
in any of its 50 frames with either the five-color or grey-only mask. The mask
removes necessary head-border evidence. Edge reduction below must not be read
as successful head detection. See the [full detector replay](../../results/aufgabe04/implementation_checks/viewer_color_edges_20260918/latest_head_replay/report.md).

The stand-axis debug viewer filters its Canny edges by HSV color support by
default. Both measured-model fitting and legacy edge fitting consume these
filtered edges. The autonomous observer's default behavior is unchanged.

Add these options to your usual viewer command:

```sh
--display-mask --display-edges
```

The mask window shows the actual allowed color support. The default
`--edge-color all` uses the physical stands' light-grey, blue, green, red, and
purple palette, including both ends of the red hue range. Use `--edge-color green` (or another
palette color) to reject objects of other colors. Color alone cannot distinguish
a stand from background objects of the same color. In particular, light-grey
walls and blue window blinds can survive; the existing candidate ROI and
geometric checks still determine which edges belong to a stand.

Initial viewer-only HSV ranges were checked against source frames from 20
workstation debug recordings dated September 11–17, 2026. The light-grey head
appears cool blue/purple under that lighting, so its range targets that cast.
An unrestricted neutral-grey range selected almost every wall and QR interior
in replay and is deliberately excluded. Retune light grey if the lighting or
white balance changes. Close views are available for grey, purple, and red;
blue and green appear as distant stands. These are initial thresholds, not a
calibrated guarantee for every lighting condition.

Use `--color green --tune` to initialize the live HSV controls from green and
override the edge color selection with the trackbar range. `--color` without
`--tune` still controls the separate side-classification mask. Use
`--no-color-edge-mask` for the previous unfiltered edge behavior.

The filter retains original Canny pixels within two processing-image pixels of
the external color contours. A 3×3 closing bridges segmentation pinholes; the
external contour suppresses QR texture enclosed by a colored frame. Support
must also remain within two pixels of the original matching color. It does
not synthesize measured edges or remove thin rails through morphological
opening. The original
image remains available for QR decoding. An empty color match yields no edge
evidence; it never falls back to background edges. Existing ROI and wall
exclusions still apply. `--no-morph` and the morphology settings control the
legacy contour/classification mask, independently of the edge support mask.

## Offline validation, 2026-09-18

One original source frame from each of 20 workstation debug recordings was
replayed with channel-union Canny, blur 5, thresholds 20/60. The mask removed
59–82% of all Canny pixels (median 65%). This measures edge reduction, not
background classification accuracy or stand detection recall. Purple/red
outlines and the cool-grey backside remain visible; the grey front still has
some QR and similarly colored window clutter. No live camera run was performed.

Local evidence: [comparison](../../results/aufgabe04/implementation_checks/viewer_color_edges_20260918/color_edge_comparison.jpg),
[per-frame counts](../../results/aufgabe04/implementation_checks/viewer_color_edges_20260918/validation.json).

The focused color, shared-acquisition, viewer, physical-fit, recording,
metric-model, raw-support, and cache checks have 120 passing tests. One cache
test expects a native QR call that is skipped; it also fails with the pipeline
loaded from HEAD. A broader run found six failures in the unchanged legacy
stand-axis image tests. These were not changed as part of the viewer mask work.
