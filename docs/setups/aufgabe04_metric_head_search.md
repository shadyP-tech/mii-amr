# Distance-constrained head-frame search

Camera exploration and the camera debug viewer share a physical head search
based on camera calibration, the measured stand profile, and optical-camera
depth. For an upright camera the nominal height is `fy * head_height_m / Z`.
The 78 mm head is approximately 125, 100, and 83 pixels high at depths of
0.4, 0.5, and 0.6 m with `fy=640`. These are examples, not measurements of the
latest recording.

Search intervals include depth uncertainty, model tolerance, unknown head yaw,
and pixel margin. Calibrated camera orientation is used by exploration and the
nearest-scan viewer mode. Frontal width is diagnostic; oblique heads are allowed
to be narrower. Size and location screening occur before contour and line
quotas, followed by the existing original-edge verification. Tracked results
must also satisfy the current metric bounds.

Exploration uses the current mapped-candidate projection, its configured
position/surface uncertainty, and the existing synchronized LiDAR search region.
The viewer's default `--head-target nearest` converts the fresh scan target into
camera coordinates using measured extrinsics. Raw LiDAR range is not used as
camera Z. Missing, stale, or ambiguous nearest-target context remains explicit.

For a measured optical depth without a known image location, append these
options to an existing calibrated physical-model viewer command:

```text
--head-target unique --head-depth-m 0.40 --head-depth-uncertainty-m 0.02
```

This manual option assumes an upright camera and searches the full processing
image by size. It requires `--stand-model-profile`; `--stand-distance-m` retains
its existing approximate-distance meaning. The viewer displays nominal height
and the allowed range. Its recording diagnostics preserve the depth and bounds.

The color mask still filters the mask/edge preview and the legacy detector.
Physical metric fitting uses original edges because a hard HSV veto erased
parts of the light-grey frame. `--no-color-edge-mask` shows the original edge
preview. Verified image borders and reliable yaw are reported separately;
uncertain yaw does not become navigation permission.

Validation covers projection at multiple distances and camera tilts, uncertainty,
clutter quota rejection, blank images, tracked size mismatch, freshness, and
separate border/yaw reporting. The recorded light-grey fixture with matching
scan, map projection, and camera transforms passes with both the region alone
and the new region-plus-size prior. The latest front-view recording lacks the
matching distance/transform context needed to establish an autonomous success
rate for this change. Live robot operation has not been tested.
