# LiDAR candidate constraints before head-border discovery

Camera exploration now uses the candidate retained from LiDAR exploration to
remove irrelevant contour edges and line segments before the head locator's
fixed work quotas. It is enabled automatically for measured physical stands.
No robot run or deployment was performed for this change.

## Geometry and association

The observer projects a possible head volume from the mapped candidate through
the camera-from-map transform already obtained at the image timestamp. The
measured head width, depth and floor-relative height define the volume. Unknown
head yaw is enclosed conservatively. Existing `--stand-uncertainty-m` and
`--stand-radius-m` settings cover position error and an exploration point on the
base/support rather than the head axis; model tolerance and a six-pixel image
margin are also included. The mask is recomputed for each image.

A fresh, synchronized scan must contain one eligible candidate cluster. If the
original narrow map cone has none, the observer searches the bearing envelope
already permitted by camera registration, retaining the original range interval
and requiring exactly one eligible cluster. It does not choose the nearest of
several clusters. This matters for the recorded candidate displaced from its
original map bearing. Missing, stale, ambiguous or invalid context disables
the mask and records the reason; existing camera and final association gates
remain active.

## Image processing

- Canny, LSD and Hough receive their original image/edge inputs.
- A separate copy of Canny edges is zeroed outside the projected region for
  contour discovery. Contours touching the mask boundary are discarded.
- LSD/Hough segments outside the region are rejected before allocating the
  rail quota. Keeping the original detector inputs avoids changing interior
  line estimates through crop or mask boundaries.
- Raw border refinement, corner evidence and competing-head checks use the
  unchanged current image and raw edges. The mask supplies no measured corners.
- Final corners must remain inside the region, including fits reached through
  pose tracking. QR identity and post-fit camera/LiDAR association remain separate.

Diagnostics include the projected bounds, scan association scope, fallback
reason, input/retained edge counts and rejected line-segment count. They are
stored in candidate-search and head-acquisition metadata.

## Validation and limits

Regression coverage includes moving-camera reprojection, orientation and
uncertainty margins, invalid/stale inputs, multiple scan clusters, contour
budget exhaustion from remote clutter, missing borders, two heads within the
region, unchanged raw evidence, tracked-fit checks and observer integration.

Validation completed with 634 tests and 729 subtests passing across 61 focused
perception regression modules. The new feature and observer integration checks
can be run with:

```sh
python -m pytest -q tests/aufgabe04/test_lidar_head_edge_region.py tests/aufgabe04/test_observer_measured_head_processing.py
```

An unmodified JPEG and sensor/transform metadata from
`mission_20260917_140949`, candidate 1, camera attempt 0, frame 25 are retained in
`tests/aufgabe04/fixtures/lidar_head_region/`. The fixture records hashes and
reconstructs the saved candidate from its recorded projection, not fitted head
pixels. The regression checks successful current-border detection after early
filtering; no expected yaw is supplied.

Offline replay of that frame retained 4,544 of 14,628 contour-search edge pixels
(about 69% removed), with a usable head fit before and after. The estimated yaw
changed from about -9.84 to -12.27 degrees as the locator's candidate set changed;
without ground-truth orientation this is not evidence of improved angle
accuracy. Replay of frame 5 found two eligible scan clusters, disabled masking,
and retained its unavailable-head outcome. These are workstation replays, not
robot timing or field-success measurements.

The LiDAR remains planar: this is a geometric search restriction, not per-pixel
depth segmentation. Background edges inside the possible head region remain and
must pass the existing visual checks. Calibration/model or map errors outside
the configured margins can exclude a real head. Live cluttered-scene validation
is still needed before claiming improved detection reliability.
