# Aufgabe 04 Perception Debug Viewer

These tools are debug-only. They visualize compressed camera frames, HSV masks, masked previews, ROI color classification results, and square-stand axis estimates.

It does not command robot motion, does not publish `/cmd_vel`, does not send Nav2 goals, and does not execute station approach behavior.

Viewer output is not real-parkour validation evidence. Real-robot Aufgabe 04 claims require the separate real-parkour checklist, run logs, and recorded mission evidence.

Keep all OpenCV window, camera, and live-debug code in this debug package. Pure perception logic belongs in `scripts/aufgabe04/perception/color_classifier.py` and should be covered by offline tests only.

## Usage

Run this only when the robot is stationary or physically secured. Confirm no autonomous mission, Nav2 goal, custom follower, or station-route runner is active.

The intended topic split is:

- TurtleBot: `camera_ros` publishes `/camera/image_raw`.
- TurtleBot: the QR scanner publishes decoded payloads on `/qr_scanner/decoded`.
- Workstation: this debug viewer subscribes to `/camera/image_raw/compressed`.

Start the TurtleBot built-in camera on the robot or ROS host:

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash

export ROS_DOMAIN_ID=30
export ROS_LOCALHOST_ONLY=0
export TURTLEBOT3_MODEL=burger
export LDS_MODEL=LDS-02

ros2 run camera_ros camera_node --ros-args \
  -p width:=320 \
  -p height:=240 \
  -p format:=BGR888
```

Run the viewer from the Apptainer workstation environment with the helper:

```bash
scripts/aufgabe04/perception/debug/run_color_mask_viewer.sh \
  --compressed-image-topic /camera/image_raw/compressed \
  --color green \
  --roi 120,100,220,180 \
  --max-display-fps 15 \
  --max-frame-age-sec 0.25 \
  --display-mode frame \
  --no-morph \
  --tune
```

The helper enters `/workspace/mii-amr`, sources ROS 2 and the TurtleBot
workspace, sets the standard TurtleBot environment defaults, and then calls the
viewer module.

If the environment is already sourced, the direct module form is:

```bash
python3 -m scripts.aufgabe04.perception.debug.color_mask_viewer \
  --compressed-image-topic /camera/image_raw/compressed \
  --color green \
  --roi 120,100,220,180 \
  --max-display-fps 15 \
  --max-frame-age-sec 0.25 \
  --display-mode frame \
  --no-morph \
  --tune
```

The viewer only supports ROS 2 `sensor_msgs/CompressedImage` input. Local OpenCV camera indexes, raw `sensor_msgs/Image` topics, and video files are intentionally not supported in this Aufgabe 04 debug tool.

The viewer uses sensor-data QoS with queue depth 1 for low-latency live video. If the viewer is running, `ros2 topic info /camera/image_raw/compressed -v` should show one subscription from `aufgabe04_color_mask_viewer`.

Audit the live topic before running:

```bash
ros2 topic list -t | grep camera
ros2 topic info /camera/image_raw/compressed -v
ros2 topic hz /camera/image_raw/compressed
```

The expected topic type is `sensor_msgs/msg/CompressedImage`. If `/camera/image_raw/compressed` is missing, fix the camera/image-transport setup on the TurtleBot side instead of falling back to raw video over Wi-Fi.

Useful keys:

- `p`: print the active `ColorRange(...)` threshold.
- `s`: save frame, mask, and preview snapshots when `--save-snapshot` is set.
- `q` or `Esc`: quit.

Tune thresholds in the actual lighting where the stands will be seen. Prefer selecting a stand ROI over classifying the full frame, because full-frame classification dilutes confidence with background pixels.

The low-latency default is a single annotated frame window. Use `--display-mode frame-mask` to also show the mask, or `--display-mode all` to show frame, mask, and masked preview. Extra windows can add visible lag over Apptainer/X11.

The ROS callback stores only the latest compressed frame bytes. JPEG decoding, masking, classification, and display happen outside the callback. The mask is built with vectorized OpenCV operations and ROI confidence is computed from `cv2.countNonZero(mask_roi)`. Lower `--max-display-fps` if the laptop is overloaded; the ROS subscriber still keeps only the latest received frame in a background thread. Duplicate-frame checks are disabled by default; enable `--detect-duplicates` only when diagnosing frozen input.

The frame overlay includes `age=...ms` when the incoming image has a ROS header stamp. Incoming stamped frames older than `--max-frame-age-sec` are dropped in the callback before conversion; use `--max-frame-age-sec 0` to disable that guard. A large displayed age means frames are still stale before display, while a small age with laggy windows points to local rendering overhead.

## Stand Axis Viewer

The stand-axis viewer uses edge detection by default, so the axis estimate does not depend on the stand color or on the QR-code side being visible. The default edge preprocessor is `outer-border`, which smooths QR/internal texture before Canny so the edge image is dominated by the stand outline rather than QR modules. It has two interpretation modes:

1. Face visible: the outer square is detected as a quadrilateral and the tool reports the left/right edge-height ratio.
2. Edge-on: the square face has collapsed to a thin vertical line and the tool reports approximate side-on / 90 degrees. In this mode it does not compute a height ratio.

In face-visible mode it overlays the detected outer quadrilateral and reports:

- `L` / `R`: apparent pixel heights of the left and right stand edges.
- `ratio`: `L / R`.
- `closer`: the side with the larger apparent edge height.
- `camera axis rot proxy`: `(ratio - 1) / (ratio + 1)`, a signed rotation cue relative to the robot camera.
- `med_ratio` / `med_proxy`: median-filtered values over the recent usable frames.

Sign convention: positive means the left image edge is taller/closer to the camera; negative means the right image edge is taller/closer.

In edge-on mode it overlays the detected vertical line and reports:

- `camera axis approx side-on / 90deg`
- `line_height_px`
- `ratio unavailable`

The console output mirrors this with either `camera_axis_rotation_proxy=...` or `camera_axis_edge_on_approx_90deg=true`.

The viewer can open up to six OpenCV windows. They all come from one process
and one camera subscription:

| Window | Enabling option | Meaning |
| --- | --- | --- |
| `aufgabe04/stand-axis` | always shown unless `--headless` is used | Final annotated camera frame. |
| `aufgabe04/stand-axis-mask` | `--display-mask` | HSV/color segmentation mask. |
| `aufgabe04/stand-axis-edges` | `--display-edges` | Canny/morphology edge image used by the edge path. |
| `aufgabe04/stand-axis-side-evidence` | `--display-face-mask` | Native-pixel cutout of untouched, color-agnostic Canny pixels selected independently near the four topology-proposed head sides. The historical option name remains for command compatibility. |
| `aufgabe04/stand-axis-rectangle` | `--display-rectangle-mask` | Native-pixel cutout of the temporally selected outer quadrilateral. In the real-camera path this preserves the common parallel-rail constraint and suppresses isolated switches to an inner Canny band. |
| `aufgabe04/stand-axis-raw-proposal` | `--display-raw-proposal` | Unfiltered per-frame detector proposal. It remains visible during temporal bootstrap/rejection for diagnosis, but never drives the accepted overlay. Recordings use `raw_proposal.avi`. |

The side evidence and fitted rectangle are independently selectable, so similar
diagnostic views do not have to be opened together. These are not additional
detectors, camera subscriptions, QR inputs, navigation inputs, or reportable
run evidence. To show only the annotated camera frame, omit `--display-mask`,
`--display-edges`, `--display-face-mask`, `--display-rectangle-mask`, and
`--display-raw-proposal`.

For a time-aligned debugging record, press `r` while an OpenCV viewer window
has focus. Press `r` again to stop. A red circle in the upper-right corner of
the annotated stand-axis window signals that recording is active; there is no
start/stop terminal message. The viewer writes one MJPEG AVI per window that
is currently displayed (the annotated frame plus each enabled diagnostic
window), so no window is hidden inside a composite video. A timestamped
recording directory is created under
`results/aufgabe04/stand_axis_debug_recordings` by default; change it with
`--record-dir PATH` or set the output rate with `--record-fps N`. The recordings
remain diagnostic-only and do not affect the detector or control the robot.

In standalone raw-simulation edge mode (`--lidar-bearing-source fixed`), Canny
is applied to the complete camera frame and `stand-axis-edges` displays that
complete edge image. The stem-anchored detector then derives a head ROI from
the detected silhouette; only `stand-axis-side-evidence` and
`stand-axis-rectangle` are cropped to that dynamic head ROI. In `map-target`
mode, the synchronized projected target crop remains the input to the edge
pipeline. The mask and full-edge windows preserve their aspect ratio and are
capped by `--diagnostic-window-size-px`. The side-evidence and rectangle
cutouts use OpenCV `WINDOW_AUTOSIZE` at native pixel size; processed pixels are
never resampled.

Because the simulated camera has zero roll/pitch and every stand head is
upright, the silhouette fit constrains the left and right head sides to the
same vertical image direction. Non-simulation edge callers jointly estimate a
single tolerant side direction instead. Top and bottom remain independently
sloped under perspective. Only sufficiently covered, outermost raw-Canny side
fits are accepted; a morphology-only rough rectangle cannot become a pose.

For the real camera, raw-supported nested candidates are resolved in favour of
the largest foreground-originated outer frame before temporal gating. Initial
tracking uses a three-of-five structural consensus over head centre, side
height, common rail direction, and neck ownership; projected width is not an
identity constraint because it contracts at oblique yaw. Once acquired, the
parallel-rail trapezoid is filtered as a line state. An isolated inward change
of only the top, bottom, left, or right boundary is held on the previous outer
border unless it persists for three frames.

Run it from the Apptainer workstation environment with:

```bash
scripts/aufgabe04/perception/debug/run_stand_axis_viewer.sh \
  --compressed-image-topic /camera/image_raw/compressed \
  --axis-source edges \
  --edge-preprocess channel-union \
  --canny-low 20 \
  --canny-high 60 \
  --max-display-fps 15 \
  --max-frame-age-sec 0.25 \
  --display-edges \
  --display-face-mask
```

For a real camera without usable calibration, candidate search can be limited
to a centered pixel window. This is an image-space diagnostic gate, not a
metric camera projection: every edge, silhouette, color-mask, and QR-geometry
head candidate is estimated from the crop, then accepted corners are translated
back onto the full displayed frame. The magenta rectangle shows the active
search boundary. Width and height default to `1.0`, and its vertical center
defaults to `0.5`, so existing commands still process the full frame.

For the current 800 x 600 camera view, this keeps columns 180 through 619 and
rows 221 through 520. Moving the vertical center down to `0.62` excludes the
room windows while retaining the stand head and upper stem:

On the QR-facing side, the viewer now prefers the largest detected QR plane
and expands it by the task fixture's configured `1.30` front-face/QR-width
ratio. This prevents the small distant stand from winning by detector return
order. If QR detection drops for a frame, the temporal gate holds the last
QR-anchored head briefly instead of switching immediately to a heater or
radiator rectangle. A stable non-QR silhouette can still be reacquired for the
plain-color side. Set the ratio to `1.0` only when explicitly testing the
silhouette fallback by itself.

The silhouette fallback does not treat every Canny pixel in this crop as part
of the stand. It ranks neck hypotheses, pairs outer head-side segments which
straddle the neck and terminate at the head-to-neck transition, and accepts a
head only when untouched raw Canny pixels independently support all four
sides. QR modules, radiator slats, window seams, and branches leaving a head
corner may remain visible in `stand-axis-edges`; they are excluded from the
stand-owned rectangle rather than globally erased. Keep global edge dilation
and closing at zero for the real-camera clutter test:

```bash
scripts/aufgabe04/perception/debug/run_stand_axis_viewer.sh \
  --compressed-image-topic /camera/image_raw/compressed \
  --axis-source edges \
  --edge-preprocess channel-union \
  --canny-low 20 \
  --canny-high 60 \
  --edge-dilate-iterations 0 \
  --edge-close-iterations 0 \
  --candidate-center-width-fraction 0.55 \
  --candidate-center-height-fraction 0.50 \
  --candidate-center-y-fraction 0.62 \
  --front-face-to-qr-width-ratio 1.30 \
  --stand-face-size-m 0.078 \
  --max-frame-age-sec 0 \
  --max-display-fps 10 \
  --display-edges \
  --display-face-mask \
  --display-rectangle-mask \
  --display-raw-proposal \
  --save-snapshot results/aufgabe04/real_camera_debug
```

For the expanded real-camera ROI that includes the complete stand base, enable
the observe-only structural gate. It accepts a missing lower head edge only
when immutable raw edges currently support the head top, both head sides, a
paired centred stem, and a bounded centred base. The base establishes target
ownership only; the displayed ratio/proxy still comes exclusively from the
head sides. Generic contour and line fallbacks cannot independently accept in
this mode.

`--structural-diagnostic` is deliberately incompatible with
`--observation-output-json` and `--observation-status-json`. Its console and
capture metadata state `observe_only=true` and `authoritative=false`; it is not
an approach, navigation, or motion input.

```bash
scripts/aufgabe04/perception/debug/run_stand_axis_viewer.sh \
  --compressed-image-topic /camera/image_raw/compressed \
  --axis-source edges \
  --structural-diagnostic \
  --edge-preprocess channel-union \
  --canny-low 20 \
  --canny-high 60 \
  --edge-dilate-iterations 0 \
  --edge-close-iterations 0 \
  --front-face-to-qr-width-ratio 1.0 \
  --no-qr-decode \
  --candidate-center-width-fraction 0.40 \
  --candidate-center-height-fraction 0.62 \
  --candidate-center-y-fraction 0.68 \
  --stand-face-size-m 0.078 \
  --head-hold-sec 1.5 \
  --axis-consensus-frames 7 \
  --print-every 1 \
  --max-frame-age-sec 0 \
  --max-display-fps 10 \
  --display-edges \
  --display-face-mask \
  --display-rectangle-mask \
  --save-snapshot results/aufgabe04/real_camera_debug/structural_base_roi
```

Pressing `s` in structural mode saves content-separated artifacts: original
compressed bytes, the unannotated decoded frame, candidate ROI, immutable raw
edges, localization edges, side/structure evidence, rectangle, annotated
display, and a JSON ledger with sensor and measurement status. This replaces
the older ambiguous two-PNG snapshot semantics for structural audits.

Direct module form when the environment is already sourced:

```bash
python3 -m scripts.aufgabe04.perception.debug.stand_axis_viewer \
  --compressed-image-topic /camera/image_raw/compressed \
  --axis-source edges \
  --edge-preprocess outer-border \
  --max-display-fps 15 \
  --display-edges \
  --display-face-mask
```

For the standalone Gazebo silhouette viewer, no stand coordinates or odometry
are required. `fixed` selects the standalone/debug path; it no longer creates a
fixed projected crop for edge detection. If `/scan` is supplied, the viewer
queries it at the detected head centre only after the full-frame silhouette is
available. The last validated head is held for up to 0.35 seconds across a
brief detector or temporal-outlier gap, preventing a one-frame
`detected head ROI unavailable` flash. Held estimates are labeled
`temporal_hold_after_<reason>` in status output and are not counted as fresh
axis-consensus samples. Set `--head-hold-sec 0` to disable the hold or tune the
duration explicitly:

```bash
scripts/aufgabe04/perception/debug/run_stand_axis_viewer.sh \
  --sim-raw-image-topic /camera/image_raw \
  --axis-source edges \
  --stand-face-size-m 0.06993 \
  --camera-fx-px 381.36246688 \
  --camera-fy-px 381.36246688 \
  --camera-cx-px 320.5 \
  --camera-cy-px 240.5 \
  --camera-forward-offset-m 0.076 \
  --scan-topic /scan \
  --use-lidar-distance \
  --lidar-bearing-source fixed \
  --head-hold-sec 0.35 \
  --edge-dilate-iterations 0 \
  --edge-close-kernel 3 \
  --edge-close-iterations 1 \
  --max-scan-age-sec 1.0 \
  --max-frame-age-sec 0 \
  --max-display-fps 15 \
  --diagnostic-window-size-px 320 \
  --display-edges \
  --display-face-mask \
  --display-rectangle-mask
```

### Gazebo exploratory angle-estimation range

A 2026-07-16, single-session, 112-frame Gazebo regression sweep exercised the
current color-agnostic raw edge estimator at four
robot/base-centre-to-stand-centre ground-plane distances (`0.30`, `0.45`,
`0.70`, and `0.95 m`) and seven stand-face yaw angles (`-60` through
`+60 degrees` in `20-degree` steps). It sampled four temporally adjacent frames
after each simulated pose change. A `0-degree` view was frontal to the stand
face, and the robot camera yaw was aimed at the head-centre bearing. The sweep
used the generated `0.06993 m` square head, the `640 x 480` simulated image,
and the intrinsics and thin-edge settings in the command above.

The camera distance below is camera-frame forward projection depth, not a
LaserScan range. Because the robot was aimed at the head and the simulated
camera has a `0.076 m` forward offset, it is
`robot-centre distance - 0.076 m` for these sweep poses. The head centre is
`0.072035 m` above the camera centre, so the corresponding full 3D
camera-centre-to-head-centre range for the recommended depth interval is
approximately `0.235-0.381 m`.

| Robot centre to stand | Camera optical depth to head | Frontal projected head width | Fresh usable estimates within +/-40 degrees | Mean absolute yaw error of usable estimates |
| ---: | ---: | ---: | ---: | ---: |
| `0.30 m` | `0.224 m` | `120.6 px` | `19/20` (`95%`) | `2.65 degrees` |
| `0.45 m` | `0.374 m` | `71.9 px` | `20/20` (`100%`) | `2.43 degrees` |
| `0.70 m` | `0.624 m` | `42.9 px` | `18/20` (`90%`) | `8.63 degrees` |
| `0.95 m` | `0.874 m` | `30.6 px` | `15/20` (`75%`) | `10.16 degrees` |

For conservative angle measurement in this simulation, this sweep supports
keeping the camera forward depth approximately between `0.224` and `0.374 m`,
keeping the stand-face yaw within `+/-40 degrees`, and keeping the projected
head width near `70 px` or larger. The pixel-width value is an empirical
transition seen in this sweep, not a hard detector threshold. The best tested
pose was `0.45 m` from the robot centre
(`0.374 m` camera depth): all 20 frames within `+/-40 degrees` produced fresh
usable estimates, with `2.43 degrees` mean absolute yaw error. At `0.624 m`
camera depth the detector still acquired the head in 90% of the tested frames,
but the `8.63-degree` mean error makes that a detection range rather than a
reliable angle-measurement range. The `0.874 m` result is not suitable for
precise axis estimation; even a frontal pose produced one `35.91-degree`
outlier.

These numbers characterize only the current Gazebo camera, model geometry, and
edge settings. They do not validate the physical TurtleBot camera. Also do not
compare the table directly with the viewer's displayed LiDAR distance: camera
projection depth and a scan ray have different origins and can intersect
different parts of the stand. The error column is computed only from accepted
frames and must be read together with the acceptance column. The sweep called
the raw estimator directly, so it did not test the viewer's temporal gate or
`0.35 s` hold. With only four adjacent frames at each exact pose, the rates are
descriptive results rather than independent-trial statistics, a reliability
guarantee, or a navigation-safe standoff recommendation. The temporary sweep
harness and data were not recorded with a repository manifest, so repeat and
persist the experiment before treating these numbers as release evidence.

For synchronized tracking of a known Gazebo stand candidate, this command
opens four distinct views:
the full annotated frame, full-frame topology edges, a native-pixel side-evidence
cutout, and a native-pixel fitted-rectangle cutout. The edge window is capped by
`--diagnostic-window-size-px`; the two cutouts are never resized. The viewer's
simulation defaults project the 0.165035 m stand-head centre from the 0.093 m
camera height. The generated Gazebo head is 0.06993 m square; 0.078 m is the
physical-stand value and biases simulation PnP.
Gazebo, the `burger` camera model, `/camera/image_raw`, `/scan`, and `/odom`
must already be running on ROS domain 31. Replace the stand coordinates below
with the selected LiDAR candidate. `map-target` synchronizes each camera frame
with odometry, computes the scan-frame bearing separately from the camera
bearing, and projects the moving target with the camera extrinsics. The cyan
rectangle in the full camera view is the exact ROI processed by every
diagnostic window. Missing/stale odometry or a target outside the camera FOV
fails closed instead of reverting to a fixed forward wall range:

```bash
cd /workspace/mii-amr

export ROS_DOMAIN_ID=31
export TURTLEBOT3_MODEL=burger

scripts/aufgabe04/perception/debug/run_stand_axis_viewer.sh \
  --sim-raw-image-topic /camera/image_raw \
  --axis-source edges \
  --stand-face-size-m 0.06993 \
  --camera-fx-px 381.36246688 \
  --camera-fy-px 381.36246688 \
  --camera-cx-px 320.5 \
  --camera-cy-px 240.5 \
  --camera-forward-offset-m 0.076 \
  --camera-lateral-offset-m 0.0 \
  --camera-yaw-offset-rad 0.0 \
  --odom-topic /odom \
  --map-frame odom \
  --base-frame base_footprint \
  --stand-x -0.395 \
  --stand-y -0.415 \
  --scan-topic /scan \
  --use-lidar-distance \
  --lidar-bearing-source map-target \
  --max-scan-age-sec 1.0 \
  --max-frame-age-sec 0 \
  --max-display-fps 15 \
  --diagnostic-window-size-px 320 \
  --display-mask \
  --display-edges \
  --display-face-mask
```

Use standalone `fixed` mode for stationary/manual full-frame silhouette
debugging. Use `map-target` when the simulated robot translates or rotates
toward a known candidate.

Approximate yaw degrees require extra geometry:

```bash
python3 -m scripts.aufgabe04.perception.debug.stand_axis_viewer \
  --compressed-image-topic /camera/image_raw/compressed \
  --axis-source edges \
  --stand-width-m 0.12 \
  --stand-distance-m 0.45
```

Without `--stand-width-m` and `--stand-distance-m`, use the displayed camera-relative proxy as a direction and relative-strength cue, not as a calibrated physical angle. At the lowest camera resolution, keep the stand face large enough that both vertical edges are at least about 8 to 10 pixels tall; otherwise the viewer will reject the estimate as `edge_too_short`.

Use these knobs when the outer square is not selected cleanly:

```bash
python3 -m scripts.aufgabe04.perception.debug.stand_axis_viewer \
  --compressed-image-topic /camera/image_raw/compressed \
  --axis-source edges \
  --edge-preprocess outer-border \
  --display-edges \
  --canny-low 40 \
  --canny-high 120 \
  --edge-close-kernel 7 \
  --edge-close-iterations 2 \
  --min-area-px 300 \
  --min-boundary-line-length-px 45 \
  --face-width-fraction 0.60 \
  --min-face-area-fraction 0.35
```

The preferred color-agnostic path is `source=edge_silhouette`: the edge image is converted into a filled stand silhouette, the broad upper face is separated from the narrow stem, and the face quadrilateral is fitted from that filled silhouette. The detector intentionally does not refine this silhouette result with raw edge lines, because those lines can snap back to QR-code texture or paper edges.

If the edge window is full of QR-code internals, keep `--edge-preprocess outer-border` and raise `--min-boundary-line-length-px` until short QR module edges stop being considered as square boundaries. If the overlay still locks onto a QR element, raise `--min-face-area-fraction` so candidates must be a larger fraction of the largest visible stand silhouette. Use `--edge-preprocess gray` only when you want to inspect the raw grayscale edge detector.

If the square face and vertical stem are the same color and physically connected, the edge contour can become one T-shaped outline. The detector handles this by first searching square-like edge contours, then extracting the broad upper band from the filled silhouette to ignore the narrow stem, then falling back to Hough line segments for the square's left and right outer edges. Tune the silhouette fallback with `--face-width-fraction`: raise it if the stem is included, lower it if the side edges are incomplete.

Tune the line fallback with:

```bash
python3 -m scripts.aufgabe04.perception.debug.stand_axis_viewer \
  --compressed-image-topic /camera/image_raw/compressed \
  --axis-source edges \
  --edge-preprocess outer-border \
  --display-edges \
  --hough-threshold 18 \
  --hough-min-line-length-px 10 \
  --hough-max-line-gap-px 6 \
  --min-boundary-line-length-px 45
```

When the silhouette fallback is used, the overlay/console reports `source=edge_silhouette`; when the line fallback is used, it reports `source=edge_lines`; when a contour quadrilateral is used, it reports `source=edges`. For the QR-facing side with many internal QR edges, `source=edge_silhouette` is usually the desired result.

When the face is nearly 90 degrees to the robot camera and only a thin line is visible, the overlay/console reports `source=edge_on_line`. This is a different interpretation from `source=edge_lines`: `edge_lines` still reconstructs a visible quadrilateral from line pairs, while `edge_on_line` reports side-on geometry without a ratio.

The legacy HSV contour mode is still available for comparison:

```bash
python3 -m scripts.aufgabe04.perception.debug.stand_axis_viewer \
  --compressed-image-topic /camera/image_raw/compressed \
  --axis-source color-mask \
  --color green \
  --display-mask \
  --tune
```

Use color-mask mode only for debugging color segmentation. For stand-axis rotation, prefer `--axis-source edges` with `source=edge_silhouette` because it stays color agnostic while using the filled outer stand shape rather than QR texture.
