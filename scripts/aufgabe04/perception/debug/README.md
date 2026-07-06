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

Run it from the Apptainer workstation environment with:

```bash
scripts/aufgabe04/perception/debug/run_stand_axis_viewer.sh \
  --compressed-image-topic /camera/image_raw/compressed \
  --axis-source edges \
  --edge-preprocess outer-border \
  --max-display-fps 15 \
  --max-frame-age-sec 0.25 \
  --display-edges
```

Direct module form when the environment is already sourced:

```bash
python3 -m scripts.aufgabe04.perception.debug.stand_axis_viewer \
  --compressed-image-topic /camera/image_raw/compressed \
  --axis-source edges \
  --edge-preprocess outer-border \
  --max-display-fps 15 \
  --display-edges
```

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
