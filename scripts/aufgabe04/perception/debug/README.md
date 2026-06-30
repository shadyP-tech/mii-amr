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

The stand-axis viewer uses the selected HSV mask to find the largest four-corner stand face in the live camera frame. It overlays the detected quadrilateral and reports:

- `L` / `R`: apparent pixel heights of the left and right stand edges.
- `ratio`: `L / R`.
- `closer`: the side with the larger apparent edge height.
- `proxy`: `(ratio - 1) / (ratio + 1)`, a signed rotation cue.
- `med_ratio` / `med_proxy`: median-filtered values over the recent usable frames.

Run it from the Apptainer workstation environment with:

```bash
scripts/aufgabe04/perception/debug/run_stand_axis_viewer.sh \
  --compressed-image-topic /camera/image_raw/compressed \
  --color green \
  --max-display-fps 15 \
  --max-frame-age-sec 0.25 \
  --display-mask \
  --tune
```

Direct module form when the environment is already sourced:

```bash
python3 -m scripts.aufgabe04.perception.debug.stand_axis_viewer \
  --compressed-image-topic /camera/image_raw/compressed \
  --color green \
  --max-display-fps 15 \
  --display-mask \
  --tune
```

Approximate yaw degrees require extra geometry:

```bash
python3 -m scripts.aufgabe04.perception.debug.stand_axis_viewer \
  --compressed-image-topic /camera/image_raw/compressed \
  --color green \
  --stand-width-m 0.12 \
  --stand-distance-m 0.45
```

Without `--stand-width-m` and `--stand-distance-m`, use the displayed ratio/proxy as a direction and relative-strength cue, not as a calibrated physical angle. At the lowest camera resolution, keep the stand face large enough that both vertical edges are at least about 8 to 10 pixels tall; otherwise the viewer will reject the estimate as `edge_too_short`.
