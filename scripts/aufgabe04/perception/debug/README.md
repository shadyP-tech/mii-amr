# Aufgabe 04 Perception Debug Viewer

This tool is debug-only. It visualizes camera frames, HSV masks, masked previews, and ROI color classification results for stand-color threshold tuning.

It does not command robot motion, does not publish `/cmd_vel`, does not send Nav2 goals, and does not execute station approach behavior.

Viewer output is not real-parkour validation evidence. Real-robot Aufgabe 04 claims require the separate real-parkour checklist, run logs, and recorded mission evidence.

Keep all OpenCV window, camera, and live-debug code in this debug package. Pure perception logic belongs in `scripts/aufgabe04/perception/color_classifier.py` and should be covered by offline tests only.

## Usage

Run this only when the robot is stationary or physically secured. Confirm no autonomous mission, Nav2 goal, custom follower, or station-route runner is active.

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

Run the viewer from an environment that can see the image topic:

```bash
python3 -m scripts.aufgabe04.perception.debug.color_mask_viewer \
  --ros-image-topic /camera/image_raw \
  --qos best-effort \
  --color green \
  --roi 120,100,220,180 \
  --max-display-fps 10 \
  --max-frame-age-sec 0.25 \
  --display-mode frame \
  --no-morph \
  --tune
```

The viewer only supports ROS 2 `sensor_msgs/Image` input. Local OpenCV camera indexes and video files are intentionally not supported in this Aufgabe 04 debug tool.

The viewer defaults to best-effort `KEEP_LAST` QoS at depth 1 for low-latency live video. If topic matching fails with another camera publisher, retry with `--qos reliable`. If the viewer is running, `ros2 topic info /camera/image_raw -v` should show one subscription.

Useful keys:

- `p`: print the active `ColorRange(...)` threshold.
- `s`: save frame, mask, and preview snapshots when `--save-snapshot` is set.
- `q` or `Esc`: quit.

Tune thresholds in the actual lighting where the stands will be seen. Prefer selecting a stand ROI over classifying the full frame, because full-frame classification dilutes confidence with background pixels.

The low-latency default is a single annotated frame window. Use `--display-mode frame-mask` to also show the mask, or `--display-mode all` to show frame, mask, and masked preview. Extra windows can add visible lag over Apptainer/X11.

The mask is built with vectorized OpenCV operations and ROI confidence is computed from `cv2.countNonZero(mask_roi)`. Lower `--max-display-fps` if the laptop is overloaded; the ROS subscriber still keeps the latest received frame in a background thread. Duplicate-frame checks are disabled by default; enable `--detect-duplicates` only when diagnosing frozen input.

The frame overlay includes `age=...ms` when the incoming image has a ROS header stamp. Incoming stamped frames older than `--max-frame-age-sec` are dropped in the callback before conversion; use `--max-frame-age-sec 0` to disable that guard. A large displayed age means frames are still stale before display, while a small age with laggy windows points to local rendering overhead.
