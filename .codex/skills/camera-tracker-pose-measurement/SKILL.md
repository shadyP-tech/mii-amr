---
name: camera-tracker-pose-measurement
description: >
  Focus all mii-amr camera-based TurtleBot pose tracker questions. Use for
  green-marker HSV tuning, OpenCV contour filtering, homography calibration
  and verification, marker height vs floor-plane geometry, pose estimation,
  results/aufgabe02/latest_tracker_pose.csv, real-run start-pose gate behavior, and
  tracker debugging.
---

# Camera Tracker / Pose Measurement

## Scope
Use this for the `mii-amr` camera tracker in `vision_tracker/`. The tracker
detects green circular markers on top of the TurtleBot, estimates `x`, `y`, and
`yaw`, writes `results/aufgabe02/latest_tracker_pose.csv`, and supports real-run
start-pose validation plus final pose measurement.

Do not use this skill for:
- general ROS2/Gazebo launch errors unless they involve the tracker
- Git conflicts unless they involve tracker files
- Sim2Real statistical analysis after pose data has already been collected
- report writing unless the section is about camera tracking or pose measurement

## Code Map
- `vision_tracker/config.py`: HSV thresholds, contour filters, camera settings,
  `WORLD_RECT_METERS`, `CENTER_FORWARD_OFFSET`, start-gate tolerances, paths.
- `vision_tracker/tune_hsv.py`: live HSV trackbars and mask validation.
- `vision_tracker/tracker.py`: HSV mask, morphology, contour filters, marker
  selection.
- `vision_tracker/calibration.py`: homography from resized image pixels to
  world meters.
- `vision_tracker/pose_estimator.py`: marker classification, center offset,
  yaw calculation, CSV writer.
- `vision_tracker/main.py`: live tracking, overlay, per-run CSV, latest-pose
  CSV.
- `vision_tracker/start_pose.py`, `vision_tracker/start_pose_gate.py`:
  start-pose validation from latest pose CSV.
- `scripts/aufgabe02/run_real_experiment.sh`: real-run orchestration; verify expected repo
  path on the current host before debugging gate failures.

## Response Contract
For normal tracker questions, answer with:
1. Diagnosis
2. Most likely cause
3. Exact change or command
4. Validation check
5. Next action

For complex tracker failures, answer with:
1. Image-processing check
2. Calibration/geometry check
3. CSV/start-gate check
4. Minimal fix
5. Verification

Response rules:
- Separate image-processing issues from geometry/calibration issues.
- Prefer concrete parameter changes in `vision_tracker/config.py`.
- For homography and pose questions, state the coordinate frame.
- For marker-height questions, distinguish floor-plane coordinates from
  marker-plane observations.
- Explain validation with screenshots, live windows, or CSV values.
- Label commands when camera host vs robot host matters.
- Never imply automatic physical TurtleBot motion; keep real-run safety explicit.

## Device / Environment Rules
Camera host:
- runs `vision_tracker/main.py`
- owns the physical camera
- writes `results/aufgabe02/latest_tracker_pose.csv`

Robot / ROS host:
- runs TurtleBot commands, bagging, and `scripts/aufgabe02/run_real_experiment.sh`
- may be inside Apptainer at `/workspace/mii-amr`

When location matters, label commands as `Camera host / MacBook`,
`Robot host / workstation`, `Apptainer shell`, or `TurtleBot shell`.
If `results/aufgabe02/latest_tracker_pose.csv` is stale or missing on the robot host,
suspect host/path mismatch before changing tracker geometry.

## Coordinate Frames
- Image frame: `(u, v)` pixels in the resized OpenCV frame. Calibration and
  tracking must use the same `RESIZE_SCALE`.
- World frame: meter coordinates defined by `WORLD_RECT_METERS` and click order
  top-left, top-right, bottom-right, bottom-left as viewed by the camera.
- Pose frame: same world frame. `yaw_rad`/`yaw_deg` are measured from world
  `+x` toward world `+y`.
- Current homography is planar: image pixels map onto the calibration plane
  represented by `WORLD_RECT_METERS`.
- If the calibration rectangle is on the floor/table but marker centers are on
  top of the robot, marker positions are marker-plane observations interpreted
  through a floor-plane homography.

If marker height matters, state whether the user needs floor-plane robot pose,
marker-plane coordinates, a homography calibrated at marker height, or an
explicit height correction.

## Debugging Order
1. Camera stream/color: wrong camera, grayscale/IR stream, exposure, backend.
2. Image processing: HSV, morphology, contour filters, false/split/missing blobs.
3. Calibration geometry: click order, `WORLD_RECT_METERS`, scale, axes,
   homography, lens distortion.
4. Pose geometry: marker layout, two large front markers, smaller rear marker,
   `CENTER_FORWARD_OFFSET`, yaw direction.
5. CSV/start gate: freshness, `valid_pose`, `num_detected`, finite values,
   tolerances, stable-time requirement, host/path mismatch.

## Image Processing
Commands:
```bash
python3 vision_tracker/list_cameras.py
python3 vision_tracker/tune_hsv.py
python3 vision_tracker/main.py
```

Validate:
- `tune_hsv.py`: mask shows exactly three solid white marker blobs and black
  background.
- `main.py`: tracking window labels markers and reports finite pose when all
  three markers are visible.
- `tracker.py`: enable `DEBUG_CONTOURS` only when contour rejection details are
  needed.

Concrete tuning moves:
- missed/dark markers: lower `HSV_LOWER[2]` after checking the mask
- glare or pale/white background included: raise `HSV_LOWER[1]` or
  `HSV_LOWER[2]`
- hue shifts: widen `HSV_LOWER[0]`/`HSV_UPPER[0]` in small steps; OpenCV hue is
  `0..179`
- small true markers rejected: lower `MIN_RADIUS` or `MIN_CONTOUR_AREA`
- noise accepted: raise `MIN_RADIUS`, `MIN_CONTOUR_AREA`, `MIN_CIRCULARITY`, or
  `MIN_FILL_RATIO`
- blobs split by glare: try `MORPH_KERNEL_SIZE` from `7` to `9`, or lower
  `MIN_FILL_RATIO` slightly after mask validation

Tuning rule: change one parameter class at a time: exposure/camera stream, HSV,
morphology, contour filters, then geometry/calibration. Do not change homography
or pose geometry to compensate for a bad mask.

## Geometry And Calibration
Commands:
```bash
python3 vision_tracker/calibration.py
python3 vision_tracker/calibration.py --verify
```

Homography checks:
- confirm `vision_tracker/data/homography.npz` exists
- confirm calibration and tracking use the same `RESIZE_SCALE`
- confirm clicked points match `WORLD_RECT_METERS` order
- verify known points and report errors in meters
- wrong scale usually means wrong `WORLD_RECT_METERS` or click order
- swapped axes/signs usually mean click-order or world-frame convention issue
- curved residual patterns suggest lens distortion; current code uses homography
  only

Pose checks:
- `classify_markers` expects two larger front markers and one smaller rear marker
- `estimate_pose` computes heading from rear marker to the straight front marker,
  then applies `CENTER_FORWARD_OFFSET` along heading
- if true chassis center is needed, consider whether a lateral offset
  perpendicular to heading is also required

## CSV And Start-Pose Gate
Check:
```bash
tail -n 2 results/aufgabe02/latest_tracker_pose.csv
tail -n 2 results/aufgabe02/real_start_pose_checks.csv
```

`results/aufgabe02/latest_tracker_pose.csv` schema:
```text
timestamp,x,y,yaw_rad,yaw_deg,valid_pose,num_detected
```

Gate acceptance requires a fresh finite pose, `valid_pose=1`,
`num_detected >= START_POSE_REQUIRED_MARKERS`, position error within
`START_POSE_POSITION_TOLERANCE_M`, yaw error within
`START_POSE_YAW_TOLERANCE_DEG`, and stability for
`START_POSE_STABLE_TIME_SEC`.

If the desired start pose moved, update `START_POSE_REF_X`, `START_POSE_REF_Y`,
and `START_POSE_REF_YAW_DEG`. Only loosen tolerances after measuring tracker
noise from stable CSV samples. `start_pose_gate.py` does not open the camera;
`vision_tracker/main.py` must already be running and writing a fresh latest-pose
CSV at the path read by the run host.

## Documentation Guidance
When documenting this tracker, explain why camera tracking was used, how
calibration maps image pixels to world meters, how marker detection produces
pose, how final pose measurements are validated, and limitations: lighting,
marker height, homography plane, lens distortion, marker visibility.
