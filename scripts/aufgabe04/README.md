# Aufgabe 04: autonomous exploration

The runtime surveys the saved occupancy map, discovers stands from stopped
LiDAR scans, plans collision-checked camera approaches, and reads QR codes. A
bounded first camera inspection that cannot identify a stand automatically
plans an opposite-side visit using the measured head orientation. Valid
unidentified observations remain explicit results.

## Run

From the repository root, validate the measured configuration without starting
ROS or sending motion commands:

```sh
python3 -m scripts.aufgabe04.core
```

On the TurtleBot ROS 2 environment, add `--execute` and enter `RUN` at the
interactive prompt. The robot drivers, AMCL on the saved map, camera, LiDAR,
odometry and calibrated TF tree must already be running. Stop other velocity
publishers before starting. Ctrl-C stops the session.

The retained measured calibration is pinned by the robot profile.
`--camera-calibration`, `--robot-profile`, `--stand-model` and `--site` select
alternative measured inputs. The site binds the map YAML and image. Defaults
use the retained TurtleBot1 profile, its calibration and the measured lab setup.

Dependencies: Python 3.10+, NumPy, OpenCV with QR decoding, and ROS 2 Humble
(`rclpy`, `tf2_ros`, `geometry_msgs`, `nav_msgs`, `sensor_msgs`, `std_srvs`).
Use the robot's compatible ROS Python environment.

## Core

| Module | Responsibility |
| --- | --- |
| `mission.py` | Mapped survey, candidate registry, front and opposite inspections |
| `planning.py` | Frozen map, coverage visibility, A*, clearance and approach geometry |
| `motion.py` | Route tracking, uncertainty budget, corner handling and swept LiDAR stop |
| `lidar.py` | Original-ray clustering, temporal support and target matching |
| `camera.py`, `vision.py` | Calibrated images, physical head geometry and QR observations |
| `ros_runtime.py` | Live sensor/TF admission, stationary epochs and control timer |
| `command_guard.py` | Independent leased publisher of the physical velocity topic |
| `model.py`, `config.py` | Immutable shared values and measured configuration |
| `__main__.py` | Input validation and explicit execution entrypoint |

The mission and sensor algorithms share typed state in one long-lived process.
There are no per-leg command subprocesses, request files or receipt chains.
The small watchdog runs separately so a stalled mission cannot keep renewing
its command lease. Only the watchdog publishes physical velocity commands.

## Operating limits

Motion requires current scan, odometry, localization and transforms, a stable
stopped starting epoch, an admitted route and exclusive velocity ownership.
The controller checks uncertainty, map alignment, route deviation, speed,
progress and the LiDAR braking sweep. The watchdog publishes zero when its
200 ms lease expires and latches that fault. Runtime faults require a new
session; there is no automatic resume from saved files.

Coverage means planned visibility of at least 95% of the mapped reachable
free cells, subject to admitted routes and live obstacles. Conservative
clearance or uncertainty checks can refuse a route. Opposite-side motion
requires bounded current head geometry and current LiDAR target support.
The mission has no QR quota, return-to-Start phase, debug viewer or replay CLI.

The software watchdog does not cover failure of its own process, the whole
computer, DDS or the motor controller. Keep the physical emergency stop
available and verify the base's command-loss behavior before a supervised
hardware trial. The [upstream TurtleBot3 Humble node](https://github.com/ROBOTIS-GIT/turtlebot3/blob/humble/turtlebot3_node/src/turtlebot3.cpp)
sends its OpenCR heartbeat independently of velocity callbacks; that heartbeat
alone is not evidence that a lost command source will stop the robot.

## Cleanup

The previous implementation, test harnesses, recorded fixtures, audit results
and clean upload archive are stored locally under
`sanitized/aufgabe04-redesign-2026-10-13/`, which is excluded by `.gitignore`.
They are not runtime dependencies. The redesign is checked offline; it has
not been validated by moving a physical robot.
