# Aufgabe 04 navigation modules

The navigation package is split by responsibility:

- `foundation/`: shared models, evidence helpers, runtime configuration, and logs
- `planning/`: map/costmap handling, A*, route geometry, and waypoint data
- `control/`: pure follower control, safety decisions, and command ownership
- `localization/`: pose/frame admission, ROS preflight, and localization reseals
- `execution/`: route certificates, revisions, motion permits, and execution gates
- `coverage/`: survey coverage, candidate lifecycle, visibility, and replanning
- `approach/`: detected-stand approach, camera admission, and viewpoints
- `missions/`: task-level planners and simulation/observation workflows
- `entrypoints/`: stable Python and shell command-line wrappers
- `station_segment/`: the validated single-segment runner
- `waypoint_follower/`: the stateful ROS follower and its runtime components

The navigation root contains package metadata only. Command wrappers delegate
to the owning package, and library consumers import from that package directly.
The only stateful ROS motion edge remains `waypoint_follower/runtime.py`.
