# Station-tour audit: 20261001T143227Z

Run `station_tour_20261001T143227_697460Z`, October 1, 2026,
**16:32:27–16:33:35 Europe/Berlin**, workstation SSH alias `mii002`,
revision `144f95425e36256b2db82b84cc555e6864c726f1`.

## Outcome

The script successfully contacted the FastAPI server, generated a random plan,
reported verified arrival at Start, and received **QR_001 / DEPOT_01** as its
next target. Navigation failed before any nonzero velocity was published.
The terminal failure was:

> stored navigation motion failed: stopped: TF transform unavailable: map <- odom

The initial Start check passed without a motion leg: position error was
**0.075302 m** (limit 0.08 m), heading error **0.093761 rad** (limit 0.15 rad).
Thus this run did not encounter the previously identified blocked-Start-goal
planning problem.

## Server evidence

`server/journal.jsonl` records successful requests and complete responses:

| Time, Berlin | Operation | Result |
|---|---|---|
| 16:32:57.397 | POST `/api/v1/robots/turtlebot1/plan/randomize` | Random plan with four numbered QR stands and three processing visits |
| 16:32:57.415 | GET `/api/v1/robots/turtlebot1/qr-mappings` | Numbered identities plus Start |
| 16:32:57.420 | GET `/api/v1/robots/turtlebot1/plan` | Plan confirmed |
| 16:32:57.427 | POST `/api/v1/qr/Start/scan` | Accepted, mission `M-00001`, state `GO_TO_DEPOT_PICKUP`, next QR `QR_001` |
| 16:32:57.448 | Navigation requested | First server-directed destination `QR_001` |

The frozen order was `Start → QR_001 → QR_004 → QR_003 → QR_001 → QR_004 → Start`.
QR_002 was scheduled for supplemental coverage after the server mission.
No QR_001 arrival was reported. The server response at Start assigned no wait
actions; the recorded zero-duration wait completed before navigation.

The old console displayed navigation but did not display server requests or
responses. Lack of visible terminal messages therefore did not imply lack of
HTTP interaction. The old startup also performed stopped localization and
Start verification before the first HTTP request.

## Navigation failure

The first QR_001 route and its temporary obstacle overlay were admitted. The
child's execution preflight passed all 15 observations and sealed an odometry
route. Its first stage was 0.265 m, with the full stored destination retained
for later localization stages.

After preflight, the motion controller created its own TF listener. During
the controller's unchanged five-second initial acquisition budget:

- `odom ← base_footprint` was fresh in all 28 lookup attempts; 94 transform
  ingestions were recorded.
- `map ← odom` was absent in all 28 attempts; **zero** receipts or ingestions
  were recorded for this edge.
- The isolated TF executor was healthy, with 100 heartbeat callbacks.
- The terminal exception was `LookupException`: the `map` frame did not exist
  in this new listener's buffer.
- The controller trace contains only zero-command startup/wait events.
  The child explicitly records `motion_published=false`, estimated distance
  **0.0 m**, and stop phase `before_motion`.

AMCL was listed both before and after execution. Preflight had obtained a
fresh `map ← odom` transform immediately before creating the motion controller.
These observations isolate the failure to delivery/acquisition in the new
listener, rather than route planning, a missing saved pose, or an unavailable
server. They do not prove whether upstream AMCL or DDS caused every missing
publication.

The implementation gap is definite: initial runtime acquisition only waited
for TF. It did not request an AMCL no-motion update. The existing AMCL refresh
path was entered through stale execution-pose recovery; with a fresh
`odom ← base_footprint` edge and a missing global edge, that path was never
entered. A preflight refresh cannot populate a TF buffer created afterward.

## Corrections

The tour startup now requests and validates the server's random plan after
local artifact validation and operator authorization, before initial Start
verification or recovery. Only a measured Start arrival can be reported to
obtain the server's first onward target. Subsequent departures still require
the server's validated next target and completion of its assigned wait.
Console progress exposes server requests, targets, waits and failures.

The follower correction requests a bounded AMCL no-motion refresh from the
new listener while holding zero. A service acknowledgement cannot authorize
motion: fresh sensor/TF delivery, ownership and frozen-frame consistency must
still pass within the original startup deadline. Existing motion-time
localization and obstacle stops remain in force.

## Evidence and validation

The read-only audit copied the selected original artifacts into
`results/implementation_checks/station_tour_failure_20261001T143227Z/workstation_evidence.json`.
It includes the server journal, frozen plan, failure summaries, Start arrival,
planning and execute localization, motion events, controller trace, node lists,
TF graph and terminal log. The audit itself issued no robot motion and made
no server mutations. Automated correction tests use simulated effects;
successful physical navigation remains unverified.

An isolated checkout of the recorded workstation revision plus only these
corrections passed **165 targeted tests in 39.078 seconds** inside the
workstation's ROS Humble container (Python 3.10). The nine test modules cover
the new AMCL refresh, initial TF acquisition, existing stale-TF recovery,
follower module boundaries, tour orchestration/runtime/client, and temporary
obstacle navigation including its planner-to-controller integration.
The test checkout was mounted read-only; live ROS motion and server writes
were not invoked. Independent code review found no remaining blocker in
these changes. This is automated validation, not a successful physical tour.
