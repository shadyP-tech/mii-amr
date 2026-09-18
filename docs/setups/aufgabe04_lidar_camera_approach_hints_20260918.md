# LiDAR suggestions for the first camera approach

Camera candidate selection now uses a multi-view LiDAR surface orientation,
when available, to preview two opposite perpendicular inspection positions.
Both positions face the candidate. The existing initial-turn risk and estimated
travel-duration policy chooses between admitted routes; candidate ordering
continues to use the established camera selection policy.

This is automatic in the autonomous camera phase when a source survey registry,
a fresh admitted planning frame, and usable completed survey scan receipts are
available. It does not add survey motion or require a new capture command.
Missing evidence, weak geometry, or two unavailable inspection routes retain
the existing robot-to-candidate approach. Camera identity, front/back admission,
certified opposite-face handling and collision checks remain authoritative.

## Evidence and geometry

- Reopen completed survey epochs and observer summaries; check the existing
  receipt file, receipt-set, configuration, survey, viewpoint and frame bindings.
- Associate returns inside the candidate's radius plus positional uncertainty
  (at most 0.15 m). Reject multiple clusters, seam fragments without topology
  proof, and returns compatible with another candidate envelope.
- Fit each scan independently in canonical odom, using at least four distinct
  beams, a 0.04–0.18 m span, minor/major variance ratio at most 0.05, and maximum
  perpendicular residual at most 0.006 m. Repeated sparse scans cannot create
  sufficient spatial support.
- Require at least three distinct scan timestamps and 75% usable scans per
  viewpoint, stationary pose spread within 0.02 m and 3 degrees, and axial
  consistency within 8 degrees within each view and between view means.
- Require at least two viewpoints separated by at least 20 degrees modulo
  180 degrees. Observing from the same or directly opposite bearing does not
  supply this independent shape check. Weight viewpoints equally.
- Rotate the retained odom tangent into the current planning frame. Bind the
  suggestion to the source registry and exact projected candidate snapshot.

The tangent's two perpendicular directions define the proposed inspection
positions at the configured camera standoff. Preview both through the existing
map, keepout and route-certification pipeline. Apply stopped-localization route
uncertainty admission to each side before choosing it. Reject endpoint angular
deviation over 10 degrees or requested-position error over the existing 0.06 m
inspection-route limit. If neither side survives, preview the ordinary approach
and apply its usual admission checks.

The selected route is reused for materialization. Its `lidar_axis_hint`
inspection-view artifact binds candidate, snapshot, start pose and suggested
direction, with `stand_axis_authorized=false` and `motion_authorized=false`.
The existing inspection-view recovery path reprojects that direction after a
localization change. No LiDAR hint can stand in for a camera backside receipt.

Selection logs contain `lidar_inspection_hints`, per-candidate acceptance or
fallback reasons, source receipt hashes, observed view-angle spread, and both
route outcomes. The selected candidate directory also retains
`lidar_initial_selection.json` and `lidar_initial_inspection_view.json`.

## Limits and validation

These thresholds are proposal heuristics, not a calibrated angle-accuracy
claim. The surface at laser height may differ from the head. A square support
can still supply the wrong head axis even if a surface fit is consistent.
Single-view or sparse candidates intentionally fall back; direct front/back
arrival is not guaranteed. The camera independently resolves the actual side.

Focused tests cover multi-view fitting, rotated localization frames, sparse and
duplicate scans, circular supports, conflicting surfaces, ambiguous association,
moving windows, missing/tampered artifacts, two-sided route selection, uncertainty
rejection, fallback, sealing and the autonomous first-view handoff. Run with:

```sh
python3 -m unittest tests.aufgabe04.test_lidar_inspection_hint tests.aufgabe04.test_camera_lidar_hint_loading tests.aufgabe04.test_lidar_inspection_planning
```

No deployment or physical robot validation was performed. Field validation
should compare suggested axes and final camera sides against measured head
orientations, including sparse, oblique and cluttered candidates.
