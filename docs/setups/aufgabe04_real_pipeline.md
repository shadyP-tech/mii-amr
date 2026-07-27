# Aufgabe 04 Real-Robot Experiment Pipeline

The real experiment path reuses the simulation pipeline's immutable station,
catalog, route, and mission contracts, but replaces simulated sensing and
runtime assumptions with dedicated hardware adapters. It is not approval for a
loaded logistics mission or a two-robot run; see
[`aufgabe04_sim_to_real_gate.md`](aufgabe04_sim_to_real_gate.md).

## Module Boundary

| Module | Responsibility | Motion capability |
| --- | --- | --- |
| `real_robot/capture_camera_calibration.py` | Capture live `CameraInfo` and the measured `base <- camera_optical` TF | None |
| `real_robot/create_hardware_profile.py` | Seal topics, namespace, frames, site, calibration, footprint, and speed limits | None |
| `real_robot/passive_viewpoint_node.py` | Synchronize image, scan, and exact-time TF; rectify the image; validate LiDAR/QR/silhouette evidence | None |
| `real_robot/prepare_passive_survey.py` | Produce immutable per-candidate observer and catalog-validation commands | None |
| `real_robot/finalize_passive_survey.py` | Freeze a complete real arrival catalog and write a real `SurveyManifest` | None |
| `real_robot/run_unloaded_segment.py` | Bind one certified route leg to the sealed hardware/site profile and existing preflight runner | Dry-run by default; explicit physical gate |

The shared `plan_synchronized_viewpoint.py` accepts
`--environment real --workflow-mode survey-only`. In real mode it cannot watch
or generate a dynamic motion route. It only validates an already committed
passive recommendation and updates the shared arrival catalog.

## 1. Inspect the Live ROS Interface

The SSH alias `turtle` must resolve from the workstation before remote
inspection. Use read-only commands first:

```bash
ssh turtle 'hostname; printenv ROS_DOMAIN_ID; ros2 topic list; ros2 node list'
ssh turtle 'ros2 topic info /scan -v; ros2 topic info /camera/camera_info -v'
ssh turtle 'ros2 run tf2_ros tf2_echo base_footprint camera_optical_frame'
```

Substitute the robot's actual namespace, camera topics, and optical frame in
all later commands. Do not infer them from Gazebo names.

## 2. Seal Calibration and Hardware Inputs

Create a physical-site descriptor whose filename stem is its stable site ID.
It should identify the measured parkour revision, station placement procedure,
map acquisition, and date. The pipeline binds its exact bytes:

```json
{
  "schema_version": 1,
  "physical_site_id": "aufgabe04_lab_v1",
  "description": "Measured Aufgabe 04 parkour, revision 1",
  "map_measurement": "replace with the measurement record",
  "station_setup": "replace with the station placement record"
}
```

Capture the calibration that is currently published by the real camera. This
is a capture step, not proof that the calibration itself is accurate:

```bash
python3 scripts/aufgabe04/real_robot/capture_camera_calibration.py \
  --namespace robot1 \
  --camera-info-topic camera/camera_info \
  --base-frame base_footprint \
  --camera-optical-frame camera_optical_frame \
  --calibration-id robot1_camera_20260727 \
  --source "measured calibration report <identifier>" \
  --output results/aufgabe04/real/profiles/robot1_camera_20260727.json
```

Measure the physical robot radius, LiDAR-to-base radial offset, and conservative
unloaded speed limits. Then seal them with the resolved runtime interface:

```bash
python3 scripts/aufgabe04/real_robot/create_hardware_profile.py \
  --profile-id robot1_unloaded_20260727 \
  --robot-id robot1 \
  --namespace robot1 \
  --scan-topic scan \
  --odom-topic odom \
  --cmd-vel-topic cmd_vel \
  --amcl-topic amcl_pose \
  --compressed-image-topic camera/image_raw/compressed \
  --camera-info-topic camera/camera_info \
  --map-frame map \
  --odom-frame odom \
  --base-frame base_footprint \
  --scan-frame base_scan \
  --localization-source amcl \
  --physical-site docs/setups/aufgabe04_lab_v1.json \
  --physical-site-id aufgabe04_lab_v1 \
  --camera-calibration results/aufgabe04/real/profiles/robot1_camera_20260727.json \
  --robot-radius-m <measured_radius> \
  --scan-origin-to-base-offset-m <measured_offset> \
  --max-linear-speed-mps <validated_unloaded_limit> \
  --max-angular-speed-radps <validated_unloaded_limit> \
  --output results/aufgabe04/real/profiles/robot1_unloaded_20260727.json
```

The profile requires `map` and `odom` to be distinct, forces
`use_sim_time=false`, resolves six distinct topics, and binds the calibration
and physical-site hashes.

## 3. Prepare and Collect a Passive Survey

Use the frozen map, detector-produced candidate snapshot, and complete station
identity registry from the successful sealed workflow:

```bash
python3 scripts/aufgabe04/real_robot/prepare_passive_survey.py \
  --robot-profile results/aufgabe04/real/profiles/robot1_unloaded_20260727.json \
  --camera-calibration results/aufgabe04/real/profiles/robot1_camera_20260727.json \
  --physical-site docs/setups/aufgabe04_lab_v1.json \
  --map maps/aufgabe03/<real_map>.yaml \
  --semantic-map-id <real_map_id> \
  --candidate-snapshot results/aufgabe04/<candidate_snapshot>.json \
  --station-identity-registry results/aufgabe04/<identity_registry>.json \
  --output-dir results/aufgabe04/real/surveys/real_survey_001 \
  --catalog results/aufgabe04/real/surveys/real_survey_001/catalog.json \
  --catalog-id real_survey_001 \
  --session-id real_survey_001 \
  --survey-manifest results/aufgabe04/real/surveys/real_survey_001/manifest.json
```

The command prints the immutable plan path. For each `candidate_runs` entry:

1. Manually place the stopped robot at a clear, localized viewpoint.
2. Run its `observer_command`.
3. Inspect `observer_status.json` and the debug images.
4. Run its `catalog_validation_command` only after the expected QR ID and
   consensus evidence were committed.

The observer creates no ROS publisher. It rejects simulated time, stale or
unsynchronized sensors, changed `CameraInfo`, changed camera extrinsics,
non-stationary poses, missing exact-time TF, projected-target/LiDAR
disagreement, weak silhouette consensus, and incorrect QR identity. A raw
compressed image is rectified into the sealed `CameraInfo.p` geometry before
the projected ROI is evaluated.

After all expected candidates are resolved, run the plan's
`finalize_command`. Finalization refuses incomplete catalogs or provenance that
differs from the sealed map, snapshot, identity registry, site, profile,
calibration, or survey configuration.

## 4. Unloaded Single-Leg Validation

`run_unloaded_segment.py` accepts the full certified mission artifact chain and
derives namespace, topics, frames, footprint, and speed limits only from the
sealed profile. Its default is a dry run:

```bash
python3 scripts/aufgabe04/real_robot/run_unloaded_segment.py \
  --robot-profile <robot_profile.json> \
  --physical-site <physical_site.json> \
  --route-csv <route.csv> \
  --diagnostics-json <route_diagnostics.json> \
  --route-certificate-json <route_certificate.json> \
  --route-bundle-json <route_bundle.json> \
  --planner-config-json <planner_config.json> \
  --mission-plan-manifest <mission_manifest.json> \
  --survey-manifest <real_survey_manifest.json> \
  --runtime-map-bundle-json <runtime_map_bundle.json> \
  --candidate-snapshot <candidate_snapshot.json> \
  --station-identity-registry <identity_registry.json> \
  --arrival-pose-catalog <real_catalog.json> \
  --task-snapshot <fresh_task_snapshot.json> \
  --leg-index 0 \
  --run-id real_unloaded_leg_001 \
  --confirm-unloaded
```

Before any physical execution, clear the arena, keep an operator beside the
robot with Ctrl+C and physical stop access, and keep a separate terminal ready
to publish one zero `Twist` to the exact resolved command topic. Verify one
localization owner and exclusive velocity ownership. Only then repeat the same
command with `--execute`; it is automatically evidence-bundled and the existing
inner runner still performs ROS preflight and requires typed `RUN`.

This adapter intentionally excludes cargo, loaded footprint, multi-leg
dispatch, strict post-arrival QR events, and fleet coordination.

## Validation

The migration modules remain ROS-free at import and test time:

```bash
python3 -m unittest tests.aufgabe04.test_real_robot_pipeline
python3 -m unittest discover -s tests/aufgabe04
python3 -m compileall -q scripts/aufgabe04/real_robot
```

Passing these checks validates contracts and command construction, not physical
calibration accuracy, localization quality, stopping distance, or route
clearance on the robot.
