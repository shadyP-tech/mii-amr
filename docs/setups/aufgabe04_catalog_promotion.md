# Offline station identity binding and checked catalog promotion

Autonomous camera exploration records the exact candidate→QR observations in
`observed_station_identities.json`. Without explicitly supplied server evidence,
its result is `server_binding_pending`: it has no invented station names and
creates no semantic identity registry. Case is significant for both QR payloads
and server station IDs. Only surrounding transport whitespace is removed.

The commands below are offline. They do not contact the server, report a scan,
move a robot, or establish fleet or loaded-robot readiness.

## Seal an already captured robot-plans response

Save the original `/api/v1/admin/robot-plans` response and its actual capture
Unix timestamp. Select the server robot ID explicitly; it need not equal the
robot profile's local ID. Sealing preserves the original plan generation time
and requires an explicit capture time:

```sh
python3 -m scripts.aufgabe04.stations.server_identity_binding seal \
  --plans-json /path/to/robot-plans.json \
  --server-robot-id Robot_Test_01 \
  --captured-unix-sec 1788957600.0 \
  --output-json /path/to/server_qr_mapping_evidence.json
```

Use the real timestamp from the capture; the number above is illustrative.
The content seal verifies integrity, not a server signature or scan ACK. The
selected robot must have exactly one plan and one-to-one `qr_mappings`.
Conflicting, duplicated, absent and cross-robot mappings fail. Plan evidence
older than the existing 3600-second server-task bound fails when consumed;
resealing cannot renew its `generated_at` time.

Supply this artifact with `--server-qr-mapping-evidence` and the explicit
`--server-robot-id` to autonomous exploration if mappings are already available.
Discovery itself needs neither argument. Existing observations can also be
bound later:

```sh
python3 -m scripts.aufgabe04.stations.server_identity_binding bind \
  --candidate-snapshot /path/to/confirmed_candidate_snapshot.json \
  --observed-identities /path/to/observed_station_identities.json \
  --server-qr-mapping-evidence /path/to/server_qr_mapping_evidence.json \
  --server-robot-id Robot_Test_01 \
  --registry-id camera_session_identities \
  --output-json /path/to/station_identity_registry.json
```

## Promote complete camera evidence

The bridge consumes the complete original LiDAR candidate pool, its confirmed
subset, the registry saved before camera decisions, the coverage plan, and the
per-observation camera projection and recommendation bindings embedded in the
facing catalog. Select one authenticated target projection as the common
map/odom frame for all catalog records. The bridge transforms original sensor
geometry into that frame and rechecks the exact target and 0.40 m terminal
corridor with the existing fixed-target planner and continuous stand-clearance
checks. Unconfirmed LiDAR hypotheses remain obstacles.

```sh
python3 -m scripts.aufgabe04.stations.promote_autonomous_arrival_catalog \
  --facing-catalog /path/to/session/stand_facing_catalog.json \
  --candidate-snapshot /path/to/source/candidate_snapshot.json \
  --confirmed-candidate-snapshot /path/to/session/confirmed_candidate_snapshot.json \
  --observed-identities /path/to/session/observed_station_identities.json \
  --source-stand-registry /path/to/session/camera_source_stand_registry.json \
  --coverage-plan /path/to/survey/coverage_plan.json \
  --target-frame-projection /path/to/arrival_frame_projection/candidate_frame_projection.json \
  --map-yaml /path/to/map.yaml --semantic-map-id arena \
  --robot-profile /path/to/robot_profile.json \
  --camera-calibration /path/to/camera_calibration.json \
  --physical-site /path/to/arena_real.json \
  --server-qr-mapping-evidence /path/to/server_qr_mapping_evidence.json \
  --server-robot-id Robot_Test_01 \
  --output-dir /path/to/new_catalog_bundle
```

If the facing catalog already binds a semantic registry, also supply its exact
`--source-identity-registry` path. That registry must agree with the supplied
server mappings. All original recommendation sensor timestamps must still be
within 300 seconds, with the original observation time preserved. Promotion
requires at least seven axis samples and committed onboard QR face evidence.
It rejects missing legacy bindings, another calibration/map/registry, foreign
map/odom frame IDs, changed source hashes and invalid corridors. A one-candidate
pilot checkpoint is not a complete survey and cannot be promoted as one.

The new output directory contains:

- `arrival_pose_catalog.json`: the frozen, validated typed catalog.
- `candidate_snapshot.json` and `station_identity_registry.json`: confirmed
  identities in the common frame.
- `obstacle_candidate_snapshot.json`: the complete candidate pool in that frame.
- `map_bundle.json`, `survey_config_<hash>.json`, `survey_input_binding.json`
  and `survey_manifest.json`: source, validation and freeze provenance.

## Plan a task route with the complete obstacle pool

Use the promoted bundle with the existing logistics route planner, a fresh
validated server task snapshot, the actual start pose, and the same arena
bounds. Pass the full obstacle snapshot explicitly:

```sh
python3 -m scripts.aufgabe04.navigation.missions.plan_arrival_catalog_route \
  --route-purpose logistics \
  --catalog /path/to/new_catalog_bundle/arrival_pose_catalog.json \
  --candidate-snapshot /path/to/new_catalog_bundle/candidate_snapshot.json \
  --obstacle-candidate-snapshot /path/to/new_catalog_bundle/obstacle_candidate_snapshot.json \
  --station-identity-registry /path/to/new_catalog_bundle/station_identity_registry.json \
  --survey-manifest /path/to/new_catalog_bundle/survey_manifest.json \
  --map-bundle-json /path/to/new_catalog_bundle/map_bundle.json \
  --map /path/to/map.yaml --map-frame map --semantic-map-id arena \
  --task-snapshot /path/to/fresh_validated_task.json --robot-id Robot_Test_01 \
  --start-x 0.0 --start-y 0.0 --start-yaw 0.0 \
  --route-csv /path/to/task_route.csv \
  --diagnostics-json /path/to/task_route_diagnostics.json
```

Replace the illustrative start pose and supply matching arena-bound arguments.
For new catalogs, omitting the full obstacle snapshot or supplying a different
hash fails before route publication. Existing catalogs without the new
obstacle binding keep their prior behavior. The route artifact remains subject
to the existing execution authorization and live sensor/localization gates;
this adapter does not implement navigation execution, post-arrival server ACK,
puck transfer or two-robot coordination.
