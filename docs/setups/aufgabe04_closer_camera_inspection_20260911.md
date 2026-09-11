# Closer camera inspection point

The autonomous exploration default for `--candidate-approach-offset-m` is now **0.50 m**, reduced from 0.70 m. This is the preferred distance from the robot base pose to the stand center. It places the first camera inspection closer and also supplies the preferred standoff for subsequent local inspection routes. Final facing offset and LiDAR survey spacing retain their separate settings.

The previous experiment command contains an explicit 0.70 m argument, which overrides the new default. Replace that argument with:

```bash
--candidate-approach-offset-m 0.50 \
```

The change is in the existing [CLI configuration](../../scripts/aufgabe04/real_robot/autonomous_runner/cli.py). Existing planners and inspection adapters already consume this parameter; no new navigation policy or module is needed. Physical clearance, map collision checks, uncertainty admission, stopped observations, seven-sample angle consensus and bounded outward framing recovery remain in force. The measured setup has a 0.330 m minimum active standoff; the candidate transit keepout and map quantization impose a higher approach-planning floor of about 0.37536 m. Neither limit was reduced.

An offline plan using the latest run's first-candidate snapshot, map, admitted start and all five candidate keepouts produced:

| Requested standoff | Planned endpoint distance after grid snapping | Route length |
| --- | ---: | ---: |
| 0.70 m | 0.699389 m | 0.097236 m |
| 0.50 m | 0.476085 m | 0.320778 m |

At comparable orientation, inverse-distance scaling suggests roughly **40–50% more pixels across the head**, or about twice the image area. This is a projection estimate, not a measurement from a new camera recording. Camera optical offset, pose and actual arrival affect the result.

Validation: CLI parsing confirms the new default and preserves explicit overrides; **29 focused tests passed** for physical clearance, inspection routing, and camera framing recovery. `git diff --check` passed. See the [offline replay](../../results/audits/stand_explore_exact2_camera_all5_20260911T135346Z/validate_closer_camera_standoff.py) and [recorded route comparison](../../results/audits/stand_explore_exact2_camera_all5_20260911T135346Z/closer_camera_standoff_validation.json).

This is a pixel-density optimization. The separately audited registration and neck-rail issues have not been changed here. No live uncertainty gate, camera measurement, ROS node, deployment or robot motion was initiated by this change; real execution still requires its normal fresh admission.
