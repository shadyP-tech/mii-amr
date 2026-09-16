"""Stopped-pose proof for a target split by one missing internal LiDAR beam.

Three distinct earlier scans must each contain one real contiguous target and
a real return spanning today's missing beam. Current fragments remain separate
and their raw count stays in the association. No endpoint join, synthetic return,
nearest-cluster preference, range relaxation or angle authority is introduced.
"""

from dataclasses import asdict, dataclass, replace
import json
import math

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.perception import candidate_lidar_association as lidar
from scripts.aufgabe04.perception.scan_topology import ScanTopology
from scripts.aufgabe04.perception.stand_axis_lidar_roi import PlainLaserScan
from scripts.aufgabe04.perception.stand_axis_handoff.geometry import rotate_vector
from scripts.aufgabe04.real_robot.observer.scan_target_geometry import scan_target_geometry
from scripts.aufgabe04.real_robot.observer.scan_witness_buffer import StoppedScanWitnessBuffer


MIN_WITNESS_SCANS = 3
MAX_HISTORY_SEC = 2.5
MAX_SCAN_GAP_SEC = 1.0
MAX_TRANSLATION_M = .01
MAX_ROTATION_RAD = math.radians(2.)
MAX_SOURCE_AGE_SEC = .5
MAX_FUTURE_SEC = .05
_CANDIDATE_GEOMETRY_FIELDS = ("candidate_x_m", "candidate_y_m", "stand_radius_m",
                              "stand_uncertainty_m", "lidar_range_tolerance_m")


@dataclass(frozen=True)
class ScanPersistenceContext:
    target_key: str
    epoch_key: str
    robot_pose: Pose2D
    scan_pose_map: Pose2D
    image_stamp_sec: float
    candidate_x_m: float
    candidate_y_m: float
    stand_radius_m: float
    stand_uncertainty_m: float
    lidar_range_tolerance_m: float
    scan_pose_robot: Pose2D


def scan_pose_relative_to_robot(robot_pose, scan_pose_map) -> Pose2D:
    """Express an exact map<-scan pose in the matching robot frame."""
    dx, dy = scan_pose_map.x_m - robot_pose.x_m, scan_pose_map.y_m - robot_pose.y_m
    c, s = math.cos(robot_pose.yaw_rad), math.sin(robot_pose.yaw_rad)
    return Pose2D(c * dx + s * dy, -s * dx + c * dy,
                  _angle(scan_pose_map.yaw_rad - robot_pose.yaw_rad))


def scan_pose_from_camera_extrinsics(base_translation, base_rotation,
                                    scan_translation, scan_rotation) -> Pose2D:
    """Compose fixed base<-camera and scan<-camera into fixed base<-scan."""
    values = (*base_translation, *base_rotation, *scan_translation, *scan_rotation)
    if (len(base_translation) != 3 or len(scan_translation) != 3
            or len(base_rotation) != 4 or len(scan_rotation) != 4
            or not all(type(v) in (int, float) and math.isfinite(v) for v in values)):
        raise ValueError("camera extrinsics must be finite rigid transforms")
    axes = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    base = tuple(zip(*(rotate_vector(axis, base_rotation) for axis in axes)))
    scan = tuple(zip(*(rotate_vector(axis, scan_rotation) for axis in axes)))
    rotation = tuple(tuple(sum(base[i][k] * scan[j][k] for k in range(3))
                           for j in range(3)) for i in range(3))
    if max(abs(rotation[2][0]), abs(rotation[2][1]), abs(rotation[0][2]),
           abs(rotation[1][2])) > 1e-6 or rotation[2][2] < 0:
        raise ValueError("fixed base-to-scan transform must be planar")
    translation = tuple(base_translation[i] - sum(rotation[i][j] * scan_translation[j]
                                                   for j in range(3)) for i in range(3))
    return Pose2D(translation[0], translation[1], math.atan2(rotation[1][0], rotation[0][0]))


def scan_pose_in_map(translation_xyz, rotation_xyzw) -> Pose2D:
    """Invert the caller's exact-time scan<-map TF without another lookup."""
    if len(translation_xyz) != 3 or len(rotation_xyzw) != 4:
        raise ValueError("scan transform dimensions are invalid")
    if not all(type(v) in (int, float) and math.isfinite(v)
               for v in (*translation_xyz, *rotation_xyzw)):
        raise ValueError("scan transform must be finite")
    columns = [rotate_vector(axis, rotation_xyzw) for axis in ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))]
    rotation = tuple(zip(*columns))
    # Temporal planar support cannot silently project a tilted laser plane.
    if max(abs(rotation[2][0]), abs(rotation[2][1]), abs(rotation[0][2]),
           abs(rotation[1][2])) > 1e-6 or rotation[2][2] < 0:
        raise ValueError("temporal scan support requires a planar transform")
    return Pose2D(-sum(rotation[j][0] * translation_xyz[j] for j in range(3)),
                  -sum(rotation[j][1] * translation_xyz[j] for j in range(3)),
                  math.atan2(rotation[0][1], rotation[0][0]))


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("scan persistence evidence must contain finite numbers")
    return float(value)


def _pose(value):
    if not isinstance(value, dict) or set(value) != {"x_m", "y_m", "yaw_rad"}:
        raise ValueError("scan persistence pose is malformed")
    return Pose2D(*(_number(value[k]) for k in ("x_m", "y_m", "yaw_rad")))


def _angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def _stationary(left, right):
    return (math.hypot(left.x_m - right.x_m, left.y_m - right.y_m) <= MAX_TRANSLATION_M
            and abs(_angle(left.yaw_rad - right.yaw_rad)) <= MAX_ROTATION_RAD)


def _same_extrinsic(left, right):
    return (math.hypot(left.x_m - right.x_m, left.y_m - right.y_m) <= 1e-6
            and abs(_angle(left.yaw_rad - right.yaw_rad)) <= 1e-6)


def _scan_entry(scan, context, now_sec, max_scan_age_sec):
    raw = asdict(scan)
    raw["ranges"] = [float(v) if v is not None and math.isfinite(float(v)) else None
                     for v in scan.ranges]
    return dict(scan=raw, context=asdict(context), now_sec=now_sec,
                max_scan_age_sec=max_scan_age_sec)


def _entry(association, scan, context, now_sec, max_scan_age_sec):
    search = association.search_association
    if search is None:
        raise ValueError("no current camera cone")
    return dict(**_scan_entry(scan, context, now_sec, max_scan_age_sec),
                parameters=dict(map_bearing_rad=association.map_bearing_rad,
                    observed_camera_bearing_rad=association.registered_search_bearing_rad,
                    cone_half_angle_rad=search.cone_half_angle_rad,
                    accepted_range_m=list(search.accepted_range_m),
                    min_cluster_sample_count=search.min_cluster_sample_count,
                    max_range_jump_m=search.max_range_jump_m, max_point_gap_m=search.max_point_gap_m,
                    max_camera_map_bearing_delta_rad=association.max_camera_map_bearing_delta_rad))


def _read_scan_context(entry):
    """Validate source-time scan/pose evidence before camera acquisition exists."""
    raw, context = entry["scan"], entry["context"]
    if not isinstance(raw, dict) or set(raw) != set(PlainLaserScan.__dataclass_fields__):
        raise ValueError("scan persistence scan is malformed")
    if not isinstance(raw["ranges"], (tuple, list)) or not 3 <= len(raw["ranges"]) <= 4096:
        raise ValueError("scan persistence scan size is invalid")
    scan = PlainLaserScan(**{**raw, "ranges": tuple(math.nan if v is None else _number(v)
                                                   for v in raw["ranges"])})
    for key in ("angle_min", "angle_increment", "range_min", "range_max"):
        _number(raw[key])
    if raw["angle_max"] is not None:
        _number(raw["angle_max"])
    if not isinstance(scan.scan_frame_id, str) or not scan.scan_frame_id:
        raise ValueError("scan persistence requires a named scan frame")
    if not isinstance(context, dict) or set(context) != set(ScanPersistenceContext.__dataclass_fields__):
        raise ValueError("scan persistence context is malformed")
    if any(not isinstance(context[k], str) or not context[k] for k in ("target_key", "epoch_key")):
        raise ValueError("scan persistence requires target and stationary epoch")
    robot, scan_pose = _pose(context["robot_pose"]), _pose(context["scan_pose_map"])
    extrinsic = _pose(context["scan_pose_robot"])
    # Robot pose is at image time; scan pose is at its exact scan time.
    # Their inferred relation may differ from the fixed measured extrinsic
    # only within the same stopped-pose bounds used throughout this proof.
    if not _stationary(extrinsic, scan_pose_relative_to_robot(robot, scan_pose)):
        raise ValueError("scan pose disagrees with the measured robot-to-scan relation")
    for key in _CANDIDATE_GEOMETRY_FIELDS:
        _number(context[key])
    stamp, now, age_limit = (_number(scan.scan_stamp_sec), _number(entry["now_sec"]),
                             _number(entry["max_scan_age_sec"]))
    if not 0 < age_limit <= MAX_SOURCE_AGE_SEC:
        raise ValueError("scan persistence age limit is invalid")
    if any(not -MAX_FUTURE_SEC <= now - _number(value) <= age_limit
           for value in (stamp, scan.receipt_sec, context["image_stamp_sec"])):
        raise ValueError("scan persistence sources are not fresh")
    if "input_source" in entry:
        if (entry["input_source"] != "independent_stopped_scan"
                or context["image_stamp_sec"] != stamp):
            raise ValueError("independent scan witness requires its exact scan-time pose")
    return scan, robot, scan_pose, now, age_limit


def _target_for_context(context):
    pose = _pose(context["scan_pose_map"])
    dx, dy = context["candidate_x_m"] - pose.x_m, context["candidate_y_m"] - pose.y_m
    c, s = math.cos(pose.yaw_rad), math.sin(pose.yaw_rad)
    return scan_target_geometry((c * dx + s * dy, -s * dx + c * dy, 0.),
        stand_radius_m=context["stand_radius_m"], stand_uncertainty_m=context["stand_uncertainty_m"],
        lidar_range_tolerance_m=context["lidar_range_tolerance_m"])


def _historical_search_bearing(current, historical):
    """Transfer the current head ray only inside an already verified stopped pose."""
    pose = _pose(current["context"]["scan_pose_map"])
    old_pose = _pose(historical["context"]["scan_pose_map"])
    distance = _target_for_context(current["context"]).center_range_m
    direction = pose.yaw_rad + current["parameters"]["observed_camera_bearing_rad"]
    x = pose.x_m + distance * math.cos(direction)
    y = pose.y_m + distance * math.sin(direction)
    return _angle(math.atan2(y - old_pose.y_m, x - old_pose.x_m) - old_pose.yaw_rad)


def _read_entry(entry):
    expected_fields = {"scan", "context", "now_sec", "max_scan_age_sec", "parameters"}
    if (not isinstance(entry, dict)
            or set(entry) not in (expected_fields, expected_fields | {"input_source"})):
        raise ValueError("scan persistence entry is malformed")
    scan, robot, scan_pose, now, age_limit = _read_scan_context(entry)
    context = entry["context"]
    parameters = entry["parameters"]
    expected = {"map_bearing_rad", "observed_camera_bearing_rad", "cone_half_angle_rad",
                "accepted_range_m", "min_cluster_sample_count", "max_range_jump_m",
                "max_point_gap_m", "max_camera_map_bearing_delta_rad"}
    if not isinstance(parameters, dict) or set(parameters) != expected:
        raise ValueError("scan persistence parameters are malformed")
    if type(parameters["min_cluster_sample_count"]) is not int or parameters["min_cluster_sample_count"] < 1:
        raise ValueError("scan persistence minimum samples is invalid")
    for key in expected - {"accepted_range_m"}:
        _number(parameters[key])
    if parameters["max_range_jump_m"] > .05 or parameters["max_point_gap_m"] > .04:
        raise ValueError("scan persistence cannot enlarge spatial gates")
    target = _target_for_context(context)
    if (abs(_angle(target.bearing_rad - parameters["map_bearing_rad"])) > 1e-9
            or len(parameters["accepted_range_m"]) != 2
            or any(not math.isclose(_number(value), expected_value, rel_tol=1e-9, abs_tol=1e-9)
                   for value, expected_value in zip(parameters["accepted_range_m"], target.accepted_range_m))):
        raise ValueError("candidate geometry, exact scan pose and admitted bearing/range disagree")
    association = lidar.associate_camera_registered_candidate_lidar_target(
        scan, **parameters, now_sec=now, max_scan_age_sec=age_limit)
    search = association.search_association
    if search is None or search.eligible_cluster_count < 1:
        raise ValueError("scan persistence has no currently eligible target")
    samples = lidar._valid_samples_in_map_cone(scan,
        map_bearing_rad=parameters["observed_camera_bearing_rad"],
        cone_half_angle_rad=parameters["cone_half_angle_rad"])
    samples = tuple(s for s in samples if parameters["accepted_range_m"][0] <= s.distance_m
                    <= parameters["accepted_range_m"][1])
    clusters = lidar._contiguous_clusters(samples,
        max_range_jump_m=parameters["max_range_jump_m"], max_point_gap_m=parameters["max_point_gap_m"],
        topology=ScanTopology(len(scan.ranges), scan.angle_min, scan.angle_increment,
                              scan.angle_max, scan.scan_topology_profile))
    eligible = tuple(c for c in clusters if len(c.samples) >= parameters["min_cluster_sample_count"])
    return scan, robot, scan_pose, association, eligible


def _xy(sample, pose):
    angle = sample.bearing_rad + pose.yaw_rad
    return (pose.x_m + sample.distance_m * math.cos(angle),
            pose.y_m + sample.distance_m * math.sin(angle))


def _resolved(current, witnesses):
    scan, robot, pose, association, clusters = _read_entry(current)
    if "input_source" in current:
        raise ValueError("current target requires current camera registration")
    if association.rejection_reason != "ambiguous_registered_camera_clusters" or len(clusters) != 2:
        raise ValueError("witnessed fragmentation requires exactly two current fragments")
    left, right = sorted(clusters, key=lambda c: c.start_index)
    # Only one missing internal beam is eligible. No seam, index leap,
    # out-of-range return, or previously rejected finite return can be filled.
    if (left.start_index > left.end_index or right.start_index > right.end_index
            or right.start_index != left.end_index + 2
            or math.isfinite(scan.ranges[left.end_index + 1])):
        raise ValueError("current fragments are not separated by one missing internal beam")
    gap_left, gap_right = left.samples[-1], right.samples[0]
    samples = tuple(s for cluster in (left, right) for s in cluster.samples)
    parameters = current["parameters"]
    max_gap, max_jump = parameters["max_point_gap_m"], parameters["max_range_jump_m"]
    if (abs(gap_right.distance_m - gap_left.distance_m) > max_jump
            or math.dist(_xy(gap_left, pose), _xy(gap_right, pose)) > max_gap):
        raise ValueError("current fragment spacing exceeds the unchanged spatial gates")
    if not isinstance(witnesses, (tuple, list)) or len(witnesses) != MIN_WITNESS_SCANS:
        raise ValueError("three independent witnessed scans are required")
    previous_stamp = None
    anchor_robot = anchor_scan_pose = None
    for entry in (*witnesses, current):
        old_scan, old_robot, old_pose, old_association, old_clusters = _read_entry(entry)
        stamp = old_scan.scan_stamp_sec
        if (previous_stamp is not None and not 0 < stamp - previous_stamp <= MAX_SCAN_GAP_SEC
                or not 0 <= scan.scan_stamp_sec - stamp <= MAX_HISTORY_SEC):
            raise ValueError("scan witnesses are duplicated, expired or separated by a gap")
        previous_stamp = stamp
        anchor_robot = old_robot if anchor_robot is None else anchor_robot
        anchor_scan_pose = old_pose if anchor_scan_pose is None else anchor_scan_pose
        if not _stationary(anchor_robot, old_robot) or not _stationary(anchor_scan_pose, old_pose):
            raise ValueError("scan witnesses did not share one stopped pose")
        if (any(entry["context"][k] != current["context"][k] for k in ("target_key", "epoch_key"))
                or any(entry["context"][k] != current["context"][k] for k in _CANDIDATE_GEOMETRY_FIELDS)
                or not _same_extrinsic(_pose(entry["context"]["scan_pose_robot"]),
                                       _pose(current["context"]["scan_pose_robot"]))
                or old_scan.scan_frame_id != scan.scan_frame_id
                or old_scan.scan_topology_profile != scan.scan_topology_profile
                or old_scan.angle_increment * scan.angle_increment <= 0
                or abs(old_scan.angle_increment / scan.angle_increment - 1.) > .1):
            raise ValueError("scan witness candidate, epoch or beam geometry changed")
        if entry is current:
            continue
        if (entry.get("input_source") == "independent_stopped_scan"
                and abs(_angle(entry["parameters"]["observed_camera_bearing_rad"]
                    - _historical_search_bearing(current, entry))) > 1e-9):
            raise ValueError("independent witness is not bound to the current head ray")
        if (not old_association.associated or len(old_clusters) != 1
                or old_clusters[0].start_index > old_clusters[0].end_index):
            raise ValueError("a witness contains competing targets or crosses the scan boundary")
        old_points = tuple(_xy(s, old_pose) for s in old_clusters[0].samples)
        if any(min(math.dist(_xy(s, pose), p) for p in old_points) > max_gap for s in samples):
            raise ValueError("a current fragment lacks persistent real support")
        if any(min(math.dist(p, _xy(s, pose)) for s in samples) > max_gap for p in old_points):
            raise ValueError("the witness extends beyond the current target fragments")
        # A genuinely observed connecting point must lie between today's
        # missing-beam endpoints after exact-time pose transformation.
        a, b = _xy(gap_left, pose), _xy(gap_right, pose)
        d = (b[0] - a[0], b[1] - a[1]); length2 = d[0] ** 2 + d[1] ** 2
        if length2 <= 0 or not any(
            0 < ((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / length2 < 1
            and max(math.dist(p, a), math.dist(p, b)) <= max_gap
            for p in old_points):
            raise ValueError("no historical real beam witnesses the missing interval")
    aggregate = lidar._build_cluster(samples)
    search = replace(association.search_association, associated=True, rejection_reason="",
        distance_m=aggregate.distance_m, selected_cluster_sample_count=len(samples),
        selected_cluster_start_index=samples[0].index, selected_cluster_end_index=samples[-1].index,
        selected_cluster_source_indices=tuple(s.index for s in samples),
        selected_cluster_wraps_scan_seam=False, selected_cluster_bearing_rad=aggregate.bearing_rad,
        selected_cluster_bearing_delta_from_map_rad=abs(_angle(
            aggregate.bearing_rad - association.registered_search_bearing_rad)),
        selection_source="witnessed_current_fragments")
    return replace(association, associated=True, distance_m=aggregate.distance_m,
                   rejection_reason="", search_association=search,
                   unique_eligible_cluster_required=False)


def validated_witnessed_fragmentation(proof, *, association=None):
    """Recompute persisted evidence; arbitrary flags/counts are never sufficient."""
    if (not isinstance(proof, dict)
            or set(proof) != {"schema_version", "kind", "current", "witnesses", "persistent_target_count"}
            or type(proof["schema_version"]) is not int or proof["schema_version"] != 1
            or proof["kind"] != "one_internal_missing_beam_witnessed"
            or type(proof["persistent_target_count"]) is not int or proof["persistent_target_count"] != 1):
        raise ValueError("witnessed fragmentation proof is malformed")
    try:
        result = replace(_resolved(proof["current"], proof["witnesses"]), witnessed_fragmentation=proof)
    except (TypeError, KeyError, ArithmeticError, AttributeError) as exc:
        raise ValueError("witnessed fragmentation evidence is malformed") from exc
    if association is not None and result != association:
        raise ValueError("witnessed fragmentation proof is not bound to the current association")
    return result


def registered_target_is_unique(association):
    """One raw cluster, or one independently recomputable persistent target."""
    if not isinstance(association, lidar.CameraRegisteredCandidateLidarAssociation) or not association.associated:
        return False
    if association.witnessed_fragmentation is None:
        return bool(association.unique_eligible_cluster_required and association.search_association
                    and association.search_association.associated
                    and association.search_association.eligible_cluster_count == 1)
    try:
        validated_witnessed_fragmentation(association.witnessed_fragmentation, association=association)
        return True
    except (TypeError, ValueError, ArithmeticError, KeyError):
        return False


def registered_target_metadata_is_unique(metadata):
    """The same proof check for observer debug/provenance dictionaries."""
    if not isinstance(metadata, dict) or metadata.get("associated") is not True:
        return False
    proof = metadata.get("witnessed_fragmentation")
    if proof is None:
        search = metadata.get("search_association") or {}
        return (metadata.get("unique_eligible_cluster_required") is True
                and search.get("associated") is True
                and type(search.get("eligible_cluster_count")) is int
                and search["eligible_cluster_count"] == 1)
    try:
        actual = asdict(validated_witnessed_fragmentation(proof))
        return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(metadata, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError, ArithmeticError, KeyError):
        return False


class StoppedScanTargetPersistence:
    """One candidate/epoch, three recent unique real witnesses, no robot I/O."""
    def __init__(self):
        self.reset()

    def reset(self):
        self._reset_resolution()
        self._pending_scans = StoppedScanWitnessBuffer()

    def _reset_resolution(self):
        self._history = []
        self._anchor = None
        self._last_stamp = None
        self.last_metadata = {}

    def ingest_scan(self, scan, *, context, now_sec, max_scan_age_sec):
        """Retain a fresh exact-time stopped scan even when camera fitting fails.

        For this scan-only input, ``context.image_stamp_sec`` names the robot
        pose timestamp and must equal the scan timestamp. No camera observation
        is implied. Only a later current head bearing can select its witnesses.
        """
        try:
            entry = dict(**_scan_entry(scan, context, now_sec, max_scan_age_sec),
                         input_source="independent_stopped_scan")
            _read_scan_context(entry)
            _target_for_context(entry["context"])
            if self._pending_scans.ingest(entry):
                self._reset_resolution()
            return True
        except (TypeError, ValueError, ArithmeticError, KeyError, AttributeError) as exc:
            self.reset()
            self.last_metadata = dict(accepted=False, reason=str(exc), input_source="independent_stopped_scan")
            return False

    @staticmethod
    def _register_scan_witness(entry, current):
        """Apply the current cone to historical raw returns, never to an angle."""
        old, new = entry["context"], current["context"]
        if (any(old[k] != new[k] for k in ("target_key", "epoch_key", *_CANDIDATE_GEOMETRY_FIELDS))
                or not _stationary(_pose(old["robot_pose"]), _pose(new["robot_pose"]))
                or not _stationary(_pose(old["scan_pose_map"]), _pose(new["scan_pose_map"]))
                or not _same_extrinsic(_pose(old["scan_pose_robot"]), _pose(new["scan_pose_robot"]))
                or entry["scan"]["scan_frame_id"] != current["scan"]["scan_frame_id"]):
            raise ValueError("independent scan witness does not share the current stopped candidate")
        target = _target_for_context(old)
        return dict(entry, parameters=dict(current["parameters"],
            map_bearing_rad=target.bearing_rad, accepted_range_m=list(target.accepted_range_m),
            observed_camera_bearing_rad=_historical_search_bearing(current, entry)))

    def resolve(self, association, scan, *, context, now_sec, max_scan_age_sec):
        """Register scan-only inputs using this frame, then recompute the proof."""
        try:
            current = _entry(association, scan, context, now_sec, max_scan_age_sec)
            _read_scan_context(current)
            # Rebind previously retained scan-only witnesses too: an earlier
            # image's head bearing cannot become this image's search authority.
            refreshed = []
            for old in self._history:
                if old.get("input_source") == "independent_stopped_scan":
                    try:
                        old = self._register_scan_witness(old, current)
                        if not _read_entry(old)[3].associated:
                            raise ValueError("independent scan witness is no longer unique in the current cone")
                    except (TypeError, ValueError, ArithmeticError, KeyError):
                        refreshed = []
                        break
                refreshed.append(old)
            self._history = refreshed
            for pending in self._pending_scans.take_before(scan.scan_stamp_sec, after_stamp=self._last_stamp):
                try:
                    old = self._register_scan_witness(pending, current)
                    old_scan, _, _, old_association, _ = _read_entry(old)
                    old_context = ScanPersistenceContext(**{**old["context"],
                        **{key: _pose(old["context"][key]) for key in
                           ("robot_pose", "scan_pose_map", "scan_pose_robot")}})
                    self._resolve(old_association, old_scan, context=old_context,
                        now_sec=old["now_sec"], max_scan_age_sec=old["max_scan_age_sec"],
                        input_source="independent_stopped_scan")
                except (TypeError, ValueError, ArithmeticError, KeyError):
                    # A contradiction consumes prior proof. Three later real
                    # scans may independently establish a new target again.
                    self._reset_resolution()
        except (TypeError, ValueError, ArithmeticError, KeyError, AttributeError):
            self.reset()
        return self._resolve(association, scan, context=context,
            now_sec=now_sec, max_scan_age_sec=max_scan_age_sec)

    def _resolve(self, association, scan, *, context, now_sec, max_scan_age_sec, input_source=None):
        """Return unchanged raw evidence unless the narrow proof recomputes."""
        try:
            entry = _entry(association, scan, context, now_sec, max_scan_age_sec)
            if input_source is not None:
                entry["input_source"] = input_source
            current_scan, robot, pose, recomputed, clusters = _read_entry(entry)
            # The resolver's clock read may be slightly later than the raw
            # association call. Re-evaluate freshness; compare every other gate.
            same_age = replace(recomputed, search_association=replace(
                recomputed.search_association, scan_age_sec=association.search_association.scan_age_sec))
            if association != same_age:
                raise ValueError("current scan association differs from recomputed inputs")
            key = (context.target_key, context.epoch_key, scan.scan_frame_id,
                   *(getattr(context, field) for field in _CANDIDATE_GEOMETRY_FIELDS))
            if (self._anchor is not None and (key != self._anchor[0]
                    or not _stationary(self._anchor[1], robot) or not _stationary(self._anchor[2], pose))
                    or self._last_stamp is not None and (scan.scan_stamp_sec < self._last_stamp
                    or scan.scan_stamp_sec - self._last_stamp > MAX_SCAN_GAP_SEC)):
                self.reset()
            if self._anchor is None:
                self._anchor = (key, robot, pose)
            self._history = [old for old in self._history
                             if 0 <= scan.scan_stamp_sec - old["scan"]["scan_stamp_sec"] <= MAX_HISTORY_SEC]
            result = association
            self.last_metadata = dict(accepted=association.associated,
                reason="unique_current_cluster" if association.associated else association.rejection_reason,
                witness_scan_count=len(self._history))
            if association.rejection_reason == "ambiguous_registered_camera_clusters":
                proof = dict(schema_version=1, kind="one_internal_missing_beam_witnessed",
                             current=entry, witnesses=[old for old in self._history
                                 if old["scan"]["scan_stamp_sec"] < scan.scan_stamp_sec][-MIN_WITNESS_SCANS:],
                             persistent_target_count=1)
                try:
                    result = validated_witnessed_fragmentation(proof)
                except (TypeError, ValueError, ArithmeticError, KeyError) as exc:
                    self.last_metadata = dict(accepted=False, reason=str(exc))
                    # A current contradictory or unresolved target cannot
                    # inherit an old connecting target on the next image.
                    self._history = []
            if association.associated and len(clusters) == 1 and scan.scan_stamp_sec != self._last_stamp:
                self._history.append(entry)
                self._history = self._history[-MIN_WITNESS_SCANS:]
            self._last_stamp = scan.scan_stamp_sec
            if result is not association:
                self.last_metadata = dict(accepted=True, reason="witnessed_current_fragments",
                    raw_eligible_cluster_count=association.search_association.eligible_cluster_count,
                    witness_scan_count=MIN_WITNESS_SCANS, persistent_target_count=1)
            return result
        except (TypeError, ValueError, ArithmeticError, KeyError) as exc:
            self.reset()
            self.last_metadata = dict(accepted=False, reason=str(exc))
            search = association.search_association
            if search is not None:
                search = replace(lidar._association_rejected_as_ambiguous(search),
                                 rejection_reason="scan_persistence_current_input_invalid")
            return replace(association, associated=False, distance_m=None,
                rejection_reason="scan_persistence_current_input_invalid",
                search_association=search, witnessed_fragmentation=None)
