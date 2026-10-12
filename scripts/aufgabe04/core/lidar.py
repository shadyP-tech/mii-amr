"""Original-ray LiDAR detection, conservative morphology and live target matching."""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Sequence, Tuple, List
import math
import statistics
from .model import Candidate, Pose2D, StoppedState, Transform2D, wrap_angle

@dataclass(frozen=True)
class Scan:
    ranges: tuple[float, ...]
    angle_min: float
    angle_increment: float
    angle_max: float
    range_min: float
    range_max: float
    stamp_sec: float
    received_monotonic: float
    frame_id: str
    map_from_scan: Transform2D
    epoch: int

    def __post_init__(self):
        if (not isinstance(self.ranges, tuple) or len(self.ranges)<3
                or any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in self.ranges)
                or not all(math.isfinite(v) for v in (self.angle_min,self.angle_increment,
                    self.angle_max,self.range_min,self.range_max,self.stamp_sec,self.received_monotonic))
                or self.angle_increment==0 or not 0<=self.range_min<self.range_max
                or self.stamp_sec<=0 or self.received_monotonic<0
                or not self.frame_id or not isinstance(self.map_from_scan,Transform2D)
                or type(self.epoch) is not int or self.epoch<1):
            raise ValueError("invalid immutable original scan")

    @property
    def scan_topology_profile(self):
        return "full_rotation"

    def topology(self):
        return ScanTopology(len(self.ranges),self.angle_min,self.angle_increment,
                            self.angle_max,"full_rotation")

@dataclass(frozen=True)
class BaseFramePoint:
    x_m: float
    y_m: float
    bearing_rad: float
    range_m: float
    source_index: int


@dataclass(frozen=True)
class StandCandidate:
    candidate_id: str
    bearing_rad: float
    distance_m: float
    approximate_width_m: float
    center_x_m: float
    center_y_m: float
    point_count: int
    confidence: float
    source_indices: Tuple[int, ...] = ()
    wraps_scan_seam: bool = False


@dataclass(frozen=True)
class LidarStandDetectorConfig:
    min_range_m: float = 0.08
    max_range_m: float = 3.5
    max_cluster_gap_m: float = 0.08
    # Sparse returns can occupy two adjacent beams. Use the scan's supplied
    # increment (about 1.6 degrees in recent real receipts); these defaults
    # are proposal heuristics, not measurements of the laser-plane stand shape.
    min_cluster_points: int = 2
    min_width_m: float = 0.03
    max_width_m: float = 0.45


def bounded_endpoint_fragments(scan, index_groups):
    """Recognize the LDS gap eligible for a search hint or temporal proof.

    This is only eligibility: callers needing target uniqueness must still
    prove continuity using independent scans. Original angles remain intact.
    """
    if len(index_groups) != 2 or any(not group for group in index_groups):
        return False
    left, right = sorted(index_groups, key=lambda group: group[0])
    if (left[0] != 0 or right[-1] != len(scan.ranges) - 1 or left[-1] >= right[0]
            or any(b != a + 1 for group in (left, right) for a, b in zip(group, group[1:]))):
        return False
    topology = ScanTopology(len(scan.ranges), scan.angle_min, scan.angle_increment,
                            scan.angle_max, scan.scan_topology_profile).evidence()
    if (topology["profile"] != "full_rotation"
            or topology["reason"] not in {"inconsistent_scan_endpoint_metadata",
                                         "seam_is_not_one_sampling_step"}
            or not 0 < abs(scan.angle_increment) <= math.radians(2.)):
        return False
    indexed, reported, error = (topology[key] for key in (
        "indexed_seam_gap_steps", "reported_seam_gap_steps", "endpoint_metadata_error_steps"))
    return (all(type(v) in (int, float) and math.isfinite(v) for v in (indexed, reported, error))
            and .9 <= indexed <= 2.1 and .9 <= reported <= 1.1 and error <= 1.1)


MAX_SEAM_STEP_ERROR = 0.10


MAX_ENDPOINT_METADATA_ERROR_STEPS = 0.25


_EPSILON = 1.0e-9


@dataclass(frozen=True)
class ScanTopology:
    sample_count: int
    angle_min_rad: float
    angle_increment_rad: float
    angle_max_rad: float | None = None
    profile: str = "linear"

    def evidence(self) -> dict[str, object]:
        result = {
            "profile": self.profile,
            "sample_count": self.sample_count if type(self.sample_count) is int else None,
            "circular_adjacency_enabled": False,
            "reason": "linear_profile",
            "indexed_seam_gap_steps": None,
            "reported_seam_gap_steps": None,
            "endpoint_metadata_error_steps": None,
            "maximum_seam_step_error": MAX_SEAM_STEP_ERROR,
            "maximum_endpoint_metadata_error_steps": MAX_ENDPOINT_METADATA_ERROR_STEPS,
        }
        for name in ("angle_min_rad", "angle_max_rad", "angle_increment_rad"):
            value = getattr(self, name)
            result[name] = value if type(value) in (int, float) and math.isfinite(value) else None
        if self.profile != "full_rotation":
            if self.profile != "linear":
                result["reason"] = "unknown_scan_topology_profile"
            return result
        if (
            type(self.sample_count) is not int or self.sample_count < 3
            or not all(type(value) in (int, float) and math.isfinite(value)
                       for value in (self.angle_min_rad, self.angle_max_rad, self.angle_increment_rad))
            or self.angle_increment_rad == 0.0
        ):
            result["reason"] = "invalid_original_scan_geometry"
            return result
        step = abs(self.angle_increment_rad)
        direction = 1.0 if self.angle_increment_rad > 0.0 else -1.0
        indexed_span = (self.sample_count - 1) * step
        reported_span = direction * (self.angle_max_rad - self.angle_min_rad)
        if not (0.0 < indexed_span < math.tau and 0.0 < reported_span < math.tau):
            result["reason"] = "partial_or_overlapping_scan_geometry"
            return result
        indexed_gap = (math.tau - indexed_span) / step
        reported_gap = (math.tau - reported_span) / step
        endpoint_error = abs(reported_span - indexed_span) / step
        result.update(indexed_seam_gap_steps=indexed_gap,
                      reported_seam_gap_steps=reported_gap,
                      endpoint_metadata_error_steps=endpoint_error)
        if endpoint_error > MAX_ENDPOINT_METADATA_ERROR_STEPS + _EPSILON:
            result["reason"] = "inconsistent_scan_endpoint_metadata"
        elif any(abs(gap - 1.0) > MAX_SEAM_STEP_ERROR + _EPSILON
                 for gap in (indexed_gap, reported_gap)):
            result["reason"] = "seam_is_not_one_sampling_step"
        else:
            result.update(circular_adjacency_enabled=True, reason="validated_full_rotation")
        return result

    def joins_endpoints(self, left_index: int, right_index: int) -> bool:
        return (
            self.evidence()["circular_adjacency_enabled"] is True
            and type(left_index) is int and type(right_index) is int
            and left_index == self.sample_count - 1 and right_index == 0
        )


def wraps_scan_seam(source_indices) -> bool:
    """Wrapped groups are represented in acquisition order across N-1 -> 0."""

    return any(right < left for left, right in zip(source_indices, source_indices[1:]))


def _valid_range(raw_range: float, config: LidarStandDetectorConfig) -> bool:
    return (
        raw_range is not None
        and math.isfinite(float(raw_range))
        and float(raw_range) >= config.min_range_m
        and float(raw_range) < config.max_range_m
    )


def scan_points_from_ranges(
    ranges: Sequence[float],
    *,
    angle_min_rad: float,
    angle_increment_rad: float,
    config: LidarStandDetectorConfig | None = None,
) -> List[BaseFramePoint]:
    cfg = config or LidarStandDetectorConfig()
    if (not all(type(value) in (int, float) and math.isfinite(value)
                for value in (angle_min_rad, angle_increment_rad))
            or angle_increment_rad == 0.0):
        return []
    points: List[BaseFramePoint] = []
    for index, raw_range in enumerate(ranges):
        if not _valid_range(raw_range, cfg):
            continue
        distance = float(raw_range)
        bearing = angle_min_rad + index * angle_increment_rad
        points.append(
            BaseFramePoint(
                x_m=distance * math.cos(bearing),
                y_m=distance * math.sin(bearing),
                bearing_rad=bearing,
                range_m=distance,
                source_index=index,
            )
        )
    return points


def _distance(a: BaseFramePoint, b: BaseFramePoint) -> float:
    return math.hypot(a.x_m - b.x_m, a.y_m - b.y_m)


def cluster_scan_points(
    points: Sequence[BaseFramePoint],
    *,
    config: LidarStandDetectorConfig | None = None,
    topology: ScanTopology | None = None,
) -> List[List[BaseFramePoint]]:
    cfg = config or LidarStandDetectorConfig()
    ordered = sorted(points, key=lambda point: point.source_index)
    clusters: List[List[BaseFramePoint]] = []
    current: List[BaseFramePoint] = []

    for point in ordered:
        if not current:
            current = [point]
            continue
        previous = current[-1]
        adjacent_index = point.source_index == previous.source_index + 1
        close_enough = _distance(previous, point) <= cfg.max_cluster_gap_m
        if adjacent_index and close_enough:
            current.append(point)
            continue
        clusters.append(current)
        current = [point]

    if current:
        clusters.append(current)
    if (
        len(clusters) > 1 and topology is not None
        and topology.joins_endpoints(clusters[-1][-1].source_index, clusters[0][0].source_index)
        and _distance(clusters[-1][-1], clusters[0][0]) <= cfg.max_cluster_gap_m
    ):
        # Keep explicit acquisition order over the seam, not a misleading
        # sorted interval that appears to include every ray between 0 and N-1.
        clusters = [clusters[-1] + clusters[0], *clusters[1:-1]]
    return clusters


def _cluster_width(cluster: Sequence[BaseFramePoint]) -> float:
    if len(cluster) < 2:
        return 0.0
    return max(_distance(a, b) for a in cluster for b in cluster)


def _candidate_from_cluster(
    cluster: Sequence[BaseFramePoint],
    *,
    candidate_index: int,
    config: LidarStandDetectorConfig,
) -> StandCandidate | None:
    if len(cluster) < config.min_cluster_points:
        return None
    width = _cluster_width(cluster)
    if width < config.min_width_m or width > config.max_width_m:
        return None

    center_x = sum(point.x_m for point in cluster) / len(cluster)
    center_y = sum(point.y_m for point in cluster) / len(cluster)
    distance = math.hypot(center_x, center_y)
    bearing = math.atan2(center_y, center_x)

    point_score = min(1.0, len(cluster) / max(config.min_cluster_points * 2.0, 1.0))
    width_midpoint = (config.min_width_m + config.max_width_m) / 2.0
    width_half_span = max((config.max_width_m - config.min_width_m) / 2.0, 0.001)
    width_score = max(0.0, 1.0 - abs(width - width_midpoint) / width_half_span)
    confidence = max(0.0, min(1.0, 0.45 + 0.30 * point_score + 0.25 * width_score))

    return StandCandidate(
        candidate_id=f"stand_candidate_{candidate_index:02d}",
        bearing_rad=bearing,
        distance_m=distance,
        approximate_width_m=width,
        center_x_m=center_x,
        center_y_m=center_y,
        point_count=len(cluster),
        confidence=confidence,
        source_indices=tuple(point.source_index for point in cluster),
        wraps_scan_seam=wraps_scan_seam(tuple(point.source_index for point in cluster)),
    )


def detect_stand_candidates(
    points: Sequence[BaseFramePoint],
    *,
    config: LidarStandDetectorConfig | None = None,
    topology: ScanTopology | None = None,
) -> List[StandCandidate]:
    cfg = config or LidarStandDetectorConfig()
    candidates: List[StandCandidate] = []
    for cluster in cluster_scan_points(points, config=cfg, topology=topology):
        candidate = _candidate_from_cluster(
            cluster,
            candidate_index=len(candidates) + 1,
            config=cfg,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def detect_stand_candidates_from_scan(
    ranges: Sequence[float],
    *,
    angle_min_rad: float,
    angle_increment_rad: float,
    angle_max_rad: float | None = None,
    scan_topology_profile: str = "linear",
    config: LidarStandDetectorConfig | None = None,
) -> List[StandCandidate]:
    cfg = config or LidarStandDetectorConfig()
    points = scan_points_from_ranges(
        ranges,
        angle_min_rad=angle_min_rad,
        angle_increment_rad=angle_increment_rad,
        config=cfg,
    )
    topology = ScanTopology(len(ranges), angle_min_rad, angle_increment_rad,
                            angle_max_rad, scan_topology_profile)
    return detect_stand_candidates(points, config=cfg, topology=topology)


@dataclass(frozen=True)
class TargetMatch:
    x_m: float
    y_m: float
    distance_m: float
    bearing_rad: float
    source_indices: tuple[int, ...]
    uncertainty_m: float


def match_target(scan: Scan, target: Candidate, *, camera_bearing=None,
                 cone_half_angle=math.radians(15.), neighbors=()) -> TargetMatch | None:
    """Match only a unique contiguous, range-gated current cluster.

    Original indices are retained. Range filtering precedes clustering, and a
    second eligible cluster is ambiguity rather than an invitation to pick the
    nearest return. Camera registration cannot move more than twelve degrees
    from the mapped target bearing.
    """
    local=scan.map_from_scan.inverse_apply(Pose2D(target.x_m,target.y_m))
    expected=math.hypot(local.x_m,local.y_m)
    bearing=math.atan2(local.y_m,local.x_m)
    search=bearing if camera_bearing is None else camera_bearing
    if (not math.isfinite(search) or expected<=0 or
            abs(wrap_angle(search-bearing))>math.radians(12.)):
        return None
    tolerance=.12+target.uncertainty_m+.04
    points=[]
    for index,distance in enumerate(scan.ranges):
        angle=scan.angle_min+index*scan.angle_increment
        if (math.isfinite(distance) and max(.08,scan.range_min)<=distance<min(3.5,scan.range_max)
                and expected-tolerance<=distance<=expected+tolerance
                and abs(wrap_angle(angle-search))<=cone_half_angle):
            points.append(BaseFramePoint(distance*math.cos(angle),distance*math.sin(angle),
                                         angle,distance,index))
    clusters=cluster_scan_points(points,config=LidarStandDetectorConfig(max_cluster_gap_m=.04),
                                 topology=scan.topology())
    eligible=[]
    for group in clusters:
        # A discontinuity in depth cannot be bridged merely because adjacent
        # angular rays happen to be close in Cartesian coordinates.
        split=[];part=[]
        for point in group:
            if part and abs(point.range_m-part[-1].range_m)>.05:
                split.append(part);part=[]
            part.append(point)
        if part:split.append(part)
        eligible.extend(split)
    if len(eligible)!=1 or len(eligible[0])<3:return None
    group=eligible[0]
    x=statistics.mean(p.x_m for p in group);y=statistics.mean(p.y_m for p in group)
    mapped=scan.map_from_scan.apply(Pose2D(x,y))
    if math.hypot(mapped.x_m-target.x_m,mapped.y_m-target.y_m)>tolerance:
        return None
    # Association must not be equally compatible with another mapped stand.
    for other in neighbors:
        if other.uid!=target.uid and math.hypot(mapped.x_m-other.x_m,mapped.y_m-other.y_m)<=max(
                tolerance,other.uncertainty_m+.06):
            return None
    spread=max(math.hypot(p.x_m-x,p.y_m-y) for p in group)
    if spread>.10:return None
    return TargetMatch(mapped.x_m,mapped.y_m,math.hypot(x,y),math.atan2(y,x),
                       tuple(p.source_index for p in group),max(.02,spread))


def _wall_extension(scan, candidate):
    source=candidate.source_indices; samples={i:r for i,r in enumerate(scan.ranges)
        if math.isfinite(r) and max(.08,scan.range_min)<=r<min(3.5,scan.range_max)}
    indices=list(source); topology=scan.topology()
    def point(i):
        a=scan.angle_min+i*scan.angle_increment
        return samples[i]*math.cos(a),samples[i]*math.sin(a)
    for endpoint,direction in ((source[0],-1),(source[-1],1)):
        previous=endpoint
        for _ in range(4):
            index=previous+direction
            if not 0<=index<len(scan.ranges):
                if not topology.joins_endpoints(len(scan.ranges)-1,0):break
                index%=len(scan.ranges)
            if index in indices or index not in samples or math.dist(point(previous),point(index))>.160:break
            indices.append(index);previous=index
    return max((math.dist(point(a),point(b)) for a in indices for b in indices),default=0.),len(indices)>len(source)


def detect_targets(scans, stopped: StoppedState, config) -> tuple[Candidate,...]:
    """Confirm physical stand tracks across distinct scans in one stopped epoch."""
    scans=tuple(scans)
    if not scans:return ()
    if any(s.epoch!=stopped.epoch for s in scans):raise ValueError("scan epoch differs from stopped capture")
    if any(b.stamp_sec<=a.stamp_sec for a,b in zip(scans,scans[1:])):
        raise ValueError("confirmation requires distinct ordered original scans")
    diameter=2.*config.survey_candidate_radius_m
    proposal_config=LidarStandDetectorConfig(min_width_m=min(.03,.25*diameter),
                                             max_width_m=max(.45,3.5*diameter))
    tracks=[]
    for scan in scans:
        proposals=detect_stand_candidates_from_scan(scan.ranges,angle_min_rad=scan.angle_min,
            angle_increment_rad=scan.angle_increment,angle_max_rad=scan.angle_max,
            scan_topology_profile="full_rotation",config=replace(proposal_config,
                min_range_m=max(.08,scan.range_min),max_range_m=min(3.5,scan.range_max)))
        observations=[]
        for raw in proposals:
            pose=scan.map_from_scan.apply(Pose2D(raw.center_x_m,raw.center_y_m))
            if raw.confidence<.55:continue
            width,extended=_wall_extension(scan,raw)
            observations.append((pose,raw,width,extended,scan.stamp_sec))
        # Freeze track centers for this scan. One scan may update each track at
        # most once; competing assignments do not create extra confirmations.
        centers=[(statistics.mean(v[0].x_m for v in t),statistics.mean(v[0].y_m for v in t)) for t in tracks]
        assignment={}
        for obs in observations:
            matches=[i for i,p in enumerate(centers) if math.hypot(obs[0].x_m-p[0],obs[0].y_m-p[1])<=.10]
            if len(matches)==1:assignment.setdefault(matches[0],[]).append(obs)
            elif not matches:tracks.append([obs])
        for index,items in assignment.items():
            if len(items)==1:tracks[index].append(items[0])
    result=[]
    for track in tracks:
        if len(track)<3:continue
        widths=sorted(item[1].approximate_width_m for item in track)
        median=statistics.median(widths)
        q=.75*(len(widths)-1);lo=int(q);hi=min(lo+1,len(widths)-1)
        upper=widths[lo]+(widths[hi]-widths[lo])*(q-lo)
        mad=statistics.median(abs(w-median) for w in widths)
        inliers=sum(.25*diameter<=w<=1.50*diameter for w in widths)/len(widths)
        affected=sum(extended and extent>1.25*diameter for _,_,extent,extended,_ in track)
        if not (.25*diameter<=median<=1.25*diameter and upper<=1.50*diameter
                and mad<=.30*diameter and inliers>=.75
                and not (affected>=3 and affected/len(track)>.25+1e-12)):
            continue
        x=statistics.mean(item[0].x_m for item in track); y=statistics.mean(item[0].y_m for item in track)
        uncertainty=max(.02,max(math.hypot(item[0].x_m-x,item[0].y_m-y) for item in track))
        result.append(Candidate(f"epoch-{stopped.epoch}-stand-{len(result)+1}",x,y,
            config.model.navigation_footprint_radius_m,uncertainty,
            statistics.mean(item[1].confidence for item in track),len(track)))
    return tuple(result)


def refine_target(scans, stopped, target, config):
    matches=[m for scan in scans if scan.epoch==stopped.epoch
             for m in [match_target(scan,target)] if m is not None]
    if len(matches)<3:return None
    x=statistics.median(m.x_m for m in matches);y=statistics.median(m.y_m for m in matches)
    if any(math.hypot(m.x_m-x,m.y_m-y)>.05 for m in matches):return None
    return replace(target,x_m=x,y_m=y,uncertainty_m=max(.02,
        max(math.hypot(m.x_m-x,m.y_m-y) for m in matches)))
