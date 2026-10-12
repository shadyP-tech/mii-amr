"""One mapped exploration loop over a single live robot owner."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time

from .model import Candidate, InspectionResult, PlannedLeg, Pose2D, StoppedState, wrap_angle


@dataclass(frozen=True)
class MissionResult:
    visited_viewpoints: int
    coverage_ratio: float
    observations: tuple[tuple[str, str], ...]


@dataclass
class _Track:
    candidate: Candidate
    viewpoints: set[int]


class CandidateRegistry:
    """Stopped multi-view detections; merging never shrinks their envelopes."""

    def __init__(self):
        self._tracks: list[_Track] = []

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        return tuple(track.candidate for track in self._tracks)

    def observe(self, observations, viewpoint: int):
        used = set()
        for incoming in observations:
            if not isinstance(incoming, Candidate):
                raise TypeError("LiDAR detection did not return a typed candidate")
            matches = [i for i, track in enumerate(self._tracks)
                       if math.hypot(incoming.x_m-track.candidate.x_m,
                                     incoming.y_m-track.candidate.y_m) <= .18]
            if len(matches) > 1 or matches and matches[0] in used:
                raise RuntimeError("ambiguous multi-view candidate correspondence")
            if not matches:
                i = len(self._tracks)
                self._tracks.append(_Track(replace(incoming, uid=f"candidate_{i:03d}"), {viewpoint}))
                used.add(i)
                continue
            i = matches[0]
            used.add(i)
            track = self._tracks[i]
            old = track.candidate
            count = old.sightings + incoming.sightings
            x = (old.x_m*old.sightings + incoming.x_m*incoming.sightings)/count
            y = (old.y_m*old.sightings + incoming.y_m*incoming.sightings)/count
            uncertainty = max(old.uncertainty_m + math.hypot(x-old.x_m, y-old.y_m),
                              incoming.uncertainty_m + math.hypot(x-incoming.x_m, y-incoming.y_m))
            track.candidate = Candidate(old.uid, x, y, max(old.radius_m, incoming.radius_m),
                uncertainty, (old.confidence*old.sightings + incoming.confidence*incoming.sightings)/count,
                count)
            track.viewpoints.add(viewpoint)

    def admitted(self) -> tuple[Candidate, ...]:
        for track in self._tracks:
            if (len(track.viewpoints) < 2 or track.candidate.sightings < 3
                    or track.candidate.confidence < .55):
                raise RuntimeError("mapped coverage leaves an unresolved provisional candidate")
        return self.candidates


def _check_state(state, *, map_id, previous_epoch=None):
    if not isinstance(state, StoppedState) or state.map_id != map_id:
        raise RuntimeError("robot returned a stopped state for the wrong map")
    if previous_epoch is not None and state.epoch <= previous_epoch:
        raise RuntimeError("completed motion did not establish a fresh stopped epoch")
    return state


def _after_stamp(state):
    stamps = dict(state.source_stamps)
    if "scan" not in stamps:
        raise RuntimeError("stopped state lacks the scan capture boundary")
    return stamps["scan"]


def _plan_candidate(planner, grid, stopped, target, population, config, **orientation):
    from .motion import require_route_uncertainty
    leg = planner.plan_candidate(grid, stopped, target, candidates=population,
        approach_offset_m=config.approach_offset_m, robot_radius_m=config.robot_radius_m,
        collision_margin_m=config.collision_margin_m, tracking_margin_m=config.tracking_margin_m,
        minimum_standoff_m=config.minimum_standoff_m,
        inflation_radius_m=config.inflation_radius_m,
        candidate_keepout_radius_m=config.candidate_keepout_radius_m,
        camera_offset_xy_m=config.camera_offset_xy_m,
        camera_heading_offset_rad=config.camera_heading_offset_rad, **orientation)
    require_route_uncertainty(leg, stopped, config.motion_config())
    return leg


def _move(robot, leg, state, target_id):
    if (not isinstance(leg, PlannedLeg) or leg.target_id != target_id
            or leg.source_epoch != state.epoch or leg.map_id != state.map_id):
        raise RuntimeError("planner changed the bound target, map or stopped epoch")
    reached = robot.execute(leg)
    return _check_state(reached, map_id=state.map_id, previous_epoch=state.epoch)


def _arrival_geometry(state, target, config):
    """Gate base range and measured optical bearing before spending camera time."""
    pose = state.pose_map
    distance = math.hypot(target.x_m-pose.x_m, target.y_m-pose.y_m)
    if not config.minimum_standoff_m <= distance <= config.approach_offset_m + .20:
        raise RuntimeError("camera arrival violates physical standoff")
    tx, ty = config.camera_offset_xy_m
    c, s = math.cos(pose.yaw_rad), math.sin(pose.yaw_rad)
    camera_x, camera_y = pose.x_m+c*tx-s*ty, pose.y_m+s*tx+c*ty
    bearing = math.atan2(target.y_m-camera_y, target.x_m-camera_x)
    if abs(wrap_angle(bearing-pose.yaw_rad-config.camera_heading_offset_rad)) > math.radians(3):
        raise RuntimeError("camera arrival optical bearing is outside the admitted window")


def _inspect(robot, camera_module, target, state, config, population, *,
             retained_axis=None, arrival_anchor=None, clock=time.monotonic):
    _arrival_geometry(state, target, config)
    session = camera_module.InspectionSession(target, state, config, neighbors=population,
        retained_axis=retained_axis, arrival_anchor=arrival_anchor)
    deadline = clock() + config.camera_timeout_sec
    after = max(dict(state.source_stamps).values())
    while True:
        remaining = deadline-clock()
        if remaining <= 0:
            return None
        frame = robot.next_camera_frame(after_stamp=after, timeout=remaining)
        stamp = frame.stamp_sec
        if not math.isfinite(stamp) or stamp <= after:
            raise RuntimeError("camera reused a frame from before the inspection boundary")
        after = stamp
        result = session.observe(frame)
        if result is None:
            continue
        if (not isinstance(result, InspectionResult) or result.target_id != target.uid
                or result.source_epoch != state.epoch or result.source_stamp_sec < stamp):
            raise RuntimeError("camera result changed its current target or stopped frame")
        return result


def _project_orientation(result, source_state, current_state):
    """Carry measured geometry through odom when localization changes while stopped."""
    old_to_odom = source_state.map_from_odom.inverse_apply
    to_current = current_state.map_from_odom.apply
    center = to_current(old_to_odom(Pose2D(result.target_x_m, result.target_y_m)))
    observer = to_current(old_to_odom(result.source_pose))
    axis = wrap_angle(result.axis_normal_rad + current_state.map_from_odom.yaw_rad
                      - source_state.map_from_odom.yaw_rad)
    toward = math.atan2(observer.y_m-center.y_m, observer.x_m-center.x_m)
    if abs(wrap_angle(axis-toward)) > math.pi/2:
        axis = wrap_angle(axis+math.pi)
    return center, wrap_angle(axis+math.pi)


def _inspect_candidate(robot, planner, detector, camera_module, grid, config,
                       state, target, population, leg, clock):
    source_state, source_target = state, target
    state = _move(robot, leg, state, target.uid)
    arrival_center = state.map_from_odom.apply(source_state.map_from_odom.inverse_apply(
        Pose2D(target.x_m, target.y_m)))
    target = replace(target, x_m=arrival_center.x_m, y_m=arrival_center.y_m)
    scans = robot.collect_scans(count=8, after_stamp=_after_stamp(state), timeout=config.startup_timeout_sec)
    refined = detector.refine_target(scans, state, target, config)
    if refined is not None:
        if refined.uid != target.uid or math.hypot(refined.x_m-target.x_m, refined.y_m-target.y_m) > .16:
            raise RuntimeError("arrival LiDAR refinement escaped its current target")
        try:
            _arrival_geometry(state, refined, config)
        except RuntimeError:
            _arrival_geometry(state, target, config)
        else:
            target = refined
    first_state = state
    first = _inspect(robot, camera_module, target, state, config, population,
        arrival_anchor=(source_state, source_target), clock=clock)
    if first is None or first.status != "unidentified":
        return first
    if first.accepted_frames < 7 or first.axis_normal_rad is None or first.axis_half_width_rad is None:
        return replace(first, status="unresolved", axis_normal_rad=None, axis_half_width_rad=None)
    state = _check_state(robot.stopped_snapshot(), map_id=grid.map_id)
    center, opposite = _project_orientation(first, first_state, state)
    opposite_target = replace(target, x_m=center.x_m, y_m=center.y_m,
                             uncertainty_m=max(target.uncertainty_m, first.target_uncertainty_m))
    scans = robot.collect_scans(count=8, after_stamp=_after_stamp(state), timeout=config.startup_timeout_sec)
    current = detector.refine_target(scans, state, opposite_target, config)
    if current is None or current.uid != target.uid:
        raise RuntimeError("opposite translation lacks current LiDAR target support")
    if math.hypot(current.x_m-opposite_target.x_m, current.y_m-opposite_target.y_m) > max(.04, opposite_target.uncertainty_m):
        raise RuntimeError("camera orientation and current LiDAR target disagree")
    leg = _plan_candidate(planner, grid, state, opposite_target, population, config,
        normal_rad=opposite, normal_half_width_rad=first.axis_half_width_rad)
    arrived = _move(robot, leg, state, target.uid)
    arrival_center = arrived.map_from_odom.apply(state.map_from_odom.inverse_apply(
        Pose2D(opposite_target.x_m, opposite_target.y_m)))
    arrival_target = replace(opposite_target, x_m=arrival_center.x_m, y_m=arrival_center.y_m)
    result = _inspect(robot, camera_module, arrival_target, arrived, config, population,
        retained_axis=first, arrival_anchor=(first_state, target), clock=clock)
    return result


def run_mission(robot, config, grid, *, planner=None, detector=None, camera_module=None,
                clock=time.monotonic) -> MissionResult:
    """Complete mapped coverage, then inspect every admitted candidate once.

    Only robot.execute can start motion. Every error, cancellation and normal
    completion ends at the owner's latched zero-command boundary.
    """
    if planner is None:
        from . import planning as planner
    if detector is None:
        from . import lidar as detector
    if camera_module is None:
        from . import camera as camera_module
    deadline = clock() + config.mission_timeout_sec
    registry = CandidateRegistry()
    observations = []
    visited = []
    try:
        state = _check_state(robot.stopped_snapshot(), map_id=grid.map_id)
        survey = planner.coverage_plan(grid, start=state.pose_map,
            stop_spacing_m=config.stop_spacing_m, lane_count=1,
            inflation_radius_m=config.inflation_radius_m,
            coverage_threshold=.95, visibility_radius_m=1.35)
        for index, viewpoint in enumerate(survey.viewpoints):
            if clock() >= deadline:
                raise RuntimeError("exploration deadline expired")
            state = _check_state(robot.stopped_snapshot(), map_id=grid.map_id)
            population = registry.candidates
            keepouts = tuple((c.x_m,c.y_m,max(config.candidate_keepout_radius_m,
                c.radius_m+c.uncertainty_m+config.robot_radius_m+config.collision_margin_m+config.tracking_margin_m)) for c in population)
            leg = planner.plan_to_pose(grid, state, viewpoint.pose,
                target_id=f"coverage_{index:03d}", kind="coverage",
                inflation_radius_m=config.inflation_radius_m, keepouts=keepouts,
                obstacles=tuple((c.x_m,c.y_m,c.radius_m+c.uncertainty_m) for c in population))
            state = _move(robot, leg, state, f"coverage_{index:03d}")
            scans = robot.collect_scans(count=8, after_stamp=_after_stamp(state), timeout=config.startup_timeout_sec)
            registry.observe(detector.detect_targets(scans, state, config), index)
            visited.append(index)
        ratio = survey.coverage_ratio(visited)
        if ratio + 1e-12 < .95:
            raise RuntimeError("completed views do not meet mapped coverage threshold")
        population = registry.admitted()
        pending = {candidate.uid: candidate for candidate in population}
        while pending:
            if clock() >= deadline:
                raise RuntimeError("exploration deadline expired")
            state = _check_state(robot.stopped_snapshot(), map_id=grid.map_id)
            options = []
            for target in pending.values():
                try:
                    leg = _plan_candidate(planner, grid, state, target, population, config)
                except ValueError:
                    continue
                distance = sum(math.hypot(b.x_m-a.x_m,b.y_m-a.y_m) for a,b in zip(leg.poses,leg.poses[1:]))
                heading = math.atan2(leg.poses[1].y_m-state.pose_map.y_m,leg.poses[1].x_m-state.pose_map.x_m)
                turn = abs(wrap_angle(heading-state.pose_map.yaw_rad))
                options.append((turn >= 3*math.pi/4, distance/config.max_linear_speed_mps
                    + turn/config.max_angular_speed_radps, -target.confidence, -target.sightings, target.uid, leg))
            if not options:
                raise RuntimeError("no pending candidate has a physically admissible route")
            selected = min(options, key=lambda value:value[:-1])
            target = pending[selected[-2]]
            result = _inspect_candidate(robot, planner, detector, camera_module, grid, config,
                state, target, population, selected[-1], clock)
            observations.append((target.uid, "unresolved" if result is None else result.qr_id or result.status))
            del pending[target.uid]
    except BaseException:
        robot.stop("exploration failed or cancelled")
        raise
    robot.stop("exploration complete")
    return MissionResult(len(visited), ratio, tuple(observations))
