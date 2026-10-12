"""One ROS owner for sensor admission, stopped inspection and guarded motion.

Planning and vision never publish velocity. The executor services subscriptions
and the bounded control timer while the caller runs stopped image processing.
An independent command-guard process owns the physical publisher and expires
leases after a main-process failure. Whole-machine or guard-process failure
still requires a verified controller-side command timeout or hardware stop.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
import threading
import time

from .config import RigidTransform
from .model import Pose2D, StoppedState, Transform2D, Velocity, finite, wrap_angle


class RuntimeFault(RuntimeError):
    """The live session has stopped and cannot resume motion."""


class TfNotReady(RuntimeFault):
    """Exact TF may arrive during a short zero-held retry window."""


class InputsNotReady(RuntimeFault):
    """Startup may wait for this input while continuously publishing zero."""


@dataclass(frozen=True)
class TimedSample:
    message: object
    stamp_sec: float
    received_monotonic: float


@dataclass(frozen=True)
class LiveState:
    pose_map: Pose2D
    pose_odom: Pose2D
    map_from_odom: Transform2D
    base_from_scan: Transform2D
    scan: object
    velocity: Velocity
    position_covariance: tuple[float, float, float]
    yaw_variance: float
    odom_stamp_sec: float
    amcl_stamp_sec: float


def require_fresh(stamp, received, *, now_ros, now_monotonic, age=.5, future=.05):
    """Source time and local delivery time are independent mandatory gates."""
    values = (stamp, received, now_ros, now_monotonic, age, future)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise InputsNotReady("non-finite sensor time")
    if stamp <= 0 or received < 0 or now_ros <= 0 or age <= 0 or future < 0:
        raise InputsNotReady("invalid sensor clock")
    if not -future <= now_ros-stamp <= age:
        raise InputsNotReady("stale or future sensor source")
    if not 0 <= now_monotonic-received <= age:
        raise InputsNotReady("stale or regressed sensor delivery clock")


def quaternion_xyzw(q):
    values = tuple(finite(getattr(q, field), "quaternion") for field in ("x", "y", "z", "w"))
    norm = math.sqrt(sum(v*v for v in values))
    if abs(norm-1.) > .01:
        raise RuntimeFault("sensor quaternion is not normalized")
    return tuple(v/norm for v in values)


def planar_yaw(q):
    x, y, z, w = quaternion_xyzw(q)
    roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(max(-1., min(1., 2*(w*y-z*x))))
    if max(abs(roll), abs(pitch)) > math.radians(5):
        raise RuntimeFault("nonplanar base/scan transform is unsupported")
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def pose_from_message(pose):
    finite(pose.position.z, "pose z")
    return Pose2D(finite(pose.position.x, "pose x"), finite(pose.position.y, "pose y"),
                  planar_yaw(pose.orientation))


def planar_transform(message):
    transform = message.transform
    finite(transform.translation.z, "transform z")
    return Transform2D(finite(transform.translation.x, "transform x"),
                       finite(transform.translation.y, "transform y"),
                       planar_yaw(transform.rotation))


def rigid_transform(message):
    transform = message.transform
    return RigidTransform(tuple(finite(getattr(transform.translation, k), "translation")
                                for k in ("x", "y", "z")),
                          quaternion_xyzw(transform.rotation))


def relative_transform(parent_from_base, parent_from_sensor):
    point = parent_from_base.inverse_apply(Pose2D(parent_from_sensor.x_m,
        parent_from_sensor.y_m, parent_from_sensor.yaw_rad))
    return Transform2D(point.x_m, point.y_m, point.yaw_rad)


def pose_near(a, b, *, translation=.01, rotation=math.radians(2)):
    return (math.hypot(a.x_m-b.x_m, a.y_m-b.y_m) <= translation
            and abs(wrap_angle(a.yaw_rad-b.yaw_rad)) <= rotation)


def validate_covariance(values):
    if len(values) != 36:
        raise RuntimeFault("AMCL covariance must be 6 by 6")
    values = tuple(finite(v, "AMCL covariance") for v in values)
    xx, xy, yx, yy, yaw = (values[i] for i in (0, 1, 6, 7, 35))
    if min(xx, yy, yaw) < 0 or abs(xy-yx) > 1e-9 or xy*xy > xx*yy+1e-15:
        raise RuntimeFault("AMCL covariance is not positive semidefinite")
    # Planning uses the actual covariance as its inflation budget as well.
    if max(xx, yy) > .10**2 or yaw > math.radians(10)**2:
        raise InputsNotReady("AMCL uncertainty exceeds the physical operating envelope")
    return (xx, xy, yy), yaw


def validate_camera_info(info, calibration, camera_frame):
    if info.header.frame_id != camera_frame or (info.width, info.height) != (calibration.width_px, calibration.height_px):
        raise RuntimeFault("camera frame or dimensions differ from measured calibration")
    if info.distortion_model != calibration.distortion_model:
        raise RuntimeFault("camera distortion model differs from measured calibration")
    for actual, expected in ((info.k, calibration.camera_matrix),
                             (info.d, calibration.distortion_coefficients),
                             (info.r, calibration.rectification_matrix),
                             (info.p, calibration.projection_matrix)):
        if len(actual) != len(expected) or any(not math.isclose(finite(a, "CameraInfo"), b,
            rel_tol=1e-6, abs_tol=1e-6) for a, b in zip(actual, expected)):
            raise RuntimeFault("CameraInfo differs from measured calibration")


def validate_camera_mount(actual, expected):
    if math.dist(actual.translation_xyz_m, expected.translation_xyz_m) > .005:
        raise RuntimeFault("camera translation differs from measured calibration")
    dot = abs(sum(a*b for a, b in zip(actual.rotation_xyzw, expected.rotation_xyzw)))
    if 2*math.acos(min(1., dot)) > math.radians(1):
        raise RuntimeFault("camera rotation differs from measured calibration")


class RobotSession:
    """Single-use live session. Any runtime fault permanently closes motion."""

    def __init__(self, config, *, map_id=None):
        import rclpy
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.duration import Duration
        from rclpy.time import Time
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import LaserScan, CompressedImage, CameraInfo
        from std_srvs.srv import Empty
        from tf2_ros import Buffer, TransformListener

        self.config = config
        self.map_id = map_id or config.expected_map_bundle_sha256
        self._ros, self._Time, self._Duration = rclpy, Time, Duration
        self._owns_ros = not rclpy.ok()
        if self._owns_ros:
            rclpy.init(args=None)
        self.node = Node("aufgabe04_exploration", namespace=config.namespace)
        if self.node.get_parameter("use_sim_time").value:
            self.node.destroy_node()
            if self._owns_ros:
                rclpy.shutdown()
            raise RuntimeFault("physical exploration rejects simulated time")
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._closed = False
        self._fault_reason = None
        self._authorized = False
        self._arming = False
        self._guard = None
        self._tf_missing_since = None
        self._ready_once = False
        self._started = time.monotonic()
        self._last_tick = self._started
        self._last_publish = self._started
        self._last_ros = None
        self._samples = {}
        self._scans = deque(maxlen=32)
        self._images = deque(maxlen=8)
        self._info = deque(maxlen=8)
        self._stationary = deque(maxlen=40)
        self._epoch = 0
        self._consumed_epochs = set()
        self._stopped = None
        self._stop_anchor = None
        self._live = None
        self._controller = None
        self._motion_finished = False
        self._moving = False
        self._last_nomotion_request = -math.inf
        self._nomotion_future = None
        self._tf = Buffer()
        self._listener = TransformListener(self._tf, self.node, spin_thread=False)
        self._subscriptions = [
            self.node.create_subscription(LaserScan, config.scan_topic,
                lambda msg: self._receive("scan", msg), qos_profile_sensor_data),
            self.node.create_subscription(Odometry, config.odom_topic,
                lambda msg: self._receive("odom", msg), qos_profile_sensor_data),
            self.node.create_subscription(PoseWithCovarianceStamped, config.amcl_topic,
                lambda msg: self._receive("amcl", msg), 10),
            self.node.create_subscription(CompressedImage, config.compressed_image_topic,
                lambda msg: self._receive("image", msg), qos_profile_sensor_data),
            self.node.create_subscription(CameraInfo, config.camera_info_topic,
                lambda msg: self._receive("camera_info", msg), qos_profile_sensor_data),
        ]
        service = (config.namespace.rstrip("/") + "/request_nomotion_update") or "/request_nomotion_update"
        self._nomotion = self.node.create_client(Empty, service)
        self._Empty = Empty
        self._timer = self.node.create_timer(.05, self._tick)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._thread = threading.Thread(target=self._spin, name="aufgabe04-safety", daemon=True)
        self._thread.start()

    @staticmethod
    def _stamp(message):
        stamp = message.header.stamp
        if type(stamp.sec) is not int or type(stamp.nanosec) is not int or not 0 <= stamp.nanosec < 1_000_000_000:
            raise RuntimeFault("malformed ROS source timestamp")
        return stamp.sec + stamp.nanosec / 1e9

    def _receive(self, name, message):
        with self._changed:
            try:
                stamp = self._stamp(message)
                old = self._samples.get(name)
                if old is not None and stamp <= old.stamp_sec:
                    return
                sample = TimedSample(message, stamp, time.monotonic())
                self._samples[name] = sample
                queue = {"scan": self._scans, "image": self._images, "camera_info": self._info}.get(name)
                if queue is not None:
                    queue.append(sample)
            except Exception as exc:
                self._latch_fault(str(exc))
            self._changed.notify_all()

    def _spin(self):
        try:
            self._executor.spin()
        except BaseException as exc:
            with self._changed:
                self._latch_fault(f"ROS executor failed: {exc}")
                self._changed.notify_all()

    def _publish(self, command=Velocity()):
        if self._guard is None:
            if command != Velocity():
                raise RuntimeFault("motion guard is not armed")
            return
        self._guard.check_health()
        self._guard.set_velocity(command)
        self._last_publish = time.monotonic()

    def _latch_fault(self, reason):
        self._fault_reason = self._fault_reason or reason
        self._controller = None
        self._moving = False
        self._stopped = None
        self._stationary.clear()
        if self._guard is not None:
            try:
                self._guard.stop()
            except Exception:
                pass  # The independent guard also stops on EOF or lease expiry.

    def _check_open(self):
        if self._closed:
            raise RuntimeFault("robot session is closed")
        if self._fault_reason:
            raise RuntimeFault(self._fault_reason)
        if not self._ros.ok():
            raise RuntimeFault("ROS context has shut down")

    def _lookup(self, target, source, stamp=None):
        try:
            result = self._tf.lookup_transform(target, source,
                self._Time() if stamp is None else self._Time(nanoseconds=round(stamp*1e9)),
                timeout=self._Duration(seconds=0.))
        except Exception as exc:
            raise TfNotReady(f"exact TF unavailable {target} <- {source}") from exc
        if result.header.frame_id != target or result.child_frame_id != source:
            raise RuntimeFault("TF result frames differ from requested frames")
        return result

    def _ownership(self):
        if self._arming:
            return  # Guard creation is bounded and no motion is possible yet.
        owners = self.node.get_publishers_info_by_topic(self.config.cmd_vel_topic)
        own = [(o.node_name, o.node_namespace.rstrip("/") or "/") for o in owners]
        if self._guard is None:
            if own:
                raise InputsNotReady("external cmd_vel publisher before authorization")
            return
        self._guard.check_health()
        expected = (self._guard.node_name, self._guard.namespace.rstrip("/") or "/")
        if own != [expected]:
            raise InputsNotReady("cmd_vel does not have exactly the command guard publisher")

    def _fresh_sample(self, name, now_ros, now, *, age=.5):
        sample = self._samples.get(name)
        if sample is None:
            raise InputsNotReady(f"missing {name}")
        require_fresh(sample.stamp_sec, sample.received_monotonic,
                      now_ros=now_ros, now_monotonic=now, age=age)
        return sample

    def _scan_value(self, sample, *, epoch):
        from .lidar import Scan
        msg = sample.message
        if msg.header.frame_id != self.config.scan_frame:
            raise RuntimeFault("scan source frame differs from configured scan")
        n = len(msg.ranges)
        if n < 8 or not all(math.isfinite(v) for v in (msg.angle_min, msg.angle_max,
                msg.angle_increment, msg.range_min, msg.range_max)):
            raise RuntimeFault("scan geometry is invalid")
        if msg.angle_increment == 0 or not 0 <= msg.range_min < msg.range_max:
            raise RuntimeFault("scan range or angular resolution is invalid")
        if abs(msg.angle_max-(msg.angle_min+(n-1)*msg.angle_increment)) > abs(msg.angle_increment)*1.5:
            raise RuntimeFault("scan angle bounds do not describe its samples")
        transform = planar_transform(self._lookup(self.config.map_frame,
            self.config.scan_frame, sample.stamp_sec))
        return Scan(tuple(float(r) for r in msg.ranges), msg.angle_min, msg.angle_increment,
            msg.angle_max, msg.range_min, msg.range_max, sample.stamp_sec,
            sample.received_monotonic, msg.header.frame_id, transform, epoch)

    def _read_live(self, now_ros, now):
        config = self.config
        self._ownership()
        odom = self._fresh_sample("odom", now_ros, now)
        scan = self._fresh_sample("scan", now_ros, now)
        amcl = self._fresh_sample("amcl", now_ros, now, age=2.)
        if odom.message.header.frame_id != config.odom_frame or odom.message.child_frame_id != config.base_frame:
            raise RuntimeFault("odometry frame binding differs from configured robot")
        if amcl.message.header.frame_id != config.map_frame:
            raise RuntimeFault("AMCL frame differs from configured map")
        pose_odom = pose_from_message(odom.message.pose.pose)
        velocity = Velocity(odom.message.twist.twist.linear.x, odom.message.twist.twist.angular.z)
        pose_from_message(amcl.message.pose.pose)  # Validate the covariance source pose as well.
        covariance, yaw_variance = validate_covariance(amcl.message.pose.covariance)
        latest_tf = self._lookup(config.map_frame, config.odom_frame)
        latest_stamp = self._stamp(latest_tf)
        if not -1.1 <= now_ros-latest_stamp <= 1.0:
            raise InputsNotReady("map/odom localization TF is stale")
        map_from_odom = planar_transform(self._lookup(config.map_frame, config.odom_frame, odom.stamp_sec))
        pose_map = map_from_odom.apply(pose_odom)
        actual_mount = rigid_transform(self._lookup(config.base_frame, config.camera_optical_frame))
        validate_camera_mount(actual_mount, config.calibration.base_to_camera)
        mount = planar_transform(self._lookup(config.base_frame, config.scan_frame))
        if math.hypot(mount.x_m, mount.y_m) > config.scan_origin_to_base_offset_m+.005:
            raise RuntimeFault("LiDAR mount exceeds measured base offset")
        raw_scan = self._scan_value(scan, epoch=max(1, self._epoch))
        base_from_scan = relative_transform(Transform2D(pose_map.x_m, pose_map.y_m, pose_map.yaw_rad), raw_scan.map_from_scan)
        return LiveState(pose_map, pose_odom, map_from_odom, base_from_scan, raw_scan,
            velocity, covariance, yaw_variance, odom.stamp_sec, amcl.stamp_sec)

    def _request_nomotion(self, now):
        if self._moving or now-self._last_nomotion_request < .5:
            return
        if self._nomotion_future is not None and not self._nomotion_future.done():
            if now-self._last_nomotion_request > 2.:
                raise InputsNotReady("AMCL no-motion update timed out")
            return
        if self._nomotion.service_is_ready():
            self._nomotion_future = self._nomotion.call_async(self._Empty.Request())
            self._last_nomotion_request = now

    def _tick(self):
        with self._changed:
            if self._closed:
                return
            now = time.monotonic()
            try:
                if self._fault_reason:
                    self._publish()
                    return
                if now-self._started > self.config.mission_timeout_sec:
                    raise RuntimeFault("mission deadline exceeded")
                if self._moving and now-self._last_tick > .25:
                    raise RuntimeFault("control watchdog expired")
                self._last_tick = now
                now_ros = self.node.get_clock().now().nanoseconds / 1e9
                if self._last_ros is not None and now_ros < self._last_ros-.05:
                    raise RuntimeFault("ROS source clock regressed")
                self._last_ros = now_ros
                self._request_nomotion(now)
                live = self._read_live(now_ros, now)
                self._tf_missing_since = None
                self._live = live
                self._ready_once = self._ready_once or self._authorized
                if self._controller is not None:
                    self._stationary.clear()
                    from .motion import require_route_uncertainty
                    require_route_uncertainty(self._controller.leg,
                        replace(self._controller.stopped,
                            position_covariance=live.position_covariance,
                            yaw_variance=live.yaw_variance), self._controller.config)
                    step = self._controller.step(live.pose_odom, live.scan, now,
                        live.map_from_odom, base_from_scan=live.base_from_scan)
                    if step.complete:
                        self._publish()
                        self._controller = None
                        self._moving = False
                        self._motion_finished = True
                        self._stopped = None
                        self._stop_anchor = None
                    else:
                        command = step.command
                        if (abs(command.linear_x_mps) > self.config.max_linear_speed_mps+1e-12
                                or abs(command.angular_z_radps) > self.config.max_angular_speed_radps+1e-12):
                            raise RuntimeFault("controller exceeded measured speed limits")
                        self._publish(command)
                else:
                    self._publish()
                    self._record_stationary(live, now)
            except TfNotReady as exc:
                self._publish()
                self._stationary.clear()
                self._tf_missing_since = now if self._tf_missing_since is None else self._tf_missing_since
                if ((self._ready_once and now-self._tf_missing_since >= .15)
                        or now-self._started >= self.config.startup_timeout_sec):
                    self._latch_fault(str(exc))
            except InputsNotReady as exc:
                self._publish()
                self._stationary.clear()
                if self._ready_once or self._moving or now-self._started >= self.config.startup_timeout_sec:
                    self._latch_fault(str(exc))
            except Exception as exc:
                self._latch_fault(str(exc))
            finally:
                self._changed.notify_all()

    def _record_stationary(self, live, now):
        if abs(live.velocity.linear_x_mps) > .01 or abs(live.velocity.angular_z_radps) > math.radians(2):
            self._stationary.clear()
            if self._stopped is not None:
                raise RuntimeFault("robot moved during stopped observation")
            return
        if self._stop_anchor is not None:
            if (not pose_near(live.pose_odom, self._stop_anchor.pose_odom)
                    or not pose_near(live.pose_map, self._stop_anchor.pose_map, translation=.03)):
                raise RuntimeFault("stopped observation pose or map alignment changed")
        if not self._stationary or live.odom_stamp_sec > self._stationary[-1][1]:
            self._stationary.append((now, live.odom_stamp_sec, live.pose_odom))
        while self._stationary and now-self._stationary[0][0] > .8:
            self._stationary.popleft()
        if self._stationary and any(not pose_near(item[2], self._stationary[0][2]) for item in self._stationary):
            self._stationary.clear()

    def authorize_motion(self, confirmation):
        if confirmation != "RUN":
            raise RuntimeFault("explicit RUN confirmation is required")
        from .command_guard import CommandGuard
        with self._changed:
            self._check_open()
            if self._authorized or self._arming:
                raise RuntimeFault("motion authorization is single-use")
            self._arming = True
        try:
            guard = CommandGuard(self.config)
            with self._changed:
                self._check_open()
                self._guard = guard
                self._authorized = True
                self._publish()
        except BaseException:
            if 'guard' in locals():
                guard.close()
            raise
        finally:
            with self._changed:
                self._arming = False

    def stopped_snapshot(self, timeout=None):
        deadline = time.monotonic()+(self.config.startup_timeout_sec if timeout is None else timeout)
        with self._changed:
            while True:
                self._check_open()
                now = time.monotonic()
                samples = self._stationary
                if (self._authorized and self._ready_once and self._controller is None
                        and self._live is not None and len(samples) >= 5
                        and samples[-1][0]-samples[0][0] >= .4 and now-samples[-1][0] < .2):
                    live = self._live
                    if self._stopped is None:
                        self._epoch += 1
                        self._stopped = StoppedState(self._epoch, now, live.pose_map, live.pose_odom,
                            live.map_from_odom, live.position_covariance, live.yaw_variance, self.map_id,
                            (("odom", live.odom_stamp_sec), ("scan", live.scan.stamp_sec), ("amcl", live.amcl_stamp_sec)))
                        self._stop_anchor = self._stopped
                    # Planning and execution bind this exact admission snapshot.
                    # Live covariance and stopped-drift checks continue each tick.
                    return self._stopped
                if now >= deadline:
                    self._latch_fault("stationary sensor admission timed out")
                    raise RuntimeFault(self._fault_reason)
                self._changed.wait(min(.05, deadline-now))

    def execute(self, leg):
        from .motion import MotionController
        stopped = self.stopped_snapshot()
        with self._changed:
            self._check_open()
            if not self._authorized:
                raise RuntimeFault("motion was not authorized by RUN")
            if (leg.source_epoch != stopped.epoch or leg.source_epoch in self._consumed_epochs
                    or leg.map_id != self.map_id):
                raise RuntimeFault("route stopped epoch or map identity changed")
            motion_config = self.config.motion_config()
            self._controller = MotionController(leg, stopped, motion_config)
            self._consumed_epochs.add(stopped.epoch)
            self._stopped = None
            self._stop_anchor = None
            self._moving = True
            self._motion_finished = False
            self._stationary.clear()
            while not self._motion_finished:
                self._check_open()
                self._changed.wait(.05)
        return self.stopped_snapshot()

    def _stationary_epoch(self):
        self._check_open()
        if self._moving or self._controller is not None or self._stopped is None:
            raise RuntimeFault("sensor capture requires an admitted stopped epoch")
        return self._stopped

    def collect_scans(self, count=8, *, after_stamp=0., timeout=None):
        if type(count) is not int or count != 8:
            raise ValueError("stand inspection requires exactly eight scans")
        stopped = self.stopped_snapshot()
        deadline = time.monotonic()+(self.config.lidar_duration_sec if timeout is None else timeout)
        floor = max(after_stamp, self.node.get_clock().now().nanoseconds/1e9)
        result = []
        with self._changed:
            while len(result) < count:
                if self._stationary_epoch().epoch != stopped.epoch:
                    raise RuntimeFault("scan capture epoch changed")
                now = time.monotonic()
                now_ros = self.node.get_clock().now().nanoseconds/1e9
                for sample in self._scans:
                    if sample.stamp_sec <= floor:
                        continue
                    require_fresh(sample.stamp_sec, sample.received_monotonic,
                        now_ros=now_ros, now_monotonic=now)
                    if sample.received_monotonic < stopped.captured_monotonic:
                        continue
                    try:
                        value = self._scan_value(sample, epoch=stopped.epoch)
                    except TfNotReady:
                        break
                    result.append(value)
                    floor = sample.stamp_sec
                    if len(result) == count:
                        break
                if len(result) == count:
                    return tuple(result)
                if now >= deadline:
                    raise RuntimeFault("eight fresh stopped scans unavailable before deadline")
                self._changed.wait(min(.05, deadline-now))

    def next_camera_frame(self, after_stamp=0., timeout=None):
        import cv2
        import numpy as np
        from .camera import FrameBundle
        stopped = self.stopped_snapshot()
        deadline = time.monotonic()+(self.config.camera_timeout_sec if timeout is None else timeout)
        with self._changed:
            while True:
                if self._stationary_epoch().epoch != stopped.epoch:
                    raise RuntimeFault("camera capture epoch changed")
                now = time.monotonic()
                now_ros = self.node.get_clock().now().nanoseconds/1e9
                images = [s for s in self._images if s.stamp_sec > after_stamp
                          and s.received_monotonic >= self._stop_anchor.captured_monotonic
                          and s.stamp_sec >= dict(self._stop_anchor.source_stamps)["odom"]]
                if images and self._scans and self._info:
                    image = images[-1]
                    scan = min(self._scans, key=lambda s: abs(s.stamp_sec-image.stamp_sec))
                    info = min(self._info, key=lambda s: abs(s.stamp_sec-image.stamp_sec))
                    try:
                        for sample in (image, scan):
                            require_fresh(sample.stamp_sec, sample.received_monotonic,
                                now_ros=now_ros, now_monotonic=now)
                        require_fresh(info.stamp_sec, info.received_monotonic,
                            now_ros=now_ros, now_monotonic=now, age=1.0)
                        if (scan.received_monotonic < self._stop_anchor.captured_monotonic
                                or scan.stamp_sec < dict(self._stop_anchor.source_stamps)["odom"]):
                            raise InputsNotReady("scan predates stopped capture epoch")
                        if abs(image.stamp_sec-scan.stamp_sec) > .10 or abs(image.stamp_sec-info.stamp_sec) > 1.0:
                            raise InputsNotReady("camera/scan timestamps are not synchronized")
                        if image.message.header.frame_id != self.config.camera_optical_frame:
                            raise RuntimeFault("image frame differs from calibrated camera")
                        validate_camera_info(info.message, self.config.calibration, self.config.camera_optical_frame)
                        camera_tf = rigid_transform(self._lookup(self.config.map_frame,
                            self.config.camera_optical_frame, image.stamp_sec))
                        odom_tf = planar_transform(self._lookup(self.config.odom_frame,
                            self.config.base_frame, image.stamp_sec))
                        pose_odom = Pose2D(odom_tf.x_m, odom_tf.y_m, odom_tf.yaw_rad)
                        map_from_odom = planar_transform(self._lookup(self.config.map_frame,
                            self.config.odom_frame, image.stamp_sec))
                        pose_map = map_from_odom.apply(pose_odom)
                        if not pose_near(pose_odom, stopped.pose_odom):
                            raise RuntimeFault("image was acquired outside stationary pose bounds")
                        scan_value = self._scan_value(scan, epoch=stopped.epoch)
                        data = bytes(image.message.data)
                        break
                    except (InputsNotReady, TfNotReady):
                        pass
                if now >= deadline:
                    raise RuntimeFault("fresh synchronized camera tuple unavailable before deadline")
                self._changed.wait(min(.05, deadline-now))
        # Decoding happens outside the safety callback lock. The final epoch
        # and original-source checks prevent a slow decoder renewing freshness.
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[:2] != (self.config.calibration.height_px, self.config.calibration.width_px):
            raise RuntimeFault("compressed camera image could not be decoded at calibrated dimensions")
        frame.setflags(write=False)
        with self._changed:
            if self._stationary_epoch().epoch != stopped.epoch:
                raise RuntimeFault("camera decode crossed a movement epoch")
            now = time.monotonic()
            now_ros = self.node.get_clock().now().nanoseconds/1e9
            for sample in (image, scan):
                require_fresh(sample.stamp_sec, sample.received_monotonic,
                    now_ros=now_ros, now_monotonic=now)
        return FrameBundle(frame, image.stamp_sec, image.received_monotonic, scan_value,
            camera_tf, pose_map, pose_odom, stopped.epoch, now_ros, selected_monotonic=now)

    def stop(self, reason="operator stop"):
        with self._changed:
            if not self._closed:
                self._latch_fault(reason)
                self._changed.notify_all()

    def close(self):
        errors = []
        with self._changed:
            if self._closed:
                return
            self._controller = None
            self._moving = False
            self._closed = True
            try:
                self._publish()
            except Exception as exc:
                errors.append(str(exc))
            self._changed.notify_all()
        # A failed lease channel must never skip publisher/process teardown.
        for operation in (self._timer.cancel,
                          lambda: self._executor.shutdown(timeout_sec=2.),
                          lambda: self._thread.join(timeout=2.),
                          lambda: self._guard.close() if self._guard is not None else None):
            try:
                operation()
            except Exception as exc:
                errors.append(str(exc))
        if self._thread.is_alive():
            errors.append("ROS executor did not stop; motion guard is closed")
        else:
            try:
                self.node.destroy_node()
            except Exception as exc:
                errors.append(str(exc))
        if self._owns_ros and self._ros.ok():
            try:
                self._ros.shutdown()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeFault("shutdown: " + "; ".join(errors))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
