"""Independent, leased publisher for the physical velocity topic.

The mission process sends bounded commands over a private socket. This process
keeps publishing zero after a missed lease, process death or ownership fault.
It is a software watchdog, not a replacement for a hardware emergency stop.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import socket
import struct
import threading
import time
import uuid

from .model import Velocity


LEASE_SEC = 0.20
PERIOD_SEC = 0.025
_PACKET = struct.Struct("!Qddd")


class GuardFault(RuntimeError):
    pass


class CommandLease:
    """Small pure state machine; every fault is latched until process restart."""

    def __init__(self, max_linear: float, max_angular: float):
        import math
        if not (math.isfinite(max_linear) and math.isfinite(max_angular)
                and max_linear > 0 and max_angular > 0):
            raise ValueError("invalid command limits")
        self.max_linear, self.max_angular = max_linear, max_angular
        self.sequence = 0
        self.issued = None
        self.velocity = Velocity()
        self.fault = None

    def fail(self, reason: str) -> None:
        self.fault = self.fault or reason
        self.velocity = Velocity()

    def accept(self, packet: bytes, now: float) -> None:
        import math
        if self.fault:
            return
        try:
            sequence, issued, linear, angular = _PACKET.unpack(packet)
            if not all(math.isfinite(v) for v in (issued, linear, angular, now)):
                raise ValueError("nonfinite command")
            if sequence <= self.sequence or not -0.02 <= now - issued <= LEASE_SEC:
                raise ValueError("old, future or reordered command")
            if not 0 <= linear <= self.max_linear or abs(angular) > self.max_angular:
                raise ValueError("command exceeds physical limits")
            # A late packet cannot revive a lease that has already expired.
            if self.issued is not None and now - self.issued > LEASE_SEC:
                raise ValueError("command lease expired")
            self.sequence, self.issued = sequence, issued
            self.velocity = Velocity(linear, angular)
        except (ValueError, struct.error) as exc:
            self.fail(str(exc))

    def command(self, now: float) -> Velocity:
        if self.issued is not None and now - self.issued > LEASE_SEC:
            self.fail("command lease expired")
        return self.velocity if self.fault is None else Velocity()


def _guard_process(channel, config, node_name, parent_pid):
    """ROS imports live only in the child; offline validation needs no ROS."""
    node = publisher = context = None
    closing = False

    def request_close(*_):
        nonlocal closing
        closing = True

    def notify(message):
        try:
            channel.send(message.encode()[:2048])
        except OSError:
            pass

    try:
        import rclpy
        from geometry_msgs.msg import Twist
        from rclpy.context import Context
        from rclpy.parameter import Parameter
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

        context = Context()
        rclpy.init(args=[], context=context)
        node = rclpy.create_node(node_name, context=context, namespace="/",
            use_global_arguments=False,
            parameter_overrides=[Parameter("use_sim_time", value=False)])
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        publisher = node.create_publisher(Twist, config.cmd_vel_topic, qos)
        signal.signal(signal.SIGINT, request_close)
        signal.signal(signal.SIGTERM, request_close)
        channel.setblocking(False)
        lease = CommandLease(config.max_linear_speed_mps, config.max_angular_speed_radps)
        ready = reported = False
        started = time.monotonic()
        next_tick = started
        while not closing and context.ok():
            now = time.monotonic()
            if os.getppid() != parent_pid:
                lease.fail("mission process exited")
                closing = True
            for _ in range(64):
                try:
                    packet = channel.recv(2048)
                except BlockingIOError:
                    break
                if packet == b"close":
                    closing = True
                    break
                lease.accept(packet, now)
            owners = node.get_publishers_info_by_topic(config.cmd_vel_topic)
            sole_owner = (len(owners) == 1 and owners[0].node_name == node_name
                          and owners[0].node_namespace == "/")
            if not ready and sole_owner:
                ready = True
                notify("ready")
            if (ready and not sole_owner) or (not ready and now - started > config.startup_timeout_sec):
                lease.fail("velocity publisher ownership lost")
            command = lease.command(now) if ready and not closing else Velocity()
            if lease.fault and not reported:
                notify("fault:" + lease.fault)
                reported = True
            message = Twist()
            message.linear.x = command.linear_x_mps
            message.angular.z = command.angular_z_radps
            publisher.publish(message)
            rclpy.spin_once(node, timeout_sec=0)
            next_tick = max(next_tick + PERIOD_SEC, time.monotonic())
            time.sleep(max(0.0, next_tick - time.monotonic()))
    except BaseException as exc:
        notify("fault:" + type(exc).__name__ + ": " + str(exc))
    finally:
        # Repeat zero across DDS scheduling and normal teardown, including faults.
        if publisher is not None:
            for _ in range(20):
                try:
                    publisher.publish(Twist())
                except Exception:
                    break
                time.sleep(PERIOD_SEC)
        notify("closed")
        if node is not None:
            node.destroy_node()
        if context is not None and context.ok():
            context.shutdown()
        channel.close()


class CommandGuard:
    """Nonblocking parent endpoint. Only the control timer refreshes its lease."""

    namespace = "/"

    def __init__(self, config):
        self.node_name = "a04_command_guard_" + uuid.uuid4().hex[:12]
        self._channel, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._channel.setblocking(False)
        self._lock = threading.Lock()
        self._sequence = 0
        self._closed = False
        self._fault = None
        self._ready = False
        self._process = multiprocessing.get_context("spawn").Process(
            target=_guard_process, args=(child, config, self.node_name, os.getpid()),
            name=self.node_name, daemon=False)
        try:
            self._process.start()
        except BaseException:
            self._channel.close()
            raise
        finally:
            child.close()
        try:
            deadline = time.monotonic() + config.startup_timeout_sec + 2.0
            while not self._ready:
                self.check_health()
                if time.monotonic() >= deadline:
                    raise GuardFault("command guard did not become ready")
                time.sleep(PERIOD_SEC)
        except BaseException:
            self.close()
            raise

    def _health_locked(self):
        for _ in range(32):
            try:
                message = self._channel.recv(2048).decode("utf-8", errors="replace")
            except BlockingIOError:
                break
            if message == "ready":
                self._ready = True
            elif message.startswith("fault:"):
                self._fault = message[6:]
            elif message == "closed":
                self._fault = self._fault or "command guard exited"
        if not self._process.is_alive():
            self._fault = self._fault or "command guard process exited"
        if self._fault:
            raise GuardFault(self._fault)

    def check_health(self):
        with self._lock:
            if self._closed:
                raise GuardFault("command guard is closed")
            self._health_locked()

    def set_velocity(self, velocity: Velocity):
        if not isinstance(velocity, Velocity):
            raise TypeError("typed velocity required")
        with self._lock:
            if self._closed:
                raise GuardFault("command guard is closed")
            self._health_locked()
            self._sequence += 1
            packet = _PACKET.pack(self._sequence, time.monotonic(),
                                  velocity.linear_x_mps, velocity.angular_z_radps)
            try:
                self._channel.send(packet)
            except OSError as exc:
                self._fault = "cannot refresh command guard lease"
                raise GuardFault(self._fault) from exc

    def stop(self):
        self.set_velocity(Velocity())

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._channel.send(b"close")
            except OSError:
                pass
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            # SIGTERM is handled by the child as repeated-zero teardown.
            self._process.terminate()
            self._process.join(timeout=2.0)
        self._channel.close()
        if self._process.is_alive():
            raise GuardFault("command guard could not shut down; physical stop required")
