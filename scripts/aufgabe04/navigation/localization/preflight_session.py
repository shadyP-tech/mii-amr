"""Session-local JSON transport for preflights across mission subprocesses.

Only the service retains TF state. Each request creates a fresh observation
node and stationary capture. No motion publishers or admission-result cache.
"""

from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import socketserver
import subprocess
import sys
import tempfile
import time


SESSION_SOCKET_ENV = "AUFGABE04_PREFLIGHT_SESSION_SOCKET"
MAX_MESSAGE_BYTES = 4 * 1024 * 1024


def _read_message(stream):
    data = stream.readline(MAX_MESSAGE_BYTES + 1)
    if len(data) > MAX_MESSAGE_BYTES or not data.endswith(b"\n"):
        raise RuntimeError("invalid preflight session message length")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise RuntimeError("preflight session message must be an object")
    return value


def _write_message(stream, value):
    data = json.dumps(value, allow_nan=False).encode() + b"\n"
    if len(data) > MAX_MESSAGE_BYTES:
        raise RuntimeError("preflight session message too large")
    stream.write(data)
    stream.flush()


def _exchange(path, message, timeout_sec):
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(timeout_sec)
            connection.connect(path)
            with connection.makefile("rwb") as stream:
                _write_message(stream, message)
                response = _read_message(stream)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"preflight session unavailable: {exc}") from exc
    if "error" in response:
        raise RuntimeError(f"preflight session failed: {response['error']}")
    return response


def collect_from_session(config, node_options):
    """Return None only when no session is configured; failures never fall back."""
    path = os.environ.get(SESSION_SOCKET_ENV)
    if not path:
        return None
    from .ros_preflight import RosObservation, RosPreflightResult

    options = dict(node_options)
    options["preflight_requirements"] = asdict(options["preflight_requirements"])
    timeout = max(30.0, options["observation_window_sec"]
                  + options["nomotion_update_timeout_sec"] + 10.0)
    payload = _exchange(path, {"config": asdict(config), "options": options}, timeout)
    payload["observations"] = [RosObservation(**item) for item in payload["observations"]]
    result = RosPreflightResult(**payload)
    result.to_json_dict()  # Apply the same evidence contract after transport.
    return result


def _wait_for_service(path, *, is_alive, timeout_sec=5.0):
    deadline = time.monotonic() + timeout_sec
    while is_alive() and time.monotonic() < deadline:
        try:
            # A socket path can exist between bind() and listen(). Readiness
            # requires a completed handshake, not merely filesystem presence.
            _exchange(path, {"ping": True}, min(0.2, max(0.001, deadline - time.monotonic())))
            return
        except RuntimeError:
            time.sleep(0.01)
    raise RuntimeError("preflight session failed to start")


@contextmanager
def mission_preflight_session():
    """Own one observation-only service for the entire autonomous mission.

    Children inherit the private socket path. ROS is loaded on the first real
    request, so argument validation/offline dry tooling needs no ROS runtime.
    """
    if os.environ.get(SESSION_SOCKET_ENV):
        # Nested mission helpers use the owner's service and cannot tear it down.
        yield
        return
    # Keep AF_UNIX paths short on both Linux and macOS; mkdtemp is mode 0700.
    with tempfile.TemporaryDirectory(prefix="a04-tf-", dir="/tmp") as directory:
        path = str(Path(directory) / "preflight.sock")
        process = subprocess.Popen(
            [sys.executable, "-m", __name__, "--serve", path, str(os.getpid())],
            cwd=Path(__file__).resolve().parents[4],
            stdin=subprocess.DEVNULL,
        )
        try:
            _wait_for_service(path, is_alive=lambda: process.poll() is None)
            os.environ[SESSION_SOCKET_ENV] = path
            yield
        finally:
            os.environ.pop(SESSION_SOCKET_ENV, None)
            if process.poll() is None:
                try:
                    _exchange(path, {"stop": True}, 2.0)
                except RuntimeError:
                    process.terminate()
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)


def serve(path, parent_pid, runtime_factory=None):
    """Serialize captures; a retained TF executor continues between requests."""
    if runtime_factory is None:
        from .preflight_session_runtime import PreflightSessionRuntime
        runtime_factory = PreflightSessionRuntime
    runtime = None
    stopping = False

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            nonlocal runtime, stopping
            self.connection.settimeout(5.0)
            try:
                request = _read_message(self.rfile)
                if request == {"ping": True}:
                    response = {"ready": True}
                elif request == {"stop": True}:
                    stopping = True
                    response = {"stopped": True}
                else:
                    if runtime is None:
                        runtime = runtime_factory()
                    response = runtime.collect(request)
                _write_message(self.wfile, response)
            except Exception as exc:
                # Transport/runtime failure cannot become successful admission.
                try:
                    _write_message(self.wfile, {"error": f"{type(exc).__name__}: {exc}"})
                except (OSError, RuntimeError):
                    pass

    try:
        with socketserver.UnixStreamServer(path, Handler) as server:
            server.timeout = 0.2
            while not stopping and os.getppid() == parent_pid:
                server.handle_request()
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--serve":
        raise SystemExit("usage: preflight_session --serve SOCKET PARENT_PID")
    serve(sys.argv[2], int(sys.argv[3]))
