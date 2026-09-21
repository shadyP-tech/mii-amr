from dataclasses import asdict
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.aufgabe04.navigation.localization import preflight_session as session
from scripts.aufgabe04.navigation.localization import preflight_session_runtime as runtime
from scripts.aufgabe04.navigation.localization import ros_preflight as preflight
from scripts.aufgabe04.navigation.foundation.ros_runtime_config import RuntimeConfig, resolve_runtime_config


def request():
    return {'config': asdict(resolve_runtime_config(RuntimeConfig())),
            'options': {'preflight_requirements': {}}}


@pytest.fixture
def owner(monkeypatch):
    stop = threading.Event()
    executor = SimpleNamespace(spin=lambda: stop.wait(), add_node=lambda n: True,
                               shutdown=lambda **kwargs: stop.set())
    monkeypatch.setitem(sys.modules, 'rclpy.executors', SimpleNamespace(
        SingleThreadedExecutor=lambda: executor))
    # Humble requires both duration keywords (even when one is None).
    def jump_threshold(*, min_forward, min_backward, on_clock_change=True):
        return (min_forward, min_backward, on_clock_change)
    monkeypatch.setitem(sys.modules, 'rclpy.clock', SimpleNamespace(JumpThreshold=jump_threshold))
    monkeypatch.setattr(preflight, 'rclpy', Mock())
    monkeypatch.setattr(preflight, 'Node', Mock())
    monkeypatch.setattr(preflight, '_node_parameter_overrides', lambda value: [])
    monkeypatch.setattr(preflight, 'Buffer', Mock())
    monkeypatch.setattr(preflight, 'Duration', Mock())
    monkeypatch.setattr(preflight, 'TransformListener', Mock())
    instance = runtime.PreflightSessionRuntime()
    try:
        yield instance
    finally:
        instance.close()


def test_two_legs_retain_tf_but_create_independent_capture_nodes(owner, monkeypatch):
    nodes, buffers = [], []

    def node_factory(config, tf_buffer, **options):
        buffers.append(tf_buffer)
        # A newly constructed capture has no previous leg's stationary samples.
        result = preflight.RosPreflightResult(True, [], [], config.as_log_dict())
        node = SimpleNamespace(collect=lambda: result, destroy_node=Mock())
        nodes.append(node)
        return node

    monkeypatch.setattr(preflight, 'RosPreflightNode', node_factory)
    first = owner.collect(request())
    second = owner.collect(request())
    assert first['stationary_amcl_samples'] == second['stationary_amcl_samples'] == []
    assert buffers[0] is buffers[1] is owner.buffer
    assert nodes[0] is not nodes[1]
    assert all(node.destroy_node.call_count == 1 for node in nodes)
    assert preflight.Buffer.call_count == preflight.TransformListener.call_count == 1


def test_configuration_change_cannot_reuse_session_buffer(owner):
    config = resolve_runtime_config(RuntimeConfig())
    owner._ensure_listener(config)
    with pytest.raises(RuntimeError, match='configuration changed'):
        owner._ensure_listener(resolve_runtime_config(RuntimeConfig(namespace='robot2')))


def test_clock_reset_during_capture_rejects_result_and_destroys_node(owner, monkeypatch):
    def collect():
        owner._clock_jumped(None)
        return preflight.RosPreflightResult(True, [], [], {})
    node = SimpleNamespace(collect=collect, destroy_node=Mock())
    monkeypatch.setattr(preflight, 'RosPreflightNode', lambda *a, **kw: node)
    with pytest.raises(RuntimeError, match='clock changed'):
        owner.collect(request())
    owner.buffer.clear.assert_called_once()
    node.destroy_node.assert_called_once()


def test_capture_exception_cleans_up_node(owner, monkeypatch):
    node = SimpleNamespace(collect=Mock(side_effect=ValueError('bad evidence')),
                           destroy_node=Mock())
    monkeypatch.setattr(preflight, 'RosPreflightNode', lambda *a, **kw: node)
    with pytest.raises(ValueError, match='bad evidence'):
        owner.collect(request())
    node.destroy_node.assert_called_once()


def test_stopped_tf_executor_blocks_further_captures(owner):
    config = resolve_runtime_config(RuntimeConfig())
    owner._ensure_listener(config)
    owner.executor_error = RuntimeError('listener failed')
    with pytest.raises(RuntimeError, match='executor stopped'):
        owner._ensure_listener(config)


def test_session_is_inherited_by_child_and_cleaned_up(monkeypatch):
    monkeypatch.delenv(session.SESSION_SOCKET_ENV, raising=False)
    with pytest.raises(ValueError, match='mission failed'):
        with session.mission_preflight_session():
            path = os.environ[session.SESSION_SOCKET_ENV]
            with session.mission_preflight_session():
                assert os.environ[session.SESSION_SOCKET_ENV] == path
            result = subprocess.run([
                sys.executable, '-c',
                'import os; from scripts.aufgabe04.navigation.localization.preflight_session '
                'import _exchange, SESSION_SOCKET_ENV; '
                'assert _exchange(os.environ[SESSION_SOCKET_ENV], {"ping": True}, 2)["ready"]',
            ], capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            raise ValueError('mission failed')
    assert session.SESSION_SOCKET_ENV not in os.environ
    assert not Path(path).exists()


def test_missing_service_fails_without_local_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv(session.SESSION_SOCKET_ENV, str(tmp_path / 'missing.sock'))
    with pytest.raises(RuntimeError, match='session unavailable'):
        preflight.run_ros_preflight(resolve_runtime_config(RuntimeConfig()))


@pytest.fixture
def short_socket_path():
    with tempfile.TemporaryDirectory(prefix='a04-test-', dir='/tmp') as directory:
        yield str(Path(directory) / 'p.sock')


def test_json_transport_round_trip_and_server_reuse(short_socket_path, monkeypatch):
    path = short_socket_path
    captures = []
    closed = []

    class Runtime:
        def collect(self, payload):
            captures.append(payload)
            return preflight.RosPreflightResult(True, [], [], {}).to_json_dict()

        def close(self):
            closed.append(True)

    factory = Mock(side_effect=Runtime)
    thread = threading.Thread(target=session.serve, args=(path, os.getppid(), factory), daemon=True)
    thread.start()
    try:
        session._wait_for_service(path, is_alive=thread.is_alive, timeout_sec=2)
        monkeypatch.setenv(session.SESSION_SOCKET_ENV, path)
        config = resolve_runtime_config(RuntimeConfig())
        assert preflight.run_ros_preflight(config).ok
        assert preflight.run_ros_preflight(config).ok
        assert factory.call_count == 1 and len(captures) == 2
    finally:
        session._exchange(path, {'stop': True}, 2)
        thread.join(timeout=2)
    assert not thread.is_alive() and closed == [True]


@pytest.mark.parametrize('data', [b'{}', b'[]\n', b'x' * (session.MAX_MESSAGE_BYTES + 1)])
def test_malformed_transport_message_rejected(data):
    with pytest.raises(RuntimeError):
        session._read_message(io.BytesIO(data))
